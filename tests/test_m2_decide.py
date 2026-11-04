"""M2 tests: rendering/rules (T-R*) and decide/record (T-D*)."""
import random

from decideonce import labels as L
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store
from decideonce.trainer import FakeTrainer


def make_service(tmp_path):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg)), store


def refund_point(**kw):
    p = {"point_id": "refund.action", "workflow": "refund_request",
         "question": "Which action?",
         "kind": "choice",
         "options": ["full_refund", "partial_refund", "replacement", "escalate_human"],
         "context_fields": {"customer_tier": {"type": "string", "enum": ["basic", "gold"]},
                            "amount_eur": {"type": "number"},
                            "days_since_delivery": {"type": "integer"},
                            "reason": {"type": "string", "max_len": 400},
                            "photo": {"type": "boolean"}},
         "required_fields": ["amount_eur", "days_since_delivery", "reason"],
         "rules": [{"id": "big", "if": [["amount_eur", ">", 500]], "only": ["escalate_human"]},
                   {"id": "late", "if": [["days_since_delivery", ">", 30]],
                    "exclude": ["full_refund"]}],
         "data_class": "public"}
    p.update(kw)
    return p


def ctx(**kw):
    c = {"customer_tier": "gold", "amount_eur": 80, "days_since_delivery": 12,
         "reason": "damaged", "photo": True}
    c.update(kw)
    return c


# -- rendering / rules ---------------------------------------------------

def test_T_R1_same_context_identical_state_text(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r1 = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    r2 = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    t1 = store.get_decision(r1["decision_id"])["state_text"]
    t2 = store.get_decision(r2["decision_id"])["state_text"]
    assert t1 == t2
    assert "workflow: refund_request" in t1
    assert "photo: yes" in t1


def test_T_R2_field_order_follows_registration(tmp_path):
    svc, _ = make_service(tmp_path)
    p = refund_point(context_fields={
        "zebra": {"type": "string"}, "apple": {"type": "integer"}, "mango": {"type": "boolean"}},
        required_fields=[], rules=[])
    assert svc.do_register(p)["ok"]
    r = svc.do_decide({"point_id": "refund.action",
                       "context": {"mango": True, "apple": 3, "zebra": "x"}})
    assert r["ok"]
    text = svc.store.get_decision(r["decision_id"])["state_text"]
    assert text.index("zebra:") < text.index("apple:") < text.index("mango:")


def test_T_R3_only_intersect_then_exclude(tmp_path):
    svc, _ = make_service(tmp_path)
    p = refund_point(rules=[
        {"id": "a", "if": [["amount_eur", ">", 10]], "only": ["full_refund", "replacement"]},
        {"id": "b", "if": [["photo", "==", True]], "only": ["replacement", "escalate_human"]},
        {"id": "c", "if": [["photo", "==", True]], "exclude": ["escalate_human"]}])
    assert svc.do_register(p)["ok"]
    # only: {full,repl} ∩ {repl,esc} = {repl}; exclude esc -> {repl} -> rule mode
    r = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    assert r["mode"] == "rule" and r["choice"] == "replacement"
    assert r["allowed_options"] == ["replacement"]


def test_T_R4_empty_set_rules_conflict(tmp_path):
    svc, _ = make_service(tmp_path)
    p = refund_point(rules=[
        {"id": "a", "if": [["amount_eur", ">", 10]], "only": ["full_refund"]},
        {"id": "b", "if": [["photo", "==", True]], "only": ["replacement"]}])
    assert svc.do_register(p)["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    assert r["ok"] is False and r["error"] == "rules_conflict"


def test_T_R5_missing_field_rule_skipped_with_warning(tmp_path):
    svc, _ = make_service(tmp_path)
    p = refund_point(required_fields=["amount_eur"],
                     rules=[{"id": "r", "if": [["nickname", "==", "x"]], "only": ["escalate_human"]}])
    p["context_fields"]["nickname"] = {"type": "string"}
    assert svc.do_register(p)["ok"]
    r = svc.do_decide({"point_id": "refund.action",
                       "context": {"amount_eur": 5, "days_since_delivery": 1, "reason": "x"}})
    assert r["ok"]
    assert any(w.startswith("rule_field_missing") for w in r["warnings"])
    assert r["allowed_options"] == p["options"]  # rule did not fire


def test_T_R6_single_remaining_option_mode_rule(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": ctx(amount_eur=600)})
    assert r["mode"] == "rule"
    assert r["choice"] == "escalate_human"
    assert r["probabilities"] == {"escalate_human": 1.0}
    row = store.get_decision(r["decision_id"])
    assert row["jev_zero_shot"] is None  # rules decide without JEV


def test_T_R7_no_eval_malicious_strings_inert(tmp_path):
    svc, _ = make_service(tmp_path)
    p = refund_point(context_fields={"__class__": {"type": "string"},
                                     "reason": {"type": "string", "max_len": 400}},
                     required_fields=["reason"], rules=[])
    assert svc.do_register(p)["ok"]
    evil = "__import__('os').system('x')"
    r = svc.do_decide({"point_id": "refund.action",
                       "context": {"reason": evil, "__class__": evil}})
    assert r["ok"]
    row = svc.store.get_decision(r["decision_id"])
    assert evil in row["state_text"]  # stored as inert text


# -- decide / record -----------------------------------------------------

def test_T_D1_missing_required_field(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": {"amount_eur": 5}})
    assert r["ok"] is False and r["error"] == "missing_field"


def test_D_nonfinite_number_context_rejected(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    for bad in (float("nan"), float("inf"), float("-inf")):
        r = svc.do_decide({"point_id": "refund.action",
                           "context": ctx(amount_eur=bad)})
        assert r["ok"] is False and r["error"] == "invalid_context", bad
    assert svc.do_decide({"point_id": "refund.action",
                          "context": ctx(amount_eur=80.5)})["ok"]


def test_T_D2_collect_teacher_no_model_call(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": ctx(),
                       "teacher_cost": {"tokens_in": 10, "usd": 0.01}})
    assert r["mode"] == "teacher" and r["choice"] is None
    assert r["probabilities"] == {}  # tiered collect calls no model
    row = store.get_decision(r["decision_id"])
    assert row["teacher_cost"] == {"tokens_in": 10, "usd": 0.01}
    assert row["probe"] is None and row["hosted"] is None
    assert row["jev_seconds"] is None


def test_T_D3_record_disallowed_choice_not_allowed(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r = svc.do_decide({"point_id": "refund.action",
                       "context": ctx(days_since_delivery=40)})  # late: no full_refund
    assert "full_refund" not in r["allowed_options"]
    bad = svc.do_record({"decision_id": r["decision_id"], "choice": "full_refund"})
    assert bad["ok"] is False and bad["error"] == "choice_not_allowed"
    assert store.get_decision(r["decision_id"])["rule_violation"] is True
    good = svc.do_record({"decision_id": r["decision_id"], "choice": r["allowed_options"][0]})
    assert good["ok"]
    row = store.get_decision(r["decision_id"])
    assert row["teacher_choice"] == row["final_choice"] == r["allowed_options"][0]


def test_S8_invalid_choice_counted_separately_no_rules(tmp_path):
    svc, store = make_service(tmp_path)
    p = refund_point()
    p.pop("rules", None)
    assert svc.do_register(p)["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    bad = svc.do_record({"decision_id": r["decision_id"],
                         "choice": "not_an_option"})
    assert bad["ok"] is False and bad["error"] == "choice_not_allowed"
    rep = svc.do_report({"point_id": "refund.action"})["points"][0]
    assert rep["invalid_choices"] == 1
    assert rep["rule_violations"] == 0
    assert rep["tier"] == "hosted_jev"  # S8.5: the effective backend


def test_S8_rule_violation_still_counted_with_rules(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]  # has rules
    r = svc.do_decide({"point_id": "refund.action",
                       "context": ctx(days_since_delivery=40)})
    bad = svc.do_record({"decision_id": r["decision_id"],
                         "choice": "full_refund"})
    assert bad["ok"] is False and bad["error"] == "choice_not_allowed"
    rep = svc.do_report({"point_id": "refund.action"})["points"][0]
    assert rep["rule_violations"] == 1
    assert rep["invalid_choices"] == 0


def test_T_D4_hosted_down_teacher_with_warning(tmp_path):
    from decideonce.deciders.base import DeciderError
    from decideonce.deciders.hosted import FakeHostedTransport

    svc, _ = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    assert svc.do_promote({"point_id": "refund.action",
                           "phase": "shadow"})["ok"]

    def boom(body):
        raise DeciderError("timeout", "scripted outage")

    svc.deciders["hosted_jev"].transport = FakeHostedTransport(fn=boom)
    r = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"
    assert "jev_unavailable" in r["warnings"]


def test_T_D5_version_bump_old_rows_excluded_from_training(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    svc.do_record({"decision_id": r["decision_id"], "choice": "full_refund"})
    assert svc.do_register(refund_point())["point"]["point_version"] == 1  # no change
    v2 = svc.do_register(refund_point(
        options=["full_refund", "partial_refund", "replacement", "escalate_human", "deny"]))
    assert v2["point"]["point_version"] == 2
    point = store.get_point("refund.action")
    rows = store.decisions_for_point("refund.action")
    assert L.labeled_rows(rows, point) == []  # option list differs -> excluded
    # same options, changed fields: old rows still usable
    p3 = dict(refund_point(options=v2["point"]["options"]))
    p3["context_fields"]["note2"] = {"type": "string"}
    assert svc.do_register(p3)["point"]["point_version"] == 3
    point3 = store.get_point("refund.action")
    rows3 = store.decisions_for_point("refund.action")
    assert L.labeled_rows(rows3, point3) == []  # still v1 options vs v2 options
    r2 = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    svc.do_record({"decision_id": r2["decision_id"], "choice": "deny"})
    got = L.labeled_rows(store.decisions_for_point("refund.action"), point3)
    assert [g[1] for g in got] == ["deny"]


def test_T_D6_too_many_options_rejected(tmp_path):
    svc, _ = make_service(tmp_path)
    r = svc.do_register(refund_point(options=[f"o{i}" for i in range(17)]))
    assert r["ok"] is False and r["error"] == "too_many_options"


def test_noul_point_two_options_used_as_is(tmp_path):
    svc, store = make_service(tmp_path)
    p = refund_point(kind="noul", options=["no", "yes"], rules=[],
                     required_fields=["reason"])
    assert svc.do_register(p)["ok"]
    bad = svc.do_register(refund_point(kind="noul", options=["a", "b", "c"], rules=[]))
    assert bad["ok"] is False
    assert svc.do_promote({"point_id": "refund.action",
                           "phase": "shadow"})["ok"]
    r = svc.do_decide({"point_id": "refund.action",
                       "context": {"amount_eur": 1, "days_since_delivery": 1, "reason": "x"}})
    assert r["ok"]
    row = store.get_decision(r["decision_id"])
    assert set(row["hosted"]["probabilities"]) == {"no", "yes"}


def test_outcome_stored_and_validated(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    r = svc.do_decide({"point_id": "refund.action", "context": ctx()})
    bad = svc.do_outcome({"decision_id": r["decision_id"], "correct_choice": "nope"})
    assert bad["ok"] is False and bad["error"] == "unknown_option"
    good = svc.do_outcome({"decision_id": r["decision_id"], "ok": True, "note": "closed"})
    assert good["ok"]
    assert store.get_decision(r["decision_id"])["outcome"]["ok"] is True
    assert svc.do_outcome({"decision_id": "missing"})["error"] == "unknown_decision"
    assert svc.do_record({"decision_id": "missing", "choice": "x"})["error"] == "unknown_decision"
    assert svc.do_decide({"point_id": "missing", "context": {}})["error"] == "unknown_point"


def test_R32_threaded_record_never_breaks_transactions(tmp_path):
    """R3-2: 40 threaded do_record calls share one Store; the RLock must
    serialize the BEGIN IMMEDIATE blocks (no 'cannot commit' errors)."""
    import threading

    svc, store = make_service(tmp_path)
    assert svc.do_register(refund_point())["ok"]
    ctx = {"customer_tier": "gold", "amount_eur": 10,
           "days_since_delivery": 3, "reason": "late", "photo": False}
    dids = [svc.do_decide({"point_id": "refund.action", "context": ctx})[
        "decision_id"] for _ in range(40)]
    errors: list = []

    def rec(did):
        try:
            r = svc.do_record({"decision_id": did, "choice": "escalate_human"})
            if not r["ok"]:
                errors.append(r)
        except Exception as e:  # noqa: BLE001 - collected, asserted below
            errors.append(e)

    threads = [threading.Thread(target=rec, args=(d,)) for d in dids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    rows = store.decisions_for_point("refund.action")
    assert sum(1 for r in rows if r.get("teacher_choice")) == 40
