"""H6 tests: plan path on hosted Jev (T-HB8)."""
import random
import time
import uuid

import pytest

import decideonce.service as S
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 0}
    base = 1761950000

    def tick():
        state["t"] += 1
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(base + state["t"]))

    monkeypatch.setattr(S, "_utcnow", tick)
    return state


def make_tiered(tmp_path, **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    return svc, store, cfg


def point(**kw):
    p = {"point_id": "h6.plan", "workflow": "hw", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"},
                            "amount": {"type": "number"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def test_T_HB8_plan_run_sends_input_fields_policy(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(policy="plan rules"))["ok"]
    for i in range(3):  # plan_eligible: at-start labelled rows
        r = svc.do_decide({"point_id": "h6.plan",
                           "context": {"label_hint": "a", "amount": 5}})
        assert svc.do_record({"decision_id": r["decision_id"],
                              "choice": "a"})["ok"]
    p = store.get_point("h6.plan")
    p["plan_phase"] = "shadow"
    store.put_point(p)
    transport = svc.deciders["hosted_jev"].transport
    calls_before = transport.calls
    r = svc.do_plan_run({"workflow": "hw", "input": "refund ticket #1",
                         "run_model": "m", "cheap_model": "c",
                         "context": {"h6.plan": {"label_hint": "a",
                                                 "amount": 5}}})
    assert r["ok"], r
    assert transport.calls == calls_before + 1
    body = transport.requests[-1]
    assert list(body["state"]) == ["policy", "input", "label_hint", "amount"]
    assert body["state"]["policy"] == "plan rules"
    assert body["state"]["input"] == "refund ticket #1"
    plan = store.get_plan(r["plan_id"])
    entry = plan["points"][0]
    assert entry["predicted"] is True
    assert set(entry["probabilities"]) == {"a", "b"}
    assert entry["decider"] == "hosted_jev"
    assert entry["hosted_cost_usd"] > 0


def craft_plan(store, pid, choice="a", pilot=False):
    plan_id = uuid.uuid4().hex
    store.put_plan({
        "plan_id": plan_id, "ts": S._utcnow(), "workflow": "hw",
        "recommendation": "rec",
        "points": [{"point_id": pid, "phase": "shadow", "status": "open",
                    "predicted": True, "choice": choice, "confidence": 0.9,
                    "probabilities": {"a": 0.9, "b": 0.1},
                    "allowed_options": ["a", "b"],
                    "decider": "hosted_jev", "hosted_cost_usd": 0.0001}],
        "point_versions": {pid: 1}, "pilot_forced": pilot})
    tid = uuid.uuid4().hex
    store.put_transcript({"transcript_id": tid, "ts": S._utcnow(),
                          "run_id": uuid.uuid4().hex, "workflow": "hw",
                          "plan_id": plan_id})
    did = uuid.uuid4().hex
    store.insert_decision({
        "decision_id": did, "point_id": pid, "point_version": 1,
        "ts": S._utcnow(), "context": {"label_hint": "a"},
        "options": ["a", "b"], "allowed_options": ["a", "b"],
        "mode": "extracted", "source": "extracted",
        "extracted_choice": choice, "transcript_id": tid, "phase": "collect"})
    return plan_id, did


def test_T_HB8_plan_fit_evidence_and_item1(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=4)
    cfg["serve"]["min_auto_support"] = 3
    cfg["train"]["min_examples"] = 2
    assert svc.do_register(point())["ok"]
    p = store.get_point("h6.plan")
    p["plan_phase"] = "shadow"
    store.put_point(p)
    plan_ids = []
    for _ in range(4):
        pid, did = craft_plan(store, "h6.plan")
        plan_ids.append(pid)
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    fit = store.active_hosted_fit("h6.plan", "plan")
    assert fit is not None and fit["n_fit"] == 4
    assert fit["threshold"] == 0.9
    assert fit["fit_ids"] == plan_ids
    pilot_pid, pilot_did = craft_plan(store, "h6.plan", pilot=True)
    assert svc.do_outcome({"decision_id": pilot_did, "ok": False,
                           "correct_choice": "a"})["ok"]
    for _ in range(2):
        _, did = craft_plan(store, "h6.plan")
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    comps = svc._hosted_plan_comparisons(store.get_point("h6.plan"))
    assert len(comps) == 6  # pilot-forced excluded
    assert pilot_pid not in [c["plan_id"] for c in comps]
    ev = svc._plan_evidence_hosted(store.get_point("h6.plan"), fit)
    assert len(ev) == 2
    r = svc.do_promote({"point_id": "h6.plan", "plan_phase": "serve",
                        "confirm": True})
    assert r["checks"]["min_examples"]["pass"] is True  # 2 evidence comps
    cfg["train"]["min_examples"] = 3
    r = svc.do_promote({"point_id": "h6.plan", "plan_phase": "serve",
                        "confirm": True})
    assert r["checks"]["min_examples"]["pass"] is False
    rep = svc.do_report({"point_id": "h6.plan"})["points"][0]
    assert rep["hosted"]["plan_fit"]["fit_label_kinds"] == {"correctness": 4}
    assert rep["hosted"]["plan_fit"]["evidence_comparisons"] == 2


# -- C1: hosted plan shadow without a trained probe --------------------


def test_C1_plan_shadow_needs_no_probe(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h6.plan", "plan_phase": "shadow"})
    assert r["ok"], r
    assert r["plan_phase"] == "shadow"
    assert store.get_point("h6.plan")["plan_phase"] == "shadow"


def test_C1_plan_shadow_no_key(tmp_path, clock):
    svc, _, _ = make_tiered(tmp_path)
    svc.deciders["hosted_jev"].key = None
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h6.plan", "plan_phase": "shadow"})
    assert r["ok"] is False and r["error"] == "no_hosted_jev_key"


def test_C1_plan_shadow_no_budget(tmp_path, clock):
    svc, _, _ = make_tiered(tmp_path, monthly_budget_usd=0)
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h6.plan", "plan_phase": "shadow"})
    assert r["ok"] is False and r["failed"] == ["budget"]


def test_C1_plan_shadow_private_needs_remote_ok(tmp_path, clock):
    svc, _, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private"))["ok"]
    r = svc.do_promote({"point_id": "h6.plan", "plan_phase": "shadow"})
    assert r["ok"] is False and r["error"] == "remote_not_allowed"


# -- C3: hosted failures surface on the plan path ----------------------


def _labelled_rows(svc, n=3, choice="a"):
    for _ in range(n):  # plan_eligible: at-start labelled rows
        r = svc.do_decide({"point_id": "h6.plan",
                           "context": {"label_hint": "a", "amount": 5}})
        assert svc.do_record({"decision_id": r["decision_id"],
                              "choice": choice})["ok"]


def _plan_call(svc):
    return svc.do_plan_run({"workflow": "hw", "input": "refund ticket #1",
                            "run_model": "m", "cheap_model": "c",
                            "context": {"h6.plan": {"label_hint": "a",
                                                    "amount": 5}}})


def test_C3_plan_path_surfaces_no_key(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    _labelled_rows(svc)
    assert svc.do_promote({"point_id": "h6.plan",
                           "plan_phase": "shadow"})["ok"]
    svc.deciders["hosted_jev"].key = None
    out = _plan_call(svc)
    assert out["ok"], out
    assert out["warnings"] == ["h6.plan: no_hosted_jev_key"]
    entry = store.get_plan(out["plan_id"])["points"][0]
    assert entry["predicted"] is False


def test_C3_plan_path_surfaces_budget_exceeded(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    _labelled_rows(svc)
    assert svc.do_promote({"point_id": "h6.plan",
                           "plan_phase": "shadow"})["ok"]
    svc.deciders["hosted_jev"].monthly_budget_usd = 0  # budget exhausted
    out = _plan_call(svc)
    assert out["ok"], out
    assert out["warnings"] == ["h6.plan: budget_exceeded"]
    entry = store.get_plan(out["plan_id"])["points"][0]
    assert entry["predicted"] is False


# -- C6: plan epochs mirror A5 -----------------------------------------


def craft_shadow_plan(store, pid, choice="a", ts=None, phase="shadow",
                      audit=False):
    plan_id = uuid.uuid4().hex
    plan_ts = ts or S._utcnow()
    store.put_plan({
        "plan_id": plan_id, "ts": plan_ts, "workflow": "hw",
        "recommendation": "rec",
        "points": [{"point_id": pid, "phase": phase, "status": "open",
                    "predicted": True, "choice": choice, "confidence": 0.9,
                    "probabilities": {"a": 0.9, "b": 0.1},
                    "allowed_options": ["a", "b"],
                    "decider": "hosted_jev", "hosted_cost_usd": 0.0001}],
        "point_versions": {pid: 1}, "pilot_forced": False, "audit": audit})
    tid = uuid.uuid4().hex
    store.put_transcript({"transcript_id": tid, "ts": plan_ts,
                          "run_id": uuid.uuid4().hex, "workflow": "hw",
                          "plan_id": plan_id})
    did = uuid.uuid4().hex
    store.insert_decision({
        "decision_id": did, "point_id": pid, "point_version": 1,
        "ts": plan_ts, "context": {"label_hint": "a"},
        "options": ["a", "b"], "allowed_options": ["a", "b"],
        "mode": "extracted", "source": "extracted",
        "extracted_choice": choice, "transcript_id": tid, "phase": "collect"})
    return plan_id, did


def test_C6_plan_comparisons_exclude_pre_epoch(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    old_pid, old_did = craft_shadow_plan(store, "h6.plan",
                                         ts="2020-01-01T00:00:00Z")
    new_pid, new_did = craft_shadow_plan(store, "h6.plan")
    for did in (old_did, new_did):
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    comps = svc._hosted_plan_comparisons(store.get_point("h6.plan"))
    assert [c["plan_id"] for c in comps] == [new_pid]


def test_C6_plan_drift_starts_new_epoch(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=2)
    cfg["serve"]["drift_window"] = 3
    cfg["serve"]["min_auto_support"] = 1
    assert svc.do_register(point())["ok"]
    for _ in range(2):  # a plan fit to deactivate on demotion
        _, did = craft_shadow_plan(store, "h6.plan")
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    assert store.active_hosted_fit("h6.plan", "plan") is not None
    p = store.get_point("h6.plan")
    p["plan_phase"] = "serve"
    store.put_point(p)
    epoch_before = store.get_point("h6.plan")["plan_fit_epoch_start"]
    decide_before = store.get_point("h6.plan")["fit_epoch_start"]
    for _ in range(3):  # a full window of pre-epoch audit errors: ignored
        _, did = craft_shadow_plan(store, "h6.plan", choice="b",
                                   ts="2020-01-01T00:00:00Z",
                                   phase="serve", audit=True)
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    assert store.get_point("h6.plan")["plan_phase"] == "serve"
    for _ in range(3):  # a full window of current-epoch errors: demotes
        _, did = craft_shadow_plan(store, "h6.plan", choice="b",
                                   phase="serve", audit=True)
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    assert store.get_point("h6.plan")["plan_phase"] == "shadow"
    assert (store.get_point("h6.plan")["plan_fit_epoch_start"]
            > epoch_before)
    # R2-1: the decide epoch is untouched by plan drift.
    assert store.get_point("h6.plan")["fit_epoch_start"] == decide_before
    assert store.active_hosted_fit("h6.plan", "plan") is None
    # the demotion's audits are pre-epoch now: re-serve and the window is
    # empty, so no instant re-demotion (the A5 shape).
    p = store.get_point("h6.plan")
    p["plan_phase"] = "serve"
    store.put_point(p)
    assert svc._plan_drift_check("h6.plan") is False
    assert store.get_point("h6.plan")["plan_phase"] == "serve"


# -- Section D: plan eligibility and label kinds -----------------------------

def _linked_plan(store, tid, injected, pilot):
    plan_id = uuid.uuid4().hex
    store.put_plan({"plan_id": plan_id, "ts": S._utcnow(), "workflow": "hw",
                    "recommendation": "rec", "points": [],
                    "point_versions": {}, "pilot_forced": pilot,
                    **({"addendum": "decided: damage"} if injected else {})})
    store.put_transcript({"transcript_id": tid, "ts": S._utcnow(),
                          "run_id": uuid.uuid4().hex, "workflow": "hw",
                          "plan_id": plan_id})
    return plan_id


def _label_row(store, pid, at_start, tid=None, extracted=False):
    row = {"decision_id": uuid.uuid4().hex, "point_id": pid,
           "point_version": 1, "ts": S._utcnow(),
           "context": {"label_hint": "a"},
           "options": ["a", "b"], "allowed_options": ["a", "b"],
           "mode": "extracted" if extracted else "teacher",
           "source": "extracted" if extracted else "do_decide",
           "at_start": at_start, "transcript_id": tid, "phase": "collect"}
    if extracted:
        row["extracted_choice"] = "a"
    else:
        row["teacher_choice"] = "a"
        row["final_choice"] = "a"
    store.insert_decision(row)


def test_D_plan_eligible_excludes_injected_pilot_gate_failed(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    for pid in ("h6.inj", "h6.pilot", "h6.gate", "h6.clean"):
        assert svc.do_register(point(point_id=pid))["ok"]
    # 9 dirty at-start rows + 1 clean not-at-start row: 0.9 either way, but
    # the dirty rows must not count, so the rate is 0/1, not 9/10.
    tid = uuid.uuid4().hex
    _linked_plan(store, tid, injected=True, pilot=False)
    for _ in range(9):
        _label_row(store, "h6.inj", True, tid=tid)
    _label_row(store, "h6.inj", False)
    tid = uuid.uuid4().hex
    _linked_plan(store, tid, injected=False, pilot=True)
    for _ in range(9):
        _label_row(store, "h6.pilot", True, tid=tid)
    _label_row(store, "h6.pilot", False)
    for _ in range(9):  # gate fails (nothing audited): excluded
        _label_row(store, "h6.gate", True, extracted=True)
    _label_row(store, "h6.gate", False)
    for _ in range(3):
        _label_row(store, "h6.clean", True)
    for pid in ("h6.inj", "h6.pilot", "h6.gate"):
        p = store.get_point(pid)
        assert svc._plan_at_start_rate(p) == (0.0, 0, 1)
        assert svc._plan_eligible_hosted(p) is False
    p = store.get_point("h6.clean")
    assert svc._plan_at_start_rate(p) == (1.0, 3, 3)
    assert svc._plan_eligible_hosted(p) is True


def test_D_plan_comparisons_exclude_teacher_kind(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    plan_id = uuid.uuid4().hex
    store.put_plan({
        "plan_id": plan_id, "ts": S._utcnow(), "workflow": "hw",
        "recommendation": "rec",
        "points": [{"point_id": "h6.plan", "phase": "shadow", "status": "open",
                    "predicted": True, "choice": "a", "confidence": 0.9,
                    "probabilities": {"a": 0.9, "b": 0.1},
                    "allowed_options": ["a", "b"],
                    "decider": "hosted_jev", "hosted_cost_usd": 0.0001}],
        "point_versions": {"h6.plan": 1}, "pilot_forced": False})
    tid = uuid.uuid4().hex
    store.put_transcript({"transcript_id": tid, "ts": S._utcnow(),
                          "run_id": uuid.uuid4().hex, "workflow": "hw",
                          "plan_id": plan_id})
    _label_row(store, "h6.plan", True, tid=tid)
    plan = store.get_plan(plan_id)
    assert svc._plan_comparisons(plan)[0]["kind"] == "teacher"
    assert svc._hosted_plan_comparisons(
        store.get_point("h6.plan")) == []


def test_D_plan_fit_stores_label_kinds(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path, fit_rows=4)
    assert svc.do_register(point())["ok"]
    for _ in range(4):
        _, did = craft_shadow_plan(store, "h6.plan")
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    fit = store.active_hosted_fit("h6.plan", "plan")
    assert fit["fit_label_kinds"] == {"correctness": 4}
    rep = svc.do_report({"point_id": "h6.plan"})["points"][0]
    assert rep["hosted"]["plan_fit"]["fit_label_kinds"] == {"correctness": 4}
    # pre-v11 row (no stored kinds): the report recomputes instead.
    store.cx.execute("UPDATE hosted_fit SET fit_label_kinds=NULL WHERE id=?",
                     (fit["id"],))
    rep = svc.do_report({"point_id": "h6.plan"})["points"][0]
    assert rep["hosted"]["plan_fit"]["fit_label_kinds"] == {"correctness": 4}


def test_D_plan_checks_detail_keys_lowercase(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    for name, check in svc._plan_serve_checks_hosted(
            store.get_point("h6.plan"), True).items():
        assert "D" + "etail" not in check, name
        assert "detail" in check, name


def test_D_plan_comparisons_skip_gate_without_extracted(tmp_path, clock,
                                                        monkeypatch):
    from decideonce import train as TR
    svc, store, _ = make_tiered(tmp_path, fit_rows=200)
    assert svc.do_register(point())["ok"]
    calls = []
    real_gate = TR.extraction_gate

    def counting(cfg, store_, point):
        calls.append(1)
        return real_gate(cfg, store_, point)

    monkeypatch.setattr(TR, "extraction_gate", counting)
    for _ in range(2):  # correctness comparisons: no extracted kind in play
        _, did = craft_shadow_plan(store, "h6.plan")
        assert svc.do_outcome({"decision_id": did, "ok": False,
                               "correct_choice": "a"})["ok"]
    comps = svc._hosted_plan_comparisons(store.get_point("h6.plan"))
    assert len(comps) == 2 and calls == []
    craft_shadow_plan(store, "h6.plan")  # extracted, gate fails: excluded
    comps = svc._hosted_plan_comparisons(store.get_point("h6.plan"))
    assert len(comps) == 2 and len(calls) == 1
