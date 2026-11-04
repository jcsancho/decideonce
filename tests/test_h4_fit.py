"""H4 tests: shadow without training, one-time fit, serve, drift (T-HJ1/2, T-HB7/11)."""
import random
import threading
import time
import uuid

import pytest

import decideonce.service as S
from decideonce import hostedfit as HF
from decideonce.stats import mask_and_renorm as _mask
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.deciders.hosted import FakeHostedTransport, HostedJevDecider
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


def make_tiered(tmp_path, backend="fake", **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = backend
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    return svc, store, cfg


def make_custom(tmp_path, fn, **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    dec = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
        key="fake", store=store, timeout_s=3,
        price_per_mtok_in=cfg["hosted_jev"]["price_per_mtok_in"],
        monthly_budget_usd=cfg["hosted_jev"]["monthly_budget_usd"],
        reserve_usd=cfg["hosted_jev"]["reserve_usd"],
        transport=FakeHostedTransport(fn=fn))
    return Service(cfg, store, {"hosted_jev": dec},
                   rng=random.Random(0)), store, cfg


def point(**kw):
    p = {"point_id": "h4.pt", "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def decide_label(svc, label, hint=None):
    r = svc.do_decide({"point_id": "h4.pt",
                       "context": {"label_hint": label if hint is None
                                   else hint}})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec
    return r["decision_id"]


# -- T-HJ1 shadow without training ------------------------------------------


def test_T_HJ1_shadow_needs_no_model_and_stores_hosted(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})
    assert r["ok"], r
    for i in range(5):
        did = decide_label(svc, "ab"[i % 2])
        row = store.get_decision(did)
        assert row["hosted"] is not None
        assert abs(sum(row["hosted"]["probabilities"].values()) - 1.0) < 1e-9
        assert row["decider"] == "hosted_jev"
        assert row["hosted"]["model"] == "typesafe/jev-1.13"


def test_T_HJ1_shadow_gates(tmp_path):
    svc, store, _ = make_tiered(tmp_path, backend="tiered")
    # tiered with no key_ref resolves no key
    assert svc.deciders["hosted_jev"].key is None
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "no_hosted_jev_key"

    svc2, store2, _ = make_tiered(tmp_path / "b0", monthly_budget_usd=0)
    assert svc2.do_register(point())["ok"]
    r = svc2.do_promote({"point_id": "h4.pt", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "promotion_refused"
    assert r["failed"] == ["budget"]

    svc3, _, _ = make_tiered(tmp_path / "priv")
    assert svc3.do_register(point(data_class="private"))["ok"]
    r = svc3.do_promote({"point_id": "h4.pt", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "remote_not_allowed"

    svc4, _, _ = make_tiered(tmp_path / "pmo")
    assert svc4.do_register(point(data_class="private",
                                  private_model_only=True))["ok"]
    r = svc4.do_promote({"point_id": "h4.pt", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"


# -- T-HJ2 one-time fit -----------------------------------------------------


def test_T_HJ2_fit_once_on_first_fit_rows(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=10)
    assert svc.do_register(point())["ok"]
    collect_ids = {decide_label(svc, "ab"[i % 2]) for i in range(3)}
    assert store.active_hosted_fit("h4.pt") is None  # collect never fits
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(9):
        decide_label(svc, "ab"[i % 2])
        assert store.active_hosted_fit("h4.pt") is None  # not before
    decide_label(svc, "b")
    fit = store.active_hosted_fit("h4.pt")
    assert fit is not None and fit["n_fit"] == 10
    assert not (collect_ids & set(fit["fit_ids"]))
    first_id, first_at = fit["id"], fit["fitted_at"]
    for i in range(5):
        decide_label(svc, "ab"[i % 2])
    fit2 = store.active_hosted_fit("h4.pt")
    assert (fit2["id"], fit2["fitted_at"], fit2["n_fit"]) == (first_id, first_at, 10)
    ev = svc._evidence_rows(store.get_point("h4.pt"), fit2)
    assert len(ev) == 5


def test_H2_same_second_splits_exactly_at_fit_rows(tmp_path, monkeypatch):
    # Prepublish review H2: with every decision in one second, the fit must
    # take exactly the first fit_rows in insertion order (old code split by
    # random decision_id, scrambling fit/evidence/late sets run to run).
    svc, store, cfg = make_tiered(tmp_path, fit_rows=200)
    monkeypatch.setattr(S, "_utcnow", lambda: "2026-11-03T12:00:00.000000Z")
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    dids = [decide_label(svc, "ab"[i % 2]) for i in range(500)]
    assert [r["seq"] for r in store.decisions_for_point("h4.pt")] == list(
        range(1, 501))
    fit = store.active_hosted_fit("h4.pt")
    assert fit is not None and fit["n_fit"] == 200
    assert fit["fit_ids"] == dids[:200]
    pt = store.get_point("h4.pt")
    assert [r["decision_id"] for r, _ in svc._evidence_rows(pt, fit)] == \
        dids[200:]
    assert svc._late_rows(pt, fit) == []


def test_T_HJ2_item1_counts_evidence_only(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=10)
    cfg["train"]["min_examples"] = 3
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    r = svc.do_promote({"point_id": "h4.pt", "phase": "serve",
                        "confirm": True})
    assert r["ok"] is False
    assert r["checks"]["min_examples"]["pass"] is False  # 0 evidence
    for i in range(3):
        decide_label(svc, "ab"[i % 2])
    r = svc.do_promote({"point_id": "h4.pt", "phase": "serve",
                        "confirm": True})
    assert r["checks"]["min_examples"]["pass"] is True  # 3 evidence, not 13


def test_T_HJ2_drift_demotion_rearms(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=10)
    cfg["serve"]["drift_window"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    fit1 = store.active_hosted_fit("h4.pt")
    assert fit1 is not None
    p = store.get_point("h4.pt")
    epoch1 = p["fit_epoch_start"]
    p["phase"] = "serve"
    store.put_point(p)
    # 5 failing audits: hosted top a, teacher b
    audit_ids = []
    for i in range(5):
        did = uuid.uuid4().hex
        audit_ids.append(did)
        store.insert_decision({
            "decision_id": did, "point_id": "h4.pt", "point_version": 1,
            "ts": S._utcnow(), "context": {"label_hint": "a"},
            "options": ["a", "b"], "allowed_options": ["a", "b"],
            "hosted": {"model": "typesafe/jev-1.13",
                       "probabilities": {"a": 0.9, "b": 0.1},
                       "cost_usd": 0.0001, "seconds": 0.1},
            "decider": "hosted_jev", "mode": "teacher", "audit": True,
            "source": "do_decide", "phase": "serve"})
    # the other four audits arrive pre-labelled; the first is labelled live
    for did in audit_ids[1:]:
        store.update_decision(did, {"teacher_choice": "b",
                                    "final_choice": "b"})
    r = svc.do_record({"decision_id": audit_ids[0], "choice": "b"})
    assert r["ok"] and r["demoted"] is True
    p = store.get_point("h4.pt")
    assert p["phase"] == "shadow"
    assert p["fit_epoch_start"] > epoch1
    assert store.active_hosted_fit("h4.pt") is None
    assert p["events"][-1]["event"] == "drift_demotion"
    assert p["events"][-1]["tier"] == "hosted_jev"
    # re-armed: fit_rows new labels refit on new-epoch rows only
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    fit2 = store.active_hosted_fit("h4.pt")
    assert fit2 is not None and fit2["id"] != fit1["id"]
    assert len(fit2["fit_ids"]) == 10
    for did in fit2["fit_ids"]:
        row = store.get_decision(did)
        assert (row["ts"], row["decision_id"]) >= (p["fit_epoch_start"], "")


# -- T-HB7 late labels ------------------------------------------------------


def test_T_HB7_late_label_in_neither_set(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path, fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    dids = []
    for i in range(11):
        r = svc.do_decide({"point_id": "h4.pt",
                           "context": {"label_hint": "ab"[i % 2]}})
        dids.append(r["decision_id"])
    for did in dids[:9] + dids[10:]:
        assert svc.do_record({"decision_id": did,
                              "choice": "a" if dids.index(did) % 2 == 0
                              else "b"})["ok"]
    fit = store.active_hosted_fit("h4.pt")
    assert fit is not None and fit["n_fit"] == 10
    assert fit["last_fit_decision_id"] == dids[10]
    assert svc.do_record({"decision_id": dids[9],
                          "choice": "b"})["ok"]  # decided before, labelled after
    pt = store.get_point("h4.pt")
    late = svc._late_rows(pt, fit)
    assert [r["decision_id"] for r, _ in late] == [dids[9]]
    assert svc._evidence_rows(pt, fit) == []
    assert store.active_hosted_fit("h4.pt")["id"] == fit["id"]


# -- T-HB11 temperature -----------------------------------------------------


def _deterministic_fn(schedule):
    state = {"n": 0}

    def fn(body):
        (pid, q), = body["questions"].items()
        opts = list(q["criteria"])
        hint = body["state"]["label_hint"]
        i = state["n"]
        state["n"] += 1
        top, p_top = schedule(i, hint, opts)
        rest = (1.0 - p_top) / (len(opts) - 1)
        probs = {o: (p_top if o == top else rest) for o in opts}
        return 200, {"answers": {pid: {"choice": top,
                                       "probabilities": probs,
                                       "confidence": p_top}},
                     "usage": {"input_tokens": 50, "output_tokens": 0,
                               "cost": 0.0001}}
    return fn


def test_T_HB11_temperature_lowers_nll(tmp_path, clock):
    def sched(i, hint, opts):
        wrong = (i % 6 == 5)
        top = hint if not wrong else [o for o in opts if o != hint][0]
        return top, 0.65

    svc, store, _ = make_custom(tmp_path, _deterministic_fn(sched),
                                fit_rows=30)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(30):
        decide_label(svc, "ab"[i % 2])
    fit = store.active_hosted_fit("h4.pt")
    assert fit is not None
    assert 0.5 <= fit["temperature"] <= 5.0
    assert fit["temperature"] < 1.0  # underconfident fake sharpens
    assert fit["temperature_skipped"] is None
    items = []
    for did in fit["fit_ids"]:
        r = store.get_decision(did)
        allowed = r["allowed_options"]
        items.append((_mask(r["hosted"]["probabilities"],
                              allowed), allowed,
                      r["teacher_choice"]))
    assert HF.nll_at(items, fit["temperature"]) < HF.nll_at(items, 1.0) - 1e-9


def test_T_HB11_error_free_gives_T1_skipped(tmp_path, clock):
    def sched(i, hint, opts):
        return hint, 0.9

    svc, store, _ = make_custom(tmp_path, _deterministic_fn(sched),
                                fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    fit = store.active_hosted_fit("h4.pt")
    assert fit["temperature"] == 1.0
    assert fit["temperature_skipped"] == "too_few_errors"


def test_T_HB11_threshold_fit_rows_only(tmp_path, clock):
    def sched(i, hint, opts):
        return hint, 0.9

    svc, store, cfg = make_custom(tmp_path, _deterministic_fn(sched),
                                  fit_rows=10)
    cfg["serve"]["min_auto_support"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    fit = store.active_hosted_fit("h4.pt")
    assert fit["threshold"] == 0.9
    for i in range(5):
        decide_label(svc, "ab"[i % 2])
    fit2 = store.active_hosted_fit("h4.pt")
    assert (fit2["id"], fit2["threshold"], fit2["fitted_at"]) == (
        fit["id"], fit["threshold"], fit["fitted_at"])


# -- hosted serve -----------------------------------------------------------


def test_hosted_serve_auto_above_threshold(tmp_path, clock):
    def sched(i, hint, opts):
        return hint, 0.9

    svc, store, cfg = make_custom(tmp_path, _deterministic_fn(sched),
                                  fit_rows=20)
    cfg["serve"]["min_auto_support"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(20):
        decide_label(svc, "ab"[i % 2])
    fit = store.active_hosted_fit("h4.pt")
    assert fit["threshold"] == 0.9
    assert fit["teacher_only_options"] == []  # 10 labels each, recall 1.0
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    r = svc.do_decide({"point_id": "h4.pt", "context": {"label_hint": "a"}})
    assert r["ok"] and r["mode"] == "auto" and r["choice"] == "a"
    assert r["threshold"] == 0.9


# -- section A anti-gaming fixes ---------------------------------------------


def test_A1_shadow_masks_probabilities_but_stores_them(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    r = svc.do_decide({"point_id": "h4.pt", "context": {"label_hint": "a"}})
    assert r["ok"] and r["mode"] == "teacher"
    assert r["probabilities"] == {} and r["confidence"] is None
    assert "audit" not in r
    row = store.get_decision(r["decision_id"])
    assert row["hosted"]["probabilities"]  # still logged for the fit


def test_A1_audit_masks_and_flags(tmp_path, clock):
    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=20)
    cfg["serve"]["min_auto_support"] = 5
    cfg["serve"]["audit_rate"] = 1.0
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(20):
        decide_label(svc, "ab"[i % 2])
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    r = svc.do_decide({"point_id": "h4.pt", "context": {"label_hint": "a"}})
    row = store.get_decision(r["decision_id"])
    assert row["audit"] is True
    assert r["mode"] == "teacher" and r["audit"] is True
    assert r["probabilities"] == {} and r["confidence"] is None


def test_A1_auto_still_shows_probabilities(tmp_path, clock):
    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=20)
    cfg["serve"]["min_auto_support"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(20):
        decide_label(svc, "ab"[i % 2])
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    r = svc.do_decide({"point_id": "h4.pt", "context": {"label_hint": "a"}})
    assert r["mode"] == "auto" and r["probabilities"]["a"] > 0.5
    assert r["confidence"] is not None and "audit" not in r


def test_A2_second_record_refused_and_label_kept(tmp_path, clock):
    svc, store, _ = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: ("a", 0.9)),
        fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "a")
    did = decide_label(svc, "b")  # hosted wrong
    pt = store.get_point("h4.pt")
    fit = svc._hosted_fit(pt)
    errs = lambda: sum(1 for _, ok in svc._score_hosted(
        pt, svc._evidence_rows(pt, fit), 1.0) if not ok)
    assert errs() == 1
    r = svc.do_record({"decision_id": did, "choice": "a"})
    assert r["ok"] is False and r["error"] == "already_recorded"
    assert store.get_decision(did)["teacher_choice"] == "b"
    assert errs() == 1


def test_A2_first_outcome_wins_and_later_ignored(tmp_path, clock):
    svc, store, _ = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=10)
    cfg_ok = svc.do_register(point())
    assert cfg_ok["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    did = decide_label(svc, "a")  # hosted top a, teacher a
    assert svc.do_outcome({"decision_id": did, "ok": False})["ok"]
    pt = store.get_point("h4.pt")
    fit = svc._hosted_fit(pt)
    errs = lambda: sum(1 for _, ok in svc._score_hosted(
        pt, svc._evidence_rows(pt, fit), 1.0) if not ok)
    assert errs() == 1
    r = svc.do_outcome({"decision_id": did, "ok": True,
                        "correct_choice": "b"})
    assert r["ok"] and r["ignored"] is True
    row = store.get_decision(did)
    assert row["outcome"] == {"ok": False, "correct_choice": None,
                              "note": None}
    assert "outcome_ignored" in (row["warnings"] or [])
    assert errs() == 1


def test_A2_labelled_ratio_gates_serve_and_shows_in_report(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path, fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    for i in range(2):  # decided but never labelled
        svc.do_decide({"point_id": "h4.pt",
                       "context": {"label_hint": "ab"[i % 2]}})
    pt = store.get_point("h4.pt")
    assert svc._shadow_label_coverage(pt) == (10, 12)
    check = svc._serve_checks_hosted(pt, True)["labelled_ratio"]
    assert check["pass"] is False  # 10/12 < 0.90
    r = svc.do_promote({"point_id": "h4.pt", "phase": "serve",
                        "confirm": True})
    assert r["ok"] is False and "labelled_ratio" in r["failed"]
    blk = svc._hosted_report_block(pt)
    assert blk["labelled_ratio"] == {"labelled": 10, "decided": 12,
                                     "ratio": 10 / 12}
    for i in range(8):
        decide_label(svc, "ab"[i % 2])
    pt = store.get_point("h4.pt")
    assert svc._serve_checks_hosted(pt, True)["labelled_ratio"][
        "pass"] is True  # 18/20 = 0.90


def test_A3_followed_ok_false_counts_as_hosted_error(tmp_path, clock):
    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=10)
    cfg["serve"]["min_auto_support"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    did = decide_label(svc, "a")  # hosted top a, teacher a
    pt = store.get_point("h4.pt")
    fit = svc._hosted_fit(pt)
    n0 = len(svc._evidence_rows(pt, fit))
    assert svc.do_outcome({"decision_id": did, "ok": False})["ok"]
    ev = svc._evidence_rows(pt, fit)
    assert len(ev) == n0  # kept, not dropped
    assert sum(1 for _, ok in svc._score_hosted(pt, ev, 1.0)
               if not ok) == 1
    # ... but probe training still never sees the known-wrong label
    from decideonce import labels as L
    row = store.get_decision(did)
    assert L.resolve_label(row)[1] == "outcome_bad_no_fix"
    assert did not in {r["decision_id"] for r, _, _, _ in
                       L.labeled_rows(store.decisions_for_point("h4.pt"),
                                      pt)}


def test_A3_unfollowed_ok_false_still_drops(tmp_path, clock):
    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: ("a", 0.9)),
        fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "a")
    did = decide_label(svc, "b")  # hosted top a, teacher b: not followed
    pt = store.get_point("h4.pt")
    fit = svc._hosted_fit(pt)
    n0 = len(svc._evidence_rows(pt, fit))
    assert svc.do_outcome({"decision_id": did, "ok": False})["ok"]
    assert len(svc._evidence_rows(pt, fit)) == n0 - 1


def test_A4_concurrent_fit_leaves_one_row(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path, fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    dids = [svc.do_decide({"point_id": "h4.pt",
                           "context": {"label_hint": "ab"[i % 2]}})[
        "decision_id"] for i in range(12)]
    for d in dids[1:10]:
        svc.do_record({"decision_id": d, "choice": "a"})
    orig = svc._fit_candidates
    barrier = threading.Barrier(2)

    def slow(p):
        out = orig(p)
        try:
            barrier.wait(timeout=5)
        except threading.BrokenBarrierError:
            pass
        return out

    svc._fit_candidates = slow
    t1 = threading.Thread(target=svc.do_record,
                          args=({"decision_id": dids[10], "choice": "a"},))
    t2 = threading.Thread(target=svc.do_record,
                          args=({"decision_id": dids[0], "choice": "a"},))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    rows = store.cx.execute("SELECT id, active FROM hosted_fit").fetchall()
    assert len(rows) == 1 and rows[0]["active"] == 1


def test_A5_drift_window_ignores_previous_epoch(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=10)
    cfg["serve"]["drift_window"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)

    def audit(top_probs, teacher, phase="serve", label=True):
        did = uuid.uuid4().hex
        store.insert_decision({
            "decision_id": did, "point_id": "h4.pt", "point_version": 1,
            "ts": S._utcnow(), "context": {"label_hint": "a"},
            "options": ["a", "b"], "allowed_options": ["a", "b"],
            "hosted": {"model": "m", "probabilities": top_probs,
                       "cost_usd": 0, "seconds": 0.1},
            "decider": "hosted_jev", "mode": "teacher", "audit": True,
            "source": "do_decide", "phase": phase,
            **({"teacher_choice": teacher, "final_choice": teacher}
               if label else {})})
        return did

    first = audit({"a": 0.9, "b": 0.1}, "b", label=False)
    for _ in range(4):
        audit({"a": 0.9, "b": 0.1}, "b")
    assert svc.do_record({"decision_id": first, "choice": "b"})[
        "demoted"] is True
    for i in range(10):  # refit on the new epoch
        decide_label(svc, "ab"[i % 2])
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    good = audit({"a": 0.9, "b": 0.1}, "a", label=False)
    assert svc.do_record({"decision_id": good, "choice": "a"})[
        "demoted"] is False


def test_A6_version_bump_returns_hosted_serve_to_shadow(tmp_path, clock):
    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=10)
    cfg["serve"]["min_auto_support"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    assert svc.do_register(point(policy="v2 rules"))["ok"]  # bump in shadow
    assert store.get_point("h4.pt")["phase"] == "shadow"
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    assert svc._hosted_fit(store.get_point("h4.pt")) is not None
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    assert svc.do_register(point(policy="v3 rules"))["ok"]
    p = store.get_point("h4.pt")
    assert (p["phase"], p["point_version"]) == ("shadow", 3)
    assert svc._hosted_fit(p) is None
    for i in range(10):  # refit fires again on shadow rows
        decide_label(svc, "ab"[i % 2])
    fit = svc._hosted_fit(store.get_point("h4.pt"))
    assert fit is not None and fit["point_version"] == 3


# -- Section D: late labels and overrides ------------------------------------

def test_D_late_labels_exclude_previous_epoch(tmp_path, clock):
    svc, store, cfg = make_tiered(tmp_path, fit_rows=4)
    cfg["serve"]["drift_window"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    for i in range(4):
        decide_label(svc, "ab"[i % 2])
    assert store.active_hosted_fit("h4.pt") is not None
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    audit_ids = []
    for _ in range(5):  # failing audits: hosted top a, teacher b
        did = uuid.uuid4().hex
        audit_ids.append(did)
        store.insert_decision({
            "decision_id": did, "point_id": "h4.pt", "point_version": 1,
            "ts": S._utcnow(), "context": {"label_hint": "a"},
            "options": ["a", "b"], "allowed_options": ["a", "b"],
            "hosted": {"model": "m", "probabilities": {"a": 0.9, "b": 0.1},
                       "cost_usd": 0, "seconds": 0.1},
            "decider": "hosted_jev", "mode": "teacher", "audit": True,
            "source": "do_decide", "phase": "serve",
            "teacher_choice": "b", "final_choice": "b"})
    assert svc._drift_check("h4.pt") is True
    for i in range(4):  # refit on the new epoch
        decide_label(svc, "ab"[i % 2])
    p = store.get_point("h4.pt")
    fit2 = svc._hosted_fit(p)
    assert fit2 is not None
    # the 4 first-epoch fit rows + 5 first-epoch audits are old-epoch rows,
    # not late labels of the new fit.
    assert svc._late_rows(p, fit2) == []
    rep = svc.do_report({"point_id": "h4.pt"})["points"][0]
    assert rep["hosted"]["late_labels_ignored"]["n"] == 0


def _auto_row(did, hosted_top, probe_top=None):
    row = {"decision_id": did, "point_id": "h4.pt", "point_version": 1,
           "ts": S._utcnow(), "context": {"label_hint": "a"},
           "options": ["a", "b"], "allowed_options": ["a", "b"],
           "hosted": {"model": "m",
                      "probabilities": {"a": 0.85, "b": 0.15}
                      if hosted_top == "a" else {"a": 0.15, "b": 0.85},
                      "cost_usd": 0, "seconds": 0.1},
           "decider": "hosted_jev", "mode": "auto", "audit": False,
           "source": "do_decide", "phase": "serve"}
    if probe_top is not None:
        row["probe"] = {"model_version": "fp-1",
                        "probabilities": {"a": 0.9, "b": 0.1}
                        if probe_top == "a" else {"a": 0.1, "b": 0.9}}
        row["jev_zero_shot"] = {"a": 0.9, "b": 0.1} \
            if probe_top == "a" else {"a": 0.1, "b": 0.9}
    return row


def test_D_record_override_uses_hosted_top(tmp_path, clock):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    p = store.get_point("h4.pt")
    p["phase"] = "serve"
    store.put_point(p)
    # hosted top a, probe/zero-shot top b: following hosted is no override.
    did1, did2 = uuid.uuid4().hex, uuid.uuid4().hex
    store.insert_decision(_auto_row(did1, "a", probe_top="b"))
    store.insert_decision(_auto_row(did2, "a", probe_top="b"))
    assert svc.do_record({"decision_id": did1, "choice": "a"})["override"] \
        is False
    assert svc.do_record({"decision_id": did2, "choice": "b"})["override"] \
        is True
    # pure hosted row (no probe/zero-shot), hosted top b: the legacy
    # comparison against {} would call following b an override.
    did3 = uuid.uuid4().hex
    store.insert_decision(_auto_row(did3, "b"))
    assert svc.do_record({"decision_id": did3, "choice": "b"})["override"] \
        is False


def test_R211_fit_bad_followed_counts_as_errors(tmp_path, clock):
    """R2-11 (A3 fit): bad_followed rows count as errors for the threshold
    and the 5-error temperature gate, excluded from per-option recall."""
    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=10)
    cfg["serve"]["min_auto_support"] = 5
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    dids = [decide_label(svc, "ab"[i % 2]) for i in range(5)]
    for d in dids:
        assert svc.do_outcome({"decision_id": d, "ok": False})["ok"]
    for i in range(5, 10):
        decide_label(svc, "ab"[i % 2])
    fit = svc._hosted_fit(store.get_point("h4.pt"))
    assert fit is not None
    # 5 bad (errors) + 5 good (correct) -> agreement 0.5, not 1.0.
    assert abs(float(fit["fit_agreement"]) - 0.5) < 1e-9
    # The clean set is perfect, so temperature still skips (nothing to fit
    # on); the bad rows count for agreement/threshold, not NLL.
    assert fit["temperature_skipped"] == "too_few_errors"
    # Per-option excludes the bad rows (2 good a, 3 good b in the tail).
    per = fit["per_option"]
    assert per["a"]["n"] == 2 and per["b"]["n"] == 3


def test_R211_A4_two_connections_leave_one_fit(tmp_path, clock):
    """R2-11 (A4): two Store/Service handles (two processes) racing the fit
    trigger leave exactly one active fit row."""
    import threading as _th

    svc, store, cfg = make_custom(
        tmp_path, _deterministic_fn(lambda i, h, o: (h, 0.9)), fit_rows=10)
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h4.pt", "phase": "shadow"})["ok"]
    dids = [svc.do_decide({"point_id": "h4.pt",
                           "context": {"label_hint": "ab"[i % 2]}})[
        "decision_id"] for i in range(11)]
    for d in dids[:9]:
        svc.do_record({"decision_id": d, "choice": "a"})
    # Second connection on the same DB.
    from decideonce.deciders import build_deciders as _bd

    store2 = Store(tmp_path / "proj")
    dec2, _ = _bd(cfg, store2)
    import random as _rnd

    svc2 = Service(cfg, store2, dec2, rng=_rnd.Random(1))
    barrier = _th.Barrier(2)
    orig1, orig2 = svc._fit_candidates, svc2._fit_candidates

    def slow1(p):
        out = orig1(p)
        try:
            barrier.wait(timeout=5)
        except _th.BrokenBarrierError:
            pass
        return out

    def slow2(p):
        out = orig2(p)
        try:
            barrier.wait(timeout=5)
        except _th.BrokenBarrierError:
            pass
        return out

    svc._fit_candidates = slow1
    svc2._fit_candidates = slow2
    t1 = _th.Thread(target=svc.do_record,
                    args=({"decision_id": dids[9], "choice": "a"},))
    t2 = _th.Thread(target=svc2.do_record,
                    args=({"decision_id": dids[10], "choice": "a"},))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    rows = store.cx.execute("SELECT id, active FROM hosted_fit").fetchall()
    assert len(rows) == 1 and rows[0]["active"] == 1
    store2.close()
