"""M4 tests: serve mode, audits, drift, promotion (T-P*)."""
import random
import uuid

import pytest

from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store
from decideonce.trainer import FakeTrainer


def make_service(tmp_path, **cfg_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    for section, vals in cfg_over.items():
        cfg[section].update(vals)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg)), store


# Small counts pin the same gates the defaults pin at 200 rows.
SMALL = {"train": {"min_train": 5, "min_examples": 10},
         "serve": {"min_shadow": 12, "min_auto_support": 3},
         "hosted_jev": {"fit_rows": 20, "min_compare": 10}}


def hint_point(pid="p.serve", options=("a", "b"), target_error=0.3, **kw):
    p = {"point_id": pid, "workflow": "w", "question": "Which?",
         "kind": "choice", "options": list(options),
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "rules": [], "data_class": "public",
         "target_error": target_error}
    p.update(kw)
    return p


def add_row(svc, pid, label, hint=None):
    ctx = {"label_hint": label if hint is None else hint}
    r = svc.do_decide({"point_id": pid, "context": ctx})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec
    return r


def flip_phase(store, pid, phase):
    point = store.get_point(pid)
    point["phase"] = phase
    store.put_point(point)


def _train_private(svc, pid):
    """Smoke then full FakeTrainer job to done; the model card is saved."""
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "smoke", r
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "smoke_done", r
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full", r
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "done", r


def _dual_ready(tmp_path, over=None):
    """Hosted fit + FakeTrainer private model; the export marker sits at
    the last fit row, so directly inserted C rows land after it."""
    svc, store = make_service(tmp_path, **(over or SMALL))
    assert svc.do_register(hint_point())["ok"]
    assert svc.do_promote({"point_id": "p.serve", "phase": "shadow"})["ok"]
    for i in range(20):
        add_row(svc, "p.serve", "ab"[i % 2])
    fit = store.active_hosted_fit("p.serve")
    assert fit is not None and fit["threshold"] == 0.9, fit
    _train_private(svc, "p.serve")
    return svc, store


def _c_row(store, pid, i, hosted_top, hosted_conf, private_top,
           private_conf, label, phase="shadow"):
    """One comparison row with crafted stored probabilities, then labeled."""
    other_h = "b" if hosted_top == "a" else "a"
    other_p = "b" if private_top == "a" else "a"
    did = uuid.uuid4().hex
    store.insert_decision({
        "decision_id": did, "point_id": pid, "point_version": 1,
        "ts": f"2027-01-01T00:{i:02d}:00Z",
        "context": {"label_hint": label}, "options": ["a", "b"],
        "allowed_options": ["a", "b"],
        "hosted": {"model": "m",
                   "probabilities": {hosted_top: hosted_conf,
                                     other_h: round(1 - hosted_conf, 2)},
                   "cost_usd": 0.0, "seconds": 0.01},
        "probe": {"model_version": "fake-private",
                  "probabilities": {private_top: private_conf,
                                    other_p: round(1 - private_conf, 2)}},
        "mode": "teacher", "source": "do_decide", "phase": phase},
        100000)
    return did


def _perfect_c(svc, store, pid, n, offset=0):
    """n C rows both models get right (hosted below the fit threshold)."""
    for i in range(n):
        label = "ab"[(offset + i) % 2]
        did = _c_row(store, pid, offset + i, label, 0.6, label, 0.95, label)
        assert svc.do_record({"decision_id": did, "choice": label})["ok"]


def test_T_P1_serve_refused_without_confirm(tmp_path):
    svc, _ = _dual_ready(tmp_path)
    _perfect_c(svc, svc.store, "p.serve", 12)
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private"})
    assert r["ok"] is False
    assert r["failed"] == ["confirm"]
    yes = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                          "tier": "private", "confirm": True})
    assert yes["ok"], yes
    assert yes["phase"] == "serve"


def _hosted_serve_ready(tmp_path, n_evidence=12, **cfg_over):
    from decideonce.deciders.hosted import FakeHostedTransport

    merged = {k: dict(v) for k, v in SMALL.items()}
    for section, vals in cfg_over.items():
        merged.setdefault(section, {}).update(vals)
    svc, store = make_service(tmp_path, **merged)
    # Fit rows at 0.7 so the fit threshold sits well below serve-time 0.9
    # (tempering float dust would break a 0.9 boundary).
    calls = {"n": 0}

    def sched(body):
        (pid, q), = body["questions"].items()
        opts = list(q["criteria"])
        hint = (body["state"] or {}).get("label_hint")
        top = hint if hint in opts else opts[0]
        calls["n"] += 1
        conf = 0.7 if calls["n"] <= 20 else 0.9
        rest = (1.0 - conf) / (len(opts) - 1)
        probs = {o: (conf if o == top else rest) for o in opts}
        return 200, {"answers": {pid: {"choice": top,
                                       "probabilities": probs,
                                       "confidence": conf}},
                     "usage": {"input_tokens": 50, "output_tokens": 0,
                               "cost": 0.0001}}

    svc.deciders["hosted_jev"].transport = FakeHostedTransport(fn=sched)
    assert svc.do_register(hint_point())["ok"]
    assert svc.do_promote({"point_id": "p.serve", "phase": "shadow"})["ok"]
    for i in range(20):
        add_row(svc, "p.serve", "ab"[i % 2])
    assert store.active_hosted_fit("p.serve")["threshold"] == 0.7
    for i in range(n_evidence):
        add_row(svc, "p.serve", "ab"[i % 2])
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "confirm": True})
    assert r["ok"], r
    return svc, store


def test_serve_mode_auto_and_audits(tmp_path):
    svc, store = _hosted_serve_ready(tmp_path)
    autos = audits = 0
    for i in range(200):
        label = "ab"[i % 2]
        r = svc.do_decide({"point_id": "p.serve",
                           "context": {"label_hint": label}})
        assert r["ok"]
        if r["mode"] == "auto":
            autos += 1
            assert r["choice"] == label
        else:
            assert r["mode"] == "teacher"
            audits += 1
            row = store.get_decision(r["decision_id"])
            assert row["audit"] is True
    assert 10 <= audits <= 30, audits  # ~10% of 200
    assert autos + audits == 200


def test_T_P2_criterion_min_examples(tmp_path):
    svc, _ = make_service(tmp_path)  # defaults: min_train 60, min_examples 200
    assert svc.do_register(hint_point())["ok"]
    assert svc.do_promote({"point_id": "p.serve", "phase": "shadow"})["ok"]
    for i in range(60):
        add_row(svc, "p.serve", "ab"[i % 2])
    _train_private(svc, "p.serve")
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert "min_examples" in r["failed"]


def test_T_P2_criterion_threshold(tmp_path):
    svc, _ = make_service(tmp_path, **SMALL)
    assert svc.do_register(hint_point())["ok"]
    assert svc.do_promote({"point_id": "p.serve", "phase": "shadow"})["ok"]
    for i in range(10):
        add_row(svc, "p.serve", "ab"[i % 2])
    # no private model trained: the private threshold is unknown
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert "threshold" in r["failed"]


def test_T_P2_criterion_evidence(tmp_path):
    svc, store = _dual_ready(tmp_path)
    for i in range(12):  # 3 private errors at high confidence
        label = "ab"[i % 2]
        ptop = ("b" if label == "a" else "a") if i < 3 else label
        did = _c_row(store, "p.serve", i, label, 0.6, ptop, 0.9, label)
        assert svc.do_record({"decision_id": did, "choice": label})["ok"]
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert "evidence" in r["failed"]  # Wilson upper(3, 12) above 0.3


def test_T_P2_criterion_margin(tmp_path):
    svc, store = _dual_ready(tmp_path)
    _perfect_c(svc, store, "p.serve", 12)
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    # H1: a perfect baseline already meets the target, so the margin check
    # is skipped instead of refusing an error-free model.
    assert "margin" not in r.get("failed", [])
    point = store.get_point("p.serve")
    detail = svc._serve_checks_private(point, True)["margin"]["detail"]
    assert "skipped" in detail


def test_H1_error_free_model_reaches_serve(tmp_path):
    svc, store = _dual_ready(tmp_path)
    _perfect_c(svc, store, "p.serve", 12)
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"], r
    assert r["phase"] == "serve"


def test_T_P2_criterion_shadow(tmp_path):
    svc, store = _dual_ready(tmp_path)
    _perfect_c(svc, store, "p.serve", 5)
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert "shadow" in r["failed"] and "shadow_agreement" in r["failed"]


def test_T_P2_criterion_shadow_agreement(tmp_path):
    svc, store = _dual_ready(tmp_path)
    for i in range(24):  # second half contradicts the private model
        label = "ab"[i % 2]
        ptop = label if i < 12 else ("b" if label == "a" else "a")
        did = _c_row(store, "p.serve", i, label, 0.6, ptop, 0.9, label)
        assert svc.do_record({"decision_id": did, "choice": label})["ok"]
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert "shadow" not in r.get("failed", [])
    assert "shadow_agreement" in r["failed"]


def test_T_P2_criterion_teacher_unreliable(tmp_path):
    svc, _ = make_service(tmp_path, **SMALL)
    assert svc.do_register(hint_point())["ok"]
    for i in range(10):
        did = add_row(svc, "p.serve", "ab"[i % 2])["decision_id"]
        svc.do_outcome({"decision_id": did, "ok": False})
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert "teacher" in r["failed"]


def test_T_P2_criterion_confirm(tmp_path):
    svc, _ = make_service(tmp_path, **SMALL)
    assert svc.do_register(hint_point())["ok"]
    r = svc.do_promote({"point_id": "p.serve", "phase": "serve",
                        "tier": "private"})
    assert "confirm" in r["failed"]


def test_promote_shadow_needs_private_model(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(hint_point(private_model_only=True))["ok"]
    r = svc.do_promote({"point_id": "p.serve", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "private_model_not_trained"
    assert svc.do_promote({"point_id": "p.serve", "phase": "paused"})["ok"]
    assert svc.do_promote({"point_id": "p.serve", "phase": "collect"})["ok"]
    assert svc.do_promote({"point_id": "nope",
                           "phase": "shadow"})["error"] == "unknown_point"


def test_T_P3_audit_rate_over_seeded_draws(tmp_path):
    svc, _ = make_service(tmp_path)
    hits = sum(svc._draw_audit() for _ in range(2000))
    assert 150 <= hits <= 250, hits


@pytest.mark.parametrize("errors,demoted", [(0, False), (2, False), (5, True), (6, True)])
def test_T_P4_drift_demotes_on_audit_evidence(tmp_path, errors, demoted):
    svc, store = make_service(tmp_path)
    assert svc.do_register(hint_point(private_model_only=True,
                                      target_error=0.02))["ok"]
    flip_phase(store, "p.serve", "serve")
    for i in range(100):
        wrong = i < errors
        store.insert_decision({
            "decision_id": uuid.uuid4().hex, "point_id": "p.serve", "point_version": 1,
            "ts": f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}Z",
            "context": {"label_hint": "a"}, "options": ["a", "b"],
            "allowed_options": ["a", "b"], "rule_hits": [],
            "probe": {"model_version": "fake-private",
                      "probabilities": {"a": 0.1, "b": 0.9} if wrong else {"a": 0.9, "b": 0.1}},
            "mode": "teacher", "audit": True, "source": "do_decide", "phase": "serve"},
            100000)
    for row in store.decisions_for_point("p.serve"):
        rec = svc.do_record({"decision_id": row["decision_id"], "choice": "a"})
        assert rec["ok"]
    assert (store.get_point("p.serve")["phase"] == "shadow") == demoted
    if demoted:
        assert store.get_point("p.serve")["events"][-1]["event"] == "drift_demotion"


def test_T_P4_no_demotion_below_window(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(hint_point(private_model_only=True,
                                      target_error=0.02))["ok"]
    flip_phase(store, "p.serve", "serve")
    for i in range(99):  # all wrong, but fewer than drift_window audits
        store.insert_decision({
            "decision_id": uuid.uuid4().hex, "point_id": "p.serve", "point_version": 1,
            "ts": "2026-01-01T00:00:00Z", "context": {}, "options": ["a", "b"],
            "allowed_options": ["a", "b"],
            "probe": {"model_version": "fake-private", "probabilities": {"a": 0.1, "b": 0.9}},
            "mode": "teacher", "audit": True, "source": "do_decide", "phase": "serve"},
            100000)
        svc.do_record({"decision_id": store.decisions_for_point("p.serve")[-1]["decision_id"],
                       "choice": "a"})
    assert store.get_point("p.serve")["phase"] == "serve"


def test_T_P5_irreversible_never_auto_by_default(tmp_path):
    svc, store = _hosted_serve_ready(tmp_path)
    point = store.get_point("p.serve")
    point["irreversible"] = True
    store.put_point(point)
    r = svc.do_decide({"point_id": "p.serve", "context": {"label_hint": "a"}})
    assert r["mode"] == "teacher"  # blocked despite confidence
    svc.cfg["serve"]["allow_auto_irreversible"] = True
    r2 = svc.do_decide({"point_id": "p.serve", "context": {"label_hint": "a"}})
    assert r2["mode"] in ("auto", "teacher")  # teacher only if drawn for audit
    if r2["mode"] == "teacher":
        assert store.get_decision(r2["decision_id"])["audit"] is True


def test_T_P6_override_recorded(tmp_path):
    svc, store = _hosted_serve_ready(tmp_path, serve={"audit_rate": 0.0})
    r = svc.do_decide({"point_id": "p.serve", "context": {"label_hint": "a"}})
    assert r["mode"] == "auto" and r["choice"] == "a"
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": "b"})
    assert rec["ok"] and rec["override"] is True
    row = store.get_decision(r["decision_id"])
    assert row["override"] is True and row["final_choice"] == "b"
