"""M3 tests: training (T-T*)."""
import random

from decideonce import labels as L
from decideonce import stats as S
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store
from decideonce.trainer import FakeTrainer, split_rows


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


def hint_point(pid="p.cls", options=("a", "b", "c"), **kw):
    p = {"point_id": pid, "workflow": "w", "question": "Which?",
         "kind": "choice", "options": list(options),
         "context_fields": {"label_hint": {"type": "string"},
                            "note": {"type": "string"}},
         "required_fields": [], "rules": [], "data_class": "public"}
    p.update(kw)
    return p


def add_row(svc, pid, label, hint=True, **extra):
    ctx = dict(extra)
    if hint:
        ctx["label_hint"] = label
    r = svc.do_decide({"point_id": pid, "context": ctx})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec
    return r["decision_id"]


def _train_done(svc, pid):
    """Smoke then full FakeTrainer job to done; returns the saved card."""
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
    return svc.store.active_model(pid, "private")["card"]


def test_T_T1_not_enough_data(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(hint_point())["ok"]
    for i in range(10):
        add_row(svc, "p.cls", "abc"[i % 3])
    r = svc.do_train({"point_id": "p.cls", "force": True, "confirm": True})
    assert r["ok"] is False and r["error"] == "not_enough_data"
    assert "need 60" in r["message"] and "have 0" in r["message"]
    # 60 rows launches even with a single rare option: the per-option
    # minimum was a probe-era gate; the trainer sees all rows.
    svc2, _ = make_service(tmp_path / "s2")
    assert svc2.do_register(hint_point())["ok"]
    for _ in range(56):
        add_row(svc2, "p.cls", "a")
    for _ in range(4):
        add_row(svc2, "p.cls", "b")
    r2 = svc2.do_train({"point_id": "p.cls", "force": True, "confirm": True})
    assert r2["ok"] and r2["status"] == "running", r2


def test_T_T2_private_model_learns_planted_signal(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(hint_point())["ok"]
    for i in range(90):
        add_row(svc, "p.cls", "abc"[i % 3])
    card = _train_done(svc, "p.cls")
    assert card["accuracy"] >= 0.9
    assert card["hint_map"] == {"a": "a", "b": "b", "c": "c"}
    # dry run uploads nothing
    before = len(svc.trainer.uploaded)
    d = svc.do_train({"point_id": "p.cls", "force": True, "dry_run": True})
    assert d["ok"] and len(svc.trainer.uploaded) == before


def test_T_T3_label_precedence():
    assert L.resolve_label({"outcome": {"correct_choice": "b", "ok": False},
                            "teacher_choice": "a"}) == ("b", "outcome_correct", 1.0)
    assert L.resolve_label({"outcome": {"ok": True}, "final_choice": "a",
                            "teacher_choice": "a"}) == ("a", "outcome_ok", 1.0)
    label, source, _ = L.resolve_label({"outcome": {"ok": False}, "final_choice": "a",
                                        "teacher_choice": "a"})
    assert label is None and source == "outcome_bad_no_fix"
    assert L.resolve_label({"teacher_choice": "a"}, teacher_weight=0.7) == ("a", "teacher", 0.7)
    assert L.resolve_label({})[0] is None


def test_T_T3_label_sources_in_upload(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(hint_point())["ok"]
    for i in range(70):
        did = add_row(svc, "p.cls", "ab"[i % 2])
        if i < 10:
            svc.do_outcome({"decision_id": did, "ok": True})
    labs = L.labeled_rows(store.decisions_for_point("p.cls"),
                          store.get_point("p.cls"), 1.0, 0.5)
    srcs: dict[str, int] = {}
    for _, _, src, _ in labs:
        srcs[src] = srcs.get(src, 0) + 1
    assert srcs == {"teacher": 60, "outcome_ok": 10}
    r = svc.do_train({"point_id": "p.cls", "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running", r
    assert len(svc.trainer.uploaded[-1]["rows"]) == 70


def test_T_T4_time_ordered_split_no_future_in_train():
    rows = [{"point_id": "p.cls", "seq": i,
             "ts": f"2026-01-01T00:00:{i:02d}Z"} for i in range(60)]
    out = split_rows(rows, 0.25, 0.15)
    by: dict[str, list[int]] = {}
    for r in out:
        by.setdefault(r["split"], []).append(r["seq"])
    assert by["train"] == list(range(36))
    assert by["calib"] == list(range(36, 45))
    assert by["heldout"] == list(range(45, 60))


def test_T_T5_threshold_point_estimate_and_wilson_bound(tmp_path):
    assert S.wilson_upper(0, 188) > 0.02
    assert S.wilson_upper(0, 189) <= 0.02
    svc, _ = make_service(tmp_path)
    assert svc.do_register(hint_point())["ok"]
    for i in range(60):
        add_row(svc, "p.cls", "abc"[i % 3])
    card = _train_done(svc, "p.cls")
    assert card["threshold"] == 0.5  # the advertised serve cutoff


def test_T_T7_rare_option_low_support_is_teacher_only(tmp_path):
    svc, store = make_service(tmp_path, hosted_jev={"fit_rows": 20})
    assert svc.do_register(hint_point())["ok"]
    assert svc.do_promote({"point_id": "p.cls", "phase": "shadow"})["ok"]
    for _ in range(19):
        add_row(svc, "p.cls", "a")
    add_row(svc, "p.cls", "b")  # 1 example < 10 -> teacher-only
    fit = store.active_hosted_fit("p.cls")
    assert "b" in fit["teacher_only_options"]
    assert "a" not in fit["teacher_only_options"]


def test_T_T8_masked_softmax_renormalises(tmp_path):
    assert S.mask_and_renorm({"a": 0.5, "b": 0.3, "c": 0.2}, ["a", "c"]) == {
        "a": 0.5 / 0.7, "c": 0.2 / 0.7}
    svc, store = make_service(tmp_path, hosted_jev={"fit_rows": 20})
    p = hint_point(options=["a", "b"])
    p["rules"] = [{"id": "no-b", "if": [["note", "==", "nob"]], "exclude": ["b"]}]
    assert svc.do_register(p)["ok"]
    assert svc.do_promote({"point_id": "p.cls", "phase": "shadow"})["ok"]
    for i in range(60):
        add_row(svc, "p.cls", "ab"[i % 2])
    assert store.active_hosted_fit("p.cls") is not None
    r = svc.do_decide({"point_id": "p.cls",
                       "context": {"label_hint": "b", "note": "nob"}})
    assert r["ok"] and r["allowed_options"] == ["a"]
    assert r["mode"] == "rule"  # one option left; the model masks the same way
    r2 = svc.do_decide({"point_id": "p.cls", "context": {"label_hint": "b"}})
    row = store.get_decision(r2["decision_id"])
    assert set(row["hosted"]["probabilities"]) == {"a", "b"}  # full logged
    assert abs(sum(row["hosted"]["probabilities"].values()) - 1.0) < 1e-9
