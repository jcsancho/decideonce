"""M12 P4 tests: pre-M12 data stays loadable (T-M12-1)."""
import random
import uuid

from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store
from decideonce.trainer import FakeTrainer


def make_service(tmp_path, **over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    for section, kv in over.items():
        cfg[section].update(kv)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg)), store, cfg


def legacy_point(**kw):
    p = {"point_id": "old.pt", "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"x": {"type": "string"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def _legacy_row(store, pid, i, label):
    """A pre-M12 decision row: probe + zero-shot + seconds, no tiered fields."""
    other = "b" if label == "a" else "a"
    store.insert_decision({
        "decision_id": uuid.uuid4().hex, "point_id": pid,
        "point_version": 1, "ts": f"2026-10-01T00:{i:02d}:00Z",
        "context": {"x": f"s{i}"}, "state_text": f"state {i}",
        "options": ["a", "b"], "allowed_options": ["a", "b"],
        "mode": "teacher", "source": "do_decide", "phase": "shadow",
        "teacher_choice": label,
        "jev_zero_shot": {label: 0.9, other: 0.1},
        "probe": {"model_version": "v1-3",
                  "probabilities": {label: 0.95, other: 0.05}},
        "jev_seconds": 0.5}, 100000)


def _legacy_db(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(legacy_point())["ok"]
    for i in range(25):
        _legacy_row(store, "old.pt", i, "ab"[i % 2])
    store.save_model("old.pt", 1, "probe", "v1-3.npz",
                     {"threshold": 0.9, "coverage": 0.99,
                      "model_version": "v1-3", "n_at_threshold": 25,
                      "error_at_threshold": 0.0,
                      "teacher_only_options": []})
    store.save_model("old.pt", 1, "plan", "v1-1.plan.npz",
                     {"plan_threshold": 0.5, "plan_eligible": True})
    return svc, store


def test_T_M12_1_legacy_db_opens_and_reports(tmp_path):
    svc, store = _legacy_db(tmp_path)
    rep = svc.do_report({"point_id": "old.pt"})["points"][0]
    assert rep["rows"] == 25
    assert rep["legacy_rows"] == 25
    assert rep["threshold"] is None  # probe card ignored
    assert rep["coverage_heldout"] is None
    assert rep["probe"] is None
    assert rep["evidence_pool"]["n"] == 0
    md = svc.do_report({"point_id": "old.pt", "format": "md"})["report"]
    assert "- legacy: 25 pre-M12 rows (shown, never evidence)" in md
    # ignored, not deleted: the model rows are still in the store.
    assert store.active_model("old.pt", "probe")["card"]["threshold"] == 0.9
    assert store.active_model("old.pt", "plan")["card"]["plan_eligible"] is True


def test_T_M12_1_nothing_promoted_from_legacy(tmp_path):
    svc, store = _legacy_db(tmp_path)
    assert svc._maybe_fit_hosted("old.pt") is None  # no hosted rows
    assert svc.do_promote({"point_id": "old.pt",
                           "phase": "shadow"})["ok"]
    r = svc.do_promote({"point_id": "old.pt", "phase": "serve",
                        "confirm": True})
    assert r["ok"] is False
    assert "threshold" in r["failed"]  # legacy rows fit nothing


def test_T_M12_1_workflow_plan_cards_ignored(tmp_path):
    from decideonce import report as RP

    svc, store = _legacy_db(tmp_path)
    rep = RP.workflow_report(svc.cfg, store, "w")
    assert rep["points"]["old.pt"]["plan_eligible"] is None
