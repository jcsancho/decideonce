"""M6 tests: report (T-Rep*), export (T-X*), Decider client."""
import json
import random
import uuid

from decideonce.client import Decider
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


def point(pid="p.rep", **kw):
    p = {"point_id": pid, "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"note": {"type": "string"}},
         "required_fields": [], "rules": [], "data_class": "private"}
    p.update(kw)
    return p


def test_T_Rep1_private_report_has_no_raw_context(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    secret = "SECRETVALUE42"
    for i in range(5):
        r = svc.do_decide({"point_id": "p.rep", "context": {"note": f"{secret}-{i}"}})
        svc.do_record({"decision_id": r["decision_id"], "choice": "ab"[i % 2]})
    rep = svc.do_report({"point_id": "p.rep"})
    assert rep["ok"]
    assert secret not in json.dumps(rep)
    md = svc.do_report({"format": "md"})
    assert md["ok"] and secret not in md["report"] and "p.rep" in md["report"]


def test_T_Rep2_savings_estimate_below_20_costs(tmp_path):
    svc, store = make_service(tmp_path)
    assert svc.do_register(point(data_class="public"))["ok"]
    for i in range(5):
        r = svc.do_decide({"point_id": "p.rep",
                           "context": {"note": "x"},
                           "teacher_cost": {"tokens_in": 100, "tokens_out": 10,
                                            "usd": 0.5, "seconds": 2.0}})
        svc.do_record({"decision_id": r["decision_id"], "choice": "a"})
    for i in range(4):  # auto rows served by the decider
        store.insert_decision({"decision_id": uuid.uuid4().hex, "point_id": "p.rep",
                               "point_version": 1, "ts": "2026-01-01T00:00:00Z",
                               "options": ["a", "b"], "allowed_options": ["a", "b"],
                               "mode": "auto", "source": "do_decide", "final_choice": "a"})
    rep = svc.do_report({"point_id": "p.rep"})["points"][0]
    assert rep["saved"]["basis"] == "estimate"
    assert rep["saved"]["usd"] == 0.5 * 4
    assert rep["saved"]["tokens_in"] == 400
    for i in range(20):  # reach 25 reported costs -> measured
        svc.do_decide({"point_id": "p.rep", "context": {"note": "y"},
                       "teacher_cost": {"usd": 0.5}})
    rep2 = svc.do_report({"point_id": "p.rep"})["points"][0]
    assert rep2["saved"]["basis"] == "measured"


def test_report_agreement_and_promotion_blockers(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(point(data_class="public"))["ok"]
    for i in range(10):
        r = svc.do_decide({"point_id": "p.rep", "context": {"note": "x"}})
        svc.do_record({"decision_id": r["decision_id"], "choice": "a"})
    rep = svc.do_report({})["points"][0]
    assert rep["rows"] == 10
    assert rep["zero_shot"]["n"] == 0  # tiered collect calls no model
    assert set(rep["zero_shot"]["per_option"]) == {"a", "b"}
    assert rep["probe"] is None
    assert rep["promotion"]["serve_ready"] is False
    assert "min_examples" in [k for k, v in rep["promotion"]["checks"].items()
                              if not v["pass"]]


def test_T_X1_export_refuses_private_without_confirm(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_register(point("p.pub", data_class="public"))["ok"]
    r = svc.do_export({"point_id": "p.rep", "path": str(tmp_path / "x.jsonl")})
    assert r["ok"] is False and r["error"] == "confirm_required"
    assert svc.do_export({"point_id": "p.rep", "path": str(tmp_path / "x.jsonl"),
                          "confirm": True})["ok"]
    assert svc.do_export({"point_id": "p.pub", "path": str(tmp_path / "y.jsonl")})["ok"]
    assert svc.do_export({"point_id": "nope", "path": "z"})["error"] == "unknown_point"


def test_T_X2_export_jsonl_shape(tmp_path):
    svc, _ = make_service(tmp_path)
    assert svc.do_register(point(data_class="public"))["ok"]
    for i in range(3):
        r = svc.do_decide({"point_id": "p.rep", "context": {"note": f"n{i}"}})
        svc.do_record({"decision_id": r["decision_id"], "choice": "ab"[i % 2]})
    out = tmp_path / "sub" / "exp.jsonl"
    r = svc.do_export({"point_id": "p.rep", "path": str(out)})
    assert r["ok"] and r["n"] == 3
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert [sorted(row) for row in rows] == [
        ["label", "label_source", "options", "state_text", "weight"]] * 3
    assert [row["label"] for row in rows] == ["a", "b", "a"]
    assert all(row["label_source"] == "teacher" and row["weight"] == 1.0 for row in rows)


def test_decider_client_round_trip(tmp_path):
    cfg_path = tmp_path / "c.toml"
    cfg_path.write_text(f'[store]\ndata_dir = "{tmp_path}/data"\nproject = "t"\n')
    d = Decider(config_path=str(cfg_path))
    assert d.register(point("p.c", data_class="public"))["ok"]
    r = d.decide("p.c", {"note": "hello"})
    assert r.mode == "teacher" and r.choice is None and r.decision_id
    assert d.record(r.decision_id, "a")["ok"]
    assert d.outcome(r.decision_id, ok=True)["ok"]
    assert d.report("p.c")["ok"]
    d.close()
