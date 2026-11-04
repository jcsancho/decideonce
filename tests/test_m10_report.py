"""M10 tests: the one-turn workflow report (§15.8)."""
import random

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
    for section, kv in over.items():
        cfg[section].update(kv)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg)), store, cfg


def point(**kw):
    p = {"point_id": "refund.action", "workflow": "refund_request",
         "question": "Which action?", "kind": "choice",
         "options": ["full_refund", "escalate_human"],
         "context_fields": {"amount_eur": {"type": "number"}},
         "required_fields": ["amount_eur"],
         "rules": [{"id": "big", "if": [["amount_eur", ">", 500]],
                    "only": ["escalate_human"]}],
         "data_class": "public"}
    p.update(kw)
    return p


INPUT = "Ticket T-1. Amount: 80 EUR."
EV0 = "Noted. I'll refund the full 80 EUR now."


def test_workflow_report_json_and_md(tmp_path):
    svc, store, _ = make_service(
        tmp_path, plan={"audit_rate": 0.0, "downgrade_audit_rate": 0.0,
                        "allow_pilot_force": True},
        extract={"audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    p = store.get_point("refund.action")
    p["plan_phase"] = "serve"
    store.put_point(p)
    assert svc.do_register({"type": "workflow", "workflow": "refund_request",
                            "downgrade_requires": ["refund.action"],
                            "check": "outcome", "run_model": "wonder",
                            "cheap_model": "spark"})["ok"]
    # extraction with one rejection + a pass-2 audit
    t = {"run_id": "rep-1", "workflow": "refund_request", "model": "wonder",
         "data_class": "public", "input": INPUT,
         "events": [{"i": 0, "type": "text", "content": EV0}],
         "final_output": "Done."}
    c1 = svc.do_extract({"transcript": t})
    good = {"point_id": "refund.action", "choice": "full_refund",
            "choice_quote": "I'll refund the full 80 EUR",
            "decision_pos": {"event": 0, "offset": EV0.find("I'll refund")},
            "context": {"amount_eur": 80},
            "evidence": {"amount_eur": {
                "pos": {"event": -1, "offset": INPUT.find("Amount: 80 EUR")},
                "quote": "Amount: 80 EUR"}}}
    bad = dict(good, choice="maybe")
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [good, bad]})
    assert len(r["accepted"]) == 1 and len(r["rejected"]) == 1
    c2 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2})
    r2 = svc.do_extract({"extraction_id": c2["extraction_id"],
                         "extractor": "spark", "records": [good]})
    assert r2["rejected"] == []
    # an injected plan with a followed run, and a pilot plan
    out = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                           "context": {"refund.action": {"amount_eur": 600}},
                           "run_model": "wonder", "cheap_model": "spark"})
    assert out["recommendation"] == "inject"
    res = svc.do_plan_result({"plan_id": out["plan_id"],
                              "transcript": dict(t, run_id="rep-run-1"),
                              "cost": {"usd": 0.5}})
    cc = svc.do_extract({"transcript_id": res["transcript_id"]})
    rr = svc.do_extract({"extraction_id": cc["extraction_id"],
                         "extractor": "spark", "records": [dict(
                             good, choice="escalate_human")]})
    assert rr["followed_rate"] == 1.0
    pf = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                          "context": {"refund.action": {"amount_eur": 600}},
                          "run_model": "wonder", "cheap_model": "spark",
                          "pilot_force": True, "pilot_mode": "downgrade"})
    assert pf["recommendation"] == "downgrade"
    rep = svc.do_report({"workflow": "refund_request"})["workflow"]
    assert rep["extraction"]["transcripts"] == 2
    assert rep["extraction"]["pass1_completed"] == 2
    assert rep["extraction"]["pass2_completed"] == 1
    assert rep["extraction"]["rejected_by_code"] == {
        "choice_not_in_options": 1}
    assert rep["extraction"]["agreement"]["refund.action"]["audited"] == 1
    assert rep["plans"]["recommendations"] == {"inject": 1}
    assert rep["plans"]["followed_rate_mean"] == 1.0
    assert rep["plans"]["pilot_forced"] == 1
    assert rep["plans"]["pilot_note"] == "pilot, not promoted"
    assert rep["trial"] is None
    assert rep["monitor"]["status"] == "insufficient_data"
    assert rep["points"]["refund.action"]["plan_phase"] == "serve"
    md = svc.do_report({"workflow": "refund_request", "format": "md"})["report"]
    assert "# one-turn report: refund_request" in md
    assert "pilot, not promoted" in md
    # unknown workflow, and point+workflow exclusivity
    assert svc.do_report({"workflow": "nope"})["error"] == "unknown_workflow"
    assert svc.do_report({"point_id": "refund.action",
                          "workflow": "refund_request"})["error"] == "invalid_args"


def test_workflow_report_pairs_costs_monitor(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_register({"type": "workflow", "workflow": "refund_request",
                            "downgrade_requires": ["refund.action"],
                            "check": "outcome", "run_model": "wonder",
                            "cheap_model": "spark"})["ok"]
    for i in range(35):  # monitor pairs, all expensive-pass / cheap-fail
        for arm, passed, usd in (("expensive", True, 0.5),
                                 ("cheap", False, 0.1)):
            store.put_plan({
                "plan_id": f"mp{i}-{arm[0]}", "ts": "2026-01-04T00:00:00Z",
                "workflow": "refund_request", "recommendation": "downgrade",
                "model": "spark", "points": [], "audit": False,
                "pair_id": f"mp{i}", "arm": arm, "pilot_forced": False,
                "conflicts": [], "point_versions": {"refund.action": 1},
                "cost": {"usd": usd},
                "check_result": {"pass": passed, "source": "user"}})
    store.put_plan({  # one downgraded single task
        "plan_id": "single-d", "ts": "2026-01-05T00:00:00Z",
        "workflow": "refund_request", "recommendation": "downgrade",
        "model": "spark", "points": [], "audit": False,
        "pilot_forced": False, "conflicts": [],
        "point_versions": {"refund.action": 1}})
    rep = svc.do_report({"workflow": "refund_request"})["workflow"]
    assert rep["monitor"]["status"] == "revoke"
    assert rep["monitor"]["pairs"] == 35
    assert rep["costs"]["monitor_arms"] == {
        "expensive": {"n": 35, "mean_usd": 0.5},
        "cheap": {"n": 35, "mean_usd": 0.1}}
    assert rep["costs"]["overhead_usd"] == 35 * 0.6
    assert rep["costs"]["downgraded_tasks"] == 1
    assert rep["costs"]["savings_usd"] == (0.5 - 0.1) - 35 * 0.6


def test_workflow_report_monitor_flag_on_full_window(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_register({"type": "workflow", "workflow": "refund_request",
                            "downgrade_requires": ["refund.action"],
                            "check": "outcome", "run_model": "wonder",
                            "cheap_model": "spark"})["ok"]
    # 120 pairs, 2 discordant against cheap: lower < -0.05, upper above it
    specs = [(True, True)] * 118 + [(True, False)] * 2
    for i, (e, c) in enumerate(specs):
        for arm, passed in (("expensive", e), ("cheap", c)):
            store.put_plan({
                "plan_id": f"fp{i}-{arm[0]}", "ts": "2026-01-04T00:00:00Z",
                "workflow": "refund_request", "recommendation": "downgrade",
                "model": "spark", "points": [], "audit": False,
                "pair_id": f"fp{i}", "arm": arm, "pilot_forced": False,
                "conflicts": [], "point_versions": {"refund.action": 1},
                "check_result": {"pass": passed, "source": "user"}})
    rep = svc.do_report({"workflow": "refund_request"})["workflow"]
    assert rep["monitor"]["pairs"] == 120
    assert rep["monitor"]["status"] == "non_inferiority_not_shown"
