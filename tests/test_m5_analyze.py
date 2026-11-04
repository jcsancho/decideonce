"""M5 tests: do_analyze rubric, hints, trace detector (T-A*)."""
import json
import random

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
                   trainer=FakeTrainer(cfg))


def cand(**over):
    c = {"draft": {"question": "Which action?", "kind": "choice",
                   "options": ["a", "b"], "context_fields": {"f": {"type": "string"}}},
         "answers": {"R1": "yes", "R2": "yes", "R3": "rule", "R4": 500, "R5": "yes",
                     "R6": {"sensitive": False, "covered": True}, "R7": "no",
                     "R8": {"tokens": 2000, "separate_call": False}}}
    for k, v in over.items():
        c["answers"][k] = v
    return c


def test_T_A1_text_hints_found_with_line_numbers(tmp_path):
    svc = make_service(tmp_path)
    text = ("# Refund skill\n"
            "Read the ticket.\n"
            "Classify the ticket into a type.\n"
            "Pick one of: full refund, partial refund, replacement.\n"
            "Options are full | partial | replacement | escalate.\n"
            "If the amount is large then escalate to a human.\n"
            "- damaged item\n"
            "- late delivery\n"
            "- wrong item\n")
    r = svc.do_analyze({"workflow": {"kind": "skill", "text": text}})
    assert r["ok"]
    assert len(r["rubric"]) == 8
    assert "answers" in r["candidate_schema"]["properties"]
    pairs = {(h["line"], h["match"]) for h in r["hints"]}
    assert (3, "classify") in pairs
    assert (4, "one of") in pairs
    assert (5, "a | b | c") in pairs
    assert (6, "if … then") in pairs
    assert any(h["line"] == 7 and "bullet" in h["match"] for h in r["hints"])


def test_T_A2_rubric_gates(tmp_path):
    svc = make_service(tmp_path)
    r = svc.do_analyze({"workflow": {"kind": "prompt", "text": "decide things"},
                        "candidates": [cand(R1="no"), cand(R3="correctness")]})
    assert [p["verdict"] for p in r["points"]] == ["keep_llm", "keep_llm"]
    assert any("gate" in reason for p in r["points"] for reason in p["reasons"])
    r2 = svc.do_analyze({"workflow": {"kind": "prompt", "text": "x"},
                         "candidates": [cand(R2="no")]})
    assert r2["points"][0]["verdict"] == "keep_llm"


def test_T_A3_scoring_thresholds(tmp_path):
    svc = make_service(tmp_path)
    good = cand()  # 2+2+2+1+1 = 8
    mid = cand(R3="other", R4=100, R5="no", R8={"tokens": 100})  # 2+0+1 = 3
    low = cand(R3="other", R4=10, R5="no",
               R6={"sensitive": True, "covered": False},
               R8={"tokens": 10})  # 2-1 = 1
    capped = cand(R7="yes")  # 8 but irreversible
    r = svc.do_analyze({"workflow": {"kind": "prompt", "text": "x"},
                        "candidates": [good, mid, low, capped]})
    pts = r["points"]
    assert [(p["score"], p["verdict"]) for p in pts] == [
        (8, "good_fit"), (3, "possible"), (1, "keep_llm"), (8, "possible")]
    assert pts[3]["draft"]["irreversible"] is True
    assert pts[0]["reasons"]  # every scoring move explained


def _write_traces(path, calls):
    with open(path, "w") as f:
        for c in calls:
            f.write(json.dumps(c) + "\n")


def test_T_A4_trace_detector_enum_json_and_freetext(tmp_path):
    svc = make_service(tmp_path)
    calls = []
    for i in range(25):  # enum group
        calls.append({"call_id": f"e{i}", "workflow": "w", "step": "triage",
                      "prompt": f"Classify ticket {i}", "output": ["bug", "spam", "bill"][i % 3],
                      "usd": 0.01})
    for i in range(25):  # JSON enum-field group
        calls.append({"call_id": f"j{i}", "workflow": "w", "step": "pri",
                      "prompt": f"Prioritise {i}",
                      "output": json.dumps({"priority": ["high", "low"][i % 2],
                                            "team": "support"}),
                      "usd": 0.02})
    for i in range(25):  # free-text group
        calls.append({"call_id": f"f{i}", "workflow": "w", "step": "write",
                      "prompt": f"Write reply {i}",
                      "output": f"Dear customer number {i}, thank you kindly {i * 7}"})
    p = tmp_path / "traces.jsonl"
    _write_traces(p, calls)
    r = svc.do_analyze({"traces": {"path": str(p)}})
    assert r["ok"]
    assert len(r["points"]) == 2, r["points"]
    enum_pt = next(pt for pt in r["points"] if "triage" in pt["draft"]["question"])
    assert enum_pt["verdict"] == "possible" and enum_pt["score"] is None
    assert sorted(enum_pt["draft"]["options"]) == ["bill", "bug", "spam"]
    assert enum_pt["draft"]["context_fields"] == {}
    assert enum_pt["est_saving_usd"] is None  # no timestamps -> no monthly rate
    json_pt = next(pt for pt in r["points"] if "pri" in pt["draft"]["question"])
    assert json_pt["draft"]["json_field"] == "priority"
    assert sorted(json_pt["draft"]["options"]) == ["high", "low"]
    assert r["trace_stats"]["groups"] == 3


def test_T_A5_min_group_respected(tmp_path):
    svc = make_service(tmp_path)
    calls = [{"call_id": f"c{i}", "workflow": "w", "step": "s",
              "prompt": "Classify", "output": "yes"} for i in range(19)]
    calls += [{"call_id": f"d{i}", "workflow": "w", "step": "t",
               "prompt": "Classify", "output": "yes"} for i in range(20)]
    p = tmp_path / "traces.jsonl"
    _write_traces(p, calls)
    r = svc.do_analyze({"traces": {"path": str(p)}})
    assert len(r["points"]) == 1
    assert r["points"][0]["n_calls"] == 20
    assert r["trace_stats"]["groups_below_min"] == 1
