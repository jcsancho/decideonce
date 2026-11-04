"""M14 demo-slice tests: T-S1 (static) ... T-S5 (eval). No live API calls."""
import json
import random
from pathlib import Path

import pytest

from decideonce.config import load_config
from decideonce.discovery import behaviour as BH
from decideonce.discovery import merge as MG
from decideonce.discovery import static_py as SP
from decideonce.deciders import build_deciders
from decideonce.trainer import FakeTrainer
from decideonce.service import Service
from decideonce.store import Store

REPO = Path(__file__).resolve().parents[1]


def make_service(tmp_path, **cfg_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    for section, values in cfg_over.items():
        cfg[section].update(values)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg))


def cfg(**over):
    c = load_config()
    c["discovery"].update(over)
    return c


# -- T-S1: static detector ---------------------------------------------------

IF_ELIF = '''
import anthropic

def decide_type(ticket, client):
    out = client.messages.create(prompt=ticket)
    label = out.strip().lower()
    if label == "damaged_item":
        return "damaged_item"
    elif label == "not_received":
        return "not_received"
    elif label == "wrong_item":
        return "wrong_item"
    return "unknown"
'''

IN_TUPLE = '''
def decide_priority(ticket, llm):
    p = llm.complete("priority? " + ticket)
    if p.strip() in ("low", "normal", "high"):
        return p.strip()
    return "normal"
'''

MATCH_CASE = '''
from openai import OpenAI

def decide_action(ticket, client):
    raw = client.chat.completions.create(model="m", messages=[])
    action = raw.strip()
    match action:
        case "full_refund":
            return issue_refund(ticket)
        case "deny":
            return deny(ticket)
        case _:
            return escalate(ticket)
'''

DICT_DISPATCH = '''
def decide_next(ticket, model):
    nxt = model.generate(ticket)
    handlers = {"retry": do_retry, "skip": do_skip, "stop": do_stop}
    return handlers[nxt.strip()]()
'''

LITERAL_ANN = '''
from typing import Literal

def decide_tier(ticket, llm):
    tier: Literal["basic", "gold"] = llm.invoke(ticket)
    return tier
'''

ENUM_ANN = '''
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3

def decide_color(ticket, llm):
    color: Color = llm.chat(ticket)
    return color
'''

ENUM_COMPARE = '''
from enum import Enum

class Color(Enum):
    RED = 1
    GREEN = 2

def decide_color(ticket, llm):
    color = llm.chat(ticket)
    if color == Color.RED:
        return "red"
    elif color == Color.GREEN:
        return "green"
    return "?"
'''

JSON_NORM = '''
import json

def decide_type(ticket, llm):
    raw = llm.complete(ticket)
    data = json.loads(raw)
    label = data["type"]
    if label == "a":
        return 1
    elif label == "b":
        return 2
    return 0
'''

FREE_TEXT = '''
def write_reply(ticket, llm):
    reply = llm.complete(ticket)
    print(reply)
    send_email(ticket["email"], reply)
    return reply
'''

SINGLE_COMPARE = '''
def check_gold(ticket, llm):
    tier = llm.complete(ticket)
    if tier == "gold":
        return True
    return False
'''

NON_LLM = '''
def decide_local(ticket):
    label = classify_local(ticket)
    if label == "a":
        return 1
    elif label == "b":
        return 2
    return 0
'''


def test_T_S1_if_elif_chain_is_candidate():
    cands = SP.find_static_candidates([{"path": "w.py", "text": IF_ELIF}],
                                      cfg())
    assert len(cands) == 1
    c = cands[0]
    assert c["step"] == "decide_type"
    assert c["location"] == "w.py:5"
    assert c["options"] == ["damaged_item", "not_received", "wrong_item"]


def test_T_S1_in_tuple_match_dict_literal():
    cands = SP.find_static_candidates([
        {"path": "a.py", "text": IN_TUPLE},
        {"path": "b.py", "text": MATCH_CASE},
        {"path": "c.py", "text": DICT_DISPATCH}], cfg())
    by_step = {c["step"]: c for c in cands}
    assert by_step["decide_priority"]["options"] == ["low", "normal", "high"]
    assert by_step["decide_action"]["options"] == ["full_refund", "deny"]
    assert by_step["decide_next"]["options"] == ["retry", "skip", "stop"]


def test_T_S1_literal_and_enum_annotations():
    tier = SP.find_static_candidates([{"path": "a.py", "text": LITERAL_ANN}],
                                     cfg())
    assert [c["options"] for c in tier] == [["basic", "gold"]]
    ann = SP.find_static_candidates([{"path": "b.py", "text": ENUM_ANN}],
                                    cfg())
    assert [c["options"] for c in ann] == [["RED", "GREEN", "BLUE"]]
    cmp = SP.find_static_candidates([{"path": "c.py", "text": ENUM_COMPARE}],
                                    cfg())
    assert [c["options"] for c in cmp] == [["Color.RED", "Color.GREEN"]]


def test_T_S1_json_loads_normalization():
    cands = SP.find_static_candidates([{"path": "w.py", "text": JSON_NORM}],
                                      cfg())
    assert [c["options"] for c in cands] == [["a", "b"]]


def test_T_S1_negatives_no_candidate():
    for text in (FREE_TEXT, SINGLE_COMPARE, NON_LLM):
        assert SP.find_static_candidates([{"path": "w.py", "text": text}],
                                         cfg()) == []


def test_T_S1_llm_calls_configurable_and_robust():
    assert SP.find_static_candidates([{"path": "w.py", "text": NON_LLM}],
                                     cfg(llm_calls=["classify_local"])) != []
    assert SP.find_static_candidates([{"path": "w.py", "text": "def broken(:\n"}],
                                     cfg()) == []
    # a reply call and a decision call in one function stay separate
    mixed = ("def handle(ticket, llm):\n"
             "    reply = llm.complete(ticket)\n"
             "    send_email('a@b.c', reply)\n"
             "    label = llm.complete(ticket)\n"
             "    if label == 'x':\n"
             "        return 1\n"
             "    elif label == 'y':\n"
             "        return 2\n"
             "    return 0\n")
    cands = SP.find_static_candidates([{"path": "w.py", "text": mixed}],
                                      cfg())
    assert len(cands) == 1 and cands[0]["options"] == ["x", "y"]


def test_T_S1_chained_normalization_and_module_dict():
    chained = ("def decide_action(ticket, llm):\n"
               "    action = llm.complete(ticket).strip()\n"
               "    match action:\n"
               '        case "a":\n'
               "            return 1\n"
               '        case "b":\n'
               "            return 2\n"
               "        case _:\n"
               "            return 0\n")
    cands = SP.find_static_candidates([{"path": "w.py", "text": chained}],
                                      cfg())
    assert [c["options"] for c in cands] == [["a", "b"]]
    mod = ('ROUTES = {"x": 1, "y": 2}\n'
           "\n"
           "def decide_route(ticket, llm):\n"
           "    r = llm.complete(ticket)\n"
           "    return ROUTES[r.strip()]\n")
    cands2 = SP.find_static_candidates([{"path": "w.py", "text": mod}], cfg())
    assert [c["options"] for c in cands2] == [["x", "y"]]


def test_T_S1_do_analyze_code_additive(tmp_path):
    svc = make_service(tmp_path)
    r = svc.do_analyze({"code": [{"path": "w.py", "text": IF_ELIF}]})
    assert r["ok"]
    disc = r["discovery"]
    assert len(disc["candidates"]) == 1
    c = disc["candidates"][0]
    assert (c["step"], c["level"]) == ("decide_type", "C")
    assert c["signals"] == {"text": False, "code": True, "behaviour": False}
    assert c["option_sources"]["damaged_item"] == ["code"]
    assert disc["rejected"] == []
    # rule 3: no new inputs -> output is the same as today (no discovery key)
    r2 = svc.do_analyze({"workflow": {"kind": "prompt", "text": "decide things"}})
    assert r2["ok"] and "discovery" not in r2
    bad = svc.do_analyze({"code": [{"path": "w.py"}]})
    assert not bad["ok"] and bad["error"] == "invalid_args"


# -- T-S2: behaviour detector with consumption --------------------------------

def _two_step_calls(n=25, outputs=("approve", "deny"), consume=True):
    """Runs of decide -> act; act's input embeds the output iff consume."""
    calls = []
    for i in range(n):
        out = outputs[i % len(outputs)]
        act_in = f"Act on decision {out} for case {i}." if consume \
            else "Do the routine thing for this case."
        rid = f"run{i:02d}"
        calls.append({"call_id": f"d{i}", "run_id": rid, "workflow": "w",
                      "step": "decide", "prompt": "Decide.", "output": out,
                      "after": [{"step": "act", "input": act_in}]})
        calls.append({"call_id": f"a{i}", "run_id": rid, "workflow": "w",
                      "step": "act", "prompt": act_in,
                      "output": f"personal note number {i} {i * 13}",
                      "after": []})
    return calls


def test_T_S2_consumed_enum_step_is_candidate():
    res = BH.detect_behaviour(_two_step_calls(), cfg())
    assert [(c["step"], c["consumption_rate"]) for c in res["candidates"]] == [
        ("decide", 1.0)]
    assert res["candidates"][0]["options"] == ["approve", "deny"]
    assert [(r["step"], r["reason"]) for r in res["rejected"]] == [
        ("act", "free text")]


def test_T_S2_free_text_step_rejected():
    calls = [{"call_id": f"f{i}", "run_id": f"run{i:02d}", "workflow": "w",
              "step": "write", "prompt": "Write.", "output": f"unique essay {i}",
              "after": []} for i in range(25)]
    res = BH.detect_behaviour(calls, cfg())
    assert res["candidates"] == []
    assert [(r["step"], r["reason"]) for r in res["rejected"]] == [
        ("write", "free text")]


def test_T_S2_unconsumed_enum_step_rejected():
    res = BH.detect_behaviour(_two_step_calls(consume=False), cfg())
    assert res["candidates"] == []
    by_step = {r["step"]: r for r in res["rejected"]}
    assert by_step["decide"]["reason"] == "not consumed"
    assert "0/25" in by_step["decide"]["detail"]


def test_T_S2_branch_counts_as_consumed():
    calls = []
    for i in range(25):
        out = "left" if i % 2 else "right"
        tool = "ship" if out == "left" else "refund"
        calls.append({"call_id": f"d{i}", "run_id": f"run{i:02d}",
                      "workflow": "w", "step": "decide", "prompt": "Decide.",
                      "output": out,
                      "after": [{"step": tool, "input": "proceed"}]})
    res = BH.detect_behaviour(calls, cfg())
    assert [(c["step"], c["consumption_rate"]) for c in res["candidates"]] == [
        ("decide", 1.0)]


def test_T_S2_min_consumed_threshold_honoured():
    calls = []
    for i in range(25):
        out = ["alpha", "beta"][i % 2]
        act_in = f"use {out} now" if i % 2 == 0 else "routine work"
        calls.append({"call_id": f"d{i}", "run_id": f"run{i:02d}",
                      "workflow": "w", "step": "decide", "prompt": "Decide.",
                      "output": out, "after": [{"step": "act",
                                                "input": act_in}]})
    assert BH.detect_behaviour(calls, cfg(min_consumed=0.8))["candidates"] == []
    got = BH.detect_behaviour(calls, cfg(min_consumed=0.4))["candidates"]
    assert len(got) == 1 and abs(got[0]["consumption_rate"] - 0.52) < 0.01


def test_T_S2_do_analyze_old_traces_unchanged_new_traces_add_discovery(tmp_path):
    svc = make_service(tmp_path)
    old = tmp_path / "old.jsonl"
    with open(old, "w") as f:
        for i in range(25):
            f.write(json.dumps({"call_id": f"e{i}", "workflow": "w",
                                "step": "triage", "prompt": f"Classify {i}",
                                "output": ["bug", "spam"][i % 2]}) + "\n")
    r = svc.do_analyze({"traces": {"path": str(old)}})
    assert r["ok"] and "discovery" not in r and len(r["points"]) == 1
    new = tmp_path / "new.jsonl"
    with open(new, "w") as f:
        for c in _two_step_calls():
            f.write(json.dumps(c) + "\n")
    r2 = svc.do_analyze({"traces": {"path": str(new)}})
    assert r2["ok"]
    assert len(r2["points"]) == 1  # legacy output unchanged (decide group)
    assert [c["step"] for c in r2["discovery"]["candidates"]] == ["decide"]
    assert r2["discovery"]["rejected"][0]["reason"] == "free text"


def test_T_S2_regenerated_traces_carry_run_id_and_after():
    p = REPO / "datasets/workflows/traces/syn-support-01.jsonl"
    if not p.exists():
        pytest.skip("private fixture datasets/ is not exported")
    rows = [json.loads(l) for l in open(p)]
    assert len(rows) == 125
    assert all("run_id" in r and isinstance(r.get("after"), list)
               for r in rows)
    by_run: dict = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    assert len(by_run) == 25
    first = by_run["syn-support-01-run00"]
    assert [c["step"] for c in first][:3] == [
        "refund_action", "escalate", "churn_save"]
    assert first[-1] == [c for c in first if c["step"] == "reply"][0]
    assert first[-1]["after"] == [] and len(first[0]["after"]) == 4


# -- T-S3: merge and evidence levels ------------------------------------------

def _st(step, options, location="w.py:1"):
    return {"step": step, "location": location, "options": list(options),
            "evidence": {"call": "llm.complete", "matches": []}}


def _bh(step, options, rate=1.0, json_field=None):
    return {"step": step, "workflow": "w", "location": None,
            "json_field": json_field, "options": list(options),
            "consumption_rate": rate, "n_calls": 25,
            "evidence": {"consumed_runs": 25, "runs_with_later": 25}}


def _tx(options):
    return {"step": None, "options": list(options),
            "evidence": {"hint": "a | b | c", "line": 1}}


def _ag(point_id, options, score=8, verdict="good_fit", irreversible=False):
    cand = {"draft": {"point_id": point_id, "question": "q?", "kind": "choice",
                      "options": list(options),
                      "context_fields": {"f": {"type": "string"}}}}
    if irreversible:
        cand["draft"]["irreversible"] = True
    scored = {"draft": cand["draft"], "score": score, "verdict": verdict,
              "reasons": ["R2: +2 (facts listable as fields)"]}
    return (cand, scored)


def test_T_S3_two_agreeing_signals_level_A():
    out = MG.merge([_st("s", ["a", "b", "c"])],
                   [_bh("s", ["b", "c", "d"])], [], [])
    assert len(out) == 1
    c = out[0]
    assert c["level"] == "A" and c["agree"] is True
    assert c["signals"] == {"text": False, "code": True, "behaviour": True}
    assert c["options"] == ["a", "b", "c", "d"]
    assert c["option_sources"] == {"a": ["code"], "b": ["code", "behaviour"],
                                   "c": ["code", "behaviour"], "d": ["behaviour"]}
    assert c["consumption_rate"] == 1.0 and c["location"] == "w.py:1"


def test_T_S3_disagreeing_options_not_A():
    out = MG.merge([_st("s", ["a", "b"])], [_bh("s", ["c", "d"])], [], [])
    assert len(out) == 1
    assert out[0]["level"] == "C" and out[0]["agree"] is False
    assert "disagree" in out[0]["reasons"][0]


def test_T_S3_one_signal_level_C():
    assert MG.merge([_st("s", ["a", "b"])], [], [], [])[0]["level"] == "C"
    assert MG.merge([], [_bh("s", ["a", "b"])], [], [])[0]["level"] == "C"
    assert MG.merge([], [], [_tx(["a", "b"])], []) == []  # text alone: dropped


def test_T_S3_agent_plus_one_signal_level_B():
    out = MG.merge([_st("s", ["a", "b"])], [],
                   [], [_ag("w.s", ["a", "b"])])
    assert len(out) == 1
    c = out[0]
    assert c["level"] == "B"
    assert (c["score"], c["verdict"]) == (8, "good_fit")
    assert c["irreversible"] is False
    assert any("R2" in r for r in c["reasons"])
    out2 = MG.merge([], [_bh("s", ["a", "b"])], [],
                    [_ag("other", ["a", "b"], irreversible=True)])
    assert out2[0]["level"] == "B" and out2[0]["irreversible"] is True
    assert MG.merge([], [], [], [_ag("w.s", ["a", "b"])]) == []  # agent alone
    # R7=yes marks the scored draft: the flag must surface from there too
    scored_r7 = {"draft": {"irreversible": True}, "score": 8,
                 "verdict": "possible", "reasons": []}
    out3 = MG.merge([_st("s", ["a", "b"])], [], [],
                    [({"draft": {"point_id": "w.s", "options": ["a", "b"]}},
                       scored_r7)])
    assert out3[0]["irreversible"] is True


def test_T_S3_text_attaches_by_jaccard():
    out = MG.merge([_st("s", ["full", "partial"])],
                   [_bh("s", ["full", "partial"])],
                   [_tx(["full", "partial", "other"])], [])
    assert out[0]["level"] == "A"
    assert out[0]["signals"]["text"] is True
    out2 = MG.merge([_st("s", ["a", "b"])], [], [_tx(["x", "y", "z"])], [])
    assert out2[0]["signals"]["text"] is False  # no overlap: dropped


def test_T_S3_distinct_steps_never_merge_by_options():
    out = MG.merge([], [_bh("s1", ["yes", "no"]), _bh("s2", ["yes", "no"])],
                   [], [])
    assert sorted(c["step"] for c in out) == ["s1", "s2"]
    assert {c["level"] for c in out} == {"C"}


def test_T_S3_rank_by_level_then_score():
    out = MG.merge([_st("c1", ["a", "b"]), _st("a1", ["x", "y"])],
                   [_bh("a1", ["x", "y"])],
                   [], [_ag("w.c1", ["a", "b"], score=3, verdict="possible"),
                        _ag("w.c2", ["m", "n"], score=8, verdict="good_fit")])
    assert [(c["step"], c["level"]) for c in out] == [
        ("a1", "A"), ("c1", "B")]
    out2 = MG.merge([_st("lo", ["a", "b"]), _st("hi", ["c", "d"])], [], [],
                    [_ag("w.lo", ["a", "b"], score=1, verdict="keep_llm"),
                     _ag("w.hi", ["c", "d"], score=9, verdict="good_fit")])
    assert [(c["step"], c["score"]) for c in out2] == [("hi", 9), ("lo", 1)]


def test_T_S3_text_candidates_from_hints():
    from decideonce import analyze as AN
    text = ("# Skill\nClassify the ticket.\n"
            "Pick: full | partial | deny.\n"
            "Choose one:\n- red\n- green\n- blue\n")
    hints = AN.find_hints(text)
    cands = MG.text_candidates(hints, text)
    assert ["full", "partial", "deny"] in [c["options"] for c in cands]
    assert ["red", "green", "blue"] in [c["options"] for c in cands]


def test_T_S3_do_analyze_merges_all_signals(tmp_path):
    svc = make_service(tmp_path)
    opts = ["damaged_item", "not_received", "wrong_item"]
    tr = tmp_path / "t.jsonl"
    with open(tr, "w") as f:
        for i in range(25):
            out = opts[i % 3]
            f.write(json.dumps(
                {"call_id": f"d{i}", "run_id": f"run{i:02d}", "workflow": "w",
                 "step": "decide_type", "prompt": "Decide.",
                 "output": out, "after": [
                     {"step": "act",
                      "input": f"Act on {out} now."}]}) + "\n")
    cand = {"draft": {"point_id": "w.decide_type", "question": "Which type?",
                      "kind": "choice", "options": opts,
                      "context_fields": {"m": {"type": "string"}}},
            "answers": {"R1": "yes", "R2": "yes", "R3": "triage", "R4": 500,
                        "R5": "yes",
                        "R6": {"sensitive": False, "covered": True},
                        "R7": "no", "R8": {"tokens": 1500}}}
    r = svc.do_analyze({
        "workflow": {"kind": "prompt",
                     "text": "Classify: damaged_item | not_received | wrong_item."},
        "code": [{"path": "w.py", "text": IF_ELIF}],
        "traces": {"path": str(tr)},
        "candidates": [cand]})
    assert r["ok"]
    assert len(r["points"]) == 2  # legacy: trace point + scored candidate
    assert len(r["discovery"]["candidates"]) == 1
    c = r["discovery"]["candidates"][0]
    assert c["level"] == "A" and c["step"] == "decide_type"
    assert c["signals"] == {"text": True, "code": True, "behaviour": True}
    assert c["score"] == 8 and c["verdict"] == "good_fit"


# -- T-S4: refund workflow as code ----------------------------------------------

def test_T_S4_workflow_traces_validate_and_cover_all_steps():
    from decideonce import schemaval as SV
    p = REPO / "pilots/refund/data/workflow_traces.jsonl"
    wf = REPO / "pilots/refund/agent_workflow.py"
    if not p.exists() or not wf.exists():
        pytest.skip("private pilots/ fixtures are not exported")
    rows = [json.loads(l) for l in open(p)]
    assert len(rows) == 200
    for r in rows:
        assert SV.validate("trace", r) == [], r["call_id"]
    by_run: dict = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    assert len(by_run) == 50
    first = by_run["refund-T-00001"]
    assert [c["step"] for c in first] == [
        "decide_type", "decide_action", "decide_priority", "write_reply"]
    assert first[-1]["after"] == []
    tools = {e["step"] for c in first for e in c["after"]}
    assert tools >= {"decide_action", "decide_priority", "write_reply"}
    assert any(t in tools for t in ("issue_refund", "ship_replacement",
                                    "escalate", "grant_store_credit",
                                    "deny_claim"))
    # the static detector finds exactly the three decisions in the workflow
    text = wf.read_text()
    cands = SP.find_static_candidates(
        [{"path": "agent_workflow.py", "text": text}], cfg())
    assert sorted(c["step"] for c in cands) == [
        "decide_action", "decide_priority", "decide_type"]


# -- T-S5: measurement ----------------------------------------------------------

def _import_eval():
    import sys
    if not (REPO / "pilots" / "discovery_eval.py").exists():
        pytest.skip("private discovery_eval pilot is not exported")
    sys.path.insert(0, str(REPO / "pilots"))
    import discovery_eval as DE
    return DE


def test_T_S5_score_math():
    DE = _import_eval()
    gold = {"s1": ["a", "b"], "s2": ["x", "y"]}
    s = DE.score_predictions([("s1", ["a", "b"]), ("zzz", ["q"])], gold)
    assert (s["hits"], s["n_pred"], s["n_gold"]) == (1, 2, 2)
    assert s["precision"] == 0.5 and s["recall"] == 0.5
    assert s["option_f1"] == 1.0 and s["strict"] == 0.5
    assert s["fp_steps"] == ["zzz"]
    s2 = DE.score_predictions([("s1", ["a", "c"])], {"s1": ["a", "b"]})
    assert s2["option_f1"] == 0.5 and s2["strict"] == 0.0


def test_T_S5_eval_runs_on_fixtures_and_writes_json(tmp_path, monkeypatch):
    DE = _import_eval()
    text = "Decide: yes | no | maybe.\n"
    tr = tmp_path / "t.jsonl"
    code = ("def decide_it(ticket, llm):\n"
            "    v = llm.complete(ticket).strip()\n"
            '    if v == "yes":\n'
            "        return 1\n"
            '    elif v == "no":\n'
            "        return 2\n"
            "    return 0\n")
    with open(tr, "w") as f:
        for i in range(25):
            out = "yes" if i % 2 else "no"
            f.write(json.dumps(
                {"call_id": f"d{i}", "run_id": f"run{i:02d}", "workflow": "w",
                 "step": "decide_it", "prompt": "Decide.", "output": out,
                 "after": [{"step": "act",
                            "input": f"Act on {out}."}]}) + "\n")
    svc = DE.make_service()
    res = DE.evaluate_workflow(svc, "fix", text, tr,
                               [{"path": "w.py", "text": code}],
                               {"decide_it": ["yes", "no"]})
    assert res["v1"]["precision"] == 1.0 and res["v1"]["recall"] == 1.0
    assert res["v2"]["precision"] == 1.0 and res["v2"]["recall"] == 1.0
    assert res["v2"]["levels"] == {"A": 1, "B": 0, "C": 0}
    results = {"generated_at": "fixture",
               "sets": {"refund": {"workflows": {"fix": res},
                                   **DE.score_set({"fix": res})},
                        "synthetic": {"workflows": {},
                                      **DE.score_set({})},
                        "p0_code": {"status": "n/a", "reason": "fixture",
                                    "workflows": []}}}
    DE.write_outputs(tmp_path, results)
    written = json.loads((tmp_path / "discovery_eval_results.json").read_text())
    assert written["sets"]["refund"]["v2"]["precision"] == 1.0
    assert (tmp_path / "DISCOVERY_EVAL.md").read_text().startswith("# Discovery")
