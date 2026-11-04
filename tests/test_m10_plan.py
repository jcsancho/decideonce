"""M10 tests: plan eligibility (T-PL1, real §15.4)."""
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
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    for section, kv in over.items():
        cfg[section].update(kv)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg)), store, cfg


def point(**kw):
    p = {"point_id": "refund.action", "workflow": "refund_request",
         "question": "Which action?", "kind": "choice",
         "options": ["full_refund", "deny"],
         "context_fields": {"amount_eur": {"type": "number"},
                            "note": {"type": "string"},
                            "label_hint": {"type": "string"}},
         "required_fields": ["amount_eur"],
         "data_class": "public"}
    p.update(kw)
    return p


def _comparison(store, pid, tag, choice, conf, label, phase="shadow",
                ts=None, eval_only=False, opts=None):
    """One shadow plan + transcript + correctness row (direct build)."""
    from decideonce.service import _utcnow

    opts = opts or ["full_refund", "deny"]
    ts = ts or _utcnow()
    other = "deny" if choice == "full_refund" else "full_refund"
    store.put_plan({
        "plan_id": f"pl{tag}", "ts": ts, "workflow": "refund_request",
        "recommendation": "none", "model": "wonder",
        "points": [{"point_id": pid, "phase": phase, "status": "open",
                    "choice": choice, "confidence": conf, "predicted": True,
                    "probabilities": {choice: conf,
                                      other: round(1 - conf, 2)},
                    "allowed_options": opts}],
        "would_have": [], "audit": False, "pilot_forced": False,
        "conflicts": [], "point_versions": {pid: 1}})
    store.put_transcript({
        "transcript_id": f"trx{tag}", "ts": ts, "run_id": f"ser{tag}",
        "workflow": "refund_request", "model": "w", "data_class": "public",
        "input": "", "events": [], "final_output": "",
        "plan_id": f"pl{tag}", "eval_only": eval_only})
    store.insert_decision({
        "decision_id": f"ex{tag}", "point_id": pid,
        "point_version": 1, "ts": ts,
        "context": {"amount_eur": 80}, "state_text": "s",
        "options": opts, "allowed_options": opts,
        "mode": "extracted", "source": "extracted",
        "extracted_choice": label, "run_id": f"ser{tag}",
        "transcript_id": f"trx{tag}", "at_start": True,
        "outcome": {"ok": True}})


def _labeled_runs(svc, at_start):
    """4 runs x 2 rows; at_start[i] flags row i. Choices alternate."""
    for i, start in enumerate(at_start):
        d = svc.do_decide({"point_id": "refund.action",
                           "run_id": f"run{i // 2}", "at_start": start,
                           "context": {"amount_eur": 80 + i}})
        svc.do_record({"decision_id": d["decision_id"],
                       "choice": ["full_refund", "deny"][i % 2]})


def test_T_PL1_eligible_uses_min_at_start(tmp_path):
    svc, store, _ = make_service(tmp_path, plan={"min_at_start": 0.9})
    assert svc.do_register(point())["ok"]
    # 6/8 at_start = 0.75 < 0.9
    _labeled_runs(svc, [True, True, True, False, True, True, False, True])
    pt = store.get_point("refund.action")
    rate, n_at, n = svc._plan_at_start_rate(pt)
    assert rate == 0.75 and (n_at, n) == (6, 8)
    assert svc._plan_eligible_hosted(pt) is False


def test_T_PL1_eligible_config_threshold(tmp_path):
    svc, store, _ = make_service(tmp_path, plan={"min_at_start": 0.5})
    assert svc.do_register(point())["ok"]
    _labeled_runs(svc, [True, True, True, False, True, True, False, True])
    pt = store.get_point("refund.action")
    assert svc._plan_at_start_rate(pt)[0] == 0.75
    assert svc._plan_eligible_hosted(pt) is True  # 0.75 >= 0.5


def test_plan_fit_agreement_counts_errors(tmp_path):
    # Tiered replacement for the plan-probe-accuracy regression (the card
    # line summed 1 per row instead of 1 per hit): the plan fit agreement
    # counts hits.
    svc, store, _ = make_service(tmp_path, hosted_jev={"fit_rows": 10},
                                 serve={"min_auto_support": 3})
    assert svc.do_register(point())["ok"]
    for i in range(10):
        label = ["full_refund", "deny"][i % 2]
        choice = label if i < 8 else ("deny" if label == "full_refund"
                                      else "full_refund")
        _comparison(store, "refund.action", f"fag{i}", choice, 0.9, label)
    fit = svc._maybe_fit_plan("refund.action")
    assert fit is not None
    assert fit["fit_agreement"] == 0.8


def test_eval_only_transcripts_never_train_or_count(tmp_path):
    # Pilot arm runs on held-out tickets: their rows must not train the
    # private model, enter plan comparisons, or count toward eligibility.
    svc, store, _ = make_service(tmp_path, train={"min_train": 1,
                                                   "min_examples": 10})
    assert svc.do_register(point())["ok"]
    for i in range(8):
        store.insert_decision({
            "decision_id": f"e{i}", "point_id": "refund.action",
            "point_version": 1, "ts": "2026-01-01T00:00:00Z",
            "context": {"amount_eur": 80 + i}, "state_text": f"estate {i}",
            "options": ["full_refund", "deny"],
            "allowed_options": ["full_refund", "deny"],
            "teacher_choice": ["full_refund", "deny"][i % 2],
            "run_id": f"erun{i // 2}", "transcript_id": f"etr{i // 2}",
            "at_start": True})
    for r in range(4):
        store.put_transcript({
            "transcript_id": f"etr{r}", "ts": "2026-01-01T00:00:00Z",
            "run_id": f"erun{r}", "workflow": "refund_request", "model": "w",
            "data_class": "public", "input": "", "events": [],
            "final_output": "", "eval_only": r == 1})
    assert store.eval_only_transcript_ids("refund_request") == {"etr1"}
    # export excludes the 2 eval rows
    r = svc.do_train({"point_id": "refund.action", "force": True,
                      "confirm": True})
    assert r["ok"] and r["status"] == "running", r
    assert len(svc.trainer.uploaded[-1]["rows"]) == 6
    # eligibility counts the 6 clean rows only
    rate, n_at, n = svc._plan_at_start_rate(
        store.get_point("refund.action"))
    assert (rate, n_at, n) == (1.0, 6, 6)
    # plan comparisons skip the eval transcript's plan
    _comparison(store, "refund.action", "ev0", "full_refund", 0.9,
                "full_refund", eval_only=True)
    _comparison(store, "refund.action", "ev1", "full_refund", 0.9,
                "full_refund")
    comps = svc._hosted_plan_comparisons(store.get_point("refund.action"))
    assert [c["plan_id"] for c in comps] == ["plev1"]


# -- plan-run helpers ------------------------------------------------------

def _plan_cfg(tmp_path, **over):
    kw = {"train": {"min_train": 1, "min_per_option": 1},
          "serve": {"min_auto_support": 1},
          "plan": {"audit_rate": 0.0, "downgrade_audit_rate": 0.0}}
    for section, vals in over.items():
        kw.setdefault(section, {}).update(vals)
    return make_service(tmp_path, **kw)


def _plan_run(svc, ctx, input="Ticket T-1."):
    return svc.do_plan_run({"workflow": "refund_request", "input": input,
                            "context": ctx, "context_source": "agent",
                            "run_model": "wonder", "cheap_model": "spark"})


def _set_phase(store, point_id, plan_phase):
    p = store.get_point(point_id)
    p["plan_phase"] = plan_phase
    store.put_point(p)


# -- T-PL2: plan shadow -> none + would_have --------------------------------

def test_T_PL2_shadow_none_with_would_have(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path)
    assert svc.do_register(point())["ok"]
    _labeled_runs(svc, [True] * 8)  # eligible: at_start 1.0
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "shadow"})
    assert r["ok"], r
    out = _plan_run(svc, {"refund.action": {"amount_eur": 80,
                                            "label_hint": "full_refund"}})
    assert out["ok"], out
    assert out["recommendation"] == "none"
    assert out["model"] == "wonder"
    assert "prompt_addendum" not in out
    assert out["audit"] is False
    assert out["points"][0]["status"] == "open"
    assert out["points"][0]["choice"] == "full_refund"
    assert len(out["would_have"]) == 1
    assert out["would_have"][0]["choice"] == "full_refund"


# -- T-PL3: decided/open, rule-decided, irreversible -------------------------

def test_T_PL3_decided_open_rule_irreversible(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, hosted_jev={"fit_rows": 20})
    assert svc.do_register(point())["ok"]
    assert svc.do_register(point(
        point_id="refund.big", options=["escalate_human", "full_refund"],
        rules=[{"id": "big", "if": [["amount_eur", ">", 500]],
                "only": ["escalate_human"]}]))["ok"]
    assert svc.do_register(point(point_id="refund.pay",
                                 irreversible=True))["ok"]
    for pid, opts in (("refund.action", ["full_refund", "deny"]),
                      ("refund.big", ["escalate_human", "full_refund"]),
                      ("refund.pay", ["full_refund", "deny"])):
        for i in range(20):
            label = opts[i % 2]
            _comparison(store, pid, f"{pid[-3:]}{i}", label, 0.7, label,
                        opts=opts)
        fit = svc._maybe_fit_plan(pid)
        assert fit is not None and 0.65 < fit["threshold"] < 0.75, (pid, fit)
        _set_phase(store, pid, "serve")
    ctx = {"refund.action": {"amount_eur": 80, "label_hint": "full_refund"},
           "refund.big": {"amount_eur": 600,
                          "label_hint": "escalate_human"},
           "refund.pay": {"amount_eur": 80, "label_hint": "full_refund"}}
    out = _plan_run(svc, ctx)
    assert out["ok"], out
    by_id = {e["point_id"]: e for e in out["points"]}
    assert by_id["refund.action"]["confidence"] > 0.85  # regime check
    assert by_id["refund.action"]["status"] == "decided"
    assert by_id["refund.big"]["status"] == "decided"
    assert by_id["refund.big"]["choice"] == "escalate_human"
    assert by_id["refund.big"]["rule"] == "big"
    assert by_id["refund.pay"]["status"] == "open"
    assert "choice" not in by_id["refund.pay"]  # never predicted
    # below-threshold read stays open
    from decideonce.deciders.hosted import FakeHostedTransport

    def low(body):
        (pid0, q), = body["questions"].items()
        opts = list(q["criteria"])
        probs = {o: 0.5 for o in opts}
        return 200, {"answers": {pid0: {"choice": opts[0],
                                        "probabilities": probs,
                                        "confidence": 0.5}},
                     "usage": {"input_tokens": 1, "output_tokens": 0,
                               "cost": 0.0}}

    orig_transport = svc.deciders["hosted_jev"].transport
    svc.deciders["hosted_jev"].transport = FakeHostedTransport(fn=low)
    out2 = _plan_run(svc, {"refund.action": {"amount_eur": 80,
                                             "label_hint": "full_refund"}})
    e2 = {e["point_id"]: e for e in out2["points"]}["refund.action"]
    assert e2["confidence"] < 0.7  # regime check
    assert e2["status"] == "open"
    svc.deciders["hosted_jev"].transport = orig_transport
    # irreversible serves when explicitly allowed
    svc.cfg["serve"]["allow_auto_irreversible"] = True
    out3 = _plan_run(svc, ctx)
    by3 = {e["point_id"]: e for e in out3["points"]}
    assert by3["refund.pay"]["status"] == "decided"


# -- T-PL4: inject / downgrade / none ---------------------------------------

def test_T_PL4_recommendation_table(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, hosted_jev={"fit_rows": 20})
    assert svc.do_register(point())["ok"]
    for i in range(20):
        label = ["full_refund", "deny"][i % 2]
        _comparison(store, "refund.action", f"pl4{i}", label, 0.7, label)
    fit = svc._maybe_fit_plan("refund.action")
    assert fit is not None and 0.65 < fit["threshold"] < 0.75
    _set_phase(store, "refund.action", "serve")
    w = svc.do_register({"type": "workflow", "workflow": "refund_request",
                         "downgrade_requires": ["refund.action"],
                         "check": "outcome", "run_model": "wonder",
                         "cheap_model": "spark"})
    assert w["ok"], w
    ctx = {"refund.action": {"amount_eur": 80, "label_hint": "full_refund"}}
    out = _plan_run(svc, ctx)  # decided but downgrade not allowed
    assert out["recommendation"] == "inject"
    assert out["model"] == "wonder"
    assert "prompt_addendum" in out
    rec = store.get_workflow("refund_request")
    rec["downgrade_allowed"] = True
    store.put_workflow(rec)
    out2 = _plan_run(svc, ctx)
    assert out2["recommendation"] == "downgrade"
    assert out2["model"] == "spark"
    assert "prompt_addendum" in out2
    from decideonce.deciders.hosted import FakeHostedTransport

    def low(body):
        (pid0, q), = body["questions"].items()
        opts = list(q["criteria"])
        probs = {o: 0.5 for o in opts}
        return 200, {"answers": {pid0: {"choice": opts[0],
                                        "probabilities": probs,
                                        "confidence": 0.5}},
                     "usage": {"input_tokens": 1, "output_tokens": 0,
                               "cost": 0.0}}

    svc.deciders["hosted_jev"].transport = FakeHostedTransport(fn=low)
    out3 = _plan_run(svc, ctx)
    assert out3["recommendation"] == "none"
    assert out3["model"] == "wonder"
    assert "prompt_addendum" not in out3


# -- pilot §15.9: argmax injection bypasses serve gating, flags teacher_only -

def test_pilot_downgrade_decides_despite_teacher_only(tmp_path):
    svc, store, cfg = _plan_cfg(
        tmp_path, plan={"allow_pilot_force": True},
        hosted_jev={"fit_rows": 10})
    cfg["plan"]["allow_pilot_force"] = True
    assert svc.do_register(point())["ok"]
    _labeled_runs(svc, [True] * 8)  # eligible (pilot still needs it)
    # 5/5 per-option support -> both options teacher-only
    for i in range(10):
        label = ["full_refund", "deny"][i % 2]
        _comparison(store, "refund.action", f"pil{i}", label, 0.9, label)
    fit = svc._maybe_fit_plan("refund.action")
    assert "full_refund" in fit["teacher_only_options"]
    w = svc.do_register({"type": "workflow", "workflow": "refund_request",
                         "downgrade_requires": ["refund.action"],
                         "check": "outcome", "run_model": "wonder",
                         "cheap_model": "spark"})
    assert w["ok"], w
    ctx = {"refund.action": {"amount_eur": 80, "label_hint": "full_refund"}}
    out = svc.do_plan_run({"workflow": "refund_request",
                           "input": "Ticket T-1.",
                           "context": ctx, "context_source": "agent",
                           "run_model": "wonder", "cheap_model": "spark",
                           "pilot_force": True, "pilot_mode": "downgrade"})
    assert out["ok"], out
    assert out["recommendation"] == "downgrade"
    assert out["points"][0]["status"] == "decided"
    assert out["points"][0]["teacher_only"] is True
    assert "prompt_addendum" in out


# -- T-PL5: addendum exact template + registration order ---------------------

def test_T_PL5_addendum_exact_template(tmp_path):
    svc, _, _ = _plan_cfg(tmp_path)
    # registered type first: addendum follows registration, not alpha order
    assert svc.do_register(point(
        point_id="refund.type", options=["damage_claim", "other"],
        context_fields={"amount_eur": {"type": "number"}},
        required_fields=[],
        rules=[{"id": "t", "if": [["amount_eur", ">", 0]],
                "only": ["damage_claim"]}]))["ok"]
    assert svc.do_register(point(
        options=["full_refund", "deny"],
        rules=[{"id": "a", "if": [["amount_eur", ">", 0]],
                "only": ["full_refund"]}]))["ok"]
    _set_phase(svc.store, "refund.type", "serve")
    _set_phase(svc.store, "refund.action", "serve")
    out = _plan_run(svc, {"refund.type": {"amount_eur": 80},
                           "refund.action": {"amount_eur": 80}})
    assert out["ok"], out
    assert out["recommendation"] == "inject"
    assert [e["point_id"] for e in out["points"]] == ["refund.type",
                                                      "refund.action"]
    assert out["prompt_addendum"] == (
        "Decisions already made for this task (do not revisit them; "
        "if a fact you find contradicts one, say\n"
        "\"DECISION CONFLICT: <point_id>\" and stop):\n"
        "- refund.type = damage_claim\n"
        "- refund.action = full_refund")


# -- T-PL6: plan audits + drift ------------------------------------------------

def _rule_point(**kw):
    p = point(options=["full_refund", "escalate_human"],
              rules=[{"id": "big", "if": [["amount_eur", ">", 500]],
                      "only": ["escalate_human"]}])
    p.update(kw)
    return p


def test_T_PL6_audits_at_rate_run_without_addendum(tmp_path):
    svc, _, _ = _plan_cfg(tmp_path, plan={"audit_rate": 0.10})
    assert svc.do_register(_rule_point())["ok"]
    _set_phase(svc.store, "refund.action", "serve")
    audits = 0
    for i in range(200):
        out = _plan_run(svc, {"refund.action": {"amount_eur": 600}},
                        input=f"Ticket T-{i}.")
        assert out["ok"], out
        if out["audit"]:
            audits += 1
            assert out["recommendation"] == "none"
            assert out["model"] == "wonder"  # run_model, no addendum
            assert "prompt_addendum" not in out
            assert len(out["would_have"]) == 1
            assert all(e["status"] == "open" for e in out["points"])
        else:
            assert out["recommendation"] == "inject"
            assert "prompt_addendum" in out
    assert 5 < audits < 35  # ~10% of 200


def _drift_transcript(run_id, amount=600):
    return {"run_id": run_id, "workflow": "refund_request", "model": "wonder",
            "data_class": "public",
            "input": f"Ticket: Amount: {amount} EUR.",
            "events": [{"i": 0, "type": "text",
                        "content": "I see it. I'll refund the full 80 EUR now."}],
            "final_output": "Done."}


def _drift_record(t, choice):
    ev0 = t["events"][0]["content"]
    q_amt = f"Amount: {t['input'].split('Amount: ')[1].split(' EUR')[0]} EUR"
    return {"point_id": "refund.action", "choice": choice,
            "choice_quote": "I'll refund the full 80 EUR",
            "decision_pos": {"event": 0, "offset": ev0.find("I'll refund")},
            "context": {"amount_eur": int(q_amt.split()[1])},
            "evidence": {"amount_eur": {
                "pos": {"event": -1, "offset": t["input"].find(q_amt)},
                "quote": q_amt}}}


def test_T_PL6_drift_demotes_to_shadow(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, plan={"audit_rate": 1.0},
                              serve={"drift_window": 3})
    assert svc.do_register(_rule_point())["ok"]
    _set_phase(store, "refund.action", "serve")
    for i in range(3):
        out = _plan_run(svc, {"refund.action": {"amount_eur": 600}},
                        input=f"Ticket T-{i}.")
        assert out["audit"] is True
        t = _drift_transcript(f"drift-{i}")
        res = svc.do_plan_result({"plan_id": out["plan_id"], "transcript": t,
                                  "final_output": "reply"})
        assert res["ok"], res
        # pass-1 extraction of the stored plan transcript; the run ignored
        # the injected escalate_human and refunded instead.
        c1 = svc.do_extract({"transcript_id": res["transcript_id"]})
        assert c1["pass"] == 1
        r = svc.do_extract({"extraction_id": c1["extraction_id"],
                            "extractor": "spark",
                            "records": [_drift_record(t, "full_refund")]})
        assert r["rejected"] == [], r
        assert r["followed_rate"] == 0.0
        want = "serve" if i < 2 else "shadow"  # window fills on the 3rd
        assert store.get_point("refund.action")["plan_phase"] == want
    ev = store.get_point("refund.action")["events"][-1]
    assert ev["event"] == "plan_drift_demotion"


def test_B10_conflict_scan_ignores_input_and_punctuation(tmp_path):
    from decideonce.service import _conflict_hits, ADDENDUM_HEAD
    # the addendum template in the input must not false-hit
    t = {"input": ADDENDUM_HEAD + "\n- refund.action = x",
         "events": [], "final_output": "Done."}
    assert _conflict_hits(t) == []
    t2 = {"input": "Ticket.", "events": [],
          "final_output": "Hmm. DECISION CONFLICT: refund.action. Stop."}
    assert _conflict_hits(t2) == ["refund.action"]


def test_B10_conflict_hits_filtered_to_workflow_points(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, plan={"audit_rate": 1.0},
                              serve={"drift_window": 1})
    assert svc.do_register(_rule_point())["ok"]
    _set_phase(store, "refund.action", "serve")
    out = _plan_run(svc, {"refund.action": {"amount_eur": 600}},
                    input="Ticket T-9.")
    assert out["audit"] is True
    t = _drift_transcript("drift-unk")
    t["final_output"] = "DECISION CONFLICT: refund.nope"
    res = svc.do_plan_result({"plan_id": out["plan_id"], "transcript": t,
                              "final_output": t["final_output"]})
    assert res["ok"] and res["conflicts"] == []


def test_B10_drift_counts_injected_conflicts(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, plan={"audit_rate": 1.0},
                              serve={"drift_window": 1})
    assert svc.do_register(_rule_point())["ok"]
    _set_phase(store, "refund.action", "serve")
    out = _plan_run(svc, {"refund.action": {"amount_eur": 600}},
                    input="Ticket T-1.")
    t = _drift_transcript("drift-ok")
    res = svc.do_plan_result({"plan_id": out["plan_id"], "transcript": t,
                              "final_output": "reply"})
    c1 = svc.do_extract({"transcript_id": res["transcript_id"]})
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [_drift_record(t, "escalate_human")]})
    assert r["rejected"] == [], r  # audit compare matches: no drift
    assert store.get_point("refund.action")["plan_phase"] == "serve"
    ats = store.get_plan(out["plan_id"])["ts"]
    store.put_plan({
        "plan_id": "pl_inj", "ts": ats,
        "workflow": "refund_request", "recommendation": "inject",
        "model": "wonder", "points": [{"point_id": "refund.action",
                                       "phase": "serve", "status": "decided",
                                       "choice": "escalate_human"}],
        "audit": False, "pilot_forced": False,
        "conflicts": ["refund.action"],
        "point_versions": {"refund.action": 1}})
    assert svc._plan_drift_check("refund.action") is True
    assert store.get_point("refund.action")["plan_phase"] == "shadow"


def test_B10_drift_never_crashes_on_skewed_window(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, plan={"audit_rate": 1.0},
                              serve={"drift_window": 1})
    assert svc.do_register(_rule_point())["ok"]
    _set_phase(store, "refund.action", "serve")
    out = _plan_run(svc, {"refund.action": {"amount_eur": 600}},
                    input="Ticket T-1.")
    t = _drift_transcript("drift-ok2")
    res = svc.do_plan_result({"plan_id": out["plan_id"], "transcript": t,
                              "final_output": "reply"})
    c1 = svc.do_extract({"transcript_id": res["transcript_id"]})
    svc.do_extract({"extraction_id": c1["extraction_id"], "extractor": "spark",
                    "records": [_drift_record(t, "escalate_human")]})
    ats = store.get_plan(out["plan_id"])["ts"]
    for i in range(5):  # five extra errors against a window of one
        store.put_plan({
            "plan_id": f"pl_inj{i}", "ts": ats,
            "workflow": "refund_request", "recommendation": "inject",
            "model": "wonder", "points": [], "audit": False,
            "pilot_forced": False, "conflicts": ["refund.action"],
            "point_versions": {"refund.action": 1}})
    assert svc._plan_drift_check("refund.action") is True  # capped, no crash


# -- T-PL7: conflicts + followed_rate ----------------------------------------

def test_T_PL7_conflict_scan_and_followed_rate(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path)
    assert svc.do_register(point(
        point_id="refund.type", options=["damage_claim", "other"],
        context_fields={"amount_eur": {"type": "number"}},
        required_fields=[],
        rules=[{"id": "t", "if": [["amount_eur", ">", 0]],
                "only": ["damage_claim"]}]))["ok"]
    assert svc.do_register(point(
        options=["full_refund", "deny"],
        rules=[{"id": "a", "if": [["amount_eur", ">", 0]],
                "only": ["full_refund"]}]))["ok"]
    _set_phase(store, "refund.type", "serve")
    _set_phase(store, "refund.action", "serve")
    out = _plan_run(svc, {"refund.type": {"amount_eur": 80},
                           "refund.action": {"amount_eur": 80}})
    assert out["recommendation"] == "inject"
    t = {"run_id": "conf-1", "workflow": "refund_request", "model": "wonder",
         "data_class": "public", "input": "Amount: 80 EUR. Claim: damage.",
         "events": [{"i": 0, "type": "text",
                     "content": "The damage claim is valid. "
                                "I'll refund the full 80 EUR now."}],
         "final_output": "Done.\nDECISION CONFLICT: refund.action"}
    res = svc.do_plan_result({"plan_id": out["plan_id"], "transcript": t,
                              "final_output": t["final_output"]})
    assert res["ok"], res
    assert res["conflicts"] == ["refund.action"]
    ev0 = t["events"][0]["content"]
    rec_action = {"point_id": "refund.action", "choice": "full_refund",
                  "choice_quote": "I'll refund the full 80 EUR",
                  "decision_pos": {"event": 0,
                                   "offset": ev0.find("I'll refund")},
                  "context": {"amount_eur": 80},
                  "evidence": {"amount_eur": {
                      "pos": {"event": -1,
                              "offset": t["input"].find("Amount: 80 EUR")},
                      "quote": "Amount: 80 EUR"}}}
    rec_type = {"point_id": "refund.type", "choice": "other",
                "choice_quote": "damage claim is valid",
                "decision_pos": {"event": 0,
                                 "offset": ev0.find("damage claim")},
                "context": {}, "evidence": {}}
    c1 = svc.do_extract({"transcript_id": res["transcript_id"]})
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [rec_action, rec_type]})
    assert r["rejected"] == [], r
    assert r["followed_rate"] == 0.5  # action matched, type did not
    plan = store.get_plan(out["plan_id"])
    assert plan["followed_rate"] == 0.5 and plan["followed_n"] == 2


def test_T_PL7_conflict_counts_as_plan_error(tmp_path):
    svc, store, _ = _plan_cfg(tmp_path, plan={"audit_rate": 1.0},
                              serve={"drift_window": 1})
    assert svc.do_register(_rule_point())["ok"]
    _set_phase(store, "refund.action", "serve")
    out = _plan_run(svc, {"refund.action": {"amount_eur": 600}},
                    input="Ticket T-9.")
    assert out["audit"] is True
    # the run obeyed the injection but flagged a conflict
    t = _drift_transcript("drift-c")
    t["final_output"] = "Ok.\nDECISION CONFLICT: refund.action"
    res = svc.do_plan_result({"plan_id": out["plan_id"], "transcript": t,
                              "final_output": t["final_output"]})
    assert res["conflicts"] == ["refund.action"]
    ev0 = t["events"][0]["content"]
    rec = {"point_id": "refund.action", "choice": "escalate_human",
           "choice_quote": "I'll refund the full 80 EUR",
           "decision_pos": {"event": 0, "offset": ev0.find("I'll refund")},
           "context": {"amount_eur": 600},
           "evidence": {"amount_eur": {
               "pos": {"event": -1, "offset": t["input"].find("Amount: 600")},
               "quote": "Amount: 600 EUR"}}}
    c1 = svc.do_extract({"transcript_id": res["transcript_id"]})
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [rec]})
    assert r["rejected"] == [], r
    # one matching comparison, but the conflict adds the error that demotes
    assert store.get_point("refund.action")["plan_phase"] == "shadow"


# -- T-PL8: plan-serve promotion, one refusal per item -----------------------

def _pl8_setup(tmp_path, n_evidence=12, n_err=0, target=0.3,
               ev_conf=0.9, unreliable=False, teacher_split=None,
               support=3):
    """Plan fit (10 perfect comparisons at 0.7) + evidence comparisons.

    No gate or training needed: the promotion logic is what is under test.
    """
    svc, store, _ = make_service(
        tmp_path, train={"min_train": 1, "min_examples": 10},
        serve={"min_auto_support": support},
        plan={"min_shadow": 10, "target_error": target},
        hosted_jev={"fit_rows": 10})
    assert svc.do_register(point())["ok"]
    for i in range(10):
        label = ["full_refund", "deny"][i % 2]
        _comparison(store, "refund.action", f"fit{i}", label, 0.7, label)
    fit = svc._maybe_fit_plan("refund.action")
    assert fit is not None, "plan fit must trigger"
    for i in range(n_evidence):
        label = ["full_refund", "deny"][i % 2]
        wrong = i >= n_evidence - n_err
        choice = ("deny" if label == "full_refund" else "full_refund") \
            if wrong else label
        _comparison(store, "refund.action", f"ev{i}", choice, ev_conf,
                    label)
    if teacher_split is not None:
        agree_n, disagree_n = teacher_split
        for i in range(agree_n + disagree_n):
            ok = i >= disagree_n
            store.insert_decision({
                "decision_id": f"t{i}", "point_id": "refund.action",
                "point_version": 1, "ts": "2026-01-01T00:00:00Z",
                "context": {"amount_eur": 80 + i}, "state_text": "s",
                "options": ["full_refund", "deny"],
                "allowed_options": ["full_refund", "deny"],
                "teacher_choice": "full_refund",
                "outcome": {"ok": ok,
                            "correct_choice": None if ok else "deny"}})
    if unreliable:
        pt = store.get_point("refund.action")
        pt["extraction_unreliable"] = True
        store.put_point(pt)
    return svc, store


def test_T_PL8_serve_granted_when_all_pass(tmp_path):
    svc, _ = _pl8_setup(tmp_path)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert r["ok"], r
    assert r["plan_phase"] == "serve"
    detail = r["checks"]["evidence"]["detail"]
    assert "correctness" in detail  # label kinds are named


def test_T_PL8_refused_min_examples(tmp_path):
    svc, _ = _pl8_setup(tmp_path, n_evidence=9)  # 9 < 10
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "min_examples" in r["failed"]


def test_T_PL8_refused_threshold(tmp_path):
    svc, _ = _pl8_setup(tmp_path, support=30)  # 10 < 30: no threshold
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "threshold" in r["failed"]


def test_T_PL8_refused_coverage(tmp_path):
    svc, _ = _pl8_setup(tmp_path, ev_conf=0.5)  # pool empty below 0.7
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "coverage" in r["failed"]


def test_T_PL8_refused_evidence_188_vs_189(tmp_path):
    svc, _ = _pl8_setup(tmp_path, n_evidence=188, target=0.02)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    # Wilson upper(0, 188) is above 0.02 and every other item passes
    assert r["failed"] == ["evidence"]


def test_T_PL8_margin_skipped_for_perfect_probe(tmp_path):
    svc, _ = _pl8_setup(tmp_path)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert r["ok"], r
    # H1: a perfect baseline already meets the target
    assert "skipped" in r["checks"]["margin"]["detail"]


def test_T_PL8_refused_shadow_count(tmp_path):
    svc, _ = _pl8_setup(tmp_path, n_evidence=9, target=0.5)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "shadow" in r["failed"]
    assert "evidence" not in r["failed"]  # upper(0, 9) is below 0.5


def test_T_PL8_refused_shadow_agreement(tmp_path):
    svc, _ = _pl8_setup(tmp_path, n_evidence=20, n_err=10)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "shadow_agreement" in r["failed"]  # latest 10 all wrong vs 1.0


def test_T_PL8_refused_extraction(tmp_path):
    svc, _ = _pl8_setup(tmp_path, unreliable=True)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "extraction" in r["failed"]


def test_T_PL8_refused_teacher(tmp_path):
    svc, _ = _pl8_setup(tmp_path, teacher_split=(10, 11))
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert "teacher" in r["failed"]


def test_T_PL8_refused_confirm(tmp_path):
    svc, _ = _pl8_setup(tmp_path)
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve"})
    assert r["ok"] is False and r["failed"] == ["confirm"]
