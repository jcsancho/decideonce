"""M10 tests: downgrade trial, monitoring, downgrade, pilot (T-PL9..14)."""
import random

import pytest

from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.trainer import FakeTrainer
from decideonce.service import Service
from decideonce.stats import newcombe_paired
from decideonce.store import Store


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
         "options": ["full_refund", "escalate_human"],
         "context_fields": {"amount_eur": {"type": "number"}},
         "required_fields": ["amount_eur"],
         "rules": [{"id": "big", "if": [["amount_eur", ">", 500]],
                    "only": ["escalate_human"]}],
         "data_class": "public"}
    p.update(kw)
    return p


def _set_phase(store, point_id, plan_phase):
    p = store.get_point(point_id)
    p["plan_phase"] = plan_phase
    store.put_point(p)


def _workflow(svc, check="outcome", **kw):
    args = {"type": "workflow", "workflow": "refund_request",
            "downgrade_requires": ["refund.action"], "check": check,
            "run_model": "wonder", "cheap_model": "spark"}
    args.update(kw)
    return svc.do_register(args)


def _trial_run(svc, amount=600):
    return svc.do_plan_run({"workflow": "refund_request",
                            "input": "Ticket.", "context": {"refund.action": {
                                "amount_eur": amount}},
                            "context_source": "agent", "run_model": "wonder",
                            "cheap_model": "spark", "trial": True})


# -- Newcombe unit pins ------------------------------------------------------

def test_newcombe_paired_matches_statsmodels():
    # statsmodels confint_proportions_paired, method="newcombe", table
    # [[10, 5], [2, 20]] (a=10, b=5, c=2, d=20), alpha=0.05:
    # diff 0.0811, CI [-0.0643926, 0.2211160]. z is the exact 95% quantile
    # (statsmodels uses ppf, not 1.96).
    diff, lower, upper = newcombe_paired(10, 5, 2, 20, 1.959963984540054)
    assert diff == pytest.approx(0.0810811, abs=1e-6)
    assert lower == pytest.approx(-0.0643926, abs=1e-6)
    assert upper == pytest.approx(0.2211160, abs=1e-6)


def test_newcombe_zero_discordance_needs_about_106_pairs():
    _, lower, _ = newcombe_paired(106, 0, 0, 0, 2.36)
    assert lower == pytest.approx(-0.05, abs=1e-3)


def test_newcombe_pocock_straddle_fixture():
    _, lo196, _ = newcombe_paired(119, 0, 1, 0, 1.96)
    _, lo236, _ = newcombe_paired(119, 0, 1, 0, 2.36)
    assert lo196 == pytest.approx(-0.0457, abs=1e-3)
    assert lo236 == pytest.approx(-0.0592, abs=1e-3)
    assert lo196 >= -0.05 > lo236


# -- T-PL9: downgrade promotion gate ------------------------------------------

def test_T_PL9_downgrade_allowed_not_settable(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    r = _workflow(svc, downgrade_allowed=True)
    assert r["ok"] is False and r["error"] == "invalid_args"


def test_T_PL9_promote_needs_trial_and_confirm(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert _workflow(svc)["ok"]
    r = svc.do_promote({"workflow": "refund_request", "downgrade": True,
                        "confirm": True})
    assert r["ok"] is False and r["failed"] == ["trial"]
    wrec = store.get_workflow("refund_request")
    wrec["trial"] = {"trial_id": "tt_x", "status": "passed", "n": 120,
                     "bound": -0.04}
    store.put_workflow(wrec)
    r2 = svc.do_promote({"workflow": "refund_request", "downgrade": True})
    assert r2["ok"] is False and r2["failed"] == ["confirm"]
    r3 = svc.do_promote({"workflow": "refund_request", "downgrade": True,
                         "confirm": True})
    assert r3["ok"] and r3["downgrade_allowed"] is True
    assert store.get_workflow("refund_request")["downgrade_allowed"] is True


def test_T_PL9_disabling_needs_only_confirm(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert _workflow(svc)["ok"]
    wrec = store.get_workflow("refund_request")
    wrec["downgrade_allowed"] = True
    store.put_workflow(wrec)
    r = svc.do_promote({"workflow": "refund_request", "downgrade": False})
    assert r["ok"] is False and r["failed"] == ["confirm"]
    r2 = svc.do_promote({"workflow": "refund_request", "downgrade": False,
                         "confirm": True})
    assert r2["ok"] and r2["downgrade_allowed"] is False


# -- T-PL10: trial issue + server-computed checks -----------------------------

def _echo_script(store):
    path = store.root / "check_echo.py"
    path.write_text("import json, sys\n"
                    "obj = json.load(sys.stdin)\n"
                    "ok = 'PASS' in obj.get('final_output', '')\n"
                    "print(json.dumps({'pass': ok}))\n")
    return str(path)


def test_B14_expensive_arm_stores_open_entries(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    b = _trial_run(svc)
    exp = store.get_plan(b["expensive"]["plan_id"])
    chp = store.get_plan(b["cheap"]["plan_id"])
    assert [e["status"] for e in exp["points"]] == ["open"]
    assert exp["would_have"][0]["choice"] == "escalate_human"
    assert [e["status"] for e in chp["points"]] == ["decided"]


def test_B14_check_socket_blocked_and_hash_recorded(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    script = store.root / "check_net.py"
    script.write_text("import json, socket, sys\n"
                      "socket.socket()\n"
                      "print(json.dumps({'pass': True}))\n")
    assert _workflow(svc, check=f"deterministic:{script}")["ok"]
    out = svc.do_plan_run({"workflow": "refund_request", "input": "Ticket.",
                           "context": {"refund.action": {"amount_eur": 600}},
                           "context_source": "agent", "run_model": "wonder",
                           "cheap_model": "spark"})
    res = svc.do_plan_result({"plan_id": out["plan_id"],
                              "final_output": "anything"})
    assert res["check_result"]["pass"] is False
    assert res["check_result"]["script_hash"] == svc._script_hash(str(script))


def test_B14_script_edit_resets_trial(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    script = _echo_script(store)
    assert _workflow(svc, check=f"deterministic:{script}")["ok"]
    b1 = _trial_run(svc)
    assert b1["ok"]
    with open(script, "a") as f:
        f.write("# edited\n")
    b2 = _trial_run(svc)
    assert b2["ok"] and b2["trial_id"] != b1["trial_id"]


def test_T_PL10_trial_bundle_and_pair_records(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    b = _trial_run(svc)
    assert b["ok"], b
    assert b["expensive"]["model"] == "wonder"
    assert b["cheap"]["model"] == "spark"
    assert "prompt_addendum" not in b["expensive"]
    assert "prompt_addendum" in b["cheap"]
    assert len(b["cheap"]["points"]) == 1
    pair = store.plans_for_pair(b["pair_id"])
    assert {(p["trial_id"], p["arm"]) for p in pair} == {
        (b["trial_id"], "expensive"), (b["trial_id"], "cheap")}
    b2 = _trial_run(svc)  # joins the running trial
    assert b2["trial_id"] == b["trial_id"]
    assert b2["pair_id"] != b["pair_id"]
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["status"] == "running" and trial["pairs_issued"] == 2


def test_T_PL10_not_downgrade_eligible(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    r = _trial_run(svc, amount=80)  # rules leave both options; no probe
    assert r["ok"] is False and r["error"] == "not_downgrade_eligible"


def test_T_PL10_check_computed_by_server(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    script = _echo_script(store)
    assert _workflow(svc, check=f"deterministic:{script}")["ok"]
    b = _trial_run(svc)
    exp_id = b["expensive"]["plan_id"]
    chp_id = b["cheap"]["plan_id"]
    r1 = svc.do_plan_result({"plan_id": exp_id, "trial_id": b["trial_id"],
                             "pair_id": b["pair_id"], "arm": "expensive",
                             "final_output": "all good, PASS"})
    assert r1["check_result"]["pass"] is True
    assert r1["check_result"]["source"] == "script"
    r2 = svc.do_plan_result({"plan_id": chp_id, "trial_id": b["trial_id"],
                             "pair_id": b["pair_id"], "arm": "cheap",
                             "final_output": "broken"})
    assert r2["check_result"]["pass"] is False
    assert r2["check_result"]["source"] == "script"
    # the agent can never supply the check: no such field exists
    r3 = svc.do_plan_result({"plan_id": exp_id, "check_result": {"pass": True}})
    assert r3["ok"] is False and r3["error"] == "invalid_args"


def test_T_PL10_check_timeout_and_bad_scripts(tmp_path):
    svc, store, _ = make_service(tmp_path, plan={"check_timeout_s": 1})
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    sleepy = store.root / "check_sleep.py"
    sleepy.write_text("import time; time.sleep(5); print('{\"pass\": true}')\n")
    assert _workflow(svc, check=f"deterministic:{sleepy}")["ok"]
    b = _trial_run(svc)
    r = svc.do_plan_result({"plan_id": b["expensive"]["plan_id"],
                            "final_output": "whatever"})
    assert r["check_result"]["pass"] is False
    assert "timeout" in r["check_result"]["error"]
    # agent rubric refused; outside-repo scripts refused
    assert _workflow(svc, check="agent:helpfulness")["error"] == "invalid_args"
    outside = tmp_path / "check_out.py"
    outside.write_text("print('{\"pass\": true}')\n")
    assert _workflow(
        svc, check=f"deterministic:{outside}")["error"] == "invalid_args"


# -- T-PL11: trial evaluation --------------------------------------------------

def _bulk_pairs(store, trial_id, specs, choice="escalate_human", ts=None,
                missing=()):
    for i, (e, c) in enumerate(specs):
        for arm, passed in (("expensive", e), ("cheap", c)):
            store.put_plan({
                "plan_id": f"{trial_id}-{i}-{arm[0]}",
                "ts": ts[i] if ts else "2026-01-03T00:00:00Z",
                "workflow": "refund_request",
                "recommendation": "downgrade" if arm == "cheap" else "none",
                "model": "spark" if arm == "cheap" else "wonder",
                "points": [{"point_id": "refund.action", "phase": "serve",
                            "status": "decided", "choice": choice,
                            "confidence": 1.0, "rule": "big",
                            "predicted": True}],
                "audit": False, "trial_id": trial_id,
                "pair_id": f"{trial_id}-{i}", "arm": arm,
                "pilot_forced": False, "conflicts": [],
                "point_versions": {"refund.action": 1},
                "check_result": (None if (i, arm) in missing
                                 else {"pass": passed, "source": "script"})})


def _running_trial(store, trial_id="tt_1", **kw):
    wrec = store.get_workflow("refund_request")
    trial = {"trial_id": trial_id, "status": "running",
             "run_model": "wonder", "cheap_model": "spark",
             "check": {"type": "outcome", "script": ""},
             "pairs_issued": 0, "last_look": 0, "n": 0, "bound": None}
    trial.update(kw)
    wrec["trial"] = trial
    store.put_workflow(wrec)
    return trial


def _spare_veto(store, svc, pair_id, injected="escalate_human"):
    """Known-incorrect decided points for a pair's cheap plan (no veto)."""
    chp = [p for p in store.plans_for_pair(pair_id) if p["arm"] == "cheap"][0]
    tid = f"trx-{pair_id}"
    store.put_transcript({
        "transcript_id": tid, "ts": "t", "run_id": f"run-{pair_id}",
        "workflow": "refund_request", "model": "w", "data_class": "public",
        "input": "", "events": [], "final_output": "", "plan_id": chp["plan_id"]})
    store.insert_decision({
        "decision_id": f"dx-{pair_id}", "point_id": "refund.action",
        "point_version": 1, "ts": "t", "context": {"amount_eur": 600},
        "state_text": "s", "options": ["full_refund", "escalate_human"],
        "allowed_options": ["escalate_human"], "mode": "extracted",
        "source": "extracted", "extracted_choice": injected,
        "run_id": f"run-{pair_id}", "transcript_id": tid, "at_start": True,
        "outcome": {"ok": False, "correct_choice":
                    "full_refund" if injected == "escalate_human"
                    else "escalate_human"}})


def test_T_PL11_straddle_passes_at_196_fails_at_236(tmp_path):
    for z, want in ((1.96, "passed"), (2.36, "running")):
        svc, store, _ = make_service(tmp_path / f"z{z}", plan={"trial_z": z})
        assert svc.do_register(point())["ok"]
        assert _workflow(svc)["ok"]
        _running_trial(store)
        specs = [(True, True)] * 119 + [(True, False)]
        _bulk_pairs(store, "tt_1", specs)
        _spare_veto(store, svc, "tt_1-119")
        svc._update_trial(store.get_plan("tt_1-0-e"))
        trial = store.get_workflow("refund_request")["trial"]
        assert trial["status"] == want, (z, trial)
        assert trial["last_look"] == 120 and trial["n"] == 120
        if z == 1.96:
            assert trial["bound"] == pytest.approx(-0.0457, abs=1e-3)
        else:
            assert trial["bound"] == pytest.approx(-0.0592, abs=1e-3)


def test_T_PL11_resolve_pair_timeout_rules():
    import time as _time
    fresh = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime())
    old = "2020-01-01T00:00:00Z"
    exp = {"ts": fresh, "check_result": {"pass": True}}
    chp = {"ts": fresh, "check_result": None}
    assert Service._resolve_pair(exp, chp, 24, _time.time()) is None
    assert Service._resolve_pair(exp, chp, 0, _time.time()) == (True, False)
    assert Service._resolve_pair({"ts": old, "check_result": None}, chp, 0,
                                 _time.time()) == (True, False)
    assert Service._resolve_pair(exp, {"ts": fresh,
                                       "check_result": {"pass": False}},
                                 24, _time.time()) == (True, False)


def test_T_PL11_missing_cheap_imputed_failure_at_look(tmp_path):
    svc, store, _ = make_service(tmp_path, plan={"pair_timeout_h": 0})
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    for i in range(60):
        b = _trial_run(svc)
        assert b["ok"], b
        exp_id = b["expensive"]["plan_id"]
        assert svc.do_outcome({"plan_id": exp_id, "ok": False})["ok"]
        # cheap arm never reports; timeout 0 imputes it immediately
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["last_look"] == 60
    assert trial["counts"] == {"a": 0, "b": 0, "c": 0, "d": 60}
    assert trial["status"] == "running"  # bound misses, looks remain


def test_T_PL11_only_first_result_counts(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    script = _echo_script(store)
    assert _workflow(svc, check=f"deterministic:{script}")["ok"]
    b = _trial_run(svc)
    exp_id, chp_id = b["expensive"]["plan_id"], b["cheap"]["plan_id"]
    r1 = svc.do_plan_result({"plan_id": exp_id, "final_output": "PASS"})
    assert r1["check_result"]["pass"] is True
    assert r1["check_result"]["source"] == "script"
    r2 = svc.do_plan_result({"plan_id": exp_id, "final_output": "broken"})
    assert r2["check_result"]["pass"] is True
    assert r2["check_result"]["source"] == "script"
    # plan outcomes are refused on deterministic workflows (B9)
    r3 = svc.do_outcome({"plan_id": exp_id, "ok": False})
    assert r3["error"] == "deterministic_check"
    # first result wins on outcome-check workflows
    assert _workflow(svc, check="outcome")["trial_reset"] is True
    assert svc.do_outcome({"plan_id": chp_id, "ok": True})["ok"]
    r4 = svc.do_plan_result({"plan_id": chp_id, "final_output": "broken"})
    assert r4["check_result"] == {"pass": True, "source": "user"}
    r5 = svc.do_outcome({"plan_id": chp_id, "ok": False})
    assert r5["error"] == "already_set"


def _veto_transcript(run_id):
    return {"run_id": run_id, "workflow": "refund_request", "model": "spark",
            "data_class": "public", "input": "Ticket: Amount: 600 EUR.",
            "events": [{"i": 0, "type": "text",
                        "content": "Noted. I'll refund the full 80 EUR now."}],
            "final_output": "Done."}


def _veto_record(t, choice):
    ev0 = t["events"][0]["content"]
    return {"point_id": "refund.action", "choice": choice,
            "choice_quote": "I'll refund the full 80 EUR",
            "decision_pos": {"event": 0, "offset": ev0.find("I'll refund")},
            "context": {"amount_eur": 600},
            "evidence": {"amount_eur": {
                "pos": {"event": -1,
                        "offset": t["input"].find("Amount: 600 EUR")},
                "quote": "Amount: 600 EUR"}}}


def test_T_PL11_veto_fires_on_unknown_or_correct(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    b = _trial_run(svc)
    exp_id, chp_id = b["expensive"]["plan_id"], b["cheap"]["plan_id"]
    assert svc.do_plan_result({"plan_id": exp_id})["ok"]
    assert svc.do_plan_result({"plan_id": chp_id})["ok"]
    assert svc.do_outcome({"plan_id": exp_id, "ok": True})["ok"]
    assert svc.do_outcome({"plan_id": chp_id, "ok": False})["ok"]
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["status"] == "failed" and trial["reason"] == "veto"


def test_T_PL11_veto_spared_when_decision_wrong(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    b = _trial_run(svc)
    exp_id, chp_id = b["expensive"]["plan_id"], b["cheap"]["plan_id"]
    t = _veto_transcript("veto-spare")
    res = svc.do_plan_result({"plan_id": chp_id, "transcript": t})
    c1 = svc.do_extract({"transcript_id": res["transcript_id"]})
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [_veto_record(t, "escalate_human")]})
    assert r["rejected"] == [], r
    did = r["accepted"][0]["decision_id"]
    assert svc.do_outcome({"decision_id": did,
                           "correct_choice": "full_refund"})["ok"]
    assert svc.do_plan_result({"plan_id": exp_id})["ok"]
    assert svc.do_outcome({"plan_id": exp_id, "ok": True})["ok"]
    assert svc.do_outcome({"plan_id": chp_id, "ok": False})["ok"]
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["status"] == "running" and trial["n"] == 1


def test_B6_look_needs_first_n_issued_resolved(tmp_path):
    # withholding the first pair must block the look, even when n later
    # pairs resolved: first-n-resolved would let a bad pair be hidden.
    svc, store, _ = make_service(tmp_path, plan={"trial_looks": [2]})
    assert svc.do_register(point())["ok"]
    assert _workflow(svc)["ok"]
    _running_trial(store)
    import time as _t
    # ordered fresh timestamps: pair 0 is oldest (first issued) but its
    # cheap arm is still pending within the timeout.
    ts = [_t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime(_t.time() - (3 - i)))
          for i in range(3)]
    _bulk_pairs(store, "tt_1", [(True, True)] * 3, ts=ts,
                missing={(0, "cheap")})
    svc._update_trial(store.get_plan("tt_1-2-e"))
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["last_look"] == 0 and trial["status"] == "running"
    pending = store.get_plan("tt_1-0-c")
    pending["check_result"] = {"pass": True, "source": "script"}
    store.put_plan(pending)
    svc._update_trial(store.get_plan("tt_1-2-e"))
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["last_look"] == 2


def test_B7_plan_run_locks_models_to_workflow_record(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    base = {"workflow": "refund_request", "input": "t",
            "context": {"refund.action": {"amount_eur": 600}}}
    assert svc.do_plan_run(base)["ok"]
    r = svc.do_plan_run({**base, "cheap_model": "spark2"})
    assert r["error"] == "invalid_args"
    r = svc.do_plan_run({**base, "run_model": "wonder2"})
    assert r["error"] == "invalid_args"


def test_B8_finished_trial_cannot_restart(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    wrec = store.get_workflow("refund_request")
    wrec["trial"] = {"trial_id": "tt_old", "status": "passed", "n": 120,
                     "bound": -0.04}
    store.put_workflow(wrec)
    assert _trial_run(svc)["error"] == "trial_finished"
    wrec = store.get_workflow("refund_request")
    wrec["trial"]["status"] = "failed"
    store.put_workflow(wrec)
    assert _trial_run(svc)["error"] == "trial_finished"
    # a version bump of a requires point is a reset condition
    p = point()
    p["options"] = ["full_refund", "escalate_human", "deny"]
    assert svc.do_register(p)["ok"]
    assert store.get_workflow("refund_request")["trial"]["status"] == "reset"
    _set_phase(store, "refund.action", "serve")  # re-promoted after the bump
    b = _trial_run(svc)
    assert b["ok"] and b["trial_id"] != "tt_old"


def test_B8_new_trial_after_reregister_with_new_model(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    wrec = store.get_workflow("refund_request")
    wrec["trial"] = {"trial_id": "tt_old", "status": "failed",
                     "reason": "bound"}
    store.put_workflow(wrec)
    assert _workflow(svc, cheap_model="spark2")["trial_reset"] is True
    b = svc.do_plan_run({"workflow": "refund_request", "input": "Ticket.",
                         "context": {"refund.action": {"amount_eur": 600}},
                         "context_source": "agent", "run_model": "wonder",
                         "cheap_model": "spark2", "trial": True})
    assert b["ok"] and b["trial_id"] != "tt_old"


def test_T_PL11_fails_at_last_look(tmp_path):
    svc, store, _ = make_service(tmp_path, plan={"trial_looks": [2]})
    assert svc.do_register(point())["ok"]
    assert _workflow(svc)["ok"]
    _running_trial(store)
    _bulk_pairs(store, "tt_1", [(True, True)] * 2)
    svc._update_trial(store.get_plan("tt_1-0-e"))
    trial = store.get_workflow("refund_request")["trial"]
    assert trial["status"] == "failed" and trial["reason"] == "bound"


def test_T_PL11_join_rejects_model_change(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    assert _trial_run(svc)["ok"]
    r = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                         "context": {"refund.action": {"amount_eur": 600}},
                         "run_model": "wonder", "cheap_model": "other",
                         "trial": True})
    assert r["ok"] is False and r["error"] == "invalid_args"


def test_T_PL11_reset_on_model_check_requires_point(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_register(point(point_id="refund.other"))["ok"]
    script = _echo_script(store)

    def _passed():
        assert _workflow(svc)["ok"]
        wrec = store.get_workflow("refund_request")
        wrec["trial"] = {"trial_id": "tt_x", "status": "passed", "n": 120,
                         "bound": -0.04}
        wrec["downgrade_allowed"] = True
        store.put_workflow(wrec)

    def _reset():
        wrec = store.get_workflow("refund_request")
        assert wrec["trial"]["status"] == "reset"
        assert wrec["downgrade_allowed"] is False

    _passed()
    assert _workflow(svc, cheap_model="spark2")["trial_reset"] is True
    _reset()
    _passed()
    assert _workflow(svc, check=f"deterministic:{script}")[
        "trial_reset"] is True
    _reset()
    _passed()
    assert _workflow(svc, downgrade_requires=["refund.action",
                                              "refund.other"])[
        "trial_reset"] is True
    _reset()
    _passed()  # identical re-register keeps the trial
    assert _workflow(svc)["trial_reset"] is False
    wrec = store.get_workflow("refund_request")
    assert wrec["trial"]["status"] == "passed"
    assert wrec["downgrade_allowed"] is True
    # a version bump of a requires point resets too
    p = point()
    p["options"] = ["full_refund", "escalate_human", "deny"]
    assert svc.do_register(p)["ok"]
    _reset()


# -- T-PL12: downgrade monitoring ----------------------------------------------

def _monitor_setup(tmp_path):
    svc, store, _ = make_service(
        tmp_path, plan={"audit_rate": 0.0, "downgrade_audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    assert _workflow(svc)["ok"]
    wrec = store.get_workflow("refund_request")
    wrec["downgrade_allowed"] = True
    store.put_workflow(wrec)
    return svc, store


def _monitor_pair(svc, exp_ok, chp_ok):
    out = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                           "context": {"refund.action": {"amount_eur": 600}},
                           "run_model": "wonder", "cheap_model": "spark"})
    assert out.get("monitor") is True, out
    exp_id, chp_id = out["expensive"]["plan_id"], out["cheap"]["plan_id"]
    assert svc.do_plan_result({"plan_id": exp_id})["ok"]
    assert svc.do_plan_result({"plan_id": chp_id})["ok"]
    assert svc.do_outcome({"plan_id": exp_id, "ok": exp_ok})["ok"]
    assert svc.do_outcome({"plan_id": chp_id, "ok": chp_ok})["ok"]
    return out["pair_id"]


def _monitor_bulk(store, tag, n, exp_pass, chp_pass, ts, cheap="spark",
                  missing_cheap=False):
    for i in range(n):
        for arm, passed in (("expensive", exp_pass), ("cheap", chp_pass)):
            store.put_plan({
                "plan_id": f"{tag}-{i}-{arm[0]}", "ts": ts,
                "workflow": "refund_request",
                "recommendation": "downgrade" if arm == "cheap" else "none",
                "model": "spark" if arm == "cheap" else "wonder",
                "cheap_model": cheap, "run_model": "wonder",
                "points": [], "audit": False, "trial_id": None,
                "pair_id": f"{tag}-{i}", "arm": arm, "pilot_forced": False,
                "conflicts": [], "point_versions": {"refund.action": 1},
                "check_result": (None if missing_cheap and arm == "cheap"
                                 else {"pass": passed, "source": "script"})})


def test_B11_monitor_imputes_timeout_as_cheap_failure(tmp_path):
    # 30 timed-out cheap arms (no result, old ts) count as (pass, fail)
    # and revoke, exactly like reported failures.
    svc, store = _monitor_setup(tmp_path)
    _monitor_bulk(store, "old", 30, True, False, "2026-01-03T00:00:00Z",
                  missing_cheap=True)
    svc._update_monitor(store.get_plan("old-0-e"))
    assert store.get_workflow("refund_request")["downgrade_allowed"] is False


def test_B11_monitor_ignores_pre_enablement_pairs(tmp_path):
    svc, store = _monitor_setup(tmp_path)
    import time as _t
    now = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
    wrec = store.get_workflow("refund_request")
    wrec["monitor"] = {"enabled_ts": now}
    store.put_workflow(wrec)
    _monitor_bulk(store, "pre", 30, True, False, "2026-01-03T00:00:00Z")
    svc._update_monitor(store.get_plan("pre-0-e"))
    assert store.get_workflow("refund_request")["downgrade_allowed"] is True


def test_B11_monitor_ignores_old_model_pairs(tmp_path):
    svc, store = _monitor_setup(tmp_path)
    import time as _t
    now = _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())
    _monitor_bulk(store, "mdl", 30, True, False, now, cheap="spark-old")
    svc._update_monitor(store.get_plan("mdl-0-e"))
    assert store.get_workflow("refund_request")["downgrade_allowed"] is True


def test_T_PL12_min_trial_bundle_shape(tmp_path):
    svc, store = _monitor_setup(tmp_path)
    out = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                           "context": {"refund.action": {"amount_eur": 600}},
                           "run_model": "wonder", "cheap_model": "spark"})
    assert out["monitor"] is True
    assert out["expensive"]["model"] == "wonder"
    assert out["cheap"]["model"] == "spark"
    assert "prompt_addendum" not in out["expensive"]
    assert "prompt_addendum" in out["cheap"]
    pair = store.plans_for_pair(out["pair_id"])
    assert len(pair) == 2
    assert all(p["trial_id"] is None for p in pair)


def test_T_PL12_insufficient_data_no_revoke(tmp_path):
    svc, store = _monitor_setup(tmp_path)
    for _ in range(10):  # all bad, but fewer than 30 pairs
        _monitor_pair(svc, True, False)
    assert store.get_workflow("refund_request")["downgrade_allowed"] is True


def test_T_PL12_revoke_on_bad_window(tmp_path):
    svc, store = _monitor_setup(tmp_path)
    revoked_at = None
    for i in range(40):
        if not store.get_workflow("refund_request")["downgrade_allowed"]:
            revoked_at = i
            break
        _monitor_pair(svc, True, False)
    assert revoked_at == 30  # first window with enough pairs revokes
    wrec = store.get_workflow("refund_request")
    assert wrec["downgrade_allowed"] is False
    assert wrec["monitor"]["n"] == 30
    assert "upper bound" in wrec["monitor"]["reason"]


def test_T_PL12_window_rolls(tmp_path):
    svc, store = _monitor_setup(tmp_path)
    _monitor_pair(svc, True, False)  # one bad pair...
    for _ in range(120):  # ...falls out of the 120 window
        _monitor_pair(svc, True, True)
    assert store.get_workflow("refund_request")["downgrade_allowed"] is True


# -- T-PL13: pilot_force ---------------------------------------------------------

def _comparison(store, pid, tag, choice, conf, label, phase="shadow",
                ts=None, eval_only=False):
    """One shadow plan + transcript + correctness row (direct build)."""
    from decideonce.service import _utcnow

    opts = ["full_refund", "escalate_human"]
    ts = ts or _utcnow()
    other = opts[1] if choice == opts[0] else opts[0]
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


def _pilot_setup(tmp_path):
    svc, store, _ = make_service(
        tmp_path, train={"min_train": 1, "min_examples": 10},
        serve={"min_auto_support": 3},
        plan={"min_shadow": 10, "allow_pilot_force": True},
        hosted_jev={"fit_rows": 10})
    assert svc.do_register(point())["ok"]
    opts = ["full_refund", "escalate_human"]
    for i in range(10):
        label = opts[i % 2]
        _comparison(store, "refund.action", f"fit{i}", label, 0.7, label)
    fit = svc._maybe_fit_plan("refund.action")
    assert fit is not None, "plan fit must trigger"
    return svc, store


def _pilot_run(svc, **kw):
    args = {"workflow": "refund_request", "input": "Ticket T-1.",
            "context": {"refund.action": {"amount_eur": 80}},
            "run_model": "wonder", "cheap_model": "spark",
            "pilot_force": True}
    args.update(kw)
    return svc.do_plan_run(args)


def test_T_PL13_refused_unless_allowed(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    r = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                         "context": {}, "run_model": "w", "cheap_model": "c",
                         "pilot_force": True})
    assert r["ok"] is False and r["error"] == "pilot_force_not_allowed"


def test_T_PL13_forced_serve_bypasses_threshold(tmp_path):
    svc, store = _pilot_setup(tmp_path)
    assert _workflow(svc)["ok"]
    normal = svc.do_plan_run({"workflow": "refund_request", "input": "t",
                              "context": {"refund.action": {
                                  "amount_eur": 80}},
                              "run_model": "wonder", "cheap_model": "spark"})
    assert normal["recommendation"] == "none"  # collect: open
    forced = _pilot_run(svc)
    assert forced["ok"], forced
    assert forced["pilot_forced"] is True
    assert forced["recommendation"] == "inject"
    assert forced["points"][0]["status"] == "decided"
    assert "prompt_addendum" in forced
    # the forced plan carries no trial/pair identity
    rec = store.get_plan(forced["plan_id"])
    assert rec["trial_id"] is None and rec["pair_id"] is None


def test_T_PL13_pilot_downgrade_bypasses_trial(tmp_path):
    svc, store = _pilot_setup(tmp_path)
    assert _workflow(svc)["ok"]
    out = _pilot_run(svc, pilot_mode="downgrade")
    assert out["ok"], out
    assert out["recommendation"] == "downgrade"
    assert out["model"] == "spark"
    assert "prompt_addendum" in out
    assert store.get_workflow("refund_request")["downgrade_allowed"] is False
    assert store.get_workflow("refund_request")["trial"] is None


def test_T_PL13_downgrade_needs_predicted(tmp_path):
    svc, _ = _pilot_setup(tmp_path)
    assert _workflow(svc)["ok"]
    r = _pilot_run(svc, pilot_mode="downgrade", context={})
    assert r["ok"] is False and r["error"] == "not_downgrade_eligible"
    r2 = _pilot_run(svc, pilot_mode="inject", context={})
    assert r2["recommendation"] == "none"  # graceful, still tagged
    assert r2["pilot_forced"] is True


def test_T_PL13_trial_and_pilot_do_not_combine(tmp_path):
    svc, _ = _pilot_setup(tmp_path)
    r = _pilot_run(svc, trial=True)
    assert r["ok"] is False and r["error"] == "invalid_args"


def test_T_PL13_pilot_plans_ignored_by_promotion(tmp_path):
    svc, store = _pilot_setup(tmp_path)
    opts = ["full_refund", "escalate_human"]
    for i in range(200):
        store.insert_decision({
            "decision_id": f"lv{i}", "point_id": "refund.action",
            "point_version": 1, "ts": "t", "context": {"amount_eur": 80},
            "state_text": "s", "options": opts, "allowed_options": opts,
            "mode": "teacher", "source": "do_decide",
            "teacher_choice": opts[i % 2], "at_start": True})
    # plan fit already present from _pilot_setup
    for i in range(5):  # pilot shadow comparisons: must not count
        pid = f"pp{i}"
        store.put_plan({
            "plan_id": pid, "ts": "t", "workflow": "refund_request",
            "recommendation": "none", "model": "wonder",
            "points": [{"point_id": "refund.action", "phase": "shadow",
                        "status": "open", "choice": "full_refund",
                        "confidence": 0.9, "predicted": True}],
            "audit": False, "pilot_forced": True, "conflicts": [],
            "point_versions": {"refund.action": 1}})
        tid = f"ptx{i}"
        store.put_transcript({
            "transcript_id": tid, "ts": "t", "run_id": f"prun-{i}",
            "workflow": "refund_request", "model": "w", "data_class": "public",
            "input": "", "events": [], "final_output": "", "plan_id": pid})
        store.insert_decision({
            "decision_id": f"px{i}", "point_id": "refund.action",
            "point_version": 1, "ts": "t", "context": {"amount_eur": 80},
            "state_text": "s", "options": opts, "allowed_options": opts,
            "mode": "extracted", "source": "extracted",
            "extracted_choice": "full_refund", "run_id": f"prun-{i}",
            "transcript_id": tid, "at_start": True,
            "outcome": {"ok": True}})
    r = svc.do_promote({"point_id": "refund.action", "plan_phase": "serve",
                        "confirm": True})
    assert r["ok"] is False
    assert "evidence" in r["failed"]
    assert "n=0" in r["checks"]["evidence"]["detail"]


# -- T-PL14: version bump resets plan state --------------------------------------

def test_T_PL14_bump_resets_phase_and_evidence(tmp_path):
    svc, store, _ = make_service(
        tmp_path, train={"min_train": 1, "min_examples": 10},
        serve={"min_auto_support": 3},
        plan={"min_shadow": 10},
        hosted_jev={"fit_rows": 10})
    assert svc.do_register(point())["ok"]
    _set_phase(store, "refund.action", "serve")
    opts = ["full_refund", "escalate_human"]
    for i in range(10):
        label = opts[i % 2]
        _comparison(store, "refund.action", f"v{i}", label, 0.7, label)
    fit = svc._maybe_fit_plan("refund.action")
    assert fit is not None
    _comparison(store, "refund.action", "vx", "full_refund", 0.9,
                "full_refund")
    assert len(svc._plan_evidence_hosted(
        store.get_point("refund.action"), fit)) == 1
    assert svc.do_register(point())["ok"]  # unchanged: no reset
    assert store.get_point("refund.action")["plan_phase"] == "serve"
    p = point()
    p["options"] = ["full_refund", "escalate_human", "deny"]
    assert svc.do_register(p)["ok"]  # version bump
    assert store.get_point("refund.action")["plan_phase"] == "collect"
    assert svc._plan_evidence_hosted(
        store.get_point("refund.action"), fit) == []
