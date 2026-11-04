"""M10 tests: do_extract two-call flow (T-EX1..14, real §15.1-15.4)."""
import json
import random

import pytest

from decideonce import labels as L
from decideonce.adapters import (adapt_jsonl, adapt_muse, adapt_plain,
                                 validate_transcript)
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


def _launch(svc, pid="refund.action"):
    """Manual smoke launch; returns the uploaded rows."""
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running", r
    return svc.trainer.uploaded[-1]["rows"]


def point(**kw):
    p = {"point_id": "refund.action", "workflow": "refund_request",
         "question": "Which action?", "kind": "choice",
         "options": ["full_refund", "deny"],
         "context_fields": {"amount_eur": {"type": "number"},
                            "reason": {"type": "string"}},
         "required_fields": ["amount_eur"],
         "data_class": "public"}
    p.update(kw)
    return p


INPUT = "Ticket T-1: customer asks for a refund. Amount: 80 EUR."
EV0 = "The customer is owed 80 EUR. I'll refund the full 80 EUR now."
FINAL = "Done: full refund issued."


def transcript(run_id="run1", **kw):
    t = {"run_id": run_id, "workflow": "refund_request", "model": "wonder",
         "data_class": "public", "input": INPUT,
         "events": [{"i": 0, "type": "text", "content": EV0}],
         "final_output": FINAL}
    t.update(kw)
    return t


def extract1(svc, t, **kw):
    return svc.do_extract({"transcript": t, **kw})


def clean_record(choice="full_refund", amount=80):
    q_choice = "I'll refund the full 80 EUR"
    q_amt = "Amount: 80 EUR"
    return {"point_id": "refund.action", "choice": choice,
            "choice_quote": q_choice,
            "decision_pos": {"event": 0, "offset": EV0.find(q_choice)},
            "context": {"amount_eur": amount},
            "evidence": {"amount_eur": {
                "pos": {"event": -1, "offset": INPUT.find(q_amt)},
                "quote": q_amt}}}


# -- T-EX1: transcript schema + positions ----------------------------------

def test_T_EX1_positions_raw_offsets_input_minus1(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_register(point(point_id="refund.type",
                                 options=["damage_claim", "other"],
                                 required_fields=[]))["ok"]
    # raw offsets: a double space counts (no normalisation)
    t = transcript(input="Note: ok. Amount:  80 EUR.",
                   events=[{"i": 0, "type": "text",
                             "content": "The customer is owed  80 EUR. Verdict: full."}])
    c1 = extract1(svc, t)
    assert c1["ok"], c1
    assert c1["pass"] == 1
    assert c1["points"][0]["point_id"] == "refund.action"
    assert "decision_pos" in c1["record_schema"]["properties"]
    assert len(c1["instructions"]) == 5
    rec_input = {"point_id": "refund.action", "choice": "full_refund",
                 "choice_quote": "EUR.",
                 "decision_pos": {"event": -1,
                                  "offset": t["input"].find("EUR.")},
                 "context": {"amount_eur": 80},
                 "evidence": {"amount_eur": {
                     "pos": {"event": -1, "offset": t["input"].find("80")},
                     "quote": "80"}}}
    ev0 = t["events"][0]["content"]
    rec_raw = {"point_id": "refund.type", "choice": "damage_claim",
               "choice_quote": "owed  80 EUR",
               "decision_pos": {"event": 0, "offset": ev0.find("owed  80 EUR")},
               "context": {}, "evidence": {}}
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [rec_input, rec_raw]})
    assert r["rejected"] == [], r
    assert len(r["accepted"]) == 2
    row = store.get_decision(r["accepted"][0]["decision_id"])
    assert row["decision_pos"] == {"event": -1,
                                   "offset": t["input"].find("EUR.")}
    assert row["mode"] == "extracted" and row["source"] == "extracted"


def test_T_EX1_final_event_usable_as_decision_pos(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="rf"))
    qf = "full refund"
    rec = {"point_id": "refund.action", "choice": "full_refund",
           "choice_quote": qf,
           "decision_pos": {"event": 1, "offset": FINAL.find(qf)},
           "context": {"amount_eur": 80},
           "evidence": {"amount_eur": {
               "pos": {"event": -1, "offset": INPUT.find("Amount: 80 EUR")},
               "quote": "Amount: 80 EUR"}}}
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [rec]})
    assert r["rejected"] == [], r
    row = store.get_decision(r["accepted"][0]["decision_id"])
    assert row["decision_pos"] == {"event": 1, "offset": FINAL.find(qf)}


# -- T-EX2: accepted pass-1 record -> decision row --------------------------

def test_T_EX2_accepted_row_shape_weight_at_start(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript())
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [clean_record()]})
    assert r["ok"] and r["rejected"] == [], r
    acc = r["accepted"][0]
    assert acc["at_start"] is True
    assert acc["rule_violation_in_run"] is False
    row = store.get_decision(acc["decision_id"])
    assert row["source"] == "extracted" and row["mode"] == "extracted"
    assert row["extracted_choice"] == "full_refund"
    assert row["teacher_choice"] is None
    assert row["run_id"] == "run1" and row["extractor"] == "spark"
    assert row["at_start"] is True
    label, source, weight = L.resolve_label(row, 1.0, 0.5)
    assert (label, source, weight) == ("full_refund", "extracted", 0.5)


def test_T_EX2_at_start_false_when_evidence_after_input(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="late"))
    q_choice = "I'll refund the full 80 EUR"
    doff = EV0.find(q_choice)
    q_amt = "owed 80 EUR"
    rec = {"point_id": "refund.action", "choice": "full_refund",
           "choice_quote": q_choice,
           "decision_pos": {"event": 0, "offset": doff},
           "context": {"amount_eur": 80},
           "evidence": {"amount_eur": {
               "pos": {"event": 0, "offset": EV0.find(q_amt)}, "quote": q_amt}}}
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [rec]})
    assert r["rejected"] == [], r
    assert r["accepted"][0]["at_start"] is False


# -- T-EX3: one test per §15.2 code ----------------------------------------

def _submit_one(svc, rec, run_id="rx"):
    c1 = extract1(svc, transcript(run_id=run_id))
    assert c1["ok"], c1
    return svc.do_extract({"extraction_id": c1["extraction_id"],
                           "extractor": "spark", "records": [rec]})


def test_T_EX3_unknown_point(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record()
    rec["point_id"] = "refund.nope"
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["unknown_point"]
    assert r["accepted"] == []


def test_T_EX3_choice_not_in_options(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record(choice="maybe")
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["choice_not_in_options"]


def test_T_EX3_quote_mismatch_choice(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record()
    rec["decision_pos"]["offset"] += 1  # quote no longer matches there
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["quote_mismatch"]


def test_T_EX3_quote_mismatch_evidence(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record()
    rec["evidence"]["amount_eur"]["quote"] = "Amount: 81 EUR"
    r = _submit_one(svc, rec)
    # one record, every violated check reported (a wrong quote also fails
    # value support for the claimed 80)
    assert [x["code"] for x in r["rejected"]] == ["quote_mismatch",
                                                  "value_unsupported"]


def test_T_EX3_future_evidence_after_decision(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    q_choice = "I'll refund the full 80 EUR"
    doff = EV0.find(q_choice)
    q_amt = "owed 80 EUR"
    eoff = EV0.find(q_amt)
    assert eoff + len(q_amt) <= doff  # control: ends before -> accepted
    rec = {"point_id": "refund.action", "choice": "full_refund",
           "choice_quote": q_choice,
           "decision_pos": {"event": 0, "offset": doff},
           "context": {"amount_eur": 80},
           "evidence": {"amount_eur": {
               "pos": {"event": 0, "offset": eoff}, "quote": q_amt}}}
    assert _submit_one(svc, rec, run_id="ctl")["rejected"] == []
    # the same quote, claimed for a decision point inside it: it starts
    # before the decision but ends after it -> future_evidence.
    rec2 = {"point_id": "refund.action", "choice": "full_refund",
            "choice_quote": "ed",
            "decision_pos": {"event": 0, "offset": eoff + 2},
            "context": {"amount_eur": 80},
            "evidence": {"amount_eur": {
                "pos": {"event": 0, "offset": eoff}, "quote": q_amt}}}
    r = _submit_one(svc, rec2, run_id="fut")
    assert [x["code"] for x in r["rejected"]] == ["future_evidence"]


def test_T_EX3_future_evidence_later_event(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record()
    rec["evidence"]["amount_eur"] = {
        "pos": {"event": 1, "offset": FINAL.find("full refund")},
        "quote": "full refund"}
    assert _submit_one(svc, rec)["rejected"][0]["code"] == "future_evidence"


def test_T_EX3_value_unsupported(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record(amount=81)  # quote says 80
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["value_unsupported"]


def test_T_EX3_value_unsupported_string(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point(required_fields=["amount_eur", "reason"]))["ok"]
    rec = clean_record()
    rec["context"]["reason"] = "late_delivery"
    rec["evidence"]["reason"] = {
        "pos": {"event": -1, "offset": INPUT.find("customer")},
        "quote": "customer asks for a refund"}
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["value_unsupported"]


def test_T_EX3_value_supported_enum_conceptual_quote(tmp_path):
    # §15.2 table: enums need only a non-empty quote - a code such as
    # "damaged_item" never appears verbatim in raw text; the judgment is
    # the extractor's, audited in §15.3. (Found by the P3 probe.)
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point(
        context_fields={"amount_eur": {"type": "number"},
                        "ticket_type": {"type": "string",
                                        "enum": ["damaged_item", "other"]}},
        required_fields=["amount_eur", "ticket_type"]))["ok"]
    rec = clean_record()
    rec["context"]["ticket_type"] = "damaged_item"
    rec["evidence"]["ticket_type"] = {
        "pos": {"event": -1, "offset": INPUT.find("customer")},
        "quote": "customer asks for a refund"}
    r = _submit_one(svc, rec)
    assert r["rejected"] == [], r["rejected"]
    assert len(r["accepted"]) == 1


def test_quote_hint_points_at_true_offset():
    hint = Service._quote_hint(
        {-1: INPUT, 0: EV0}, {"event": -1, "offset": 0}, "80 EUR")
    assert "quote occurs at offset" in hint
    assert str(INPUT.find("80 EUR")) in hint
    missing = Service._quote_hint(
        {-1: INPUT, 0: EV0}, {"event": -1, "offset": 0}, "nope")
    assert "not found in event -1" in missing
    elsewhere = Service._quote_hint(
        {-1: INPUT, 0: EV0}, {"event": -1, "offset": 0}, "owed 80 EUR")
    assert "quote is in event 0" in elsewhere


def test_B1_mistyped_required_field_rejected(tmp_path):
    # amount_eur: "80" (string) was accepted, then silently dropped from the
    # stored context, hiding rule hits. Now it rejects with the wanted type.
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record()
    rec["context"]["amount_eur"] = "80"
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["missing_field"]
    assert "number" in r["rejected"][0]["message"]
    assert r["accepted"] == []


def test_B1_wrong_enum_member_rejected(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point(
        context_fields={"amount_eur": {"type": "number"},
                        "ticket_type": {"type": "string",
                                        "enum": ["damaged_item", "other"]}},
        required_fields=["amount_eur", "ticket_type"]))["ok"]
    rec = clean_record()
    rec["context"]["ticket_type"] = "nonsense"
    rec["evidence"]["ticket_type"] = {
        "pos": {"event": -1, "offset": INPUT.find("customer")},
        "quote": "customer asks for a refund"}
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["missing_field"]
    assert "nonsense" in r["rejected"][0]["message"]


def test_B2_unevidenced_known_field_rejected(tmp_path):
    # a claimed value without evidence skips quote/future/at_start checks.
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point(
        context_fields={"amount_eur": {"type": "number"},
                        "tier": {"type": "string", "enum": ["basic", "gold"]}},
        required_fields=["amount_eur"]))["ok"]
    rec = clean_record()
    rec["context"]["tier"] = "gold"  # no evidence entry
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["missing_field"]
    assert "no evidence" in r["rejected"][0]["message"]


def test_number_support_is_numeric_not_substring(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    assert Service._number_supported(80.0, "Amount: 80 EUR")
    assert Service._number_supported(80.0, "Amount: 80.00 EUR")
    assert Service._number_supported(1234.5, "total 1,234.50 EUR")
    assert not Service._number_supported(8.0, "Amount: 80 EUR")
    assert not Service._number_supported(80.0, "Amount: 180 EUR")
    # end to end: 81 claimed against an 80 quote still rejects
    rec = clean_record(amount=81)
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["value_unsupported"]


def test_T_EX3_missing_field(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    rec = clean_record()
    rec["context"] = {}
    rec["evidence"] = {}
    r = _submit_one(svc, rec)
    assert [x["code"] for x in r["rejected"]] == ["missing_field"]
    # value without evidence is also missing
    rec2 = clean_record()
    rec2["evidence"] = {}
    r2 = _submit_one(svc, rec2, run_id="rx2")
    assert [x["code"] for x in r2["rejected"]] == ["missing_field"]


def test_T_EX3_duplicate_record(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="dup"))
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [clean_record(), clean_record()]})
    assert len(r["accepted"]) == 1
    assert [x["code"] for x in r["rejected"]] == ["duplicate_record"]
    # ... and across submissions for the same transcript
    r2 = svc.do_extract({"extraction_id": c1["extraction_id"],
                         "extractor": "spark", "records": [clean_record()]})
    assert [x["code"] for x in r2["rejected"]] == ["duplicate_record"]
    assert len(store.decisions_for_transcript(c1["transcript_id"])) == 1


def test_T_EX3_duplicate_allowed_when_repeatable(tmp_path):
    svc, _, _ = make_service(tmp_path)
    assert svc.do_register(point(repeatable=True))["ok"]
    c1 = extract1(svc, transcript(run_id="rep"))
    second = clean_record()
    second["decision_pos"] = {"event": 1, "offset": FINAL.find("full refund")}
    second["choice_quote"] = "full refund"
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [clean_record(), second]})
    assert r["rejected"] == [], r
    assert len(r["accepted"]) == 2


def test_B5_identical_repeatable_resubmit_rejected(tmp_path):
    # repeatable accepts distinct records, but an identical resubmit
    # (retry, reopened pass 1) must not duplicate the row.
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point(repeatable=True))["ok"]
    c1 = extract1(svc, transcript(run_id="rep2"))
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [clean_record(), clean_record()]})
    assert len(r["accepted"]) == 1
    assert [x["code"] for x in r["rejected"]] == ["duplicate_record"]
    r2 = svc.do_extract({"extraction_id": c1["extraction_id"],
                         "extractor": "spark", "records": [clean_record()]})
    assert [x["code"] for x in r2["rejected"]] == ["duplicate_record"]
    assert len(store.decisions_for_transcript(c1["transcript_id"])) == 1


def test_B4_outcome_ok_keeps_extracted_label(tmp_path):
    label, source, weight = L.resolve_label(
        {"extracted_choice": "full_refund", "outcome": {"ok": True}})
    assert (label, source, weight) == ("full_refund", "outcome_ok", 1.0)


# -- T-EX4: rule-forbidden choice kept + flagged ----------------------------

def test_T_EX4_rule_violation_kept_and_flagged(tmp_path):
    svc, store, _ = make_service(tmp_path)
    p = point(options=["full_refund", "escalate_human"],
              rules=[{"id": "big", "if": [["amount_eur", ">", 500]],
                      "only": ["escalate_human"]}])
    assert svc.do_register(p)["ok"]
    t = transcript(run_id="viol", input="Amount: 600 EUR. Escalation territory.")
    c1 = extract1(svc, t)
    q = "Amount: 600 EUR"
    rec = {"point_id": "refund.action", "choice": "full_refund",
           "choice_quote": "I'll refund the full 80 EUR",
           "decision_pos": {"event": 0, "offset": EV0.find("I'll refund")},
           "context": {"amount_eur": 600},
           "evidence": {"amount_eur": {
               "pos": {"event": -1, "offset": t["input"].find(q)}, "quote": q}}}
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [rec]})
    assert r["rejected"] == [], r
    assert r["accepted"][0]["rule_violation_in_run"] is True
    row = store.get_decision(r["accepted"][0]["decision_id"])
    assert row["rule_violation_in_run"] is True
    assert row["extracted_choice"] == "full_refund"


# -- T-EX5: privacy gate ----------------------------------------------------

def test_T_EX5_private_needs_confirm(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    t = transcript(run_id="priv", data_class="private")
    r = extract1(svc, t)
    assert r["ok"] is False and r["error"] == "private_needs_confirm"
    assert store.transcript_by_run("priv") is None  # refused = not stored
    r2 = extract1(svc, t, contributor_off_confirmed=True)
    assert r2["ok"], r2
    t3 = transcript(run_id="priv3", data_class="private")
    r3 = extract1(svc, t3, extractor_local=True)
    assert r3["ok"], r3


# -- T-EX6: pass 2 -> audits, never trains ----------------------------------

def test_T_EX6_pass2_audits_never_train(tmp_path):
    svc, store, _ = make_service(tmp_path, train={"min_train": 1,
                                                   "min_per_option": 1},
                                 extract={"min_audited": 2, "min_agreement": 0.9,
                                          "audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    for i, choice in enumerate(["full_refund", "deny"] * 2):
        c1 = extract1(svc, transcript(run_id=f"p2-{i}"))
        rec = clean_record(choice=choice)
        r1 = svc.do_extract({"extraction_id": c1["extraction_id"],
                             "extractor": "spark", "records": [rec]})
        assert r1["rejected"] == [], r1
        c2 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2})
        assert c2["pass"] == 2
        r2 = svc.do_extract({"extraction_id": c2["extraction_id"],
                             "extractor": "spark",
                             "records": [clean_record(choice=choice)]})
        assert r2["rejected"] == [], r2

        assert "audit_id" in r2["accepted"][0]
    assert len(store.audits_for_transcript(
        store.transcript_by_run("p2-0")["transcript_id"])) == 1
    rows = store.decisions_for_point("refund.action")
    assert len(rows) == 4  # pass-1 only
    for row in rows:
        ex = store.get_extraction(row["extraction_id"])
        assert ex["pass"] == 1
    up = _launch(svc)
    assert len(up) == 4  # the 4 audits never train (audits are not rows)
    from decideonce import train as TR

    gate = TR.extraction_gate(svc.cfg, store, store.get_point("refund.action"))
    assert gate["passed"] is True
    # empty records = "found no decision": valid, completes the extraction
    c3 = extract1(svc, transcript(run_id="p2-empty"))
    r3 = svc.do_extract({"extraction_id": c3["extraction_id"],
                         "extractor": "spark", "records": []})
    assert r3["ok"] and r3["accepted"] == [] and r3["rejected"] == []


def test_B3_pass2_refused_when_not_drawn(tmp_path):
    # the caller cannot choose which transcripts to audit: extra audits
    # would inflate the gate's audited share.
    svc, _, _ = make_service(tmp_path, extract={"audit_rate": 0.0})
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="nodraw"))
    assert c1["audit_required"] is False
    c2 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2})
    assert c2["error"] == "audit_not_required"


def test_B3_pass2_privacy_gate(tmp_path):
    svc, _, _ = make_service(tmp_path, extract={"audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    t = transcript(run_id="priv2")
    t["data_class"] = "private"
    c1 = svc.do_extract({"transcript": t, "contributor_off_confirmed": True})
    assert c1["ok"]
    c2 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2})
    assert c2["error"] == "private_needs_confirm"
    c3 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2,
                         "contributor_off_confirmed": True})
    assert c3["ok"] and c3["pass"] == 2


# -- T-EX7: the §15.3 gate (fails closed) -----------------------------------

def _live_rows(svc, n=4):
    for i in range(n):
        d = svc.do_decide({"point_id": "refund.action",
                           "context": {"amount_eur": 80 + i, "reason": "x"}})
        svc.do_record({"decision_id": d["decision_id"],
                       "choice": ["full_refund", "deny"][i % 2]})


def _double_extract(svc, run_id, agree=True, choice="full_refund"):
    c1 = extract1(svc, transcript(run_id=run_id))
    r1 = svc.do_extract({"extraction_id": c1["extraction_id"],
                         "extractor": "spark",
                         "records": [clean_record(choice=choice)]})
    assert r1["rejected"] == [], r1
    c2 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2})
    other = choice if agree else ("deny" if choice == "full_refund"
                                  else "full_refund")
    r2 = svc.do_extract({"extraction_id": c2["extraction_id"],
                         "extractor": "spark",
                         "records": [clean_record(choice=other)]})
    assert r2["rejected"] == [], r2


def test_T_EX7_gate_audited_leg(tmp_path):
    svc, _, _ = make_service(tmp_path, train={"min_train": 1,
                                               "min_per_option": 1},
                             extract={"min_audited": 2, "min_agreement": 0.9,
                                      "audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    _live_rows(svc)
    _double_extract(svc, "g-a")  # 1 audited < 2 -> extracted excluded
    assert len(_launch(svc)) == 4
    from decideonce import train as TR

    gate = TR.extraction_gate(svc.cfg, svc.store, svc.store.get_point("refund.action"))
    assert gate["passed"] is False


def test_T_EX7_gate_agreement_leg(tmp_path):
    svc, _, _ = make_service(tmp_path, train={"min_train": 1,
                                               "min_per_option": 1},
                             extract={"min_audited": 1, "min_agreement": 0.9,
                                      "audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    _live_rows(svc)
    _double_extract(svc, "g-b", agree=False)  # agreement 0 < 0.9
    assert len(_launch(svc)) == 4
    from decideonce import train as TR

    gate = TR.extraction_gate(svc.cfg, svc.store, svc.store.get_point("refund.action"))
    assert gate["agreement"] == 0.0


def test_T_EX7_gate_share_leg(tmp_path):
    svc, _, _ = make_service(tmp_path, train={"min_train": 1,
                                               "min_per_option": 1},
                             extract={"min_audited": 1, "min_agreement": 0.9,
                                      "audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    _live_rows(svc)
    _double_extract(svc, "g-c")  # audited...
    for i in range(4):  # ...but the share is 1/5 < 0.8
        c1 = extract1(svc, transcript(run_id=f"g-c-single-{i}"))
        svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [clean_record()]})
    assert len(_launch(svc)) == 4
    from decideonce import train as TR

    gate = TR.extraction_gate(svc.cfg, svc.store, svc.store.get_point("refund.action"))
    assert gate["share"] == pytest.approx(0.2)


def test_T_EX7_gate_passes_and_reports(tmp_path):
    svc, _, _ = make_service(tmp_path, train={"min_train": 1,
                                               "min_per_option": 1},
                             extract={"min_audited": 1, "min_agreement": 0.9,
                                      "audit_rate": 1.0})
    assert svc.do_register(point())["ok"]
    _live_rows(svc)
    _double_extract(svc, "g-d")
    up = _launch(svc)
    assert len(up) == 5
    assert sum(1 for r in up if r["weight"] == 0.5) == 1  # extracted joins
    from decideonce import train as TR

    gate = TR.extraction_gate(svc.cfg, svc.store, svc.store.get_point("refund.action"))
    assert gate["passed"] is True and gate["audited"] == 1


def _gate_fixture(store, run_id, r1, r2):
    """One transcript with a completed pass 1 (zero or one row) and pass 2."""
    tid = f"tr_{run_id}"
    store.put_transcript({"transcript_id": tid, "ts": "2026-01-01T00:00:00Z",
                          "run_id": run_id, "workflow": "refund_request",
                          "model": "w", "data_class": "public", "input": "",
                          "events": [], "final_output": ""})
    store.put_extraction({"extraction_id": f"ex1_{run_id}", "ts": "t",
                          "transcript_id": tid, "pass": 1})
    store.put_extraction({"extraction_id": f"ex2_{run_id}", "ts": "t",
                          "transcript_id": tid, "pass": 2})
    store.complete_extraction(f"ex1_{run_id}", "spark", 0, 0, {}, [])
    store.complete_extraction(f"ex2_{run_id}", "spark", 0, 0, {}, [])
    if r1 is not None:
        store.insert_decision({
            "decision_id": f"d_{run_id}", "point_id": "refund.action",
            "point_version": 1, "ts": "t", "context": r1.get("context"),
            "state_text": "s", "options": ["full_refund", "deny"],
            "allowed_options": ["full_refund", "deny"], "mode": "extracted",
            "source": "extracted", "extracted_choice": r1["choice"],
            "run_id": run_id, "transcript_id": tid,
            "extraction_id": f"ex1_{run_id}", "extractor": "spark",
            "decision_pos": r1.get("decision_pos"), "at_start": r1.get("at_start")})
    if r2 is not None:
        store.audit_insert({"audit_id": f"ea_{run_id}", "ts": "t",
                            "transcript_id": tid,
                            "extraction_id": f"ex2_{run_id}",
                            "point_id": "refund.action", "record": r2})


def _rec(choice="full_refund", at_start=True, event=0, amount=80):
    return {"choice": choice, "at_start": at_start,
            "decision_pos": {"event": event, "offset": 3},
            "context": {"amount_eur": amount}}


def test_T_EX7_agreement_dimensions(tmp_path):
    from decideonce import train as TR
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    p = store.get_point("refund.action")
    _gate_fixture(store, "same", _rec(), _rec())
    _gate_fixture(store, "choice", _rec(), _rec(choice="deny"))
    _gate_fixture(store, "start", _rec(), _rec(at_start=False))
    _gate_fixture(store, "value", _rec(), _rec(amount=81))
    _gate_fixture(store, "event", _rec(), _rec(event=1))
    _gate_fixture(store, "none", None, None)  # both found no decision
    _gate_fixture(store, "half", _rec(), None)  # only pass 1 found one
    gate = TR.extraction_gate(cfg, store, p)
    assert gate["audited"] == 7
    # identical only: on a non-repeatable point a double skip is an
    # extractor failure, not agreement (B14)
    assert gate["agree"] == 1
    assert gate["agreement"] == pytest.approx(1 / 7)


def test_B14_both_none_agrees_only_when_repeatable(tmp_path):
    from decideonce import train as TR
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(point(repeatable=True))["ok"]
    p = store.get_point("refund.action")
    _gate_fixture(store, "same", _rec(), _rec())
    _gate_fixture(store, "none", None, None)
    gate = TR.extraction_gate(cfg, store, p)
    assert gate["agree"] == 2 and gate["agreement"] == pytest.approx(1.0)


def test_B14_repeatable_compares_full_identities(tmp_path):
    from decideonce import train as TR
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(point(repeatable=True))["ok"]
    p = store.get_point("refund.action")
    tid = "tr_multi"
    store.put_transcript({"transcript_id": tid, "ts": "2026-01-01T00:00:00Z",
                          "run_id": "multi", "workflow": "refund_request",
                          "model": "w", "data_class": "public", "input": "",
                          "events": [], "final_output": ""})
    store.put_extraction({"extraction_id": "ex1m", "ts": "t",
                          "transcript_id": tid, "pass": 1})
    store.put_extraction({"extraction_id": "ex2m", "ts": "t",
                          "transcript_id": tid, "pass": 2})
    store.complete_extraction("ex1m", "spark", 0, 0, {}, [])
    store.complete_extraction("ex2m", "spark", 0, 0, {}, [])
    for i, (choice, ev) in enumerate([("full_refund", 0), ("deny", 0)]):
        store.insert_decision({
            "decision_id": f"dm{i}", "point_id": "refund.action",
            "point_version": 1, "ts": "t", "context": {"amount_eur": 80},
            "state_text": "s", "options": ["full_refund", "deny"],
            "allowed_options": ["full_refund", "deny"], "mode": "extracted",
            "source": "extracted", "extracted_choice": choice,
            "run_id": "multi", "transcript_id": tid,
            "extraction_id": "ex1m", "extractor": "spark",
            "decision_pos": {"event": ev, "offset": 1}, "at_start": True})
    for i, (choice, ev) in enumerate([("full_refund", 0), ("deny", 1)]):
        store.audit_insert({"audit_id": f"eam{i}", "ts": "t",
                            "transcript_id": tid, "extraction_id": "ex2m",
                            "point_id": "refund.action",
                            "record": {"choice": choice, "at_start": True,
                                       "decision_pos": {"event": ev,
                                                        "offset": 1},
                                       "context": {"amount_eur": 80}}})
    gate = TR.extraction_gate(cfg, store, p)
    # same choice multiset, but the deny moved events -> disagree
    assert gate["agree"] == 0


def test_B14_agreement_compares_all_context_fields(tmp_path):
    from decideonce import train as TR
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(point(
        context_fields={"amount_eur": {"type": "number"},
                        "note": {"type": "string"}},
        required_fields=["amount_eur"]))["ok"]
    p = store.get_point("refund.action")
    a = _rec()
    b = _rec()
    a["context"]["note"] = "x"
    b["context"]["note"] = "y"  # optional field differs -> disagree
    _gate_fixture(store, "opt", a, b)
    gate = TR.extraction_gate(cfg, store, p)
    assert gate["agree"] == 0


# -- T-EX8: precedence + run_outcome labels nothing -------------------------

def test_T_EX8_outcome_over_teacher_over_extracted(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="prec"))
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [clean_record(choice="deny")]})
    did = r["accepted"][0]["decision_id"]
    row = store.get_decision(did)
    assert L.resolve_label(row)[:2] == ("deny", "extracted")
    assert svc.do_record({"decision_id": did,
                          "choice": "full_refund"})["ok"]
    row = store.get_decision(did)
    assert L.resolve_label(row)[:2] == ("full_refund", "teacher")
    assert svc.do_outcome({"decision_id": did, "ok": True})["ok"]
    row = store.get_decision(did)
    assert L.resolve_label(row)[:2] == ("full_refund", "outcome_ok")
    r2 = svc.do_outcome({"decision_id": did, "correct_choice": "deny"})
    assert r2["ok"] and r2["ignored"] is True  # first outcome wins (A2)
    row = store.get_decision(did)
    assert L.resolve_label(row)[:2] == ("full_refund", "outcome_ok")
    # outcome_correct still outranks outcome_ok within a single outcome
    c2 = extract1(svc, transcript(run_id="prec2"))
    r = svc.do_extract({"extraction_id": c2["extraction_id"],
                        "extractor": "spark",
                        "records": [clean_record(choice="deny")]})
    did2 = r["accepted"][0]["decision_id"]
    assert svc.do_record({"decision_id": did2,
                          "choice": "full_refund"})["ok"]
    assert svc.do_outcome({"decision_id": did2, "ok": True,
                           "correct_choice": "deny"})["ok"]
    row2 = store.get_decision(did2)
    assert L.resolve_label(row2)[:2] == ("deny", "outcome_correct")


def test_T_EX8_run_outcome_labels_no_decision(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    t = transcript(run_id="ro", run_outcome={"ok": False, "note": "user left"})
    r = extract1(svc, t)
    assert r["ok"], r
    stored = store.get_transcript(r["transcript_id"])
    assert stored["run_outcome"] == {"ok": False, "note": "user left"}
    r2 = svc.do_extract({"extraction_id": r["extraction_id"],
                         "extractor": "spark", "records": [clean_record()]})
    row = store.get_decision(r2["accepted"][0]["decision_id"])
    assert row["outcome"] is None
    assert L.resolve_label(row)[:2] == ("full_refund", "extracted")


# -- T-EX9: run-split -------------------------------------------------------

def test_T_EX9_runs_stay_whole_across_split(tmp_path):
    svc, _, _ = make_service(tmp_path, train={"min_train": 1,
                                               "min_per_option": 1,
                                               "holdout": 0.25})
    assert svc.do_register(point())["ok"]
    for i in range(10):  # run A first (older)
        d = svc.do_decide({"point_id": "refund.action", "run_id": "runA",
                           "context": {"amount_eur": 80 + i, "reason": "x"}})
        svc.do_record({"decision_id": d["decision_id"],
                       "choice": ["full_refund", "deny"][i % 2]})
    for i in range(6):  # run B last
        d = svc.do_decide({"point_id": "refund.action", "run_id": "runB",
                           "context": {"amount_eur": 90 + i, "reason": "y"}})
        svc.do_record({"decision_id": d["decision_id"],
                       "choice": ["full_refund", "deny"][i % 2]})
    up = _launch(svc)
    # row-split would hold out ~4 rows; run-split holds the whole last run
    tr = [r for r in up if r["split"] == "train"]
    ho = [r for r in up if r["split"] == "heldout"]
    assert len(tr) == 10 and len(ho) == 6
    svc_store = svc.store
    runs_tr = {svc_store.get_decision(r["decision_id"])["run_id"] for r in tr}
    runs_ho = {svc_store.get_decision(r["decision_id"])["run_id"] for r in ho}
    assert runs_tr == {"runA"} and runs_ho == {"runB"}


# -- T-EX10: extracted rows never in the §6.4 evidence pool -----------------

def _hinted_transcript(i, choice):
    return transcript(
        run_id=f"ex10-{i}",
        input=f"Ticket T-{i}.\nlabel_hint: {choice}\nAmount: 80 EUR.")


def _rec_for(t, choice):
    ev0 = t["events"][0]["content"]
    q_choice = "I'll refund the full 80 EUR"
    q_amt = "Amount: 80 EUR"
    return {"point_id": "refund.action", "choice": choice,
            "choice_quote": q_choice,
            "decision_pos": {"event": 0, "offset": ev0.find(q_choice)},
            "context": {"amount_eur": 80},
            "evidence": {"amount_eur": {
                "pos": {"event": -1, "offset": t["input"].find(q_amt)},
                "quote": q_amt}}}


def test_T_EX10_extracted_never_in_evidence_pool(tmp_path):
    svc, store, _ = make_service(tmp_path, train={"min_train": 1,
                                                   "min_per_option": 1},
                                 serve={"min_auto_support": 3},
                                 hosted_jev={"fit_rows": 5},
                                 extract={"min_audited": 1, "min_agreement": 0.9,
                                          "audit_rate": 1.0})
    p = point()
    p["context_fields"]["label_hint"] = {"type": "string"}
    assert svc.do_register(p)["ok"]
    assert svc.do_promote({"point_id": "refund.action",
                           "phase": "shadow"})["ok"]
    for i in range(12):  # live shadow rows, hinted
        choice = ["full_refund", "deny"][i % 2]
        d = svc.do_decide({"point_id": "refund.action",
                           "context": {"amount_eur": 80, "reason": f"ticket {i}",
                                       "label_hint": choice}})
        svc.do_record({"decision_id": d["decision_id"], "choice": choice})
    for i in range(8):  # extracted rows, hinted in the ticket input
        choice = ["full_refund", "deny"][i % 2]
        t = _hinted_transcript(i, choice)
        c1 = extract1(svc, t)
        assert c1["audit_required"] is True
        r = svc.do_extract({"extraction_id": c1["extraction_id"],
                            "extractor": "spark", "records": [_rec_for(t, choice)]})
        assert r["rejected"] == [], r
        # pass 2 is only for drawn transcripts (B3): audit all eight
        c2 = svc.do_extract({"transcript_id": c1["transcript_id"], "pass": 2})
        r2 = svc.do_extract({"extraction_id": c2["extraction_id"],
                             "extractor": "spark",
                             "records": [_rec_for(t, choice)]})
        assert r2["rejected"] == [], r2
    from decideonce import train as TR

    gate = TR.extraction_gate(svc.cfg, store, store.get_point("refund.action"))
    assert gate["passed"] is True
    # fit on the first 5 live rows; evidence is the 7 later live rows only
    checks = svc._serve_checks(store.get_point("refund.action"), True)
    assert "7 labelled evidence rows" in checks["min_examples"]["detail"]


# -- T-EX11: jev_disagreements only at or above the threshold ---------------

def test_T_EX11_tiered_extract_lists_no_probe_disagreements(tmp_path):
    # Mode C audit needs the probe's raw backend; tiered extract (no raw
    # backend) completes without disagreements instead of crashing.
    svc, store, _ = make_service(tmp_path, train={"min_train": 1,
                                                   "min_per_option": 1})
    assert svc.do_register(point())["ok"]
    t = transcript(run_id="nc-1")
    c1 = extract1(svc, t)
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark",
                        "records": [_rec_for(t, "deny")]})
    assert r["rejected"] == [], r
    assert r["jev_disagreements"] == []


# -- T-EX12: new_points never auto-registered --------------------------------

def _candidate():
    return {"draft": {"point_id": "refund.new", "workflow": "refund_request",
                      "question": "New?", "kind": "choice",
                      "options": ["a", "b"]},
            "answers": {"R1": "yes", "R2": "yes", "R3": "rule", "R4": 500,
                        "R5": "yes", "R6": {"sensitive": False, "covered": True},
                        "R7": "no", "R8": {}}}


def test_T_EX12_new_points_scored_never_registered(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="np"))
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [clean_record()],
                        "new_points": [_candidate(), {"nope": True}]})
    assert r["rejected"] == [], r
    assert r["new_points"][0]["verdict"] == "good_fit"
    assert r["new_points"][1]["verdict"] == "invalid"
    assert store.get_point("refund.new") is None
    assert [p["point_id"] for p in store.list_points()] == ["refund.action"]


# -- T-EX13: adapters round-trip ---------------------------------------------

def test_T_EX13_plain_jsonl_adapters_round_trip(tmp_path):
    t = adapt_plain("hello log", "run-p", "refund_request", model="wonder",
                    data_class="public", input="in", final_output="out")
    assert validate_transcript(t) == []
    assert t["events"] == [{"i": 0, "type": "text", "content": "hello log"}]
    back = adapt_jsonl(json.dumps(t) + "\n" + json.dumps(t).replace(
        "run-p", "run-q") + "\n")
    assert len(back) == 2
    assert back[0] == t
    assert back[1]["run_id"] == "run-q"
    assert validate_transcript(back[1]) == []


def test_T_EX13_muse_adapter_from_real_fixture():
    import pathlib
    src = pathlib.Path(__file__).parent / "fixtures" / "muse_exec_sample.jsonl"
    t = adapt_muse(src.read_text(), "refund_request")
    assert validate_transcript(t) == []
    assert t["run_id"] == "01a1996c-8f16-7920-aac2-f71692dd108e"
    assert t["model"] == "muse-spark-1.3-contributor"
    assert t["input"].startswith("Run the shell command")
    kinds = [e["type"] for e in t["events"]]
    assert "tool_result" in kinds and "text" in kinds
    tool = next(e for e in t["events"] if e["type"] == "tool_result")
    assert tool["name"] == "bash" and "hello-world" in tool["content"]
    assert t["final_output"] == "hello-world"
    assert t["cost"]["tokens_in"] == 47430
    assert t["cost"]["tokens_out"] == 386
    assert t["run_outcome"] == {"ok": True}


# -- T-EX14: transcript retention purge ---------------------------------------

def test_B13_purge_covers_events_only_extracted_text_and_plans(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="oldie2"))
    rec = clean_record()
    r = svc.do_extract({"extraction_id": c1["extraction_id"],
                        "extractor": "spark", "records": [rec]})
    assert r["accepted"]
    # events-only transcript: empty input/output but text in events
    store.put_transcript({"transcript_id": "tr_ev", "ts": "2020-01-01T00:00:00Z",
                          "run_id": "ev", "workflow": "refund_request",
                          "model": "w", "data_class": "private", "input": "",
                          "events": [{"i": 0, "content": "secret"}],
                          "final_output": ""})
    store.cx.execute("UPDATE transcripts SET ts='2020-01-01T00:00:00Z' "
                     "WHERE transcript_id=?", (c1["transcript_id"],))
    store.cx.execute("UPDATE decisions SET ts='2020-01-01T00:00:00Z' "
                     "WHERE transcript_id=?", (c1["transcript_id"],))
    store.put_plan({"plan_id": "pl_old", "ts": "2020-01-01T00:00:00Z",
                    "workflow": "refund_request", "input": "secret task",
                    "recommendation": "none", "model": "w", "points": [],
                    "audit": False, "pilot_forced": False, "conflicts": [],
                    "point_versions": {}, "final_output": "secret reply"})
    store.cx.commit()
    extract1(svc, transcript(run_id="fresh2"))  # any call purges
    assert store.get_transcript("tr_ev")["events"] == []
    row = store.decisions_for_transcript(c1["transcript_id"])[0]
    assert row["state_text"] is None
    # plan purge rides on plan calls
    svc.do_plan_run({"workflow": "refund_request", "input": "t", "context": {},
                     "run_model": "w", "cheap_model": "s"})
    got = store.get_plan("pl_old")
    assert got["input"] == "" and got["final_output"] == ""


def test_T_EX14_old_transcripts_purged_on_next_extract(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(point())["ok"]
    c1 = extract1(svc, transcript(run_id="oldie"))
    store.cx.execute("UPDATE transcripts SET ts='2020-01-01T00:00:00Z' "
                     "WHERE transcript_id=?", (c1["transcript_id"],))
    store.cx.commit()
    extract1(svc, transcript(run_id="fresh"))  # any call purges
    got = store.get_transcript(c1["transcript_id"])
    assert got["input"] == "" and got["events"] == [] and got["final_output"] == ""
    assert got["run_id"] == "oldie" and got["workflow"] == "refund_request"
    assert store.get_transcript(
        store.transcript_by_run("fresh")["transcript_id"])["input"] == INPUT
