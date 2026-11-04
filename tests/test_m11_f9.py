"""M11 wiring/money/privacy tests (F9): T-FW2/5/6/7/8/11 offline on fakes."""
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

import decideonce.fireworks_worker as W
from decideonce import autopilot as AP
from decideonce.config import ConfigError, load_config
from decideonce.deciders import build_deciders
from decideonce.deciders.base import DecideRequest, DeciderError
from decideonce.deciders.session import FireworksSessionDecider
from decideonce.fireworks_trainer import FireworksTrainer
from decideonce.service import Service
from decideonce.store import Store
from decideonce.trainer import FakeTrainer
from decideonce.trainer_log import redact_account

SCRIPT = str(Path(W.__file__).resolve())
CANARY = "fw_canary_9z9z9z9z9z9z"
ACCT = "acct-test"
OLD = "accounts/acct-test/models/dk-old"
NEW = "accounts/acct-test/models/dk-new"

CHILD = '''
import sys
mode = sys.argv[1]
if mode == "acctfail":
    print("denied for accounts/acct-test/models/m", file=sys.stderr,
          flush=True)
    sys.exit(1)
if mode == "prosefail":
    print("Auto-resolved account ID: acct-test", flush=True)
    for i in range(25):
        print(f"filler {i}", flush=True)
    print("denied for account acct-test", file=sys.stderr, flush=True)
    sys.exit(1)
'''


def _live_failed_job(tmp_path, mode):
    """Service + live fake child to terminal failed; returns svc, store, jid."""
    child = tmp_path / f"child-{mode}.py"
    child.write_text(CHILD)
    svc, store, cfg = _svc(tmp_path, trainer_kind="none", key=CANARY)
    cb, short = _cookbook(tmp_path)
    cfg["fireworks"].update({"cookbook_dir": str(cb),
                             "cookbook_commit": short})
    svc.trainer = FireworksTrainer(
        cfg, tmp_path / "jobs", key=CANARY,
        recipe=[sys.executable, str(child), mode])
    assert svc.do_register(_point())["ok"]
    _add_rows(svc, "f9.pt")
    r = svc.do_train({"point_id": "f9.pt", "force": True, "confirm": True})
    assert r["ok"] and r["status"] in ("running", "launching"), r
    jid = r["job_id"]
    deadline = time.monotonic() + 30
    while True:
        r = svc.do_train({"point_id": "f9.pt", "force": True,
                          "confirm": True})
        assert r["ok"], r
        if r["status"] == "failed":
            return svc, store, jid
        assert r["status"] in ("running", "launching"), r
        assert time.monotonic() < deadline, "live child never failed"
        time.sleep(0.3)


class _Runner:
    """Fake spawn: records (argv, env, cwd), runs nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, env, cwd):
        self.calls.append({"argv": argv, "env": dict(env), "cwd": cwd})


_CB_N = {"n": 0}


def _cookbook(tmp_path):
    _CB_N["n"] += 1
    cb = tmp_path / f"cb{_CB_N['n']}"
    py = cb / "training" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    py.chmod(0o755)
    (cb / "README.md").write_text("cookbook\n")
    env = {**os.environ, "GIT_CONFIG_NOSYSTEM": "1",
           "HOME": str(tmp_path)}
    subprocess.run(["git", "init", "-q", "-b", "main", str(cb)],
                   check=True, env=env, capture_output=True)
    subprocess.run(["git", "-C", str(cb), "add", "-A"],
                   check=True, env=env, capture_output=True)
    subprocess.run(["git", "-C", str(cb), "-c", "user.email=t@t",
                    "-c", "user.name=t", "commit", "-q", "--no-gpg-sign",
                    "-m", "x"], check=True, env=env, capture_output=True)
    short = subprocess.run(
        ["git", "-C", str(cb), "rev-parse", "--short", "HEAD"],
        check=True, capture_output=True, text=True).stdout.strip()
    return cb, short


def _svc(tmp_path, trainer_kind="fake", key=None, fw=None, **sections):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["train"]["min_train"] = 6
    fwo = {"monthly_budget_usd": 100.0, "allow_session_serving": True}
    fwo.update(fw or {})
    cfg["fireworks"].update(fwo)
    for section, kv in sections.items():
        cfg[section].update(kv)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    if trainer_kind == "fake":
        trainer = FakeTrainer(cfg)
    else:
        trainer = None
    svc = Service(cfg, store, deciders, rng=random.Random(0),
                  trainer=trainer, fireworks_key=key)
    return svc, store, cfg


def _point(pid="f9.pt", **kw):
    p = {"point_id": pid, "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def _add_rows(svc, pid, n=8, invert=False):
    for i in range(n):
        label = "ab"[i % 2]
        hint = "ab"[(i + 1) % 2] if invert else label
        r = svc.do_decide({"point_id": pid,
                           "context": {"label_hint": hint}})
        assert r["ok"], r
        rec = svc.do_record({"decision_id": r["decision_id"],
                             "choice": label})
        assert rec["ok"], rec


def _month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _session_svc(tmp_path, pid="f9.pt", point_kw=None, fake_args=("ok",),
                 serving=True, budget=5.0, key="k", argv_calls=None,
                 sections=None):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["train"]["min_train"] = 6
    cfg["fireworks"]["monthly_budget_usd"] = budget
    cfg["fireworks"]["allow_session_serving"] = serving
    for section, kv in (sections or {}).items():
        cfg[section].update(kv)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    argv = [sys.executable, SCRIPT, "fake", *[str(a) for a in fake_args]]

    def _argv(model):
        if argv_calls is not None:
            argv_calls.append(model)
        return list(argv)

    dec = FireworksSessionDecider(store, cfg, key, worker_argv_fn=_argv)
    deciders["private"] = dec
    svc = Service(cfg, store, deciders, rng=random.Random(0),
                  fireworks_key=key)
    assert svc.do_register(_point(pid, **(point_kw or {})))["ok"]
    return svc, store, cfg, dec


# -- T-FW2: confirm + dry_run -------------------------------------------------

def test_T_FW2_confirm_required_and_dry_run_clean(tmp_path):
    svc, store, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    _add_rows(svc, "f9.pt")
    r = svc.do_train({"point_id": "f9.pt", "force": True})
    assert r["ok"] is False and r["error"] == "confirm_required"
    assert store.jobs_running() == [] and store.jobs_launching() == []
    assert svc.trainer.uploaded == []
    r = svc.do_train({"point_id": "f9.pt", "force": True, "dry_run": True})
    assert r["ok"] and r["dry_run"] is True and r["kind"] == "smoke"
    assert r["estimate"]["estimate_usd"] > 0
    assert r["estimate"]["train_rows"] > 0
    assert svc.trainer.uploaded == []  # uploads nothing
    assert store.jobs_running() == [] and store.jobs_launching() == []


# -- T-FW5: key and account privacy -------------------------------------------

def test_T_FW5_redact_account_unit():
    assert redact_account(
        "using accounts/acct-test/models/m") == \
        "using accounts/acct-…/models/m"
    assert redact_account("Auto-resolved account ID: acct-test") == \
        "Auto-resolved account ID: acct-…"
    assert redact_account(
        "accounts/fireworks/models/muse-glimmer-30b") == \
        "accounts/fireworks/models/muse-glimmer-30b"
    assert redact_account("accounts/acct-test/models/x") == \
        "accounts/acct-…/models/x"
    assert redact_account("dk-abc123") == "dk-abc123"


def test_T_FW5_redact_gaps_unit():
    # accounts/<id> with no trailing slash.
    assert redact_account("denied for accounts/someuser-1a2b3c") == \
        "denied for accounts/acct-…"
    assert redact_account("base=accounts/fireworks") == \
        "base=accounts/fireworks"
    # the resume hint, with no Auto-resolved line anywhere.
    assert redact_account(
        "pass init_from_checkpoint='someuser-1a2b3c/run-x/c' next") == \
        "pass init_from_checkpoint='acct-…/run-x/c' next"
    # the literal id parsed from the Auto-resolved line is replaced
    # everywhere, including bare prose no safe pattern can match.
    text = ("Auto-resolved account ID: someuser-1a2b3c\n"
            "bill account someuser-1a2b3c for the run")
    assert redact_account(text) == (
        "Auto-resolved account ID: acct-…\n"
        "bill account acct-… for the run")
    # ... or passed by a caller that parsed the full log.
    assert redact_account("denied for account someuser-1a2b3c",
                          "someuser-1a2b3c") == \
        "denied for account acct-…"
    # idempotent: re-redaction never doubles the ellipsis.
    once = redact_account(text)
    assert redact_account(once) == once
    assert redact_account(
        "Auto-resolved account ID: acct-…") == \
        "Auto-resolved account ID: acct-…"
    # a suspiciously short id is left alone in prose (the line still
    # masks, so a mistyped id degrades to one masked line).
    assert redact_account("Auto-resolved account ID: ab\nsee ab") == \
        "Auto-resolved account ID: acct-…\nsee ab"


def test_T_FW5_tail_without_resolve_line_still_masked(tmp_path):
    # X2: the 20-line tail holds bare prose but not the Auto-resolved
    # line; status() threads the id parsed from the full log.
    svc, store, jid = _live_failed_job(tmp_path, "prosefail")
    log = (tmp_path / "jobs" / jid / "train.log").read_text()
    assert "Auto-resolved account ID: acct-test" in log
    assert "denied for account acct-test" in log
    tail = "\n".join(log.splitlines()[-20:])
    assert "Auto-resolved" not in tail
    err = store.job_get(jid)["error"]
    assert "acct-test" not in err, err
    assert "denied for account acct-…" in err, err


def test_T_FW5_key_only_in_child_env(tmp_path):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cb, short = _cookbook(tmp_path)
    cfg["fireworks"].update({"cookbook_dir": str(cb),
                             "cookbook_commit": short})
    runner = _Runner()
    tr = FireworksTrainer(cfg, tmp_path / "jobs", key=CANARY, runner=runner)
    rows = [{"point_id": "p", "split": s,
             "decision_id": f"d-{s}-{i}", "state": {"policy": "P"},
             "questions": {"p": {"type": "choice", "instructions": "Q?",
                                 "criteria": {"a": "Ay", "b": "Bee"},
                                 "label": "ab"[i % 2], "weight": 1.0}}}
            for i, s in enumerate(["train", "train", "calib", "heldout"])]
    job_id = tr.launch(rows, "smoke", "idem-f9-canary")
    assert runner.calls[0]["env"]["FIREWORKS_API_KEY"] == CANARY
    jd = tmp_path / "jobs" / job_id
    for name in ("run.sh", "meta.json", "train.jsonl", "dev.jsonl",
                 "heldout.jsonl"):
        assert CANARY not in (jd / name).read_text()
    assert CANARY not in " ".join(runner.calls[0]["argv"])


def test_T_FW5_failed_job_masks_account_everywhere(tmp_path):
    svc, store, jid = _live_failed_job(tmp_path, "acctfail")
    jd = tmp_path / "jobs" / jid
    assert ACCT in (jd / "train.log").read_text()  # raw: job dir only
    assert store.job_get(jid)["error"] is not None
    assert ACCT not in store.job_get(jid)["error"]
    assert "acct-…" in store.job_get(jid)["error"]
    rep = svc.do_report({"point_id": "f9.pt"})
    blob = json.dumps(rep)
    assert ACCT not in blob and CANARY not in blob
    md = svc.do_report({"point_id": "f9.pt", "format": "md"})["report"]
    assert ACCT not in md and CANARY not in md
    for f in (tmp_path / "proj").glob("decide.db*"):
        raw = f.read_bytes()
        assert CANARY.encode() not in raw
        assert ACCT.encode() not in raw


def test_T_FW5_no_key_names_it_and_collect_works(tmp_path):
    svc, store, _ = _svc(tmp_path, trainer_kind="none",
                         fw={"allow_session_serving": False})
    assert svc.do_register(_point(
        "f9.priv", data_class="private", private_model_only=True))["ok"]
    _add_rows(svc, "f9.priv")
    r = svc.do_train({"point_id": "f9.priv"})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"
    assert r["export"]["n_labeled"] == 8
    # collect needs no key and no model.
    r = svc.do_decide({"point_id": "f9.priv",
                       "context": {"label_hint": "a"}})
    assert r["ok"] and r["mode"] == "teacher"
    # shadow with a trained model but no key names the key, not the model.
    store.save_model("f9.priv", 1, "private", "v1-1.json",
                     {"model_ref": OLD, "temperature": 1.0, "job_id": "j"})
    svc.trainer = FakeTrainer(svc.cfg)
    svc.trainer.needs_key = True
    r = svc.do_promote({"point_id": "f9.priv", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"
    # without the serving flag the trained point reports blocked.
    rep = svc.do_report({"point_id": "f9.priv"})["points"][0]
    assert rep["private"]["blocked"] == "serving_not_allowed"
    md = svc.do_report({"point_id": "f9.priv", "format": "md"})["report"]
    assert "blocked(serving_not_allowed)" in md


def test_T_FW5_report_masks_live_model_ref(tmp_path):
    svc, store, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    store.save_model("f9.pt", 1, "private", "v1-9.json",
                     {"model_ref": f"accounts/{ACCT}/models/dk-x",
                      "temperature": 0.8, "job_id": "j9", "accuracy": 0.9,
                      "threshold": 0.5,
                      "metrics": {"n_holdout": 10}})
    rep = svc.do_report({"point_id": "f9.pt"})["points"][0]
    assert rep["private"]["model"]["model"] == \
        "accounts/acct-…/models/dk-x"
    md = svc.do_report({"point_id": "f9.pt", "format": "md"})["report"]
    assert "accounts/acct-…/models/dk-x" in md
    assert ACCT not in md
    # the card itself keeps the full ref.
    assert store.active_model("f9.pt", "private")["card"]["model_ref"] == \
        f"accounts/{ACCT}/models/dk-x"


# -- T-FW6: budget guard ------------------------------------------------------

def test_T_FW6_80_percent_warns(tmp_path):
    svc, store, _ = _svc(tmp_path, fw={"monthly_budget_usd": 1.0})
    assert svc.do_register(_point())["ok"]
    store.spend_add(_month(), "fireworks", 0.85)
    rep = svc.do_report({"point_id": "f9.pt"})["points"][0]["private"]
    assert rep["spend_month_usd"] == pytest.approx(0.85)
    assert rep["budget_usd"] == pytest.approx(1.0)
    assert rep["budget_warning_80"] is True
    md = svc.do_report({"point_id": "f9.pt", "format": "md"})["report"]
    assert "WARNING: over 80% of budget" in md


def test_T_FW6_100_percent_teacher_and_train_refuses(tmp_path):
    svc, store, _ = _svc(tmp_path, fw={"monthly_budget_usd": 1.0})
    assert svc.do_register(_point())["ok"]
    _add_rows(svc, "f9.pt")
    store.spend_add(_month(), "fireworks", 1.0)
    r = svc.do_train({"point_id": "f9.pt", "force": True, "confirm": True})
    assert r["ok"] is False and r["error"] == "budget_exceeded"
    assert svc.trainer.uploaded == []


def test_T_FW6_decide_over_budget_is_teacher(tmp_path):
    svc, store, _, dec = _session_svc(
        tmp_path, budget=1.0,
        point_kw={"data_class": "private", "private_model_only": True,
                  "remote_ok": ["fireworks"], "confirm": True})
    try:
        store.save_model("f9.pt", 1, "private", "v1-1.json",
                         {"model_ref": OLD, "temperature": 1.0,
                          "job_id": "j"})
        p = store.get_point("f9.pt")
        p["phase"] = "shadow"
        store.put_point(p)
        store.spend_add(_month(), "fireworks", 1.0)
        r = svc.do_decide({"point_id": "f9.pt",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"
        assert "budget_exceeded" in r["warnings"]
        row = store.get_decision(r["decision_id"])
        assert row["probe"] is None  # no worker call, nothing stored
    finally:
        dec.close()


# -- T-FW7: remote_ok ---------------------------------------------------------

def test_T_FW7_private_without_fireworks_never_sent(tmp_path):
    calls: list = []
    svc, store, _, dec = _session_svc(
        tmp_path, pid="f9.priv",
        point_kw={"data_class": "private", "private_model_only": True},
        argv_calls=calls)
    try:
        store.save_model("f9.priv", 1, "private", "v1-1.json",
                         {"model_ref": OLD, "temperature": 1.0,
                          "job_id": "j"})
        p = store.get_point("f9.priv")
        p["phase"] = "shadow"
        store.put_point(p)
        req = DecideRequest(
            point_id="f9.priv", state={"policy": "P", "label_hint": "a"},
            instructions="Which?", criteria={"a": "a", "b": "b"})
        with pytest.raises(DeciderError) as e:
            dec.decide([req])
        assert e.value.code == "remote_not_allowed"
        with pytest.raises(DeciderError) as e2:
            dec.decide_as(OLD, 1.0, [req])
        assert e2.value.code == "remote_not_allowed"
        assert calls == []  # no worker argv built, nothing spawned
        r = svc.do_decide({"point_id": "f9.priv",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"
        assert r["warnings"] == []
        assert calls == []
        assert store.get_decision(r["decision_id"])["probe"] is None
    finally:
        dec.close()


def test_T_FW7_disallowed_trigger_never_trained(tmp_path):
    svc, store, _ = _svc(tmp_path)
    assert svc.do_register(_point(
        "f9.priv", data_class="private", private_model_only=True))["ok"]
    _add_rows(svc, "f9.priv")
    joins = svc.autopilot.join_candidates()
    assert [j for j in joins if j["point_id"] == "f9.priv"][0][
        "allowed"] is False
    r = svc.do_train({"point_id": "f9.priv", "dry_run": True})
    assert r["ok"] and r["estimate"]["train_rows"] == 0
    r = svc.do_train({"point_id": "f9.priv", "confirm": True})
    assert r["ok"] is False and r["error"] == "remote_not_allowed"
    assert svc.trainer.uploaded == []


def test_T_FW7_boolean_remote_ok_reads_as_both(tmp_path):
    calls: list = []
    svc, store, _, dec = _session_svc(
        tmp_path, pid="f9.priv",
        point_kw={"data_class": "private",
                  "remote_ok": True, "confirm": True},
        argv_calls=calls)
    try:
        assert store.get_point("f9.priv")["remote_ok"] == [
            "fireworks", "hosted_jev"]
        store.save_model("f9.priv", 1, "private", "v1-1.json",
                         {"model_ref": OLD, "temperature": 1.0,
                          "job_id": "j"})
        p = store.get_point("f9.priv")
        p["tier"] = "private"
        p["phase"] = "shadow"
        store.put_point(p)
        r = svc.do_decide({"point_id": "f9.priv",
                           "context": {"label_hint": "a"}})
        assert r["ok"]
        assert calls == [OLD]  # the worker was built and called
        row = store.get_decision(r["decision_id"])
        assert row["probe"]["model_version"] == OLD
        assert row["probe"]["probabilities"]["a"] > 0.9
    finally:
        dec.close()


# -- T-FW8: retraining switch ---------------------------------------------------

def _switch_cards(store, pid, new_ref=NEW):
    store.save_model(pid, 1, "private", "v1-old.json",
                     {"model_ref": OLD, "temperature": 1.0,
                      "threshold": 0.5, "job_id": "j-old",
                      "export_seq": 0, "metrics": {}})
    mid = store.save_model(
        pid, 1, "private", "v1-new.json",
        {"model_ref": new_ref, "temperature": 1.0, "threshold": 0.5,
         "job_id": "j-new", "export_seq": 0,
         "metrics": {"accuracy": 1.0, "coverage": 1.0, "n_holdout": 10,
                     "error_at_threshold": 0.0}}, active=False)
    p = store.get_point(pid)
    p["phase"] = "serve"
    p["pending_private_model"] = {"model_id": mid, "job_id": "j-new"}
    store.put_point(p)
    return mid


def _switch_svc(tmp_path, pid="f9.sw"):
    return _session_svc(
        tmp_path, pid=pid,
        point_kw={"data_class": "private", "private_model_only": True,
                  "remote_ok": ["fireworks"], "confirm": True,
                  "target_error": 0.3},
        sections={"train": {"min_examples": 5},
                  "serve": {"min_shadow": 5, "audit_rate": 1.0}})


def test_T_FW8_serve_keeps_old_until_switch(tmp_path):
    svc, store, _, dec = _switch_svc(tmp_path)
    try:
        mid = _switch_cards(store, "f9.sw")
        for i in range(8):
            label = "ab"[i % 2]
            r = svc.do_decide({"point_id": "f9.sw",
                               "context": {"label_hint": label}})
            assert r["ok"] and r["mode"] == "teacher" and r.get("audit")
            rec = svc.do_record({"decision_id": r["decision_id"],
                                 "choice": label})
            assert rec["ok"], rec
        rows = [r for r in store.decisions_for_point("f9.sw")
                if r.get("pending_probe")]
        assert len(rows) == 8
        for r in rows:
            assert r["audit"] is True
            assert r["probe"]["model_version"] == OLD  # still serving old
            assert r["pending_probe"]["model_version"] == NEW  # logging new
        # §6.4 holds for the new checkpoint, but no confirm: no switch.
        r = svc.do_promote({"point_id": "f9.sw", "phase": "serve"})
        assert r["ok"] is False and r["failed"] == ["confirm"]
        assert store.active_model("f9.sw", "private")["card"][
            "model_ref"] == OLD
        r = svc.do_promote({"point_id": "f9.sw", "phase": "serve",
                            "confirm": True})
        assert r["ok"] and r["switched"] is True, r
        assert store.active_model("f9.sw", "private")["id"] == mid
        assert "pending_private_model" not in store.get_point("f9.sw")
        r = svc.do_decide({"point_id": "f9.sw",
                           "context": {"label_hint": "a"}})
        assert r["ok"]
        assert store.get_decision(
            r["decision_id"])["probe"]["model_version"] == NEW
        assert store.get_decision(
            r["decision_id"]).get("pending_probe") is None
    finally:
        dec.close()


def test_T_FW8_regressing_checkpoint_refused(tmp_path):
    svc, store, _, dec = _switch_svc(tmp_path, pid="f9.rg")
    try:
        _switch_cards(store, "f9.rg")
        for i in range(8):
            label = "ab"[i % 2]
            r = svc.do_decide({"point_id": "f9.rg",
                               "context": {"label_hint": label}})
            assert r["ok"]
            # inverted labels: every pending row is wrong.
            rec = svc.do_record({"decision_id": r["decision_id"],
                                 "choice": "ab"[(i + 1) % 2]})
            assert rec["ok"], rec
        r = svc.do_promote({"point_id": "f9.rg", "phase": "serve",
                            "confirm": True})
        assert r["ok"] is False
        assert r["error"] == "private_not_better", r
        assert store.active_model("f9.rg", "private")["card"][
            "model_ref"] == OLD
        assert "pending_private_model" in store.get_point("f9.rg")
    finally:
        dec.close()


def test_T_FW8_collect_and_shadow_switch_at_once(tmp_path):
    svc, store, _ = _svc(tmp_path)
    for pid in ("f9.c", "f9.s"):
        assert svc.do_register(_point(
            pid, data_class="private", private_model_only=True,
            remote_ok=["fireworks"], confirm=True))["ok"]
        _add_rows(svc, pid)
    store.save_model("f9.s", 1, "private", "v1-old.json",
                     {"model_ref": OLD, "temperature": 1.0, "job_id": "j0",
                      "sampler": "fakeckpt:j0"})
    p = store.get_point("f9.s")
    p["phase"] = "shadow"
    store.put_point(p)
    r = svc.do_train({"point_id": "f9.c", "confirm": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "smoke", r
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": "f9.c", "confirm": True})
    assert r["ok"] and r["status"] == "smoke_done", r
    r = svc.do_train({"point_id": "f9.c", "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full", r
    full = r["job_id"]
    svc.trainer.complete(full)
    r = svc.do_train({"point_id": "f9.c", "confirm": True})
    assert r["ok"] and r["status"] == "done", r
    for pid in ("f9.c", "f9.s"):
        assert store.active_model(pid, "private")["card"]["job_id"] == full
        assert "pending_private_model" not in store.get_point(pid)


# -- Z3: pending lifecycle ----------------------------------------------------------

def test_Z3_drift_demotion_clears_pending(tmp_path):
    svc, store, _, dec = _switch_svc(tmp_path, pid="f9.zd")
    try:
        _switch_cards(store, "f9.zd")
        assert store.get_point("f9.zd")["pending_private_model"]
        for i in range(100):
            label = "ab"[i % 2]
            r = svc.do_decide({"point_id": "f9.zd",
                               "context": {"label_hint": label}})
            assert r["ok"], r
            rec = svc.do_record({"decision_id": r["decision_id"],
                                 "choice": "ab"[(i + 1) % 2]})
            assert rec["ok"], rec
        assert store.get_point("f9.zd")["phase"] == "shadow"
        assert "pending_private_model" not in store.get_point("f9.zd")
        assert svc._pending_private_model(
            store.get_point("f9.zd")) is None
    finally:
        dec.close()


def test_Z3_manual_demotion_clears_pending(tmp_path):
    svc, store, _, dec = _switch_svc(tmp_path, pid="f9.zm")
    try:
        _switch_cards(store, "f9.zm")
        assert store.get_point("f9.zm")["pending_private_model"]
        r = svc.do_promote({"point_id": "f9.zm", "phase": "collect"})
        assert r["ok"], r
        assert "pending_private_model" not in store.get_point("f9.zm")
    finally:
        dec.close()


def test_Z3_reregister_carries_pending(tmp_path):
    svc, store, _, dec = _switch_svc(tmp_path, pid="f9.zr")
    try:
        mid = _switch_cards(store, "f9.zr")
        r = svc.do_register(_point(
            "f9.zr", data_class="private", private_model_only=True,
            remote_ok=["fireworks"], confirm=True, target_error=0.3))
        assert r["ok"], r
        assert store.get_point("f9.zr")["pending_private_model"] == {
            "model_id": mid, "job_id": "j-new"}
    finally:
        dec.close()


# -- Z4: shadow serving gate + plan label -------------------------------------------

def test_Z4_private_shadow_needs_serving_gate_and_budget(tmp_path):
    svc, store, cfg = _svc(tmp_path, fw={"allow_session_serving": False})
    assert svc.do_register(_point("f9.sh", private_model_only=True))["ok"]
    store.save_model("f9.sh", 1, "private", "v1-1.json",
                     {"model_ref": "m", "temperature": 1.0, "job_id": "j0"})
    r = svc.do_promote({"point_id": "f9.sh", "phase": "shadow"})
    assert r["ok"] is False and r["error"] == "serving_not_allowed", r
    cfg["fireworks"]["allow_session_serving"] = True
    cfg["fireworks"]["monthly_budget_usd"] = 0.0
    r = svc.do_promote({"point_id": "f9.sh", "phase": "shadow"})
    assert r["ok"] is False and r["failed"] == ["budget"], r
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    r = svc.do_promote({"point_id": "f9.sh", "phase": "shadow"})
    assert r["ok"], r


def test_Z4_plan_gp11_label(tmp_path):
    from decideonce.deciders.session import FireworksSessionDecider

    svc, store, cfg = _svc(tmp_path, fw={"allow_session_serving": False})
    assert svc.do_register(_point("f9.pl", private_model_only=True))["ok"]
    p = store.get_point("f9.pl")
    p["plan_phase"] = "shadow"
    store.put_point(p)
    svc.deciders["private"] = FireworksSessionDecider(store, cfg, "k")
    svc._plan_eligible_hosted = lambda _p: True  # orthogonal to the label
    _, warns = svc._plan_predict(store.get_point("f9.pl"), "do it", {}, False)
    assert warns == ["f9.pl: serving_not_allowed"]


# -- T-FW11: paid-job gate ------------------------------------------------------

def _trained_points(svc, *pids):
    for pid in pids:
        assert svc.do_register(_point(pid))["ok"]
        _add_rows(svc, pid)


def test_T_FW11_first_launch_is_smoke_and_full_needs_free_flag(tmp_path):
    svc, store, _ = _svc(tmp_path)
    _trained_points(svc, "f9.g")
    r = svc.do_train({"point_id": "f9.g", "force": True, "dry_run": True})
    assert r["ok"] and r["kind"] == "smoke"
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "smoke", r
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "smoke_done", r
    assert store.smoke_done_exists()
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True})
    assert r["ok"] is False and r["error"] == "paid_job_not_approved"
    assert r["estimate_usd"] > 0


def test_T_FW11_after_cutoff_needs_paid_approval(tmp_path, monkeypatch):
    svc, _, cfg = _svc(tmp_path)
    _trained_points(svc, "f9.g")
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True})
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True})
    assert r["status"] == "smoke_done", r
    monkeypatch.setattr(AP, "MAIN_FREE_CUTOFF_UTC",
                        datetime(2020, 6, 1, tzinfo=timezone.utc))
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] is False and r["error"] == "paid_job_not_approved"
    cfg["fireworks"]["allow_paid_training"] = True
    r = svc.do_train({"point_id": "f9.g", "force": True, "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full", r


def test_T_FW11_budget_zero_blocks_everything(tmp_path):
    svc, store, _, dec = _session_svc(tmp_path, budget=0.0)
    try:
        _add_rows(svc, "f9.pt")
        svc.trainer = FakeTrainer(svc.cfg)
        r = svc.do_train({"point_id": "f9.pt", "force": True,
                          "confirm": True})
        assert r["ok"] is False and r["error"] == "budget_exceeded"
        assert svc.trainer.uploaded == []
        rep = svc.do_report({})["fireworks"]
        assert rep["serving_allowed"] is False
    finally:
        dec.close()


def test_T_FW11_leftover_jev_config_refused(tmp_path):
    jev_toml = tmp_path / "jev.toml"
    jev_toml.write_text('[jev]\nurl = "http://127.0.0.1:8090"\n')
    with pytest.raises(ConfigError) as e:
        load_config(jev_toml)
    assert e.value.code == "config_v16_migration"


# -- Y1/Y2: idem validation + reuse ----------------------------------------------

def test_Y1_do_train_idem_schema(tmp_path):
    svc, _, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    for bad in ("../../evil-12345", "short", "has space!!"):
        r = svc.do_train({"point_id": "f9.pt", "force": True,
                          "confirm": True, "idem": bad})
        assert r["ok"] is False and r["error"] == "invalid_args", (bad, r)


def test_Y2_sequential_reuse_refused(tmp_path):
    svc, _, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    _add_rows(svc, "f9.pt")
    r = svc.do_train({"point_id": "f9.pt", "force": True, "confirm": True,
                      "idem": "idem-reuse-1"})
    assert r["status"] == "running"
    c = svc.do_train({"point_id": "f9.pt", "force": True, "cancel": True})
    assert c["status"] == "cancelled"
    r2 = svc.do_train({"point_id": "f9.pt", "force": True, "confirm": True,
                       "idem": "idem-reuse-1"})
    assert r2["ok"] is False and r2["error"] == "idem_in_use", r2
    assert len(svc.trainer.uploaded) == 1


# -- Y10: cancel races ------------------------------------------------------------

def _y10_row(idem, train_usd=1.5, counted_usd=2.0):
    return {"job_id": f"launching-{idem[:12]}", "kind": "smoke",
            "status": "launching", "estimate_usd": 2.0,
            "counted_usd": counted_usd, "train_usd": train_usd,
            "points": [{"point_id": "f9.pt"}], "autopilot": False,
            "created_at": "2026-11-04T00:00:00Z",
            "idempotency_key": idem}


def test_Y10_cancel_loses_to_eval_claim(tmp_path):
    svc, store, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    store.job_insert_launching(_y10_row("idem-eval-race-lost"))
    jid = "launching-idem-eval-ra"
    store.job_update(jid, {"status": "evaluating"})
    stale = store.job_get(jid)
    assert stale["status"] == "evaluating"
    assert store.job_claim_evaluated(jid)  # the eval thread wins
    assert svc.autopilot._mark_cancelled(stale) is False
    assert store.job_get(jid)["status"] == "done"
    assert store.spend_get(_month(), "fireworks")["usd"] == 0


def test_Y10_cancel_wins_on_evaluating(tmp_path):
    svc, store, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    store.job_insert_launching(_y10_row("idem-eval-race-won"))
    jid = "launching-idem-eval-ra"
    store.job_update(jid, {"status": "evaluating"})
    assert svc.autopilot._mark_cancelled(store.job_get(jid)) is True
    row = store.job_get(jid)
    assert row["status"] == "cancelled"
    assert row["spend_usd"] == pytest.approx(1.5)
    # Y9: the launch estimate stays for the monthly cap (sums estimates).
    assert row["counted_usd"] == pytest.approx(2.0)
    assert store.spend_get(_month(), "fireworks")["usd"] == pytest.approx(1.5)


def test_Y10_launch_loses_claim_to_cancel(tmp_path):
    svc, store, _ = _svc(tmp_path)
    assert svc.do_register(_point())["ok"]
    _add_rows(svc, "f9.pt")
    real_launch = svc.trainer.launch
    real_cancel = svc.trainer.cancel
    kills: list = []

    def launch(rows, kind, idem=""):
        # a concurrent cancel lands mid-spawn, before our claim.
        store.job_update(f"launching-{idem[:12]}", {"status": "cancelled",
                                                   "error": "cancelled by user"})
        return real_launch(rows, kind, idem)

    def cancel(job_id):
        kills.append(job_id)
        return real_cancel(job_id)

    svc.trainer.launch = launch
    svc.trainer.cancel = cancel
    idem = "idem-cancel-race"
    r = svc.do_train({"point_id": "f9.pt", "force": True, "confirm": True,
                      "idem": idem})
    assert r["ok"] and r["status"] == "cancelled", r
    assert r["job_id"] == f"launching-{idem[:12]}"
    assert len(kills) == 1  # our just-spawned child was killed, not orphaned
    assert svc.trainer.jobs[kills[0]]["status"] == "cancelled"
