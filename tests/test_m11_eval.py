"""M11 background-evaluation tests (F8): evaluating flow, T fit, GP7 gate."""
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
from decideonce import hostedfit as HF
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.deciders.session import FireworksSessionDecider
from decideonce.fireworks_trainer import FireworksTrainer
from decideonce.fireworks_worker import tokens_of
from decideonce.service import Service
from decideonce.store import Store

SCRIPT = str(Path(W.__file__).resolve())
MODEL = "accounts/acct-test/models/dk-t-0001"
RECIPE_T = 0.79

FULL_LOG = (
    '{"type": "metrics", "step": 10, "kway_loss": 0.41}\n'
    '{"type": "metrics", "step": 10, "eval_loss": 0.35, '
    '"eval_accuracy": 0.88}\n'
    "tasks=150\n"
    "Decision calibration: T={'choice': 0.79}\n"
    "Sampler checkpoint 'ckpt-1' saved\n"
    "Promoting session checkpoint 'ckpt-1' -> model "
    f"'{MODEL}'\n"
    "Promoted!\n")

SMOKE_LOG = (
    '{"type": "metrics", "step": 2, "eval_loss": 0.61, '
    '"eval_accuracy": 0.71}\n'
    "tasks=32\n"
    "Decision calibration: T={'choice': 1.0}\n"
    "Sampler checkpoint 'ckpt-s' saved\n")


class _Runner:
    def __call__(self, argv, env, cwd):
        pass


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


def _setup(tmp_path, fake_args=("ok",), budget=100.0, eval_ready=600):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["fireworks"]["monthly_budget_usd"] = budget
    cfg["fireworks"]["allow_paid_training"] = True
    cfg["fireworks"]["allow_session_serving"] = True
    cfg["fireworks"]["eval_ready_s"] = eval_ready
    cb, short = _cookbook(tmp_path)
    cfg["fireworks"].update({"cookbook_dir": str(cb),
                             "cookbook_commit": short})
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    trainer = FireworksTrainer(cfg, tmp_path / "jobs", key="k",
                               runner=_Runner())
    svc = Service(cfg, store, deciders, rng=random.Random(0),
                  trainer=trainer, fireworks_key="k")
    svc.autopilot.worker_argv_fn = lambda _m: [
        sys.executable, SCRIPT, "fake", *[str(a) for a in fake_args]]
    assert svc.do_register(
        {"point_id": "e.pt", "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b", "c"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"})["ok"]
    return svc, store, cfg


def _add_row(svc, label, hint=None):
    ctx = {"label_hint": label if hint is None else hint}
    r = svc.do_decide({"point_id": "e.pt", "context": ctx})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec


def _finish_smoke(svc, pid="e.pt"):
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "smoke", r
    jd = Path(svc.cfg["store"]["data_dir"]) / "jobs" / r["job_id"]
    (jd / "train.log").write_text(SMOKE_LOG)
    (jd / "exit_code").write_text("0")
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "smoke_done", r
    return r["job_id"]


def _launch_full(svc, pid="e.pt"):
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full", r
    return r["job_id"]


def _finish_full(svc, job_id, log=FULL_LOG):
    jd = Path(svc.cfg["store"]["data_dir"]) / "jobs" / job_id
    # Y12: the log must promote the id meta.json asked for, like the
    # real recipe does; splice the job's own --output-model-id in.
    argv = json.loads((jd / "meta.json").read_text())["argv"]
    mid = argv[argv.index("--output-model-id") + 1]
    (jd / "train.log").write_text(log.replace("dk-t-0001", mid))
    (jd / "exit_code").write_text("0")
    return f"accounts/acct-test/models/{mid}"


def _poll_eval(svc, pid="e.pt"):
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "evaluating", r
    return r["job_id"]


def _file_rows(svc, job_id, name):
    jd = Path(svc.cfg["store"]["data_dir"]) / "jobs" / job_id
    return [json.loads(line) for line in
            (jd / f"{name}.jsonl").read_text().splitlines()]


def _month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


# -- T-FW4 pure: the private temperature fit -----------------------------------

def _items(n_right, n_wrong):
    right = [([-0.105, -8.0, -8.0], 0)]
    wrong = [([-8.0, -0.105, -8.0], 0)]
    return right * n_right + wrong * n_wrong


def test_T_FW4_fit_lowers_nll():
    items = _items(4, 6)
    T, skipped = HF.fit_temperature_private(items, RECIPE_T)
    assert skipped is None
    assert 0.05 <= T <= 10.0
    assert T != pytest.approx(1.0)
    assert HF.nll_at_logits(items, T) < HF.nll_at_logits(items, 1.0)


def test_T_FW4_fallback_below_5_errors():
    assert HF.fit_temperature_private(_items(6, 4), RECIPE_T) == (
        RECIPE_T, "too_few_errors")
    assert HF.fit_temperature_private([], RECIPE_T) == (
        RECIPE_T, "too_few_errors")
    for bad in (None, "x", -1.0, float("nan")):
        assert HF.fit_temperature_private(_items(6, 4), bad) == (
            1.0, "too_few_errors")


# -- live status enrichment -----------------------------------------------------

def test_eval_status_enriches_from_log(tmp_path):
    svc, _, _ = _setup(tmp_path)
    tr = svc.trainer
    jd = Path(svc.cfg["store"]["data_dir"]) / "jobs" / "jd1-eval-0001"
    jd.mkdir(parents=True)
    (jd / "meta.json").write_text(json.dumps(
        {"kind": "full", "n_train": 150, "estimate": {"eval_rows": 40},
         "argv": ["recipe", "--output-model-id", "dk-t-0001"]}))
    (jd / "train.log").write_text(FULL_LOG)
    (jd / "exit_code").write_text("0")
    st = tr.status("jd1-eval-0001")
    assert st["status"] == "done"
    assert st["model"] == MODEL
    assert st["recipe_temperature"] == pytest.approx(RECIPE_T)
    assert st["metrics"]["eval_accuracy"] == pytest.approx(0.88)
    assert st["metrics"]["n_train"] == 150
    rows = tr.eval_rows("jd1-eval-0001")
    assert rows == {"calib": [], "heldout": []}  # missing files, no crash


# -- T-FW4 integration: eval fits T and writes the card --------------------------

def _heldout_hint_accuracy(svc, job_id):
    n = ok = 0
    for r in _file_rows(svc, job_id, "heldout"):
        q = next(iter(r["questions"].values()))
        keys = list(q["criteria"])
        hint = r["state"].get("label_hint")
        if hint in keys and q["label"] in keys:
            n += 1
            ok += hint == q["label"]
    return ok / n, n


def test_T_FW4_eval_fits_T(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        label = "abc"[i % 3]
        hint = label if i < 45 else "abc"[(i + 1) % 3]  # late rows wrong
        _add_row(svc, label, hint)
    sid = _finish_smoke(svc)
    smoke_counted = store.job_get(sid)["counted_usd"]
    jid = _launch_full(svc)
    launch_counted = store.job_get(jid)["counted_usd"]
    want_model = _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert svc.autopilot.join_eval(jid, timeout=60)
    job = store.job_get(jid)
    assert job["status"] == "done" and job["finalized"]
    # Y9: the monthly cap sums launch estimates, so counted never resets.
    assert job["counted_usd"] == pytest.approx(launch_counted)
    assert job["train_usd"] > 0
    card = store.active_model("e.pt", "private")["card"]
    assert card["scoring"] == "session_forward"
    assert card["model_ref"] == want_model
    assert card["temperature_source"] == "fit"
    assert card["temperature"] != pytest.approx(RECIPE_T)
    assert card["recipe_temperature"] == pytest.approx(RECIPE_T)
    assert card["prompt_renderer"].startswith("cookbook@")
    want_acc, n = _heldout_hint_accuracy(svc, jid)
    assert n > 0
    assert card["accuracy"] == pytest.approx(want_acc)
    assert card["metrics"]["n_holdout"] == n
    assert card["metrics"]["n_calib"] > 0
    # T-T6: random-escalation error is the overall error.
    assert card["metrics"]["random_escalation_error"] == pytest.approx(
        1.0 - card["metrics"]["accuracy"])
    # No double counting: smoke counted + train ledger + settled eval.
    assert store.spend_get(_month(), "fireworks")["usd"] == pytest.approx(
        smoke_counted + job["train_usd"]
        + card["metrics"]["eval_cost_usd"])


def test_T_FW4_eval_recipe_fallback(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])  # all hints right: 0 errors
    _finish_smoke(svc)
    jid = _launch_full(svc)
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert svc.autopilot.join_eval(jid, timeout=60)
    card = store.active_model("e.pt", "private")["card"]
    assert card["temperature_source"] == "recipe"
    assert card["temperature"] == pytest.approx(RECIPE_T)
    assert card["accuracy"] == pytest.approx(1.0)
    assert card["metrics"]["ece"] == pytest.approx(0.0, abs=0.2)


# -- T-FW3: failure leaves the old model -----------------------------------------

def _old_card(store):
    store.save_model("e.pt", 1, "private", "v1-1.json",
                     {"model_ref": "old-model", "temperature": 1.0,
                      "job_id": "j0"})
    d = store.models_dir / "e.pt"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_T_FW3_failed_job_leaves_old_model(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    models = _old_card(store)
    before = sorted(p.name for p in models.iterdir())
    _finish_smoke(svc)
    jid = _launch_full(svc)
    jd = Path(svc.cfg["store"]["data_dir"]) / "jobs" / jid
    (jd / "train.log").write_text("boom\n")
    (jd / "exit_code").write_text("1")
    r = svc.do_train({"point_id": "e.pt", "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "failed", r
    assert store.job_get(jid)["status"] == "failed"
    assert store.active_model("e.pt", "private")["card"]["job_id"] == "j0"
    assert sorted(p.name for p in models.iterdir()) == before


def test_T_FW3_failed_eval_leaves_old_model(tmp_path):
    svc, store, _ = _setup(tmp_path, fake_args=("error-after", "1"))
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    models = _old_card(store)
    before = sorted(p.name for p in models.iterdir())
    sid = _finish_smoke(svc)
    smoke_counted = store.job_get(sid)["counted_usd"]
    jid = _launch_full(svc)
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert svc.autopilot.join_eval(jid, timeout=60)
    job = store.job_get(jid)
    assert job["status"] == "eval_failed"
    assert "fake boom" in (job["error"] or "")
    assert store.active_model("e.pt", "private")["card"]["job_id"] == "j0"
    assert sorted(p.name for p in models.iterdir()) == before
    # Train ledger recorded, first call's spend stays, nothing else.
    calib_toks = sum(tokens_of([r]) for r in _file_rows(svc, jid, "dev"))
    assert store.spend_get(_month(), "fireworks")["usd"] == pytest.approx(
        smoke_counted + job["train_usd"] + calib_toks * 1.96 / 1e6)


# -- T-PT10: lock, restart, retry --------------------------------------------------

def test_T_PT10_no_launch_while_evaluating(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    _finish_smoke(svc)
    jid = _launch_full(svc)
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert store.job_get(jid)["status"] == "evaluating"
    # A second launch sees the live job instead of launching.
    r = svc.do_train({"point_id": "e.pt", "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "evaluating" and r["job_id"] == jid
    assert store.jobs_launching() == []
    assert svc.autopilot.join_eval(jid, timeout=60)
    assert store.job_get(jid)["status"] == "done"


def test_T_PT10_restart_and_retry(tmp_path):
    svc, store, _ = _setup(tmp_path, fake_args=("slow", "2"))
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    sid = _finish_smoke(svc)
    smoke_counted = store.job_get(sid)["counted_usd"]
    jid = _launch_full(svc)
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    # The first worker call takes 2 s; at 0.5 s the thread is mid-call.
    time.sleep(0.5)
    # Restart mid-eval: the stuck job becomes eval_failed ...
    out = svc.autopilot.recover_interrupted_evals()
    assert [a["action"] for a in out] == ["eval_failed"]
    job = store.job_get(jid)
    assert job["status"] == "eval_failed" and job["error"] == "server_restart"
    assert svc.autopilot.join_eval(jid, timeout=60)  # thread exits quietly
    mid = store.spend_get(_month(), "fireworks")["usd"]
    calib_toks = sum(tokens_of([r]) for r in _file_rows(svc, jid, "dev"))
    held_toks = sum(tokens_of([r]) for r in _file_rows(svc, jid, "heldout"))
    # ... the in-flight call still settles (it ran), then retry re-runs.
    assert mid == pytest.approx(
        smoke_counted + job["train_usd"] + calib_toks * 1.96 / 1e6)
    assert svc.do_train(
        {"point_id": "e.pt", "retry_eval": True,
         "force": True})["error"] == "confirm_required"
    r = svc.do_train({"point_id": "e.pt", "retry_eval": True,
                      "confirm": True, "force": True})
    assert r["ok"] and r["status"] == "evaluating" and r["job_id"] == jid
    assert svc.autopilot.join_eval(jid, timeout=120)
    assert store.job_get(jid)["status"] == "done"
    assert store.active_model("e.pt", "private")["card"]["job_id"] == jid
    assert store.spend_get(_month(), "fireworks")["usd"] == pytest.approx(
        mid + (calib_toks + held_toks) * 1.96 / 1e6)
    r = svc.do_train({"point_id": "e.pt", "retry_eval": True,
                      "confirm": True, "force": True})
    assert r["error"] == "nothing_to_retry"


def test_eval_cancel_mid_run(tmp_path):
    svc, store, _ = _setup(tmp_path, fake_args=("slow", "2"))
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    sid = _finish_smoke(svc)
    smoke_counted = store.job_get(sid)["counted_usd"]
    jid = _launch_full(svc)
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    # The first worker call takes 2 s; at 0.5 s the thread is mid-call.
    time.sleep(0.5)
    r = svc.do_train({"point_id": "e.pt", "cancel": True, "force": True})
    assert r["ok"] and r["status"] == "cancelled", r
    assert svc.autopilot.join_eval(jid, timeout=60)
    job = store.job_get(jid)
    assert job["status"] == "cancelled"
    assert job["spend_usd"] == pytest.approx(job["train_usd"])
    assert store.active_model("e.pt", "private") is None
    calib_toks = sum(tokens_of([r]) for r in _file_rows(svc, jid, "dev"))
    assert store.spend_get(_month(), "fireworks")["usd"] == pytest.approx(
        smoke_counted + job["train_usd"] + calib_toks * 1.96 / 1e6)


# -- T-PT9: GP7 private plan gate --------------------------------------------------

def _psetup(tmp_path):
    svc, store, cfg = _setup(tmp_path)
    dec = FireworksSessionDecider(
        store, cfg, "k",
        worker_argv_fn=lambda _m: [sys.executable, SCRIPT, "fake", "ok"])
    svc.deciders["private"] = dec
    p = store.get_point("e.pt")
    p["tier"] = "private"
    p["plan_phase"] = "serve"
    store.put_point(p)
    store.save_model("e.pt", 1, "private", "v1-1.json",
                     {"model_ref": MODEL, "temperature": 1.0,
                      "job_id": "j0"})
    svc._plan_eligible_hosted = lambda _p: True  # orthogonal, covered in m9
    return svc, store, dec


def _predict(svc, pilot=False):
    point = svc.store.get_point("e.pt")
    return svc._plan_predict(point, "do the thing", {"label_hint": "a"},
                             pilot)


def _plan_fit(store, threshold):
    store.save_hosted_fit({
        "point_id": "e.pt", "point_version": 1, "kind": "plan",
        "tier": "private", "epoch_start_ts": "", "fitted_at": "",
        "n_fit": 20, "last_fit_ts": "", "last_fit_decision_id": "p1",
        "temperature": 1.0, "threshold": threshold})


def test_T_PT9_private_plan_gate(tmp_path):
    svc, store, dec = _psetup(tmp_path)
    try:
        # No fit: predicted (comparisons accumulate) but open.
        e, w = _predict(svc)
        assert e["predicted"] and e["status"] == "open" and w == []
        assert dec._sessions  # the worker was consulted
        # Pilot_force does not bypass the missing fit.
        e, _ = _predict(svc, pilot=True)
        assert e["status"] == "open"
        # A null threshold keeps the point open.
        _plan_fit(store, None)
        e, _ = _predict(svc)
        assert e["status"] == "open"
        # A threshold decides (and pilot decides with a fit).
        _plan_fit(store, 0.5)
        e, _ = _predict(svc)
        assert e["status"] == "decided" and e["choice"] == "a"
        e, _ = _predict(svc, pilot=True)
        assert e["status"] == "decided"
    finally:
        dec.close()


# -- Y3: eval thread owns its store ----------------------------------------------

def test_Y3_eval_survives_main_store_close(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    _finish_smoke(svc)
    jid = _launch_full(svc)
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    store.close()  # the server tick's finally, mid-eval
    assert svc.autopilot.join_eval(jid, timeout=120)
    store2 = Store(store.root)
    try:
        assert store2.job_get(jid)["status"] == "done"
        assert store2.active_model("e.pt", "private") is not None
    finally:
        store2.close()


def test_Y5_eval_uses_eval_timeout(tmp_path):
    import threading

    svc, store, _ = _setup(tmp_path, fake_args=("slow", "4"))
    sess = svc.autopilot._eval_worker(MODEL)
    try:
        row = {"state": {"policy": "P"},
               "questions": {"e.pt": {
                   "type": "choice", "instructions": "Q?",
                   "criteria": {"a": "a", "b": "b"}, "label": "a"}}}
        out = svc.autopilot._eval_wait_ready(
            sess, [row], threading.Event(), "j-eval-timeout", store)
        assert len(out["logprobs"]) == 1
    finally:
        sess.close()


# -- Z1: private-tier serve path -------------------------------------------------

def test_Z1_pmo_shadow_to_serve_with_real_private_rows(tmp_path):
    svc, store, cfg = _setup(tmp_path)
    cfg["train"]["min_examples"] = 10
    cfg["serve"]["min_shadow"] = 12
    cfg["serve"]["min_auto_support"] = 3
    svc.deciders["private"] = FireworksSessionDecider(
        store, cfg, "k",
        worker_argv_fn=lambda _m: [sys.executable, SCRIPT, "fake", "ok"])
    dec = svc.deciders["private"]
    try:
        p = store.get_point("e.pt")
        p["private_model_only"] = True
        p["target_error"] = 0.3
        store.put_point(p)
        for i in range(70):
            _add_row(svc, "abc"[i % 3])
        _finish_smoke(svc)
        jid = _launch_full(svc)
        _finish_full(svc, jid)
        assert _poll_eval(svc) == jid
        assert svc.autopilot.join_eval(jid, timeout=60)
        assert store.job_get(jid)["status"] == "done"
        card = store.active_model("e.pt", "private")["card"]
        assert card["model_ref"]
        r = svc.do_promote({"point_id": "e.pt", "phase": "shadow"})
        assert r["ok"], r
        for i in range(12):
            _add_row(svc, "abc"[i % 3])
        ev = svc._private_evidence_rows(
            store.get_point("e.pt"), card)
        assert len(ev) == 12
        assert {r["probe"]["model_version"] for r, _ in ev} == {
            card["model_ref"]}
        assert all(r["decider"] == "private" for r, _ in ev)
        r = svc.do_promote({"point_id": "e.pt", "phase": "serve",
                            "confirm": True})
        assert r["ok"], r
        assert store.get_point("e.pt")["phase"] == "serve"
    finally:
        dec.close()


# -- Y9: spend accounting ------------------------------------------------------

def test_Y9_train_ledger_includes_dev_eval(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    _finish_smoke(svc)
    dry = svc.autopilot.dry_run(store.get_point("e.pt"))
    assert dry["estimate"]["dev_eval_usd"] > 0
    jid = _launch_full(svc)
    job = store.job_get(jid)
    assert job["train_usd"] == pytest.approx(
        dry["estimate"]["train_usd"] + dry["estimate"]["dev_eval_usd"])


def test_Y9_free_path_counts_eval_estimate(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    _finish_smoke(svc)
    r = svc.do_train({"point_id": "e.pt", "force": True, "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full", r
    jid = r["job_id"]
    launch = store.job_get(jid)
    assert launch["train_usd"] == 0
    assert launch["counted_usd"] > 0  # the eval part, not $0
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert svc.autopilot.join_eval(jid, timeout=60)
    job = store.job_get(jid)
    assert job["status"] == "done"
    assert job["counted_usd"] == pytest.approx(launch["counted_usd"])


def test_Y9_eval_counts_train_commit(tmp_path):
    svc, store, _ = _setup(tmp_path)
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    sid = _finish_smoke(svc)
    smoke_counted = store.job_get(sid)["counted_usd"]
    jid = _launch_full(svc)
    train = store.job_get(jid)["train_usd"]
    assert train > 0
    # Budget covers settled + train/2 + the first batch, but not the
    # full train commit: without the train term the eval would proceed.
    svc.cfg["fireworks"]["monthly_budget_usd"] = smoke_counted + train / 2
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert svc.autopilot.join_eval(jid, timeout=60)
    job = store.job_get(jid)
    assert job["status"] == "eval_failed"
    assert "budget_exceeded" in (job["error"] or "")
    # Train settles (it ran); no eval call was ever scored.
    assert store.spend_get(_month(), "fireworks")["usd"] == pytest.approx(
        smoke_counted + train)


def test_Y9_eval_counts_open_reservations(tmp_path):
    svc, store, _ = _setup(tmp_path)  # budget 100
    for i in range(70):
        _add_row(svc, "abc"[i % 3])
    _finish_smoke(svc)
    jid = _launch_full(svc)
    train = store.job_get(jid)["train_usd"]
    settled = store.spend_get(_month(), "fireworks")["usd"]
    # An open reservation that fits on its own, but with the train
    # commit leaves no room for the first eval batch.
    rid = store.hosted_reserve(_month(), "fireworks",
                               100.0 - settled - train / 2, 100.0, 600)
    assert rid is not None
    _finish_full(svc, jid)
    assert _poll_eval(svc) == jid
    assert svc.autopilot.join_eval(jid, timeout=60)
    job = store.job_get(jid)
    assert job["status"] == "eval_failed"
    assert "budget_exceeded" in (job["error"] or "")
