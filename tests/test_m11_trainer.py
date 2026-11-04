"""M11 trainer tests (F1–F5): FireworksTrainer launch/estimate/pin/lifecycle/export."""
import json
import os
import re
import stat
import subprocess
from pathlib import Path

import pytest

from decideonce import labels as L
from decideonce import render as R
from decideonce.autopilot import _fireworks_allowed
from decideonce.config import load_config
from decideonce.fireworks_export import render_export_row
from decideonce.fireworks_trainer import RECIPE_MODULE, FireworksTrainer
from decideonce.trainer import split_rows


def _row(pid="p", label="a", split="train", policy="POLICY", **ctx):
    state = {"policy": policy}
    state.update(ctx)
    return {"point_id": pid, "split": split, "decision_id": f"d-{split}-{label}",
            "state": state,
            "questions": {pid: {"type": "choice", "instructions": "Which?",
                                "criteria": {"a": "Ay", "b": "Bee"},
                                "label": label, "weight": 1.0}}}


def _cfg(tmp_path, **over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["store"]["project"] = "DemoProj"
    for section, kv in over.items():
        cfg[section].update(kv)
    return cfg


_CB_N = {"n": 0}


def _cookbook(tmp_path, name=None, commit=True):
    """A cookbook-looking dir: venv python + optional git checkout."""
    _CB_N["n"] += 1
    cb = tmp_path / (name or f"cb{_CB_N['n']}")
    py = cb / "training" / ".venv" / "bin" / "python"
    py.parent.mkdir(parents=True)
    py.touch()
    py.chmod(0o755)
    if commit:
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
    return cb, None


class _Runner:
    """Fake spawn: records (argv, env, cwd), runs nothing."""

    def __init__(self):
        self.calls = []

    def __call__(self, argv, env, cwd):
        self.calls.append({"argv": argv, "env": dict(env), "cwd": cwd})


def _trainer(tmp_path, runner=None, **fw_over):
    cfg = _cfg(tmp_path)
    cb, short = _cookbook(tmp_path)
    cfg["fireworks"].update({"cookbook_dir": str(cb),
                             "cookbook_commit": short})
    cfg["fireworks"].update(fw_over)
    jobs = tmp_path / "jobs"
    return (FireworksTrainer(cfg, jobs, key="fw_test_key_123",
                             runner=runner or _Runner()),
            cfg, jobs, cb)


# -- T-PT2: pin check -------------------------------------------------------

def test_T_PT2_pin_ok(tmp_path):
    tr, _, _, _ = _trainer(tmp_path)
    st = tr.pin_status()
    assert st["status"] == "ok"


def test_T_PT2_pin_missing_dir(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["fireworks"]["cookbook_dir"] = str(tmp_path / "nope")
    tr = FireworksTrainer(cfg, tmp_path / "jobs", key="k")
    assert tr.pin_status()["status"] == "cookbook_missing"


def test_T_PT2_pin_missing_venv(tmp_path):
    cfg = _cfg(tmp_path)
    cb = tmp_path / "cb"
    cb.mkdir()
    cfg["fireworks"]["cookbook_dir"] = str(cb)
    tr = FireworksTrainer(cfg, tmp_path / "jobs", key="k")
    assert tr.pin_status()["status"] == "cookbook_missing"


def test_T_PT2_pin_mismatch(tmp_path):
    tr, _, _, _ = _trainer(tmp_path, cookbook_commit="0000000")
    st = tr.pin_status()
    assert st["status"] == "cookbook_mismatch"


def test_T_PT2_launch_refuses_bad_pin_and_missing_key(tmp_path):
    import pytest

    tr, _, _, _ = _trainer(tmp_path, cookbook_commit="0000000")
    with pytest.raises(RuntimeError, match="cookbook_mismatch"):
        tr.launch([_row()], "smoke", "idem-x")
    tr2, _, _, _ = _trainer(tmp_path)
    tr2.key = None
    with pytest.raises(RuntimeError, match="no_fireworks_key"):
        tr2.launch([_row()], "smoke", "idem-x")


# -- T-PT1: argv, env, cwd ---------------------------------------------------

def _launch_calls(kind, tmp_path, rows=None, **fw_over):
    runner = _Runner()
    tr, cfg, jobs, cb = _trainer(tmp_path, runner, **fw_over)
    rows = rows if rows is not None else [
        _row(split="train"), _row(split="train", label="b"),
        _row(split="calib"), _row(split="heldout")]
    job_id = tr.launch(rows, kind, f"idem-{kind}")
    assert job_id == f"idem-{kind}"
    assert len(runner.calls) == 1
    return tr, cfg, jobs, cb, runner.calls[0], jobs / f"idem-{kind}"


def _assert_common_argv(call, cb, job_dir, n_train):
    argv = call["argv"]
    assert argv[0] == str(job_dir / "run.sh")
    assert argv[1] == str(cb / "training" / ".venv" / "bin" / "python")
    assert argv[2:4] == ["-m", RECIPE_MODULE]
    flags = argv[4:]
    assert ["--train-path", str(job_dir / "train.jsonl")] == flags[0:2]
    assert ["--dev-path", str(job_dir / "dev.jsonl")] == flags[2:4]
    assert "--skip-download" in flags
    assert ["--lora-rank", "64"] == [
        flags[flags.index("--lora-rank")], flags[flags.index("--lora-rank") + 1]]
    assert flags[flags.index("--learning-rate") + 1] == "0.0001"
    assert flags[flags.index("--batch-size") + 1] == str(min(24, n_train))
    assert flags[flags.index("--epochs") + 1] == "2"
    assert ["--log-path", str(job_dir / "cookbook")] == [
        flags[flags.index("--log-path")], flags[flags.index("--log-path") + 1]]
    return flags


def test_T_PT1_smoke_argv_env_cwd_and_files(tmp_path):
    tr, cfg, jobs, cb, call, job_dir = _launch_calls("smoke", tmp_path)
    flags = _assert_common_argv(call, cb, job_dir, 2)
    assert ["--max-examples", "32"] == [
        flags[flags.index("--max-examples")],
        flags[flags.index("--max-examples") + 1]]
    assert "--output-model-id" not in flags
    # env: key only in env, never in argv
    assert call["env"]["FIREWORKS_API_KEY"] == "fw_test_key_123"
    assert "fw_test_key_123" not in " ".join(call["argv"])
    assert call["cwd"] == str(cb)
    # files: rendered rows only (state + questions), split files
    for name, want in (("train.jsonl", 2), ("dev.jsonl", 1),
                       ("heldout.jsonl", 1)):
        lines = (job_dir / name).read_text().splitlines()
        assert len(lines) == want
        for line in lines:
            assert sorted(json.loads(line)) == ["questions", "state"]
    meta = json.loads((job_dir / "meta.json").read_text())
    assert meta["kind"] == "smoke" and meta["n_train"] == 2
    assert meta["estimate"]["kind"] == "smoke"
    assert stat.S_IMODE((job_dir / "run.sh").stat().st_mode) == 0o700
    assert stat.S_IMODE((job_dir / "train.log").stat().st_mode) == 0o600


def test_T_PT1_full_model_id_and_batch_cap(tmp_path):
    rows = [_row(split="train", label="ab"[i % 2]) for i in range(30)]
    tr, cfg, jobs, cb, call, job_dir = _launch_calls(
        "full", tmp_path, rows=rows)
    flags = _assert_common_argv(call, cb, job_dir, 30)
    assert "--max-examples" not in flags
    mid = flags[flags.index("--output-model-id") + 1]
    assert re.fullmatch(r"dk-demoproj-\d{8}-\d{4}", mid), mid
    assert len(mid) <= 63
    tiny = [_row(split="train")]
    _, _, _, _, call2, _ = _launch_calls("full", tmp_path / "t2", rows=tiny)
    assert call2["argv"][call2["argv"].index("--batch-size") + 1] == "1"


def test_X3_job_jsonl_0600(tmp_path):
    _tr, _cfg, _jobs, _cb, _call, job_dir = _launch_calls(
        "smoke", tmp_path)
    for name in ("train.jsonl", "dev.jsonl", "heldout.jsonl"):
        mode = stat.S_IMODE((job_dir / name).stat().st_mode)
        assert mode == 0o600, (name, oct(mode))


# -- Y1/Y2: idem validation + reuse --------------------------------------------

def test_Y1_launch_rejects_bad_idem(tmp_path):
    tr, _, jobs, _ = _trainer(tmp_path)
    rows = [_row(split="train")]
    for bad in ("../../evil-traversal", "short", "has space here!!",
                "", "ok-but-way-too-long-" + "x" * 60):
        with pytest.raises(ValueError, match="bad idem"):
            tr.launch(rows, "smoke", bad)
    assert not jobs.exists() or list(jobs.iterdir()) == []
    assert not (tmp_path / "evil-traversal").exists()


def test_Y1_job_dir_rejects_traversal(tmp_path):
    tr, _, _, _ = _trainer(tmp_path)
    with pytest.raises(ValueError, match="bad job_id"):
        tr.status("../../evil-12345678")
    with pytest.raises(ValueError, match="bad job_id"):
        tr.cancel("x")


def test_Y2_trainer_refuses_existing_dir(tmp_path):
    from decideonce.trainer import IdemInUse

    tr, _, jobs, _ = _trainer(tmp_path)
    rows = [_row(split="train")]
    tr.launch(rows, "smoke", "idem-reuse-dir")
    with pytest.raises(IdemInUse):
        tr.launch(rows, "smoke", "idem-reuse-dir")
    assert [p.name for p in jobs.iterdir()] == ["idem-reuse-dir"]


def test_Y8_cancel_skips_reused_pid(tmp_path):
    import subprocess as _sp
    import time as _t

    from decideonce.fireworks_trainer import FireworksTrainer

    cfg = _cfg(tmp_path)
    jobs = tmp_path / "jobs"
    tr = FireworksTrainer(cfg, jobs, key="k", runner=_Runner())
    victim = _sp.Popen(["sleep", "60"])
    try:
        jd = jobs / "idem-reused-pid"
        jd.mkdir(parents=True)
        (jd / "meta.json").write_text('{"kind": "smoke"}')
        (jd / "pid").write_text(f"{victim.pid}\n")
        (jd / "started_at").write_text(f"{int(_t.time()) - 3600}\n")
        out = tr.cancel("idem-reused-pid")
        assert out["status"] == "cancelled"
        assert victim.poll() is None  # the stranger survives
        assert (jd / "cancelled").is_file()
        assert (jd / "exit_code").read_text().strip() == "125"
    finally:
        victim.kill()
        victim.wait()


def test_Y8_status_files_atomic(tmp_path):
    from decideonce.fireworks_trainer import RUN_SH

    assert "mv -f" in RUN_SH
    tr, jobs = _live_trainer(tmp_path, "smoke-ok", name="atomic")
    job_id = tr.launch(_rows(), "smoke", "idem-atomic")
    try:
        assert _poll(tr, job_id)["status"] == "done"
        names = {p.name for p in (jobs / job_id).iterdir()}
        assert not [n for n in names if n.endswith(".tmp")]
        assert names >= {"pid", "started_at", "exit_code", "finished_at",
                         "train.log"}
        int((jobs / job_id / "pid").read_text().strip())
        int((jobs / job_id / "exit_code").read_text().strip())
    finally:
        tr.cancel(job_id)


# -- estimate (GP12) ----------------------------------------------------------

def _toks(row):
    payload = {"state": row["state"], "questions": row["questions"]}
    return max(1, len(json.dumps(payload, sort_keys=True)) // 4)


def test_F1_estimate_full_counts_train_dev_calib_heldout(tmp_path):
    tr, _, _, _ = _trainer(tmp_path)
    rows = ([_row(split="train", label="ab"[i % 2]) for i in range(10)]
            + [_row(split="calib") for _ in range(4)]
            + [_row(split="heldout") for _ in range(6)])
    est = tr.estimate(rows, "full")
    train_toks = sum(_toks(r) for r in rows if r["split"] == "train")
    dev_toks = sum(_toks(r) for r in rows if r["split"] == "calib")
    held_toks = sum(_toks(r) for r in rows if r["split"] == "heldout")
    assert est["train_tokens"] == train_toks
    assert est["train_usd"] == train_toks * 2 * 5.86 / 1e6
    # in-job dev (epochs+1 passes) + post-job calib + heldout scoring
    assert est["eval_tokens"] == 3 * dev_toks + dev_toks + held_toks
    assert est["eval_usd"] == est["eval_tokens"] * 1.96 / 1e6
    assert est["estimate_usd"] == est["train_usd"] + est["eval_usd"]
    assert est["train_rows"] == 10 and est["epochs"] == 2


def test_F1_estimate_smoke_caps_train_and_skips_post_scoring(tmp_path):
    tr, _, _, _ = _trainer(tmp_path)
    rows = ([_row(split="train", label="ab"[i % 2]) for i in range(40)]
            + [_row(split="calib") for _ in range(4)]
            + [_row(split="heldout") for _ in range(6)])
    est = tr.estimate(rows, "smoke")
    assert est["train_rows"] == 32  # --max-examples cap
    dev_toks = sum(_toks(r) for r in rows if r["split"] == "calib")
    assert est["eval_tokens"] == 3 * dev_toks  # in-job dev only
    assert est["eval_rows"] == 3 * 4


def test_F1_estimate_empty_is_zero(tmp_path):
    tr, _, _, _ = _trainer(tmp_path)
    est = tr.estimate([], "full")
    assert est["estimate_usd"] == 0.0 and est["train_rows"] == 0


def test_Y9_estimate_splits_dev_eval(tmp_path):
    tr, _, _, _ = _trainer(tmp_path)
    rows = ([_row(split="train", label="ab"[i % 2]) for i in range(10)]
            + [_row(split="calib") for _ in range(4)]
            + [_row(split="heldout") for _ in range(6)])
    est = tr.estimate(rows, "full")
    dev_toks = sum(_toks(r) for r in rows if r["split"] == "calib")
    assert est["dev_eval_usd"] == 3 * dev_toks * 1.96 / 1e6
    assert est["dev_eval_usd"] < est["eval_usd"]  # post-job part on top
    smoke = tr.estimate(rows, "smoke")
    assert smoke["dev_eval_usd"] == smoke["eval_usd"]  # in-job dev only
    assert tr.estimate([], "full")["dev_eval_usd"] == 0.0


# -- T-PT3: log parser on redacted pilot fixtures ------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def test_T_PT3_smoke_log(tmp_path):
    from decideonce.trainer_log import parse_log

    out = parse_log((FIXTURES / "fw_log_smoke.out").read_text())
    assert [m["step"] for m in out["metrics"]] == [0, 1, 2, 3, 4, 4]
    assert out["last_step"] == 4
    assert out["last_eval"]["eval_accuracy"] == 0.8866666666666667
    assert out["last_loss"] == 0.4719843864440918
    assert out["calibration_T"] == {"choice": 1.0059159722032391}
    assert out["sampler_checkpoint"]["name"] == "step-4-cal"
    assert out["sampler_checkpoint"]["server_stored"] == (
        "run-test…-step-4-cal-18145e12")
    assert out["model"] is None and out["promoted"] is False
    assert out["account_id"] == "acct-test"
    assert out["init_from_checkpoint"].startswith("acct-test/run-test")
    assert out["n_train"] == 32


def test_T_PT3_full_log(tmp_path):
    from decideonce.trainer_log import parse_log

    out = parse_log((FIXTURES / "fw_log_full.out").read_text())
    assert len(out["metrics"]) == 61
    assert out["last_step"] == 57
    assert out["last_eval"]["eval_accuracy"] == 0.94
    assert out["calibration_T"] == {"choice": 0.7943445437494694}
    assert out["sampler_checkpoint"]["name"] == "step-57-cal"
    assert out["model"] == "accounts/acct-test/models/refund-decide-v1"
    assert out["promoted"] is True
    assert out["account_id"] == "acct-test"
    assert out["n_train"] == 150


def test_Y12_bare_model_line_ignored(tmp_path):
    from decideonce.trainer_log import parse_log

    out = parse_log(
        "echo --output-model-id dk-real\n"
        "debug: resolved -> model 'accounts/x/models/evil'\n"
        "Promoting session checkpoint 'c' -> model "
        "'accounts/acct-test/models/dk-real'\n"
        "Promoted!\n")
    assert out["model"] == "accounts/acct-test/models/dk-real"
    assert out["promoted"] is True
    out = parse_log("note -> model 'accounts/x/models/evil'\n")
    assert out["model"] is None


def test_Y12_promotion_must_match_meta(tmp_path):
    tr, _, jobs, _ = _trainer(tmp_path)
    jd = jobs / "jd1-y12-0001"
    jd.mkdir(parents=True)
    (jd / "meta.json").write_text(json.dumps(
        {"kind": "full",
         "argv": ["recipe", "--output-model-id", "dk-real-0001"]}))
    (jd / "train.log").write_text(
        "Promoting session checkpoint 'c' -> model "
        "'accounts/acct-test/models/dk-evil-0002'\n"
        "Promoted!\n")
    (jd / "exit_code").write_text("0")
    st = tr.status("jd1-y12-0001")
    assert st["status"] == "failed", st
    assert "model_mismatch" in st["error"]
    assert "acct-test" not in st["error"]  # tails only
    # The same log with the requested id confirms done.
    (jd / "train.log").write_text(
        "Promoting session checkpoint 'c' -> model "
        "'accounts/acct-test/models/dk-real-0001'\n"
        "Promoted!\n")
    st = tr.status("jd1-y12-0001")
    assert st["status"] == "done", st
    assert st["model"] == "accounts/acct-test/models/dk-real-0001"


def test_T_PT3_malformed_input_never_raises():
    from decideonce.trainer_log import parse_log

    out = parse_log('{"type": "metrics", oops\nDecision calibration: T=zzz\n'
                    "Sampler checkpoint 'x' saved\ntrailing line\n")
    assert out["metrics"] == [] and out["last_step"] is None
    assert out["calibration_T"] is None and out["last_eval"] is None
    assert out["sampler_checkpoint"] == {"name": "x", "server_stored": None}
    assert out["model"] is None and out["promoted"] is False
    assert out["account_id"] is None and out["n_train"] is None


def test_Y13_missing_cookbook_reason(tmp_path):
    from decideonce.store import Store
    from decideonce.trainer import live_trainer_unavailable_reason

    cfg = _cfg(tmp_path)
    cfg["fireworks"]["cookbook_dir"] = str(tmp_path / "nope")
    store = Store(tmp_path / "proj")
    try:
        assert live_trainer_unavailable_reason(
            cfg, store, "k") == "cookbook_missing"
    finally:
        store.close()


# -- T-PT4: detached lifecycle with a fake child script -------------------------

import sys as _sys
import time as _time

CHILD = '''
import os, sys, time
mode = sys.argv[1]
if mode == "smoke-ok":
    print("Sampler checkpoint 's-cal' saved (1.0s)", flush=True)
elif mode == "pwd-ok":
    print(os.getcwd(), flush=True)
    print("Sampler checkpoint 's-cal' saved (1.0s)", flush=True)
elif mode == "full-ok":
    _mid = sys.argv[sys.argv.index("--output-model-id") + 1]
    print("Promoting session checkpoint 'x' -> model "
          f"'accounts/a/models/{_mid}'", flush=True)
    print("Promoted! Model state=READY, kind=HF_PEFT_ADDON", flush=True)
elif mode == "slow-ok":
    time.sleep(3)
    print("Sampler checkpoint 's-cal' saved (1.0s)", flush=True)
elif mode == "fail":
    print("boom happened", file=sys.stderr, flush=True)
    sys.exit(1)
elif mode == "key":
    print("using key fw_abcdefgh1234567890 for upload", flush=True)
    sys.exit(1)
elif mode == "hang":
    print("hanging", flush=True)
    time.sleep(600)
# bare: print nothing, exit 0
'''


def _live_trainer(tmp_path, mode, name="live", **fw_over):
    """Real spawns, fake child: recipe ignores the cookbook flags."""
    from decideonce.fireworks_trainer import FireworksTrainer

    child = tmp_path / f"child-{name}.py"
    child.write_text(CHILD)
    cfg = _cfg(tmp_path)
    cb, short = _cookbook(tmp_path)
    cfg["fireworks"].update({"cookbook_dir": str(cb),
                             "cookbook_commit": short})
    cfg["fireworks"].update(fw_over)
    jobs = tmp_path / f"jobs-{name}"
    tr = FireworksTrainer(cfg, jobs, key="fw_test_key_123",
                          recipe=[_sys.executable, str(child), mode])
    return tr, jobs


def _rows(kind_split="train"):
    return [_row(split="train"), _row(split="calib", label="b")]


def _poll(tr, job_id, timeout_s=30.0):
    deadline = _time.monotonic() + timeout_s
    st = tr.status(job_id)
    while st["status"] == "running" and _time.monotonic() < deadline:
        _time.sleep(0.2)
        st = tr.status(job_id)
    return st


def test_T_PT4_running_to_done_smoke(tmp_path):
    tr, jobs = _live_trainer(tmp_path, "slow-ok")
    job_id = None
    try:
        job_id = tr.launch(_rows(), "smoke", "idem-f3-slow")
        assert tr.status(job_id)["status"] == "running"
        st = _poll(tr, job_id)
        assert st["status"] == "done", st
        assert (jobs / job_id / "finished_at").is_file()
    finally:
        if job_id:
            tr.cancel(job_id)


def test_T_PT4_done_full_and_failed_and_bare(tmp_path):
    tr, _ = _live_trainer(tmp_path, "full-ok", name="a")
    job_id = None
    try:
        job_id = tr.launch(_rows(), "full", "idem-f3-full")
        assert _poll(tr, job_id)["status"] == "done"
    finally:
        if job_id:
            tr.cancel(job_id)
    tr2, _ = _live_trainer(tmp_path, "fail", name="b")
    job_id = None
    try:
        job2 = tr2.launch(_rows(), "smoke", "idem-f3-fail")
        st = _poll(tr2, job2)
        assert st["status"] == "failed" and "boom" in st["error"], st
    finally:
        if job2:
            tr2.cancel(job2)
    tr3, _ = _live_trainer(tmp_path, "bare", name="c")
    job_id = None
    try:
        job3 = tr3.launch(_rows(), "smoke", "idem-f3-bare")
        st = _poll(tr3, job3)
        assert st["status"] == "failed" and "without" in st["error"], st
    finally:
        if job3:
            tr3.cancel(job3)


def test_Z6_recipe_runs_at_cookbook_root(tmp_path):
    tr, jobs = _live_trainer(tmp_path, "pwd-ok", name="z6cwd")
    job_id = tr.launch(_rows(), "smoke", "idem-z6-cwd")
    try:
        assert _poll(tr, job_id)["status"] == "done"
        log = (jobs / job_id / "train.log").read_text().splitlines()
        assert log[0] == str(tr._cookbook())
        assert (jobs / job_id / "pid").is_file()  # markers stay in the job dir
    finally:
        tr.cancel(job_id)


def test_X5_no_pid_gone_race_on_fast_exit(tmp_path):
    # X5: status() once reported "pid gone with no exit_code" when run.sh
    # finished between its exit check and its pid check. Hammer fast
    # failures with tight polls: the error must always be the real one.
    for i in range(25):
        tr, _ = _live_trainer(tmp_path, "fail", name=f"x5-{i}")
        job_id = tr.launch(_rows(), "smoke", f"idem-x5-{i:02d}")
        try:
            for _ in range(500):
                st = tr.status(job_id)
                if st["status"] != "running":
                    break
            assert st["status"] == "failed", st
            assert "boom" in st["error"], st
        finally:
            tr.cancel(job_id)


def test_T_PT4_restart_sees_same_state(tmp_path):
    from decideonce.fireworks_trainer import FireworksTrainer

    tr, jobs = _live_trainer(tmp_path, "hang")
    job_id = None
    try:
        job_id = tr.launch(_rows(), "smoke", "idem-f3-restart")
        assert tr.status(job_id)["status"] == "running"
        tr2 = FireworksTrainer(tr.cfg, jobs, key="other")
        assert tr2.status(job_id)["status"] == "running"
    finally:
        if job_id:
            tr.cancel(job_id)


def test_T_PT4_pid_reuse_detected(tmp_path):
    import os as _os

    tr, _, jobs, _ = _trainer(tmp_path)
    d = jobs / "idem-reuse"
    d.mkdir(parents=True)
    (d / "pid").write_text(f"{_os.getpid()}\n")
    (d / "started_at").write_text("1\n")  # 1970: not our process
    (d / "meta.json").write_text('{"kind": "smoke"}')
    st = tr.status("idem-reuse")
    assert st["status"] == "failed" and "pid_reuse" in st["error"], st


def test_T_PT4_timeout_kills_group(tmp_path):
    import os as _os

    tr, jobs = _live_trainer(tmp_path, "hang", name="t",
                             job_max_hours=1 / 3600)
    job_id = tr.launch(_rows(), "smoke", "idem-f3-timeout")
    try:
        assert tr.status(job_id)["status"] == "running"
        _time.sleep(1.3)
        st = tr.status(job_id)
        assert st["status"] == "failed" and st["error"] == "job_timeout", st
        pid = int((jobs / job_id / "pid").read_text())
        tr.status(job_id)  # reaps the handle, so the zombie is gone
        try:
            _os.kill(pid, 0)
            alive = True
        except OSError:
            alive = False
        assert not alive
        assert (jobs / job_id / "timeout").is_file()
        assert (jobs / job_id / "exit_code").read_text().strip() == "124"
        assert tr.status(job_id)["error"] == "job_timeout"
    finally:
        if job_id:
            tr.cancel(job_id)


def test_T_PT4_cancel(tmp_path):
    tr, jobs = _live_trainer(tmp_path, "hang", name="x")
    job_id = tr.launch(_rows(), "smoke", "idem-f3-cancel")
    try:
        assert tr.status(job_id)["status"] == "running"
        assert tr.cancel(job_id)["status"] == "cancelled"
        assert tr.status(job_id)["status"] == "cancelled"
        assert tr.cancel(job_id)["status"] == "cancelled"  # idempotent
    finally:
        if job_id:
            tr.cancel(job_id)
    tr2, _ = _live_trainer(tmp_path, "fail", name="y")
    job_id = None
    try:
        job2 = tr2.launch(_rows(), "smoke", "idem-f3-cancel2")
        st = _poll(tr2, job2)
        assert st["status"] == "failed"
        assert tr2.cancel(job2)["status"] == "failed"  # finished: as-is
    finally:
        if job2:
            tr2.cancel(job2)


def test_T_PT4_pipeline_masks_keys(tmp_path):
    tr, jobs = _live_trainer(tmp_path, "key", name="k")
    job_id = None
    try:
        job_id = tr.launch(_rows(), "smoke", "idem-f3-key")
        st = _poll(tr, job_id)
        assert st["status"] == "failed"
        log = (jobs / job_id / "train.log").read_text()
        assert "fw_***" in log and "fw_abcdefgh1234567890" not in log
        assert "fw_abcdefgh1234567890" not in st["error"]
    finally:
        if job_id:
            tr.cancel(job_id)


# -- T-PT5: reconcile, idem, cancel, recovery -----------------------------------

_SENTINEL = object()


def _svc4(tmp_path, trainer=_SENTINEL, **over):
    from decideonce.service import Service
    from decideonce.store import Store
    from decideonce.trainer import FakeTrainer

    cfg = _cfg(tmp_path)
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    cfg["train"].update({"min_train": 5, "min_examples": 6})
    for section, kv in over.items():
        cfg[section].update(kv)
    store = Store(tmp_path / "proj")
    if trainer is _SENTINEL:
        trainer = FakeTrainer(cfg)
    svc = Service(cfg, store, {}, trainer=trainer)
    return svc, store


def _point4(pid="pt4"):
    return {"point_id": pid, "workflow": "w", "question": "Which?",
            "kind": "choice", "options": ["a", "b"],
            "context_fields": {"label_hint": {"type": "string"}},
            "required_fields": [], "data_class": "public"}


def _label_rows4(svc, pid, n=10):
    for i in range(n):
        d = svc.do_decide({"point_id": pid,
                           "context": {"label_hint": "ab"[i % 2]}})
        assert svc.do_record({"decision_id": d["decision_id"],
                              "choice": "ab"[i % 2]})["ok"]


def _launching_row(idem, created_at, kind="smoke", autopilot=False):
    return {"job_id": f"launching-{idem[:12]}", "kind": kind,
            "status": "launching", "estimate_usd": 0.01, "counted_usd": 0.01,
            "points": [{"point_id": "pt4"}], "autopilot": autopilot,
            "created_at": created_at, "idempotency_key": idem}


def test_T_PT5_crash_between_spawn_and_claim(tmp_path):
    tr, _ = _live_trainer(tmp_path, "hang", name="r")
    tr.launch(_rows(), "smoke", "idem-reconcile")
    try:
        svc, store = _svc4(tmp_path / "db", trainer=tr)
        assert svc.do_register(_point4())["ok"]
        store.job_insert_launching(_launching_row(
            "idem-reconcile", "2026-01-01T00:00:00Z"))
        out = svc.autopilot.poll_and_finalize()
        assert out == []  # adopted, still running: nothing to report
        assert store.job_get("idem-reconcile")["status"] == "running"
    finally:
        tr.cancel("idem-reconcile")


def test_T_PT5_young_launching_left_alone(tmp_path):
    from decideonce import service as _S

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    store.job_insert_launching(_launching_row("idem-young", _S._utcnow()))
    assert svc.autopilot.poll_and_finalize() == []
    assert store.job_get("launching-idem-young")["status"] == "launching"


def test_T_PT5_same_idem_launches_once(tmp_path):
    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    _label_rows4(svc, "pt4")
    r1 = svc.do_train({"point_id": "pt4", "force": True, "confirm": True,
                       "idem": "idem-dup"})
    assert r1["status"] == "running"
    r2 = svc.do_train({"point_id": "pt4", "force": True, "confirm": True,
                       "idem": "idem-dup"})
    assert r2["job_id"] == r1["job_id"]
    assert len(svc.trainer.uploaded) == 1


def test_Y11_join_needs_reason_pmo_or_force(tmp_path):
    svc, store = _svc4(tmp_path)
    trig = _point4("pt4")
    trig["private_model_only"] = True
    assert svc.do_register(trig)["ok"]
    pj1 = _point4("pj1")
    pj1["private_model_only"] = True
    assert svc.do_register(pj1)["ok"]
    assert svc.do_register(_point4("pj2"))["ok"]  # plain, hosted-served
    _label_rows4(svc, "pt4")
    _label_rows4(svc, "pj1")
    _label_rows4(svc, "pj2")
    dry = svc.autopilot.dry_run(store.get_point("pt4"))
    marks = {j["point_id"]: j["join_ok"] for j in dry["joins"]}
    assert marks == {"pt4": True, "pj1": True, "pj2": False}
    forced = svc.autopilot.dry_run(store.get_point("pt4"), force=True)
    assert {j["point_id"] for j in forced["joins"] if j["join_ok"]} == {
        "pt4", "pj1", "pj2"}
    r = svc.do_train({"point_id": "pt4", "confirm": True})
    assert r["ok"] and r["status"] == "running", r
    assert {row["point_id"] for row in svc.trainer.uploaded[-1]["rows"]} == {
        "pt4", "pj1"}


def test_T_PT5_failing_trainer_backoff(tmp_path):
    import types as _t

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    _label_rows4(svc, "pt4")
    r = svc.do_train({"point_id": "pt4", "force": True, "confirm": True})
    assert r["status"] == "running"

    def boom(job_id):
        raise RuntimeError("flaky")

    svc.trainer = _t.SimpleNamespace(status=boom)
    assert svc.autopilot.poll_and_finalize() == []
    assert svc.autopilot.poll_and_finalize() == []
    out = svc.autopilot.poll_and_finalize()
    assert out[0]["action"] == "train_failed"
    assert store.job_get(r["job_id"])["error"] == "trainer unreachable (3 polls)"


def test_Z7_keyerror_backs_off_like_other_errors(tmp_path):
    import types as _t

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    _label_rows4(svc, "pt4")
    r = svc.do_train({"point_id": "pt4", "force": True, "confirm": True})
    assert r["status"] == "running"

    def bug(job_id):
        raise KeyError("internal oops")

    svc.trainer = _t.SimpleNamespace(status=bug)
    assert svc.autopilot.poll_and_finalize() == []
    assert store.job_get(r["job_id"])["status"] == "running"
    assert svc.autopilot.poll_and_finalize() == []
    out = svc.autopilot.poll_and_finalize()
    assert out[0]["action"] == "train_failed"
    assert store.job_get(r["job_id"])["error"] == "trainer unreachable (3 polls)"


def test_T_PT5_cancel_never_finalizes(tmp_path):
    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    _label_rows4(svc, "pt4")
    r = svc.do_train({"point_id": "pt4", "force": True, "confirm": True})
    c = svc.do_train({"point_id": "pt4", "force": True, "cancel": True})
    assert c["status"] == "cancelled" and c["job_id"] == r["job_id"]
    assert svc.trainer.jobs[r["job_id"]]["status"] == "cancelled"
    svc.trainer.complete(r["job_id"])  # trainer finished anyway
    assert svc.autopilot.poll_and_finalize() == []
    assert store.job_get(r["job_id"])["status"] == "cancelled"
    assert store.active_model("pt4", "private") is None
    c2 = svc.do_train({"point_id": "pt4", "force": True, "cancel": r["job_id"]})
    assert c2["error"] == "job_not_running"  # already terminal


def test_T_PT5_launching_status_stale_and_launcher(tmp_path):
    import types as _t

    from decideonce import service as _S

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    store.job_insert_launching(_launching_row("idem-st", _S._utcnow()))
    r = svc.do_train({"point_id": "pt4", "force": True})
    assert (r["status"], r["stale"], r["launcher"]) == (
        "launching", False, "manual")

    def boom(job_id):
        raise RuntimeError("flaky")

    svc.trainer = _t.SimpleNamespace(status=boom)
    store.cx.execute("UPDATE jobs SET created_at=? WHERE job_id=?",
                     ("2026-01-01T00:00:00Z", "launching-idem-st"))
    r = svc.do_train({"point_id": "pt4", "force": True})
    assert (r["status"], r["stale"], r["launcher"]) == (
        "launching", True, "manual")


def test_Y9_stale_empty_launching_fails_with_estimate(tmp_path):
    import time as _time

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    row = _launching_row("idem-stale-empty", "2026-01-01T00:00:00Z")
    store.job_insert_launching(row)
    out = svc.autopilot.poll_and_finalize()
    assert [a["action"] for a in out] == ["train_failed"]
    job = store.job_get(row["job_id"])
    assert job["status"] == "failed"
    assert job["error"] == "crashed during launch"
    month = _time.strftime("%Y-%m", _time.gmtime())
    assert store.spend_get(month, "fireworks")["usd"] == pytest.approx(0.01)
    assert job["spend_usd"] == pytest.approx(0.01)


def test_Y9_stale_ran_failed_keeps_error_and_spend(tmp_path):
    import time as _time

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    svc.trainer.jobs["idem-stale-failed"] = {
        "status": "failed", "kind": "smoke", "error": "recipe boom"}
    row = _launching_row("idem-stale-failed", "2026-01-01T00:00:00Z")
    store.job_insert_launching(row)
    out = svc.autopilot.poll_and_finalize()
    assert [a["action"] for a in out] == ["train_failed"]
    job = store.job_get(row["job_id"])
    assert job["status"] == "failed"
    assert job["error"] == "recipe boom"
    month = _time.strftime("%Y-%m", _time.gmtime())
    assert store.spend_get(month, "fireworks")["usd"] == pytest.approx(0.01)


def test_Y9_launching_window_is_10_min(tmp_path):
    import time as _time

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    now = _time.time()
    young = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(now - 9 * 60))
    old = _time.strftime("%Y-%m-%dT%H:%M:%SZ", _time.gmtime(now - 11 * 60))
    row = _launching_row("idem-win-10min", young)
    store.job_insert_launching(row)
    assert svc.autopilot.poll_and_finalize() == []
    assert store.job_get(row["job_id"])["status"] == "launching"
    store.cx.execute("UPDATE jobs SET created_at=? WHERE job_id=?",
                     (old, row["job_id"]))
    store.cx.commit()
    out = svc.autopilot.poll_and_finalize()
    assert [a["action"] for a in out] == ["train_failed"]
    assert store.job_get(row["job_id"])["status"] == "failed"


def test_T_PT5_crash_between_mark_done_and_store(tmp_path, monkeypatch):
    import pytest as _pt

    svc, store = _svc4(tmp_path)
    assert svc.do_register(_point4())["ok"]
    _label_rows4(svc, "pt4")
    r = svc.do_train({"point_id": "pt4", "force": True, "confirm": True})
    svc.trainer.complete(r["job_id"])
    assert svc.do_train({"point_id": "pt4", "force": True})["status"] == "smoke_done"
    r = svc.do_train({"point_id": "pt4", "force": True, "confirm": True,
                      "training_free_confirmed": True})
    assert r["status"] == "running"
    svc.trainer.complete(r["job_id"])
    ap = svc.autopilot
    orig = ap._save_private_model
    calls = []

    def boom_once(join, job, model, store=None):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("crash now")
        return orig(join, job, model, store)

    monkeypatch.setattr(ap, "_save_private_model", boom_once)
    _month = store.job_get(r["job_id"])["created_at"][:7]
    before = store.spend_get(_month, "fireworks")["usd"]
    with _pt.raises(RuntimeError, match="crash now"):
        ap.poll_and_finalize()
    job = store.job_get(r["job_id"])
    assert job["status"] == "done" and not job["finalized"]
    assert store.active_model("pt4", "private") is None
    monkeypatch.setattr(ap, "_save_private_model", orig)
    out = ap.poll_and_finalize()
    assert out[0]["action"] == "finalize_recovered"
    assert store.active_model("pt4", "private")["card"]["job_id"] == r["job_id"]
    month = job["created_at"][:7]
    counted = store.job_get(r["job_id"])["counted_usd"]
    assert store.spend_get(month, "fireworks")["usd"] == before + counted
    assert ap.poll_and_finalize() == []  # no double spend, no dup models
    assert store.spend_get(month, "fireworks")["usd"] == before + counted


# -- F5 fixtures ------------------------------------------------------------

def _xpoint(pid="ex", kind="choice", options=("c", "a", "b"), helps=None):
    return {"point_id": pid, "kind": kind, "options": list(options),
            "option_help": dict(helps or {}),
            "question": f"Which {pid}?",
            "policy": "POLICY",
            "context_fields": {"amount": {"max_len": 64},
                               "vip": {"max_len": 8}}}


def _xraw(pid, i, label, weight=1.0, source="teacher", split="train",
          run=None, ctx=None):
    return {"point_id": pid, "ts": f"2026-01-01T00:00:{i:02d}Z",
            "decision_id": f"d{i:02d}", "seq": i, "run_id": run,
            "context": dict(ctx) if ctx is not None
            else {"amount": f"{i}", "vip": i % 2 == 0},
            "label": label, "weight": weight, "source": source,
            "split": split}


# -- T-PT6: GP3 export ------------------------------------------------------

def test_T_PT6_render_shape():
    p = _xpoint(helps={"a": "Ay"})
    raw = _xraw("ex", 3, "a", weight=0.5, source="extracted", split="calib",
                ctx={"amount": "42 EUR", "vip": True})
    r = render_export_row(p, raw)
    assert list(r["questions"]) == ["ex"]  # one question per row
    q = r["questions"]["ex"]
    assert q["type"] == "choice"
    assert q["instructions"] == "Which ex?"
    assert list(q["criteria"]) == ["c", "a", "b"]  # registered, not sorted
    assert q["criteria"] == {"c": "c", "a": "Ay", "b": "b"}  # help or option
    assert (q["label"], q["weight"]) == ("a", 0.5)
    assert r["state"] == R.render_state_obj(p, raw["context"])
    assert "input" not in r["state"]  # rows store no task_input
    for k in ("point_id", "ts", "decision_id", "seq", "run_id", "context",
              "label", "weight", "source", "split"):
        assert r[k] == raw[k]  # identity carried through
    assert "state" not in raw  # input not mutated


def test_T_PT6_noul_is_two_option_choice():
    p = _xpoint(pid="nu", kind="noul", options=("no", "yes"))
    q = render_export_row(p, _xraw("nu", 1, "yes"))["questions"]["nu"]
    assert q["type"] == "choice"
    assert list(q["criteria"]) == ["no", "yes"]
    assert q["label"] in q["criteria"]


def test_T_PT6_permission_matrix():
    pub = {"data_class": "public", "remote_ok": []}
    assert _fireworks_allowed(pub) is True
    assert _fireworks_allowed(
        {"data_class": "private", "remote_ok": ["fireworks"]}) is True
    assert _fireworks_allowed(
        {"data_class": "private", "remote_ok": []}) is False
    assert _fireworks_allowed(
        {"data_class": "private", "remote_ok": ["other"]}) is False
    assert _fireworks_allowed({"data_class": "private"}) is False
    # dry_run skips non-allowed points and launch excludes their rows (T-AP7).


def test_T_PT6_files_no_heldout_leak_and_marker(tmp_path):
    p = _xpoint()
    raws = [_xraw("ex", i, "abc"[i % 3],
                  split=("train", "calib", "heldout")[i % 3])
            for i in range(9)]
    rows = [render_export_row(p, r) for r in raws]
    tr, _, jobs, _ = _trainer(tmp_path)
    tr.launch(rows, "smoke", "idem-pt6-x")
    want = {}
    for r in rows:
        want.setdefault(r["split"], []).append(
            {"questions": r["questions"], "state": r["state"]})
    for split, name in (("train", "train"), ("calib", "dev"),
                        ("heldout", "heldout")):
        got = [json.loads(line) for line in
               (jobs / "idem-pt6-x" / f"{name}.jsonl").read_text().splitlines()]
        assert got == want.get(split, [])
    held_ids = {id(r) for r in want.get("heldout", [])}
    assert not (held_ids & {id(r) for r in want.get("train", [])})
    meta = json.loads((jobs / "idem-pt6-x" / "meta.json").read_text())
    assert meta["export_marker"] == {"ts": "2026-01-01T00:00:08Z",
                                     "decision_id": "d08", "seq": 8}


# -- T-FW1 (updated): labels, weights, splits --------------------------------

def _dec(options, i, **kw):
    r = {"options": list(options), "ts": f"2026-01-01T00:00:{i:02d}Z",
         "decision_id": f"d{i:02d}", "seq": i}
    r.update(kw)
    return r


def test_T_FW1_label_precedence_and_weights():
    opts = ["a", "b", "c"]
    p = _xpoint(options=opts)
    rows = [
        _dec(opts, 1, outcome={"correct_choice": "b"}, teacher_choice="a",
             extracted_choice="c"),                       # outcome_correct
        _dec(opts, 2, outcome={"ok": True}, final_choice="c",
             teacher_choice="a"),                          # outcome_ok
        _dec(opts, 3, teacher_choice="a",
             extracted_choice="b"),                        # teacher
        _dec(opts, 4, extracted_choice="b"),               # extracted
        _dec(opts, 5, outcome={"ok": False}),              # excluded
        _dec(opts, 6),                                     # unlabeled: excluded
        _dec(opts, 7, teacher_choice="zzz"),               # not an option
    ]
    got = L.labeled_rows(rows, p, teacher_weight=0.8, extract_weight=0.5)
    assert [(lab, src, w) for _, lab, src, w in got] == [
        ("b", "outcome_correct", 1.0), ("c", "outcome_ok", 1.0),
        ("a", "teacher", 0.8), ("b", "extracted", 0.5)]
    for (_r, lab, _src, w) in got:
        q = render_export_row(
            p, _xraw("ex", 0, lab, weight=w))["questions"]["ex"]
        assert (q["label"], q["weight"]) == (lab, w)


def test_T_FW1_run_split_and_time_order():
    rows = []
    for run in ("run1", "run2"):
        for i in range(4):
            n = 0 if run == "run1" else 4
            rows.append(_xraw("ex", n + i, "a", source="extracted",
                              run=run))
    out = split_rows(rows, 0.3, 0.0)
    by_run = {}
    for r in out:
        by_run.setdefault(r["run_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_run.values())  # whole runs, one side
    assert by_run["run1"] == {"train"}  # earliest run always trains
    assert by_run["run2"] == {"heldout"}  # latest run held out
    assert [r["seq"] for r in out] == sorted(r["seq"] for r in out)


def test_T_FW1_synthetic_train_only():
    rows = [_xraw("ex", i, "a",
                  source="synthetic_gt" if i >= 8 else "teacher")
            for i in range(10)]
    out = split_rows(rows, 0.2, 0.1)
    assert all(r["split"] == "train"
               for r in out if r["source"] == "synthetic_gt")
    control = split_rows([dict(r, source="teacher") for r in rows], 0.2, 0.1)
    assert {r["split"] for r in control if r["seq"] >= 8} != {"train"}


# -- T-FW10 (changed): pilot schema parity + cookbook checks ------------------

_PILOT = (Path(__file__).parent.parent / "pilots" / "refund" / "data"
          / "fireworks" / "decision_train.jsonl")


def _check_schema(row):
    assert set(row) == {"state", "questions"}
    assert isinstance(row["state"], dict) and row["state"]
    assert isinstance(row["questions"], dict) and row["questions"]
    for q in row["questions"].values():
        assert set(q) == {"type", "instructions", "criteria", "label",
                          "weight"}
        assert q["type"] == "choice"
        assert isinstance(q["instructions"], str) and q["instructions"]
        assert isinstance(q["criteria"], dict) and len(q["criteria"]) >= 2
        assert all(isinstance(k, str) and isinstance(v, str)
                   for k, v in q["criteria"].items())
        assert q["label"] in q["criteria"]
        assert isinstance(q["weight"], (int, float)) and q["weight"] > 0


def _three_fixtures():
    ch = render_export_row(
        _xpoint(helps={"a": "Ay"}),
        _xraw("ex", 1, "a", ctx={"amount": "12 EUR", "vip": False}))
    nu = render_export_row(
        _xpoint(pid="nu", kind="noul", options=("no", "yes")),
        _xraw("nu", 2, "yes", ctx={"amount": "7", "vip": True}))
    ex = render_export_row(
        _xpoint(),
        _xraw("ex", 3, "b", weight=0.5, source="extracted"))
    return [ch, nu, ex]


def test_T_FW10_schema_matches_pilot():
    if not _PILOT.is_file():
        pytest.skip("private pilot fixtures are not exported")
    pilot_rows = [json.loads(line) for line in
                  _PILOT.read_text().splitlines()[:3]]
    assert len(pilot_rows) == 3
    for pr in pilot_rows:
        _check_schema(pr)
    for ours in _three_fixtures():
        _check_schema({k: ours[k] for k in ("state", "questions")})


def _cookbook_dir():
    for cand in (os.environ.get("DECIDEONCE_COOKBOOK_DIR", ""),
                 os.path.expanduser("~/projects/fw-cookbook-src/cookbook")):
        if cand and (Path(cand) / "training" / ".venv" / "bin"
                     / "python").is_file():
            return Path(cand)
    return None


_CK_SCRIPT = (
    "import json, sys\n"
    "from training.utils.decision import label_index, options_for\n"
    "rows = json.load(open(sys.argv[1]))\n"
    "for r in rows:\n"
    "    for q in r['questions'].values():\n"
    "        keys = [k for k, _ in options_for(q)]\n"
    "        assert len(keys) >= 2\n"
    "        assert label_index(q, keys) is not None\n"
    "print('COOKBOOK-OK', len(rows))\n")


@pytest.mark.cookbook
def test_T_FW10_cookbook_checks(tmp_path):
    cb = _cookbook_dir()
    if cb is None:
        pytest.skip("cookbook venv not found")
    if not _PILOT.is_file():
        pytest.skip("private pilot fixtures are not exported")
    pilot = json.loads(_PILOT.read_text().splitlines()[0])
    rows = [pilot] + [{k: r[k] for k in ("state", "questions")}
                      for r in _three_fixtures()]
    data = tmp_path / "rows.json"
    data.write_text(json.dumps(rows))
    proc = subprocess.run(
        [str(cb / "training" / ".venv" / "bin" / "python"), "-c",
         _CK_SCRIPT, str(data)],
        capture_output=True, text=True, cwd=str(cb), timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "COOKBOOK-OK 4" in proc.stdout
