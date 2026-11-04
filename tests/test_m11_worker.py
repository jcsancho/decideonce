"""M11 worker tests (F6): WorkerSession against FakeWorker over real pipes."""
import ast
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import decideonce.fireworks_worker as W
from decideonce.config import load_config
from decideonce.fireworks_worker import (
    WorkerError,
    WorkerSession,
    WorkerUnavailable,
    insert_dummy_labels,
    reorder_to_registered,
    tokens_of,
)

SCRIPT = str(Path(W.__file__).resolve())


def _wcfg(tmp_path, timeout_s=3.0, idle_min=30):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["timeout_s"] = timeout_s
    cfg["fireworks"]["worker_idle_min"] = idle_min
    return cfg


def _sess(tmp_path, *fake_args, clock=None, **cfg_over):
    cfg = _wcfg(tmp_path, **cfg_over)
    argv = [sys.executable, SCRIPT, "fake", *[str(a) for a in fake_args]]
    return WorkerSession(cfg, "accounts/acct-test/models/dk-t-0001",
                         tmp_path / "jobs", key="k", worker_argv=argv,
                         clock=clock)


def _srow(hint, options=("c", "a", "b")):
    return {"state": {"policy": "P", "amount": "1", "label_hint": hint},
            "questions": {"q": {"type": "choice", "instructions": "Which?",
                                "criteria": {o: o for o in options},
                                "label": None, "weight": 1.0}}}


def _peak(lp):
    return max(range(len(lp)), key=lp.__getitem__)


# -- T-PT7 ------------------------------------------------------------------

def test_T_PT7_request_response(tmp_path):
    rows = [_srow("a"), _srow("b")]
    with _sess(tmp_path, "ok") as s:
        out = s.score(rows)
        assert len(out["logprobs"]) == 2
        assert _peak(out["logprobs"][0]) == 1  # a in [c, a, b]
        assert _peak(out["logprobs"][1]) == 2  # b in [c, a, b]
        assert out["tokens_in"] == tokens_of(rows)
        assert out["seconds"] >= 0
        assert (s.dir / "worker.log").is_file()
        first_pid = s._proc.pid
        s.score(rows)  # same worker serves again
        assert s._proc.pid == first_pid
    assert s._proc is None


def test_T_PT7_error_stays_alive(tmp_path):
    with _sess(tmp_path, "error", "bad row") as s:
        with pytest.raises(WorkerError, match="bad row"):
            s.score([_srow("a")])
        pid = s._proc.pid
        with pytest.raises(WorkerError, match="bad row"):
            s.score([_srow("a")])
        assert s._proc.pid == pid  # no restart on request errors


def test_T_PT7_timeout(tmp_path):
    with _sess(tmp_path, "slow", "30", timeout_s=0.2) as s:
        t0 = time.monotonic()
        with pytest.raises(WorkerUnavailable) as e:
            s.score([_srow("a")])
        assert e.value.reason == "timeout"
        assert time.monotonic() - t0 < 5
        assert s._proc is None  # stuck worker killed and reaped


def test_T_PT7_death_restart_throttle(tmp_path):
    now = [1000.0]
    with _sess(tmp_path, "die-after", "1", clock=lambda: now[0]) as s:
        assert _peak(s.score([_srow("a")])["logprobs"][0]) == 1
        pid0 = s._proc.pid
        with pytest.raises(WorkerUnavailable) as e:
            s.score([_srow("a")])
        assert e.value.reason == "worker_died"  # F7 maps this to teacher mode
        assert _peak(s.score([_srow("b")])["logprobs"][0]) == 2  # restart 1
        assert s._proc.pid != pid0
        with pytest.raises(WorkerUnavailable) as e:
            s.score([_srow("a")])
        assert e.value.reason == "worker_died"
        with pytest.raises(WorkerUnavailable) as e:
            s.score([_srow("a")])
        assert e.value.reason == "restart_throttled"  # 1 restart per 10 min
        now[0] += 601
        assert _peak(s.score([_srow("a")])["logprobs"][0]) == 1  # window ends


def test_T_PT7_idle_close_restarts_freely(tmp_path):
    with _sess(tmp_path, "idle", "0.2") as s:
        assert _peak(s.score([_srow("a")])["logprobs"][0]) == 1
        old = s._proc
        time.sleep(0.6)
        assert old.poll() == 0  # worker-side idle exit closes the session
        out = s.score([_srow("b")])  # lazy restart, not throttled
        assert _peak(out["logprobs"][0]) == 2
        assert s._proc.pid != old.pid


def test_T_PT7_one_in_flight_8_threads(tmp_path):
    hints = "abcabcab"
    with _sess(tmp_path, "ok", "0.05") as s:
        barrier = threading.Barrier(8)
        errors = []

        def one(i):
            try:
                barrier.wait(timeout=10)
                lp = s.score([_srow(hints[i])])["logprobs"][0]
                assert _peak(lp) == "cab".index(hints[i]), (i, lp)
            except Exception as e:  # noqa: BLE001 - collected, asserted below
                errors.append(e)

        threads = [threading.Thread(target=one, args=(i,)) for i in range(8)]
        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        dt = time.monotonic() - t0
        assert not errors
        assert not [t for t in threads if t.is_alive()]
        assert dt >= 0.35  # 8 x 0.05s serialized, not interleaved


def test_T_PT7_dummy_label_first_option():
    rows = [{"state": {}, "questions": {
        "q1": {"type": "choice", "criteria": {"b": "B", "a": "A"}},
        "q2": {"type": "choice", "criteria": {"x": "X"}, "label": "x"}}}]
    labelled = insert_dummy_labels(rows)
    assert labelled[0]["questions"]["q1"]["label"] == "b"  # first, received
    assert labelled[0]["questions"]["q2"]["label"] == "x"
    assert "label" not in rows[0]["questions"]["q1"]  # input not mutated
    assert reorder_to_registered(
        {"type": "choice", "criteria": {"b": 1, "a": 1, "c": 1}},
        [1.0, 2.0, 3.0]) == [2.0, 1.0, 3.0]  # sorted [a,b,c] -> [b,a,c]
    assert reorder_to_registered({"type": "noul"}, [1.0, 2.0]) == [1.0, 2.0]


def test_T_PT7_module_is_stdlib_only():
    """The venv runs this file directly: top level must be stdlib-only,
    training.* only inside functions (GP1)."""
    tree = ast.parse(Path(SCRIPT).read_text())
    stdlib = sys.stdlib_module_names
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name.split(".")[0] in stdlib, a.name
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0
            assert (node.module or "").split(".")[0] in stdlib, node.module
    func_src = [n for n in ast.walk(tree) if isinstance(n, (ast.Import,
                                                       ast.ImportFrom))]
    training = [n for n in func_src
                if "training" in ast.dump(n)]
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert training and not any(n in top for n in training)


def test_T_PT7_serve_refuses_without_cookbook(tmp_path):
    proc = subprocess.run(
        [sys.executable, SCRIPT, "serve", str(tmp_path), "0"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=120,
        env={**os.environ, "DECIDEONCE_MODEL_REF": "m"})
    assert proc.returncode == 1
    assert "training" in proc.stderr


def test_X3_serve_refuses_without_model_env(tmp_path):
    env = {k: v for k, v in os.environ.items()
           if k != "DECIDEONCE_MODEL_REF"}
    proc = subprocess.run(
        [sys.executable, SCRIPT, "serve", str(tmp_path), "0"],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=120,
        env=env)
    assert proc.returncode == 1
    assert "DECIDEONCE_MODEL_REF" in proc.stderr


def test_X3_spawn_argv_carries_no_model(tmp_path):
    from decideonce.fireworks_worker import WorkerSession

    cfg = _wcfg(tmp_path)
    s = WorkerSession(cfg, "accounts/someuser-1a2b3c/models/dk-x",
                      tmp_path / "jobs", key="k")
    try:
        argv, _cwd, needs_key = s._spawn_argv()
        assert needs_key is True
        cmd = " ".join(argv)
        assert "someuser-1a2b3c" not in cmd
        assert "accounts/someuser-1a2b3c/models/dk-x" not in cmd
        # the workdir carries the tail only (no account id): stable names
        assert "worker-dk-x" in cmd
    finally:
        s.close()


def test_X3_worker_log_0600_and_masked(tmp_path):
    from decideonce.fireworks_worker import WorkerSession

    cfg = _wcfg(tmp_path)
    script = ("import sys, time\n"
              "sys.stdout.write('READY\\n'); sys.stdout.flush()\n"
              "sys.stderr.write('leaked fw_abcdefgh12345678 and '\n"
              "  'accounts/someuser-1a2b3c/models/m\\n'); sys.stderr.flush()\n"
              "time.sleep(30)\n")
    argv = [sys.executable, "-c", script]
    s = WorkerSession(cfg, "accounts/acct-test/models/dk-t-0001",
                      tmp_path / "jobs", key="k", worker_argv=argv)
    try:
        s._ensure_started()
        deadline = time.monotonic() + 10
        log = s.dir / "worker.log"
        while time.monotonic() < deadline:
            if log.is_file() and "leaked" in log.read_text():
                break
            time.sleep(0.1)
        body = log.read_text()
        assert "fw_abcdefgh12345678" not in body
        assert "someuser-1a2b3c" not in body
        assert "fw_***" in body
        assert "accounts/acct-…/models/m" in body
        assert log.stat().st_mode & 0o777 == 0o600
    finally:
        s.close()
