"""Fireworks scoring worker (M11 F6, GP6): the proven session method.

This file is THREE things at once, so the protocol has one home:

1. The REAL worker program (``serve``): run with the cookbook venv
   python, cwd = cookbook root, key only in its environment. It opens
   one serverless session (``setup_serverless_training``, as proven
   on the private refund pilot), loads the adapter, prints READY,
   then serves JSON-line requests ``{"id", "rows"}`` with ``{"id",
   "logprobs", "tokens_in", "seconds"}`` or ``{"id", "error"}``.
   Live-only (needs the key, the session and the adapter). The
   protocol rides a dedicated pipe (Y6), never stdout: prints go to
   the log stream, so cookbook chatter can't corrupt replies.
2. The FAKE worker program (``fake``): the same JSON lines with a
   planted ``label_hint`` signal, scripted failures for tests. Runs on
   any python, stdlib only, fully offline.
3. The CLIENT (``WorkerSession``): spawns 1 or 2, one request in
   flight (a lock; gateway threads queue), ``decider.timeout_s`` per
   call, at most one failure-restart per 10 minutes, worker-side idle
   exit after ``fireworks.worker_idle_min``.

Venv-safety: this file imports STDLIB ONLY at module level (pinned by
T-PT7: an AST test). The real worker imports ``training.*`` inside
``real_main`` — the server process never does (GP1).

Protocol details:
- Labels: the cookbook skips unlabelled questions, so the worker
  inserts the FIRST option (received order) as a dummy label in each
  question before rendering (``insert_dummy_labels``); the output
  carries no label.
- Logprob order: the cookbook sorts criteria when it assigns option
  letters; the worker reorders each question's logprobs back to the
  received (registered) order (``reorder_to_registered``). One entry
  per scored question, in (row, sorted question-id) order — 1:1 with
  rows for our one-question export rows.
- ``tokens_in`` is the GP12 formula (rendered json / 4, same as
  ``FireworksTrainer._row_tokens``), so estimates and settled scoring
  spend use one formula until L3 reconciles against the bill.
- READY: the worker's first protocol line, after the adapter loads.
  The client waits up to ``decider.timeout_s`` (wall time) at spawn,
  then reports ``warming`` until READY arrives. Prints (recipe
  chatter) travel the log stream, never the protocol pipe.
- Idle: the worker exits 0 after ``idle_s`` without a request, which
  closes the serverless session; the client restarts lazily without
  counting it as a failure. Any other death, a timeout or a protocol
  violation counts as a failure: the next spawn is a restart, throttled
  to one per 10 minutes (``restart_throttled`` until the window ends).
- Topology (Z6): the MCP server and the gateway are separate processes
  with separate session deciders, so concurrent private traffic in both
  holds two serverless sessions. Spawns are lazy (first scored call)
  and idle workers exit after ``worker_idle_min``, which bounds the
  overlap; sharing one session across processes would need a worker
  socket server, which is a bigger change than this item.
"""
from __future__ import annotations

import json
import os
import re
import select
import subprocess
import sys
import threading
import time
from pathlib import Path

#: Failure-restart throttle window (GP6: at most one restart per 10 min).
RESTART_WINDOW_S = 600.0

#: env var carrying the model ref to the worker (X3: it holds the
#: account id, so like the key it travels in the environment, never argv).
MODEL_REF_ENV = "DECIDEONCE_MODEL_REF"

#: env var carrying the protocol pipe's fd number into the worker (Y6:
#: the JSON-lines protocol rides a dedicated pipe, never stdout, so
#: cookbook prints can't corrupt it and buffered reads can't hide it).
PROTO_FD_ENV = "DECIDEONCE_PROTO_FD"

_FW_KEY_RE = re.compile(r"fw_[A-Za-z0-9]{8,}")


def _mask_log_line(line: str) -> str:
    """Mask secrets in a worker.log line (X3). Server-side only: the
    worker programs never call it (venv-safety, see the module doc)."""
    from .trainer_log import redact_account

    return redact_account(_FW_KEY_RE.sub("fw_***", line))


def _proto_fd():
    """The protocol sink (Y6): the parent's dedicated pipe, or None
    when the worker wasn't spawned by a session (never stdout: the
    protocol must not share a fd with prints)."""
    try:
        return os.fdopen(int(os.environ.get(PROTO_FD_ENV, "")), "w",
                         buffering=1)
    except (ValueError, OSError):
        return None
#: Forward chunk, as option_logprobs' default.
CHUNK = 64


class WorkerError(Exception):
    """The worker is alive but refused the request (its ``error`` text)."""


class WorkerUnavailable(Exception):
    """No score available; ``reason`` is one of warming, timeout,
    worker_died, restart_throttled, protocol, spawn_failed, closed."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


# -- pure protocol helpers ---------------------------------------------------

def insert_dummy_labels(rows: list[dict]) -> list[dict]:
    """Copies of ``rows`` with each question labelled by its first option.

    The input is not mutated. Questions without a non-empty criteria
    mapping are left alone (the cookbook then raises, and the worker
    answers ``error``).
    """
    out = []
    for row in rows:
        questions = row.get("questions")
        if not isinstance(questions, dict):
            out.append(row)
            continue
        new_qs = {}
        for qid, q in questions.items():
            if not isinstance(q, dict):
                new_qs[qid] = q
                continue
            crit = q.get("criteria")
            if isinstance(crit, dict) and crit:
                q = {**q, "label": next(iter(crit))}
            new_qs[qid] = q
        out.append({**row, "questions": new_qs})
    return out


def reorder_to_registered(question: dict, logprobs: list[float]) -> list[float]:
    """Cookbook (sorted-criteria) logprobs back to received option order."""
    if question.get("type") != "choice":
        return list(logprobs)
    keys = list((question.get("criteria") or {}))
    order = sorted(keys)
    if keys == order:
        return list(logprobs)
    return [logprobs[order.index(k)] for k in keys]


def tokens_of(rows: list[dict]) -> int:
    """GP12 tokens over rendered rows (same formula as the estimate)."""
    total = 0
    for r in rows:
        payload = {"state": r.get("state"), "questions": r.get("questions")}
        total += max(1, len(json.dumps(payload, sort_keys=True)) // 4)
    return total


def _model_tail(model: str) -> str:
    tail = (model or "model").rsplit("/", 1)[-1] or "model"
    return re.sub(r"[^A-Za-z0-9]+", "-", tail).strip("-")[:40] or "model"


# -- client -------------------------------------------------------------------

class WorkerSession:
    """One worker process, one request in flight. Thread-safe.

    ``worker_argv`` overrides the spawn command for fakes (a list whose
    first item is the python); the default spawns the real worker with
    the cookbook venv python. ``clock`` is an injectable monotonic
    source for the restart throttle (tests); grace waits use wall time.
    """

    def __init__(self, cfg: dict, model: str, jobs_dir,
                 key: str, worker_argv: list[str] | None = None,
                 clock=None):
        self.timeout_s = float(cfg["decider"]["timeout_s"])
        self.idle_s = float(cfg["fireworks"]["worker_idle_min"]) * 60.0
        self.cookbook = str(cfg["fireworks"]["cookbook_dir"])
        self.model = model
        self.key = key
        self.dir = Path(jobs_dir) / f"worker-{_model_tail(model)}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._argv = worker_argv
        self._clock = clock or time.monotonic
        self._lock = threading.Lock()
        self._log = None
        self._log_lock = threading.Lock()
        self._err_pumps: list[threading.Thread] = []
        self._proc: subprocess.Popen | None = None
        self._proto_r: int | None = None
        self._proto_buf = bytearray()
        self._last_rc: int | None = None
        self._ready = False
        self._closed = False
        self._restarts: list[float] = []
        self._ids = 0

    # -- lifecycle ---------------------------------------------------------
    def _spawn_argv(self) -> tuple[list[str], str | None, bool]:
        if self._argv is not None:
            return list(self._argv), None, False
        venv = str(Path(self.cookbook) / "training" / ".venv" / "bin"
                   / "python")
        # X3: the model ref is passed in the environment (below), never
        # in argv: process listings and audit logs capture command lines.
        return ([venv, str(Path(__file__).resolve()), "serve",
                 str(self.dir), str(self.idle_s)],
                self.cookbook, True)

    def _open_log(self) -> None:
        """Open worker.log 0600 (X3: it can carry keys and model refs)."""
        if self._log is not None:
            return
        path = self.dir / "worker.log"
        fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        os.chmod(path, 0o600)  # tighten files from before X3
        self._log = os.fdopen(fd, "a")

    def _pump_stderr(self, proc: subprocess.Popen) -> None:
        """Drain one child's stderr into worker.log, masked (X3). Ends on
        EOF; never touches self._lock, so close() can't deadlock it."""
        log = self._log
        try:
            for line in proc.stderr:
                with self._log_lock:
                    log.write(_mask_log_line(line))
                    log.flush()
        except (ValueError, OSError):
            pass  # closed under us

    def _spawn(self) -> None:
        argv, cwd, needs_key = self._spawn_argv()
        if needs_key and not self.key:
            raise WorkerUnavailable("spawn_failed", "no fireworks key")
        self._open_log()
        # Y6: the protocol rides a dedicated pipe (never stdout, so
        # cookbook prints can't corrupt it). The fd number travels in
        # the environment; our write end closes right after the spawn
        # (else the read end never sees EOF when the child dies).
        proto_r, proto_w = os.pipe()
        env = dict(os.environ)
        env[PROTO_FD_ENV] = str(proto_w)
        if needs_key:
            env["FIREWORKS_API_KEY"] = self.key
            env[MODEL_REF_ENV] = self.model
        try:
            self._proc = subprocess.Popen(
                argv, cwd=cwd, env=env, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                pass_fds=(proto_w,),
                start_new_session=True, text=True, bufsize=1)
        except OSError as e:
            os.close(proto_r)
            os.close(proto_w)
            raise WorkerUnavailable("spawn_failed", str(e))
        os.close(proto_w)
        self._proto_r = proto_r
        self._proto_buf = bytearray()
        pump = threading.Thread(target=self._pump_stderr,
                                args=(self._proc,), daemon=True)
        pump.start()
        self._err_pumps.append(pump)
        self._ready = False
        # Block up to one call timeout for READY (wall time); a slow
        # start keeps going in the background and reads as warming.
        deadline = time.monotonic() + self.timeout_s
        while True:
            line = self._proto_readline(max(0.0, deadline - time.monotonic()))
            if line is None:
                return
            if line == "":
                self._reap()
                raise WorkerUnavailable("worker_died",
                                        "exited before READY")
            if line.strip() == "READY":
                self._ready = True
                return
            with self._log_lock:
                self._log.write(_mask_log_line(f"[early] {line}"))
                self._log.flush()

    def _proto_readline(self, timeout: float) -> str | None:
        """One protocol line (Y6): unbuffered os.read under select.

        Returns "" on EOF, None on timeout. Buffered readline() is
        banned here: its readahead hides data from select, which reads
        a ready worker as warming forever.
        """
        buf = self._proto_buf
        deadline = time.monotonic() + max(timeout, 0.0)
        while True:
            nl = buf.find(b"\n")
            if nl >= 0:
                line = bytes(buf[:nl + 1])
                del buf[:nl + 1]
                return line.decode("utf-8", "replace")
            wait = max(0.0, deadline - time.monotonic())
            ready, _, _ = select.select([self._proto_r], [], [], wait)
            if not ready:
                return None
            try:
                chunk = os.read(self._proto_r, 65536)
            except OSError:
                return ""
            if not chunk:
                return ""
            buf += chunk

    def _reap(self) -> int | None:
        proc, self._proc = self._proc, None
        if self._proto_r is not None:
            try:
                os.close(self._proto_r)
            except OSError:
                pass
            self._proto_r = None
        self._proto_buf = bytearray()
        if proc is None:
            return self._last_rc
        try:
            rc = proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = proc.wait(timeout=5)
        self._last_rc = rc
        return rc

    def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)

    def _note_restart_or_raise(self) -> None:
        now = self._clock()
        self._restarts = [t for t in self._restarts
                          if now - t < RESTART_WINDOW_S]
        if self._restarts:
            raise WorkerUnavailable("restart_throttled",
                                    "one restart per 10 min")
        self._restarts.append(now)

    def _ensure_started(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if self._proc is not None:
            self._reap()
        if self._last_rc is not None:
            # Account the reaped exit once: non-zero is a failure, so
            # this spawn is a throttled restart; 0 is an idle exit.
            # Y7: clear only after the throttle check passes. Clearing
            # first lets a throttled call forget the death, so the
            # next call respawns inside the window instead of waiting.
            rc = self._last_rc
            if rc != 0:
                self._note_restart_or_raise()
            self._last_rc = None
        self._spawn()

    def _poll_ready(self) -> bool:
        """Consume the protocol pipe until READY (True), EOF (raises)
        or dry (False)."""
        while True:
            line = self._proto_readline(0)
            if line is None:
                return False
            if line == "":
                self._reap()
                raise WorkerUnavailable("worker_died",
                                        "exited while warming")
            if line.strip() == "READY":
                self._ready = True
                return True
            with self._log_lock:
                self._log.write(_mask_log_line(f"[early] {line}"))
                self._log.flush()

    # -- protocol ------------------------------------------------------------
    def score(self, rows: list[dict], timeout_s: float | None = None) -> dict:
        """Score rows: {logprobs, tokens_in, seconds}.

        Raises WorkerError (worker refused) or WorkerUnavailable
        (retry later; F7 maps these to teacher mode). Y5: callers
        pass their own timeout (eval batches need far more than the
        3 s serve default); None keeps it.
        """
        if timeout_s is None:
            timeout_s = self.timeout_s
        with self._lock:
            if self._closed:
                raise WorkerUnavailable("closed", "session closed")
            self._ensure_started()
            if not self._ready and not self._poll_ready():
                raise WorkerUnavailable("warming", "worker starting")
            self._ids += 1
            req = {"id": self._ids, "rows": rows}
            try:
                self._proc.stdin.write(json.dumps(req) + "\n")
                self._proc.stdin.flush()
            except (BrokenPipeError, ValueError):
                self._reap()
                raise WorkerUnavailable("worker_died", "stdin broken")
            line = self._proto_readline(timeout_s)
            if line is None:
                self._kill()
                self._reap()
                raise WorkerUnavailable("timeout",
                                        f"no reply in {timeout_s}s")
            if line == "":
                self._reap()
                raise WorkerUnavailable("worker_died", "protocol EOF")
            try:
                resp = json.loads(line)
            except ValueError:
                self._kill()
                self._reap()
                raise WorkerUnavailable("protocol", "bad json")
            if not isinstance(resp, dict) or resp.get("id") != req["id"]:
                self._kill()
                self._reap()
                raise WorkerUnavailable("protocol", "id mismatch")
            if "error" in resp:
                raise WorkerError(str(resp["error"]))
            return {"logprobs": resp["logprobs"],
                    "tokens_in": resp["tokens_in"],
                    "seconds": resp["seconds"]}

    def __enter__(self) -> "WorkerSession":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def status(self) -> dict:
        """F9: live state for do_report (no account ids leave here)."""
        proc = self._proc
        alive = proc is not None and proc.poll() is None
        return {"running": alive, "ready": bool(alive and self._ready),
                "restarts_10min": len([
                    t for t in self._restarts
                    if self._clock() - t < RESTART_WINDOW_S])
                if alive else 0}

    def close(self) -> None:
        """Terminate the worker. Idempotent; the session is not reused."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._kill()
                self._reap()
            finally:
                pumps, self._err_pumps = self._err_pumps, []
                for pump in pumps:
                    pump.join(timeout=5)
                if self._log is not None:
                    with self._log_lock:
                        self._log.close()
                    self._log = None


# -- real worker program (live-only; runs in the cookbook venv) ----------------

def _serve_loop(handle, idle_s: float, out) -> int:
    """Serve stdin until EOF or (idle_s > 0) an idle gap (0).

    ``handle`` maps rows -> logprobs; rows are dummy-labelled first
    (the cookbook skips unlabelled questions). ``out`` is the Y6
    protocol sink (never stdout: prints would corrupt the stream).
    """
    while True:
        if idle_s > 0:
            ready, _, _ = select.select([sys.stdin], [], [], idle_s)
            if not ready:
                return 0
        line = sys.stdin.readline()
        if line == "":
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            rows = req["rows"]
            if not isinstance(rows, list):
                raise ValueError("rows must be a list")
        except (ValueError, KeyError, AttributeError) as e:
            rid = req.get("id") if isinstance(req, dict) else None
            out.write(json.dumps(
                {"id": rid, "error": f"bad request: {e}"}) + "\n")
            out.flush()
            continue
        t0 = time.time()
        try:
            logprobs = handle(insert_dummy_labels(rows))
            resp = {"id": req.get("id"), "logprobs": logprobs,
                    "tokens_in": tokens_of(rows),
                    "seconds": time.time() - t0}
        except Exception as e:  # noqa: BLE001 - reported, worker survives
            resp = {"id": req.get("id"),
                    "error": f"{type(e).__name__}: {e}"}
        out.write(json.dumps(resp) + "\n")
        out.flush()


def real_main(argv: list[str]) -> int:
    """serve <workdir> <idle_s>; the model ref comes from the environment.

    X3: the ref carries the account id, so like the key it travels in
    DECIDEONCE_MODEL_REF, never in argv (process listings and audit
    logs capture command lines). Imports training.* (venv-only).
    """
    model = os.environ.get(MODEL_REF_ENV, "")
    if not model:
        sys.stderr.write(f"worker: {MODEL_REF_ENV} is not set\n")
        return 1
    workdir, idle_s = argv[0], float(argv[1])
    # NOTE: every training.* import lives in this function: the server
    # process must never import the cookbook (GP1), and the module top
    # level stays stdlib-only so the venv can run this file directly.
    from contextlib import ExitStack

    from training.examples.decision.kalshi_v1 import \
        train_kalshi_decision as tk
    from training.recipes import sft_loop
    from training.renderer import get_renderer
    from training.renderer.tokenizer import get_tokenizer
    from training.utils import read_api_extra_headers_env
    from training.utils.decision import decision_datums, option_logprobs
    from training.utils.serverless import setup_serverless_training
    args = tk.parse_args(["--log-path", workdir])
    cfg = tk.build_config(args)
    renderer_name = sft_loop._resolved_renderer_name(
        tokenizer_model=cfg.tokenizer_model, renderer_name=cfg.renderer_name,
        thinking_trace_history_mode=cfg.thinking_trace_history_mode,
        renderer_name_is_resolved=cfg.renderer_name_is_resolved)
    renderer = get_renderer(renderer_name,
                            get_tokenizer(cfg.tokenizer_model))
    stack = ExitStack()
    (_service, client, _ckpt, _job_id, max_seq_len) = \
        setup_serverless_training(
            cfg, api_key=os.environ["FIREWORKS_API_KEY"],
            base_url=os.environ.get("FIREWORKS_BASE_URL",
                                    "https://api.fireworks.ai"),
            additional_headers=read_api_extra_headers_env(), stack=stack,
            close_session_on_exit=True)
    client.load_adapter(model)
    proto = _proto_fd()
    if proto is None:
        sys.stderr.write(f"worker: {PROTO_FD_ENV} is not set\n")
        return 1
    proto.write("READY\n")
    proto.flush()

    def handle(rows: list[dict]) -> list[list[float]]:
        datums = []
        owners = []
        for i, row in enumerate(rows):
            questions = row.get("questions")
            if not isinstance(questions, dict) or not questions:
                raise ValueError(f"row {i}: no questions")
            for qid in sorted(questions):
                one = {"state": row.get("state"),
                       "questions": {qid: questions[qid]}}
                got = decision_datums(
                    one, renderer=renderer, renderer_name=renderer_name,
                    max_seq_len=max_seq_len)
                if not got:
                    raise ValueError(
                        f"row {i} question {qid!r}: exceeds max_seq_len")
                owners.append(questions[qid])
                datums.extend(d for _, d in got)
        rows_out = option_logprobs(client, datums, chunk=CHUNK)
        return [reorder_to_registered(q, lp)
                for q, lp in zip(owners, rows_out)]

    with stack:
        return _serve_loop(handle, idle_s, proto)


# -- fake worker program (offline; any python) -----------------------------------

def _fake_logprobs(rows: list[dict]) -> list[list[float]]:
    import math

    out = []
    for row in rows:
        questions = row.get("questions") or {}
        for qid in sorted(questions):
            q = questions[qid]
            keys = list((q.get("criteria") or {}))
            hint = (row.get("state") or {}).get("label_hint")
            if hint in keys:
                out.append([-0.105 if k == hint else -8.0 for k in keys])
            else:
                out.append([math.log(1.0 / len(keys))] * len(keys)
                          if keys else [])
    return out


def fake_main(argv: list[str]) -> int:
    """fake <mode> [param]: ok [delay] | error [msg] | slow <s> |
    die-after <n> | idle <s> | mute [s] | error-after <n>. Same JSON
    lines as serve, on the same protocol pipe (stdout when hand-run
    without a session); mute never prints READY (warming tests)."""
    mode = argv[0] if argv else "ok"
    param = argv[1] if len(argv) > 1 else ""
    if mode == "mute":
        time.sleep(float(param or 30))
        return 0
    out = _proto_fd() or sys.stdout
    out.write("READY\n")
    out.flush()
    served = 0
    idle_s = float(param) if mode == "idle" else 0.0
    delay = 0.0
    if mode == "ok" and param:
        delay = float(param)
    while True:
        if idle_s > 0:
            ready, _, _ = select.select([sys.stdin], [], [], idle_s)
            if not ready:
                return 0
        line = sys.stdin.readline()
        if line == "":
            return 0
        line = line.strip()
        if not line:
            continue
        t0 = time.time()
        try:
            req = json.loads(line)
            rows = req["rows"]
        except (ValueError, KeyError, AttributeError) as e:
            rid = req.get("id") if isinstance(req, dict) else None
            out.write(json.dumps(
                {"id": rid, "error": f"bad request: {e}"}) + "\n")
            out.flush()
            continue
        if mode == "error":
            resp = {"id": req.get("id"), "error": param or "fake boom"}
        elif mode == "error-after" and served >= int(param or 0):
            resp = {"id": req.get("id"), "error": "fake boom"}
        elif mode == "die-after" and served >= int(param or 0):
            return 1
        else:
            if mode == "slow":
                time.sleep(float(param or 0))
            elif delay:
                time.sleep(delay)
            served += 1
            resp = {"id": req.get("id"), "logprobs": _fake_logprobs(rows),
                    "tokens_in": tokens_of(rows),
                    "seconds": time.time() - t0}
        out.write(json.dumps(resp) + "\n")
        out.flush()


def main(argv: list[str]) -> int:
    # Y6: prints (ours, the cookbook's, the interpreter's) go to the
    # log stream; only the JSON-lines protocol uses the pipe.
    os.dup2(2, 1)
    if not argv or argv[0] not in ("serve", "fake"):
        sys.stderr.write("usage: fireworks_worker.py serve|fake ...\n")
        return 2
    try:
        if argv[0] == "serve":
            return real_main(argv[1:])
        return fake_main(argv[1:])
    except Exception as e:  # noqa: BLE001 - worker fails loudly, exits 1
        import traceback
        traceback.print_exc()
        sys.stderr.write(f"worker failed: {e}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
