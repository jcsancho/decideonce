"""Live Fireworks trainer (M11, GP1–GP3): the pinned cookbook recipe runs
as a detached child process, never as an import.

Rendered export rows (state + questions, F5) go in; `estimate()` prices
exactly what Fireworks sees (GP12). F1 covers the pin check, argv/env/cwd
construction and estimate; F3 the lifecycle (pid-reuse guard, log proof,
timeout, cancel); F4 the reconcile rule.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from .store import row_key
from .trainer import Trainer
from .trainer_log import parse_account_id, parse_log

RECIPE_MODULE = "training.examples.decision.kalshi_v1.train_kalshi_decision"
MODEL_ID_PREFIX = "dk"
# Pid-reuse skew (GP2), SIGTERM grace before SIGKILL, and our exit codes.
REUSE_SKEW_S = 5
KILL_GRACE_S = 30
EXIT_TIMEOUT = 124
EXIT_CANCELLED = 125

RUN_SH = """#!/usr/bin/env bash
cd -- "$(dirname -- "$0")"  # pid/log/exit land here, not in the spawn cwd
echo $$ > pid; date +%s > started_at
set -o pipefail
"$@" 2>&1 | sed -E 's/fw_[A-Za-z0-9]{8,}/fw_***/g' > train.log
echo $? > exit_code; date +%s > finished_at
"""
# NOTE: no `sed -l` (GNU-only): macOS sed rejects it, and status reads the
# exit_code file, so block-buffered log writes are harmless.


def _default_runner(argv: list[str], env: dict, cwd: str):
    """Detached spawn (GP2): a new session, so the job survives us and the
    whole group dies on killpg. run.sh redirects everything to train.log."""
    return subprocess.Popen(argv, env=env, cwd=cwd, start_new_session=True,
                            stdin=subprocess.DEVNULL,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def _parse_etime(out: str) -> float | None:
    """`ps -o etime=` ([DD-]HH:MM:SS, or MM:SS) to seconds. 1s resolution."""
    s = out.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        try:
            days = int(d)
        except ValueError:
            return None
    parts = s.split(":")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, sec = nums
    elif len(nums) == 2:
        h, m, sec = 0, nums[0], nums[1]
    else:
        return None
    return float(days * 86400 + h * 3600 + m * 60 + sec)


class FireworksTrainer(Trainer):
    """Fireworks Serverless Training via the pinned cookbook (GP1).

    key is resolved by the caller (envsave, F9) and set only in the
    child's environment. runner(argv, env, cwd) is a test hook (T-PT1);
    recipe replaces the cookbook argv prefix (F3's fake child script).
    """

    needs_key = True

    def __init__(self, cfg: dict, jobs_dir: str | Path,
                 key: str | None = None, runner=None,
                 recipe: list[str] | None = None):
        self.cfg = cfg
        self.fw = cfg.get("fireworks", {})
        self.jobs_dir = Path(jobs_dir)
        self.key = key
        self._runner = runner or _default_runner
        self._recipe = recipe
        # Live Popen handles, reaped on status: without poll() every
        # finished run.sh lingers as our zombie until this process exits.
        # (A restart is still safe: orphans reparent to init, which reaps.)
        self._procs: dict[str, subprocess.Popen] = {}

    # -- pin check (GP1, T-PT2) ------------------------------------------
    def _cookbook(self) -> Path:
        return Path(self.fw.get("cookbook_dir") or "")

    def _venv_python(self) -> Path:
        return self._cookbook() / "training" / ".venv" / "bin" / "python"

    def pin_status(self) -> dict:
        """ok | cookbook_missing | cookbook_mismatch (GP1 startup check)."""
        cb = self._cookbook()
        if not str(cb) or not cb.is_dir():
            return {"status": "cookbook_missing",
                    "detail": f"cookbook_dir is not a directory: {cb}"}
        if not self._venv_python().is_file():
            return {"status": "cookbook_missing",
                    "detail": f"cookbook venv python missing: {self._venv_python()}"}
        want = self.fw.get("cookbook_commit", "")
        try:
            proc = subprocess.run(
                ["git", "-C", str(cb), "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=10,
                stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return {"status": "cookbook_mismatch",
                    "detail": "git rev-parse failed"}
        got = (proc.stdout or "").strip()
        if proc.returncode != 0 or not got:
            return {"status": "cookbook_mismatch",
                    "detail": "cookbook_dir is not a git checkout"}
        if got != want:
            return {"status": "cookbook_mismatch",
                    "detail": f"cookbook at {got}, want {want}"}
        return {"status": "ok", "detail": f"cookbook at {got}"}

    # -- estimate (GP12) ---------------------------------------------------
    @staticmethod
    def _row_tokens(row: dict) -> int:
        """Tokens of exactly what Fireworks sees (json / 4, policy in state)."""
        payload = {"state": row["state"], "questions": row["questions"]}
        return max(1, len(json.dumps(payload, sort_keys=True)) // 4)

    def estimate(self, rows: list[dict], kind: str) -> dict:
        """GP12 on rendered rows (each carries split: train|calib|heldout).

        Training: train tokens × epochs × train price. Evaluation: in-job
        dev scoring (one per epoch plus step 0) for every kind, plus the
        post-job calibration + held-out scoring for full jobs (smoke jobs
        are checked from their own eval lines, GP6).
        """
        epochs = int(self.fw.get("epochs", 2) or 2)
        train_p = float(self.fw.get("train_price_per_mtok", 5.86))
        score_p = float(self.fw.get("score_price_per_mtok", 1.96))
        smoke_n = int(self.fw.get("smoke_examples", 32) or 32)
        train = [r for r in rows if r.get("split") == "train"]
        if kind == "smoke":
            train = train[:smoke_n]
        dev = [r for r in rows if r.get("split") == "calib"]
        held = [r for r in rows if r.get("split") == "heldout"]
        train_toks = sum(self._row_tokens(r) for r in train)
        dev_toks = sum(self._row_tokens(r) for r in dev)
        train_usd = train_toks * epochs * train_p / 1e6
        eval_toks = (epochs + 1) * dev_toks
        eval_rows = (epochs + 1) * len(dev)
        if kind == "full":
            post = dev + held
            eval_toks += sum(self._row_tokens(r) for r in post)
            eval_rows += len(post)
        eval_usd = eval_toks * score_p / 1e6
        return {"estimate_usd": train_usd + eval_usd,
                "train_usd": train_usd, "eval_usd": eval_usd,
                "train_rows": len(train), "eval_rows": eval_rows,
                "train_tokens": train_toks, "eval_tokens": eval_toks,
                "epochs": epochs, "kind": kind}

    # -- argv (GP1–GP2, T-PT1) ----------------------------------------------
    def recipe_argv(self, kind: str, job_dir: Path, n_train: int) -> list[str]:
        """[python, -m, module, flags...] for the recipe (or the test recipe)."""
        fw = self.fw
        batch = max(1, min(int(fw.get("batch_rows", 24) or 24),
                           n_train or 1))
        argv = [
            *self._recipe_prefix(),
            "--train-path", str(job_dir / "train.jsonl"),
            "--dev-path", str(job_dir / "dev.jsonl"),
            "--skip-download",
            "--lora-rank", str(int(fw.get("lora_rank", 64) or 64)),
            "--learning-rate", str(float(fw.get("learning_rate", 1e-4))),
            "--batch-size", str(batch),
            "--epochs", str(int(fw.get("epochs", 2) or 2)),
            "--log-path", str(job_dir / "cookbook"),
        ]
        if kind == "smoke":
            argv += ["--max-examples",
                    str(int(fw.get("smoke_examples", 32) or 32))]
        else:
            argv += ["--output-model-id", self._model_id()]
        return argv

    def _recipe_prefix(self) -> list[str]:
        if self._recipe is not None:
            return list(self._recipe)
        return [str(self._venv_python()), "-m", RECIPE_MODULE]

    def _model_id(self) -> str:
        """dk-<project>-<yyyymmdd-hhmm>, lowercase [a-z0-9-], ≤ 63 chars."""
        proj = re.sub(r"[^a-z0-9]+", "-",
                      str(self.cfg.get("store", {}).get("project", "job")
                          ).lower()).strip("-") or "job"
        stamp = time.strftime("%Y%m%d-%H%M", time.gmtime())
        proj = proj[:63 - len(MODEL_ID_PREFIX) - 1 - len(stamp) - 1]
        return f"{MODEL_ID_PREFIX}-{proj}-{stamp}"

    # -- launch (GP2) ------------------------------------------------------
    @staticmethod
    def _rendered(row: dict) -> dict:
        return {"state": row["state"], "questions": row["questions"]}

    def launch(self, rows: list[dict], kind: str, idem: str) -> str:
        """Write the idem-named job directory and spawn run.sh detached.

        Returns idem as the job_id: the jobs row and the directory share it
        (GP2). Held-out rows are written for estimate/evaluation but never
        passed to the training job (only train/dev paths go in argv).
        """
        pin = self.pin_status()
        if pin["status"] != "ok":
            raise RuntimeError(f"{pin['status']}: {pin['detail']}")
        if not self.key:
            raise RuntimeError("no_fireworks_key: resolve fireworks.key_ref first")
        job_dir = self.jobs_dir / idem
        job_dir.mkdir(parents=True, exist_ok=True)
        by_split = {"train": [], "calib": [], "heldout": []}
        for r in rows:
            if r.get("split") in by_split:
                by_split[r["split"]].append(self._rendered(r))
        (job_dir / "train.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in by_split["train"]))
        (job_dir / "dev.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in by_split["calib"]))
        (job_dir / "heldout.jsonl").write_text(
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in by_split["heldout"]))
        for name in ("train.jsonl", "dev.jsonl", "heldout.jsonl"):
            (job_dir / name).chmod(0o600)  # X3: customer rows, tight perms
        argv = self.recipe_argv(kind, job_dir, len(by_split["train"]))
        latest = max(rows, key=row_key, default=None)
        marker = {"ts": latest.get("ts", "") if latest else "",
                  "decision_id": latest.get("decision_id", "") if latest else "",
                  "seq": latest.get("seq") if latest else None}
        (job_dir / "meta.json").write_text(json.dumps({
            "kind": kind, "idem": idem, "argv": argv,
            "export_marker": marker,
            "n_train": len(by_split["train"]),
            "n_dev": len(by_split["calib"]),
            "n_heldout": len(by_split["heldout"]),
            "estimate": self.estimate(rows, kind),
            "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }, indent=2, sort_keys=True))
        run_sh = job_dir / "run.sh"
        run_sh.write_text(RUN_SH)
        run_sh.chmod(0o700)
        log_path = job_dir / "train.log"
        log_path.touch(exist_ok=True)
        log_path.chmod(0o600)  # the redirect inherits these permissions
        env = {**os.environ, "FIREWORKS_API_KEY": self.key}
        proc = self._runner([str(run_sh), *argv], env, str(self._cookbook()))
        if isinstance(proc, subprocess.Popen):
            self._procs[idem] = proc
        if self._runner is _default_runner:
            self._await_start(job_dir)
        return idem

    @staticmethod
    def _await_start(job_dir: Path, timeout_s: float = 10.0) -> None:
        """Wait until run.sh announces itself (pid) or already finished."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if (job_dir / "pid").is_file() \
                    or (job_dir / "exit_code").is_file():
                return
            time.sleep(0.05)

    # -- status (GP2 lifecycle) --------------------------------------------
    def _job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except (OSError, ValueError, OverflowError):
            return False
        return True

    @staticmethod
    def _elapsed_s(pid: int) -> float | None:
        """Process age in seconds via ps (etimes on Linux, etime fallback)."""
        for flag in ("etimes=", "etime="):
            try:
                proc = subprocess.run(
                    ["ps", "-o", flag, "-p", str(pid)],
                    capture_output=True, text=True, timeout=5,
                    stdin=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                return None
            if proc.returncode != 0:
                continue
            out = (proc.stdout or "").strip()
            if flag == "etimes=":
                try:
                    return float(out)
                except ValueError:
                    continue  # macOS: no etimes keyword
            return _parse_etime(out)
        return None

    def _pid_reused(self, pid: int, job_dir: Path) -> bool:
        """True when pid is alive but started far from started_at (GP2)."""
        try:
            started = int((job_dir / "started_at").read_text().strip())
        except (OSError, ValueError):
            return False  # cannot verify: trust kill -0
        elapsed = self._elapsed_s(pid)
        if elapsed is None:
            return False
        return abs((time.time() - elapsed) - started) > REUSE_SKEW_S

    def _overdue(self, job_dir: Path) -> bool:
        try:
            started = int((job_dir / "started_at").read_text().strip())
        except (OSError, ValueError):
            return False
        max_s = float(self.fw.get("job_max_hours", 3) or 3) * 3600
        return time.time() - started > max_s

    def status(self, job_id: str) -> dict:
        """running | done | failed | cancelled from pid + exit_code (GP2).

        done needs exit 0 plus the log proof (Promoted! for full jobs, a
        sampler checkpoint line for smoke). A live pid whose start time
        misses started_at is reuse, not running. An overdue live job is
        killed here (SIGTERM, SIGKILL after the grace) and fails as
        job_timeout.
        """
        self._reap(job_id)
        job_dir = self._job_dir(job_id)
        kind = self._meta_kind(job_dir)
        if not job_dir.is_dir():
            return {"status": "failed", "kind": kind, "error": "unknown job"}
        exit_file = job_dir / "exit_code"
        if exit_file.is_file():
            try:
                code = int(exit_file.read_text().strip())
            except ValueError:
                code = 1
            if code == 0:
                ok, why = self._log_confirms(job_dir, kind)
                if ok:
                    return {"status": "done", "kind": kind,
                            **self._done_details(job_dir)}
                return {"status": "failed", "kind": kind, "error": why}
            if (job_dir / "timeout").is_file():
                return {"status": "failed", "kind": kind,
                        "error": "job_timeout"}
            if (job_dir / "cancelled").is_file():
                return {"status": "cancelled", "kind": kind}
            return {"status": "failed", "kind": kind,
                    "error": self._log_tail(job_dir),
                    "account_id": self._log_account_id(job_dir)}
        pid = self._read_pid(job_dir)
        if pid is None:
            return {"status": "failed", "kind": kind,
                    "error": "no pid and no exit_code"}
        if not self._pid_alive(pid) or self._is_zombie(pid):
            # A zombie leader is already dead (kill -0 still answers).
            return {"status": "failed", "kind": kind,
                    "error": "pid gone with no exit_code"}
        if self._pid_reused(pid, job_dir):
            return {"status": "failed", "kind": kind,
                    "error": f"pid_reuse: pid {pid} is a different process"}
        if self._overdue(job_dir):
            self._kill_group(pid)
            self._write_marker(job_dir, "timeout", EXIT_TIMEOUT,
                               "job_timeout: killed past job_max_hours")
            return {"status": "failed", "kind": kind, "error": "job_timeout"}
        return {"status": "running", "kind": kind}

    def _log_confirms(self, job_dir: Path, kind: str) -> tuple[bool, str]:
        try:
            parsed = parse_log((job_dir / "train.log").read_text())
        except OSError:
            parsed = {}
        if kind == "smoke":
            if (parsed.get("sampler_checkpoint") or {}).get("name"):
                return True, ""
            return False, "exited 0 without a sampler checkpoint line"
        if parsed.get("promoted"):
            return True, ""
        return False, "exited 0 without Promoted!"

    def _reap(self, job_id: str) -> None:
        proc = self._procs.get(job_id)
        if proc is not None and proc.poll() is not None:
            del self._procs[job_id]

    @staticmethod
    def _is_zombie(pid: int) -> bool:
        try:
            proc = subprocess.run(
                ["ps", "-o", "stat=", "-p", str(pid)],
                capture_output=True, text=True, timeout=5,
                stdin=subprocess.DEVNULL)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return proc.returncode == 0 and (proc.stdout or "").strip().startswith("Z")

    @staticmethod
    def _kill_group(pid: int, grace_s: float = KILL_GRACE_S) -> None:
        """SIGTERM the session group, SIGKILL what survives the grace.

        PermissionError means nothing signalable is left: our groups hold
        only our own children, so that is zombies (macOS answers EPERM
        for a zombie-only group, not ESRCH).
        """
        try:
            os.killpg(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return
        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            try:
                os.killpg(pid, 0)
            except (ProcessLookupError, PermissionError):
                return
            time.sleep(0.2)
        try:
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    @staticmethod
    def _write_marker(job_dir: Path, name: str, code: int, note: str) -> None:
        """Our own exit record: run.sh died with the group, so it never
        wrote one. The marker disambiguates our codes from the recipe's."""
        (job_dir / name).write_text(note + "\n")
        (job_dir / "exit_code").write_text(f"{code}\n")
        (job_dir / "finished_at").write_text(f"{int(time.time())}\n")
        with open(job_dir / "train.log", "a") as f:
            f.write(note + "\n")

    def cancel(self, job_id: str) -> dict:
        """Kill a live job's group; finished or unknown jobs are returned
        as-is (idempotent: nothing to kill is already cancelled)."""
        job_dir = self._job_dir(job_id)
        kind = self._meta_kind(job_dir)
        if not job_dir.is_dir():
            return {"status": "cancelled", "kind": kind}
        if (job_dir / "exit_code").is_file():
            return self.status(job_id)
        pid = self._read_pid(job_dir)
        if pid is not None and self._pid_alive(pid):
            self._kill_group(pid)
        self._write_marker(job_dir, "cancelled", EXIT_CANCELLED,
                           "cancelled by user")
        return {"status": "cancelled", "kind": kind}

    def _meta_kind(self, job_dir: Path) -> str:
        try:
            return json.loads((job_dir / "meta.json").read_text()).get(
                "kind", "")
        except (OSError, ValueError):
            return ""

    @staticmethod
    def _read_pid(job_dir: Path):
        try:
            return int((job_dir / "pid").read_text().strip())
        except (OSError, ValueError):
            return None

    @staticmethod
    def _log_tail(job_dir: Path, n: int = 20) -> str:
        try:
            lines = (job_dir / "train.log").read_text().splitlines()
        except OSError:
            return ""
        return "\n".join(lines[-n:])

    @staticmethod
    def _log_account_id(job_dir: Path) -> str | None:
        """The resolved id from the full log (X2: the 20-line error tail
        rarely contains the Auto-resolved line itself)."""
        try:
            text = (job_dir / "train.log").read_text()
        except OSError:
            return None
        return parse_account_id(text)

    def checkpoint(self, job_id: str) -> dict:
        raise KeyError(f"job {job_id} has no checkpoint (F8 wires evaluation)")

    def _done_details(self, job_dir: Path) -> dict:
        """F8: model + recipe T + log metrics for a done job (GP5/GP6).

        The smoke check uses the job's own eval lines and T from the
        log; the background eval takes the model and the recipe T.
        Never raises (missing/partial logs read as None).
        """
        try:
            parsed = parse_log((job_dir / "train.log").read_text())
        except OSError:
            parsed = {}
        try:
            meta = json.loads((job_dir / "meta.json").read_text())
        except (OSError, ValueError):
            meta = {}
        last = parsed.get("last_eval") or {}
        cal_t = parsed.get("calibration_T") or {}
        return {
            "metrics": {
                "n_train": parsed.get("n_train") or meta.get("n_train"),
                "n_eval": (meta.get("estimate") or {}).get("eval_rows"),
                "eval_loss": last.get("eval_loss"),
                "eval_accuracy": last.get("eval_accuracy"),
            },
            "model": parsed.get("model"),
            "recipe_temperature": cal_t.get("choice"),
            "account_id": parsed.get("account_id"),
        }

    def progress(self, job_id: str) -> dict | None:
        """F9: live progress from the log (step/loss/eval), or None."""
        try:
            parsed = parse_log(
                (self._job_dir(job_id) / "train.log").read_text())
        except OSError:
            return None
        last = parsed.get("last_eval") or {}
        out = {"step": parsed.get("last_step"),
               "loss": parsed.get("last_loss"),
               "eval_accuracy": last.get("eval_accuracy"),
               "eval_loss": last.get("eval_loss")}
        if all(v is None for v in out.values()):
            return None
        return out

    def eval_rows(self, job_id: str) -> dict:
        """F8: the job's scored splits, read back from the job files."""
        job_dir = self._job_dir(job_id)
        out = {}
        for split, name in (("calib", "dev"), ("heldout", "heldout")):
            rows = []
            try:
                text = (job_dir / f"{name}.jsonl").read_text()
            except OSError:
                text = ""
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            out[split] = rows
        return out
