"""Autopilot engine (M16).

Inside user-approved limits, the server promotes collect -> shadow -> serve
and launches training jobs by itself. Outside them (or while off) behaviour
is exactly v1.8. Plan serve (§15.6) and the downgrade trial (§15.8) are out
of scope and keep their confirmations.

All clock reads go through service._utcnow (attribute lookup per call) so
the tests' fake clock applies; the per-point once-a-minute limiter uses a
monotonic source (injectable on Service for tests).
"""
from __future__ import annotations

import fnmatch
import hashlib
import json
import shlex
import subprocess
import threading
import time
from datetime import datetime, timezone

from . import fireworks_export as FE
from . import hostedfit as HF
from . import labels as L
from . import stats as S
from .fireworks_trainer import FireworksTrainer
from .fireworks_worker import WorkerError, WorkerSession, WorkerUnavailable
from .store import row_key as _row_key
from .store import row_seq as _row_seq

APPROVED_HASH_KEY = "autopilot_approved_hash"
NOTIFIED_HASH_KEY = "autopilot_notified_hash"
APPROVED_SIG_KEY = "autopilot_approved_sig"
APPROVED_AT_KEY = "autopilot_approved_at"
APPROVED_NOTIFY_KEY = "autopilot_approved_notify"
TRUST_FP_KEY = "autopilot_trust_fingerprint"
TRUST_NOTIFIED_KEY = "autopilot_trust_notified"
SIGNING_KEY_NAME = "DECIDEONCE_AUTOPILOT_SIGNING_KEY"
SIGNING_KEY_NAME_LEGACY = "DECIDE_AUTOPILOT_SIGNING_KEY"  # pre-rename fallback
SMOKE_DONE_AT_KEY = "autopilot_smoke_done_at"

# G2: transient status() failures before a running job is declared lost.
STATUS_LOST_AFTER = 3
STALE_LAUNCH_S = 10 * 60  # Y9: GP2 says 10 min, not 15
# G5: backoff after a failed autopilot job before the next launch.
FAIL_BACKOFF_S = 24 * 3600


class JobRunning(Exception):
    """Another job is launching/running (G3 race lost)."""


class JobCancelled(Exception):
    """Y10: our launch lost its claim to a concurrent cancel.

    Carries the temp row id; the row is already cancelled and the
    just-spawned child was killed, so the caller reports cancelled.
    """


#: Worker calls per evaluation batch (GP6).
EVAL_BATCH = 32
#: Warming poll step while an eval session starts (cap: eval_ready_s).
EVAL_READY_POLL_S = 0.2


class _EvalAbort(Exception):
    """Evaluation failed: the job becomes eval_failed with this error."""


class _EvalCancel(Exception):
    """Cancel won mid-evaluation: the thread exits quietly (the row is
    already terminal)."""


def _fmt_num(v, digits: int = 4) -> str:
    if v is None:
        return "?"
    try:
        return f"{float(v):.{digits}f}"
    except (TypeError, ValueError, OverflowError):
        return "?"

# Main spec §7.3: the free-training confirmation only counts before
# 2026-12-01 00:00 Pacific.
MAIN_FREE_CUTOFF_UTC = datetime(2026, 12, 1, 8, 0, 0, tzinfo=timezone.utc)

# _serve_checks names in §6.4 item order (confirm/item 6 replaced by the
# approved limits; private_item8/item 8 never automatic). labelled_ratio is
# the hosted-tier §7.7 extra: autopilot requires it, reported by name.
GATE_ORDER = [
    ("min_examples", "1"),
    ("threshold", "2"),
    ("coverage", "2"),
    ("evidence", "3"),
    ("margin", "3"),
    ("shadow", "4"),
    ("shadow_agreement", "4"),
    ("teacher", "5"),
    ("gateway_format", "7"),
    ("labelled_ratio", "labelled_ratio"),
]


def section_hash(section: dict) -> str:
    """SHA-256 over the canonical JSON of a section (or effective limits)."""
    canon = json.dumps(section, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def effective_limits(cfg: dict) -> dict:
    """E5: the limits autopilot's gates and estimates read.

    [autopilot] plus every key the promotion gates, the training gates and
    the cost estimates read: full [serve] and [train], the hosted budget /
    fit keys, the gateway format gates, and the Fireworks prices, budgets
    and estimate params. Connection keys (urls, models, key_refs) are not
    limits and are excluded.
    """
    ap = cfg.get("autopilot", {}) or {}
    serve = cfg.get("serve", {}) or {}
    train = cfg.get("train", {}) or {}
    hj = cfg.get("hosted_jev", {}) or {}
    gw = cfg.get("gateway", {}) or {}
    fw = cfg.get("fireworks", {}) or {}
    return {
        "autopilot": {k: ap.get(k) for k in sorted(ap)},
        "serve": {k: serve.get(k) for k in sorted(serve)},
        "train": {k: train.get(k) for k in sorted(train)},
        "hosted_jev": {k: hj.get(k) for k in (
            "price_per_mtok_in", "monthly_budget_usd", "fit_rows",
            "min_private_gain", "min_compare", "redemote_window_days",
            "reserve_usd") if k in hj},
        "gateway": {k: gw.get(k) for k in (
            "format_min_n", "format_match_min", "max_unparsed") if k in gw},
        "fireworks": {k: fw.get(k) for k in (
            "monthly_budget_usd", "allow_paid_training", "smoke_examples",
            "epochs", "train_price_per_mtok",
            "eval_price_per_decision", "cookbook_dir", "cookbook_commit",
            "batch_rows", "lora_rank", "learning_rate", "job_max_hours",
            "worker_idle_min", "score_price_per_mtok", "reserve_usd",
            "allow_session_serving") if k in fw},
    }


def effective_hash(cfg: dict) -> str:
    """SHA-256 over the effective limits (E5: the approved hash)."""
    return section_hash(effective_limits(cfg))


def format_section(section: dict) -> str:
    lines = ["[autopilot]"]
    for k in sorted(section):
        lines.append(f"{k} = {json.dumps(section[k])}")
    return "\n".join(lines)


def format_effective_limits(cfg: dict) -> str:
    """Printable effective limits for approval (E5)."""
    limits = effective_limits(cfg)
    lines = []
    for section in ("autopilot", "serve", "train", "hosted_jev",
                    "gateway", "fireworks"):
        lines.append(f"[{section}]")
        for k in sorted(limits.get(section, {})):
            lines.append(f"{k} = {json.dumps(limits[section][k])}")
    return "\n".join(lines)


_NOTIFY_THREADS: list = []
_NOTIFY_LOCK = None


def _notify_lock():
    global _NOTIFY_LOCK
    if _NOTIFY_LOCK is None:
        import threading as _th

        _NOTIFY_LOCK = _th.Lock()
    return _NOTIFY_LOCK


def notify(cmd, line: str) -> None:
    """Run notify_cmd (no shell) with the one-line message appended.

    G8: `--` separates the message from flags, and the run is asynchronous
    (daemon thread) so a slow notifier never blocks do_record/do_outcome.
    Never raises. Tests join via sync_notifies().
    """
    try:
        if not cmd:
            return
        if isinstance(cmd, str):
            argv = shlex.split(cmd)
        else:
            argv = [str(a) for a in cmd]
        if not argv:
            return
        msg = " ".join(str(line).split())
        import threading as _th

        def _run():
            try:
                subprocess.run(argv + ["--", msg], stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
            except Exception:
                pass

        t = _th.Thread(target=_run, daemon=True)
        with _notify_lock():
            # Z7: prune finished threads (production never calls
            # sync_notifies, so the list would grow without bound).
            _NOTIFY_THREADS[:] = [u for u in _NOTIFY_THREADS if u.is_alive()]
            _NOTIFY_THREADS.append(t)
        t.start()
    except Exception:
        pass


def sync_notifies(timeout: float = 5.0) -> None:
    """Join pending notify threads (tests only)."""
    with _notify_lock():
        pending = list(_NOTIFY_THREADS)
        _NOTIFY_THREADS.clear()
    for t in pending:
        try:
            t.join(timeout=timeout)
        except Exception:
            pass


def fingerprint_pubkey(b64_pub: str) -> str | None:
    """SHA-256 hex (first 16) of the raw Ed25519 public key, or None."""
    import base64

    try:
        raw = base64.b64decode(b64_pub.strip())
    except Exception:
        return None
    if len(raw) != 32:
        return None
    return hashlib.sha256(raw).hexdigest()[:16]


def _sign_message(project: str, eff_hash: str, approved_at: str) -> bytes:
    return f"{project}\n{eff_hash}\n{approved_at}".encode()


def sign_approval(raw_priv_32: bytes, project: str, eff_hash: str,
                  approved_at: str) -> str:
    """Base64 Ed25519 signature over project||hash||time."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey)

    priv = Ed25519PrivateKey.from_private_bytes(raw_priv_32)
    sig = priv.sign(_sign_message(project, eff_hash, approved_at))
    import base64 as _b64

    return _b64.b64encode(sig).decode()


def verify_approval(b64_pub: str, b64_sig: str, project: str, eff_hash: str,
                    approved_at: str) -> bool:
    """True when the signature matches the public key and message."""
    import base64

    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PublicKey)

    try:
        pub = Ed25519PublicKey.from_public_bytes(
            base64.b64decode(b64_pub.strip()))
        pub.verify(base64.b64decode(b64_sig.strip()),
                   _sign_message(project, eff_hash, approved_at))
        return True
    except Exception:
        return False


def _parse_envsave_show(text: str, key_name: str) -> str | None:
    """Extract the secret from `envsave show` output, or None.

    Real format: `  DECIDEONCE_AUTOPILOT_SIGNING_KEY = <secret>`.
    Anything else fails loudly instead of decoding a prompt line.
    """
    import re as _re

    pat = _re.compile(r"^\s*" + _re.escape(key_name) + r"\s*=\s*(\S+)\s*$")
    for ln in text.splitlines():
        m = pat.match(ln)
        if m:
            return m.group(1)
    return None


def _read_envsave_secret(key_name: str, out, env_runner=None) -> str | None:
    """Read a secret via `envsave show`. Returns the secret or None.

    The password prompt is expected on stderr (Unix convention: data on
    stdout, prompts on stderr), so stderr and stdin stay attached to the
    terminal while stdout is captured. env_runner (tests) replaces the
    subprocess call.
    """
    import sys

    if env_runner is not None:
        try:
            res = env_runner(["envsave", "show", key_name])
        except Exception as e:
            out(f"error: envsave failed: {e}")
            return None
        if isinstance(res, tuple):
            code, stdout = res[0], res[1]
        else:
            code, stdout = 0, res
        if code != 0:
            out("error: envsave show failed (is the vault unlocked?)")
            return None
        text = stdout if isinstance(stdout, str) else stdout.decode(errors="replace")
    else:
        out("envsave will ask for your master password if the vault is locked...")
        try:
            proc = subprocess.run(
                ["envsave", "show", key_name], stdin=sys.stdin,
                stdout=subprocess.PIPE, stderr=sys.stderr, timeout=120)
        except Exception as e:
            out(f"error: envsave failed: {e}")
            return None
        if proc.returncode != 0:
            out("error: envsave show failed (is the vault unlocked?)")
            return None
        try:
            text = proc.stdout.decode(errors="replace")
        except Exception:
            text = ""
    secret = _parse_envsave_show(text, key_name)
    if not secret:
        out(f"error: envsave show did not print '{key_name} = <secret>'")
        return None
    return secret


def _read_signing_secret(out, env_runner=None) -> str | None:
    """Signing key from envsave: the new name first, then the pre-rename
    fallback. Per-name failures stay silent; only a total miss reports."""
    def _quiet(msg: str) -> None:
        pass
    secret = _read_envsave_secret(SIGNING_KEY_NAME, _quiet, env_runner)
    if secret is None:
        secret = _read_envsave_secret(SIGNING_KEY_NAME_LEGACY, _quiet,
                                      env_runner)
    if secret is None:
        out(f"error: envsave has neither '{SIGNING_KEY_NAME}' nor "
            f"'{SIGNING_KEY_NAME_LEGACY}'; run `decideonce autopilot keygen`")
    return secret


def run_keygen(cfg_path, store, out, env_runner=None) -> str:
    """Generate an Ed25519 signing key, store it in envsave, write the public
    key to the config. Returns ok|error. env_runner replaces subprocess in tests."""
    import base64
    import sys
    from pathlib import Path as _Path

    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import (
        Ed25519PrivateKey)

    priv = Ed25519PrivateKey.generate()
    raw_priv = priv.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
        serialization.NoEncryption())
    raw_pub = priv.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    b64_priv = base64.b64encode(raw_priv).decode()
    b64_pub = base64.b64encode(raw_pub).decode()
    fp = fingerprint_pubkey(b64_pub) or "?"
    # Store the private key in envsave (value on stdin, never argv). The
    # password prompt (if any) is expected on /dev/tty or stderr, not stdin.
    # R3-1: exit 0 alone proves nothing (without Touch ID nothing is stored),
    # so stdout must carry Added:/Updated: before the public key is written.
    set_stdout = ""
    if env_runner is not None:
        try:
            res = env_runner(["envsave", "set", SIGNING_KEY_NAME], b64_priv)
        except Exception as e:
            out(f"error: envsave set failed: {e}")
            return "error"
        if isinstance(res, tuple):
            code, set_stdout = res[0], (res[1] if len(res) > 1 else "")
        else:
            code = res
        if code != 0:
            out("error: envsave set failed")
            return "error"
    else:
        out("storing the private key in envsave "
            f"({SIGNING_KEY_NAME}); approve the Touch ID / password prompt...")
        try:
            proc = subprocess.run(
                ["envsave", "set", SIGNING_KEY_NAME],
                input=(b64_priv + "\n").encode(),
                stdout=subprocess.PIPE, stderr=sys.stderr, timeout=120)
        except Exception as e:
            out(f"error: envsave set failed: {e}")
            return "error"
        if proc.returncode != 0:
            out("error: envsave set failed; the key was not stored. "
                "Run it yourself: envsave set " + SIGNING_KEY_NAME)
            return "error"
        try:
            set_stdout = proc.stdout.decode(errors="replace")
        except Exception:
            set_stdout = ""
    for ln in set_stdout.splitlines():
        if ln.strip():
            out(ln.strip())
    if "Added:" not in set_stdout and \
            f"Updated: {SIGNING_KEY_NAME}" not in set_stdout:
        out("error: envsave set stored nothing (no Added:/Updated: line "
            "on stdout; the vault may need Touch ID). Run "
            f"`envsave set {SIGNING_KEY_NAME}` yourself or enable Touch ID, "
            "then re-run keygen.")
        return "error"
    # Wipe the private key from memory (best effort in Python).
    try:
        b64_priv = "0" * len(b64_priv)
    except Exception:
        pass
    del raw_priv
    # Write the public key to the config file.
    try:
        import tomllib

        path = _Path(str(cfg_path))
        data = {}
        if path.is_file():
            with open(path, "rb") as f:
                data = tomllib.load(f) or {}
        if not isinstance(data, dict):
            data = {}
        trust = dict(data.get("autopilot_trust") or {})
        trust["public_key"] = b64_pub
        # Preserve the file's other text: rewrite via a minimal TOML merge.
        lines = path.read_text().splitlines() if path.is_file() else []
        out_lines: list[str] = []
        in_trust = False
        trust_written = False
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("["):
                if in_trust and not trust_written:
                    out_lines.append(f'public_key = "{b64_pub}"')
                    trust_written = True
                in_trust = stripped == "[autopilot_trust]"
                out_lines.append(ln)
                continue
            if in_trust and stripped.startswith("public_key"):
                out_lines.append(f'public_key = "{b64_pub}"')
                trust_written = True
                continue
            out_lines.append(ln)
        if in_trust and not trust_written:
            out_lines.append(f'public_key = "{b64_pub}"')
            trust_written = True
        if not trust_written:
            if out_lines and out_lines[-1].strip():
                out_lines.append("")
            out_lines.append("[autopilot_trust]")
            out_lines.append(f'public_key = "{b64_pub}"')
        path.write_text("\n".join(out_lines) + "\n")
    except Exception as e:
        out(f"error: could not write the public key to {cfg_path}: {e}")
        return "error"
    out(f"public key fingerprint: {fp}")
    out(f"public key written to {cfg_path} [autopilot_trust]")
    return "ok"


def run_approve(cfg: dict, store, *, is_tty: bool, out,
                env_runner=None, read_line=None) -> str:
    """Terminal approval via Ed25519 (user decision 2026-11-03).

    Prints the effective limits and SHA-256, reads the signing key via
    `envsave show` (user types the master password), signs
    project||hash||time and stores signature+hash+time. Never prints the
    hash on the non-TTY path. Returns ok|not_interactive|error.
    read_line is accepted for backward compat and ignored.
    """
    if not is_tty:
        out("error: not_interactive: approval needs an interactive terminal "
            "(the user types the envsave master password)")
        return "not_interactive"
    h = effective_hash(cfg)
    out(format_effective_limits(cfg))
    out(f"SHA-256: {h}")
    b64_pub = (cfg.get("autopilot_trust", {}) or {}).get("public_key", "") or ""
    fp = fingerprint_pubkey(b64_pub)
    if not fp:
        out("error: no [autopilot_trust] public_key in config; run "
            "`decideonce autopilot keygen` first")
        return "error"
    out(f"trust key fingerprint: {fp}")
    project = (cfg.get("store", {}) or {}).get("project", "default")
    secret = _read_signing_secret(out, env_runner)
    if not secret:
        return "error"
    import base64

    try:
        raw_priv = base64.b64decode(secret.strip(), validate=True)
    except Exception:
        out("error: the stored signing key is not valid base64")
        return "error"
    finally:
        try:
            secret = "0" * len(secret)
        except Exception:
            pass
    if len(raw_priv) != 32:
        out("error: the stored signing key is not 32 bytes")
        return "error"
    approved_at = _now_iso()
    try:
        sig = sign_approval(raw_priv, project, h, approved_at)
    except Exception as e:
        out(f"error: signing failed: {e}")
        return "error"
    finally:
        try:
            raw_priv = b"\x00" * len(raw_priv)
        except Exception:
            pass
        del raw_priv
    # Sanity: the signature must verify with the configured public key.
    if not verify_approval(b64_pub, sig, project, h, approved_at):
        out("error: the signature does not verify with [autopilot_trust] "
            "public_key (wrong key in envsave?)")
        return "error"
    store.meta_put(APPROVED_HASH_KEY, h)
    store.meta_put(APPROVED_SIG_KEY, sig)
    store.meta_put(APPROVED_AT_KEY, approved_at)
    store.meta_put(APPROVED_NOTIFY_KEY,
                   json.dumps((cfg.get("autopilot", {}) or {}).get("notify_cmd", "")))
    store.meta_put(TRUST_FP_KEY, fp)
    out(f"approved {h[:12]} (signed {approved_at}): autopilot limits active")
    return "ok"


def _now_iso() -> str:
    from . import service as S

    return S._utcnow()


def _parse_ts(s) -> float | None:
    """Parse second- or microsecond-resolution UTC stamps (H2 mixes both)."""
    if not isinstance(s, str):
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _parse_when(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _point_target(point: dict, cfg: dict) -> float:
    return float(point.get("target_error") or cfg["serve"]["target_error"])


def _in_scope(point_id: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(point_id, p) for p in (patterns or []))


def _fireworks_allowed(point: dict) -> bool:
    if point.get("data_class", "private") == "public":
        return True
    ro = point.get("remote_ok", [])
    return ro is True or "fireworks" in (ro or [])


class Autopilot:
    """Eligibility, checks and training automation. Pure status stays pure:
    point_status() and status_report() never write, upload or launch."""

    def __init__(self, svc):
        self.svc = svc
        self.checks_run = 0
        # F8 background evaluation: live threads and their cancel flags
        # by job_id, plus a worker-argv hook (tests inject FakeWorker).
        self._eval_threads: dict[str, threading.Thread] = {}
        self._eval_cancel: dict[str, threading.Event] = {}
        self.worker_argv_fn = None

    # -- global state --------------------------------------------------
    @property
    def cfg(self) -> dict:
        return self.svc.cfg

    @property
    def store(self):
        return self.svc.store

    @property
    def section(self) -> dict:
        return self.cfg.get("autopilot", {}) or {}

    def hashes(self) -> dict:
        current = effective_hash(self.cfg)
        approved = self.store.meta_get(APPROVED_HASH_KEY)
        return {"approved": approved, "current": current,
                "match": self._signature_valid(current)}

    def _signature_valid(self, current: str | None = None) -> bool:
        """True when the stored signature verifies the current limits."""
        if current is None:
            current = effective_hash(self.cfg)
        approved = self.store.meta_get(APPROVED_HASH_KEY)
        sig = self.store.meta_get(APPROVED_SIG_KEY)
        approved_at = self.store.meta_get(APPROVED_AT_KEY)
        if not approved or not sig or not approved_at:
            return False
        if approved != current:
            return False
        b64_pub = (self.cfg.get("autopilot_trust", {}) or {}).get(
            "public_key", "") or ""
        if not b64_pub:
            return False
        project = (self.cfg.get("store", {}) or {}).get("project", "default")
        return verify_approval(b64_pub, sig, project, approved, approved_at)

    def _trust_key_changed(self) -> bool:
        """True when the configured public key differs from the approved one."""
        b64_pub = (self.cfg.get("autopilot_trust", {}) or {}).get(
            "public_key", "") or ""
        fp_now = fingerprint_pubkey(b64_pub)
        fp_approved = self.store.meta_get(TRUST_FP_KEY)
        return bool(fp_approved) and fp_now != fp_approved

    def global_state(self) -> str:
        """ok | disabled | not_approved | trust_key_changed."""
        if not self.section.get("enabled", False):
            return "disabled"
        if self._trust_key_changed():
            return "trust_key_changed"
        return "ok" if self._signature_valid() else "not_approved"

    def _approved_notify_cmd(self):
        """F1: the notify_cmd from the last valid approval (never the edited one)."""
        raw = self.store.meta_get(APPROVED_NOTIFY_KEY)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def _notify_cmd_for(self, fallback_to_current: bool = False):
        """Notify command to run: approved only (F1), optionally falling back
        to the current section when no approval exists yet (tests that approve
        by hash directly)."""
        cmd = self._approved_notify_cmd()
        if cmd is not None:
            return cmd
        if fallback_to_current:
            return self.section.get("notify_cmd")
        return ""

    def _reload_section(self) -> None:
        """Re-read the effective limits and trust key from the config file.

        Services built from a dict (tests) have no path and keep in-memory
        config. A missing or unparseable file falls back to defaults, which
        stops autopilot (fail closed) until the file is fixed and re-approved.
        """
        path = getattr(self.svc, "_config_path", None)
        if not path:
            return
        from .config import DEFAULTS

        data: dict = {}
        try:
            import tomllib
            from pathlib import Path as _Path

            with open(_Path(path), "rb") as f:
                data = tomllib.load(f)
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}
        for _sec in ("autopilot", "autopilot_trust", "serve", "train",
                     "hosted_jev", "gateway", "fireworks"):
            section = data.get(_sec) if isinstance(data.get(_sec), dict) else {}
            merged = dict(DEFAULTS.get(_sec, {}))
            merged.update(section)
            self.svc.cfg[_sec] = merged

    # -- per-point status (pure) ----------------------------------------
    def _teacher_unreliable(self, point: dict) -> bool:
        rows = self.store.decisions_for_point(point["point_id"])
        both, disagree = L.teacher_disagreement(rows)
        return both > 0 and disagree / both > self.cfg["train"]["teacher_alarm"]

    def _cooldown_left_s(self, point: dict) -> float:
        ts = _parse_ts(point.get("autopilot_demotion_ts"))
        if ts is None:
            return 0.0
        now = _parse_ts(_now_iso())
        if now is None:
            return 0.0
        try:
            days = float(self.section.get("cooldown_days", 7))
        except (TypeError, ValueError):
            days = 7.0
        return ts + max(days, 0.0) * 86400 - now

    def _demotion_since(self, point: dict):
        ts = point.get("autopilot_demotion_ts")
        return (ts, "") if ts else None

    def _hosted_exhausted(self) -> bool:
        hj = self.cfg.get("hosted_jev", {})
        try:
            budget = float(hj.get("monthly_budget_usd") or 0)
        except (TypeError, ValueError):
            return True
        if budget <= 0:
            return False  # not configured: no_decider, not budget
        spent = self.store.spend_get(_now_iso()[:7], "hosted_jev")["usd"]
        return spent >= budget

    def _fireworks_headroom(self) -> float | None:
        """Remaining Fireworks budget, or None when no spending approval."""
        fw = self.cfg.get("fireworks", {})
        try:
            budget = float(fw.get("monthly_budget_usd") or 0)
        except (TypeError, ValueError):
            return None
        if budget <= 0:
            return None
        spent = self.store.spend_get(_now_iso()[:7], "fireworks")["usd"]
        return budget - spent

    def _shadow_gate(self, point: dict) -> str | None:
        """None when a decider exists for collect -> shadow, else a reason."""
        svc = self.svc
        if svc._hosted_tier(point):
            if svc._hosted_shadow_gate(point, "") is not None:
                return "no_decider"
            if self._hosted_exhausted():
                return "budget"
            return None
        # Private tier (private_model_only): a trained private model.
        model = self.store.active_model(point["point_id"], "private")
        if model is None or model.get("point_version") != point.get(
                "point_version", 1):
            return "no_decider"
        if getattr(svc.trainer, "needs_key", True) and not svc.fireworks_key:
            return "no_decider"
        if not self.cfg.get("fireworks", {}).get("allow_session_serving",
                                                 False):
            return "serving_not_allowed"  # GP11: report + no auto-shadow
        if self._fireworks_headroom() is not None \
                and self._fireworks_headroom() <= 0:
            return "budget"
        return None

    def _serve_gate(self, point: dict) -> str | None:
        """None when §6.4 items 1-5+7 (+labelled_ratio) hold, else gate:<item>."""
        checks = self.svc._serve_checks(point, True, None,
                                        since=self._demotion_since(point))
        for name, item in GATE_ORDER:
            c = checks.get(name)
            if c is None:
                continue  # gateway_format on non-gateway points
            if not c["pass"]:
                return f"gate:{item}"
        return None

    def _training_trigger(self, point: dict) -> bool:
        if point.get("private_model_only"):
            return len(self._labeled(point)) >= self.cfg["train"]["min_train"]
        return bool(self.svc._train_reasons(point))

    def point_status(self, point: dict) -> tuple[str, str]:
        """(status, waiting_for). Status is on|off|blocked(<reason>). Pure."""
        ap = self.section
        g = self.global_state()
        if g == "disabled":
            return "blocked(disabled)", "disabled"
        if g == "trust_key_changed":
            return "blocked(trust_key_changed)", "trust_key_changed"
        if g != "ok":
            return "blocked(not_approved)", "not_approved"
        if not _in_scope(point["point_id"], ap.get("points", [])):
            return "blocked(not_in_points)", "not_in_points"
        if _point_target(point, self.cfg) > float(ap.get("max_target_error", 0.02)):
            return "blocked(target_error_above_limit)", "target_error_above_limit"
        if point.get("suspected", False):
            return "blocked(suspected)", "suspected"
        if self._teacher_unreliable(point):
            return "blocked(teacher_unreliable)", "teacher_unreliable"
        if point.get("autopilot_stopped", False) or point.get("phase") == "paused":
            return "blocked(stopped_by_call)", "stopped_by_call"

        phase = point.get("phase", "collect")
        promo: tuple | None = None  # ("ready", action) | ("reason", r)
        if phase == "collect" and ap.get("auto_serve", True):
            r = self._shadow_gate(point)
            promo = ("ready", "collect_to_shadow") if r is None else ("reason", r)
        elif phase == "shadow" and ap.get("auto_serve", True):
            if self._cooldown_left_s(point) > 0:
                promo = ("reason", "cooldown")
            elif point.get("irreversible", False):
                promo = ("reason", "irreversible")
            elif self._hosted_exhausted() \
                    and self.svc._hosted_tier(point):
                promo = ("reason", "budget")
            else:
                r = self._serve_gate(point)
                promo = ("ready", "shadow_to_serve") if r is None else ("reason", r)

        train: tuple | None = None
        if ap.get("auto_train", True) and phase in ("collect", "shadow", "serve") \
                and self._training_trigger(point):
            r = self._train_block_reason(point)
            train = ("ready", "train") if r is None else ("reason", r)

        if promo is not None and promo[0] == "ready":
            return "on", promo[1]
        if train is not None and train[0] == "ready":
            return "on", train[1]
        if promo is not None and promo[0] == "reason":
            # Training unblocks promotion only when the missing decider is a
            # model training would produce.
            if phase == "collect" and promo[1] == "no_decider" \
                    and self._model_unblocks(point) \
                    and train is not None and train[0] == "reason":
                return f"blocked({train[1]})", train[1]
            return f"blocked({promo[1]})", promo[1]
        if train is not None and train[0] == "reason":
            return f"blocked({train[1]})", train[1]
        return "off", "idle"

    def _model_unblocks(self, point: dict) -> bool:
        """True when training could produce the missing collect decider."""
        svc = self.svc
        return point.get("private_model_only", False) \
            or (svc._effective_tier(point) == "private")

    # -- training dry-run / gates (pure: estimate computes, never uploads) --
    def _labeled(self, point: dict) -> list:
        from . import train as TR

        labeled = L.labeled_rows(
            self.store.decisions_for_point(point["point_id"]), point,
            self.cfg["train"]["teacher_only_weight"],
            self.cfg["extract"]["weight"])
        # Eval-only transcripts (pilot arm on held-out tickets) never train:
        # their rows would leak the eval set.
        eval_ids = self.store.eval_only_transcript_ids(point["workflow"])
        if eval_ids:
            labeled = [(r, lab, src, w) for r, lab, src, w in labeled
                       if (r.get("transcript_id") or "") not in eval_ids]
        # §15.3 gate: extracted rows join the export only while the point
        # passes it.
        if any(src == "extracted" for _, _, src, _ in labeled) \
                and not TR.extraction_gate(self.cfg, self.store, point)["passed"]:
            labeled = [(r, lab, src, w) for r, lab, src, w in labeled
                       if src != "extracted"]
        return labeled

    def join_candidates(self) -> list[dict]:
        """Points with >= min_train labelled rows (§7.2 step 1), with flags."""
        out = []
        for p in self.store.list_points():
            n = len(self._labeled(p))
            if n < self.cfg["train"]["min_train"]:
                continue
            out.append({"point_id": p["point_id"], "n_rows": n,
                        "allowed": _fireworks_allowed(p),
                        "eligible": self._join_eligible(p)})
        return out

    def _join_eligible(self, point: dict) -> bool:
        ap = self.section
        return bool(
            _in_scope(point["point_id"], ap.get("points", []))
            and _point_target(point, self.cfg) <= float(ap.get("max_target_error", 0.02))
            and not point.get("suspected", False)
            and not self._teacher_unreliable(point)
            and not point.get("autopilot_stopped", False)
            and point.get("phase") != "paused")

    def dry_run(self, point: dict, force: bool = False) -> dict:
        """Kind, join list and estimate for the point's next job (no upload).

        Y11: the trigger always rides (it passed its own gate); any other
        join needs train_recommended, private_model_only, or a manual
        force — a point whose hosted Jev is good enough must not silently
        join a paid job. Excluded joins stay listed with join_ok False.
        """
        from . import trainer as TR

        joins = self.join_candidates()
        by_id: dict[str, dict] = {}
        for j in joins:
            p = self.store.get_point(j["point_id"])
            if p is not None:
                by_id[p["point_id"]] = p
            j["join_ok"] = force or j["point_id"] == point["point_id"] or (
                p is not None and (p.get("private_model_only")
                                   or bool(self.svc._train_reasons(p))))
        rows: list[dict] = []
        pts: dict[str, dict] = {}
        for j in joins:
            if not j["allowed"]:
                continue  # §7.6: never exported, trained or sent
            if not j["join_ok"]:
                continue  # Y11: hosted Jev serves it; leave it out
            p = by_id.get(j["point_id"])
            if p is None:
                continue
            pts[p["point_id"]] = p
            for r, label, src, weight in self._labeled(p):
                ctx = r.get("context")
                rows.append({"point_id": p["point_id"], "ts": r.get("ts") or "",
                             "decision_id": r.get("decision_id") or "",
                             "seq": r.get("seq"),
                             "run_id": r.get("run_id"),
                             "context": ctx if isinstance(ctx, dict) else {},
                             "label": label, "weight": weight,
                             "source": src})
        rows.sort(key=_row_key)
        rows = TR.split_rows(rows, self.cfg["train"]["holdout"],
                             self.cfg["train"].get("calib", 0.15))
        # F5: estimate and upload see the rendered export rows (§0).
        rows = [FE.render_export_row(pts[r["point_id"]], r) for r in rows]
        kind = "full" if self.store.smoke_done_exists() else "smoke"
        est = self.svc.trainer.estimate(rows, kind) if self.svc.trainer else {
            "estimate_usd": 0.0, "train_usd": 0.0, "eval_usd": 0.0,
            "dev_eval_usd": 0.0,
            "train_rows": 0, "eval_rows": 0, "kind": kind}
        return {"kind": kind, "joins": joins, "rows": rows, "estimate": est,
                "trigger": point["point_id"]}

    def _fail_backoff_left_s(self) -> float:
        """G5: seconds until the 24 h post-failure backoff ends (0 when none)."""
        failed_at = self.store.latest_autopilot_failure_at()
        if not failed_at:
            return 0.0
        fts = _parse_ts(failed_at)
        now = _parse_ts(_now_iso())
        if fts is None or now is None:
            return 0.0
        return max(0.0, fts + FAIL_BACKOFF_S - now)

    def _train_block_reason(self, point: dict) -> str | None:
        """None when autopilot may launch now, else a reason. Pure."""
        if self.svc.trainer is None:
            return "no_decider"  # no usable trainer until M11
        if getattr(self.svc.trainer, "needs_key", True) \
                and not self.svc.fireworks_key:
            return "no_decider"
        if self.store.jobs_running() or self.store.jobs_launching():
            return "job_running"
        if self._fail_backoff_left_s() > 0:
            return "backoff"
        if not self._new_rows_ok(point):
            return "not_enough_new_rows"
        dry = self.dry_run(point)
        for j in dry["joins"]:
            if not j["eligible"] or not j["allowed"]:
                return "privacy"
        if self._counted_usd(dry) is None:
            return "budget"
        # Y14: §2 step 2 on the paid path too — a full job needs a
        # post-smoke re-approval, not just an open budget. (The free
        # path already requires it through free_path_ok's G6.)
        if dry["kind"] == "full" and not self._smoke_reapproved():
            return "reapproval"
        return None

    def _new_rows_ok(self, point: dict) -> bool:
        """min_new_rows labelled rows since the point's last train export."""
        if not point.get("last_train_export_ts"):
            return True  # first job: none needed
        try:
            need = int(self.section.get("min_new_rows", 100))
        except (TypeError, ValueError):
            need = 100
        mseq = point.get("last_train_export_seq")
        legacy = None
        if not isinstance(mseq, int):
            row = self.store.get_decision(
                point.get("last_train_export_id") or "")
            if row is not None and isinstance(row.get("seq"), int):
                mseq = row["seq"]
            else:
                legacy = (point.get("last_train_export_ts"),
                          point.get("last_train_export_id"))
        n = 0
        for r, _lab, _src, _w in self._labeled(point):
            if legacy is not None:
                after = (r.get("ts") or "", r.get("decision_id") or "") > legacy
            else:
                after = _row_seq(r) > mseq
            n += after
        return n >= need

    def free_path_ok(self) -> bool:
        """Free full job: free_training_until set and future, before the
        main-spec cut-off, and (G6) approved after the smoke job finished
        (the approval confirms the user checked the smoke was not billed)."""
        raw = (self.section.get("free_training_until") or "").strip()
        if not raw:
            return False
        until = _parse_when(raw)
        if until is None:
            return False
        now_ts = _parse_ts(_now_iso())
        if now_ts is None:
            return False
        now = datetime.fromtimestamp(now_ts, tz=timezone.utc)
        if not (now < until and now < MAIN_FREE_CUTOFF_UTC):
            return False
        return self._smoke_reapproved()

    def _smoke_reapproved(self) -> bool:
        """G6 (§2 step 2): an approval newer than the last smoke job.

        Shared by the free path (via free_path_ok) and Y14's paid gate:
        the user saw the smoke outcome and re-approved before any full
        job may launch automatically.
        """
        smoke_at = self.store.meta_get(SMOKE_DONE_AT_KEY)
        approved_at = self.store.meta_get(APPROVED_AT_KEY)
        return bool(smoke_at and approved_at and approved_at > smoke_at)

    def manual_free_ok(self, confirmed: bool) -> bool:
        """Manual full job on the free path (§7.3): the call confirms the
        smoke job was not billed, and the launch is before the cut-off."""
        if not confirmed:
            return False
        now_ts = _parse_ts(_now_iso())
        if now_ts is None:
            return False
        return datetime.fromtimestamp(
            now_ts, tz=timezone.utc) < MAIN_FREE_CUTOFF_UTC

    def _limits_ok(self, counted: float) -> bool:
        ap = self.section
        try:
            if float(ap.get("max_job_usd", 0)) <= 0:
                return False  # 0 = autopilot never trains
            if float(ap.get("max_monthly_jobs_usd", 0)) <= 0:
                return False
        except (TypeError, ValueError):
            return False
        if counted > float(ap.get("max_job_usd", 0)):
            return False
        month = _now_iso()[:7]
        if counted + self.store.autopilot_month_counted(month) > float(
                ap.get("max_monthly_jobs_usd", 0)):
            return False
        headroom = self._fireworks_headroom()
        if headroom is None or counted > headroom:
            return False
        return True

    def _counted_usd(self, dry: dict) -> float | None:
        """The estimate the three limits are checked against, or None."""
        est = dry["estimate"]
        if dry["kind"] == "smoke":
            e = float(est["estimate_usd"])
            return e if self._limits_ok(e) else None
        if self.free_path_ok():
            # G7: the promotion may not cover decision jobs, so the full
            # estimate (not just the eval part) must fit the limits. The
            # counted spend stays the eval part (likely actual cost).
            full = float(est["estimate_usd"])
            if not self._limits_ok(full):
                return None
            e = float(est.get("eval_usd", est["estimate_usd"]))
            return e if self._limits_ok(e) else None
        if not self.cfg.get("fireworks", {}).get("allow_paid_training", False):
            return None
        e = float(est["estimate_usd"])
        return e if self._limits_ok(e) else None

    # -- checks (write) --------------------------------------------------
    def kick(self, point_id: str) -> None:
        """Trigger a check after do_record/do_outcome (<= once/min/point).

        F3: reloads before checking approval so a terminal re-approval takes
        effect without a restart. F4: never raises (the label is already
        committed); errors are notified with the approved command.
        """
        try:
            self._reload_section()
            if self.global_state() != "ok":
                self._notify_approval_block()
                return
            now = self.svc._monotonic()
            last = self.svc._ap_last_check.get(point_id)
            if last is not None and now - last < 60:
                return
            self.svc._ap_last_check[point_id] = now
            self.check([point_id])
        except Exception as e:
            try:
                notify(self._notify_cmd_for(),
                       f"autopilot check failed for {point_id}: "
                       f"{type(e).__name__}")
            except Exception:
                pass

    def check(self, point_ids: list[str] | None = None) -> dict:
        """Run autopilot once. Returns {checked, actions}."""
        self.checks_run += 1
        self._reload_section()
        result: dict = {"checked": [], "actions": []}
        if self.global_state() != "ok":
            self._notify_approval_block()
            return result
        for fin in self.poll_and_finalize():
            result["actions"].append(fin)
        if point_ids is None:
            points = self.store.list_points()
        else:
            points = [p for pid in point_ids
                      if (p := self.store.get_point(pid)) is not None]
        for point in points:
            for act in self._check_point(point):
                result["actions"].append(act)
            result["checked"].append(point["point_id"])
        return result

    def _notify_hash_change(self) -> None:
        """Backward-compat alias (tests): approval-block notification."""
        self._notify_approval_block()

    def _notify_approval_block(self) -> None:
        """Notify once when approval blocks autopilot (F1: approved cmd only)."""
        state = self.global_state()
        if state == "trust_key_changed":
            b64_pub = (self.cfg.get("autopilot_trust", {}) or {}).get(
                "public_key", "") or ""
            fp_now = fingerprint_pubkey(b64_pub) or "none"
            if self.store.meta_get(TRUST_NOTIFIED_KEY) == fp_now:
                return
            notify(self._notify_cmd_for(),
                   "autopilot stopped: trust key changed "
                   f"(now {fp_now}), needs terminal re-approval; possible key "
                   "replacement attack, see trust_key_changed")
            self.store.meta_put(TRUST_NOTIFIED_KEY, fp_now)
            return
        if state != "not_approved":
            return
        current = effective_hash(self.cfg)
        if self.store.meta_get(NOTIFIED_HASH_KEY) == current:
            return
        approved = self.store.meta_get(APPROVED_HASH_KEY)
        a = (approved or "none")[:6]
        notify(self._notify_cmd_for(),
               f"autopilot stopped: not approved (approved {a} != "
               f"current {current[:6]}), needs terminal re-approval")
        self.store.meta_put(NOTIFIED_HASH_KEY, current)

    def _check_point(self, point: dict) -> list[dict]:
        actions = []
        status, waiting = self.point_status(point)
        if status != "on":
            return actions
        ap = self.section
        if waiting == "collect_to_shadow" and ap.get("auto_serve", True):
            actions.append(self._promote(point, "shadow"))
            point = self.store.get_point(point["point_id"]) or point
            status, waiting = self.point_status(point)
            if status != "on":
                return actions
        if waiting == "shadow_to_serve" and ap.get("auto_serve", True):
            actions.append(self._promote(point, "serve"))
        elif waiting == "train" and ap.get("auto_train", True):
            actions.append(self._launch(point))
        return actions

    def _gates_for_promote(self, point: dict, phase: str) -> dict:
        if phase == "shadow":
            return {"decider": self._decider_line(point)}
        checks = self.svc._serve_checks(point, True, None,
                                        since=self._demotion_since(point))
        gates = {}
        for name, _item in GATE_ORDER:
            c = checks.get(name)
            if c is not None:
                gates[name] = {"pass": c["pass"], "detail": c.get("detail")}
        return gates

    def _decider_line(self, point: dict) -> str:
        svc = self.svc
        if svc._hosted_tier(point):
            return "hosted Jev key + budget + remote_ok"
        return "trained private model"

    def _promote(self, point: dict, phase: str) -> dict:
        gates = self._gates_for_promote(point, phase)
        # E3: a concurrent user pause/demotion wins. Re-read and compare
        # phase, stop flag, version, suspected, irreversible and the
        # demotion cooldown inside one BEGIN IMMEDIATE; on mismatch skip
        # without logging or notifying (no stale overwrite).
        fresh = self.store.cas_point_phase(
            point["point_id"], point.get("phase", "collect"),
            point.get("point_version", 1),
            bool(point.get("autopilot_stopped", False)), phase,
            bool(point.get("suspected", False)),
            bool(point.get("irreversible", False)),
            point.get("autopilot_demotion_ts"))
        if fresh is None:
            return {"point_id": point["point_id"],
                    "action": "promotion_skipped",
                    "reason": "concurrent_modification"}
        h = effective_hash(self.cfg)
        line = self._promote_line(fresh, phase, gates)
        self.store.autopilot_log_add(
            {"ts": _now_iso(), "point_id": point["point_id"],
             "action": f"promoted_to_{phase}", "gates": gates,
             "config_hash": h})
        notify(self._notify_cmd_for(fallback_to_current=True), line)
        return {"point_id": point["point_id"], "action": f"promoted_to_{phase}",
                "line": line}

    def _promote_line(self, point: dict, phase: str, gates: dict) -> str:
        pid = point["point_id"]
        if phase == "shadow":
            return f"{pid} promoted to shadow: {gates['decider']}"
        ev = (gates.get("evidence") or {}).get("detail", "")
        mg = (gates.get("margin") or {}).get("detail", "")
        return f"{pid} promoted to serve: {ev}; {mg}"

    def monitored(self, point: dict) -> bool:
        """Points autopilot watches (demotions logged) and may act on."""
        if self.global_state() != "ok":
            return False
        ap = self.section
        return bool(
            _in_scope(point["point_id"], ap.get("points", []))
            and _point_target(point, self.cfg)
            <= float(ap.get("max_target_error", 0.02))
            and not point.get("suspected", False)
            and not self._teacher_unreliable(point)
            and not point.get("autopilot_stopped", False)
            and point.get("phase") != "paused")

    def note_demotion(self, point: dict, detail: dict) -> None:
        """Log + announce an automatic drift demotion (starts the cooldown)."""
        if not self.monitored(point):
            return
        h = effective_hash(self.cfg)
        line = (f"{point['point_id']} demoted to shadow (drift): "
                f"{detail.get('errors')}/{detail.get('n')} audit errors")
        self.store.autopilot_log_add(
            {"ts": _now_iso(), "point_id": point["point_id"],
             "action": "demoted", "gates": detail, "config_hash": h})
        notify(self._notify_cmd_for(fallback_to_current=True), line)

    # -- training launch / poll / finalize -------------------------------
    def _launch(self, point: dict) -> dict:
        dry = self.dry_run(point)
        counted = self._counted_usd(dry)
        assert counted is not None  # point_status said ready
        try:
            job = self.launch_job(
                point, dry, counted, autopilot=True,
                free_train=dry["kind"] == "full" and self.free_path_ok())
        except JobRunning:
            return {"point_id": point["point_id"], "action": "train_skipped",
                    "reason": "job_running"}
        except JobCancelled:
            return {"point_id": point["point_id"], "action": "train_skipped",
                    "reason": "job_cancelled"}
        h = effective_hash(self.cfg)
        gates = {"kind": dry["kind"], "estimate_usd": dry["estimate"]["estimate_usd"],
                 "counted_usd": counted, "free": dry["kind"] == "full" and self.free_path_ok(),
                 "joins": [j["point_id"] for j in dry["joins"]
                           if j.get("allowed") and j.get("join_ok", True)]}
        line = (f"{point['point_id']} training launched ({dry['kind']} job "
                f"{job['job_id']}): estimate ${dry['estimate']['estimate_usd']:.4f}")
        self.store.autopilot_log_add(
            {"ts": _now_iso(), "point_id": point["point_id"],
             "action": "train_launched", "gates": gates, "config_hash": h,
             "estimate_usd": counted})
        notify(self._notify_cmd_for(fallback_to_current=True), line)
        return {"point_id": point["point_id"], "action": "train_launched",
                "job_id": job["job_id"], "line": line}

    def launch_job(self, point: dict, dry: dict, counted: float,
                   autopilot: bool, idem: str | None = None,
                   free_train: bool = False) -> dict:
        """Export allowed rows, upload and record a running job. Shared by
        manual do_train and autopilot (the callers passed their own gates).

        G1: a launching row with an idempotency key is written before the
        upload; a crash there reconciles on the next check (no silent
        duplicate). G3: the check-and-insert is one transaction plus the
        unique partial index; the loser gets JobRunning. F4: the caller
        may pass idem (do_train idem); two launchers with the same idem
        share the one live job.
        """
        import sqlite3 as _sqlite3
        import uuid as _uuid

        from .trainer import IdemInUse

        # Y11: upload only joins the dry-run marked (allowed and
        # join_ok). An unmarked dry is fail-closed: only the trigger rides.
        marks = {j["point_id"]: (j.get("allowed", False) and j.get(
            "join_ok", j["point_id"] == point["point_id"]))
            for j in dry["joins"]}
        upload = [r for r in dry["rows"]
                  if marks.get(r["point_id"], False)]
        joins = []
        for j in dry["joins"]:
            rs = [r for r in upload if r["point_id"] == j["point_id"]]
            latest = max(rs, key=_row_key, default=None)
            joins.append({**j, "n_upload": len(rs),
                          "export_ts": latest["ts"] if latest else "",
                          "export_id": latest["decision_id"]
                          if latest else "",
                          "export_seq": latest.get("seq")
                          if latest else None})
        idem = idem or _uuid.uuid4().hex
        # Y2: a used idem shares its live job, but a terminal row
        # strands its idem (relaunching would adopt a stale exit_code
        # and orphan a paid job). No DB unique index: temp_id derives
        # from idem, so a concurrent same-idem insert always collides
        # on the temp PK and lands in JobRunning below.
        prior = self.store.job_by_idem(idem)
        if prior is not None:
            if prior["status"] in ("running", "launching", "evaluating"):
                raise JobRunning()
            raise IdemInUse(f"idem {idem!r} was already used by "
                            f"{prior['job_id']} ({prior['status']})")
        temp_id = f"launching-{idem[:12]}"
        # F8: the train ledger amount (0 when the free path covers the
        # train part). Y9: it includes the in-job dev-eval part, which
        # bills inside the training session; finalize settles the
        # ledger while counted_usd keeps the launch estimate (the
        # monthly cap is the sum of autopilot job estimates).
        train_ledger = 0.0 if (free_train and dry["kind"] == "full") \
            else float(dry["estimate"].get("train_usd", 0)) + float(
                dry["estimate"].get("dev_eval_usd", 0))
        launching = {"job_id": temp_id, "kind": dry["kind"],
                     "status": "launching",
                     "estimate_usd": dry["estimate"]["estimate_usd"],
                     "counted_usd": counted, "spend_usd": 0.0,
                     "train_usd": train_ledger,
                     "points": joins, "autopilot": autopilot,
                     "created_at": _now_iso(), "idempotency_key": idem}
        try:
            self.store.job_insert_launching(launching)
        except RuntimeError:
            raise JobRunning()
        except _sqlite3.IntegrityError:
            raise JobRunning()
        try:
            remote_id = self.svc.trainer.launch(upload, dry["kind"], idem)
        except Exception:
            self.store.job_update(temp_id, {"status": "failed",
                                            "error": "launch failed",
                                            "failed_at": _now_iso()})
            raise
        if not self.store.job_claim_launch(temp_id, remote_id, idem):
            # Lost the claim (reconciled concurrently); treat as running-loss.
            row = self.store.job_get(temp_id)
            if row is not None and row.get("status") == "cancelled":
                # Y10: a cancel landed mid-spawn. Our child is already
                # up under remote_id: kill it (it is untracked now),
                # then report the cancelled row, not a phantom job.
                try:
                    self.svc.trainer.cancel(remote_id)
                except Exception:
                    pass
                raise JobCancelled(temp_id)
            raise JobRunning()
        return self.store.job_get(remote_id) or {**launching, "job_id": remote_id,
                                                 "status": "running"}

    def _launch_stale(self, job: dict) -> bool:
        """A launching row older than the claim window (GP2: 10 min)."""
        ts = _parse_ts(job.get("created_at"))
        if ts is None:
            return True
        return time.time() - ts >= STALE_LAUNCH_S

    def reconcile_launching(self) -> list[dict]:
        """G1/F4: adopt or fail stale launching rows (crash between insert
        and claim). Young rows are left alone; a stale row is adopted when
        a job still runs (or finished) under its idem, else failed with
        its estimate as spend (GP2/G5: it may already have billed). A
        status check that errors (other than confirmed absence) leaves
        the row: absence must be proven before failing.
        """
        out = []
        for job in self.store.jobs_launching():
            if not self._launch_stale(job):
                continue
            idem = job.get("idempotency_key") or ""
            adopted = False
            st: dict | None = None
            if idem and self.svc.trainer is not None:
                try:
                    st = self.svc.trainer.status(idem)
                except Exception:
                    continue  # uncertain: absence must be proven
                if st is not None and st.get("status") in ("running", "done"):
                    adopted = self.store.job_claim_launch(
                        job["job_id"], idem, idem)
            if adopted:
                continue
            error = "crashed during launch"
            account_id = None
            if st is not None and st.get("status") == "failed" \
                    and st.get("error") != "unknown job":
                # Y9: the job actually ran and failed: fail it with the
                # estimate as spend, keeping the trainer's error. Z7: an
                # "unknown job" is the crash-before-spawn case above.
                error = st.get("error") or "failed"
                account_id = st.get("account_id")
            self._mark_failed(job, error, account_id)
            out.append(self._failed_action({**job, "status": "failed"},
                                           error, account_id=account_id))
        return out

    def _recover_done(self) -> list[dict]:
        """F4: re-run the tails of done jobs that crashed between mark_done
        and store. The tail is idempotent (conditional spend, job-tagged
        models); a tail that still cannot run waits for the next poll."""
        out = []
        if self.svc.trainer is None:
            return out
        for job in self.store.jobs_done_unfinalized():
            try:
                st = self.svc.trainer.status(job["job_id"])
            except Exception:
                continue
            try:
                self._finalize_tail(
                    job, st, self.store.job_get_eval(job["job_id"]))
            except Exception:
                continue
            trigger = (job.get("points") or [{}])[0].get("point_id", "")
            out.append({"point_id": trigger, "action": "finalize_recovered",
                        "job_id": job["job_id"]})
        return out

    def poll_and_finalize(self) -> list[dict]:
        """Poll running jobs; finalize finished ones. Returns actions."""
        out = []
        for rec in self.reconcile_launching():
            out.append(rec)
        for rec in self._recover_done():
            out.append(rec)
        self._reap_eval_threads()
        if self.svc.trainer is None:
            return out
        for job in self.store.jobs_running():
            if job["status"] == "evaluating":
                continue  # F8: owned by its eval thread, not the poll
            try:
                st = self.svc.trainer.status(job["job_id"])
            except Exception:
                # G2: transient errors must not fail-and-relaunch. Only the
                # Nth consecutive failure declares the job lost. Z7: that
                # includes KeyError (a trainer bug, not proof of absence;
                # unknown jobs report failed/"unknown job" instead).
                n = self.store.job_note_status_failure(job["job_id"])
                if n >= STATUS_LOST_AFTER:
                    self._mark_failed(job, "trainer unreachable "
                                           f"({n} polls)")
                    out.append(self._failed_action(
                        job, f"trainer unreachable ({n} polls)"))
                continue
            self.store.job_reset_status_failures(job["job_id"])
            if st.get("status") == "running":
                continue
            if st.get("status") == "failed":
                self._mark_failed(job, st.get("error", "failed"),
                                  st.get("account_id"))
                out.append(self._failed_action(
                    job, st.get("error", "failed"),
                    account_id=st.get("account_id")))
                continue
            if st.get("status") != "done":
                # Trainer-cancelled behind our back: terminal, never
                # finalized (no backoff: a cancel is not a failure).
                if not self._mark_cancelled(job):
                    continue  # finalize won concurrently; leave it done
                trigger = (job.get("points") or [{}])[0].get("point_id", "")
                out.append({"point_id": trigger, "action": "train_cancelled",
                            "job_id": job["job_id"]})
                continue
            if job["kind"] == "full" and isinstance(
                    self.svc.trainer, FireworksTrainer):
                out.append(self._start_evaluation(job))
                continue
            out.append(self._finalize_job(job, st))
        return out

    def _mark_failed(self, job: dict, error: str,
                     account_id: str | None = None) -> None:
        """G5: fail a job, record its estimated spend, stamp backoff time."""
        from .trainer_log import redact_account

        error = redact_account(error, account_id)
        self.store.job_update(job["job_id"], {"status": "failed",
                                              "error": error,
                                              "failed_at": _now_iso()})
        month = _now_iso()[:7]
        counted = float(job.get("counted_usd", 0))
        if counted > 0:
            self.store.spend_add(month, "fireworks", counted)
            self.store.job_update(job["job_id"], {"spend_usd": counted})

    def _failed_action(self, job: dict, error: str,
                       action: str = "train_failed",
                       account_id: str | None = None,
                       store=None) -> dict:
        from .trainer_log import redact_account

        if store is None:
            store = self.store
        error = redact_account(error, account_id)
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        base: dict = {"point_id": trigger, "action": action,
                      "job_id": job["job_id"]}
        if not job.get("autopilot"):
            return base
        h = effective_hash(self.cfg)
        if action == "eval_failed":
            line = (f"{trigger} evaluation failed (job {job['job_id']}): "
                    f"{error}; do_train retry_eval to retry")
        else:
            line = (f"{trigger} training failed ({job['kind']} job "
                    f"{job['job_id']}): {error}; retry after 24 h backoff")
        store.autopilot_log_add(
            {"ts": _now_iso(), "point_id": trigger,
             "action": action, "gates": {"error": error},
             "config_hash": h})
        notify(self._notify_cmd_for(fallback_to_current=True), line)
        return {**base, "line": line}

    def _finalize_job(self, job: dict, st: dict) -> dict:
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        if job["kind"] == "smoke":
            # G4: claim first; the loser records no spend.
            if not self.store.job_finalize(job["job_id"], "smoke_done"):
                return {"point_id": trigger, "action": "finalize_skipped",
                        "job_id": job["job_id"]}
        elif not self.store.job_finalize(job["job_id"], "done"):
            return {"point_id": trigger, "action": "finalize_skipped",
                    "job_id": job["job_id"]}
        return self._finalize_tail(job, st)

    def _finalize_tail(self, job: dict, st: dict,
                       eval: dict | None = None, store=None) -> dict:
        """Idempotent finalize tail (F4): models, atomic spend, log line.

        Runs after the G4 claim and from recovery. Models are skipped
        when this job's card is already active; spend settles exactly
        once via job_record_spend_finalized. F8: evaluated jobs pass
        their eval doc (per-point T/metrics instead of the checkpoint).
        Y3: the eval thread passes its own store (the tick's is closed).
        """
        if store is None:
            store = self.store
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        month = _now_iso()[:7]
        if job["kind"] == "smoke":
            store.meta_put(SMOKE_DONE_AT_KEY, _now_iso())
            self._record_spend(job, month, store)
            metrics = st.get("metrics", {})
            if job.get("autopilot"):
                return self._log_action(
                    trigger, "smoke_done",
                    {"job_id": job["job_id"], **metrics,
                     "recipe_temperature": st.get("recipe_temperature")},
                    f"{trigger} smoke job done ({job['job_id']}): "
                    f"{self._smoke_detail(st)}",
                    estimate_usd=job.get("counted_usd"),
                    account_id=st.get("account_id"), store=store)
            return {"point_id": trigger, "action": "smoke_done",
                    "job_id": job["job_id"]}
        if eval is not None:
            per_point = eval.get("points", {})
        else:
            ckpt = self.svc.trainer.checkpoint(job["job_id"])
            per_point = (ckpt or {}).get("per_point", {})
        for j in job.get("points", []):
            if not j.get("allowed", True) or not j.get("n_upload"):
                continue
            if eval is not None and j["point_id"] not in per_point:
                continue  # unevaluated join: no card without evidence
            self._save_private_model(j, job, (per_point.get(j["point_id"]) or {}),
                                     store)
        settle = float(job.get("train_usd") or 0) \
            if eval is not None else None
        self._record_spend(job, month, store, settle)
        if job.get("autopilot"):
            if eval is not None:
                return self._log_action(
                    trigger, "trained",
                    {"job_id": job["job_id"],
                     "points": sorted(per_point),
                     "eval_cost_usd": eval.get("eval_cost_usd"),
                     "eval_tokens": eval.get("eval_tokens"),
                     "accuracy": {p: m.get("accuracy")
                                  for p, m in per_point.items()}},
                    f"{trigger} trained (job {job['job_id']}): "
                    f"{eval.get('summary', '')}",
                    estimate_usd=job.get("counted_usd"),
                    account_id=st.get("account_id"), store=store)
            metrics = st.get("metrics", {})
            return self._log_action(
                trigger, "trained",
                {"job_id": job["job_id"], **metrics,
                 "points": [j["point_id"] for j in job.get("points", [])]},
                f"{trigger} trained (job {job['job_id']}): "
                f"accuracy {metrics.get('accuracy')}, "
                f"{metrics.get('n_train')} train rows",
                estimate_usd=job.get("counted_usd"),
                account_id=st.get("account_id"), store=store)
        return {"point_id": trigger, "action": "trained",
                "job_id": job["job_id"]}

    @staticmethod
    def _smoke_detail(st: dict) -> str:
        """One-line smoke summary from fake or log-parsed metrics (F8)."""
        metrics = st.get("metrics", {})
        parts = []
        if metrics.get("n_train") is not None:
            parts.append(f"{metrics['n_train']} train rows")
        if metrics.get("n_eval") is not None:
            parts.append(f"{metrics['n_eval']} eval calls")
        if metrics.get("eval_accuracy") is not None:
            parts.append(f"eval acc {metrics['eval_accuracy']:.4f}")
        if metrics.get("eval_loss") is not None:
            parts.append(f"eval loss {metrics['eval_loss']:.4f}")
        if st.get("recipe_temperature") is not None:
            parts.append(f"T {st['recipe_temperature']}")
        return ", ".join(parts) if parts else "no metrics"

    # -- background evaluation (M11 F8, GP6) ----------------------------------
    def _start_evaluation(self, job: dict) -> dict:
        """Claim running -> evaluating and spawn the eval thread."""
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        if not self.store.job_begin_eval(job["job_id"]):
            return {"point_id": trigger, "action": "finalize_skipped",
                    "job_id": job["job_id"]}
        self._spawn_eval_thread(job["job_id"])
        return {"point_id": trigger, "action": "eval_started",
                "job_id": job["job_id"]}

    def retry_evaluation(self, job: dict) -> dict:
        """Claim eval_failed -> evaluating and re-run the thread."""
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        if not self.store.job_retry_eval(job["job_id"]):
            return {"point_id": trigger, "action": "finalize_skipped",
                    "job_id": job["job_id"]}
        self._spawn_eval_thread(job["job_id"])
        return {"point_id": trigger, "action": "eval_started",
                "job_id": job["job_id"]}

    def _spawn_eval_thread(self, job_id: str) -> None:
        self._eval_cancel[job_id] = threading.Event()
        t = threading.Thread(target=self._run_evaluation, args=(job_id,),
                             daemon=True, name=f"eval-{job_id[:12]}")
        self._eval_threads[job_id] = t
        t.start()

    def join_eval(self, job_id: str, timeout: float | None = None) -> bool:
        """Wait for a job's eval thread. True when none is alive."""
        t = self._eval_threads.get(job_id)
        if t is None:
            return True
        t.join(timeout)
        if t.is_alive():
            return False
        self._eval_threads.pop(job_id, None)
        self._eval_cancel.pop(job_id, None)
        return True

    def _reap_eval_threads(self) -> None:
        for jid, t in list(self._eval_threads.items()):
            if not t.is_alive():
                self._eval_threads.pop(jid, None)
                self._eval_cancel.pop(jid, None)

    def _run_evaluation(self, job_id: str) -> None:
        """Thread body: evaluate, or terminalize on any failure. Never
        raises (a dead thread with status evaluating would wedge the
        one-job lock until a restart).

        Y3: the thread opens its own Store. The server tick builds a
        fresh Service per check and closes its store when the check
        returns; sharing it would kill the eval mid-run with a closed
        connection (stuck evaluating, model never saved, spend lost).
        """
        from .store import Store

        estore = Store(self.store.root)
        try:
            self._do_evaluation(job_id, estore)
        except _EvalCancel:
            pass  # cancel won; the row is already terminal
        except _EvalAbort as e:
            try:
                job = estore.job_get(job_id)
                if job is not None and job["status"] == "evaluating":
                    self._eval_failed(job, str(e), estore)
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001 - last resort, still terminal
            try:
                job = estore.job_get(job_id)
                if job is not None and job["status"] == "evaluating":
                    self._eval_failed(job, f"eval crashed: {e}", estore)
            except Exception:
                pass
        finally:
            estore.close()

    def _eval_worker(self, model: str) -> WorkerSession:
        argv = None
        if self.worker_argv_fn is not None:
            argv = self.worker_argv_fn(model)
        return WorkerSession(
            self.cfg, model, self.store.root / "jobs",
            self.svc.fireworks_key or "", worker_argv=argv)

    def _do_evaluation(self, job_id: str, store) -> None:
        job = store.job_get(job_id)
        if job is None or job["status"] != "evaluating":
            return
        st = self.svc.trainer.status(job_id)
        if st.get("status") != "done":
            raise _EvalAbort("trainer lost the job")
        model = st.get("model")
        if not model:
            raise _EvalAbort("no promoted model in the job log")
        recipe_t = st.get("recipe_temperature")
        try:
            splits = self.svc.trainer.eval_rows(job_id)
        except Exception as e:
            raise _EvalAbort(f"no eval rows: {e}")
        calib = splits.get("calib", [])
        held = splits.get("heldout", [])
        cancel_ev = self._eval_cancel.get(job_id) or threading.Event()
        sess = self._eval_worker(model)
        try:
            scored_c = self._eval_score(sess, job, calib, cancel_ev, store)
            scored_h = self._eval_score(sess, job, held, cancel_ev, store)
        finally:
            sess.close()
        self._raise_if_cancelled(job_id, cancel_ev, store)
        uploads = {j["point_id"]: j.get("n_upload", 0)
                   for j in job.get("points", [])}
        total_up = sum(uploads.values())
        train_ledger = float(job.get("train_usd") or 0)
        per_point = {}
        for pid in sorted({s[1] for s in scored_c + scored_h}):
            c = [s for s in scored_c if s[1] == pid]
            h = [s for s in scored_h if s[1] == pid]
            share = (uploads.get(pid, 0) / total_up) if total_up else 0.0
            per_point[pid] = self._eval_fit_point(
                pid, c, h, recipe_t, model,
                train_ledger * share, job.get("created_at", ""),
                uploads.get(pid, 0), store)
        unevaluated = sorted(
            p for p in uploads if p not in per_point and uploads[p])
        doc = {
            "model": model, "recipe_temperature": recipe_t,
            "points": per_point,
            "eval_cost_usd": sum(m["eval_cost_usd"]
                                 for m in per_point.values()),
            "eval_tokens": sum(m["eval_tokens"] for m in per_point.values()),
            "n_calib": len(scored_c), "n_heldout": len(scored_h),
            "unevaluated": unevaluated,
            "summary": "; ".join(
                f"{p}: acc {_fmt_num(m.get('accuracy'))}, "
                f"T {_fmt_num(m.get('temperature'))}"
                for p, m in sorted(per_point.items())) or "no eval rows",
            "finished_at": _now_iso(),
        }
        store.job_save_eval(job_id, doc)
        # Y9: counted_usd keeps the launch estimate (the monthly cap
        # sums estimates); the tail settles the train ledger while the
        # per-call eval spend already settled live.
        self._raise_if_cancelled(job_id, cancel_ev, store)
        if not store.job_claim_evaluated(job_id):
            return  # lost the race (cancel/retry/recovery claimed first)
        self._finalize_tail(store.job_get(job_id), st, doc, store)

    def _raise_if_cancelled(self, job_id: str,
                            cancel_ev: threading.Event, store) -> None:
        if cancel_ev.is_set():
            raise _EvalCancel()
        job = store.job_get(job_id)
        if job is None or job["status"] != "evaluating":
            raise _EvalCancel()

    def _eval_score(self, sess: WorkerSession, job: dict, rows: list[dict],
                    cancel_ev: threading.Event, store) -> list[tuple]:
        """Score rows in batches of 32: [(row, pid, logprobs, label_idx)].

        Each worker call settles immediately (spend_add); a batch that
        would exceed the monthly budget aborts the eval (GP11: inside
        the budget). Malformed rows abort loudly: eval rows come from
        our own export, so anything else is corruption.
        """
        from .fireworks_worker import tokens_of as _toks

        scored = []
        if not rows:
            return scored
        fw = self.cfg.get("fireworks", {})
        price = float(fw.get("score_price_per_mtok", 1.96))
        try:
            budget = float(fw.get("monthly_budget_usd") or 0)
        except (TypeError, ValueError):
            budget = 0.0
        month = _now_iso()[:7]
        # Y9: the running job's train part settles at finalize, so it is
        # committed spend for this check, like open serving reservations.
        train_commit = float(job.get("train_usd") or 0)
        for b in range(0, len(rows), EVAL_BATCH):
            batch = rows[b:b + EVAL_BATCH]
            self._raise_if_cancelled(job["job_id"], cancel_ev, store)
            est = sum(_toks([r]) for r in batch) * price / 1e6
            spent = store.spend_get(month, "fireworks")["usd"]
            open_usd = store.spend_open(month, "fireworks")["usd"]
            committed = spent + open_usd + train_commit
            if budget <= 0 or committed + est > budget:
                raise _EvalAbort(
                    f"budget_exceeded: eval batch ${est:.4f} over "
                    f"remaining ${max(budget - committed, 0.0):.4f}")
            try:
                out = self._eval_wait_ready(sess, batch, cancel_ev,
                                            job["job_id"], store)
            except WorkerError as e:
                raise _EvalAbort(f"worker: {e}"[:200])
            except WorkerUnavailable as e:
                raise _EvalAbort(f"worker_{e.reason}")
            lpss = out.get("logprobs")
            if not isinstance(lpss, list) or len(lpss) != len(batch):
                raise _EvalAbort("worker logprobs do not match rows")
            toks = out.get("tokens_in")
            if not isinstance(toks, (int, float)) or toks < 0:
                toks = sum(_toks([r]) for r in batch)
            actual = toks * price / 1e6
            store.spend_add(month, "fireworks", actual)
            denom = sum(_toks([r]) for r in batch) or 1
            for r, lps in zip(batch, lpss):
                pid, keys, y = self._eval_row_label(r)
                if not isinstance(lps, list) or len(lps) != len(keys):
                    raise _EvalAbort(
                        "worker logprobs do not match options")
                try:
                    lps = [float(v) for v in lps]
                except (TypeError, ValueError):
                    raise _EvalAbort(
                        "worker logprobs are not floats")
                rt = _toks([r])
                scored.append((r, pid, lps, y, actual * rt / denom,
                               toks * rt / denom))
        return scored

    @staticmethod
    def _eval_row_label(row: dict) -> tuple[str, list[str], int]:
        """(point_id, option keys, label index) of one eval row."""
        questions = row.get("questions")
        if not isinstance(questions, dict) or len(questions) != 1:
            raise _EvalAbort("eval rows must carry exactly one question")
        pid, q = next(iter(questions.items()))
        keys = list((q.get("criteria") or {}))
        label = q.get("label")
        if label not in keys:
            raise _EvalAbort("eval row label is not an option")
        return pid, keys, keys.index(label)

    def _eval_wait_ready(self, sess: WorkerSession, batch: list[dict],
                         cancel_ev: threading.Event, job_id: str,
                         store) -> dict:
        """score() through the warming window (bounded by eval_ready_s)."""
        try:
            cap = float(self.cfg.get("fireworks", {}).get("eval_ready_s",
                                                           600))
        except (TypeError, ValueError):
            cap = 600.0
        try:
            call_timeout = float(self.cfg.get("fireworks", {}).get(
                "eval_timeout_s", 600))
        except (TypeError, ValueError):
            call_timeout = 600.0
        deadline = time.monotonic() + max(cap, 0.0)
        while True:
            try:
                return sess.score(batch, call_timeout)
            except WorkerUnavailable as e:
                if e.reason != "warming":
                    raise
            self._raise_if_cancelled(job_id, cancel_ev, store)
            if time.monotonic() >= deadline:
                raise _EvalAbort(
                    f"worker_warming_timeout after {cap:g}s")
            time.sleep(EVAL_READY_POLL_S)

    def _eval_fit_point(self, pid: str, calib: list[tuple],
                        held: list[tuple], recipe_t: float | None,
                        model: str, train_cost: float,
                        created_at: str, n_upload: int, store) -> dict:
        """Per-point T (GP4), §6.1 threshold and §7.2 step 4 metrics."""
        point = store.get_point(pid)
        target = float((point or {}).get("target_error")
                       or self.cfg["serve"]["target_error"])
        support = int(self.cfg["serve"]["min_auto_support"])
        min_recall = float(self.cfg["serve"]["min_option_recall"])
        items_c = [(lps, y) for (_r, _p, lps, y, _c, _t) in calib]
        T, skipped = HF.fit_temperature_private(items_c, recipe_t or 1.0)
        pairs = []
        tops = []
        labels = []
        options: list[str] = []
        for (_r, _p, lps, y, _c, _t) in held:
            probs = S.softmax([v / T for v in lps])
            top = max(range(len(probs)), key=probs.__getitem__)
            tops.append(top)
            labels.append(y)
            pairs.append((max(probs), top == y))
        if held:
            r0 = held[0][0]
            q0 = next(iter(r0["questions"].values()))
            options = list((q0.get("criteria") or {}))
        n_h = len(held)
        errors = sum(1 for _, ok in pairs if not ok)
        accuracy = (n_h - errors) / n_h if n_h else None
        per_option = {}
        for i, o in enumerate(options):
            idx = [k for k, y in enumerate(labels) if y == i]
            rec = sum(1 for k in idx if tops[k] == i) / len(idx) \
                if idx else 0.0
            pred = [k for k, t in enumerate(tops) if t == i]
            prec = sum(1 for k in pred if labels[k] == i) / len(pred) \
                if pred else 0.0
            per_option[o] = {"recall": rec, "precision": prec,
                             "n": len(idx)}
        teacher_only = [o for o in options
                        if per_option[o]["recall"] < min_recall
                        or per_option[o]["n"] < 10]
        macro = self._macro_f1(per_option, options)
        maj = max(set(labels), key=labels.count) if labels else None
        base_acc = (labels.count(maj) / n_h) if maj is not None else None
        base_macro = self._majority_macro_f1(labels, options, maj)
        threshold = HF.fit_threshold(pairs, target, support) if pairs \
            else None
        if threshold is not None:
            at = [c for conf, c in pairs if conf >= threshold]
            coverage = len(at) / len(pairs)
            err_n = sum(1 for c in at if not c)
            err_at = err_n / len(at)
            wilson = S.wilson_upper(err_n, len(at))
        else:
            coverage, err_at, wilson = 0.0, None, None
        sources: dict[str, int] = {}
        for (r, _p, _l, _y, _c, _t) in calib + held:
            src = r.get("source")
            if src:
                sources[src] = sources.get(src, 0) + 1
        eval_cost = sum(c for (_r, _p, _l, _y, c, _t) in calib + held)
        eval_toks = sum(t for (_r, _p, _l, _y, _c, t) in calib + held)
        try:
            job_min = max(0.0, (time.time() - _parse_ts(created_at)) / 60)
        except (TypeError, ValueError):
            job_min = None
        return {
            "threshold": threshold, "accuracy": accuracy,
            "temperature": T,
            "temperature_source": "recipe" if skipped else "fit",
            "recipe_temperature": recipe_t,
            "model_ref": model,
            "eval_tokens": eval_toks, "eval_cost_usd": eval_cost,
            "scoring": "session_forward",
            "prompt_renderer": (
                f"cookbook@{self.cfg['fireworks']['cookbook_commit']}"),
            "metrics": {
                "n_train": max(0, n_upload - len(calib) - len(held)),
                "n_calib": len(calib), "n_holdout": n_h,
                "label_sources": sources,
                "accuracy": accuracy, "macro_f1": macro,
                "baseline_accuracy": base_acc,
                "baseline_macro_f1": base_macro,
                "per_option": per_option,
                "ece": S.ece(pairs),
                "threshold": threshold, "coverage": coverage,
                "error_at_threshold": err_at,
                "error_at_threshold_wilson_upper": wilson,
                "random_escalation_error":
                    (1.0 - accuracy) if accuracy is not None else None,
                "teacher_only_options": teacher_only,
                "job_minutes": job_min,
                "train_cost_usd": train_cost,
                "eval_cost_usd": eval_cost,
            },
        }

    @staticmethod
    def _macro_f1(per_option: dict, options: list[str]) -> float | None:
        if not options:
            return None
        f1s = []
        for o in options:
            p = per_option[o]["precision"]
            r = per_option[o]["recall"]
            f1s.append(2 * p * r / (p + r) if p + r > 0 else 0.0)
        return sum(f1s) / len(f1s)

    @staticmethod
    def _majority_macro_f1(labels: list[int], options: list[str],
                           maj: int | None) -> float | None:
        if maj is None or not options:
            return None
        n = len(labels)
        f1s = []
        for i in range(len(options)):
            if i == maj:
                tp = labels.count(maj)
                prec = tp / n
                rec = 1.0
            else:
                prec, rec = 0.0, 0.0
            f1s.append(2 * prec * rec / (prec + rec)
                       if prec + rec > 0 else 0.0)
        return sum(f1s) / len(f1s)

    def _eval_failed(self, job: dict, error: str, store=None) -> dict:
        """Terminalize a failed evaluation: the train part records, the
        settled per-call spend stays, the old model stays, retry_eval
        can retry. Spend finalizes exactly once (conditional)."""
        from .trainer_log import redact_account

        if store is None:
            store = self.store
        error = redact_account(error)
        train = float(job.get("train_usd") or 0)
        month = _now_iso()[:7]
        # Y9: counted_usd keeps the launch estimate for the monthly cap.
        store.job_record_spend_finalized(job["job_id"], train, month)
        store.job_update(job["job_id"], {"status": "eval_failed",
                                         "error": error,
                                         "failed_at": _now_iso()})
        # No thread-tracking cleanup here: a live thread may still be
        # settling its in-flight call; join_eval and the poll reaper own
        # that once the thread actually exits.
        return self._failed_action(job, error, action="eval_failed",
                                   store=store)

    def recover_interrupted_evals(self) -> list[dict]:
        """Server startup (F9 wires it): jobs still evaluating died with
        the old process; they become eval_failed (retryable)."""
        return [self._eval_failed(job, "server_restart")
                for job in self.store.jobs_evaluating()]

    def _record_spend(self, job: dict, month: str, store=None,
                      settle: float | None = None) -> None:
        """Settle the job's in-session spend exactly once (conditional).

        Y9: an evaluated job settles its train ledger (the post-job eval
        part already settled per call); anything else settles the
        counted estimate. counted_usd itself never changes after launch.
        """
        if store is None:
            store = self.store
        if settle is None:
            settle = float(job.get("counted_usd", 0))
        store.job_record_spend_finalized(job["job_id"], settle, month)

    def _mark_cancelled(self, job: dict) -> bool:
        """F4: cancel a job: no relaunch, ever. Spend counts like a failure
        (conservative: a cancelled job may already have billed).

        Y10: compare-and-set against the finalize claims. True when we
        won; False (row already done/failed/cancelled) records nothing,
        so a cancel that loses to job_claim_evaluated can neither
        double-count the train spend nor strand a done job as cancelled.
        """
        if job.get("status") == "evaluating":
            # F8: stop the eval thread; the train part records (it ran)
            # and the settled per-call spend stays. Never the full
            # counted estimate: the eval part would double-count.
            ev = self._eval_cancel.get(job["job_id"])
            if ev is not None:
                ev.set()
            train = float(job.get("train_usd") or 0)
            month = _now_iso()[:7]
            if not self.store.job_mark_cancelled(job["job_id"], train, None):
                return False
            # Y9: counted_usd keeps the launch estimate for the monthly cap.
            self.store.spend_add(month, "fireworks", train)
            return True
        counted = float(job.get("counted_usd", 0))
        if not self.store.job_mark_cancelled(job["job_id"], None,
                                             _now_iso()):
            return False
        if counted > 0:
            self.store.spend_add(_now_iso()[:7], "fireworks", counted)
        return True

    def _save_private_model(self, join: dict, job: dict, model: dict,
                              store=None) -> None:
        """Model card file + active kind=private model + export marker."""
        import json as _json

        if store is None:
            store = self.store
        point = store.get_point(join["point_id"])
        if point is None:
            return
        cur = store.active_model(join["point_id"], "private")
        if cur is not None and cur.get("card", {}).get("job_id") == job["job_id"]:
            return  # F4: recovery re-run; this job's card is already active
        pending = point.get("pending_private_model") or {}
        if pending.get("job_id") == job["job_id"]:
            return  # F9: recovery re-run; this job's card is already pending
        ver = point.get("point_version", 1)
        dest = store.models_dir / point["point_id"]
        dest.mkdir(parents=True, exist_ok=True)
        n = 1
        while (dest / f"v{ver}-{n}.json").exists():
            n += 1
        card = {"point_id": point["point_id"], "point_version": ver,
                "kind": "private", "job_id": job["job_id"],
                "threshold": model.get("threshold"),
                "accuracy": model.get("accuracy"),
                "export_ts": join.get("export_ts", ""),
                "export_decision_id": join.get("export_id", ""),
                "export_seq": join.get("export_seq"),
                "n_upload": join.get("n_upload", 0)}
        if model.get("scoring") == "session_forward":
            # F8 eval card (§7.2 step 4): the full name stays in
            # model_ref (reports print the redacted form, GP9/F9).
            ref = model.get("model_ref") or ""
            card.update({
                "model_ref": ref,
                "model_tail": ref.rsplit("/", 1)[-1],
                "temperature": model.get("temperature"),
                "temperature_source": model.get("temperature_source"),
                "recipe_temperature": model.get("recipe_temperature"),
                "metrics": model.get("metrics", {}),
                "scoring": "session_forward",
                "prompt_renderer": model.get("prompt_renderer"),
                "eval_tokens": model.get("eval_tokens"),
                "eval_cost_usd": model.get("eval_cost_usd")})
        else:
            # Fake-era card (offline gate tests pin this shape). Z2: it
            # still names its model, so evidence pins to this checkpoint.
            card["hint_map"] = model.get("hint_map", {})
            card["sampler"] = f"fakeckpt:{job['job_id']}"
            card["model_ref"] = card["sampler"]
        (dest / f"v{ver}-{n}.json").write_text(_json.dumps(card, indent=1))
        # F9 (§7.2 step 5): a serve-phase private-tier point keeps
        # answering with its current checkpoint; the new card waits as
        # pending until §6.4 holds for it and the user confirms. Every
        # other phase/tier switches at once.
        serve_private = point.get("phase") == "serve" \
            and self.svc._effective_tier(point) == "private"
        mid = store.save_model(point["point_id"], ver, "private",
                               f"v{ver}-{n}.json", card,
                               active=not serve_private)
        if not serve_private and "pending_private_model" in point:
            # Z3: immediate activation voids any stale pending marker (a
            # demotion should already have cleared it; this is the backstop
            # so an older checkpoint can never be switched in later).
            del point["pending_private_model"]
            store.put_point(point)
        if serve_private:
            point["pending_private_model"] = {"model_id": mid,
                                              "job_id": job["job_id"]}
        point["last_train_export_ts"] = join.get("export_ts", "")
        point["last_train_export_id"] = join.get("export_id", "")
        point["last_train_export_seq"] = join.get("export_seq")
        store.put_point(point)

    def _log_action(self, point_id: str, action: str, gates: dict,
                    line: str, estimate_usd=None,
                    account_id: str | None = None, store=None) -> dict:
        from .trainer_log import redact_account, redact_obj

        if store is None:
            store = self.store
        line = redact_account(line, account_id)
        gates = redact_obj(gates, account_id)
        h = effective_hash(self.cfg)
        entry: dict = {"ts": _now_iso(), "point_id": point_id,
                       "action": action, "gates": gates, "config_hash": h}
        if estimate_usd is not None:
            entry["estimate_usd"] = estimate_usd
        store.autopilot_log_add(entry)
        notify(self._notify_cmd_for(fallback_to_current=True), line)
        return {"point_id": point_id, "action": action, "line": line}

    # -- read-only reports (pure) ------------------------------------------
    def block(self) -> dict:
        """do_report's Autopilot block: hashes + last 20 actions, one line each."""
        self._reload_section()  # memory only; the read stays DB-pure
        h = self.hashes()
        actions = []
        for e in self.store.autopilot_log_recent(20):
            actions.append({"ts": e["ts"], "point_id": e["point_id"],
                            "action": e["action"],
                            "line": _log_line(e),
                            "estimate_usd": e.get("estimate_usd")})
        return {"enabled": bool(self.section.get("enabled", False)),
                "approved_hash": h["approved"], "current_hash": h["current"],
                "match": h["match"], "actions": actions}

    def status_report(self) -> dict:
        """do_autopilot_status body: hashes, eligible points, per-point gates."""
        self._reload_section()  # memory only; the read stays DB-pure
        h = self.hashes()
        points = {}
        eligible = []
        for p in self.store.list_points():
            status, waiting = self.point_status(p)
            points[p["point_id"]] = {"status": status, "waiting_for": waiting,
                                     "phase": p.get("phase", "collect")}
            if self.monitored(p):
                eligible.append(p["point_id"])
        return {"enabled": bool(self.section.get("enabled", False)),
                "approved_hash": h["approved"], "current_hash": h["current"],
                "match": h["match"], "eligible_points": sorted(eligible),
                "points": points}


def _log_line(e: dict) -> str:
    pid, action, gates = e["point_id"], e["action"], e.get("gates") or {}
    if action == "promoted_to_shadow":
        return f"{pid} promoted to shadow: {gates.get('decider', '')}"
    if action == "promoted_to_serve":
        ev = (gates.get("evidence") or {}).get("detail", "")
        mg = (gates.get("margin") or {}).get("detail", "")
        return f"{pid} promoted to serve: {ev}; {mg}"
    if action == "train_launched":
        return (f"{pid} training launched ({gates.get('kind')} job): "
                f"estimate ${gates.get('estimate_usd', 0):.4f}")
    if action == "smoke_done":
        return (f"{pid} smoke job done ({gates.get('job_id')}): "
                f"{gates.get('n_train')} train rows")
    if action == "trained":
        return (f"{pid} trained (job {gates.get('job_id')}): "
                f"accuracy {gates.get('accuracy')}")
    if action == "train_failed":
        return f"{pid} training failed: {gates.get('error', '')}"
    if action == "demoted":
        return (f"{pid} demoted to shadow (drift): "
                f"{gates.get('errors')}/{gates.get('n')} audit errors")
    return f"{pid} {action}"
