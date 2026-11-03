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

import calendar
import fnmatch
import hashlib
import json
import shlex
import subprocess
import time
from datetime import datetime, timedelta, timezone

from . import labels as L

APPROVED_HASH_KEY = "autopilot_approved_hash"
NOTIFIED_HASH_KEY = "autopilot_notified_hash"
APPROVED_SIG_KEY = "autopilot_approved_sig"
APPROVED_AT_KEY = "autopilot_approved_at"
APPROVED_NOTIFY_KEY = "autopilot_approved_notify"
TRUST_FP_KEY = "autopilot_trust_fingerprint"
TRUST_NOTIFIED_KEY = "autopilot_trust_notified"
SIGNING_KEY_NAME = "DECIDE_AUTOPILOT_SIGNING_KEY"
SMOKE_DONE_AT_KEY = "autopilot_smoke_done_at"

# G2: transient status() failures before a running job is declared lost.
STATUS_LOST_AFTER = 3
# G5: backoff after a failed autopilot job before the next launch.
FAIL_BACKOFF_S = 24 * 3600


class JobRunning(Exception):
    """Another job is launching/running (G3 race lost)."""

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
            "eval_price_per_decision") if k in fw},
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
    import base64

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


def _read_envsave_secret(key_name: str, out, env_runner=None) -> str | None:
    """Read a secret via `envsave show`. Returns the stripped secret or None.

    The password prompt is expected on stderr (Unix convention: data on
    stdout, prompts on stderr), so stderr and stdin stay attached to the
    terminal while stdout is captured. If envsave prints the prompt on
    stdout instead, the last non-empty line is taken as the secret and any
    earlier lines are echoed so the user sees them. env_runner (tests)
    replaces the subprocess call.
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
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return lines[-1].strip() if lines else None
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
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        out("error: envsave returned no secret")
        return None
    if len(lines) > 1:
        for ln in lines[:-1]:
            try:
                print(ln, file=sys.stderr)
            except Exception:
                pass
    return lines[-1].strip()


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
    if env_runner is not None:
        try:
            code = env_runner(["envsave", "set", SIGNING_KEY_NAME], b64_priv)
        except Exception as e:
            out(f"error: envsave set failed: {e}")
            return "error"
        if isinstance(code, tuple):
            code = code[0]
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
                stdout=sys.stdout, stderr=sys.stderr, timeout=120)
        except Exception as e:
            out(f"error: envsave set failed: {e}")
            return "error"
        if proc.returncode != 0:
            out("error: envsave set failed; the key was not stored. "
                "Run it yourself: envsave set " + SIGNING_KEY_NAME)
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
    secret = _read_envsave_secret(SIGNING_KEY_NAME, out, env_runner)
    if not secret:
        return "error"
    import base64

    try:
        raw_priv = base64.b64decode(secret.strip())
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
    try:
        return float(calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ")))
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
        if not svc._legacy_mode and svc._hosted_tier(point):
            if svc._hosted_shadow_gate(point, "") is not None:
                return "no_decider"
            if self._hosted_exhausted():
                return "budget"
            return None
        if not svc._legacy_mode:
            # Private tier (private_model_only): a trained private model.
            model = self.store.active_model(point["point_id"], "private")
            if model is None or model.get("point_version") != point.get(
                    "point_version", 1):
                return "no_decider"
            if getattr(svc.trainer, "needs_key", True) and not svc.fireworks_key:
                return "no_decider"
            if self._fireworks_headroom() is not None \
                    and self._fireworks_headroom() <= 0:
                return "budget"
            return None
        if svc._current_model(point) is None:
            return "no_decider"
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
            rows = L.labeled_rows(
                self.store.decisions_for_point(point["point_id"]), point,
                self.cfg["train"]["teacher_only_weight"],
                self.cfg["extract"]["weight"])
            return len(rows) >= self.cfg["train"]["min_train"]
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
            elif not self.svc._legacy_mode and self._hosted_exhausted() \
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
        if svc._legacy_mode:
            return svc._current_model(point) is None
        return point.get("private_model_only", False) \
            or (svc._effective_tier(point) == "private")

    # -- training dry-run / gates (pure: estimate computes, never uploads) --
    def _labeled(self, point: dict) -> list:
        return L.labeled_rows(
            self.store.decisions_for_point(point["point_id"]), point,
            self.cfg["train"]["teacher_only_weight"],
            self.cfg["extract"]["weight"])

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

    def dry_run(self, point: dict) -> dict:
        """Kind, join list and estimate for the point's next job (no upload)."""
        from . import trainer as TR

        joins = self.join_candidates()
        rows: list[dict] = []
        for j in joins:
            if not j["allowed"]:
                continue  # §7.6: never exported, trained or sent
            p = self.store.get_point(j["point_id"])
            if p is None:
                continue
            for r, label, _src, weight in self._labeled(p):
                ctx = r.get("context")
                rows.append({"point_id": p["point_id"], "ts": r.get("ts") or "",
                             "decision_id": r.get("decision_id") or "",
                             "context": ctx if isinstance(ctx, dict) else {},
                             "label": label, "weight": weight})
        rows.sort(key=lambda r: (r["ts"], r["decision_id"]))
        rows = TR.split_rows(rows, self.cfg["train"]["holdout"],
                             self.cfg["train"].get("calib", 0.15))
        kind = "full" if self.store.smoke_done_exists() else "smoke"
        est = self.svc.trainer.estimate(rows, kind) if self.svc.trainer else {
            "estimate_usd": 0.0, "train_usd": 0.0, "eval_usd": 0.0,
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
        return None

    def _new_rows_ok(self, point: dict) -> bool:
        """min_new_rows labelled rows since the point's last train export."""
        marker = (point.get("last_train_export_ts"),
                  point.get("last_train_export_id"))
        if not marker[0]:
            return True  # first job: none needed
        try:
            need = int(self.section.get("min_new_rows", 100))
        except (TypeError, ValueError):
            need = 100
        n = sum(1 for r, _lab, _src, _w in self._labeled(point)
                if (r.get("ts") or "", r.get("decision_id") or "") > marker)
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
        smoke_at = self.store.meta_get(SMOKE_DONE_AT_KEY)
        approved_at = self.store.meta_get(APPROVED_AT_KEY)
        if not smoke_at or not approved_at:
            return False
        return approved_at > smoke_at

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
        if not svc._legacy_mode and svc._hosted_tier(point):
            return "hosted Jev key + budget + remote_ok"
        if not svc._legacy_mode:
            return "trained private model"
        return "trained probe"

    def _promote(self, point: dict, phase: str) -> dict:
        gates = self._gates_for_promote(point, phase)
        # E3: a concurrent user pause/demotion wins. Re-read and compare
        # phase, stop flag and version inside one BEGIN IMMEDIATE; on
        # mismatch skip without logging or notifying (no stale overwrite).
        fresh = self.store.cas_point_phase(
            point["point_id"], point.get("phase", "collect"),
            point.get("point_version", 1),
            bool(point.get("autopilot_stopped", False)), phase)
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
            job = self.launch_job(point, dry, counted, autopilot=True)
        except JobRunning:
            return {"point_id": point["point_id"], "action": "train_skipped",
                    "reason": "job_running"}
        h = effective_hash(self.cfg)
        gates = {"kind": dry["kind"], "estimate_usd": dry["estimate"]["estimate_usd"],
                 "counted_usd": counted, "free": dry["kind"] == "full" and self.free_path_ok(),
                 "joins": [j["point_id"] for j in dry["joins"]]}
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
                   autopilot: bool) -> dict:
        """Export allowed rows, upload and record a running job. Shared by
        manual do_train and autopilot (the callers passed their own gates).

        G1: a launching row with an idempotency key is written before the
        upload; a crash there reconciles on the next check (no silent
        duplicate). G3: the check-and-insert is one transaction plus the
        unique partial index; the loser gets JobRunning.
        """
        import sqlite3 as _sqlite3
        import uuid as _uuid

        by_point: dict[str, list] = {}
        for r in dry["rows"]:
            by_point.setdefault(r["point_id"], []).append(r)
        upload = [r for pid, rs in by_point.items()
                  for r in rs
                  if next((j for j in dry["joins"]
                           if j["point_id"] == pid), {}).get("allowed", False)]
        joins = []
        for j in dry["joins"]:
            rs = [r for r in upload if r["point_id"] == j["point_id"]]
            marker = max(((r["ts"], r["decision_id"]) for r in rs),
                         default=("", ""))
            joins.append({**j, "n_upload": len(rs),
                          "export_ts": marker[0], "export_id": marker[1]})
        idem = _uuid.uuid4().hex
        temp_id = f"launching-{idem[:12]}"
        launching = {"job_id": temp_id, "kind": dry["kind"],
                     "status": "launching",
                     "estimate_usd": dry["estimate"]["estimate_usd"],
                     "counted_usd": counted, "spend_usd": 0.0,
                     "points": joins, "autopilot": autopilot,
                     "created_at": _now_iso(), "idempotency_key": idem}
        try:
            self.store.job_insert_launching(launching)
        except RuntimeError:
            raise JobRunning()
        except _sqlite3.IntegrityError:
            raise JobRunning()
        try:
            remote_id = self.svc.trainer.launch(upload, dry["kind"])
        except Exception:
            self.store.job_update(temp_id, {"status": "failed",
                                            "error": "launch failed",
                                            "failed_at": _now_iso()})
            raise
        if not self.store.job_claim_launch(temp_id, remote_id, idem):
            # Lost the claim (reconciled concurrently); treat as running-loss.
            raise JobRunning()
        return self.store.job_get(remote_id) or {**launching, "job_id": remote_id,
                                                 "status": "running"}

    def reconcile_launching(self) -> list[dict]:
        """G1: adopt or fail stale launching rows (crash between insert and
        claim). With no live trainer the remote state is unknown, so mark
        failed with backoff (no spend: nothing was confirmed uploaded)."""
        out = []
        for job in self.store.jobs_launching():
            self.store.job_update(job["job_id"], {"status": "failed",
                                                  "error": "crashed during launch",
                                                  "failed_at": _now_iso()})
            out.append(self._failed_action({**job, "status": "failed"},
                                           "crashed during launch"))
        return out

    def poll_and_finalize(self) -> list[dict]:
        """Poll running jobs; finalize finished ones. Returns actions."""
        out = []
        for rec in self.reconcile_launching():
            out.append(rec)
        if self.svc.trainer is None:
            return out
        for job in self.store.jobs_running():
            try:
                st = self.svc.trainer.status(job["job_id"])
            except KeyError:
                # Confirmed absence: the trainer has no such job.
                self._mark_failed(job, "trainer lost the job")
                out.append(self._failed_action(job, "trainer lost the job"))
                continue
            except Exception:
                # G2: transient errors must not fail-and-relaunch. Only the
                # Nth consecutive failure declares the job lost.
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
                self._mark_failed(job, st.get("error", "failed"))
                out.append(self._failed_action(
                    job, st.get("error", "failed")))
                continue
            out.append(self._finalize_job(job, st))
        return out

    def _mark_failed(self, job: dict, error: str) -> None:
        """G5: fail a job, record its estimated spend, stamp backoff time."""
        self.store.job_update(job["job_id"], {"status": "failed",
                                              "error": error,
                                              "failed_at": _now_iso()})
        month = _now_iso()[:7]
        counted = float(job.get("counted_usd", 0))
        if counted > 0:
            self.store.spend_add(month, "fireworks", counted)
            self.store.job_update(job["job_id"], {"spend_usd": counted})

    def _failed_action(self, job: dict, error: str) -> dict:
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        base: dict = {"point_id": trigger, "action": "train_failed",
                      "job_id": job["job_id"]}
        if not job.get("autopilot"):
            return base
        h = effective_hash(self.cfg)
        line = (f"{trigger} training failed ({job['kind']} job "
                f"{job['job_id']}): {error}; retry after 24 h backoff")
        self.store.autopilot_log_add(
            {"ts": _now_iso(), "point_id": trigger,
             "action": "train_failed", "gates": {"error": error},
             "config_hash": h})
        notify(self._notify_cmd_for(fallback_to_current=True), line)
        return {**base, "line": line}

    def _finalize_job(self, job: dict, st: dict) -> dict:
        trigger = (job.get("points") or [{}])[0].get("point_id", "")
        month = _now_iso()[:7]
        if job["kind"] == "smoke":
            # G4: claim first; the loser records no spend.
            if not self.store.job_finalize(job["job_id"], "smoke_done"):
                return {"point_id": trigger, "action": "finalize_skipped",
                        "job_id": job["job_id"]}
            self.store.meta_put(SMOKE_DONE_AT_KEY, _now_iso())
            self._record_spend(job, month)
            metrics = st.get("metrics", {})
            if job.get("autopilot"):
                return self._log_action(
                    trigger, "smoke_done",
                    {"job_id": job["job_id"], **metrics},
                    f"{trigger} smoke job done ({job['job_id']}): "
                    f"{metrics.get('n_train')} train rows, "
                    f"{metrics.get('n_eval')} eval calls",
                    estimate_usd=job.get("counted_usd"))
            return {"point_id": trigger, "action": "smoke_done",
                    "job_id": job["job_id"]}
        # G4: claim first; the loser saves no models and records no spend.
        if not self.store.job_finalize(job["job_id"], "done"):
            return {"point_id": trigger, "action": "finalize_skipped",
                    "job_id": job["job_id"]}
        ckpt = self.svc.trainer.checkpoint(job["job_id"])
        per_point = (ckpt or {}).get("per_point", {})
        for j in job.get("points", []):
            if not j.get("allowed", True) or not j.get("n_upload"):
                continue
            self._save_private_model(j, job, (per_point.get(j["point_id"]) or {}))
        self._record_spend(job, month)
        metrics = st.get("metrics", {})
        if job.get("autopilot"):
            return self._log_action(
                trigger, "trained",
                {"job_id": job["job_id"], **metrics,
                 "points": [j["point_id"] for j in job.get("points", [])]},
                f"{trigger} trained (job {job['job_id']}): "
                f"accuracy {metrics.get('accuracy')}, "
                f"{metrics.get('n_train')} train rows",
                estimate_usd=job.get("counted_usd"))
        return {"point_id": trigger, "action": "trained",
                "job_id": job["job_id"]}

    def _record_spend(self, job: dict, month: str) -> None:
        counted = float(job.get("counted_usd", 0))
        self.store.spend_add(month, "fireworks", counted)
        self.store.job_update(job["job_id"], {"spend_usd": counted})

    def _save_private_model(self, join: dict, job: dict, model: dict) -> None:
        """Model card file + active kind=private model + export marker."""
        import json as _json

        point = self.store.get_point(join["point_id"])
        if point is None:
            return
        ver = point.get("point_version", 1)
        dest = self.store.models_dir / point["point_id"]
        dest.mkdir(parents=True, exist_ok=True)
        n = 1
        while (dest / f"v{ver}-{n}.json").exists():
            n += 1
        card = {"point_id": point["point_id"], "point_version": ver,
                "kind": "private", "job_id": job["job_id"],
                "threshold": model.get("threshold"),
                "accuracy": model.get("accuracy"),
                "hint_map": model.get("hint_map", {}),
                "sampler": f"fakeckpt:{job['job_id']}",
                "export_ts": join.get("export_ts", ""),
                "export_decision_id": join.get("export_id", ""),
                "n_upload": join.get("n_upload", 0)}
        (dest / f"v{ver}-{n}.json").write_text(_json.dumps(card, indent=1))
        self.store.save_model(point["point_id"], ver, "private",
                              f"v{ver}-{n}.json", card)
        point["last_train_export_ts"] = join.get("export_ts", "")
        point["last_train_export_id"] = join.get("export_id", "")
        self.store.put_point(point)

    def _log_action(self, point_id: str, action: str, gates: dict,
                    line: str, estimate_usd=None) -> dict:
        h = effective_hash(self.cfg)
        entry: dict = {"ts": _now_iso(), "point_id": point_id,
                       "action": action, "gates": gates, "config_hash": h}
        if estimate_usd is not None:
            entry["estimate_usd"] = estimate_usd
        self.store.autopilot_log_add(entry)
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
