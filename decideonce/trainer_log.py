"""Log parser for cookbook training jobs (GP5). Pure functions over text.

Reads the named lines the recipe prints: metrics JSON, the calibration T,
the sampler checkpoint pair, the promotion line, the account id, the
resume hint and the render task count (n_train for smoke jobs).
"""
from __future__ import annotations

import ast
import json
import re

_METRICS_PREFIX = '{"type": "metrics"'
_ACCOUNT_RE = re.compile(r"Auto-resolved account ID: (\S+)")
_REDACT_ACCOUNTS = re.compile(
    r"accounts/(?!fireworks(?:/|\s|$))([^/\s'\"]+)")
_REDACT_BARE_ID = re.compile(
    r"([Aa]uto-resolved [Aa]ccount ID: )(?!acct-…)"
    r"([A-Za-z0-9_][A-Za-z0-9_.\-]*)")
_REDACT_INIT = re.compile(r"init_from_checkpoint='([^'/]+)/")

_MASKED = "acct-…"


def parse_account_id(text: str) -> str | None:
    """The literal id from an ``Auto-resolved account ID:`` line, if any."""
    if not text:
        return None
    m = _ACCOUNT_RE.search(text)
    if not m:
        return None
    return m.group(1).rstrip(".,;:!?'\"")


def redact_account(text: str, account_id: str | None = None) -> str:
    """GP9: mask the Fireworks account id outside the job dir.

    Three layers, idempotent: the literal id (passed by callers that
    know it, else parsed from an ``Auto-resolved account ID:`` line in
    the text) is replaced everywhere — this catches the bare shapes
    (``init_from_checkpoint='<id>/…'``, ``account <id>`` prose) that no
    safe pattern can; then the ``accounts/<id>`` shape (trailing slash
    or not, except the public ``accounts/fireworks/`` vendor prefix);
    then the resume hint's first segment for tails that never name the
    id (X2).
    """
    if not text:
        return text
    if account_id is None:
        account_id = parse_account_id(text)
    if account_id and len(account_id) >= 4 and account_id != _MASKED:
        text = text.replace(account_id, _MASKED)
    if "account" not in text.lower() and "init_from_checkpoint" not in text:
        return text
    text = _REDACT_ACCOUNTS.sub(f"accounts/{_MASKED}", text)
    text = _REDACT_BARE_ID.sub(f"\\1{_MASKED}", text)
    return _REDACT_INIT.sub(f"init_from_checkpoint='{_MASKED}/", text)


def redact_obj(obj, account_id: str | None = None):
    """redact_account over every string in a JSON-shaped value."""
    if isinstance(obj, str):
        return redact_account(obj, account_id)
    if isinstance(obj, list):
        return [redact_obj(v, account_id) for v in obj]
    if isinstance(obj, dict):
        return {k: redact_obj(v, account_id) for k, v in obj.items()}
    return obj
_INIT_RE = re.compile(r"init_from_checkpoint='([^']+)'")
_TASKS_RE = re.compile(r"tasks=(\d+)")
_CAL_RE = "Decision calibration: T="
_SAMPLER_RE = re.compile(r"Sampler checkpoint '([^']+)' saved")
_STORED_RE = re.compile(r"Sampler server-stored name '([^']+)' differs")
_MODEL_RE = re.compile(r"-> model '([^']+)'")


def parse_log(text: str) -> dict:
    """Parse a recipe log. Never raises on malformed input."""
    lines = text.splitlines()
    metrics = []
    for line in lines:
        s = line.strip()
        if not s.startswith(_METRICS_PREFIX):
            continue
        try:
            m = json.loads(s)
        except ValueError:
            continue
        if isinstance(m, dict) and m.get("type") == "metrics" \
                and isinstance(m.get("step"), int):
            metrics.append(m)
    evals = [m for m in metrics if "eval_accuracy" in m]
    trains = [m for m in metrics if "kway_loss" in m]
    sampler: dict | None = None
    for i, line in enumerate(lines):
        m = _SAMPLER_RE.search(line)
        if not m:
            continue
        stored = None
        if i + 1 < len(lines):
            m2 = _STORED_RE.search(lines[i + 1])
            if m2:
                stored = m2.group(1)
        sampler = {"name": m.group(1), "server_stored": stored}
    return {
        "metrics": metrics,
        "last_step": max((m["step"] for m in metrics), default=None),
        "last_eval": evals[-1] if evals else None,
        "last_loss": trains[-1]["kway_loss"] if trains else None,
        "calibration_T": _calibration(lines),
        "sampler_checkpoint": sampler,
        "model": _first_group(lines, _MODEL_RE),
        "promoted": any("Promoted!" in line for line in lines),
        "account_id": _first_group(lines, _ACCOUNT_RE),
        "init_from_checkpoint": _first_group(lines, _INIT_RE),
        "n_train": _first_int(lines, _TASKS_RE),
    }


def _first_group(lines: list[str], rx: re.Pattern) -> str | None:
    for line in lines:
        m = rx.search(line)
        if m:
            return m.group(1)
    return None


def _first_int(lines: list[str], rx: re.Pattern) -> int | None:
    g = _first_group(lines, rx)
    return int(g) if g is not None else None


def _calibration(lines: list[str]) -> dict | None:
    for line in lines:
        i = line.find(_CAL_RE)
        if i < 0:
            continue
        try:
            t = ast.literal_eval(line[i + len(_CAL_RE):].strip())
        except (ValueError, SyntaxError):
            return None
        return dict(t) if isinstance(t, dict) else None
    return None
