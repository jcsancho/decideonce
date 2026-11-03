"""Context validation (§1.1 types) and state_text rendering (§7.1)."""
from __future__ import annotations

import math

DEFAULT_MAX_LEN = 400


def validate_context(point: dict, context: dict
                     ) -> tuple[dict, list[str], list[str]]:
    """Returns (validated, missing_required, type_errors).

    Unknown fields are dropped; null counts as missing. No coercion: types must match.
    """
    fields = point.get("context_fields", {})
    required = point.get("required_fields", [])
    validated: dict = {}
    missing: list[str] = []
    type_errors: list[str] = []
    for name in required:
        if name not in fields:
            continue  # register-time validation rejects this; never happens for stored points
        if name not in context or context[name] is None:
            missing.append(name)
    for name, spec in fields.items():
        if name not in context or context[name] is None:
            continue
        value = context[name]
        if _type_ok(spec, value):
            validated[name] = value
        else:
            type_errors.append(name)
    return validated, missing, type_errors


def _type_ok(spec: dict, value) -> bool:
    t = spec.get("type")
    if t == "string":
        if not isinstance(value, str):
            return False
    elif t == "number":
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            # D: NaN/inf would serialise to invalid JSON downstream.
            return False
    elif t == "integer":
        if not isinstance(value, int) or isinstance(value, bool):
            return False
    elif t == "boolean":
        if not isinstance(value, bool):
            return False
    elif t == "enum":
        if not isinstance(value, str):
            return False
    else:
        return False
    enum = spec.get("enum")
    if enum is not None and value not in enum:
        return False
    return True


def _format_value(spec: dict, value) -> str:
    t = spec.get("type")
    if t == "boolean":
        return "yes" if value else "no"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    if isinstance(value, (int, float)):
        return str(value)
    s = str(value)
    return s[: spec.get("max_len", DEFAULT_MAX_LEN)]


def render_state_obj(point: dict, context: dict,
                     task_input: str | None = None,
                     max_input_chars: int = 8000) -> dict:
    """Ordered state object for the hosted decider (build §2).

    Key order: "policy" (when the point has one), then "input" (only when
    task_input is given, trimmed to max_input_chars), then each registered
    field present in context, in registered order. Booleans stay booleans
    (true/false JSON), numbers as given, strings trimmed to max_len.
    """
    out: dict = {}
    if point.get("policy"):
        out["policy"] = point["policy"]
    if task_input is not None:
        out["input"] = task_input[:max_input_chars]
    fields = point.get("context_fields", {})
    for name, spec in fields.items():
        if name not in context or context[name] is None:
            continue
        value = context[name]
        if isinstance(value, str):
            value = value[: spec.get("max_len", DEFAULT_MAX_LEN)]
        out[name] = value
    return out


def render_state(point: dict, context: dict) -> str:
    """Deterministic state_text: header, one line per field in registered order (§7.1),
    then option_help lines. Same context -> same text."""
    fields = point.get("context_fields", {})
    lines = [f"workflow: {point['workflow']}", f"question: {point['question']}"]
    for name, spec in fields.items():
        if name in context and context[name] is not None:
            lines.append(f"{name}: {_format_value(spec, context[name])}")
    for opt in point["options"]:
        help_text = (point.get("option_help") or {}).get(opt)
        if help_text:
            lines.append(f"help {opt}: {help_text}")
    return "\n".join(lines)
