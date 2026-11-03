"""Rules: checked in code before JEV (§5). No expressions, no eval."""
from __future__ import annotations


def _cond_true(field_value, op: str, expected) -> bool:
    """Pure-Python comparison; type mismatches mean the condition is false (never an error)."""
    try:
        if op == "==":
            return bool(field_value == expected)
        if op == "!=":
            return bool(field_value != expected)
        if op in (">", ">=", "<", "<="):
            if isinstance(field_value, bool) or isinstance(expected, bool):
                return False
            if not isinstance(field_value, (int, float)) or not isinstance(expected, (int, float)):
                return False
            return {"<": field_value < expected, "<=": field_value <= expected,
                    ">": field_value > expected, ">=": field_value >= expected}[op]
        if op == "in":
            return isinstance(expected, list) and field_value in expected
        if op == "not_in":
            return isinstance(expected, list) and field_value not in expected
        if op == "contains":
            return (isinstance(field_value, str) and isinstance(expected, str)
                    and expected.lower() in field_value.lower())
    except Exception:
        return False
    return False


def apply_rules(point: dict, context: dict) -> tuple[list[str], list[str], list[str]]:
    """Returns (allowed_options, fired_rule_ids, warnings).

    All `only` rules intersect first, then all `exclude` rules subtract.
    A rule whose condition references a missing field does not fire and yields
    a `rule_field_missing` warning.
    """
    options = list(point["options"])
    allowed = list(options)
    fired: list[str] = []
    warnings: list[str] = []

    def matches(rule: dict, idx: int) -> bool:
        for field, op, expected in rule["if"]:
            if field not in context or context[field] is None:
                warnings.append(f"rule_field_missing:{rule.get('id', f'rule_{idx}')}:{field}")
                return False
            if not _cond_true(context[field], op, expected):
                return False
        return True

    for idx, rule in enumerate(point.get("rules", [])):
        if "only" in rule and matches(rule, idx):
            keep = set(rule["only"])
            allowed = [o for o in allowed if o in keep]
            fired.append(rule.get("id", f"rule_{idx}"))
    for idx, rule in enumerate(point.get("rules", [])):
        if "exclude" in rule and "only" not in rule and matches(rule, idx):
            drop = set(rule["exclude"])
            allowed = [o for o in allowed if o not in drop]
            fired.append(rule.get("id", f"rule_{idx}"))
    return allowed, fired, sorted(set(warnings))
