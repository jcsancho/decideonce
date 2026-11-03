"""Gateway output formats: learning, parsing, rendering (§14.8)."""
from __future__ import annotations

import json


def norm_surface(s: str) -> str:
    return s.strip().lower().strip("\"'`.,;:!?()[]{}").strip()


def learn_format(answers: list[str], options: list[str]) -> dict | None:
    """Learn enum_text or json_enum from logged answers, or None (unsupported)."""
    if len(answers) < 20:
        return None
    opt_norm = {norm_surface(o): o for o in options}
    trimmed = [a.strip() for a in answers]
    if all(norm_surface(a) in opt_norm for a in trimmed):
        return {"type": "enum_text",
                "surface": {o: _surface_for(o, trimmed) for o in options}}
    return _learn_json_enum(trimmed, options)


def _surface_for(option: str, answers: list[str]) -> str:
    """Most common raw surface form of an option."""
    from collections import Counter
    c = Counter(a.strip() for a in answers if norm_surface(a) == norm_surface(option))
    return c.most_common(1)[0][0] if c else option


def _learn_json_enum(trimmed: list[str], options: list[str]) -> dict | None:
    objs = []
    for a in trimmed:
        try:
            o = json.loads(a)
        except (ValueError, TypeError):
            return None
        if not isinstance(o, dict):
            return None
        objs.append(o)
    opt_set = set(options)
    paths = _string_paths(objs[0])
    field_path = None
    for path in sorted(paths):
        vals = [_dig(o, path) for o in objs]
        if all(isinstance(v, str) and v in opt_set for v in vals):
            field_path = path
            break
    if field_path is None:
        return None
    # every other field constant across >= 95%
    from collections import Counter
    all_paths = set()
    for o in objs:
        all_paths.update(_all_paths(o))
    all_paths.discard(field_path)
    for path in all_paths:
        vals = [json.dumps(_dig(o, path), sort_keys=True) for o in objs]
        top = Counter(vals).most_common(1)[0][1]
        if top / len(vals) < 0.95:
            return None
    common = Counter(trimmed).most_common(1)[0][0]
    return {"type": "json_enum", "path": field_path,
            "skeleton": _skeleton_of(json.loads(common), field_path),
            "separators": _separators_of(common)}


def _string_paths(obj: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in obj.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, str):
            out.append(p)
        elif isinstance(v, dict):
            out.extend(_string_paths(v, p))
    return out


def _all_paths(obj: dict, prefix: str = "") -> list[str]:
    out = []
    for k, v in obj.items():
        p = f"{prefix}.{k}" if prefix else k
        out.append(p)
        if isinstance(v, dict):
            out.extend(_all_paths(v, p))
    return out


def _dig(obj: dict, path: str):
    cur: object = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _skeleton_of(obj: dict, field_path: str) -> dict:
    sk = json.loads(json.dumps(obj))
    parts = field_path.split(".")
    cur = sk
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = ""
    return sk


def _separators_of(raw: str) -> list[str]:
    item = ", " if ", " in raw or "," not in raw else ","
    key = ": " if ": " in raw else ":"
    return [item, key]


def parse_answer(text: str, output_format: dict, options: list[str]) -> str | None:
    """Parse an upstream answer into an option, or None when unparsed."""
    kind = output_format.get("type")
    if kind == "enum_text":
        n = norm_surface(text)
        for o in options:
            if norm_surface(o) == n:
                return o
        return None
    if kind == "json_enum":
        try:
            o = json.loads(text.strip())
        except (ValueError, TypeError):
            return None
        if not isinstance(o, dict):
            return None
        v = _dig(o, output_format["path"])
        return v if isinstance(v, str) and v in options else None
    return None


def render_answer(option: str, output_format: dict) -> str:
    """Render the would-be / auto answer for an option."""
    kind = output_format.get("type")
    if kind == "enum_text":
        return output_format.get("surface", {}).get(option, option)
    if kind == "json_enum":
        sk = json.loads(json.dumps(output_format["skeleton"]))
        parts = output_format["path"].split(".")
        cur = sk
        for part in parts[:-1]:
            cur = cur[part]
        cur[parts[-1]] = option
        item, key = output_format.get("separators", [", ", ": "])
        return json.dumps(sk, separators=(item, key))
    raise ValueError(f"unsupported output format {kind!r}")
