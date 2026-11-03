"""Gateway templates: discovery keys, learning, validation, matching (§14.5)."""
from __future__ import annotations

import hashlib

from .analyze import skeleton


def norm_system(system: str) -> str:
    return system.replace("\r\n", "\n").replace("\r", "\n").strip()


def discovery_key(route: str, model: str, system: str) -> str:
    return hashlib.sha256("\x00".join([route, model, skeleton(norm_system(system))]).encode()
                          ).hexdigest()


def fingerprint(route: str, model: str, system: str, prefix: str, suffix: str) -> str:
    return hashlib.sha256("\x00".join([route, model, norm_system(system), prefix, suffix]
                                      ).encode()).hexdigest()


def _common_prefix(strs: list[str]) -> str:
    if not strs:
        return ""
    p = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(p) and i < len(s) and p[i] == s[i]:
            i += 1
        p = p[:i]
    return p


def _common_suffix(strs: list[str]) -> str:
    if not strs:
        return ""
    rev = [s[::-1] for s in strs]
    return _common_prefix(rev)[::-1]


def _cut_prefix(common: str) -> str:
    """Cut back to the last line/sentence boundary (newline, '. ', ': ')."""
    idx = max(common.rfind("\n"), common.rfind(". "), common.rfind(": "))
    if idx < 0:
        return ""
    keep = idx + (1 if common[idx] == "\n" else 2)
    return common[:keep]


def _cut_suffix(common: str) -> str:
    """Cut forward to the first boundary, keeping it."""
    cands = [i for i in (common.find("\n"), common.find(". "), common.find(": "))
             if i >= 0]
    if not cands:
        return ""
    return common[min(cands):]


def middle_of(user: str, prefix: str, suffix: str) -> str | None:
    if not user.startswith(prefix) or (suffix and not user.endswith(suffix)):
        return None
    end = len(user) - len(suffix) if suffix else len(user)
    if end < len(prefix):
        return None
    return user[len(prefix):end]


def template_matches(template: dict, system: str, user: str) -> bool:
    if norm_system(system) != norm_system(template["system"]):
        return False
    return middle_of(user, template["user_prefix"], template["user_suffix"]) is not None


def learn_template(pairs: list[tuple[str, str]], cfg: dict) -> dict:
    """Learn a template from (system, user) pairs.

    Returns {"template": {...}, "fingerprint_parts": ..., "match_rate": ...} or
    {"error": "template_too_weak"|"template_overfit", ...}."""
    gw = cfg["gateway"]
    if not pairs:
        return {"error": "template_too_weak", "reason": "no traces"}
    systems = {norm_system(s) for s, _ in pairs}
    if len(systems) != 1:
        return {"error": "template_too_weak", "reason": "system prompt differs across traces"}
    system = next(iter(systems))
    users = [u for _, u in pairs]
    prefix, suffix = _cut_prefix(_common_prefix(users)), _cut_suffix(_common_suffix(users))
    for _ in users:  # overlap: shorten the suffix until none overlaps
        if any(len(prefix) + len(suffix) > len(u) for u in users) and suffix:
            suffix = suffix[1:]
        else:
            break
    if any(len(prefix) + len(suffix) > len(u) for u in users):
        return {"error": "template_too_weak", "reason": "prefix/suffix overlap"}
    empty = sum(1 for u in users
                if middle_of(u, prefix, suffix) is not None
                and middle_of(u, prefix, suffix) == "")
    if users and empty / len(users) > 0.05:
        return {"error": "template_too_weak", "reason": "middle empty in >5% of traces"}
    if len(system) + len(prefix) + len(suffix) < gw["min_template_chars"]:
        return {"error": "template_too_weak",
                "reason": f"fixed text below min_template_chars ({gw['min_template_chars']})"}
    # 80/20 validation: learn on the first 80%, check the rest
    cut = max(1, int(len(pairs) * 0.8))
    train, held = pairs[:cut], pairs[cut:]
    t_users = [u for _, u in train]
    t_pre, t_suf = _cut_prefix(_common_prefix(t_users)), _cut_suffix(_common_suffix(t_users))
    sys_train = {norm_system(s) for s, _ in train}
    if len(sys_train) != 1 or norm_system(next(iter(sys_train))) != system:
        return {"error": "template_overfit", "reason": "system differs in held-out split"}
    if held:
        ok = sum(1 for s, u in held
                 if norm_system(s) == system and middle_of(u, t_pre, t_suf) is not None)
        rate = ok / len(held)
        if rate < gw["template_min_match"]:
            return {"error": "template_overfit",
                    "reason": f"held-out match {rate:.2f} below {gw['template_min_match']}",
                    "match_rate": rate}
    else:
        rate = 1.0
    return {"template": {"system": system, "user_prefix": prefix, "user_suffix": suffix},
            "match_rate": rate}


def cluster_traces(items: list[dict], cfg: dict) -> list[dict]:
    """Split traces into template clusters (§14.5). Each item needs system/user keys.

    Returns [{n, template?, match_rate?, error?, reason?, items}] (items dropped
    by callers that only report)."""
    min_n = 20
    out: list[dict] = []

    def attempt(group: list[dict]) -> None:
        if len(group) < min_n:
            out.append({"n": len(group), "items": group, "error": "too_small",
                        "reason": f"fewer than {min_n} traces"})
            return
        r = learn_template([(g["system"], g["user"]) for g in group], cfg)
        if "template" in r:
            out.append({"n": len(group), "items": group, **r})
            return
        if r.get("error") != "template_too_weak" or "min_template_chars" not in r.get(
                "reason", ""):
            # Only short fixed text triggers a split; overfit and other
            # failures are terminal for the cluster.
            out.append({"n": len(group), "items": group, **r})
            return
        # split by first line, then by last line
        by_first: dict[str, list] = {}
        for g in group:
            by_first.setdefault(g["user"].split("\n", 1)[0], []).append(g)
        if len(by_first) > 1:
            for sub in sorted(by_first.values(), key=len, reverse=True):
                attempt(sub)
            return
        by_last: dict[str, list] = {}
        for g in group:
            by_last.setdefault(g["user"].rsplit("\n", 1)[-1], []).append(g)
        if len(by_last) > 1:
            for sub in sorted(by_last.values(), key=len, reverse=True):
                attempt(sub)
            return
        out.append({"n": len(group), "items": group, **{k: v for k, v in r.items()}})

    attempt(items)
    return out
