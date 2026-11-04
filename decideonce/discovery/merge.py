"""Merge text-hint, static and behaviour candidates into evidence levels.

Merge rule: detector candidates join by step name, else (when at least one
side has no step) by option Jaccard >= 0.5. Text candidates without options,
and text/agent candidates matching nothing, are dropped: hints stay visible
in the legacy output and agent candidates in the scored points.

Levels: A = >= 2 detector signals agree (same step, option Jaccard >= 0.5);
B = >= 1 detector signal plus an agent-proposed candidate (and not A, with
agreement unless there is a single detector signal); C = detector signal(s)
only. Rank by level, then rubric score. Options of a level-A candidate are
the union of the agreeing signals' options, with each value's sources.
"""
from __future__ import annotations

import re

MERGE_JACCARD = 0.5
LEVEL_ORDER = {"A": 0, "B": 1, "C": 2}

BULLET_ITEM_RE = re.compile(r"^\s*[-*]\s+(\S.*)")
PIPE_SPLIT_RE = re.compile(r"\s*\|\s*")


def option_jaccard(a: list, b: list) -> float:
    sa = {str(x).strip().lower() for x in a}
    sb = {str(x).strip().lower() for x in b}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _pipe_options(line: str) -> list[str]:
    """Options from a pipe-list line ("Pick: a | b | c.")."""
    parts = [p.strip().strip("\"'`") for p in PIPE_SPLIT_RE.split(line)]
    if parts and ":" in parts[0]:
        parts[0] = parts[0].rsplit(":", 1)[-1].strip()
    options = [p.strip(".,;:!?") for p in parts]
    return [o for o in options if o]


def text_candidates(hints: list[dict], text: str) -> list[dict]:
    """Text-hint candidates: pipe lists and bullet runs carry options."""
    lines = text.splitlines()
    out: list[dict] = []
    for h in hints or []:
        options: list[str] = []
        if h.get("match") == "a | b | c":
            options = _pipe_options(h.get("text", ""))
        elif str(h.get("match", "")).startswith("bullet list"):
            start = (h.get("line") or 1) - 1
            for line in lines[start:]:
                m = BULLET_ITEM_RE.match(line)
                if not m or len(line.strip()) >= 120:
                    break
                options.append(m.group(1).strip()[:120])
        if len(options) >= 2:
            out.append({"step": None, "options": options,
                        "evidence": {"hint": h.get("match"),
                                     "line": h.get("line"),
                                     "text": h.get("text", "")[:200]}})
    return out


def _norm_static(c: dict) -> dict:
    return {"signal": "code", "step": c.get("step"),
            "json_field": None, "options": list(c.get("options", [])),
            "location": c.get("location"), "consumption_rate": None,
            "evidence": c.get("evidence", {})}


def _norm_behaviour(c: dict) -> dict:
    return {"signal": "behaviour", "step": c.get("step"),
            "json_field": c.get("json_field"),
            "options": list(c.get("options", [])),
            "location": c.get("location"), "n_calls": c.get("n_calls"),
            "consumption_rate": c.get("consumption_rate"),
            "evidence": c.get("evidence", {})}


def _norm_text(c: dict) -> dict:
    return {"signal": "text", "step": None, "json_field": None,
            "options": list(c.get("options", [])), "location": None,
            "consumption_rate": None, "evidence": c.get("evidence", {})}


def _norm_agent(cand: dict, scored: dict) -> dict:
    draft = cand.get("draft", {}) if isinstance(cand, dict) else {}
    pid = draft.get("point_id") if isinstance(draft, dict) else None
    step = pid.rsplit(".", 1)[-1] if isinstance(pid, str) and pid else None
    options = list(draft.get("options", []) or [])
    scored_draft = scored.get("draft", {}) if isinstance(scored, dict) else {}
    return {"signal": "agent", "step": step, "json_field": None,
            "options": options, "location": None, "consumption_rate": None,
            "score": scored.get("score"), "verdict": scored.get("verdict"),
            "reasons": list(scored.get("reasons", [])),
            # R7=yes marks the scored draft irreversible; explicit True stays.
            "irreversible": bool(draft.get("irreversible", False)
                                 or scored_draft.get("irreversible", False)),
            "point_id": pid,
            "evidence": {"point_id": pid, "verdict": scored.get("verdict")}}


def _group_options(group: list[dict]) -> list[str]:
    seen: list[str] = []
    for m in group:
        for o in m["options"]:
            if o not in seen:
                seen.append(o)
    return seen


def _group_step(group: list[dict]) -> str | None:
    for m in group:
        if m["step"] is not None:
            return m["step"]
    return None


def _detectors(group: list[dict]) -> list[dict]:
    return [m for m in group if m["signal"] in ("code", "behaviour", "text")]


def _mergeable_by_jaccard(a: list[dict], b: list[dict]) -> bool:
    """Jaccard merge only when at least one side is stepless.

    Two stepped groups never merge by options alone: distinct steps with
    identical options (e.g. two yes/no calls) are distinct decisions.
    """
    sa, sb = _group_step(a), _group_step(b)
    if sa is not None and sb is not None:
        return False
    return option_jaccard(_group_options(a), _group_options(b)) >= MERGE_JACCARD


def _initial_groups(static: list[dict], behaviour: list[dict]) -> list[list[dict]]:
    # by step name (and json_field: one candidate per JSON enum field)
    keyed: dict = {}
    order: list = []
    for entry in ([_norm_static(c) for c in static]
                  + [_norm_behaviour(c) for c in behaviour]):
        key = (entry["step"], entry["json_field"])
        if entry["step"] is None:
            order.append([entry])
        elif key not in keyed:
            keyed[key] = [entry]
            order.append(keyed[key])
        else:
            keyed[key].append(entry)
    # ... else by option Jaccard (stepless sides only), best pair first
    while True:
        best = None
        for i in range(len(order)):
            for j in range(i + 1, len(order)):
                if not _mergeable_by_jaccard(order[i], order[j]):
                    continue
                jacc = option_jaccard(_group_options(order[i]),
                                      _group_options(order[j]))
                if best is None or jacc > best[0]:
                    best = (jacc, i, j)
        if best is None:
            return order
        _, i, j = best
        order[i] = order[i] + order[j]
        del order[j]


def _attach_text(groups: list[list[dict]], text: list[dict]) -> None:
    for c in text:
        entry = _norm_text(c)
        best, best_j = None, 0.0
        for g in groups:
            j = option_jaccard(entry["options"], _group_options(g))
            if j >= MERGE_JACCARD and j > best_j:
                best, best_j = g, j
        if best is not None:
            best.append(entry)
        # unmatched text candidates are dropped (legacy hints keep them)


def _attach_agent(groups: list[list[dict]], agent: list[dict]) -> None:
    for entry in agent:
        best, best_key = None, None
        for g in groups:
            step_hit = entry["step"] is not None \
                and entry["step"] == _group_step(g)
            j = option_jaccard(entry["options"], _group_options(g))
            opt_hit = j >= MERGE_JACCARD
            if not step_hit and not opt_hit:
                continue
            key = (0 if step_hit else 1, -(j if opt_hit else 0.0))
            if best_key is None or key < best_key:
                best, best_key = g, key
        if best is not None:
            best.append(entry)
        # unmatched agent candidates stay in the scored legacy points only


def _agree(members: list[dict]) -> bool:
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if option_jaccard(members[i]["options"],
                              members[j]["options"]) < MERGE_JACCARD:
                return False
    return True


def _build_candidate(group: list[dict]) -> dict:
    members = _detectors(group)
    agents = [m for m in group if m["signal"] == "agent"]
    present = [m["signal"] for m in members]
    agree = _agree(members)
    agent = agents[0] if agents else None
    if len(members) >= 2 and agree:
        level = "A"
        reason = f"level A: {'+'.join(present)} agree"
    elif agent is not None and (agree or len(members) == 1):
        level = "B"
        reason = f"level B: {'+'.join(present)} + agent candidate"
    else:
        level = "C"
        if len(members) >= 2:
            reason = f"level C: {'+'.join(present)} disagree on options"
        else:
            reason = f"level C: {'+'.join(present)} signal only"
    options: list[str] = []
    sources: dict[str, list[str]] = {}
    for m in members:
        for o in m["options"]:
            if o not in sources:
                sources[o] = []
                options.append(o)
            if m["signal"] not in sources[o]:
                sources[o].append(m["signal"])
    evidence: dict[str, list] = {"text": [], "code": [], "behaviour": []}
    for m in members:
        evidence[m["signal"]].append(m["evidence"])
    consumption = next((m["consumption_rate"] for m in members
                        if m["consumption_rate"] is not None), None)
    location = next((m["location"] for m in members
                     if m.get("location")), None)
    fields = {m["json_field"] for m in members if m.get("json_field")}
    reasons = [reason] + (agent["reasons"] if agent else [])
    return {
        "step": _group_step(group),
        "location": location,
        "json_field": next(iter(fields)) if len(fields) == 1 else None,
        "options": options,
        "option_sources": sources,
        "signals": {s: s in present for s in ("text", "code", "behaviour")},
        "signal_evidence": evidence,
        "level": level,
        "agree": agree,
        "score": agent["score"] if agent else None,
        "verdict": agent["verdict"] if agent else None,
        "reasons": reasons,
        "irreversible": agent["irreversible"] if agent else False,
        "consumption_rate": consumption,
    }


def merge(static: list[dict], behaviour: list[dict], text: list[dict],
          agent_scored: list[tuple]) -> list[dict]:
    """Merge detector candidates into ranked discovery candidates."""
    groups = _initial_groups(static, behaviour)
    _attach_text(groups, text)
    _attach_agent(groups, [_norm_agent(c, s) for c, s in agent_scored])
    out = [_build_candidate(g) for g in groups if _detectors(g)]
    out.sort(key=lambda c: (LEVEL_ORDER[c["level"]], c["score"] is None,
                            -(c["score"] or 0),
                            c["step"] or "", c["location"] or ""))
    return out
