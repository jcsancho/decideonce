"""Fit scoring, text hints and the trace detector for do_analyze (§3)."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime

from . import schemaval as SV

RUBRIC = [
    {"id": "R1",
     "question": "Is the output one option from a fixed list (≤ 16) or yes/no?",
     "effect": "no -> keep_llm (gate)"},
    {"id": "R2",
     "question": "Can the facts that decide it be listed as fields "
                 "(≤ 12 fields, text ≤ 400 chars each)?",
     "effect": "yes -> +R2; no -> keep_llm (gate)"},
    {"id": "R3",
     "question": "Is it rule matching, routing, triage, relevance or priority (vs. judging "
                 "correctness of code/maths or multi-step reasoning)?",
     "effect": "rule/routing/triage -> +R3; correctness -> keep_llm (gate)"},
    {"id": "R4",
     "question": "Expected volume per month (number)?",
     "effect": ">= 300 -> +R4_high; 50-299 -> +R4_mid; else 0"},
    {"id": "R5",
     "question": "Will the real outcome be observable later?",
     "effect": "yes -> +R5"},
    {"id": "R6",
     "question": "Can one small detail flip the answer (a date, an amount, a negation)? "
                 "Answer {sensitive, covered}: is that detail a field or a rule?",
     "effect": "sensitive and not covered -> +R6 (negative)"},
    {"id": "R7",
     "question": "Is the action irreversible (payment, delete, external email)?",
     "effect": "yes -> possible at best, irreversible: true"},
    {"id": "R8",
     "question": "How much does the agent reason before deciding today? "
                 "Answer {tokens, separate_call}.",
     "effect": "tokens > 1000 or a separate LLM call -> +R8"},
]


def candidate_schema() -> dict:
    return SV.load_schema("candidate")


def score_candidate(candidate: dict, cfg: dict) -> dict:
    """Score one candidate. Returns {draft, score, verdict, reasons[, errors]}."""
    draft = candidate.get("draft", {}) if isinstance(candidate, dict) else {}
    errors = SV.validate("candidate", candidate) if isinstance(candidate, dict) else ["not an object"]
    if errors:
        return {"draft": draft, "score": None, "verdict": "invalid",
                "reasons": [], "errors": errors[:5]}
    a = candidate["answers"]
    w = cfg["analyze"]["weights"]
    score, reasons = 0, []
    gate: str | None = None
    if a["R1"] == "no":
        gate = "R1: output is not one option from a fixed list"
    if a["R2"] == "yes":
        score += w["R2"]
        reasons.append(f"R2: +{w['R2']} (facts listable as fields)")
    else:
        gate = "R2: deciding facts cannot be listed as fields"
    if a["R3"] in ("rule", "routing", "triage"):
        score += w["R3"]
        reasons.append(f"R3: +{w['R3']} ({a['R3']})")
    elif a["R3"] == "correctness":
        gate = "R3: judging correctness needs reasoning"
    else:
        reasons.append(f"R3: +0 ({a['R3']})")
    vol = a["R4"]
    if vol >= 300:
        score += w["R4_high"]
        reasons.append(f"R4: +{w['R4_high']} ({vol}/month)")
    elif vol >= 50:
        score += w["R4_mid"]
        reasons.append(f"R4: +{w['R4_mid']} ({vol}/month)")
    else:
        reasons.append(f"R4: +0 ({vol}/month)")
    if a["R5"] == "yes":
        score += w["R5"]
        reasons.append(f"R5: +{w['R5']} (outcome observable)")
    if a["R6"]["sensitive"] and not a["R6"]["covered"]:
        score += w["R6"]
        reasons.append(f"R6: {w['R6']:+d} (flippable detail not covered by a field/rule)")
    irreversible = a["R7"] == "yes"
    if irreversible:
        reasons.append("R7: irreversible -> possible at best")
    r8 = a.get("R8") or {}
    if (r8.get("tokens") or 0) > 1000 or r8.get("separate_call"):
        score += w["R8"]
        reasons.append(f"R8: +{w['R8']} (heavy reasoning or separate call today)")
    if gate:
        return {"draft": draft, "score": score, "verdict": "keep_llm",
                "reasons": reasons + [f"gate: {gate}"]}
    if score >= cfg["analyze"]["good_fit"]:
        verdict = "good_fit"
    elif score >= cfg["analyze"]["possible"]:
        verdict = "possible"
    else:
        verdict = "keep_llm"
    if verdict == "good_fit" and irreversible:
        verdict = "possible"
    if irreversible:
        draft = {**draft, "irreversible": True}
    return {"draft": draft, "score": score, "verdict": verdict, "reasons": reasons}


# -- text hints (§2.1a) ----------------------------------------------------

KEYWORDS = ["classify", "choose", "decide", "one of", "whether", "route", "priority",
            "category", "approve", "escalate", "is this"]
PIPES_RE = re.compile(r"\w[\w .'-]{0,30}(\s*\|\s*\w[\w .'-]{0,30}){2,}")
BULLET_RE = re.compile(r"^\s*[-*]\s+\S")


def find_hints(text: str) -> list[dict]:
    """Deterministic keyword / enum-list hits with line numbers (1-based)."""
    hints: list[dict] = []
    lines = text.splitlines()
    for n, line in enumerate(lines, start=1):
        low = line.lower()
        for kw in KEYWORDS:
            if kw in low:
                hints.append({"line": n, "match": kw, "text": line.strip()[:200]})
                break
        if re.search(r"\bif\b", low) and re.search(r"\bthen\b", low):
            hints.append({"line": n, "match": "if … then", "text": line.strip()[:200]})
        if len(line) < 300 and PIPES_RE.search(line):
            hints.append({"line": n, "match": "a | b | c", "text": line.strip()[:200]})
    # bullet runs of 2-16 short items
    run: list[int] = []
    for n, line in enumerate(lines, start=1):
        if BULLET_RE.match(line) and len(line.strip()) < 120:
            run.append(n)
        else:
            if 2 <= len(run) <= 16:
                hints.append({"line": run[0], "match": f"bullet list ({len(run)} items)",
                              "text": lines[run[0] - 1].strip()[:200]})
            run = []
    if 2 <= len(run) <= 16:
        hints.append({"line": run[0], "match": f"bullet list ({len(run)} items)",
                      "text": lines[run[0] - 1].strip()[:200]})
    hints.sort(key=lambda h: (h["line"], h["match"]))
    return hints


# -- trace detector (§3.2) ---------------------------------------------------

UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                     r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
DQ_RE = re.compile(r'"[^"\n]{0,200}"')
SQ_RE = re.compile(r"'[^'\n]{0,200}'")
NUM_RE = re.compile(r"\d+")


def skeleton(prompt: str) -> str:
    s = UUID_RE.sub("<UUID>", prompt)
    s = EMAIL_RE.sub("<EMAIL>", s)
    s = DATE_RE.sub("<DATE>", s)
    s = DQ_RE.sub("<STR>", s)
    s = SQ_RE.sub("<STR>", s)
    return NUM_RE.sub("<N>", s)


def group_key(call: dict) -> tuple:
    if call.get("workflow") is not None and call.get("step") is not None:
        return ("ws", call["workflow"], call["step"])
    prompt = call.get("prompt", "")
    h = hashlib.sha256(skeleton(prompt)[:400].encode()).hexdigest()
    return ("tpl", h)


def norm_output(output) -> str:
    """Strip JSON wrapping, whitespace and punctuation; lowercase for grouping."""
    s = output
    if isinstance(s, str):
        t = s.strip()
        try:
            parsed = json.loads(t)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, str):
            t = parsed
        elif isinstance(parsed, dict) and len(parsed) == 1:
            v = next(iter(parsed.values()))
            if isinstance(v, (str, int, float, bool)):
                t = str(v)
        elif isinstance(parsed, list) and len(parsed) == 1 and isinstance(
                parsed[0], (str, int, float, bool)):
            t = str(parsed[0])
        s = t.strip().strip("\"'`.,;:!?()[]{}").strip().lower()
        return s
    return str(s).strip().lower()


def _top_coverage(values: list[str], k: int = 16) -> tuple[float, list[str]]:
    from collections import Counter
    counts = Counter(values)
    top = counts.most_common(k)
    covered = sum(c for _, c in top)
    return covered / len(values), [v for v, _ in top]


def _json_enum_fields(outputs: list) -> dict[str, list] | None:
    """If >= 90% parse as objects with only enum/bool/small-int differing fields,
    return {field: distinct values (most frequent first)} for the differing fields."""
    objs = []
    for o in outputs:
        if isinstance(o, dict):
            objs.append(o)
            continue
        if isinstance(o, str):
            try:
                p = json.loads(o.strip())
            except (ValueError, TypeError):
                return None
            if not isinstance(p, dict):
                return None
            objs.append(p)
        else:
            return None
    if len(objs) < 0.9 * len(outputs):
        return None
    fields: dict[str, list] = {}
    for o in objs:
        for k, v in o.items():
            fields.setdefault(k, []).append(v)
    out: dict[str, list] = {}
    for k, vals in fields.items():
        present = [v for v in vals if v is not None]
        if not present:
            continue
        if any(isinstance(v, (dict, list, float)) for v in present):
            return None
        if any(isinstance(v, bool) for v in present) and not all(
                isinstance(v, bool) for v in present):
            return None
        if any(isinstance(v, int) and not isinstance(v, bool) for v in present) and not all(
                isinstance(v, int) and not isinstance(v, bool) for v in present):
            return None
        if any(isinstance(v, str) for v in present) and not all(
                isinstance(v, str) for v in present):
            return None
        distinct = list(dict.fromkeys(present))  # order-preserving
        if len(distinct) <= 1:
            continue  # constant field
        if len(distinct) > 16:
            return None
        from collections import Counter
        out[k] = [v for v, _ in Counter(present).most_common()]
    return out


def detect_traces(calls: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """Group calls into candidate decision points. Returns (points, stats)."""
    from collections import Counter
    groups: dict[tuple, list[dict]] = {}
    for c in calls:
        if not isinstance(c, dict) or "output" not in c:
            continue
        groups.setdefault(group_key(c), []).append(c)
    min_group = cfg["analyze"]["min_group"]
    points: list[dict] = []
    for key, members in sorted(groups.items(), key=lambda kv: str(kv[0])):
        if len(members) < min_group:
            continue
        outputs = [m["output"] for m in members]
        enum_fields = _json_enum_fields(outputs)
        if enum_fields:
            for field, values in sorted(enum_fields.items()):
                points.append(_trace_point(key, members, [str(v) for v in values],
                                           field=field, coverage=1.0))
            continue
        norms = [norm_output(o) for o in outputs]
        coverage, top = _top_coverage(norms)
        if coverage < 0.90:  # top-16 values must cover >= 90% of outputs
            continue
        # surface forms: most frequent raw trimmed output per normalized value
        surfaces: dict[str, Counter] = {}
        for o, n in zip(outputs, norms):
            raw = o.strip()[:120] if isinstance(o, str) else str(o)
            surfaces.setdefault(n, Counter())[raw] += 1
        options = [surfaces[v].most_common(1)[0][0] for v in top]
        points.append(_trace_point(key, members, options, coverage=coverage))
    return points, {"groups": len(groups),
                    "groups_below_min": sum(1 for m in groups.values() if len(m) < min_group)}


def _trace_point(key, members, options, field=None, coverage=1.0) -> dict:
    draft: dict = {"kind": "choice", "options": options, "context_fields": {}}
    if field is not None:
        draft["json_field"] = field
    if key[0] == "ws":
        draft["workflow"], draft["question"] = key[1], f"Decide at step {key[2]}?"
    else:
        draft["question"] = "Decide (fill in from the prompt template)?"
    monthly = _monthly_calls(members)
    usds = sorted(m["usd"] for m in members if isinstance(m.get("usd"), (int, float)))
    saving = monthly * usds[len(usds) // 2] if monthly and usds else None
    reasons = [f"trace detector: {len(members)} calls, {len(options)} options, "
               f"coverage {coverage:.0%}"]
    if field is not None:
        reasons.append(f"JSON enum field: {field}")
    reasons.append("context fields left to the agent (verdict at most possible)")
    return {"draft": draft, "score": None, "verdict": "possible", "reasons": reasons,
            "n_calls": len(members), "est_monthly_calls": monthly,
            "est_saving_usd": saving}


def _monthly_calls(members: list[dict]) -> float | None:
    tss = []
    for m in members:
        ts = m.get("ts")
        if isinstance(ts, str):
            try:
                tss.append(datetime.fromisoformat(ts.replace("Z", "+00:00")))
            except ValueError:
                pass
    if len(set(tss)) < 2:
        return None
    span_days = (max(tss) - min(tss)).total_seconds() / 86400
    if span_days <= 0:
        return None
    return len(members) / span_days * 30
