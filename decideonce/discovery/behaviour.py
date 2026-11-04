"""Behaviour detector with consumption (M14 demo slice S2).

Wraps `analyze.detect_traces` (grouping and the enum test unchanged, run once
per group so each point maps exactly to its member calls), then keeps a group
as a behaviour candidate only if its output is consumed in at least
`discovery.min_consumed` of the runs where a later step exists: the
normalised output appears in a later step's input, or the later steps differ
by output value (a branch). Groups failing the enum test are reported as
rejected ("free text"); groups failing consumption as rejected ("not
consumed").
"""
from __future__ import annotations

from .. import analyze as AN

MIN_SUBSTRING_LEN = 2


def has_new_trace_fields(calls: list[dict]) -> bool:
    """True when any call carries the v2 trace fields (run_id/after)."""
    return any(isinstance(c, dict) and ("after" in c or "run_id" in c)
               for c in calls)


def _group_calls(calls: list[dict]) -> dict:
    groups: dict = {}
    for c in calls:
        if not isinstance(c, dict) or "output" not in c:
            continue
        groups.setdefault(AN.group_key(c), []).append(c)
    return groups


def _after_events(call: dict) -> list[dict]:
    after = call.get("after") or []
    if not isinstance(after, list):
        return []
    return [e for e in after if isinstance(e, dict)]


def _step_of(key) -> str | None:
    return key[2] if key[0] == "ws" else None


def _workflow_of(key) -> str | None:
    return key[1] if key[0] == "ws" else None


def _substring_hit(output, events: list[dict]) -> bool:
    v = AN.norm_output(output)
    if len(v) < MIN_SUBSTRING_LEN:
        return False
    for e in events:
        text = e.get("input")
        if isinstance(text, str) and v in text.lower():
            return True
    return False


def _consumption(members: list[dict]) -> tuple[int, int, list[dict]]:
    """Per-run consumption. Returns (consumed_runs, runs_with_later, runs).

    runs: [{run_id, value, later: [step names], hit}] in first-seen order.
    """
    by_run: dict = {}
    singletons = 0
    for m in members:
        rid = m.get("run_id")
        if rid is None:
            rid = f"_solo_{singletons}"
            singletons += 1
        by_run.setdefault(rid, []).append(m)
    runs = []
    for rid, ms in by_run.items():
        later = [e.get("step") for m in ms for e in _after_events(m)]
        value = AN.norm_output(ms[0]["output"])
        hit = any(_substring_hit(m["output"], _after_events(m)) for m in ms)
        runs.append({"run_id": rid, "value": value,
                     "later": [s for s in later if s is not None], "hit": hit})
    with_later = [r for r in runs if r["later"]]
    # branch: another run with a different output and different later steps
    for r in with_later:
        if r["hit"]:
            continue
        r_steps = sorted(set(r["later"]))
        for o in with_later:
            if o is not r and o["value"] != r["value"] \
                    and sorted(set(o["later"])) != r_steps:
                r["hit"] = True
                r["branch"] = True
                break
    consumed = sum(1 for r in with_later if r["hit"])
    return consumed, len(with_later), runs


def detect_behaviour(calls: list[dict], cfg: dict) -> dict:
    """Behaviour candidates and rejections over trace calls.

    Returns {"candidates": [...], "rejected": [...]}. A candidate carries the
    `detect_traces` point fields plus step/workflow/consumption_rate; a
    rejection carries step/workflow/reason/detail.
    """
    min_group = cfg["analyze"]["min_group"]
    min_consumed = cfg.get("discovery", {}).get("min_consumed", 0.8)
    candidates: list[dict] = []
    rejected: list[dict] = []
    groups = _group_calls(calls)
    for key in sorted(groups, key=str):
        members = groups[key]
        if len(members) < min_group:
            continue
        step, workflow = _step_of(key), _workflow_of(key)
        sub_points, _ = AN.detect_traces(members, cfg)
        if not sub_points:
            norms = [AN.norm_output(m["output"]) for m in members]
            coverage, _ = AN._top_coverage(norms)
            rejected.append({
                "step": step, "workflow": workflow,
                "location": None, "signal": "behaviour",
                "reason": "free text",
                "detail": (f"{len(members)} calls, top-16 cover "
                           f"{coverage:.0%} (< 90%), no JSON enum field"),
            })
            continue
        consumed, denom, _runs = _consumption(members)
        rate = consumed / denom if denom else 0.0
        if denom == 0 or rate < min_consumed:
            why = "no runs with later steps" if denom == 0 else (
                f"consumed in {consumed}/{denom} runs "
                f"({rate:.0%} < {min_consumed:.0%})")
            for pt in sub_points:
                rejected.append({
                    "step": step, "workflow": workflow,
                    "location": None, "signal": "behaviour",
                    "reason": "not consumed", "detail": why,
                    "json_field": pt["draft"].get("json_field"),
                    "consumption_rate": rate,
                })
            continue
        for pt in sub_points:
            candidates.append({
                "step": step, "workflow": workflow,
                "location": None, "json_field": pt["draft"].get("json_field"),
                "options": list(pt["draft"]["options"]),
                "consumption_rate": rate,
                "n_calls": pt["n_calls"],
                "est_monthly_calls": pt.get("est_monthly_calls"),
                "est_saving_usd": pt.get("est_saving_usd"),
                "reasons": list(pt["reasons"]),
                "evidence": {
                    "consumed_runs": consumed, "runs_with_later": denom,
                    "min_consumed": min_consumed,
                },
            })
    return {"candidates": candidates, "rejected": rejected}
