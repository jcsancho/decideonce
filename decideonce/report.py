"""do_report: per-point metrics (JSON + Markdown). Never prints raw context values."""
from __future__ import annotations

import statistics
from collections import Counter

from . import gateway as GW
from . import labels as L
from . import probe as P
from . import stats as S


def _agreement(rows: list[dict], options: list[str], key: str) -> dict:
    """Teacher agreement of a stored distribution (jev_zero_shot or probe.probabilities)."""
    agree = n = 0
    per: dict[str, dict] = {o: {"agree": 0, "n": 0} for o in options}
    for r in rows:
        teacher = r.get("teacher_choice")
        dist = r.get(key) if key == "jev_zero_shot" else (r.get("probe") or {}).get("probabilities")
        if teacher is None or not dist:
            continue
        allowed = r.get("allowed_options") or options
        top = P.argmax(P.mask_and_renorm(dist, allowed), allowed)
        n += 1
        if top == teacher:
            agree += 1
        if teacher in per:
            per[teacher]["n"] += 1
            per[teacher]["agree"] += top == teacher
    return {"agreement": agree / n if n else None, "n": n,
            "per_option": {o: {"agreement": (v["agree"] / v["n"] if v["n"] else None),
                                 "n": v["n"]} for o, v in per.items()}}


def _still_needed(errors: int, n: int, target: float, cap: int = 5000) -> int | None:
    if S.wilson_upper(errors, n) <= target:
        return 0
    for x in range(1, cap + 1):
        if S.wilson_upper(errors, n + x) <= target:
            return x
    return None


def _pctile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    import math
    return sorted_vals[min(len(sorted_vals) - 1, max(0, math.ceil(q * len(sorted_vals)) - 1))]


def _gateway_section(cfg: dict, store, point: dict, rows: list[dict],
                     modes: Counter) -> dict:
    """Per-gateway-point block (§14.12). No raw text, only counts and rates."""
    pid = point["point_id"]
    n_auto = modes.get("auto", 0)
    jev_ms = sorted(r["jev_seconds"] * 1000 for r in rows
                    if r.get("jev_seconds") is not None)
    unav = store.counter_get("jev_unavailable", pid)
    unav_rate = unav / len(rows) if rows else None
    variants = store.counter_get("fingerprint_variant", pid)
    denom = len(rows) + variants
    fe = GW.format_evidence(cfg, store, point)
    fwd = [t for t in store.traces_for_point(pid, point.get("point_version", 1))
           if t.get("mode") == "forwarded"]
    toks = []
    for t in fwd:
        u = t.get("usage") or {}
        pi, co = u.get("prompt_tokens"), u.get("completion_tokens")
        if isinstance(pi, (int, float)) or isinstance(co, (int, float)):
            toks.append((pi or 0) + (co or 0))
    med_tok = statistics.median(toks) if toks else 0.0
    usds = [r["teacher_cost"]["usd"] for r in rows
            if isinstance((r.get("teacher_cost") or {}).get("usd"), (int, float))]
    med_usd = statistics.median(usds) if usds else 0.0
    overhead = sorted(t["latency_ms"] - t["upstream_ms"] for t in fwd
                      if t.get("latency_ms") is not None
                      and t.get("upstream_ms") is not None)
    auto_jev = sorted(r["jev_seconds"] * 1000 for r in rows
                      if r.get("mode") == "auto" and r.get("jev_seconds") is not None)
    return {
        "forwarded": modes.get("forwarded", 0),
        "auto": n_auto,
        "rule": modes.get("rule", 0),
        "template_match_rate_live": len(rows) / denom if denom else None,
        "fingerprint_variant": variants,
        "unparsed": sum(1 for r in rows if r.get("unparsed")),
        "missing_guard_field": store.counter_get("missing_guard_field", pid),
        "missing_required_field": store.counter_get("missing_required_field", pid),
        "jev_unavailable": {"n": unav, "rate": unav_rate,
                            "warning": unav_rate is not None and unav_rate > 0.10},
        "jev_latency_ms": {"p50": _pctile(jev_ms, 0.50), "p95": _pctile(jev_ms, 0.95),
                           "n": len(jev_ms)},
        "format": fe,
        "avoided": {"basis": "estimate", "n_auto": n_auto,
                    "tokens_per_call_median": med_tok, "tokens": med_tok * n_auto,
                    "usd_per_call_median": med_usd, "usd": med_usd * n_auto,
                    "n_usage": len(toks), "n_costs": len(usds)},
        "added_latency_ms": {"forwarded_overhead_p50": _pctile(overhead, 0.50),
                             "forwarded_overhead_n": len(overhead),
                             "auto_jev_p50": _pctile(auto_jev, 0.50),
                             "auto_jev_n": len(auto_jev)},
    }


def point_report(cfg: dict, store, point: dict, serve_checks: dict | None = None) -> dict:
    options = list(point["options"])
    rows = store.decisions_for_point(point["point_id"])
    modes = Counter(r.get("mode") for r in rows)
    sources = Counter(r.get("source") for r in rows)
    labeled = L.labeled_rows(rows, point, cfg["train"]["teacher_only_weight"],
                             cfg["extract"]["weight"])
    label_sources = Counter(src for _, _, src, _ in labeled)
    unlabeled = len(rows) - len(labeled) - sum(
        1 for r in rows if L.resolve_label(r)[1] == "outcome_bad_no_fix")
    excluded_bad = sum(1 for r in rows if L.resolve_label(r)[1] == "outcome_bad_no_fix")

    model = store.active_model(point["point_id"], "probe")
    card = model["card"] if model else {}
    threshold = card.get("threshold")

    audits = [r for r in rows if r.get("audit") and r.get("teacher_choice")]
    audit_errs = 0
    for r in audits:
        probs = (r.get("probe") or {}).get("probabilities") or r.get("jev_zero_shot") or {}
        if probs and P.argmax(P.mask_and_renorm(probs, r.get("allowed_options") or options),
                              r.get("allowed_options") or options) != r["teacher_choice"]:
            audit_errs += 1

    costs = [r["teacher_cost"] for r in rows if r.get("teacher_cost")]
    medians: dict[str, float] = {}
    for k in ("tokens_in", "tokens_out", "usd", "seconds"):
        vals = [c[k] for c in costs if isinstance(c.get(k), (int, float))]
        if vals:
            medians[k] = statistics.median(vals)
    n_auto = modes.get("auto", 0)
    saved = {k: medians[k] * n_auto for k in medians}
    basis = "measured" if len(costs) >= 20 else "estimate"

    rule_hits: Counter = Counter()
    for r in rows:
        rule_hits.update(r.get("rule_hits") or [])

    both, disagree = _teacher_disagree(rows)
    unreliable = both > 0 and disagree / both > cfg["train"]["teacher_alarm"]

    window = cfg["serve"]["drift_window"]
    recent = audits[-window:]
    drift_events = [e for e in point.get("events", []) if e.get("event") == "drift_demotion"]

    target = point.get("target_error", cfg["serve"]["target_error"])
    pool_n = int(card.get("n_at_threshold") or 0)
    pool_e = int(round(float(card.get("error_at_threshold") or 0.0) * pool_n)) if pool_n else 0
    gw = (_gateway_section(cfg, store, point, rows, modes)
          if point.get("source") == "gateway" else None)
    rep = {
        "point_id": point["point_id"],
        "workflow": point["workflow"],
        "kind": point["kind"],
        "phase": point.get("phase", "collect"),
        "point_version": point.get("point_version", 1),
        "gateway": gw,
        "options": options,
        "target_error": target,
        "data_class": point.get("data_class", "private"),
        "irreversible": bool(point.get("irreversible", False)),
        "rows": len(rows),
        "rows_by_mode": dict(modes),
        "rows_by_source": dict(sources),
        "label_sources": dict(label_sources),
        "unlabeled": unlabeled,
        "excluded_outcome_bad": excluded_bad,
        "zero_shot": _agreement(rows, options, "jev_zero_shot"),
        "probe": ({**_agreement(rows, options, "probe"),
                   "model_version": card.get("model_version")} if model else None),
        "threshold": threshold,
        "coverage_heldout": card.get("coverage"),
        "teacher_only_options": card.get("teacher_only_options", []),
        "auto": {"n": n_auto, "share": n_auto / len(rows) if rows else 0.0},
        "audit_error": {"errors": audit_errs, "n": len(audits),
                        "rate": audit_errs / len(audits) if audits else None,
                        "wilson_upper": S.wilson_upper(audit_errs, len(audits))
                        if audits else None},
        "overrides": sum(1 for r in rows if r.get("override")),
        "rule_hits": dict(rule_hits),
        "rule_violations": sum(1 for r in rows if r.get("rule_violation")),
        "rule_violations_in_run": sum(1 for r in rows if r.get("rule_violation_in_run")),
        "teacher_unreliable": unreliable,
        "teacher_disagreement": {"disagree": disagree, "n": both},
        "teacher_cost": {"n": len(costs), "median": medians},
        "saved": {"tokens_in": saved.get("tokens_in", 0.0),
                  "tokens_out": saved.get("tokens_out", 0.0),
                  "usd": saved.get("usd", 0.0),
                  "seconds": saved.get("seconds", 0.0),
                  "basis": basis, "n_auto": n_auto, "n_costs": len(costs)},
        "drift": {"status": ("demoted_to_shadow" if drift_events and point.get("phase") == "shadow"
                             else "ok"),
                  "demotions": len(drift_events),
                  "recent_audits": {"errors": sum(
                      1 for r in recent
                      if ((r.get("probe") or {}).get("probabilities") and P.argmax(
                          P.mask_and_renorm((r["probe"])["probabilities"],
                                            r.get("allowed_options") or options),
                          r.get("allowed_options") or options) != r["teacher_choice"])),
                      "n": len(recent), "window": window}},
        "evidence_pool": {"n": pool_n, "errors": pool_e,
                          "wilson_upper": S.wilson_upper(pool_e, pool_n) if pool_n else None,
                          "still_needed": _still_needed(pool_e, pool_n, target)},
    }
    if serve_checks is not None:
        rep["promotion"] = {"serve_ready": all(v["pass"] for v in serve_checks.values()),
                            "checks": serve_checks}
    return rep


def _teacher_disagree(rows: list[dict]) -> tuple[int, int]:
    return L.teacher_disagreement(rows)


def render_autopilot_md(block: dict) -> str:
    """The Autopilot block at the top of md reports (M16). Pure."""
    state = "on" if block.get("enabled") and block.get("match") else (
        "not_approved" if block.get("enabled") else "off")
    lines = ["# Autopilot", "", f"- state: {state}"]
    ap = (block.get("approved_hash") or "none")[:12]
    cu = (block.get("current_hash") or "none")[:12]
    lines.append(f"- approved: {ap} | current: {cu}")
    for a in block.get("actions", []):
        lines.append(f"- {a.get('ts', '')} {a.get('line', '')}")
    lines.append("")
    return "\n".join(lines)


def render_md(reports: list[dict]) -> str:
    lines = ["# decideonce report", ""]
    for rep in reports:
        lines.append(f"## {rep['point_id']} ({rep['phase']}, v{rep['point_version']})")
        lines.append("")
        if rep.get("gateway"):
            g = rep["gateway"]
            ju, jl = g["jev_unavailable"], g["jev_latency_ms"]
            lines.append(f"- gateway: forwarded {g['forwarded']}, auto {g['auto']}, "
                         f"rule {g['rule']} | template match (live) "
                         f"{_pct(g['template_match_rate_live'])} | variants "
                         f"{g['fingerprint_variant']} | unparsed {g['unparsed']} | "
                         f"missing guard/required {g['missing_guard_field']}/"
                         f"{g['missing_required_field']}")
            lines.append(f"- jev_unavailable: {ju['n']} ({_pct(ju['rate'])}) | JEV "
                         f"latency p50 {_ms(jl['p50'])}, p95 {_ms(jl['p95'])} "
                         f"(n={jl['n']})"
                         + (" **WARNING: above 10% — the gateway may silently "
                            "never serve**" if ju["warning"] else ""))
            f = g["format"]
            lines.append(f"- format: {f['n_match']}/{f['n_compared']} match "
                         f"(Wilson lower {_pct(f['wilson_lower'])}) | unparsed "
                         f"{f['n_unparsed']}/{f['n_answered']} "
                         f"({_pct(f['unparsed_rate'])})")
            av = g["avoided"]
            lines.append(f"- avoided ({av['basis']}, {av['n_auto']} auto × median of "
                         f"{av['n_usage']} usage reports): {av['tokens']:.0f} tokens, "
                         f"${av['usd']:.2f}")
            al = g["added_latency_ms"]
            lines.append(f"- added latency: forwarded overhead p50 {_ms(al['forwarded_overhead_p50'])} "
                         f"(n={al['forwarded_overhead_n']}), auto JEV p50 "
                         f"{_ms(al['auto_jev_p50'])} (n={al['auto_jev_n']})")
        lines.append(f"- workflow: {rep['workflow']} | kind: {rep['kind']} | "
                     f"target error: {rep['target_error']}")
        if rep.get("tier"):
            lines.append(f"- tier: {rep['tier']}")
        h = rep.get("hosted")
        if h:
            lines.append(
                f"- hosted: {h['calls_month']} calls, ${h['spend_month_usd']:.4f} / "
                f"${h['budget_usd']:.2f} this month"
                + (" **WARNING: over 80% of budget**" if h["budget_warning_80"] else "")
                + f" | median latency {_s(h['median_latency_s'])} | errors "
                + ", ".join(f"{k} {v}" for k, v in sorted(h["errors"].items())))
            lines.append(f"  ({h['spend_note']})")
            f = h["fit"]
            if f:
                lines.append(f"- hosted fit: T={f['temperature']:.3f}, threshold="
                             f"{f['threshold']}, coverage={f['coverage']:.3f}, "
                             f"teacher_only={f['teacher_only_options']} "
                             f"(n_fit={f['n_fit']})")
            else:
                fp = h["fit_progress"]
                lines.append(f"- hosted fit: pre-fit ({fp['labelled']}/{fp['fit_rows']} labelled)")
            ll = h["late_labels_ignored"]
            lines.append(f"- hosted evidence: {h['evidence_rows']} rows"
                         + (f", ~{h['still_needed']} more error-free cases needed"
                            if h["still_needed"] else "")
                         + f" | late ignored: {ll['n']} (err {_pct(ll['error_rate'])})"
                         + (" **late_label_bias**" if h["late_label_bias"] else ""))
            lines.append(f"- train_recommended: {h['train_recommended']}"
                         + (f" ({', '.join(h['train_reasons'])})" if h["train_reasons"] else ""))
            pc = h.get("private_comparison")
            if pc:
                lines.append(f"- private vs hosted on C: n={pc['n']}, private cov="
                             f"{pc['private']['coverage']:.3f} err={pc['private']['error']}, hosted cov="
                             f"{pc['hosted']['coverage']:.3f} err={pc['hosted']['error']}, discordant "
                             f"{pc['discordant_hosted_only']}/{pc['discordant_private_only']}, McNemar p="
                             f"{pc['mcnemar_p']:.4f}")
            pf = h.get("plan_fit") or {}
            if pf.get("fitted"):
                lines.append(f"- hosted plan fit: threshold={pf['threshold']} "
                             f"(n_fit={pf['n_fit']}, kinds={pf['fit_label_kinds']}); "
                             f"evidence {pf['evidence_comparisons']} "
                             f"(kinds={pf['evidence_label_kinds']})")
        lines.append(f"- rows: {rep['rows']} (modes: "
                     + ", ".join(f"{k} {v}" for k, v in sorted(rep["rows_by_mode"].items()))
                     + ")")
        lines.append("- label sources: "
                     + (", ".join(f"{k} {v}" for k, v in sorted(rep["label_sources"].items()))
                        or "none"))
        zs, pr = rep["zero_shot"], rep["probe"]
        lines.append(f"- teacher agreement: zero-shot "
                     f"{_pct(zs['agreement'])} (n={zs['n']})"
                     + (f", probe {_pct(pr['agreement'])} (n={pr['n']})" if pr else ", probe: none"))
        if pr:
            per = ", ".join(f"{o} {_pct(v['agreement'])}" for o, v in pr["per_option"].items())
            lines.append(f"  - per option (probe): {per}")
        lines.append(f"- threshold: {rep['threshold']} | coverage (held-out): "
                     f"{rep['coverage_heldout']} | auto: {rep['auto']['n']} "
                     f"({rep['auto']['share']:.0%})")
        ae = rep["audit_error"]
        lines.append(f"- audit error: {ae['errors']}/{ae['n']} "
                     f"({_pct(ae['rate'])}; Wilson upper {_pct(ae['wilson_upper'])})")
        lines.append(f"- overrides: {rep['overrides']} | rule hits: {rep['rule_hits']} | "
                     f"rule violations: {rep['rule_violations']}")
        if rep["teacher_unreliable"]:
            lines.append(f"- **teacher_unreliable**: {rep['teacher_disagreement']}")
        sv = rep["saved"]
        lines.append(f"- saved ({sv['basis']}, {sv['n_auto']} auto × median of "
                     f"{sv['n_costs']} reported costs): ${sv['usd']:.2f}, "
                     f"{sv['tokens_in']:.0f} in / {sv['tokens_out']:.0f} out tokens, "
                     f"{sv['seconds']:.1f}s")
        lines.append(f"- drift: {rep['drift']['status']} "
                     f"(demotions: {rep['drift']['demotions']})")
        ep = rep["evidence_pool"]
        lines.append(f"- evidence for serve: n={ep['n']}, errors={ep['errors']}, "
                     f"Wilson upper={_pct(ep['wilson_upper'])}"
                     + (f", ~{ep['still_needed']} more error-free cases needed"
                        if ep["still_needed"] else ""))
        if "promotion" in rep:
            failed = [k for k, v in rep["promotion"]["checks"].items() if not v["pass"]]
            lines.append("- serve promotion: "
                         + ("READY" if rep["promotion"]["serve_ready"]
                            else f"blocked on {', '.join(failed)}"))
        lines.append("")
    return "\n".join(lines)


def _pct(x) -> str:
    return "n/a" if x is None else f"{x:.1%}"


def _s(x) -> str:
    return "n/a" if x is None else f"{x:.3f}s"


def _ms(x) -> str:
    return "n/a" if x is None else f"{x:.1f}ms"


def _mean(vals: list[float]) -> float | None:
    return sum(vals) / len(vals) if vals else None


def workflow_report(cfg: dict, store, workflow: str) -> dict:
    """One-turn workflow report (§15.8): extraction, plans, trial, costs."""
    from . import train as TR
    points = store.list_points(workflow)
    transcripts = store.transcripts_for_workflow(workflow)
    n_pass1 = n_pass2 = 0
    rejected: Counter = Counter()
    disagreements: list[dict] = []
    for t in transcripts:
        exs = store.extractions_for_transcript(t["transcript_id"])
        if any(e["pass"] == 1 and e["status"] == "completed" for e in exs):
            n_pass1 += 1
        if any(e["pass"] == 2 and e["status"] == "completed" for e in exs):
            n_pass2 += 1
        for e in exs:
            full = store.get_extraction(e["extraction_id"])
            if full is None:
                continue
            rejected.update(full.get("rejected_codes") or {})
            disagreements.extend(full.get("disagreements") or [])
    gates = {p["point_id"]: TR.extraction_gate(cfg, store, p) for p in points}
    cards = {}
    for p in points:
        model = (store.active_model(p["point_id"], "plan")
                 or store.active_model(p["point_id"], "probe"))
        cards[p["point_id"]] = model["card"] if model else {}

    plans = store.plans_for_workflow(workflow)
    real = [p for p in plans if not p.get("pilot_forced")]
    pilot = [p for p in plans if p.get("pilot_forced")]
    singles = [p for p in real if not p.get("pair_id")]
    followed = [p["followed_rate"] for p in real
                if p.get("followed_rate") is not None]
    audits = [p for p in real if p.get("audit")]
    drift = [e for p in points for e in p.get("events", [])
             if e.get("event") == "plan_drift_demotion"]

    by_pair: dict[str, dict] = {}
    for p in real:
        if p.get("pair_id"):
            by_pair.setdefault(p["pair_id"], {})[p["arm"]] = p
    trial_pairs = {k: v for k, v in by_pair.items()
                   if list(v.values())[0].get("trial_id")}
    mon_pairs = {k: v for k, v in by_pair.items()
                 if not list(v.values())[0].get("trial_id")}

    def arm_means(pairs: dict) -> dict:
        out = {}
        for arm in ("expensive", "cheap"):
            vals = [a["cost"]["usd"] for a in
                    (arms.get(arm) or {} for arms in pairs.values())
                    if isinstance((a.get("cost") or {}).get("usd"), (int, float))]
            out[arm] = {"n": len(vals), "mean_usd": _mean(vals)}
        return out

    trial_costs = arm_means(trial_pairs)
    mon_costs = arm_means(mon_pairs)
    exp_vals = [a["cost"]["usd"]
                for arms in list(trial_pairs.values()) + list(mon_pairs.values())
                for a in (arms.get("expensive") or {},)
                if isinstance((a.get("cost") or {}).get("usd"), (int, float))]
    chp_vals = [a["cost"]["usd"]
                for arms in list(trial_pairs.values()) + list(mon_pairs.values())
                for a in (arms.get("cheap") or {},)
                if isinstance((a.get("cost") or {}).get("usd"), (int, float))]

    def _usd(p: dict) -> float:
        c = p.get("cost") or {}
        return c.get("usd") if isinstance(c.get("usd"), (int, float)) else 0.0

    overhead = (sum(_usd(p) for p in audits)
                + sum(_usd(a) for arms in list(trial_pairs.values())
                      + list(mon_pairs.values()) for a in arms.values()))
    downgraded = sum(1 for p in singles if p["recommendation"] == "downgrade")
    mean_exp, mean_chp = _mean(exp_vals), _mean(chp_vals)
    savings = ((downgraded * (mean_exp - mean_chp) - overhead)
               if mean_exp is not None and mean_chp is not None else None)

    complete = []
    for pair_id, arms in mon_pairs.items():
        if "expensive" not in arms or "cheap" not in arms:
            continue
        e = (arms["expensive"].get("check_result") or {}).get("pass")
        c = (arms["cheap"].get("check_result") or {}).get("pass")
        if e is None or c is None:
            continue
        complete.append((min(arms["expensive"]["ts"], arms["cheap"]["ts"]),
                         pair_id, bool(e), bool(c)))
    complete.sort()
    window = complete[-cfg["plan"]["monitor_window"]:]
    monitor: dict = {"pairs": len(window), "status": "insufficient_data",
                       "bound": None, "upper": None, "diff": None}
    if len(window) >= cfg["plan"]["monitor_min_pairs"]:
        a = sum(1 for _, _, e, c in window if e and c)
        b = sum(1 for _, _, e, c in window if c and not e)
        c = sum(1 for _, _, e, c in window if e and not c)
        d = sum(1 for _, _, e, c in window if not e and not c)
        diff, lower, upper = S.newcombe_paired(a, b, c, d, 1.96)
        monitor.update(bound=lower, upper=upper, diff=diff,
                       counts={"a": a, "b": b, "c": c, "d": d})
        max_drop = float(cfg["plan"]["max_quality_drop"])
        if upper < -max_drop:
            monitor["status"] = "revoke"
        elif len(window) >= cfg["plan"]["monitor_window"] and lower < -max_drop:
            monitor["status"] = "non_inferiority_not_shown"
        else:
            monitor["status"] = "ok"

    wrec = store.get_workflow(workflow) or {}
    return {
        "workflow": workflow,
        "extraction": {
            "transcripts": len(transcripts),
            "pass1_completed": n_pass1,
            "pass2_completed": n_pass2,
            "rejected_by_code": dict(rejected),
            "agreement": {pid: {"audited": g["audited"],
                                "agreement": g["agreement"],
                                "share": g["share"], "passed": g["passed"]}
                          for pid, g in gates.items()},
            "jev_disagreements": disagreements,
        },
        "plans": {
            "recommendations": dict(Counter(
                p["recommendation"] for p in singles)),
            "followed_rate_mean": _mean(followed),
            "followed_n": len(followed),
            "conflicts": sum(len(p.get("conflicts", [])) for p in real),
            "audits": len(audits),
            "drift_demotions": len(drift),
            "pilot_forced": len(pilot),
            "pilot_note": ("pilot, not promoted" if pilot else None),
        },
        "trial": (wrec.get("trial") or None),
        "downgrade_allowed": wrec.get("downgrade_allowed", False),
        "monitor": monitor,
        "costs": {
            "trial_arms": trial_costs,
            "monitor_arms": mon_costs,
            "overhead_usd": overhead,
            "downgraded_tasks": downgraded,
            "savings_usd": savings,
        },
        "points": {p["point_id"]: {
            "plan_phase": p.get("plan_phase", "collect"),
            "plan_eligible": (cards[p["point_id"]].get("plan_eligible")
                              if cards[p["point_id"]] else None),
            "agreement": (cards[p["point_id"]].get("agreement")
                          if cards[p["point_id"]] else None),
            "correctness": (cards[p["point_id"]].get("correctness")
                            if cards[p["point_id"]] else None),
        } for p in points},
    }


def render_workflow_md(rep: dict) -> str:
    lines = [f"# one-turn report: {rep['workflow']}", ""]
    ex = rep["extraction"]
    lines.append(f"- extraction: {ex['transcripts']} transcripts, "
                 f"{ex['pass1_completed']} pass-1, {ex['pass2_completed']} "
                 "pass-2")
    lines.append("- rejections by code: "
                 + (", ".join(f"{k} {v}" for k, v in
                              sorted(ex["rejected_by_code"].items())) or "none"))
    for pid, g in ex["agreement"].items():
        lines.append(f"- {pid}: audited {g['audited']}, agreement "
                     f"{_pct(g['agreement'])}, share {_pct(g['share'])} "
                     f"({'pass' if g['passed'] else 'excluded'})")
    lines.append(f"- jev_disagreements: {len(ex['jev_disagreements'])}")
    pl = rep["plans"]
    lines.append("- recommendations: "
                 + (", ".join(f"{k} {v}" for k, v in
                              sorted(pl["recommendations"].items())) or "none"))
    lines.append(f"- followed_rate: {_pct(pl['followed_rate_mean'])} "
                 f"(n={pl['followed_n']}) | conflicts: {pl['conflicts']} | "
                 f"audits: {pl['audits']} | drift demotions: "
                 f"{pl['drift_demotions']}")
    if pl["pilot_forced"]:
        lines.append(f"- pilot plans: {pl['pilot_forced']} "
                     f"({pl['pilot_note']})")
    trial = rep["trial"] or {}
    lines.append(f"- trial: {trial.get('status', 'none')}"
                 + (f" (n={trial.get('n')}, bound={trial.get('bound')})"
                    if trial else ""))
    lines.append(f"- downgrade_allowed: {rep['downgrade_allowed']}")
    mo = rep["monitor"]
    lines.append(f"- monitor: {mo['status']} (pairs={mo['pairs']})")
    co = rep["costs"]
    for name in ("trial_arms", "monitor_arms"):
        arms = co[name]
        lines.append(f"- {name}: expensive n={arms['expensive']['n']} "
                     f"mean ${arms['expensive']['mean_usd']} | cheap "
                     f"n={arms['cheap']['n']} mean ${arms['cheap']['mean_usd']}")
    lines.append(f"- savings: ${co['savings_usd']} "
                 f"({co['downgraded_tasks']} downgraded, overhead "
                 f"${co['overhead_usd']})")
    for pid, p in rep["points"].items():
        lines.append(f"- {pid}: plan_phase={p['plan_phase']}, eligible="
                     f"{p['plan_eligible']}, agreement={_pct(p['agreement'])}, "
                     f"correctness={_pct(p['correctness'])}")
    return "\n".join(lines)
