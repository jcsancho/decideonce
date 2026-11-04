"""Post-probe training helpers (M12 P3): the extraction gate, the
test-only fake-private stand-in, and model versioning. Local probe
training was deleted with the local probe; private-tier training runs
on Fireworks (trainer.py)."""
from __future__ import annotations

from .store import row_key as _row_key


def _norm_value(v):
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        return ("str", v[:40])
    return ("other", repr(v))


def _record_key(r: dict) -> tuple:
    """Canonical agreement identity: choice, at_start, decision event, and
    normalized context (§15.3 equality: numbers equal, strings' first 40)."""
    c = r.get("context") or {}
    return (r.get("choice"), bool(r.get("at_start")),
            (r.get("decision_pos") or {}).get("event"),
            tuple(sorted((k, _norm_value(v)) for k, v in c.items())))


def _records_agree(r1: dict | None, r2: dict | None) -> bool:
    """Single-record agreement (§15.3): same choice, same at_start, key
    context values equal, decision_pos in the same event. Every stored
    context field is key, not only required ones (B14)."""
    if (r1 is None) != (r2 is None):
        return False
    if r1 is None or r2 is None:
        return True
    return _record_key(r1) == _record_key(r2)


def extraction_gate(cfg: dict, store, point: dict) -> dict:
    """The §15.3 gate for a point: its extracted rows train only when all
    three legs hold (>= min_audited audited transcripts, agreement >=
    min_agreement, audited share >= 0.8 x audit_rate)."""
    wf = point["workflow"]
    n1 = n_both = agree = 0
    for t in store.transcripts_for_workflow(wf):
        if t.get("eval_only"):
            continue  # pilot arm runs never gate extraction quality
        exs = store.extractions_for_transcript(t["transcript_id"])
        p1 = any(e["pass"] == 1 and e["status"] == "completed" for e in exs)
        if not p1:
            continue
        n1 += 1
        if not any(e["pass"] == 2 and e["status"] == "completed" for e in exs):
            continue
        n_both += 1
        rows1 = [{
            "choice": r.get("extracted_choice"), "at_start": r.get("at_start"),
            "decision_pos": r.get("decision_pos"), "context": r.get("context"),
        } for r in store.decisions_for_transcript(t["transcript_id"])
            if r["point_id"] == point["point_id"] and r.get("source") == "extracted"]
        recs2 = [a["record"] for a in
                 store.audits_for_transcript(t["transcript_id"], point["point_id"])]
        if len(rows1) != len(recs2):
            continue
        if not rows1:
            # both passes found no decision: agreement only for repeatable
            # points (a run may genuinely hold none). For other points a
            # double skip is an extractor failure, not agreement (B14).
            agree += bool(point.get("repeatable"))
        elif len(rows1) == 1:
            agree += _records_agree(rows1[0], recs2[0])
        else:  # repeatable: full-identity multisets must match
            agree += sorted(_record_key(r) for r in rows1) == sorted(
                _record_key(r) for r in recs2)
    agreement = agree / n_both if n_both else None
    share = n_both / n1 if n1 else 1.0
    audited_ok = n_both >= cfg["extract"]["min_audited"]
    agree_ok = agreement is not None and agreement >= cfg["extract"]["min_agreement"]
    share_ok = share >= 0.8 * cfg["extract"]["audit_rate"]
    return {"audited": n_both, "n_pass1": n1, "agree": agree,
            "agreement": agreement, "share": share,
            "passed": bool(audited_ok and agree_ok and share_ok),
            "unreliable": bool(n_both >= cfg["extract"]["min_audited"]
                               and agreement is not None
                               and agreement < cfg["extract"]["min_agreement"])}


def train_fake_private(store, point_id: str, threshold: float = 0.7,
                     model_version: str = "fake-private-1") -> dict:
    """Test-only private-tier stand-in (H5): records the export marker (the
    latest decision) and a serve threshold as a kind="private" model. The
    inference probabilities come from the test's FakeDecider. M11 replaces
    this with the Fireworks LoRA job."""
    from .errors import ToolError  # local import: errors has no deps
    point = store.get_point(point_id)
    if point is None:
        raise ToolError("unknown_point", f"unknown point_id {point_id!r}")
    rows = store.decisions_for_point(point_id)
    latest = max(rows, key=_row_key, default=None)
    card = {"point_id": point_id,
            "point_version": point.get("point_version", 1),
            "kind": "private", "model_version": model_version,
            "threshold": threshold,
            "export_ts": latest.get("ts") or "" if latest else "",
            "export_decision_id": (latest.get("decision_id") or ""
                                   if latest else ""),
            "export_seq": latest.get("seq") if latest else None,
            "fake": True}
    store.save_model(point_id, point.get("point_version", 1), "private",
                     f"{model_version}.json", card)
    return {"point_id": point_id, "model_version": model_version,
            "card": card}


def _next_version(store, point_id: str, point_version: int) -> tuple[str, str]:
    dest = store.models_dir / point_id
    n = 0
    if dest.is_dir():
        for f in dest.glob(f"v{point_version}-*.npz"):
            try:
                n = max(n, int(f.stem.split("-")[1]))
            except (IndexError, ValueError):
                pass
    n += 1
    return f"v{point_version}-{n}", f"v{point_version}-{n}.npz"
