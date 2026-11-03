"""Training labels: precedence (§1.3) and row selection for training."""
from __future__ import annotations

# Row sources that carry live teacher labels and may count as promotion
# evidence. Extracted rows may train (and fit thresholds) but are never
# §6.4 evidence (§15.4, T-EX10).
LIVE_SOURCES = ("do_decide", "dm_decide", "gateway")  # dm_ kept for pre-M15 rows


def resolve_label(row: dict, teacher_weight: float = 1.0,
                  extract_weight: float = 0.5) -> tuple[str | None, str, float]:
    """Returns (label, source, weight). source is one of outcome_correct, outcome_ok,
    outcome_bad_no_fix (excluded), teacher, extracted, no_final_choice, unlabeled,
    unparsed (excluded: the teacher answer never parsed into an option, §14.7)."""
    if row.get("unparsed"):
        return None, "unparsed", 0.0
    outcome = row.get("outcome") or {}
    correct = outcome.get("correct_choice")
    if correct:
        return correct, "outcome_correct", 1.0
    ok = outcome.get("ok")
    if ok is True:
        final = (row.get("final_choice") or row.get("teacher_choice")
                 or row.get("extracted_choice"))
        if final:
            return final, "outcome_ok", 1.0
        return None, "no_final_choice", 0.0
    if ok is False:
        return None, "outcome_bad_no_fix", 0.0
    if row.get("teacher_choice"):
        return row["teacher_choice"], "teacher", teacher_weight
    if row.get("extracted_choice"):
        return row["extracted_choice"], "extracted", extract_weight
    return None, "unlabeled", 0.0


def hosted_top(row: dict) -> str | None:
    """Masked top of the stored hosted-Jev probabilities (None when absent).

    Mirrors the shown top: tempering never changes the argmax, so the
    untempered masked top is what the teacher would have followed.
    """
    probs = (row.get("hosted") or {}).get("probabilities") or {}
    allowed = row.get("allowed_options") or list(probs)
    if not probs or not allowed:
        return None
    return max(allowed, key=lambda o: (probs.get(o, 0.0),
                                       -allowed.index(o)))


def teacher_followed_hosted_top(row: dict) -> bool:
    """True when the teacher's choice equals hosted Jev's shown top."""
    teacher = row.get("teacher_choice")
    return teacher is not None and teacher == hosted_top(row)


def teacher_disagreement(rows: list[dict]) -> tuple[int, int]:
    """(n_both, n_disagree) over rows with a teacher choice and an outcome (§1.3)."""
    both = disagree = 0
    for r in rows:
        t = r.get("teacher_choice")
        oc = r.get("outcome") or {}
        if t is None or (oc.get("ok") is None and not oc.get("correct_choice")):
            continue
        both += 1
        cc = oc.get("correct_choice")
        if cc is not None:
            if cc != t:
                disagree += 1
        elif oc.get("ok") is False:
            disagree += 1
    return both, disagree


def is_training_row(row: dict, point: dict) -> bool:
    """Rows of older versions are usable only when the option list is identical."""
    return row.get("options") == list(point["options"])


def labeled_rows(rows: list[dict], point: dict, teacher_weight: float = 1.0,
                 extract_weight: float = 0.5,
                 ) -> list[tuple[dict, str, str, float]]:
    """(row, label, source, weight) for rows usable in training (option list identical,
    label present, label in current options). Rows must already be in time order."""
    out = []
    options = set(point["options"])
    for row in rows:
        if not is_training_row(row, point):
            continue
        label, source, weight = resolve_label(row, teacher_weight, extract_weight)
        if label is None or label not in options:
            continue
        out.append((row, label, source, weight))
    return out
