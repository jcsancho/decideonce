"""Rendered export rows for the Fireworks decision recipe (GP3, §7.1).

One row per labelled decision: the cookbook payload (``state`` +
``questions``, one question per row) plus the row identity the pipeline
needs (point/ts/decision id/seq/run/split/label/weight/source). Only
``state``/``questions`` reach the train/dev/heldout files; the rest
stays in-process (file writing strips them, estimate ignores them).

Contract notes:
- criteria carry the FULL option list in REGISTERED order (dict order
  follows point["options"]); the recipe sorts criteria internally when
  it assigns option letters, so our order is a stable convention, not
  a letter assignment. noul points render as a 2-option choice.
- label/weight come from the §1.3 caller (labels.labeled_rows): the
  label is always a current option, the weight the precedence weight.
- state has no "input" key: decision rows do not store task_input.
- source "synthetic_gt" (no producer yet; a future synthetic import
  marks its rows with it) always splits to train, never to calib or
  heldout, so synthetic rows can train but never enter the evidence
  pool (T-FW1). Evidence scoring additionally admits only
  labels.LIVE_SOURCES.
"""
from __future__ import annotations

from . import render as R

#: Sources that split to train even when time order would hold them out.
TRAIN_ONLY_SOURCES = frozenset({"synthetic_gt"})


def render_export_row(point: dict, row: dict) -> dict:
    """Render one labelled row: identity keys plus state/questions.

    ``row`` is an upload-shaped row (point_id/ts/decision_id/seq/
    run_id/context/label/weight/source/split); ``point`` the registered
    point. Returns a new dict; the input is not mutated.
    """
    ctx = row.get("context")
    if not isinstance(ctx, dict):
        ctx = {}
    helps = point.get("option_help") or {}
    criteria = {opt: helps.get(opt, opt) for opt in point["options"]}
    qid = point["point_id"]
    return {**row,
            "state": R.render_state_obj(point, ctx),
            "questions": {qid: {
                "type": "choice",
                "instructions": point["question"],
                "criteria": criteria,
                "label": row["label"],
                "weight": row["weight"]}}}
