"""do_export: JSONL of {state_text, options, label, label_source, weight} (§2.9)."""
from __future__ import annotations

import json
from pathlib import Path

from . import labels as L


def export_point(cfg: dict, store, point_id: str, path: str) -> dict:
    point = store.get_point(point_id)
    if point is None:
        from .errors import ToolError
        raise ToolError("unknown_point", f"unknown point_id {point_id!r}")
    rows = store.decisions_for_point(point_id)
    labeled = L.labeled_rows(rows, point, cfg["train"]["teacher_only_weight"],
                             cfg["extract"]["weight"])
    dest = Path(path)
    if dest.parent != Path(""):
        dest.parent.mkdir(parents=True, exist_ok=True)
    n = skipped = 0
    with open(dest, "w") as f:
        for row, label, source, weight in labeled:
            if not row.get("state_text"):
                skipped += 1
                continue
            f.write(json.dumps({"state_text": row["state_text"],
                                "options": list(point["options"]),
                                "label": label, "label_source": source,
                                "weight": weight}) + "\n")
            n += 1
    return {"point_id": point_id, "path": str(dest), "n": n, "skipped_no_text": skipped}
