"""plain adapter: one text blob -> one transcript with a single text event."""
from __future__ import annotations

from . import TranscriptError


def adapt(source: str, run_id: str, workflow: str, model: str = "",
          data_class: str = "private", input: str = "",
          final_output: str = "") -> dict:
    if not isinstance(source, str) or not source.strip():
        raise TranscriptError("invalid_transcript", "plain transcript is empty")
    if not run_id or not workflow:
        raise TranscriptError("invalid_transcript", "plain adapt needs run_id and workflow")
    if data_class not in ("public", "private"):
        raise TranscriptError("invalid_transcript",
                              f"bad data_class {data_class!r}")
    return {
        "run_id": run_id,
        "workflow": workflow,
        "model": model,
        "data_class": data_class,
        "input": input,
        "events": [{"i": 0, "type": "text", "content": source}],
        "final_output": final_output,
    }
