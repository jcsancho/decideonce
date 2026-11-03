"""jsonl adapter: one full §15.1 transcript object per non-blank line."""
from __future__ import annotations

import json

from . import TranscriptError


def adapt(source: str) -> list[dict]:
    if not isinstance(source, str) or not source.strip():
        raise TranscriptError("invalid_transcript", "jsonl transcript is empty")
    from . import validate_transcript
    out = []
    for i, line in enumerate(source.splitlines()):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            raise TranscriptError("invalid_transcript",
                                  f"line {i + 1} is not JSON") from None
        if not isinstance(obj, dict):
            raise TranscriptError("invalid_transcript",
                                  f"line {i + 1} is not an object")
        errors = validate_transcript(obj)
        if errors:
            raise TranscriptError("invalid_transcript",
                                  f"line {i + 1}: {errors[0]}")
        out.append(obj)
    if not out:
        raise TranscriptError("invalid_transcript", "jsonl transcript has no lines")
    return out
