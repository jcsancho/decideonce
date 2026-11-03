"""Transcript adapters (§15.1): plain text, JSONL, and muse exec --json.

Each adapter builds §15.1 transcript dicts (run_id, workflow, model, input,
events, final_output, ...). The skill/pilot adapts raw logs, then submits the
transcript object to do_extract call 1.
"""
from __future__ import annotations


class TranscriptError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


from . import jsonl as _jsonl  # noqa: E402
from . import muse as _muse  # noqa: E402
from . import plain as _plain  # noqa: E402

FORMATS = ("plain", "jsonl", "muse")


def adapt_plain(source: str, run_id: str, workflow: str, model: str = "",
                data_class: str = "private", input: str = "",
                final_output: str = "") -> dict:
    """One text blob -> one transcript with a single text event."""
    return _plain.adapt(source, run_id, workflow, model, data_class, input,
                        final_output)


def adapt_jsonl(source: str) -> list[dict]:
    """One full §15.1 transcript object per non-blank line."""
    return _jsonl.adapt(source)


def adapt_muse(source: str, workflow: str, run_id: str | None = None,
               model: str | None = None, data_class: str = "private") -> dict:
    """`muse exec --json` JSONL -> one transcript."""
    return _muse.adapt(source, workflow, run_id, model, data_class)


def validate_transcript(t: dict) -> list[str]:
    """Structural validation against schemas/transcript.json (+ index order)."""
    from .. import schemaval as SV
    errors = SV.validate("transcript", t) if isinstance(t, dict) else ["<root>: not an object"]
    if errors or not isinstance(t, dict):
        return errors
    for n, e in enumerate(t.get("events", [])):
        if e.get("i") != n:
            errors.append(f"events[{n}]: i should be {n}")
    return errors
