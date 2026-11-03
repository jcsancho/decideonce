"""muse adapter: `muse exec --json` JSONL onto a §15.1 transcript.

Written against real captured output (Muse Code 1.4.14). Observed envelope: every line is
{..., stream, payload_type, payload}. Mapping:
- run.model.configured -> transcript model (payload.model_id)
- run_stream.id -> default run_id
- first turn.input.user prompt -> transcript input; later ones -> user events
- run.output.delta text -> text events (consecutive deltas concatenate)
- task.lifecycle.side_effect_intent with operation "tool:<name>" -> tool_call
  (the envelope carries no arguments; name + call id survive)
- tool.result -> tool_result (name from correlation_facts, text verbatim)
- run.terminal.* -> run_outcome {ok: terminal == "completed"}; its text is
  final_output (it duplicates the last text event when deltas carried it)
- run.usage -> cost (tokens summed over models)
Everything else (task lifecycle, session/stream bookkeeping, reconciliation,
the task.lifecycle.output duplicate of tool results) is harness noise and is
skipped. Unknown payload types are skipped too: adapters must not break when
the harness adds event kinds. Reasoning is usually absent from exec output
(hidden by the model); extraction then works from the other events.
"""
from __future__ import annotations

import json

from . import TranscriptError


def adapt(source: str, workflow: str, run_id: str | None = None,
          model: str | None = None, data_class: str = "private") -> dict:
    if not isinstance(source, str) or not source.strip():
        raise TranscriptError("invalid_transcript", "muse transcript is empty")
    if not workflow:
        raise TranscriptError("invalid_transcript", "muse adapt needs a workflow")
    if data_class not in ("public", "private"):
        raise TranscriptError("invalid_transcript",
                              f"bad data_class {data_class!r}")
    events: list[dict] = []
    input_text = ""
    model_id = model
    exec_run_id = run_id
    terminal: str | None = None
    terminal_text = ""
    cost: dict | None = None

    def text_sink() -> dict:
        if events and events[-1]["type"] == "text":
            return events[-1]
        events.append({"type": "text", "content": ""})
        return events[-1]

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
        if exec_run_id is None:
            rs = ((obj.get("payload") or {}).get("run_stream") or {})
            if isinstance(rs.get("id"), str):
                exec_run_id = rs["id"]
        pt = obj.get("payload_type", "")
        payload = obj.get("payload") or {}
        if pt == "run.model.configured":
            if model_id is None and isinstance(payload.get("model_id"), str):
                model_id = payload["model_id"]
        elif pt == "turn.input.user":
            prompt = payload.get("prompt")
            if isinstance(prompt, str) and prompt:
                if not input_text:
                    input_text = prompt
                else:
                    events.append({"type": "user", "content": prompt})
        elif pt == "run.output.delta":
            if isinstance(payload.get("text"), str) and payload["text"]:
                text_sink()["content"] += payload["text"]
        elif pt == "task.lifecycle.side_effect_intent":
            event = payload.get("event") or {}
            op = event.get("operation", "")
            if isinstance(op, str) and op.startswith("tool:"):
                key = event.get("idempotency_key", "")
                item: dict = {"type": "tool_call", "name": op[5:], "content": ""}
                if isinstance(key, str) and key.startswith("tool:"):
                    item["name"] = item["name"] or key[5:]
                events.append(item)
        elif pt == "tool.result":
            facts = payload.get("correlation_facts") or {}
            text = payload.get("text")
            events.append({"type": "tool_result",
                           "name": facts.get("tool_name", "unknown"),
                           "content": text if isinstance(text, str) else ""})
        elif pt == "run.usage":
            models = payload.get("models") or []
            cost = {"tokens_in": sum(m.get("input_tokens", 0) for m in models),
                    "tokens_out": sum(m.get("output_tokens", 0) for m in models),
                    "cached_tokens": sum(m.get("cached_tokens", 0) for m in models),
                    "calls": payload.get("calls", 0)}
        elif pt.startswith("run.terminal."):
            terminal = pt.split("run.terminal.", 1)[1]
            text = payload.get("text")
            if isinstance(text, str) and text:
                terminal_text = text
        # else: harness noise or unknown future kind -> skip
    if exec_run_id is None:
        raise TranscriptError("invalid_transcript",
                              "muse transcript has no run_stream id (pass run_id)")
    for n, e in enumerate(events):
        e["i"] = n
    t: dict = {
        "run_id": exec_run_id,
        "workflow": workflow,
        "model": model_id or "",
        "data_class": data_class,
        "input": input_text,
        "events": events,
        "final_output": terminal_text or next(
            (e["content"] for e in reversed(events) if e["type"] == "text"), ""),
    }
    if cost is not None:
        t["cost"] = cost
    if terminal is not None:
        t["run_outcome"] = {"ok": terminal == "completed"}
    return t
