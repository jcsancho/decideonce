"""Zero-edit gateway (§14): OpenAI/Anthropic-compatible front end on 127.0.0.1:8787.

Matched calls become decision rows (mode auto/rule/forwarded, source gateway);
unmatched traffic is only logged when log_unmatched/log_allow say so.
Stdlib HTTP only (no new dependencies).
"""
from __future__ import annotations

import concurrent.futures
import gzip
import http.client
import ipaddress
import json
import math
import random
import re
import ssl
import sys
import threading
import time
import urllib.parse
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import gwformat as GF
from . import gwtemplate as GT
from . import hostedfit as HF
from . import probe as P
from . import render as R
from . import rules as RU
from . import stats as S
from .config import data_root, load_config
from .deciders import DeciderError
from .deciders.hosted import record_hosted_error
from .jev import JevError
from .store import Store

try:
    import re2 as _re2  # type: ignore
except ImportError:
    _re2 = None

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailer", "transfer-encoding", "upgrade"}
# Never stored or logged (auth headers still go upstream; x-decideonce-token does not).
SECRET_HEADERS = {"authorization", "x-api-key", "proxy-authorization", "cookie",
                  "set-cookie", "openai-organization", "openai-project", "x-decideonce-token",
                  "x-decide-token"}  # old header accepted (M15)

MAX_BODY = 10 * 1024 * 1024


def _utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code = status, code


# --------------------------------------------------------------------------
# request parsing (§14.4 shapes)
# --------------------------------------------------------------------------

def _text_of_content(content, details: str) -> str | None:
    """String or text-only parts -> text; anything else -> None (forward)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict):
                return None
            if part.get("type") in ("text", "input_text") and isinstance(
                    part.get("text"), str):
                texts.append(part["text"])
            else:
                return None
        return "\n".join(texts)
    return None


def parse_openai(body: dict) -> dict:
    """Normalise a chat-completions request. Raises GatewayError(400) when the
    body is not a completions request at all; shape problems return shape=None."""
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise GatewayError(400, "bad_request", "not a chat-completions body")
    messages = body["messages"]
    if not messages or not all(isinstance(m, dict) for m in messages):
        raise GatewayError(400, "bad_request", "messages must be a non-empty list")
    bad = False
    system, rest = "", messages
    if rest and rest[0].get("role") in ("system", "developer"):
        system = _text_of_content(rest[0].get("content"), "system") or ""
        if rest[0].get("content") is not None and system == "" and rest[0].get(
                "content") != "":
            return {"shape": None, "model": body.get("model", "")}
        rest = rest[1:]
    single = (len(rest) == 1 and rest[0].get("role") == "user"
              and not any(m.get("role") != "user" for m in rest))
    user, conversation = "", ""
    if single:
        t = _text_of_content(rest[0].get("content"), "user")
        if t is None:
            bad = True
        else:
            user = t
    else:
        for m in rest:
            t = _text_of_content(m.get("content"), m.get("role", ""))
            if t is None:
                bad = True
                break
            conversation += f"{m.get('role')}: {t}\n"
        conversation = conversation.rstrip("\n")
    if body.get("tools") or body.get("functions") or body.get("tool_choice"):
        bad = True
    if body.get("n") is not None and body.get("n") != 1:
        bad = True
    if body.get("logprobs") or body.get("top_logprobs"):
        bad = True
    if body.get("audio") or body.get("modalities"):
        bad = True
    rf = body.get("response_format")
    if rf is not None and rf.get("type") not in ("text", "json_object", "json_schema"):
        bad = True
    stops = body.get("stop") or []
    stops = [stops] if isinstance(stops, str) else stops
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens"))
    return {"shape": None if bad else ("single" if single else "multi"),
            "model": body.get("model", ""),
            "system": system, "user": user, "conversation": conversation,
            "stream": bool(body.get("stream")),
            "include_usage": bool((body.get("stream_options") or {}).get("include_usage")),
            "stop": [s for s in stops if isinstance(s, str)], "max_tokens": max_tokens,
            "response_format": rf}


def parse_anthropic(body: dict) -> dict:
    if not isinstance(body, dict) or not isinstance(body.get("messages"), list):
        raise GatewayError(400, "bad_request", "not a messages body")
    messages = body["messages"]
    if not messages or not all(isinstance(m, dict) for m in messages):
        raise GatewayError(400, "bad_request", "messages must be a non-empty list")
    bad = False
    system = body.get("system", "")
    if isinstance(system, list):
        system = _text_of_content(system, "system")
        if system is None:
            return {"shape": None, "model": body.get("model", "")}
    elif not isinstance(system, str):
        return {"shape": None, "model": body.get("model", "")}
    single = len(messages) == 1 and messages[0].get("role") == "user"
    user, conversation = "", ""
    if single:
        t = _text_of_content(messages[0].get("content"), "user")
        if t is None:
            bad = True
        else:
            user = t
    else:
        for m in messages:
            t = _text_of_content(m.get("content"), m.get("role", ""))
            if t is None:
                bad = True
                break
            conversation += f"{m.get('role')}: {t}\n"
        conversation = conversation.rstrip("\n")
    if body.get("tools") or body.get("tool_choice") or body.get("thinking"):
        bad = True
    stops = body.get("stop_sequences") or []
    return {"shape": None if bad else ("single" if single else "multi"),
            "model": body.get("model", ""),
            "system": system, "user": user, "conversation": conversation,
            "stream": bool(body.get("stream")),
            "include_usage": True,  # anthropic streams always carry usage
            "stop": [s for s in stops if isinstance(s, str)],
            "max_tokens": body.get("max_tokens"), "response_format": None}


def answer_text_openai(payload: bytes) -> tuple[str | None, dict]:
    try:
        body = json.loads(payload.decode("utf-8", "replace"))
    except ValueError:
        return None, {}
    if isinstance(body, dict) and isinstance(body.get("choices"), list) and body["choices"]:
        msg = body["choices"][0].get("message", {})
        content = msg.get("content")
        if isinstance(content, str):
            return content, _usage_openai(body)
    return None, _usage_openai(body) if isinstance(body, dict) else {}


def _usage_openai(body: dict) -> dict:
    u = body.get("usage") or {}
    return {"prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}


def answer_text_anthropic(payload: bytes) -> tuple[str | None, dict]:
    try:
        body = json.loads(payload.decode("utf-8", "replace"))
    except ValueError:
        return None, {}
    if isinstance(body, dict) and isinstance(body.get("content"), list):
        texts = [b.get("text", "") for b in body["content"] if b.get("type") == "text"]
        u = body.get("usage") or {}
        usage = {"prompt_tokens": u.get("input_tokens", 0),
                 "completion_tokens": u.get("output_tokens", 0)}
        return ("\n".join(texts), usage) if texts else (None, usage)
    return None, {}


def sse_answer_openai(chunks: bytes) -> tuple[str, dict]:
    text, usage = "", {}
    for line in chunks.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except ValueError:
            continue
        for ch in ev.get("choices", []):
            delta = ch.get("delta", {})
            if isinstance(delta.get("content"), str):
                text += delta["content"]
        if ev.get("usage"):
            usage = _usage_openai(ev)
    return text, usage


def sse_answer_anthropic(chunks: bytes) -> tuple[str, dict]:
    text, usage = "", {}
    for line in chunks.decode("utf-8", "replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        try:
            ev = json.loads(line[5:].strip())
        except ValueError:
            continue
        if ev.get("type") == "content_block_delta":
            d = ev.get("delta", {})
            if d.get("type") == "text_delta" and isinstance(d.get("text"), str):
                text += d["text"]
        if ev.get("type") == "message_delta" and ev.get("usage"):
            u = ev["usage"]
            usage["completion_tokens"] = u.get("output_tokens", 0)
        if ev.get("type") == "message_start":
            u = (ev.get("message", {}).get("usage") or {})
            usage["prompt_tokens"] = u.get("input_tokens", 0)
            usage.setdefault("completion_tokens", u.get("output_tokens", 0))
    return text, usage


# --------------------------------------------------------------------------
# synthesized responses (§14.11, field-exact)
# --------------------------------------------------------------------------

def synth_openai(text: str, model: str) -> dict:
    return {"id": f"chatcmpl-decide-{uuid.uuid4().hex[:8]}", "object": "chat.completion",
            "created": int(time.time()), "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text,
                                                "refusal": None},
                         "logprobs": None, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "system_fingerprint": "decide-gateway"}


def synth_openai_stream(text: str, model: str, include_usage: bool) -> list[bytes]:
    rid, now = f"chatcmpl-decide-{uuid.uuid4().hex[:8]}", int(time.time())

    def chunk(choices, usage=None):
        body: dict = {"id": rid, "object": "chat.completion.chunk", "created": now,
                      "model": model, "choices": choices}
        if usage is not None:
            body["usage"] = usage
        return ("data: " + json.dumps(body) + "\n\n").encode()

    out = [chunk([{"index": 0, "delta": {"role": "assistant", "content": ""},
                         "finish_reason": None}]),
           chunk([{"index": 0, "delta": {"content": text}, "finish_reason": None}]),
           chunk([{"index": 0, "delta": {}, "finish_reason": "stop"}])]
    if include_usage:
        out.append(chunk([], {"prompt_tokens": 0, "completion_tokens": 0}))
    out.append(b"data: [DONE]\n\n")
    return out


def synth_anthropic(text: str, model: str) -> dict:
    return {"id": f"msg_decide_{uuid.uuid4().hex[:12]}", "type": "message",
            "role": "assistant", "model": model,
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}


def synth_anthropic_stream(text: str, model: str) -> list[bytes]:
    rid = f"msg_decide_{uuid.uuid4().hex[:12]}"

    def event(etype: str, payload: dict) -> bytes:
        return f"event: {etype}\ndata: {json.dumps(payload)}\n\n".encode()

    return [
        event("message_start", {"type": "message_start",
                                "message": {"id": rid, "type": "message", "role": "assistant",
                                            "model": model, "content": [], "stop_reason": None,
                                            "stop_sequence": None,
                                            "usage": {"input_tokens": 0, "output_tokens": 0}}}),
        event("content_block_start", {"type": "content_block_start", "index": 0,
                                      "content_block": {"type": "text", "text": ""}}),
        event("content_block_delta", {"type": "content_block_delta", "index": 0,
                                      "delta": {"type": "text_delta", "text": text}}),
        event("content_block_stop", {"type": "content_block_stop", "index": 0}),
        event("message_delta", {"type": "message_delta",
                                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                "usage": {"output_tokens": 0}}),
        event("message_stop", {"type": "message_stop"}),
    ]


# --------------------------------------------------------------------------
# field extraction (§14.7)
# --------------------------------------------------------------------------

def _search(pattern: str, text: str):
    engine = _re2.search if _re2 is not None else re.search
    box: list = []
    th = threading.Thread(target=lambda: box.append(engine(pattern, text)), daemon=True)
    th.start()
    th.join(timeout=0.05)
    return box[0] if box else None


def _cast(raw: str, cast: str):
    try:
        if cast == "string":
            return raw
        if cast == "number":
            # D: float("nan")/float("inf") parse but would serialise to
            # invalid JSON downstream; treat as no match.
            v = float(raw)
            return v if math.isfinite(v) else None
        if cast == "integer":
            v = float(raw)
            if not math.isfinite(v):
                return None
            return int(v) if v.is_integer() else None
        if cast == "boolean":
            low = raw.strip().lower()
            if low in ("true", "yes", "1"):
                return True
            if low in ("false", "no", "0"):
                return False
            return None
    except (ValueError, TypeError):
        return None
    return None


def extract_fields(extract: dict, text: str) -> dict:
    """First match wins per field; pattern > 200 chars never matches."""
    out = {}
    for field, spec in (extract or {}).items():
        pattern = spec.get("regex", "")
        if not pattern or len(pattern) > 200:
            continue
        try:
            m = _search(pattern, text)
        except Exception:
            continue
        if not m:
            continue
        try:
            raw = m.group(spec.get("group", 0))
        except Exception:
            continue
        if raw is None:
            continue
        value = _cast(str(raw), spec.get("cast", "string"))
        if value is not None:
            out[field] = value
    return out


def guard_fields_of(point: dict) -> set[str]:
    if "guard_fields" in point and point["guard_fields"] is not None:
        return set(point["guard_fields"])
    fields: set[str] = set()
    for rule in point.get("rules", []):
        if isinstance(rule, dict) and rule.get("guard") is False:
            continue
        for cond in rule.get("if", []):
            fields.add(cond[0])
    return fields


# --------------------------------------------------------------------------
# gateway flow
# --------------------------------------------------------------------------

class Gateway:
    def __init__(self, cfg: dict, store, jev, rng: random.Random | None = None):
        self.cfg = cfg
        self.store = store
        if isinstance(jev, dict):
            self.deciders = jev
            self.jev = None
            for d in jev.values():
                raw = getattr(d, "jev", None)
                if raw is not None:
                    self.jev = raw
                    break
        else:
            self.jev = jev
        self.rng = rng or random.Random()
        self.port = cfg["gateway"]["port"]
        self._jev_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        self._ssl = ssl.create_default_context()
        self._last_purge = 0.0
        from .service import Service
        self.svc = Service(cfg, store, jev, rng=self.rng)

    # -- decider call with timeout -----------------------------------------
    def _decide_call(self, decider, req):
        """decider.decide in the pool: slow backends time out (T-G15).

        B8: on the hosted path the outer pool timeout is twice the inner
        decider timeout, so the decider's own timeout always fires first
        (releasing its reservation and mapping cleanly); a pool timeout
        surfaces as DeciderError("timeout") so it is counted as
        hosted_error_timeout. R2-6: the local probe keeps a single timeout.
        """
        from .deciders.hosted import HostedJevDecider

        inner = self.cfg["gateway"]["jev_timeout_s"]
        outer = inner * 2 if isinstance(decider, HostedJevDecider) else inner
        fut = self._jev_pool.submit(decider.decide, [req], inner)
        try:
            return fut.result(timeout=outer)[0]
        except concurrent.futures.TimeoutError as e:
            raise DeciderError("timeout", "decider timed out") from e

    # -- point lookup ------------------------------------------------------
    def _gateway_points(self, route: str, model: str) -> list[dict]:
        return [p for p in self.store.list_points()
                if p.get("source") == "gateway" and p.get("route") == route
                and p.get("model") == model]

    def _match_point(self, route: str, model: str, system: str, user: str
                     ) -> tuple[dict | None, list[dict]]:
        """(best template match, points sharing the discovery key)."""
        cands = self._gateway_points(route, model)
        dk = GT.discovery_key(route, model, system)
        same_dk = [p for p in cands if p.get("discovery_key") == dk]
        best, best_len = None, -1
        for p in cands:
            t = p.get("template") or {}
            if not t or not GT.template_matches(t, system, user):
                continue
            fixed = len(t.get("system", "")) + len(t.get("user_prefix", "")) + len(
                t.get("user_suffix", ""))
            if (fixed, p["point_id"]) > (best_len, best["point_id"] if best else ""):
                best, best_len = p, fixed
        return best, same_dk

    # -- main entry --------------------------------------------------------
    def handle_decision(self, route: str, req: dict, raw_body: bytes) -> dict:
        """Route a parsed decision request. Returns a plan the handler executes:
        {"action": "auto", ...} or {"action": "forward", ...}."""
        t0 = time.monotonic()
        self._maybe_purge()
        model = req.get("model", "")
        match_user = req.get("user") or self._last_user(req)
        point, same_dk = self._match_point(route, model, req.get("system", ""), match_user)
        match_text = match_user
        if point is None and req.get("shape") == "multi":
            # multi_turn points pin the whole conversation, not the last line
            point2, _ = self._match_point(route, model, req.get("system", ""),
                                          req.get("conversation", ""))
            if point2 is not None and point2.get("multi_turn"):
                point, match_text = point2, req.get("conversation", "")
        if point is None:
            for p in same_dk:
                self.store.counter_inc("fingerprint_variant", p["point_id"])
            return {"action": "forward", "point": None, "t0": t0, "raw_body": raw_body}
        if req.get("shape") is None:
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["shape_not_allowed"])
        template = point.get("template") or {}
        middle = GT.middle_of(match_text, template.get("user_prefix", ""),
                              template.get("user_suffix", ""))
        if middle is not None and len(middle) > self.cfg["gateway"]["max_state_chars"]:
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["middle_too_long"])
        state_text = self._trim_state(self._state_text(point, req))
        extract_text = req["user"] if req.get("shape") == "single" else req.get(
            "conversation", "")
        context = extract_fields(point.get("extract"), extract_text)
        if req.get("shape") == "multi" and not point.get("multi_turn"):
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["multi_turn_not_enabled"],
                                        context=context, state_text=state_text)
        if not self._response_format_ok(req, point):
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["response_format_not_allowed"],
                                        context=context, state_text=state_text)
        missing = sorted(guard_fields_of(point) - set(context))
        if missing:
            self.store.counter_inc("missing_guard_field", point["point_id"])
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["missing_guard_field:" + ",".join(missing)],
                                        context=context, state_text=state_text)
        missing_req = sorted(set(point.get("required_fields") or []) - set(context))
        if missing_req:
            self.store.counter_inc("missing_required_field", point["point_id"])
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["missing_required_field:" + ",".join(
                                            missing_req)],
                                        context=context, state_text=state_text)
        allowed, rule_hits, warnings = RU.apply_rules(point, context)
        if not allowed:
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=["rules_conflict"], context=context,
                                        state_text=state_text)
        if len(allowed) == 1:
            return self._auto(route, req, point, allowed[0], {allowed[0]: 1.0}, 1.0,
                              None, None, context, rule_hits, warnings, t0, raw_body,
                              mode="rule")
        if point.get("phase") == "paused":
            return self._forward_logged(route, req, point, t0, raw_body, context=context,
                                        state_text=state_text, allowed=allowed,
                                        rule_hits=rule_hits, warnings=warnings)
        options = list(point["options"])
        decider = self.svc._decider_for(point)
        if decider is None:
            # tiered collect/paused (or a tier without a decider): forward
            # with no model call.
            return self._forward_logged(route, req, point, t0, raw_body,
                                        context=context, state_text=state_text,
                                        allowed=allowed, rule_hits=rule_hits,
                                        warnings=warnings)
        try:
            model_row = self.svc._current_model(point)
            dreq = self.svc._decide_request(point, context,
                                            state_text=state_text)
            if not self.svc._legacy_mode:
                dreq = self._gateway_request(point, context, dreq)
            res = self._decide_call(decider, dreq)
            hosted_field: dict | None = None
            decider_tag: str | None = None
            if self.svc._legacy_mode:
                if res._hidden is not None and point.get("phase") in (
                        "shadow", "serve"):
                    from . import train as TR
                    TR.cache_vector(self.store, point, state_text, res._hidden)
                zs_full = res._zs_full or {}
                probe_full = res._probe_full
                shown_full: dict = res.probs
                probe_field: dict | None = (
                    {"model_version": model_row["card"].get("model_version"),
                     "probabilities": probe_full}
                    if probe_full is not None else None)
            elif res.decider == "private":
                probe_field = {"model_version": res.model,
                               "probabilities": res.probs}
                decider_tag = res.decider
                zs_full, shown_full = {}, res.probs
            else:
                hosted_field = {
                    "model": res.model, "probabilities": res.probs,
                    "cost_usd": res.cost_usd, "seconds": res.seconds}
                decider_tag = res.decider
                tmp: dict = {}
                self.svc._maybe_store_private(point, dreq, tmp)
                probe_field = tmp.get("probe")
                zs_full, shown_full = {}, res.probs
                if res.warnings:
                    warnings = warnings + list(res.warnings)
            jev_seconds = res.seconds
        except (JevError, DeciderError) as e:
            tag = "jev_unavailable"
            if not self.svc._legacy_mode and isinstance(e, DeciderError):
                record_hosted_error(self.store, point["point_id"], e)
                tag = {"budget_exceeded": "budget_exceeded",
                       "no_key": "no_hosted_jev_key"}.get(e.code, tag)
            self.store.counter_inc("jev_unavailable", point["point_id"])
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=warnings + [tag],
                                        context=context, state_text=state_text,
                                        allowed=allowed, rule_hits=rule_hits)
        if (not self.svc._legacy_mode and decider_tag == "hosted_jev"
                and self.svc._hosted_tier(point)):
            shown = HF.mask_temper(shown_full, allowed,
                                   self.svc._hosted_temperature(point))
        else:
            shown = P.mask_and_renorm(shown_full, allowed)
        confidence = max(shown.values())
        top = P.argmax(shown, allowed)
        if point.get("phase") != "serve":
            return self._forward_logged(route, req, point, t0, raw_body, context=context,
                                        state_text=state_text, allowed=allowed,
                                        rule_hits=rule_hits, warnings=warnings,
                                        jev_zero_shot=zs_full, probe=probe_field,
                                        jev_seconds=jev_seconds, hosted=hosted_field,
                                        decider=decider_tag)
        mode, _, audit = self.svc._select_mode(point, confidence, top,
                                               probe_field is not None)
        if mode != "auto":
            return self._forward_logged(route, req, point, t0, raw_body, context=context,
                                        state_text=state_text, allowed=allowed,
                                        rule_hits=rule_hits, warnings=warnings,
                                        jev_zero_shot=zs_full, probe=probe_field,
                                        jev_seconds=jev_seconds, hosted=hosted_field,
                                        decider=decider_tag, audit=audit)
        return self._auto(route, req, point, top, shown, confidence, zs_full, probe_field,
                          context, rule_hits, warnings, t0, raw_body, mode="auto",
                          jev_seconds=jev_seconds, hosted=hosted_field,
                          decider=decider_tag)

    @staticmethod
    def _gateway_request(point: dict, context: dict, req):
        """Hosted gateway state: policy + registered fields + the extracted
        template fields (build §10)."""
        import dataclasses
        state = dict(req.state)
        for k, v in (context or {}).items():
            if v is None or k in state:
                continue
            state[k] = v[:R.DEFAULT_MAX_LEN] if isinstance(v, str) else v
        return dataclasses.replace(req, state=state)

    @staticmethod
    def _last_user(req: dict) -> str:
        conv = req.get("conversation", "")
        for line in reversed(conv.splitlines()):
            if line.startswith("user:"):
                return line[5:].strip()
        return ""

    @staticmethod
    def _state_text(point: dict, req: dict) -> str:
        if req.get("shape") == "multi":
            user = req.get("conversation", "")
        else:
            user = req.get("user", "")
        return f"{req.get('system', '')}\n\n{user}" if req.get("system") else user

    def _trim_state(self, text: str) -> str:
        limit = self.cfg["gateway"]["max_state_chars"]
        if len(text) <= limit:
            return text
        half = limit // 2
        return text[:half] + text[len(text) - half:]

    def _response_format_ok(self, req: dict, point: dict) -> bool:
        rf = req.get("response_format")
        if rf is None or rf.get("type") in ("text", "json_object"):
            return True
        if rf.get("type") != "json_schema":
            return False
        of = point.get("output_format") or {}
        if of.get("type") != "json_enum":
            return False
        try:
            node = (rf.get("json_schema") or {}).get("schema", {})
            for part in of["path"].split("."):
                node = node.get("properties", {}).get(part, node.get(part, {}))
            return set(node.get("enum", [])) == set(point["options"])
        except (AttributeError, TypeError, KeyError):
            return False

    def _forward_logged(self, route, req, point, t0, raw_body, warnings=None,
                        context=None, state_text=None, allowed=None, rule_hits=None,
                        jev_zero_shot=None, probe=None, jev_seconds=None, audit=False,
                        hosted=None, decider=None):
        return {"action": "forward", "point": point, "t0": t0, "raw_body": raw_body,
                "warnings": warnings or [], "context": context, "state_text": state_text,
                "allowed": allowed or list(point["options"]), "rule_hits": rule_hits or [],
                "jev_zero_shot": jev_zero_shot, "probe": probe, "jev_seconds": jev_seconds,
                "audit": audit, "hosted": hosted, "decider": decider}

    def _auto(self, route, req, point, choice, shown, confidence, zs_full, probe_field,
              context, rule_hits, warnings, t0, raw_body, mode, jev_seconds=None,
              hosted=None, decider=None):
        text = GF.render_answer(choice, point.get("output_format") or {})
        for stop in req.get("stop") or []:  # stop/max_tokens gate (§14.4)
            if stop and stop in text:
                return self._forward_logged(route, req, point, t0, raw_body,
                                            warnings=warnings + ["stop_conflict"],
                                            context=context,
                                            state_text=self._trim_state(
                                                self._state_text(point, req)),
                                            jev_zero_shot=zs_full, probe=probe_field,
                                            jev_seconds=jev_seconds,
                                            hosted=hosted, decider=decider)
        mt = req.get("max_tokens")
        if isinstance(mt, int) and -(-len(text) // 3) > mt:
            return self._forward_logged(route, req, point, t0, raw_body,
                                        warnings=warnings + ["max_tokens_conflict"],
                                        context=context,
                                        state_text=self._trim_state(
                                            self._state_text(point, req)),
                                        jev_zero_shot=zs_full, probe=probe_field,
                                        jev_seconds=jev_seconds,
                                        hosted=hosted, decider=decider)
        return {"action": "auto", "point": point, "choice": choice, "text": text,
                "shown": shown, "confidence": confidence, "jev_zero_shot": zs_full,
                "probe": probe_field, "context": context, "rule_hits": rule_hits,
                "warnings": warnings, "t0": t0, "raw_body": raw_body, "mode": mode,
                "jev_seconds": jev_seconds, "hosted": hosted, "decider": decider,
                "state_text": self._trim_state(self._state_text(point, req))}

    # -- logging -----------------------------------------------------------
    def _should_log_unmatched(self, route: str, model: str, dk: str) -> bool:
        if self.cfg["gateway"]["log_unmatched"]:
            return True
        allow = self.cfg["gateway"]["log_allow"]
        return dk in allow or f"{route}:{model}" in allow

    def _log_forwarded_unmatched(self, route, req, raw_body, status, answer, usage,
                                 latency_ms, upstream_ms) -> None:
        dk = GT.discovery_key(route, req.get("model", ""), req.get("system", ""))
        if not self._should_log_unmatched(route, req.get("model", ""), dk):
            self.store.counter_inc(
                "unregistered", f"{route}:{req.get('model', '')}:{status}")
            return
        self.store.trace_insert({
            "ts": _utcnow(), "route": route, "model": req.get("model", ""),
            "discovery_key": dk, "mode": "forwarded",
            "raw_request": {"system": req.get("system", ""), "user": req.get(
                "user", "") or req.get("conversation", "")},
            "raw_response": answer, "latency_ms": latency_ms,
            "upstream_ms": upstream_ms, "usage": usage, "status": status,
            "data_class": self.cfg["gateway"]["trace_data_class"]})

    def _insert_decision_row(self, point, plan, req, parsed, unparsed, mode,
                             final_choice, decision_id=None) -> str:
        did = decision_id or uuid.uuid4().hex
        self.store.insert_decision({
            "decision_id": did, "point_id": point["point_id"],
            "point_version": point.get("point_version", 1), "ts": _utcnow(),
            "context": plan.get("context"), "state_text": plan.get("state_text"),
            "options": list(point["options"]), "allowed_options": plan.get("allowed"),
            "rule_hits": plan.get("rule_hits"), "jev_zero_shot": plan.get("jev_zero_shot"),
            "probe": plan.get("probe"), "mode": mode, "audit": plan.get("audit", False),
            "source": "gateway", "teacher_choice": parsed,
            "final_choice": final_choice,
            "phase": point.get("phase", "collect"),
            "jev_seconds": plan.get("jev_seconds"), "unparsed": unparsed,
            "hosted": plan.get("hosted"), "decider": plan.get("decider"),
            "warnings": plan.get("warnings") or None}, self.cfg["store"]["max_rows_per_point"])
        if parsed is not None:
            # gateway teacher labels land at insert: they can complete fits.
            self.svc._maybe_fit_hosted(point["point_id"])
            self.svc._maybe_fit_hosted(point["point_id"], "plan")
        return did

    def _log_auto(self, route, req, point, plan, latency_ms) -> str:
        did = self._insert_decision_row(point, plan, req, None, False, plan["mode"],
                                        plan["choice"], None)
        dk = GT.discovery_key(route, req.get("model", ""), req.get("system", ""))
        self.store.trace_insert({
            "ts": _utcnow(), "route": route, "model": req.get("model", ""),
            "discovery_key": dk, "fingerprint": point.get("fingerprint"),
            "point_id": point["point_id"], "point_version": point.get("point_version", 1),
            "decision_id": did, "mode": plan["mode"],
            "raw_request": {"system": req.get("system", ""),
                            "user": req.get("user", "") or req.get("conversation", "")},
            "raw_response": plan["text"], "parsed_option": plan["choice"],
            "latency_ms": latency_ms, "status": 200,
            "data_class": self.cfg["gateway"]["trace_data_class"]})
        return did

    def _post_serve_format_check(self, point: dict) -> None:
        traces = [t for t in self.store.traces_for_point(
            point["point_id"], point.get("point_version", 1)) if t.get("format_compared")]
        # NOTE: scoped to the current point version: a version bump (§14.9)
        # resets the format-check evidence along with the promotion gate.
        last = traces[-100:]
        if len(last) < 100:
            return
        mism = sum(1 for t in last if not t.get("format_match"))
        if S.wilson_upper(mism, len(last)) > 0.05:
            fresh = self.store.get_point(point["point_id"])
            if fresh is None or fresh.get("phase") != "serve":
                return
            fresh["phase"] = "shadow"
            # E4: a format demotion starts the cooldown like any demotion.
            from .service import Service as _Svc

            _Svc._note_demotion_cooldown(fresh)
            fresh.setdefault("events", []).append(
                {"ts": _utcnow(), "event": "format_demotion",
                 "mismatches": mism, "n": len(last)})
            self.store.put_point(fresh)
            try:
                self.svc.autopilot.note_demotion(
                    fresh, {"errors": mism, "n": len(last),
                            "format_demotion": True})
            except Exception:
                pass

    def _maybe_purge(self) -> None:
        now = time.time()
        if now - self._last_purge < 60:
            return
        self._last_purge = now
        purge_retention(self.store, self.cfg)


def format_evidence(cfg: dict, store, point: dict) -> dict:
    """Promotion item 7 numbers (§14.9) for the point's current version."""
    traces = store.traces_for_point(point["point_id"], point.get("point_version", 1))
    answered = [t for t in traces
                if t.get("mode") == "forwarded" and t.get("raw_response") is not None]
    compared = [t for t in answered if t.get("format_compared")]
    matches = sum(1 for t in compared if t.get("format_match"))
    n_unparsed = sum(1 for t in answered if t.get("unparsed"))
    n, mism = len(compared), len(compared) - matches
    return {
        "n_compared": n,
        "n_match": matches,
        "match_rate": matches / n if n else None,
        "wilson_lower": 1.0 - S.wilson_upper(mism, n) if n else None,
        "n_answered": len(answered),
        "n_unparsed": n_unparsed,
        "unparsed_rate": n_unparsed / len(answered) if answered else None,
    }


def purge_retention(store, cfg: dict, now_s: float | None = None) -> dict:
    """Purge old unregistered traces and old registered text (§14.6)."""
    now = now_s if now_s is not None else time.time()
    day = 86400
    unreg_cut = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                              time.gmtime(now - cfg["gateway"]["trace_retention_days"] * day))
    reg_cut = time.strftime("%Y-%m-%dT%H:%M:%SZ",
                            time.gmtime(now - cfg["gateway"]["point_retention_days"] * day))
    return {"unregistered_deleted": store.purge_unregistered_traces(unreg_cut),
            "registered_text_purged": store.text_purge_registered(reg_cut)}


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

def check_gateway_local(host: str) -> None:
    from .config import ConfigError
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ConfigError("gateway_not_local",
                      f"gateway.host must be a loopback IP literal, got {host!r}")


def split_route(path: str) -> tuple[str | None, str]:
    """("openai"|"anthropic", upstream path) or (None, "") for non-API paths."""
    for name in ("openai", "anthropic"):
        if path == f"/{name}" or path.startswith(f"/{name}/"):
            return name, path[len(name) + 1:] or "/"
    return None, ""


def check_upstream_path(upath: str) -> str:
    if "://" in upath or not upath.startswith("/v1/") or ".." in upath or "//" in upath:
        raise GatewayError(400, "bad_path", f"refusing to forward {upath!r}")
    return upath


class Handler(BaseHTTPRequestHandler):
    server_version = "decide-gateway/0.1"
    protocol_version = "HTTP/1.1"
    gateway: Gateway  # set by serve()

    def log_message(self, fmt, *args):  # quiet: never log request content
        pass

    # -- entry ----------------------------------------------------------------
    def _guard(self) -> None:
        gw = self.gateway
        port = gw.server_port
        host = (self.headers.get("Host") or "")
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"):
            raise GatewayError(403, "forbidden_host", "bad Host header")
        if self.headers.get("Origin") is not None:
            raise GatewayError(403, "forbidden_origin", "Origin header not allowed")
        token = gw.cfg["gateway"]["local_token"]
        presented = self.headers.get("x-decideonce-token")
        if presented is None:  # M15 compat: old header accepted
            presented = self.headers.get("x-decide-token")
        if token and presented != token:
            raise GatewayError(403, "forbidden_token", "bad or missing x-decideonce-token")

    def _read_body(self) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            raise GatewayError(400, "bad_request", "bad Content-Length")
        if length > MAX_BODY:
            raise GatewayError(413, "body_too_large", f"cap {MAX_BODY} bytes")
        return self.rfile.read(length) if length else b""

    def _send_json(self, status: int, obj: dict, extra: dict | None = None) -> None:
        payload = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(payload)

    def _handle(self) -> None:
        gw = self.gateway
        try:
            self._guard()
        except GatewayError as e:
            self._send_json(e.status, {"error": e.code, "message": str(e)})
            return
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/health" and self.command == "GET":
            self._send_json(200, {"ok": True})
            return
        route, upath = split_route(parsed.path)
        if route is None:
            self._send_json(404, {"error": "not_found"})
            return
        try:
            decision = (route == "openai" and upath == "/v1/chat/completions") or (
                route == "anthropic" and upath == "/v1/messages")
            if self.command == "POST" and decision:
                self._decision(route)
                return
            self._forward_generic(
                route, upath + (("?" + parsed.query) if parsed.query else ""))
        except GatewayError as e:
            self._send_json(e.status, {"error": e.code})
        except BrokenPipeError:
            pass
        except Exception:
            # Never render exception text: it can echo request content (§14.3).
            try:
                self._send_json(500, {"error": "internal"})
            except (BrokenPipeError, ConnectionResetError):
                pass

    do_GET = _handle
    do_POST = _handle
    do_PUT = _handle
    do_DELETE = _handle
    do_PATCH = _handle

    # -- decision routes ------------------------------------------------------
    def _decision(self, route: str) -> None:
        gw = self.gateway
        try:
            raw_body = self._read_body()
            body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except (ValueError, UnicodeDecodeError):
            self._send_json(400, {"error": "bad_request", "message": "body is not JSON"})
            return
        try:
            req = parse_openai(body) if route == "openai" else parse_anthropic(body)
        except GatewayError as e:
            self._send_json(e.status, {"error": e.code, "message": str(e)})
            return
        plan = gw.handle_decision(route, req, raw_body)
        if plan["action"] == "auto":
            self._respond_auto(route, req, plan)
        else:
            self._respond_forward(route, req, plan, self._upstream_path(route))

    def _upstream_path(self, route: str) -> str:
        parsed = urllib.parse.urlsplit(self.path)
        _, upath = split_route(parsed.path)
        return upath + (("?" + parsed.query) if parsed.query else "")

    def _respond_auto(self, route: str, req: dict, plan: dict) -> None:
        gw = self.gateway
        point, model = plan["point"], req.get("model", "")
        if req.get("stream"):
            if route == "openai":
                chunks = synth_openai_stream(plan["text"], model, req.get("include_usage"))
            else:
                chunks = synth_anthropic_stream(plan["text"], model)
            did = gw._log_auto(route, req, point, plan,
                               (time.monotonic() - plan["t0"]) * 1000)
            payload = b"".join(chunks)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("x-decide-mode", plan["mode"])
            self.send_header("x-decide-id", did)
            self.send_header("x-decide-point", point["point_id"])
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
        else:
            obj = synth_openai(plan["text"], model) if route == "openai" else \
                synth_anthropic(plan["text"], model)
            did = gw._log_auto(route, req, point, plan,
                               (time.monotonic() - plan["t0"]) * 1000)
            self._send_json(200, obj, {"x-decide-mode": plan["mode"], "x-decide-id": did,
                                       "x-decide-point": point["point_id"]})

    # -- forwarding -----------------------------------------------------------
    def _forward_headers(self) -> dict:
        # Auth headers go upstream (else calls fail); only x-decideonce-token is
        # stripped. Nothing header-shaped is ever persisted (traces keep parsed
        # bodies only), which is what §14.3 requires.
        out = {}
        for k, v in self.headers.items():
            kl = k.lower()
            if kl in HOP_BY_HOP or kl in ("host", "x-decideonce-token", "x-decide-token"):
                continue
            out[k] = v
        return out

    def _call_upstream(self, route: str, upath: str, method: str, body: bytes,
                       stream: bool):
        gw = self.gateway
        base = gw.cfg["gateway"]["upstreams"][route]
        parts = urllib.parse.urlsplit(base)
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(
                parts.hostname, parts.port or 443, context=gw._ssl,
                timeout=gw.cfg["gateway"]["upstream_timeout_s"])
        else:
            conn = http.client.HTTPConnection(
                parts.hostname, parts.port or 80,
                timeout=gw.cfg["gateway"]["upstream_timeout_s"])
        headers = self._forward_headers()
        headers["Host"] = parts.hostname + (f":{parts.port}" if parts.port else "")
        if body:
            headers["Content-Length"] = str(len(body))
        t0 = time.monotonic()
        conn.request(method, upath, body=body, headers=headers)
        resp = conn.getresponse()
        upstream_ms = (time.monotonic() - t0) * 1000
        return conn, resp, upstream_ms

    def _copy_response(self, resp, extra: dict | None = None) -> bytes:
        payload = resp.read()
        self._send_full(resp, extra, payload)
        return payload

    def _send_stream_headers(self, resp, extra: dict | None = None) -> None:
        """Headers for a relayed stream: bounded when the upstream gave a
        length, else close-delimited (HTTP/1.1 keep-alive would hang)."""
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                continue
            self.send_header(k, v)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        length = resp.getheader("Content-Length")
        if length is not None:
            self.send_header("Content-Length", length)
        else:
            self.send_header("Connection", "close")
            self.close_connection = True
        self.end_headers()

    def _relay(self, resp) -> bytes:
        # Small reads: http.client read(n) waits for n bytes, which would hold
        # back the first SSE chunk until the upstream finishes. 16 bytes is
        # smaller than any SSE line, so each upstream flush reaches the client.
        buf = b""
        while True:
            chunk = resp.read(16)
            if not chunk:
                break
            buf += chunk
            self.wfile.write(chunk)
            self.wfile.flush()
        return buf

    def _send_full(self, resp, extra: dict | None, payload: bytes) -> None:
        self.send_response(resp.status)
        for k, v in resp.getheaders():
            if k.lower() in HOP_BY_HOP or k.lower() == "content-length":
                continue
            self.send_header(k, v)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_forward(self, route: str, req: dict, plan: dict, upath: str) -> None:
        gw = self.gateway
        try:
            check_upstream_path(upath)
        except GatewayError as e:
            self._send_json(e.status, {"error": e.code})
            return
        stream = bool(req.get("stream"))
        try:
            conn, resp, upstream_ms = self._call_upstream(
                route, upath, self.command, plan["raw_body"], stream)
        except Exception:
            self._send_json(502, {"error": "upstream_unreachable"})
            return
        point = plan.get("point")
        try:
            if point is None:
                if stream:
                    # Unbuffered relay: logging necessarily lands after.
                    self._send_stream_headers(resp)
                    payload = self._relay(resp)
                else:
                    payload = resp.read()
                answer, usage = self._extract_answer(route, resp, payload)
                latency_ms = (time.monotonic() - plan["t0"]) * 1000
                if not stream:
                    # Log before responding: the client must never observe a
                    # call the store does not know about yet.
                    gw._log_forwarded_unmatched(
                        route, req, plan["raw_body"], resp.status, answer, usage,
                        latency_ms, upstream_ms)
                    self._send_full(resp, None, payload)
                else:
                    gw._log_forwarded_unmatched(
                        route, req, plan["raw_body"], resp.status, answer, usage,
                        latency_ms, upstream_ms)
            else:
                self._buffered_forward(route, req, plan, resp, stream, upstream_ms)
        finally:
            conn.close()

    def _buffered_forward(self, route, req, plan, resp, stream, upstream_ms) -> None:
        """Forward while capturing the answer; decision id is known up front."""
        gw = self.gateway
        point = plan["point"]
        did = uuid.uuid4().hex
        extra = {"x-decide-mode": "forwarded", "x-decide-id": did,
                 "x-decide-point": point["point_id"]}
        if stream:
            # SSE still streams to the client; the id is in the headers
            # before the first byte, the row lands after the last one.
            self._send_stream_headers(resp, extra)
            payload = self._relay(resp)
        else:
            payload = resp.read()
        if resp.status >= 400:
            # Upstream errors are transport, not labels: trace only, no row.
            gw.store.trace_insert({
                "ts": _utcnow(), "route": route, "model": req.get("model", ""),
                "discovery_key": GT.discovery_key(route, req.get("model", ""),
                                                  req.get("system", "")),
                "fingerprint": point.get("fingerprint"), "point_id": point["point_id"],
                "point_version": point.get("point_version", 1),
                "decision_id": None, "mode": "forwarded",
                "raw_request": {"system": req.get("system", ""),
                                "user": req.get("user", "") or req.get("conversation", "")},
                "raw_response": None, "latency_ms": (time.monotonic() - plan["t0"]) * 1000,
                "upstream_ms": upstream_ms, "usage": {}, "status": resp.status,
                "data_class": gw.cfg["gateway"]["trace_data_class"]})
            if not stream:
                self._send_full(resp, extra, payload)
            return
        answer, usage = self._extract_answer(route, resp, payload)
        # insert the decision row with the pre-generated id
        parsed = GF.parse_answer(answer or "", point.get("output_format") or {},
                                 list(point["options"])) if answer is not None else None
        unparsed = answer is None or parsed is None
        gw._insert_decision_row(point, plan, req, parsed, unparsed, "forwarded",
                                parsed if not unparsed else None, did)
        # Format check (§14.9): render what the gateway would have answered
        # for the upstream's parsed option and compare with the final text.
        compared = matched_fmt = False
        if not unparsed and (point.get("output_format") or {}).get("type") in (
                "enum_text", "json_enum"):
            try:
                want = GF.render_answer(parsed, point.get("output_format") or {})
            except (ValueError, KeyError, TypeError):
                want = None
            if want is not None:
                compared, matched_fmt = True, want.rstrip() == (answer or "").rstrip()
        dk = GT.discovery_key(route, req.get("model", ""), req.get("system", ""))
        gw.store.trace_insert({
            "ts": _utcnow(), "route": route, "model": req.get("model", ""),
            "discovery_key": dk, "fingerprint": point.get("fingerprint"),
            "point_id": point["point_id"], "point_version": point.get("point_version", 1),
            "decision_id": did, "mode": "forwarded",
            "raw_request": {"system": req.get("system", ""),
                            "user": req.get("user", "") or req.get("conversation", "")},
            "raw_response": answer, "parsed_option": parsed, "unparsed": unparsed,
            "format_compared": compared, "format_match": matched_fmt,
            "latency_ms": (time.monotonic() - plan["t0"]) * 1000,
            "upstream_ms": upstream_ms, "usage": usage, "status": resp.status,
            "data_class": gw.cfg["gateway"]["trace_data_class"]})
        if not stream:
            # Row + trace are durable before the client sees the answer, so
            # x-decide-id is usable (do_outcome) the moment it arrives.
            self._send_full(resp, extra, payload)
        if plan.get("audit"):
            gw.svc._drift_check(point["point_id"])
        if compared and point.get("phase") == "serve":
            gw._post_serve_format_check(point)

    def _extract_answer(self, route: str, resp, payload: bytes) -> tuple[str | None, dict]:
        encoding = (resp.getheader("Content-Encoding") or "").lower()
        raw = payload
        if encoding == "gzip" and payload:
            try:
                raw = gzip.decompress(payload)
            except OSError:
                raw = payload
        ctype = (resp.getheader("Content-Type") or "").lower()
        if "text/event-stream" in ctype:
            if route == "openai":
                return sse_answer_openai(raw)
            return sse_answer_anthropic(raw)
        if route == "openai":
            return answer_text_openai(raw)
        return answer_text_anthropic(raw)

    def _forward_generic(self, route: str, upath: str) -> None:
        try:
            check_upstream_path(upath)
        except GatewayError as e:
            self._send_json(e.status, {"error": e.code})
            return
        try:
            raw_body = self._read_body()
        except GatewayError as e:
            self._send_json(e.status, {"error": e.code})
            return
        try:
            conn, resp, _ = self._call_upstream(route, upath, self.command, raw_body,
                                                False)
        except Exception:
            self._send_json(502, {"error": "upstream_unreachable"})
            return
        try:
            self._copy_response(resp)
        finally:
            conn.close()


def serve(cfg=None, jev=None, rng=None, block=True):
    """Start the gateway. Returns (server, thread) when block=False (tests)."""
    from .config import ConfigError
    cfg = cfg or load_config()
    check_gateway_local(cfg["gateway"]["host"])
    if jev is not None:
        gw = Gateway(cfg, Store(data_root(cfg)), jev, rng=rng)
    else:
        from .deciders import build_deciders
        store = Store(data_root(cfg))
        deciders, _ = build_deciders(cfg, store)
        gw = Gateway(cfg, store, deciders, rng=rng)
    try:
        server = ThreadingHTTPServer((cfg["gateway"]["host"], cfg["gateway"]["port"]),
                                     Handler)
    except OSError as e:
        print(f"decide-gateway: cannot bind: {e}", file=sys.stderr)
        raise SystemExit(1)
    server.daemon_threads = True
    server.gateway = gw
    Handler.gateway = gw
    gw.server_port = server.server_address[1]
    if block:
        print(f"decide-gateway on http://{cfg['gateway']['host']}:{gw.server_port} "
              f"(openai/anthropic)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        return server, None
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def main() -> None:
    from .config import ConfigError
    try:
        serve()
    except ConfigError as e:
        print(f"decide-gateway: refusing to start: {e.code}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

