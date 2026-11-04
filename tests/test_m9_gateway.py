"""M9 tests: gateway (§14), stub upstream + fake deciders (T-G1..T-G20)."""
import gzip
import http.client
import json
import os
import random
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from decideonce import gateway as GW
from decideonce import gwformat as GF
from decideonce import gwtemplate as GT
from decideonce.config import ConfigError, load_config
from decideonce.deciders import build_deciders
from decideonce.trainer import FakeTrainer
from decideonce.service import Service
from decideonce.store import Store

SYS = ("You are a ticket classifier. Answer with exactly one word: "
       "approve, deny, or escalate.")
PREFIX = "Classify this ticket: "
MODEL = "stub"
OPTIONS = ["approve", "deny", "escalate"]


def make_cfg(tmp_path, **over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["store"]["project"] = "proj"
    cfg["gateway"]["port"] = 0
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    for section, vals in over.items():
        cfg[section].update(vals)
    return cfg


def make_service(tmp_path, **over):
    cfg = make_cfg(tmp_path, **over)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    return Service(cfg, store, deciders, rng=random.Random(0),
                   trainer=FakeTrainer(cfg)), store, cfg


def _train_private(svc, pid):
    """Smoke then full FakeTrainer job to done."""
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "smoke", r
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "smoke_done", r
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full", r
    svc.trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": pid, "force": True, "confirm": True})
    assert r["ok"] and r["status"] == "done", r


def ticket_point(**kw):
    p = {"point_id": "gw.tickets", "workflow": "support",
         "question": "What should happen to this ticket?",
         "kind": "choice", "options": list(OPTIONS),
         "context_fields": {}, "required_fields": [], "rules": [],
         "data_class": "public", "source": "gateway",
         "route": "openai", "model": MODEL,
         "template": {"system": SYS, "user_prefix": PREFIX, "user_suffix": ""},
         "output_format": {"type": "enum_text"},
         "extract": {}, "multi_turn": False}
    p.update(kw)
    return p


def oai_request(user, system=SYS, model=MODEL, **kw):
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": user})
    body = {"model": model, "messages": msgs}
    body.update(kw)
    return body


# --------------------------------------------------------------------------
# stub upstream
# --------------------------------------------------------------------------

class StubUpstream:
    """Scripted OpenAI/Anthropic-shaped upstream. `answer_fn(user_text) -> str`
    labels requests; `mode` switches the response shape."""

    def __init__(self):
        self.requests: list[dict] = []
        self.answer_fn = lambda user: "approve"
        self.mode = "json"  # json | sse | gzip | error500 | redirect | sse_wait
        self.status = 200
        self.extra_headers: dict[str, str] = {}
        self.sse_first: list[bytes] = []
        self.sse_rest: list[bytes] = []
        self.sse_gate: threading.Event | None = None
        stub = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _handle(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                stub.requests.append({"path": self.path,
                                      "headers": dict(self.headers), "body": body})
                if stub.mode == "redirect":
                    payload = b"redirect-body"
                    self.send_response(302)
                    self.send_header("Location", "/v1/other")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                if stub.mode == "error500":
                    payload = b'{"error": "stub broke"}'
                    self.send_response(500)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("x-stub", "yes")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return
                try:
                    req = json.loads(body.decode()) if body else {}
                except ValueError:
                    req = {}
                user = ""
                for m in req.get("messages", []):
                    if m.get("role") == "user":
                        c = m.get("content")
                        user = c if isinstance(c, str) else ""
                answer = stub.answer_fn(user)
                if stub.mode == "sse_wait":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    for c in stub.sse_first:
                        self.wfile.write(c)
                        self.wfile.flush()
                    assert stub.sse_gate is not None
                    stub.sse_gate.wait(timeout=15)
                    for c in stub.sse_rest:
                        self.wfile.write(c)
                        self.wfile.flush()
                    self.close_connection = True
                    return
                if self.path.endswith("/v1/messages"):
                    obj = {"id": "msg_stub", "type": "message", "role": "assistant",
                           "model": req.get("model", MODEL),
                           "content": [{"type": "text", "text": answer}],
                           "stop_reason": "end_turn", "stop_sequence": None,
                           "usage": {"input_tokens": 11, "output_tokens": 3}}
                else:
                    obj = {"id": "chatcmpl-stub", "object": "chat.completion",
                           "created": 1, "model": req.get("model", MODEL),
                           "choices": [{"index": 0, "message": {"role": "assistant",
                                                               "content": answer},
                                        "finish_reason": "stop"}],
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2}}
                payload = json.dumps(obj).encode()
                headers = {"Content-Type": "application/json", "x-stub": "yes"}
                headers.update(stub.extra_headers)
                if stub.mode == "gzip":
                    payload = gzip.compress(payload)
                    headers["Content-Encoding"] = "gzip"
                if stub.mode == "sse":
                    text = answer
                    first = {"choices": [{"delta": {"content": text}}]}
                    payload = (f"data: {json.dumps(first)}\n\n"
                               "data: [DONE]\n\n").encode()
                    headers = {"Content-Type": "text/event-stream"}
                self.send_response(stub.status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            do_GET = _handle
            do_POST = _handle

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


@pytest.fixture()
def stub():
    s = StubUpstream()
    yield s
    s.close()


def start_gateway(cfg, **kw):
    server, thread = GW.serve(cfg, rng=random.Random(0), block=False, **kw)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    return server, thread, base


@pytest.fixture()
def gw_factory():
    servers = []

    def _start(cfg, **kw):
        server, thread, base = start_gateway(cfg, **kw)
        servers.append(server)
        return server, thread, base

    yield _start
    for s in servers:
        s.shutdown()
        s.server_close()


def gw_post(base, path, body, headers=None):
    """POST dict (as JSON) or raw bytes; returns (status, headers, body)."""
    host, port = base.rsplit(":", 1)
    host = host.split("://", 1)[1]
    conn = http.client.HTTPConnection(host, int(port), timeout=15)
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    hdrs = dict(headers or {})
    conn.request("POST", path, body=data, headers=hdrs)
    resp = conn.getresponse()
    payload = resp.read()
    out = (resp.status, dict(resp.getheaders()), payload)
    conn.close()
    return out


def gw_get(base, path, headers=None):
    host, port = base.rsplit(":", 1)
    host = host.split("://", 1)[1]
    conn = http.client.HTTPConnection(host, int(port), timeout=15)
    conn.request("GET", path, headers=dict(headers or {}))
    resp = conn.getresponse()
    out = (resp.status, dict(resp.getheaders()), resp.read())
    conn.close()
    return out


def ticket_call(base, text, headers=None, **kw):
    body = oai_request(PREFIX + text, **kw)
    return gw_post(base, "/openai/v1/chat/completions", body, headers)


# --------------------------------------------------------------------------
# T-G1: forwarding fidelity
# --------------------------------------------------------------------------

def test_T_G1_forwarded_byte_for_byte_and_gzip(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    # exact bytes incl. odd spacing must reach the stub unchanged
    raw = b'{"model" : "stub" , "messages" : [{"role":"user","content":"hi"}]}'
    status, headers, payload = gw_post(base, "/openai/v1/chat/completions", raw)
    assert status == 200
    assert stub.requests[-1]["body"] == raw
    assert stub.requests[-1]["path"] == "/v1/chat/completions"
    # status + headers preserved (minus hop-by-hop)
    assert headers.get("x-stub") == "yes"
    assert "transfer-encoding" not in {k.lower() for k in headers}
    # gzip body passes through untouched
    stub.mode = "gzip"
    stub.requests.clear()
    status, headers, payload = gw_post(base, "/openai/v1/chat/completions", raw)
    assert status == 200
    assert headers.get("Content-Encoding") == "gzip"
    assert json.loads(gzip.decompress(payload))["object"] == "chat.completion"
    # unregistered + log_unmatched=false: no trace rows, only the counter
    assert store.traces_by_discovery("nope") == []
    assert store.counter_get("unregistered", f"openai:{MODEL}:200") == 2


# --------------------------------------------------------------------------
# T-G2: bind/host/origin/token
# --------------------------------------------------------------------------

def test_T_G2_binds_refused():
    for host in ("localhost", "0.0.0.0", "::"):
        with pytest.raises(ConfigError) as ei:
            GW.check_gateway_local(host)
        assert ei.value.code == "gateway_not_local"
    GW.check_gateway_local("127.0.0.1")
    GW.check_gateway_local("::1")


def test_T_G2_host_origin_token(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    cfg["gateway"]["local_token"] = "sekrit"
    _, _, base = gw_factory(cfg)
    port = int(base.rsplit(":", 1)[1])

    def raw(payload: bytes) -> bytes:
        s = socket.create_connection(("127.0.0.1", port), timeout=10)
        s.sendall(payload)
        chunks = []
        try:
            while True:
                b = s.recv(65536)
                if not b:
                    break
                chunks.append(b)
        except socket.timeout:
            pass
        s.close()
        return b"".join(chunks)

    body = b"{}"
    # bad Host -> 403
    resp = raw(b"POST /openai/v1/chat/completions HTTP/1.1\r\nHost: evil.com\r\n"
               b"Content-Length: 2\r\nConnection: close\r\n\r\n" + body)
    assert resp.split(b"\r\n", 1)[0].startswith(b"HTTP/1.1 403")
    # Origin -> 403
    resp = raw(("POST /openai/v1/chat/completions HTTP/1.1\r\n"
                f"Host: 127.0.0.1:{port}\r\n"
                "Origin: https://evil.test\r\n"
                "Content-Length: 2\r\nConnection: close\r\n\r\n").encode() + body)
    assert resp.split(b"\r\n", 1)[0].startswith(b"HTTP/1.1 403")
    # token enforced
    status, _, _ = gw_post(base, "/openai/v1/chat/completions", {"model": MODEL})
    assert status == 403
    status, _, _ = gw_post(base, "/openai/v1/chat/completions", {"model": MODEL},
                           {"x-decideonce-token": "wrong"})
    assert status == 403
    # right token: forwarded, and stripped before forwarding
    stub.requests.clear()
    status, _, _ = gw_post(base, "/openai/v1/chat/completions",
                           oai_request(PREFIX + "hi"),
                           {"x-decideonce-token": "sekrit"})
    assert status == 200
    sent = {k.lower(): v for k, v in stub.requests[-1]["headers"].items()}
    assert "x-decideonce-token" not in sent


# --------------------------------------------------------------------------
# T-G3: templates
# --------------------------------------------------------------------------

def insert_trace(store, system, user, answer, route="openai", model=MODEL,
                 ts="2026-01-01T00:00:00Z"):
    store.trace_insert({
        "ts": ts, "route": route, "model": model,
        "discovery_key": GT.discovery_key(route, model, system),
        "mode": "forwarded",
        "raw_request": {"system": system, "user": user},
        "raw_response": answer, "latency_ms": 1.0, "upstream_ms": 0.5,
        "usage": {"prompt_tokens": 10, "completion_tokens": 2}, "status": 200})


def test_T_G3_learn_template_matches_new_tickets(tmp_path):
    svc, store, _ = make_service(tmp_path)
    middles = [f"ticket {i} about order {1000 + i} refund request" for i in range(30)]
    for i, m in enumerate(middles):
        insert_trace(store, SYS, PREFIX + m, OPTIONS[i % 3])
    r = svc.do_analyze({"gateway": True})
    assert r["ok"], r
    assert len(r["groups"]) == 1
    clusters = r["groups"][0]["clusters"]
    assert len(clusters) == 1
    cl = clusters[0]
    assert cl["n"] == 30
    assert cl["template"]["system"] == SYS
    assert cl["template"]["user_prefix"] == PREFIX
    assert cl["template"]["user_suffix"] == ""
    assert cl["template_match_rate"] == 1.0
    assert sorted(cl["options"]) == sorted(OPTIONS)
    assert cl["output_format"]["type"] == "enum_text"
    assert "fingerprint" in cl
    # the learned template matches new tickets
    assert GT.template_matches(cl["template"], SYS, PREFIX + "a brand new ticket")


def test_T_G3_changed_instruction_is_variant(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    # baseline: one matched call logs a row
    status, _, _ = ticket_call(base, "order 5 refund")
    assert status == 200
    assert len(store.decisions_for_point("gw.tickets")) == 1
    # changed instruction in the system prompt (skeleton-preserving: v1 -> v2)
    sys2 = SYS.replace("classifier", "classifier v1")
    assert svc.do_register(ticket_point(
        point_id="gw.v", template={"system": sys2, "user_prefix": PREFIX,
                                   "user_suffix": ""}))["ok"]
    sys3 = sys2.replace("classifier v1", "classifier v2")
    assert GT.discovery_key("openai", MODEL, sys2) == GT.discovery_key(
        "openai", MODEL, sys3)
    status, _, _ = ticket_call(base, "order 6 refund", system=sys3)
    assert status == 200
    assert store.counter_get("fingerprint_variant", "gw.v") == 1
    assert len(store.decisions_for_point("gw.v")) == 0
    # changed instruction in the user prefix -> variant on gw.tickets
    body = oai_request("Categorize this ticket: order 7 refund")
    status, _, _ = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert store.counter_get("fingerprint_variant", "gw.tickets") == 1
    assert len(store.decisions_for_point("gw.tickets")) == 1


def test_T_G3_short_prefix_too_weak_and_boundary_trim(tmp_path):
    svc, store, _ = make_service(tmp_path)
    # no system prompt and a short prefix: not enough fixed text
    dk_weak = GT.discovery_key("openai", MODEL, "")
    for i in range(20):
        insert_trace(store, "", f"Q: rate it\nbody {i}\nEND", "approve")
    r = svc.do_analyze({"gateway": True, "discovery_key": dk_weak})
    assert r["ok"], r
    assert r["groups"][0]["clusters"][0]["error"] == "template_too_weak"
    # shared opening words of the inputs do not leak into the prefix
    for i in range(20):
        insert_trace(store, SYS, PREFIX + f"My order {i} never arrived, refund please",
                     OPTIONS[i % 3])
    dk = GT.discovery_key("openai", MODEL, SYS)
    r = svc.do_analyze({"gateway": True, "discovery_key": dk})
    assert r["ok"], r
    cl = r["groups"][0]["clusters"][0]
    assert cl["template"]["user_prefix"] == PREFIX, cl["template"]


def test_T_G3_overfit_and_two_clusters(tmp_path):
    svc, store, _ = make_service(tmp_path)
    # held-out mismatch: first 80% share a prefix the last 20% lack
    users = [PREFIX + f"ticket {i}" for i in range(16)]
    users += [f"Sort this ticket: ticket {i}" for i in range(16, 20)]
    for i, u in enumerate(users):
        insert_trace(store, SYS, u, OPTIONS[i % 3])
    dk = GT.discovery_key("openai", MODEL, SYS)
    r = svc.do_analyze({"gateway": True, "discovery_key": dk})
    assert r["ok"], r
    cl = r["groups"][0]["clusters"][0]
    assert cl["error"] == "template_overfit", cl
    assert cl["match_rate"] < 0.95
    # two templates under one short system prompt are split into two clusters
    sys_short = "You answer with one word."
    for i in range(20):
        insert_trace(store, sys_short, f"Classify this ticket:\nbody A-{i}", "approve")
        insert_trace(store, sys_short, f"Rate this ticket:\nbody B-{i}", "deny")
    dk2 = GT.discovery_key("openai", MODEL, sys_short)
    r = svc.do_analyze({"gateway": True, "discovery_key": dk2})
    assert r["ok"], r
    clusters = r["groups"][0]["clusters"]
    assert len(clusters) == 2, clusters
    assert {c["n"] for c in clusters} == {20}
    prefixes = {c["template"]["user_prefix"] for c in clusters}
    assert prefixes == {"Classify this ticket:\n", "Rate this ticket:\n"}


def test_T_G3_late_instructions_pinned(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    sys_long = "Background: " + "x" * 400 + " Return exactly one word."
    assert svc.do_register(ticket_point(
        point_id="gw.long",
        template={"system": sys_long, "user_prefix": PREFIX, "user_suffix": ""}))["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    # exact system incl. the late instruction matches and logs a row
    body = oai_request(PREFIX + "ticket 1", system=sys_long)
    status, _, _ = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert len(store.decisions_for_point("gw.long")) == 1
    # one word changed after character 400 -> forwarded, never answered
    sys_changed = sys_long.replace("exactly one word", "exactly one token")
    body = oai_request(PREFIX + "ticket 2", system=sys_changed)
    status, hdrs, _ = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert "x-decide-id" not in {k.lower() for k in hdrs}
    assert len(store.decisions_for_point("gw.long")) == 1


# --------------------------------------------------------------------------
# T-G4: non-answerable shapes are forwarded
# --------------------------------------------------------------------------

def test_T_G4_shapes_forwarded(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    stub.answer_fn = lambda user: "deny"

    def call(body):
        return gw_post(base, "/openai/v1/chat/completions", body)

    cases = [
        ("tools", oai_request(PREFIX + "t", tools=[{"type": "function"}])),
        ("n2", oai_request(PREFIX + "t", n=2)),
        ("logprobs", oai_request(PREFIX + "t", logprobs=True)),
        ("multi", {"model": MODEL, "messages": [
            {"role": "system", "content": SYS},
            {"role": "user", "content": PREFIX + "t1"},
            {"role": "assistant", "content": "deny"},
            {"role": "user", "content": PREFIX + "t2"}]}),
        ("json_schema_no_enum", oai_request(
            PREFIX + "t", response_format={"type": "json_schema", "json_schema": {
                "name": "d", "schema": {"type": "object",
                                        "properties": {"x": {"type": "string"}}}}})),
    ]
    for name, body in cases:
        status, hdrs, payload = call(body)
        assert status == 200, name
        # forwarded: the stub's answer comes back, with a logged row
        assert json.loads(payload)["choices"][0]["message"]["content"] == "deny", name
        assert hdrs.get("x-decide-mode") == "forwarded", name
    # image content has no extractable text: forwarded unmatched, no row
    status, hdrs, payload = call({"model": MODEL, "messages": [
        {"role": "system", "content": SYS},
        {"role": "user", "content": [{"type": "text", "text": PREFIX + "t"},
                                     {"type": "image_url",
                                      "image_url": {"url": "http://x/y.png"}}]}]})
    assert status == 200
    assert json.loads(payload)["choices"][0]["message"]["content"] == "deny"
    assert "x-decide-mode" not in {k.lower() for k in hdrs}
    rows = store.decisions_for_point("gw.tickets")
    assert len(rows) == len(cases)
    assert all(r["mode"] == "forwarded" for r in rows)
    assert all(r["teacher_choice"] == "deny" for r in rows)
    warnings = [w for r in rows for w in (r["warnings"] or [])]
    assert "shape_not_allowed" in warnings
    assert "multi_turn_not_enabled" in warnings
    assert "response_format_not_allowed" in warnings
    # anthropic thinking shape is forwarded too
    assert svc.do_register(ticket_point(
        point_id="gw.anth", route="anthropic",
        template={"system": SYS, "user_prefix": PREFIX, "user_suffix": ""}))["ok"]
    status, _, payload = gw_post(
        base, "/anthropic/v1/messages",
        {"model": MODEL, "system": SYS,
         "messages": [{"role": "user", "content": PREFIX + "t"}],
         "thinking": {"type": "enabled", "budget_tokens": 100},
         "max_tokens": 10})
    assert status == 200
    rows = store.decisions_for_point("gw.anth")
    assert len(rows) == 1 and rows[0]["mode"] == "forwarded"
    assert "shape_not_allowed" in (rows[0]["warnings"] or [])


def test_T_G4_stop_and_max_tokens_conflict(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    # rule-decided point (would auto-answer "escalate") to reach the gate
    p = ticket_point(
        point_id="gw.rule", context_fields={"amount": {"type": "number"}},
        extract={"amount": {"regex": r"amount: (\d+)", "group": 1, "cast": "number"}},
        rules=[{"id": "big", "if": [["amount", ">", 500]], "only": ["escalate"]}],
        output_format={"type": "enum_text", "surface": {"escalate": "escalate"}})
    assert svc.do_register(p)["ok"], svc.do_register(p)
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    stub.answer_fn = lambda user: "approve"
    _, _, base = gw_factory(cfg)
    # stop string present in the would-be answer -> forwarded
    status, hdrs, payload = ticket_call(
        base, "ticket amount: 600", headers=None, stop=["escalate"])
    assert status == 200
    assert hdrs.get("x-decide-mode") == "forwarded"
    assert json.loads(payload)["choices"][0]["message"]["content"] == "approve"
    # max_tokens too small for the would-be answer -> forwarded
    status, hdrs, _ = ticket_call(base, "ticket amount: 601", max_tokens=1)
    assert status == 200
    assert hdrs.get("x-decide-mode") == "forwarded"
    rows = store.decisions_for_point("gw.rule")
    assert len(rows) == 2
    warnings = [w for r in rows for w in (r["warnings"] or [])]
    assert "stop_conflict" in warnings
    assert "max_tokens_conflict" in warnings


# --------------------------------------------------------------------------
# T-G5: collect/shadow rows + unparsed exclusion
# --------------------------------------------------------------------------

def test_T_G5_collect_shadow_unparsed(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    stub.answer_fn = lambda user: "Approve!" if "good" in user else "no idea, sorry"
    _, _, base = gw_factory(cfg)
    status, _, _ = ticket_call(base, "good ticket")
    assert status == 200
    status, _, _ = ticket_call(base, "weird ticket")
    assert status == 200
    rows = store.decisions_for_point("gw.tickets")
    assert len(rows) == 2
    assert rows[0]["mode"] == "forwarded" and rows[0]["teacher_choice"] == "approve"
    assert rows[1]["unparsed"] is True and rows[1]["teacher_choice"] is None
    # unparsed rows never train, even with an outcome attached later
    from decideonce import labels as L
    assert L.resolve_label(rows[1])[1] == "unparsed"
    point = store.get_point("gw.tickets")
    assert L.labeled_rows(rows, point, 1.0, 0.5) != []
    assert all(not r.get("unparsed") for r, _, _, _ in
               L.labeled_rows(rows, point, 1.0, 0.5))
    # shadow phase still forwards (fit or not)
    point["phase"] = "shadow"
    store.put_point(point)
    status, hdrs, _ = ticket_call(base, "good ticket again")
    assert status == 200
    assert hdrs.get("x-decide-mode") == "forwarded"
    rows = store.decisions_for_point("gw.tickets")
    assert rows[-1]["phase"] == "shadow" and rows[-1]["teacher_choice"] == "approve"


# --------------------------------------------------------------------------
# T-G10: upstream errors pass through unchanged
# --------------------------------------------------------------------------

def test_T_G10_upstream_error_passthrough(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    stub.mode = "error500"
    _, _, base = gw_factory(cfg)
    status, hdrs, payload = ticket_call(base, "ticket 1")
    assert status == 500
    assert payload == b'{"error": "stub broke"}'
    assert hdrs.get("x-stub") == "yes"
    assert hdrs.get("x-decide-mode") == "forwarded"
    # trace only, no decision row: errors are transport, not labels
    assert store.decisions_for_point("gw.tickets") == []
    traces = store.traces_for_point("gw.tickets")
    assert len(traces) == 1 and traces[0]["status"] == 500


# --------------------------------------------------------------------------
# T-G11: extractor feeds rules; missing guard forwards
# --------------------------------------------------------------------------

def test_T_G11_extract_rule_and_guard(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    p = ticket_point(
        point_id="gw.amt", context_fields={"amount": {"type": "number"}},
        extract={"amount": {"regex": r"amount: (\d+)", "group": 1, "cast": "number"}},
        rules=[{"id": "big", "if": [["amount", ">", 500]], "only": ["escalate"]}])
    assert svc.do_register(p)["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    stub.answer_fn = lambda user: "approve"
    _, _, base = gw_factory(cfg)
    # amount extracted -> rule fires -> auto "escalate" (stub would say approve)
    body = oai_request(PREFIX + "ticket amount: 600")
    status, hdrs, payload = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert hdrs.get("x-decide-mode") == "rule"
    assert json.loads(payload)["choices"][0]["message"]["content"] == "escalate"
    # no amount -> guard (rule field) missing -> forwarded, never auto
    body = oai_request(PREFIX + "ticket with no amount")
    status, hdrs, payload = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert hdrs.get("x-decide-mode") == "forwarded"
    assert json.loads(payload)["choices"][0]["message"]["content"] == "approve"
    assert store.counter_get("missing_guard_field", "gw.amt") == 1
    rows = store.decisions_for_point("gw.amt")
    assert [r["mode"] for r in rows] == ["rule", "forwarded"]
    assert rows[1]["teacher_choice"] == "approve"


def test_D_extract_cast_rejects_nonfinite():
    assert GW._cast("nan", "number") is None
    assert GW._cast("inf", "number") is None
    assert GW._cast("nan", "integer") is None
    assert GW._cast("3.5", "number") == 3.5
    assert GW._cast("3.0", "integer") == 3
    assert GW.extract_fields(
        {"amount": {"regex": r"amount: (\S+)", "group": 1,
                    "cast": "number"}},
        "ticket amount: nan") == {}


# --------------------------------------------------------------------------
# serve-ready fixture (fake deciders plant signal through real HTTP)
# --------------------------------------------------------------------------

def hint_ticket(i, options=("approve", "deny")):
    label = options[i % len(options)]
    return f"ticket {i}\nlabel_hint: {label}", label


def hint_answer_fn(user):
    import re
    m = re.search(r"label_hint: (\w+)", user)
    return m.group(1) if m else "approve"


def serve_ready_gateway(tmp_path, stub, gw_factory, n_collect=175, n_shadow=30,
                        funky=(), unparsed=(), **cfg_over):
    """Full collect -> train -> shadow run over HTTP. `funky` call indices get a
    parseable-but-mismatching surface; `unparsed` indices get junk answers."""
    # §14.13 demo config + SMALL fit/promote budgets: 30 shadow rows fit
    # at 20 (10/10 per option, no teacher_only) leaving 10 post-fit rows
    merged: dict = {"serve": {"min_margin_vs_random": 0, "min_shadow": 10,
                               "min_auto_support": 10},
                    "train": {"min_train": 5, "min_examples": 10},
                    "hosted_jev": {"fit_rows": 20, "min_compare": 10}}
    for section, vals in cfg_over.items():
        merged.setdefault(section, {}).update(vals)
    svc, store, cfg = make_service(tmp_path, **merged)
    p = ticket_point(point_id="gw.serve", options=["approve", "deny"],
                     target_error=0.3,
                     output_format={"type": "enum_text",
                                    "surface": {"approve": "approve", "deny": "deny"}},
                     # the planted signal, extracted into the decider state
                     extract={"label_hint": {"regex": "label_hint: (\\w+)",
                                             "group": 1, "cast": "string"}})
    assert svc.do_register(p)["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    calls = {"n": 0}

    def answer(user):
        i = calls["n"]
        calls["n"] += 1
        label = hint_answer_fn(user)
        if i in set(unparsed):
            return "no idea, sorry"
        if i in set(funky):
            return label.upper() + "!!!"
        return label

    stub.answer_fn = answer
    _, _, base = gw_factory(cfg)
    for i in range(n_collect):
        text, _ = hint_ticket(i)
        status, _, _ = ticket_call(base, text)
        assert status == 200
    _train_private(svc, "gw.serve")
    assert svc.do_promote({"point_id": "gw.serve", "phase": "shadow"})["ok"]
    for i in range(n_collect, n_collect + n_shadow):
        text, _ = hint_ticket(i)
        status, _, _ = ticket_call(base, text)
        assert status == 200
    return svc, store, cfg, base


# --------------------------------------------------------------------------
# T-G6: synthesized response shapes
# --------------------------------------------------------------------------

def test_T_G6_synth_openai_auto_exact(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(
        tmp_path, stub, gw_factory, serve={"audit_rate": 0.0})
    r = svc.do_promote({"point_id": "gw.serve", "phase": "serve", "confirm": True})
    assert r["ok"], r
    text, label = hint_ticket(1001)
    status, hdrs, payload = ticket_call(base, text)
    assert status == 200
    assert hdrs["Content-Type"] == "application/json"
    assert hdrs.get("x-decide-mode") == "auto"
    assert hdrs.get("x-decide-point") == "gw.serve"
    did = hdrs.get("x-decide-id")
    assert did and store.get_decision(did)["mode"] == "auto"
    obj = json.loads(payload)
    assert obj["id"].startswith("chatcmpl-decide-")
    assert obj["object"] == "chat.completion"
    assert isinstance(obj["created"], int)
    assert obj["model"] == MODEL  # requested model echoed
    assert obj["choices"] == [{"index": 0, "message": {"role": "assistant",
                                                      "content": label,
                                                      "refusal": None},
                               "logprobs": None, "finish_reason": "stop"}]
    assert obj["usage"] == {"prompt_tokens": 0, "completion_tokens": 0,
                            "total_tokens": 0}
    assert obj["system_fingerprint"] == "decideonce-gateway"
    assert store.get_decision(did)["final_choice"] == label


def test_T_G6_synth_anthropic_exact(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    p = ticket_point(point_id="gw.arule", route="anthropic",
                     context_fields={"amount": {"type": "number"}},
                     extract={"amount": {"regex": r"amount: (\d+)", "group": 1,
                                         "cast": "number"}},
                     rules=[{"id": "big", "if": [["amount", ">", 500]],
                             "only": ["escalate"]}],
                     output_format={"type": "enum_text",
                                    "surface": {"escalate": "escalate"}})
    assert svc.do_register(p)["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    status, hdrs, payload = gw_post(
        base, "/anthropic/v1/messages",
        {"model": MODEL, "system": SYS,
         "messages": [{"role": "user",
                       "content": PREFIX + "ticket amount: 700"}],
         "max_tokens": 10})
    assert status == 200
    assert hdrs.get("x-decide-mode") == "rule"
    assert hdrs.get("x-decide-point") == "gw.arule"
    obj = json.loads(payload)
    assert obj["id"].startswith("msg_decide_")
    assert obj["type"] == "message" and obj["role"] == "assistant"
    assert obj["model"] == MODEL
    assert obj["content"] == [{"type": "text", "text": "escalate"}]
    assert obj["stop_reason"] == "end_turn"
    assert obj["stop_sequence"] is None
    assert obj["usage"] == {"input_tokens": 0, "output_tokens": 0}


# --------------------------------------------------------------------------
# T-G7: synthesized SSE sequences
# --------------------------------------------------------------------------

def test_T_G7_synth_sse_exact(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(
        tmp_path, stub, gw_factory, serve={"audit_rate": 0.0})
    assert svc.do_promote({"point_id": "gw.serve", "phase": "serve",
                           "confirm": True})["ok"]
    text, label = hint_ticket(2001)
    # OpenAI stream with include_usage
    body = oai_request(PREFIX + text, stream=True,
                       stream_options={"include_usage": True})
    status, hdrs, payload = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert hdrs["Content-Type"] == "text/event-stream"
    assert hdrs.get("x-decide-mode") == "auto"
    lines = [ln for ln in payload.decode().split("\n") if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[5:].strip()) for ln in lines[:-1]]
    assert len(chunks) == 4
    assert {c["id"] for c in chunks} == {chunks[0]["id"]}
    assert all(c["object"] == "chat.completion.chunk" for c in chunks)
    assert all(c["model"] == MODEL for c in chunks)
    assert chunks[0]["choices"] == [{"index": 0, "delta": {"role": "assistant",
                                                          "content": ""},
                                     "finish_reason": None}]
    assert chunks[1]["choices"] == [{"index": 0, "delta": {"content": label},
                                     "finish_reason": None}]
    assert chunks[2]["choices"] == [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}]
    assert chunks[3]["choices"] == []
    assert chunks[3]["usage"] == {"prompt_tokens": 0, "completion_tokens": 0}
    # Anthropic stream via rule point (same wire code as auto)
    p = ticket_point(point_id="gw.srule", route="anthropic",
                     context_fields={"amount": {"type": "number"}},
                     extract={"amount": {"regex": r"amount: (\d+)", "group": 1,
                                         "cast": "number"}},
                     rules=[{"id": "big", "if": [["amount", ">", 500]],
                             "only": ["escalate"]}])
    assert svc.do_register(p)["ok"]
    status, hdrs, payload = gw_post(
        base, "/anthropic/v1/messages",
        {"model": MODEL, "system": SYS, "stream": True,
         "messages": [{"role": "user",
                       "content": PREFIX + "ticket amount: 800"}],
         "max_tokens": 10})
    assert status == 200
    assert hdrs["Content-Type"] == "text/event-stream"
    events = []
    for block in payload.decode().split("\n\n"):
        if not block.strip():
            continue
        kind, data = block.split("\n", 1)
        events.append((kind.split(": ", 1)[1], json.loads(data.split(": ", 1)[1])))
    assert [e for e, _ in events] == ["message_start", "content_block_start",
                                      "content_block_delta", "content_block_stop",
                                      "message_delta", "message_stop"]
    by_name = dict(events)
    assert by_name["message_start"]["message"]["content"] == []
    assert by_name["message_start"]["message"]["usage"] == {"input_tokens": 0,
                                                            "output_tokens": 0}
    assert by_name["content_block_delta"]["delta"] == {"type": "text_delta",
                                                       "text": "escalate"}
    assert by_name["message_delta"]["delta"] == {"stop_reason": "end_turn",
                                                 "stop_sequence": None}
    assert by_name["message_delta"]["usage"] == {"output_tokens": 0}


# --------------------------------------------------------------------------
# T-G8: format promotion gate
# --------------------------------------------------------------------------

def test_T_G8_count_pass_and_version_reset(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(tmp_path, stub, gw_factory)
    fe = GW.format_evidence(cfg, store, store.get_point("gw.serve"))
    assert fe["n_compared"] == 205 and fe["n_match"] == 205
    assert fe["wilson_lower"] >= 0.97
    assert fe["unparsed_rate"] == 0.0
    r = svc.do_promote({"point_id": "gw.serve", "phase": "serve", "confirm": True})
    assert r["ok"], r  # 205 compared passes the default format_min_n=200
    # version bump resets the counters (0 compared < 200) -> refused again
    p = store.get_point("gw.serve")
    p["extract"] = {"amount": {"regex": "x", "group": 0, "cast": "string"}}
    assert svc.do_register(p)["ok"]
    p2 = store.get_point("gw.serve")
    assert p2["point_version"] == 2 and p2["phase"] == "shadow"
    r = svc.do_promote({"point_id": "gw.serve", "phase": "serve", "confirm": True})
    assert r["ok"] is False and "gateway_format" in r["failed"], r


def test_T_G8_wilson_bounds_and_unparsed(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(tmp_path, stub, gw_factory)
    # top up from 205 to exactly the crafted totals via direct traces
    dk = GT.discovery_key("openai", MODEL, SYS)
    n0 = GW.format_evidence(cfg, store, store.get_point("gw.serve"))["n_compared"]
    assert n0 == 205
    # 199/200 passes at 0.97: add nothing, drop 5 -> use a fresh store slice:
    # instead verify the bound math on crafted evidence sets
    fe = GW.format_evidence(cfg, store, store.get_point("gw.serve"))
    assert fe["wilson_lower"] is not None
    # now break it: 6 mismatches over the next 95 forwarded calls
    for i in range(95):
        store.trace_insert({"ts": "2026-01-02T00:00:00Z", "route": "openai",
                            "model": MODEL, "discovery_key": dk,
                            "point_id": "gw.serve", "point_version": 1,
                            "mode": "forwarded", "raw_request": {"system": SYS, "user": "u"},
                            "raw_response": "approve", "parsed_option": "approve",
                            "format_compared": True, "format_match": i >= 6,
                            "status": 200})
    fe = GW.format_evidence(cfg, store, store.get_point("gw.serve"))
    assert fe["n_compared"] == 300 and fe["n_match"] == 294
    from decideonce import stats as S
    assert abs(fe["wilson_lower"] - (1.0 - S.wilson_upper(6, 300))) < 1e-9
    r = svc.do_promote({"point_id": "gw.serve", "phase": "serve", "confirm": True})
    assert r["ok"] is False and "gateway_format" in r["failed"], r
    # exact boundary: fresh point with exactly 199/200 then 198/200
    for n_match in (199, 198):
        svc2, store2, cfg2 = make_service(tmp_path / f"b{n_match}")
        p = ticket_point(point_id="gw.b", options=["approve", "deny"])
        assert svc2.do_register(p)["ok"]
        dk2 = GT.discovery_key("openai", MODEL, SYS)
        for i in range(200):
            store2.trace_insert({"ts": "2026-01-02T00:00:00Z", "route": "openai",
                                 "model": MODEL, "discovery_key": dk2,
                                 "point_id": "gw.b", "point_version": 1,
                                 "mode": "forwarded",
                                 "raw_request": {"system": SYS, "user": "u"},
                                 "raw_response": "approve", "parsed_option": "approve",
                                 "format_compared": True, "format_match": i < n_match,
                                 "status": 200})
        fe2 = GW.format_evidence(cfg2, store2, store2.get_point("gw.b"))
        if n_match == 199:
            assert fe2["wilson_lower"] >= 0.97, fe2
        else:
            assert fe2["wilson_lower"] < 0.97, fe2
    # unparsed rate above max_unparsed fails the gate
    svc3, store3, _ = make_service(tmp_path / "unparsed")
    assert svc3.do_register(ticket_point(point_id="gw.u"))["ok"]
    for i in range(200):
        store3.trace_insert({"ts": "2026-01-02T00:00:00Z", "route": "openai",
                             "model": MODEL, "discovery_key": dk,
                             "point_id": "gw.u", "point_version": 1,
                             "mode": "forwarded",
                             "raw_request": {"system": SYS, "user": "u"},
                             "raw_response": "huh?" if i < 5 else "approve",
                             "parsed_option": None if i < 5 else "approve",
                             "unparsed": i < 5, "format_compared": i >= 5,
                             "format_match": i >= 5, "status": 200})
    checks = svc3._serve_checks(store3.get_point("gw.u"), True)
    assert checks["gateway_format"]["pass"] is False, checks["gateway_format"]


def test_T_G8_mismatch_burst_demotes(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(
        tmp_path, stub, gw_factory, serve={"audit_rate": 1.0})
    assert svc.do_promote({"point_id": "gw.serve", "phase": "serve",
                           "confirm": True})["ok"]
    # every call is now an audit (forwarded): funky surfaces mismatch format
    stub.answer_fn = lambda user: hint_answer_fn(user).upper() + "!!!"
    for i in range(100):
        text, _ = hint_ticket(5000 + i)
        status, _, _ = ticket_call(base, text)
        assert status == 200
    assert store.get_point("gw.serve")["phase"] == "shadow"
    events = [e for e in store.get_point("gw.serve").get("events", [])
              if e.get("event") == "format_demotion"]
    assert len(events) == 1 and events[0]["n"] == 100


# --------------------------------------------------------------------------
# T-G9: json_enum formats
# --------------------------------------------------------------------------

def test_T_G9_json_enum_learn_parse_render(tmp_path):
    answers = [f'{{"decision": "{o}", "version": 1}}'
               for i, o in enumerate(OPTIONS * 10)]
    fmt = GF.learn_format(answers, OPTIONS)
    assert fmt is not None and fmt["type"] == "json_enum"
    assert fmt["path"] == "decision"
    assert json.loads(GF.render_answer("deny", fmt)) == {"decision": "deny",
                                                         "version": 1}
    # only the path field replaced, key order kept
    assert GF.render_answer("deny", fmt) == '{"decision": "deny", "version": 1}'
    assert GF.parse_answer('{"decision": "escalate", "version": 1}', fmt,
                           OPTIONS) == "escalate"
    assert GF.parse_answer("not json", fmt, OPTIONS) is None
    # a varying free-text field -> unsupported
    bad = [f'{{"decision": "{o}", "reason": "because {i}"}}'
           for i, o in enumerate(OPTIONS * 10)]
    assert GF.learn_format(bad, OPTIONS) is None
    # and registration refuses it once 20+ such answers are logged
    svc, store, _ = make_service(tmp_path)
    for i, a in enumerate(bad[:25]):
        insert_trace(store, SYS, PREFIX + f"ticket {i}", a)
    r = svc.do_register(ticket_point(
        output_format={"type": "json_enum", "path": "decision", "skeleton": {}}))
    assert r["ok"] is False and r["error"] == "format_unsupported", r


# --------------------------------------------------------------------------
# T-G12: canary key never persisted
# --------------------------------------------------------------------------

def test_T_G12_canary_never_stored(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    canary = "sk-canary-g12-9f8e7d"
    hdrs = {"authorization": f"Bearer {canary}", "x-api-key": canary}
    # normal matched + unmatched forwards carry the key upstream
    ticket_call(base, "ticket 1", headers=hdrs)
    gw_post(base, "/openai/v1/chat/completions", oai_request("unmatched", system="other"),
            hdrs)
    # an upstream error with the key
    stub.mode = "error500"
    ticket_call(base, "ticket 2", headers=hdrs)
    stub.mode = "json"
    # exception paths: malformed JSON, bad path
    gw_post(base, "/openai/v1/chat/completions", b"{oops", hdrs)
    gw_get(base, "/openai/v1/../x", hdrs)
    assert stub.requests, "stub saw nothing"
    assert any(canary in str(r["headers"]) for r in stub.requests)
    # scan every file the gateway+MCP processes wrote
    found = []
    for root, _, files in os.walk(tmp_path):
        for f in files:
            p = Path(root) / f
            if canary.encode() in p.read_bytes():
                found.append(str(p))
    assert found == []


# --------------------------------------------------------------------------
# T-G13: unregistered never answered; serve needs confirm
# --------------------------------------------------------------------------

def test_T_G13_unregistered_never_auto(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(
        tmp_path, stub, gw_factory, serve={"audit_rate": 0.0})
    r = svc.do_promote({"point_id": "gw.serve", "phase": "serve"})
    assert r["ok"] is False and "confirm" in r["failed"], r
    assert svc.do_promote({"point_id": "gw.serve", "phase": "serve",
                           "confirm": True})["ok"]
    # same discovery key, unknown template, strong signal: still forwarded
    stub.answer_fn = lambda user: "deny"
    text, _ = hint_ticket(3001)
    body = oai_request("Something else entirely: " + text)
    status, hdrs, payload = gw_post(base, "/openai/v1/chat/completions", body)
    assert status == 200
    assert "x-decide-mode" not in {k.lower() for k in hdrs}
    assert json.loads(payload)["choices"][0]["message"]["content"] == "deny"
    assert store.counter_get("fingerprint_variant", "gw.serve") == 1


# --------------------------------------------------------------------------
# T-G14: audits are forwarded with teacher labels
# --------------------------------------------------------------------------

def test_T_G14_audits_forwarded_with_teacher_choice(tmp_path, stub, gw_factory):
    svc, store, cfg, base = serve_ready_gateway(
        tmp_path, stub, gw_factory, serve={"audit_rate": 1.0})
    assert svc.do_promote({"point_id": "gw.serve", "phase": "serve",
                           "confirm": True})["ok"]
    text, label = hint_ticket(4001)
    status, hdrs, _ = ticket_call(base, text)
    assert status == 200
    assert hdrs.get("x-decide-mode") == "forwarded"
    row = store.get_decision(hdrs["x-decide-id"])
    assert row["mode"] == "forwarded" and row["audit"] is True
    assert row["teacher_choice"] == label


# --------------------------------------------------------------------------
# T-G15: JEV down / timeout
# --------------------------------------------------------------------------

def test_T_G15_jev_down_and_timeout(tmp_path, stub, gw_factory):
    from decideonce.deciders.base import DeciderError
    from decideonce.deciders.hosted import FakeHostedTransport
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    # tiered collect makes no model call: shadow reaches the decider
    assert svc.do_promote({"point_id": "gw.tickets",
                           "phase": "shadow"})["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    deciders, _ = build_deciders(cfg, store)

    def boom(body):
        raise DeciderError("timeout", "scripted outage")

    deciders["hosted_jev"].transport = FakeHostedTransport(fn=boom)
    _, _, base = gw_factory(cfg, deciders=deciders)
    status, hdrs, payload = ticket_call(base, "ticket 1")
    assert status == 200  # the gateway never fails a request because of JEV
    assert hdrs.get("x-decide-mode") == "forwarded"
    assert store.counter_get("jev_unavailable", "gw.tickets") == 1
    row = store.get_decision(hdrs["x-decide-id"])
    assert "jev_unavailable" in (row["warnings"] or [])
    # slow JEV past jev_timeout_s -> same path (direct Gateway, no HTTP)
    class SlowJev:
        def hidden(self, *a):
            time.sleep(0.3)
            raise AssertionError("must time out first")

        def decide(self, *a):
            time.sleep(0.3)
            raise AssertionError("must time out first")

    cfg2 = make_cfg(tmp_path / "slow", gateway={"jev_timeout_s": 0.05})
    store2 = Store(tmp_path / "slow" / "proj")
    deciders2, _ = build_deciders(cfg2, store2)
    svc2 = Service(cfg2, store2, deciders2, rng=random.Random(0),
                   trainer=FakeTrainer(cfg2))
    assert svc2.do_register(ticket_point())["ok"]
    assert svc2.do_promote({"point_id": "gw.tickets",
                            "phase": "shadow"})["ok"]
    gw = GW.Gateway(cfg2, store2, {"hosted_jev": SlowJev()})
    req = GW.parse_openai(oai_request(PREFIX + "ticket 2"))
    plan = gw.handle_decision("openai", req, b"{}")
    assert plan["action"] == "forward" and plan["point"] is not None
    assert "jev_unavailable" in plan["warnings"]
    assert store2.counter_get("jev_unavailable", "gw.tickets") == 1


# --------------------------------------------------------------------------
# T-G16: forwarded SSE streams unbuffered
# --------------------------------------------------------------------------

def test_T_G16_forwarded_sse_unbuffered(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    port = int(base.rsplit(":", 1)[1])
    stub.mode = "sse_wait"
    stub.sse_first = [b'data: {"choices": [{"delta": {"content": "ap"}}]}\n\n']
    stub.sse_rest = [b'data: {"choices": [{"delta": {"content": "prove"}}]}\n\n',
                     b"data: [DONE]\n\n"]
    stub.sse_gate = threading.Event()
    body = json.dumps(oai_request(PREFIX + "ticket 1", stream=True)).encode()
    s = socket.create_connection(("127.0.0.1", port), timeout=15)
    s.sendall(b"POST /openai/v1/chat/completions HTTP/1.1\r\n"
              + f"Host: 127.0.0.1:{port}\r\n".encode()
              + b"Content-Type: application/json\r\n"
              + b"Content-Length: " + str(len(body)).encode() + b"\r\n"
              + b"Connection: close\r\n\r\n" + body)
    buf = b""
    try:
        while b"data:" not in buf:
            chunk = s.recv(16)
            assert chunk, "connection closed before the first SSE event"
            buf += chunk
        # the first chunk arrived while the stub was still waiting -> unbuffered
        assert not stub.sse_gate.is_set()
        stub.sse_gate.set()
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    finally:
        s.close()
    assert b"data: [DONE]" in buf
    assert buf.count(b"data:") >= 3


# --------------------------------------------------------------------------
# T-G17: bad paths + redirects
# --------------------------------------------------------------------------

def test_T_G17_bad_path_and_redirect(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    for path in ("/openai/v1/../x", "/openai//v1/models",
                 "/openai/http://evil.com/x", "/anthropic/v1/a/../b"):
        status, _, payload = gw_get(base, path)
        assert status == 400, path
        assert json.loads(payload)["error"] == "bad_path", path
    assert stub.requests == []
    # redirects are returned as-is, never followed
    stub.mode = "redirect"
    status, hdrs, payload = gw_get(base, "/openai/v1/models")
    assert status == 302
    assert hdrs.get("Location") == "/v1/other"
    assert payload == b"redirect-body"
    assert len(stub.requests) == 1


# --------------------------------------------------------------------------
# T-G18: do_outcome via x-decide-id
# --------------------------------------------------------------------------

def test_T_G18_outcome_via_decision_id(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    _, hdrs, _ = ticket_call(base, "ticket 1")
    did = hdrs["x-decide-id"]
    r = svc.do_outcome({"decision_id": did, "ok": True, "note": "customer happy"})
    assert r["ok"], r
    row = store.get_decision(did)
    assert row["outcome"] == {"ok": True, "correct_choice": None,
                              "note": "customer happy"}


# --------------------------------------------------------------------------
# T-G19: log allow-list, samples gate, retention
# --------------------------------------------------------------------------

def test_T_G19_allow_samples_retention(tmp_path, stub, gw_factory):
    svc, store, cfg = make_service(tmp_path)
    cfg["gateway"]["upstreams"] = {"openai": stub.url, "anthropic": stub.url}
    _, _, base = gw_factory(cfg)
    # log_unmatched=false + empty log_allow: only counters
    status, _, _ = ticket_call(base, "ticket 1")
    assert status == 200
    assert store.counter_get("unregistered", f"openai:{MODEL}:200") == 1
    assert store.unregistered_traces() == []
    # allow-listed discovery key: full trace logged
    dk = GT.discovery_key("openai", MODEL, SYS)
    cfg["gateway"]["log_allow"] = [dk]
    status, _, _ = ticket_call(base, "ticket 2")
    assert status == 200
    traces = store.unregistered_traces()
    assert len(traces) == 1
    assert traces[0]["raw_request"]["user"] == PREFIX + "ticket 2"
    assert traces[0]["data_class"] == "private"  # gateway.trace_data_class
    # samples only with show_samples + confirm
    r = svc.do_analyze({"gateway": True})
    assert r["ok"] and "samples" not in r, r
    r = svc.do_analyze({"gateway": True, "show_samples": True})
    assert r["ok"] is False and r["error"] == "confirm_required", r
    r = svc.do_analyze({"gateway": True, "show_samples": True, "confirm": True})
    assert r["ok"] and len(r["samples"]) == 1, r
    assert r["samples"][0]["user"] == PREFIX + "ticket 2"
    # retention purges by age
    old = {"ts": "2020-01-01T00:00:00Z", "route": "openai", "model": MODEL,
           "discovery_key": dk, "mode": "forwarded",
           "raw_request": {"system": SYS, "user": "old"},
           "raw_response": "approve", "status": 200}
    store.trace_insert(old)
    assert svc.do_register(ticket_point())["ok"]
    store.trace_insert({**old, "point_id": "gw.tickets", "point_version": 1})
    store.insert_decision({"decision_id": "d1", "point_id": "gw.tickets",
                           "point_version": 1, "ts": "2020-01-01T00:00:00Z",
                           "context": {"a": 1}, "state_text": "old text",
                           "options": list(OPTIONS),
                           "allowed_options": list(OPTIONS), "mode": "forwarded",
                           "source": "gateway", "phase": "collect"})
    got = GW.purge_retention(store, cfg)
    assert got["unregistered_deleted"] == 1
    assert got["registered_text_purged"] == 1
    assert all(t["raw_request"] is None
               for t in store.traces_for_point("gw.tickets"))
    row = store.get_decision("d1")
    assert row["state_text"] is None and row["context"] is None
    assert row["mode"] == "forwarded"  # compact record stays


# --------------------------------------------------------------------------
# vector caching for retraining (§14.6)
# --------------------------------------------------------------------------

# test_gateway_caches_shadow_vectors deleted in P2: probe hidden-vector
# caching is a local_legacy mechanism; tiered training exports rows.


# --------------------------------------------------------------------------
# T-G20: duplicate fingerprint + option cap
# --------------------------------------------------------------------------

def test_T_G20_fingerprint_in_use_and_option_cap(tmp_path):
    svc, store, _ = make_service(tmp_path)
    assert svc.do_register(ticket_point())["ok"]
    r = svc.do_register(ticket_point(point_id="gw.clone"))
    assert r["ok"] is False and r["error"] == "fingerprint_in_use", r
    many = [f"opt{i}" for i in range(17)]
    r = svc.do_register(ticket_point(point_id="gw.many", options=many))
    assert r["ok"] is False and r["error"] == "too_many_options", r
