"""H2 tests: HostedJevDecider, key/host checks, spend + budget (T-HB1-5, T-HJ6, T-HJ8)."""
import json
import random
import socket
import subprocess
import threading
import time

import pytest

from decideonce.config import ConfigError, load_config
from decideonce.deciders import build_deciders
from decideonce.deciders.base import DecideRequest, DeciderError
from decideonce.deciders.hosted import (
    FakeHostedTransport,
    HostedJevDecider,
    UrllibTransport,
    build_hosted_deciders,
    resolve_key,
)
from decideonce.service import Service
from decideonce.store import Store


def make_tiered(tmp_path, **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"  # tiered with fake transports
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    return svc, store, cfg


def make_custom(tmp_path, fn, **hj_over):
    """Tiered service with a scripted FakeHostedTransport."""
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    transport = FakeHostedTransport(fn=fn)
    dec = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
        key="fake", store=store, timeout_s=cfg["decider"]["timeout_s"],
        price_per_mtok_in=cfg["hosted_jev"]["price_per_mtok_in"],
        monthly_budget_usd=cfg["hosted_jev"]["monthly_budget_usd"],
        reserve_usd=cfg["hosted_jev"]["reserve_usd"], transport=transport)
    svc = Service(cfg, store, {"hosted_jev": dec}, rng=random.Random(0))
    return svc, store, cfg, transport


def choice_point(**kw):
    p = {"point_id": "hb.shape", "workflow": "w", "question": "Which action?",
         "kind": "choice", "options": ["full_refund", "partial_refund", "escalate"],
         "option_help": {"escalate": "send to a human"},
         "context_fields": {"customer_tier": {"type": "string"},
                            "amount": {"type": "number"},
                            "photo": {"type": "boolean"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def to_shadow(store, pid="hb.shape"):
    p = store.get_point(pid)
    p["phase"] = "shadow"  # H4 wires do_promote; H2 sets the phase directly
    store.put_point(p)


def ctx(**kw):
    c = {"customer_tier": "gold", "amount": 80, "photo": True}
    c.update(kw)
    return c


# -- T-HB1 request shape ----------------------------------------------------


def test_T_HB1_request_shape_and_golden_body(tmp_path):
    svc, store, cfg = make_tiered(tmp_path)
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"
    transport = svc.deciders["hosted_jev"].transport
    assert transport.calls == 1
    body = transport.requests[0]
    assert body["model"] == "typesafe/jev-1.13"
    assert list(body["state"]) == ["customer_tier", "amount", "photo"]
    assert body["state"] == {"customer_tier": "gold", "amount": 80,
                             "photo": True}
    (pid, q), = body["questions"].items()
    assert pid == "hb.shape"
    assert q["type"] == "choice"
    assert q["instructions"] == "Which action?"
    assert list(q["criteria"]) == ["full_refund", "partial_refund", "escalate"]
    assert q["criteria"] == {"full_refund": "full_refund",
                             "partial_refund": "partial_refund",
                             "escalate": "send to a human"}
    golden = {"model": "typesafe/jev-1.13",
              "state": {"customer_tier": "gold", "amount": 80, "photo": True},
              "questions": {"hb.shape": {
                  "type": "choice", "instructions": "Which action?",
                  "criteria": {"full_refund": "full_refund",
                               "partial_refund": "partial_refund",
                               "escalate": "send to a human"}}}}
    assert body == golden
    assert json.dumps(body, sort_keys=True) == json.dumps(golden, sort_keys=True)


def test_T_HB1_noul_sent_as_two_option_choice(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    p = choice_point(point_id="hb.noul", kind="noul", options=["no", "yes"],
                     option_help={})
    assert svc.do_register(p)["ok"]
    to_shadow(store, "hb.noul")
    r = svc.do_decide({"point_id": "hb.noul", "context": ctx()})
    assert r["ok"]
    body = svc.deciders["hosted_jev"].transport.requests[0]
    (pid, q), = body["questions"].items()
    assert pid == "hb.noul" and q["type"] == "choice"
    assert list(q["criteria"]) == ["no", "yes"]


# -- T-HB2 response parsing -------------------------------------------------


def scripted(probs=None, status=200, point_id="hb.shape", usage=None, **extra):
    def fn(body):
        q = body["questions"][point_id]
        opts = list(q["criteria"])
        if probs is None:
            p = {o: 1.0 / len(opts) for o in opts}
        else:
            p = dict(probs)
        ans = {"choice": max(p, key=p.get), "probabilities": p,
               "confidence": max(p.values()) if p else 0}
        resp = {"answers": {point_id: ans},
                "usage": {"input_tokens": 100, "output_tokens": 0,
                          "cost": 0.0001}}
        if usage is not None:
            resp["usage"] = usage
        resp.update(extra)
        return status, resp
    return fn


def test_T_HB2_missing_option_zero_and_renormalise(tmp_path):
    svc, store, _, _ = make_custom(
        tmp_path, scripted({"full_refund": 0.6, "partial_refund": 0.2}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"
    row = store.get_decision(r["decision_id"])
    hp = row["hosted"]["probabilities"]
    assert hp["escalate"] == 0
    assert abs(sum(hp.values()) - 1.0) < 1e-9
    assert abs(hp["full_refund"] - 0.75) < 1e-9
    assert row["decider"] == "hosted_jev"


def _assert_teacher_unavailable(svc, store):
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"  # no exception escapes
    assert r["choice"] is None and r["probabilities"] is None
    assert "jev_unavailable" in r["warnings"]
    row = store.get_decision(r["decision_id"])
    assert row["hosted"] is None


def test_T_HB2_missing_point_id(tmp_path):
    def fn(body):
        return 200, {"answers": {"other": {"choice": "x",
                                           "probabilities": {"x": 1.0},
                                           "confidence": 1.0}},
                     "usage": {"cost": 0.0001}}
    svc, store, _, _ = make_custom(tmp_path, fn)
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    _assert_teacher_unavailable(svc, store)


def test_T_HB2_all_zeros(tmp_path):
    svc, store, _, _ = make_custom(tmp_path, scripted(
        {"full_refund": 0.0, "partial_refund": 0.0, "escalate": 0.0}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    _assert_teacher_unavailable(svc, store)


def test_T_HB2_nan(tmp_path):
    svc, store, _, _ = make_custom(tmp_path, scripted(
        {"full_refund": float("nan"), "partial_refund": 0.5, "escalate": 0.5}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    _assert_teacher_unavailable(svc, store)


def test_T_HB2_unknown_option(tmp_path):
    svc, store, _, _ = make_custom(tmp_path, scripted(
        {"full_refund": 0.5, "partial_refund": 0.3, "bribe": 0.2}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    _assert_teacher_unavailable(svc, store)


# -- T-HB3 collect/paused call nothing --------------------------------------


def test_T_HB3_collect_and_paused_make_no_call(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(choice_point())["ok"]
    transport = svc.deciders["hosted_jev"].transport
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"
    assert transport.calls == 0
    p = store.get_point("hb.shape")
    p["phase"] = "paused"
    store.put_point(p)
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"
    assert transport.calls == 0


# -- T-HB4 host checks ------------------------------------------------------


def _cfg_with_url(tmp_path, url):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["url"] = url
    cfg["hosted_jev"]["monthly_budget_usd"] = 1.0
    return cfg


@pytest.mark.parametrize("url", [
    "http://openrouter.ai/api/v1/systemone",
    "https://evil.com/api/v1/systemone",
    "https://openrouter.ai@evil.com/api/v1/systemone",
    "https://user:pass@openrouter.ai/api/v1/systemone",
    "https://openrouter.ai:8443/api/v1/systemone",
    "https://openrouter.ai.evil.com/api/v1/systemone",
    "https://api.typesafe.ai.evil.com/v1/systemone",
])
def test_T_HB4_bad_hosts_refused_at_startup(tmp_path, url):
    cfg = _cfg_with_url(tmp_path, url)
    store = Store(tmp_path / "proj")
    with pytest.raises(ConfigError) as ei:
        build_hosted_deciders(cfg, store)
    assert ei.value.code == "decider_host_not_allowed"


@pytest.mark.parametrize("url", [
    "https://openrouter.ai/api/v1/systemone",
    "https://api.typesafe.ai/v1/systemone",
    "https://openrouter.ai:443/api/v1/systemone",
])
def test_T_HB4_good_hosts_start(tmp_path, url):
    cfg = _cfg_with_url(tmp_path, url)
    store = Store(tmp_path / "proj")
    deciders, _ = build_hosted_deciders(cfg, store)
    assert "hosted_jev" in deciders


# -- Section D: pack refusal and URL validation -------------------------------

def test_D_pack_true_refused_at_startup(tmp_path):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 1.0
    cfg["hosted_jev"]["pack"] = True
    store = Store(tmp_path / "proj")
    with pytest.raises(ConfigError) as ei:
        build_hosted_deciders(cfg, store)
    assert ei.value.code == "pack_not_supported"


@pytest.mark.parametrize("url", [
    "https://openrouter.ai/a b",
    "https://openrouter.ai/x?y=1 2",
    "https://openrouter.ai/\x00",
    "https://openrouter.ai/\xff",
    "https://openrouter.ai/\t",
])
def test_D_check_host_refuses_what_urllib_refuses(tmp_path, url):
    cfg = _cfg_with_url(tmp_path, url)
    store = Store(tmp_path / "proj")
    with pytest.raises(ConfigError) as ei:
        build_hosted_deciders(cfg, store)
    assert ei.value.code == "decider_host_not_allowed"


def test_T_HB4_302_maps_to_http(tmp_path):
    svc, store, _, _ = make_custom(tmp_path, scripted(status=302))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    _assert_teacher_unavailable(svc, store)


def test_T_HB4_real_opener_follows_no_redirect():
    import http.server
    import socketserver

    hits2 = []

    class H2(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            hits2.append(1)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass

    s2 = socketserver.TCPServer(("127.0.0.1", 0), H2)
    port2 = s2.server_address[1]
    t2 = threading.Thread(target=s2.handle_request, daemon=True)
    t2.start()

    class H1(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            self.send_response(302)
            self.send_header("Location",
                             f"http://127.0.0.1:{port2}/")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *a):
            pass

    s1 = socketserver.TCPServer(("127.0.0.1", 0), H1)
    port1 = s1.server_address[1]
    t1 = threading.Thread(target=s1.handle_request, daemon=True)
    t1.start()
    try:
        transport = UrllibTransport()
        with pytest.raises(DeciderError) as ei:
            transport(f"http://127.0.0.1:{port1}/", {}, b"{}", 3)
        assert ei.value.code == "http"
        t1.join(timeout=5)
        time.sleep(0.2)
        assert hits2 == []  # the redirect target never saw the request
    finally:
        s1.server_close()
        s2.server_close()


# -- T-HB5 spend ------------------------------------------------------------


def test_T_HB5_two_connections_sum_exact(tmp_path):
    s1 = Store(tmp_path / "proj")
    s2 = Store(tmp_path / "proj")
    month, errors = "2026-11", []

    def worker(store, tag):
        try:
            for i in range(10):
                store.hosted_settle(f"{tag}-{i}", month, "hosted_jev", 0.0001)
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(s, t))
               for s, t in [(s1, "a"), (s2, "b")]]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    got = s1.spend_get(month, "hosted_jev")
    assert got["calls"] == 20
    assert abs(got["usd"] - 0.002) < 1e-9


def test_T_HB5_budget_zero_makes_no_call(tmp_path):
    svc, store, _ = make_tiered(tmp_path, monthly_budget_usd=0)
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    transport = svc.deciders["hosted_jev"].transport
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"
    assert r["probabilities"] is None
    assert "budget_exceeded" in r["warnings"]
    assert transport.calls == 0


def test_T_HB5_concurrent_budget_admits_at_most_ten(tmp_path):
    reserve = 0.0005
    svc, store, _ = make_tiered(tmp_path, monthly_budget_usd=10 * reserve)
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    # all 20 threads reserve before any proceeds, so the admitted set is
    # exactly the first 10 and no reservation is released early; the
    # transport barrier then holds those 10 together until all are in.
    all_reserved = threading.Barrier(20)
    orig_reserve = store.hosted_reserve

    def gated_reserve(*a, **k):
        rid = orig_reserve(*a, **k)
        all_reserved.wait(timeout=30)
        return rid

    store.hosted_reserve = gated_reserve
    barrier = threading.Barrier(10)
    transport = svc.deciders["hosted_jev"].transport
    default = transport._default

    def gated(body):
        barrier.wait(timeout=30)
        return default(body)

    transport.fn = gated
    errors = []

    def worker(i):
        try:
            svc.do_decide({"point_id": "hb.shape",
                           "context": ctx(amount=i)})
        except Exception as e:  # noqa: BLE001 - surfaced below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert errors == []
    assert transport.calls <= 10
    assert transport.calls == 10


def test_T_HB5_stale_reservation_released(tmp_path):
    store = Store(tmp_path / "proj")
    month = "2026-11"
    store.cx.execute(
        "INSERT INTO spend_reservations(id, month, tier, usd, created_at)"
        " VALUES('stale', ?, 'hosted_jev', 0.0005, '2020-01-01T00:00:00Z')",
        (month,))
    store.cx.commit()
    # budget fits exactly one reservation: admitted only if stale is freed
    rid = store.hosted_reserve(month, "hosted_jev", 0.0005, 0.0005, 30)
    assert rid is not None
    assert store.spend_open(month, "hosted_jev")["n"] == 1


def test_T_HB5_budget_below_reservation_refuses_startup(tmp_path):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 0.0001
    cfg["hosted_jev"]["reserve_usd"] = 0.0005
    store = Store(tmp_path / "proj")
    with pytest.raises(ConfigError) as ei:
        build_hosted_deciders(cfg, store)
    assert ei.value.code == "budget_below_reservation"


def test_T_HB5_months_are_independent(tmp_path):
    store = Store(tmp_path / "proj")
    store.hosted_settle("a", "2026-10", "hosted_jev", 0.001)
    store.hosted_settle("b", "2026-11", "hosted_jev", 0.002)
    assert store.spend_get("2026-10", "hosted_jev") == {"usd": 0.001,
                                                       "calls": 1}
    assert store.spend_get("2026-11", "hosted_jev") == {"usd": 0.002,
                                                       "calls": 1}


# -- T-HJ6 key canary -------------------------------------------------------


def test_T_HJ6_key_never_leaks(tmp_path):
    key = "CANARY_KEY_7f3a9c_hosted"
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    store = Store(tmp_path / "proj")

    def flaky(body):
        flaky.n += 1
        if flaky.n == 1:
            return 500, {"error": "boom"}
        if flaky.n == 2:
            raise DeciderError("timeout", "slow")
        return 200, {"answers": {}, "usage": {}}

    flaky.n = 0
    dec = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
        key=key, store=store, timeout_s=3,
        price_per_mtok_in=0.042, monthly_budget_usd=5.0,
        reserve_usd=0.0005, transport=FakeHostedTransport(fn=flaky))
    svc = Service(cfg, store, {"hosted_jev": dec}, rng=random.Random(0))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    texts = [repr(dec)]
    for _ in range(3):
        texts.append(json.dumps(svc.do_decide(
            {"point_id": "hb.shape", "context": ctx()}), default=str))
    texts.append(json.dumps(svc.do_report({"point_id": "hb.shape"}),
                            default=str))
    for table in ("decisions", "spend", "spend_reservations", "points",
                  "gateway_counters"):
        try:
            rows = store.cx.execute(f"SELECT * FROM {table}").fetchall()
        except Exception:
            continue
        texts.append(json.dumps([dict(r) for r in rows], default=str))
    for t in texts:
        assert key not in t


# -- T-HJ8 budget -----------------------------------------------------------


def test_T_HJ8_cost_from_usage_counts_against_hosted_budget(tmp_path):
    svc, store, _, transport = make_custom(
        tmp_path, scripted(usage={"input_tokens": 100, "output_tokens": 0,
                                  "cost": 0.0003}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"]
    row = store.get_decision(r["decision_id"])
    assert row["hosted"]["cost_usd"] == 0.0003
    import time as _t
    month = _t.strftime("%Y-%m", _t.gmtime())
    assert store.spend_get(month, "hosted_jev")["usd"] == 0.0003
    assert transport.calls == 1


def test_T_HJ8_token_fallback_and_estimate(tmp_path):
    svc, store, _, _ = make_custom(
        tmp_path, scripted(usage={"input_tokens": 1000, "output_tokens": 0}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    row = store.get_decision(r["decision_id"])
    assert abs(row["hosted"]["cost_usd"] - 1000 * 0.042 / 1e6) < 1e-12

    svc2, store2, _, _ = make_custom(
        tmp_path / "est", scripted(usage={}))
    assert svc2.do_register(choice_point())["ok"]
    to_shadow(store2)
    r2 = svc2.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert "cost_estimated" in (r2["warnings"] or [])
    assert store2.get_decision(r2["decision_id"])["hosted"]["cost_usd"] > 0


# -- Section B: robustness and security --------------------------------------

def _raw_server(behave):
    """Raw-socket stub: behave(conn) after the request headers arrive."""
    ls = socket.socket()
    ls.bind(("127.0.0.1", 0))
    ls.listen(5)

    def read_req(c):
        data = b""
        while b"\r\n\r\n" not in data:
            chunk = c.recv(4096)
            if not chunk:
                return False
            data += chunk
        return True

    def loop():
        while True:
            try:
                c, _ = ls.accept()
            except OSError:
                return
            def run(c=c):
                try:
                    if read_req(c):
                        behave(c)
                except OSError:
                    pass
                finally:
                    try:
                        c.close()
                    except OSError:
                        pass
            threading.Thread(target=run, daemon=True).start()

    threading.Thread(target=loop, daemon=True).start()
    return ls


def _close_noresp(c):
    c.shutdown(socket.SHUT_RDWR)


def _short_body(c):
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n{\"ans")


def _garbage_status(c):
    c.sendall(b"GARBAGE\r\n\r\n")


def _drip(c):
    c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 8\r\n\r\n")
    for ch in b"{\"a\":1} ":
        time.sleep(0.7)
        c.sendall(bytes([ch]))


def test_B1_raw_socket_failures_map_to_http():
    for behave in (_close_noresp, _short_body, _garbage_status):
        ls = _raw_server(behave)
        try:
            port = ls.getsockname()[1]
            with pytest.raises(DeciderError) as ei:
                UrllibTransport()(f"http://127.0.0.1:{port}/", {}, b"{}", 3)
            assert ei.value.code == "http"
        finally:
            ls.close()


def test_B1_transport_error_releases_reservation_and_falls_back(tmp_path):
    def boom(body):
        raise ConnectionResetError("reset by peer")

    svc, store, _, _ = make_custom(tmp_path, boom)
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"] and r["mode"] == "teacher"  # fallback, not "internal"
    month = time.strftime("%Y-%m", time.gmtime())
    assert store.spend_open(month, "hosted_jev")["n"] == 0
    assert store.counter_get("hosted_error_http", "hb.shape") == 1


def test_B1_huge_integers_do_not_escape(tmp_path):
    big = 10 ** 400
    req = DecideRequest("hb.shape", {"a": 1}, "Q?",
                        {"x": "x", "y": "y"})
    t = FakeHostedTransport(
        fn=lambda b: (200, {"answers": {"hb.shape": {"probabilities": {
            "x": big, "y": 0}}}, "usage": {"cost": 0.0001}}))
    d = HostedJevDecider(url="https://openrouter.ai/x", model="m", key="k",
                         store=Store(tmp_path / "p1"),
                         monthly_budget_usd=5.0, transport=t)
    with pytest.raises(DeciderError):
        d.decide([req])
    # an unrepresentable cost falls back to the estimate instead of escaping
    t2 = FakeHostedTransport(
        fn=lambda b: (200, {"answers": {"hb.shape": {"probabilities": {
            "x": 0.5, "y": 0.5}}}, "usage": {"cost": big}}))
    d2 = HostedJevDecider(url="https://openrouter.ai/x", model="m", key="k",
                          store=Store(tmp_path / "p2"),
                          monthly_budget_usd=5.0, transport=t2)
    res = d2.decide([req])[0]
    assert "cost_estimated" in res.warnings


def test_B2_key_validated_at_resolve_and_stdin_devnull(monkeypatch):
    seen = {}

    class Proc:
        returncode = 0
        stdout = "sk-valid-key_123"

    def fake_run(cmd, **kw):
        seen.update(kw)
        return Proc()

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert resolve_key("es_key") == "sk-valid-key_123"
    assert seen.get("stdin") == subprocess.DEVNULL  # B3
    Proc.stdout = "sk-CANARY\nextra"
    assert resolve_key("es_key") is None
    Proc.stdout = "has space"
    assert resolve_key("es_key") is None
    assert resolve_key("") is None


def test_B2_newline_key_direct_never_echoed(tmp_path):
    store = Store(tmp_path / "proj")
    req = DecideRequest("hb.shape", {"a": 1}, "Q?",
                        {"x": "x", "y": "y"})
    ls = _raw_server(_close_noresp)
    try:
        port = ls.getsockname()[1]
        d = HostedJevDecider(url=f"http://127.0.0.1:{port}/", model="m",
                             key="sk-CANARY-9f2\nextra", store=store,
                             monthly_budget_usd=5.0)
        with pytest.raises(DeciderError) as ei:
            d.decide([req])
        assert "CANARY" not in str(ei.value)
        month = time.strftime("%Y-%m", time.gmtime())
        assert store.spend_open(month, "hosted_jev")["n"] == 0
    finally:
        ls.close()


def test_B4_slow_drip_hits_overall_deadline():
    ls = _raw_server(_drip)
    try:
        port = ls.getsockname()[1]
        t0 = time.monotonic()
        with pytest.raises(DeciderError) as ei:
            UrllibTransport()(f"http://127.0.0.1:{port}/", {}, b"{}", 1)
        elapsed = time.monotonic() - t0
        assert ei.value.code == "timeout"
        assert elapsed < 4  # the full drip takes ~5.6 s
    finally:
        ls.close()


def test_B4_stale_release_at_10x_max_timeouts(tmp_path):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["decider"]["timeout_s"] = 3
    cfg["gateway"]["jev_timeout_s"] = 5
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    assert deciders["hosted_jev"].stale_after_s == 50


def test_B5_bad_budget_configs_refused(tmp_path):
    bad = [{"monthly_budget_usd": float("nan")},
           {"monthly_budget_usd": -1.0},
           {"monthly_budget_usd": 0.001, "reserve_usd": -1.0},
           {"monthly_budget_usd": 0.001,
            "reserve_usd": float("nan")},
           {"monthly_budget_usd": 0.001, "reserve_usd": 0},
           {"monthly_budget_usd": 0.001, "reserve_usd": float("inf")}]
    for i, over in enumerate(bad):
        cfg = load_config()
        cfg["store"]["data_dir"] = str(tmp_path / f"c{i}")
        cfg["decider"]["backend"] = "tiered"
        cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
        cfg["hosted_jev"].update(over)
        store = Store(tmp_path / f"c{i}" / "proj")
        with pytest.raises(ConfigError) as ei:
            build_hosted_deciders(cfg, store)
        assert ei.value.code == "invalid_budget"


def test_B7_stale_window_is_per_reservation(tmp_path):
    store = Store(tmp_path / "proj")
    month = "2026-11"
    # budget fits exactly one reservation: a second reserve is admitted
    # only if the first row counts as stale.
    r1 = store.hosted_reserve(month, "hosted_jev", 0.0005, 0.0005, 50)
    assert r1 is not None
    store.cx.execute(
        "UPDATE spend_reservations SET created_at=? WHERE id=?",
        (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                       time.gmtime(time.time() - 40)), r1))
    store.cx.commit()
    # 40 s old: stale under the caller's 30 s window, live under the row's
    # own 50 s window -> the row survives and the new call is refused.
    assert store.hosted_reserve(month, "hosted_jev", 0.0005, 0.0005,
                                30) is None
    assert store.spend_open(month, "hosted_jev")["n"] == 1
    # past the row's own window it is freed.
    store.cx.execute(
        "UPDATE spend_reservations SET created_at=? WHERE id=?",
        (time.strftime("%Y-%m-%dT%H:%M:%SZ",
                       time.gmtime(time.time() - 60)), r1))
    store.cx.commit()
    assert store.hosted_reserve(month, "hosted_jev", 0.0005, 0.0005,
                                30) is not None


def test_B7_failed_settle_keeps_cost(tmp_path):
    svc, store, _, _ = make_custom(
        tmp_path, scripted(usage={"input_tokens": 100, "output_tokens": 0,
                                  "cost": 0.0003}))
    assert svc.do_register(choice_point())["ok"]
    to_shadow(store)

    def always_fail(*a, **k):
        raise RuntimeError("store is down")

    store.hosted_settle = always_fail
    r = svc.do_decide({"point_id": "hb.shape", "context": ctx()})
    assert r["ok"]
    month = time.strftime("%Y-%m", time.gmtime())
    assert store.spend_get(month, "hosted_jev")["usd"] == 0.0003
    assert store.spend_open(month, "hosted_jev")["n"] == 0


def test_B9_store_required_outside_tests(tmp_path):
    with pytest.raises(TypeError):
        HostedJevDecider(url="https://openrouter.ai/x", model="m",
                         key="k", monthly_budget_usd=0)
    d = HostedJevDecider(url="https://openrouter.ai/x", model="m", key="k",
                         monthly_budget_usd=0, allow_no_store=True,
                         transport=FakeHostedTransport(fn=scripted()))
    req = DecideRequest("hb.shape", {}, "Q?", {"a": "a", "b": "b"})
    assert d.decide([req])[0].decider == "hosted_jev"


def test_R211_B3_real_subprocess_stdin_devnull(tmp_path, monkeypatch):
    """R2-11 (B3): resolve_key really passes stdin=DEVNULL (real subprocess).

    A fake `envsave` records what it reads on stdin while fd 0 carries a
    canary. DEVNULL reads EOF, never the canary.
    """
    import os

    from decideonce.deciders.hosted import resolve_key

    bindir = tmp_path / "bin"
    bindir.mkdir()
    stdin_log = tmp_path / "stdin.txt"
    (bindir / "envsave").write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        f"log = {str(stdin_log)!r}\n"
        "data = sys.stdin.read()\n"
        "open(log, 'w').write(data)\n"
        "print('sk-REALKEY')\n")
    (bindir / "envsave").chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])
    # fd 0 carries a canary; DEVNULL must not read it.
    r, w = os.pipe()
    os.write(w, b"CANARY-STDIN")
    os.close(w)
    saved0 = os.dup(0)
    try:
        os.dup2(r, 0)
        assert resolve_key("es_test") == "sk-REALKEY"
    finally:
        os.dup2(saved0, 0)
        os.close(saved0)
        os.close(r)
    assert stdin_log.read_text() == ""


def test_R211_B2_no_key_echo_through_run_wrapper(tmp_path):
    """R2-11 (B2): a bad-key failure through the server _run wrapper returns
    teacher fallback (or internal) with no key bytes in the JSON."""
    import json as _json

    from decideonce.config import load_config
    from decideonce.deciders.hosted import FakeHostedTransport, HostedJevDecider
    from decideonce.service import Service
    from decideonce.store import Store

    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    store = Store(tmp_path / "proj")
    dec = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
        key="sk-CANARY\nwith-newline", store=store, timeout_s=3,
        monthly_budget_usd=5.0, reserve_usd=0.0005,
        transport=FakeHostedTransport())
    svc = Service(cfg, store, {"hosted_jev": dec})

    def _run(fn, payload):
        try:
            return _json.dumps(fn(payload or {}))
        except Exception as e:
            return _json.dumps({"ok": False, "error": "internal",
                                "message": f"{type(e).__name__}: {e}"})

    out = _run(svc.do_register, {"point_id": "hb.key", "workflow": "w",
                                 "question": "Q?", "kind": "choice",
                                 "options": ["a", "b"],
                                 "context_fields": {"x": {"type": "string"}},
                                 "required_fields": [],
                                 "data_class": "public"})
    assert "CANARY" not in out
    out = _run(svc.do_decide, {"point_id": "hb.key",
                               "context": {"x": "y"}})
    assert "CANARY" not in out


def test_H3_fake_backends_warn_once_on_stderr(tmp_path, capsys):
    from decideonce import deciders as D
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    store = Store(tmp_path / "proj")
    D._WARNED.clear()
    cfg["decider"]["backend"] = "fake"
    build_deciders(cfg, store)
    build_deciders(cfg, store)
    err = capsys.readouterr().err
    assert err.count("FAKE decider") == 1
    assert "scripted answers, not a real model" in err
    D._WARNED.clear()
    # P3: local_legacy is refused outright, never warned-then-built.
    from decideonce.config import ConfigError
    cfg["decider"]["backend"] = "local_legacy"
    cfg["jev"] = {"url": "fake:", "timeout_s": 10}
    with pytest.raises(ConfigError):
        build_deciders(cfg, store)
    assert load_config()["hosted_jev"]["fake_accuracy"] == 1.0
