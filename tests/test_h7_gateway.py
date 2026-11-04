"""H7 tests: gateway on the tiered decider (T-HB9)."""
import random
import time

from decideonce import gateway as GW
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store

SYS = ("You are a ticket classifier. Answer with exactly one word: "
       "approve, deny, or escalate.")
PREFIX = "Classify this ticket: "
MODEL = "stub"
OPTIONS = ["approve", "deny", "escalate"]


def make_tiered(tmp_path, **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["store"]["project"] = "proj"
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    gw = GW.Gateway(cfg, store, deciders, rng=random.Random(1))
    return svc, gw, store, cfg


def gw_point(**kw):
    p = {"point_id": "gw.h7", "workflow": "support",
         "question": "What should happen to this ticket?",
         "kind": "choice", "options": list(OPTIONS),
         "context_fields": {}, "required_fields": [], "rules": [],
         "data_class": "public", "source": "gateway",
         "route": "openai", "model": MODEL,
         "template": {"system": SYS, "user_prefix": PREFIX, "user_suffix": ""},
         "output_format": {"type": "enum_text"},
         "extract": {"label_hint": {"regex": r"label_hint: (\w+)",
                                    "group": 1, "cast": "string"}},
         "multi_turn": False, "policy": "gateway policy"}
    p.update(kw)
    return p


def gw_req(user):
    return {"model": MODEL, "system": SYS, "user": PREFIX + user,
            "shape": "single"}


def test_T_HB9_gateway_tiered_call_state_spend_fit(tmp_path):
    svc, gw, store, _ = make_tiered(tmp_path, fit_rows=6)
    assert svc.do_register(gw_point())["ok"]
    transport = gw.svc.deciders["hosted_jev"].transport
    plan = gw.handle_decision("openai", gw_req("ticket 0 label_hint: approve"),
                              b"")
    assert plan["action"] == "forward"  # collect: no model call
    assert transport.calls == 0
    assert svc.do_promote({"point_id": "gw.h7",
                           "phase": "shadow"})["ok"]
    for i in range(6):
        req = gw_req(f"ticket {i} label_hint: approve")
        plan = gw.handle_decision("openai", req, b"")
        assert plan["action"] == "forward"  # shadow always forwards
        gw._insert_decision_row(store.get_point("gw.h7"), plan, req,
                                "approve", False, "forwarded", "approve")
    assert transport.calls == 6  # same budget, shared spend table
    body = transport.requests[0]
    assert list(body["state"])[0] == "policy"
    assert body["state"]["policy"] == "gateway policy"
    assert body["state"]["label_hint"] == "approve"
    month = time.strftime("%Y-%m", time.gmtime())
    assert store.spend_get(month, "hosted_jev")["calls"] == 6
    fit = store.active_hosted_fit("gw.h7")
    assert fit is not None and fit["n_fit"] == 6
    rows = store.decisions_for_point("gw.h7")
    assert len(rows) == 6 and all(r["hosted"] for r in rows)
    assert {r["decider"] for r in rows} == {"hosted_jev"}


# -- Section B: robustness and security --------------------------------------

def test_B8_pool_timeout_counted_as_hosted_error_timeout(tmp_path):
    import pytest

    from decideonce.deciders.base import DeciderError

    svc, gw, store, cfg = make_tiered(tmp_path)
    cfg["gateway"]["jev_timeout_s"] = 0.05
    assert svc.do_register(gw_point())["ok"]
    p = store.get_point("gw.h7")
    p["phase"] = "shadow"
    store.put_point(p)

    class Hang:
        def decide(self, reqs, timeout_s=None):
            time.sleep(0.5)
            raise AssertionError("pool must time out first")

    gw.svc.deciders["hosted_jev"] = Hang()
    plan = gw.handle_decision(
        "openai", gw_req("ticket 1 label_hint: approve"), b"")
    assert plan["action"] == "forward"
    assert "jev_unavailable" in plan["warnings"]
    assert store.counter_get("hosted_error_timeout", "gw.h7") == 1
    # direct: the pool timeout surfaces as DeciderError("timeout")
    with pytest.raises(DeciderError) as ei:
        gw._decide_call(Hang(), None)
    assert ei.value.code == "timeout"


def test_B8_outer_timeout_exceeds_inner(tmp_path):
    import time as _time

    from decideonce.deciders.base import DecideRequest, DeciderError
    from decideonce.deciders.hosted import FakeHostedTransport, HostedJevDecider

    svc, gw, store, cfg = make_tiered(tmp_path)
    cfg["gateway"]["jev_timeout_s"] = 0.5

    def slow_fn(body):
        _time.sleep(0.75)  # past the inner 0.5, inside the outer 1.0
        (pid, q), = body["questions"].items()
        opts = list(q["criteria"])
        probs = {o: 1.0 / len(opts) for o in opts}
        return 200, {"answers": {pid: {"choice": opts[0],
                                       "probabilities": probs,
                                       "confidence": 0.5}},
                     "usage": {"input_tokens": 10, "output_tokens": 0,
                               "cost": 0.0001}}

    dec = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model="m", key="fake", store=store,
        monthly_budget_usd=5.0, transport=FakeHostedTransport(fn=slow_fn))
    req = DecideRequest("gw.h7", {}, "Q?", {"a": "a", "b": "b"})
    assert gw._decide_call(dec, req).decider == "hosted_jev"

    # R2-6: the local probe keeps a single timeout.
    sentinel = object()

    class SlowLocal:
        def decide(self, reqs, timeout_s=None):
            _time.sleep(0.75)
            return [sentinel]

    try:
        gw._decide_call(SlowLocal(), None)
        assert False, "local slow call should time out at 0.5"
    except DeciderError as e:
        assert e.code == "timeout"


# -- Section D: gateway version bump runs the promote checks -----------------

def test_D_gateway_bump_collect_stays_without_gate(tmp_path):
    svc, _, store, _ = make_tiered(tmp_path, monthly_budget_usd=0)
    assert svc.do_register(gw_point())["ok"]
    assert store.get_point("gw.h7")["phase"] == "collect"
    # no budget: the bump must not start paid shadow calls by itself
    assert svc.do_register(gw_point(
        options=["approve", "deny", "escalate", "hold"]))["ok"]
    p = store.get_point("gw.h7")
    assert p["point_version"] == 2 and p["phase"] == "collect"
    # and do_promote still refuses for the same reason
    r = svc.do_promote({"point_id": "gw.h7", "phase": "shadow"})
    assert r["ok"] is False and r["failed"] == ["budget"]
    # paused likewise stays paused
    p["phase"] = "paused"
    store.put_point(p)
    assert svc.do_register(gw_point(
        options=["approve", "deny", "escalate", "hold", "xfer"]))["ok"]
    p = store.get_point("gw.h7")
    assert p["point_version"] == 3 and p["phase"] == "paused"


def test_D_gateway_bump_collect_stays_with_gate(tmp_path):
    # R2-4 supersedes the old moves-with-gate rule: a bump never moves
    # collect/paused (paid calls, undoes pause); do_promote admits it.
    svc, _, store, _ = make_tiered(tmp_path)
    assert svc.do_register(gw_point())["ok"]
    assert svc.do_register(gw_point(
        options=["approve", "deny", "escalate", "hold"]))["ok"]
    p = store.get_point("gw.h7")
    assert p["point_version"] == 2 and p["phase"] == "collect"
    assert svc.do_promote(
        {"point_id": "gw.h7", "phase": "shadow"})["ok"]


def test_D_gateway_bump_serve_goes_shadow_regardless(tmp_path):
    svc, _, store, _ = make_tiered(tmp_path, monthly_budget_usd=0)
    assert svc.do_register(gw_point())["ok"]
    p = store.get_point("gw.h7")
    p["phase"] = "serve"
    store.put_point(p)
    assert svc.do_register(gw_point(
        options=["approve", "deny", "escalate", "hold"]))["ok"]
    p = store.get_point("gw.h7")
    assert p["point_version"] == 2 and p["phase"] == "shadow"
