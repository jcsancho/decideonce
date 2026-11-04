"""M11 session-decider tests (F7): FireworksSessionDecider + FakeWorker."""
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

import decideonce.fireworks_worker as W
from decideonce import stats as S
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.deciders.base import DecideRequest, DeciderError
from decideonce.deciders.session import FireworksSessionDecider
from decideonce.fireworks_worker import tokens_of
from decideonce.service import Service
from decideonce.store import Store

SCRIPT = str(Path(W.__file__).resolve())
MODEL = "accounts/acct-test/models/dk-t-0001"


def _month():
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _setup(tmp_path, fake_args=("ok",), serving=True, budget=5.0,
           timeout_s=3.0, temperature=1.0, model=MODEL, card=True):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["decider"]["timeout_s"] = timeout_s
    cfg["fireworks"]["allow_session_serving"] = serving
    cfg["fireworks"]["monthly_budget_usd"] = budget
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    argv = [sys.executable, SCRIPT, "fake", *[str(a) for a in fake_args]]
    deciders["private"] = FireworksSessionDecider(
        store, cfg, "k", worker_argv_fn=lambda _m: list(argv))
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    assert svc.do_register(
        {"point_id": "s.pt", "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"})["ok"]
    p = store.get_point("s.pt")
    p["tier"] = "private"
    p["phase"] = "shadow"
    store.put_point(p)
    if card:
        store.save_model("s.pt", 1, "private", "v1-1.json",
                         {"model_ref": model, "temperature": temperature,
                          "job_id": "j1"})
    return svc, store, cfg


def _req(pid="s.pt", hint="a"):
    return DecideRequest(
        point_id=pid, state={"policy": "P", "label_hint": hint},
        instructions="Which?", criteria={"a": "a", "b": "b"})


# -- T-PT8 ------------------------------------------------------------------

def test_T_PT8_warming_teacher_and_counted(tmp_path):
    svc, store, _ = _setup(tmp_path, fake_args=("mute", "30"),
                           timeout_s=0.2)
    dec = svc.deciders["private"]
    try:
        for n in (1, 2):
            r = svc.do_decide({"point_id": "s.pt",
                               "context": {"label_hint": "a"}})
            assert r["ok"] and r["mode"] == "teacher"
            assert "jev_unavailable" in r["warnings"]
            assert store.counter_get("private_warming", "s.pt") == n
        assert store.counter_get("hosted_error_timeout", "s.pt") == 0
    finally:
        dec.close()


def test_T_PT8_tempered_softmax_and_spend(tmp_path):
    svc, store, _ = _setup(tmp_path, temperature=0.5)
    dec = svc.deciders["private"]
    try:
        assert repr(dec) == "FireworksSessionDecider(key=set)"
        res = dec.decide([_req()])[0]
        assert res.decider == "private" and res.model == MODEL
        assert list(res.probs) == ["a", "b"]
        want = S.softmax([-0.105 / 0.5, -8.0 / 0.5])
        assert res.probs["a"] == pytest.approx(want[0])
        assert res.probs["b"] == pytest.approx(want[1])
        toks = tokens_of([dec._worker_row(_req())])
        assert res.tokens_in == toks
        assert res.cost_usd == pytest.approx(toks * 1.96 / 1e6)
        got = store.spend_get(_month(), "fireworks")
        assert got == {"usd": pytest.approx(res.cost_usd), "calls": 1}
        r = svc.do_decide({"point_id": "s.pt",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"  # shadow hides the answer
        row = store.get_decision(r["decision_id"])
        assert row["decider"] == "private"
        assert row["probe"]["model_version"] == MODEL
        assert row["probe"]["probabilities"]["a"] == pytest.approx(want[0])
    finally:
        dec.close()


def test_T_PT8_gate_silent_teacher(tmp_path):
    svc, store, _ = _setup(tmp_path, serving=False)
    dec = svc.deciders["private"]
    try:
        with pytest.raises(DeciderError) as e:
            dec.decide([_req()])
        assert (e.value.code, e.value.msg) == ("remote_not_allowed",
                                               "serving_not_allowed")
        r = svc.do_decide({"point_id": "s.pt",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"
        assert r["warnings"] == []  # silent, like collect
        assert dec._sessions == {}  # no worker ever started
    finally:
        dec.close()


def test_T_PT8_budget_blocks_before_spawn(tmp_path):
    svc, store, _ = _setup(tmp_path, budget=0)
    dec = svc.deciders["private"]
    try:
        with pytest.raises(DeciderError) as e:
            dec.decide([_req()])
        assert e.value.code == "budget_exceeded"
        r = svc.do_decide({"point_id": "s.pt",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"
        assert "budget_exceeded" in r["warnings"]
        (sess,) = dec._sessions.values()
        assert sess._proc is None  # reserved first: nothing spawned
    finally:
        dec.close()


def test_T_PT8_no_model(tmp_path):
    svc, store, _ = _setup(tmp_path, card=False)
    dec = svc.deciders["private"]
    try:
        with pytest.raises(DeciderError) as e:
            dec.decide([_req()])
        assert (e.value.code, e.value.msg) == ("unavailable",
                                               "no_private_model")
        r = svc.do_decide({"point_id": "s.pt",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"
        assert "jev_unavailable" in r["warnings"]
        assert store.counter_get("private_error_unavailable", "s.pt") == 1
    finally:
        dec.close()


def test_T_PT8_timeout_counted_private(tmp_path):
    svc, store, _ = _setup(tmp_path, fake_args=("slow", "30"),
                           timeout_s=0.2)
    dec = svc.deciders["private"]
    try:
        r = svc.do_decide({"point_id": "s.pt",
                           "context": {"label_hint": "a"}})
        assert r["ok"] and r["mode"] == "teacher"
        assert "jev_unavailable" in r["warnings"]
        assert store.counter_get("private_error_timeout", "s.pt") == 1
        assert store.counter_get("hosted_error_timeout", "s.pt") == 0
        assert store.spend_open(_month(), "fireworks")["n"] == 1  # kept: R2-8
    finally:
        dec.close()


def test_T_PT8_error_releases(tmp_path):
    svc, store, _ = _setup(tmp_path, fake_args=("error", "boom"))
    dec = svc.deciders["private"]
    try:
        with pytest.raises(DeciderError) as e:
            dec.decide([_req()])
        assert e.value.code == "bad_response"
        assert store.spend_open(_month(), "fireworks")["n"] == 0
        assert store.spend_get(_month(), "fireworks") == {"usd": 0.0,
                                                          "calls": 0}
    finally:
        dec.close()


def test_T_PT8_two_models_two_sessions(tmp_path):
    svc, store, _ = _setup(tmp_path)
    dec = svc.deciders["private"]
    try:
        assert svc.do_register(
            {"point_id": "s.q", "workflow": "w", "question": "Which?",
             "kind": "choice", "options": ["a", "b"],
             "context_fields": {"label_hint": {"type": "string"}},
             "required_fields": [], "data_class": "public"})["ok"]
        store.save_model("s.q", 1, "private", "v1-1.json",
                         {"model_ref": MODEL + "-2", "temperature": 1.0,
                          "job_id": "j2"})
        out = dec.decide([_req("s.pt", "a"), _req("s.q", "b")])
        assert [r.model for r in out] == [MODEL, MODEL + "-2"]
        assert out[1].probs["b"] > out[1].probs["a"]
        assert len(dec._sessions) == 2
    finally:
        dec.close()


def test_T_PT8_bad_temperature_falls_back(tmp_path):
    svc, store, _ = _setup(tmp_path)
    dec = svc.deciders["private"]
    try:
        want = S.softmax([-0.105, -8.0])
        for bad in ("junk", 0, -2.0, None):
            store.save_model("s.pt", 1, "private", "v1-9.json",
                             {"model_ref": MODEL, "temperature": bad,
                              "job_id": "j9"})
            res = dec.decide([_req()])[0]
            assert res.probs["a"] == pytest.approx(want[0]), bad
    finally:
        dec.close()
