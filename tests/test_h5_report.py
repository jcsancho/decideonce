"""H5 tests: train_recommended, do_train precondition, §6.4 item 8 (T-HJ3/4/5)."""
import random
import time

import pytest

import decideonce.service as S
from decideonce import train as TR
from decideonce.config import load_config
from decideonce.deciders.fake import FakeDecider
from decideonce.deciders.hosted import FakeHostedTransport, HostedJevDecider
from decideonce.service import Service
from decideonce.store import Store


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 0}
    base = 1761950000

    def tick():
        state["t"] += 1
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(base + state["t"]))

    monkeypatch.setattr(S, "_utcnow", tick)
    return state


def make_dual(tmp_path, hosted_fn, private_fn=None,
              model_version="fake-private-1", **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    hosted = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
        key="fake", store=store, timeout_s=3,
        price_per_mtok_in=cfg["hosted_jev"]["price_per_mtok_in"],
        monthly_budget_usd=cfg["hosted_jev"]["monthly_budget_usd"],
        reserve_usd=cfg["hosted_jev"]["reserve_usd"],
        transport=FakeHostedTransport(fn=hosted_fn))
    deciders = {"hosted_jev": hosted}
    if private_fn is not None:
        deciders["private"] = FakeDecider(
            fn=private_fn, decider_tag="private", model_tag=model_version)
    return Service(cfg, store, deciders, rng=random.Random(0)), store, cfg


def scripted_hosted(schedule):
    """schedule(i, hint, opts) -> (top, p_top); i = call index."""
    state = {"n": 0}

    def fn(body):
        (pid, q), = body["questions"].items()
        opts = list(q["criteria"])
        hint = body["state"]["label_hint"]
        i = state["n"]
        state["n"] += 1
        top, p_top = schedule(i, hint, opts)
        rest = (1.0 - p_top) / (len(opts) - 1)
        probs = {o: (p_top if o == top else rest) for o in opts}
        return 200, {"answers": {pid: {"choice": top,
                                       "probabilities": probs,
                                       "confidence": p_top}},
                     "usage": {"input_tokens": 50, "output_tokens": 0,
                               "cost": 0.0001}}
    return fn


def scripted_private(schedule):
    """schedule(i, hint, opts) -> (top, p_top); i = dual-call index."""
    state = {"n": 0}

    def fn(req):
        opts = list(req.criteria)
        hint = req.state["label_hint"]
        i = state["n"]
        state["n"] += 1
        top, p_top = schedule(i, hint, opts)
        rest = (1.0 - p_top) / (len(opts) - 1)
        return {o: (p_top if o == top else rest) for o in opts}
    return fn


def point(**kw):
    p = {"point_id": "h5.pt", "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def decide_label(svc, label):
    r = svc.do_decide({"point_id": "h5.pt",
                       "context": {"label_hint": label}})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec
    return r["decision_id"]


def other(hint, opts):
    return [o for o in opts if o != hint][0]


# -- T-HJ3 train_recommended ------------------------------------------------


def test_T_HJ3_reason_low_coverage_threshold_null(tmp_path, clock):
    def sched(i, hint, opts):
        return (other(hint, opts) if i in (2, 5, 8) else hint), 0.9

    svc, store, cfg = make_dual(tmp_path, scripted_hosted(sched),
                                fit_rows=10)
    cfg["serve"]["min_auto_support"] = 3
    cfg["train"]["min_examples"] = 3
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    assert store.active_hosted_fit("h5.pt")["threshold"] is None
    for i in range(5):
        decide_label(svc, "ab"[i % 2])
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_recommended"] is True
    assert "low_coverage" in rep["hosted"]["train_reasons"]


def test_T_HJ3_reason_low_coverage_on_evidence(tmp_path, clock):
    def sched(i, hint, opts):
        return hint, 0.9 if i < 10 else 0.55

    svc, store, cfg = make_dual(tmp_path, scripted_hosted(sched),
                                fit_rows=10)
    cfg["serve"]["min_auto_support"] = 3
    cfg["train"]["min_examples"] = 3
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    assert store.active_hosted_fit("h5.pt")["threshold"] == 0.9
    for i in range(5):
        decide_label(svc, "ab"[i % 2])
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert "low_coverage" in rep["hosted"]["train_reasons"]


def test_T_HJ3_reason_margin_not_met(tmp_path, clock):
    def sched(i, hint, opts):
        if i < 10:
            return hint, 0.9
        j = i - 10
        return (other(hint, opts), 0.9) if j in (1, 4, 7) else (hint, 0.95)

    svc, store, cfg = make_dual(tmp_path, scripted_hosted(sched),
                                fit_rows=10)
    cfg["serve"]["min_auto_support"] = 3
    cfg["train"]["min_examples"] = 3
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for _ in range(10):
        decide_label(svc, "a")
    for _ in range(10):
        decide_label(svc, "a")
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_reasons"] == ["margin_not_met"]


def test_T_HJ3_reason_repeated_drift(tmp_path):
    svc, store, _ = make_dual(tmp_path, scripted_hosted(
        lambda i, hint, opts: (hint, 0.9)))
    assert svc.do_register(point())["ok"]
    p = store.get_point("h5.pt")
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p["events"] = [{"ts": now, "event": "drift_demotion", "tier": "hosted_jev"},
                   {"ts": now, "event": "drift_demotion", "tier": "hosted_jev"}]
    store.put_point(p)
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_reasons"] == ["repeated_drift"]
    p["events"] = [{"ts": "2020-01-01T00:00:00Z", "event": "drift_demotion",
                    "tier": "hosted_jev"}]
    store.put_point(p)
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_recommended"] is False


def test_T_HJ3_reason_teacher_only_heavy(tmp_path, clock):
    svc, store, cfg = make_dual(
        tmp_path, scripted_hosted(lambda i, hint, opts: (hint, 0.9)),
        fit_rows=10)
    cfg["serve"]["min_auto_support"] = 3
    cfg["train"]["min_examples"] = 3
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for i in range(7):
        decide_label(svc, "a")
    for i in range(3):
        decide_label(svc, "b")
    # Z5: rule 4 counts evidence rows (post-fit), not the fit rows.
    for i in range(7):
        decide_label(svc, "a")
    for i in range(3):
        decide_label(svc, "b")
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_reasons"] == ["teacher_only_heavy"]


def test_T_HJ3_not_otherwise(tmp_path, clock):
    svc, store, cfg = make_dual(
        tmp_path, scripted_hosted(lambda i, hint, opts: (hint, 0.9)),
        fit_rows=20)
    cfg["serve"]["min_auto_support"] = 5
    assert svc.do_register(point())["ok"]
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_recommended"] is False  # fresh
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for i in range(5):
        decide_label(svc, "ab"[i % 2])
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_recommended"] is False  # pre-trigger
    for i in range(15):
        decide_label(svc, "ab"[(i + 5) % 2])
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_reasons"] == []  # balanced, clean
    assert rep["hosted"]["fit"]["teacher_only_options"] == []


def test_Z6_no_private_scoring_on_unaudited_serve(tmp_path, clock):
    svc, store, cfg = make_dual(
        tmp_path, scripted_hosted(lambda i, hint, opts: (hint, 0.9)),
        scripted_private(lambda i, hint, opts: (hint, 0.9)), fit_rows=10)
    cfg["serve"]["audit_rate"] = 0.0
    assert svc.do_register(point())["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for _ in range(10):
        decide_label(svc, "a")
    TR.train_fake_private(store, "h5.pt", threshold=0.7)
    for _ in range(3):
        decide_label(svc, "a")
    rows = store.decisions_for_point("h5.pt")
    assert all(r.get("probe") for r in rows[-3:])  # shadow still logs
    p = store.get_point("h5.pt")
    p["phase"] = "serve"
    store.put_point(p)
    for _ in range(3):
        decide_label(svc, "a")
    rows = store.decisions_for_point("h5.pt")
    assert all(r.get("hosted") for r in rows[-3:])
    assert not any(r.get("audit") for r in rows[-3:])
    assert not any(r.get("probe") for r in rows[-3:])


# -- T-HJ4 do_train precondition --------------------------------------------


def test_T_HJ4_hosted_sufficient_then_not_built(tmp_path, clock):
    svc, store, cfg = make_dual(
        tmp_path, scripted_hosted(lambda i, hint, opts: (hint, 0.9)),
        fit_rows=10)
    cfg["serve"]["min_auto_support"] = 3
    cfg["train"]["min_examples"] = 3
    assert svc.do_register(point())["ok"]
    r = svc.do_train({"point_id": "h5.pt"})
    assert r["ok"] is False and r["error"] == "hosted_jev_sufficient"
    assert "hosted" in r and r["train_recommended"] is False
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for _ in range(7):
        decide_label(svc, "a")
    for _ in range(3):
        decide_label(svc, "b")
    # Z5: rule 4 needs mature evidence rows, not just the fit rows.
    for _ in range(7):
        decide_label(svc, "a")
    for _ in range(3):
        decide_label(svc, "b")
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["train_recommended"] is True
    r = svc.do_train({"point_id": "h5.pt"})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"
    assert "export" in r


def test_T_HJ4_private_only_and_force(tmp_path, clock):
    svc, _, _ = make_dual(tmp_path, scripted_hosted(
        lambda i, hint, opts: (hint, 0.9)))
    assert svc.do_register(point(point_id="h5.priv", data_class="private",
                                 private_model_only=True))["ok"]
    r = svc.do_train({"point_id": "h5.priv"})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"
    assert svc.do_register(point())["ok"]
    r = svc.do_train({"point_id": "h5.pt", "force": True})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"


# -- T-HJ5 item 8 -----------------------------------------------------------


def hj5_hosted(fit_wrong, c_wrong):
    def sched(i, hint, opts):
        if i < 10:
            return (other(hint, opts) if i in fit_wrong else hint), 0.9
        j = i - 10
        return (other(hint, opts) if j in c_wrong else hint), 0.6
    return scripted_hosted(sched)


def hj5_private(wrong):
    def sched(i, hint, opts):
        if i in wrong:
            return other(hint, opts), 0.6
        return hint, 0.95
    return scripted_private(sched)


def hj5_setup(tmp_path, clock, hosted_fn, private_fn, n_c=12):
    svc, store, cfg = make_dual(tmp_path, hosted_fn, private_fn,
                                fit_rows=10)
    cfg["serve"]["min_auto_support"] = 3
    cfg["serve"]["min_shadow"] = 12
    cfg["train"]["min_examples"] = 10
    cfg["hosted_jev"]["min_compare"] = 10
    assert svc.do_register(point(target_error=0.3))["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    for i in range(10):
        decide_label(svc, "ab"[i % 2])
    assert store.active_hosted_fit("h5.pt")["threshold"] == 0.9
    TR.train_fake_private(store, "h5.pt", threshold=0.7)
    for i in range(n_c):
        decide_label(svc, "ab"[i % 2])
    return svc, store, cfg


def test_T_HJ5_pass_promotes(tmp_path, clock):
    svc, store, _ = hj5_setup(tmp_path, clock, hj5_hosted({4}, {2, 10}),
                              hj5_private({6, 7}))
    comp = svc._private_comparison(store.get_point("h5.pt"))
    assert comp["n"] == 12
    assert comp["discordant_hosted_only"] == 2
    assert comp["discordant_private_only"] == 2
    assert 0.0 <= comp["mcnemar_p"] <= 1.0
    assert comp["pass"] is True
    rep = svc.do_report({"point_id": "h5.pt"})["points"][0]
    assert rep["hosted"]["private_comparison"]["pass"] is True
    r = svc.do_promote({"point_id": "h5.pt", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"], r
    assert store.get_point("h5.pt")["tier"] == "private"


def test_T_HJ5_small_c_refused(tmp_path, clock):
    svc, store, _ = hj5_setup(tmp_path, clock, hj5_hosted({4}, {2}),
                              hj5_private(set()), n_c=5)
    r = svc.do_promote({"point_id": "h5.pt", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"] is False and r["error"] == "private_not_better"


def test_T_HJ5_no_gain_refused(tmp_path, clock):
    svc, store, _ = hj5_setup(tmp_path, clock, hj5_hosted({4}, {2, 10}),
                              scripted_private(
                                  lambda i, hint, opts: (
                                      (other(hint, opts)
                                       if i in (2, 10) else hint), 0.6)))
    r = svc.do_promote({"point_id": "h5.pt", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"] is False and r["error"] == "private_not_better"


def test_T_HJ5_discordant_loss_refused(tmp_path, clock):
    svc, store, _ = hj5_setup(tmp_path, clock, hj5_hosted({4}, {10}),
                              hj5_private({0, 1, 2}))
    comp = svc._private_comparison(store.get_point("h5.pt"))
    assert comp["discordant_hosted_only"] == 3
    assert comp["discordant_private_only"] == 1
    r = svc.do_promote({"point_id": "h5.pt", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"] is False and r["error"] == "private_not_better"


# -- Section D: private serve checks -----------------------------------------

def test_D_private_item1_counts_labelled_rows_not_c(tmp_path, clock):
    # |C| = 5 < min_examples = 10, but 15 labelled tier rows: item 1
    # passes while item 8 still binds on min_compare.
    svc, store, _ = hj5_setup(tmp_path, clock, hj5_hosted({4}, {2}),
                              hj5_private(set()), n_c=5)
    checks = svc._serve_checks_private(store.get_point("h5.pt"), True)
    assert checks["min_examples"]["pass"] is True
    assert checks["private_item8"]["pass"] is False


def test_D_private_item8_vacuous_for_private_only(tmp_path, clock):
    svc, store, _ = make_dual(tmp_path, scripted_hosted(
        lambda i, hint, opts: (hint, 0.9)))
    assert svc.do_register(point(point_id="h5.pt", data_class="private",
                                 private_model_only=True))["ok"]
    checks = svc._serve_checks_private(store.get_point("h5.pt"), True)
    assert checks["private_item8"]["pass"] is True
    assert "not required" in checks["private_item8"]["detail"]


def test_D_item8_needs_active_hosted_fit(tmp_path, clock):
    # a private pool that would pass cond2 on its own: no hosted fit, so
    # item 8 must still fail.
    svc, store, cfg = make_dual(
        tmp_path, scripted_hosted(lambda i, hint, opts: (hint, 0.9)),
        scripted_private(lambda i, hint, opts: (hint, 0.95)),
        fit_rows=200)
    cfg["hosted_jev"]["min_compare"] = 3
    assert svc.do_register(point(target_error=0.3))["ok"]
    assert svc.do_promote({"point_id": "h5.pt", "phase": "shadow"})["ok"]
    TR.train_fake_private(store, "h5.pt", threshold=0.7)
    for i in range(5):
        decide_label(svc, "ab"[i % 2])
    assert store.active_hosted_fit("h5.pt") is None
    comp = svc._private_comparison(store.get_point("h5.pt"))
    assert comp["n"] == 5
    assert comp["pass"] is False


def test_D_check_detail_keys_lowercase(tmp_path, clock):
    svc, store, _ = hj5_setup(tmp_path, clock, hj5_hosted({4}, {2, 10}),
                              hj5_private({6, 7}))
    point = store.get_point("h5.pt")
    for checks in (svc._serve_checks_hosted(point, True),
                   svc._serve_checks_private(point, True)):
        for name, check in checks.items():
            assert "D" + "etail" not in check, name
            assert "detail" in check, name
