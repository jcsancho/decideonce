"""H8 tests: decider-construction contract (T-HB10, P3). Service takes a
deciders map only; local_legacy is refused; this file pins the rules."""
import random

import pytest

from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.config import ConfigError
from decideonce.service import Service
from decideonce.store import Store


def test_T_HB10_default_backend_is_tiered(tmp_path):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    assert cfg["decider"]["backend"] == "tiered"
    assert "jev" not in cfg


def test_T_HB10_local_legacy_refused(tmp_path):
    # P3 deleted the probe with the local_legacy branch: no URL form
    # revives it.
    for url in ("fake:", "fake:down", "http://127.0.0.1:8090"):
        cfg = load_config()
        cfg["store"]["data_dir"] = str(tmp_path)
        cfg["decider"]["backend"] = "local_legacy"
        cfg["jev"] = {"url": url, "timeout_s": 10}
        store = Store(tmp_path / "proj")
        with pytest.raises(ConfigError) as ei:
            build_deciders(cfg, store)
        assert ei.value.code == "bad_decider_backend"


def test_do_train_fake_backend_reports_dry_run_export(tmp_path):
    from decideonce.deciders import build_deciders
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    store = Store(tmp_path / "proj")
    deciders, raw = build_deciders(cfg, store)
    assert raw is None  # fake backend: hosted fake + standalone private
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    assert svc.do_register({"point_id": "h8t", "workflow": "w",
                            "question": "Q?", "kind": "choice",
                            "options": ["a", "b"],
                            "context_fields": {"label_hint": {"type": "string"}},
                            "required_fields": [],
                            "data_class": "public"})["ok"]
    r = svc.do_train({"point_id": "h8t", "force": True})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"
    assert r["export"]["n_labeled"] == 0
    for i in range(70):
        d = svc.do_decide({"point_id": "h8t",
                           "context": {"label_hint": "ab"[i % 2]}})
        assert svc.do_record({"decision_id": d["decision_id"],
                              "choice": "ab"[i % 2]})["ok"]
    r = svc.do_train({"point_id": "h8t", "force": True})
    assert r["ok"] is False and r["error"] == "no_fireworks_key"
    assert r["export"]["dry_run"] is True
    assert r["export"]["n_labeled"] == 70
    assert r["export"]["per_option"] == {"a": 35, "b": 35}


def test_T_HB10_bare_backend_refused(tmp_path):
    """P2: Service takes a deciders map only; a bare backend is refused."""
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    store = Store(tmp_path / "proj")
    with pytest.raises(TypeError):
        Service(cfg, store, object(), rng=random.Random(0))
