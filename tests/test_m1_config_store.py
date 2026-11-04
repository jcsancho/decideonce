"""M1 tests: config (T-C*), store (T-S*), FakeDecider basics."""
import pytest

from decideonce import config as C
from decideonce.deciders.base import DecideRequest, DeciderError
from decideonce.deciders.fake import FakeDecider
from decideonce.store import SCHEMA_VERSION, Store


def test_T_C1_config_defaults_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DECIDEONCE_CONFIG", raising=False)
    monkeypatch.delenv("DECIDEONCE_PROJECT", raising=False)
    cfg = C.load_config()
    assert cfg["train"]["min_train"] == 60
    assert cfg["train"]["min_examples"] == 200
    assert cfg["train"]["holdout"] == 0.25
    assert cfg["serve"]["target_error"] == 0.02
    assert cfg["serve"]["min_auto_support"] == 30
    assert cfg["serve"]["audit_rate"] == 0.10
    assert cfg["serve"]["drift_window"] == 100
    assert cfg["analyze"]["min_group"] == 20
    assert cfg["store"]["max_rows_per_point"] == 50000
    assert cfg["gateway"]["port"] == 8787
    assert cfg["extract"]["weight"] == 0.5
    assert cfg["plan"]["trial_z"] == 2.36
    # a TOML file overlays defaults
    (tmp_path / "custom.toml").write_text('[train]\nmin_train = 5\n')
    cfg2 = C.load_config(tmp_path / "custom.toml")
    assert cfg2["train"]["min_train"] == 5
    assert cfg2["train"]["min_examples"] == 200


def test_T_C2_env_var_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg_file = tmp_path / "c.toml"
    cfg_file.write_text('[store]\nproject = "fileproj"\n')
    monkeypatch.setenv("DECIDEONCE_CONFIG", str(cfg_file))
    monkeypatch.setenv("DECIDEONCE_PROJECT", "envproj")
    cfg = C.load_config()
    assert cfg["store"]["project"] == "envproj"
    monkeypatch.delenv("DECIDEONCE_PROJECT")
    assert C.load_config()["store"]["project"] == "fileproj"
    # explicit missing path fails loudly
    monkeypatch.setenv("DECIDEONCE_CONFIG", str(tmp_path / "nope.toml"))
    with pytest.raises(C.ConfigError):
        C.load_config()


def test_P1_config_v16_migration(tmp_path):
    assert C.load_config()["decider"]["backend"] == "tiered"
    jev_toml = tmp_path / "jev.toml"
    jev_toml.write_text('[jev]\nurl = "http://127.0.0.1:8090"\n')
    with pytest.raises(C.ConfigError) as e:
        C.load_config(jev_toml)
    assert e.value.code == "config_v16_migration"
    assert e.value.args[0] == 'remove [jev]; set [decider] backend = "tiered"'
    legacy_toml = tmp_path / "legacy.toml"
    legacy_toml.write_text('[decider]\nbackend = "local_legacy"\n')
    with pytest.raises(C.ConfigError) as e2:
        C.load_config(legacy_toml)
    assert e2.value.code == "config_v16_migration"


def _point(pid="p.action"):
    return {"point_id": pid, "workflow": "w", "kind": "choice",
            "question": "q?", "options": ["a", "b"],
            "context_fields": {"f": {"type": "string"}},
            "phase": "collect", "point_version": 1}


def _row(pid="p.action", did="d1", ts="2026-01-01T00:00:00Z"):
    return {"decision_id": did, "point_id": pid, "point_version": 1, "ts": ts,
            "mode": "teacher", "source": "do_decide"}


def test_T_S1_schema_migration_on_empty_db(tmp_path):
    root = tmp_path / "data" / "proj"
    s = Store(root)
    assert (root / "decide.db").exists()
    assert s.version() == SCHEMA_VERSION
    assert s.get_point("missing") is None
    s.put_point(_point())
    assert s.get_point("p.action")["workflow"] == "w"
    s.insert_decision(_row())
    assert s.get_decision("d1")["mode"] == "teacher"
    s.close()
    # reopen: migrate is idempotent
    s2 = Store(root)
    assert s2.version() == SCHEMA_VERSION
    assert s2.get_decision("d1")["mode"] == "teacher"
    s2.close()


def test_T_S2_retention_trims_oldest_rows(tmp_path):
    s = Store(tmp_path / "r")
    s.put_point(_point())
    for i in range(10):
        s.insert_decision(_row(did=f"d{i:02d}", ts=f"2026-01-01T00:00:{i:02d}Z"),
                          max_rows_per_point=4)
    assert s.count_decisions("p.action") == 4
    assert s.get_decision("d00") is None
    assert s.get_decision("d09") is not None
    s.close()


def _req(hint=None):
    return DecideRequest(point_id="p", state={"label_hint": hint} if hint else {},
                         instructions="q?", criteria={"a": "a", "b": "b"})


def test_fakedecider_deterministic_shape():
    d = FakeDecider()
    p1 = d.decide([_req()])[0].probs
    d2 = FakeDecider()
    p2 = d2.decide([_req()])[0].probs
    assert p1 == p2  # same seed, same call index
    assert set(p1) == {"a", "b"}
    assert abs(sum(p1.values()) - 1.0) < 1e-9


def test_fakedecider_planted_signal_tracks_hint():
    d = FakeDecider()
    probs_a = d.decide([_req("a")])[0].probs
    assert max(probs_a, key=probs_a.get) == "a"
    d2 = FakeDecider()
    probs_b = d2.decide([_req("b")])[0].probs
    assert max(probs_b, key=probs_b.get) == "b"
    assert probs_b["b"] > 0.5
    # accuracy 0 never peaks the hint
    d3 = FakeDecider(accuracy=0.0)
    probs_miss = d3.decide([_req("a")])[0].probs
    assert max(probs_miss, key=probs_miss.get) == "b"


def test_T_S0_discovery_defaults_and_overlay(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DECIDEONCE_CONFIG", raising=False)
    monkeypatch.delenv("DECIDEONCE_PROJECT", raising=False)
    cfg = C.load_config()
    assert cfg["discovery"]["min_consumed"] == 0.8
    assert cfg["discovery"]["llm_calls"] == [
        "chat", "complete", "completions.create", "messages.create",
        "do_decide", "invoke", "generate"]
    (tmp_path / "d.toml").write_text(
        "[discovery]\nmin_consumed = 0.5\nllm_calls = [\"chat\"]\n")
    cfg2 = C.load_config(tmp_path / "d.toml")
    assert cfg2["discovery"]["min_consumed"] == 0.5
    assert cfg2["discovery"]["llm_calls"] == ["chat"]


def test_T_S0_discovery_validation_and_unknown_keys(tmp_path):
    bad = ["[discovery]\nmin_consumed = 1.5\n",
           "[discovery]\nmin_consumed = -0.1\n",
           "[discovery]\nmin_consumed = \"high\"\n",
           "[discovery]\nllm_calls = \"chat\"\n",
           "[discovery]\nllm_calls = []\n",
           "[discovery]\nllm_calls = [\"chat\", 3]\n",
           "[discovery]\nmin_consumed = 0.8\nextra_key = 1\n",
           "[nope]\nx = 1\n",
           "[train]\nno_such_key = 1\n"]
    for i, text in enumerate(bad):
        p = tmp_path / f"bad{i}.toml"
        p.write_text(text)
        with pytest.raises(C.ConfigError):
            C.load_config(p)


def test_fakedecider_scripted_failure_propagates():
    def boom(req):
        raise DeciderError("timeout", "scripted")
    d = FakeDecider(fn=boom)
    with pytest.raises(DeciderError):
        d.decide([_req("a")])
