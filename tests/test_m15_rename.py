"""M15 tests: old names still work with a warning (T-RN1)."""
import random

from decideonce.config import load_config, resolve_config_path
from decideonce.service import Service
from decideonce.store import Store


def test_T_RN1_old_service_names_work_with_warning(tmp_path, capsys):
    from decideonce.config import load_config as _lc

    cfg = _lc()
    cfg["store"]["data_dir"] = str(tmp_path)
    store = Store(tmp_path / "proj")
    svc = Service(cfg, store, {}, rng=random.Random(0))
    p = {"point_id": "rn.pt", "workflow": "w", "question": "Q?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"x": {"type": "string"}}, "required_fields": [],
         "data_class": "public"}
    assert svc.dm_register(p)["ok"]
    assert "deprecated" in capsys.readouterr().err
    assert store.get_point("rn.pt") is not None


def test_T_RN1_old_env_names_work_with_warning(tmp_path, monkeypatch, capsys):
    cfg_file = tmp_path / "old.toml"
    cfg_file.write_text("[store]\nproject = \"fileproj\"\n")
    monkeypatch.setenv("DECIDE_MCP_CONFIG", str(cfg_file))
    monkeypatch.delenv("DECIDEONCE_CONFIG", raising=False)
    assert resolve_config_path() == cfg_file
    assert "deprecated" in capsys.readouterr().err
    monkeypatch.setenv("DECIDE_MCP_PROJECT", "envproj")
    monkeypatch.delenv("DECIDEONCE_PROJECT", raising=False)
    cfg = load_config(str(cfg_file))
    assert cfg["store"]["project"] == "envproj"
    assert "deprecated" in capsys.readouterr().err
