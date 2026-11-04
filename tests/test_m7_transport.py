"""M7 tests: MCP stdio transport (T-E*)."""
import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = str(Path(__file__).resolve().parents[1])


def child_env(config_path):
    return {**os.environ, "DECIDEONCE_CONFIG": config_path,
            "PYTHONPATH": REPO + os.pathsep + os.environ.get("PYTHONPATH", "")}


def write_config(path, tmp_path):
    path.write_text(f'[store]\ndata_dir = "{tmp_path}/data"\nproject = "t"\n'
                    '[decider]\nbackend = "fake"\n')
    return str(path)


async def roundtrip(config_path):
    params = StdioServerParameters(command=sys.executable, args=["-m", "decideonce.server"],
                                   env=child_env(config_path))
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()
            tools = await s.list_tools()
            point = {"point_id": "triage.route", "workflow": "support",
                     "question": "Where to route?", "kind": "choice",
                     "options": ["billing", "tech"],
                     "context_fields": {"msg": {"type": "string"}},
                     "required_fields": ["msg"], "data_class": "public"}
            reg = await s.call_tool("do_register", {"payload": point})
            dec = await s.call_tool("do_decide", {"payload": {"point_id": "triage.route",
                                                              "context": {"msg": "my bill"}}})
            did = json.loads(dec.content[0].text)["decision_id"]
            rec = await s.call_tool("do_record", {"payload": {"decision_id": did,
                                                              "choice": "billing"}})
            return tools, reg, dec, rec


def test_T_E1_stdio_round_trip(tmp_path):
    cfg = write_config(tmp_path / "c.toml", tmp_path)
    tools, reg, dec, rec = asyncio.run(roundtrip(cfg))
    names = {t.name for t in tools.tools}
    assert {"do_register", "do_decide", "do_record"} <= names
    assert json.loads(reg.content[0].text)["ok"]
    assert "snippet" in json.loads(reg.content[0].text)
    assert json.loads(dec.content[0].text)["mode"] == "teacher"
    assert json.loads(rec.content[0].text)["ok"]


def test_P1_server_refuses_pre_m12_config(tmp_path):
    # Replaces T-E2 (non-local jev url): the probe is gone, so a [jev]
    # section or backend = "local_legacy" refuses with the migration code.
    legacy = tmp_path / "legacy.toml"
    legacy.write_text('[store]\ndata_dir = "x"\nproject = "t"\n'
                      '[jev]\nurl = "http://127.0.0.1:8090"\n')
    p = subprocess.run([sys.executable, "-m", "decideonce.server"],
                       capture_output=True, text=True, timeout=60,
                       env=child_env(str(legacy)))
    assert p.returncode != 0
    assert "config_v16_migration" in p.stderr
    assert 'remove [jev]; set [decider] backend = "tiered"' in p.stderr
    old_backend = tmp_path / "old.toml"
    old_backend.write_text('[store]\ndata_dir = "x"\nproject = "t"\n'
                           '[decider]\nbackend = "local_legacy"\n')
    p = subprocess.run([sys.executable, "-m", "decideonce.server"],
                       capture_output=True, text=True, timeout=60,
                       env=child_env(str(old_backend)))
    assert p.returncode != 0
    assert "config_v16_migration" in p.stderr


def test_S5_server_help_and_bogus(monkeypatch, capsys):
    from decideonce import server as SRV
    monkeypatch.setattr(sys, "argv", ["decideonce", "--help"])
    with pytest.raises(SystemExit) as e:
        SRV.main()
    assert e.value.code == 0
    out = capsys.readouterr().out
    assert "With no arguments, starts the server" in out
    assert "autopilot" in out
    monkeypatch.setattr(sys, "argv", ["decideonce", "bogus"])
    with pytest.raises(SystemExit) as e2:
        SRV.main()
    assert e2.value.code == 2


def test_S5_gateway_help_does_not_start(monkeypatch, capsys):
    from decideonce import gateway as GW
    monkeypatch.setattr(sys, "argv", ["decideonce-gateway", "--help"])
    with pytest.raises(SystemExit) as e:
        GW.main()
    assert e.value.code == 0
    assert "[gateway]" in capsys.readouterr().out
