"""Config loader: decideonce.toml defaults (§8) + file overlay + env overrides."""
from __future__ import annotations

import copy
import ipaddress
import os
import tomllib
import urllib.parse
from pathlib import Path

DEFAULTS: dict = {
    "store": {"data_dir": "~/.decideonce", "project": "default", "max_rows_per_point": 50000},
    "jev": {"url": "http://127.0.0.1:8090", "timeout_s": 10},
    "decider": {"backend": "local_legacy", "timeout_s": 3},
    "hosted_jev": {
        "url": "https://openrouter.ai/api/v1/systemone",
        "model": "typesafe/jev-1.13",
        "key_ref": "",
        "price_per_mtok_in": 0.042,
        "monthly_budget_usd": 0,
        "fit_rows": 200,
        "min_private_gain": 0.05,
        "min_compare": 100,
        "redemote_window_days": 30,
        "pack": False,
        "reserve_usd": 0.0005,
        "fake_accuracy": 0.97,
    },
    "analyze": {
        "min_group": 20,
        "weights": {"R2": 2, "R3": 2, "R4_high": 2, "R4_mid": 1, "R5": 1, "R6": -1, "R8": 1},
        "good_fit": 6,
        "possible": 3,
    },
    "train": {
        "min_train": 60,
        "min_examples": 200,
        "min_per_option": 5,
        "holdout": 0.25,
        "calib": 0.15,
        "teacher_only_weight": 1.0,
        "teacher_alarm": 0.10,
    },
    "fireworks": {
        "url": "https://api.fireworks.ai",
        "key_ref": "",
        "monthly_budget_usd": 0,
        "allow_paid_training": False,
        "smoke_examples": 32,
        "epochs": 2,
        "train_price_per_mtok": 5.86,
        "eval_price_per_decision": 0.0012,
    },
    "autopilot": {
        "enabled": False,
        "points": [],
        "max_target_error": 0.02,
        "auto_serve": True,
        "auto_train": True,
        "max_job_usd": 0.0,
        "max_monthly_jobs_usd": 0.0,
        "min_new_rows": 100,
        "free_training_until": "",
        "cooldown_days": 7,
        "check_interval_min": 10,
        "notify_cmd": "",
    },
    "autopilot_trust": {
        "public_key": "",
    },
    "serve": {
        "target_error": 0.02,
        "min_auto_support": 30,
        "min_coverage": 0.20,
        "min_margin_vs_random": 0.02,
        "min_option_recall": 0.8,
        "min_shadow": 30,
        "audit_rate": 0.10,
        "drift_window": 100,
        "allow_auto_irreversible": False,
    },
    "gateway": {
        "host": "127.0.0.1",
        "port": 8787,
        "local_token": "",
        "upstreams": {"openai": "https://api.openai.com", "anthropic": "https://api.anthropic.com"},
        "log_unmatched": False,
        "log_allow": [],
        "trace_data_class": "private",
        "trace_retention_days": 30,
        "point_retention_days": 90,
        "min_template_chars": 40,
        "template_min_match": 0.95,
        "max_state_chars": 8000,
        "format_min_n": 200,
        "format_match_min": 0.97,
        "max_unparsed": 0.02,
        "jev_timeout_s": 5,
        "upstream_timeout_s": 120,
    },
    "extract": {
        "audit_rate": 0.2,
        # §15.3 gate: a point's extracted rows train only with >= min_audited
        # audited transcripts, agreement >= min_agreement, and audited share
        # >= 0.8 x audit_rate.
        "min_audited": 30,
        "min_agreement": 0.9,
        "weight": 0.5,
        "transcript_retention_days": 90,
    },
    "plan": {
        "min_at_start": 0.9,
        "target_error": 0.02,
        "min_shadow": 30,
        "audit_rate": 0.10,
        "max_quality_drop": 0.05,
        "trial_looks": [60, 120, 200, 400],
        "trial_z": 2.36,
        "pair_timeout_h": 24,
        "downgrade_audit_rate": 0.2,
        "monitor_window": 120,
        "monitor_min_pairs": 30,
        "allow_pilot_force": False,
        "check_timeout_s": 30,
    },
}


class ConfigError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in overlay.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _old_env_warn(old: str, new: str) -> None:
    import sys as _sys

    print(f"decideonce: env {old} is deprecated, use {new}",
          file=_sys.stderr)


def resolve_config_path(path: str | os.PathLike | None = None) -> Path:
    """The config file load_config would read (may not exist)."""
    if path:
        return Path(path)
    if os.environ.get("DECIDEONCE_CONFIG"):
        return Path(os.environ["DECIDEONCE_CONFIG"])
    if os.environ.get("DECIDE_MCP_CONFIG"):
        _old_env_warn("DECIDE_MCP_CONFIG", "DECIDEONCE_CONFIG")
        return Path(os.environ["DECIDE_MCP_CONFIG"])
    # M15 compat: old filename still read when the new one is absent.
    if not Path("decideonce.toml").is_file() \
            and Path("decide-mcp.toml").is_file():
        import sys as _sys

        print("decideonce: decide-mcp.toml is deprecated, use decideonce.toml",
              file=_sys.stderr)
        return Path("decide-mcp.toml")
    return Path("decideonce.toml")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config: DEFAULTS <- TOML file <- env. Missing file is fine (defaults)."""
    cfg = copy.deepcopy(DEFAULTS)
    cfg_path = resolve_config_path(path)
    if cfg_path.is_file():
        with open(cfg_path, "rb") as f:
            cfg = _deep_merge(cfg, tomllib.load(f))
    elif "DECIDEONCE_CONFIG" in os.environ or "DECIDE_MCP_CONFIG" in os.environ \
            or path is not None:
        # Explicit path that does not exist: fail loudly, not silently default.
        raise ConfigError("config_not_found", f"config file not found: {cfg_path}")
    if os.environ.get("DECIDEONCE_PROJECT"):
        cfg["store"]["project"] = os.environ["DECIDEONCE_PROJECT"]
    elif os.environ.get("DECIDE_MCP_PROJECT"):
        _old_env_warn("DECIDE_MCP_PROJECT", "DECIDEONCE_PROJECT")
        cfg["store"]["project"] = os.environ["DECIDE_MCP_PROJECT"]
    return cfg


def data_root(cfg: dict) -> Path:
    root = Path(os.path.expanduser(cfg["store"]["data_dir"])).expanduser() \
        / cfg["store"]["project"]
    # M15: move ~/.decide-mcp/<project> to ~/.decideonce/<project> when the
    # new one is absent; never delete.
    try:
        old_base = Path(os.path.expanduser("~/.decide-mcp"))
        new_base = Path(os.path.expanduser("~/.decideonce"))
        if root.parent == new_base and not root.exists():
            old = old_base / cfg["store"]["project"]
            if old.is_dir():
                import shutil as _shutil
                import sys as _sys

                new_base.mkdir(parents=True, exist_ok=True)
                _shutil.move(str(old), str(root))
                print(f"decideonce: moved data dir {old} -> {root}",
                      file=_sys.stderr)
    except Exception:
        pass
    return root


def check_jev_local(url: str) -> None:
    """Refuse to start unless jev.url resolves to loopback (§7.3)."""
    host = urllib.parse.urlparse(url).hostname or ""
    if host in ("localhost",):
        return
    try:
        if ipaddress.ip_address(host).is_loopback:
            return
    except ValueError:
        pass
    raise ConfigError("jev_not_local", f"jev.url must be local (127.0.0.1/localhost), got {url!r}")
