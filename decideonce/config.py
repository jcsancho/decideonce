"""Config loader: decideonce.toml defaults (§8) + file overlay + env overrides."""
from __future__ import annotations

import copy
import os
import tomllib
from pathlib import Path

DEFAULTS: dict = {
    "store": {"data_dir": "~/.decideonce", "project": "default", "max_rows_per_point": 50000},
    "decider": {"backend": "tiered", "timeout_s": 3},
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
        "fake_accuracy": 1.0,
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
        "cookbook_dir": "",
        "cookbook_commit": "57968a2",
        "batch_rows": 24,
        "lora_rank": 64,
        "learning_rate": 1e-4,
        "base_model": "accounts/fireworks/models/muse-glimmer-30b",
        "job_max_hours": 3,
        "worker_idle_min": 30,
        "eval_ready_s": 600,
        "eval_timeout_s": 600,
        "score_price_per_mtok": 1.96,
        "reserve_usd": 0.005,
        "allow_session_serving": False,
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
    "discovery": {
        "llm_calls": ["chat", "complete", "completions.create",
                      "messages.create", "do_decide", "invoke", "generate"],
        "min_consumed": 0.8,
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


def _check_overlay_keys(overlay: dict) -> None:
    """Reject unknown sections and keys (two levels; nested tables unchecked)."""
    for section, values in overlay.items():
        if section not in DEFAULTS:
            raise ConfigError("unknown_config_key",
                              f"unknown config section [{section}]")
        if not isinstance(values, dict):
            raise ConfigError("unknown_config_key",
                              f"config section [{section}] must be a table")
        for key in values:
            if key not in DEFAULTS[section]:
                raise ConfigError("unknown_config_key",
                                  f"unknown config key [{section}] {key!r}")


def _check_discovery(cfg: dict) -> None:
    """Validate the effective [discovery] values (defaults or file overlay)."""
    disc = cfg.get("discovery", {})
    calls = disc.get("llm_calls")
    if (not isinstance(calls, list) or not calls
            or any(not isinstance(c, str) or not c for c in calls)):
        raise ConfigError("bad_discovery_config",
                          "[discovery] llm_calls must be a non-empty list of strings")
    mc = disc.get("min_consumed")
    if isinstance(mc, bool) or not isinstance(mc, (int, float)) \
            or not 0 <= mc <= 1:
        raise ConfigError("bad_discovery_config",
                          "[discovery] min_consumed must be a number in 0-1")


def load_config(path: str | os.PathLike | None = None) -> dict:
    """Load config: DEFAULTS <- TOML file <- env. Missing file is fine (defaults)."""
    cfg = copy.deepcopy(DEFAULTS)
    cfg_path = resolve_config_path(path)
    if cfg_path.is_file():
        with open(cfg_path, "rb") as f:
            overlay = tomllib.load(f)
        # M12 P1: the local probe is gone; refuse pre-M12 configs loudly.
        if "jev" in overlay or (isinstance(overlay.get("decider"), dict)
                               and overlay["decider"].get("backend") == "local_legacy"):
            raise ConfigError("config_v16_migration",
                              'remove [jev]; set [decider] backend = "tiered"')
        _check_overlay_keys(overlay)
        cfg = _deep_merge(cfg, overlay)
    elif "DECIDEONCE_CONFIG" in os.environ or "DECIDE_MCP_CONFIG" in os.environ \
            or path is not None:
        # Explicit path that does not exist: fail loudly, not silently default.
        raise ConfigError("config_not_found", f"config file not found: {cfg_path}")
    _check_discovery(cfg)
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

