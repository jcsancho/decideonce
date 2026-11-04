"""Deciders package: the Decider interface and its backends (build §2)."""
import sys

from .base import DecideRequest, DecideResult, Decider, DeciderError
from .fake import FakeDecider

__all__ = ["DecideRequest", "DecideResult", "Decider", "DeciderError",
           "FakeDecider", "build_deciders"]

_WARNED: set[str] = set()


def _warn_fake(reason: str) -> None:
    """Loud once-per-process stderr warning for scripted backends (H3)."""
    if reason in _WARNED:
        return
    _WARNED.add(reason)
    print(f"decideonce: FAKE decider ({reason}): scripted answers, "
          "not a real model. Never use in production.", file=sys.stderr)


def build_deciders(cfg: dict, store):
    """(deciders map, None) from [decider] backend (build §2).

    fake -> hosted fake + standalone FakeDecider as "private".
    tiered -> hosted only. (M12 P3 deleted the local_legacy branch with
    the probe; pre-M12 configs are refused at load.)
    """
    from ..config import ConfigError

    backend = cfg.get("decider", {}).get("backend", "tiered")
    if backend in ("tiered", "fake"):
        if backend == "fake":
            _warn_fake('[decider] backend = "fake"')
        from .hosted import build_hosted_deciders

        return build_hosted_deciders(cfg, store)
    raise ConfigError("bad_decider_backend",
                      f"[decider] backend must be tiered|fake, got {backend!r}")
