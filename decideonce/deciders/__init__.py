"""Deciders package: the Decider interface and its backends (build §2)."""
from .base import DecideRequest, DecideResult, Decider, DeciderError
from .fake import FakeDecider
from .local import LocalProbeDecider

__all__ = ["DecideRequest", "DecideResult", "Decider", "DeciderError",
           "FakeDecider", "LocalProbeDecider", "build_deciders"]


def build_deciders(cfg: dict, store):
    """(deciders map, raw JevBackend|None) from [decider] backend (build §2).

    local_legacy -> {"legacy": LocalProbeDecider|FakeDecider} + the raw
    backend (train keeps receiving it). tiered/fake land in H2.
    """
    from ..config import ConfigError
    from ..jev import FakeJev, HttpJev
    from ..config import check_jev_local

    backend = cfg.get("decider", {}).get("backend", "local_legacy")
    if backend == "local_legacy":
        url = cfg["jev"]["url"]
        if url.startswith("fake:"):
            jev = FakeJev(down=url == "fake:down")
            return {"legacy": FakeDecider(jev, store, cfg)}, jev
        check_jev_local(url)  # raises ConfigError(jev_not_local)
        jev = HttpJev(url, cfg["jev"]["timeout_s"])
        return {"legacy": LocalProbeDecider(jev, store, cfg)}, jev
    if backend in ("tiered", "fake"):
        from .hosted import build_hosted_deciders  # noqa: lazy, H2

        return build_hosted_deciders(cfg, store)
    raise ConfigError("bad_decider_backend",
                      f"[decider] backend must be tiered|local_legacy|fake, got {backend!r}")
