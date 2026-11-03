"""FakeDecider: FakeJev wrapped as a Decider (tests).

FakeDecider(jev, store): legacy-compatible fake (decider tag "fake").
FakeDecider(fn=...): scripted private-tier stand-in for T-HJ5 (H5): fn(req)
returns the full probability dict.
"""
from __future__ import annotations

import time

from .base import DecideRequest, DecideResult
from .local import LocalProbeDecider


class FakeDecider(LocalProbeDecider):
    name = "fake"
    decider_tag = "fake"
    model_tag = "fake"

    def __init__(self, jev=None, store=None, cfg: dict | None = None,
                 fn=None, decider_tag: str | None = None,
                 model_tag: str | None = None):
        self.jev = jev
        self.store = store
        self.cfg = cfg or {}
        self.fn = fn
        if decider_tag is not None:
            self.decider_tag = decider_tag
        if model_tag is not None:
            self.model_tag = model_tag

    def decide(self, reqs: list[DecideRequest],
               timeout_s: float | None = None,
               model_kind: str = "probe") -> list[DecideResult]:
        if self.fn is None:
            assert self.jev is not None and self.store is not None
            return super().decide(reqs, timeout_s, model_kind)
        out = []
        for req in reqs:
            t0 = time.monotonic()
            out.append(DecideResult(
                probs=dict(self.fn(req)), decider=self.decider_tag,
                model=self.model_tag, seconds=time.monotonic() - t0,
                cost_usd=0.0, tokens_in=None))
        return out
