"""FakeDecider: scripted Decider for tests (no probe, no JEV).

Default: a peaked distribution on the request state's `label_hint`, read
from the structured state. fn(req): full scripted probabilities
(private-tier stand-ins).
"""
from __future__ import annotations

import json
import random
import time

from .base import DecideRequest, DecideResult, Decider


class FakeDecider(Decider):
    name = "fake"
    decider_tag = "fake"
    model_tag = "fake"

    def __init__(self, store=None, cfg: dict | None = None, fn=None,
                 accuracy: float = 1.0, seed: int = 0,
                 decider_tag: str | None = None,
                 model_tag: str | None = None):
        self.store = store
        self.cfg = cfg or {}
        self.fn = fn
        self.accuracy = accuracy
        self.seed = seed
        self._n = 0
        if decider_tag is not None:
            self.decider_tag = decider_tag
        if model_tag is not None:
            self.model_tag = model_tag

    def decide(self, reqs: list[DecideRequest],
               timeout_s: float | None = None) -> list[DecideResult]:
        _ = timeout_s  # scripted answers never block
        return [self._decide_one(r) for r in reqs]

    def _decide_one(self, req: DecideRequest) -> DecideResult:
        t0 = time.monotonic()
        if self.fn is not None:
            probs = dict(self.fn(req))
            toks = None
        else:
            options = list(req.criteria)
            hint = (req.state or {}).get("label_hint")
            self._n += 1
            rng = random.Random(f"{self.seed}:{self._n}:{req.point_id}")
            if hint in options and rng.random() < self.accuracy:
                top = hint
            elif hint in options:
                top = rng.choice([o for o in options if o != hint])
            else:
                top = rng.choice(options)
            rest = 0.1 / max(1, len(options) - 1)
            probs = {o: (0.9 if o == top else rest) for o in options}
            s = sum(probs.values())
            probs = {o: v / s for o, v in probs.items()}
            toks = max(1, len(json.dumps(req.state or {})) // 4)
        return DecideResult(
            probs=probs, decider=self.decider_tag, model=self.model_tag,
            seconds=time.monotonic() - t0, cost_usd=0.0, tokens_in=toks)
