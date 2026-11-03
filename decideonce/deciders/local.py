"""LocalProbeDecider: the current probe path wrapped as a Decider (G5).

Wraps a JevBackend (HttpJev/SlowJev duck-types) plus the probe .npz: probe
probabilities when a probe exists for the current version, else zero-shot.
"""
from __future__ import annotations

import time

from .. import probe as P
from .. import render as R
from .base import DecideRequest, DecideResult, Decider, DeciderError


class LocalProbeDecider(Decider):
    name = "local_probe"
    decider_tag = "local_probe"
    model_tag = "local-probe"

    def __init__(self, jev, store, cfg: dict | None = None):
        self.jev = jev
        self.store = store
        self.cfg = cfg or {}

    def decide(self, reqs: list[DecideRequest],
               timeout_s: float | None = None,
               model_kind: str = "probe") -> list[DecideResult]:
        _ = timeout_s  # the local backend owns its timeout (jev.timeout_s)
        return [self._decide_one(r, model_kind) for r in reqs]

    def _decide_one(self, req: DecideRequest, model_kind: str) -> DecideResult:
        point = self.store.get_point(req.point_id)
        if point is None:
            raise DeciderError("bad_response",
                               f"unknown point {req.point_id!r}")
        options = list(point["options"])
        if req.state_text is not None:
            state_text = req.state_text
        else:
            fields = point.get("context_fields", {})
            ctx = {k: req.state.get(k) for k in fields if k in req.state}
            state_text = R.render_state(point, ctx)
        model = self._current_model(point, model_kind)
        t0 = time.monotonic()
        try:
            if model is not None:
                vector, zs_list, toks = self.jev.hidden(
                    point["kind"], state_text, point["question"], options)
                seconds = time.monotonic() - t0
                probe_full = self._probe_probs(model, vector, point)
                zs_full = dict(zip(options, zs_list))
                shown = probe_full if probe_full is not None else zs_full
            else:
                zs_list = self.jev.decide(
                    point["kind"], state_text, point["question"], options)
                seconds = time.monotonic() - t0
                vector, toks, probe_full = None, None, None
                zs_full = dict(zip(options, zs_list))
                shown = zs_full
        except DeciderError:
            raise
        except Exception as e:
            raise DeciderError("http", f"legacy jev failed: {e}") from e
        return DecideResult(
            probs=dict(shown), decider=self.decider_tag,
            model=self.model_tag, seconds=seconds, cost_usd=0.0,
            tokens_in=toks, _hidden=vector, _zs_full=zs_full,
            _probe_full=probe_full)

    def _current_model(self, point: dict, kind: str) -> dict | None:
        model = self.store.active_model(point["point_id"], kind)
        if model is None or model.get("point_version") != point.get(
                "point_version", 1):
            return None
        return model

    def _probe_probs(self, model: dict, vector: list,
                     point: dict) -> dict | None:
        try:
            mp = self.store.models_dir / point["point_id"] / model["path"]
            params = P.load_model(mp)
        except Exception:
            return None
        try:
            return P.predict_full(params, vector, list(point["options"]))
        except Exception:
            return None
