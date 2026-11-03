"""Hosted-tier one-time fit: temperature, threshold, tempering (build §7, G8).

Pure functions over stored hosted probabilities; the trigger and storage live
in Service (do_record/do_outcome -> _maybe_fit_hosted) and Store (hosted_fit).
"""
from __future__ import annotations

import math

from . import probe as P

TEMP_LO, TEMP_HI = 0.5, 5.0
PROB_FLOOR = 1e-9


def temper_full(probs: dict[str, float], T: float) -> dict[str, float]:
    """softmax(log(p) / T) over the full distribution (order-preserving)."""
    if T == 1.0:
        return dict(probs)
    logs = [math.log(max(probs[o], PROB_FLOOR)) / T for o in probs]
    m = max(logs)
    exps = [math.exp(x - m) for x in logs]
    s = sum(exps)
    return {o: e / s for o, e in zip(probs, exps)}


def mask_temper(raw: dict[str, float], allowed: list[str],
                T: float) -> dict[str, float]:
    """Allowed-option renormalisation, then temperature (build §7)."""
    masked = P.mask_and_renorm(raw, allowed)
    return temper_full(masked, T)


def nll_at(items: list[tuple[dict, list, str]], T: float) -> float:
    """Mean NLL of the label under softmax(log(p_masked) / T)."""
    total = 0.0
    for masked, allowed, label in items:
        t = temper_full(masked, T)
        total += -math.log(max(t.get(label, 0.0), PROB_FLOOR))
    return total / len(items)


def fit_temperature(items: list[tuple[dict, list, str]]
                    ) -> tuple[float, str | None]:
    """Golden-section search on [0.5, 5] (G8); T = 1 with fewer than 5 errors."""
    errs = sum(1 for masked, allowed, label in items
               if P.argmax(masked, allowed) != label)
    if errs < 5:
        return 1.0, "too_few_errors"
    lo, hi = TEMP_LO, TEMP_HI
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = nll_at(items, c), nll_at(items, d)
    for _ in range(80):
        if fc < fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = nll_at(items, c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = nll_at(items, d)
    return (lo + hi) / 2.0, None


def fit_threshold(pairs: list[tuple[float, bool]], target: float,
                  min_support: int) -> float | None:
    """§6.1 on the given (confidence, correct) pairs: lowest t with
    point-estimate error <= target and >= min_support cases; else null."""
    for t, _ in sorted(pairs, key=lambda p: p[0]):
        idx = [correct for conf, correct in pairs if conf >= t]
        if len(idx) >= min_support and \
                sum(1 for c in idx if not c) / len(idx) <= target:
            return float(t)
    return None
