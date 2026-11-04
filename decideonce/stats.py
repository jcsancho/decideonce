"""Small-sample statistics: Wilson score bounds (95%, z=1.96) plus the
shared probability helpers (moved from probe.py in M12 P3)."""
from __future__ import annotations

import math

Z95 = 1.96


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    s = sum(e)
    return [x / s for x in e]


def mask_and_renorm(full: dict[str, float], allowed: list[str]) -> dict[str, float]:
    """Masked softmax at inference: rule-removed options get 0, renormalise over allowed."""
    vals = {o: max(full.get(o, 0.0), 0.0) for o in allowed}
    s = sum(vals.values())
    if s <= 0:
        n = len(allowed)
        return {o: 1.0 / n for o in allowed} if n else {}
    return {o: v / s for o, v in vals.items()}


def argmax(probs: dict[str, float], order: list[str]) -> str:
    best = max(order, key=lambda o: (probs.get(o, 0.0), -order.index(o)))
    return best


def wilson_bounds(errors: int, n: int, z: float = Z95) -> tuple[float, float]:
    """(lower, upper) Wilson score bounds for the error rate errors/n."""
    if n <= 0:
        return 0.0, 1.0
    p = errors / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def wilson_upper(errors: int, n: int, z: float = Z95) -> float:
    return wilson_bounds(errors, n, z)[1]


def wilson_lower(errors: int, n: int, z: float = Z95) -> float:
    return wilson_bounds(errors, n, z)[0]


def ece(pairs: list[tuple[float, bool]], bins: int = 10) -> float | None:
    """Expected calibration error over (confidence, correct) pairs.

    Ten equal-width confidence bins; None when there are no pairs.
    """
    if not pairs or bins <= 0:
        return None
    n = len(pairs)
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        in_bin = [(c, ok) for c, ok in pairs
                  if (c > lo or (b == 0 and c == 0.0)) and c <= hi]
        if not in_bin:
            continue
        acc = sum(1 for _, ok in in_bin if ok) / len(in_bin)
        conf = sum(c for c, _ in in_bin) / len(in_bin)
        total += len(in_bin) / n * abs(acc - conf)
    return total


def newcombe_paired(a: int, b: int, c: int, d: int, z: float
                    ) -> tuple[float, float, float]:
    """Newcombe's method 10 for paired proportions (Statistics in Medicine
    1998, 17:2635-2650): (diff, lower, upper) for cheap − expensive success.

    a = both pass, b = cheap pass / expensive fail, c = cheap fail /
    expensive pass, d = both fail. z is given (Pocock 2.36 at trial looks,
    1.96 one-sided for downgrade monitoring). Verified against statsmodels
    confint_proportions_paired(method="newcombe"): (10, 5, 2, 20) ->
    -0.0643926..0.2211160 at z = 1.959963984540054.
    """
    n = a + b + c + d
    if n <= 0:
        return 0.0, -1.0, 1.0
    zsq = z * z
    diff = (b - c) / n
    # marginal Wilson intervals: pA = cheap pass rate, pB = expensive pass rate
    x5, x6 = a + b, c + d
    la = (2 * x5 + zsq - z * math.sqrt(zsq + 4 * x5 * x6 / n)) / (2 * (n + zsq))
    ua = (2 * x5 + zsq + z * math.sqrt(zsq + 4 * x5 * x6 / n)) / (2 * (n + zsq))
    x7, x8 = a + c, b + d
    lb = (2 * x7 + zsq - z * math.sqrt(zsq + 4 * x7 * x8 / n)) / (2 * (n + zsq))
    ub = (2 * x7 + zsq + z * math.sqrt(zsq + 4 * x7 * x8 / n)) / (2 * (n + zsq))
    pa, pb = x5 / n, x7 / n
    lo_a, hi_a = pa - la, ua - pa
    lo_b, hi_b = pb - lb, ub - pb
    # adjusted phi (bias correction -n/2 when positive)
    prod = x5 * x6 * x7 * x8
    if prod == 0:
        phi = 0.0
    else:
        det = a * d - b * c
        adj = max(det - n / 2, 0.0) if det > 0 else det
        phi = adj / math.sqrt(prod)
    lower = diff - math.sqrt(max(lo_a * lo_a - 2 * phi * lo_a * hi_b
                                 + hi_b * hi_b, 0.0))
    upper = diff + math.sqrt(max(hi_a * hi_a - 2 * phi * hi_a * lo_b
                                 + lo_b * lo_b, 0.0))
    return diff, lower, upper
