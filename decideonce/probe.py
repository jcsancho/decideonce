"""Probe inference: linear head on frozen JEV hidden states.

Model file: models/<point_id>/v<point_version>-<n>.npz with arrays W (D, K), b (K,),
temp (scalar) and classes (K option names in column order). A JSON card with the
same basename sits next to it.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def softmax(logits: list[float]) -> list[float]:
    m = max(logits)
    e = [math.exp(x - m) for x in logits]
    s = sum(e)
    return [x / s for x in e]


def load_model(npz_path: str | Path) -> dict:
    z = np.load(str(npz_path), allow_pickle=False)
    return {"W": z["W"], "b": z["b"], "temp": float(z["temp"]),
            "classes": [str(c) for c in z["classes"]]}


def predict_full(model: dict, vector: list[float], options: list[str]) -> dict[str, float]:
    """Full probe distribution over the point's options (aligned by class name)."""
    v = np.asarray(vector, dtype=np.float64)
    logits = (v @ model["W"] + model["b"]) / model["temp"]
    probs = softmax([float(x) for x in logits])
    by_class = dict(zip(model["classes"], probs))
    return {o: by_class.get(o, 0.0) for o in options}


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
