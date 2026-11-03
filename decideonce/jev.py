"""JEV backends: HttpJev (jev-mlx server) and FakeJev (tests) (§7.4)."""
from __future__ import annotations

import hashlib
import json
import math
import random
import urllib.request
from abc import ABC, abstractmethod


class JevError(Exception):
    pass


class JevBackend(ABC):
    @abstractmethod
    def decide(self, kind: str, state: str, question: str, options: list[str]) -> list[float]:
        """Zero-shot probabilities aligned with options."""

    @abstractmethod
    def hidden(self, kind: str, state: str, question: str,
               options: list[str]) -> tuple[list[float], list[float], int]:
        """(hidden vector, zero-shot probabilities, prompt tokens)."""


class HttpJev(JevBackend):
    def __init__(self, url: str, timeout_s: float = 10):
        self.url = url.rstrip("/")
        self.timeout_s = timeout_s

    def _post(self, path: str, body: dict) -> dict:
        try:
            req = urllib.request.Request(
                self.url + path, data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.load(resp)
        except Exception as e:
            raise JevError(f"JEV {path} failed: {e}") from e

    def decide(self, kind, state, question, options):
        body: dict = {"kind": kind, "state": state, "question": question}
        if kind == "choice":
            body["options"] = options
        return list(self._post("/v1/decide", body)["probabilities"])

    def hidden(self, kind, state, question, options):
        body: dict = {"kind": kind, "state": state, "question": question}
        if kind == "choice":
            body["options"] = options
        r = self._post("/v1/hidden", body)
        return list(r["hidden"]), list(r["probabilities"]), int(r["prompt_tokens"])


class FakeJev(JevBackend):
    """Deterministic test backend. Hidden vectors are hash features of the prompt text with a
    planted signal: a `label_hint: <option>` line in the state boosts the hidden dims of that
    option's index, so training tests can check that the probe learns. Zero-shot decide() is
    deliberately weak (near uniform), so a trained probe beats it (T-T2)."""

    DIM = 64
    SIGNAL = 5.0

    def __init__(self, down: bool = False, seed: int = 0):
        self.down = down
        self.seed = seed
        self.calls: list[dict] = []

    def _check(self):
        if self.down:
            raise JevError("FakeJev is down")

    @staticmethod
    def _stable_seed(*parts: str) -> int:
        h = hashlib.sha256("\x00".join(parts).encode()).digest()
        return int.from_bytes(h[:8], "little")

    def _hint_index(self, state: str, options: list[str]) -> int | None:
        for line in state.splitlines():
            if line.startswith("label_hint:"):
                hint = line.split(":", 1)[1].strip()
                if hint in options:
                    return options.index(hint)
        return None

    def hidden(self, kind, state, question, options):
        self._check()
        self.calls.append({"op": "hidden", "kind": kind})
        if kind == "noul":
            options = ["false", "true"]
        rng = random.Random(self._stable_seed(str(self.seed), kind, state, question,
                                              ",".join(options)))
        vec = [rng.gauss(0, 0.5) for _ in range(self.DIM)]
        idx = self._hint_index(state, options)
        if idx is not None:
            vec[idx % self.DIM] += self.SIGNAL
            vec[(idx + 16) % self.DIM] += self.SIGNAL / 2
        probs = self._weak_probs(kind, state, question, options)
        return vec, probs, len(state.split()) + len(question.split())

    def _weak_probs(self, kind, state, question, options):
        rng = random.Random(self._stable_seed("zs", str(self.seed), kind, state, question))
        w = [1.0 + rng.uniform(-0.05, 0.05) for _ in options]
        s = sum(w)
        return [x / s for x in w]

    def decide(self, kind, state, question, options):
        self._check()
        self.calls.append({"op": "decide", "kind": kind})
        if kind == "noul":
            options = ["false", "true"]
        return self._weak_probs(kind, state, question, options)
