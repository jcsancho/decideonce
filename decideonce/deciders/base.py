"""Decider interface (hosted-jev build §2): all decision backends behind one call."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class DeciderError(Exception):
    """Backend failure. code in {"timeout", "http", "bad_response",
    "budget_exceeded", "no_key", "host_not_allowed", "remote_not_allowed",
    "unavailable"}. msg carries detail (e.g. "auth_failed",
    "rate_limited" for "http"; "private_warming" for "unavailable")."""

    def __init__(self, code: str, msg: str = ""):
        super().__init__(f"{code}: {msg}" if msg else code)
        self.code = code
        self.msg = msg


def record_decider_error(store, decider, point_id: str,
                         e: DeciderError) -> None:
    """Count a decider failure under its own backend (F7: private errors
    must not inflate the hosted counters)."""
    if getattr(decider, "name", "") == "private":
        from .session import record_private_error

        record_private_error(store, point_id, e)
    else:
        from .hosted import record_hosted_error

        record_hosted_error(store, point_id, e)


@dataclass(frozen=True)
class DecideRequest:
    point_id: str
    state: dict  # ordered: "policy" (if any), then "input", then registered fields
    instructions: str  # the point's question
    criteria: dict  # every registered option -> option_help or the option name
    state_text: str | None = None  # legacy backends: exact text for jev.hidden


@dataclass
class DecideResult:
    probs: dict  # over ALL registered options, sums to 1 (unmasked)
    decider: str  # "hosted_jev" | "private" | "fake"
    model: str  # e.g. "typesafe/jev-1.13"
    seconds: float
    cost_usd: float  # 0.0 for fake
    tokens_in: int | None = None
    warnings: list = field(default_factory=list)


class Decider(ABC):
    name: str

    @abstractmethod
    def decide(self, reqs: list[DecideRequest],
               timeout_s: float | None = None) -> list[DecideResult]:
        """One result per request, in order. Raises DeciderError on failure."""
