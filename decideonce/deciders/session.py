"""FireworksSessionDecider (M11 F7): the private tier behind one call.

Live shadow/serve scoring through a WorkerSession per trained model
(model + T from the point's active private card, F8). One DecideRequest
becomes one worker row (state + single choice question, label None:
the worker inserts the dummy label); the worker's logprobs come back
in registered order and leave here as softmax(logits / T) over ALL
registered options (Service masks + renorms over allowed, which is the
masked softmax with T).

Gates (GP11): without ``fireworks.allow_session_serving`` every call
refuses with ``remote_not_allowed`` (silent teacher, like collect) and
no worker ever starts; the worker also needs ``monthly_budget_usd > 0``
(the reserve refuses otherwise). Evaluation (F8) scores through the
worker directly, not through this gate.

Spend (GP12): reserve-then-settle under tier ``fireworks`` with
``fireworks.reserve_usd``, settled at ``score_price_per_mtok`` over the
worker's measured tokens_in — the same discipline as HostedJevDecider
(B4/B5/B7/R2-8). A worker timeout keeps its reservation (the call may
still bill); other failures release it.

Errors: warming reads as ``("unavailable", "private_warming")`` →
teacher mode with ``jev_unavailable``, counted for the report. Worker
timeouts read as ``"timeout"`` (kept reservation, counted separately
from hosted timeouts by decider name). A point with no usable private
model reads as ``("unavailable", "no_private_model")``.
"""
from __future__ import annotations

import math
import threading
from datetime import datetime, timezone

from .. import stats as S
from ..fireworks_worker import WorkerError, WorkerSession, WorkerUnavailable
from ..fireworks_worker import tokens_of as _worker_tokens
from .base import DecideRequest, DecideResult, Decider, DeciderError


def _utc_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _temperature(card: dict | None) -> float:
    """Effective T from the card (F8 fits it); 1.0 when missing/garbage."""
    try:
        t = float((card or {}).get("temperature", 1.0))
    except (TypeError, ValueError, OverflowError):
        return 1.0
    return t if (math.isfinite(t) and t > 0) else 1.0


def record_private_error(store, point_id: str, e: DeciderError) -> None:
    """Count a private-tier failure for do_report (F7; F9 displays)."""
    if e.msg == "private_warming":
        store.counter_inc("private_warming", point_id)
    elif e.code == "timeout":
        store.counter_inc("private_error_timeout", point_id)
    elif e.code == "bad_response":
        store.counter_inc("private_error_bad_response", point_id)
    elif e.code == "unavailable":
        store.counter_inc("private_error_unavailable", point_id)


class FireworksSessionDecider(Decider):
    name = "private"

    def __init__(self, store, cfg: dict, key: str | None,
                 worker_argv_fn=None, clock=None):
        from ..config import ConfigError

        if store is None:  # B9: no store means no model, no budget
            raise TypeError("FireworksSessionDecider requires a store")
        fw = cfg.get("fireworks", {})
        try:
            budget = float(fw.get("monthly_budget_usd") or 0)
            reserve = float(fw.get("reserve_usd"))
            price = float(fw.get("score_price_per_mtok"))
        except (TypeError, ValueError, OverflowError) as e:
            raise ConfigError("invalid_budget",
                              "fireworks budget/reserve/price must be "
                              "numbers") from e
        if not math.isfinite(budget) or budget < 0:  # B5
            raise ConfigError(
                "invalid_budget",
                "fireworks.monthly_budget_usd must be finite and >= 0, "
                f"got {fw.get('monthly_budget_usd')!r}")
        if not math.isfinite(reserve) or reserve <= 0:
            raise ConfigError(
                "invalid_budget",
                "fireworks.reserve_usd must be finite and > 0, "
                f"got {fw.get('reserve_usd')!r}")
        if not math.isfinite(price) or price < 0:
            raise ConfigError(
                "invalid_budget",
                "fireworks.score_price_per_mtok must be finite and >= 0, "
                f"got {fw.get('score_price_per_mtok')!r}")
        self.store = store
        self.cfg = cfg
        self.key = key
        self.timeout_s = float(cfg["decider"]["timeout_s"])
        self.monthly_budget_usd = budget
        self.reserve_usd = reserve
        self.score_price = price
        self.stale_after_s = 10 * max(
            float(cfg["decider"]["timeout_s"]),
            float(cfg["gateway"]["jev_timeout_s"]))
        self._worker_argv_fn = worker_argv_fn
        self._clock = clock
        self._sessions: dict[str, WorkerSession] = {}
        self._lock = threading.Lock()

    def __repr__(self):  # the key never appears (canary T-HJ6)
        return f"FireworksSessionDecider(key={'set' if self.key else 'unset'})"

    def close(self) -> None:
        """Terminate all worker sessions. Idempotent."""
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()
        for s in sessions:
            try:
                s.close()
            except Exception:
                pass

    def sessions_summary(self) -> dict:
        """F9: worker state by model (GP9: the account id masked, so
        reports show ``accounts/acct-…/models/<id>``)."""
        from ..trainer_log import redact_account

        with self._lock:
            items = list(self._sessions.items())
        return {redact_account(m): s.status() for m, s in items}

    # -- model lookup ------------------------------------------------------
    def _model_for(self, point_id: str) -> tuple[str, float]:
        point = self.store.get_point(point_id)
        if point is None:
            raise DeciderError("unavailable", "no_private_model")
        m = self.store.active_model(point_id, "private")
        if m is None or m.get("point_version") != point.get(
                "point_version", 1):
            raise DeciderError("unavailable", "no_private_model")
        card = m.get("card") or {}
        model = card.get("model_ref") or ""
        if not model:
            raise DeciderError("unavailable", "no_private_model")
        return model, _temperature(card)

    def _session_for(self, model: str) -> WorkerSession:
        with self._lock:
            sess = self._sessions.get(model)
            if sess is None:
                argv = None
                if self._worker_argv_fn is not None:
                    argv = self._worker_argv_fn(model)
                sess = WorkerSession(
                    self.cfg, model, self.store.root / "jobs", self.key or "",
                    worker_argv=argv, clock=self._clock)
                self._sessions[model] = sess
            return sess

    # -- Decide --------------------------------------------------------------
    def _check_remote(self, reqs: list[DecideRequest]) -> None:
        """§7.6: a private point without "fireworks" in remote_ok is never
        sent. Fakes stay ungated (T-HJ7: they send nothing); this live
        decider refuses before any worker starts."""
        for r in reqs:
            point = self.store.get_point(r.point_id)
            if point is None:
                continue
            if point.get("data_class", "private") == "public":
                continue
            ro = point.get("remote_ok", [])
            if ro is True or "fireworks" in (ro or []):
                continue
            raise DeciderError(
                "remote_not_allowed",
                f"{r.point_id} is private without 'fireworks' in "
                "remote_ok")

    def decide(self, reqs: list[DecideRequest],
               timeout_s: float | None = None) -> list[DecideResult]:
        if not self.cfg["fireworks"].get("allow_session_serving"):
            # GP11: live serving needs the explicit flag; silent teacher.
            raise DeciderError("remote_not_allowed", "serving_not_allowed")
        if not self.key:
            raise DeciderError("unavailable", "no_fireworks_key")
        self._check_remote(reqs)
        timeout = timeout_s or self.timeout_s
        return [self._decide_one(r, timeout) for r in reqs]

    def decide_as(self, model: str, temperature: float,
                  reqs: list[DecideRequest],
                  timeout_s: float | None = None) -> list[DecideResult]:
        """Score with an explicit model + T (F9 §7.2 step 5: the pending
        checkpoint logs beside the serving one). Same gates and spend."""
        if not self.cfg["fireworks"].get("allow_session_serving"):
            raise DeciderError("remote_not_allowed", "serving_not_allowed")
        if not self.key:
            raise DeciderError("unavailable", "no_fireworks_key")
        self._check_remote(reqs)
        timeout = timeout_s or self.timeout_s
        return [self._score_with(model, temperature, r, timeout)
                for r in reqs]

    @staticmethod
    def _worker_row(req: DecideRequest) -> dict:
        return {"state": req.state,
                "questions": {req.point_id: {
                    "type": "choice", "instructions": req.instructions,
                    "criteria": dict(req.criteria), "label": None,
                    "weight": 1.0}}}

    def _decide_one(self, req: DecideRequest, timeout: float) -> DecideResult:
        model, temp = self._model_for(req.point_id)
        return self._score_with(model, temp, req, timeout)

    def _score_with(self, model: str, temp: float, req: DecideRequest,
                    timeout: float) -> DecideResult:
        try:
            temp = float(temp)
        except (TypeError, ValueError, OverflowError):
            temp = 1.0
        if not math.isfinite(temp) or temp <= 0:
            temp = 1.0
        sess = self._session_for(model)
        rows = [self._worker_row(req)]
        month = _utc_month()
        window = 10 * timeout  # B4/B7, as the hosted decider
        window = max(window, self.stale_after_s)
        try:
            reservation = self.store.hosted_reserve(
                month, "fireworks", self.reserve_usd,
                self.monthly_budget_usd, window)
        except Exception as e:
            raise DeciderError("http", f"spend busy: {e}") from e
        if reservation is None:
            raise DeciderError(
                "budget_exceeded",
                f"fireworks month {month}: budget "
                f"${self.monthly_budget_usd:.4f} exhausted")
        try:
            out = sess.score(rows)
        except WorkerError as e:
            self._release(reservation)
            raise DeciderError("bad_response",
                               f"worker: {e}"[:160]) from e
        except WorkerUnavailable as e:
            if e.reason == "warming":
                self._release(reservation)
                raise DeciderError("unavailable",
                                   "private_warming") from e
            if e.reason == "timeout":
                # R2-8: the worker may still bill; keep the reservation
                # for the stale reaper instead of freeing it.
                raise DeciderError("timeout", "worker_timeout") from e
            self._release(reservation)
            raise DeciderError("unavailable",
                               f"worker_{e.reason}") from e
        except Exception as e:  # B1: never "internal"
            self._release(reservation)
            raise DeciderError("bad_response",
                               f"worker failed: {e}"[:160]) from e
        keys = list(req.criteria)
        lps = (out.get("logprobs") or [None])[0]
        if not isinstance(lps, list) or len(lps) != len(keys):
            self._release(reservation)
            raise DeciderError("bad_response",
                               "worker logprobs do not match options")
        try:
            lps = [float(v) for v in lps]
            if not all(math.isfinite(v) for v in lps):
                raise ValueError("non-finite")
        except (TypeError, ValueError):
            self._release(reservation)
            raise DeciderError("bad_response",
                               "worker logprobs are not finite floats")
        toks = out.get("tokens_in")
        warnings: list[str] = []
        if isinstance(toks, bool) or not isinstance(toks, (int, float)) \
                or not math.isfinite(toks) or toks < 0:
            toks = _worker_tokens(rows)  # GP12 estimate fallback
            warnings.append("cost_estimated")
        else:
            toks = int(toks)
        cost = toks * self.score_price / 1e6
        if not math.isfinite(cost) or cost < 0:
            cost = 0.0
        self._settle(reservation, month, cost)
        probs = dict(zip(keys, S.softmax([v / temp for v in lps])))
        return DecideResult(probs=probs, decider="private", model=model,
                            seconds=float(out.get("seconds") or 0.0),
                            cost_usd=cost, tokens_in=toks,
                            warnings=warnings)

    # -- spend ---------------------------------------------------------------
    def _release(self, reservation: str) -> None:
        try:
            self.store.hosted_release(reservation)
        except Exception:
            pass

    def _settle(self, reservation: str, month: str, cost: float) -> None:
        try:
            self.store.hosted_settle(reservation, month, "fireworks", cost)
            return
        except Exception:
            pass
        # B7: a failed settle must not lose the cost (as hosted).
        try:
            self.store.hosted_settle(reservation, month, "fireworks", cost)
            return
        except Exception:
            pass
        try:
            self.store.spend_add(month, "fireworks", cost)
        except Exception:
            pass
        finally:
            self._release(reservation)
