"""HostedJevDecider: System One (OpenRouter/TypeSafe) behind the Decider interface.

Request shape verified live 2026-11-02.
One request per point (pack = false, G7). No retries: every failure is a
DeciderError and do_decide falls back to teacher mode.
"""
from __future__ import annotations

import http.client
import json
import math
import random
import re
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from .base import DecideRequest, DecideResult, Decider, DeciderError

ALLOWED_HOSTS = ("openrouter.ai", "api.typesafe.ai")

# A resolved key must be printable non-space ASCII (B2): anything else (an
# inner newline in particular) makes http.client raise a ValueError whose
# text carries the header value.
KEY_RE = re.compile(r"[\x21-\x7e]+")


def check_host(url: str) -> None:
    """Refuse any hosted-Jev host but the two allowed ones (startup check)."""
    from ..config import ConfigError

    def refuse(why: str) -> None:
        raise ConfigError("decider_host_not_allowed",
                          f"hosted_jev.url refused ({why}): {url!r}")

    # D: urllib/http.client refuse URLs with spaces, controls or
    # non-ASCII (InvalidURL/UnicodeEncodeError at call time); refuse them
    # here instead, at startup.
    if not isinstance(url, str) or KEY_RE.fullmatch(url) is None:
        refuse("URL must be printable ASCII without spaces")
    try:
        parts = urllib.parse.urlparse(url)
    except Exception:
        refuse("unparseable")
    if parts.scheme != "https":
        refuse("scheme must be https")
    if parts.username is not None or parts.password is not None:
        refuse("userinfo not allowed")
    try:
        port = parts.port
    except ValueError:
        refuse("bad port")
    if port is not None and port != 443:
        refuse("port must be absent or 443")
    if (parts.hostname or "") not in ALLOWED_HOSTS:
        refuse(f"host must be one of {list(ALLOWED_HOSTS)}")


def resolve_key(key_ref: str) -> str | None:
    """Resolve an envsave opaque key once, at startup. None when empty/failed.

    B3: stdin is DEVNULL so a vault prompt can never eat the MCP stdio
    handshake. B2: a resolved value outside printable non-space ASCII is
    rejected (None), since it would break header framing downstream.
    """
    if not key_ref:
        return None
    try:
        proc = subprocess.run(["envsave", "get", key_ref],
                              capture_output=True, text=True, timeout=10,
                              stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    key = proc.stdout.strip() or None
    if key is not None and KEY_RE.fullmatch(key) is None:
        return None
    return key


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects: urllib would forward Authorization to the target."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise DeciderError("http", f"redirect refused ({code})")


def _safe_http_message(e: BaseException, key: str | None) -> str:
    """DeciderError detail that never echoes header material (B2).

    http.client raises ValueError with the offending header value inline, so
    ValueErrors map to a static message. Anything else is echoed only after
    redacting the key itself.
    """
    if isinstance(e, ValueError):
        return "invalid header value"
    text = str(e)
    if key and key in text:
        text = text.replace(key, "[redacted]")
    if not text:
        return type(e).__name__
    return f"{type(e).__name__}: {text}"


class UrllibTransport:
    """Default transport: POST with no redirects. Returns (status, payload).

    timeout_s is one overall deadline per call (B4): the request runs on a
    daemon thread and the caller gives up after timeout_s, so a slow-drip
    body cannot stretch the call past the deadline the way a per-operation
    socket timeout lets it.
    """

    def __init__(self):
        opener = urllib.request.OpenerDirector()
        opener.add_handler(urllib.request.HTTPHandler())
        opener.add_handler(urllib.request.HTTPSHandler())
        # HTTPErrorProcessor routes non-2xx through parent.error, so 3xx
        # reaches _NoRedirect (which raises) and 4xx/5xx raise HTTPError.
        opener.add_handler(urllib.request.HTTPErrorProcessor())
        opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
        opener.add_handler(_NoRedirect())
        self.opener = opener

    def __call__(self, url: str, headers: dict, body: bytes,
                 timeout_s: float) -> tuple[int, bytes]:
        box: dict = {}

        def run() -> None:
            try:
                box["result"] = self._post(url, headers, body, timeout_s)
            except BaseException as e:  # re-raised on the caller thread
                box["error"] = e

        worker = threading.Thread(target=run, daemon=True)
        worker.start()
        worker.join(timeout_s)
        if "result" in box:
            return box["result"]
        if "error" in box:
            raise box["error"]
        raise DeciderError("timeout", f"call exceeded {timeout_s}s deadline")

    def _post(self, url: str, headers: dict, body: bytes,
              timeout_s: float) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=body, headers=headers,
                                     method="POST")
        try:
            with self.opener.open(req, timeout=timeout_s) as resp:
                return int(resp.status), resp.read()
        except DeciderError:
            raise
        except urllib.error.HTTPError as e:
            try:
                payload = e.read()
            except Exception:
                payload = b""
            return int(e.code), payload
        except (TimeoutError, socket.timeout) as e:
            raise DeciderError("timeout", str(e)) from e
        except urllib.error.URLError as e:
            if isinstance(e.reason, (TimeoutError, socket.timeout)):
                raise DeciderError("timeout", str(e.reason)) from e
            raise DeciderError("http", f"connection failed: {e.reason}") from e
        except (OSError, http.client.HTTPException) as e:
            # RemoteDisconnected, ConnectionReset, IncompleteRead,
            # BadStatusLine and friends (B1). socket.timeout subclasses
            # OSError but is caught above, so this stays "http".
            raise DeciderError("http", f"connection failed: {e}") from e
        except ValueError as e:
            # Invalid header framing (B1); never echo: the text carries the
            # header value (B2).
            raise DeciderError("http", "invalid header value") from e


class FakeHostedTransport:
    """Test transport (build §4): fn(body) -> (status, json); records bodies.

    The default fn plants FakeJev-style signal: with probability `accuracy`
    the distribution is peaked on the state's label_hint field, else on
    another option. fn may raise DeciderError (e.g. timeout) to script
    failures.
    """

    def __init__(self, fn=None, accuracy: float = 0.97, seed: int = 0):
        self.fn = fn
        self.accuracy = accuracy
        self.seed = seed
        self.requests: list[dict] = []
        self._n = 0

    @property
    def calls(self) -> int:
        return len(self.requests)

    def __call__(self, url: str, headers: dict, body: bytes,
                 timeout_s: float) -> tuple[int, bytes]:
        payload = json.loads(body.decode("utf-8"))
        self.requests.append(payload)
        self._n += 1
        status, resp = (self.fn or self._default)(payload)
        return status, json.dumps(resp).encode()

    def _default(self, body: dict) -> tuple[int, dict]:
        (point_id, q), = body["questions"].items()
        options = list(q["criteria"])
        hint = (body.get("state") or {}).get("label_hint")
        rng = random.Random(f"{self.seed}:{self._n}:{point_id}")
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
        toks = max(1, len(json.dumps(body)) // 4)
        return 200, {
            "answers": {point_id: {"choice": top, "probabilities": probs,
                                   "confidence": probs[top]}},
            "usage": {"input_tokens": toks, "output_tokens": 0,
                      "cost": 0.00004}}


def _utc_month() -> str:
    return time.strftime("%Y-%m", time.gmtime())


def _finite_float(v) -> float | None:
    """Guarded float(): None for bools, non-numbers, NaN/inf and values
    float() cannot represent (huge ints raise OverflowError). Never raises."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    try:
        f = float(v)
    except (OverflowError, ValueError):
        return None
    return f if math.isfinite(f) else None


class HostedJevDecider(Decider):
    name = "hosted_jev"

    def __init__(self, *, url: str, model: str, key: str | None,
                 store=None, timeout_s: float = 3,
                 price_per_mtok_in: float = 0.042,
                 monthly_budget_usd: float = 0,
                 reserve_usd: float = 0.0005,
                 stale_after_s: float | None = None,
                 transport=None, allow_no_store: bool = False):
        from ..config import ConfigError
        if store is None and not allow_no_store:
            # B9: without a store there is no budget and no privacy gate.
            raise TypeError("HostedJevDecider requires a store "
                            "(tests may pass allow_no_store=True)")
        try:
            budget = float(monthly_budget_usd or 0)
            reserve = float(reserve_usd)
        except (TypeError, ValueError, OverflowError) as e:
            raise ConfigError("invalid_budget",
                              "hosted_jev budget/reserve must be numbers") from e
        # B5: NaN admits every call, a negative reserve admits every call,
        # and reserve 0 used to silently become 0.0005.
        if not math.isfinite(budget) or budget < 0:
            raise ConfigError(
                "invalid_budget",
                f"hosted_jev.monthly_budget_usd must be finite and >= 0, "
                f"got {monthly_budget_usd!r}")
        if not math.isfinite(reserve) or reserve <= 0:
            raise ConfigError(
                "invalid_budget",
                f"hosted_jev.reserve_usd must be finite and > 0, "
                f"got {reserve_usd!r}")
        self.url = url
        self.model = model
        self.key = key
        self.store = store
        self.timeout_s = timeout_s
        self.price_per_mtok_in = price_per_mtok_in
        self.monthly_budget_usd = budget
        self.reserve_usd = reserve
        self.stale_after_s = stale_after_s
        self.transport = transport or UrllibTransport()

    def __repr__(self):  # the key never appears (canary T-HJ6)
        return (f"HostedJevDecider(url={self.url!r}, model={self.model!r}, "
                f"key={'set' if self.key else 'unset'})")

    def decide(self, reqs: list[DecideRequest],
               timeout_s: float | None = None) -> list[DecideResult]:
        return [self._decide_one(r, timeout_s or self.timeout_s)
                for r in reqs]

    @staticmethod
    def _remote_ok(point: dict) -> bool:
        if point.get("data_class", "private") == "public":
            return True
        ro = point.get("remote_ok", [])
        if ro is True:  # v1.6 boolean: both hosts allowed
            return True
        return "hosted_jev" in (ro or [])

    def _decide_one(self, req: DecideRequest, timeout: float) -> DecideResult:
        if self.store is not None:
            point = self.store.get_point(req.point_id)
            if point is not None and not self._remote_ok(point):
                raise DeciderError("remote_not_allowed", req.point_id)
        if not self.key:
            raise DeciderError("no_key",
                               "hosted_jev.key_ref is empty or unresolvable")
        body = {"model": self.model, "state": req.state,
                "questions": {req.point_id: {
                    "type": "choice", "instructions": req.instructions,
                    "criteria": req.criteria}}}
        body_bytes = json.dumps(body).encode()
        month = _utc_month()
        reservation = None
        if self.store is not None:
            # Stale window: 10 x max timeouts (B4), never below the live
            # call's own 10 x timeout. Each reservation carries its window,
            # so processes with different timeouts never free each other's
            # live reservations (B7).
            window = 10 * timeout
            if self.stale_after_s:
                window = max(window, self.stale_after_s)
            try:
                reservation = self.store.hosted_reserve(
                    month, "hosted_jev", self.reserve_usd,
                    self.monthly_budget_usd, window)
            except Exception as e:
                raise DeciderError("http", f"spend busy: {e}") from e
            if reservation is None:
                raise DeciderError(
                    "budget_exceeded",
                    f"hosted_jev month {month}: budget "
                    f"${self.monthly_budget_usd:.4f} exhausted")
        t0 = time.monotonic()
        try:
            status, payload = self.transport(
                self.url, {"Content-Type": "application/json",
                           "Authorization": "Bearer " + self.key},
                body_bytes, timeout)
            seconds = time.monotonic() - t0
            probs, cost, toks, warnings = self._parse(
                status, payload, req, body)
        except DeciderError as e:
            # R2-8: on timeout the worker thread outlives the deadline and
            # the upstream call may still be billed. Keep the reservation
            # for the stale reaper (10x window) instead of freeing it while
            # billing may land. Other errors never billed: release.
            if reservation is not None and e.code != "timeout":
                self._release(reservation)
            raise
        except Exception as e:
            # B1: anything the transport or parse did not already map (a
            # scripted-transport error, an OverflowError, ...) still frees
            # the reservation and becomes a DeciderError, never "internal".
            if reservation is not None:
                self._release(reservation)
            raise DeciderError("http",
                               _safe_http_message(e, self.key)) from e
        if reservation is not None:
            self._settle(reservation, month, cost)
        return DecideResult(probs=probs, decider="hosted_jev",
                            model=self.model, seconds=seconds,
                            cost_usd=cost, tokens_in=toks,
                            warnings=warnings)

    def _release(self, reservation: str) -> None:
        try:
            assert self.store is not None
            self.store.hosted_release(reservation)
        except Exception:
            pass

    def _settle(self, reservation: str, month: str, cost: float) -> None:
        assert self.store is not None
        try:
            self.store.hosted_settle(reservation, month, "hosted_jev", cost)
            return
        except Exception:
            pass
        # B7: a failed settle must not lose the cost. hosted_settle rolls
        # back as a unit, so retry once; if the store is still failing,
        # record the spend directly, then always free the reservation.
        try:
            self.store.hosted_settle(reservation, month, "hosted_jev", cost)
            return
        except Exception:
            pass
        try:
            self.store.spend_add(month, "hosted_jev", cost)
        except Exception:
            pass
        finally:
            self._release(reservation)

    def _parse(self, status: int, payload: bytes, req: DecideRequest,
               body: dict) -> tuple[dict, float, int | None, list]:
        if status in (401, 403):
            raise DeciderError("http", "auth_failed")
        if status == 429:
            raise DeciderError("http", "rate_limited")
        if status != 200:
            raise DeciderError("http", f"status {status}")
        try:
            resp = json.loads(payload.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise DeciderError("bad_response", "invalid JSON")
        answers = resp.get("answers") if isinstance(resp, dict) else None
        ans = answers.get(req.point_id) if isinstance(answers, dict) else None
        if not isinstance(ans, dict):
            raise DeciderError("bad_response", "missing point answer")
        raw = ans.get("probabilities")
        if not isinstance(raw, dict):
            raise DeciderError("bad_response", "missing probabilities")
        for k in raw:
            if k not in req.criteria:
                raise DeciderError("bad_response",
                                   f"unknown option {k!r}")
        probs: dict[str, float] = {}
        for o in req.criteria:
            v = raw.get(o, 0.0)
            # B1: math.isfinite passes for huge ints, but float() then
            # raises OverflowError, so convert (guarded) before judging.
            f = _finite_float(v)
            if f is None or f < 0 or f > 1:
                raise DeciderError("bad_response",
                                   f"bad probability for {o!r}")
            probs[o] = f
        s = sum(probs.values())
        if s <= 0:
            raise DeciderError("bad_response", "all zeros")
        probs = {o: v / s for o, v in probs.items()}
        usage = resp.get("usage") if isinstance(resp.get("usage"), dict) \
            else {}
        cost = _finite_float(usage.get("cost"))
        if cost is not None and cost < 0:
            cost = None
        toks = usage.get("input_tokens")
        if isinstance(toks, bool) or not isinstance(toks, (int, float)):
            toks = None
        else:
            try:
                toks = int(toks)
            except (OverflowError, ValueError):
                toks = None
        warnings: list[str] = []
        if cost is None:
            if toks is not None:
                try:
                    cost = toks * self.price_per_mtok_in / 1e6
                except OverflowError:
                    cost = None
                if cost is not None and not math.isfinite(cost):
                    cost = None
            if cost is None and toks is not None:
                toks = None
        if cost is None:
            est = max(1, len(json.dumps(body)) // 4)
            cost = est * self.price_per_mtok_in / 1e6
            toks = est
            warnings.append("cost_estimated")
        return probs, float(cost), toks, warnings


def record_hosted_error(store, point_id: str, e: DeciderError) -> None:
    """Count a hosted failure for do_report's errors-by-type (H5)."""
    if e.code == "timeout":
        store.counter_inc("hosted_error_timeout", point_id)
    elif e.code == "http" and e.msg == "auth_failed":
        store.counter_inc("hosted_error_auth_failed", point_id)
    elif e.code == "http" and e.msg == "rate_limited":
        store.counter_inc("hosted_error_rate_limited", point_id)
    elif e.code == "bad_response":
        store.counter_inc("hosted_error_bad_response", point_id)
    elif e.code == "http":
        store.counter_inc("hosted_error_http", point_id)


def build_hosted_deciders(cfg: dict, store):
    """(deciders map, None) for backend tiered/fake. Raises ConfigError to
    refuse startup (decider_host_not_allowed, pack_not_supported,
    budget_below_reservation)."""
    from ..config import ConfigError
    from ..jev import FakeJev
    from .fake import FakeDecider

    hj = cfg["hosted_jev"]
    check_host(hj["url"])
    if hj.get("pack"):
        # D: packing several points into one request is not implemented
        # (one request per point, G7); refuse instead of silently
        # ignoring the flag.
        raise ConfigError(
            "pack_not_supported",
            "hosted_jev.pack = true is not supported: the decider sends "
            "one request per point (G7)")
    # B5: no `or` masking (reserve 0 used to silently become 0.0005, NaN
    # passed every check); HostedJevDecider validates finiteness and sign.
    budget = hj.get("monthly_budget_usd", 0)
    budget = 0 if budget is None else budget
    reserve = hj.get("reserve_usd", 0.0005)
    reserve = 0.0005 if reserve is None else reserve
    def _ok_num(v) -> bool:
        return (isinstance(v, (int, float)) and not isinstance(v, bool)
                and math.isfinite(v))

    if _ok_num(budget) and _ok_num(reserve) and 0 < budget < reserve:
        raise ConfigError(
            "budget_below_reservation",
            f"hosted_jev.monthly_budget_usd ${budget} is below one "
            f"reservation ${reserve}: every call would be silently blocked")
    # B4: stale release at 10 x max timeouts. Both processes build from the
    # same config, so the MCP server (decider.timeout_s) and the gateway
    # (gateway.jev_timeout_s) share one window.
    stale_after_s = 10 * max(float(cfg["decider"]["timeout_s"]),
                             float(cfg["gateway"]["jev_timeout_s"]))
    backend = cfg.get("decider", {}).get("backend", "tiered")
    if backend == "fake":
        fake_transport = FakeHostedTransport(
            accuracy=float(hj.get("fake_accuracy", 0.97)))
        fake_jev = FakeJev()
        return {
            "hosted_jev": HostedJevDecider(
                url=hj["url"], model=hj["model"], key="fake", store=store,
                timeout_s=cfg["decider"]["timeout_s"],
                price_per_mtok_in=hj["price_per_mtok_in"],
                monthly_budget_usd=budget, reserve_usd=reserve,
                stale_after_s=stale_after_s,
                transport=fake_transport),
            "private": FakeDecider(fake_jev, store, cfg,
                                     decider_tag="private",
                                     model_tag="fake-private"),
        }, fake_jev
    key = resolve_key(hj.get("key_ref", ""))
    return {
        "hosted_jev": HostedJevDecider(
            url=hj["url"], model=hj["model"], key=key, store=store,
            timeout_s=cfg["decider"]["timeout_s"],
            price_per_mtok_in=hj["price_per_mtok_in"],
            monthly_budget_usd=budget, reserve_usd=reserve,
            stale_after_s=stale_after_s),
    }, None
