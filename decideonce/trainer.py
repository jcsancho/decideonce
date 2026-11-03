"""Trainer interface (§7.5) and FakeTrainer (tests, autopilot M16).

Main spec §7.5: `estimate(rows)`, `launch(rows) -> job_id`, `status(job_id)`,
`checkpoint(job_id)`. The live `FireworksTrainer` is M11 (needs the cookbook
recipe, the API client and P4's measurements); until then `Service(trainer=...)`
takes a `FakeTrainer` in tests and `None` in production, where training
automation stays dormant and manual tiered `do_train` keeps reporting
`private_tier_not_built`.
"""
from __future__ import annotations

import json
import math
import uuid
from abc import ABC, abstractmethod


class Trainer(ABC):
    """Async training-job backend (main spec §7.5)."""

    # A live trainer needs the Fireworks key; fakes never touch the network
    # and skip the key gate the way FakeJev/FakeHostedTransport do.
    needs_key = True

    @abstractmethod
    def estimate(self, rows: list[dict], kind: str) -> dict:
        """Cost estimate for a job over rows. Returns at least estimate_usd."""

    @abstractmethod
    def launch(self, rows: list[dict], kind: str) -> str:
        """Upload rows and launch a job. Returns a job_id (status running)."""

    @abstractmethod
    def status(self, job_id: str) -> dict:
        """Poll a job: {"status": running|done|failed, ...metrics when done}."""

    @abstractmethod
    def checkpoint(self, job_id: str) -> dict:
        """Per-point models of a finished job: {"per_point": {pid: {...}}}."""


def check_fireworks_host(url: str) -> None:
    """Refuse any Fireworks host but api.fireworks.ai (startup check, §7.6)."""
    import urllib.parse

    from .config import ConfigError

    def refuse(why: str) -> None:
        raise ConfigError("decider_host_not_allowed",
                          f"fireworks.url refused ({why}): {url!r}")

    try:
        parts = urllib.parse.urlparse(url)
    except Exception:
        refuse("unparseable")
    if parts.scheme != "https":
        refuse("scheme must be https")
    try:
        userinfo = bool(parts.username or parts.password)
        port = parts.port
        host = (parts.hostname or "")
    except ValueError:
        refuse("unparseable host/port")
    if userinfo:
        refuse("userinfo not allowed")
    if port is not None and port != 443:
        refuse("port must be absent or 443")
    if host != "api.fireworks.ai":
        refuse("host must be api.fireworks.ai")


def _tokens_of(context: dict) -> int:
    return max(1, len(json.dumps(context, sort_keys=True)) // 4)


class FakeTrainer(Trainer):
    """Offline trainer: learns a planted `label_hint` signal (§7.5).

    estimate() prices the §7.3 formula (train tokens × epochs plus evaluation
    decisions) from the [fireworks] prices, so autopilot budget tests run
    offline. `estimate_override` scripts an exact estimate instead.
    Jobs stay running until the test calls complete()/fail() explicitly, so
    job_running tests are deterministic. `uploaded` records every launch's
    rows for no-upload assertions (T-AP7).
    """

    needs_key = False

    def __init__(self, cfg: dict | None = None, estimate_override: float | None = None,
                 smoke_decisions: int = 10):
        fw = (cfg or {}).get("fireworks", {}) if isinstance(cfg, dict) else {}
        self.epochs = int(fw.get("epochs", 2) or 2)
        self.smoke_examples = int(fw.get("smoke_examples", 32) or 32)
        self.train_price_per_mtok = float(fw.get("train_price_per_mtok", 5.86))
        self.eval_price_per_decision = float(fw.get("eval_price_per_decision", 0.0012))
        self.estimate_override = estimate_override
        self.smoke_decisions = smoke_decisions
        self.jobs: dict[str, dict] = {}
        self.uploaded: list[dict] = []  # one entry per launch: {job_id, kind, rows}

    def estimate(self, rows: list[dict], kind: str) -> dict:
        train_rows = rows if kind == "full" else rows[:self.smoke_examples]
        train_toks = sum(_tokens_of(r.get("context") or {}) for r in train_rows)
        train_usd = train_toks * self.epochs * self.train_price_per_mtok / 1e6
        n_eval = sum(1 for r in rows if r.get("split") in ("calib", "heldout"))
        if kind == "smoke":
            n_eval += self.smoke_decisions
        eval_usd = n_eval * self.eval_price_per_decision
        total = train_usd + eval_usd
        if self.estimate_override is not None:
            total = float(self.estimate_override)
        return {"estimate_usd": total, "train_usd": train_usd, "eval_usd": eval_usd,
                "train_rows": len(train_rows), "eval_rows": n_eval,
                "train_tokens": train_toks, "epochs": self.epochs, "kind": kind}

    def launch(self, rows: list[dict], kind: str) -> str:
        job_id = uuid.uuid4().hex[:12]
        self.jobs[job_id] = {"status": "running", "kind": kind,
                             "rows": [dict(r) for r in rows]}
        self.uploaded.append({"job_id": job_id, "kind": kind,
                              "rows": [dict(r) for r in rows]})
        return job_id

    def complete(self, job_id: str) -> dict:
        """Test hook: finish a running job successfully. Returns its status."""
        job = self.jobs[job_id]
        if job["status"] != "running":
            return self.status(job_id)
        rows = job["rows"]
        train = rows if job["kind"] == "full" else rows[:self.smoke_examples]
        counts: dict[str, dict[str, dict[str, int]]] = {}
        for r in train:
            hint = (r.get("context") or {}).get("label_hint")
            if hint is None:
                continue
            per = counts.setdefault(r["point_id"], {})
            per.setdefault(str(hint), {}).update(
                {r["label"]: per.get(str(hint), {}).get(r["label"], 0) + 1})
        learned = {pid: {h: max(v, key=lambda lab: (v[lab], lab))
                        for h, v in per.items()}
                   for pid, per in counts.items()}
        held = [r for r in rows if r.get("split") == "heldout"] or rows
        ok = sum(1 for r in held
                 if learned.get(r["point_id"], {}).get(
                     str((r.get("context") or {}).get("label_hint"))) == r["label"])
        n_eval = len(held) + (self.smoke_decisions if job["kind"] == "smoke" else 0)
        toks = sum(_tokens_of(r.get("context") or {}) for r in train)
        job["status"] = "done"
        job["metrics"] = {
            "accuracy": (ok / len(held)) if held else None,
            "n_train": len(train), "n_eval": n_eval, "train_tokens": toks,
            "hint_points": sorted(learned),
        }
        job["learned"] = learned
        return self.status(job_id)

    def fail(self, job_id: str, message: str = "fake failure") -> dict:
        """Test hook: fail a running job."""
        self.jobs[job_id]["status"] = "failed"
        self.jobs[job_id]["error"] = message
        return self.status(job_id)

    def status(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        out: dict = {"status": job["status"], "kind": job["kind"]}
        if job["status"] == "done":
            out["metrics"] = dict(job["metrics"])
        if job["status"] == "failed":
            out["error"] = job.get("error", "failed")
        return out

    def checkpoint(self, job_id: str) -> dict:
        job = self.jobs[job_id]
        if job["status"] != "done":
            raise KeyError(f"job {job_id} has no checkpoint ({job['status']})")
        per_point = {}
        for pid in {r["point_id"] for r in job["rows"]}:
            per_point[pid] = {"threshold": 0.5,
                              "accuracy": (job["metrics"] or {}).get("accuracy"),
                              "hint_map": (job.get("learned") or {}).get(pid, {})}
        return {"per_point": per_point, "sampler": f"fakeckpt:{job_id}"}


def split_rows(rows: list[dict], holdout: float, calib: float) -> list[dict]:
    """Time-ordered per-point split (§7.2 step 1): train/calib/heldout tails."""
    by_point: dict[str, list[dict]] = {}
    for r in rows:
        by_point.setdefault(r["point_id"], []).append(r)
    out = []
    for pid, rs in by_point.items():
        n = len(rs)
        n_hold = min(n - 1, int(round(n * holdout))) if n > 1 else 0
        n_calib = min(max(n - n_hold - 1, 0), int(round(n * calib))) if n > 1 else 0
        for i, r in enumerate(rs):
            split = "train"
            if i >= n - n_hold:
                split = "heldout"
            elif i >= n - n_hold - n_calib:
                split = "calib"
            out.append({**r, "split": split})
    out.sort(key=lambda r: (r.get("ts", ""), r.get("decision_id", "")))
    return out


def estimate_tokens(rows: list[dict]) -> int:
    """Total ~tokens over export rows (chars // 4 each)."""
    return sum(_tokens_of(r.get("context") or {}) for r in rows)
