"""Decider: use decideonce from Python without the MCP round trip (§4).

from decideonce.client import Decider
d = Decider(project="support")
r = d.decide("refund.action", ctx)
choice = r.choice if r.mode in ("auto", "rule") else my_llm_classify(ctx, r.allowed_options)
d.record(r.decision_id, choice)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field


@dataclass
class Decision:
    decision_id: str
    mode: str
    choice: str | None
    allowed_options: list
    probabilities: dict | None = None
    confidence: float | None = None
    threshold: float | None = None
    instruction: str = ""
    warnings: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)


class Decider:
    """Same store, same code paths as the MCP tools."""

    def __init__(self, project: str | None = None, config_path: str | None = None,
                 jev=None, seed: int | None = None):
        from .config import data_root, load_config
        from .deciders import build_deciders
        from .service import Service
        from .store import Store
        cfg = load_config(config_path)
        if project is not None:
            cfg["store"]["project"] = project
        self.cfg = cfg
        self.store = Store(data_root(cfg))
        if jev is None:
            jev, _ = build_deciders(cfg, self.store)
        self.svc = Service(cfg, self.store, jev,
                           rng=random.Random(seed) if seed is not None else None)

    def close(self) -> None:
        self.store.close()

    def decide(self, point_id: str, context: dict, teacher_cost: dict | None = None) -> Decision:
        args: dict = {"point_id": point_id, "context": context}
        if teacher_cost is not None:
            args["teacher_cost"] = teacher_cost
        r = self.svc.do_decide(args)
        if not r["ok"]:
            raise RuntimeError(f"do_decide failed: {r['error']}: {r.get('message')}")
        return Decision(decision_id=r["decision_id"], mode=r["mode"], choice=r["choice"],
                        allowed_options=r["allowed_options"], probabilities=r["probabilities"],
                        confidence=r["confidence"], threshold=r["threshold"],
                        instruction=r["instruction"], warnings=r.get("warnings") or [], raw=r)

    def record(self, decision_id: str, choice: str, teacher_cost: dict | None = None) -> dict:
        args: dict = {"decision_id": decision_id, "choice": choice}
        if teacher_cost is not None:
            args["teacher_cost"] = teacher_cost
        return self.svc.do_record(args)

    def outcome(self, decision_id: str, ok: bool | None = None,
                correct_choice: str | None = None, note: str | None = None) -> dict:
        args: dict = {"decision_id": decision_id}
        if ok is not None:
            args["ok"] = ok
        if correct_choice is not None:
            args["correct_choice"] = correct_choice
        if note is not None:
            args["note"] = note
        return self.svc.do_outcome(args)

    def register(self, point: dict) -> dict:
        return self.svc.do_register(point)

    def train(self, point_id: str, dry_run: bool = False) -> dict:
        return self.svc.do_train({"point_id": point_id, "dry_run": dry_run})

    def promote(self, point_id: str, phase: str, confirm: bool = False) -> dict:
        return self.svc.do_promote({"point_id": point_id, "phase": phase, "confirm": confirm})

    def report(self, point_id: str | None = None, format: str = "json") -> dict:
        args: dict = {"format": format}
        if point_id is not None:
            args["point_id"] = point_id
        return self.svc.do_report(args)

    def extract_call1(self, transcript: dict | None = None, path: str | None = None,
                      transcript_id: str | None = None, pass_: int = 1,
                      **kw) -> dict:
        args: dict = dict(kw)
        if transcript is not None:
            args["transcript"] = transcript
        if path is not None:
            args["path"] = path
        if transcript_id is not None:
            args["transcript_id"] = transcript_id
        if pass_ == 2:
            args["pass"] = 2
        return self.svc.do_extract(args)

    def extract_call2(self, extraction_id: str, extractor: str,
                      records: list, new_points: list | None = None) -> dict:
        args: dict = {"extraction_id": extraction_id, "extractor": extractor,
                      "records": records}
        if new_points is not None:
            args["new_points"] = new_points
        return self.svc.do_extract(args)

    def plan_run(self, workflow: str, input: str, context: dict, **kw) -> dict:
        return self.svc.do_plan_run({"workflow": workflow, "input": input,
                                     "context": context, **kw})

    def plan_result(self, plan_id: str, **kw) -> dict:
        return self.svc.do_plan_result({"plan_id": plan_id, **kw})
