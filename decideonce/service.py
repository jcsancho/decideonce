"""Service: the do_* tools as plain methods (shared by the MCP server and Decider).

Every method takes a dict and returns {ok: true, ...} or {ok: false, error, message}.
"""
from __future__ import annotations

import json
import os
import random
import time
import uuid

from . import hostedfit as HF
from . import labels as L
from . import stats as S
from . import render as R
from . import rules as RU
from . import schemaval as SV
from .store import row_key as _row_key
from .store import row_seq as _row_seq
from .deciders import DecideRequest, Decider, DeciderError
from .deciders.base import record_decider_error
from .errors import err, ok

TEACHER_INSTRUCTION = ("Decide yourself among allowed_options, then call "
                       "do_record(decision_id, choice).")
AUTO_INSTRUCTION = "Use choice. Do not re-decide."

# Row sources that carry live teacher labels and may count as promotion evidence.
LIVE_SOURCES = L.LIVE_SOURCES

# do_extract call 1 returns this shape for the extractor child to fill (§15.2).
RECORD_SCHEMA = {
    "type": "object",
    "required": ["point_id", "choice", "choice_quote", "decision_pos",
                 "context", "evidence"],
    "properties": {
        "point_id": {"type": "string"},
        "choice": {"type": "string"},
        "choice_quote": {
            "type": "string",
            "description": "exact raw transcript text starting at decision_pos",
        },
        "decision_pos": {
            "type": "object",
            "required": ["event", "offset"],
            "properties": {
                "event": {"type": "integer",
                          "description": "event index; input is -1"},
                "offset": {"type": "integer", "minimum": 0,
                           "description": "0-based char offset into raw content"},
            },
        },
        "context": {"type": "object",
                    "description": "field values for this point's context_fields"},
        "evidence": {
            "type": "object",
            "description": "one entry per context field, all before decision_pos",
            "additionalProperties": {
                "type": "object",
                "required": ["pos", "quote"],
                "properties": {
                    "pos": {"type": "object",
                            "required": ["event", "offset"],
                            "properties": {
                                "event": {"type": "integer"},
                                "offset": {"type": "integer", "minimum": 0}}},
                    "quote": {"type": "string"},
                },
            },
        },
    },
}

EXTRACT_INSTRUCTIONS = [
    "one record per decision actually taken",
    "decision_pos = where the choice is first committed (not a later restatement)",
    "quote the choice starting exactly there",
    "quote evidence for every context field",
    "never use text at or after the decision",
]

ADDENDUM_HEAD = ("Decisions already made for this task (do not revisit them; if a fact you find "
                 "contradicts one, say\n\"DECISION CONFLICT: <point_id>\" and stop):")

CONFLICT_RE = "DECISION CONFLICT:\\s*(\\S+)"


def _addendum(decided_entries: list[dict]) -> str:
    """Fixed prompt_addendum template (§15.5): one line per decided point."""
    lines = [ADDENDUM_HEAD]
    for e in decided_entries:
        lines.append(f"- {e['point_id']} = {e['choice']}")
    return "\n".join(lines)


def _conflict_hits(t: dict) -> list[str]:
    """Point ids named by DECISION CONFLICT markers in model-produced text.

    The input is NOT scanned: it holds the addendum template itself
    ("DECISION CONFLICT: <point_id>"), which would false-hit every
    injected run. Trailing punctuation is stripped ("refund.action.")."""
    import re
    texts = [e.get("content", "") for e in t.get("events", [])]
    texts.append(t.get("final_output", ""))
    hits = []
    for text in texts:
        for m in re.finditer(CONFLICT_RE, text or ""):
            hit = m.group(1).rstrip(".,;:!?)]}\"'")
            if hit and hit not in hits:
                hits.append(hit)
    return hits


def _fmt_num(x) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _after_marker(r: dict, mseq: int | None, legacy: tuple | None) -> bool:
    """Row strictly after a fit/export marker (H2: seq; legacy tuple fallback)."""
    if legacy is not None:
        return (r.get("ts") or "", r.get("decision_id") or "") > legacy
    assert mseq is not None
    return _row_seq(r) > mseq


def _at_or_before_marker(r: dict, mseq: int | None,
                         legacy: tuple | None) -> bool:
    """Row at/before a fit marker (H2: seq; legacy tuple fallback)."""
    if legacy is not None:
        return (r.get("ts") or "", r.get("decision_id") or "") <= legacy
    assert mseq is not None
    return _row_seq(r) <= mseq


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def _norm_remote_ok(value) -> list[str]:
    """remote_ok normal form: v1.6 boolean true reads as both hosts (v1.7)."""
    if value is True:
        return ["fireworks", "hosted_jev"]
    if not value:
        return []
    return sorted(set(value))


def _mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar p over b+c discordant pairs (no scipy)."""
    import math
    n = b + c
    if n == 0:
        return 1.0
    p = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * p)


def _still_needed(errors: int, n: int, target: float,
                  cap: int = 5000) -> int | None:
    from . import stats as S
    if S.wilson_upper(errors, n) <= target:
        return 0
    for x in range(1, cap + 1):
        if S.wilson_upper(errors, n + x) <= target:
            return x
    return None


def snippet_for(point: dict) -> str:
    fields = ", ".join(point.get("context_fields", {}).keys())
    pid = point["point_id"]
    return (
        f"### Decision: {pid}\n"
        f"Call `do_decide` with point_id `{pid}` and context\n"
        f"{{{fields}}}.\n"
        '- mode "auto" or "rule": use `choice`. Do not re-decide.\n'
        '- mode "teacher": decide yourself among `allowed_options`, then call\n'
        "  `do_record(decision_id, choice)`.\n"
        "When the result is known (ticket closed, customer reply), call "
        "`do_outcome(decision_id, ok)`.\n"
    )


class Service:
    def __init__(self, cfg: dict, store, deciders: dict,
                 rng: random.Random | None = None, trainer=None,
                 fireworks_key: str | None = None, monotonic=None,
                 config_path=None):
        self.cfg = cfg
        self.store = store
        self.rng = rng or random.Random()
        # M16 autopilot: trainer is None in production until M11 (live
        # FireworksTrainer); tests pass a FakeTrainer. fireworks_key is the
        # resolved key (memory only). monotonic paces the per-point check
        # limiter (tests inject a fake).
        self.trainer = trainer
        self.fireworks_key = fireworks_key
        self._config_path = config_path
        self._monotonic = monotonic or time.monotonic
        self._ap_last_check: dict[str, float] = {}
        from .autopilot import Autopilot  # lazy: autopilot reads service._utcnow

        self.autopilot = Autopilot(self)
        if not isinstance(deciders, dict):
            raise TypeError(
                "Service needs a deciders map from build_deciders; "
                "a bare backend was removed in M12 P2")
        self.deciders: dict[str, Decider] = deciders

    def _decider_for(self, point: dict) -> Decider | None:
        """The decider for a point, or None when no model is called.

        Collect/paused call no model (H2); a private point without
        "hosted_jev" in remote_ok calls nothing, and a private-tier or
        private-only point never sees hosted Jev (H3).
        """
        if point.get("phase", "collect") in ("collect", "paused"):
            return None
        tier = point.get("tier") or ("private" if point.get(
            "private_model_only") else "hosted_jev")
        if tier == "private" or point.get("private_model_only"):
            # Never falls back to hosted Jev; None until the private tier
            # exists (H5 fakes, M11 Fireworks).
            return self.deciders.get("private")
        if point.get("data_class", "private") == "private":
            ro = point.get("remote_ok", [])
            allowed = True if ro is True else "hosted_jev" in (ro or [])
            if not allowed:
                return None
        return self.deciders.get("hosted_jev")

    def _decide_request(self, point: dict, context: dict,
                        task_input: str | None = None,
                        state_text: str | None = None) -> DecideRequest:
        helps = point.get("option_help") or {}
        return DecideRequest(
            point_id=point["point_id"],
            state=R.render_state_obj(point, context, task_input,
                                     self.cfg["gateway"]["max_state_chars"]),
            instructions=point["question"],
            criteria={o: helps.get(o) or o for o in point["options"]},
            state_text=state_text)

    # -- do_register -----------------------------------------------------
    def do_register(self, args: dict) -> dict:
        if not isinstance(args, dict):
            return err("invalid_args", "do_register takes the decision point object")
        if args.get("type") == "workflow":
            return self._register_workflow(args)
        errors = SV.validate("point", args)
        if errors:
            return err("invalid_point", "; ".join(errors[:5]))
        point = self._normalize_point(args)
        serr = self._check_point(point)
        if serr:
            return serr
        if point["private_model_only"] and "hosted_jev" in point["remote_ok"]:
            return err("privacy_conflict",
                       "private_model_only with 'hosted_jev' in remote_ok")
        stored = self.store.get_point(point["point_id"])
        if stored is None:
            # D: a new private point that may call a remote host from birth
            # needs the user's confirm, exactly like widening later does.
            if point["data_class"] == "private" and point["remote_ok"]:
                if not args.get("confirm", False):
                    return err("confirm_required",
                               "a new private point with remote_ok needs "
                               "confirm: true")
        elif stored.get("data_class", "private") == "private":
            # D: making a private point public exposes its future calls;
            # narrowing (public -> private) is the safe direction.
            if point["data_class"] != "private":
                if not args.get("confirm", False):
                    return err("confirm_required",
                               "making a private point public needs "
                               "confirm: true")
            elif (set(point["remote_ok"]) != set(_norm_remote_ok(
                        stored.get("remote_ok", [])))
                    or point["private_model_only"] != bool(
                        stored.get("private_model_only", False))):
                if not args.get("confirm", False):
                    return err("confirm_required",
                               "changing remote_ok/private_model_only on a "
                               "private point needs confirm: true")
        if stored is not None:
            # E2: clearing suspected/irreversible drops autopilot's blocks
            # (suspected/irreversible), so it needs the user's confirm.
            for _flag in ("suspected", "irreversible"):
                if stored.get(_flag, False) and not point.get(_flag, False):
                    if not args.get("confirm", False):
                        return err("confirm_required",
                                   f"clearing {_flag} needs confirm: true")
        if stored is None:
            point["point_version"] = 1
            point["phase"] = "collect"
            point["plan_phase"] = "collect"
            point["tier"] = ("private" if point["private_model_only"]
                             else "hosted_jev")
            point["fit_epoch_start"] = _utcnow()
            point["plan_fit_epoch_start"] = point["fit_epoch_start"]
        else:
            point["point_version"] = stored.get("point_version", 1)
            point["phase"] = stored.get("phase", "collect")
            point["plan_phase"] = stored.get("plan_phase", "collect")
            point["tier"] = stored.get("tier") or (
                "private" if point["private_model_only"] else "hosted_jev")
            # tier follows private_model_only (server-managed): a point that
            # becomes private-only leaves the hosted tier, and one that stops
            # being private-only returns to it.
            if point["private_model_only"]:
                point["tier"] = "private"
            elif (stored.get("private_model_only")
                    and point["tier"] == "private"):
                point["tier"] = "hosted_jev"
            point["fit_epoch_start"] = stored.get("fit_epoch_start", "")
            # E1: a plain re-register must not wipe server-managed autopilot
            # state (stop, cooldown, export markers, events, extraction flag).
            for _k in ("autopilot_stopped", "autopilot_demotion_ts",
                       "last_train_export_ts", "last_train_export_id",
                       "last_train_export_seq", "extraction_unreliable"):
                if _k in stored:
                    point[_k] = stored[_k]
            if "events" in stored:
                point["events"] = [dict(e) for e in stored["events"]]
            # R2-1: plan has its own epoch; old points backfill from decide.
            point["plan_fit_epoch_start"] = stored.get(
                "plan_fit_epoch_start", stored.get("fit_epoch_start", ""))
            # E2: any change to suspected/irreversible starts the cooldown,
            # so autopilot never serves a point whose flags just changed.
            if (bool(stored.get("suspected", False)) != bool(point.get("suspected", False))
                    or bool(stored.get("irreversible", False)) != bool(point.get("irreversible", False))):
                self._note_demotion_cooldown(point)
            if self._versioned_differs(stored, point):
                point["point_version"] += 1
                # A version bump resets plan evidence (§15.4): plan_phase back
                # to collect; old shadow comparisons are filtered by version.
                point["plan_phase"] = "collect"
                point["fit_epoch_start"] = _utcnow()
                point["plan_fit_epoch_start"] = point["fit_epoch_start"]
                self.store.deactivate_hosted_fit(point["point_id"])
                _prev_phase = point["phase"]
                if point["source"] == "gateway":
                    # R2-4: a bump never moves collect/paused (it would start
                    # paid calls and undo the user's pause); the point stays
                    # put until do_promote admits it. serve -> shadow is the
                    # demotion; shadow stays.
                    if point["phase"] == "serve":
                        point["phase"] = "shadow"
                elif (point["phase"] == "serve"
                        and self._hosted_tier(point)):
                    # A6: a bumped point has no fit; without this the point
                    # serves uncalibrated and the fit never refires (fit rows
                    # are shadow rows). Gateway points already go back.
                    point["phase"] = "shadow"
                # E4: a version-bump demotion (serve -> shadow) starts the
                # cooldown like any other demotion.
                if _prev_phase == "serve" and point["phase"] == "shadow":
                    self._note_demotion_cooldown(point)
        self.store.put_point(point)
        if stored is not None and point["point_version"] != stored.get(
                "point_version", 1):
            # A bumped point resets trial + monitor evidence of workflows
            # that depend on it (§15.8, B11).
            for wrec in self.store.list_workflows():
                if point["point_id"] in wrec["downgrade_requires"]:
                    self._reset_workflow_evidence(
                        wrec["workflow"],
                        f"point {point['point_id']} bumped to "
                        f"v{point['point_version']}")
        return ok(point=point, snippet=snippet_for(point))

    @staticmethod
    def _normalize_point(args: dict) -> dict:
        point = {
            "point_id": args["point_id"],
            "workflow": args["workflow"],
            "question": args["question"],
            "kind": args.get("kind", "choice"),
            "options": list(args["options"]),
            "option_help": dict(args.get("option_help") or {}),
            "context_fields": dict(args.get("context_fields") or {}),
            "required_fields": list(args.get("required_fields") or []),
            "rules": [dict(r) for r in args.get("rules", [])],
            "irreversible": bool(args.get("irreversible", False)),
            "suspected": bool(args.get("suspected", False)),
            "repeatable": bool(args.get("repeatable", False)),
            "data_class": args.get("data_class", "private"),
            "target_error": args.get("target_error"),
            "source": args.get("source", "do_decide"),
            "remote_ok": _norm_remote_ok(args.get("remote_ok", [])),
            "private_model_only": bool(args.get("private_model_only", False)),
        }
        if args.get("policy") is not None:
            point["policy"] = args["policy"]
        if point["source"] == "gateway":
            for k in ("route", "model", "template", "output_format", "extract",
                      "guard_fields", "multi_turn"):
                if k in args:
                    point[k] = args[k]
            point["multi_turn"] = bool(point.get("multi_turn", False))
        return point

    def _check_point(self, p: dict) -> dict | None:
        if p["kind"] == "choice":
            if len(p["options"]) > 16:
                return err("too_many_options",
                           f"choice points hold at most 16 options, got {len(p['options'])}; "
                           "split the point")
            if len(p["options"]) < 2:
                return err("invalid_point", "choice points need at least 2 options")
        else:  # noul: exactly 2 options, mapped to false/true by position
            if len(p["options"]) != 2:
                return err("invalid_point",
                           f"noul points need exactly 2 options, got {len(p['options'])}")
        if len(set(p["options"])) != len(p["options"]):
            return err("invalid_point", "options must be unique")
        for name in p["option_help"]:
            if name not in p["options"]:
                return err("invalid_point", f"option_help key {name!r} is not an option")
        fields = p["context_fields"]
        # B6: render_state_obj writes the real policy and task input under
        # "policy"/"input" first, so registered fields with those names
        # would silently overwrite them in the hosted state.
        for name in ("policy", "input"):
            if name in fields:
                return err("invalid_point",
                           f"context field {name!r} is reserved (policy/input) "
                           "and cannot be registered")
        # gateway points draw fields from extract as well as context_fields
        universe = set(fields) | set((p.get("extract") or {}))
        for name in p["required_fields"]:
            if name not in universe:
                return err("invalid_point", f"required field {name!r} not in context_fields/extract")
        for fspec in fields.values():
            if fspec.get("type") == "enum" and not fspec.get("enum"):
                return err("invalid_point", "enum fields need a non-empty enum list")
        for i, rule in enumerate(p["rules"]):
            rid = rule.get("id", f"rule_{i}")
            has_only, has_excl = "only" in rule, "exclude" in rule
            if has_only == has_excl:
                return err("invalid_point", f"rule {rid}: exactly one of only/exclude required")
            targets = rule["only"] if has_only else rule["exclude"]
            if not targets:
                return err("invalid_point", f"rule {rid}: only/exclude must be non-empty")
            for o in targets:
                if o not in p["options"]:
                    return err("invalid_point", f"rule {rid}: {o!r} is not an option")
            for cond in rule["if"]:
                if cond[0] not in universe:
                    return err("invalid_point",
                               f"rule {rid}: field {cond[0]!r} not in context_fields/extract")
        if p["source"] == "gateway":
            gerr = self._check_gateway_point(p)
            if gerr:
                return gerr
        if p["target_error"] is None:
            p["target_error"] = self.cfg["serve"]["target_error"]
        return None

    def _check_gateway_point(self, p: dict) -> dict | None:
        from . import gwformat as GF
        from . import gwtemplate as GT
        if p.get("route") not in ("openai", "anthropic"):
            return err("invalid_point", "gateway points need route openai|anthropic")
        if not p.get("model") or not isinstance(p.get("model"), str):
            return err("invalid_point", "gateway points need a model string")
        template = p.get("template") or {}
        if not all(isinstance(template.get(k), str)
                   for k in ("system", "user_prefix", "user_suffix")):
            return err("invalid_point",
                       "gateway points need template {system, user_prefix, user_suffix}")
        fixed = len(GT.norm_system(template["system"])) + len(
            template["user_prefix"]) + len(template["user_suffix"])
        if fixed < self.cfg["gateway"]["min_template_chars"]:
            return err("template_too_weak",
                       f"fixed text {fixed} chars below min_template_chars")
        for field, spec in (p.get("extract") or {}).items():
            if not isinstance(spec, dict) or not isinstance(spec.get("regex"), str):
                return err("invalid_point", f"extract {field}: need {{regex, group?, cast?}}")
            if len(spec["regex"]) > 200:
                return err("invalid_point", f"extract {field}: pattern over 200 chars")
            if spec.get("cast", "string") not in ("string", "number", "integer", "boolean"):
                return err("invalid_point", f"extract {field}: bad cast")
            if not isinstance(spec.get("group", 0), int) or spec.get("group", 0) < 0:
                return err("invalid_point", f"extract {field}: bad group")
        if "guard_fields" in p and p["guard_fields"] is not None and not isinstance(
                p["guard_fields"], list):
            return err("invalid_point", "guard_fields must be a list")
        of = p.get("output_format") or {}
        if of.get("type") not in ("enum_text", "json_enum"):
            return err("invalid_point", "gateway points need output_format enum_text|json_enum")
        dk = GT.discovery_key(p["route"], p["model"], template["system"])
        p["discovery_key"] = dk
        p["fingerprint"] = GT.fingerprint(p["route"], p["model"], template["system"],
                                          template["user_prefix"], template["user_suffix"])
        for other in self.store.list_points():
            if other.get("source") == "gateway" and other["point_id"] != p["point_id"] \
                    and other.get("fingerprint") == p["fingerprint"]:
                return err("fingerprint_in_use",
                           f"fingerprint already registered by {other['point_id']}")
        # §14.10: when ≥ 20 template-matching answers are logged, the format
        # must be learnable from them (server-learned details win). Matching
        # only (not the whole discovery key): one key can hold two templates.
        matching = [t.get("raw_response") or ""
                    for t in self.store.traces_by_discovery(dk)
                    if t.get("raw_request") and t.get("raw_response")
                    and GT.template_matches(
                        template, (t["raw_request"] or {}).get("system", ""),
                        (t["raw_request"] or {}).get("user", ""))]
        if len(matching) >= 20:
            learned = GF.learn_format(matching, list(p["options"]))
            if learned is None or learned["type"] != of["type"]:
                return err("format_unsupported",
                           "output format not learnable from 20+ logged answers")
            p["output_format"] = learned
        return None

    def _register_workflow(self, args: dict) -> dict:
        wf = args.get("workflow")
        if not isinstance(wf, str) or not wf:
            return err("invalid_args", "workflow records need a workflow name")
        if "downgrade_allowed" in args:
            return err("invalid_args", "downgrade_allowed is set only by "
                       "do_promote(workflow, downgrade: true, confirm: true)")
        req = args.get("downgrade_requires")
        if (not isinstance(req, list) or not req
                or not all(isinstance(p, str) and p for p in req)):
            return err("invalid_args", "downgrade_requires must be a non-empty "
                       "list of point ids")
        for pid in req:
            p = self.store.get_point(pid)
            if p is None:
                return err("unknown_point",
                           f"downgrade_requires: unknown point {pid!r}")
            if p.get("workflow") != wf:
                return err("invalid_args",
                           f"downgrade_requires: {pid!r} is in workflow "
                           f"{p.get('workflow')!r}, not {wf!r}")
        check = args.get("check")
        if check == "outcome":
            check_type, check_script = "outcome", ""
        elif isinstance(check, str) and check.startswith("deterministic:"):
            resolved = self._resolve_check_script(check.split(":", 1)[1])
            if resolved is None:
                return err("invalid_args",
                           "check script must be a local file inside the repo "
                           "or workflow folder")
            check_type, check_script = "deterministic", resolved
        else:
            return err("invalid_args", "check must be 'outcome' or "
                       "'deterministic:<script>'; a rubric scored by the "
                       "calling agent is not accepted")
        run_model, cheap_model = args.get("run_model"), args.get("cheap_model")
        if (not isinstance(run_model, str) or not run_model
                or not isinstance(cheap_model, str) or not cheap_model):
            return err("invalid_args", "workflow records need run_model and "
                       "cheap_model strings")
        stored = self.store.get_workflow(wf)
        trial = monitor = None
        allowed, reset = False, True
        script_changed = False
        if stored is not None and check_type == "deterministic":
            recorded = ((stored.get("trial") or {}).get("check") or {}).get(
                "hash")
            if recorded is not None and self._script_hash(
                    check_script) != recorded:
                script_changed = True
        if stored is not None and (stored["downgrade_requires"] == req
                                   and stored["check_type"] == check_type
                                   and stored["check_script"] == check_script
                                   and stored["cheap_model"] == cheap_model
                                   and not script_changed):
            trial = stored.get("trial")
            monitor = stored.get("monitor")
            allowed = stored.get("downgrade_allowed", False)
            reset = False
        elif stored is not None and stored.get("trial") is not None:
            changed = []
            if stored["downgrade_requires"] != req:
                changed.append("downgrade_requires")
            if (stored["check_type"] != check_type
                    or stored["check_script"] != check_script
                    or script_changed):
                changed.append("check")
            if stored["cheap_model"] != cheap_model:
                changed.append("cheap_model")
            trial = {"status": "reset",
                     "reason": f"re-registered with changed "
                               f"{', '.join(changed)}",
                     "prev_trial_id": (stored.get("trial") or {}).get(
                         "trial_id"),
                     "reset_ts": _utcnow()}
            monitor = {"enabled_ts": _utcnow(),
                       "reset_reason": "re-registered with changes"}
        self.store.put_workflow({
            "workflow": wf, "downgrade_requires": req,
            "check_type": check_type, "check_script": check_script,
            "run_model": run_model, "cheap_model": cheap_model,
            "downgrade_allowed": allowed, "trial": trial, "monitor": monitor})
        return ok(workflow=wf, downgrade_requires=req, check=check,
                  run_model=run_model, cheap_model=cheap_model,
                  downgrade_allowed=allowed, trial_reset=reset)

    def _resolve_check_script(self, script: str) -> str | None:
        """Absolute path of a check script inside the repo or data root."""
        from pathlib import Path
        try:
            p = Path(script)
            if not p.is_absolute():
                p = Path.cwd() / p
            resolved = p.resolve()
        except OSError:
            return None
        roots = [Path.cwd().resolve()]
        try:
            roots.append(Path(self.store.root).resolve())
        except OSError:
            pass
        if not resolved.is_file():
            return None
        if not any(resolved.is_relative_to(r) for r in roots):
            return None
        return str(resolved)

    @staticmethod
    def _plan_epoch(point: dict) -> str:
        """R2-1: plan epoch (own clock; old points fall back to decide)."""
        return point.get("plan_fit_epoch_start",
                         point.get("fit_epoch_start", ""))

    @staticmethod
    def _versioned_differs(a: dict, b: dict) -> bool:
        # D: question/option_help change what is sent to Jev (instructions
        # + criteria), so they bump the version and reset the fit like the
        # other prompt-shaping fields.
        keys = ["options", "question", "option_help", "context_fields",
                "rules", "policy"]
        if (b.get("source") or "do_decide") == "gateway":
            keys += ["template", "output_format", "extract", "guard_fields",
                     "multi_turn", "fingerprint"]
        for k in keys:
            if json.dumps(a.get(k), sort_keys=True) != json.dumps(b.get(k), sort_keys=True):
                return True
        return False

    # -- do_decide -------------------------------------------------------
    def do_decide(self, args: dict) -> dict:
        errors = SV.validate("do_decide", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        point = self.store.get_point(args["point_id"])
        if point is None:
            return err("unknown_point", f"unknown point_id {args['point_id']!r}")
        context, missing, type_errors = R.validate_context(point, args["context"])
        if missing:
            return err("missing_field",
                       f"missing required field(s): {', '.join(missing)} (never guessed)")
        if type_errors:
            return err("invalid_context",
                       f"wrong type or enum value for field(s): {', '.join(type_errors)}")
        allowed, rule_hits, warnings = RU.apply_rules(point, context)
        if not allowed:
            return err("rules_conflict", "rules left no allowed option")
        options = list(point["options"])
        base_row: dict = {
            "decision_id": uuid.uuid4().hex,
            "point_id": point["point_id"],
            "point_version": point.get("point_version", 1),
            "ts": _utcnow(),
            "context": context,
            "options": options,
            "allowed_options": allowed,
            "rule_hits": rule_hits,
            "mode": "teacher",
            "audit": False,
            "source": "do_decide",
            "teacher_cost": args.get("teacher_cost"),
            "warnings": warnings or None,
            "phase": point.get("phase", "collect"),
            "run_id": args.get("run_id"),
            "decision_pos": args.get("decision_pos"),
            "at_start": bool(args.get("at_start", True)),
        }
        if len(allowed) == 1:
            # Rules decided: no JEV call (they run in code before JEV, §5).
            base_row.update(mode="rule", final_choice=allowed[0])
            self._insert(base_row)
            return ok(decision_id=base_row["decision_id"], mode="rule", choice=allowed[0],
                      allowed_options=allowed, probabilities={allowed[0]: 1.0},
                      confidence=1.0, threshold=None, instruction=AUTO_INSTRUCTION,
                      warnings=warnings)
        state_text = R.render_state(point, context)
        base_row["state_text"] = state_text
        decider = self._decider_for(point)
        if decider is None:
            # tiered collect/paused (or a tier without a decider): no model call.
            self._insert(base_row)
            return ok(decision_id=base_row["decision_id"], mode="teacher", choice=None,
                      allowed_options=allowed, probabilities={}, confidence=None,
                      threshold=self._threshold_of(point), instruction=TEACHER_INSTRUCTION,
                      warnings=warnings)
        try:
            req = self._decide_request(point, context,
                                       state_text=state_text)
            res = decider.decide([req])[0]
            base_row["jev_seconds"] = res.seconds
            if res.decider == "private":
                base_row["probe"] = {
                    "model_version": res.model,
                    "probabilities": res.probs}
                self._maybe_store_pending(point, req, base_row)
            else:
                base_row["hosted"] = {
                    "model": res.model, "probabilities": res.probs,
                    "cost_usd": res.cost_usd, "seconds": res.seconds}
                self._maybe_store_private(point, req, base_row)
            base_row["decider"] = res.decider
            if res.warnings:
                warnings = warnings + list(res.warnings)
                base_row["warnings"] = warnings or None
            shown_full = res.probs
        except DeciderError as e:
            tag = "jev_unavailable"
            if e.code == "remote_not_allowed":
                # blocked before any call: like collect, no warning.
                self._insert(base_row)
                return ok(decision_id=base_row["decision_id"], mode="teacher",
                          choice=None, allowed_options=allowed, probabilities={},
                          confidence=None, threshold=self._threshold_of(point),
                          instruction=TEACHER_INSTRUCTION, warnings=warnings)
            record_decider_error(self.store, decider, point["point_id"], e)
            tag = {"budget_exceeded": "budget_exceeded",
                   "no_key": "no_hosted_jev_key"}.get(e.code, tag)
            base_row["warnings"] = (warnings + [tag]) or None
            self._insert(base_row)
            return ok(decision_id=base_row["decision_id"], mode="teacher", choice=None,
                      allowed_options=allowed, probabilities=None, confidence=None,
                      threshold=self._threshold_of(point), instruction=TEACHER_INSTRUCTION,
                      warnings=base_row["warnings"] or [])
        probe_used = base_row.get("probe") is not None
        if (base_row.get("decider") == "hosted_jev"
                and self._hosted_tier(point)):
            shown = HF.mask_temper(shown_full, allowed,
                                   self._hosted_temperature(point))
        else:
            shown = S.mask_and_renorm(shown_full, allowed)
        confidence = max(shown.values()) if shown else 0.0
        top = S.argmax(shown, allowed)
        mode, choice, audit = self._select_mode(point, confidence, top, probe_used)
        base_row["mode"] = mode
        base_row["audit"] = audit
        if mode == "auto":
            base_row["final_choice"] = choice
        self._insert(base_row)
        if audit or point.get("phase") == "shadow":
            # A1: the teacher must not see the answer it grades. In shadow
            # and in audits the probabilities would anchor the label and
            # inflate the fit and the evidence (main spec §2.3).
            shown_out: dict = {}
            conf_out = None
        else:
            shown_out, conf_out = shown, confidence
        resp = ok(decision_id=base_row["decision_id"], mode=mode, choice=choice,
                  allowed_options=allowed, probabilities=shown_out,
                  confidence=conf_out,
                  threshold=self._threshold_of(point),
                  instruction=AUTO_INSTRUCTION if mode == "auto" else TEACHER_INSTRUCTION,
                  warnings=warnings)
        if audit:
            resp["audit"] = True
        return resp

    def _threshold_of(self, point: dict) -> float | None:
        if not self._hosted_tier(point):
            model = self.store.active_model(point["point_id"], "private")
            if model is None or model.get("point_version") != point.get(
                    "point_version", 1):
                return None
            return model.get("card", {}).get("threshold")
        fit = self._hosted_fit(point)
        return fit.get("threshold") if fit else None

    # -- hosted tier (build §7) --------------------------------------------
    def _hosted_tier(self, point: dict) -> bool:
        if point.get("private_model_only"):
            return False
        return (point.get("tier") or "hosted_jev") == "hosted_jev"

    def _fit_marker_seq(self, fit: dict, kind: str
                        ) -> tuple[int | None, tuple | None]:
        """(seq, legacy) splitting rows at the last fit row (H2).

        New fits carry last_fit_seq. Pre-v14 fits resolve by stored id; when
        that row is gone (trimmed), the legacy (ts, id) tuple is returned
        for the old comparison (deterministic on stored data).
        """
        s = fit.get("last_fit_seq")
        if isinstance(s, int):
            return s, None
        rid = fit.get("last_fit_decision_id") or ""
        row = self.store.get_plan(rid) if kind == "plan" \
            else self.store.get_decision(rid)
        if row is not None and isinstance(row.get("seq"), int):
            return row["seq"], None
        return None, (fit.get("last_fit_ts") or "", rid)

    def _export_marker_seq(self, card: dict
                           ) -> tuple[int | None, tuple | None]:
        """(seq, legacy) for a private-model export marker (H2). Same fallback."""
        s = card.get("export_seq")
        if isinstance(s, int):
            return s, None
        rid = card.get("export_decision_id") or ""
        row = self.store.get_decision(rid)
        if row is not None and isinstance(row.get("seq"), int):
            return row["seq"], None
        return None, (card.get("export_ts", ""), rid)

    def _hosted_fit(self, point: dict, kind: str = "decide") -> dict | None:
        """Active hosted fit for the point's current version, else None."""
        fit = self.store.active_hosted_fit(point["point_id"], kind)
        if fit is None or fit.get("point_version") != point.get(
                "point_version", 1):
            return None
        return fit

    def _private_plan_fit(self, point: dict) -> dict | None:
        """GP7: active private plan fit for the current version, else None."""
        fit = self.store.active_hosted_fit(point["point_id"], "plan",
                                           "private")
        if fit is None or fit.get("point_version") != point.get(
                "point_version", 1):
            return None
        return fit

    def _plan_fit_for(self, point: dict) -> dict | None:
        """The plan fit that tempers and gates this point (GP7: private
        tier reads the private plan fit, never the hosted one)."""
        if self._effective_tier(point) == "private":
            return self._private_plan_fit(point)
        return self._hosted_fit(point, "plan")

    def _hosted_temperature(self, point: dict) -> float:
        fit = self._hosted_fit(point)
        return float(fit["temperature"]) if fit else 1.0

    def _maybe_store_private(self, point: dict, req, base_row: dict) -> None:
        """While a private model is evaluated, its probabilities are stored
        beside hosted Jev's (comparison set C, §6.4 item 8). Best effort: a
        failing private call never breaks the hosted decision."""
        if not self._hosted_tier(point):
            return
        model = self.store.active_model(point["point_id"], "private")
        if model is None or model.get("point_version") != point.get(
                "point_version", 1):
            return
        dec = self.deciders.get("private")
        if dec is None:
            return
        try:
            pres = dec.decide([req])[0]
        except DeciderError:
            return
        base_row["probe"] = {"model_version": pres.model,
                             "probabilities": pres.probs}

    def _pending_private_model(self, point: dict) -> dict | None:
        """F9 (§7.2 step 5): the pending checkpoint, or None when absent,
        active (raced the switch) or version-stale."""
        marker = point.get("pending_private_model") or {}
        mid = marker.get("model_id")
        if not isinstance(mid, int):
            return None
        model = self.store.get_model(mid)
        if model is None or model.get("active"):
            return None
        if model.get("point_version") != point.get("point_version", 1):
            return None
        return model

    def _maybe_store_pending(self, point: dict, req, base_row: dict) -> None:
        """While a retrained checkpoint is pending, its probabilities are
        logged beside the serving one's (switch evidence, §7.2 step 5).
        Best effort, like _maybe_store_private."""
        if point.get("phase") != "serve":
            return
        model = self._pending_private_model(point)
        if model is None:
            return
        dec = self.deciders.get("private")
        if dec is None or not hasattr(dec, "decide_as"):
            return
        card = model.get("card", {})
        if not card.get("model_ref"):
            return
        try:
            pres = dec.decide_as(card["model_ref"],
                                 card.get("temperature", 1.0), [req])[0]
        except DeciderError:
            return
        base_row["pending_probe"] = {"model_version": pres.model,
                                     "probabilities": pres.probs}

    @staticmethod
    def _tier_label(row: dict, point: dict) -> str | None:
        """Independent label for hosted-tier fits/evidence (build §7).

        Teacher, outcome_correct or outcome_ok only: never extracted, never
        synthetic_gt. The label must be a current option. A3: an
        outcome_bad_no_fix row where the teacher followed hosted Jev's top
        choice keeps its (wrong) teacher label so the row counts as a hosted
        error instead of silently dropping out.
        """
        label, source, _ = L.resolve_label(row)
        if source == "outcome_bad_no_fix":
            if L.teacher_followed_hosted_top(row):
                teacher = row.get("teacher_choice")
                return teacher if teacher in point["options"] else None
            return None
        if label is None or source not in ("teacher", "outcome_correct",
                                           "outcome_ok"):
            return None
        return label if label in point["options"] else None

    def _tier_rows(self, point: dict) -> list[tuple[dict, str]]:
        """Labelled hosted-tier rows, current version, in decision order."""
        ver = point.get("point_version", 1)
        out = []
        for r in self.store.decisions_for_point(point["point_id"]):
            if not r.get("hosted"):
                continue
            if r.get("source") not in LIVE_SOURCES:
                continue
            if r.get("point_version") != ver:
                continue
            label = self._tier_label(r, point)
            if label is None:
                continue
            out.append((r, label))
        out.sort(key=lambda t: _row_key(t[0]))
        return out

    def _fit_candidates(self, point: dict) -> list[tuple[dict, str]]:
        """Labelled shadow rows of the current epoch, in decision order."""
        epoch = point.get("fit_epoch_start", "")
        return [(r, lab) for r, lab in self._tier_rows(point)
                if r.get("phase") == "shadow"
                and (r.get("ts") or "") >= epoch]

    def _evidence_rows(self, point: dict, fit: dict,
                       since: tuple | None = None
                       ) -> list[tuple[dict, str]]:
        """Labelled rows after the last fit row: shadow rows plus audited
        serve rows (unaudited auto rows have no independent label). since
        (M16) further keeps rows decided after a demotion only."""
        mseq, legacy = self._fit_marker_seq(fit, "decide")
        return [(r, lab) for r, lab in self._tier_rows(point)
                if (r.get("phase") == "shadow"
                    or (r.get("phase") == "serve" and r.get("audit")))
                and _after_marker(r, mseq, legacy)
                and (since is None or (r.get("ts") or "") >= since[0])]

    def _late_rows(self, point: dict, fit: dict) -> list[tuple[dict, str]]:
        """Labelled rows decided at/before the last fit row that are not fit
        rows: decided before, labelled after (build §7, G2). Current epoch
        only: after a drift + refit, previous-epoch rows would otherwise all
        count as late (D)."""
        mseq, legacy = self._fit_marker_seq(fit, "decide")
        epoch = point.get("fit_epoch_start", "")
        ids = set(fit.get("fit_ids") or [])
        return [(r, lab) for r, lab in self._tier_rows(point)
                if (r.get("phase") == "shadow"
                    or (r.get("phase") == "serve" and r.get("audit")))
                and (r.get("ts") or "") >= epoch
                and _at_or_before_marker(r, mseq, legacy)
                and (r.get("decision_id") or "") not in ids]

    def _shadow_label_coverage(self, point: dict,
                               since: tuple | None = None
                               ) -> tuple[int, int]:
        """(labelled, decided) shadow rows of the current version and epoch.

        A2: unlabelled shadow rows silently drop out of the fit and the
        evidence, so the agent chooses which rows count. Rule-mode rows are
        excluded: rules decide them, no teacher label is needed. since (M16)
        further keeps rows decided after a demotion only.
        """
        ver = point.get("point_version", 1)
        epoch = point.get("fit_epoch_start", "")
        decided = labelled = 0
        for r in self.store.decisions_for_point(point["point_id"]):
            if r.get("phase") != "shadow":
                continue
            if r.get("source") not in LIVE_SOURCES:
                continue
            if r.get("mode") == "rule":
                continue
            if r.get("point_version") != ver:
                continue
            if (r.get("ts") or "") < epoch:
                continue
            if since is not None and (r.get("ts") or "") < since[0]:
                continue
            decided += 1
            if self._tier_label(r, point) is not None:
                labelled += 1
        return labelled, decided

    def _score_hosted(self, point: dict, pairs: list[tuple[dict, str]],
                      T: float) -> list[tuple[float, bool]]:
        """(tempered masked confidence, top-is-label) per row. A3: a followed
        row the outcome marked wrong is an error even though the top choice
        equals the (wrong) teacher label."""
        options = list(point["options"])
        out = []
        for r, label in pairs:
            allowed = r.get("allowed_options") or options
            tm = HF.mask_temper(r["hosted"]["probabilities"], allowed, T)
            top = S.argmax(tm, allowed)
            bad_followed = L.resolve_label(r)[1] == "outcome_bad_no_fix"
            out.append((max(tm.values()) if tm else 0.0,
                        False if bad_followed else top == label))
        return out

    def _maybe_fit_hosted(self, point_id: str,
                          kind: str = "decide") -> dict | None:
        """One-time fit at fit_rows labelled rows (build §7). Returns the fit
        (new or existing) or None."""
        if kind == "plan":
            return self._maybe_fit_plan(point_id)
        point = self.store.get_point(point_id)
        if point is None or not self._hosted_tier(point):
            return None
        fit = self.store.active_hosted_fit(point_id, kind)
        if fit is not None:
            if fit.get("point_version") == point.get("point_version", 1):
                return fit
            self.store.deactivate_hosted_fit(point_id, kind)
        cands = self._fit_candidates(point)
        need = int(self.cfg["hosted_jev"]["fit_rows"])
        if len(cands) < need:
            return None
        chosen = cands[:need]
        options = list(point["options"])
        # R2-2 (A3): bad_followed rows (teacher followed, outcome not OK)
        # have no true label for temperature, but count as errors for the
        # threshold/agreement (mirror _score_hosted).
        items = []
        bad = []
        for r, label in chosen:
            allowed = r.get("allowed_options") or options
            items.append((S.mask_and_renorm(r["hosted"]["probabilities"],
                                            allowed), allowed, label, r))
            bad.append(L.resolve_label(r)[1] == "outcome_bad_no_fix")
        clean = [(m, a, l) for (m, a, l, _r), b in zip(items, bad) if not b]
        n_bad = sum(1 for b in bad if b)
        n_clean_err = sum(1 for m, a, l in clean if S.argmax(m, a) != l)
        if n_bad + n_clean_err < 5 or not clean:
            T, skipped = 1.0, "too_few_errors"
        else:
            T, skipped = HF.fit_temperature(clean)
        scored = []
        tops: list[str] = []
        for (masked, allowed, label, _r), is_bad in zip(items, bad):
            tm = HF.temper_full(masked, T)
            top = S.argmax(tm, allowed)
            tops.append(top)
            scored.append((max(tm.values()) if tm else 0.0,
                          False if is_bad else top == label))
        target = point.get("target_error", self.cfg["serve"]["target_error"])
        threshold = HF.fit_threshold(scored, target,
                                     self.cfg["serve"]["min_auto_support"])
        if threshold is not None:
            at = [c for conf, c in scored if conf >= threshold]
            coverage = len(at) / len(scored)
            err_at = sum(1 for c in at if not c) / len(at)
        else:
            coverage, err_at = 0.0, None
        min_recall = self.cfg["serve"]["min_option_recall"]
        per_option: dict[str, dict] = {}
        teacher_only = []
        for o in options:
            # R2-2: bad_followed rows carry no true label, excluded here.
            idx = [i for i, (_, label) in enumerate(chosen)
                   if label == o and not bad[i]]
            rec = (sum(1 for i in idx if tops[i] == o) / len(idx)
                   if idx else 0.0)
            prec_d = [i for i in range(len(chosen))
                      if tops[i] == o and not bad[i]]
            prec = (sum(1 for i in prec_d if chosen[i][1] == o)
                    / len(prec_d) if prec_d else 0.0)
            per_option[o] = {"recall": rec, "precision": prec, "n": len(idx)}
            if rec < min_recall or len(idx) < 10:
                teacher_only.append(o)
        last = chosen[-1][0]
        fit_row = {
            "point_id": point_id,
            "point_version": point.get("point_version", 1), "kind": kind,
            "epoch_start_ts": point.get("fit_epoch_start", ""),
            "fitted_at": _utcnow(), "n_fit": len(chosen),
            "last_fit_ts": last.get("ts") or "",
            "last_fit_decision_id": last.get("decision_id") or "",
            "last_fit_seq": last.get("seq"),
            "temperature": T, "temperature_skipped": skipped,
            "threshold": threshold, "coverage": coverage,
            "error_at_threshold": err_at,
            "fit_agreement": (sum(1 for _, c in scored if c)
                              / len(scored)) if scored else None,
            "teacher_only_options": teacher_only, "per_option": per_option,
            "fit_ids": [r.get("decision_id") or "" for r, _ in chosen],
        }
        # A4: the re-check and the insert are one BEGIN IMMEDIATE, so two
        # threads crossing the trigger together leave exactly one fit row.
        return self.store.save_hosted_fit_once(fit_row)

    def _hosted_plan_comparisons(self, point: dict) -> list[dict]:
        """Plan comparisons for the hosted plan fit (build §9): shadow plans
        (or plan audits) with a stored model prediction and a label.
        Correctness labels always count; extracted ones only while the
        point passes the §15.3 gate; teacher labels never count. Never
        pilot-forced plans."""
        from . import train as TR
        gate: dict | None = None
        # C6: comparisons of the current plan epoch only (R2-1: own clock,
        # not the decide epoch). Without this a demoted point refits on the
        # pre-demotion comparisons that caused the demotion.
        epoch = Service._plan_epoch(point)
        eval_ids = self.store.eval_only_transcript_ids(point["workflow"])
        out = []
        for plan in self.store.plans_for_workflow(point["workflow"]):
            if plan["ts"] < epoch:
                continue
            if plan.get("pilot_forced"):
                continue
            t = self.store.transcript_for_plan(plan["plan_id"])
            if t is not None and (t.get("transcript_id") or "") in eval_ids:
                continue  # P2 parity: eval-only runs never fit or count
            if plan.get("trial_id") or plan.get("pair_id"):
                continue
            is_audit = bool(plan.get("audit"))
            for c in self._plan_comparisons(plan):
                if c["point_id"] != point["point_id"]:
                    continue
                if c["phase"] != "shadow" and not is_audit:
                    continue
                if not c.get("probabilities"):
                    continue  # rule entries: no model call, no evidence
                if c["kind"] == "teacher":
                    continue  # D: build §9 admits correctness labels and
                    # (gated) extracted choices only; a live teacher label
                    # on the linked row is not plan evidence.
                if c["kind"] == "extracted":
                    if gate is None:
                        # D: the gate scans every transcript + extraction,
                        # so it runs only when an extracted comparison is
                        # actually in play, not on every outcome.
                        gate = TR.extraction_gate(self.cfg, self.store, point)
                    if not gate["passed"]:
                        continue
                out.append({"plan_id": plan["plan_id"], "ts": plan["ts"],
                            "seq": plan.get("seq"), **c})
        out.sort(key=lambda c: (_row_seq(c), c["ts"], c["plan_id"]))
        return out

    def _plan_evidence_hosted(self, point: dict, fit: dict) -> list[dict]:
        """Hosted plan evidence: comparisons after the last fit comparison."""
        mseq, legacy = self._fit_marker_seq(fit, "plan")
        out = []
        for c in self._hosted_plan_comparisons(point):
            if legacy is not None:
                after = (c["ts"], c["plan_id"]) > legacy
            else:
                assert mseq is not None
                after = _row_seq(c) > mseq
            if after:
                out.append(c)
        return out

    def _score_plan_hosted(self, point: dict, comps: list[dict],
                           T: float) -> list[tuple[float, bool]]:
        """(tempered masked confidence, top-is-label) per plan comparison."""
        options = list(point["options"])
        out = []
        for c in comps:
            allowed = c.get("allowed_options") or options
            tm = HF.mask_temper(c["probabilities"], allowed, T)
            top = S.argmax(tm, allowed)
            out.append((max(tm.values()) if tm else 0.0, top == c["label"]))
        return out

    def _maybe_fit_plan(self, point_id: str) -> dict | None:
        """One-time plan fit at fit_rows plan comparisons (build §9)."""
        point = self.store.get_point(point_id)
        if point is None or not self._hosted_tier(point):
            return None
        fit = self.store.active_hosted_fit(point_id, "plan")
        if fit is not None:
            if fit.get("point_version") == point.get("point_version", 1):
                return fit
            self.store.deactivate_hosted_fit(point_id, "plan")
        return self._fit_plan_core(
            point, self._hosted_plan_comparisons(point), "hosted_jev")

    def _private_plan_comparisons(self, point: dict) -> list[dict]:
        """GP7: plan comparisons predicted by the private tier (at-start
        contexts, like all plan predictions)."""
        return [c for c in self._hosted_plan_comparisons(point)
                if c.get("decider") == "private"]

    def _maybe_fit_private_plan(self, point_id: str) -> dict | None:
        """GP7: one-time private plan fit, the same procedure as the
        hosted plan fit, over private-tier comparisons."""
        point = self.store.get_point(point_id)
        if point is None or self._effective_tier(point) != "private":
            return None
        fit = self.store.active_hosted_fit(point_id, "plan", "private")
        if fit is not None:
            if fit.get("point_version") == point.get("point_version", 1):
                return fit
            self.store.deactivate_hosted_fit(point_id, "plan", "private")
        return self._fit_plan_core(
            point, self._private_plan_comparisons(point), "private")

    def _fit_plan_core(self, point: dict, comps: list[dict],
                       tier: str) -> dict | None:
        """Shared plan-fit procedure (hosted + GP7 private): T, §6.1
        threshold, per-option recall/precision, one stored fit row."""
        need = int(self.cfg["hosted_jev"]["fit_rows"])
        if len(comps) < need:
            return None
        chosen = comps[:need]
        options = list(point["options"])
        items = []
        for c in chosen:
            allowed = c.get("allowed_options") or options
            items.append((S.mask_and_renorm(c["probabilities"], allowed),
                          allowed, c["label"]))
        T, skipped = HF.fit_temperature(items)
        scored, tops = [], []
        for masked, allowed, label in items:
            tm = HF.temper_full(masked, T)
            top = S.argmax(tm, allowed)
            tops.append(top)
            scored.append((max(tm.values()) if tm else 0.0, top == label))
        threshold = HF.fit_threshold(scored, self.cfg["plan"]["target_error"],
                                     self.cfg["serve"]["min_auto_support"])
        if threshold is not None:
            at = [c for conf, c in scored if conf >= threshold]
            coverage = len(at) / len(scored)
            err_at = sum(1 for c in at if not c) / len(at)
        else:
            coverage, err_at = 0.0, None
        min_recall = self.cfg["serve"]["min_option_recall"]
        per_option: dict[str, dict] = {}
        teacher_only = []
        for o in options:
            idx = [i for i, c in enumerate(chosen) if c["label"] == o]
            rec = (sum(1 for i in idx if tops[i] == o) / len(idx)
                   if idx else 0.0)
            prec_d = [i for i in range(len(chosen)) if tops[i] == o]
            prec = (sum(1 for i in prec_d if chosen[i]["label"] == o)
                    / len(prec_d) if prec_d else 0.0)
            per_option[o] = {"recall": rec, "precision": prec, "n": len(idx)}
            if rec < min_recall or len(idx) < 10:
                teacher_only.append(o)
        last = chosen[-1]
        # D: the fit records which label kinds it was fitted on; the report
        # reads this instead of recomputing (comparison labels drift as
        # outcomes land later, and recomputing re-runs the gate scan).
        fit_kinds: dict[str, int] = {}
        for c in chosen:
            fit_kinds[c["kind"]] = fit_kinds.get(c["kind"], 0) + 1
        # A4: same race as the decide fit; the re-check and the insert are
        # one BEGIN IMMEDIATE.
        return self.store.save_hosted_fit_once({
            "point_id": point.get("point_id"),
            "point_version": point.get("point_version", 1), "kind": "plan",
            "tier": tier,
            "epoch_start_ts": Service._plan_epoch(point),
            "fitted_at": _utcnow(), "n_fit": len(chosen),
            "last_fit_ts": last["ts"], "last_fit_decision_id": last["plan_id"],
            "last_fit_seq": last.get("seq"),
            "temperature": T, "temperature_skipped": skipped,
            "threshold": threshold, "coverage": coverage,
            "error_at_threshold": err_at,
            "fit_agreement": (sum(1 for _, c in scored if c)
                              / len(scored)) if scored else None,
            "teacher_only_options": teacher_only, "per_option": per_option,
            "fit_ids": [c["plan_id"] for c in chosen],
            "fit_label_kinds": fit_kinds,
        })

    # -- private comparison + train reasons (H5) ---------------------------
    def _effective_tier(self, point: dict) -> str:
        if point.get("private_model_only"):
            return "private"
        return point.get("tier") or "hosted_jev"

    def _comparison_rows(self, point: dict, card: dict,
                           since: tuple | None = None
                           ) -> list[tuple[dict, str]]:
        """Comparison set C (§6.4 item 8): tier-labelled rows decided after
        the private export, in shadow (or audited serve), with both models'
        probabilities stored, and no hosted fit row. since (M16) further
        keeps rows decided after a demotion only."""
        mseq, legacy = self._export_marker_seq(card)
        fit = self._hosted_fit(point)
        fit_ids = set(fit.get("fit_ids") or []) if fit else set()
        return [(r, lab) for r, lab in self._tier_rows(point)
                if r.get("probe")
                and (r.get("phase") == "shadow"
                     or (r.get("phase") == "serve" and r.get("audit")))
                and _after_marker(r, mseq, legacy)
                and (since is None or (r.get("ts") or "") >= since[0])
                and (r.get("decision_id") or "") not in fit_ids
                # A3: on a followed row the outcome marked wrong the true
                # label is unknown, so neither model can be scored on it.
                and L.resolve_label(r)[1] != "outcome_bad_no_fix"]

    def _pending_rows(self, point: dict, card: dict,
                      ) -> list[tuple[dict, str]]:
        """F9 (§7.2 step 5): labelled rows with this pending checkpoint's
        probabilities stored, in shadow (or audited serve), decided after
        its export. Mirrors _comparison_rows (same unbiased rule)."""
        mseq, legacy = self._export_marker_seq(card)
        ver = point.get("point_version", 1)
        ref = card.get("model_ref")
        out = []
        for r in self.store.decisions_for_point(point["point_id"]):
            pp = r.get("pending_probe") or {}
            if not pp.get("probabilities") or pp.get("model_version") != ref:
                continue
            if r.get("source") not in LIVE_SOURCES:
                continue
            if r.get("point_version") != ver:
                continue
            if not (r.get("phase") == "shadow"
                    or (r.get("phase") == "serve" and r.get("audit"))):
                continue
            if not _after_marker(r, mseq, legacy):
                continue
            if L.resolve_label(r)[1] == "outcome_bad_no_fix":
                continue
            label = self._tier_label(r, point)
            if label is None:
                continue
            out.append((r, label))
        out.sort(key=lambda t: _row_key(t[0]))
        return out

    def _pending_switch_checks(self, point: dict, pending: dict,
                               confirm: bool) -> dict:
        """F9 (§7.2 step 5): §6.4 for the pending checkpoint. Items 1-7
        over its held-out card metrics plus pending rows; item 8 adapts
        to old-vs-new (no hosted arm serves private-tier traffic): the
        paired McNemar must not favor the old checkpoint, the new pool
        error must meet the target, and coverage must not collapse."""
        from . import stats as S
        tc, sc = self.cfg["train"], self.cfg["serve"]
        target = point.get("target_error", sc["target_error"])
        card = pending.get("card", {})
        metrics = card.get("metrics", {})
        tp = card.get("threshold")
        options = list(point["options"])
        n_h = int(metrics.get("n_holdout") or 0)
        cov_h = metrics.get("coverage")
        err_h = metrics.get("error_at_threshold")
        n_at_h = int(round((cov_h or 0.0) * n_h)) if n_h else 0
        e_at_h = int(round((err_h or 0.0) * n_at_h)) if n_at_h else 0
        rows = self._pending_rows(point, card)
        scored, olds = [], []
        for r, label in rows:
            allowed = r.get("allowed_options") or options
            pm = S.mask_and_renorm(
                (r.get("pending_probe") or {}).get("probabilities") or {},
                allowed)
            om = S.mask_and_renorm(
                (r.get("probe") or {}).get("probabilities") or {}, allowed)
            scored.append((max(pm.values()) if pm else 0.0,
                           S.argmax(pm, allowed) == label if pm else False))
            olds.append((max(om.values()) if om else 0.0,
                         S.argmax(om, allowed) == label if om else False))
        checks: dict[str, dict] = {}
        n_labelled = len(L.labeled_rows(
            self.store.decisions_for_point(point["point_id"]), point,
            self.cfg["train"]["teacher_only_weight"],
            self.cfg["extract"]["weight"]))
        checks["min_examples"] = {
            "pass": n_labelled >= tc["min_examples"],
            "detail": f"{n_labelled} labelled rows, need "
                      f"{tc['min_examples']}"}
        checks["threshold"] = {
            "pass": tp is not None,
            "detail": f"pending threshold={tp}"}
        coverage = metrics.get("coverage")
        checks["coverage"] = {
            "pass": tp is not None and coverage is not None
            and coverage >= sc["min_coverage"],
            "detail": f"held-out coverage={coverage}, need "
                      f"{sc['min_coverage']}"}
        pool = [(c, ok) for c, ok in scored if tp is not None and c >= tp]
        n_pool = n_at_h + len(pool)
        e_pool = e_at_h + sum(1 for _, ok in pool if not ok)
        upper = S.wilson_upper(e_pool, n_pool) if n_pool else 1.0
        checks["evidence"] = {
            "pass": n_pool > 0 and upper <= target,
            "detail": f"pool n={n_pool} (held-out {n_at_h} + pending "
                      f"{len(pool)}), errors={e_pool}, Wilson upper="
                      f"{upper:.4f}, target={target}"}
        pool_err = (e_pool / n_pool) if n_pool else 1.0
        acc_h = metrics.get("accuracy")
        agree_p = (sum(1 for _, ok in scored if ok) / len(scored)
                   if scored else None)
        if acc_h is not None and n_h and agree_p is not None and scored:
            baseline = 1.0 - (acc_h * n_h + agree_p * len(scored)) / (
                n_h + len(scored))
        elif acc_h is not None:
            baseline = 1.0 - acc_h
        else:
            baseline = 1.0
        checks["margin"] = self._margin_check(
            n_pool, pool_err, baseline, target, sc["min_margin_vs_random"],
            "combined baseline")
        checks["shadow"] = {
            "pass": len(rows) >= sc["min_shadow"],
            "detail": f"{len(rows)} pending rows, need {sc['min_shadow']}"}
        checks["shadow_agreement"] = {
            "pass": agree_p is not None and acc_h is not None
            and len(rows) >= sc["min_shadow"]
            and abs(agree_p - acc_h) <= 0.05,
            "detail": f"pending agreement {agree_p} vs held-out {acc_h} "
                      "(need within 0.05)"}
        both, disagree = L.teacher_disagreement(
            self.store.decisions_for_point(point["point_id"]))
        unreliable = both > 0 and disagree / both > tc["teacher_alarm"]
        checks["teacher"] = {
            "pass": not unreliable,
            "detail": f"outcome/teacher disagreements {disagree}/{both} "
                      f"(alarm above {tc['teacher_alarm']})"}
        checks["confirm"] = {
            "pass": bool(confirm),
            "detail": ("the user agreed" if confirm
                       else "confirm: true required")}
        if point.get("source") == "gateway":
            from . import gateway as GW
            fe = GW.format_evidence(self.cfg, self.store, point)
            gcfg = self.cfg["gateway"]
            n_ok = fe["n_compared"] >= gcfg["format_min_n"]
            wl = fe["wilson_lower"]
            w_ok = wl is not None and wl >= gcfg["format_match_min"]
            ur = fe["unparsed_rate"]
            u_ok = ur is not None and ur <= gcfg["max_unparsed"]
            checks["gateway_format"] = {
                "pass": n_ok and w_ok and u_ok,
                "detail": (f"compared={fe['n_compared']}/"
                           f"{gcfg['format_min_n']}, unparsed "
                           f"{fe['n_unparsed']}/{fe['n_answered']}")}
        old = self.store.active_model(point["point_id"], "private")
        if old is None:
            checks["private_item8"] = {
                "pass": True,
                "detail": "no serving checkpoint to regress from"}
            return checks
        b = sum(1 for (_, oo), (_, ok) in zip(olds, scored) if oo and not ok)
        c = sum(1 for (_, oo), (_, ok) in zip(olds, scored) if ok and not oo)
        cov_o = (sum(1 for co, _ in olds if tp is not None and co >= tp)
                 / len(olds)) if olds else 0.0
        cov_n = (len(pool) / len(scored)) if scored else 0.0
        gain = self.cfg["hosted_jev"]["min_private_gain"]
        mcnemar = b <= c
        err_ok = (e_pool / n_pool) <= target if n_pool else False
        cov_ok = cov_n >= cov_o - gain
        checks["private_item8"] = {
            "pass": bool(scored and mcnemar and err_ok and cov_ok),
            "detail": (f"pending n={len(scored)}, new err="
                       f"{(e_pool / n_pool) if n_pool else None}, "
                       f"new cov={cov_n:.3f} vs old cov={cov_o:.3f}, "
                       f"discordant old-only/new-only {b}/{c}, McNemar p="
                       f"{_mcnemar_p(b, c):.4f}")}
        return checks

    def _private_comparison(self, point: dict,
                            since: tuple | None = None) -> dict | None:
        """Paired private-vs-hosted scoring over C (§6.4 item 8). None when
        no private model is under evaluation."""
        model = self.store.active_model(point["point_id"], "private")
        if model is None or model.get("point_version") != point.get(
                "point_version", 1):
            return None
        card = model.get("card", {})
        rows = self._comparison_rows(point, card, since)
        options = list(point["options"])
        T = self._hosted_temperature(point)
        hfit = self._hosted_fit(point)
        th = (hfit or {}).get("threshold")
        tp = card.get("threshold")
        hs, ps = [], []
        for r, label in rows:
            allowed = r.get("allowed_options") or options
            tm = HF.mask_temper(r["hosted"]["probabilities"], allowed, T)
            pm = S.mask_and_renorm(
                (r.get("probe") or {}).get("probabilities") or {}, allowed)
            hs.append((max(tm.values()) if tm else 0.0,
                       S.argmax(tm, allowed) == label if tm else False))
            ps.append((max(pm.values()) if pm else 0.0,
                       S.argmax(pm, allowed) == label if pm else False))

        def pool_stats(scored, t):
            if t is None:
                return 0.0, None, 0
            at = [ok for c, ok in scored if c >= t]
            cov = len(at) / len(scored) if scored else 0.0
            return cov, (sum(1 for ok in at if not ok) / len(at)
                         if at else None), len(at)

        cov_h, err_h, n_h = pool_stats(hs, th)
        cov_p, err_p, n_p = pool_stats(ps, tp)
        b = sum(1 for (_, oh), (_, op) in zip(hs, ps) if oh and not op)
        c = sum(1 for (_, oh), (_, op) in zip(hs, ps) if op and not oh)
        target = point.get("target_error", self.cfg["serve"]["target_error"])
        cond1 = n_h > 0 and n_p > 0 and (err_h or 0) > target \
            and (err_p if err_p is not None else 1) <= target
        cond2 = n_p > 0 and (err_p if err_p is not None else 1) <= target \
            and cov_p - cov_h >= self.cfg["hosted_jev"]["min_private_gain"]
        enough = len(rows) >= self.cfg["hosted_jev"]["min_compare"]
        # D: without an active hosted fit there is no hosted arm (th is
        # None, coverage 0), so cond2 passes trivially on the private pool
        # alone; item 8 needs both arms.
        passed = (hfit is not None and tp is not None and enough
                  and (cond1 or cond2) and b <= c)
        return {"n": len(rows),
                "min_compare": self.cfg["hosted_jev"]["min_compare"],
                "hosted": {"coverage": cov_h, "error": err_h,
                           "n_at_threshold": n_h, "threshold": th},
                "private": {"coverage": cov_p, "error": err_p,
                            "n_at_threshold": n_p, "threshold": tp},
                "discordant_hosted_only": b, "discordant_private_only": c,
                "mcnemar_p": _mcnemar_p(b, c),
                "scored_private": ps,
                "pass": bool(passed)}

    def _train_reasons(self, point: dict) -> list[str]:
        """The four train_recommended rules of main spec §7.7 (H5). Rules 1-2
        need a mature point (fit fired + min_examples evidence rows)."""
        if not self._hosted_tier(point):
            return []
        reasons = []
        fit = self._hosted_fit(point)
        T = float(fit["temperature"]) if fit else 1.0
        ev = self._evidence_rows(point, fit) if fit else []
        scored = self._score_hosted(point, ev, T)
        t = fit.get("threshold") if fit else None
        mature = fit is not None and \
            len(ev) >= self.cfg["train"]["min_examples"]
        if mature:
            pool = [(c, ok) for c, ok in scored
                    if t is not None and c >= t]
            coverage = len(pool) / len(scored) if scored else 0.0
            if t is None or coverage < self.cfg["serve"]["min_coverage"]:
                reasons.append("low_coverage")
            if not self._serve_checks_hosted(point, True)["margin"]["pass"]:
                reasons.append("margin_not_met")
        window_s = float(
            self.cfg["hosted_jev"]["redemote_window_days"]) * 86400
        now = time.time()
        recent = 0
        for e in point.get("events", []):
            if e.get("event") != "drift_demotion" \
                    or e.get("tier") != "hosted_jev":
                continue
            try:
                from datetime import datetime
                raw = e.get("ts", "")
                ts = datetime.fromisoformat(
                    raw.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError, AttributeError):
                continue
            if now - ts <= window_s:
                recent += 1
        if recent >= 2:
            reasons.append("repeated_drift")
        if fit:
            to = fit.get("teacher_only_options") or []
            if to:
                labs = [lab for _, lab in self._tier_rows(point)]
                if labs and sum(1 for lab in labs if lab in to) / len(labs) \
                        > 0.2:
                    reasons.append("teacher_only_heavy")
        return reasons

    def _hosted_report_block(self, point: dict) -> dict:
        """do_report's hosted-tier block (build §8, H5)."""
        import statistics
        month = time.strftime("%Y-%m", time.gmtime())
        spend = self.store.spend_get(month, "hosted_jev")
        budget = float(self.cfg["hosted_jev"].get("monthly_budget_usd") or 0)
        open_r = self.store.spend_open(month, "hosted_jev")
        fit = self._hosted_fit(point)
        T = float(fit["temperature"]) if fit else 1.0
        t = fit.get("threshold") if fit else None
        ev = self._evidence_rows(point, fit) if fit else []
        scored = self._score_hosted(point, ev, T)
        pool = [(c, ok) for c, ok in scored if t is not None and c >= t]
        e_pool = sum(1 for _, ok in pool if not ok)
        target = point.get("target_error", self.cfg["serve"]["target_error"])
        late = self._late_rows(point, fit) if fit else []
        late_scored = self._score_hosted(point, late, T)
        late_sub = [ok for c, ok in late_scored
                    if t is None or c >= t]
        late_err = (sum(1 for ok in late_sub if not ok) / len(late_sub)
                    if late_sub else None)
        ev_err = (e_pool / len(pool) if pool
                  else ((sum(1 for _, ok in scored if not ok) / len(scored))
                        if scored else None))
        bias = late_err is not None and ev_err is not None \
            and late_err > ev_err + 2 * target
        ver = point.get("point_version", 1)
        lats = sorted(r["hosted"]["seconds"]
                      for r in self.store.decisions_for_point(
                          point["point_id"])
                      if r.get("hosted") and r.get("point_version") == ver
                      and isinstance((r["hosted"] or {}).get("seconds"),
                                     (int, float)))
        cands = self._fit_candidates(point)
        labelled_cov = self._shadow_label_coverage(point)
        reasons = self._train_reasons(point)
        pfit = self._hosted_fit(point, "plan")
        if pfit:
            if pfit.get("fit_label_kinds") is not None:
                # D: the fit recorded its kinds; read them (labels drift as
                # outcomes land, so recomputing would misstate the fit).
                fit_kinds = dict(pfit["fit_label_kinds"])
            else:
                # pre-v11 fit row: fall back to recomputing from live rows.
                pcomps = self._hosted_plan_comparisons(point)
                pfit_ids = set(pfit.get("fit_ids") or [])
                fit_kinds = {}
                for c in pcomps:
                    if c["plan_id"] in pfit_ids:
                        fit_kinds[c["kind"]] = fit_kinds.get(c["kind"], 0) + 1
            pev = self._plan_evidence_hosted(point, pfit)
            ev_kinds: dict[str, int] = {}
            for c in pev:
                ev_kinds[c["kind"]] = ev_kinds.get(c["kind"], 0) + 1
            plan_block: dict = {
                "fitted": True, "threshold": pfit["threshold"],
                "n_fit": pfit["n_fit"], "fit_label_kinds": fit_kinds,
                "evidence_comparisons": len(pev),
                "evidence_label_kinds": ev_kinds}
        else:
            plan_block = {
                "fitted": False,
                "comparisons": len(self._hosted_plan_comparisons(point)),
                "fit_rows": int(self.cfg["hosted_jev"]["fit_rows"])}
        fake = self.cfg.get("decider", {}).get("backend") == "fake"
        return {
            "calls_month": spend["calls"],
            "spend_month_usd": spend["usd"], "budget_usd": budget,
            "budget_warning_80": bool(budget and spend["usd"] >= 0.8 * budget),
            "open_reservations_usd": open_r["usd"],
            "open_reservations_n": open_r["n"],
            "spend_simulated": fake,
            "spend_note": ("simulated spend: the fake backend bills "
                           "nothing" if fake else
                           "settled cost is an estimate from usage; "
                           "compare with the OpenRouter bill"),
            "median_latency_s": (statistics.median(lats) if lats else None),
            "errors": {
                "timeout": self.store.counter_get("hosted_error_timeout",
                                                  point["point_id"]),
                "auth_failed": self.store.counter_get(
                    "hosted_error_auth_failed", point["point_id"]),
                "rate_limited": self.store.counter_get(
                    "hosted_error_rate_limited", point["point_id"]),
                "bad_response": self.store.counter_get(
                    "hosted_error_bad_response", point["point_id"])},
            "fit": ({"temperature": fit["temperature"],
                     "temperature_skipped": fit.get("temperature_skipped"),
                     "threshold": fit["threshold"],
                     "coverage": fit["coverage"],
                     "error_at_threshold": fit.get("error_at_threshold"),
                     "teacher_only_options": fit["teacher_only_options"],
                     "n_fit": fit["n_fit"],
                     "fitted_at": fit["fitted_at"]} if fit else None),
            "fit_progress": {"labelled": len(cands),
                             "fit_rows": int(
                                 self.cfg["hosted_jev"]["fit_rows"])},
            "evidence_rows": len(ev),
            "labelled_ratio": {
                "labelled": labelled_cov[0], "decided": labelled_cov[1],
                "ratio": (labelled_cov[0] / labelled_cov[1]
                          if labelled_cov[1] else None)},
            "still_needed": (_still_needed(e_pool, len(pool), target)
                             if t is not None else None),
            "late_labels_ignored": {"n": len(late), "error_rate": late_err},
            "late_label_bias": bool(bias),
            "train_recommended": bool(reasons),
            "train_reasons": reasons,
            "private_comparison": self._private_comparison(point),
            "plan_fit": plan_block,
        }

    def _draw_audit(self) -> bool:
        return self.rng.random() < float(self.cfg["serve"]["audit_rate"])

    def _select_mode(self, point: dict, confidence: float, top: str, probe_used: bool
                     ) -> tuple[str, str | None, bool]:
        """Mode selection (§6.2). Returns (mode, choice, audit)."""
        if point.get("phase") != "serve":
            return "teacher", None, False
        return self._select_mode_hosted(point, confidence, top)

    def _select_mode_private(self, point: dict, confidence: float,
                               top: str) -> tuple[str, str | None, bool]:
        """Mode selection on the private tier (F9, §6.2): the active card's
        threshold and teacher-only options, then the same audit sample as
        hosted. Without a card for this version, teacher with no audit."""
        model = self.store.active_model(point["point_id"], "private")
        card = {}
        if model is not None and model.get("point_version") == point.get(
                "point_version", 1):
            card = model.get("card", {})
        threshold = card.get("threshold")
        if threshold is None or confidence < threshold:
            return "teacher", None, False
        if top in ((card.get("metrics") or {}).get("teacher_only_options")
                   or []):
            return "teacher", None, False
        if point.get("irreversible") and not self.cfg["serve"]["allow_auto_irreversible"]:
            return "teacher", None, False
        if self._draw_audit():
            return "teacher", None, True
        return "auto", top, False

    def _select_mode_hosted(self, point: dict, confidence: float, top: str
                            ) -> tuple[str, str | None, bool]:
        """Mode selection on the hosted tier: fit threshold, teacher-only
        options and the tempered masked confidence (§6.2 rest unchanged)."""
        if not self._hosted_tier(point):
            return self._select_mode_private(point, confidence, top)
        fit = self._hosted_fit(point)
        threshold = fit.get("threshold") if fit else None
        if threshold is None or confidence < threshold:
            return "teacher", None, False
        if top in ((fit or {}).get("teacher_only_options") or []):
            return "teacher", None, False
        if point.get("irreversible") and not self.cfg["serve"]["allow_auto_irreversible"]:
            return "teacher", None, False
        if self._draw_audit():
            return "teacher", None, True
        return "auto", top, False

    @staticmethod
    def _probe_top(row: dict) -> str | None:
        probs = (row.get("probe") or {}).get("probabilities")
        if not probs:
            return None
        allowed = row.get("allowed_options") or list(probs)
        return S.argmax(S.mask_and_renorm(probs, allowed), allowed)

    def _drift_check(self, point_id: str) -> bool:
        """Demote serve -> shadow on audit evidence (§6.2). Returns True when demoted."""
        from . import stats as S
        point = self.store.get_point(point_id)
        if point is None or point.get("phase") != "serve":
            return False
        if self._hosted_tier(point):
            return self._drift_check_hosted(point)
        # Private tier: drift on the stored private probabilities.
        target = point.get("target_error", self.cfg["serve"]["target_error"])
        window = self.cfg["serve"]["drift_window"]
        audits = [r for r in self.store.decisions_for_point(point_id)
                  if r.get("audit") and r.get("teacher_choice")
                  and r.get("source") in LIVE_SOURCES]
        last = audits[-window:]
        if len(last) < window:
            return False
        n = len(last)
        errs = sum(1 for r in last if self._probe_top(r) != r["teacher_choice"])
        if errs / n > 2 * target and S.wilson_lower(errs, n) > target:
            point["phase"] = "shadow"
            self._note_demotion_cooldown(point)
            point.setdefault("events", []).append(
                {"ts": _utcnow(), "event": "drift_demotion",
                 "errors": errs, "n": n, "target": target})
            self.store.put_point(point)
            self.autopilot.note_demotion(
                point, {"errors": errs, "n": n, "target": target})
            return True
        return False

    def _drift_check_hosted(self, point: dict) -> bool:
        """Drift on the hosted tier: tempered masked hosted top on audits. A
        demotion starts a new epoch and re-arms the fit trigger (build §7)."""
        from . import stats as S
        target = point.get("target_error", self.cfg["serve"]["target_error"])
        window = self.cfg["serve"]["drift_window"]
        options = list(point["options"])
        T = self._hosted_temperature(point)
        # A5: the window covers the current epoch and version only. Without
        # this a re-promoted point is demoted again by pre-demotion audits,
        # and train_recommended gets a false repeated_drift.
        ver = point.get("point_version", 1)
        epoch = point.get("fit_epoch_start", "")
        audits = [r for r in self.store.decisions_for_point(point["point_id"])
                  if r.get("audit") and r.get("teacher_choice")
                  and r.get("source") in LIVE_SOURCES and r.get("hosted")
                  and r.get("point_version") == ver
                  and (r.get("ts") or "") >= epoch]
        last = audits[-window:]
        if len(last) < window:
            return False
        n = len(last)
        errs = 0
        for r in last:
            allowed = r.get("allowed_options") or options
            tm = HF.mask_temper(r["hosted"]["probabilities"], allowed, T)
            if S.argmax(tm, allowed) != r["teacher_choice"]:
                errs += 1
        if errs / n > 2 * target and S.wilson_lower(errs, n) > target:
            point["phase"] = "shadow"
            point["fit_epoch_start"] = _utcnow()
            self._note_demotion_cooldown(point)
            self.store.deactivate_hosted_fit(point["point_id"], "decide")
            point.setdefault("events", []).append(
                {"ts": _utcnow(), "event": "drift_demotion",
                 "tier": "hosted_jev",
                 "errors": errs, "n": n, "target": target})
            self.store.put_point(point)
            self.autopilot.note_demotion(
                point, {"errors": errs, "n": n, "target": target,
                        "tier": "hosted_jev"})
            return True
        return False

    def _insert(self, row: dict) -> None:
        self.store.insert_decision(row, self.cfg["store"]["max_rows_per_point"])

    # -- do_record -------------------------------------------------------
    def do_record(self, args: dict) -> dict:
        errors = SV.validate("do_record", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        row = self.store.get_decision(args["decision_id"])
        if row is None:
            return err("unknown_decision", f"unknown decision_id {args['decision_id']!r}")
        choice = args["choice"]
        if choice not in (row.get("allowed_options") or []):
            self.store.update_decision(row["decision_id"], {"rule_violation": True})
            return err("choice_not_allowed",
                       f"{choice!r} is not in allowed_options {row.get('allowed_options')}")
        override = False
        if row.get("mode") == "auto":
            hosted = (row.get("hosted") or {}).get("probabilities") or {}
            if hosted:
                # D: a hosted-serve auto choice came from the hosted
                # tempered top (tempering never moves the argmax).
                returned = L.hosted_top(row)
            else:
                shown = (row.get("probe") or {}).get("probabilities") or row.get("jev_zero_shot") or {}
                returned = S.argmax(shown, row.get("allowed_options") or [])
            override = choice != returned
        # R2-7: the already_recorded check and the write are one
        # BEGIN IMMEDIATE (MCP server and gateway are separate processes).
        tc = args.get("teacher_cost", row.get("teacher_cost"))
        status, prev = self.store.try_record_teacher(
            row["decision_id"], choice, tc, override)
        if status == "unknown":
            return err("unknown_decision", f"unknown decision_id {args['decision_id']!r}")
        if status == "already":
            return err("already_recorded",
                       f"decision {args['decision_id']!r} already recorded "
                       f"as {prev!r} (first record wins)")
        self._maybe_fit_hosted(row["point_id"])
        demoted = bool(row.get("audit")) and self._drift_check(row["point_id"])
        self.autopilot.kick(row["point_id"])
        return ok(decision_id=row["decision_id"], override=override, demoted=demoted)

    # -- do_extract ------------------------------------------------------
    def do_extract(self, args: dict) -> dict:
        errors = SV.validate("do_extract", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
            time.time() - self.cfg["extract"]["transcript_retention_days"] * 86400))
        self.store.purge_transcripts(cutoff)
        self.store.purge_extracted_text(cutoff)
        if args.get("extraction_id") is not None:
            if args.get("transcript") is not None or args.get("path") is not None:
                return err("invalid_args", "call 2 takes extraction_id, not a transcript")
            return self._extract_call2(args)
        if args.get("pass", 1) == 2:
            return self._extract_call1_pass2(args)
        return self._extract_call1(args)

    def _extract_points(self, workflow: str) -> list[dict]:
        return [p for p in self.store.list_points(workflow)
                if (p.get("source") or "do_decide") in ("do_decide", "dm_decide")]

    @staticmethod
    def _extract_point_brief(p: dict) -> dict:
        return {"point_id": p["point_id"], "options": list(p["options"]),
                "option_help": dict(p.get("option_help") or {}),
                "fields": dict(p.get("context_fields") or {}),
                "required_fields": list(p.get("required_fields") or []),
                "repeatable": bool(p.get("repeatable", False))}

    def _extract_call1(self, args: dict) -> dict:
        from .adapters import validate_transcript
        given = [k for k in ("transcript", "path", "transcript_id")
                 if args.get(k) is not None]
        if len(given) != 1:
            return err("invalid_args", "call 1 needs exactly one of "
                       "transcript/path/transcript_id")
        if args.get("transcript_id") is not None:
            # pass-1 extraction of a stored transcript (e.g. linked by
            # do_plan_result). The privacy gate applies to the stored class.
            stored = self.store.get_transcript(args["transcript_id"])
            if stored is None:
                return err("unknown_transcript",
                           f"unknown transcript_id {args['transcript_id']!r}")
            if stored.get("data_class", "private") == "private" and not (
                    args.get("contributor_off_confirmed")
                    or args.get("extractor_local")):
                return err("private_needs_confirm",
                           "private transcript: confirm the extractor child runs "
                           "with (Muse) Contributor mode off "
                           "(contributor_off_confirmed: true) or extracts "
                           "locally (extractor_local: true)")
            points = self._extract_points(stored["workflow"])
            if not points:
                return err("unknown_workflow",
                           f"no decision points registered for workflow "
                           f"{stored['workflow']!r}")
            extraction_id = f"ex_{uuid.uuid4().hex[:8]}"
            audit_required = self.rng.random() < float(
                self.cfg["extract"]["audit_rate"])
            self.store.put_extraction({
                "extraction_id": extraction_id, "ts": _utcnow(),
                "transcript_id": stored["transcript_id"], "pass": 1,
                "audit_required": audit_required})
            out = ok(transcript_id=stored["transcript_id"],
                     extraction_id=extraction_id,
                     audit_required=audit_required,
                     points=[self._extract_point_brief(p) for p in points],
                     record_schema=RECORD_SCHEMA,
                     instructions=EXTRACT_INSTRUCTIONS)
            out["pass"] = 1
            return out
        if args.get("path") is not None:
            try:
                with open(args["path"]) as f:
                    transcript = json.load(f)
            except OSError:
                return err("transcript_not_found", f"cannot read {args['path']}")
            except ValueError as e:
                return err("invalid_transcript", f"bad JSON in {args['path']}: {e}")
        else:
            transcript = args["transcript"]
        if not isinstance(transcript, dict):
            return err("invalid_transcript", "transcript must be an object")
        verr = validate_transcript(transcript)
        if verr:
            return err("invalid_transcript", "; ".join(verr[:3]))
        points = self._extract_points(transcript["workflow"])
        if not points:
            return err("unknown_workflow",
                       f"no decision points registered for workflow "
                       f"{transcript['workflow']!r}")
        if transcript.get("data_class", "private") == "private" and not (
                args.get("contributor_off_confirmed") or args.get("extractor_local")):
            return err("private_needs_confirm",
                       "private transcript: confirm the extractor child runs with "
                       "(Muse) Contributor mode off "
                       "(contributor_off_confirmed: true) or "
                       "extracts locally (extractor_local: true)")
        if self.store.transcript_by_run(transcript["run_id"]) is not None:
            return err("duplicate_run",
                       f"run_id {transcript['run_id']!r} already stored")
        transcript_id = f"tr_{uuid.uuid4().hex[:8]}"
        self.store.put_transcript({
            "transcript_id": transcript_id, "ts": _utcnow(),
            "run_id": transcript["run_id"], "workflow": transcript["workflow"],
            "model": transcript.get("model", ""),
            "data_class": transcript.get("data_class", "private"),
            "input": transcript.get("input", ""),
            "events": transcript.get("events", []),
            "final_output": transcript.get("final_output", ""),
            "cost": transcript.get("cost"), "run_outcome": transcript.get("run_outcome"),
            "eval_only": bool(args.get("eval_only"))})
        extraction_id = f"ex_{uuid.uuid4().hex[:8]}"
        audit_required = self.rng.random() < float(self.cfg["extract"]["audit_rate"])
        self.store.put_extraction({
            "extraction_id": extraction_id, "ts": _utcnow(),
            "transcript_id": transcript_id, "pass": 1,
            "audit_required": audit_required})
        out = ok(transcript_id=transcript_id, extraction_id=extraction_id,
                 audit_required=audit_required,
                 points=[self._extract_point_brief(p) for p in points],
                 record_schema=RECORD_SCHEMA, instructions=EXTRACT_INSTRUCTIONS)
        out["pass"] = 1
        return out

    def _extract_call1_pass2(self, args: dict) -> dict:
        if args.get("transcript_id") is None:
            return err("invalid_args", "pass 2 call 1 needs {transcript_id, pass: 2}")
        if args.get("transcript") is not None or args.get("path") is not None:
            return err("invalid_args", "pass 2 works on a stored transcript_id")
        stored = self.store.get_transcript(args["transcript_id"])
        if stored is None:
            return err("unknown_transcript",
                       f"unknown transcript_id {args['transcript_id']!r}")
        if stored.get("data_class", "private") == "private" and not (
                args.get("contributor_off_confirmed")
                or args.get("extractor_local")):
            return err("private_needs_confirm",
                       "private transcript: confirm the pass-2 extractor child "
                       "runs with (Muse) Contributor mode off "
                       "(contributor_off_confirmed: true) or extracts "
                       "locally (extractor_local: true)")
        p1 = [e for e in self.store.extractions_for_transcript(
            stored["transcript_id"]) if e["pass"] == 1]
        if not p1 or not any(e.get("audit_required") for e in p1):
            return err("audit_not_required",
                       "pass 2 is only for transcripts drawn for audit "
                       "(pass-1 audit_required)")
        points = self._extract_points(stored["workflow"])
        if not points:
            return err("unknown_workflow",
                       f"no decision points registered for workflow "
                       f"{stored['workflow']!r}")
        extraction_id = f"ex_{uuid.uuid4().hex[:8]}"
        self.store.put_extraction({
            "extraction_id": extraction_id, "ts": _utcnow(),
            "transcript_id": stored["transcript_id"], "pass": 2,
            "audit_required": False})
        out = ok(transcript_id=stored["transcript_id"],
                 extraction_id=extraction_id, audit_required=False,
                 points=[self._extract_point_brief(p) for p in points],
                 record_schema=RECORD_SCHEMA, instructions=EXTRACT_INSTRUCTIONS)
        out["pass"] = 2
        return out

    @staticmethod
    def _transcript_texts(t: dict) -> dict[int, str]:
        """Event index -> raw text. Input is event -1; a non-empty
        final_output is appended as synthetic last event (§15.1)."""
        texts = {-1: t.get("input", "")}
        for e in t.get("events", []):
            texts[e["i"]] = e.get("content", "")
        if t.get("final_output"):
            texts[len(t.get("events", []))] = t["final_output"]
        return texts

    @staticmethod
    def _quote_hint(texts: dict[int, str], pos: dict, quote: str) -> str:
        """Repair feedback for a mismatched quote: what the raw text holds
        at the claimed offset, and where the quote actually occurs (nearest
        occurrence to the claim, same event first). Bounded to one line."""
        ev, off = pos["event"], pos["offset"]
        raw = texts.get(ev, "")
        there = raw[off:off + len(quote)][:60]
        occs = []
        start = 0
        while len(quote) > 0 and (at := raw.find(quote, start)) != -1:
            occs.append(at)
            start = at + 1
        if occs:
            near = min(occs, key=lambda a: abs(a - off))
            return (f"raw text there: {there!r}; quote occurs at offset "
                    f"{near}" + (" (nearest)" if len(occs) > 1 else ""))
        for oev, otext in sorted(texts.items()):
            if oev != ev and quote and quote in otext:
                return (f"raw text there: {there!r}; quote is in event "
                        f"{oev} at offset {otext.find(quote)}")
        return f"raw text there: {there!r}; quote not found in event {ev}"

    @staticmethod
    def _number_supported(value: float, quote: str) -> bool:
        """Numeric value support: the quote must contain the number as a
        number, not as a substring (8 must not match "80"; "1,234.50"
        matches 1234.5)."""
        import re
        for tok in re.findall(r"-?[0-9][0-9,]*(?:\.[0-9]+)?", quote):
            try:
                if abs(float(tok.replace(",", "")) - value) <= max(
                        1e-9, 1e-9 * abs(value)):
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _type_complaint(fspec: dict, raw) -> str:
        """Why a claimed value was dropped from the validated context."""
        want = fspec.get("type", "?")
        if isinstance(raw, bool):
            got = "boolean"
        elif isinstance(raw, int):
            got = "integer"
        elif isinstance(raw, float):
            got = "number"
        elif isinstance(raw, str):
            got = "string"
        else:
            got = type(raw).__name__
        enum = fspec.get("enum")
        if enum is not None and isinstance(raw, str) and raw not in enum:
            return (f"has the wrong value {raw!r} (want one of "
                    f"{list(enum)})")
        return f"has the wrong JSON type (want {want}, got {got})"

    @staticmethod
    def _record_identity(choice, pos: dict, context: dict) -> str:
        """Canonical identity of an accepted record: same choice, same
        decision position, same validated context."""
        pos = pos or {}
        return json.dumps(
            [choice, pos.get("event"), pos.get("offset"), context],
            sort_keys=True, default=str)

    def _validate_record(self, rec: dict, texts: dict[int, str],
                         points_by_id: dict[str, dict], seen: dict[str, int],
                         prior_points: set[str], seen_ids: set[str],
                         prior_ids: set[str]
                         ) -> tuple[dict | None, list[tuple[str, str]] | None]:
        """One §15.2 record -> (accepted, None) or (None, [(code, message)]).
        Structural faults fail fast; content checks (§15.2 table rows) all
        report, so one repair round sees every fault in the record."""
        if not isinstance(rec, dict):
            return None, [("invalid_record", "record must be an object")]
        pid = rec.get("point_id")
        point = points_by_id.get(pid) if isinstance(pid, str) else None
        if point is None:
            return None, [("unknown_point",
                           f"unknown point {pid!r} for this workflow")]
        choice = rec.get("choice")
        if choice not in point["options"]:
            return None, [("choice_not_in_options",
                           f"{choice!r} is not in {list(point['options'])}")]
        pos = rec.get("decision_pos")
        if (not isinstance(pos, dict) or not isinstance(pos.get("event"), int)
                or not isinstance(pos.get("offset"), int) or pos["offset"] < 0):
            return None, [("invalid_record",
                           "decision_pos needs {event: int, offset: int >= 0}")]
        quote = rec.get("choice_quote")
        if not isinstance(quote, str) or not quote:
            return None, [("invalid_record",
                           "choice_quote must be a non-empty string")]
        context = rec.get("context")
        if not isinstance(context, dict):
            return None, [("invalid_record", "context must be an object")]
        evidence = rec.get("evidence")
        if not isinstance(evidence, dict):
            return None, [("invalid_record", "evidence must be an object")]
        for field, entry in evidence.items():
            if (not isinstance(entry, dict) or not isinstance(entry.get("pos"), dict)
                    or not isinstance(entry["pos"].get("event"), int)
                    or not isinstance(entry["pos"].get("offset"), int)
                    or entry["pos"]["offset"] < 0
                    or not isinstance(entry.get("quote"), str)):
                return None, [("invalid_record",
                               f"evidence {field!r} needs "
                               "{pos: {event, offset}, quote}")]
        bad: list[tuple[str, str]] = []
        if pos["event"] not in texts:
            bad.append(("quote_mismatch",
                        f"decision event {pos['event']} is outside the transcript"))
        else:
            dtext = texts[pos["event"]]
            if dtext[pos["offset"]:pos["offset"] + len(quote)] != quote:
                bad.append(("quote_mismatch",
                            "choice_quote is not the raw text at decision_pos "
                            f"({self._quote_hint(texts, pos, quote)})"))
        ends: dict[str, tuple[int, int]] = {}
        for field, entry in evidence.items():
            epos, equote = entry["pos"], entry["quote"]
            if epos["event"] not in texts:
                bad.append(("quote_mismatch",
                            f"evidence {field!r}: event is outside the transcript"))
                continue
            etext = texts[epos["event"]]
            if etext[epos["offset"]:epos["offset"] + len(equote)] != equote:
                bad.append(("quote_mismatch",
                            f"evidence {field!r}: quote is not the raw text "
                            f"at pos ({self._quote_hint(texts, epos, equote)})"))
            else:
                ends[field] = (epos["event"], epos["offset"] + len(equote))
        for field, (ev, end) in ends.items():
            if ev > pos["event"] or (ev == pos["event"] and end > pos["offset"]):
                bad.append(("future_evidence",
                            f"evidence {field!r} ends after decision_pos"))
        fields = point.get("context_fields", {}) or {}
        # validated early: a required field whose claim is dropped (wrong JSON
        # type or not an allowed enum member) is missing, not stored (B1).
        validated_pre, _, _ = R.validate_context(point, context)
        for field in evidence:
            v = context.get(field)
            q = evidence[field]["quote"]
            if v is None:
                if field in (point.get("required_fields") or []):
                    continue  # reported as missing_field below
                bad.append(("value_unsupported",
                            f"evidence {field!r} has no claimed value"))
                continue
            fspec = fields.get(field) or {}
            is_enum = (fspec.get("type") == "enum"
                       or fspec.get("enum") is not None)
            if isinstance(v, bool) or is_enum:
                # §15.2 table: booleans and enums need only a non-empty
                # quote; the judgment itself is the extractor's, audited
                # in §15.3 (a code like "damaged_item" never appears
                # verbatim in raw text).
                supported = bool(q)
            elif isinstance(v, (int, float)):
                supported = self._number_supported(float(v), q)
            elif isinstance(v, str):
                supported = (v in q) or (v[:40] in q)
            else:
                supported = False
            if not supported:
                bad.append(("value_unsupported",
                            f"value of {field!r} is not supported by its quote"))
        for name in point.get("required_fields") or []:
            if name not in point.get("context_fields", {}):
                continue
            if name in validated_pre and name in evidence:
                continue
            raw = context.get(name)
            if raw is None or name not in evidence:
                bad.append(("missing_field",
                            f"required field {name!r} needs a value and evidence"))
            else:
                bad.append(("missing_field",
                            f"required field {name!r} "
                            f"{self._type_complaint(fields[name], raw)}"))
        for field, v in context.items():
            if (field in fields and v is not None and field not in evidence
                    and field not in (point.get("required_fields") or [])):
                bad.append(("missing_field",
                            f"field {field!r} has a value but no evidence"))
        if not bad and (seen.get(pid, 0) >= 1 or pid in prior_points):
            if not point.get("repeatable"):
                bad.append(("duplicate_record",
                            f"second record for {pid!r} "
                            "(point is not repeatable)"))
        if not bad and point.get("repeatable"):
            # repeatable points accept distinct records, but a resubmitted
            # identical record (retry, reopened pass 1) is still a duplicate.
            ident = self._record_identity(choice, pos, validated_pre)
            if ident in seen_ids or ident in prior_ids:
                bad.append(("duplicate_record",
                            f"identical record for {pid!r} already stored"))
        if bad:
            return None, bad
        # validated context: unknown and mistyped fields drop (validate_context
        # semantics); the extractor quoted what it saw, the row keeps what the
        # point understands. Rule errors are ignored: only the allowed/forbidden
        # verdict matters (§15.2 keeps forbidden choices, flagged).
        validated, _, _ = R.validate_context(point, context)
        allowed, rule_hits, warnings = RU.apply_rules(point, validated)
        at_start = all(evidence[f]["pos"]["event"] == -1 for f in evidence)
        return {"point": point, "choice": choice,
                "decision_pos": {"event": pos["event"], "offset": pos["offset"]},
                "context": validated, "at_start": at_start, "allowed": allowed,
                "rule_hits": rule_hits, "warnings": warnings,
                "identity": self._record_identity(choice, pos, validated),
                "rule_violation_in_run": choice not in allowed}, None

    @staticmethod
    def _state_slice(t: dict, pos: dict, budget: int = 4000) -> str:
        """Transcript text strictly before decision_pos: the state the run
        model saw when it decided. Over budget, head + tail survive (the input
        carries the ticket, the tail the decision context)."""
        ev, off = pos["event"], pos["offset"]
        if ev == -1:
            text = t.get("input", "")[:off]
        else:
            parts = [t.get("input", "")]
            for e in t.get("events", []):
                if e["i"] < ev:
                    parts.append(e.get("content", ""))
                elif e["i"] == ev:
                    parts.append(e.get("content", "")[:off])
            if ev == len(t.get("events", [])) and t.get("final_output"):
                parts.append(t["final_output"][:off])
            text = "\n\n".join(p for p in parts if p)
        if len(text) > budget:
            text = text[:1500] + "\n…\n" + text[len(text) - (budget - 1500):]
        return text

    def _extract_call2(self, args: dict) -> dict:
        from . import analyze as AN
        ex = self.store.get_extraction(args["extraction_id"])
        if ex is None:
            return err("unknown_extraction",
                       f"unknown extraction_id {args['extraction_id']!r}")
        extractor = args.get("extractor")
        if not isinstance(extractor, str) or not extractor:
            return err("invalid_args",
                       "call 2 needs extractor (the extracting model id)")
        records = args.get("records")
        if not isinstance(records, list):
            return err("invalid_args", "records must be a list")
        new_points = args.get("new_points", [])
        if not isinstance(new_points, list):
            return err("invalid_args", "new_points must be a list")
        t = self.store.get_transcript(ex["transcript_id"])
        assert t is not None
        points = self._extract_points(t["workflow"])
        by_id = {p["point_id"]: p for p in points}
        texts = self._transcript_texts(t)
        if ex["pass"] == 1:
            rows = self.store.decisions_for_transcript(t["transcript_id"])
            prior = {r["point_id"] for r in rows}
            prior_ids = {self._record_identity(
                r.get("extracted_choice"), r.get("decision_pos") or {},
                r.get("context") or {}) for r in rows}
        else:
            audits = self.store.audits_for_transcript(t["transcript_id"])
            prior = {a["point_id"] for a in audits}
            prior_ids = {self._record_identity(
                a["record"].get("choice"),
                a["record"].get("decision_pos") or {},
                a["record"].get("context") or {}) for a in audits}
        accepted: list[dict] = []
        detail: list[dict] = []
        rejected: list[dict] = []
        seen: dict[str, int] = {}
        seen_ids: set[str] = set()
        for rec in records:
            good, bad = self._validate_record(rec, texts, by_id, seen, prior,
                                              seen_ids, prior_ids)
            if bad is not None:
                pid = rec.get("point_id") if isinstance(rec, dict) else None
                for code, msg in bad:
                    rejected.append({"point_id": pid, "code": code,
                                     "message": msg})
                continue
            assert good is not None
            seen[good["point"]["point_id"]] = seen.get(
                good["point"]["point_id"], 0) + 1
            seen_ids.add(good["identity"])
            entry = self._store_accepted(ex, t, extractor, good)
            accepted.append(entry["response"])
            detail.append(entry["detail"])
        disagreements, jev_errors = self._jev_disagreements(ex, detail)
        codes: dict[str, int] = dict(ex.get("rejected_codes") or {})
        for r in rejected:
            codes[r["code"]] = codes.get(r["code"], 0) + 1
        self.store.complete_extraction(
            ex["extraction_id"], extractor,
            (ex.get("n_accepted") or 0) + len(accepted),
            (ex.get("n_rejected") or 0) + len(rejected),
            codes, list(ex.get("disagreements") or []) + disagreements)
        if ex["pass"] == 1:
            for pid in {a.get("point_id") for a in accepted
                        if a.get("point_id")}:
                self._maybe_fit_hosted(pid, "plan")
                self._maybe_fit_private_plan(pid)
        scored = [AN.score_candidate(np, self.cfg) for np in new_points]
        out = ok(extraction_id=ex["extraction_id"],
                 transcript_id=t["transcript_id"],
                 accepted=accepted, rejected=rejected,
                 jev_disagreements=disagreements, new_points=scored)
        out["pass"] = ex["pass"]
        if jev_errors:
            out["jev_errors"] = sorted(set(jev_errors))
        if ex["pass"] == 1 and t.get("plan_id"):
            rate = self._update_followed(t["plan_id"], t["transcript_id"])
            if rate is not None:
                out["followed_rate"] = rate
        return out

    def _store_accepted(self, ex: dict, t: dict, extractor: str,
                        good: dict) -> dict:
        point = good["point"]
        if ex["pass"] == 2:
            audit_id = f"ea_{uuid.uuid4().hex[:8]}"
            self.store.audit_insert({
                "audit_id": audit_id, "ts": _utcnow(),
                "transcript_id": t["transcript_id"],
                "extraction_id": ex["extraction_id"],
                "point_id": point["point_id"],
                "record": {"choice": good["choice"],
                           "decision_pos": good["decision_pos"],
                           "context": good["context"],
                           "at_start": good["at_start"]}})
            return {"response": {
                        "audit_id": audit_id, "point_id": point["point_id"],
                        "choice": good["choice"], "at_start": good["at_start"],
                        "rule_violation_in_run": good["rule_violation_in_run"]},
                    "detail": None}
        state_text = self._state_slice(t, good["decision_pos"])
        did = uuid.uuid4().hex
        self.store.insert_decision({
            "decision_id": did, "point_id": point["point_id"],
            "point_version": point.get("point_version", 1), "ts": _utcnow(),
            "context": good["context"], "state_text": state_text,
            "options": list(point["options"]),
            "allowed_options": good["allowed"], "rule_hits": good["rule_hits"],
            "mode": "extracted", "source": "extracted",
            "extracted_choice": good["choice"], "run_id": t["run_id"],
            "transcript_id": t["transcript_id"],
            "extraction_id": ex["extraction_id"], "extractor": extractor,
            "decision_pos": good["decision_pos"], "at_start": good["at_start"],
            "rule_violation_in_run": good["rule_violation_in_run"],
            "warnings": good["warnings"] or None,
            "phase": point.get("phase", "collect")})
        return {"response": {
                    "decision_id": did, "point_id": point["point_id"],
                    "choice": good["choice"], "at_start": good["at_start"],
                    "rule_violation_in_run": good["rule_violation_in_run"]},
                "detail": {"decision_id": did, "point": point,
                           "choice": good["choice"], "state_text": state_text,
                           "allowed": good["allowed"]}}

    def _jev_disagreements(self, ex: dict, detail: list[dict]
                           ) -> tuple[list[dict], list[str]]:
        """Mode C audit (§15.2): M12 P3 deleted the probe, so there is no
        probe top to compare against; extract completes with no
        disagreements. Read-only; never changes state."""
        return [], []

    # -- do_train --------------------------------------------------------
    def do_train(self, args: dict) -> dict:
        errors = SV.validate("do_train", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        return self._dm_train_tiered(args)

    def _dm_train_tiered(self, args: dict) -> dict:
        """v1.7 precondition (H5, G6): a hosted-tier point trains only when
        train_recommended, private_model_only or force; until M11 the private
        tier itself is not built, so allowed calls report what would train."""
        point = self.store.get_point(args["point_id"])
        if point is None:
            return err("unknown_point",
                       f"unknown point_id {args['point_id']!r}")
        if self._hosted_tier(point):
            reasons = self._train_reasons(point)
            if not reasons and not point.get("private_model_only") \
                    and not args.get("force", False):
                return err("hosted_jev_sufficient",
                           "hosted Jev serves this point; train only when "
                           "train_recommended, private_model_only or force",
                           train_recommended=False,
                           hosted=self._hosted_metrics(point))
        if self.trainer is not None:
            return self._dm_train_job(args, point)
        # F9: no trainer was built; say exactly why (key or cookbook pin).
        from .trainer import live_trainer_unavailable_reason

        labeled = L.labeled_rows(
            self.store.decisions_for_point(args["point_id"]), point,
            self.cfg["train"]["teacher_only_weight"],
            self.cfg["extract"]["weight"])
        by_opt: dict[str, int] = {}
        for _, lab, _, _ in labeled:
            by_opt[lab] = by_opt.get(lab, 0) + 1
        summary = {"n_labeled": len(labeled), "per_option": by_opt,
                   "dry_run": True}
        reason = live_trainer_unavailable_reason(
            self.cfg, self.store, self.fireworks_key) or "no_fireworks_key"
        detail = {"no_fireworks_key": "set fireworks.key_ref to train",
                  "cookbook_missing": "fireworks.cookbook_dir is not a "
                                      "cookbook checkout",
                  "cookbook_mismatch": "cookbook checkout is not at "
                                       "fireworks.cookbook_commit"}[reason]
        # F9: private_tier_not_built is removed; do_train names the
        # reason (no_fireworks_key / cookbook_missing / cookbook_mismatch).
        return err(reason, f"private-tier training is unavailable: {detail}; "
                           "nothing was trained",
                   export=summary)

    def _dm_train_job(self, args: dict, point: dict) -> dict:
        """Tiered do_train on a trainer (M16, §2.6): dry_run estimates, launch
        with confirm, status polling, smoke-first then paid-gated full jobs."""
        ap = self.autopilot
        finals = ap.poll_and_finalize()
        if finals:
            fin = finals[-1]
            job = self.store.job_get(fin.get("job_id", ""))
            status = {"train_failed": "failed",
                      "eval_failed": "eval_failed",
                      "eval_started": "evaluating"}.get(fin["action"])
            if status is None:
                status = ("smoke_done"
                          if (job or {}).get("kind") == "smoke" else "done")
            return ok(job_id=fin.get("job_id"), status=status,
                      kind=(job or {}).get("kind"), metrics=(
                          self._job_metrics(job) if job else {}))
        if args.get("cancel"):
            return self._dm_train_cancel(args, point)
        if args.get("retry_eval"):
            return self._dm_train_retry_eval(args, point)
        running = self.store.jobs_running()
        launching = self.store.jobs_launching()
        if (running or launching) and not args.get("dry_run", False):
            if running:
                job = running[0]
                return ok(job_id=job["job_id"], status=job["status"],
                          kind=job["kind"], estimate_usd=job["estimate_usd"])
            return self._launching_response(launching[0])
        dry = ap.dry_run(point)
        if args.get("dry_run", False):
            return ok(dry_run=True, kind=dry["kind"], joins=dry["joins"],
                      estimate=dry["estimate"], trigger=point["point_id"])
        from .autopilot import _fireworks_allowed
        if not _fireworks_allowed(point):
            return err("remote_not_allowed",
                       f"{point['point_id']} is private without 'fireworks' "
                       "in remote_ok: never exported, trained or sent")
        trig_rows = next((j["n_rows"] for j in dry["joins"]
                          if j["point_id"] == point["point_id"]), 0)
        if trig_rows < self.cfg["train"]["min_train"]:
            return err("not_enough_data",
                       f"need {self.cfg['train']['min_train']} labelled rows; "
                       f"have {trig_rows}")
        if getattr(self.trainer, "needs_key", True) and not self.fireworks_key:
            return err("no_fireworks_key",
                       "set fireworks.key_ref to train (do_train)")
        if not args.get("confirm", False):
            return err("confirm_required",
                       "re-run with confirm: true to launch the job")
        est = dry["estimate"]
        headroom = ap._fireworks_headroom()
        free_ok = False
        if dry["kind"] == "full":
            free_ok = ap.manual_free_ok(bool(args.get("training_free_confirmed")))
            counted = float(est.get("eval_usd", est["estimate_usd"])) \
                if free_ok else float(est["estimate_usd"])
            if not free_ok and not self.cfg.get("fireworks", {}).get(
                    "allow_paid_training", False):
                return err("paid_job_not_approved",
                           "a full job needs training_free_confirmed: true "
                           "(before 2026-12-01 Pacific) or "
                           "fireworks.allow_paid_training",
                           estimate_usd=est["estimate_usd"])
        else:
            counted = float(est["estimate_usd"])
        if headroom is None or counted > headroom:
            return err("budget_exceeded",
                       f"estimate ${counted:.4f} exceeds the remaining "
                       f"Fireworks budget", estimate_usd=counted)
        from .autopilot import JobRunning

        try:
            job = ap.launch_job(point, dry, counted, autopilot=False,
                                idem=args.get("idem"), free_train=free_ok)
        except JobRunning:
            running = self.store.jobs_running()
            launching = self.store.jobs_launching()
            job0 = (running or launching or [{}])[0]
            if job0.get("status") == "launching":
                return self._launching_response(job0)
            return ok(job_id=job0.get("job_id"),
                      status=job0.get("status", "running"),
                      kind=job0.get("kind"), estimate_usd=job0.get("estimate_usd", 0))
        return ok(job_id=job["job_id"], status="running", kind=job["kind"],
                  estimate_usd=job["estimate_usd"])

    def _launching_response(self, job: dict) -> dict:
        """F4: a launching poll carries (status, stale) plus the launcher
        kind. The identity is masked to manual/autopilot: no pids or hosts
        leak, but a second launcher can see who is on it."""
        return ok(job_id=job["job_id"], status="launching",
                  stale=self.autopilot._launch_stale(job),
                  launcher=("autopilot" if job.get("autopilot") else "manual"),
                  kind=job["kind"], estimate_usd=job["estimate_usd"])

    def _dm_train_cancel(self, args: dict, point: dict) -> dict:
        """F4: do_train cancel (a job_id, or True for this point's live job).

        Kills the trainer job and marks the row cancelled: it never
        finalizes and never relaunches."""
        ap = self.autopilot
        target = args.get("cancel")
        job = None
        if target is True:
            live = self.store.jobs_running() + self.store.jobs_launching()
            job = next((j for j in live
                        if any(p.get("point_id") == point["point_id"]
                               for p in j.get("points", []))), None)
            if job is None:
                return err("no_running_job",
                           f"no launching or running job for {point['point_id']!r}")
        else:
            job = self.store.job_get(target)
            if job is None:
                return err("unknown_job", f"unknown job_id {target!r}")
            if job["status"] not in ("running", "launching", "evaluating"):
                return err("job_not_running",
                           f"job {target!r} is {job['status']}")
        tid = job["job_id"] if job["status"] == "running" \
            else (job.get("idempotency_key") or job["job_id"])
        out = ok(job_id=job["job_id"], status="cancelled", kind=job["kind"])
        if self.trainer is None:
            out["warning"] = ("no trainer: marked cancelled, but a detached "
                              "job may still run")
        else:
            try:
                self.trainer.cancel(tid)
            except Exception as e:
                return err("cancel_failed", f"trainer cancel failed: {e}")
        ap._mark_cancelled(job)
        return out

    def _dm_train_retry_eval(self, args: dict, point: dict) -> dict:
        """F8: do_train retry_eval (re-run a failed evaluation)."""
        from .fireworks_trainer import FireworksTrainer

        ap = self.autopilot
        if not args.get("confirm", False):
            return err("confirm_required",
                       "re-run with confirm: true to retry the evaluation")
        if not isinstance(self.trainer, FireworksTrainer):
            return err("retry_not_possible",
                       "evaluation needs the live Fireworks trainer")
        if self.store.jobs_running() or self.store.jobs_launching():
            return err("job_running",
                       "another job is live; retry when it finishes")
        cands = [j for j in self.store.jobs_eval_failed()
                 if any(p.get("point_id") == point["point_id"]
                        for p in j.get("points", []))]
        if not cands:
            return err("nothing_to_retry",
                       f"no failed evaluation for {point['point_id']!r}")
        job = max(cands, key=lambda j: (j.get("created_at", ""),
                                        j.get("job_id", "")))
        ap.retry_evaluation(job)
        return ok(job_id=job["job_id"], status="evaluating",
                  kind=job["kind"], estimate_usd=job["estimate_usd"])

    @staticmethod
    def _job_metrics(job: dict) -> dict:
        return {"estimate_usd": job.get("estimate_usd", 0),
                "spend_usd": job.get("spend_usd", 0),
                "points": [p.get("point_id") for p in job.get("points", [])],
                "error": job.get("error")}

    def _hosted_metrics(self, point: dict) -> dict:
        """Hosted numbers attached to hosted_jev_sufficient (H5)."""
        fit = self._hosted_fit(point)
        T = float(fit["temperature"]) if fit else 1.0
        ev = self._evidence_rows(point, fit) if fit else []
        scored = self._score_hosted(point, ev, T)
        t = fit.get("threshold") if fit else None
        pool = [ok for c, ok in scored if t is not None and c >= t]
        return {
            "fit": ({"temperature": fit["temperature"],
                     "threshold": fit["threshold"],
                     "coverage": fit["coverage"],
                     "n_fit": fit["n_fit"]} if fit else None),
            "evidence_rows": len(ev),
            "evidence_coverage": (len(pool) / len(scored)
                                  if scored and t is not None else 0.0),
            "evidence_error": (sum(1 for ok in pool if not ok) / len(pool)
                               if pool else None),
        }

    # -- do_promote ------------------------------------------------------
    def do_promote(self, args: dict) -> dict:
        errors = SV.validate("do_promote", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        if args.get("tier") is not None and args.get("phase") != "serve":
            return err("invalid_args", "tier is only valid with phase: serve")
        if args.get("downgrade") is not None:
            return self._promote_downgrade(args)
        if args.get("point_id") is None:
            return err("invalid_args", "do_promote needs point_id or workflow")
        if args.get("phase") is not None and args.get("plan_phase") is not None:
            return err("invalid_args", "move one of phase/plan_phase per call")
        if args.get("plan_phase") is not None:
            return self._promote_plan_phase(args)
        if args.get("phase") is None:
            return err("invalid_args", "do_promote needs phase, plan_phase or downgrade")
        point = self.store.get_point(args["point_id"])
        if point is None:
            return err("unknown_point", f"unknown point_id {args['point_id']!r}")
        target = args["phase"]
        if target == "shadow":
            tier = point.get("tier") or ("private" if point.get(
                "private_model_only") else "hosted_jev")
            if tier == "private" or point.get("private_model_only"):
                return self._promote_shadow_private(point)
            return self._promote_shadow_hosted(point)
        if target == "serve":
            tier_err = self._check_promote_tier(point, args.get("tier"))
            if tier_err:
                return tier_err
            eff = args.get("tier") or self._effective_tier(point)
            pending = self._pending_private_model(point)
            if point.get("phase") == "serve" and pending is not None \
                    and eff == "private" and args.get("tier") is None:
                # F9 (§7.2 step 5): switch to the pending checkpoint.
                return self._promote_switch_pending(point, pending, args)
            checks = self._serve_checks(point, bool(args.get("confirm", False)),
                                        eff)
            failed = [k for k, v in checks.items() if not v["pass"]]
            if failed:
                if eff == "private" and not checks.get(
                        "private_item8", {}).get("pass", True):
                    return err("private_not_better",
                               "§6.4 item 8 not met: the private model has not "
                               "beaten hosted Jev on C",
                               failed=failed, checks=checks)
                return err("promotion_refused",
                           f"serve promotion refused: {', '.join(failed)}",
                           failed=failed, checks=checks)
            if args.get("tier") is not None:
                point["tier"] = args["tier"]
            point["phase"] = "serve"
            self.store.put_point(point)
            return ok(point_id=point["point_id"], phase="serve", checks=checks)
        self._autopilot_note_manual_move(point, target)
        point["phase"] = target
        self.store.put_point(point)
        return ok(point_id=point["point_id"], phase=target)

    def _hosted_shadow_gate(self, point: dict, scope: str) -> dict | None:
        """Key, budget and privacy checks shared by decide-phase and
        plan-phase shadow promotion on the hosted tier (review C1). None
        when all pass, else the error to return."""
        if point.get("data_class", "private") == "private":
            ro = point.get("remote_ok", [])
            allowed = True if ro is True else "hosted_jev" in (ro or [])
            if not allowed:
                return err("remote_not_allowed",
                           f"{point['point_id']} is private without "
                           "'hosted_jev' in remote_ok")
        dec = self.deciders.get("hosted_jev")
        if dec is None or not getattr(dec, "key", None):
            return err("no_hosted_jev_key",
                       "set hosted_jev.key_ref to call hosted Jev")
        if float(self.cfg["hosted_jev"].get("monthly_budget_usd") or 0) <= 0:
            checks = {"budget": {
                "pass": False,
                "detail": f"{scope}shadow needs "
                          "hosted_jev.monthly_budget_usd > 0"}}
            return err("promotion_refused",
                       f"{scope}shadow needs a hosted budget",
                       failed=["budget"], checks=checks)
        return None

    def _promote_shadow_hosted(self, point: dict) -> dict:
        """Shadow on the hosted tier: no training; needs the key, a budget
        and the privacy gate (build §7)."""
        gate = self._hosted_shadow_gate(point, "")
        if gate is not None:
            return gate
        self._autopilot_note_manual_move(point, "shadow")
        point["phase"] = "shadow"
        self.store.put_point(point)
        return ok(point_id=point["point_id"], phase="shadow")

    def _promote_shadow_private(self, point: dict) -> dict:
        """Shadow on the private tier (§2.7): the Fireworks key (fakes skip
        it) plus a trained private model for the current version. The key
        gates first: training needs it too, so a keyless point is told to
        set the key before it is told to train."""
        if getattr(self.trainer, "needs_key", True) and not self.fireworks_key:
            return err("no_fireworks_key",
                       "set fireworks.key_ref to serve the private tier")
        model = self.store.active_model(point["point_id"], "private")
        if model is None or model.get("point_version") != point.get(
                "point_version", 1):
            return err("private_model_not_trained",
                       "shadow on the private tier needs a trained private "
                       "model (do_train)")
        self._autopilot_note_manual_move(point, "shadow")
        point["phase"] = "shadow"
        self.store.put_point(point)
        return ok(point_id=point["point_id"], phase="shadow")

    def _promote_switch_pending(self, point: dict, pending: dict,
                                args: dict) -> dict:
        """F9 (§7.2 step 5): serve-to-serve switch onto the pending
        checkpoint (§6.4 for the new checkpoint + the user's confirm)."""
        checks = self._pending_switch_checks(
            point, pending, bool(args.get("confirm", False)))
        failed = [k for k, v in checks.items() if not v["pass"]]
        if failed:
            if not checks.get("private_item8", {}).get("pass", True):
                return err("private_not_better",
                           "§6.4 item 8 not met for the pending checkpoint: "
                           "it regresses against the serving one",
                           failed=failed, checks=checks)
            return err("promotion_refused",
                       f"checkpoint switch refused: {', '.join(failed)}",
                       failed=failed, checks=checks)
        self.store.activate_model(pending["id"])
        if "pending_private_model" in point:
            del point["pending_private_model"]
        self.store.put_point(point)
        return ok(point_id=point["point_id"], phase="serve", switched=True,
                  checks=checks)

    @staticmethod
    def _note_demotion_cooldown(point: dict) -> None:
        """E4: one helper every demotion path calls to start the cooldown."""
        point["autopilot_demotion_ts"] = _utcnow()

    @staticmethod
    def _autopilot_note_manual_move(point: dict, target: str) -> None:
        """M16 bookkeeping for a successful manual do_promote (decide phase).

        Tool calls can't tell the user from the agent, so every call counts:
        paused/collect stops the point (until the next shadow call), shadow
        resumes it, and any demotion from serve starts the cooldown (E4: the
        serve -> paused/collect leg sets it, so a later shadow resume stays
        in cooldown).
        Inert when autopilot is off (T-AP1): plain point fields.
        """
        prev = point.get("phase", "collect")
        if target == "shadow":
            point["autopilot_stopped"] = False
            if prev == "serve":
                Service._note_demotion_cooldown(point)
        elif target in ("paused", "collect"):
            point["autopilot_stopped"] = True
            if prev == "serve":
                Service._note_demotion_cooldown(point)

    def _check_promote_tier(self, point: dict, requested: str | None) -> dict | None:
        """v1.7 tier switch on promote-to-serve (item 8 in _serve_checks)."""
        if requested is None:
            return None
        if requested == "hosted_jev" and point.get("private_model_only"):
            return err("privacy_conflict",
                       "a private_model_only point cannot use hosted_jev")
        return None

    @staticmethod
    def _margin_check(n_pool: int, pool_err: float, baseline: float,
                      target: float, min_margin: float,
                      baseline_label: str) -> dict:
        """Margin of the serve pool over the escalation baseline (§6.4).

        When the baseline already meets the target, no margin is possible or
        needed (a perfect model can't beat a perfect baseline), so the check
        is skipped instead of refusing. The detail always says which rule ran.
        """
        if baseline <= target:
            return {"pass": n_pool > 0,
                    "detail": f"pool error={pool_err:.4f}, {baseline_label}="
                              f"{baseline:.4f} already ≤ target {target}: "
                              "margin check skipped"}
        return {"pass": n_pool > 0
                and pool_err <= baseline - min_margin,
                "detail": f"pool error={pool_err:.4f}, {baseline_label}="
                          f"{baseline:.4f}, need margin {min_margin}"}

    def _serve_checks(self, point: dict, confirm: bool,
                      tier: str | None = None,
                      since: tuple | None = None) -> dict:
        """Promotion criteria of §6.4 (item 7 is added for gateway points in M9).

        since (M16, autopilot re-promotion only): live evidence rows decided
        at/before it are excluded. None (manual promotion, reports) is v1.8.
        """
        eff = tier or self._effective_tier(point)
        if eff == "private":
            return self._serve_checks_private(point, confirm, since)
        return self._serve_checks_hosted(point, confirm, since)


    def _serve_checks_hosted(self, point: dict, confirm: bool,
                             since: tuple | None = None) -> dict:
        """§6.4 for the hosted tier (main spec §7.7): the threshold comes from
        the fit rows; item 1, coverage, the Wilson bound and the margin are
        computed on the evidence rows."""
        from . import stats as S
        tc, sc = self.cfg["train"], self.cfg["serve"]
        target = point.get("target_error", sc["target_error"])
        fit = self._hosted_fit(point)
        T = float(fit["temperature"]) if fit else 1.0
        ev = self._evidence_rows(point, fit, since) if fit else []
        scored = self._score_hosted(point, ev, T)
        checks: dict[str, dict] = {}
        checks["min_examples"] = {
            "pass": len(ev) >= tc["min_examples"],
            "detail": f"{len(ev)} labelled evidence rows, need "
                      f"{tc['min_examples']} (fit rows never count)"}
        threshold = fit.get("threshold") if fit else None
        checks["threshold"] = {
            "pass": threshold is not None,
            "detail": f"threshold={threshold} "
                      f"(fit on {fit['n_fit'] if fit else 0} rows)"}
        pool = [(c, ok) for c, ok in scored
                if threshold is not None and c >= threshold]
        coverage = len(pool) / len(scored) if scored else 0.0
        checks["coverage"] = {
            "pass": threshold is not None and coverage >= sc["min_coverage"],
            "detail": f"coverage={coverage:.3f}, need {sc['min_coverage']}"}
        n_pool, e_pool = len(pool), sum(1 for _, ok in pool if not ok)
        upper = S.wilson_upper(e_pool, n_pool) if n_pool else 1.0
        checks["evidence"] = {
            "pass": n_pool > 0 and upper <= target,
            "detail": f"pool n={n_pool}, errors={e_pool}, "
                      f"Wilson upper={upper:.4f}, target={target}"}
        acc = (sum(1 for _, ok in scored if ok) / len(scored)
               if scored else 0.0)
        baseline = 1.0 - acc
        pool_err = (e_pool / n_pool) if n_pool else 1.0
        checks["margin"] = self._margin_check(
            n_pool, pool_err, baseline, target, sc["min_margin_vs_random"],
            "evidence baseline")
        checks["shadow"] = {
            "pass": len(ev) >= sc["min_shadow"],
            "detail": f"{len(ev)} evidence rows, need {sc['min_shadow']}"}
        latest = scored[-sc["min_shadow"]:] if scored else []
        agree = (sum(1 for _, ok in latest if ok) / len(latest)
                 if latest else None)
        fit_agree = fit.get("fit_agreement") if fit else None
        checks["shadow_agreement"] = {
            "pass": agree is not None and fit_agree is not None
            and len(latest) >= sc["min_shadow"]
            and abs(agree - fit_agree) <= 0.05,
            "detail": f"evidence agreement={agree} (latest {len(latest)}), "
                      f"fit={fit_agree} (need within 0.05)"}
        rows = self.store.decisions_for_point(point["point_id"])
        both, disagree = L.teacher_disagreement(rows)
        unreliable = both > 0 and disagree / both > tc["teacher_alarm"]
        checks["teacher"] = {
            "pass": not unreliable,
            "detail": f"outcome/teacher disagreements {disagree}/{both} "
                      f"(alarm above {tc['teacher_alarm']})"}
        checks["confirm"] = {"pass": bool(confirm),
                             "detail": "the user agreed" if confirm else "confirm: true required"}
        labelled, decided = self._shadow_label_coverage(point, since)
        ratio = labelled / decided if decided else 0.0
        checks["labelled_ratio"] = {
            "pass": decided > 0 and ratio >= 0.9,
            "detail": f"{labelled}/{decided} decided shadow rows labelled "
                      f"(ratio {ratio:.3f}, need 0.90)"}
        if point.get("source") == "gateway":
            from . import gateway as GW
            fe = GW.format_evidence(self.cfg, self.store, point)
            gcfg = self.cfg["gateway"]
            n_ok = fe["n_compared"] >= gcfg["format_min_n"]
            wl = fe["wilson_lower"]
            w_ok = wl is not None and wl >= gcfg["format_match_min"]
            ur = fe["unparsed_rate"]
            u_ok = ur is not None and ur <= gcfg["max_unparsed"]
            checks["gateway_format"] = {
                "pass": n_ok and w_ok and u_ok,
                "detail": (f"compared={fe['n_compared']}/{gcfg['format_min_n']}, "
                           f"match {fe['n_match']}/{fe['n_compared']} "
                           f"(Wilson lower {_fmt_num(wl)}, need "
                           f"{gcfg['format_match_min']}), unparsed "
                           f"{fe['n_unparsed']}/{fe['n_answered']} "
                           f"({_fmt_num(ur)}, max {gcfg['max_unparsed']})")}
        return checks

    def _serve_checks_private(self, point: dict, confirm: bool,
                              since: tuple | None = None) -> dict:
        """§6.4 for the private tier on fakes (H5; M11 refines): items 1-7
        over the comparison set C with the private threshold, plus item 8."""
        from . import stats as S
        tc, sc = self.cfg["train"], self.cfg["serve"]
        target = point.get("target_error", sc["target_error"])
        comp = self._private_comparison(point, since)
        priv = (comp or {}).get("private") or {}
        scored = (comp or {}).get("scored_private") or []
        n = (comp or {}).get("n", 0)
        tp = priv.get("threshold")
        checks: dict[str, dict] = {}
        # D: item 1 counts labelled rows (§6.4), not |C|: over C it would
        # dominate min_compare (200 > 100) and the item-8 size gate in
        # _private_comparison would never bind.
        n_labelled = len(self._tier_rows(point))
        checks["min_examples"] = {
            "pass": n_labelled >= tc["min_examples"],
            "detail": f"{n_labelled} labelled tier rows, need "
                      f"{tc['min_examples']}"}
        checks["threshold"] = {
            "pass": tp is not None,
            "detail": f"private threshold={tp}"}
        pool = [(c, ok) for c, ok in scored if tp is not None and c >= tp]
        coverage = len(pool) / len(scored) if scored else 0.0
        checks["coverage"] = {
            "pass": tp is not None and coverage >= sc["min_coverage"],
            "detail": f"coverage={coverage:.3f}, need {sc['min_coverage']}"}
        n_pool, e_pool = len(pool), sum(1 for _, ok in pool if not ok)
        upper = S.wilson_upper(e_pool, n_pool) if n_pool else 1.0
        checks["evidence"] = {
            "pass": n_pool > 0 and upper <= target,
            "detail": f"pool n={n_pool}, errors={e_pool}, "
                      f"Wilson upper={upper:.4f}, target={target}"}
        acc = (sum(1 for _, ok in scored if ok) / len(scored)
               if scored else 0.0)
        baseline = 1.0 - acc
        pool_err = (e_pool / n_pool) if n_pool else 1.0
        checks["margin"] = self._margin_check(
            n_pool, pool_err, baseline, target, sc["min_margin_vs_random"],
            "C baseline")
        checks["shadow"] = {
            "pass": n >= sc["min_shadow"],
            "detail": f"{n} comparison rows, need {sc['min_shadow']}"}
        latest = scored[-sc["min_shadow"]:] if scored else []
        agree = (sum(1 for _, ok in latest if ok) / len(latest)
                 if latest else None)
        checks["shadow_agreement"] = {
            "pass": agree is not None and len(latest) >= sc["min_shadow"]
            and abs(agree - acc) <= 0.05,
            "detail": f"C agreement latest {agree} vs overall {acc:.3f} "
                      "(need within 0.05)"}
        rows = self.store.decisions_for_point(point["point_id"])
        both, disagree = L.teacher_disagreement(rows)
        unreliable = both > 0 and disagree / both > tc["teacher_alarm"]
        checks["teacher"] = {
            "pass": not unreliable,
            "detail": f"outcome/teacher disagreements {disagree}/{both} "
                      f"(alarm above {tc['teacher_alarm']})"}
        checks["confirm"] = {"pass": bool(confirm),
                             "detail": "the user agreed" if confirm else "confirm: true required"}
        if point.get("source") == "gateway":
            from . import gateway as GW
            fe = GW.format_evidence(self.cfg, self.store, point)
            gcfg = self.cfg["gateway"]
            n_ok = fe["n_compared"] >= gcfg["format_min_n"]
            wl = fe["wilson_lower"]
            w_ok = wl is not None and wl >= gcfg["format_match_min"]
            ur = fe["unparsed_rate"]
            u_ok = ur is not None and ur <= gcfg["max_unparsed"]
            checks["gateway_format"] = {
                "pass": n_ok and w_ok and u_ok,
                "detail": (f"compared={fe['n_compared']}/{gcfg['format_min_n']}, "
                           f"match {fe['n_match']}/{fe['n_compared']} "
                           f"(Wilson lower {_fmt_num(wl)}, need "
                           f"{gcfg['format_match_min']}), unparsed "
                           f"{fe['n_unparsed']}/{fe['n_answered']} "
                           f"({_fmt_num(ur)}, max {gcfg['max_unparsed']})")}
        if point.get("private_model_only"):
            # D: §6.4 item 8 is not required for private_model_only points
            # (no hosted-Jev arm to beat).
            checks["private_item8"] = {
                "pass": True,
                "detail": "not required for private_model_only "
                          "(no hosted-Jev arm)"}
            return checks
        item8 = comp is not None and comp["pass"]
        checks["private_item8"] = {
            "pass": item8,
            "detail": ("no private model under evaluation" if comp is None
                       else f"C n={comp['n']}, private cov="
                       f"{comp['private']['coverage']:.3f} err="
                       f"{comp['private']['error']}, hosted cov="
                       f"{comp['hosted']['coverage']:.3f} err="
                       f"{comp['hosted']['error']}, discordant "
                       f"{comp['discordant_hosted_only']}/"
                       f"{comp['discordant_private_only']}, McNemar p="
                       f"{comp['mcnemar_p']:.4f}")}
        return checks

    def _promote_plan_phase(self, args: dict) -> dict:
        point = self.store.get_point(args["point_id"])
        if point is None:
            return err("unknown_point", f"unknown point_id {args['point_id']!r}")
        target = args["plan_phase"]
        if target == "shadow":
            if self._hosted_tier(point):
                # C1: hosted plan shadow needs no trained model; the key,
                # budget and privacy gate admit it, as for decide shadow.
                gate = self._hosted_shadow_gate(point, "plan ")
                if gate is not None:
                    return gate
            else:
                if getattr(self.trainer, "needs_key", True) \
                        and not self.fireworks_key:
                    return err("no_fireworks_key",
                               "set fireworks.key_ref to serve the private "
                               "tier")
                model = self.store.active_model(point["point_id"],
                                                "private")
                if model is None or model.get("point_version") != point.get(
                        "point_version", 1):
                    return err("private_model_not_trained",
                               "plan shadow on the private tier needs a "
                               "trained private model (do_train)")
            point["plan_phase"] = "shadow"
            self.store.put_point(point)
            return ok(point_id=point["point_id"], plan_phase="shadow")
        if target == "serve":
            checks = self._plan_serve_checks(
                point, bool(args.get("confirm", False)))
            failed = [k for k, v in checks.items() if not v["pass"]]
            if failed:
                return err("promotion_refused",
                           f"plan serve promotion refused: {', '.join(failed)}",
                           failed=failed, checks=checks)
            point["plan_phase"] = "serve"
            self.store.put_point(point)
            return ok(point_id=point["point_id"], plan_phase="serve",
                      checks=checks)
        point["plan_phase"] = target
        self.store.put_point(point)
        return ok(point_id=point["point_id"], plan_phase=target)

    def _plan_serve_checks(self, point: dict, confirm: bool) -> dict:
        """Promotion criteria of §15.6 (§6.4 restated for plans)."""
        return self._plan_serve_checks_hosted(point, confirm)


    def _plan_serve_checks_hosted(self, point: dict,
                                    confirm: bool) -> dict:
        """§15.6 for hosted plan points (build §9): item 1 counts evidence
        comparisons, item 2 is the plan threshold from the plan fit, items
        3-7 run on the evidence comparisons."""
        from . import stats as S
        tc = self.cfg["train"]
        target = self.cfg["plan"]["target_error"]
        fit = self._hosted_fit(point, "plan")
        T = float(fit["temperature"]) if fit else 1.0
        ev = self._plan_evidence_hosted(point, fit) if fit else []
        scored = self._score_plan_hosted(point, ev, T)
        checks: dict[str, dict] = {}
        checks["min_examples"] = {
            "pass": len(ev) >= tc["min_examples"],
            "detail": f"{len(ev)} evidence comparisons, "
                      f"need {tc['min_examples']}"}
        pt = fit.get("threshold") if fit else None
        checks["threshold"] = {
            "pass": pt is not None,
            "detail": f"plan_threshold={pt}"}
        pool = [(c, ok) for c, ok in scored if pt is not None and c >= pt]
        coverage = len(pool) / len(scored) if scored else 0.0
        checks["coverage"] = {
            "pass": pt is not None and coverage >= self.cfg["serve"]["min_coverage"],
            "detail": f"plan coverage={coverage:.3f}, "
                      f"need {self.cfg['serve']['min_coverage']}"}
        n_pool, e_pool = len(pool), sum(1 for _, ok in pool if not ok)
        kinds: dict[str, int] = {}
        for c in ev:
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
        upper = S.wilson_upper(e_pool, n_pool) if n_pool else 1.0
        checks["evidence"] = {
            "pass": n_pool > 0 and upper <= target,
            "detail": f"pool n={n_pool}, errors={e_pool}, "
                      f"Wilson upper={upper:.4f}, target={target}; "
                      f"label kinds: {kinds or 'none'}"}
        acc = (sum(1 for _, ok in scored if ok) / len(scored)
               if scored else 0.0)
        baseline = 1.0 - acc
        pool_err = (e_pool / n_pool) if n_pool else 1.0
        checks["margin"] = self._margin_check(
            n_pool, pool_err, baseline, target,
            self.cfg["serve"]["min_margin_vs_random"], "baseline")
        checks["shadow"] = {
            "pass": len(ev) >= self.cfg["plan"]["min_shadow"],
            "detail": f"{len(ev)} evidence comparisons, "
                      f"need {self.cfg['plan']['min_shadow']}"}
        latest = scored[-self.cfg["plan"]["min_shadow"]:] if scored else []
        agree = (sum(1 for _, ok in latest if ok) / len(latest)
                 if latest else None)
        fit_agree = fit.get("fit_agreement") if fit else None
        checks["shadow_agreement"] = {
            "pass": agree is not None and fit_agree is not None
            and len(latest) >= self.cfg["plan"]["min_shadow"]
            and abs(agree - fit_agree) <= 0.05,
            "detail": f"evidence agreement={agree}, "
                      f"fit={fit_agree} (need within 0.05)"}
        checks["extraction"] = {
            "pass": not point.get("extraction_unreliable", False),
            "detail": ("extraction_unreliable" if point.get(
                "extraction_unreliable", False) else "extraction reliable")}
        rows = self.store.decisions_for_point(point["point_id"])
        both, disagree = L.teacher_disagreement(rows)
        unreliable = both > 0 and disagree / both > tc["teacher_alarm"]
        checks["teacher"] = {
            "pass": not unreliable,
            "detail": f"outcome/teacher disagreements {disagree}/{both} "
                      f"(alarm above {tc['teacher_alarm']})"}
        checks["confirm"] = {"pass": bool(confirm),
                             "detail": "the user agreed" if confirm else "confirm: true required"}
        return checks

    def _promote_downgrade(self, args: dict) -> dict:
        if (args.get("point_id") is not None or args.get("phase") is not None
                or args.get("plan_phase") is not None):
            return err("invalid_args", "one move per call")
        wf = args.get("workflow")
        if not wf:
            return err("invalid_args", "downgrade needs workflow")
        wrec = self.store.get_workflow(wf)
        if wrec is None:
            return err("unknown_workflow", f"unknown workflow {wf!r}")
        confirm = bool(args.get("confirm", False))
        if args.get("downgrade") is False:
            # disabling is the safe direction: confirm only, no trial needed.
            if not confirm:
                return err("promotion_refused",
                           "confirm: true required to change downgrade",
                           failed=["confirm"],
                           checks={"confirm": {"pass": False,
                                               "detail": "confirm: true required"}})
            wrec["downgrade_allowed"] = False
            self.store.put_workflow(wrec)
            return ok(workflow=wf, downgrade_allowed=False)
        trial = wrec.get("trial") or {}
        checks = {
            "trial": {"pass": trial.get("status") == "passed",
                      "detail": f"trial status={trial.get('status', 'none')}, "
                                "need a passed trial"},
            "confirm": {"pass": confirm,
                        "detail": "the user agreed" if confirm
                                  else "confirm: true required"},
        }
        failed = [k for k, v in checks.items() if not v["pass"]]
        if failed:
            return err("promotion_refused",
                       f"downgrade refused: {', '.join(failed)}",
                       failed=failed, checks=checks)
        wrec["downgrade_allowed"] = True
        wrec["monitor"] = {"enabled_ts": _utcnow(),
                           "trial_id": trial.get("trial_id")}
        self.store.put_workflow(wrec)
        return ok(workflow=wf, downgrade_allowed=True,
                  trial={"trial_id": trial.get("trial_id"),
                         "n": trial.get("n"), "bound": trial.get("bound")})

    def _update_pair_state(self, plan: dict) -> None:
        if plan.get("trial_id"):
            self._update_trial(plan)
        elif plan.get("pair_id"):
            self._update_monitor(plan)

    def _trial_pairs(self, workflow: str, trial_id: str) -> list:
        """(issue_ts, pair_id, expensive, cheap) in issue order."""
        by_pair: dict[str, dict] = {}
        for p in self.store.plans_for_trial(trial_id):
            by_pair.setdefault(p["pair_id"], {})[p["arm"]] = p
        pairs = []
        for pair_id, arms in by_pair.items():
            if "expensive" not in arms or "cheap" not in arms:
                continue
            exp, chp = arms["expensive"], arms["cheap"]
            pairs.append((min(exp["ts"], chp["ts"]), pair_id, exp, chp))
        pairs.sort()
        return pairs

    @staticmethod
    def _resolve_pair(exp: dict, chp: dict, timeout_h: float,
                      now: float) -> tuple[bool, bool] | None:
        """A pair's (expensive_pass, cheap_pass), or None when pending.

        No cherry-picking (§15.8): every issued pair counts. A cheap arm with
        no result past the timeout counts as a cheap failure; a missing
        expensive arm counts the pair as expensive success and cheap failure
        (missing data counts against the cheap arm).
        """
        from datetime import datetime
        e = (exp.get("check_result") or {}).get("pass")
        c = (chp.get("check_result") or {}).get("pass")
        if e is not None and c is not None:
            return bool(e), bool(c)
        issue = min(exp["ts"], chp["ts"])
        try:
            age_h = (now - datetime.fromisoformat(
                issue.replace("Z", "+00:00")).timestamp()) / 3600
        except ValueError:
            age_h = 0.0
        if age_h <= timeout_h:
            return None
        if e is None:
            return True, False
        return bool(e), False

    def _trial_veto(self, chp: dict) -> bool:
        """True when the cheap plan's decided points were all correct or
        unknown: with the expensive arm passing and the cheap arm failing,
        the cheap model can't carry out the decisions. Unknown correctness
        does not exonerate (missing data counts against the cheap arm)."""
        t = self.store.transcript_for_plan(chp["plan_id"])
        rows: dict[str, dict] = {}
        if t is not None:
            for r in self.store.decisions_for_transcript(t["transcript_id"]):
                rows.setdefault(r["point_id"], r)
        for e in chp.get("points", []):
            if e.get("status") != "decided":
                continue
            row = rows.get(e["point_id"])
            if row is None:
                continue  # unknown: counts as correct (fail closed)
            oc = row.get("outcome") or {}
            cc = oc.get("correct_choice")
            if cc is not None:
                if cc != e.get("choice"):
                    return False
            elif oc.get("ok") is False:
                return False
        return True

    def _update_trial(self, plan: dict) -> None:
        from . import stats as S
        wrec = self.store.get_workflow(plan["workflow"])
        if wrec is None:
            return
        trial = wrec.get("trial") or {}
        if trial.get("trial_id") != plan.get("trial_id"):
            return  # stale pair from a superseded trial
        if trial.get("status") != "running":
            return
        pairs = self._trial_pairs(plan["workflow"], trial["trial_id"])
        timeout_h = float(self.cfg["plan"]["pair_timeout_h"])
        now = time.time()
        resolved = []
        for _, _, exp, chp in pairs:
            r = self._resolve_pair(exp, chp, timeout_h, now)
            if r is not None:
                resolved.append((r[0], r[1], chp))
        for e_pass, c_pass, chp in resolved:
            if e_pass and not c_pass and self._trial_veto(chp):
                trial.update(status="failed", reason="veto",
                             n=len(resolved), failed_ts=_utcnow())
                wrec["trial"] = trial
                self.store.put_workflow(wrec)
                return
        if self._trial_check_changed(wrec, trial):
            self._reset_workflow_evidence(
                plan["workflow"], "check script changed mid-trial")
            return
        looks = sorted(self.cfg["plan"]["trial_looks"])
        z = float(self.cfg["plan"]["trial_z"])
        max_drop = float(self.cfg["plan"]["max_quality_drop"])
        evaluated_here = False
        for look in looks:
            if look <= (trial.get("last_look") or 0):
                continue
            if len(pairs) < look:
                break
            # a look at n needs ALL of the first n issued pairs resolved
            # (by result or by timeout): using the first n resolved lets the
            # caller withhold a bad pair and pass (B6).
            first_n = []
            for _, _, exp, chp in pairs[:look]:
                r = self._resolve_pair(exp, chp, timeout_h, now)
                if r is None:
                    break
                first_n.append((r[0], r[1]))
            if len(first_n) < look:
                break
            first = [(e, c, None) for e, c in first_n]
            a = sum(1 for e, c, _ in first if e and c)
            b = sum(1 for e, c, _ in first if c and not e)
            c = sum(1 for e, c, _ in first if e and not c)
            d = sum(1 for e, c, _ in first if not e and not c)
            diff, lower, _ = S.newcombe_paired(a, b, c, d, z)
            trial.update(n=look, bound=lower, diff=diff, last_look=look,
                         counts={"a": a, "b": b, "c": c, "d": d})
            evaluated_here = True
            if lower >= -max_drop:
                trial.update(status="passed", passed_ts=_utcnow())
                wrec["trial"] = trial
                self.store.put_workflow(wrec)
                return
            if look == looks[-1]:
                trial.update(status="failed", reason="bound",
                             failed_ts=_utcnow())
                wrec["trial"] = trial
                self.store.put_workflow(wrec)
                return
        if not evaluated_here:
            trial["n"] = len(resolved)
        wrec["trial"] = trial
        self.store.put_workflow(wrec)

    def _update_monitor(self, plan: dict) -> None:
        from . import stats as S
        wrec = self.store.get_workflow(plan["workflow"])
        if wrec is None or not wrec.get("downgrade_allowed"):
            return
        trial = wrec.get("trial") or {}
        if trial and self._trial_check_changed(wrec, trial):
            self._reset_workflow_evidence(
                plan["workflow"], "check script changed during monitoring")
            return
        enabled_ts = (wrec.get("monitor") or {}).get("enabled_ts") or ""
        cheap_model = wrec.get("cheap_model")
        timeout_h = float(self.cfg["plan"]["pair_timeout_h"])
        now = time.time()
        by_pair: dict[str, dict] = {}
        for p in self.store.plans_for_workflow(plan["workflow"]):
            if not p.get("pair_id") or p.get("trial_id"):
                continue
            if p.get("pilot_forced"):
                continue
            by_pair.setdefault(p["pair_id"], {})[p["arm"]] = p
        pairs = []
        for pair_id, arms in by_pair.items():
            if "expensive" not in arms or "cheap" not in arms:
                continue
            exp, chp = arms["expensive"], arms["cheap"]
            if min(exp["ts"], chp["ts"]) < enabled_ts:
                continue  # issued under an older enablement (B11)
            if chp.get("cheap_model") != cheap_model:
                continue  # scored with an older cheap model (B11)
            r = self._resolve_pair(exp, chp, timeout_h, now)
            if r is None:
                continue  # pending within the timeout
            pairs.append((min(exp["ts"], chp["ts"]), pair_id, r[0], r[1]))
        pairs.sort()
        window = pairs[-self.cfg["plan"]["monitor_window"]:]
        n = len(window)
        if n < self.cfg["plan"]["monitor_min_pairs"]:
            return  # insufficient data: cannot detect drops yet
        a = sum(1 for _, _, e, c in window if e and c)
        b = sum(1 for _, _, e, c in window if c and not e)
        c = sum(1 for _, _, e, c in window if e and not c)
        d = sum(1 for _, _, e, c in window if not e and not c)
        max_drop = float(self.cfg["plan"]["max_quality_drop"])
        diff, _, upper = S.newcombe_paired(a, b, c, d, 1.96)
        if upper < -max_drop:
            wrec["downgrade_allowed"] = False
            monitor = wrec.get("monitor") or {}
            monitor.update(revoked_ts=_utcnow(),
                           reason=f"upper bound {upper:.4f} below -{max_drop}",
                           n=n, upper=upper, diff=diff)
            wrec["monitor"] = monitor
            self.store.put_workflow(wrec)

    # -- do_analyze ------------------------------------------------------
    def do_analyze(self, args: dict) -> dict:
        from . import analyze as AN
        errors = SV.validate("do_analyze", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        if "workflow" not in args and "traces" not in args \
                and "code" not in args and not args.get("gateway"):
            return err("invalid_args",
                       "do_analyze needs workflow, traces, code or gateway")
        out: dict = {"points": [], "rubric": AN.RUBRIC,
                     "candidate_schema": AN.candidate_schema()}
        if args.get("gateway"):
            return self._dm_analyze_gateway(args, out)
        workflow_text = None
        if "workflow" in args:
            text = self._workflow_text(args["workflow"])
            if isinstance(text, dict):  # error dict
                return text
            out["hints"] = AN.find_hints(text)
            workflow_text = text
        trace_calls = None
        if "traces" in args:
            calls = self._trace_calls(args["traces"])
            if isinstance(calls, dict):
                return calls
            points, stats = AN.detect_traces(calls, self.cfg)
            out["points"].extend(points)
            out["trace_stats"] = stats
            trace_calls = calls
        scored_agent = [(cand, AN.score_candidate(cand, self.cfg))
                        for cand in args.get("candidates") or []]
        for _, scored in scored_agent:
            out["points"].append(scored)
        static_cands: list[dict] = []
        if "code" in args:
            from .discovery import static_py as SP
            static_cands = SP.find_static_candidates(args["code"], self.cfg)
        behaviour_res = None
        if trace_calls is not None:
            from .discovery import behaviour as BH
            if BH.has_new_trace_fields(trace_calls):
                behaviour_res = BH.detect_behaviour(trace_calls, self.cfg)
        if "code" in args or behaviour_res is not None:
            from .discovery import merge as MG
            text_cands = MG.text_candidates(out.get("hints", []),
                                            workflow_text or "")
            merged = MG.merge(
                static_cands,
                behaviour_res["candidates"] if behaviour_res else [],
                text_cands if workflow_text is not None else [],
                scored_agent)
            out["discovery"] = {
                "candidates": merged,
                "rejected": behaviour_res["rejected"] if behaviour_res else [],
            }
        return ok(**out)

    def _dm_analyze_gateway(self, args: dict, out: dict) -> dict:
        from . import gwformat as GF
        from . import gwtemplate as GT
        from .analyze import _json_enum_fields, _top_coverage, norm_output
        if args.get("show_samples") and not args.get("confirm"):
            return err("confirm_required",
                       "raw samples need show_samples + confirm (private traces)")
        dk_filter = args.get("discovery_key")
        groups: dict[str, list[dict]] = {}
        # discovery runs on allow-listed (unregistered) traces with raw text
        for t in self.store.unregistered_traces():
            if dk_filter and t["discovery_key"] != dk_filter:
                continue
            groups.setdefault(t["discovery_key"], []).append(t)
        result_groups = []
        for dk in sorted(groups):
            traces = groups[dk]
            items = [{"system": (t["raw_request"] or {}).get("system", ""),
                      "user": (t["raw_request"] or {}).get("user", ""),
                      "answer": t.get("raw_response") or ""} for t in traces]
            clusters = []
            tagged = [{"system": i["system"], "user": i["user"], "_i": j}
                      for j, i in enumerate(items)]
            for cl in GT.cluster_traces(tagged, self.cfg):
                entry: dict = {"n": cl["n"]}
                if cl.get("error"):
                    entry.update({k: cl[k] for k in ("error", "reason", "match_rate")
                                  if k in cl})
                    clusters.append(entry)
                    continue
                template = cl["template"]
                # options from the cluster's answers (§3.2 step 3)
                answers = [items[g["_i"]]["answer"] for g in cl["items"]]
                options: list[str] | None = None
                norms = [norm_output(a) for a in answers]
                coverage, top = _top_coverage(norms)
                if coverage >= 0.90 and len(set(norms)) <= 16:
                    surfaces: dict[str, dict] = {}
                    for a, nv in zip(answers, norms):
                        surfaces.setdefault(nv, {}).setdefault(a.strip()[:120], 0)
                        surfaces[nv][a.strip()[:120]] += 1
                    options = [max(surfaces[v], key=surfaces[v].get) for v in top]
                else:
                    fields = _json_enum_fields(answers)
                    if fields is not None and len(fields) == 1:
                        ((_, values),) = fields.items()
                        if all(isinstance(v, str) for v in values):
                            options = [str(v) for v in values]
                if options is None:
                    entry.update({"template": template,
                                  "template_match_rate": cl.get("match_rate"),
                                  "error": "format_unsupported",
                                  "reason": "answers are not an enum (top-16 cover "
                                            f"{coverage:.0%}, no single JSON enum field)"})
                    clusters.append(entry)
                    continue
                output_format = GF.learn_format(answers, options)
                if output_format is None:
                    entry.update({"template": template,
                                  "template_match_rate": cl.get("match_rate"),
                                  "options": options, "error": "format_unsupported",
                                  "reason": "no learnable output format (need 20+ enum/json answers)"})
                else:
                    entry.update({"template": template,
                                  "template_match_rate": cl.get("match_rate"),
                                  "fingerprint": GT.fingerprint(
                                      traces[0]["route"], traces[0]["model"],
                                      template["system"], template["user_prefix"],
                                      template["user_suffix"]),
                                  "options": options, "output_format": output_format})
                clusters.append(entry)
            group: dict = {"discovery_key": dk, "route": traces[0]["route"],
                           "model": traces[0]["model"], "n_traces": len(traces),
                           "clusters": clusters}
            result_groups.append(group)
        out.update({"gateway": True, "groups": result_groups})
        if args.get("show_samples"):
            samples = []
            for dk in sorted(groups):
                for t in groups[dk][:5 - len(samples)]:
                    samples.append({"system": (t["raw_request"] or {}).get("system", ""),
                                    "user": (t["raw_request"] or {}).get("user", ""),
                                    "answer": t.get("raw_response")})
                    if len(samples) >= 5:
                        break
                if len(samples) >= 5:
                    break
            out["samples"] = samples
        return ok(**out)

    @staticmethod
    def _workflow_text(workflow: dict):
        if workflow.get("text") is not None:
            return workflow["text"]
        path = workflow.get("path")
        if not path:
            return err("invalid_args", "workflow needs text or path")
        try:
            size = os.path.getsize(path)
            if size > 2_000_000:
                return err("workflow_too_large", f"{path} is {size} bytes (cap 2 MB)")
            with open(path) as f:
                return f.read()
        except OSError as e:
            return err("workflow_not_found", f"cannot read {path}: {e}")

    @staticmethod
    def _trace_calls(traces: dict):
        import json as _json
        try:
            calls = []
            with open(traces["path"]) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        calls.append(_json.loads(line))
            return calls
        except OSError as e:
            return err("traces_not_found", f"cannot read {traces['path']}: {e}")
        except ValueError as e:
            return err("invalid_args", f"bad JSONL in {traces['path']}: {e}")

    # -- do_plan_run / do_plan_result --------------------------------------
    def _purge_plan_content(self) -> None:
        """Plan input/output share the run-content lifecycle (B13)."""
        cutoff = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(
            time.time() - self.cfg["extract"]["transcript_retention_days"] * 86400))
        self.store.purge_plans(cutoff)

    def do_plan_run(self, args: dict) -> dict:
        errors = SV.validate("do_plan_run", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        self._purge_plan_content()
        wf = args["workflow"]
        points = [p for p in self.store.points_in_registration_order(wf)
                  if (p.get("source") or "do_decide") in ("do_decide", "dm_decide")]
        if not points:
            return err("unknown_workflow",
                       f"no decision points registered for workflow {wf!r}")
        wrec = self.store.get_workflow(wf)
        run_model = args.get("run_model") or (wrec or {}).get("run_model") or ""
        cheap_model = args.get("cheap_model") or (wrec or {}).get("cheap_model") or ""
        if not run_model or not cheap_model:
            return err("invalid_args", "run_model and cheap_model are required "
                       "(pass them or register the workflow record)")
        if wrec is not None:
            # the trial passed with the registered models; serving or
            # trialling another cheap model needs re-registration (which
            # resets the trial, §15.8) (B7).
            for key in ("run_model", "cheap_model"):
                if (args.get(key) is not None and args[key] != wrec.get(key)):
                    return err("invalid_args",
                               f"{key} {args[key]!r} differs from the workflow "
                               f"record ({wrec.get(key)!r}): re-register to "
                               "change it (resets the trial)")
        pilot = bool(args.get("pilot_force"))
        if pilot and not self.cfg["plan"]["allow_pilot_force"]:
            return err("pilot_force_not_allowed",
                       "plan.allow_pilot_force is false "
                       "(enable only in pilot configs)")
        trial = bool(args.get("trial", False))
        if trial and pilot:
            return err("invalid_args", "trial and pilot_force do not combine")
        if args.get("pilot_mode") is not None and not pilot:
            return err("invalid_args", "pilot_mode needs pilot_force: true")
        task_input = args.get("input", "")
        context = args.get("context", {})
        entries: list[dict] = []
        warnings: list[str] = []
        for p in points:
            e, w = self._plan_predict(p, task_input,
                                      context.get(p["point_id"]), pilot)
            entries.append(e)
            warnings.extend(w)
        by_id = {e["point_id"]: e for e in entries}
        requires = (wrec or {}).get("downgrade_requires", [])
        requires_decided = (
            bool(requires) and all(by_id[r]["status"] == "decided"
                                   for r in requires))
        if trial:
            if wrec is None:
                return err("not_downgrade_eligible",
                           "trial needs the workflow record "
                           "(downgrade_requires + check)")
            if not requires_decided:
                return err("not_downgrade_eligible",
                           "trial needs every downgrade_requires point decided "
                           "at plan serve")
            return self._trial_bundle(args, wrec, entries, points, run_model,
                                      cheap_model, task_input, context)
        forced = args.get("pilot_mode") if pilot else None
        if forced == "downgrade":
            unpredicted = [r for r in requires if not by_id[r]["predicted"]]
            if unpredicted:
                return err("not_downgrade_eligible",
                           "pilot downgrade needs every downgrade_requires "
                           f"point predicted, missing {unpredicted}")
            would = "downgrade"
        elif forced == "inject":
            would = ("inject" if any(e["status"] == "decided" for e in entries)
                     else "none")
        elif wrec and wrec.get("downgrade_allowed") and requires_decided:
            would = "downgrade"
        elif any(e["status"] == "decided" for e in entries):
            would = "inject"
        else:
            would = "none"
        if would == "downgrade" and not pilot and self.rng.random() < float(
                self.cfg["plan"]["downgrade_audit_rate"]):
            return self._monitor_bundle(args, entries, points, run_model,
                                        cheap_model, task_input, context)
        decided_pre = [e for e in entries if e["status"] == "decided"]
        audit = (would in ("inject", "downgrade") and not pilot
                 and self.rng.random() < float(self.cfg["plan"]["audit_rate"]))
        would_have: list[dict] = []
        if audit:
            for e in entries:
                e["status"] = "open"
            recommendation, model, addendum = "none", run_model, None
            would_have = list(decided_pre)
        elif would == "none":
            recommendation, model, addendum = "none", run_model, None
            would_have = [e for e in entries if e["predicted"]]
        else:
            recommendation = would
            model = cheap_model if would == "downgrade" else run_model
            addendum = _addendum(decided_pre)
        plan_id = f"pl_{uuid.uuid4().hex[:8]}"
        record = {"plan_id": plan_id, "ts": _utcnow(), "workflow": wf,
                  "task_ref": args.get("task_ref", ""), "input": task_input,
                  "context": context,
                  "context_source": args.get("context_source", "agent"),
                  "run_model": run_model, "cheap_model": cheap_model,
                  "recommendation": recommendation, "model": model,
                  "addendum": addendum, "points": entries,
                  "would_have": ([self._would_entry(e) for e in would_have]
                                 if would_have else None),
                  "audit": audit, "pilot_forced": pilot, "conflicts": [],
                  "point_versions": {p["point_id"]: p.get("point_version", 1)
                                     for p in points}}
        self.store.put_plan(record)
        out = ok(plan_id=plan_id, recommendation=recommendation, model=model,
                 audit=audit,
                 points=[self._public_entry(e) for e in entries],
                 context_source=record["context_source"], pilot_forced=pilot)
        if addendum is not None:
            out["prompt_addendum"] = addendum
        if would_have:
            out["would_have"] = [self._would_entry(e) for e in would_have]
        if warnings:
            out["warnings"] = warnings
        return out

    def _plan_predict(self, point: dict, task_input: str, pctx, pilot: bool
                      ) -> tuple[dict, list[str]]:
        """Step 1-2 of §15.5 for one point (pre-audit statuses)."""
        entry: dict = {"point_id": point["point_id"],
                       "phase": point.get("plan_phase", "collect"),
                       "status": "open", "predicted": False}
        warns: list[str] = []
        validated, missing, type_errors = R.validate_context(point, pctx or {})
        allowed, hits, _ = RU.apply_rules(point, validated)
        if len(allowed) == 1:
            entry.update(choice=allowed[0], confidence=1.0,
                         rule=hits[-1] if hits else None, predicted=True)
            if self._plan_decided_hosted(point, entry, pilot, via_rule=True):
                entry["status"] = "decided"
            return entry, warns
        if missing or type_errors:
            return entry, warns
        if point.get("irreversible") and not self.cfg["serve"]["allow_auto_irreversible"]:
            return entry, warns
        return self._plan_predict_hosted(point, task_input, validated,
                                         allowed, entry, warns, pilot)

    # -- hosted plan path (build §9, G3) -----------------------------------
    def _plan_decider_for(self, point: dict):
        """Plan decider: tier + privacy only (plan_phase gates the call, not
        the point's decide phase)."""
        if point.get("private_model_only"):
            return self.deciders.get("private")
        if (point.get("tier") or "hosted_jev") == "private":
            return self.deciders.get("private")
        if point.get("data_class", "private") == "private":
            ro = point.get("remote_ok", [])
            allowed = True if ro is True else "hosted_jev" in (ro or [])
            if not allowed:
                return None
        return self.deciders.get("hosted_jev")

    def _plan_at_start_rate(self, point: dict
                            ) -> tuple[float | None, int, int]:
        """(at-start rate, n_at_start, n) over clean labelled rows (build §9).

        D: injected and pilot-forced runs parrot the addendum (B12), and
        extracted rows count only while the §15.3 gate passes, so none of
        them may inflate eligibility. Unknown at_start is not at_start.
        P3's eligibility_cutoffs reads this live from the store.
        """
        from . import train as TR
        labeled = L.labeled_rows(
            self.store.decisions_for_point(point["point_id"]), point,
            self.cfg["train"]["teacher_only_weight"],
            self.cfg["extract"]["weight"])
        gate: dict | None = None
        plan_flags = self.store.transcript_plan_flags(point["workflow"])
        eval_only = self.store.eval_only_transcript_ids(point["workflow"])
        clean = []
        for r, _, src, _ in labeled:
            if src == "extracted":
                if gate is None:
                    gate = TR.extraction_gate(self.cfg, self.store, point)
                if not gate["passed"]:
                    continue
            if (r.get("transcript_id") or "") in eval_only:
                continue  # P2 parity: eval-only runs never count
            f = plan_flags.get(r.get("transcript_id") or "")
            if f and (f["injected"] or f["pilot"]):
                continue
            clean.append(r)
        if not clean:
            return None, 0, 0
        n_at_start = sum(1 for r in clean if r.get("at_start"))
        return n_at_start / len(clean), n_at_start, len(clean)

    def _plan_eligible_hosted(self, point: dict) -> bool:
        """plan_eligible without training: the at-start share of labelled
        rows is at least plan.min_at_start (build §9)."""
        rate, _, _ = self._plan_at_start_rate(point)
        return rate is not None and rate >= self.cfg["plan"]["min_at_start"]

    def _plan_predict_hosted(self, point: dict, task_input: str,
                             validated: dict, allowed: list, entry: dict,
                             warns: list, pilot: bool
                             ) -> tuple[dict, list[str]]:
        """Plan prediction on hosted Jev (build §9): policy + task input +
        at-start fields; no plan training."""
        if point.get("plan_phase", "collect") in ("collect", "paused") \
                and not pilot:
            return entry, warns  # no paid calls before plan shadow
        if not self._plan_eligible_hosted(point):
            return entry, warns
        decider = self._plan_decider_for(point)
        if decider is None:
            return entry, warns
        try:
            req = self._decide_request(
                point, validated,
                task_input=task_input or None, state_text=None)
            res = decider.decide([req])[0]
        except DeciderError as e:
            # C3: the plan path surfaces hosted failures like do_decide
            # does, so prereg can tell no key / no budget / down apart
            # instead of marking every ticket silently unpredicted.
            tag = "jev_unavailable"
            if e.code == "remote_not_allowed":
                # R2-10: unlike do_decide's collect-like silence, a plan
                # row blocked by privacy must warn so P3 counts it as a
                # failure instead of silently unpredicted.
                warns.append(f"{point['point_id']}: remote_not_allowed")
                return entry, warns
            record_decider_error(self.store, decider, point["point_id"], e)
            tag = {"budget_exceeded": "budget_exceeded",
                   "no_key": "no_hosted_jev_key"}.get(e.code, tag)
            warns.append(f"{point['point_id']}: {tag}")
            return entry, warns
        T = 1.0
        fit = self._plan_fit_for(point)
        if fit is not None:
            T = float(fit["temperature"])
        masked = HF.mask_temper(res.probs, allowed, T)
        top = S.argmax(masked, allowed)
        entry.update(choice=top, confidence=max(masked.values())
                     if masked else 0.0, predicted=True,
                     probabilities=dict(res.probs), decider=res.decider,
                     hosted_cost_usd=res.cost_usd,
                     allowed_options=list(allowed))
        entry["teacher_only"] = top in (
            (fit or {}).get("teacher_only_options") or [])
        if self._plan_decided_hosted(point, entry, pilot, via_rule=False):
            entry["status"] = "decided"
        return entry, warns

    def _plan_decided_hosted(self, point: dict, entry: dict, pilot: bool,
                             via_rule: bool) -> bool:
        """Plan serve gating on the plan fit (T = 1 before a fit; pilot
        injects the argmax with no threshold)."""
        if not pilot and point.get("plan_phase", "collect") != "serve":
            return False
        if via_rule:
            return True
        if self._effective_tier(point) == "private":
            # GP7: private-tier plan decisions stay open until a private
            # plan fit has a threshold; pilot_force never bypasses this.
            fit = self._private_plan_fit(point)
            if fit is None or fit.get("threshold") is None:
                return False
            if pilot:
                return True
            pt = fit["threshold"]
            if (entry.get("confidence") or 0.0) < pt:
                return False
            if entry.get("choice") in (fit.get("teacher_only_options")
                                       or []):
                return False
            return True
        if not pilot:
            fit = self._hosted_fit(point, "plan")
            pt = (fit or {}).get("threshold")
            if pt is None or (entry.get("confidence") or 0.0) < pt:
                return False
            if entry.get("choice") in ((fit or {}).get(
                    "teacher_only_options") or []):
                return False
        return True

    @staticmethod
    def _public_entry(e: dict) -> dict:
        out = {"point_id": e["point_id"], "status": e["status"]}
        for k in ("choice", "confidence", "rule"):
            if e.get(k) is not None:
                out[k] = e[k]
        if e.get("predicted"):
            out["predicted"] = True
        if e.get("teacher_only"):
            out["teacher_only"] = True
        return out

    @staticmethod
    def _would_entry(e: dict) -> dict:
        out = {"point_id": e["point_id"], "choice": e["choice"]}
        for k in ("confidence", "rule"):
            if e.get(k) is not None:
                out[k] = e[k]
        return out

    def _store_arm_plan(self, args: dict, entries: list[dict], points: list[dict],
                        run_model: str, cheap_model: str, task_input: str,
                        context: dict, recommendation: str, model: str,
                        addendum: str | None, would_have: list[dict] | None,
                        trial_id: str | None, pair_id: str,
                        arm: str) -> dict:
        plan_id = f"pl_{uuid.uuid4().hex[:8]}"
        record = {"plan_id": plan_id, "ts": _utcnow(), "workflow": args["workflow"],
                  "task_ref": args.get("task_ref", ""), "input": task_input,
                  "context": context,
                  "context_source": args.get("context_source", "agent"),
                  "run_model": run_model, "cheap_model": cheap_model,
                  "recommendation": recommendation, "model": model,
                  "addendum": addendum, "points": [dict(e) for e in entries],
                  "would_have": ([self._would_entry(e) for e in would_have]
                                 if would_have else None),
                  "audit": False, "trial_id": trial_id, "pair_id": pair_id,
                  "arm": arm, "pilot_forced": False, "conflicts": [],
                  "point_versions": {p["point_id"]: p.get("point_version", 1)
                                     for p in points}}
        self.store.put_plan(record)
        return record

    def _public_plan(self, record: dict) -> dict:
        out = ok(plan_id=record["plan_id"],
                 recommendation=record["recommendation"], model=record["model"],
                 audit=record["audit"],
                 points=[self._public_entry(e) for e in record["points"]],
                 context_source=record["context_source"], pilot_forced=False)
        if record.get("addendum") is not None:
            out["prompt_addendum"] = record["addendum"]
        if record.get("would_have"):
            out["would_have"] = record["would_have"]
        return out

    @staticmethod
    def _script_hash(script: str) -> str | None:
        """sha256 of a check script's bytes, None when unreadable."""
        import hashlib
        try:
            with open(script, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()
        except OSError:
            return None

    def _trial_check_changed(self, wrec: dict, trial: dict) -> bool:
        """True when the check script's content differs from what the trial
        recorded (B14: editing the script resets the trial)."""
        recorded = (trial.get("check") or {}).get("hash")
        if recorded is None:
            return False  # legacy trial: no baseline recorded
        if wrec.get("check_type") != "deterministic":
            return False
        return self._script_hash(wrec.get("check_script", "")) != recorded

    def _reset_workflow_evidence(self, workflow: str, reason: str) -> None:
        """Clear trial + monitor basis after a point-version bump or a check
        edit (§15.8): the old evidence no longer describes the workflow.
        Downgrade drops (fail closed); a new trial may start."""
        wrec = self.store.get_workflow(workflow)
        if wrec is None:
            return
        trial = wrec.get("trial") or {}
        wrec["trial"] = {"status": "reset", "reason": reason,
                         "prev_trial_id": trial.get("trial_id"),
                         "reset_ts": _utcnow()}
        wrec["downgrade_allowed"] = False
        wrec["monitor"] = {"enabled_ts": _utcnow(), "reset_reason": reason}
        self.store.put_workflow(wrec)

    def _trial_bundle(self, args: dict, wrec: dict, entries: list[dict],
                      points: list[dict], run_model: str, cheap_model: str,
                      task_input: str, context: dict) -> dict:
        trial = wrec.get("trial") or {}
        if trial and self._trial_check_changed(wrec, trial):
            self._reset_workflow_evidence(
                wrec["workflow"], "check script changed")
            wrec = self.store.get_workflow(wrec["workflow"]) or wrec
            trial = wrec.get("trial") or {}
        status = trial.get("status")
        if status == "running":
            if (trial.get("cheap_model") != cheap_model
                    or trial.get("run_model") != run_model):
                return err("invalid_args",
                           "trial already running with different models")
        elif status in (None, "reset") or not trial:
            check_hash = (self._script_hash(wrec["check_script"])
                          if wrec["check_type"] == "deterministic" else None)
            if wrec["check_type"] == "deterministic" and check_hash is None:
                return err("check_missing",
                           "deterministic trial needs a readable check script")
            trial = {"trial_id": f"tt_{uuid.uuid4().hex[:8]}", "status": "running",
                     "run_model": run_model, "cheap_model": cheap_model,
                     "check": {"type": wrec["check_type"],
                               "script": wrec["check_script"],
                               "hash": check_hash},
                     "pairs_issued": 0, "last_look": 0, "n": 0, "bound": None,
                     "started_ts": _utcnow()}
        else:
            # passed / failed: a finished trial cannot be restarted for
            # another shot (B8). Reset paths: re-register with changed
            # models/check/requires, or bump a required point's version.
            return err("trial_finished",
                       f"trial {trial.get('trial_id')} already {status}: "
                       "re-register with changed models, check or requires "
                       "(resets the trial) to start a new one")
        trial["pairs_issued"] += 1
        wrec["trial"] = trial
        self.store.put_workflow(wrec)
        pair_id = f"pr_{uuid.uuid4().hex[:8]}"
        decided = [e for e in entries if e["status"] == "decided"]
        # the expensive arm runs without the addendum: its entries are
        # would-have-been (open), not injected (decided). Storing them as
        # decided dilutes followed_rate (B14).
        exp_entries = [{**e, "status": "open"} for e in entries]
        exp = self._store_arm_plan(args, exp_entries, points, run_model,
                                   cheap_model, task_input, context, "none",
                                   run_model, None, decided, trial["trial_id"],
                                   pair_id, "expensive")
        chp = self._store_arm_plan(args, entries, points, run_model, cheap_model,
                                   task_input, context, "downgrade", cheap_model,
                                   _addendum(decided), None, trial["trial_id"],
                                   pair_id, "cheap")
        return ok(trial_id=trial["trial_id"], pair_id=pair_id,
                  expensive=self._public_plan(exp),
                  cheap=self._public_plan(chp))

    def _monitor_bundle(self, args: dict, entries: list[dict], points: list[dict],
                        run_model: str, cheap_model: str, task_input: str,
                        context: dict) -> dict:
        pair_id = f"pr_{uuid.uuid4().hex[:8]}"
        decided = [e for e in entries if e["status"] == "decided"]
        exp_entries = [{**e, "status": "open"} for e in entries]
        exp = self._store_arm_plan(args, exp_entries, points, run_model,
                                   cheap_model, task_input, context, "none",
                                   run_model, None, decided, None, pair_id,
                                   "expensive")
        chp = self._store_arm_plan(args, entries, points, run_model, cheap_model,
                                   task_input, context, "downgrade", cheap_model,
                                   _addendum(decided), None, None, pair_id, "cheap")
        return ok(pair_id=pair_id, monitor=True,
                  expensive=self._public_plan(exp),
                  cheap=self._public_plan(chp))

    def do_plan_result(self, args: dict) -> dict:
        errors = SV.validate("do_plan_result", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        self._purge_plan_content()
        plan = self.store.get_plan(args["plan_id"])
        if plan is None:
            return err("unknown_plan", f"unknown plan_id {args['plan_id']!r}")
        for k in ("trial_id", "pair_id", "arm"):
            if args.get(k) is not None and args[k] != plan.get(k):
                return err("invalid_args",
                           f"{k} {args[k]!r} does not match the plan")
        out: dict = {"plan_id": plan["plan_id"]}
        if args.get("cost") is not None:
            if not isinstance(args["cost"], dict):
                return err("invalid_args", "cost must be an object")
            plan["cost"] = args["cost"]
        if args.get("conflict") is not None:
            cp = self.store.get_point(args["conflict"])
            if cp is None or cp.get("workflow") != plan["workflow"]:
                return err("invalid_args",
                           f"conflict: unknown point {args['conflict']!r} "
                           f"for workflow {plan['workflow']!r}")
            if args["conflict"] not in plan["conflicts"]:
                plan["conflicts"].append(args["conflict"])
        if args.get("transcript") is not None:
            from .adapters import validate_transcript
            t = args["transcript"]
            verr = validate_transcript(t) if isinstance(t, dict) else ["not an object"]
            if verr:
                return err("invalid_transcript", "; ".join(verr[:3]))
            if t["workflow"] != plan["workflow"]:
                return err("invalid_args", "transcript workflow does not match "
                           "the plan")
            if self.store.transcript_by_run(t["run_id"]) is not None:
                return err("duplicate_run",
                           f"run_id {t['run_id']!r} already stored")
            transcript_id = f"tr_{uuid.uuid4().hex[:8]}"
            self.store.put_transcript({
                "transcript_id": transcript_id, "ts": _utcnow(),
                "run_id": t["run_id"], "workflow": t["workflow"],
                "model": t.get("model", ""),
                "data_class": t.get("data_class", "private"),
                "input": t.get("input", ""), "events": t.get("events", []),
                "final_output": t.get("final_output", ""),
                "cost": t.get("cost"), "run_outcome": t.get("run_outcome"),
                "plan_id": plan["plan_id"],
                "eval_only": bool(args.get("eval_only"))})
            for hit in _conflict_hits(t):
                if hit in plan["conflicts"]:
                    continue
                cp = self.store.get_point(hit)
                if cp is not None and cp.get("workflow") == plan["workflow"]:
                    plan["conflicts"].append(hit)
            out["transcript_id"] = transcript_id
        if args.get("final_output") is not None:
            plan["final_output"] = args["final_output"]
        wrec = self.store.get_workflow(plan["workflow"])
        if (wrec and wrec["check_type"] == "deterministic"
                and plan.get("final_output") and not plan.get("check_result")):
            plan["check_result"] = self._run_check(wrec, plan)
            out["check_result"] = plan["check_result"]
        elif plan.get("check_result") is not None:
            out["check_result"] = plan["check_result"]
        self.store.put_plan(plan)
        out["conflicts"] = plan["conflicts"]
        if plan.get("pair_id"):
            self._update_pair_state(plan)
        elif plan.get("conflicts"):
            for pid in plan["conflicts"]:
                point = self.store.get_point(pid)
                if point is not None and point.get("workflow") == plan["workflow"]:
                    self._plan_drift_check(point["point_id"])
        response = ok(**out)
        return response

    # Wrapper for .py check scripts: blocks pure-Python socket use. This is
    # not a sandbox (subprocesses and C extensions bypass it); full network
    # isolation needs OS-level sandboxing, out of scope for v1.
    _CHECK_WRAPPER = (
        "import runpy, socket, sys; "
        "_blocked = lambda *a, **k: (_ for _ in ()).throw("
        "OSError('network disabled for check scripts')); "
        "socket.socket = _blocked; socket.create_connection = _blocked; "
        "sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')")

    def _run_check(self, wrec: dict, plan: dict) -> dict:
        """Run the workflow's deterministic check script (server-side, §15.8).

        Protocol: argv [script], stdin JSON {input, final_output, task_ref},
        stdout JSON {"pass": bool, "reasons"?}. Timeout plan.check_timeout_s
        (default 30 s). Anything unparseable, slow or crashing counts as a
        fail. .py scripts run with socket creation disabled (partial
        isolation, see _CHECK_WRAPPER); scripts are local files only.
        """
        import subprocess
        import sys
        from pathlib import Path
        script = wrec["check_script"]
        script_hash = self._script_hash(script)
        payload = {"input": plan.get("input", ""),
                   "final_output": plan.get("final_output", ""),
                   "task_ref": plan.get("task_ref", "")}
        p = Path(script)
        if script.endswith(".py"):
            cmd = [sys.executable, "-c", self._CHECK_WRAPPER, script]
        elif p.is_file() and os.access(script, os.X_OK):
            cmd = [script]
        else:
            return {"pass": False, "source": "script",
                    "script_hash": script_hash,
                    "error": f"check script not runnable: {script!r}"}
        timeout = float(self.cfg["plan"]["check_timeout_s"])
        try:
            proc = subprocess.run(cmd, input=json.dumps(payload),
                                  capture_output=True, text=True,
                                  timeout=timeout, cwd=str(p.parent))
        except subprocess.TimeoutExpired:
            return {"pass": False, "source": "script",
                    "script_hash": script_hash,
                    "error": f"timeout ({timeout:g} s)"}
        except OSError as e:
            return {"pass": False, "source": "script",
                    "script_hash": script_hash, "error": str(e)}
        stdout = (proc.stdout or "").strip()
        try:
            obj = json.loads(stdout)
            if isinstance(obj, dict) and isinstance(obj.get("pass"), bool):
                result: dict = {"pass": obj["pass"], "source": "script",
                                "script_hash": script_hash}
                if isinstance(obj.get("reasons"), list):
                    result["reasons"] = obj["reasons"][:10]
                return result
        except ValueError:
            pass
        first = stdout.splitlines()[0].strip().lower() if stdout else ""
        if first in ("pass", "true", "yes", "1"):
            return {"pass": True, "source": "script",
                    "script_hash": script_hash}
        if first in ("fail", "false", "no", "0"):
            return {"pass": False, "source": "script",
                    "script_hash": script_hash}
        return {"pass": False, "source": "script",
                "script_hash": script_hash,
                "error": f"unparseable check output: {stdout[:120]!r}"}

    def _update_followed(self, plan_id: str, transcript_id: str) -> float | None:
        """followed_rate from a later extraction (§15.5): the share of injected
        decisions that the run's extracted decision matches."""
        plan = self.store.get_plan(plan_id)
        if plan is None:
            return None
        decided = [e for e in plan.get("points", []) if e.get("status") == "decided"]
        if not decided and plan.get("audit"):
            # audit plans store entries as open: compare what would have
            # been injected.
            decided = [{"point_id": e["point_id"], "choice": e["choice"]}
                       for e in plan.get("would_have", [])]
        if not decided:
            return None
        rows: dict[str, dict] = {}
        for r in self.store.decisions_for_transcript(transcript_id):
            rows.setdefault(r["point_id"], r)
        matched = sum(1 for e in decided
                      if rows.get(e["point_id"], {}).get("extracted_choice")
                      == e.get("choice"))
        plan["followed_rate"] = matched / len(decided)
        plan["followed_n"] = len(decided)
        self.store.put_plan(plan)
        if plan.get("audit"):
            for e in decided:
                self._plan_drift_check(e["point_id"])
        return plan["followed_rate"]

    def _comparison_label(self, row: dict) -> tuple[str | None, str | None]:
        """(label, kind) for plan evidence: correctness when available,
        otherwise the live teacher label, otherwise the extracted choice."""
        outcome = row.get("outcome") or {}
        cc = outcome.get("correct_choice")
        if cc is not None:
            return cc, "correctness"
        if outcome.get("ok") is True:
            choice = (row.get("final_choice") or row.get("teacher_choice")
                      or row.get("extracted_choice"))
            return (choice, "correctness") if choice else (None, None)
        if outcome.get("ok") is False:
            return None, None  # known wrong, right answer unknown: unscorable
        if row.get("teacher_choice") is not None:
            return row["teacher_choice"], "teacher"
        if row.get("extracted_choice") is not None:
            return row["extracted_choice"], "extracted"
        return None, None

    def _plan_comparisons(self, plan: dict) -> list[dict]:
        """Per-point prediction-vs-label comparisons for one plan, from its
        linked transcript's extracted rows (computed lazily: outcomes landing
        later are picked up)."""
        out = []
        t = self.store.transcript_for_plan(plan["plan_id"])
        rows: dict[str, dict] = {}
        if t is not None:
            for r in self.store.decisions_for_transcript(t["transcript_id"]):
                rows.setdefault(r["point_id"], r)
        versions = plan.get("point_versions") or {}
        for e in plan.get("points", []):
            pid = e.get("point_id")
            if e.get("choice") is None:
                continue
            point = self.store.get_point(pid)
            if point is None:
                continue
            if versions.get(pid, point.get("point_version", 1)) != point.get(
                    "point_version", 1):
                continue  # stale version: plan evidence cleared (§15.4)
            row = rows.get(pid)
            if row is None:
                continue  # not extracted yet
            label, kind = self._comparison_label(row)
            if label is None:
                continue
            out.append({"point_id": pid, "predicted": e["choice"],
                        "confidence": e.get("confidence"),
                        "phase": e.get("phase"), "label": label, "kind": kind,
                        "match": e["choice"] == label,
                        "probabilities": e.get("probabilities"),
                        "decider": e.get("decider"),
                        "allowed_options": e.get("allowed_options")})
        return out

    def _plan_point_failed(self, plan: dict, point_id: str) -> bool:
        """A conflict naming the point, or a decision-level outcome failure
        without a known right answer, on one plan's run."""
        if point_id in (plan.get("conflicts") or []):
            return True
        t = self.store.transcript_for_plan(plan["plan_id"])
        if t is not None:
            for r in self.store.decisions_for_transcript(t["transcript_id"]):
                if r["point_id"] != point_id:
                    continue
                oc = r.get("outcome") or {}
                if oc.get("ok") is False and not oc.get("correct_choice"):
                    return True
        return False

    def _plan_drift_check(self, point_id: str) -> bool:
        """Demote plan serve -> shadow on audit evidence (§15.7)."""
        from . import stats as S
        point = self.store.get_point(point_id)
        if point is None or point.get("plan_phase") != "serve":
            return False
        target = self.cfg["plan"]["target_error"]
        window = self.cfg["serve"]["drift_window"]
        # C6: the window covers the current plan epoch only (R2-1); stale
        # versions are already skipped inside _plan_comparisons.
        epoch = Service._plan_epoch(point)
        compares: list[tuple[str, str, bool]] = []  # (ts, plan_id, match)
        for plan in self.store.plans_for_workflow(point["workflow"]):
            if plan.get("pilot_forced") or not plan.get("audit"):
                continue
            if plan.get("trial_id") or plan.get("pair_id"):
                continue
            if plan["ts"] < epoch:
                continue
            for c in self._plan_comparisons(plan):
                if c["point_id"] == point_id and c["phase"] == "serve":
                    compares.append((plan["ts"], plan["plan_id"], c["match"]))
        if len(compares) < window:
            return False
        win = compares[-window:]
        floor = win[0][0]
        mismatched = {pid for _, pid, m in win if not m}
        errs = len(mismatched)
        # conflicts and decision-level outcome failures add to the errors
        # (§15.7). They live on injected runs (audit plans carry no
        # addendum, so nothing conflicts with them), hence the scan over
        # non-pilot, non-pair plans in the window's span. Each plan counts
        # once; plans already counted as mismatches are not double-counted.
        for plan in self.store.plans_for_workflow(point["workflow"]):
            if plan.get("pilot_forced"):
                continue
            if plan.get("trial_id") or plan.get("pair_id"):
                continue
            if plan["ts"] < floor:
                continue
            if plan["plan_id"] in mismatched:
                continue
            if self._plan_point_failed(plan, point_id):
                errs += 1
        # saturation: more error events than window tasks means every task
        # errored. Wilson needs errs <= n; without the cap a skewed window
        # crashes it (ValueError).
        n = len(win)
        errs = min(errs, n)
        if errs / n > 2 * target and S.wilson_lower(errs, n) > target:
            point["plan_phase"] = "shadow"
            # C6: a demotion starts a new plan epoch and re-arms the plan
            # fit trigger (R2-1: the decide epoch is untouched).
            point["plan_fit_epoch_start"] = _utcnow()
            self.store.deactivate_hosted_fit(point["point_id"], "plan")
            point.setdefault("events", []).append(
                {"ts": _utcnow(), "event": "plan_drift_demotion",
                 "errors": errs, "n": n, "target": target})
            self.store.put_point(point)
            return True
        return False

    # -- do_report -------------------------------------------------------
    def _private_report_block(self, point: dict) -> dict:
        """do_report's private-tier block (F9). Model names print as the
        tail only (GP9); the full ref never leaves the card."""
        import statistics

        from .trainer_log import redact_account

        pid = point["point_id"]
        month = time.strftime("%Y-%m", time.gmtime())
        spend = self.store.spend_get(month, "fireworks")
        budget = float(self.cfg["fireworks"].get("monthly_budget_usd") or 0)
        ver = point.get("point_version", 1)
        rows = [r for r in self.store.decisions_for_point(pid)
                if r.get("decider") == "private"
                and r.get("point_version") == ver]
        lats = sorted(r["jev_seconds"] for r in rows
                      if isinstance(r.get("jev_seconds"), (int, float)))

        def _model_block(card: dict) -> dict | None:
            if not card:
                return None
            metrics = card.get("metrics", {}) or {}
            ref = card.get("model_ref") or card.get("sampler") or None
            return {
                "model": redact_account(ref) if ref else None,
                "temperature": card.get("temperature"),
                "job_id": card.get("job_id"),
                "accuracy": card.get("accuracy"),
                "n_holdout": metrics.get("n_holdout"),
                "threshold": card.get("threshold")}

        model = self.store.active_model(pid, "private")
        card = {}
        if model is not None and model.get("point_version") == ver:
            card = model.get("card", {})
        pending = self._pending_private_model(point)
        blocked = None
        if (card or pending) and not self._hosted_tier(point) \
                and not self.cfg["fireworks"].get("allow_session_serving",
                                                 False):
            # GP11: a trained private-tier point with the serving gate
            # closed stays collect; the report says so either way.
            blocked = "serving_not_allowed"
        return {
            "blocked": blocked,
            "model": _model_block(card),
            "pending_model": _model_block((pending or {}).get("card", {})),
            "calls": len(rows),
            "median_latency_s": (statistics.median(lats) if lats else None),
            "calls_month": spend["calls"],
            "spend_month_usd": spend["usd"], "budget_usd": budget,
            "budget_warning_80": bool(budget and spend["usd"] >= 0.8 * budget),
            "warming": self.store.counter_get("private_warming", pid),
            "errors": {
                "timeout": self.store.counter_get(
                    "private_error_timeout", pid),
                "bad_response": self.store.counter_get(
                    "private_error_bad_response", pid),
                "unavailable": self.store.counter_get(
                    "private_error_unavailable", pid)},
            "measured": False,  # unmeasured until L3 reconciles the bill
        }

    def _fireworks_report_block(self) -> dict:
        """do_report's Fireworks-global block (F9): the live job, the
        worker sessions, the cookbook pin and the serving gate."""
        from datetime import datetime, timezone

        from .fireworks_trainer import FireworksTrainer

        pin = getattr(self.trainer, "pin_status", None)
        if callable(pin):
            pin = self.trainer.pin_status()
        else:
            pin = FireworksTrainer(
                self.cfg, self.store.root / "jobs").pin_status()
        fw = self.cfg["fireworks"]
        try:
            budget = float(fw.get("monthly_budget_usd") or 0)
        except (TypeError, ValueError):
            budget = 0.0
        serving = bool(fw.get("allow_session_serving", False)) \
            and budget > 0
        live = self.store.jobs_running() + self.store.jobs_launching()
        cur = None
        if live:
            job = live[0]
            prog: dict = {}
            if self.trainer is not None:
                try:
                    prog = self.trainer.progress(job["job_id"]) or {}
                except Exception:
                    prog = {}
            try:
                created = datetime.fromisoformat(job.get("created_at", ""))
                elapsed = (datetime.now(timezone.utc) - created
                           ).total_seconds()
            except (TypeError, ValueError):
                elapsed = None
            cur = {"job_id": job["job_id"], "kind": job["kind"],
                   "state": job["status"], "elapsed_s": elapsed,
                   "step": prog.get("step"), "loss": prog.get("loss"),
                   "eval_accuracy": prog.get("eval_accuracy")}
        dec = self.deciders.get("private")
        if dec is None:
            worker = None
        elif hasattr(dec, "sessions_summary"):
            worker = {"sessions": dec.sessions_summary()}
        else:
            worker = {"backend": getattr(dec, "name", "fake")}
        return {"current_job": cur, "worker": worker, "pin": pin,
                "serving_allowed": serving}

    @staticmethod
    def _redact_report(obj):
        """GP9 backstop: mask account ids anywhere in a report payload."""
        from .trainer_log import redact_account

        if isinstance(obj, str):
            return redact_account(obj)
        if isinstance(obj, list):
            return [Service._redact_report(v) for v in obj]
        if isinstance(obj, dict):
            return {k: Service._redact_report(v) for k, v in obj.items()}
        return obj

    def do_report(self, args: dict) -> dict:
        from . import report as RP
        from .trainer_log import redact_account
        errors = SV.validate("do_report", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        if args.get("point_id") is not None and args.get("workflow") is not None:
            return err("invalid_args", "one of point_id/workflow per call")
        if args.get("workflow") is not None:
            wf = args["workflow"]
            if not self.store.list_points(wf) and self.store.get_workflow(wf) is None:
                return err("unknown_workflow", f"unknown workflow {wf!r}")
            rep = RP.workflow_report(self.cfg, self.store, wf)
            fw = self._fireworks_report_block()
            if args.get("format") == "md":
                return ok(format="md", report=redact_account(
                    RP.render_autopilot_md(self.autopilot.block())
                    + RP.render_fireworks_md(fw)
                    + RP.render_workflow_md(rep)))
            return ok(**self._redact_report(
                {"workflow": rep, "autopilot": self.autopilot.block(),
                 "fireworks": fw}))
        if args.get("point_id") is not None:
            point = self.store.get_point(args["point_id"])
            if point is None:
                return err("unknown_point", f"unknown point_id {args['point_id']!r}")
            points = [point]
        else:
            points = self.store.list_points()
        reps = []
        for p in points:
            rep = RP.point_report(self.cfg, self.store, p,
                                  self._serve_checks(p, True))
            rep["tier"] = self._effective_tier(p)
            rep["hosted"] = self._hosted_report_block(p)
            rep["private"] = self._private_report_block(p)
            rep["autopilot"], _ = self.autopilot.point_status(p)
            reps.append(rep)
        fw = self._fireworks_report_block()
        if args.get("format") == "md":
            return ok(format="md", report=redact_account(
                RP.render_autopilot_md(self.autopilot.block())
                + RP.render_fireworks_md(fw) + RP.render_md(reps)))
        return ok(**self._redact_report(
            {"points": reps, "autopilot": self.autopilot.block(),
             "fireworks": fw}))

    # -- do_autopilot_status (M16, read-only) ------------------------------
    def do_autopilot_status(self, args: dict) -> dict:
        errors = SV.validate("do_autopilot_status", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        return ok(**self.autopilot.status_report())

    def autopilot_check(self, point_ids: list[str] | None = None) -> dict:
        """Run one autopilot check (do_record/do_outcome triggers and the
        background loop call this; tests call it directly)."""
        return self.autopilot.check(point_ids)

    # -- do_export -------------------------------------------------------
    def do_export(self, args: dict) -> dict:
        from . import export as EX
        from .errors import ToolError
        errors = SV.validate("do_export", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        point = self.store.get_point(args["point_id"])
        if point is None:
            return err("unknown_point", f"unknown point_id {args['point_id']!r}")
        if point.get("data_class", "private") == "private" and not args.get("confirm", False):
            return err("confirm_required",
                       f"{args['point_id']} is private: re-run with confirm: true to export")
        try:
            return ok(**EX.export_point(self.cfg, self.store, args["point_id"], args["path"]))
        except ToolError as e:
            return err(e.code, str(e))

    # -- do_outcome ------------------------------------------------------
    def do_outcome(self, args: dict) -> dict:
        errors = SV.validate("do_outcome", args) if isinstance(args, dict) else ["<root>: not an object"]
        if errors:
            return err("invalid_args", "; ".join(errors[:3]))
        if args.get("plan_id") is not None:
            return self._outcome_plan(args)
        if args.get("decision_id") is None:
            return err("invalid_args", "do_outcome needs decision_id or plan_id")
        row = self.store.get_decision(args["decision_id"])
        if row is None:
            return err("unknown_decision", f"unknown decision_id {args['decision_id']!r}")
        cc = args.get("correct_choice")
        if cc is not None:
            point = self.store.get_point(row["point_id"])
            if point is None or cc not in point["options"]:
                return err("unknown_option", f"{cc!r} is not an option of {row['point_id']}")
        # R2-7: first-outcome-wins is one BEGIN IMMEDIATE.
        status, _ = self.store.try_set_outcome(row["decision_id"], {
            "ok": args.get("ok"), "correct_choice": cc,
            "note": args.get("note")})
        if status == "unknown":
            return err("unknown_decision", f"unknown decision_id {args['decision_id']!r}")
        if status == "ignored":
            return ok(decision_id=row["decision_id"], ignored=True)
        self._maybe_fit_hosted(row["point_id"])
        self._maybe_fit_hosted(row["point_id"], "plan")
        self._maybe_fit_private_plan(row["point_id"])
        if row.get("transcript_id"):
            for plan in self.store.plans_for_transcript_link(row["transcript_id"]):
                if plan.get("audit"):
                    self._plan_drift_check(row["point_id"])
        self.autopilot.kick(row["point_id"])
        return ok(decision_id=row["decision_id"])

    def _outcome_plan(self, args: dict) -> dict:
        """User-supplied check result for outcome-check workflows (§15.5)."""
        if args.get("decision_id") is not None:
            return err("invalid_args", "one target per call: decision_id or plan_id")
        if args.get("correct_choice") is not None:
            return err("invalid_args", "plan outcomes take ok, not correct_choice")
        if args.get("ok") is None:
            return err("invalid_args", "plan outcomes need ok")
        plan = self.store.get_plan(args["plan_id"])
        if plan is None:
            return err("unknown_plan", f"unknown plan_id {args['plan_id']!r}")
        wrec = self.store.get_workflow(plan["workflow"])
        if wrec is not None and wrec.get("check_type") == "deterministic":
            # the server runs the script; a plan outcome here would let the
            # calling agent grade its own run (B9).
            return err("deterministic_check",
                       "plan outcomes are refused on deterministic workflows: "
                       "the check script scores the run")
        if plan.get("check_result") is not None:
            return err("already_set", "check_result already set (first result wins)")
        plan["check_result"] = {"pass": bool(args["ok"]), "source": "user"}
        if args.get("note") is not None:
            plan["check_result"]["note"] = args["note"]
        self.store.put_plan(plan)
        if plan.get("pair_id"):
            self._update_pair_state(plan)
        return ok(plan_id=plan["plan_id"], check_result=plan["check_result"])

    # -- M15 compat: old dm_ names answer for one release with a warning ----
    def _dm_warn(self, old: str, new: str) -> None:
        import sys as _sys

        print(f"decideonce: {old} is deprecated, use {new}",
              file=_sys.stderr)

    def dm_register(self, args: dict) -> dict:
        self._dm_warn("dm_register", "do_register")
        return self.do_register(args)

    def dm_decide(self, args: dict) -> dict:
        self._dm_warn("dm_decide", "do_decide")
        return self.do_decide(args)

    def dm_record(self, args: dict) -> dict:
        self._dm_warn("dm_record", "do_record")
        return self.do_record(args)

    def dm_extract(self, args: dict) -> dict:
        self._dm_warn("dm_extract", "do_extract")
        return self.do_extract(args)

    def dm_train(self, args: dict) -> dict:
        self._dm_warn("dm_train", "do_train")
        return self.do_train(args)

    def dm_promote(self, args: dict) -> dict:
        self._dm_warn("dm_promote", "do_promote")
        return self.do_promote(args)

    def dm_analyze(self, args: dict) -> dict:
        self._dm_warn("dm_analyze", "do_analyze")
        return self.do_analyze(args)

    def dm_plan_run(self, args: dict) -> dict:
        self._dm_warn("dm_plan_run", "do_plan_run")
        return self.do_plan_run(args)

    def dm_plan_result(self, args: dict) -> dict:
        self._dm_warn("dm_plan_result", "do_plan_result")
        return self.do_plan_result(args)

    def dm_report(self, args: dict) -> dict:
        self._dm_warn("dm_report", "do_report")
        return self.do_report(args)

    def dm_autopilot_status(self, args: dict) -> dict:
        self._dm_warn("dm_autopilot_status", "do_autopilot_status")
        return self.do_autopilot_status(args)

    def dm_export(self, args: dict) -> dict:
        self._dm_warn("dm_export", "do_export")
        return self.do_export(args)

    def dm_outcome(self, args: dict) -> dict:
        self._dm_warn("dm_outcome", "do_outcome")
        return self.do_outcome(args)
