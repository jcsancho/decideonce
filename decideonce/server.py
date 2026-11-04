"""decideonce MCP server (stdio). No LLM inside: the only outbound calls are
to hosted Jev (OpenRouter/TypeSafe) and, from M11, Fireworks. Config path
from DECIDEONCE_CONFIG.

Test hook: [decider] backend = "fake" selects scripted fakes.
"""
from __future__ import annotations

import json
import sys

try:  # MCP SDK v2: FastMCP renamed to MCPServer
    from mcp.server.mcpserver import MCPServer as _ServerClass

    _HAS_MCP = True
except ImportError:
    try:  # MCP SDK v1
        from mcp.server.fastmcp import FastMCP as _ServerClass

        _HAS_MCP = True
    except ImportError:
        _ServerClass = None  # type: ignore
        _HAS_MCP = False

from .config import data_root, load_config
from .deciders import build_deciders
from .service import Service
from .store import Store

TRANSPORT = "stdio"

DESCRIPTIONS = {
    "do_analyze": ("Find decision points in a workflow. Input: {workflow: {kind, "
                   "text|path}, [traces: {path}], [candidates]}. First call returns "
                   "rubric + candidate_schema + hints; with candidates, returns score "
                   "and verdict per candidate."),
    "do_register": ("Create or update a decision point (point fields at top level: "
                    "point_id*, workflow*, question*, kind, options*, context_fields*, "
                    "required_fields, rules, irreversible, data_class, target_error). "
                    "Returns the stored point + a call snippet to paste into the skill. "
                    "With type: workflow, registers a workflow record "
                    "(downgrade_requires, check, run_model, cheap_model)."),
    "do_decide": ("Ask for a decision. Input: {point_id*, context*, [teacher_cost]}. "
                  "Returns {decision_id, mode, choice, allowed_options, probabilities, "
                  "confidence, threshold, instruction}. mode teacher: decide yourself, "
                  "then do_record. mode auto/rule: use choice."),
    "do_record": ("Record the teacher's choice. Input: {decision_id*, choice*, "
                  "[teacher_cost]}. choice must be in allowed_options."),
    "do_outcome": ("Attach the real outcome. Input: {decision_id*, [ok], "
                  "[correct_choice], [note]}. Either ok or correct_choice may be given. "
                  "With plan_id + ok, records a user check result for a plan."),
    "do_train": ("Train the decision model for a point. Input: {point_id*, "
                 "[dry_run], [confirm], [training_free_confirmed], [force], "
                 "[idem], [cancel]}. "
                 "dry_run shows the estimate and joining points; confirm "
                 "launches the job (smoke first, then the paid-gated full "
                 "job); repeat calls poll status. idem names the launch for "
                 "crash recovery; cancel (a job_id, or true for this point's "
                 "live job) kills it for good."),
    "do_promote": ("Move a point: {point_id*, phase* (collect|shadow|serve|paused), "
                   "[confirm]}. shadow needs the "
                   "hosted key + a budget + remote_ok (no training); serve "
                   "needs every serve-gate check plus confirm:true (ask the user first). "
                   "With plan_phase, moves plan phases through the plan-serve checks; "
                   "with workflow + downgrade, enables downgrade after a passed trial."),
    "do_report": ("Per-point metrics. Input: {[point_id], [format json|md]}. Agreement, "
                 "coverage, auto share, audits, savings, drift, promotion blockers. "
                 "With workflow, the one-turn workflow report."),
    "do_export": ("Export labelled rows as JSONL. Input: {point_id*, path*, [confirm]}. "
                 "Private points need confirm:true."),
    "do_extract": ("Extract decision labels from a one-turn transcript (two "
                   "calls). Call 1: {transcript|path|transcript_id, "
                   "[contributor_off_confirmed], [extractor_local]} or "
                   "{transcript_id, pass: 2} for the audit pass; returns "
                   "transcript_id + extraction_id + points + record_schema. "
                   "Call 2: {extraction_id*, extractor*, records*, "
                   "[new_points]}; returns accepted + rejected (one code per "
                   "check) + jev_disagreements."),
    "do_plan_run": ("Plan one-turn work. Input: {workflow*, input*, context*, "
                    "[context_source], [run_model], [cheap_model], [trial], "
                    "[pilot_force], [pilot_mode], [task_ref]}. Returns the "
                    "recommendation (inject/downgrade/none) + prompt_addendum "
                    "or would_have."),
    "do_plan_result": ("Record a planned run's result. Input: {plan_id*, "
                       "[transcript], [final_output], [cost], [conflict], "
                       "[trial_id], [pair_id], [arm]}. Runs the workflow "
                       "check when deterministic."),
    "do_autopilot_status": ("Autopilot state (read-only). Returns the approved "
                            "and current [autopilot] hashes, the eligible "
                            "points, and per point its status (on|off|"
                            "blocked(reason)) and the gate it waits for."),
}


def build_service() -> Service:
    from .config import resolve_config_path
    from .deciders.hosted import resolve_key
    from .trainer import build_live_trainer, check_fireworks_host

    cfg = load_config()
    check_fireworks_host(cfg["fireworks"].get("url", ""))  # refuse non-allow-listed hosts
    store = Store(data_root(cfg))
    deciders, _ = build_deciders(cfg, store)  # raises ConfigError: refuse to start
    fw_key = resolve_key(cfg["fireworks"].get("key_ref", ""))
    trainer, _ = build_live_trainer(cfg, store, fw_key)
    svc = Service(cfg, store, deciders, trainer=trainer,
                  fireworks_key=fw_key,
                  config_path=resolve_config_path())
    # F8/F9: a restart orphans evaluating rows (their thread died with the
    # old process); they become eval_failed (retryable). Only here, never
    # in the tick loop: another process may be evaluating right now.
    svc.autopilot.recover_interrupted_evals()
    return svc


def _autopilot_loop() -> None:
    """Background autopilot check every check_interval_min.

    Each tick builds fresh config/store/service handles like the main
    service (F3: same trainer/key/config_path wiring): no shared state
    with the request path, and config edits are picked up. A failing tick
    is logged to stderr and notified with the approved command (F3: never
    silent); the next tick retries.
    """
    import sys as _sys
    import time as _time

    _time.sleep(5)  # let the server finish starting
    while True:
        cfg: dict = {}
        try:
            cfg = load_config()
            mins = float(cfg.get("autopilot", {}).get("check_interval_min", 10))
        except Exception as e:
            print(f"decideonce autopilot: config load failed: {e}",
                  file=_sys.stderr)
            mins = 10
        try:
            if cfg.get("autopilot", {}).get("enabled", False):
                from .config import resolve_config_path
                from .deciders.hosted import resolve_key
                from .trainer import build_live_trainer

                store = Store(data_root(cfg))
                try:
                    deciders, _ = build_deciders(cfg, store)
                    fw_key = resolve_key(
                        cfg.get("fireworks", {}).get("key_ref", ""))
                    trainer, _ = build_live_trainer(cfg, store, fw_key)
                    svc = Service(cfg, store, deciders, trainer=trainer,
                                  fireworks_key=fw_key,
                                  config_path=resolve_config_path())
                    svc.autopilot_check()
                finally:
                    store.close()
        except Exception as e:
            print(f"decideonce autopilot: tick failed: "
                  f"{type(e).__name__}: {e}", file=_sys.stderr)
            try:
                from .autopilot import notify as _notify

                store2 = Store(data_root(cfg)) if cfg else None
                if store2 is not None:
                    try:
                        import json as _json

                        raw = store2.meta_get("autopilot_approved_notify")
                        cmd = _json.loads(raw) if raw else ""
                        _notify(cmd, f"autopilot background check failed: "
                                     f"{type(e).__name__}")
                    finally:
                        store2.close()
            except Exception:
                pass
        _time.sleep(max(mins * 60, 60))


def _cli_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="decideonce",
        description="DecideOnce MCP server (stdio). With no arguments, "
                    "starts the server; 'autopilot keygen|approve' is "
                    "terminal-only trust setup.")
    sub = p.add_subparsers(dest="command")
    ap = sub.add_parser("autopilot", help="terminal-only trust setup")
    ap.add_argument("action", choices=["keygen", "approve"],
                    help="'keygen' creates the signing key once; 'approve' "
                         "signs the effective limits")
    return p


def _cli(argv: list[str]) -> None:
    """decideonce autopilot keygen|approve: terminal-only trust setup."""
    ns = _cli_parser().parse_args(argv)  # --help exits 0, bad usage exits 2
    if ns.command is None:
        _cli_parser().print_usage(sys.stderr)
        sys.exit(2)
    from .config import resolve_config_path

    try:
        cfg = load_config()
        store = Store(data_root(cfg))
    except Exception as e:
        print(f"decideonce: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        if ns.action == "keygen":
            from .autopilot import run_keygen

            res = run_keygen(resolve_config_path(), store, out=print)
        else:
            from .autopilot import run_approve

            res = run_approve(cfg, store, is_tty=sys.stdin.isatty(),
                              out=print)
    finally:
        store.close()
    sys.exit(0 if res == "ok" else (2 if res == "not_interactive" else 1))


def main() -> None:
    if len(sys.argv) > 1:
        _cli(sys.argv[1:])
        return
    if not _HAS_MCP:
        print("decideonce: MCP SDK not installed.", file=sys.stderr)
        sys.exit(1)
    from .config import ConfigError
    try:
        svc = build_service()
    except ConfigError as e:
        print(f"decideonce: refusing to start: {e.code}: {e}", file=sys.stderr)
        sys.exit(1)
    server = _ServerClass("decideonce")

    def _run(fn, payload):
        try:
            return json.dumps(fn(payload or {}))
        except Exception as e:  # never crash the server on a bad call
            return json.dumps({"ok": False, "error": "internal",
                               "message": f"{type(e).__name__}: {e}"})

    @server.tool(description=DESCRIPTIONS["do_analyze"])
    def do_analyze(payload: dict) -> str:
        return _run(svc.do_analyze, payload)

    @server.tool(description=DESCRIPTIONS["do_register"])
    def do_register(payload: dict) -> str:
        return _run(svc.do_register, payload)

    @server.tool(description=DESCRIPTIONS["do_decide"])
    def do_decide(payload: dict) -> str:
        return _run(svc.do_decide, payload)

    @server.tool(description=DESCRIPTIONS["do_record"])
    def do_record(payload: dict) -> str:
        return _run(svc.do_record, payload)

    @server.tool(description=DESCRIPTIONS["do_outcome"])
    def do_outcome(payload: dict) -> str:
        return _run(svc.do_outcome, payload)

    @server.tool(description=DESCRIPTIONS["do_train"])
    def do_train(payload: dict) -> str:
        return _run(svc.do_train, payload)

    @server.tool(description=DESCRIPTIONS["do_promote"])
    def do_promote(payload: dict) -> str:
        return _run(svc.do_promote, payload)

    @server.tool(description=DESCRIPTIONS["do_report"])
    def do_report(payload: dict) -> str:
        return _run(svc.do_report, payload)

    @server.tool(description=DESCRIPTIONS["do_export"])
    def do_export(payload: dict) -> str:
        return _run(svc.do_export, payload)

    @server.tool(description=DESCRIPTIONS["do_extract"])
    def do_extract(payload: dict) -> str:
        return _run(svc.do_extract, payload)

    @server.tool(description=DESCRIPTIONS["do_plan_run"])
    def do_plan_run(payload: dict) -> str:
        return _run(svc.do_plan_run, payload)

    @server.tool(description=DESCRIPTIONS["do_plan_result"])
    def do_plan_result(payload: dict) -> str:
        return _run(svc.do_plan_result, payload)

    @server.tool(description=DESCRIPTIONS["do_autopilot_status"])
    def do_autopilot_status(payload: dict) -> str:
        return _run(svc.do_autopilot_status, payload)

    # M15 compat: old dm_ tool names answer for one release with a warning.
    def _dm_tool_warn(old: str) -> None:
        print(f"decideonce: MCP tool {old} is deprecated, use do_"
              f"{old[3:]}", file=sys.stderr)

    @server.tool(description="Deprecated alias of do_analyze. " + DESCRIPTIONS["do_analyze"])
    def dm_analyze(payload: dict) -> str:
        _dm_tool_warn("dm_analyze")
        return _run(svc.do_analyze, payload)

    @server.tool(description="Deprecated alias of do_register. " + DESCRIPTIONS["do_register"])
    def dm_register(payload: dict) -> str:
        _dm_tool_warn("dm_register")
        return _run(svc.do_register, payload)

    @server.tool(description="Deprecated alias of do_decide. " + DESCRIPTIONS["do_decide"])
    def dm_decide(payload: dict) -> str:
        _dm_tool_warn("dm_decide")
        return _run(svc.do_decide, payload)

    @server.tool(description="Deprecated alias of do_record. " + DESCRIPTIONS["do_record"])
    def dm_record(payload: dict) -> str:
        _dm_tool_warn("dm_record")
        return _run(svc.do_record, payload)

    @server.tool(description="Deprecated alias of do_outcome. " + DESCRIPTIONS["do_outcome"])
    def dm_outcome(payload: dict) -> str:
        _dm_tool_warn("dm_outcome")
        return _run(svc.do_outcome, payload)

    @server.tool(description="Deprecated alias of do_train. " + DESCRIPTIONS["do_train"])
    def dm_train(payload: dict) -> str:
        _dm_tool_warn("dm_train")
        return _run(svc.do_train, payload)

    @server.tool(description="Deprecated alias of do_promote. " + DESCRIPTIONS["do_promote"])
    def dm_promote(payload: dict) -> str:
        _dm_tool_warn("dm_promote")
        return _run(svc.do_promote, payload)

    @server.tool(description="Deprecated alias of do_report. " + DESCRIPTIONS["do_report"])
    def dm_report(payload: dict) -> str:
        _dm_tool_warn("dm_report")
        return _run(svc.do_report, payload)

    @server.tool(description="Deprecated alias of do_export. " + DESCRIPTIONS["do_export"])
    def dm_export(payload: dict) -> str:
        _dm_tool_warn("dm_export")
        return _run(svc.do_export, payload)

    @server.tool(description="Deprecated alias of do_extract. " + DESCRIPTIONS["do_extract"])
    def dm_extract(payload: dict) -> str:
        _dm_tool_warn("dm_extract")
        return _run(svc.do_extract, payload)

    @server.tool(description="Deprecated alias of do_plan_run. " + DESCRIPTIONS["do_plan_run"])
    def dm_plan_run(payload: dict) -> str:
        _dm_tool_warn("dm_plan_run")
        return _run(svc.do_plan_run, payload)

    @server.tool(description="Deprecated alias of do_plan_result. " + DESCRIPTIONS["do_plan_result"])
    def dm_plan_result(payload: dict) -> str:
        _dm_tool_warn("dm_plan_result")
        return _run(svc.do_plan_result, payload)

    @server.tool(description="Deprecated alias of do_autopilot_status. " + DESCRIPTIONS["do_autopilot_status"])
    def dm_autopilot_status(payload: dict) -> str:
        _dm_tool_warn("dm_autopilot_status")
        return _run(svc.do_autopilot_status, payload)

    import threading

    threading.Thread(target=_autopilot_loop, daemon=True).start()
    try:
        server.run(transport=TRANSPORT)
    except TypeError:  # MCP SDK v1 FastMCP.run() takes no transport arg; stdio-only
        server.run()


if __name__ == "__main__":
    main()
