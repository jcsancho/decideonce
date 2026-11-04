"""SQLite store: <data_dir>/<project>/decide.db, WAL mode, busy timeout 5 s (§7.3).

Schema is versioned in `meta`; migrate() applies pending migrations in order.
"""
from __future__ import annotations

import functools
import json
import sqlite3
import threading
from pathlib import Path

SCHEMA_VERSION = 17

SEQ_TOP = 10 ** 18  # sort key for rows without seq (hand-built fixtures)


def _utcnow() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z"


def row_seq(r: dict) -> int:
    """Monotonic insertion order of a decision/plan row (H2)."""
    s = r.get("seq")
    return s if isinstance(s, int) else SEQ_TOP


def row_key(r: dict) -> tuple:
    """Total decision-time order: seq first, ts/id break fabricated ties."""
    return (row_seq(r), r.get("ts") or "", r.get("decision_id") or "")


MIGRATIONS: dict[int, list[str]] = {
    1: [
        """CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)""",
        """CREATE TABLE IF NOT EXISTS points (
          point_id TEXT PRIMARY KEY,
          workflow TEXT NOT NULL,
          kind TEXT NOT NULL,
          source TEXT NOT NULL DEFAULT 'do_decide',
          version INTEGER NOT NULL DEFAULT 1,
          phase TEXT NOT NULL DEFAULT 'collect',
          plan_phase TEXT NOT NULL DEFAULT 'collect',
          data TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS decisions (
          decision_id TEXT PRIMARY KEY,
          point_id TEXT NOT NULL,
          point_version INTEGER NOT NULL,
          ts TEXT NOT NULL,
          context TEXT,
          state_text TEXT,
          allowed_options TEXT,
          rule_hits TEXT,
          jev_zero_shot TEXT,
          probe TEXT,
          mode TEXT NOT NULL,
          audit INTEGER NOT NULL DEFAULT 0,
          source TEXT NOT NULL DEFAULT 'do_decide',
          teacher_choice TEXT,
          extracted_choice TEXT,
          run_id TEXT,
          extraction_id TEXT,
          extractor TEXT,
          decision_pos TEXT,
          at_start INTEGER,
          final_choice TEXT,
          outcome TEXT,
          teacher_cost TEXT,
          jev_seconds REAL,
          unparsed INTEGER NOT NULL DEFAULT 0,
          rule_violation INTEGER NOT NULL DEFAULT 0,
          rule_violation_in_run INTEGER NOT NULL DEFAULT 0,
          override INTEGER NOT NULL DEFAULT 0,
          warnings TEXT,
          created_at TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_decisions_point_ts
           ON decisions (point_id, ts, decision_id)""",
        """CREATE TABLE IF NOT EXISTS models (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          point_id TEXT NOT NULL,
          point_version INTEGER NOT NULL,
          kind TEXT NOT NULL DEFAULT 'probe',
          path TEXT NOT NULL,
          card TEXT NOT NULL,
          created_at TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1
        )""",
        """CREATE INDEX IF NOT EXISTS idx_models_point ON models (point_id, kind, active)""",
    ],
    2: [
        # Snapshot of the point's full option list at decide time: older-version rows
        # are usable for training only when it matches the current list (§2.2).
        """ALTER TABLE decisions ADD COLUMN options TEXT""",
    ],
    3: [
        # Point phase at decide time (shadow evidence, audits, drift).
        """ALTER TABLE decisions ADD COLUMN phase TEXT""",
    ],
    4: [
        """CREATE TABLE IF NOT EXISTS gateway_traces (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          route TEXT NOT NULL,
          model TEXT NOT NULL,
          discovery_key TEXT NOT NULL,
          fingerprint TEXT,
          point_id TEXT,
          point_version INTEGER,
          decision_id TEXT,
          mode TEXT,
          raw_request TEXT,
          raw_response TEXT,
          parsed_option TEXT,
          unparsed INTEGER NOT NULL DEFAULT 0,
          format_compared INTEGER NOT NULL DEFAULT 0,
          format_match INTEGER NOT NULL DEFAULT 0,
          latency_ms REAL,
          upstream_ms REAL,
          usage TEXT,
          status INTEGER,
          data_class TEXT NOT NULL DEFAULT 'private'
        )""",
        """CREATE INDEX IF NOT EXISTS idx_gwtrace_point
           ON gateway_traces (point_id, point_version, id)""",
        """CREATE INDEX IF NOT EXISTS idx_gwtrace_dk ON gateway_traces (discovery_key, id)""",
        """CREATE TABLE IF NOT EXISTS gateway_counters (
          name TEXT NOT NULL,
          scope TEXT NOT NULL DEFAULT '',
          n INTEGER NOT NULL DEFAULT 0,
          PRIMARY KEY (name, scope)
        )""",
    ],
    5: [
        """CREATE TABLE IF NOT EXISTS transcripts (
          transcript_id TEXT PRIMARY KEY,
          ts TEXT NOT NULL,
          run_id TEXT NOT NULL UNIQUE,
          workflow TEXT NOT NULL,
          model TEXT NOT NULL DEFAULT '',
          data_class TEXT NOT NULL DEFAULT 'private',
          input TEXT NOT NULL DEFAULT '',
          events TEXT NOT NULL DEFAULT '[]',
          final_output TEXT NOT NULL DEFAULT '',
          cost TEXT,
          run_outcome TEXT,
          plan_id TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS extractions (
          extraction_id TEXT PRIMARY KEY,
          ts TEXT NOT NULL,
          transcript_id TEXT NOT NULL,
          pass INTEGER NOT NULL DEFAULT 1,
          audit_required INTEGER NOT NULL DEFAULT 0,
          status TEXT NOT NULL DEFAULT 'open',
          extractor TEXT,
          n_accepted INTEGER NOT NULL DEFAULT 0,
          n_rejected INTEGER NOT NULL DEFAULT 0,
          rejected_codes TEXT,
          disagreements TEXT
        )""",
        """CREATE TABLE IF NOT EXISTS extraction_audits (
          audit_id TEXT PRIMARY KEY,
          ts TEXT NOT NULL,
          transcript_id TEXT NOT NULL,
          extraction_id TEXT NOT NULL,
          point_id TEXT NOT NULL,
          record TEXT NOT NULL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_extractions_transcript
           ON extractions (transcript_id, pass, status)""",
        """CREATE INDEX IF NOT EXISTS idx_audits_transcript
           ON extraction_audits (transcript_id, point_id)""",
    ],
    6: [
        "ALTER TABLE decisions ADD COLUMN transcript_id TEXT",
        """CREATE TABLE IF NOT EXISTS workflows (
          workflow TEXT PRIMARY KEY,
          downgrade_requires TEXT NOT NULL,
          check_type TEXT NOT NULL DEFAULT 'outcome',
          check_script TEXT NOT NULL DEFAULT '',
          run_model TEXT NOT NULL DEFAULT '',
          cheap_model TEXT NOT NULL DEFAULT '',
          downgrade_allowed INTEGER NOT NULL DEFAULT 0,
          trial TEXT,
          monitor TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )""",
        """CREATE TABLE IF NOT EXISTS plan_records (
          plan_id TEXT PRIMARY KEY,
          ts TEXT NOT NULL,
          workflow TEXT NOT NULL,
          task_ref TEXT NOT NULL DEFAULT '',
          input TEXT NOT NULL DEFAULT '',
          context TEXT,
          context_source TEXT NOT NULL DEFAULT 'agent',
          run_model TEXT NOT NULL DEFAULT '',
          cheap_model TEXT NOT NULL DEFAULT '',
          recommendation TEXT NOT NULL,
          model TEXT NOT NULL DEFAULT '',
          addendum TEXT,
          points TEXT NOT NULL DEFAULT '[]',
          would_have TEXT,
          audit INTEGER NOT NULL DEFAULT 0,
          trial_id TEXT,
          pair_id TEXT,
          arm TEXT,
          pilot_forced INTEGER NOT NULL DEFAULT 0,
          cost TEXT,
          check_result TEXT,
          conflicts TEXT NOT NULL DEFAULT '[]',
          followed_rate REAL,
          followed_n INTEGER NOT NULL DEFAULT 0,
          point_versions TEXT,
          final_output TEXT
        )""",
        """CREATE INDEX IF NOT EXISTS idx_plans_workflow ON plan_records (workflow, ts)""",
        """CREATE INDEX IF NOT EXISTS idx_plans_pair ON plan_records (pair_id, arm)""",
        """CREATE INDEX IF NOT EXISTS idx_plans_trial ON plan_records (trial_id, pair_id)""",
    ],
    7: [
        # Eval-only transcripts (pilot arm runs): never train, never gate.
        "ALTER TABLE transcripts ADD COLUMN eval_only INTEGER NOT NULL DEFAULT 0",
    ],
    8: [
        # Hosted tier (build §5/§6): settled spend + open reservations,
        # per-decision hosted probabilities + decider, one-time fits.
        """CREATE TABLE IF NOT EXISTS spend (
          month TEXT NOT NULL,
          tier TEXT NOT NULL,
          usd REAL NOT NULL,
          calls INTEGER NOT NULL,
          PRIMARY KEY (month, tier)
        )""",
        """CREATE TABLE IF NOT EXISTS spend_reservations (
          id TEXT PRIMARY KEY,
          month TEXT NOT NULL,
          tier TEXT NOT NULL,
          usd REAL NOT NULL,
          created_at TEXT NOT NULL
        )""",
        "ALTER TABLE decisions ADD COLUMN hosted TEXT",
        "ALTER TABLE decisions ADD COLUMN decider TEXT",
        """CREATE TABLE IF NOT EXISTS hosted_fit (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          point_id TEXT NOT NULL,
          point_version INTEGER NOT NULL,
          kind TEXT NOT NULL,
          epoch_start_ts TEXT NOT NULL,
          fitted_at TEXT NOT NULL,
          n_fit INTEGER NOT NULL,
          last_fit_ts TEXT NOT NULL,
          last_fit_decision_id TEXT NOT NULL,
          temperature REAL NOT NULL,
          temperature_skipped TEXT,
          threshold REAL,
          coverage REAL,
          error_at_threshold REAL,
          fit_agreement REAL,
          teacher_only_options TEXT NOT NULL,
          per_option TEXT NOT NULL,
          active INTEGER NOT NULL DEFAULT 1
        )""",
        """CREATE INDEX IF NOT EXISTS idx_hosted_fit
           ON hosted_fit (point_id, kind, active)""",
    ],
    9: [
        # Fit-row membership: late labels (decided before the last fit row,
        # labelled after) are identified by exclusion from fit_ids (build §7).
        "ALTER TABLE hosted_fit ADD COLUMN fit_ids TEXT",
    ],
    10: [
        # B7: each reservation carries its own stale window, so a process
        # with a short timeout never frees a live reservation made by a
        # process with a longer one (gateway 5 s vs MCP 3 s).
        "ALTER TABLE spend_reservations ADD COLUMN stale_after_s REAL",
    ],
    11: [
        # D9: plan fits record which label kinds they were fitted on, so
        # the report reads them instead of recomputing from live rows
        # (whose labels can change as outcomes land later).
        "ALTER TABLE hosted_fit ADD COLUMN fit_label_kinds TEXT",
    ],
    12: [
        # M16 autopilot: action log (§2) and async training jobs (§2.6/§7.2).
        # jobs.counted_usd is the estimate the launch limits were checked
        # against (full E, or the eval part on the free path).
        """CREATE TABLE IF NOT EXISTS autopilot_log (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          ts TEXT NOT NULL,
          point_id TEXT NOT NULL,
          action TEXT NOT NULL,
          gates TEXT,
          config_hash TEXT,
          estimate_usd REAL
        )""",
        """CREATE INDEX IF NOT EXISTS idx_autopilot_log_ts
           ON autopilot_log (id)""",
        """CREATE TABLE IF NOT EXISTS jobs (
          job_id TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          status TEXT NOT NULL,
          estimate_usd REAL NOT NULL DEFAULT 0,
          counted_usd REAL NOT NULL DEFAULT 0,
          spend_usd REAL NOT NULL DEFAULT 0,
          points TEXT,
          autopilot INTEGER NOT NULL DEFAULT 0,
          error TEXT,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        )""",
    ],
    13: [
        # G1/G2/G5: idempotency key (launching rows), transient-failure
        # counter, and failure time for the 24 h backoff.
        "ALTER TABLE jobs ADD COLUMN idempotency_key TEXT",
        "ALTER TABLE jobs ADD COLUMN status_failures INTEGER NOT NULL DEFAULT 0",
        "ALTER TABLE jobs ADD COLUMN failed_at TEXT",
        # G3: at most one launching/running job (check+insert is also one
        # transaction; this index is the cross-process backstop).
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_running
           ON jobs((1)) WHERE status IN ('running', 'launching')""",
    ],
    14: [
        # Prepublish review H2: monotonic per-DB sequence for decisions and
        # plans, so fit/evidence/late/drift/plan splits order by insertion
        # instead of (second-resolution ts, random id). Backfill follows
        # rowid order, which is insertion order; new rows continue past MAX.
        "ALTER TABLE decisions ADD COLUMN seq INTEGER",
        "UPDATE decisions SET seq = rowid",
        """CREATE INDEX IF NOT EXISTS idx_decisions_point_seq
           ON decisions (point_id, seq)""",
        "ALTER TABLE plan_records ADD COLUMN seq INTEGER",
        "UPDATE plan_records SET seq = rowid",
        """CREATE INDEX IF NOT EXISTS idx_plans_workflow_seq
           ON plan_records (workflow, seq)""",
        # The last fit row's seq (decision seq for decide fits, plan seq
        # for plan fits). Pre-v14 fits carry NULL and resolve by stored id.
        "ALTER TABLE hosted_fit ADD COLUMN last_fit_seq INTEGER",
    ],
    15: [
        # M11 F4: the finalize tail (models + spend) completed. A done job
        # without it crashed between mark_done and store; the recovery pass
        # re-runs the tail. Set atomically with the spend write.
        "ALTER TABLE jobs ADD COLUMN finalized INTEGER NOT NULL DEFAULT 0",
    ],
    16: [
        # M11 F8: background evaluation. train_usd is the ledger amount
        # for the train part (train estimate; 0.0 when the free path
        # covers it); counted_usd is reset to it when evaluation ends so
        # the cap counts training estimate + settled evaluation spend.
        # eval_json carries the eval results (per-point T/metrics) from
        # the eval thread to the finalize tail and the recovery pass.
        "ALTER TABLE jobs ADD COLUMN train_usd REAL NOT NULL DEFAULT 0",
        "ALTER TABLE jobs ADD COLUMN eval_json TEXT",
        # M11 F8 (GP7): private plan fits share the fits table, keyed by
        # tier so they never clobber the hosted fits.
        "ALTER TABLE hosted_fit ADD COLUMN tier TEXT NOT NULL "
        "DEFAULT 'hosted_jev'",
        # M11 F8: evaluating counts as running for one-job-at-a-time.
        "DROP INDEX IF EXISTS idx_jobs_one_running",
        """CREATE UNIQUE INDEX IF NOT EXISTS idx_jobs_one_running
           ON jobs((1)) WHERE status IN ('running', 'launching',
           'evaluating')""",
    ],
    17: [
        # M11 F9 (§7.2 step 5): the pending checkpoint's probabilities,
        # logged beside the serving one on retrained serve points.
        "ALTER TABLE decisions ADD COLUMN pending_probe TEXT",
    ],
}


class Store:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "decide.db"
        self.models_dir = self.root / "models"
        self.models_dir.mkdir(exist_ok=True)
        # check_same_thread=False + RLock: the MCP server and gateway call from
        # worker threads; SQLite itself is in WAL mode with a busy timeout.
        self._lock = threading.RLock()
        self.cx = sqlite3.connect(str(self.db_path), timeout=5.0, isolation_level=None,
                                  check_same_thread=False)
        self.cx.row_factory = sqlite3.Row
        self.cx.execute("PRAGMA journal_mode=WAL")
        self.cx.execute("PRAGMA busy_timeout=5000")
        self.migrate()

    def close(self) -> None:
        self.cx.close()

    # -- migrations ------------------------------------------------------
    def version(self) -> int:
        try:
            row = self.cx.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row["value"]) if row else 0

    def migrate(self) -> int:
        # F5: two processes starting together must not both run ALTER TABLE.
        # Each version is applied inside one BEGIN IMMEDIATE with the version
        # re-checked inside it; the loser sees the winner's commit and skips.
        for _ in range(self.version() + 1, SCHEMA_VERSION + 1):
            self.cx.execute("BEGIN IMMEDIATE")
            try:
                try:
                    row = self.cx.execute(
                        "SELECT value FROM meta WHERE key='schema_version'"
                    ).fetchone()
                except sqlite3.OperationalError:
                    row = None
                current = int(row["value"]) if row else 0
                nxt = current + 1
                if nxt > SCHEMA_VERSION:
                    self.cx.execute("ROLLBACK")
                    break
                for stmt in MIGRATIONS[nxt]:
                    self.cx.execute(stmt)
                self.cx.execute(
                    "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (str(nxt),),
                )
                self.cx.execute("COMMIT")
            except Exception:
                try:
                    self.cx.execute("ROLLBACK")
                except Exception:
                    pass
                raise
        return self.version()

    # -- points ----------------------------------------------------------
    def put_point(self, point: dict) -> None:
        now = _utcnow()
        with self.cx:
            self.cx.execute(
                """INSERT INTO points(point_id, workflow, kind, source, version, phase,
                                      plan_phase, data, created_at, updated_at)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(point_id) DO UPDATE SET workflow=excluded.workflow,
                     kind=excluded.kind, source=excluded.source, version=excluded.version,
                     phase=excluded.phase, plan_phase=excluded.plan_phase, data=excluded.data,
                     updated_at=excluded.updated_at""",
                (
                    point["point_id"], point["workflow"], point["kind"],
                    point.get("source", "do_decide"), point.get("point_version", 1),
                    point.get("phase", "collect"), point.get("plan_phase", "collect"),
                    json.dumps(point), now, now,
                ),
            )

    def get_point(self, point_id: str) -> dict | None:
        row = self.cx.execute("SELECT data FROM points WHERE point_id=?", (point_id,)).fetchone()
        return json.loads(row["data"]) if row else None

    def cas_point_phase(self, point_id: str, expected_phase: str,
                        expected_version: int, expected_stopped: bool,
                        new_phase: str) -> dict | None:
        """E3: atomic compare-and-set for autopilot promotion.

        Re-reads the point and checks phase, stop flag and version inside
        one BEGIN IMMEDIATE. Returns the fresh point after the update, or
        None when the point is missing or any expectation mismatches (a
        concurrent user pause/demotion wins; the caller must not promote).
        """
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            row = self.cx.execute(
                "SELECT data FROM points WHERE point_id=?", (point_id,)).fetchone()
            if row is None:
                self.cx.execute("ROLLBACK")
                return None
            fresh = json.loads(row["data"])
            if (fresh.get("phase", "collect") != expected_phase
                    or fresh.get("point_version", 1) != expected_version
                    or bool(fresh.get("autopilot_stopped", False)) != bool(expected_stopped)):
                self.cx.execute("ROLLBACK")
                return None
            fresh["phase"] = new_phase
            now = _utcnow()
            self.cx.execute(
                """UPDATE points SET workflow=?, kind=?, source=?, version=?,
                                      phase=?, plan_phase=?, data=?, updated_at=?
                   WHERE point_id=?""",
                (fresh["workflow"], fresh["kind"],
                 fresh.get("source", "do_decide"),
                 fresh.get("point_version", 1), fresh.get("phase", "collect"),
                 fresh.get("plan_phase", "collect"), json.dumps(fresh),
                 now, point_id),
            )
            self.cx.execute("COMMIT")
            return fresh
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def list_points(self, workflow: str | None = None) -> list[dict]:
        q = "SELECT data FROM points ORDER BY point_id"
        args: tuple = ()
        if workflow is not None:
            q = "SELECT data FROM points WHERE workflow=? ORDER BY point_id"
            args = (workflow,)
        return [json.loads(r["data"]) for r in self.cx.execute(q, args)]

    def points_in_registration_order(self, workflow: str) -> list[dict]:
        """Points of a workflow in registration order (§15.5 step 1)."""
        return [json.loads(r["data"]) for r in self.cx.execute(
            "SELECT data FROM points WHERE workflow=? ORDER BY rowid", (workflow,))]

    # -- decisions -------------------------------------------------------
    @staticmethod
    def _j(v) -> str | None:
        return None if v is None else json.dumps(v)

    def insert_decision(self, row: dict, max_rows_per_point: int = 50000) -> None:
        with self.cx:
            self.cx.execute(
                """INSERT INTO decisions(decision_id, point_id, point_version, ts, context,
                   state_text, options, allowed_options, rule_hits, jev_zero_shot, probe, mode,
                   audit, source, teacher_choice, extracted_choice, run_id, extraction_id,
                   extractor, decision_pos, at_start, final_choice, outcome, teacher_cost,
                   jev_seconds, unparsed, rule_violation, rule_violation_in_run, override,
                   warnings, phase, created_at, transcript_id, hosted, decider, seq,
                   pending_probe)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           (SELECT COALESCE(MAX(seq), 0) + 1 FROM decisions), ?)""",
                (
                    row["decision_id"], row["point_id"], row["point_version"], row["ts"],
                    self._j(row.get("context")), row.get("state_text"),
                    self._j(row.get("options")),
                    self._j(row.get("allowed_options")), self._j(row.get("rule_hits")),
                    self._j(row.get("jev_zero_shot")), self._j(row.get("probe")),
                    row.get("mode", "teacher"), int(bool(row.get("audit", False))),
                    row.get("source", "do_decide"), row.get("teacher_choice"),
                    row.get("extracted_choice"), row.get("run_id"), row.get("extraction_id"),
                    row.get("extractor"), self._j(row.get("decision_pos")),
                    row.get("at_start"), row.get("final_choice"), self._j(row.get("outcome")),
                    self._j(row.get("teacher_cost")), row.get("jev_seconds"),
                    int(bool(row.get("unparsed", False))),
                    int(bool(row.get("rule_violation", False))),
                    int(bool(row.get("rule_violation_in_run", False))),
                    int(bool(row.get("override", False))), self._j(row.get("warnings")),
                    row.get("phase"), _utcnow(),
                    row.get("transcript_id"), self._j(row.get("hosted")),
                    row.get("decider"), self._j(row.get("pending_probe")),
                ),
            )
            self._trim(row["point_id"], max_rows_per_point)

    def _trim(self, point_id: str, limit: int) -> None:
        if limit <= 0:
            return
        self.cx.execute(
            """DELETE FROM decisions WHERE point_id=? AND decision_id NOT IN (
                 SELECT decision_id FROM decisions WHERE point_id=?
                 ORDER BY seq DESC LIMIT ?)""",
            (point_id, point_id, limit),
        )

    def update_decision(self, decision_id: str, fields: dict) -> None:
        sets, args = [], []
        for k, v in fields.items():
            if k in ("context", "allowed_options", "rule_hits", "jev_zero_shot", "probe",
                      "outcome", "teacher_cost", "decision_pos", "warnings", "hosted"):
                v = self._j(v)
            elif k in ("audit", "unparsed", "rule_violation", "rule_violation_in_run", "override"):
                v = int(bool(v))
            sets.append(f"{k}=?")
            args.append(v)
        if not sets:
            return
        args.append(decision_id)
        with self.cx:
            self.cx.execute(f"UPDATE decisions SET {', '.join(sets)} WHERE decision_id=?", args)

    @staticmethod
    def _parse(row: sqlite3.Row) -> dict:
        d = dict(row)
        for k in ("context", "options", "allowed_options", "rule_hits", "jev_zero_shot",
                  "probe", "outcome", "teacher_cost", "decision_pos", "warnings",
                  "hosted", "pending_probe"):
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        for k in ("audit", "unparsed", "rule_violation", "rule_violation_in_run", "override"):
            d[k] = bool(d.get(k))
        if d.get("at_start") is not None:
            d["at_start"] = bool(d["at_start"])
        return d

    def get_decision(self, decision_id: str) -> dict | None:
        row = self.cx.execute(
            "SELECT * FROM decisions WHERE decision_id=?", (decision_id,)).fetchone()
        return self._parse(row) if row else None

    def try_record_teacher(self, decision_id: str, choice: str,
                           teacher_cost=None, override: bool = False
                           ) -> tuple[str, str | None]:
        """R2-7: atomic first-record-wins. Returns (ok|already|unknown, prev)."""
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            row = self.cx.execute(
                "SELECT teacher_choice FROM decisions WHERE decision_id=?",
                (decision_id,)).fetchone()
            if row is None:
                self.cx.execute("ROLLBACK")
                return "unknown", None
            prev = row["teacher_choice"]
            if prev is not None:
                self.cx.execute("ROLLBACK")
                return "already", prev
            self.cx.execute(
                "UPDATE decisions SET teacher_choice=?, final_choice=?, "
                "teacher_cost=?, override=? WHERE decision_id=?",
                (choice, choice,
                 self._j(teacher_cost) if teacher_cost is not None else None,
                 int(bool(override)), decision_id))
            self.cx.execute("COMMIT")
            return "ok", None
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def try_set_outcome(self, decision_id: str, outcome: dict
                        ) -> tuple[str, dict | None]:
        """R2-7: atomic first-outcome-wins. Returns (ok|ignored|unknown, row?)."""
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            row = self.cx.execute(
                "SELECT outcome, warnings FROM decisions WHERE decision_id=?",
                (decision_id,)).fetchone()
            if row is None:
                self.cx.execute("ROLLBACK")
                return "unknown", None
            prev = json.loads(row["outcome"]) if row["outcome"] else {}
            if prev.get("ok") is not None or prev.get("correct_choice") is not None:
                warnings = json.loads(row["warnings"]) if row["warnings"] else []
                if "outcome_ignored" not in warnings:
                    warnings.append("outcome_ignored")
                    self.cx.execute(
                        "UPDATE decisions SET warnings=? WHERE decision_id=?",
                        (json.dumps(warnings), decision_id))
                self.cx.execute("COMMIT")
                return "ignored", None
            self.cx.execute(
                "UPDATE decisions SET outcome=? WHERE decision_id=?",
                (json.dumps(outcome), decision_id))
            self.cx.execute("COMMIT")
            return "ok", None
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def decisions_for_point(self, point_id: str, limit: int | None = None) -> list[dict]:
        # seq order = insertion = time order (H2: ts alone is ambiguous).
        q = "SELECT * FROM decisions WHERE point_id=? ORDER BY seq"
        args: tuple = (point_id,)
        if limit is not None:
            q += " LIMIT ?"
            args = (point_id, limit)
        return [self._parse(r) for r in self.cx.execute(q, args)]

    def count_decisions(self, point_id: str) -> int:
        row = self.cx.execute(
            "SELECT COUNT(*) c FROM decisions WHERE point_id=?", (point_id,)).fetchone()
        return int(row["c"])

    # -- models ----------------------------------------------------------
    def save_model(self, point_id: str, point_version: int, kind: str,
                   path: str, card: dict, active: bool = True) -> int:
        with self.cx:
            if active:
                self.cx.execute(
                    "UPDATE models SET active=0 WHERE point_id=? AND kind=?",
                    (point_id, kind))
            cur = self.cx.execute(
                """INSERT INTO models(point_id, point_version, kind, path, card, created_at, active)
                   VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (point_id, point_version, kind, path, json.dumps(card), _utcnow(),
                 1 if active else 0),
            )
            return int(cur.lastrowid)

    def activate_model(self, model_id: int) -> None:
        """F9 (§7.2 step 5): switch the active private model (the pending
        checkpoint earned serve)."""
        with self.cx:
            row = self.cx.execute(
                "SELECT point_id, kind FROM models WHERE id=?",
                (model_id,)).fetchone()
            if row is None:
                return
            self.cx.execute(
                "UPDATE models SET active=0 WHERE point_id=? AND kind=?",
                (row["point_id"], row["kind"]))
            self.cx.execute("UPDATE models SET active=1 WHERE id=?",
                            (model_id,))

    def active_model(self, point_id: str, kind: str = "probe") -> dict | None:
        row = self.cx.execute(
            """SELECT * FROM models WHERE point_id=? AND kind=? AND active=1
               ORDER BY id DESC LIMIT 1""", (point_id, kind)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["card"] = json.loads(d["card"])
        return d

    def get_model(self, model_id: int) -> dict | None:
        """F9: a model row by id (the pending checkpoint), or None."""
        row = self.cx.execute(
            "SELECT * FROM models WHERE id=?", (model_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["card"] = json.loads(d["card"])
        return d

    # -- gateway traces + counters ---------------------------------------
    def trace_insert(self, t: dict) -> int:
        with self.cx:
            cur = self.cx.execute(
                """INSERT INTO gateway_traces(ts, route, model, discovery_key, fingerprint,
                   point_id, point_version, decision_id, mode, raw_request, raw_response,
                   parsed_option, unparsed, format_compared, format_match, latency_ms,
                   upstream_ms, usage, status, data_class)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (t["ts"], t["route"], t["model"], t["discovery_key"], t.get("fingerprint"),
                 t.get("point_id"), t.get("point_version"), t.get("decision_id"),
                 t.get("mode"), self._j(t.get("raw_request")), t.get("raw_response"),
                 t.get("parsed_option"), int(bool(t.get("unparsed", False))),
                 int(bool(t.get("format_compared", False))),
                 int(bool(t.get("format_match", False))), t.get("latency_ms"),
                 t.get("upstream_ms"), self._j(t.get("usage")), t.get("status"),
                 t.get("data_class", "private")))
            return int(cur.lastrowid)

    def traces_by_discovery(self, discovery_key: str) -> list[dict]:
        return [self._parse_trace(r) for r in self.cx.execute(
            "SELECT * FROM gateway_traces WHERE discovery_key=? ORDER BY id", (discovery_key,))]

    def unregistered_traces(self) -> list[dict]:
        return [self._parse_trace(r) for r in self.cx.execute(
            "SELECT * FROM gateway_traces WHERE point_id IS NULL AND raw_request IS NOT NULL "
            "ORDER BY id")]

    def traces_for_point(self, point_id: str, point_version: int | None = None) -> list[dict]:
        if point_version is None:
            q = "SELECT * FROM gateway_traces WHERE point_id=? ORDER BY id"
            args: tuple = (point_id,)
        else:
            q = "SELECT * FROM gateway_traces WHERE point_id=? AND point_version=? ORDER BY id"
            args = (point_id, point_version)
        return [self._parse_trace(r) for r in self.cx.execute(q, args)]

    @staticmethod
    def _parse_trace(row) -> dict:
        d = dict(row)
        if d.get("raw_request") is not None:
            d["raw_request"] = json.loads(d["raw_request"])
        if d.get("usage") is not None:
            d["usage"] = json.loads(d["usage"])
        d["unparsed"] = bool(d.get("unparsed"))
        d["format_compared"] = bool(d.get("format_compared"))
        d["format_match"] = bool(d.get("format_match"))
        return d

    def counter_inc(self, name: str, scope: str = "", by: int = 1) -> None:
        with self.cx:
            self.cx.execute(
                "INSERT INTO gateway_counters(name, scope, n) VALUES(?, ?, ?) "
                "ON CONFLICT(name, scope) DO UPDATE SET n=n+excluded.n",
                (name, scope, by))

    def counter_get(self, name: str, scope: str = "") -> int:
        row = self.cx.execute(
            "SELECT n FROM gateway_counters WHERE name=? AND scope=?", (name, scope)
        ).fetchone()
        return int(row["n"]) if row else 0

    # -- hosted spend: reserve, then reconcile (build §5) --------------------
    def hosted_reserve(self, month: str, tier: str, reserve_usd: float,
                       budget_usd: float, stale_after_s: float) -> str | None:
        """Reserve one hosted call's budget, atomically (BEGIN IMMEDIATE).

        Releases stale reservations, then admits the call iff settled + open
        + reserve <= budget. Returns the reservation id, or None when the
        call must not happen. A budget of 0 admits nothing.

        B7: staleness is per reservation row (each row carries the window of
        the process that made it), so a short-timeout caller never frees a
        live reservation of a longer-timeout process. Rows predating the
        window column fall back to this call's stale_after_s.
        """
        import uuid
        if budget_usd <= 0:
            return None
        rid = uuid.uuid4().hex
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            self.cx.execute(
                "DELETE FROM spend_reservations WHERE "
                "(strftime('%s', 'now') - strftime('%s', created_at)) > "
                "COALESCE(stale_after_s, ?)",
                (stale_after_s,))
            row = self.cx.execute(
                "SELECT usd FROM spend WHERE month=? AND tier=?",
                (month, tier)).fetchone()
            settled = float(row["usd"]) if row else 0.0
            row = self.cx.execute(
                "SELECT COALESCE(SUM(usd), 0) s FROM spend_reservations "
                "WHERE month=? AND tier=?", (month, tier)).fetchone()
            open_sum = float(row["s"])
            # epsilon: usd is float, and 10 x 0.0005 must fit a 0.005 budget.
            if settled + open_sum + reserve_usd > budget_usd + 1e-9:
                self.cx.execute("ROLLBACK")
                return None
            self.cx.execute(
                "INSERT INTO spend_reservations(id, month, tier, usd, created_at,"
                " stale_after_s) VALUES(?, ?, ?, ?, ?, ?)",
                (rid, month, tier, float(reserve_usd), _utcnow(),
                 float(stale_after_s)))
            self.cx.execute("COMMIT")
            return rid
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def hosted_settle(self, reservation_id: str, month: str, tier: str,
                      real_cost: float) -> None:
        """Delete the reservation and add the real cost to settled spend."""
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            self.cx.execute("DELETE FROM spend_reservations WHERE id=?",
                            (reservation_id,))
            self.cx.execute(
                "INSERT INTO spend(month, tier, usd, calls) VALUES(?, ?, ?, 1)"
                " ON CONFLICT(month, tier) DO UPDATE SET usd=usd+excluded.usd,"
                " calls=calls+1", (month, tier, float(real_cost)))
            self.cx.execute("COMMIT")
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def hosted_release(self, reservation_id: str) -> None:
        """Delete a reservation without settling (failed call)."""
        with self.cx:
            self.cx.execute("DELETE FROM spend_reservations WHERE id=?",
                            (reservation_id,))

    def spend_add(self, month: str, tier: str, usd: float) -> None:
        """Add settled spend without touching reservations (B7: last-resort
        settle fallback, so a failed settle never loses the real cost)."""
        with self.cx:
            self.cx.execute(
                "INSERT INTO spend(month, tier, usd, calls) VALUES(?, ?, ?, 1)"
                " ON CONFLICT(month, tier) DO UPDATE SET usd=usd+excluded.usd,"
                " calls=calls+1", (month, tier, float(usd)))

    def spend_get(self, month: str, tier: str) -> dict:
        row = self.cx.execute(
            "SELECT usd, calls FROM spend WHERE month=? AND tier=?",
            (month, tier)).fetchone()
        if row is None:
            return {"usd": 0.0, "calls": 0}
        return {"usd": float(row["usd"]), "calls": int(row["calls"])}

    def spend_open(self, month: str, tier: str) -> dict:
        row = self.cx.execute(
            "SELECT COALESCE(SUM(usd), 0) usd, COUNT(*) n"
            " FROM spend_reservations WHERE month=? AND tier=?",
            (month, tier)).fetchone()
        return {"usd": float(row["usd"]), "n": int(row["n"])}

    # -- meta kv ---------------------------------------------------------
    def meta_get(self, key: str) -> str | None:
        row = self.cx.execute("SELECT value FROM meta WHERE key=?",
                              (key,)).fetchone()
        return row["value"] if row else None

    def meta_put(self, key: str, value: str) -> None:
        with self.cx:
            self.cx.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value))

    # -- autopilot log + training jobs (M16) -------------------------------
    def autopilot_log_add(self, entry: dict) -> int:
        """Append {ts, point_id, action, gates?, config_hash?, estimate_usd?}."""
        with self.cx:
            cur = self.cx.execute(
                """INSERT INTO autopilot_log(ts, point_id, action, gates,
                                             config_hash, estimate_usd)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (entry["ts"], entry["point_id"], entry["action"],
                 json.dumps(entry.get("gates")) if entry.get("gates") is not None else None,
                 entry.get("config_hash"), entry.get("estimate_usd")))
            return int(cur.lastrowid)

    def autopilot_log_recent(self, n: int = 20) -> list[dict]:
        rows = self.cx.execute(
            "SELECT * FROM autopilot_log ORDER BY id DESC LIMIT ?",
            (n,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["gates"] = json.loads(d["gates"]) if d.get("gates") is not None else None
            out.append(d)
        return out

    def job_put(self, job: dict) -> None:
        with self.cx:
            self.cx.execute(
                """INSERT INTO jobs(job_id, kind, status, estimate_usd, counted_usd,
                                    spend_usd, points, autopilot, error,
                                    created_at, updated_at)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job["job_id"], job["kind"], job["status"],
                 float(job.get("estimate_usd", 0)), float(job.get("counted_usd", 0)),
                 float(job.get("spend_usd", 0)),
                 json.dumps(job.get("points", [])),
                 1 if job.get("autopilot") else 0, job.get("error"),
                 job["created_at"], job.get("updated_at", job["created_at"])))

    def job_get(self, job_id: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM jobs WHERE job_id=?",
                              (job_id,)).fetchone()
        return self._parse_job(row) if row else None

    def job_update(self, job_id: str, fields: dict) -> None:
        fields = dict(fields)
        if "points" in fields:
            fields["points"] = json.dumps(fields["points"])
        fields["updated_at"] = _utcnow()
        keys = [k for k in fields
                if k in ("status", "estimate_usd", "counted_usd", "spend_usd",
                         "points", "error", "updated_at", "idempotency_key",
                         "status_failures", "failed_at")]
        if not keys:
            return
        with self.cx:
            self.cx.execute(
                f"UPDATE jobs SET {', '.join(f'{k}=?' for k in keys)} "
                "WHERE job_id=?",
                tuple(fields[k] for k in keys) + (job_id,))

    def job_insert_launching(self, job: dict) -> None:
        """G1/G3: atomic check-and-insert of a launching row.

        Inside one BEGIN IMMEDIATE: fail when a launching/running job
        already exists, else insert. The unique partial index is the
        cross-process backstop (IntegrityError when the race slips through).
        """
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            row = self.cx.execute(
                "SELECT 1 FROM jobs WHERE status IN ('running', 'launching',"
                " 'evaluating') LIMIT 1").fetchone()
            if row is not None:
                self.cx.execute("ROLLBACK")
                raise RuntimeError("job_running")
            self.cx.execute(
                """INSERT INTO jobs(job_id, kind, status, estimate_usd, counted_usd,
                                    spend_usd, points, autopilot, error,
                                    created_at, updated_at, idempotency_key,
                                    status_failures, failed_at, train_usd)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (job["job_id"], job["kind"], job["status"],
                 float(job.get("estimate_usd", 0)), float(job.get("counted_usd", 0)),
                 float(job.get("spend_usd", 0)),
                 json.dumps(job.get("points", [])),
                 1 if job.get("autopilot") else 0, job.get("error"),
                 job["created_at"], job.get("updated_at", job["created_at"]),
                 job.get("idempotency_key"),
                 int(job.get("status_failures", 0)), job.get("failed_at"),
                 float(job.get("train_usd", 0))))
            self.cx.execute("COMMIT")
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise

    def job_claim_launch(self, temp_id: str, remote_id: str,
                         idempotency_key: str) -> bool:
        """G1: adopt the remote job id for a launching row. True when claimed."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE jobs SET job_id=?, status='running', "
                "idempotency_key=?, updated_at=? "
                "WHERE job_id=? AND status='launching'",
                (remote_id, idempotency_key, _utcnow(), temp_id))
            return cur.rowcount == 1

    def job_finalize(self, job_id: str, new_status: str) -> bool:
        """G4: conditional running -> done/smoke_done. True when we won."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE jobs SET status=?, updated_at=? "
                "WHERE job_id=? AND status='running'",
                (new_status, _utcnow(), job_id))
            return cur.rowcount == 1

    def job_note_status_failure(self, job_id: str) -> int:
        """G2: bump the transient-failure counter. Returns the new count."""
        with self.cx:
            self.cx.execute(
                "UPDATE jobs SET status_failures=COALESCE(status_failures,0)+1, "
                "updated_at=? WHERE job_id=?", (_utcnow(), job_id))
            row = self.cx.execute(
                "SELECT status_failures FROM jobs WHERE job_id=?",
                (job_id,)).fetchone()
            return int(row["status_failures"]) if row else 0

    def job_reset_status_failures(self, job_id: str) -> None:
        with self.cx:
            self.cx.execute(
                "UPDATE jobs SET status_failures=0 WHERE job_id=?", (job_id,))

    def jobs_launching(self) -> list[dict]:
        rows = self.cx.execute(
            "SELECT * FROM jobs WHERE status='launching' ORDER BY created_at"
        ).fetchall()
        return [self._parse_job(r) for r in rows]

    def latest_autopilot_failure_at(self) -> str | None:
        """G5: newest failed_at of a failed autopilot job, or None."""
        row = self.cx.execute(
            "SELECT MAX(failed_at) m FROM jobs WHERE autopilot=1 AND "
            "status='failed' AND failed_at IS NOT NULL").fetchone()
        return row["m"] if row and row["m"] else None

    @staticmethod
    def _parse_job(row) -> dict:
        d = dict(row)
        d["points"] = json.loads(d["points"]) if d.get("points") else []
        d["autopilot"] = bool(d.get("autopilot"))
        for k in ("estimate_usd", "counted_usd", "spend_usd", "train_usd"):
            d[k] = float(d.get(k) or 0)
        return d

    def jobs_running(self) -> list[dict]:
        """Running + evaluating jobs (F8: one job at a time covers both)."""
        rows = self.cx.execute(
            "SELECT * FROM jobs WHERE status IN ('running', 'evaluating') "
            "ORDER BY created_at").fetchall()
        return [self._parse_job(r) for r in rows]

    def jobs_evaluating(self) -> list[dict]:
        """Jobs stuck mid-evaluation (F8 restart recovery)."""
        rows = self.cx.execute(
            "SELECT * FROM jobs WHERE status='evaluating' "
            "ORDER BY created_at").fetchall()
        return [self._parse_job(r) for r in rows]

    def jobs_eval_failed(self) -> list[dict]:
        """Jobs whose evaluation failed (F8 retry_eval)."""
        rows = self.cx.execute(
            "SELECT * FROM jobs WHERE status='eval_failed' "
            "ORDER BY created_at").fetchall()
        return [self._parse_job(r) for r in rows]

    def job_begin_eval(self, job_id: str) -> bool:
        """F8: conditional running -> evaluating (one eval thread wins)."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE jobs SET status='evaluating', updated_at=? "
                "WHERE job_id=? AND status='running'",
                (_utcnow(), job_id))
            return cur.rowcount == 1

    def job_claim_evaluated(self, job_id: str) -> bool:
        """F8: conditional evaluating -> done (the eval thread's G4)."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE jobs SET status='done', updated_at=? "
                "WHERE job_id=? AND status='evaluating'",
                (_utcnow(), job_id))
            return cur.rowcount == 1

    def job_retry_eval(self, job_id: str) -> bool:
        """F8: conditional eval_failed -> evaluating (retry_eval)."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE jobs SET status='evaluating', error=NULL, "
                "eval_json=NULL, updated_at=? "
                "WHERE job_id=? AND status='eval_failed'",
                (_utcnow(), job_id))
            return cur.rowcount == 1

    def job_save_eval(self, job_id: str, eval_doc: dict) -> None:
        """F8: persist eval results for the tail and the recovery pass."""
        with self.cx:
            self.cx.execute(
                "UPDATE jobs SET eval_json=?, updated_at=? WHERE job_id=?",
                (json.dumps(eval_doc), _utcnow(), job_id))

    def job_get_eval(self, job_id: str) -> dict | None:
        """F8: the persisted eval results, or None when absent."""
        row = self.cx.execute(
            "SELECT eval_json FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        if not row or not row["eval_json"]:
            return None
        try:
            return json.loads(row["eval_json"])
        except ValueError:
            return None

    def jobs_done_unfinalized(self) -> list[dict]:
        """M11 F4: done/smoke_done jobs whose tail never completed (crash
        between mark_done and store). Recovery re-runs their tails."""
        rows = self.cx.execute(
            "SELECT * FROM jobs WHERE status IN ('done', 'smoke_done') "
            "AND COALESCE(finalized, 0)=0 ORDER BY created_at").fetchall()
        return [self._parse_job(r) for r in rows]

    def job_record_spend_finalized(self, job_id: str, counted: float,
                                   month: str) -> bool:
        """M11 F4: settle spend and set finalized=1 atomically. Only the
        first call wins (concurrent recovery is safe)."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE jobs SET spend_usd=?, finalized=1, updated_at=? "
                "WHERE job_id=? AND COALESCE(finalized, 0)=0",
                (float(counted), _utcnow(), job_id))
            if cur.rowcount != 1:
                return False
            self.cx.execute(
                "INSERT INTO spend(month, tier, usd, calls) VALUES(?, ?, ?, 1)"
                " ON CONFLICT(month, tier) DO UPDATE SET usd=usd+excluded.usd,"
                " calls=calls+1", (month, "fireworks", float(counted)))
            return True

    def jobs_latest(self) -> dict | None:
        row = self.cx.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC, rowid DESC LIMIT 1").fetchone()
        return self._parse_job(row) if row else None

    def smoke_done_exists(self) -> bool:
        row = self.cx.execute(
            "SELECT 1 FROM jobs WHERE kind='smoke' AND status IN "
            "('smoke_done', 'done') LIMIT 1").fetchone()
        return row is not None

    def autopilot_month_counted(self, month: str) -> float:
        """Sum of counted estimates of autopilot jobs created in a UTC month."""
        row = self.cx.execute(
            "SELECT COALESCE(SUM(counted_usd), 0) s FROM jobs "
            "WHERE autopilot=1 AND substr(created_at, 1, 7)=?",
            (month,)).fetchone()
        return float(row["s"])

    # -- hosted fits (build §7) --------------------------------------------
    def save_hosted_fit(self, fit: dict) -> int:
        tier = fit.get("tier", "hosted_jev")
        with self.cx:
            self.cx.execute(
                "UPDATE hosted_fit SET active=0 WHERE point_id=? AND kind=? "
                "AND tier=?",
                (fit["point_id"], fit["kind"], tier))
            cur = self.cx.execute(
                """INSERT INTO hosted_fit(point_id, point_version, kind,
                   tier, epoch_start_ts, fitted_at, n_fit, last_fit_ts,
                   last_fit_decision_id, last_fit_seq, temperature,
                   temperature_skipped, threshold, coverage, error_at_threshold,
                   fit_agreement, teacher_only_options, per_option, fit_ids,
                   fit_label_kinds, active)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, 1)""",
                (fit["point_id"], fit["point_version"], fit["kind"],
                 fit.get("tier", "hosted_jev"),
                 fit["epoch_start_ts"], fit["fitted_at"], fit["n_fit"],
                 fit["last_fit_ts"], fit["last_fit_decision_id"],
                 fit.get("last_fit_seq"),
                 fit["temperature"], fit.get("temperature_skipped"),
                 fit.get("threshold"), fit.get("coverage"),
                 fit.get("error_at_threshold"), fit.get("fit_agreement"),
                 json.dumps(fit.get("teacher_only_options", [])),
                 json.dumps(fit.get("per_option", {})),
                 self._j(fit.get("fit_ids")),
                 self._j(fit.get("fit_label_kinds"))))
            return int(cur.lastrowid)

    def save_hosted_fit_once(self, fit: dict) -> dict:
        """Insert a one-time fit unless one already won the race (A4).

        The re-check and the insert run inside one BEGIN IMMEDIATE, so two
        threads/processes crossing the fit trigger together leave exactly
        one active fit row. Returns the active fit: ours, or the winner's
        (same point version) when we lost.
        """
        self.cx.execute("BEGIN IMMEDIATE")
        try:
            tier = fit.get("tier", "hosted_jev")
            row = self.cx.execute(
                """SELECT * FROM hosted_fit
                   WHERE point_id=? AND kind=? AND tier=? AND active=1
                   ORDER BY id DESC LIMIT 1""",
                (fit["point_id"], fit["kind"], tier)).fetchone()
            if row is not None:
                existing = self._parse_fit(row)
                if existing.get("point_version") == fit.get("point_version"):
                    self.cx.execute("ROLLBACK")
                    return existing
            self.cx.execute(
                "UPDATE hosted_fit SET active=0 WHERE point_id=? AND kind=? "
                "AND tier=?",
                (fit["point_id"], fit["kind"], tier))
            self.cx.execute(
                """INSERT INTO hosted_fit(point_id, point_version, kind,
                   tier, epoch_start_ts, fitted_at, n_fit, last_fit_ts,
                   last_fit_decision_id, last_fit_seq, temperature,
                   temperature_skipped, threshold, coverage, error_at_threshold,
                   fit_agreement, teacher_only_options, per_option, fit_ids,
                   fit_label_kinds, active)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                   ?, ?, ?, 1)""",
                (fit["point_id"], fit["point_version"], fit["kind"],
                 fit.get("tier", "hosted_jev"),
                 fit["epoch_start_ts"], fit["fitted_at"], fit["n_fit"],
                 fit["last_fit_ts"], fit["last_fit_decision_id"],
                 fit.get("last_fit_seq"),
                 fit["temperature"], fit.get("temperature_skipped"),
                 fit.get("threshold"), fit.get("coverage"),
                 fit.get("error_at_threshold"), fit.get("fit_agreement"),
                 json.dumps(fit.get("teacher_only_options", [])),
                 json.dumps(fit.get("per_option", {})),
                 self._j(fit.get("fit_ids")),
                 self._j(fit.get("fit_label_kinds"))))
            self.cx.execute("COMMIT")
        except Exception:
            try:
                self.cx.execute("ROLLBACK")
            except Exception:
                pass
            raise
        out = self.active_hosted_fit(fit["point_id"], fit["kind"],
                                       fit.get("tier", "hosted_jev"))
        assert out is not None
        return out

    @staticmethod
    def _parse_fit(row) -> dict:
        d = dict(row)
        d["active"] = bool(d["active"])
        for k in ("teacher_only_options", "per_option", "fit_ids"):
            d[k] = json.loads(d[k]) if d.get(k) is not None else (
                [] if k != "per_option" else {})
        # D9: plan fits record this; decide fits and pre-v11 rows carry None.
        d["fit_label_kinds"] = (json.loads(d["fit_label_kinds"])
                                if d.get("fit_label_kinds") is not None
                                else None)
        return d

    def active_hosted_fit(self, point_id: str,
                          kind: str = "decide",
                          tier: str = "hosted_jev") -> dict | None:
        row = self.cx.execute(
            """SELECT * FROM hosted_fit WHERE point_id=? AND kind=? AND tier=?
               AND active=1
               ORDER BY id DESC LIMIT 1""",
            (point_id, kind, tier)).fetchone()
        return self._parse_fit(row) if row else None

    def deactivate_hosted_fit(self, point_id: str,
                              kind: str | None = None,
                              tier: str | None = None) -> None:
        with self.cx:
            if kind is None and tier is None:
                self.cx.execute(
                    "UPDATE hosted_fit SET active=0 WHERE point_id=?",
                    (point_id,))
            elif tier is None:
                self.cx.execute(
                    "UPDATE hosted_fit SET active=0 WHERE point_id=? AND kind=?",
                    (point_id, kind))
            else:
                self.cx.execute(
                    "UPDATE hosted_fit SET active=0 WHERE point_id=? AND kind=? "
                    "AND tier=?",
                    (point_id, kind, tier))

    def purge_unregistered_traces(self, older_than_ts: str) -> int:
        with self.cx:
            cur = self.cx.execute(
                "DELETE FROM gateway_traces WHERE point_id IS NULL AND ts < ?",
                (older_than_ts,))
            return int(cur.rowcount)

    # -- transcripts + extractions -----------------------------------------
    def put_transcript(self, t: dict) -> None:
        with self.cx:
            self.cx.execute(
                """INSERT INTO transcripts(transcript_id, ts, run_id, workflow, model,
                   data_class, input, events, final_output, cost, run_outcome, plan_id,
                   eval_only)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (t["transcript_id"], t["ts"], t["run_id"], t["workflow"],
                 t.get("model", ""), t.get("data_class", "private"),
                 t.get("input", ""), self._j(t.get("events", [])),
                 t.get("final_output", ""), self._j(t.get("cost")),
                 self._j(t.get("run_outcome")), t.get("plan_id"),
                 1 if t.get("eval_only") else 0))

    def get_transcript(self, transcript_id: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM transcripts WHERE transcript_id=?",
                              (transcript_id,)).fetchone()
        return self._parse_transcript(row) if row else None

    def transcript_by_run(self, run_id: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM transcripts WHERE run_id=?",
                              (run_id,)).fetchone()
        return self._parse_transcript(row) if row else None

    def transcripts_for_workflow(self, workflow: str) -> list[dict]:
        return [self._parse_transcript(r) for r in self.cx.execute(
            "SELECT * FROM transcripts WHERE workflow=? ORDER BY ts, transcript_id",
            (workflow,))]

    @staticmethod
    def _parse_transcript(row) -> dict:
        d = dict(row)
        d["events"] = json.loads(d["events"]) if d["events"] else []
        if d.get("cost") is not None:
            d["cost"] = json.loads(d["cost"])
        if d.get("run_outcome") is not None:
            d["run_outcome"] = json.loads(d["run_outcome"])
        d["eval_only"] = bool(d.get("eval_only"))
        return d

    def eval_only_transcript_ids(self, workflow: str) -> set[str]:
        """Transcripts excluded from training and gating (pilot arm runs)."""
        return {r["transcript_id"] for r in self.cx.execute(
            "SELECT transcript_id FROM transcripts WHERE workflow=? AND eval_only=1",
            (workflow,))}

    def put_extraction(self, e: dict) -> None:
        with self.cx:
            self.cx.execute(
                """INSERT INTO extractions(extraction_id, ts, transcript_id, pass,
                   audit_required, status, extractor)
                   VALUES(?, ?, ?, ?, ?, ?, ?)""",
                (e["extraction_id"], e["ts"], e["transcript_id"], e.get("pass", 1),
                 int(bool(e.get("audit_required", False))), e.get("status", "open"),
                 e.get("extractor")))

    def get_extraction(self, extraction_id: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM extractions WHERE extraction_id=?",
                              (extraction_id,)).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["audit_required"] = bool(d["audit_required"])
        if d.get("rejected_codes") is not None:
            d["rejected_codes"] = json.loads(d["rejected_codes"])
        if d.get("disagreements") is not None:
            d["disagreements"] = json.loads(d["disagreements"])
        return d

    def complete_extraction(self, extraction_id: str, extractor: str,
                            n_accepted: int, n_rejected: int,
                            rejected_codes: dict, disagreements: list) -> None:
        with self.cx:
            self.cx.execute(
                """UPDATE extractions SET status='completed', extractor=?,
                   n_accepted=?, n_rejected=?, rejected_codes=?, disagreements=?
                   WHERE extraction_id=?""",
                (extractor, n_accepted, n_rejected, json.dumps(rejected_codes),
                 json.dumps(disagreements), extraction_id))

    def extractions_for_transcript(self, transcript_id: str) -> list[dict]:
        rows = self.cx.execute(
            "SELECT * FROM extractions WHERE transcript_id=? ORDER BY ts, extraction_id",
            (transcript_id,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["audit_required"] = bool(d["audit_required"])
            out.append(d)
        return out

    def audit_insert(self, a: dict) -> None:
        """Store one accepted pass-2 record (§15.2: never trains)."""
        with self.cx:
            self.cx.execute(
                """INSERT INTO extraction_audits(audit_id, ts, transcript_id,
                   extraction_id, point_id, record)
                   VALUES(?, ?, ?, ?, ?, ?)""",
                (a["audit_id"], a["ts"], a["transcript_id"], a["extraction_id"],
                 a["point_id"], json.dumps(a["record"])))

    def audits_for_transcript(self, transcript_id: str,
                              point_id: str | None = None) -> list[dict]:
        q = "SELECT * FROM extraction_audits WHERE transcript_id=?"
        args: tuple = (transcript_id,)
        if point_id is not None:
            q += " AND point_id=?"
            args = (transcript_id, point_id)
        out = []
        for r in self.cx.execute(q + " ORDER BY ts, audit_id", args):
            d = dict(r)
            d["record"] = json.loads(d["record"])
            out.append(d)
        return out

    def decisions_for_run(self, run_id: str) -> list[dict]:
        return [self._parse(r) for r in self.cx.execute(
            "SELECT * FROM decisions WHERE run_id=? ORDER BY rowid",
            (run_id,))]

    def decisions_for_transcript(self, transcript_id: str) -> list[dict]:
        return [self._parse(r) for r in self.cx.execute(
            "SELECT * FROM decisions WHERE transcript_id=? ORDER BY rowid",
            (transcript_id,))]

    def purge_transcripts(self, older_than_ts: str) -> int:
        """Drop content of old transcripts; metadata row stays (T-EX14).

        Every older row is purged, even one with empty input/final_output:
        its events can still hold private text (B13)."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE transcripts SET input='', events='[]', final_output='' "
                "WHERE ts < ?", (older_than_ts,))
            return int(cur.rowcount)

    def purge_extracted_text(self, older_than_ts: str) -> int:
        """Drop state_text of old extracted rows: up to 4,000 characters of
        transcript per row that no purge cleared (B13). Vectors stay."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE decisions SET state_text=NULL "
                "WHERE source='extracted' AND ts < ? "
                "AND state_text IS NOT NULL", (older_than_ts,))
            return int(cur.rowcount)

    def purge_plans(self, older_than_ts: str) -> int:
        """Drop input/final_output of old plan records (B13). Decisions,
        costs and check results stay."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE plan_records SET input='', final_output='' "
                "WHERE ts < ?", (older_than_ts,))
            return int(cur.rowcount)

    # -- workflows + plans -------------------------------------------------
    def put_workflow(self, w: dict) -> None:
        now = _utcnow()
        with self.cx:
            self.cx.execute(
                """INSERT INTO workflows(workflow, downgrade_requires, check_type,
                   check_script, run_model, cheap_model, downgrade_allowed, trial,
                   monitor, created_at, updated_at)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(workflow) DO UPDATE SET downgrade_requires=excluded.downgrade_requires,
                     check_type=excluded.check_type, check_script=excluded.check_script,
                     run_model=excluded.run_model, cheap_model=excluded.cheap_model,
                     downgrade_allowed=excluded.downgrade_allowed, trial=excluded.trial,
                     monitor=excluded.monitor, updated_at=excluded.updated_at""",
                (w["workflow"], json.dumps(w.get("downgrade_requires", [])),
                 w.get("check_type", "outcome"), w.get("check_script", ""),
                 w.get("run_model", ""), w.get("cheap_model", ""),
                 int(bool(w.get("downgrade_allowed", False))),
                 self._j(w.get("trial")), self._j(w.get("monitor")), now, now))

    def get_workflow(self, workflow: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM workflows WHERE workflow=?",
                              (workflow,)).fetchone()
        if row is None:
            return None
        return self._parse_workflow(row)

    def list_workflows(self) -> list[dict]:
        return [self._parse_workflow(r) for r in self.cx.execute(
            "SELECT * FROM workflows ORDER BY workflow")]

    @staticmethod
    def _parse_workflow(row) -> dict:
        d = dict(row)
        d["downgrade_requires"] = json.loads(d["downgrade_requires"])
        d["downgrade_allowed"] = bool(d["downgrade_allowed"])
        if d.get("trial") is not None:
            d["trial"] = json.loads(d["trial"])
        if d.get("monitor") is not None:
            d["monitor"] = json.loads(d["monitor"])
        return d

    def put_plan(self, p: dict) -> None:
        with self.cx:
            self.cx.execute(
                """INSERT INTO plan_records(plan_id, ts, workflow, task_ref, input,
                   context, context_source, run_model, cheap_model, recommendation,
                   model, addendum, points, would_have, audit, trial_id, pair_id, arm,
                   pilot_forced, cost, check_result, conflicts, followed_rate,
                   followed_n, point_versions, final_output, seq)
                   VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                           ?, ?, ?, ?, ?, ?,
                           (SELECT COALESCE(MAX(seq), 0) + 1 FROM plan_records))
                   ON CONFLICT(plan_id) DO UPDATE SET cost=excluded.cost,
                     check_result=excluded.check_result, conflicts=excluded.conflicts,
                     followed_rate=excluded.followed_rate,
                     followed_n=excluded.followed_n,
                     final_output=excluded.final_output""",
                (p["plan_id"], p["ts"], p["workflow"], p.get("task_ref", ""),
                 p.get("input", ""), self._j(p.get("context")),
                 p.get("context_source", "agent"), p.get("run_model", ""),
                 p.get("cheap_model", ""), p["recommendation"], p.get("model", ""),
                 p.get("addendum"), self._j(p.get("points", [])),
                 self._j(p.get("would_have")), int(bool(p.get("audit", False))),
                 p.get("trial_id"), p.get("pair_id"), p.get("arm"),
                 int(bool(p.get("pilot_forced", False))), self._j(p.get("cost")),
                 self._j(p.get("check_result")), json.dumps(p.get("conflicts", [])),
                 p.get("followed_rate"), p.get("followed_n", 0),
                 self._j(p.get("point_versions")), p.get("final_output")))

    def get_plan(self, plan_id: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM plan_records WHERE plan_id=?",
                              (plan_id,)).fetchone()
        return self._parse_plan(row) if row else None

    def plans_for_pair(self, pair_id: str) -> list[dict]:
        return [self._parse_plan(r) for r in self.cx.execute(
            "SELECT * FROM plan_records WHERE pair_id=? ORDER BY arm", (pair_id,))]

    def plans_for_trial(self, trial_id: str) -> list[dict]:
        return [self._parse_plan(r) for r in self.cx.execute(
            "SELECT * FROM plan_records WHERE trial_id=? ORDER BY ts, plan_id",
            (trial_id,))]

    def plans_for_workflow(self, workflow: str) -> list[dict]:
        return [self._parse_plan(r) for r in self.cx.execute(
            "SELECT * FROM plan_records WHERE workflow=? ORDER BY seq",
            (workflow,))]

    def plans_for_transcript_link(self, transcript_id: str) -> list[dict]:
        t = self.get_transcript(transcript_id)
        if t is None or not t.get("plan_id"):
            return []
        p = self.get_plan(t["plan_id"])
        return [p] if p else []

    def transcript_for_plan(self, plan_id: str) -> dict | None:
        row = self.cx.execute("SELECT * FROM transcripts WHERE plan_id=?",
                              (plan_id,)).fetchone()
        return self._parse_transcript(row) if row else None

    def transcript_plan_flags(self, workflow: str) -> dict[str, dict]:
        """transcript_id -> {"injected": bool, "pilot": bool} for transcripts
        linked to a plan (B12: injected runs parrot the addendum, so their
        rows must not train the plan probe or count toward min_examples)."""
        out: dict[str, dict] = {}
        for tid, addendum, pilot in self.cx.execute(
                """SELECT t.transcript_id, p.addendum, p.pilot_forced
                   FROM transcripts t JOIN plan_records p
                     ON t.plan_id = p.plan_id
                   WHERE t.workflow = ?""", (workflow,)):
            out[tid] = {"injected": addendum is not None and addendum != "",
                        "pilot": bool(pilot)}
        return out

    @staticmethod
    def _parse_plan(row) -> dict:
        d = dict(row)
        d["audit"] = bool(d["audit"])
        d["pilot_forced"] = bool(d["pilot_forced"])
        d["points"] = json.loads(d["points"]) if d["points"] else []
        d["conflicts"] = json.loads(d["conflicts"]) if d["conflicts"] else []
        for k in ("context", "would_have", "cost", "check_result", "point_versions"):
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        return d

    def text_purge_registered(self, older_than_ts: str) -> int:
        """Drop text of old registered rows; the compact record + vector stay."""
        with self.cx:
            cur = self.cx.execute(
                "UPDATE decisions SET state_text=NULL, context=NULL "
                "WHERE source='gateway' AND ts < ? AND state_text IS NOT NULL",
                (older_than_ts,))
            n = int(cur.rowcount)
            self.cx.execute(
                "UPDATE gateway_traces SET raw_request=NULL, raw_response=NULL "
                "WHERE point_id IS NOT NULL AND ts < ?", (older_than_ts,))
            return n


def _locked(fn):
    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            try:
                return fn(self, *args, **kwargs)
            finally:
                # End any open transaction (a no-op in autocommit mode): with
                # implicit transactions, SELECTs would otherwise hold one
                # snapshot and never see the other process's commits (the MCP
                # server and the gateway share this DB).
                if fn.__name__ != "close":
                    self.cx.commit()
    return wrapper


# Every connection touchpoint runs under the instance RLock (MCP/gateway threads).
for _name in ("version", "migrate", "close", "put_point", "get_point", "list_points",
              "insert_decision", "_trim", "update_decision", "get_decision",
              "decisions_for_point", "count_decisions", "save_model", "active_model",
              "trace_insert", "traces_by_discovery", "traces_for_point",
              "unregistered_traces",
              "counter_inc", "counter_get", "purge_unregistered_traces",
              "hosted_reserve", "hosted_settle", "hosted_release",
              "spend_get", "spend_open", "spend_add",
              "save_hosted_fit", "save_hosted_fit_once", "active_hosted_fit",
              "deactivate_hosted_fit",
              "text_purge_registered",
              "put_transcript", "get_transcript", "transcript_by_run",
              "transcripts_for_workflow", "put_extraction", "get_extraction",
              "complete_extraction", "extractions_for_transcript",
              "audit_insert", "audits_for_transcript",
              "decisions_for_run", "decisions_for_transcript", "purge_transcripts",
              "points_in_registration_order",
              "put_workflow", "get_workflow", "list_workflows", "put_plan", "get_plan",
              "plans_for_pair", "plans_for_trial", "plans_for_workflow",
              "plans_for_transcript_link", "transcript_for_plan",
              "meta_get", "meta_put",
              "autopilot_log_add", "autopilot_log_recent",
              "job_put", "job_get", "job_update", "jobs_running",
              "jobs_latest", "smoke_done_exists",
              "job_insert_launching", "job_claim_launch", "job_finalize",
              "job_note_status_failure", "job_reset_status_failures",
              "jobs_launching", "latest_autopilot_failure_at",
              "cas_point_phase", "try_record_teacher", "try_set_outcome",
              "eval_only_transcript_ids", "purge_extracted_text",
              "purge_plans", "transcript_plan_flags",
              "autopilot_month_counted"):
    setattr(Store, _name, _locked(getattr(Store, _name)))
del _name
