"""M16 autopilot tests (T-AP1..T-AP13): promote/train on their own within
user-approved limits. Fakes only: FakeDecider/FakeHostedTransport/FakeTrainer,
a fake clock (service._utcnow), an injected monotonic, and file-backed
notify_cmd / approve TTY doubles."""
import json
import random
import sys
import time

import pytest

import decideonce.service as S
from decideonce import train as TR
from decideonce.autopilot import (
    APPROVED_AT_KEY,
    APPROVED_HASH_KEY,
    APPROVED_NOTIFY_KEY,
    APPROVED_SIG_KEY,
    TRUST_FP_KEY,
    effective_hash,
    fingerprint_pubkey,
    run_approve,
    run_keygen,
    section_hash,
    sign_approval,
)
from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.deciders.fake import FakeDecider
from decideonce.deciders.hosted import FakeHostedTransport, HostedJevDecider
from decideonce.service import Service
from decideonce.store import Store
from decideonce.trainer import FakeTrainer


@pytest.fixture
def clock(monkeypatch):
    state = {"t": 0}
    base = 1761950000  # 2025-10-31 UTC: before the free-training cut-off

    def tick():
        state["t"] += 1
        return time.strftime("%Y-%m-%dT%H:%M:%SZ",
                             time.gmtime(base + state["t"]))

    monkeypatch.setattr(S, "_utcnow", tick)
    return state


@pytest.fixture
def mono():
    return {"t": 1000.0}


# Test signing key (deterministic): 32 bytes 0..31.
import base64 as _b64

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey as _TestPriv)

_TEST_PRIV_RAW = bytes(range(32))
_TEST_PRIV = _TestPriv.from_private_bytes(_TEST_PRIV_RAW)
_TEST_PUB_B64 = _b64.b64encode(_TEST_PRIV.public_key().public_bytes_raw()).decode()
_TEST_PRIV_B64 = _b64.b64encode(_TEST_PRIV_RAW).decode()
_TEST_FP = fingerprint_pubkey(_TEST_PUB_B64)


def _sign_cfg(cfg, store):
    """Approve cfg's effective limits with the test key (direct, no TTY)."""
    cfg.setdefault("autopilot_trust", {})["public_key"] = _TEST_PUB_B64
    h = effective_hash(cfg)
    at = S._utcnow()
    project = (cfg.get("store", {}) or {}).get("project", "default")
    sig = sign_approval(_TEST_PRIV_RAW, project, h, at)
    store.meta_put(APPROVED_HASH_KEY, h)
    store.meta_put(APPROVED_SIG_KEY, sig)
    store.meta_put(APPROVED_AT_KEY, at)
    store.meta_put(APPROVED_NOTIFY_KEY,
                   json.dumps((cfg.get("autopilot", {}) or {}).get("notify_cmd", "")))
    store.meta_put(TRUST_FP_KEY, _TEST_FP)


def make_svc(tmp_path, backend="fake", trainer=None, mono=None, **over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = backend
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    for section, values in over.items():
        cfg[section].update(values)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0), trainer=trainer,
                  monotonic=(lambda: mono["t"]) if mono is not None else None)
    return svc, store, cfg


def make_scripted(tmp_path, schedule, private_schedule=None, trainer=None,
                  mono=None, **over):
    """Tiered service with a scripted hosted fn (+ scripted private dual)."""
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "tiered"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    for section, values in over.items():
        cfg[section].update(values)
    store = Store(tmp_path / "proj")
    hosted = HostedJevDecider(
        url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
        key="fake", store=store, timeout_s=3,
        price_per_mtok_in=cfg["hosted_jev"]["price_per_mtok_in"],
        monthly_budget_usd=cfg["hosted_jev"]["monthly_budget_usd"],
        reserve_usd=cfg["hosted_jev"]["reserve_usd"],
        transport=FakeHostedTransport(fn=scripted_hosted(schedule)))
    deciders = {"hosted_jev": hosted}
    if private_schedule is not None:
        deciders["private"] = FakeDecider(
            fn=scripted_private(private_schedule), decider_tag="private",
            model_tag="fake-private-1")
    svc = Service(cfg, store, deciders, rng=random.Random(0), trainer=trainer,
                  monotonic=(lambda: mono["t"]) if mono is not None else None)
    return svc, store, cfg


def scripted_hosted(schedule):
    """schedule(i, hint, opts) -> (top, p_top); i = call index."""
    state = {"n": 0}

    def fn(body):
        (pid, q), = body["questions"].items()
        opts = list(q["criteria"])
        hint = (body.get("state") or {}).get("label_hint")
        i = state["n"]
        state["n"] += 1
        top, p_top = schedule(i, hint, opts)
        rest = (1.0 - p_top) / (len(opts) - 1)
        probs = {o: (p_top if o == top else rest) for o in opts}
        return 200, {"answers": {pid: {"choice": top,
                                       "probabilities": probs,
                                       "confidence": p_top}},
                     "usage": {"input_tokens": 50, "output_tokens": 0,
                               "cost": 0.0001}}

    return fn


def scripted_private(schedule):
    """schedule(i, hint, opts) -> (top, p_top); i = dual-call index."""
    state = {"n": 0}

    def fn(req):
        opts = list(req.criteria)
        hint = req.state.get("label_hint")
        i = state["n"]
        state["n"] += 1
        top, p_top = schedule(i, hint, opts)
        rest = (1.0 - p_top) / (len(opts) - 1)
        return {o: (p_top if o == top else rest) for o in opts}

    return fn


def other(hint, opts):
    return [o for o in opts if o != hint][0]


def point(pid="ap.pt", **kw):
    p = {"point_id": pid, "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def decide_label(svc, pid, label, hint=None):
    r = svc.do_decide({"point_id": pid,
                       "context": {"label_hint": label if hint is None
                                 else hint}})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec
    return r["decision_id"]


def notify_cmd_for(path):
    return [sys.executable, "-c",
            "import sys;a=sys.argv;open(a[1],'a').write("
            "(a[3] if len(a)>3 and a[2]=='--' else a[2])+'\\n')",
            str(path)]


def enable(cfg, store, notify_path=None, **over):
    """Set [autopilot] limits and approve them (directly, not via TTY)."""
    section = {"enabled": True, "points": ["ap.*"], "max_target_error": 0.5,
               "auto_serve": True, "auto_train": False, "max_job_usd": 10.0,
               "max_monthly_jobs_usd": 100.0, "min_new_rows": 5,
               "free_training_until": "", "cooldown_days": 7,
               "check_interval_min": 10, "notify_cmd": ""}
    section.update(over)
    if notify_path is not None:
        section["notify_cmd"] = notify_cmd_for(notify_path)
    cfg["autopilot"].update(section)
    _sign_cfg(cfg, store)
    return cfg["autopilot"]


def notify_lines(path):
    from decideonce.autopilot import sync_notifies

    sync_notifies()
    try:
        return path.read_text().splitlines()
    except FileNotFoundError:
        return []


# -- T-AP1: off means v1.8 ----------------------------------------------------


def test_T_AP1_disabled_is_v18(tmp_path, clock, mono):
    svc, store, cfg = make_svc(tmp_path, mono=mono)
    notify_path = tmp_path / "notify.log"
    cfg["autopilot"]["notify_cmd"] = notify_cmd_for(notify_path)
    assert cfg["autopilot"]["enabled"] is False
    assert cfg["autopilot"]["points"] == []
    assert cfg["fireworks"]["monthly_budget_usd"] == 0
    assert cfg["fireworks"]["allow_paid_training"] is False
    assert svc.do_register(point())["ok"]
    # record/outcome cycles change nothing by themselves.
    for i in range(6):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    assert store.get_point("ap.pt")["phase"] == "collect"
    assert store.autopilot_log_recent() == []
    assert store.jobs_running() == [] and store.jobs_latest() is None
    assert notify_lines(notify_path) == []
    assert svc.autopilot.check() == {"checked": [], "actions": []}
    rep = svc.do_report({})["points"][0]
    assert rep["autopilot"] == "blocked(disabled)"
    st = svc.do_autopilot_status({})
    assert st["ok"] and st["enabled"] is False
    assert st["eligible_points"] == []
    assert st["points"]["ap.pt"]["status"] == "blocked(disabled)"


# -- T-AP2: collect -> shadow -> serve ----------------------------------------


def serve_schedule(wrong=(), low=()):
    """Correct 0.9 except wrong@0.6 (errors) and low@0.5 (below threshold)."""

    def sched(i, hint, opts):
        if i in wrong:
            return other(hint, opts), 0.6
        if i in low:
            return hint, 0.5
        return hint, 0.9

    return sched


def test_T_AP2_collect_shadow_serve_no_confirm(tmp_path, clock, mono):
    notify_path = tmp_path / "notify.log"
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong={24, 25}), mono=mono,
        hosted_jev={"fit_rows": 20}, serve={"min_auto_support": 5},
        train={"min_examples": 10})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store, notify_path)
    assert svc.do_register(point(target_error=0.3))["ok"]
    # 20 fit rows + 14 evidence rows (12 @0.9 + 2 wrong@0.6 for the margin).
    for i in range(34):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    assert store.get_point("ap.pt")["phase"] == "shadow"  # via first kick
    assert store.active_hosted_fit("ap.pt")["threshold"] == 0.9
    assert svc.do_report({"point_id": "ap.pt"})["points"][0][
        "autopilot"] == "on"
    out = svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "serve"
    assert [a["action"] for a in out["actions"]] == ["promoted_to_serve"]
    log = store.autopilot_log_recent(5)
    assert [e["action"] for e in log] == ["promoted_to_serve",
                                          "promoted_to_shadow"]
    assert log[0]["config_hash"] == effective_hash(cfg)
    assert "Wilson upper" in log[0]["gates"]["evidence"]["detail"]
    lines = notify_lines(notify_path)
    assert any("promoted to shadow" in ln for ln in lines)
    serve_line = next(ln for ln in lines if "promoted to serve" in ln)
    assert "Wilson upper" in serve_line and "margin" in serve_line
    # And it actually serves automatically now.
    r = svc.do_decide({"point_id": "ap.pt", "context": {"label_hint": "a"}})
    assert r["mode"] == "auto" and r["choice"] == "a"


# -- T-AP3: never promotes the excluded ---------------------------------------


def test_T_AP3_excluded_points_never_promoted(tmp_path, clock, mono):
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(), mono=mono, hosted_jev={"fit_rows": 20},
        serve={"min_auto_support": 5})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store, max_target_error=0.3)
    assert svc.do_register(point("ap.irr", irreversible=True))["ok"]
    assert svc.do_register(point("ap.sus", suspected=True))["ok"]
    assert svc.do_register(point("ap.unr"))["ok"]
    assert svc.do_register(point("other.pt"))["ok"]
    assert svc.do_register(point("ap.lax", target_error=0.4))["ok"]
    for pid in ("ap.irr", "ap.sus", "ap.unr", "other.pt", "ap.lax"):
        for i in range(24):
            decide_label(svc, pid, "ab"[i % 2])
    # two teacher/outcome disagreements -> teacher_unreliable.
    for did in [d["decision_id"]
                for d in store.decisions_for_point("ap.unr")[:2]]:
        row = store.get_decision(did)
        flipped = "b" if row["teacher_choice"] == "a" else "a"
        assert svc.do_outcome({"decision_id": did,
                               "correct_choice": flipped})["ok"]
    svc.autopilot_check()
    assert store.get_point("ap.irr")["phase"] == "shadow"  # collect ok...
    assert store.get_point("ap.sus")["phase"] == "collect"
    # ap.unr reached shadow while still reliable; now it is stuck there.
    assert store.get_point("ap.unr")["phase"] == "shadow"
    assert store.get_point("other.pt")["phase"] == "collect"
    assert store.get_point("ap.lax")["phase"] == "collect"
    # ...but serve is barred for all of them, each with its reason.
    rep = {p["point_id"]: p["autopilot"]
           for p in svc.do_report({})["points"]}
    assert rep["ap.irr"] == "blocked(irreversible)"
    assert rep["ap.sus"] == "blocked(suspected)"
    assert rep["ap.unr"] == "blocked(teacher_unreliable)"
    assert rep["other.pt"] == "blocked(not_in_points)"
    assert rep["ap.lax"] == "blocked(target_error_above_limit)"
    # allow_auto_irreversible does not move autopilot (reapprove: E5 covers serve).
    cfg["serve"]["allow_auto_irreversible"] = True
    reapprove(cfg, store)
    svc.autopilot_check()
    assert store.get_point("ap.irr")["phase"] == "shadow"
    st = svc.do_autopilot_status({})
    assert st["points"]["ap.irr"]["waiting_for"] == "irreversible"


# -- T-AP4: approval ----------------------------------------------------------


def test_T_AP4_hash_change_stops_and_notifies(tmp_path, clock, mono):
    notify_path = tmp_path / "notify.log"
    svc, store, cfg = make_scripted(tmp_path, serve_schedule(), mono=mono)
    enable(cfg, store, notify_path)
    assert svc.do_register(point())["ok"]
    decide_label(svc, "ap.pt", "a")
    assert store.get_point("ap.pt")["phase"] == "shadow"
    notify_lines(notify_path)  # flush async promotion notice
    notify_path.write_text("")  # only the stop below may notify
    cfg["autopilot"]["cooldown_days"] = 8  # edited after approval
    out = svc.autopilot_check()
    assert out["actions"] == []
    assert store.get_point("ap.pt")["phase"] == "shadow"
    rep = svc.do_report({"point_id": "ap.pt"})["points"][0]
    assert rep["autopilot"] == "blocked(not_approved)"
    block = svc.do_report({})["autopilot"]
    assert block["match"] is False
    assert block["approved_hash"] != block["current_hash"]
    lines = notify_lines(notify_path)
    assert len(lines) == 1 and "stopped" in lines[0]
    svc.autopilot_check()  # no duplicate notification
    assert notify_lines(notify_path) == lines


def test_T_AP4_approve_tty_flow(tmp_path, clock):
    svc, store, cfg = make_svc(tmp_path)
    cfg["autopilot_trust"]["public_key"] = _TEST_PUB_B64
    out: list[str] = []
    # Non-TTY prints no hash (F2).
    assert run_approve(cfg, store, is_tty=False,
                       out=out.append) == "not_interactive"
    assert store.meta_get(APPROVED_HASH_KEY) is None
    assert not any("SHA-256" in ln for ln in out)
    assert not any(effective_hash(cfg) in ln for ln in out)
    # TTY without a trust key fails; with the key + stubbed envsave it signs.
    cfg["autopilot_trust"]["public_key"] = ""
    assert run_approve(cfg, store, is_tty=True, out=out.append,
                       env_runner=lambda argv: (0, "  DECIDE_AUTOPILOT_SIGNING_KEY = "
                                                   + _TEST_PRIV_B64 + "\n")) == "error"
    cfg["autopilot_trust"]["public_key"] = _TEST_PUB_B64
    h = effective_hash(cfg)
    assert run_approve(cfg, store, is_tty=True, out=out.append,
                       env_runner=lambda argv: (0, "  DECIDE_AUTOPILOT_SIGNING_KEY = "
                                                   + _TEST_PRIV_B64 + "\n")) == "ok"
    assert store.meta_get(APPROVED_HASH_KEY) == h
    assert store.meta_get(APPROVED_SIG_KEY)
    assert any("SHA-256" in ln for ln in out)


def test_S4_signing_key_rename_and_fallback(tmp_path, clock):
    from decideonce.autopilot import (SIGNING_KEY_NAME,
                                      SIGNING_KEY_NAME_LEGACY)
    assert SIGNING_KEY_NAME == "DECIDEONCE_AUTOPILOT_SIGNING_KEY"
    assert SIGNING_KEY_NAME_LEGACY == "DECIDE_AUTOPILOT_SIGNING_KEY"
    # New name answers: the legacy name is never consulted.
    svc, store, cfg = make_svc(tmp_path / "new")
    cfg["autopilot_trust"]["public_key"] = _TEST_PUB_B64
    seen: list[str] = []

    def runner_new(argv):
        seen.append(argv[-1])
        if argv[-1] == SIGNING_KEY_NAME:
            return 0, f"  {SIGNING_KEY_NAME} = {_TEST_PRIV_B64}\n"
        return 0, "\n"

    assert run_approve(cfg, store, is_tty=True, out=lambda m: None,
                       env_runner=runner_new) == "ok"
    assert seen == [SIGNING_KEY_NAME]
    # Only the old name answers: the fallback signs.
    svc2, store2, cfg2 = make_svc(tmp_path / "old")
    cfg2["autopilot_trust"]["public_key"] = _TEST_PUB_B64
    seen2: list[str] = []

    def runner_old(argv):
        seen2.append(argv[-1])
        if argv[-1] == SIGNING_KEY_NAME_LEGACY:
            return 0, f"  {SIGNING_KEY_NAME_LEGACY} = {_TEST_PRIV_B64}\n"
        return 0, "\n"

    assert run_approve(cfg2, store2, is_tty=True, out=lambda m: None,
                       env_runner=runner_old) == "ok"
    assert seen2 == [SIGNING_KEY_NAME, SIGNING_KEY_NAME_LEGACY]


def test_T_AP4_approve_cli(tmp_path, clock, monkeypatch, capsys):
    from decideonce import server as SRV

    toml_path = tmp_path / "decideonce.toml"
    toml_path.write_text("[store]\n"
                         f"data_dir = {json.dumps(str(tmp_path / 'data'))}\n"
                         "project = \"ap\"\n"
                         "[autopilot]\n"
                         "enabled = true\n"
                         "points = [\"ap.*\"]\n")
    monkeypatch.setenv("DECIDEONCE_CONFIG", str(toml_path))

    class Stdin:
        def __init__(self, tty):
            self._tty = tty

        def isatty(self):
            return self._tty

    monkeypatch.setattr(sys, "stdin", Stdin(False))
    with pytest.raises(SystemExit) as e:
        SRV._cli(["autopilot", "approve"])
    assert e.value.code == 2
    captured = capsys.readouterr().out
    assert "not_interactive" in captured
    assert "SHA-256" not in captured

    with pytest.raises(SystemExit) as e:
        SRV._cli(["bogus"])
    assert e.value.code == 2


def test_F_trust_valid_and_tampered(tmp_path, clock):
    svc, store, cfg = make_svc(tmp_path)
    enable(cfg, store)
    assert svc.autopilot.global_state() == "ok"
    # Tampered limits.
    cfg["serve"]["min_shadow"] = 9999
    assert svc.autopilot.global_state() == "not_approved"
    assert svc.autopilot.point_status({"point_id": "ap.pt"})[1] == "not_approved"
    reapprove(cfg, store)
    assert svc.autopilot.global_state() == "ok"
    # Tampered signature.
    store.meta_put(APPROVED_SIG_KEY, "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==")
    assert svc.autopilot.global_state() == "not_approved"
    reapprove(cfg, store)
    # Tampered public key -> trust_key_changed.
    other = _TestPriv.from_private_bytes(bytes([9] * 32))
    other_pub = _b64.b64encode(other.public_key().public_bytes_raw()).decode()
    cfg["autopilot_trust"]["public_key"] = other_pub
    assert svc.autopilot.global_state() == "trust_key_changed"
    assert svc.autopilot.point_status(
        {"point_id": "ap.pt"})[1] == "trust_key_changed"


def test_F_keygen_stub(tmp_path, clock):
    toml = tmp_path / "k.toml"
    toml.write_text("[autopilot]\nenabled = true\n")
    load_config(str(toml))
    store = Store(tmp_path / "proj")
    out: list[str] = []
    seen = {}

    def runner(argv, value=None):
        seen["argv"] = argv
        seen["value"] = value
        assert argv[:2] == ["envsave", "set"]
        assert value and len(value) == 44  # base64 32 bytes, never on argv
        assert value not in " ".join(argv)
        return 0, "Added: DECIDEONCE_AUTOPILOT_SIGNING_KEY (es_testkey)\n"

    assert run_keygen(str(toml), store, out=out.append,
                      env_runner=runner) == "ok"
    assert seen["argv"][2] == "DECIDEONCE_AUTOPILOT_SIGNING_KEY"
    text = toml.read_text()
    assert "[autopilot_trust]" in text and "public_key" in text
    assert any("fingerprint" in ln for ln in out)


def test_F1_approved_notify_only(tmp_path, clock, mono):
    good = tmp_path / "good.log"
    evil = tmp_path / "evil.log"
    svc, store, cfg = make_scripted(tmp_path, serve_schedule(), mono=mono)
    enable(cfg, store, good)
    assert svc.do_register(point())["ok"]
    decide_label(svc, "ap.pt", "a")
    assert store.get_point("ap.pt")["phase"] == "shadow"
    notify_lines(good)  # flush async promotion notice
    good.write_text("")
    # Edit the limits AND the notify command after approval: the stop notice
    # must run the approved command only, never the edited one.
    cfg["autopilot"]["cooldown_days"] = 8
    cfg["autopilot"]["notify_cmd"] = notify_cmd_for(evil)
    svc.autopilot_check()
    assert svc.autopilot.global_state() == "not_approved"
    assert len(notify_lines(good)) == 1
    assert notify_lines(evil) == []


# -- training helpers --------------------------------------------------------


def train_point(pid="ap.tr", **kw):
    p = {"point_id": pid, "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "private",
         "private_model_only": True, "remote_ok": ["fireworks"],
         "confirm": True}
    p.update(kw)
    return p


def train_setup(tmp_path, n_rows=10, estimate_override=None, **fw_over):
    """Tiered + FakeTrainer + approved autopilot (train only, no serve)."""
    trainer_holder: dict = {}

    def _build():
        cfg = load_config()
        cfg["store"]["data_dir"] = str(tmp_path)
        cfg["decider"]["backend"] = "tiered"
        cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
        cfg["train"]["min_train"] = 10
        cfg["fireworks"]["monthly_budget_usd"] = 100.0
        cfg["fireworks"].update(fw_over)
        store = Store(tmp_path / "proj")
        trainer = FakeTrainer(cfg, estimate_override=estimate_override)
        trainer_holder["trainer"] = trainer
        hosted = HostedJevDecider(
            url=cfg["hosted_jev"]["url"], model=cfg["hosted_jev"]["model"],
            key="fake", store=store, timeout_s=3,
            price_per_mtok_in=cfg["hosted_jev"]["price_per_mtok_in"],
            monthly_budget_usd=cfg["hosted_jev"]["monthly_budget_usd"],
            reserve_usd=cfg["hosted_jev"]["reserve_usd"],
            transport=FakeHostedTransport(fn=scripted_hosted(
                lambda i, h, o: (h, 0.9))))
        svc = Service(cfg, store, {"hosted_jev": hosted},
                      rng=random.Random(0), trainer=trainer)
        return svc, store, cfg

    svc, store, cfg = _build()
    enable(cfg, store, auto_serve=False, auto_train=True)
    assert svc.do_register(train_point())["ok"]
    for i in range(n_rows):
        decide_label(svc, "ap.tr", "ab"[i % 2])
    return svc, store, cfg, trainer_holder["trainer"]


def fake_month():
    return S._utcnow()[:7]


def reapprove(cfg, store):
    _sign_cfg(cfg, store)


# -- T-AP5: training budgets --------------------------------------------------


def test_T_AP5_smoke_then_full_on_free_path(tmp_path, clock):
    notify_path = tmp_path / "notify.log"
    svc, store, cfg, trainer = train_setup(
        tmp_path, estimate_override=1.0, allow_paid_training=False)
    cfg["autopilot"]["notify_cmd"] = notify_cmd_for(notify_path)
    reapprove(cfg, store)
    # First launch is the smoke job (full estimate counts, never free).
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("on", "train")
    out = svc.autopilot_check()
    assert [a["action"] for a in out["actions"]] == ["train_launched"]
    job = store.jobs_latest()
    assert job["kind"] == "smoke" and job["status"] == "running"
    assert job["counted_usd"] == 1.0 and job["autopilot"] is True
    assert len(trainer.uploaded) == 1
    assert len(trainer.uploaded[0]["rows"]) == 10
    assert all(r["point_id"] == "ap.tr" for r in trainer.uploaded[0]["rows"])
    trainer.complete(job["job_id"])
    svc.autopilot_check()
    job = store.job_get(job["job_id"])
    assert job["status"] == "smoke_done"
    assert store.spend_get(fake_month(), "fireworks")["usd"] == 1.0
    assert store.active_model("ap.tr", "private") is None  # smoke: no card
    # Full job on the free path: training $0, the eval part counts.
    cfg["autopilot"]["free_training_until"] = "2026-06-01T00:00:00-08:00"
    reapprove(cfg, store)
    out = svc.autopilot_check()
    assert [a["action"] for a in out["actions"]] == ["train_launched"]
    full = store.jobs_latest()
    assert full["kind"] == "full" and full["status"] == "running"
    assert full["counted_usd"] < 1.0  # eval part only
    assert full["estimate_usd"] == 1.0  # full E still recorded
    trainer.complete(full["job_id"])
    svc.autopilot_check()
    full = store.job_get(full["job_id"])
    assert full["status"] == "done"
    card_path = store.models_dir / "ap.tr" / "v1-1.json"
    assert card_path.is_file()
    model = store.active_model("ap.tr", "private")
    assert model is not None and model["card"]["job_id"] == full["job_id"]
    assert model["card"]["export_ts"] != ""
    pt = store.get_point("ap.tr")
    assert pt["last_train_export_ts"] == model["card"]["export_ts"]
    spent = store.spend_get(fake_month(), "fireworks")["usd"]
    assert spent == pytest.approx(1.0 + full["counted_usd"])
    actions = [e["action"] for e in store.autopilot_log_recent(10)]
    assert actions == ["trained", "train_launched", "smoke_done",
                       "train_launched"]
    lines = notify_lines(notify_path)
    assert any("smoke job done" in ln for ln in lines)
    assert any("trained (job" in ln for ln in lines)


def test_T_AP5_free_expiry_and_cutoff(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path, estimate_override=1.0)
    # run the smoke job to completion first.
    svc.autopilot_check()
    assert store.jobs_latest()["kind"] == "smoke"
    trainer.complete(store.jobs_latest()["job_id"])
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "smoke_done"
    # free_training_until in the past -> refused (paid not allowed).
    cfg["autopilot"]["free_training_until"] = "2025-01-01T00:00:00-08:00"
    reapprove(cfg, store)
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(budget)", "budget")
    assert svc.autopilot_check()["actions"] == []
    # free_training_until in the future, but past the main cut-off -> refused.
    cfg["autopilot"]["free_training_until"] = "2027-06-01T00:00:00-08:00"
    clock["t"] += 500 * 86400  # now past 2026-12-01 Pacific
    reapprove(cfg, store)
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(budget)", "budget")
    assert svc.autopilot_check()["actions"] == []
    assert len(trainer.uploaded) == 1  # smoke only; no full upload


def test_T_AP5_paid_refusals(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path, estimate_override=5.0)
    cfg["autopilot"]["max_job_usd"] = 2.0
    reapprove(cfg, store)
    # smoke exceeds max_job_usd.
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(budget)", "budget")
    assert svc.autopilot_check()["actions"] == []
    assert trainer.uploaded == []
    # paid path refused without allow_paid_training (full after smoke).
    cfg["autopilot"]["max_job_usd"] = 10.0
    reapprove(cfg, store)
    svc.autopilot_check()
    assert store.jobs_latest()["kind"] == "smoke"
    trainer.complete(store.jobs_latest()["job_id"])
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "smoke_done"
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(budget)", "budget")
    # ...and allowed with it.
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    out = svc.autopilot_check()
    assert [a["action"] for a in out["actions"]] == ["train_launched"]
    assert store.jobs_latest()["kind"] == "full"


def test_Y14_paid_full_needs_post_smoke_reapproval(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path, estimate_override=1.0)
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    svc.autopilot_check()  # smoke launches
    assert store.jobs_latest()["kind"] == "smoke"
    trainer.complete(store.jobs_latest()["job_id"])
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "smoke_done"
    # Paid and within limits, but no post-smoke approval: blocked.
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(reapproval)", "reapproval")
    assert svc.autopilot_check()["actions"] == []
    assert len(trainer.uploaded) == 1  # smoke only
    # Re-approving after the smoke unblocks the full job.
    reapprove(cfg, store)
    out = svc.autopilot_check()
    assert [a["action"] for a in out["actions"]] == ["train_launched"]
    assert store.jobs_latest()["kind"] == "full"


def test_T_AP5_month_cap(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path, estimate_override=1.0)
    cfg["autopilot"]["max_monthly_jobs_usd"] = 1.0
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    svc.autopilot_check()  # smoke: 1.0 + 0 <= 1.0
    assert store.jobs_latest()["kind"] == "smoke"
    trainer.complete(store.jobs_latest()["job_id"])
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "smoke_done"
    # full would add ~1.0 more -> over the monthly cap.
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(budget)", "budget")
    assert svc.autopilot_check()["actions"] == []
    assert store.autopilot_month_counted(fake_month()) == 1.0


def test_T_AP5_manual_train_flow(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path)
    r = svc.do_train({"point_id": "ap.tr", "dry_run": True})
    assert r["ok"] and r["kind"] == "smoke"
    assert r["joins"][0]["point_id"] == "ap.tr"
    assert r["estimate"]["estimate_usd"] > 0
    assert trainer.uploaded == []  # dry_run never uploads
    r = svc.do_train({"point_id": "ap.tr"})
    assert r["ok"] is False and r["error"] == "confirm_required"
    cfg["fireworks"]["monthly_budget_usd"] = 0.0  # no spending approval
    r = svc.do_train({"point_id": "ap.tr", "confirm": True})
    assert r["ok"] is False and r["error"] == "budget_exceeded"
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    r = svc.do_train({"point_id": "ap.tr", "confirm": True})
    assert r["ok"] and r["status"] == "running"
    r = svc.do_train({"point_id": "ap.tr", "confirm": True})
    assert r["ok"] and r["status"] == "running"  # polled while running
    trainer.complete(r["job_id"])
    r = svc.do_train({"point_id": "ap.tr", "confirm": True})
    assert r["ok"] and r["status"] == "smoke_done"
    # full needs the paid-job gate.
    r = svc.do_train({"point_id": "ap.tr", "confirm": True})
    assert r["ok"] is False and r["error"] == "paid_job_not_approved"
    r = svc.do_train({"point_id": "ap.tr", "confirm": True,
                      "training_free_confirmed": True})
    assert r["ok"] and r["status"] == "running" and r["kind"] == "full"


# -- T-AP6: one job at a time, retrain needs new rows -------------------------


def test_T_AP6_job_running_and_min_new_rows(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path)
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "running"
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(job_running)", "job_running")
    trainer.complete(store.jobs_latest()["job_id"])
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "smoke_done"
    reapprove(cfg, store)  # Y14: the full job needs a post-smoke approval
    svc.autopilot_check()  # full launches (no marker yet)
    assert store.jobs_latest()["kind"] == "full"
    trainer.complete(store.jobs_latest()["job_id"])
    svc.autopilot_check()
    assert store.jobs_latest()["status"] == "done"
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(not_enough_new_rows)",
                                      "not_enough_new_rows")
    for i in range(5):
        decide_label(svc, "ap.tr", "ab"[i % 2])
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("on", "train")
    out = svc.autopilot_check()
    assert [a["action"] for a in out["actions"]] == ["train_launched"]


# -- T-AP7: privacy -----------------------------------------------------------


def test_T_AP7_joined_privacy_and_no_decider(tmp_path, clock):
    svc, store, cfg, trainer = train_setup(tmp_path)
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    # point B joins by row count but must never leave the machine.
    assert svc.do_register(train_point("ap.joined", private_model_only=False,
                                       remote_ok=[]))["ok"]
    for i in range(10):
        decide_label(svc, "ap.joined", "ab"[i % 2])
    joins = {j["point_id"]: j for j in svc.autopilot.dry_run(
        store.get_point("ap.tr"))["joins"]}
    assert joins["ap.joined"]["allowed"] is False
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(privacy)", "privacy")
    assert svc.autopilot_check()["actions"] == []
    assert trainer.uploaded == []  # blocked: no upload at all
    # manual launch excludes B's rows instead.
    r = svc.do_train({"point_id": "ap.tr", "confirm": True})
    assert r["ok"] and r["status"] == "running"
    assert {row["point_id"] for row in trainer.uploaded[-1]["rows"]} == {"ap.tr"}
    # a private point without hosted_jev stays in collect.
    assert svc.do_register(point("ap.priv", data_class="private"))["ok"]
    cfg["autopilot"]["auto_serve"] = True
    reapprove(cfg, store)
    svc.autopilot_check()
    assert store.get_point("ap.priv")["phase"] == "collect"
    assert svc.do_autopilot_status(
        {})["points"]["ap.priv"]["waiting_for"] == "no_decider"


# -- T-AP8: cooldown ----------------------------------------------------------


def test_T_AP8_cooldown_after_drift_and_manual_shadow(tmp_path, clock, mono):
    notify_path = tmp_path / "notify.log"
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong={24, 25, 63, 64, 77, 78}), mono=mono,
        hosted_jev={"fit_rows": 20}, serve={"min_auto_support": 5},
        train={"min_examples": 10})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["drift_window"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store, notify_path)
    assert svc.do_register(point(target_error=0.3))["ok"]
    for i in range(34):  # 20 fit + 14 evidence
        decide_label(svc, "ap.pt", "ab"[i % 2])
    svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "serve"
    # five bad audits -> drift demotion, logged and announced.
    cfg["serve"]["audit_rate"] = 1.0
    reapprove(cfg, store)
    mono["t"] += 61
    for i in range(5):
        hint = "ab"[i % 2]
        r = svc.do_decide({"point_id": "ap.pt",
                           "context": {"label_hint": hint}})
        assert r["audit"] is True
        rec = svc.do_record({"decision_id": r["decision_id"],
                             "choice": "b" if hint == "a" else "a"})
        assert rec["ok"], rec
    assert store.get_point("ap.pt")["phase"] == "shadow"
    assert store.get_point("ap.pt")["autopilot_demotion_ts"] != ""
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["waiting_for"] == "cooldown"
    assert any(e["action"] == "demoted"
               for e in store.autopilot_log_recent(10))
    assert any("demoted to shadow (drift)" in ln
               for ln in notify_lines(notify_path))
    cfg["serve"]["audit_rate"] = 0.0
    reapprove(cfg, store)
    mono["t"] += 61
    svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "shadow"  # still cooling
    # after the cooldown, pre-demotion evidence does not count...
    clock["t"] += 8 * 86400
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["waiting_for"] == "gate:1"
    # ...so the point refits and re-serves on post-demotion rows only.
    for i in range(34):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "serve"
    # a manual serve -> shadow also starts the cooldown (fit stays active).
    assert svc.do_promote(
        {"point_id": "ap.pt", "phase": "shadow"})["ok"]
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["waiting_for"] == "cooldown"
    clock["t"] += 8 * 86400
    mono["t"] += 61
    # fit still active, but the only evidence rows are pre-demotion ones.
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["waiting_for"] == "gate:1"
    assert svc.autopilot_check()["actions"] == []
    for i in range(14):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "serve"


# -- T-AP9: explicit stops ----------------------------------------------------


def test_T_AP9_paused_stops_until_next_shadow(tmp_path, clock, mono):
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong={22, 23}), mono=mono,
        hosted_jev={"fit_rows": 20}, serve={"min_auto_support": 5},
        train={"min_examples": 10})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store)
    assert svc.do_register(point(target_error=0.3))["ok"]
    for i in range(24):  # fit, but only 4 evidence rows (need 10)
        decide_label(svc, "ap.pt", "ab"[i % 2])
    assert store.get_point("ap.pt")["phase"] == "shadow"
    assert svc.do_promote(
        {"point_id": "ap.pt", "phase": "paused"})["ok"]
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["status"] == "blocked(stopped_by_call)"
    # paused points sit out (paused rows are not evidence either)...
    for i in range(10):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "paused"
    # ...until the next manual shadow call resumes them.
    assert svc.do_promote(
        {"point_id": "ap.pt", "phase": "shadow"})["ok"]
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["waiting_for"] == "gate:1"
    for i in range(10):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    out = svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "serve"
    assert [a["action"] for a in out["actions"]] == ["promoted_to_serve"]


# -- T-AP10: no automatic tier switch -----------------------------------------


def test_T_AP10_item8_holds_but_tier_never_switches(tmp_path, clock, mono):
    # hosted: 18 @0.9 + 4 low@0.5 + 2 wrong@0.6; private: one early error
    # below its threshold (margin spread without breaking agreement).
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong={12, 13}, low={14, 15, 16, 17}),
        private_schedule=lambda i, h, o: (other(h, o), 0.6) if i == 0
        else (h, 0.9), mono=mono,
        hosted_jev={"fit_rows": 10, "min_compare": 10},
        serve={"min_auto_support": 5}, train={"min_examples": 10})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store)
    assert svc.do_register(point(target_error=0.3))["ok"]
    for i in range(11):  # row 0 is collect (no hosted call): 10 candidates
        decide_label(svc, "ap.pt", "ab"[i % 2])
    assert store.active_hosted_fit("ap.pt")["threshold"] == 0.9
    TR.train_fake_private(store, "ap.pt", threshold=0.7)
    for i in range(24):
        decide_label(svc, "ap.pt", "ab"[i % 2])
    comp = svc._private_comparison(store.get_point("ap.pt"))
    assert comp["n"] == 24 and comp["pass"] is True
    out = svc.autopilot_check()
    assert store.get_point("ap.pt")["phase"] == "serve"
    assert store.get_point("ap.pt")["tier"] == "hosted_jev"  # no switch
    assert [a["action"] for a in out["actions"]] == ["promoted_to_serve"]
    # ...while the manual switch with confirm still works.
    r = svc.do_promote({"point_id": "ap.pt", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"], r
    assert store.get_point("ap.pt")["tier"] == "private"


# -- T-AP11: reads never change state -----------------------------------------


TABLES = ["points", "decisions", "models", "jobs", "autopilot_log", "meta",
          "spend", "spend_reservations", "hosted_fit"]


def db_snapshot(store):
    snap = {}
    for t in TABLES:
        try:
            rows = store.cx.execute(
                f"SELECT * FROM {t} ORDER BY rowid").fetchall()
        except Exception:
            rows = []
        snap[t] = [tuple(r) for r in rows]
    return snap


def test_T_AP11_report_and_status_are_read_only(tmp_path, clock, mono):
    notify_path = tmp_path / "notify.log"
    svc, store, cfg, trainer = train_setup(tmp_path)
    cfg["autopilot"]["notify_cmd"] = notify_cmd_for(notify_path)
    reapprove(cfg, store)
    svc.autopilot_check()  # launch the smoke job (writes expected here)
    assert store.jobs_latest()["status"] == "running"
    before = db_snapshot(store)
    notify_before = notify_lines(notify_path)
    assert svc.do_report({})["ok"]
    assert svc.do_report({"point_id": "ap.tr"})["ok"]
    assert svc.do_report({"format": "md"})["ok"]
    assert svc.do_report({"point_id": "ap.tr", "format": "md"})["ok"]
    assert svc.do_autopilot_status({})["ok"]
    assert db_snapshot(store) == before
    assert notify_lines(notify_path) == notify_before
    block = svc.do_report({})["autopilot"]
    assert block["enabled"] is True and block["match"] is True
    assert [a["action"] for a in block["actions"]] == ["train_launched"]


# -- T-AP12: toggles, budgets, limiter ----------------------------------------


def serve_ready_setup(tmp_path, clock, mono, pid="ap.serve", wrong=frozenset()):
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong=wrong), mono=mono,
        hosted_jev={"fit_rows": 20}, serve={"min_auto_support": 5},
        train={"min_examples": 10})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store)
    assert svc.do_register(point(pid, target_error=0.3))["ok"]
    for i in range(34):
        decide_label(svc, pid, "ab"[i % 2])
    return svc, store, cfg


def test_T_AP12_auto_serve_off_disables_only_serve(tmp_path, clock, mono):
    svc, store, cfg = serve_ready_setup(tmp_path, clock, mono,
                                        wrong={24, 25})
    cfg["autopilot"]["auto_serve"] = False
    cfg["autopilot"]["auto_train"] = True
    cfg["train"]["min_train"] = 10
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    cfg["fireworks"]["allow_paid_training"] = True
    trainer = FakeTrainer(cfg)
    svc.trainer = trainer
    reapprove(cfg, store)
    assert svc.do_register(train_point())["ok"]
    for i in range(10):
        decide_label(svc, "ap.tr", "ab"[i % 2])
    out = svc.autopilot_check()
    # serve-ready point untouched...
    assert store.get_point("ap.serve")["phase"] == "shadow"
    assert svc.do_autopilot_status(
        {})["points"]["ap.serve"]["status"] == "off"
    # ...while training proceeds.
    assert [a["action"] for a in out["actions"]] == ["train_launched"]
    assert store.jobs_latest()["status"] == "running"


def test_T_AP12_auto_train_off_disables_only_train(tmp_path, clock, mono):
    svc, store, cfg = serve_ready_setup(tmp_path, clock, mono,
                                        wrong={24, 25})
    cfg["train"]["min_train"] = 10
    cfg["fireworks"]["monthly_budget_usd"] = 100.0
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    svc.trainer = FakeTrainer(cfg)
    assert svc.do_register(train_point())["ok"]
    for i in range(10):
        decide_label(svc, "ap.tr", "ab"[i % 2])
    out = svc.autopilot_check()
    assert store.get_point("ap.serve")["phase"] == "serve"
    assert store.jobs_latest() is None  # no training without auto_train
    assert [a["action"] for a in out["actions"]] == ["promoted_to_serve"]


def test_T_AP12_budgets_at_100_block(tmp_path, clock, mono):
    svc, store, cfg, trainer = train_setup(tmp_path)
    cfg["fireworks"]["allow_paid_training"] = True
    reapprove(cfg, store)
    month = fake_month()
    store.spend_add(month, "fireworks", 100.0)
    assert svc.autopilot.point_status(
        store.get_point("ap.tr")) == ("blocked(budget)", "budget")
    store.spend_add(month, "hosted_jev", 5.0)
    cfg["autopilot"]["auto_serve"] = True
    reapprove(cfg, store)
    assert svc.do_register(point())["ok"]
    assert svc.autopilot.point_status(
        store.get_point("ap.pt")) == ("blocked(budget)", "budget")
    svc2, store2, cfg2 = serve_ready_setup(tmp_path / "s", clock, mono,
                                           wrong={24, 25})
    store2.spend_add(fake_month(), "hosted_jev", 5.0)
    assert svc2.autopilot.point_status(
        store2.get_point("ap.serve")) == ("blocked(budget)", "budget")
    assert svc2.autopilot_check()["actions"] == []


def test_T_AP12_per_point_minute_limit(tmp_path, clock, mono):
    svc, store, cfg = make_svc(tmp_path, mono=mono)
    enable(cfg, store)
    assert svc.do_register(point())["ok"]
    dids = []
    for i in range(101):
        r = svc.do_decide({"point_id": "ap.pt",
                           "context": {"label_hint": "a"}})
        dids.append(r["decision_id"])
    assert svc.autopilot.checks_run == 0  # decide never triggers
    for did in dids[:100]:
        assert svc.do_outcome({"decision_id": did, "ok": True})["ok"]
    assert svc.autopilot.checks_run == 1
    mono["t"] += 61
    assert svc.do_outcome({"decision_id": dids[100], "ok": True})["ok"]
    assert svc.autopilot.checks_run == 2


# -- T-AP13: gateway item 7 ----------------------------------------------------


GW_SYS = ("You are a support triage classifier. Read the ticket and answer "
          "with exactly one word: approve or deny.")
GW_PREFIX = "Ticket text follows. Classify it now: "


def gw_point(**kw):
    p = {"point_id": "ap.gw", "workflow": "support",
         "question": "What should happen to this ticket?",
         "kind": "choice", "options": ["approve", "deny"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public",
         "source": "gateway", "route": "openai", "model": "m",
         "template": {"system": GW_SYS, "user_prefix": GW_PREFIX,
                      "user_suffix": ""},
         "output_format": {"type": "enum_text"},
         "extract": {}, "multi_turn": False}
    p.update(kw)
    return p


def gw_label(svc, label):
    r = svc.do_decide({"point_id": "ap.gw",
                       "context": {"label_hint": label}})
    assert r["ok"], r
    rec = svc.do_record({"decision_id": r["decision_id"], "choice": label})
    assert rec["ok"], rec
    return r["decision_id"]


def test_T_AP13_gateway_point_needs_item7(tmp_path, clock, mono):
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong={12, 13}), mono=mono,
        hosted_jev={"fit_rows": 10},
        serve={"min_auto_support": 5}, train={"min_examples": 10},
        gateway={"format_min_n": 5, "format_match_min": 0.5})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store)
    assert svc.do_register(gw_point(target_error=0.3))["ok"]
    for i in range(11):  # row 0 is collect (no hosted call): 10 candidates
        gw_label(svc, ["approve", "deny"][i % 2])
    assert store.active_hosted_fit("ap.gw")["threshold"] == 0.9
    for i in range(13):
        gw_label(svc, ["approve", "deny"][i % 2])
    assert store.get_point("ap.gw")["phase"] == "shadow"
    assert svc.do_autopilot_status(
        {})["points"]["ap.gw"]["waiting_for"] == "gate:7"
    assert svc.autopilot_check()["actions"] == []
    for i in range(5):
        store.trace_insert(
            {"ts": S._utcnow(), "route": "openai", "model": "m",
             "discovery_key": "k", "point_id": "ap.gw", "point_version": 1,
             "mode": "forwarded", "raw_response": "approve",
             "format_compared": 1, "format_match": 1, "unparsed": 0})
    out = svc.autopilot_check()
    assert store.get_point("ap.gw")["phase"] == "serve"
    assert [a["action"] for a in out["actions"]] == ["promoted_to_serve"]


# -- units: hashing, dates, notify, trainer -----------------------------------


def test_units_section_hash_stable(tmp_path, clock):
    a = {"enabled": True, "points": ["x"], "n": 1}
    b = {"n": 1, "points": ["x"], "enabled": True}
    assert section_hash(a) == section_hash(b)
    assert section_hash({**a, "n": 2}) != section_hash(a)
    assert len(section_hash(a)) == 64


def test_units_free_path_dates(tmp_path, clock):
    from decideonce.autopilot import APPROVED_AT_KEY, SMOKE_DONE_AT_KEY

    svc, store, cfg = make_svc(tmp_path)
    ap = svc.autopilot
    assert ap.free_path_ok() is False  # unset
    cfg["autopilot"]["free_training_until"] = "not-a-date"
    assert ap.free_path_ok() is False
    # G6: approval must postdate the smoke job.
    store.meta_put(SMOKE_DONE_AT_KEY, S._utcnow())
    store.meta_put(APPROVED_AT_KEY, S._utcnow())
    cfg["autopilot"]["free_training_until"] = "2026-06-01T00:00:00-08:00"
    assert ap.free_path_ok() is True  # fake now is Oct 2025
    cfg["autopilot"]["free_training_until"] = "2026-06-01T00:00:00"  # naive UTC
    assert ap.free_path_ok() is True
    cfg["autopilot"]["free_training_until"] = "2025-01-01T00:00:00-08:00"
    assert ap.free_path_ok() is False
    # Approval predating the smoke job blocks the free path (G6).
    cfg["autopilot"]["free_training_until"] = "2026-06-01T00:00:00-08:00"
    store.meta_put(SMOKE_DONE_AT_KEY, S._utcnow())
    assert ap.free_path_ok() is False
    assert ap.manual_free_ok(False) is False
    assert ap.manual_free_ok(True) is True
    clock["t"] += 500 * 86400
    assert ap.manual_free_ok(True) is False  # past the main cut-off


def test_units_notify_never_runs_a_shell(tmp_path, clock):
    from decideonce.autopilot import notify, sync_notifies

    marker = tmp_path / "MARKER"
    notify(f"echo hi; touch {marker}", "hello")  # metachars are argv, not shell
    notify([sys.executable, "-c", "import sys"], "hello")  # exits 0, ignored
    notify("definitely-not-a-binary-xyz", "hello")  # swallowed
    notify("", "hello")
    notify(None, "hello")
    sync_notifies()
    assert not marker.exists()


def test_units_fireworks_host_check():
    from decideonce.trainer import check_fireworks_host

    check_fireworks_host("https://api.fireworks.ai")
    check_fireworks_host("https://api.fireworks.ai:443/v1")
    for bad in ("http://api.fireworks.ai", "https://evil.example.com",
                "https://user@api.fireworks.ai",
                "https://api.fireworks.ai:8443", "not a url"):
        with pytest.raises(Exception):
            check_fireworks_host(bad)


def test_units_fake_trainer_learns_and_prices(tmp_path, clock):
    cfg = load_config()
    trainer = FakeTrainer(cfg)
    rows = [{"point_id": "p", "ts": f"2025-01-01T00:00:{i:02d}Z",
             "decision_id": f"d{i}", "context": {"label_hint": "ab"[i % 2]},
             "label": "ab"[i % 2], "weight": 1.0, "split": "train"}
            for i in range(10)]
    est = trainer.estimate(rows, "full")
    assert est["estimate_usd"] > 0 and est["train_rows"] == 10
    smoke = trainer.estimate(rows, "smoke")
    assert smoke["train_rows"] == 10  # fewer than smoke_examples: all used
    job = trainer.launch(rows, "full", "idem-1")
    assert trainer.status(job)["status"] == "running"
    done = trainer.complete(job)
    assert done["status"] == "done"
    assert done["metrics"]["accuracy"] == 1.0
    ckpt = trainer.checkpoint(job)
    assert ckpt["per_point"]["p"]["hint_map"] == {"a": "a", "b": "b"}
    job2 = trainer.launch(rows, "full", "idem-2")
    assert trainer.fail(job2)["status"] == "failed"
    trainer2 = FakeTrainer(cfg, estimate_override=3.5)
    assert trainer2.estimate(rows, "full")["estimate_usd"] == 3.5


def test_units_check_rereads_section_from_file(tmp_path, clock, mono):
    toml = tmp_path / "c.toml"
    toml.write_text("[autopilot]\nenabled = true\npoints = [\"ap.*\"]\n"
                    "max_target_error = 0.5\n"
                    "[hosted_jev]\nmonthly_budget_usd = 5.0\n"
                    f"[autopilot_trust]\npublic_key = \"{_TEST_PUB_B64}\"\n")
    cfg = load_config(str(toml))
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0),
                  monotonic=lambda: mono["t"], config_path=str(toml))
    _sign_cfg(cfg, store)
    assert svc.do_register(point())["ok"]
    decide_label(svc, "ap.pt", "a")
    assert store.get_point("ap.pt")["phase"] == "shadow"
    toml.write_text("[autopilot]\nenabled = true\npoints = [\"ap.*\"]\n"
                    "max_target_error = 0.5\ncooldown_days = 8\n"
                    "[hosted_jev]\nmonthly_budget_usd = 5.0\n"
                    f"[autopilot_trust]\npublic_key = \"{_TEST_PUB_B64}\"\n")
    out = svc.autopilot_check()  # file edit picked up, no restart
    assert out["actions"] == []
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["status"] == "blocked(not_approved)"
    toml.unlink()  # missing file -> defaults -> still stopped
    assert svc.autopilot_check()["actions"] == []
    assert svc.do_autopilot_status(
        {})["points"]["ap.pt"]["status"] == "blocked(disabled)"


def test_units_join_candidacy_and_labelled_ratio(tmp_path, clock, mono):
    # ap.small's 8 shadow calls sit between ap.big rows 19 and 20, so ap.big
    # evidence calls are fn 28-40: wrong rows at fn {32, 33}.
    svc, store, cfg = make_scripted(
        tmp_path, serve_schedule(wrong={32, 33}), mono=mono,
        hosted_jev={"fit_rows": 20},
        serve={"min_auto_support": 5}, train={"min_examples": 10,
                                              "min_train": 10})
    cfg["serve"]["min_shadow"] = 5
    cfg["serve"]["audit_rate"] = 0.0
    enable(cfg, store)
    assert svc.do_register(point("ap.big", target_error=0.3))["ok"]
    assert svc.do_register(point("ap.small"))["ok"]
    for i in range(20):
        decide_label(svc, "ap.big", "ab"[i % 2])
    for i in range(9):
        decide_label(svc, "ap.small", "ab"[i % 2])
    joins = {j["point_id"]: j["n_rows"]
             for j in svc.autopilot.dry_run(
                 store.get_point("ap.big"))["joins"]}
    assert joins == {"ap.big": 20}  # 9 rows does not join
    # four unlabelled shadow rows sink the labelled ratio below 0.9.
    for i in range(14):
        decide_label(svc, "ap.big", "ab"[i % 2])
    for i in range(4):
        r = svc.do_decide({"point_id": "ap.big",
                           "context": {"label_hint": "a"}})
        assert r["ok"]
    assert svc.do_autopilot_status(
        {})["points"]["ap.big"]["waiting_for"] == "gate:labelled_ratio"


def test_R31_approve_rejects_unparseable_show_output(tmp_path, clock):
    svc, store, cfg = make_svc(tmp_path)
    cfg["autopilot_trust"]["public_key"] = _TEST_PUB_B64
    out: list[str] = []
    # The whole-line format must not be decoded as a key (R3-1).
    assert run_approve(cfg, store, is_tty=True, out=out.append,
                       env_runner=lambda argv: (0, _TEST_PRIV_B64 + "\n")) == "error"
    assert store.meta_get(APPROVED_HASH_KEY) is None
    assert run_approve(cfg, store, is_tty=True, out=out.append,
                       env_runner=lambda argv: (0, " gibberish line\n")) == "error"
    assert store.meta_get(APPROVED_HASH_KEY) is None


def test_R31_keygen_requires_added_or_updated(tmp_path, clock):
    toml = tmp_path / "k.toml"
    toml.write_text("[autopilot]\nenabled = true\n")
    store = Store(tmp_path / "proj")
    out: list[str] = []
    # Exit 0 with no marker (vault without Touch ID): no public key written.
    assert run_keygen(str(toml), store, out=out.append,
                      env_runner=lambda argv, value=None: (0, "")) == "error"
    assert "public_key" not in toml.read_text()
    assert any("Touch ID" in ln for ln in out)
    # Updated: path also accepted.
    out.clear()
    assert run_keygen(str(toml), store, out=out.append,
                      env_runner=lambda argv, value=None:
                      (0, "Updated: DECIDEONCE_AUTOPILOT_SIGNING_KEY\n")) == "ok"
    assert "public_key" in toml.read_text()
