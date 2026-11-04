"""H3 tests: point fields (tier/remote_ok/private_model_only/policy), v1.7 inputs."""
import random

from decideonce.config import load_config
from decideonce.deciders import build_deciders
from decideonce.service import Service
from decideonce.store import Store


def make_tiered(tmp_path, **hj_over):
    cfg = load_config()
    cfg["store"]["data_dir"] = str(tmp_path)
    cfg["decider"]["backend"] = "fake"
    cfg["hosted_jev"]["monthly_budget_usd"] = 5.0
    cfg["hosted_jev"].update(hj_over)
    store = Store(tmp_path / "proj")
    deciders, _ = build_deciders(cfg, store)
    svc = Service(cfg, store, deciders, rng=random.Random(0))
    return svc, store, cfg


def point(**kw):
    p = {"point_id": "h3.pt", "workflow": "w", "question": "Which?",
         "kind": "choice", "options": ["a", "b"],
         "context_fields": {"label_hint": {"type": "string"}},
         "required_fields": [], "data_class": "public"}
    p.update(kw)
    return p


def to_shadow(store, pid="h3.pt"):
    p = store.get_point(pid)
    p["phase"] = "shadow"
    store.put_point(p)


# -- T-HB6 policy versions --------------------------------------------------


def test_T_HB6_policy_bump_resets_plan_phase_and_epoch(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(policy="v1 rules"))["ok"]
    p = store.get_point("h3.pt")
    assert p["point_version"] == 1 and p["policy"] == "v1 rules"
    assert p["fit_epoch_start"]
    p["plan_phase"] = "shadow"
    p["fit_epoch_start"] = "2020-01-01T00:00:00Z"
    store.put_point(p)
    assert svc.do_register(point(policy="v2 rules"))["ok"]
    p2 = store.get_point("h3.pt")
    assert p2["point_version"] == 2
    assert p2["plan_phase"] == "collect"
    assert p2["fit_epoch_start"] > "2020-01-01T00:00:00Z"


def test_Z7_reregister_without_policy_keeps_it(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(policy="v1 rules"))["ok"]
    assert store.get_point("h3.pt")["point_version"] == 1
    assert svc.do_register(point())["ok"]  # policy omitted: not a removal
    p = store.get_point("h3.pt")
    assert p["policy"] == "v1 rules"
    assert p["point_version"] == 1


def test_T_HB6_policy_first_state_key(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(policy="refund rules: be kind"))["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "h3.pt",
                       "context": {"label_hint": "a"}})
    assert r["ok"]
    body = svc.deciders["hosted_jev"].transport.requests[0]
    assert list(body["state"])[0] == "policy"
    assert body["state"]["policy"] == "refund rules: be kind"


def test_T_HB6_policy_too_long_rejected(tmp_path):
    svc, _, _ = make_tiered(tmp_path)
    r = svc.do_register(point(policy="x" * 20001))
    assert r["ok"] is False


# -- v1.7 register inputs ---------------------------------------------------


def test_register_tier_is_server_managed(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(tier="private"))["ok"]
    assert store.get_point("h3.pt")["tier"] == "hosted_jev"
    assert svc.do_register(point(point_id="h3.priv", data_class="private",
                                 private_model_only=True))["ok"]
    assert store.get_point("h3.priv")["tier"] == "private"


def test_register_privacy_conflict(tmp_path):
    svc, _, _ = make_tiered(tmp_path)
    r = svc.do_register(point(data_class="private", private_model_only=True,
                              remote_ok=["hosted_jev"]))
    assert r["ok"] is False and r["error"] == "privacy_conflict"


def test_register_remote_ok_true_read_as_both(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private", remote_ok=True,
                                 confirm=True))["ok"]
    assert store.get_point("h3.pt")["remote_ok"] == ["fireworks", "hosted_jev"]


def test_register_privacy_change_needs_confirm(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private"))["ok"]
    base = point(data_class="private")
    r = svc.do_register({**base, "remote_ok": ["hosted_jev"]})
    assert r["ok"] is False and r["error"] == "confirm_required"
    assert svc.do_register({**base, "remote_ok": ["hosted_jev"],
                            "confirm": True})["ok"]
    assert store.get_point("h3.pt")["remote_ok"] == ["hosted_jev"]
    r = svc.do_register({**base, "remote_ok": ["fireworks"],
                         "private_model_only": True})
    assert r["ok"] is False and r["error"] == "confirm_required"
    assert svc.do_register({**base, "remote_ok": ["fireworks"],
                            "private_model_only": True,
                            "confirm": True})["ok"]
    p = store.get_point("h3.pt")
    assert p["tier"] == "private"  # tier follows private_model_only


def test_register_public_point_needs_no_confirm(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    assert svc.do_register(point(remote_ok=["hosted_jev"]))["ok"]
    assert store.get_point("h3.pt")["remote_ok"] == ["hosted_jev"]


# -- T-HJ7 privacy gate -----------------------------------------------------


def test_T_HJ7_private_model_only_never_calls_hosted(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private",
                                 private_model_only=True))["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "h3.pt", "context": {"label_hint": "a"}})
    assert r["ok"] and r["mode"] == "teacher"
    assert svc.deciders["hosted_jev"].transport.calls == 0
    row = store.get_decision(r["decision_id"])
    assert row["hosted"] is None and row["decider"] == "private"


def test_T_HJ7_private_tier_never_falls_back(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    p = store.get_point("h3.pt")
    p["tier"] = "private"  # H5/M11 promote path; H3 pins the routing
    p["phase"] = "shadow"
    store.put_point(p)
    r = svc.do_decide({"point_id": "h3.pt", "context": {"label_hint": "a"}})
    assert r["ok"]
    assert svc.deciders["hosted_jev"].transport.calls == 0
    assert store.get_decision(r["decision_id"])["decider"] == "private"


def test_T_HJ7_private_needs_remote_ok(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private"))["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "h3.pt", "context": {"label_hint": "a"}})
    assert r["ok"] and r["mode"] == "teacher"
    assert svc.deciders["hosted_jev"].transport.calls == 0
    assert svc.do_register(point(data_class="private",
                                 remote_ok=["hosted_jev"],
                                 confirm=True))["ok"]
    to_shadow(store)
    r = svc.do_decide({"point_id": "h3.pt", "context": {"label_hint": "a"}})
    assert r["ok"]
    assert svc.deciders["hosted_jev"].transport.calls == 1


def test_T_HJ7_public_needs_nothing(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    to_shadow(store)
    assert svc.do_decide({"point_id": "h3.pt",
                          "context": {"label_hint": "a"}})["ok"]
    assert svc.deciders["hosted_jev"].transport.calls == 1


# -- do_promote tier (H3) ---------------------------------------------------


def test_promote_tier_only_with_serve(tmp_path):
    svc, _, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h3.pt", "phase": "shadow",
                        "tier": "private"})
    assert r["ok"] is False and r["error"] == "invalid_args"


def test_promote_tier_private_not_better(tmp_path):
    svc, _, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    r = svc.do_promote({"point_id": "h3.pt", "phase": "serve",
                        "tier": "private", "confirm": True})
    assert r["ok"] is False and r["error"] == "private_not_better"


def test_promote_tier_hosted_on_private_only_conflicts(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private",
                                 private_model_only=True))["ok"]
    r = svc.do_promote({"point_id": "h3.pt", "phase": "serve",
                        "tier": "hosted_jev", "confirm": True})
    assert r["ok"] is False and r["error"] == "privacy_conflict"


# -- Section D: privacy confirms ---------------------------------------------

def test_D_new_private_remote_ok_needs_confirm(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    r = svc.do_register(point(data_class="private",
                              remote_ok=["hosted_jev"]))
    assert r["ok"] is False and r["error"] == "confirm_required"
    r = svc.do_register(point(data_class="private", remote_ok=True))
    assert r["ok"] is False and r["error"] == "confirm_required"
    assert svc.do_register(point(data_class="private",
                                 remote_ok=["hosted_jev"],
                                 confirm=True))["ok"]
    assert store.get_point("h3.pt")["remote_ok"] == ["hosted_jev"]


def test_D_new_private_without_remote_ok_needs_no_confirm(tmp_path):
    svc, _, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private"))["ok"]
    assert svc.do_register(point(point_id="h3.pmo", data_class="private",
                                 private_model_only=True))["ok"]


def test_D_private_to_public_needs_confirm(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point(data_class="private"))["ok"]
    r = svc.do_register(point(data_class="public"))
    assert r["ok"] is False and r["error"] == "confirm_required"
    assert svc.do_register(point(data_class="public",
                                 confirm=True))["ok"]
    assert store.get_point("h3.pt")["data_class"] == "public"
    # narrowing is the safe direction: no confirm needed
    assert svc.do_register(point(data_class="private"))["ok"]


# -- Section D: prompt changes bump the version ----------------------------

def test_D_question_and_option_help_bump_version(tmp_path):
    svc, store, _ = make_tiered(tmp_path)
    assert svc.do_register(point())["ok"]
    assert store.get_point("h3.pt")["point_version"] == 1
    assert svc.do_register(point(question="Which one?"))["ok"]
    assert store.get_point("h3.pt")["point_version"] == 2
    assert svc.do_register(point(option_help={"a": "first"}))["ok"]
    assert store.get_point("h3.pt")["point_version"] == 3
    # identical re-register: no bump
    assert svc.do_register(point(option_help={"a": "first"}))["ok"]
    assert store.get_point("h3.pt")["point_version"] == 3


# -- Section B: robustness and security --------------------------------------

def test_B6_policy_and_input_field_names_rejected(tmp_path):
    svc, _, _ = make_tiered(tmp_path)
    for name in ("policy", "input"):
        r = svc.do_register(point(point_id=f"h3.{name}",
                                  context_fields={name: {"type": "string"}}))
        assert r["ok"] is False and r["error"] == "invalid_point"
