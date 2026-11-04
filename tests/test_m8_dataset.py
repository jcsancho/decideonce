"""M8 tests: workflow dataset (T-W*)."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
WF = REPO / "datasets" / "workflows"

if not (WF / "workflows.jsonl").exists():
    pytest.skip("private dataset fixtures are not exported",
                allow_module_level=True)

sys.path.insert(0, str(WF))
sys.path.insert(0, str(REPO / "pilots"))

import build  # noqa: E402
import p0_eval  # noqa: E402


def _rows():
    return [json.loads(line) for line in (WF / "workflows.jsonl").read_text().splitlines()]


def test_T_W1_every_row_validates_and_has_gold():
    rows = _rows()
    assert len(rows) >= 150, len(rows)
    for row in rows:
        assert build.validate_row(row) == [], row["id"]
        gold_file = WF / "gold" / f"{row['id']}.json"
        assert gold_file.is_file(), row["id"]
        gold = json.loads(gold_file.read_text())
        assert build.validate_gold(gold) == [], row["id"]
        assert gold["id"] == row["id"]
        assert (WF / "sources" / f"{row['id']}.md").is_file()


def test_T_W2_build_idempotent():
    def snapshot():
        h = hashlib.sha256()
        h.update((WF / "workflows.jsonl").read_bytes())
        h.update((WF / "README.md").read_bytes())
        for g in sorted((WF / "gold").glob("*.json")):
            h.update(g.read_bytes())
        return h.hexdigest()

    r1 = subprocess.run([sys.executable, "build.py"], cwd=WF, capture_output=True, text=True)
    assert r1.returncode == 0, r1.stderr
    mid = snapshot()
    r2 = subprocess.run([sys.executable, "build.py"], cwd=WF, capture_output=True, text=True)
    assert r2.returncode == 0, r2.stderr
    # second run changes nothing (first run may normalise a dirty tree)
    assert snapshot() == mid, "build.py is not idempotent"
    assert json.loads(r1.stdout)["rows"] >= 150


def test_T_W3_secret_scan_blocks_fixture():
    assert "sk-" in build.secret_hits("key = sk-abcdefghijklmnopqrst")
    assert "AKIA" in build.secret_hits("id AKIAIOSFODNN7EXAMPLE ok")
    assert "xox" in build.secret_hits("token xoxb-123456789012-abcdefgh")
    assert build.secret_hits("nothing secret here, just a skill") == []


def test_T_W4_synthetic_gold_matches_planted():
    import synth as S
    from decideonce.analyze import score_candidate
    from decideonce.config import load_config
    cfg = load_config(str(REPO / "decideonce.toml"))
    n = 0
    for area in S.AREAS:
        for idx in range(1, 11):
            _, planted, planted_non, wid = S.render(area, idx)
            gold = json.loads((WF / "gold" / f"{wid}.json").read_text())
            assert gold["annotator"] == "synthetic"
            assert [p["draft"]["point_id"] for p in gold["points"]] == [
                p["draft"]["point_id"] for p in planted]
            assert [p["line_span"] for p in gold["points"]] == [
                p["line_span"] for p in planted]
            assert [p["verdict"] for p in gold["points"]] == [
                p["verdict"] for p in planted]
            assert len(gold["non_points"]) == len(planted_non)
            for p in gold["points"]:  # planted verdict == scorer verdict
                got = score_candidate({"draft": {"options": p["draft"]["options"]},
                                       "answers": p["rubric_answers"]}, cfg)
                assert got["verdict"] == p["verdict"], (wid, p["draft"]["point_id"])
            n += 1
    assert n == 60


def test_T_W5_p0_eval_matching_fixture():
    gold = [{"line_span": [3, 3], "options": ["a", "b"], "verdict": "good_fit"},
            {"line_span": [9, 10], "options": ["x", "y"], "verdict": "possible"}]
    pred = [{"line_span": [3, 4], "options": ["a", "b", "c"], "verdict": "good_fit"},
            {"line_span": [20, 20], "options": ["p", "q"], "verdict": "possible"}]
    hits, misses, false = p0_eval.match_points(gold, pred)
    assert len(hits) == 1 and len(misses) == 1 and len(false) == 1
    assert hits[0][0]["line_span"] == [3, 3]
    # Jaccard 2/4 = 0.5 passes; below 0.5 fails
    assert p0_eval.option_jaccard(["a", "b"], ["a", "b", "c", "d"]) == 0.5
    hits2, _, _ = p0_eval.match_points(
        [{"line_span": [1, 1], "options": ["a", "b"], "verdict": "good_fit"}],
        [{"line_span": [1, 1], "options": ["a", "c", "d"], "verdict": "good_fit"}])
    assert hits2 == []


def _refund_ticket():
    path = REPO / "pilots" / "refund" / "data" / "tickets.jsonl"
    for line in path.read_text().splitlines():
        t = json.loads(line)
        if t["ground_truth"]["refund_action"] == "full_refund":
            return t
    raise AssertionError("no full_refund ticket")


def test_check_stdio_protocol():
    check = REPO / "pilots" / "refund" / "check.py"
    t = _refund_ticket()
    amt = t["ground_truth"]["refund_amount_eur"]
    payload = json.dumps({"task_ref": t["ticket_id"],
                          "final_output": f"We will issue a full refund.\nAction: full refund; Amount: {amt:.2f} EUR"})
    r = subprocess.run([sys.executable, str(check)], input=payload,
                       capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert json.loads(r.stdout)["pass"] is True
    bad = subprocess.run(
        [sys.executable, str(check)],
        input=json.dumps({"task_ref": t["ticket_id"],
                          "final_output": "Sorry, no refund is possible."}),
        capture_output=True, text=True, timeout=30)
    assert json.loads(bad.stdout)["pass"] is False
    unknown = subprocess.run(
        [sys.executable, str(check)],
        input=json.dumps({"task_ref": "T-NOPE", "final_output": "x"}),
        capture_output=True, text=True, timeout=30)
    assert json.loads(unknown.stdout)["pass"] is False


def test_check_cli_unchanged(tmp_path):
    check = REPO / "pilots" / "refund" / "check.py"
    t = _refund_ticket()
    amt = t["ground_truth"]["refund_amount_eur"]
    reply = tmp_path / "reply.txt"
    reply.write_text(f"We will issue a full refund.\nAction: full refund; Amount: {amt:.2f} EUR")
    r = subprocess.run([sys.executable, str(check), t["ticket_id"],
                        str(reply)], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0
    assert json.loads(r.stdout)["pass"] is True
