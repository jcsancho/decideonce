"""Probe training (§7.2): vectors, logistic probe, calibration, threshold, model card."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from . import labels as L
from . import probe as P
from . import stats as S

C_GRID = [0.01, 0.1, 1.0, 10.0]


def prompt_hash(kind: str, state_text: str, question: str, options: list[str]) -> str:
    return hashlib.sha256("\x00".join(
        [kind, state_text, question, ",".join(options)]).encode()).hexdigest()


class VectorCache:
    """vectors/<point_id>.npy (float32 rows) + .index.json {hash: row}."""

    def __init__(self, vectors_dir: Path, point_id: str):
        self.npy = vectors_dir / f"{point_id}.npy"
        self.idx_path = vectors_dir / f"{point_id}.index.json"
        self.dim: int | None = None
        self.index: dict[str, int] = {}
        self.vecs: np.ndarray | None = None
        self._load()

    def _load(self) -> None:
        if self.idx_path.is_file() and self.npy.is_file():
            try:
                meta = json.loads(self.idx_path.read_text())
                vecs = np.load(str(self.npy))
                if vecs.ndim == 2 and vecs.shape[0] == len(meta.get("hashes", {})):
                    self.dim = int(vecs.shape[1])
                    self.index = dict(meta["hashes"])
                    self.vecs = vecs
                    return
            except Exception:
                pass
        self.dim, self.index, self.vecs = None, {}, None

    def save(self) -> None:
        if self.vecs is None:
            return
        np.save(str(self.npy), self.vecs.astype(np.float32))
        self.idx_path.write_text(json.dumps({"dim": self.dim, "hashes": self.index}))

    def get(self, h: str) -> list[float] | None:
        if h in self.index and self.vecs is not None:
            return [float(x) for x in self.vecs[self.index[h]]]
        return None

    def add(self, h: str, vec: list[float]) -> None:
        if h in self.index:
            return
        arr = np.asarray(vec, dtype=np.float32)
        if self.dim is None:
            self.dim = int(arr.shape[0])
            self.vecs = arr.reshape(1, -1)
        elif arr.shape[0] != self.dim:
            # Backend changed underneath us: restart the cache.
            self.dim, self.index = int(arr.shape[0]), {}
            self.vecs = arr.reshape(1, -1)
        else:
            assert self.vecs is not None
            self.vecs = np.vstack([self.vecs, arr.reshape(1, -1)])
        self.index[h] = int(self.vecs.shape[0]) - 1


def cache_vector(store, point: dict, state_text: str, vector: list[float]) -> None:
    """On-line caching for shadow/serve (Q2 default); failures are silent."""
    try:
        c = VectorCache(store.vectors_dir, point["point_id"])
        c.add(prompt_hash(point["kind"], state_text, point["question"],
                          list(point["options"])), vector)
        c.save()
    except Exception:
        pass


def _fit_temperature(oof_logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling on CV out-of-fold logits: grid NLL minimization."""
    best_t, best_nll = 1.0, float("inf")
    for _ in range(2):
        lo, hi = best_t / 4.0, best_t * 4.0
        grid = np.exp(np.linspace(np.log(max(lo, 1e-3)), np.log(hi), 120))
        for t in grid:
            z = oof_logits / float(t)
            z = z - z.max(axis=1, keepdims=True)
            logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            nll = float(-logp[np.arange(len(y)), y].mean())
            if nll < best_nll:
                best_nll, best_t = nll, float(t)
    return best_t


def _ece(y_true: list[int], proba: np.ndarray, n_bins: int = 10) -> float:
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    ece, n = 0.0, len(y_true)
    if n == 0:
        return 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        mask = (conf > lo) & (conf <= hi) if b else (conf <= hi)
        m = int(mask.sum())
        if m:
            acc = float((pred[mask] == np.asarray(y_true)[mask]).mean())
            ece += m / n * abs(acc - float(conf[mask].mean()))
    return ece


def train_point(cfg: dict, store, jev, point_id: str, dry_run: bool = False,
                seed: int = 0, sources: list[str] | None = None,
                ignore_gate: bool = False) -> dict:
    """Train a probe for a point. Returns the do_train response body (without ok wrapper).

    sources/ignore_gate exist for one pilot measure (§15.9: a probe trained
    on the extracted rows alone); production training never uses them.
    """
    from .errors import ToolError  # local import: errors has no deps
    t_start = time.monotonic()
    point = store.get_point(point_id)
    if point is None:
        raise ToolError("unknown_point", f"unknown point_id {point_id!r}")
    options = list(point["options"])
    rows = store.decisions_for_point(point_id)  # time order
    labeled = L.labeled_rows(rows, point, cfg["train"]["teacher_only_weight"],
                             cfg["extract"]["weight"])
    if sources is not None:
        labeled = [(r, lab, src, w) for r, lab, src, w in labeled
                   if src in sources]
    # eval-only transcripts (pilot arm runs on held-out tickets) never train:
    # their rows would leak the eval set into the probe.
    eval_ids = store.eval_only_transcript_ids(point["workflow"])
    n_eval_excluded = sum(1 for r, _, _, _ in labeled
                          if (r.get("transcript_id") or "") in eval_ids)
    labeled = [(r, lab, src, w) for r, lab, src, w in labeled
               if (r.get("transcript_id") or "") not in eval_ids]
    # §15.3 gate: extracted rows train only while the point passes it. Rows
    # whose label comes from an outcome or the live teacher stay: those labels
    # do not depend on extraction quality.
    gate = extraction_gate(cfg, store, point)
    if not gate["passed"] and not ignore_gate:
        labeled = [(r, lab, src, w) for r, lab, src, w in labeled
                   if src != "extracted"]
    min_train = cfg["train"]["min_train"]
    min_per_option = cfg["train"]["min_per_option"]
    counts: dict[str, int] = {}
    for _, label, _, _ in labeled:
        counts[label] = counts.get(label, 0) + 1
    well = sum(1 for c in counts.values() if c >= min_per_option)
    if len(labeled) < min_train or well < 2:
        raise ToolError("not_enough_data",
                        f"need {min_train} labelled rows and {min_per_option}+ examples of "
                        f"2+ options; have {len(labeled)} rows, {well} options qualify")

    # 1. vectors (cached, else lazy from stored state_text)
    cache = VectorCache(store.vectors_dir, point_id)
    X, y, weights, kept = [], [], [], []
    n_cached, n_computed, n_no_text = 0, 0, 0
    opt_index = {o: i for i, o in enumerate(options)}
    for row, label, source, weight in labeled:
        h = None
        if row.get("state_text"):
            h = prompt_hash(point["kind"], row["state_text"], point["question"], options)
            vec = cache.get(h)
            if vec is not None:
                n_cached += 1
            else:
                try:
                    vec, _, _ = jev.hidden(point["kind"], row["state_text"],
                                           point["question"], options)
                except Exception as e:
                    raise ToolError("jev_unavailable", f"JEV hidden failed: {e}")
                cache.add(h, vec)
                n_computed += 1
        else:
            n_no_text += 1
            continue
        assert vec is not None
        X.append(vec)
        y.append(opt_index[label])
        weights.append(weight)
        kept.append((row, label, source, weight))
    cache.save()
    kept_counts: dict[str, int] = {}
    for _, lab, _, _ in kept:
        kept_counts[lab] = kept_counts.get(lab, 0) + 1
    kept_well = sum(1 for c in kept_counts.values() if c >= min_per_option)
    if len(kept) < min_train or kept_well < 2:
        raise ToolError("not_enough_data",
                        f"only {len(kept)} rows have vectors (need {min_train})")
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    w = np.asarray(weights, dtype=np.float64)

    # 2. run-aware split (§15.4): whole runs go to one side (no shuffling
    # across time); rows without run_id split by row as before.
    holdout = cfg["train"]["holdout"]
    run_order: list = []
    run_idx: dict = {}
    solo_idx = []
    for i, (row, _, _, _) in enumerate(kept):
        rid = row.get("run_id")
        if rid is None:
            solo_idx.append(i)
        elif rid not in run_idx:
            run_idx[rid] = [i]
            run_order.append(rid)
        else:
            run_idx[rid].append(i)
    hold_runs: set = set()
    if run_order:
        n_run_hold = max(1, int(round(len(run_order) * holdout)))
        if len(run_order) >= 2:
            n_run_hold = min(n_run_hold, len(run_order) - 1)
        hold_runs = set(run_order[len(run_order) - n_run_hold:])
    hold_set = {i for rid in hold_runs for i in run_idx[rid]}
    if solo_idx:
        if not run_order:
            n_solo = max(1, int(round(len(solo_idx) * holdout)))
            n_solo = min(n_solo, len(solo_idx) - 1)
        else:
            n_solo = int(round(len(solo_idx) * holdout))
        if n_solo:
            hold_set |= set(solo_idx[len(solo_idx) - n_solo:])
    if not hold_set:
        hold_set = {len(kept) - 1}
    tr_idx = [i for i in range(len(kept)) if i not in hold_set]
    ho_idx = sorted(hold_set)
    if not tr_idx:
        raise ToolError("not_enough_data", "run split left no training rows")
    Xtr, ytr, wtr = X[tr_idx], y[tr_idx], w[tr_idx]
    Xho, yho = X[ho_idx], y[ho_idx]
    hold_rows = [kept[i] for i in ho_idx]
    n_hold, n_train_rows = len(ho_idx), len(tr_idx)

    # 3. C by CV on the training part
    classes_tr, class_counts = np.unique(ytr, return_counts=True)
    k = min(5, int(class_counts.min())) if len(classes_tr) >= 2 else 0
    oof_logits: np.ndarray | None = None
    ytr_m = np.searchsorted(classes_tr, ytr)  # contiguous labels for oof columns
    if k >= 2:
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        best_c, best_score = C_GRID[0], -1.0
        cv_scores = {}
        for c in C_GRID:
            clf = LogisticRegression(C=c, class_weight="balanced", max_iter=2000)
            preds = cross_val_predict(clf, Xtr, ytr, cv=cv)
            s = float(accuracy_score(ytr, preds))
            cv_scores[str(c)] = s
            if s > best_score:
                best_score, best_c = s, c
        final = LogisticRegression(C=best_c, class_weight="balanced", max_iter=2000)
        oof = cross_val_predict(final, Xtr, ytr, cv=cv, method="decision_function")
        if oof.ndim == 1:  # binary: >0 means the larger class -> two columns
            oof = np.vstack([-oof, oof]).T
        oof_logits = np.asarray(oof)
        final.fit(Xtr, ytr, sample_weight=wtr)
    else:  # degenerate: single class in train; constant model
        best_c, cv_scores = 1.0, {}
        final = LogisticRegression(C=1.0, max_iter=2000)
        try:
            final.fit(Xtr, ytr, sample_weight=wtr)
        except ValueError:
            final = None
    if final is None:
        raise ToolError("not_enough_data", "training part holds a single class")

    # 4. temperature scaling on CV predictions
    temp = _fit_temperature(oof_logits, ytr_m) if oof_logits is not None else 1.0

    # probe predictions on holdout (masked over allowed, like inference)
    W, b = _coef_to_full(final, X.shape[1], options)
    model = {"W": W, "b": b, "temp": temp, "classes": options}
    ho_pred, ho_conf, ho_proba = [], [], []
    for i, (row, label, source, weight) in enumerate(hold_rows):
        full = P.predict_full(model, [float(x) for x in Xho[i]], options)
        masked = P.mask_and_renorm(full, row.get("allowed_options") or options)
        ho_pred.append(P.argmax(masked, options))
        ho_conf.append(max(masked.values()))
        ho_proba.append([masked.get(o, 0.0) for o in options])
    ho_proba = np.asarray(ho_proba)
    yho_labels = [options[i] for i in yho]

    # 4b. plan probe (§15.4): trained only on rows with at_start: true.
    # Unknown (None) is not at_start (fail closed). Injected and
    # pilot-forced runs are excluded: they parrot the addendum, so the
    # probe would confirm its own injected choices (B12). One training
    # row per train decision: X = its run's mean vector over kept rows.
    plan_flags = store.transcript_plan_flags(point["workflow"])

    def _at_start_row(tup) -> bool:
        return bool(tup[0].get("at_start"))

    def _plan_clean(tup) -> bool:
        if (tup[0].get("transcript_id") or "") in eval_ids:
            return False
        f = plan_flags.get(tup[0].get("transcript_id") or "")
        return not (f and (f["injected"] or f["pilot"]))

    def _plan_row(tup) -> bool:
        return _at_start_row(tup) and _plan_clean(tup)

    plan_model = None
    plan_W = plan_b = None
    run_mean: dict = {}
    for rid in run_order:
        vecs = [X[i] for i in tr_idx if kept[i][0].get("run_id") == rid
                and _plan_row(kept[i])]
        if vecs:
            run_mean[rid] = np.mean(vecs, axis=0)
    plan_train_n = 0
    if len(run_mean) >= 2:
        p_idx = [i for i in tr_idx if kept[i][0].get("run_id") in run_mean
                 and _plan_row(kept[i])]
        plan_train_n = len(p_idx)
        Xp = np.asarray([run_mean[kept[i][0]["run_id"]] for i in p_idx])
        yp = np.asarray([y[i] for i in p_idx])
        run_w = {rid: float(np.mean([w[i] for i in tr_idx
                                     if kept[i][0].get("run_id") == rid
                                     and _plan_row(kept[i])]))
                 for rid in run_mean}
        wp = np.asarray([run_w[kept[i][0]["run_id"]] for i in p_idx])
        if len(np.unique(yp)) >= 2:
            pclf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
            try:
                pclf.fit(Xp, yp, sample_weight=wp)
                plan_W, plan_b = _coef_to_full(pclf, X.shape[1], options)
                plan_model = {"W": plan_W, "b": plan_b, "temp": 1.0,
                              "classes": options}
            except ValueError:
                plan_model = None
    # plan-probe confidence on held-out grouped rows (for the plan threshold)
    plan_eval: list[tuple[float, bool]] = []
    if plan_model is not None:
        ho_mean: dict = {}
        for rid in hold_runs:
            vecs = [X[j] for j in ho_idx if kept[j][0].get("run_id") == rid
                    and _plan_row(kept[j])]
            if vecs:
                ho_mean[rid] = np.mean(vecs, axis=0)
        for k, i in enumerate(ho_idx):
            rid = kept[i][0].get("run_id")
            if rid not in ho_mean or not _plan_row(kept[i]):
                continue
            full = P.predict_full(plan_model, [float(x) for x in ho_mean[rid]], options)
            masked = P.mask_and_renorm(
                full, hold_rows[k][0].get("allowed_options") or options)
            plan_eval.append((max(masked.values()),
                              P.argmax(masked, options) == yho_labels[k]))

    # 5. zero-shot + majority on the same held-out rows
    zs_pred, zs_true = [], []
    for (row, label, _, _), _ in zip(hold_rows, yho_labels):
        zs = row.get("jev_zero_shot")
        if zs:
            masked = P.mask_and_renorm(zs, row.get("allowed_options") or options)
            zs_pred.append(P.argmax(masked, options))
            zs_true.append(label)
    majority = max(counts, key=lambda o: (counts[o], -options.index(o)))
    maj_acc = float(sum(1 for t in yho_labels if t == majority) / len(yho_labels))

    probe_acc = float(accuracy_score(yho_labels, ho_pred))
    # held-out metrics per label source; agreement (vs extracted choice) and
    # correctness (vs outcome labels) reported separately (§15.4).
    by_source: dict[str, dict] = {}
    for (_, _, source, _), pred, true in zip(hold_rows, ho_pred, yho_labels):
        e = by_source.setdefault(source, {"n": 0, "correct": 0})
        e["n"] += 1
        e["correct"] += pred == true
    for e in by_source.values():
        e["acc"] = e["correct"] / e["n"] if e["n"] else None
    ex = by_source.get("extracted", {"n": 0, "correct": 0})
    oc = by_source.get("outcome_correct", {"n": 0, "correct": 0})
    oo = by_source.get("outcome_ok", {"n": 0, "correct": 0})
    cn = oc["n"] + oo["n"]
    card: dict = {
        "point_id": point_id,
        "point_version": point.get("point_version", 1),
        "options": options,
        "kind": point["kind"],
        "n_labeled": len(labeled),
        "n_train": int(n_train_rows),
        "n_holdout": int(n_hold),
        "split": {"runs_train": len(run_order) - len(hold_runs),
                  "runs_hold": len(hold_runs),
                  "solo_hold": sum(1 for i in ho_idx if kept[i][0].get("run_id") is None)},
        "extraction_gate": gate,
        "plan_train_n": plan_train_n,
        "label_sources": _count_sources(kept),
        "eval_only_excluded": n_eval_excluded,
        "heldout_by_source": by_source,
        "agreement": (ex["correct"] / ex["n"] if ex["n"] else None),
        "correctness": ((oc["correct"] + oo["correct"]) / cn if cn else None),
        "probe_accuracy": probe_acc,
        "probe_macro_f1": float(f1_score(yho_labels, ho_pred, average="macro",
                                         zero_division=0)),
        "zero_shot_accuracy": (float(accuracy_score(zs_true, zs_pred)) if zs_true else None),
        "zero_shot_macro_f1": (float(f1_score(zs_true, zs_pred, average="macro",
                                              zero_division=0)) if zs_true else None),
        "zero_shot_n": len(zs_true),
        "majority_accuracy": maj_acc,
        "majority_class": majority,
        "per_option": _per_option(options, yho_labels, ho_pred),
        "ece": _ece(list(yho), ho_proba),
        "random_baseline_error": 1.0 - probe_acc,
        "C": best_c,
        "cv_scores": cv_scores,
        "temperature": temp,
        "vectors": {"cached": n_cached, "computed": n_computed, "excluded_no_text": n_no_text},
        "train_seconds": round(time.monotonic() - t_start, 3),
    }

    # 6. threshold (§6.1): lowest t with point-estimate error <= target and count >= support
    target = point.get("target_error", cfg["serve"]["target_error"])
    min_support = cfg["serve"]["min_auto_support"]
    order = sorted(range(len(hold_rows)), key=lambda i: ho_conf[i])
    threshold: float | None = None
    for i in order:
        t = ho_conf[i]
        idx = [j for j in range(len(hold_rows)) if ho_conf[j] >= t]
        errs = sum(1 for j in idx if ho_pred[j] != yho_labels[j])
        if len(idx) >= min_support and errs / len(idx) <= target:
            threshold = float(t)
            break
    if threshold is not None:
        idx = [j for j in range(len(hold_rows)) if ho_conf[j] >= threshold]
        errs = sum(1 for j in idx if ho_pred[j] != yho_labels[j])
        live = [j for j in idx
                if hold_rows[j][0].get("source") in L.LIVE_SOURCES]
        e_live = sum(1 for j in live if ho_pred[j] != yho_labels[j])
        card.update({
            "threshold": threshold,
            "coverage": len(idx) / len(hold_rows),
            "n_at_threshold": len(idx),
            "error_at_threshold": errs / len(idx),
            "error_at_threshold_wilson_upper": S.wilson_upper(errs, len(idx)),
            # T-EX10: §6.4 evidence never counts extracted rows.
            "n_at_threshold_live": len(live),
            "error_at_threshold_live": (e_live / len(live)) if live else None,
        })
    else:
        card.update({"threshold": None, "coverage": 0.0, "n_at_threshold": 0,
                     "error_at_threshold": None, "error_at_threshold_wilson_upper": None,
                     "n_at_threshold_live": 0, "error_at_threshold_live": None})

    # plan threshold (§15.6 item 2): fitted as in §6.1 on run-split held-out
    # rows, with the plan target error.
    p_target = cfg["plan"]["target_error"]
    plan_threshold: float | None = None
    porder = sorted(range(len(plan_eval)), key=lambda i: plan_eval[i][0])
    for i in porder:
        t = plan_eval[i][0]
        pidx = [j for j in range(len(plan_eval)) if plan_eval[j][0] >= t]
        perrs = sum(1 for j in pidx if not plan_eval[j][1])
        if len(pidx) >= min_support and perrs / len(pidx) <= p_target:
            plan_threshold = float(t)
            break
    if plan_threshold is not None:
        pidx = [j for j in range(len(plan_eval)) if plan_eval[j][0] >= plan_threshold]
        perrs = sum(1 for j in pidx if not plan_eval[j][1])
        card.update({
            "plan_threshold": plan_threshold,
            "plan_coverage": len(pidx) / len(plan_eval),
            "plan_n_at_threshold": len(pidx),
            "plan_error_at_threshold": perrs / len(pidx),
            "plan_heldout_agreement": 1.0 - perrs / len(pidx),
        })
    else:
        card.update({"plan_threshold": None, "plan_coverage": 0.0,
                     "plan_n_at_threshold": 0, "plan_error_at_threshold": None,
                     "plan_heldout_agreement": None})
    card["plan_probe_accuracy"] = (sum(1 for _, ok_ in plan_eval if ok_)
                                   / len(plan_eval) if plan_eval else None)
    card["plan_eval_n"] = len(plan_eval)

    min_recall = cfg["serve"]["min_option_recall"]
    teacher_only = [o for o in options
                    if card["per_option"][o]["recall"] < min_recall
                    or card["per_option"][o]["n"] < 10]
    card["teacher_only_options"] = teacher_only
    def _at_start(r):
        return bool(r.get("at_start"))  # unknown is not at_start (B14)

    n_at_start = sum(1 for r, _, _, _ in labeled if _at_start(r))
    at_start_rate = (n_at_start / len(labeled)) if labeled else 0.0
    card["plan_eligible"] = bool(at_start_rate >= cfg["plan"]["min_at_start"])
    card["plan_eligible_detail"] = {"at_start_rate": at_start_rate,
                                    "n_at_start": n_at_start,
                                    "n_labeled": len(labeled),
                                    "min_at_start": cfg["plan"]["min_at_start"]}

    if dry_run:
        card["model_version"] = "dry-run"
        return {"point_id": point_id, "point_version": card["point_version"],
                "model_version": "dry-run", "path": None, "card": card, "dry_run": True}
    if point.get("extraction_unreliable", False) != gate["unreliable"]:
        point["extraction_unreliable"] = gate["unreliable"]
        store.put_point(point)
    model_version, fname = _next_version(store, point_id, card["point_version"])
    card["model_version"] = model_version
    card["created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    dest_dir = store.models_dir / point_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    np.savez(str(dest_dir / fname),
             W=np.asarray(W, dtype=np.float32), b=np.asarray(b, dtype=np.float32),
             temp=np.asarray(temp, dtype=np.float32),
             classes=np.asarray(options, dtype="<U64"))
    if plan_model is not None:
        card["plan_path"] = fname.replace(".npz", ".plan.npz")
    (dest_dir / fname).with_suffix(".json").write_text(json.dumps(card, indent=1))
    store.save_model(point_id, card["point_version"], "probe", fname, card)
    if plan_model is not None:
        np.savez(str(dest_dir / card["plan_path"]),
                 W=np.asarray(plan_W, dtype=np.float32),
                 b=np.asarray(plan_b, dtype=np.float32),
                 temp=np.asarray(1.0, dtype=np.float32),
                 classes=np.asarray(options, dtype="<U64"))
        store.save_model(point_id, card["point_version"], "plan", card["plan_path"], card)
    return {"point_id": point_id, "point_version": card["point_version"],
            "model_version": model_version, "path": fname, "card": card, "dry_run": False}


def _coef_to_full(clf, dim: int, options: list[str]):
    """Expand a fitted LogisticRegression to full (D, K) W and (K,) b."""
    W = np.zeros((dim, len(options)))
    b = np.zeros(len(options))
    if len(clf.classes_) == 2:
        # binary coef_ has shape (1, D): row 0 points at the larger class.
        c0, c1 = int(clf.classes_[0]), int(clf.classes_[1])
        W[:, c1] = clf.coef_[0]
        W[:, c0] = -clf.coef_[0]
        b[c1] = float(clf.intercept_[0])
        b[c0] = -float(clf.intercept_[0])
    else:
        for j, cls in enumerate(clf.classes_):
            W[:, int(cls)] = clf.coef_[j]
            b[int(cls)] = float(clf.intercept_[j])
    return W, b


def _count_sources(kept) -> dict[str, int]:
    out: dict[str, int] = {}
    for _, _, source, _ in kept:
        out[source] = out.get(source, 0) + 1
    return out


def _per_option(options, y_true, y_pred) -> dict[str, dict]:
    out = {}
    for o in options:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == o and p == o)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == o and p != o)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != o and p == o)
        out[o] = {"recall": tp / (tp + fn) if (tp + fn) else 0.0,
                  "precision": tp / (tp + fp) if (tp + fp) else 0.0,
                  "n": tp + fn}
    return out


def _norm_value(v):
    if isinstance(v, bool):
        return ("bool", v)
    if isinstance(v, (int, float)):
        return ("num", float(v))
    if isinstance(v, str):
        return ("str", v[:40])
    return ("other", repr(v))


def _record_key(r: dict) -> tuple:
    """Canonical agreement identity: choice, at_start, decision event, and
    normalized context (§15.3 equality: numbers equal, strings' first 40)."""
    c = r.get("context") or {}
    return (r.get("choice"), bool(r.get("at_start")),
            (r.get("decision_pos") or {}).get("event"),
            tuple(sorted((k, _norm_value(v)) for k, v in c.items())))


def _records_agree(r1: dict | None, r2: dict | None) -> bool:
    """Single-record agreement (§15.3): same choice, same at_start, key
    context values equal, decision_pos in the same event. Every stored
    context field is key, not only required ones (B14)."""
    if (r1 is None) != (r2 is None):
        return False
    if r1 is None or r2 is None:
        return True
    return _record_key(r1) == _record_key(r2)


def extraction_gate(cfg: dict, store, point: dict) -> dict:
    """The §15.3 gate for a point: its extracted rows train only when all
    three legs hold (>= min_audited audited transcripts, agreement >=
    min_agreement, audited share >= 0.8 x audit_rate)."""
    wf = point["workflow"]
    n1 = n_both = agree = 0
    for t in store.transcripts_for_workflow(wf):
        if t.get("eval_only"):
            continue  # pilot arm runs never gate extraction quality
        exs = store.extractions_for_transcript(t["transcript_id"])
        p1 = any(e["pass"] == 1 and e["status"] == "completed" for e in exs)
        if not p1:
            continue
        n1 += 1
        if not any(e["pass"] == 2 and e["status"] == "completed" for e in exs):
            continue
        n_both += 1
        rows1 = [{
            "choice": r.get("extracted_choice"), "at_start": r.get("at_start"),
            "decision_pos": r.get("decision_pos"), "context": r.get("context"),
        } for r in store.decisions_for_transcript(t["transcript_id"])
            if r["point_id"] == point["point_id"] and r.get("source") == "extracted"]
        recs2 = [a["record"] for a in
                 store.audits_for_transcript(t["transcript_id"], point["point_id"])]
        if len(rows1) != len(recs2):
            continue
        if not rows1:
            # both passes found no decision: agreement only for repeatable
            # points (a run may genuinely hold none). For other points a
            # double skip is an extractor failure, not agreement (B14).
            agree += bool(point.get("repeatable"))
        elif len(rows1) == 1:
            agree += _records_agree(rows1[0], recs2[0])
        else:  # repeatable: full-identity multisets must match
            agree += sorted(_record_key(r) for r in rows1) == sorted(
                _record_key(r) for r in recs2)
    agreement = agree / n_both if n_both else None
    share = n_both / n1 if n1 else 1.0
    audited_ok = n_both >= cfg["extract"]["min_audited"]
    agree_ok = agreement is not None and agreement >= cfg["extract"]["min_agreement"]
    share_ok = share >= 0.8 * cfg["extract"]["audit_rate"]
    return {"audited": n_both, "n_pass1": n1, "agree": agree,
            "agreement": agreement, "share": share,
            "passed": bool(audited_ok and agree_ok and share_ok),
            "unreliable": bool(n_both >= cfg["extract"]["min_audited"]
                               and agreement is not None
                               and agreement < cfg["extract"]["min_agreement"])}


def train_fake_private(store, point_id: str, threshold: float = 0.7,
                     model_version: str = "fake-private-1") -> dict:
    """Test-only private-tier stand-in (H5): records the export marker (the
    latest decision) and a serve threshold as a kind="private" model. The
    inference probabilities come from the test's FakeDecider. M11 replaces
    this with the Fireworks LoRA job."""
    from .errors import ToolError  # local import: errors has no deps
    point = store.get_point(point_id)
    if point is None:
        raise ToolError("unknown_point", f"unknown point_id {point_id!r}")
    rows = store.decisions_for_point(point_id)
    marker = max(((r.get("ts") or "", r.get("decision_id") or "")
                  for r in rows), default=("", ""))
    card = {"point_id": point_id,
            "point_version": point.get("point_version", 1),
            "kind": "private", "model_version": model_version,
            "threshold": threshold, "export_ts": marker[0],
            "export_decision_id": marker[1], "fake": True}
    store.save_model(point_id, point.get("point_version", 1), "private",
                     f"{model_version}.json", card)
    return {"point_id": point_id, "model_version": model_version,
            "card": card}


def _next_version(store, point_id: str, point_version: int) -> tuple[str, str]:
    dest = store.models_dir / point_id
    n = 0
    if dest.is_dir():
        for f in dest.glob(f"v{point_version}-*.npz"):
            try:
                n = max(n, int(f.stem.split("-")[1]))
            except (IndexError, ValueError):
                pass
    n += 1
    return f"v{point_version}-{n}", f"v{point_version}-{n}.npz"
