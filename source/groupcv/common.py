"""
Shared helpers for the patient-grouped CV work (agent: lgbm-groupcv).

Everything here touches DS1 only for fitting / model selection.  DS2 is read
only by `load_ds2()` and only used for the final, reported test metrics.

Grouped CV
----------
StratifiedGroupKFold(5) by record on DS1.  With this data it is
deterministic (shuffle seed does not change the partition) and gives:

    fold 0: 201 215 223
    fold 1: 114 118 119 205 230          (R: 118)
    fold 2: 101 115 208 209
    fold 3: 108 112 203 207              (L+R: 207)
    fold 4: 106 109 116 122 124 220      (L: 109, R: 124)

So every training fold still contains at least one LBBB and one RBBB patient,
and each LBBB/RBBB patient is validated by a model that never saw it.

Objective = macro-F1 of the *pooled* out-of-fold predictions (all 50,307 DS1
beats predicted by a model that did not see that patient).  Per-fold macro-F1
is unstable here because several folds contain no L or R beats at all.
SMOTE (or nothing, for class-weight variants) is fit inside each training fold.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "datasets" / "processed"
MODEL_DIR = ROOT / "models"
CKPT_DIR = MODEL_DIR / "checkpoints"
OUT_DIR = ROOT / "reports" / "lgbm-groupcv"

SEED = 42
N_JOBS = int(os.environ.get("GROUPCV_NJOBS", "8"))
LABELS = ["N", "L", "R", "V", "A"]

FEATURE_NAMES = [str(s) for s in np.load(
    PROC_DIR / "mit_bih_features.npz", allow_pickle=True)["feature_names"]]

# Feature groups (same slices as evaluate.py ABLATION_CONFIGS, B split in two)
FEATURE_GROUPS = {
    "A_stat": list(range(0, 6)),
    "B1_intervals": list(range(6, 11)),
    "B2_morph": list(range(11, 19)),
    "C_dwt": list(range(19, 31)),
    "D1_hrv": list(range(31, 36)),
    "D2_nonlin": list(range(36, 40)),
}

PAPER_PARAMS = {
    "n_estimators": 1450, "max_depth": 11, "learning_rate": 0.05,
    "num_leaves": 38, "subsample": 0.85, "colsample_bytree": 0.75,
}
# best params of the existing models/lgbm_ds1ds2.pkl (beat-level CV, trial #94)
OLD_DS1DS2_PARAMS = {
    "n_estimators": 1282, "max_depth": 12, "learning_rate": 0.02797,
    "num_leaves": 46, "subsample": 0.6093, "colsample_bytree": 0.9151,
}


def load_ds1():
    d = np.load(PROC_DIR / "split_ds1ds2.npz", allow_pickle=True)
    return d["X_train"], d["y_train"], d["record_ids_train"]


def load_ds2():
    d = np.load(PROC_DIR / "split_ds1ds2.npz", allow_pickle=True)
    return d["X_test"], d["y_test"], d["record_ids_test"]


def group_folds(y, groups, n_splits: int = 5, seed: int = SEED):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(np.zeros(len(y)), y, groups))


def logo_folds(groups):
    recs = np.unique(groups)
    return [(np.where(groups != r)[0], np.where(groups == r)[0]) for r in recs]


def make_clf(params: dict, class_weight=None, device: str = "cpu",
             n_jobs: int = N_JOBS) -> LGBMClassifier:
    p = dict(params)
    p.setdefault("subsample_freq", 1)
    return LGBMClassifier(
        **p, class_weight=class_weight, device=device,
        random_state=SEED, n_jobs=n_jobs, verbose=-1,
    )


def fit_model(X, y, params, sampler: str = "smote", class_weight=None,
              device: str = "cpu", n_jobs: int = N_JOBS):
    """sampler: 'smote' (paper pipeline) or 'none'."""
    if sampler == "smote":
        X, y = SMOTE(random_state=SEED).fit_resample(X, y)
    clf = make_clf(params, class_weight=class_weight, device=device, n_jobs=n_jobs)
    clf.fit(X, y)
    return clf


def grouped_cv(X, y, folds, params, sampler="smote", class_weight=None,
               feat_idx=None, device="cpu", n_jobs=N_JOBS, return_models=False):
    """Pooled out-of-fold evaluation. Returns dict of scores + OOF arrays."""
    if feat_idx is not None:
        X = X[:, feat_idx]
    oof = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(LABELS)))
    fold_f1, models = [], []
    t0 = time.time()
    for tr, va in folds:
        m = fit_model(X[tr], y[tr], params, sampler, class_weight, device, n_jobs)
        oof[va] = m.predict(X[va])
        p = m.predict_proba(X[va])
        cols = [LABELS.index(c) for c in m.classes_]
        proba[np.ix_(va, cols)] = p
        fold_f1.append(f1_score(y[va], oof[va], average="macro", zero_division=0))
        if return_models:
            models.append(m)
    res = scores(y, oof.astype(str), proba)
    res["fold_macro_f1"] = [float(f) for f in fold_f1]
    res["mean_fold_macro_f1"] = float(np.mean(fold_f1))
    res["seconds"] = time.time() - t0
    res["oof_pred"] = oof.astype(str)
    res["oof_proba"] = proba
    if return_models:
        res["models"] = models
    return res


def to_binary(y):
    return np.where(np.asarray(y) == "N", 0, 1).astype(int)


def scores(y, y_pred, proba=None) -> dict:
    """Same metric set as evaluate.run_main_eval (labels in LABELS order)."""
    y = np.asarray(y).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    yb, pb = to_binary(y), to_binary(y_pred)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, y_pred, labels=LABELS, zero_division=0)
    out = {
        "acc": float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "bin_p": float(precision_score(yb, pb, zero_division=0)),
        "bin_r": float(recall_score(yb, pb, zero_division=0)),
        "bin_f1": float(f1_score(yb, pb, zero_division=0)),
        "per_class": {c: {"p": float(prec[i]), "r": float(rec[i]),
                          "f1": float(f1[i]), "n": int(sup[i])}
                      for i, c in enumerate(LABELS)},
    }
    if proba is not None:
        try:
            # sklearn requires `labels` sorted (as evaluate.py's model.classes_ is)
            order = sorted(LABELS)
            cols = [LABELS.index(c) for c in order]
            out["ovr_auc"] = float(roc_auc_score(
                y, proba[:, cols], multi_class="ovr", average="macro", labels=order))
        except Exception:
            out["ovr_auc"] = float("nan")
        anom = proba[:, [i for i, c in enumerate(LABELS) if c != "N"]].sum(1)
        out["bin_auc"] = float(roc_auc_score(yb, anom))
    return out


def eval_ds2(model, feat_idx=None) -> dict:
    X, y, _ = load_ds2()
    if feat_idx is not None:
        X = X[:, feat_idx]
    pred = model.predict(X)
    p = model.predict_proba(X)
    proba = np.zeros((len(y), len(LABELS)))
    proba[:, [LABELS.index(c) for c in model.classes_]] = p
    return scores(y, pred, proba)


def fmt(res: dict, name: str = "") -> str:
    pc = res["per_class"]
    return (f"{name:<38} acc={res['acc']:.4f} mF1={res['macro_f1']:.4f} "
            f"AUC={res.get('ovr_auc', float('nan')):.4f} "
            f"binF1={res['bin_f1']:.4f} binAUC={res.get('bin_auc', float('nan')):.4f} | "
            + " ".join(f"{c}={pc[c]['f1']:.3f}" for c in LABELS))


def strip(res: dict) -> dict:
    """JSON-serialisable copy (drops arrays/models)."""
    return {k: v for k, v in res.items()
            if k not in ("oof_pred", "oof_proba", "models")}
