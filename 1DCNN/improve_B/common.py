"""
Shared helpers for farag-improve (B1-B5).

Data: full DS1 (22 records) with per-beat record ids, DS2 (test only).
Weights: w_c = (1/freq_c)^p normalised so that sum_c n_c w_c = n
         (p=0 -> no weighting, p=1 -> sklearn 'balanced').
Decision rule: y = argmax(log P + [0, bS, bV])  (per-class logit offsets).
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE   = Path(__file__).resolve().parent              # $ROOT/1DCNN/improve_B
ROOT   = HERE.parents[1]
PAPER3 = ROOT / "HasanML" / "paper3"
EXPS   = ROOT / "1DCNN" / "1DCNN experiments"
CACHE  = HERE / "cache"
RUNS   = HERE / "runs"
RES    = HERE / "results"
for d in (CACHE, RUNS, RES):
    d.mkdir(exist_ok=True, parents=True)

if str(PAPER3) not in sys.path:
    sys.path.insert(0, str(PAPER3))

K = 3
CLS = ["N", "S", "V"]
N_FOLDS = 5

# Farag 2023 Table 4 (verified by farag-gap, reports/farag-gap.md). ACC is PLAIN accuracy.
# Model 6 = 3-class, NK=32, derivative, MF-trainable, NO class weights (our B1 config).
# Model 2 = same but with class weights (our official P1 config).
SE_S_TARGET, SE_V_TARGET = 0.816, 0.950   # Model 6 Se_S / Se_V


def load_cfg():
    import yaml
    with open(PAPER3 / "configs" / "default.yaml") as fh:
        cfg = yaml.safe_load(fh)
    cfg["input"] = {**cfg["input"], "mode": "derivative", "nk": 32}
    cfg["task"] = {**cfg["task"], "init": "mf_trainable", "classes": 3}
    return cfg


def load_data():
    d = np.load(CACHE / "data.npz", allow_pickle=False)
    return {k: d[k] for k in d.files}


def fold_assignment():
    with open(CACHE / "folds.json") as fh:
        return json.load(fh)   # {"folds": [[rec,...],...]}


def class_weights(y, p):
    n_c = np.bincount(y, minlength=K).astype(float)
    w = (1.0 / np.maximum(n_c, 1)) ** p
    w = w * len(y) / float((n_c * w).sum())
    return {i: float(w[i]) for i in range(K)}


# ---------------------------------------------------------------- metrics
def confusion(y, yp):
    return np.bincount(y * K + yp, minlength=K * K).reshape(K, K)


def metrics_from_cm(cm):
    cm = np.asarray(cm, float)
    n = cm.sum()
    tp = np.diag(cm); fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = n - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nan_to_num(tp / (tp + fn)); ppv = np.nan_to_num(tp / (tp + fp))
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn)); sp = np.nan_to_num(tn / (tn + fp))
    m = {"accuracy": tp.sum() / n, "ovr_accuracy": float(np.mean((tp + tn) / n)),
         "macro_f1": f1.mean(), "macro_ppv": ppv.mean(), "macro_se": se.mean()}
    for i, c in enumerate(CLS):
        m[f"se_{c}"] = se[i]; m[f"ppv_{c}"] = ppv[i]; m[f"f1_{c}"] = f1[i]; m[f"sp_{c}"] = sp[i]
    m = {k: float(v) for k, v in m.items()}
    m["cm"] = cm.astype(int).tolist()
    return m


def predict_offsets(proba, bS=0.0, bV=0.0):
    lp = np.log(np.clip(proba, 1e-12, 1.0))
    return np.argmax(lp + np.array([0.0, bS, bV]), axis=1)


def evaluate(y, proba, bS=0.0, bV=0.0):
    return metrics_from_cm(confusion(y, predict_offsets(proba, bS, bV)))


# ---------------------------------------------------------------- calibration
GRID = np.round(np.arange(-6.0, 6.0001, 0.1), 2)


def offset_grid_cms(y, proba, grid=GRID):
    """Confusion matrices for every (bS, bV) in grid x grid -> array (G, G, K, K)."""
    lp = np.log(np.clip(proba, 1e-12, 1.0))
    dS = lp[:, 1] - lp[:, 0]
    dV = lp[:, 2] - lp[:, 0]
    G = len(grid)
    out = np.zeros((G, G, K, K), dtype=np.int64)
    for i, bS in enumerate(grid):
        s = dS + bS
        for j, bV in enumerate(grid):
            v = dV + bV
            yp = np.where((s < 0) & (v < 0), 0, np.where(s >= v, 1, 2))
            out[i, j] = np.bincount(y * K + yp, minlength=K * K).reshape(K, K)
    return out


def grid_metrics(cms):
    """Vectorised metrics over (G,G,K,K) confusion matrices."""
    cm = cms.astype(float)
    n = cm.sum(axis=(-1, -2))
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    fn = cm.sum(-1) - tp
    fp = cm.sum(-2) - tp
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.nan_to_num(tp / (tp + fn)); ppv = np.nan_to_num(tp / (tp + fp))
        f1 = np.nan_to_num(2 * tp / (2 * tp + fp + fn))
    return {"acc": tp.sum(-1) / n, "macro_f1": f1.mean(-1), "macro_ppv": ppv.mean(-1),
            "se": se, "ppv": ppv, "f1": f1}


def choose_points(y, proba, grid=GRID):
    """Pick operating points on calibration data (DS1 only).
    Returns dict name -> (bS, bV, calib metrics)."""
    cms = offset_grid_cms(y, proba, grid)
    gm = grid_metrics(cms)
    pts = {}

    def _pick(score, name):
        i, j = np.unravel_index(np.nanargmax(score), score.shape)
        pts[name] = (float(grid[i]), float(grid[j]))

    # small tie-break toward offsets near 0
    reg = -1e-6 * (np.abs(grid)[:, None] + np.abs(grid)[None, :])
    _pick(gm["macro_f1"] + reg, "maxF1")
    _pick(gm["acc"] + reg, "maxAcc")          # == max OvR acc (OvR = 1-2e/K)
    feas = (gm["se"][..., 1] >= SE_S_TARGET) & (gm["se"][..., 2] >= SE_V_TARGET)
    if feas.any():
        _pick(np.where(feas, gm["acc"], -np.inf) + reg, "SeConstr")
        pts["SeConstr_feasible"] = True
    else:
        # Targets unreachable on held-out-patient data: fall back to the point that
        # maximises macro Se (balanced accuracy, the paper's "avg Se").
        _pick(gm["se"].mean(-1) + reg, "SeConstr")
        pts["SeConstr_feasible"] = False
    return pts, gm


def fmt_row(name, m, extra=""):
    return ("| {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | {:.2f} | "
            "{:.2f} | {:.2f} | {:.2f} |{}").format(
        name, 100 * m["accuracy"], 100 * m["ovr_accuracy"], 100 * m["macro_f1"],
        100 * m["macro_ppv"], 100 * m["macro_se"],
        100 * m["se_N"], 100 * m["ppv_N"], 100 * m["se_S"], 100 * m["ppv_S"],
        100 * m["se_V"], 100 * m["ppv_V"], extra)


TABLE_HDR = ("| Config | Acc | OvR Acc | macro F1 | macro PREC | macro Se | Se_N | PPV_N | "
             "Se_S | PPV_S | Se_V | PPV_V |")
TABLE_SEP = "|" + "---|" * 12
PAPER_ROW = ("| **Paper Model 6 (no CW, seed mean)** | 98.18 | 98.79 | 92.17 | 92.44 | 91.90 | 99.10 | 99.00 | "
             "81.60 | 82.68 | 95.00 | 95.63 |\n"
             "| **Paper Model 2 (CW, seed mean)** | 95.58 | 97.05 | 84.79 | 79.63 | 93.34 | 95.86 | 99.47 | "
             "88.51 | 51.49 | 95.65 | 87.93 |")
