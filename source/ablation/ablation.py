"""
Feature-group ablation A1->A4 on the normalised (2026-11-13) features.

Two modes
---------
  --mode random   Calls source/evaluate.run_ablation() unchanged (random split,
                  paper params, SMOTE, 5-class LightGBM). The only change is
                  that LightGBM is forced to N_THREADS threads (evaluate.py uses
                  n_jobs=-1, which would grab every core on the shared box).

  --mode ds1ds2   Same A1->A4 configs (plus single-group and drop-one-group
                  extras with --extras) trained on DS1, tested on DS2.
                  For every config two models are fitted with paper params:
                    * 5-class LightGBM + SMOTE  (same as evaluate.run_ablation)
                    * binary LightGBM (N vs L/R/V/A) + SMOTE
                  DS2 is only used for the final test numbers, never for any
                  choice.

Usage (from $ROOT):
  .venv/bin/python source/ablation/ablation.py --mode random
  .venv/bin/python source/ablation/ablation.py --mode ds1ds2 --extras
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

N_THREADS = int(os.environ.get("ABL_THREADS", "6"))
os.environ.setdefault("OMP_NUM_THREADS", str(N_THREADS))

import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

_HERE    = Path(__file__).resolve().parent
ROOT     = _HERE.parent.parent
PROC_DIR = ROOT / "datasets" / "processed"
OUT_DIR  = _HERE / "results"
sys.path.insert(0, str(ROOT / "source"))

SEED = 42
LABELS = ["N", "L", "R", "V", "A"]
PAPER_PARAMS = {
    "n_estimators": 1450, "max_depth": 11, "learning_rate": 0.05,
    "num_leaves": 38, "subsample": 0.85, "colsample_bytree": 0.75,
}
GROUPS = {
    "A": list(range(0, 6)),    # statistical
    "B": list(range(6, 19)),   # fiducial
    "C": list(range(19, 31)),  # wavelet
    "D": list(range(31, 40)),  # HRV + non-linear
}
CONFIGS = [
    ("A1 Statistical", "A"),
    ("A2 +Fiducial", "AB"),
    ("A3 +Wavelet", "ABC"),
    ("A4 +HRV/NonLin", "ABCD"),
]
EXTRA_CONFIGS = [
    ("only Fiducial", "B"),
    ("only Wavelet", "C"),
    ("only HRV/NonLin", "D"),
    ("all - Statistical", "BCD"),
    ("all - Fiducial", "ACD"),
    ("all - Wavelet", "ABD"),
    ("all - HRV/NonLin", "ABC"),
]


def _cols(groups: str) -> list[int]:
    return [c for g in groups for c in GROUPS[g]]


def _lgbm_capped(**kw):
    """LGBMClassifier factory with n_jobs forced to N_THREADS."""
    kw["n_jobs"] = N_THREADS
    return LGBMClassifier(**kw)


def run_random_paper() -> None:
    import evaluate  # source/evaluate.py, read-only
    evaluate.LGBMClassifier = _lgbm_capped
    evaluate.run_ablation()


def _fit_eval(Xtr, ytr5, Xte, yte5) -> dict:
    ytr_b = (ytr5 != "N").astype(int)
    yte_b = (yte5 != "N").astype(int)
    out = {}

    # 5-class
    Xr, yr = SMOTE(random_state=SEED).fit_resample(Xtr, ytr5)
    m = LGBMClassifier(**PAPER_PARAMS, subsample_freq=1, random_state=SEED,
                       n_jobs=N_THREADS, verbose=-1).fit(Xr, yr)
    pred = m.predict(Xte)
    proba = m.predict_proba(Xte)
    anom = proba[:, [i for i, c in enumerate(m.classes_) if c != "N"]].sum(1)
    pb = (pred != "N").astype(int)
    out["5c"] = {
        "acc": accuracy_score(yte5, pred),
        "macro_f1": f1_score(yte5, pred, average="macro", labels=LABELS, zero_division=0),
        "per_class_f1": dict(zip(LABELS, f1_score(yte5, pred, average=None,
                                                   labels=LABELS, zero_division=0).tolist())),
        "anom_prec": precision_score(yte_b, pb, zero_division=0),
        "anom_rec": recall_score(yte_b, pb, zero_division=0),
        "anom_f1": f1_score(yte_b, pb, zero_division=0),
        "anom_auc": roc_auc_score(yte_b, anom),
    }

    # binary objective
    Xr, yr = SMOTE(random_state=SEED).fit_resample(Xtr, ytr_b)
    mb = LGBMClassifier(**PAPER_PARAMS, objective="binary", subsample_freq=1,
                        random_state=SEED, n_jobs=N_THREADS, verbose=-1).fit(Xr, yr)
    p = mb.predict_proba(Xte)[:, 1]
    pb = (p >= 0.5).astype(int)
    out["bin"] = {
        "acc": accuracy_score(yte_b, pb),
        "prec": precision_score(yte_b, pb, zero_division=0),
        "rec": recall_score(yte_b, pb, zero_division=0),
        "f1": f1_score(yte_b, pb, zero_division=0),
        "auc": roc_auc_score(yte_b, p),
        "rec_by_class": {c: float(pb[yte5 == c].mean()) if c != "N"
                         else float(1 - pb[yte5 == c].mean()) for c in LABELS},
    }
    return out


def run_ds1ds2(extras: bool) -> None:
    d = np.load(PROC_DIR / "split_ds1ds2.npz", allow_pickle=True)
    Xtr, ytr, Xte, yte = d["X_train"], d["y_train"], d["X_test"], d["y_test"]
    configs = CONFIGS + (EXTRA_CONFIGS if extras else [])
    OUT_DIR.mkdir(exist_ok=True)
    out_json = OUT_DIR / "ablation_ds1ds2.json"
    results = json.loads(out_json.read_text()) if out_json.exists() else {}

    for name, groups in configs:
        if name in results:
            print(f"[cached] {name}")
            continue
        cols = _cols(groups)
        t0 = time.time()
        r = _fit_eval(Xtr[:, cols], ytr, Xte[:, cols], yte)
        r["n_feats"] = len(cols)
        r["groups"] = groups
        r["secs"] = round(time.time() - t0, 1)
        results[name] = r
        out_json.write_text(json.dumps(results, indent=2))
        print(f"done {name} in {r['secs']}s", flush=True)

    print("\nDS1 -> DS2, paper params, SMOTE")
    print(f"{'Config':<20}{'#F':>4} | {'5c Acc':>7}{'mF1':>7}{'F1_N':>6}{'F1_L':>6}"
          f"{'F1_R':>6}{'F1_V':>6}{'F1_A':>6}{'cAnF1':>7}{'cAUC':>7} | "
          f"{'bin P':>6}{'R':>6}{'F1':>6}{'AUC':>6}{'rL':>6}{'rR':>6}{'rV':>6}{'rA':>6}")
    for name, _ in configs:
        r = results[name]; a = r["5c"]; b = r["bin"]; pc = a["per_class_f1"]; rc = b["rec_by_class"]
        print(f"{name:<20}{r['n_feats']:>4} | {a['acc']:>7.3f}{a['macro_f1']:>7.3f}"
              + "".join(f"{pc[c]:>6.3f}" for c in LABELS)
              + f"{a['anom_f1']:>7.3f}{a['anom_auc']:>7.3f} | "
              f"{b['prec']:>6.3f}{b['rec']:>6.3f}{b['f1']:>6.3f}{b['auc']:>6.3f}"
              + "".join(f"{rc[c]:>6.3f}" for c in "LRVA"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["random", "ds1ds2"], required=True)
    ap.add_argument("--extras", action="store_true")
    a = ap.parse_args()
    if a.mode == "random":
        run_random_paper()
    else:
        run_ds1ds2(a.extras)
