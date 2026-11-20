"""
Task 1: how much of the CV-vs-DS2 gap does patient-grouped CV explain?

Scores the existing model's params (models/lgbm_ds1ds2.pkl, beat-level-CV-tuned)
and the paper params under:
  (a) beat-level StratifiedKFold(5) on DS1   (what Optuna optimised before)
  (b) StratifiedGroupKFold(5) by record on DS1
  (c) leave-one-record-out on DS1 (22 folds)   [old params only]
All with SMOTE inside each training fold, pooled OOF metrics.
"""
import json, pickle
import numpy as np
from sklearn.model_selection import StratifiedKFold
from common import *

OUT_DIR.mkdir(parents=True, exist_ok=True)
X, y, g = load_ds1()

with open(MODEL_DIR / "lgbm_ds1ds2.pkl", "rb") as f:
    old = pickle.load(f)
old_params = dict(old["best_params"])
print("old params:", old_params, "stored beat-CV:", old["cv_score"], flush=True)

beat = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y))
grp = group_folds(y, g)
for i, (_, va) in enumerate(grp):
    print(f"  group fold {i}: records {sorted(set(g[va]))}", flush=True)

results, oof = {}, {}
runs = [
    ("old_params/beat5", old_params, beat),  # ran once (log of first run): mF1 0.9862
    ("old_params/group5", old_params, grp),
    ("paper_params/group5", PAPER_PARAMS, grp),
    # ("old_params/logo22", old_params, logo_folds(g)),  # dropped: ~40 min on the loaded box
]
for name, params, folds in runs:
    r = grouped_cv(X, y, folds, params)
    print(fmt(r, name), f"meanFold={r['mean_fold_macro_f1']:.4f} ({r['seconds']:.0f}s)", flush=True)
    results[name] = strip(r)
    oof[name.replace("/", "__")] = r["oof_pred"]
    # per-record view of grouped OOF predictions
    if "beat" not in name:
        per_rec = {}
        for rec in np.unique(g):
            m = g == rec
            yt, yp = y[m], r["oof_pred"][m]
            per_rec[str(rec)] = {
                "acc": float((yt == yp).mean()),
                "true": {c: int((yt == c).sum()) for c in LABELS if (yt == c).any()},
                "pred": {c: int((yp == c).sum()) for c in LABELS if (yp == c).any()},
            }
        results[name]["per_record"] = per_rec

json.dump(results, open(OUT_DIR / "task1_baseline.json", "w"), indent=1)
np.savez_compressed(OUT_DIR / "task1_oof.npz", y=y, groups=g, **oof)
print("saved", OUT_DIR / "task1_baseline.json")
