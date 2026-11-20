"""
farag-gap A5: RR-only XGBoost baseline variants vs Farag (2023) Sec. 3
(paper: acc 93%; recall/precision/F1  N 97/97/97, SVEB 60/36/45, VEB 58/75/65).

Uses the cached 4-feature RR inputs from 1DCNN/1DCNN experiments/.cache (identical
to the CNN's RR branch input).  DS2 is evaluated only; never used to fit/tune.

Run (from $ROOT):
  PYTHONPATH=1DCNN/gap_analysis/pylib OMP_NUM_THREADS=2 \
      .venv/bin/python 1DCNN/gap_analysis/rr_baseline_variants.py
"""
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np
import xgboost as xgb
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_sample_weight

ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "1DCNN" / "1DCNN experiments" / ".cache"
OUT = Path(__file__).resolve().parent / "rr_baseline"
OUT.mkdir(exist_ok=True)

d = {k: np.load(C / f"{k}.npz") for k in ("ds1_train", "ds1_val", "ds1_all", "ds2")}
Xte, yte = d["ds2"]["X_rr"], d["ds2"]["y"]


def fit_eval(name, Xtr, ytr, weighted=False, **kw):
    params = dict(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="mlogloss",
                  random_state=0, n_jobs=2, verbosity=0)
    params.update(kw)
    m = xgb.XGBClassifier(**params)
    sw = compute_sample_weight("balanced", ytr) if weighted else None
    m.fit(Xtr, ytr, sample_weight=sw)
    p = m.predict(Xte)
    pr, rc, f1, _ = precision_recall_fscore_support(yte, p, labels=[0, 1, 2], zero_division=0)
    r = {"name": name, "acc": float((p == yte).mean()),
         "recall": rc.tolist(), "precision": pr.tolist(), "f1": f1.tolist(),
         "macro_recall": float(rc.mean()), "macro_precision": float(pr.mean()),
         "macro_f1": float(f1.mean()),
         "cm": confusion_matrix(yte, p, labels=[0, 1, 2]).tolist()}
    print(f"{name:<48} acc {100*r['acc']:.2f} | Se N/S/V {100*rc[0]:.1f}/{100*rc[1]:.1f}/{100*rc[2]:.1f}"
          f" | PPV {100*pr[0]:.1f}/{100*pr[1]:.1f}/{100*pr[2]:.1f} | F1 {100*f1[0]:.1f}/{100*f1[1]:.1f}/{100*f1[2]:.1f}"
          f" | macro Se {100*rc.mean():.1f} PPV {100*pr.mean():.1f}")
    return r


res = []
print("Paper (Farag 2023 Sec.3): acc 93 | Se N/S/V 97/60/58 | PPV 97/36/75 | F1 97/45/65")
res.append(fit_eval("A DS1-train (12 rec), 300x d6 [= CLI]", d["ds1_train"]["X_rr"], d["ds1_train"]["y"]))
res.append(fit_eval("B full DS1 (22 rec), 300x d6", d["ds1_all"]["X_rr"], d["ds1_all"]["y"]))
res.append(fit_eval("C full DS1, 100x d3", d["ds1_all"]["X_rr"], d["ds1_all"]["y"], n_estimators=100, max_depth=3))
res.append(fit_eval("D full DS1, balanced weights, 300x d6", d["ds1_all"]["X_rr"], d["ds1_all"]["y"], weighted=True))
res.append(fit_eval("E full DS1, balanced weights, 100x d3", d["ds1_all"]["X_rr"], d["ds1_all"]["y"], weighted=True,
                    n_estimators=100, max_depth=3))
res.append(fit_eval("F DS1-train, balanced weights, 300x d6", d["ds1_train"]["X_rr"], d["ds1_train"]["y"], weighted=True))
json.dump(res, open(OUT / "rr_baseline_variants.json", "w"), indent=1)
