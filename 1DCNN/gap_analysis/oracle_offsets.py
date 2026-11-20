"""
farag-gap: how much of the gap can decision-threshold changes alone close?
Adds log-prior offsets (b_S, b_V) to the ensemble log-probabilities.

  (1) LEGIT: offsets chosen on the DS1 random-15% hold-out (ds1_is_val), applied to DS2.
  (2) ORACLE (diagnostic only, NOT a reportable result): offsets chosen on DS2 itself,
      giving an upper bound on what any re-thresholding of this ensemble could achieve.

Run (from $ROOT): OMP_NUM_THREADS=2 .venv/bin/python 1DCNN/gap_analysis/oracle_offsets.py
"""
import json
from pathlib import Path
import numpy as np

G = Path(__file__).resolve().parent
d = np.load(G / "ensemble_probs.npz")
grid = np.arange(-6.0, 2.01, 0.25)


def stats(y, p):
    cm = np.zeros((3, 3), int); np.add.at(cm, (y, p), 1)
    tp = np.diag(cm); se = tp / cm.sum(1); ppv = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * tp / (cm.sum(1) + cm.sum(0))
    return dict(acc=tp.sum() / cm.sum(), se=se, ppv=ppv, f1=f1, mf1=f1.mean(), cm=cm)


def predict(P, bS, bV):
    L = np.log(np.clip(P, 1e-9, 1)); L[:, 1] += bS; L[:, 2] += bV
    return L.argmax(1)


def search(P, y, key):
    best = None
    for bS in grid:
        for bV in grid:
            s = stats(y, predict(P, bS, bV))
            if best is None or s[key] > best[0]:
                best = (s[key], bS, bV)
    return best


val = d["ds1_is_val"]
P1, y1 = d["ds1_proba"][val], d["ds1_y"][val].astype(int)
P2, y2 = d["ds2_proba"], d["ds2_y"].astype(int)
out = {}
for key in ("mf1", "acc"):
    for src, (P, y) in (("ds1val", (P1, y1)), ("ORACLE_ds2", (P2, y2))):
        _, bS, bV = search(P, y, key)
        s = stats(y2, predict(P2, bS, bV))
        tag = f"{src}_max_{key}"
        out[tag] = dict(bS=float(bS), bV=float(bV), acc=float(s["acc"]), macro_f1=float(s["mf1"]),
                        se=s["se"].tolist(), ppv=s["ppv"].tolist(), f1=s["f1"].tolist(), cm=s["cm"].tolist())
        print(f"{tag:<22} bS={bS:+.2f} bV={bV:+.2f} | DS2 acc {100*s['acc']:.2f} macroF1 {100*s['mf1']:.2f} "
              f"| Se {np.round(100*s['se'],1)} PPV {np.round(100*s['ppv'],1)}")
json.dump(out, open(G / "oracle_offsets.json", "w"), indent=1)
