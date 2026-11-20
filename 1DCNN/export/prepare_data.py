"""
Build the arrays the export pipeline needs, numpy only (works in any venv).

P1 (3-class N/S/V): reuses the preprocessed caches of the official P1 run
    (1DCNN/1DCNN experiments/.cache/ds1_all.npz, ds2.npz) and re-creates the
    exact 15 % random-beat hold-out of run_official_p1.py (rng seed 42), so
    "train" here == the beats the official P1 models were trained on, and
    "hold" == their early-stopping hold-out.
P2 (binary AAMI): uses 1DCNN/export/.cache/bin_aami_{ds1_all,ds2}.npz written
    by train_binary_fullds1.py, same split rule.

Output: 1DCNN/export/.cache/data_{p1,p2}.npz with
    X_ecg_train, X_rr_train, y_train, X_ecg_hold, ..., X_ecg_ds2, ...
"""
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXP_CACHE = ROOT / "1DCNN" / "1DCNN experiments" / ".cache"
CACHE = HERE / ".cache"


def split(n, val_frac=0.15, seed=42):
    idx = np.random.default_rng(seed).permutation(n)
    k = max(1, int(n * val_frac))
    return idx[k:], idx[:k]


def build(tag, ds1_path, ds2_path):
    a, b = np.load(ds1_path), np.load(ds2_path)
    ti, vi = split(len(a["y"]))
    out = {}
    for nm, ix in (("train", ti), ("hold", vi)):
        for k in ("X_ecg", "X_rr", "y"):
            out[f"{k}_{nm}"] = a[k][ix]
    for k in ("X_ecg", "X_rr", "y"):
        out[f"{k}_ds2"] = b[k]
    CACHE.mkdir(exist_ok=True)
    np.savez(CACHE / f"data_{tag}.npz", **out)
    print(tag, {k: v.shape for k, v in out.items() if k.startswith("y_")},
          "hold class counts", np.bincount(out["y_hold"]))


if __name__ == "__main__":
    which = sys.argv[1:] or ["p1"]
    if "p1" in which:
        build("p1", EXP_CACHE / "ds1_all.npz", EXP_CACHE / "ds2.npz")
    if "p2" in which:
        build("p2", CACHE / "bin_aami_ds1_all.npz", CACHE / "bin_aami_ds2.npz")
