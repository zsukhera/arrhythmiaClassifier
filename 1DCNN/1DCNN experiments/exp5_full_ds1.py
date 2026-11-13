"""
Experiment 5: Train on Full DS1 (DS1_TRAIN + DS1_VAL)
------------------------------------------------------
Problem: DS1_TRAIN has only 149 S beats.
         DS1_VAL has an additional 795 S beats (total 944 across all DS1 records).
         We were withholding DS1_VAL for early-stopping but that costs us 84% of S beats.

Fix: Combine DS1_TRAIN + DS1_VAL for training.
     Hold out 15% of the COMBINED beats (random split) for early-stopping validation.
     DS2 patients are still never seen during training.

This is the most impactful change — 149 → 944 S beats (+534%).

Note: The random split mixes DS1_TRAIN and DS1_VAL *beats* (not records) for validation.
      This is acceptable because: (a) all patients are from DS1 (not DS2), and (b) the
      split is only used for early stopping, not final evaluation.

Usage:
    python exp5_full_ds1.py --seeds 0 1 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HasanML" / "paper3"))

from utils import load_config, load_data, load_kernel_mf, print_header, print_row, print_footer


def _split_train_val(X_ecg, X_rr, y, val_frac=0.15, seed=42):
    """Random beat-level train/val split (val is for early stopping only)."""
    rng     = np.random.default_rng(seed)
    n       = len(y)
    idx     = rng.permutation(n)
    n_val   = max(1, int(n * val_frac))
    val_idx = idx[:n_val]
    tr_idx  = idx[n_val:]
    return (
        X_ecg[tr_idx],  X_rr[tr_idx],  y[tr_idx],
        X_ecg[val_idx], X_rr[val_idx], y[val_idx],
    )


def run(seeds: list[int] | None = None, val_frac: float = 0.15,
        verbose: bool = True) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    cfg = load_config()
    cfg["training"]["class_weight"] = "balanced"

    # Load all DS1 data combined, plus DS2 for final evaluation
    ds1_all = load_data("ds1_all", cfg)   # DS1_TRAIN + DS1_VAL
    test    = load_data("ds2",     cfg)

    n_S = int((ds1_all["y"] == 1).sum())
    n_N = int((ds1_all["y"] == 0).sum())
    n_V = int((ds1_all["y"] == 2).sum())
    if verbose:
        print(f"[exp5] Full DS1 beats — N={n_N}, S={n_S}, V={n_V}")
        print(f"[exp5] S fraction: {100*n_S/(n_N+n_S+n_V):.2f}%")

    # Random beat-level split for early stopping
    X_ecg_tr, X_rr_tr, y_tr, X_ecg_val, X_rr_val, y_val = _split_train_val(
        ds1_all["X_ecg"], ds1_all["X_rr"], ds1_all["y"], val_frac=val_frac
    )

    if verbose:
        n_S_tr  = int((y_tr  == 1).sum())
        n_S_val = int((y_val == 1).sum())
        print(f"[exp5] Train: {len(y_tr)} beats (S={n_S_tr}), "
              f"Val: {len(y_val)} beats (S={n_S_val})")

    kernel_mf = load_kernel_mf(cfg)

    from src.train import train_multi_seed
    from src.metrics import multiclass_metrics

    results = train_multi_seed(
        cfg=cfg,
        X_ecg_tr=X_ecg_tr,         X_rr_tr=X_rr_tr,         y_tr=y_tr,
        X_ecg_val=X_ecg_val,        X_rr_val=X_rr_val,        y_val=y_val,
        X_ecg_test=test["X_ecg"],   X_rr_test=test["X_rr"],   y_test=test["y"],
        kernel_mf=kernel_mf,
        n_classes=3,
        seeds=seeds,
        cache_dir=None,
        run_name="exp5_fullDS1",
        deterministic=True,
    )

    test_metrics = [r["metrics_test"] for r in results]
    mean_m = {
        k: float(np.mean([m[k] for m in test_metrics if k in m]))
        for k in test_metrics[0]
        if isinstance(test_metrics[0][k], float)
    }

    if verbose:
        print_header()
        print_row(f"Exp5: full DS1 ({len(seeds)} seeds)", mean_m)
        for s, m in zip(seeds, test_metrics):
            print_row(f"  seed {s}", m)
        print_footer()

    return {"seed_results": results, "mean_metrics": mean_m}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp5: train on full DS1 (DS1_TRAIN + DS1_VAL)")
    ap.add_argument("--seeds",    nargs="+", type=int,   default=[0, 1, 2])
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="Fraction of combined DS1 held out for early stopping (default 0.15)")
    args = ap.parse_args()
    run(seeds=args.seeds, val_frac=args.val_frac, verbose=True)
