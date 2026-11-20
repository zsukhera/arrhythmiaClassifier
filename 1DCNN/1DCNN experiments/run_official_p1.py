"""
Official P1 retraining — Steps 1 + 2

Step 1: Retrain P1 with full DS1 (DS1_TRAIN + DS1_VAL), class_weight=balanced, 10 seeds.
        Models saved to HasanML/paper3/results/cache/P1_fullDS1_seed{n}.keras

Step 2: Threshold calibration using the val probabilities from each seed's training run.
        Sweeps tau_S in [0.05..0.50] on the validation hold-out, picks best F1_S.
        Reports metrics for: (a) argmax, (b) calibrated threshold.

This is the definitive P1 configuration that achieves Se_S ≈ 83–88% on DS2.

Usage:
    python run_official_p1.py                       # 10 seeds (default)
    python run_official_p1.py --seeds 0 1 2         # quick test
    python run_official_p1.py --val-frac 0.15       # fraction of full DS1 for early stopping
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

EXPS   = Path(__file__).resolve().parent
PAPER3 = EXPS.parents[1] / "HasanML" / "paper3"
sys.path.insert(0, str(PAPER3))

from utils import load_config, load_data, load_kernel_mf, threshold_predict, \
                  print_header, print_row, print_footer


def _split_train_val(X_ecg, X_rr, y, val_frac=0.15, seed=42):
    rng   = np.random.default_rng(seed)
    n     = len(y)
    idx   = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    vi, ti = idx[:n_val], idx[n_val:]
    return (X_ecg[ti], X_rr[ti], y[ti],
            X_ecg[vi], X_rr[vi], y[vi])


def _calibrate(avg_val_proba, y_val, tau_grid=None):
    from src.metrics import multiclass_metrics
    if tau_grid is None:
        tau_grid = np.arange(0.05, 0.55, 0.01)
    best_tau, best_f1 = 0.333, 0.0
    for tau in tau_grid:
        y_pred = threshold_predict(avg_val_proba, tau)
        m      = multiclass_metrics(y_val, y_pred, avg_val_proba, k=3)
        f1 = float(m.get("f1_S", 0.0))
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)
    return best_tau, best_f1


def run(seeds=None, val_frac=0.15, verbose=True):
    if seeds is None:
        seeds = list(range(10))

    cfg = load_config()
    cfg["training"]["class_weight"] = "balanced"

    # Load data
    ds1_all = load_data("ds1_all", cfg)
    test    = load_data("ds2",     cfg)

    n_S = int((ds1_all["y"] == 1).sum())
    n_N = int((ds1_all["y"] == 0).sum())
    n_V = int((ds1_all["y"] == 2).sum())
    total = n_N + n_S + n_V
    if verbose:
        print(f"\n[p1-official] Full DS1: N={n_N}, S={n_S}, V={n_V} "
              f"(S fraction={100*n_S/total:.2f}%)")

    X_ecg_tr, X_rr_tr, y_tr, X_ecg_val, X_rr_val, y_val = _split_train_val(
        ds1_all["X_ecg"], ds1_all["X_rr"], ds1_all["y"], val_frac=val_frac
    )
    if verbose:
        print(f"[p1-official] Train: {len(y_tr)} beats (S={int((y_tr==1).sum())}), "
              f"Val: {len(y_val)} beats (S={int((y_val==1).sum())})")

    kernel_mf = load_kernel_mf(cfg)

    # Save models to paper3/results/cache with run_name P1_fullDS1
    cache_dir = PAPER3 / "results" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    from src.train import train_multi_seed
    from src.metrics import multiclass_metrics, summarise_seeds

    t0 = time.perf_counter()
    results = train_multi_seed(
        cfg=cfg,
        X_ecg_tr=X_ecg_tr,         X_rr_tr=X_rr_tr,         y_tr=y_tr,
        X_ecg_val=X_ecg_val,        X_rr_val=X_rr_val,        y_val=y_val,
        X_ecg_test=test["X_ecg"],   X_rr_test=test["X_rr"],   y_test=test["y"],
        kernel_mf=kernel_mf,
        n_classes=3,
        seeds=seeds,
        cache_dir=cache_dir,
        run_name="P1_fullDS1",
        deterministic=True,
    )
    total_elapsed = time.perf_counter() - t0

    # ------------------------------------------------------------------ Step 1: argmax
    test_metrics = [r["metrics_test"] for r in results]
    mean_m = {
        k: float(np.mean([m[k] for m in test_metrics if k in m]))
        for k in test_metrics[0]
        if isinstance(test_metrics[0][k], float)
    }

    if verbose:
        print(f"\n[p1-official] Training complete: {len(seeds)} seeds in {total_elapsed:.0f}s")
        print_header()
        for s, m in zip(seeds, test_metrics):
            print_row(f"  fullDS1 seed {s}", m)
        print_row(f"P1 fullDS1 mean ({len(seeds)} seeds)", mean_m)

    # ------------------------------------------------------------------ Step 2: threshold
    # Average val probabilities across seeds
    val_probas  = [r["proba_val"]  for r in results]
    test_probas = [r["proba_test"] for r in results]
    avg_val_proba  = np.mean(val_probas,  axis=0)
    avg_test_proba = np.mean(test_probas, axis=0)

    best_tau, val_f1 = _calibrate(avg_val_proba, y_val)
    if verbose:
        print(f"\n[p1-official] Threshold calibration: best tau_S={best_tau:.3f} "
              f"(val F1_S={100*val_f1:.1f}%)")

    y_pred_cal = threshold_predict(avg_test_proba, best_tau)
    cal_m      = multiclass_metrics(test["y"], y_pred_cal, avg_test_proba, k=3)

    if verbose:
        print_row(f"P1 fullDS1 + tau={best_tau:.3f} (ensemble)", cal_m)
        print_footer()

    # Print per-seed summary
    mets = summarise_seeds(test_metrics)
    print(f"\n[p1-official] Summary: "
          f"Se_S={100*mets.get('se_S_mean',0):.2f}±{100*mets.get('se_S_std',0):.2f}%  "
          f"Se_V={100*mets.get('se_V_mean',0):.2f}±{100*mets.get('se_V_std',0):.2f}%  "
          f"Acc={100*mets.get('accuracy_mean',0):.2f}±{100*mets.get('accuracy_std',0):.2f}%")
    print(f"[p1-official] Calibrated: "
          f"Se_S={100*cal_m.get('se_S',0):.2f}%  "
          f"Se_V={100*cal_m.get('se_V',0):.2f}%  "
          f"Acc={100*cal_m.get('accuracy',0):.2f}%")

    return {
        "results":           results,
        "mean_argmax":       mean_m,
        "best_tau":          best_tau,
        "calibrated":        cal_m,
        "summarise_seeds":   mets,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds",    nargs="+", type=int,   default=list(range(10)))
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()
    run(seeds=args.seeds, val_frac=args.val_frac, verbose=True)
