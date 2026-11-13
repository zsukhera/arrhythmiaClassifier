"""
Experiment 1: Threshold Calibration
------------------------------------
Problem: Se_S is low because the model is calibrated for argmax (tau=0.33 implicit).
         Only 149 S beats in DS1-train → class S probabilities are systematically low.

Fix: Lower the decision threshold for class S.
     Sweep tau_S in [0.05, 0.50] on DS1-val, pick the threshold that maximises
     F1_S on DS1-val, then lock it in and evaluate on DS2 (never re-using DS2 to pick tau).

No retraining needed — works on existing P1_seed*.keras models in paper3/results/cache/.

Usage:
    cd /home/sirajmunir/fyp/HasanML/paper3
    python -m 1DCNN.1DCNN\ experiments.exp1_threshold

    OR run directly:
    cd /home/sirajmunir/fyp/1DCNN/1DCNN\ experiments
    python exp1_threshold.py
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HasanML" / "paper3"))

from utils import (
    load_config, load_data, available_p1_seeds, load_p1_model,
    threshold_predict, print_header, print_row, print_footer,
)


def _get_proba(model, X_ecg: np.ndarray, X_rr: np.ndarray,
               batch_size: int = 1024) -> np.ndarray:
    return model.predict([X_ecg, X_rr], batch_size=batch_size, verbose=0)


def _metrics(y_true, y_pred, proba, k=3):
    from src.metrics import multiclass_metrics
    return multiclass_metrics(y_true, y_pred, proba, k=k)


def _f1_S(m: dict) -> float:
    return float(m.get("f1_S", 0.0))


def run(seeds: list[int] | None = None, tau_grid: np.ndarray | None = None,
        verbose: bool = True) -> dict:
    """
    Returns dict with keys:
      best_tau, baseline_metrics, calibrated_metrics
    """
    cfg   = load_config()
    val   = load_data("ds1_val",  cfg)
    test  = load_data("ds2",      cfg)

    if seeds is None:
        seeds = available_p1_seeds()
    if not seeds:
        raise RuntimeError(
            "No P1 models found in paper3/results/cache/.\n"
            "Run:  python -m src.run_experiment --p1 --class-weight balanced"
        )
    if verbose:
        print(f"[exp1] Loaded {len(seeds)} P1 models: seeds={seeds}")

    # Collect probabilities from each seed
    val_probas  = []
    test_probas = []
    for s in seeds:
        m = load_p1_model(s)
        val_probas.append(_get_proba(m, val["X_ecg"], val["X_rr"]))
        test_probas.append(_get_proba(m, test["X_ecg"], test["X_rr"]))

    # Average probabilities across seeds (soft ensemble for threshold selection)
    avg_val_proba  = np.mean(val_probas,  axis=0)   # (N_val,  3)
    avg_test_proba = np.mean(test_probas, axis=0)   # (N_test, 3)

    # Baseline (argmax, tau=0.333)
    y_pred_base = np.argmax(avg_test_proba, axis=1)
    baseline_m  = _metrics(test["y"], y_pred_base, avg_test_proba)

    # Threshold sweep on DS1-val
    if tau_grid is None:
        tau_grid = np.arange(0.05, 0.55, 0.01)

    best_tau, best_f1 = 0.333, 0.0
    for tau in tau_grid:
        y_pred_v = threshold_predict(avg_val_proba, tau)
        m_v      = _metrics(val["y"], y_pred_v, avg_val_proba)
        f1 = _f1_S(m_v)
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)

    if verbose:
        print(f"[exp1] Best tau_S={best_tau:.3f}  (val F1_S={100*best_f1:.1f}%)")

    # Evaluate on DS2 with best tau
    y_pred_cal = threshold_predict(avg_test_proba, best_tau)
    calibrated_m = _metrics(test["y"], y_pred_cal, avg_test_proba)

    if verbose:
        print_header()
        print_row("P1 baseline (argmax)",          baseline_m)
        print_row(f"Exp1: threshold tau={best_tau:.3f}", calibrated_m)
        print_footer()

    return {
        "best_tau":           best_tau,
        "baseline_metrics":   baseline_m,
        "calibrated_metrics": calibrated_m,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp1: threshold calibration for class S")
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Which P1 seeds to use (default: all available)")
    args = ap.parse_args()
    run(seeds=args.seeds, verbose=True)
