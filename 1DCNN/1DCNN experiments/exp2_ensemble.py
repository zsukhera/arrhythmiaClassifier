"""
Experiment 2: Model Ensemble
-----------------------------
Problem: Each seed converges to a slightly different local minimum.
         Averaging their probability outputs (soft voting) reduces variance
         and usually improves minority-class sensitivity.

Method: Average softmax probabilities across all 10 available seeds, then
        apply the threshold calibration from Exp1 to further boost Se_S.

No retraining — requires the same P1_seed*.keras models as Exp1.

Usage:
    python exp2_ensemble.py
    python exp2_ensemble.py --seeds 0 1 2 3 4   # use only 5 seeds
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


def _get_proba(model, X_ecg, X_rr, batch_size=1024):
    return model.predict([X_ecg, X_rr], batch_size=batch_size, verbose=0)


def _metrics(y_true, y_pred, proba, k=3):
    from src.metrics import multiclass_metrics
    return multiclass_metrics(y_true, y_pred, proba, k=k)


def _best_tau(avg_val_proba, y_val, tau_grid=None):
    if tau_grid is None:
        tau_grid = np.arange(0.05, 0.55, 0.01)
    best_tau, best_f1 = 0.333, 0.0
    for tau in tau_grid:
        y_pred = threshold_predict(avg_val_proba, tau)
        m      = _metrics(y_val, y_pred, avg_val_proba)
        f1 = float(m.get("f1_S", 0.0))
        if f1 > best_f1:
            best_f1  = f1
            best_tau = float(tau)
    return best_tau, best_f1


def run(seeds: list[int] | None = None, verbose: bool = True) -> dict:
    cfg  = load_config()
    val  = load_data("ds1_val", cfg)
    test = load_data("ds2",     cfg)

    if seeds is None:
        seeds = available_p1_seeds()
    if not seeds:
        raise RuntimeError(
            "No P1 models found. Run:\n"
            "  python -m src.run_experiment --p1 --class-weight balanced"
        )
    if verbose:
        print(f"[exp2] Ensembling {len(seeds)} seeds: {seeds}")

    val_probas, test_probas = [], []
    for s in seeds:
        model = load_p1_model(s)
        val_probas.append(_get_proba(model, val["X_ecg"], val["X_rr"]))
        test_probas.append(_get_proba(model, test["X_ecg"], test["X_rr"]))

    # Per-seed baselines (for comparison)
    seed_metrics = []
    for proba_t in test_probas:
        y_pred = np.argmax(proba_t, axis=1)
        seed_metrics.append(_metrics(test["y"], y_pred, proba_t))

    avg_se_S = np.mean([m.get("se_S", 0) for m in seed_metrics])

    # Soft-vote ensemble
    avg_val_proba  = np.mean(val_probas,  axis=0)
    avg_test_proba = np.mean(test_probas, axis=0)

    # Ensemble + argmax
    y_pred_ens = np.argmax(avg_test_proba, axis=1)
    ens_m      = _metrics(test["y"], y_pred_ens, avg_test_proba)

    # Ensemble + threshold calibration
    tau, val_f1 = _best_tau(avg_val_proba, val["y"])
    if verbose:
        print(f"[exp2] Best tau_S={tau:.3f}  (val F1_S={100*val_f1:.1f}%)")

    y_pred_cal = threshold_predict(avg_test_proba, tau)
    cal_m      = _metrics(test["y"], y_pred_cal, avg_test_proba)

    if verbose:
        print_header()
        print_row(f"Per-seed mean  ({len(seeds)} seeds)",
                  {"accuracy": np.mean([m.get("accuracy",0) for m in seed_metrics]),
                   "se_N": np.mean([m.get("se_N",0) for m in seed_metrics]),
                   "se_S": avg_se_S,
                   "se_V": np.mean([m.get("se_V",0) for m in seed_metrics]),
                   "macro_f1": np.mean([m.get("macro_f1",0) for m in seed_metrics])})
        print_row(f"Exp2a: soft ensemble ({len(seeds)} seeds)", ens_m)
        print_row(f"Exp2b: ensemble + tau={tau:.3f}",           cal_m)
        print_footer()

    return {
        "seed_metrics":   seed_metrics,
        "ensemble_argmax_metrics": ens_m,
        "best_tau":       tau,
        "ensemble_calibrated_metrics": cal_m,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp2: model ensemble + threshold calibration")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    args = ap.parse_args()
    run(seeds=args.seeds, verbose=True)
