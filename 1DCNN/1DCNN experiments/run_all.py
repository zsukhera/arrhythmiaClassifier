"""
Master runner — executes all 5 experiments sequentially and prints a comparison table.

Exp1 and Exp2 require existing P1 models (run paper3 P1 training first).
Exp3-5 retrain from scratch (3 seeds each by default).

Usage:
    # Run everything (3 seeds for exp3-5):
    python run_all.py

    # Faster smoke-test (1 seed per retrain experiment):
    python run_all.py --seeds 0

    # Skip experiments that require existing models:
    python run_all.py --skip-no-model

    # Only run specific experiments:
    python run_all.py --only 1 5
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HasanML" / "paper3"))

import numpy as np
from utils import (
    load_config, load_data, available_p1_seeds,
    print_header, print_row, print_footer,
)


def _baseline_metrics(seeds, cfg, test) -> dict:
    """Compute mean per-seed baseline from existing P1 models."""
    from utils import load_p1_model
    from src.metrics import multiclass_metrics
    all_m = []
    for s in seeds:
        try:
            m = load_p1_model(s)
            proba = m.predict([test["X_ecg"], test["X_rr"]], batch_size=1024, verbose=0)
            y_pred = np.argmax(proba, axis=1)
            all_m.append(multiclass_metrics(test["y"], y_pred, proba, k=3))
        except FileNotFoundError:
            pass
    if not all_m:
        return {}
    return {k: float(np.mean([m[k] for m in all_m if k in m]))
            for k in all_m[0] if isinstance(all_m[0][k], float)}


def main():
    ap = argparse.ArgumentParser(description="Run all sensitivity experiments")
    ap.add_argument("--seeds",          nargs="+", type=int, default=[0, 1, 2],
                    help="Seeds for exp3-5 (default: 0 1 2)")
    ap.add_argument("--skip-no-model",  action="store_true",
                    help="Skip exp1/exp2 if no P1 models exist")
    ap.add_argument("--only",           nargs="+", type=int, default=None,
                    help="Run only listed experiment numbers (e.g. --only 1 5)")
    args = ap.parse_args()

    want = set(args.only) if args.only else {1, 2, 3, 4, 5}

    cfg  = load_config()
    test = load_data("ds2", cfg)
    p1_seeds = available_p1_seeds()

    rows = []  # (label, metrics_dict)

    # Baseline from existing models (if any)
    if p1_seeds:
        base_m = _baseline_metrics(p1_seeds, cfg, test)
        if base_m:
            rows.append((f"Baseline P1 (mean {len(p1_seeds)} seeds)", base_m))

    # Exp 1
    if 1 in want:
        if not p1_seeds and args.skip_no_model:
            print("[run_all] Skipping exp1 (no P1 models found)")
        else:
            print("\n=== Experiment 1: Threshold Calibration ===")
            import exp1_threshold
            r1 = exp1_threshold.run(seeds=p1_seeds or None, verbose=True)
            rows.append(("Exp1: threshold calibration", r1["calibrated_metrics"]))

    # Exp 2
    if 2 in want:
        if not p1_seeds and args.skip_no_model:
            print("[run_all] Skipping exp2 (no P1 models found)")
        else:
            print("\n=== Experiment 2: Model Ensemble ===")
            import exp2_ensemble
            r2 = exp2_ensemble.run(seeds=p1_seeds or None, verbose=True)
            rows.append(("Exp2: soft ensemble + threshold", r2["ensemble_calibrated_metrics"]))

    # Exp 3
    if 3 in want:
        print(f"\n=== Experiment 3: Data Augmentation (seeds={args.seeds}) ===")
        import exp3_augmentation
        r3 = exp3_augmentation.run(seeds=args.seeds, verbose=True)
        rows.append((f"Exp3: augmentation ({len(args.seeds)} seeds)", r3["mean_metrics"]))

    # Exp 4
    if 4 in want:
        print(f"\n=== Experiment 4: Focal Loss (seeds={args.seeds}) ===")
        import exp4_focal_loss
        r4 = exp4_focal_loss.run(seeds=args.seeds, gammas=[2.0], verbose=True)
        best_gamma_m = r4[2.0]["mean"]
        rows.append((f"Exp4: focal loss gamma=2 ({len(args.seeds)} seeds)", best_gamma_m))

    # Exp 5
    if 5 in want:
        print(f"\n=== Experiment 5: Full DS1 Training (seeds={args.seeds}) ===")
        import exp5_full_ds1
        r5 = exp5_full_ds1.run(seeds=args.seeds, verbose=True)
        rows.append((f"Exp5: full DS1 ({len(args.seeds)} seeds)", r5["mean_metrics"]))

    # Final comparison table
    if rows:
        print("\n" + "=" * 70)
        print("FINAL COMPARISON — DS2 test set")
        print("=" * 70)
        print_header()
        for label, m in rows:
            print_row(label, m)
        print_footer()


if __name__ == "__main__":
    main()
