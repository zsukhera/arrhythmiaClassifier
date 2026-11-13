"""
Experiment 3: Data Augmentation for S-class Beats
---------------------------------------------------
Problem: DS1-train has only 149 S beats (0.59%) vs 23,932 N beats.
         Even balanced class weights can't conjure more signal from so few examples.

Fix: Synthetically expand the S-class with 6 augmentation operations:
     1. Gaussian noise (sigma = 0.05 × segment std)
     2. Amplitude scale ×0.85
     3. Amplitude scale ×1.15
     4. Time shift −3 samples (np.roll)
     5. Time shift +3 samples (np.roll)
     6. Noise + scale combination

Each of the 149 S beats produces 6 extra copies → ~1,043 S beats total,
bringing S-class fraction from 0.59% to ~3.9%.
Still using class_weight=balanced to cover remaining imbalance.

Retrains from scratch (3 seeds by default; pass --seeds for more).

Usage:
    cd /home/sirajmunir/fyp/HasanML/paper3
    python -m 1DCNN.1DCNN\ experiments.exp3_augmentation

    OR:
    python exp3_augmentation.py --seeds 0 1 2
    python exp3_augmentation.py --seeds 0 1 2 3 4   # more seeds on GPU
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HasanML" / "paper3"))

from utils import load_config, load_data, load_kernel_mf, print_header, print_row, print_footer


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def _augment_S_beats(X_ecg: np.ndarray, X_rr: np.ndarray,
                     y: np.ndarray, s_class: int = 1,
                     rng: np.random.Generator | None = None) -> tuple:
    """
    Returns (X_ecg_aug, X_rr_aug, y_aug) with augmented S beats appended.
    X_ecg shape: (N, NK, C).  Augmentation applied per segment.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    s_idx = np.where(y == s_class)[0]
    segs  = X_ecg[s_idx]        # (n_S, NK, C)
    rrs   = X_rr[s_idx]         # (n_S, 4)
    n_S, NK, C = segs.shape

    new_segs, new_rrs = [], []

    for seg, rr in zip(segs, rrs):
        std = seg.std() + 1e-8

        # 1. Gaussian noise
        noise = rng.normal(0, 0.05 * std, seg.shape)
        new_segs.append(seg + noise)
        new_rrs.append(rr)

        # 2. Amplitude scale ×0.85
        new_segs.append(seg * 0.85)
        new_rrs.append(rr)

        # 3. Amplitude scale ×1.15
        new_segs.append(seg * 1.15)
        new_rrs.append(rr)

        # 4. Time shift −3
        new_segs.append(np.roll(seg, -3, axis=0))
        new_rrs.append(rr)

        # 5. Time shift +3
        new_segs.append(np.roll(seg, +3, axis=0))
        new_rrs.append(rr)

        # 6. Noise + scale
        scaled = seg * rng.uniform(0.9, 1.1)
        noise2 = rng.normal(0, 0.03 * std, scaled.shape)
        new_segs.append(scaled + noise2)
        new_rrs.append(rr)

    new_segs = np.stack(new_segs, axis=0).astype(np.float32)
    new_rrs  = np.stack(new_rrs,  axis=0).astype(np.float32)
    new_y    = np.full(len(new_segs), s_class, dtype=np.int32)

    X_ecg_aug = np.concatenate([X_ecg, new_segs], axis=0)
    X_rr_aug  = np.concatenate([X_rr,  new_rrs],  axis=0)
    y_aug     = np.concatenate([y,      new_y],    axis=0)

    return X_ecg_aug, X_rr_aug, y_aug


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seeds: list[int] | None = None, verbose: bool = True) -> dict:
    if seeds is None:
        seeds = [0, 1, 2]

    cfg = load_config()
    # Force class_weight=balanced (augmentation works best with it)
    cfg["training"]["class_weight"] = "balanced"

    train = load_data("ds1_train", cfg)
    val   = load_data("ds1_val",   cfg)
    test  = load_data("ds2",       cfg)

    # Count S beats before and after
    n_S_before = int((train["y"] == 1).sum())
    X_ecg_aug, X_rr_aug, y_aug = _augment_S_beats(
        train["X_ecg"], train["X_rr"], train["y"]
    )
    n_S_after = int((y_aug == 1).sum())
    if verbose:
        print(f"[exp3] S beats: {n_S_before} → {n_S_after} "
              f"(×{n_S_after/n_S_before:.1f} augmentation)")

    kernel_mf = load_kernel_mf(cfg)

    # Import training infrastructure from paper3
    from src.train import train_multi_seed
    from src.metrics import multiclass_metrics

    results = train_multi_seed(
        cfg=cfg,
        X_ecg_tr=X_ecg_aug,     X_rr_tr=X_rr_aug,     y_tr=y_aug,
        X_ecg_val=val["X_ecg"], X_rr_val=val["X_rr"], y_val=val["y"],
        X_ecg_test=test["X_ecg"], X_rr_test=test["X_rr"], y_test=test["y"],
        kernel_mf=kernel_mf,
        n_classes=3,
        seeds=seeds,
        cache_dir=None,          # don't pollute P1 cache
        run_name="exp3_aug",
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
        print_row(f"Exp3: augmentation ({len(seeds)} seeds)", mean_m)
        for i, (s, m) in enumerate(zip(seeds, test_metrics)):
            print_row(f"  seed {s}", m)
        print_footer()

    return {"seed_results": results, "mean_metrics": mean_m}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp3: S-beat augmentation + retrain")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = ap.parse_args()
    run(seeds=args.seeds, verbose=True)
