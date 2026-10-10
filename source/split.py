# %% [markdown]

# # MIT-BIH — Step 6: Train/Test Split
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Two splits (clarityBase.txt §9.1):
#
# 1. RANDOM STRATIFIED (80/20 by beat label, seed=42)
#    Replicates the paper's leaderboard numbers.  Patient identity can bleed
#    across train/test, so scores are optimistic.
#
# 2. INTER-PATIENT DS1/DS2 (de Chazal 2004)
#    DS1 (22 records) → training; DS2 (22 records) → test.
#    No patient appears in both sets → honest, publishable evaluation.
#
# Input:  datasets/processed/mit_bih_features.npz
# Output: datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz
#
# Each output stores:
#   indices_train, indices_test   — positions into the features file
#   X_train, X_test               — feature matrices (float32)
#   y_train, y_test               — label arrays
#   record_ids_train, record_ids_test

# %%

import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

# %%

_HERE      = Path(__file__).resolve().parent
PROC_DIR   = _HERE.parent / "datasets" / "processed"
INPUT_PATH = PROC_DIR / "mit_bih_features.npz"

RANDOM_SEED = 42
TEST_SIZE   = 0.20

# ── de Chazal (2004) inter-patient DS1/DS2 ────────────────────────────────────
# Paced records 102/104/107/217 are already absent from the feature file.
DS1_RECORDS = {
    "101", "106", "108", "109", "112", "114", "115", "116",
    "118", "119", "122", "124", "201", "203", "205", "207",
    "208", "209", "215", "220", "223", "230",
}
DS2_RECORDS = {
    "100", "103", "105", "111", "113", "117", "121", "123",
    "200", "202", "210", "212", "213", "214", "219", "221",
    "222", "228", "231", "232", "233", "234",
}

LABEL_ORDER = ["N", "L", "R", "V", "A"]
LABEL_NAMES = {"N": "Normal", "L": "LBBB", "R": "RBBB", "V": "PVC", "A": "APC"}

# %%

def _print_stats(
    name: str,
    y_tr: np.ndarray, rec_tr: np.ndarray,
    y_te: np.ndarray, rec_te: np.ndarray,
) -> None:
    tc, vc = Counter(y_tr), Counter(y_te)
    print(f"\n{'─'*60}")
    print(f"Split : {name}")
    print(f"  Train : {len(y_tr):>7,} beats  |  {len(np.unique(rec_tr))} records")
    print(f"  Test  : {len(y_te):>7,} beats  |  {len(np.unique(rec_te))} records")
    print(f"\n  {'Cls':<5}{'Name':<10}{'Train':>8}{'Test':>8}")
    print(f"  {'─'*34}")
    for cls in LABEL_ORDER:
        if cls in tc or cls in vc:
            print(f"  {cls:<5}{LABEL_NAMES.get(cls, ''):<10}"
                  f"{tc.get(cls, 0):>8,}{vc.get(cls, 0):>8,}")
    print(f"  {'─'*34}")


# %%

def random_split(
    indices: np.ndarray,
    y: np.ndarray,
) -> tuple:
    """Stratified 80/20 split by beat label."""
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        stratify=y[indices],
        random_state=RANDOM_SEED,
    )
    return train_idx, test_idx


def ds1ds2_split(
    indices: np.ndarray,
    record_ids: np.ndarray,
) -> tuple:
    """
    Inter-patient split following de Chazal (2004).
    DS1 records → training; DS2 records → test.
    """
    recs       = record_ids[indices]
    train_mask = np.array([r in DS1_RECORDS for r in recs])
    test_mask  = np.array([r in DS2_RECORDS for r in recs])

    unmatched  = indices[~train_mask & ~test_mask]
    if len(unmatched):
        bad = np.unique(record_ids[unmatched])
        print(f"  [warn] {len(unmatched)} beats from unrecognised records: {bad}")

    return indices[train_mask], indices[test_mask]


# %%

if __name__ == "__main__":
    print(f"Loading {INPUT_PATH.name} ...")
    data       = np.load(INPUT_PATH, allow_pickle=True)
    X          = data["X"]           # (N, 40)
    y          = data["y"]           # (N,)
    record_ids = data["record_ids"]  # (N,)
    r_samples  = data["r_samples"]   # (N,)

    N       = len(y)
    indices = np.arange(N)
    print(f"  {N:,} beats  |  feature shape: {X.shape}\n")

    # ── 1. Random stratified split ────────────────────────────────────────────
    r_tr, r_te = random_split(indices, y)
    _print_stats(
        "Random stratified 80/20",
        y[r_tr], record_ids[r_tr],
        y[r_te], record_ids[r_te],
    )
    np.savez_compressed(
        PROC_DIR / "split_random.npz",
        indices_train=r_tr,   indices_test=r_te,
        X_train=X[r_tr],      X_test=X[r_te],
        y_train=y[r_tr],      y_test=y[r_te],
        record_ids_train=record_ids[r_tr],
        record_ids_test=record_ids[r_te],
    )
    print(f"  Saved → {PROC_DIR / 'split_random.npz'}")

    # ── 2. Inter-patient DS1/DS2 split ───────────────────────────────────────
    d_tr, d_te = ds1ds2_split(indices, record_ids)
    _print_stats(
        "Inter-patient DS1/DS2  (de Chazal 2004)",
        y[d_tr], record_ids[d_tr],
        y[d_te], record_ids[d_te],
    )
    np.savez_compressed(
        PROC_DIR / "split_ds1ds2.npz",
        indices_train=d_tr,   indices_test=d_te,
        X_train=X[d_tr],      X_test=X[d_te],
        y_train=y[d_tr],      y_test=y[d_te],
        record_ids_train=record_ids[d_tr],
        record_ids_test=record_ids[d_te],
    )
    print(f"  Saved → {PROC_DIR / 'split_ds1ds2.npz'}")

    print("\nDone.")
