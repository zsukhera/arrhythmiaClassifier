# %% [markdown]

# # MIT-BIH — Step 9: Validate Paper Claims
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# The paper (clarityBase.txt §9.1) never specifies its train/test split.
# This script tests four scenarios using ONLY the paper's own hyperparameters
# and SMOTE — no Optuna, no saved models — to isolate the effect of split choice.
#
# Scenarios
# ---------
#   S1  REPLICATE   Train on random-split train  → test on random-split test
#                   Expected: matches paper's 0.989 accuracy / 0.928 binary-F1
#
#   S2  EXPOSE      Same model as S1             → test on DS2 (cross-patient)
#                   Paper never reports this.  Expected: much worse for L/R.
#
#   S3  HONEST      Train on DS1                 → test on DS2
#                   True inter-patient evaluation.
#
#   S4  ABLATION    A1→A4 on DS1→DS2 split
#                   Tests whether the ablation table (§8.2) holds cross-patient.
#
# Input:  datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz
# Output: printed tables (no files written)

# %%

import warnings
from pathlib import Path

import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# %%

_HERE      = Path(__file__).resolve().parent
PROC_DIR   = _HERE.parent / "datasets" / "processed"
RESULT_DIR = _HERE.parent / "results"

SEED        = 42
LABEL_ORDER = ["N", "L", "R", "V", "A"]
LABEL_NAMES = {"N": "Normal", "L": "LBBB", "R": "RBBB", "V": "PVC", "A": "APC"}

PAPER_PARAMS = {
    "n_estimators":     1450,
    "max_depth":        11,
    "learning_rate":    0.05,
    "num_leaves":       38,
    "subsample":        0.85,
    "colsample_bytree": 0.75,
}

ABLATION_CONFIGS = [
    ("A1 Statistical (6)",      slice(0, 6)),
    ("A2 +Fiducial (19)",       slice(0, 19)),
    ("A3 +Wavelet (31)",        slice(0, 31)),
    ("A4 +HRV/Non-linear (40)", slice(0, 40)),
]

# %%

def _train(X_tr, y_tr) -> LGBMClassifier:
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    clf = LGBMClassifier(
        **PAPER_PARAMS, subsample_freq=1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(X_res, y_res)
    return clf


def _to_binary(y: np.ndarray) -> np.ndarray:
    return np.where(y == "N", 0, 1).astype(int)


def _binary_auc(clf, X, y_bin) -> float:
    proba     = clf.predict_proba(X)
    classes   = list(clf.classes_)
    anom_cols = [i for i, c in enumerate(classes) if c != "N"]
    scores    = proba[:, anom_cols].sum(axis=1)
    try:
        return float(roc_auc_score(y_bin, scores))
    except ValueError:
        return float("nan")


def _5class_auc(clf, X, y) -> float:
    try:
        return float(roc_auc_score(
            y, clf.predict_proba(X),
            multi_class="ovr", average="macro", labels=clf.classes_,
        ))
    except Exception:
        return float("nan")


def _metrics(clf, X, y) -> dict:
    y_pred     = clf.predict(X)
    y_bin      = _to_binary(y)
    y_bin_pred = _to_binary(np.asarray(y_pred))
    return {
        "acc":      float(accuracy_score(y, y_pred)),
        "macro_f1": float(f1_score(y, y_pred, average="macro", zero_division=0)),
        "auc_5":    _5class_auc(clf, X, y),
        "bin_p":    float(precision_score(y_bin, y_bin_pred, zero_division=0)),
        "bin_r":    float(recall_score(y_bin, y_bin_pred, zero_division=0)),
        "bin_f1":   float(f1_score(y_bin, y_bin_pred, zero_division=0)),
        "bin_auc":  _binary_auc(clf, X, y_bin),
        "y_pred":   y_pred,
    }


def _print_per_class(clf, X, y, indent="  "):
    y_pred  = clf.predict(X)
    labels  = [c for c in LABEL_ORDER if c in set(y)]
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, y_pred, labels=labels, zero_division=0,
    )
    print(f"{indent}{'Cls':<5}{'Name':<10}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Support':>10}")
    print(f"{indent}{'─'*49}")
    for i, cls in enumerate(labels):
        print(f"{indent}{cls:<5}{LABEL_NAMES.get(cls,''):<10}"
              f"{prec[i]:>8.3f}{rec[i]:>8.3f}{f1[i]:>8.3f}{sup[i]:>10,}")
    print(f"{indent}{'─'*49}")


def _print_summary(tag: str, m: dict) -> None:
    print(f"\n  5-class  acc={m['acc']:.4f}  macro-F1={m['macro_f1']:.4f}  AUC={m['auc_5']:.4f}")
    print(f"  Binary   P={m['bin_p']:.4f}  R={m['bin_r']:.4f}  F1={m['bin_f1']:.4f}  AUC={m['bin_auc']:.4f}")


def _save_cm(clf, X, y, tag: str) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred  = clf.predict(X)
    labels  = [c for c in LABEL_ORDER if c in set(y)]
    cm      = confusion_matrix(y, y_pred, labels=labels)
    disp    = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(tag)
    fig.tight_layout()
    out = RESULT_DIR / f"cm_{tag.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {out.name}")

# %%

def load_splits() -> tuple:
    rand   = np.load(PROC_DIR / "split_random.npz",  allow_pickle=True)
    ds1ds2 = np.load(PROC_DIR / "split_ds1ds2.npz",  allow_pickle=True)
    return rand, ds1ds2


def run_s1_s2(rand, ds1ds2) -> LGBMClassifier:
    """
    S1: Train on random-split train, test on random-split test  (replicates paper).
    S2: Same model tested on DS2                                (exposes cross-patient gap).
    Returns the trained model for reuse.
    """
    print(f"\n{'═'*62}")
    print("S1  REPLICATE  —  random-split train → random-split test")
    print("    Expected to match paper: acc≈0.989  binary-F1≈0.928")
    print(f"{'═'*62}")

    print(f"\n  Training on {len(rand['y_train']):,} beats (random split) ...")
    clf = _train(rand["X_train"], rand["y_train"])

    m = _metrics(clf, rand["X_test"], rand["y_test"])
    _print_summary("S1", m)
    _print_per_class(clf, rand["X_test"], rand["y_test"])
    _save_cm(clf, rand["X_test"], rand["y_test"], "S1_random_test")

    print(f"\n{'═'*62}")
    print("S2  EXPOSE  —  same model → DS2 test  (paper never reports this)")
    print(f"{'═'*62}")

    m2 = _metrics(clf, ds1ds2["X_test"], ds1ds2["y_test"])
    _print_summary("S2", m2)
    _print_per_class(clf, ds1ds2["X_test"], ds1ds2["y_test"])
    _save_cm(clf, ds1ds2["X_test"], ds1ds2["y_test"], "S2_random_model_DS2_test")

    return clf


def run_s3(ds1ds2) -> None:
    """S3: Train on DS1, test on DS2 (true inter-patient evaluation)."""
    print(f"\n{'═'*62}")
    print("S3  HONEST  —  DS1 train → DS2 test  (true inter-patient)")
    print(f"{'═'*62}")

    print(f"\n  Training on {len(ds1ds2['y_train']):,} beats (DS1) ...")
    clf = _train(ds1ds2["X_train"], ds1ds2["y_train"])

    m = _metrics(clf, ds1ds2["X_test"], ds1ds2["y_test"])
    _print_summary("S3", m)
    _print_per_class(clf, ds1ds2["X_test"], ds1ds2["y_test"])
    _save_cm(clf, ds1ds2["X_test"], ds1ds2["y_test"], "S3_DS1_train_DS2_test")


def run_s4(ds1ds2) -> None:
    """S4: Ablation A1→A4 on DS1→DS2.  Tests if paper's ablation table holds cross-patient."""
    print(f"\n{'═'*62}")
    print("S4  ABLATION on DS1→DS2  (paper only shows random-split ablation)")
    print(f"{'═'*62}")

    X_tr, y_tr = ds1ds2["X_train"], ds1ds2["y_train"]
    X_te, y_te = ds1ds2["X_test"],  ds1ds2["y_test"]
    y_bin_te   = _to_binary(y_te)

    header = f"  {'Config':<30}{'#Feats':>7}{'Acc':>8}{'AnomRec':>10}{'AnomF1':>9}{'AUC':>8}"
    sep    = f"  {'─'*72}"
    print(f"\n{header}\n{sep}")

    for label, feat_slice in ABLATION_CONFIGS:
        n_feats = feat_slice.stop - feat_slice.start
        print(f"  {label:<30}{n_feats:>7}  training...", end="\r", flush=True)
        clf = _train(X_tr[:, feat_slice], y_tr)
        y_pred     = clf.predict(X_te[:, feat_slice])
        y_bin_pred = _to_binary(np.asarray(y_pred))
        acc    = float(accuracy_score(y_te, y_pred))
        anom_r = float(recall_score(y_bin_te, y_bin_pred, zero_division=0))
        anom_f = float(f1_score(y_bin_te, y_bin_pred, zero_division=0))
        auc    = _binary_auc(clf, X_te[:, feat_slice], y_bin_te)
        print(f"  {label:<30}{n_feats:>7}{acc:>8.3f}{anom_r:>10.3f}{anom_f:>9.3f}{auc:>8.3f}")

    print(sep)
    print("  Paper §8.2 (random split, NOT inter-patient):")
    print(f"    6 feats  acc=0.985  AnomRec=0.862  AnomF1=0.860  AUC=0.971")
    print(f"   19 feats  acc=0.987  AnomRec=0.905  AnomF1=0.901  AUC=0.979")
    print(f"   31 feats  acc=0.988  AnomRec=0.919  AnomF1=0.921  AUC=0.983")
    print(f"   40 feats  acc=0.989  AnomRec=0.925  AnomF1=0.928  AUC=0.985")


# %%

def _print_gap_summary(rand, ds1ds2) -> None:
    """
    Print a one-page comparison table: paper's reported numbers vs what we measured
    on each split.  Drives the central argument that the paper's split was not inter-patient.
    """
    print(f"\n{'═'*62}")
    print("SUMMARY — paper claims vs measured across splits")
    print(f"{'═'*62}")
    rows = [
        ("Metric",          "Paper",   "S1 random", "S2 paper→DS2", "S3 DS1→DS2"),
        ("─"*20,            "─"*7,     "─"*10,      "─"*13,         "─"*11),
        ("Accuracy",        "0.989",   "(see S1)",  "(see S2)",      "(see S3)"),
        ("Binary F1",       "0.928",   "(see S1)",  "(see S2)",      "(see S3)"),
        ("Binary AUC",      "0.985",   "(see S1)",  "(see S2)",      "(see S3)"),
        ("L (LBBB) F1",     "0.915",   "(see S1)",  "(see S2)",      "(see S3)"),
        ("R (RBBB) F1",     "0.921",   "(see S1)",  "(see S2)",      "(see S3)"),
    ]
    for r in rows:
        print(f"  {r[0]:<22}{r[1]:<10}{r[2]:<12}{r[3]:<15}{r[4]}")
    print()
    print("  Interpretation:")
    print("  - S1 ≈ paper  → confirms random split replication")
    print("  - S2 << paper → paper's own model fails cross-patient")
    print("  - S3 ≈ S2     → inter-patient is inherently harder (not a code bug)")


# %%

if __name__ == "__main__":
    print("\nCLARITY-AI 2.0 — Paper Claim Validation")
    print("=" * 62)
    print("Uses paper hyperparameters only — no Optuna, no saved models.")
    print("All models trained fresh in this script.\n")

    rand, ds1ds2 = load_splits()

    run_s1_s2(rand, ds1ds2)
    run_s3(ds1ds2)
    run_s4(ds1ds2)
    _print_gap_summary(rand, ds1ds2)

    print("\nDone.")
