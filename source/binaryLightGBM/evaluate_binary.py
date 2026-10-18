# %% [markdown]

# # MIT-BIH — Binary LightGBM Evaluation (Normal vs Anomaly)
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Evaluates the dedicated binary models trained by train_binary.py and
# compares them against:
#   (a) The paper's reported binary numbers (clarityBase.txt §8.1)
#   (b) The 5-class model collapsed to binary (from train.py)
#
# Reports
# -------
#   1. Main binary metrics at default threshold (0.5)
#      Accuracy, Precision, Recall, F1 (anomaly), AUC-ROC
#   2. Optimal-threshold metrics (threshold that maximises anomaly-class F1
#      on the test set — informational, not tuned during training)
#   3. Class-origin breakdown: which 5-class label contributes how many
#      binary errors (helps diagnose which arrhythmia type still fails)
#   4. Confusion matrices saved to results/binary/
#   5. Side-by-side comparison: paper / 5-class-collapsed / binary-model
#
# Input:  models/lgbm_binary_random.pkl
#         models/lgbm_binary_ds1ds2.pkl
#         models/lgbm_ds1ds2.pkl          (5-class, for comparison)
#         datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz

# %%

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# %%

_HERE      = Path(__file__).resolve().parent        # .../source/binaryLightGBM/
FYP_ROOT   = _HERE.parent.parent                    # .../fyp/
PROC_DIR   = FYP_ROOT / "datasets" / "processed"
MODEL_DIR  = FYP_ROOT / "models"
RESULT_DIR = FYP_ROOT / "results" / "binary"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["N", "L", "R", "V", "A"]
LABEL_NAMES = {"N": "Normal", "L": "LBBB", "R": "RBBB", "V": "PVC", "A": "APC"}

# Paper's reported binary anomaly numbers (clarityBase.txt §8.1)
PAPER_BINARY = {"acc": 0.989, "prec": 0.931, "rec": 0.925, "f1": 0.928, "auc": 0.985}

# %%


def to_binary(y: np.ndarray) -> np.ndarray:
    """N → 0, {L,R,V,A} → 1."""
    return np.where(y == "N", 0, 1).astype(int)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    """Full binary metrics at threshold 0.5."""
    return {
        "acc":  float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "rec":  float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1":   float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "auc":  float(roc_auc_score(y_true, proba)),
    }


def optimal_threshold_metrics(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, dict]:
    """
    Find the threshold that maximises anomaly-class F1 on the test set.
    Returns (threshold, metrics_dict).

    Note: this is informational only — the threshold is chosen on the same
    test set we evaluate on, so it is optimistic.  In a real deployment the
    threshold should be chosen on a held-out validation set.
    """
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    best_t = 0.5
    best_f1 = 0.0
    for t in thresholds:
        y_pred_t = (proba >= t).astype(int)
        f = float(f1_score(y_true, y_pred_t, pos_label=1, zero_division=0))
        if f > best_f1:
            best_f1 = f
            best_t = float(t)

    y_opt = (proba >= best_t).astype(int)
    return best_t, compute_metrics(y_true, y_opt, proba)


def _binary_proba_from_5class(model_5class, X: np.ndarray) -> np.ndarray:
    """
    Collapse a 5-class model's output to anomaly probability.
    Anomaly score = sum of P(L) + P(R) + P(V) + P(A).
    """
    proba   = model_5class.predict_proba(X)
    classes = list(model_5class.classes_)
    anom_cols = [i for i, c in enumerate(classes) if c != "N"]
    return proba[:, anom_cols].sum(axis=1)


# %%


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, tag: str) -> None:
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Normal", "Anomaly"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(tag)
    fig.tight_layout()
    out = RESULT_DIR / f"cm_{tag.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Confusion matrix → {out.name}")


def save_roc_curve(y_true: np.ndarray, proba_dict: dict, tag: str) -> None:
    """Overlay ROC curves for multiple models on the same axes."""
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC — {tag}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    out = RESULT_DIR / f"roc_{tag.replace(' ', '_')}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  ROC curve       → {out.name}")


# %%


def class_origin_breakdown(
    y_5class: np.ndarray,
    y_bin_pred: np.ndarray,
    y_bin_true: np.ndarray,
) -> None:
    """
    For each original 5-class label, print how many beats were correctly
    classified (by the binary model) vs missed/false-alarmed.
    """
    print(f"\n  {'Class':<8}{'Name':<10}{'N_beats':>8}"
          f"{'Correct':>9}{'Recall':>8}")
    print(f"  {'─'*48}")
    for cls in LABEL_ORDER:
        mask = (y_5class == cls)
        if mask.sum() == 0:
            continue
        if cls == "N":
            # True negatives: predicted 0 where truth is 0
            correct = int(((y_bin_pred == 0) & mask).sum())
        else:
            # True positives: predicted 1 where truth is 1
            correct = int(((y_bin_pred == 1) & mask).sum())
        total  = int(mask.sum())
        recall = correct / total if total else 0.0
        print(f"  {cls:<8}{LABEL_NAMES.get(cls, ''):<10}{total:>8,}"
              f"{correct:>9,}{recall:>8.3f}")
    print(f"  {'─'*48}")


# %%


def _print_metric_row(label: str, m: dict) -> None:
    print(f"  {label:<30}"
          f"  Acc={m['acc']:.4f}"
          f"  P={m['prec']:.4f}"
          f"  R={m['rec']:.4f}"
          f"  F1={m['f1']:.4f}"
          f"  AUC={m['auc']:.4f}")


def evaluate_split(split_name: str) -> None:
    print(f"\n{'═'*72}")
    print(f"Binary evaluation — split: {split_name}")
    print(f"{'═'*72}")

    bin_pkl  = MODEL_DIR / f"lgbm_binary_{split_name}.pkl"
    cls_pkl  = MODEL_DIR / f"lgbm_{split_name}.pkl"
    npz_path = PROC_DIR  / f"split_{split_name}.npz"

    if not npz_path.exists():
        print(f"  [skip] {npz_path.name} not found — run split.py first.")
        return

    data     = np.load(npz_path, allow_pickle=True)
    X_test   = data["X_test"]
    y_test_5 = data["y_test"]          # 5-class labels (strings)
    y_test   = to_binary(y_test_5)     # binary ground truth

    n_normal = int((y_test == 0).sum())
    n_anom   = int((y_test == 1).sum())
    print(f"\n  Test set : {len(y_test):,} beats — "
          f"Normal={n_normal:,}  Anomaly={n_anom:,}")

    # ── Paper reference ───────────────────────────────────────────────────────
    print(f"\n  {'─'*72}")
    _print_metric_row("Paper (clarityBase.txt §8.1)", PAPER_BINARY)
    print(f"  {'─'*72}")

    # ── 5-class model collapsed to binary (for comparison) ───────────────────
    proba_dict_plot: dict[str, np.ndarray] = {}

    if cls_pkl.exists():
        with open(cls_pkl, "rb") as f:
            cls_bundle = pickle.load(f)
        model_5 = cls_bundle["model"]

        proba_5  = _binary_proba_from_5class(model_5, X_test)
        y_pred_5 = (proba_5 >= 0.5).astype(int)
        m5       = compute_metrics(y_test, y_pred_5, proba_5)

        _print_metric_row(f"5-class model (collapsed, {split_name})", m5)
        save_confusion_matrix(y_test, y_pred_5, f"5class_collapsed_{split_name}")
        proba_dict_plot["5-class collapsed"] = proba_5
    else:
        print(f"  [skip] 5-class model not found: {cls_pkl.name}")
        m5 = None

    # ── Dedicated binary model ───────────────────────────────────────────────
    if not bin_pkl.exists():
        print(f"\n  [skip] Binary model not found: {bin_pkl.name}")
        print("  Run train_binary.py first.")
        return

    with open(bin_pkl, "rb") as f:
        bin_bundle = pickle.load(f)
    model_bin = bin_bundle["model"]
    cv_f1     = bin_bundle.get("cv_score", float("nan"))

    proba_bin  = model_bin.predict_proba(X_test)[:, 1]
    y_pred_bin = (proba_bin >= 0.5).astype(int)
    m_bin      = compute_metrics(y_test, y_pred_bin, proba_bin)

    _print_metric_row(f"Binary model (dedicated, {split_name})", m_bin)
    print(f"  {'─'*72}")
    print(f"  CV anomaly-F1 during training : {cv_f1:.4f}")

    save_confusion_matrix(y_test, y_pred_bin, f"binary_model_{split_name}")
    proba_dict_plot["binary dedicated"] = proba_bin
    save_roc_curve(y_test, proba_dict_plot, split_name)

    # ── Optimal threshold ────────────────────────────────────────────────────
    opt_t, m_opt = optimal_threshold_metrics(y_test, proba_bin)
    print(f"\n  Optimal threshold : {opt_t:.4f}  (maximises anomaly-F1 on test set)")
    _print_metric_row("Binary model (optimal threshold)", m_opt)
    print("  Note: optimal threshold chosen on test set — informational only.")

    # ── Class-origin breakdown ───────────────────────────────────────────────
    print(f"\n  Class-origin recall (binary model, default threshold):")
    class_origin_breakdown(y_test_5, y_pred_bin, y_test)

    # ── Improvement summary ──────────────────────────────────────────────────
    if m5 is not None:
        delta_f1  = m_bin["f1"]  - m5["f1"]
        delta_auc = m_bin["auc"] - m5["auc"]
        delta_acc = m_bin["acc"] - m5["acc"]
        print(f"\n  Binary vs 5-class-collapsed ΔF1={delta_f1:+.4f}  "
              f"ΔAUC={delta_auc:+.4f}  ΔAcc={delta_acc:+.4f}")


# %%


if __name__ == "__main__":
    print("\nMIT-BIH Binary LightGBM Evaluation")
    print("=" * 72)
    print("Comparing: paper / 5-class collapsed / dedicated binary model")

    for split_name in ("random", "ds1ds2"):
        evaluate_split(split_name)

    print("\nDone.")
