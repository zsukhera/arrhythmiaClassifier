# %% [markdown]
# ## metrics.py — Full metrics suite for inter-patient ECG evaluation
#
# Always report BOTH:
#   plain accuracy     (1 - e)      = fraction correctly classified
#   OvR accuracy       (1 - 2e/K)   = mean one-vs-rest accuracy
#   "predict all N" baseline        = how well trivial predictor scores
#
# Plain accuracy looks high when N = 90% of DS2 (predict-all-N = ~90%).
# OvR accuracy looks even higher at the same error rate.
# Minority-class sensitivity (Se_S, Se_V) is the real quality measure.
#
# AAMI EC57 gross statistics:
#   Se = TP / (TP + FN)    per class
#  +P  = TP / (TP + FP)    per class  (PPV)
#
# References:
#   Farag (2023): uses OvR "average accuracy" = 1-2e/K.
#   ANSI/AAMI EC57:2012 standard for arrhythmia detector evaluation.

# %%
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


# %%
def ovr_accuracy(y_true: np.ndarray, y_pred: np.ndarray, k: int) -> float:
    """
    Mean one-vs-rest accuracy = 1 - 2 * total_errors / (K * n_samples).
    Equivalent to: mean over classes of (TP + TN) / n.
    """
    n = len(y_true)
    if n == 0:
        return float("nan")
    total_err = int((y_true != y_pred).sum())
    return 1.0 - 2.0 * total_err / (k * n)


def majority_accuracy(y: np.ndarray) -> float:
    """Accuracy of a classifier that always predicts the most common class."""
    if len(y) == 0:
        return float("nan")
    counts = np.bincount(y[y >= 0])
    return float(counts.max() / len(y)) if len(counts) > 0 else float("nan")


def macro_f1_from_preds(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fast macro-F1 for early stopping (no sklearn overhead)."""
    classes = np.unique(y_true)
    f1s = []
    for c in classes:
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f1s.append(2 * prec * rec / (prec + rec + 1e-10))
    return float(np.mean(f1s)) if f1s else 0.0


# %%
def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       proba: Optional[np.ndarray] = None,
                       k: int = 5) -> dict:
    """
    Full metrics dict for a multi-class run.

    Returns keys: accuracy, ovr_accuracy, majority_acc, macro_se, macro_ppv,
    macro_f1, weighted_f1, cm (list of lists), per-class se_X/ppv_X/sp_X/f1_X/n_X,
    and optionally auc_macro if proba is provided.
    """
    from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                                  recall_score, confusion_matrix)

    n = len(y_true)
    if n == 0:
        return {}

    acc    = float((y_true == y_pred).mean())
    ovr    = ovr_accuracy(y_true, y_pred, k)
    maj    = majority_accuracy(y_true)

    # Per-class metrics
    classes = list(range(k))
    se_per   = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    ppv_per  = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_per   = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred, labels=classes)

    # Specificity per class
    sp_per = []
    for i in classes:
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = n - tp - fn - fp
        sp_per.append(tn / (tn + fp + 1e-10))

    out = {
        "accuracy":    acc,
        "ovr_accuracy":ovr,
        "majority_acc":maj,
        "macro_se":    float(se_per.mean()),
        "macro_ppv":   float(ppv_per.mean()),
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro",   zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted",zero_division=0)),
        "cm":          cm.tolist(),
    }
    cls_names = AAMI_CLASSES[:k]
    for i, c in enumerate(cls_names):
        out[f"se_{c}"]  = float(se_per[i])
        out[f"ppv_{c}"] = float(ppv_per[i])
        out[f"sp_{c}"]  = float(sp_per[i])
        out[f"f1_{c}"]  = float(f1_per[i])
        out[f"n_{c}"]   = int((y_true == i).sum())

    if proba is not None:
        try:
            if proba.shape[1] == 2:
                auc = roc_auc_score(y_true, proba[:, 1])
            else:
                auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
            out["auc_macro"] = float(auc)
        except Exception:
            out["auc_macro"] = float("nan")

    return out


def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   proba_pos: Optional[np.ndarray] = None) -> dict:
    """Binary classification metrics."""
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  confusion_matrix, f1_score)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    se  = tp / (tp + fn + 1e-10)
    sp  = tn / (tn + fp + 1e-10)
    ppv = tp / (tp + fp + 1e-10)
    f1  = 2 * se * ppv / (se + ppv + 1e-10)
    out = {"se": se, "sp": sp, "ppv": ppv, "f1": f1,
           "tp": tp, "tn": tn, "fp": fp, "fn": fn}
    if proba_pos is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            out["pr_auc"]  = float(average_precision_score(y_true, proba_pos))
        except Exception:
            pass
    return out


# %%
def print_metrics_table(run_name: str, agg: dict, n_classes: int = 5) -> None:
    """Print a formatted metrics summary to stdout."""
    pct  = lambda k: f"{100*agg.get(k+'_mean', float('nan')):6.2f} ± {100*agg.get(k+'_std',0):4.2f}%"
    pct1 = lambda k: f"{100*agg.get(k+'_mean', float('nan')):6.2f}%"

    print(f"\n{'='*70}")
    print(f"  Run: {run_name}  ({agg.get('n_seeds', '?')} seeds)")
    print(f"{'='*70}")
    print(f"  {'Plain accuracy':<30} {pct('accuracy')}")
    print(f"  {'OvR accuracy (1-2e/K)':<30} {pct('ovr_accuracy')}")
    print(f"  {'Majority-class baseline':<30} {pct1('majority_acc')}")
    print(f"  {'-'*50}")
    print(f"  {'Macro sensitivity':<30} {pct('macro_se')}")
    print(f"  {'Macro precision (PPV)':<30} {pct('macro_ppv')}")
    print(f"  {'Macro F1':<30} {pct('macro_f1')}")
    if agg.get("auc_macro_mean") is not None:
        print(f"  {'Macro ROC-AUC':<30} {pct('auc_macro')}")
    print(f"  {'-'*50}")
    cls_names = AAMI_CLASSES[:n_classes]
    for c in cls_names:
        se  = agg.get(f"se_{c}_mean",  float("nan"))
        ppv = agg.get(f"ppv_{c}_mean", float("nan"))
        f1  = agg.get(f"f1_{c}_mean",  float("nan"))
        n   = int(agg.get(f"n_{c}_mean", 0))
        print(f"  Class {c:<4}  Se={100*se:5.1f}%  PPV={100*ppv:5.1f}%  F1={100*f1:5.1f}%  n={n:,}")
    print(f"  {'-'*50}")
    print(f"  Seeds:       {agg.get('n_seeds', '?')}")
    print(f"  Total time:  {agg.get('total_time_s', 0)/60:.1f} min")
    print(f"{'='*70}\n")


def aggregate_seeds(seed_results: list) -> dict:
    """Compute mean ± std of every float metric across seeds."""
    if not seed_results:
        return {}
    keys = [k for k, v in seed_results[0].get("metrics_test", {}).items()
            if isinstance(v, float) and not np.isnan(v)]
    out = {}
    for k in keys:
        vals = [r["metrics_test"][k] for r in seed_results
                if k in r.get("metrics_test", {}) and not np.isnan(r["metrics_test"][k])]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"]  = float(np.std(vals, ddof=0))
    out["n_seeds"]      = len(seed_results)
    out["total_time_s"] = sum(r.get("elapsed", 0) for r in seed_results)
    return out


def plot_confusion(cm: np.ndarray, class_names: list, title: str,
                   out_path=None) -> None:
    """Save or display a confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        return

    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_xticklabels(class_names)
    ax.set_yticks(range(len(class_names))); ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, f"{cm[i,j]:,}",
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black", fontsize=9)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
