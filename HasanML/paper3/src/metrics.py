# %% [markdown]
# ## metrics.py — All evaluation metrics for multiclass and binary tasks
#
# Reference: Farag (2023), Sections 5 and 6; de Chazal (2004) AAMI EC57 style.
#
# Key design decision: report BOTH accuracy definitions side by side because
# the paper's "average accuracy" is ambiguous.  Investigation of paper2 (Yakut
# 2020) revealed that "average accuracy" there meant mean one-vs-rest accuracy
# (1 - 2e/K), not plain multiclass accuracy (1 - e).  We compute both and let
# the per-class confusion matrix decide which definition matches Table 4.
#
# Majority-class baseline: N is ~89.5 % of DS2 beats (paper Sec 5.1).
# Any classifier that simply predicts "Normal" achieves ~89.5 %, so plain
# accuracy of e.g. 97 % means the model gains ~7.5 pp over the trivial baseline.

# %%
import io
import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

log = logging.getLogger(__name__)

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


# %%
def _safe_div(a, b, fill=np.nan):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.where(b > 0, a / b, fill)


def confusion_matrix_k(y_true: np.ndarray, y_pred: np.ndarray,
                        k: int) -> np.ndarray:
    """K×K integer confusion matrix; rows=truth, cols=prediction."""
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       proba: Optional[np.ndarray] = None,
                       k: Optional[int] = None) -> dict:
    """
    Full set of multiclass metrics from predictions.

    Returns dict with:
      accuracy        : plain accuracy 1 - e
      ovr_accuracy    : mean one-vs-rest accuracy 1 - 2e/K  (paper's "avg accuracy")
      majority_acc    : predict-all-N baseline
      macro_se, macro_ppv, macro_sp, macro_f1 : macro-averaged
      weighted_se, ...weighted F1
      se_N, ppv_N, sp_N, f1_N, ... (per class, up to 5)
      auc_macro, auc_micro (if proba supplied)
      cm              : K×K confusion matrix as list
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if k is None:
        k = max(y_true.max(), y_pred.max()) + 1

    cm   = confusion_matrix_k(y_true, y_pred, k)
    n    = cm.sum()
    tp   = np.diag(cm).astype(float)
    fn   = cm.sum(axis=1) - tp
    fp   = cm.sum(axis=0) - tp
    tn   = n - tp - fn - fp
    sup  = cm.sum(axis=1)
    pres = sup > 0

    se   = _safe_div(tp, tp + fn)       # sensitivity (recall)
    sp   = _safe_div(tn, tn + fp)       # specificity
    ppv  = _safe_div(tp, tp + fp)       # precision (PPV)
    f1   = _safe_div(2*tp, 2*tp + fp + fn)
    w    = sup / max(n, 1)

    out = {
        "accuracy":     float(tp.sum() / n),
        "ovr_accuracy": float(np.mean((tp + tn) / n)),   # 1 - 2e/K in K-class case
        "majority_acc": float(sup.max() / n),
    }

    for name, v in (("se", se), ("ppv", ppv), ("sp", sp), ("f1", f1)):
        # macro: skip absent classes (undefined metric -> nan -> excluded from mean)
        vals_pres = v[pres]
        vals_pres_nonan = np.where(np.isnan(vals_pres), 0.0, vals_pres)
        out[f"macro_{name}"]    = float(vals_pres_nonan.mean()) if len(vals_pres_nonan) else np.nan
        out[f"weighted_{name}"] = float(np.nansum(np.where(np.isnan(v), 0.0, v) * w))

    for i, cname in enumerate(AAMI_CLASSES[:k]):
        for metric, v in (("se", se), ("ppv", ppv), ("sp", sp), ("f1", f1)):
            out[f"{metric}_{cname}"] = float(v[i])
        out[f"n_{cname}"] = int(sup[i])

    out["n_test"] = int(n)
    out["cm"] = cm.tolist()

    # ROC-AUC (requires predicted probabilities)
    if proba is not None:
        proba = np.asarray(proba)
        present = np.unique(y_true)
        Y = label_binarize(y_true, classes=np.arange(k))
        if k == 2 and Y.shape[1] == 1:   # sklearn returns one column for 2 classes
            Y = np.column_stack([1 - Y[:, 0], Y[:, 0]])
        try:
            if len(present) > 1:
                out["auc_macro"] = float(
                    np.mean([roc_auc_score(Y[:, c], proba[:, c]) for c in present])
                )
                out["auc_micro"] = float(
                    roc_auc_score(Y[:, present].ravel(), proba[:, present].ravel())
                )
            else:
                out["auc_macro"] = out["auc_micro"] = np.nan
        except Exception as exc:
            log.warning("AUC computation failed: %s", exc)
            out["auc_macro"] = out["auc_micro"] = np.nan

    return out


def ovr_accuracy_check(cm: np.ndarray) -> dict:
    """
    Reproduce the paper's accuracy check: given the K×K confusion matrix,
    compute both accuracy definitions and return a comparison.
    Use this to decide whether the paper's "avg accuracy" is 1-e or 1-2e/K.
    """
    n  = cm.sum()
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    e  = 1 - tp.sum() / n                # multiclass error
    k  = cm.shape[0]
    return {
        "plain_accuracy_pct":  100 * (1 - e),
        "ovr_accuracy_pct":    100 * float(np.mean((tp + tn) / n)),
        "note": f"For K={k}, OvR accuracy = 1 - 2e/K.  "
                f"At plain_acc={100*(1-e):.2f}%, OvR={100*float(np.mean((tp+tn)/n)):.2f}%."
    }


# %%
def binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                   proba_pos: Optional[np.ndarray] = None) -> dict:
    """
    Binary classification metrics.
    y_true, y_pred: 0 = Normal, 1 = Arrhythmia.
    proba_pos     : probability of class 1 (for AUC).
    """
    y_true = np.asarray(y_true, int)
    y_pred = np.asarray(y_pred, int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    n = tp + fp + fn + tn

    se   = tp / (tp + fn) if (tp + fn) > 0 else np.nan  # sensitivity / recall
    sp   = tn / (tn + fp) if (tn + fp) > 0 else np.nan  # specificity
    ppv  = tp / (tp + fp) if (tp + fp) > 0 else np.nan  # precision / PPV
    f1   = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else np.nan
    acc  = (tp + tn) / n if n > 0 else np.nan

    out = {
        "accuracy": acc, "sensitivity": se, "specificity": sp,
        "ppv": ppv, "f1_arr": f1,
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "n_test": int(n),
        "cm": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }

    if proba_pos is not None:
        proba_pos = np.asarray(proba_pos, float)
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, proba_pos))
            out["pr_auc"]  = float(average_precision_score(y_true, proba_pos))
        except Exception as exc:
            log.warning("binary AUC failed: %s", exc)
            out["roc_auc"] = out["pr_auc"] = np.nan

    return out


def find_optimal_threshold(proba_pos: np.ndarray, y_true: np.ndarray,
                            start: float = 0.01, stop: float = 0.99,
                            step: float = 0.01) -> tuple:
    """
    Sweep decision thresholds on the validation set.
    Return (best_threshold, best_f1_arrhythmia) where best = max F1(class=1).
    """
    thresholds = np.arange(start, stop + step / 2, step)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (proba_pos >= t).astype(int)
        m = binary_metrics(y_true, pred)
        if m["f1_arr"] > best_f1:
            best_f1, best_t = m["f1_arr"], float(t)
    return best_t, best_f1


# %%
def plot_confusion(cm: np.ndarray, class_names: list, title: str,
                   out_path: Optional[Path] = None) -> None:
    """Save a confusion matrix heat-map."""
    fig, ax = plt.subplots(figsize=(4 + 0.5 * len(class_names),
                                    3 + 0.5 * len(class_names)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() * 0.5 else "black",
                    fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def metrics_to_md_row(name: str, m: dict, paper_row: Optional[dict] = None) -> str:
    """Format one result row for the results_summary.md table."""
    pct = lambda v: f"{100*v:.2f}" if v is not None and not np.isnan(v) else "—"
    row = (f"| {name} | {pct(m.get('accuracy'))} | {pct(m.get('ovr_accuracy'))} | "
           f"{pct(m.get('majority_acc'))} | {pct(m.get('macro_se'))} | "
           f"{pct(m.get('macro_ppv'))} | {pct(m.get('macro_f1'))} | "
           f"{pct(m.get('se_S'))} | {pct(m.get('se_V'))} |")
    if paper_row:
        row += (f" {paper_row.get('ovr_accuracy', '—')} | "
                f"{paper_row.get('macro_f1', '—')} |")
    return row


def summarise_seeds(metrics_list: list) -> dict:
    """Average and std of metrics over multiple seed runs."""
    keys = [k for k in metrics_list[0] if isinstance(metrics_list[0][k], float) and not np.isnan(metrics_list[0][k])]
    out = {}
    for k in keys:
        vals = [m[k] for m in metrics_list if k in m and not np.isnan(m[k])]
        out[f"{k}_mean"] = float(np.mean(vals))
        out[f"{k}_std"]  = float(np.std(vals, ddof=0))
    return out
