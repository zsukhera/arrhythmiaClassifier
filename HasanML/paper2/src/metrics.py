"""Multiclass metrics under every averaging scheme, from a confusion matrix.

With K classes and multiclass error rate e, the one-vs-rest quantities collapse to:
  micro sensitivity = micro precision = micro F1 = plain accuracy = 1 - e
  mean OvR binary accuracy = 1 - 2e/K
  micro specificity        = 1 - e/(K-1)
The paper's "Accuracy / Sensitivity / Specificity / F1" match these (see README).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from .data import CLASSES


def confusion(y_true, y_pred, k: int = len(CLASSES)) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def _div(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.full(np.broadcast(a, b).shape, np.nan), where=b > 0)


def metrics_from_cm(cm: np.ndarray) -> dict:
    """All scalar metrics (fractions, not %) for a KxK confusion matrix (rows = truth)."""
    cm = np.asarray(cm, dtype=np.int64)
    k, n = cm.shape[0], cm.sum()
    tp = np.diag(cm).astype(float)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = n - tp - fn - fp
    support = cm.sum(axis=1)
    present = support > 0

    sens, spec, prec = _div(tp, tp + fn), _div(tn, tn + fp), _div(tp, tp + fp)
    f1 = _div(2 * tp, 2 * tp + fp + fn)
    w = support / max(n, 1)

    out = {
        "accuracy": tp.sum() / n,                              # plain multiclass accuracy
        "ovr_accuracy": float(np.mean((tp + tn) / n)),          # mean one-vs-rest accuracy
        "majority_accuracy": support.max() / n,                 # predict-all-majority (N) baseline
        "micro_sensitivity": tp.sum() / (tp + fn).sum(),
        "micro_specificity": tn.sum() / (tn + fp).sum(),
        "micro_precision": tp.sum() / (tp + fp).sum(),
        "micro_f1": 2 * tp.sum() / (2 * tp.sum() + fp.sum() + fn.sum()),
    }
    # Macro: classes absent from the test set are skipped (their sensitivity is undefined).
    for name, v in (("sensitivity", sens), ("specificity", spec), ("precision", prec), ("f1", f1)):
        vp = np.where(np.isnan(v[present]), 0.0, v[present])   # 0/0 precision (never predicted) -> 0
        out[f"macro_{name}"] = float(vp.mean())
        out[f"weighted_{name}"] = float(np.sum(np.where(np.isnan(v), 0.0, v) * w))
    for i, c in enumerate(CLASSES[:k]):
        out[f"se_{c}"] = sens[i]
        out[f"ppv_{c}"] = prec[i]
        out[f"n_{c}"] = int(support[i])
    out["n_test"] = int(n)
    return out


def paper_metrics_from_error(e: float, k: int = len(CLASSES)) -> dict:
    """The micro-averaging hypothesis: the paper's four numbers implied by error rate e."""
    return {"accuracy": 1 - 2 * e / k, "sensitivity": 1 - e, "specificity": 1 - e / (k - 1), "f1": 1 - e}


def auc_scores(y_true: np.ndarray, proba: np.ndarray, k: int = len(CLASSES)) -> dict:
    """Macro one-vs-rest AUC over the classes present in y_true, and micro AUC."""
    present = np.unique(y_true)
    Y = label_binarize(y_true, classes=np.arange(k))
    macro = np.mean([roc_auc_score(Y[:, c], proba[:, c]) for c in present]) if len(present) > 1 else np.nan
    micro = roc_auc_score(Y[:, present].ravel(), proba[:, present].ravel())
    return {"auc_macro": float(macro), "auc_micro": float(micro)}


def roc_curves(y_true: np.ndarray, proba: np.ndarray, k: int = len(CLASSES)) -> dict:
    """{class name | 'micro': (fpr, tpr, auc)} for the classes present in y_true."""
    Y = label_binarize(y_true, classes=np.arange(k))
    out = {}
    for c in np.unique(y_true):
        fpr, tpr, _ = roc_curve(Y[:, c], proba[:, c])
        out[CLASSES[c]] = (fpr, tpr, roc_auc_score(Y[:, c], proba[:, c]))
    present = np.unique(y_true)
    fpr, tpr, _ = roc_curve(Y[:, present].ravel(), proba[:, present].ravel())
    out["micro"] = (fpr, tpr, roc_auc_score(Y[:, present].ravel(), proba[:, present].ravel()))
    return out
