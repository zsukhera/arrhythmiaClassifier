# %% [markdown]

# # MIT-BIH — Binary LightGBM Evaluation (Normal vs Anomaly)
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Evaluates the dedicated binary models trained by train_binary.py against
#   (a) the paper's reported binary numbers (clarityBase.txt §8.1)
#   (b) the 5-class model (models/lgbm_{split}.pkl) collapsed to binary
#
# Revision 2026-11-18 (lgbm-binary agent):
#   * --tags to evaluate any lgbm_binary_{tag}.pkl (e.g. ds1ds2_stratifiedcv)
#   * Reports three operating points for the binary model:
#       - threshold 0.5
#       - the OOF threshold stored in the bundle (chosen on training-set
#         out-of-fold probabilities — the honest operating point)
#       - the test-set-optimal threshold, labelled ORACLE (informational only,
#         never used as a result)
#   * Recall broken down by original class (L/R/V/A) and N specificity for each
#     operating point, for both the binary model and the 5-class collapsed model.
#   * All numbers written to results/binary/binary_eval_{tag}.json
#
# Usage (from $ROOT):
#   .venv/bin/python source/binaryLightGBM/evaluate_binary.py --tags random ds1ds2

# %%

import argparse
import json
import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings("ignore")

# %%

_HERE      = Path(__file__).resolve().parent
FYP_ROOT   = _HERE.parent.parent
PROC_DIR   = FYP_ROOT / "datasets" / "processed"
MODEL_DIR  = FYP_ROOT / "models"
RESULT_DIR = FYP_ROOT / "results" / "binary"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER = ["N", "L", "R", "V", "A"]
LABEL_NAMES = {"N": "Normal", "L": "LBBB", "R": "RBBB", "V": "PVC", "A": "APC"}

PAPER_BINARY = {"acc": 0.989, "prec": 0.931, "rec": 0.925, "f1": 0.928, "auc": 0.985}

# %%


def to_binary(y: np.ndarray) -> np.ndarray:
    return np.where(y == "N", 0, 1).astype(int)


def compute_metrics(y_true, proba, thr, y5) -> dict:
    y_pred = (proba >= thr).astype(int)
    m = {
        "threshold": float(thr),
        "acc":  float(accuracy_score(y_true, y_pred)),
        "prec": float(precision_score(y_true, y_pred, zero_division=0)),
        "rec":  float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":   float(f1_score(y_true, y_pred, zero_division=0)),
        "auc":  float(roc_auc_score(y_true, proba)),
        # recall by original class; for N this is specificity (pred 0)
        "rec_by_class": {},
        "n_by_class": {},
    }
    for c in LABEL_ORDER:
        mask = y5 == c
        if mask.sum() == 0:
            continue
        hit = (y_pred[mask] == 0) if c == "N" else (y_pred[mask] == 1)
        m["rec_by_class"][c] = float(hit.mean())
        m["n_by_class"][c] = int(mask.sum())
    return m


def oracle_threshold(y_true, proba) -> float:
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = (2 * prec * rec / np.clip(prec + rec, 1e-12, None))[:-1]
    return float(thr[int(np.nanargmax(f1))])


def _binary_proba_from_5class(model_5class, X):
    proba = model_5class.predict_proba(X)
    anom_cols = [i for i, c in enumerate(model_5class.classes_) if c != "N"]
    return proba[:, anom_cols].sum(axis=1)


# %%


def save_confusion_matrix(y_true, y_pred, tag):
    cm   = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Anomaly"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(tag)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"cm_{tag.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def save_roc_curve(y_true, proba_dict, tag):
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, proba in proba_dict.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, proba):.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title(f"ROC — {tag}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULT_DIR / f"roc_{tag.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


# %%


def _row(label, m):
    rc = m["rec_by_class"]
    print(f"  {label:<38} thr={m['threshold']:.3f}  Acc={m['acc']:.4f}  P={m['prec']:.4f}"
          f"  R={m['rec']:.4f}  F1={m['f1']:.4f}  AUC={m['auc']:.4f}  |  "
          + "  ".join(f"{c}={rc[c]:.3f}" for c in LABEL_ORDER if c in rc))


def evaluate_tag(tag: str) -> dict:
    split_name = tag.split("_")[0]
    print(f"\n{'═'*100}\nBinary evaluation — model tag: {tag}  (test split: {split_name})\n{'═'*100}")
    data = np.load(PROC_DIR / f"split_{split_name}.npz", allow_pickle=True)
    X_test, y5 = data["X_test"], data["y_test"]
    y_test = to_binary(y5)
    print(f"  Test set : {len(y_test):,} beats — Normal={int((y_test==0).sum()):,}  "
          f"Anomaly={int((y_test==1).sum()):,}")
    print("  Columns after '|': per-original-class recall (N column = specificity)")
    print(f"  Paper (§8.1): Acc={PAPER_BINARY['acc']}  P={PAPER_BINARY['prec']}  "
          f"R={PAPER_BINARY['rec']}  F1={PAPER_BINARY['f1']}  AUC={PAPER_BINARY['auc']}")

    out = {"tag": tag, "paper": PAPER_BINARY}
    proba_plot = {}

    cls_pkl = MODEL_DIR / f"lgbm_{split_name}.pkl"
    if cls_pkl.exists():
        with open(cls_pkl, "rb") as f:
            model_5 = pickle.load(f)["model"]
        p5 = _binary_proba_from_5class(model_5, X_test)
        # The 5-class model's own argmax decision (what evaluate.py reports)
        pred5 = to_binary(np.asarray(model_5.predict(X_test)))
        m5_arg = compute_metrics(y_test, pred5.astype(float), 0.5, y5)
        m5_arg["auc"] = float(roc_auc_score(y_test, p5))
        m5_arg["threshold"] = float("nan")
        m5_05 = compute_metrics(y_test, p5, 0.5, y5)
        _row("5-class collapsed (argmax)", m5_arg)
        _row("5-class collapsed (sum P(anom)>=0.5)", m5_05)
        out["five_class_argmax"] = m5_arg
        out["five_class_sum05"] = m5_05
        proba_plot["5-class collapsed"] = p5

    bin_pkl = MODEL_DIR / f"lgbm_binary_{tag}.pkl"
    with open(bin_pkl, "rb") as f:
        bundle = pickle.load(f)
    model = bundle["model"]
    p = model.predict_proba(X_test)[:, 1]
    proba_plot["binary"] = p

    m05 = compute_metrics(y_test, p, 0.5, y5)
    _row("Binary @0.5", m05)
    out["binary_05"] = m05
    if "threshold" in bundle:
        m_oof = compute_metrics(y_test, p, bundle["threshold"], y5)
        _row(f"Binary @OOF thr ({bundle.get('cv_kind')} CV)", m_oof)
        out["binary_oofthr"] = m_oof
        save_confusion_matrix(y_test, (p >= bundle["threshold"]).astype(int),
                              f"binary_{tag}_oofthr")
    t_or = oracle_threshold(y_test, p)
    m_or = compute_metrics(y_test, p, t_or, y5)
    _row("Binary @ORACLE test thr (not a result)", m_or)
    out["binary_oracle"] = m_or
    save_confusion_matrix(y_test, (p >= 0.5).astype(int), f"binary_{tag}_05")
    save_roc_curve(y_test, proba_plot, tag)

    out["bundle"] = {k: v for k, v in bundle.items() if k != "model"}
    print(f"  Training CV objective ({bundle.get('objective')}): {bundle.get('cv_score'):.4f}   "
          f"OOF AUC: {bundle.get('oof_auc', float('nan')):.4f}")

    # Per-record view on the test split (where do the misses come from?)
    rec_ids = data["record_ids_test"]
    thr = bundle.get("threshold", 0.5)
    per_rec = {}
    for r in sorted(set(rec_ids)):
        mask = rec_ids == r
        yt, yp = y_test[mask], (p[mask] >= thr).astype(int)
        per_rec[str(r)] = {
            "n": int(mask.sum()), "n_anom": int(yt.sum()),
            "tp": int(((yp == 1) & (yt == 1)).sum()), "fp": int(((yp == 1) & (yt == 0)).sum()),
            "fn": int(((yp == 0) & (yt == 1)).sum()),
        }
    out["per_record_at_used_thr"] = per_rec
    worst = sorted(per_rec.items(), key=lambda kv: -(kv[1]["fn"] + kv[1]["fp"]))[:8]
    print("  Records with most errors (at OOF thr):  " +
          ", ".join(f"{r}: FN={v['fn']} FP={v['fp']}" for r, v in worst))

    (RESULT_DIR / f"binary_eval_{tag}.json").write_text(json.dumps(out, indent=2, default=str))
    return out


# %%

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", default=["random", "ds1ds2"])
    a = ap.parse_args()
    for t in a.tags:
        if not (MODEL_DIR / f"lgbm_binary_{t}.pkl").exists():
            print(f"[skip] lgbm_binary_{t}.pkl not found")
            continue
        evaluate_tag(t)
    print("\nDone.")
