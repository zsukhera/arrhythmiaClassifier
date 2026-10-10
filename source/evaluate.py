# %% [markdown]

# # MIT-BIH — Step 8: Evaluate
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Loads models from train.py and computes the full result set from clarityBase.txt §8:
#
#   1. 5-class metrics  — accuracy, per-class P/R/F1, macro-F1, OVR AUC
#   2. Binary anomaly   — N vs {L,R,V,A}: precision, recall, F1, AUC  (§9.14)
#   3. Ablation A1→A4  — LGBM (paper params) on 6/19/31/40 features    (§8.2)
#   4. Synergy          — GBC vs LGBM on 6 vs 40 features               (§8.3)
#   5. Confusion matrices saved to results/
#
# Input:  models/lgbm_random.pkl, models/lgbm_ds1ds2.pkl
#         datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz
# Output: printed tables
#         results/cm_*.png

# %%

import pickle
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

# %%

_HERE      = Path(__file__).resolve().parent
PROC_DIR   = _HERE.parent / "datasets" / "processed"
MODEL_DIR  = _HERE.parent / "models"
RESULT_DIR = _HERE.parent / "results"

SEED        = 42
LABEL_ORDER = ["N", "L", "R", "V", "A"]
LABEL_NAMES = {"N": "Normal", "L": "LBBB", "R": "RBBB", "V": "PVC", "A": "APC"}

# Feature group slices into the 40-column feature matrix (clarityBase.txt §5, §9.7)
#   [0:6]   A  Statistical           (6)
#   [6:19]  B  Fiducial B1+B2       (13)  paper says 14, counts don't add up (§9.7)
#   [19:31] C  Wavelet DWT          (12)
#   [31:40] D  HRV + Non-linear      (9)
ABLATION_CONFIGS = [
    ("A1 Statistical (6)",      slice(0, 6)),
    ("A2 +Fiducial (19)",       slice(0, 19)),
    ("A3 +Wavelet (31)",        slice(0, 31)),
    ("A4 +HRV/Non-linear (40)", slice(0, 40)),
]

PAPER_PARAMS = {
    "n_estimators":     1450,
    "max_depth":        11,
    "learning_rate":    0.05,
    "num_leaves":       38,
    "subsample":        0.85,
    "colsample_bytree": 0.75,
}

# %%

def to_binary(y: np.ndarray) -> np.ndarray:
    """N → 0, {L,R,V,A} → 1."""
    return np.where(y == "N", 0, 1).astype(int)


def binary_auc(model, X_test: np.ndarray, y_bin: np.ndarray) -> float:
    """
    ROC-AUC for binary anomaly detection.
    Anomaly score = sum of predicted probabilities for all non-N classes.
    """
    proba     = model.predict_proba(X_test)
    classes   = list(model.classes_)
    anom_cols = [i for i, c in enumerate(classes) if c != "N"]
    scores    = proba[:, anom_cols].sum(axis=1)
    try:
        return float(roc_auc_score(y_bin, scores))
    except ValueError:
        return float("nan")


def compute_5class_auc(model, X_test: np.ndarray, y_test: np.ndarray) -> float:
    try:
        proba = model.predict_proba(X_test)
        return float(roc_auc_score(
            y_test, proba,
            multi_class="ovr", average="macro",
            labels=model.classes_,
        ))
    except Exception:
        return float("nan")

# %%

def print_per_class(model, X_test: np.ndarray, y_test: np.ndarray) -> None:
    y_pred  = model.predict(X_test)
    labels  = [c for c in LABEL_ORDER if c in set(y_test)]
    prec, rec, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=labels, zero_division=0,
    )
    print(f"\n  {'Cls':<5}{'Name':<10}{'Prec':>8}{'Rec':>8}{'F1':>8}{'Support':>10}")
    print(f"  {'─'*49}")
    for i, cls in enumerate(labels):
        print(f"  {cls:<5}{LABEL_NAMES.get(cls, ''):<10}"
              f"{prec[i]:>8.3f}{rec[i]:>8.3f}{f1[i]:>8.3f}{support[i]:>10,}")
    print(f"  {'─'*49}")


def save_confusion_matrix(
    model, X_test: np.ndarray, y_test: np.ndarray, tag: str
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    y_pred  = model.predict(X_test)
    labels  = [c for c in LABEL_ORDER if c in set(y_test)]
    cm      = confusion_matrix(y_test, y_pred, labels=labels)
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

def run_main_eval() -> None:
    print(f"\n{'═'*62}")
    print("MAIN EVALUATION — tuned models from train.py")
    print(f"{'═'*62}")

    for split_name in ("random", "ds1ds2"):
        pkl_path   = MODEL_DIR / f"lgbm_{split_name}.pkl"
        split_path = PROC_DIR  / f"split_{split_name}.npz"

        if not pkl_path.exists():
            print(f"\n  [skip] {pkl_path.name} not found — run train.py first.")
            continue

        with open(pkl_path, "rb") as f:
            bundle = pickle.load(f)
        model  = bundle["model"]
        cv_f1  = bundle.get("cv_score", float("nan"))

        data   = np.load(split_path, allow_pickle=True)
        X_test = data["X_test"]
        y_test = data["y_test"]

        y_pred    = model.predict(X_test)
        y_bin     = to_binary(y_test)
        y_bin_pred = to_binary(np.asarray(y_pred))

        acc      = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
        auc_5    = compute_5class_auc(model, X_test, y_test)
        anom_p   = float(precision_score(y_bin, y_bin_pred, zero_division=0))
        anom_r   = float(recall_score(y_bin, y_bin_pred, zero_division=0))
        anom_f1  = float(f1_score(y_bin, y_bin_pred, zero_division=0))
        anom_auc = binary_auc(model, X_test, y_bin)

        print(f"\n  Split  : {split_name}")
        print(f"  Test   : {len(y_test):,} beats")
        print(f"  CV F1  : {cv_f1:.4f}  (training objective)")
        print(f"\n  ── 5-class ─────────────────────────────────────────")
        print(f"  Accuracy   : {acc:.4f}   (paper: 0.989)")
        print(f"  Macro-F1   : {macro_f1:.4f}   (paper: ~0.937 computed)")
        print(f"  OVR AUC    : {auc_5:.4f}   (paper: 0.985)")
        print_per_class(model, X_test, y_test)
        print(f"\n  ── Binary anomaly (N vs {{L,R,V,A}}) ────────────────")
        print(f"  Precision  : {anom_p:.4f}   (paper: 0.931)")
        print(f"  Recall     : {anom_r:.4f}   (paper: 0.925)")
        print(f"  F1         : {anom_f1:.4f}   (paper: 0.928)")
        print(f"  AUC        : {anom_auc:.4f}   (paper: 0.985)")

        save_confusion_matrix(model, X_test, y_test, f"{split_name}_5class")

# %%

def _lgbm_with_paper_params(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> dict:
    """Train LGBM (paper params + SMOTE) and return binary anomaly + accuracy metrics."""
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    model = LGBMClassifier(
        **PAPER_PARAMS, subsample_freq=1,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    model.fit(X_res, y_res)
    y_pred     = model.predict(X_te)
    y_bin      = to_binary(y_te)
    y_bin_pred = to_binary(np.asarray(y_pred))
    return {
        "acc":      float(accuracy_score(y_te, y_pred)),
        "anom_rec": float(recall_score(y_bin, y_bin_pred, zero_division=0)),
        "anom_f1":  float(f1_score(y_bin, y_bin_pred, zero_division=0)),
        "auc":      binary_auc(model, X_te, y_bin),
    }


def run_ablation() -> None:
    print(f"\n{'═'*62}")
    print("ABLATION A1→A4  (LightGBM, paper params, random split)")
    print("cf. clarityBase.txt §8.2")
    print(f"{'═'*62}")

    split_path = PROC_DIR / "split_random.npz"
    if not split_path.exists():
        print("  [skip] split_random.npz not found.")
        return

    data    = np.load(split_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    header = f"  {'Config':<30}{'#Feats':>7}{'Acc':>8}{'AnomRec':>10}{'AnomF1':>9}{'AUC':>8}"
    sep    = f"  {'─'*72}"
    print(f"\n{header}\n{sep}")

    for label, feat_slice in ABLATION_CONFIGS:
        n_feats = feat_slice.stop - feat_slice.start
        print(f"  {label:<30}{n_feats:>7}  training...", end="\r", flush=True)
        m = _lgbm_with_paper_params(
            X_train[:, feat_slice], y_train,
            X_test[:, feat_slice],  y_test,
        )
        print(f"  {label:<30}{n_feats:>7}{m['acc']:>8.3f}"
              f"{m['anom_rec']:>10.3f}{m['anom_f1']:>9.3f}{m['auc']:>8.3f}")

    print(sep)
    print(f"  Paper (§8.2):         6 feats  0.985   0.862    0.860   0.971")
    print(f"                       19 feats  0.987   0.905    0.901   0.979")
    print(f"                       31 feats  0.988   0.919    0.921   0.983")
    print(f"                       40 feats  0.989   0.925    0.928   0.985")

# %%

def _gbc_binary_f1(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
) -> float:
    """Train GBC (default params, original CLARITY-AI baseline) + SMOTE → binary F1."""
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_tr, y_tr)
    model        = GradientBoostingClassifier(random_state=SEED)
    model.fit(X_res, y_res)
    y_bin      = to_binary(y_te)
    y_bin_pred = to_binary(np.asarray(model.predict(X_te)))
    return float(f1_score(y_bin, y_bin_pred, zero_division=0))


def run_synergy() -> None:
    print(f"\n{'═'*62}")
    print("MODEL × FEATURE-SET SYNERGY  (binary anomaly F1)")
    print("cf. clarityBase.txt §8.3")
    print(f"{'═'*62}")

    split_path = PROC_DIR / "split_random.npz"
    if not split_path.exists():
        print("  [skip] split_random.npz not found.")
        return

    data    = np.load(split_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test  = data["X_test"]
    y_test  = data["y_test"]

    s6  = slice(0, 6)
    s40 = slice(0, 40)

    print("\n  Training 4 models ...", flush=True)
    gbc_6   = _gbc_binary_f1(X_train[:, s6],  y_train, X_test[:, s6],  y_test)
    gbc_40  = _gbc_binary_f1(X_train[:, s40], y_train, X_test[:, s40], y_test)
    lgbm_6  = _lgbm_with_paper_params(
        X_train[:, s6],  y_train, X_test[:, s6],  y_test)["anom_f1"]
    lgbm_40 = _lgbm_with_paper_params(
        X_train[:, s40], y_train, X_test[:, s40], y_test)["anom_f1"]

    sep = f"  {'─'*44}"
    print(f"\n  {'Model':<22}{'6 features':>12}{'40 features':>12}")
    print(sep)
    print(f"  {'GBC (original)':<22}{gbc_6:>12.3f}{gbc_40:>12.3f}")
    print(f"  {'LightGBM':<22}{lgbm_6:>12.3f}{lgbm_40:>12.3f}")
    print(sep)
    print(f"  Paper (§8.3):")
    print(f"  {'GBC (original)':<22}{'0.841':>12}{'0.912':>12}")
    print(f"  {'LightGBM':<22}{'0.860':>12}{'0.928':>12}")

# %%

if __name__ == "__main__":
    run_main_eval()
    run_ablation()
    run_synergy()
    print("\nDone.")
