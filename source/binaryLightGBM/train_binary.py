# %% [markdown]

# # MIT-BIH — Binary LightGBM Training (Normal vs Anomaly)
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Trains a dedicated binary classifier: 0 = Normal, 1 = Anomaly {L, R, V, A}.
#
# Revision 2026-11-18 (lgbm-binary agent):
#   * CLI: --split, --cv {stratified,grouped}, --n-trials, --threads, --objective
#   * --cv grouped = StratifiedGroupKFold by record (patient). SMOTE is applied
#     inside each training fold only. This is the default for ds1ds2 because
#     beat-level folds leak patients (CV F1 0.99 vs DS2 0.4 in the 5-class run).
#   * Out-of-fold (OOF) probabilities are pooled; the trial objective can be
#     F1@0.5 (fold mean, the old objective), the best-threshold OOF F1, or OOF AUC.
#   * Decision threshold is chosen on the pooled OOF probabilities of the best
#     params (training data only — DS2 is never touched here) and saved in the
#     model bundle as "threshold".
#   * Balancing strategy is part of the search: smote | scale_pos_weight | none.
#   * Wider regularisation ranges (min_child_samples up to 1000, shallow trees).
#
# Input:  datasets/processed/split_{random,ds1ds2}.npz
# Output: models/lgbm_binary_{tag}.pkl, models/lgbm_binary_{tag}_optuna.json
#         models/checkpoints/lgbm_binary_{tag}_optuna.db   (resumable)
#         models/checkpoints/lgbm_binary_{tag}_oof.npz     (OOF probs, best params)
#   tag = split for the default CV of that split (random→stratified,
#         ds1ds2→grouped), otherwise f"{split}_{cv}cv".
#
# Usage (from $ROOT):
#   .venv/bin/python source/binaryLightGBM/train_binary.py --split ds1ds2 --n-trials 80
#   .venv/bin/python source/binaryLightGBM/train_binary.py --split random --n-trials 25

# %%

import argparse
import contextlib
import json
import os
import pickle
import subprocess
import time
from pathlib import Path

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%

_HERE          = Path(__file__).resolve().parent        # .../source/binaryLightGBM/
FYP_ROOT       = _HERE.parent.parent                    # project root
PROC_DIR       = FYP_ROOT / "datasets" / "processed"
MODEL_DIR      = FYP_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# %%

N_TRIALS   = 50
N_CV_FOLDS = 5
SEED       = 42
N_THREADS  = 6

DEFAULT_CV = {"random": "stratified", "ds1ds2": "grouped"}

# Paper's reported hyperparameters (clarityBase.txt §6) — enqueued as trial 0.
PAPER_PARAMS = {
    "n_estimators":     1450,
    "max_depth":        11,
    "learning_rate":    0.05,
    "num_leaves":       38,
    "subsample":        0.85,
    "colsample_bytree": 0.75,
}

# reg_alpha/reg_lambda use 1e-8 (not 0) because log-uniform needs > 0.
PAPER_PARAMS_BINARY = {
    **PAPER_PARAMS,
    "min_child_samples": 20,
    "reg_alpha":         1e-8,
    "reg_lambda":        1e-8,
    "balance":           "smote",
}

# %%


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect C-level fd-2 to /dev/null temporarily."""
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    saved_fd   = os.dup(2)
    os.dup2(devnull_fd, 2)
    try:
        yield
    finally:
        os.dup2(saved_fd, 2)
        os.close(devnull_fd)
        os.close(saved_fd)


def _probe_device() -> str:
    """'cuda' | 'gpu' | 'cpu'. Override with LGBM_DEVICE. Defaults to cpu
    unless LGBM_DEVICE is set, because the data are small and the GPU is shared."""
    forced = os.environ.get("LGBM_DEVICE", "cpu").lower()
    if forced in ("gpu", "cuda", "cpu"):
        return forced
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except Exception:
        return "cpu"
    rng = np.random.default_rng(0)
    X_t = rng.random((200, 10)).astype(np.float32)
    y_t = (X_t[:, 0] > 0.5).astype(int)
    for dev in ("gpu", "cuda"):
        try:
            with _suppress_c_stderr():
                LGBMClassifier(device=dev, n_estimators=5, verbose=-1,
                               objective="binary").fit(X_t, y_t)
            return dev
        except Exception:
            pass
    return "cpu"


DEVICE = _probe_device()

# %%


def to_binary(y: np.ndarray) -> np.ndarray:
    """Convert 5-class labels to binary: N → 0, {L,R,V,A} → 1."""
    return np.where(y == "N", 0, 1).astype(int)


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    """Return (threshold, F1) maximising anomaly-class F1 over all cut points."""
    prec, rec, thr = precision_recall_curve(y_true, proba)
    f1 = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    f1 = f1[:-1]                       # last point has no threshold
    i = int(np.nanargmax(f1))
    return float(thr[i]), float(f1[i])


def make_cv(cv_kind: str):
    if cv_kind == "grouped":
        return StratifiedGroupKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)
    return StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)


def build_model(params: dict, n_threads: int, n_pos: int = 0, n_neg: int = 0) -> tuple[LGBMClassifier, bool]:
    """Return (model, use_smote) from a param dict containing 'balance'."""
    p = dict(params)
    balance = p.pop("balance", "smote")
    extra = {}
    if balance == "scale_pos_weight" and n_pos > 0:
        extra["scale_pos_weight"] = n_neg / n_pos
    model = LGBMClassifier(
        **p, **extra,
        objective="binary",
        subsample_freq=1,
        device=DEVICE,
        random_state=SEED,
        n_jobs=n_threads,
        verbose=-1,
    )
    return model, balance == "smote"


def fit_balanced(params: dict, X: np.ndarray, y: np.ndarray, n_threads: int) -> LGBMClassifier:
    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum())
    model, use_smote = build_model(params, n_threads, n_pos, n_neg)
    if use_smote:
        X, y = SMOTE(random_state=SEED).fit_resample(X, y)
    model.fit(X, y)
    return model


def oof_predict(params: dict, X: np.ndarray, y: np.ndarray, groups: np.ndarray,
                y5: np.ndarray, cv_kind: str, n_threads: int,
                trial: optuna.Trial | None = None) -> tuple[np.ndarray, list[float]]:
    """Pooled out-of-fold anomaly probabilities. SMOTE only inside train folds.
    Grouped CV stratifies on the 5-class label so L/R records spread over folds."""
    cv = make_cv(cv_kind)
    oof = np.zeros(len(y), dtype=float)
    fold_f1 = []
    split_iter = (cv.split(X, y5, groups) if cv_kind == "grouped" else cv.split(X, y))
    for k, (tr, va) in enumerate(split_iter):
        m = fit_balanced(params, X[tr], y[tr], n_threads)
        oof[va] = m.predict_proba(X[va])[:, 1]
        fold_f1.append(float(f1_score(y[va], (oof[va] >= 0.5).astype(int), zero_division=0)))
        if trial is not None:
            trial.report(float(np.mean(fold_f1)), k)
    return oof, fold_f1


# %%


def suggest_params(trial: optuna.Trial) -> dict:
    return {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 2000),
        "max_depth":         trial.suggest_int("max_depth", 3, 15),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 7, 63),
        "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 1000, log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "balance":           trial.suggest_categorical(
                                 "balance", ["smote", "scale_pos_weight", "none"]),
    }


def make_objective(X, y, groups, y5, cv_kind, objective_kind, n_threads):
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        t0 = time.time()
        oof, fold_f1 = oof_predict(params, X, y, groups, y5, cv_kind, n_threads)
        thr, f1_best = best_f1_threshold(y, oof)
        auc = float(roc_auc_score(y, oof))
        f1_05 = float(np.mean(fold_f1))
        trial.set_user_attr("f1_05_foldmean", f1_05)
        trial.set_user_attr("f1_05_fold", fold_f1)
        trial.set_user_attr("oof_f1_best", f1_best)
        trial.set_user_attr("oof_best_thr", thr)
        trial.set_user_attr("oof_auc", auc)
        trial.set_user_attr("secs", round(time.time() - t0, 1))
        print(f"  trial {trial.number:>3}: F1@0.5={f1_05:.4f}  OOF bestF1={f1_best:.4f}"
              f"@{thr:.3f}  AUC={auc:.4f}  ({time.time()-t0:.0f}s)  {params}", flush=True)
        return {"f1_05": f1_05, "f1_best": f1_best, "auc": auc}[objective_kind]
    return objective


# %%


def get_or_create_study(tag: str) -> optuna.Study:
    db_path = CHECKPOINT_DIR / f"lgbm_binary_{tag}_optuna.db"
    print(f"\n  Optuna checkpoint: {db_path}")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"lgbm_binary_{tag}",
        storage=f"sqlite:///{db_path.resolve()}",
        load_if_exists=True,
    )
    for t in study.trials:
        if t.state == optuna.trial.TrialState.RUNNING:
            try:
                study._storage.set_trial_state_values(
                    t._trial_id, state=optuna.trial.TrialState.FAIL)
            except Exception:
                pass
    return study


# %%


def train_and_save(split_name: str, cv_kind: str, n_trials: int,
                   objective_kind: str, n_threads: int, force_refit: bool,
                   tag_suffix: str = "") -> None:
    tag = split_name if cv_kind == DEFAULT_CV[split_name] else f"{split_name}_{cv_kind}cv"
    if tag_suffix:
        tag = f"{tag}_{tag_suffix}"
    split_path = PROC_DIR / f"split_{split_name}.npz"

    print(f"\n{'═'*62}")
    print(f"Binary split : {split_name}   CV: {cv_kind}   objective: {objective_kind}   tag: {tag}")
    print(f"{'═'*62}")

    pkl_path  = MODEL_DIR / f"lgbm_binary_{tag}.pkl"
    meta_path = MODEL_DIR / f"lgbm_binary_{tag}_optuna.json"
    oof_path  = CHECKPOINT_DIR / f"lgbm_binary_{tag}_oof.npz"

    data   = np.load(split_path, allow_pickle=True)
    X      = data["X_train"]
    y5     = data["y_train"]
    groups = data["record_ids_train"]
    y      = to_binary(y5)

    n_neg = int((y == 0).sum()); n_pos = int((y == 1).sum())
    print(f"  Train  : {len(y):,} beats × {X.shape[1]} features, {len(set(groups))} records")
    print(f"  Binary : N(0)={n_neg:,}  Anomaly(1)={n_pos:,}  ratio={n_neg/n_pos:.2f}:1")
    if cv_kind == "grouped":
        for k, (tr, va) in enumerate(make_cv("grouped").split(X, y5, groups)):
            recs = sorted(set(groups[va]))
            u, c = np.unique(y5[va], return_counts=True)
            print(f"  fold {k}: {len(recs)} records {recs}  {dict(zip(u.tolist(), c.tolist()))}")

    study = get_or_create_study(tag)
    if len(study.trials) == 0:
        print("  New study — enqueuing paper hyperparameters as trial 0.")
        study.enqueue_trial(PAPER_PARAMS_BINARY)

    n_done = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
    remaining = max(0, n_trials - n_done)
    print(f"  Completed trials: {n_done}   target: {n_trials}   remaining: {remaining}")
    if remaining > 0:
        study.optimize(make_objective(X, y, groups, y5, cv_kind, objective_kind, n_threads),
                       n_trials=remaining)

    best = study.best_trial
    best_params = dict(best.params)
    t0 = study.trials[0]
    print(f"\n  Trial 0 (paper params): obj={t0.value:.4f}  {t0.user_attrs}")
    print(f"  Best trial #{best.number}: obj={best.value:.4f}")
    print(f"    {best_params}")
    print(f"    {best.user_attrs}")

    if pkl_path.exists() and not force_refit:
        print(f"  ✓ {pkl_path.name} exists — skipping refit (use --force-refit).")
        return

    # OOF probabilities of the best params → decision threshold (train data only)
    print("\n  Recomputing OOF probabilities for best params ...")
    oof, fold_f1 = oof_predict(best_params, X, y, groups, y5, cv_kind, n_threads)
    thr, f1_thr = best_f1_threshold(y, oof)
    oof_auc = float(roc_auc_score(y, oof))
    print(f"  OOF: F1@0.5 fold-mean={np.mean(fold_f1):.4f}  best-thr={thr:.4f} "
          f"(F1={f1_thr:.4f})  AUC={oof_auc:.4f}")
    np.savez(oof_path, oof=oof, y=y, y5=y5, groups=groups, threshold=thr)

    print("  Refitting on full training set ...")
    final_model = fit_balanced(best_params, X, y, n_threads)

    bundle = {
        "model":        final_model,
        "best_params":  best_params,
        "cv_score":     best.value,
        "cv_kind":      cv_kind,
        "objective":    objective_kind,
        "threshold":    thr,
        "oof_f1_at_threshold": f1_thr,
        "oof_auc":      oof_auc,
        "oof_f1_05_foldmean": float(np.mean(fold_f1)),
        "split":        split_name,
        "tag":          tag,
        "task":         "binary",
        "label_map":    {"N": 0, "L": 1, "R": 1, "V": 1, "A": 1},
    }
    with open(pkl_path, "wb") as f:
        pickle.dump(bundle, f)
    with open(meta_path, "w") as f:
        json.dump({k: v for k, v in bundle.items() if k != "model"} | {
            "n_trials": n_done if remaining == 0 else n_trials,
            "n_cv_folds": N_CV_FOLDS, "paper_params": PAPER_PARAMS,
            "trial0_value": t0.value, "trial0_attrs": t0.user_attrs,
            "best_trial": best.number, "best_attrs": best.user_attrs,
            "device": DEVICE,
        }, f, indent=2, default=str)
    print(f"  ✓ Model → {pkl_path}\n  ✓ Meta  → {meta_path}\n  ✓ OOF   → {oof_path}")


# %%


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["random", "ds1ds2", "both"], default="both")
    ap.add_argument("--cv", choices=["default", "stratified", "grouped"], default="default")
    ap.add_argument("--n-trials", type=int, default=N_TRIALS)
    ap.add_argument("--objective", choices=["default", "f1_05", "f1_best", "auc"],
                    default="default",
                    help="default: f1_best for grouped CV, f1_05 for stratified")
    ap.add_argument("--threads", type=int, default=N_THREADS)
    ap.add_argument("--force-refit", action="store_true")
    ap.add_argument("--tag-suffix", default="",
                    help="extra model/study tag, e.g. 'auc', or 'paper' with --n-trials 1 "
                         "(= paper params only, trial 0)")
    a = ap.parse_args()

    print(f"Device: {DEVICE}   threads: {a.threads}")
    splits = ["random", "ds1ds2"] if a.split == "both" else [a.split]
    for s in splits:
        cv = DEFAULT_CV[s] if a.cv == "default" else a.cv
        obj = a.objective if a.objective != "default" else (
            "f1_best" if cv == "grouped" else "f1_05")
        train_and_save(s, cv, a.n_trials, obj, a.threads, a.force_refit, a.tag_suffix)
    print("Done.")
