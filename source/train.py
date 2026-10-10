# %% [markdown]

# # MIT-BIH — Step 7: Train
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Trains LightGBM with SMOTE and Optuna on both splits:
#   - Random stratified 80/20  (split_random.npz)
#   - Inter-patient DS1/DS2    (split_ds1ds2.npz)
#
# Key choices (clarityBase.txt §6, §9.2, §9.13):
#   - SMOTE inside imblearn Pipeline → applied per CV fold only, not before (§9.2)
#   - subsample_freq=1 required for subsample to take effect in LightGBM (§9.13)
#   - Optuna TPE sampler, objective = stratified 5-fold macro-F1
#   - Paper's reported final params (§6) seeded as trial 0
#   - Final model refit on all resampled training data
#
# Input:  datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz
# Output: models/lgbm_random.pkl
#         models/lgbm_ds1ds2.pkl
#         models/lgbm_random_optuna.json
#         models/lgbm_ds1ds2_optuna.json

# %%

import json
import pickle
import subprocess
from pathlib import Path

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%

_HERE     = Path(__file__).resolve().parent
PROC_DIR  = _HERE.parent / "datasets" / "processed"
MODEL_DIR = _HERE.parent / "models"


def _probe_device() -> str:
    """
    Return 'cuda', 'gpu', or 'cpu' — whichever LightGBM can actually use.
    Tries CUDA first (LightGBM >= 4.0 bundles CUDA support in the standard wheel),
    then OpenCL ('gpu'), then falls back to CPU.
    """
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except Exception:
        print("[device] No NVIDIA GPU detected — using CPU.")
        return "cpu"

    rng = np.random.default_rng(0)
    X_t = rng.random((200, 10)).astype(np.float32)
    y_t = (X_t[:, 0] > 0.5).astype(int)

    for dev in ("cuda", "gpu"):
        try:
            LGBMClassifier(device=dev, n_estimators=5, verbose=-1).fit(X_t, y_t)
            print(f"[device] LightGBM will use device='{dev}'")
            return dev
        except Exception:
            pass

    print("[device] GPU probe failed — falling back to CPU.")
    return "cpu"


DEVICE = _probe_device()

N_TRIALS   = 100
N_CV_FOLDS = 5
SEED       = 42

# Paper's reported final hyperparameters (clarityBase.txt §6) — seeded as trial 0
# so the paper's configuration is always evaluated even if Optuna finds better params.
PAPER_PARAMS = {
    "n_estimators":     1450,
    "max_depth":        11,
    "learning_rate":    0.05,
    "num_leaves":       38,
    "subsample":        0.85,
    "colsample_bytree": 0.75,
}

# %%

def make_objective(X_train: np.ndarray, y_train: np.ndarray):
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 500, 2000),
            "max_depth":        trial.suggest_int("max_depth", 5, 15),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":       trial.suggest_int("num_leaves", 20, 50),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        pipeline = Pipeline([
            ("smote", SMOTE(random_state=SEED)),
            ("clf",   LGBMClassifier(
                **params,
                subsample_freq=1,   # §9.13: required for subsample to take effect
                device=DEVICE,
                random_state=SEED,
                n_jobs=1 if DEVICE != "cpu" else -1,  # GPU handles its own parallelism
                verbose=-1,
            )),
        ])
        # n_jobs=1 here avoids nested parallelism between Optuna and LightGBM threads
        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv, scoring="f1_macro", n_jobs=1,
        )
        return float(scores.mean())

    return objective

# %%

def train_and_save(split_name: str, split_path: Path) -> None:
    print(f"\n{'═'*62}")
    print(f"Split : {split_name}  ({split_path.name})")
    print(f"{'═'*62}")

    data    = np.load(split_path, allow_pickle=True)
    X_train = data["X_train"]
    y_train = data["y_train"]

    print(f"  Train  : {X_train.shape[0]:,} beats × {X_train.shape[1]} features")
    dist = {cls: int((y_train == cls).sum()) for cls in sorted(set(y_train))}
    print(f"  Classes: {dist}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=f"lgbm_{split_name}",
    )
    study.enqueue_trial(PAPER_PARAMS)

    print(f"\n  Running {N_TRIALS} Optuna trials ({N_CV_FOLDS}-fold CV, macro-F1) ...")
    study.optimize(
        make_objective(X_train, y_train),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_score  = study.best_value
    print(f"\n  Best CV macro-F1 : {best_score:.4f}")
    print(f"  Best params vs paper:")
    for k in PAPER_PARAMS:
        print(f"    {k:<20s}: {best_params[k]}   (paper: {PAPER_PARAMS[k]})")

    # Refit on the full training set (SMOTE applied once to all of training, not per fold)
    print("\n  Refitting on full training set with SMOTE ...")
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    res_dist = {cls: int((y_res == cls).sum()) for cls in sorted(set(y_res))}
    print(f"  After SMOTE: {X_res.shape[0]:,} beats  {res_dist}")

    model = LGBMClassifier(
        **best_params,
        subsample_freq=1,
        device=DEVICE,
        random_state=SEED,
        n_jobs=1 if DEVICE != "cpu" else -1,
        verbose=-1,
    )
    model.fit(X_res, y_res)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    pkl_path  = MODEL_DIR / f"lgbm_{split_name}.pkl"
    meta_path = MODEL_DIR / f"lgbm_{split_name}_optuna.json"

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":       model,
            "best_params": best_params,
            "cv_score":    best_score,
            "split":       split_name,
        }, f)

    with open(meta_path, "w") as f:
        json.dump({
            "split":        split_name,
            "best_params":  best_params,
            "cv_score":     best_score,
            "n_trials":     N_TRIALS,
            "n_cv_folds":   N_CV_FOLDS,
            "paper_params": PAPER_PARAMS,
        }, f, indent=2)

    print(f"\n  Saved model → {pkl_path}")
    print(f"  Saved meta  → {meta_path}")

# %%

if __name__ == "__main__":
    for name in ("random", "ds1ds2"):
        train_and_save(name, PROC_DIR / f"split_{name}.npz")
    print("\nDone.")
