# %% [markdown]

# # MIT-BIH — Binary LightGBM Training (Normal vs Anomaly)
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Trains a dedicated binary classifier: 0 = Normal, 1 = Anomaly {L, R, V, A}.
#
# A dedicated binary model should outperform the 5-class model collapsed to
# binary (the current DS1/DS2 binary F1 is ~0.34) because:
#   - objective='binary' uses log-loss directly, not multiclass softmax
#   - The decision boundary simplifies: "is this beat abnormal?" rather than
#     distinguishing LBBB from RBBB from PVC, which fails cross-patient
#   - SMOTE balances N vs {L+R+V+A} directly (3:1 ratio in DS1)
#
# Starting point: clarityBase.txt §6 paper hyperparameters seeded as trial 0.
# Extended Optuna search adds min_child_samples, reg_alpha, reg_lambda for
# better cross-patient regularisation (key improvement over train.py).
#
# GPU probe: cuda → gpu → cpu (set LGBM_DEVICE env-var to override).
#
# Input:  datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz
#
# Output: models/lgbm_binary_random.pkl
#         models/lgbm_binary_ds1ds2.pkl
#         models/lgbm_binary_random_optuna.json
#         models/lgbm_binary_ds1ds2_optuna.json
#
# Checkpoints:
#         models/checkpoints/lgbm_binary_random_optuna.db
#         models/checkpoints/lgbm_binary_ds1ds2_optuna.db

# %%

import contextlib
import json
import os
import pickle
import subprocess
from pathlib import Path

import numpy as np
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

# %%

_HERE         = Path(__file__).resolve().parent        # .../source/binaryLightGBM/
FYP_ROOT      = _HERE.parent.parent                    # .../fyp/
PROC_DIR      = FYP_ROOT / "datasets" / "processed"
MODEL_DIR     = FYP_ROOT / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# %%

N_TRIALS   = 50
N_CV_FOLDS = 5
SEED       = 42

# Paper's reported hyperparameters (clarityBase.txt §6).
# These are seeded as trial 0 so they are always evaluated even if
# Optuna finds better parameters elsewhere in the search space.
PAPER_PARAMS = {
    "n_estimators":     1450,
    "max_depth":        11,
    "learning_rate":    0.05,
    "num_leaves":       38,
    "subsample":        0.85,
    "colsample_bytree": 0.75,
}

# Extended starting point for binary: adds LightGBM defaults for the
# regularisation params that are new to this search space.
# reg_alpha/reg_lambda use 1e-8 (not 0.0) because suggest_float(..., log=True)
# requires strictly positive values; 0.0 would cause trial 0 to FAIL.
PAPER_PARAMS_BINARY = {
    **PAPER_PARAMS,
    "min_child_samples": 20,
    "reg_alpha":         1e-8,
    "reg_lambda":        1e-8,
}

# Custom scorer: F1 for the anomaly class (pos_label=1).
# Used as the Optuna CV objective — directly optimises what we care about.
ANOMALY_F1_SCORER = make_scorer(f1_score, pos_label=1, average="binary", zero_division=0)

# %%


@contextlib.contextmanager
def _suppress_c_stderr():
    """Redirect C-level fd-2 to /dev/null temporarily.

    LightGBM prints [Fatal] messages directly via C, bypassing Python's
    sys.stderr, so only an fd-level redirect can suppress them.
    """
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
    """
    Return 'cuda', 'gpu', or 'cpu' — whichever LightGBM can actually use.

    Override by setting LGBM_DEVICE=cuda|gpu|cpu in the environment.
    WSL2 note: the standard PyPI wheel is NOT compiled with CUDA support.
    Run source/setup_gpu.sh once to rebuild LightGBM with CUDA enabled.
    """
    forced = os.environ.get("LGBM_DEVICE", "").lower()
    if forced in ( "gpu", "cuda","cpu"):
        print(f"[device] LGBM_DEVICE override → '{forced}'")
        return forced

    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
    except Exception:
        print("[device] No NVIDIA GPU detected — using CPU.")
        return "cpu"

    rng = np.random.default_rng(0)
    X_t = rng.random((200, 10)).astype(np.float32)
    y_t = (X_t[:, 0] > 0.5).astype(int)
    failures: dict[str, str] = {}

    for dev in ("gpu", "cuda"):
        try:
            with _suppress_c_stderr():
                LGBMClassifier(device=dev, n_estimators=5, verbose=-1,
                               objective="binary").fit(X_t, y_t)
            print(f"[device] LightGBM will use device='{dev}'")
            return dev
        except Exception as exc:
            failures[dev] = str(exc).splitlines()[0]

    print("[device] GPU probe failed — falling back to CPU.")
    for dev, msg in failures.items():
        print(f"  [{dev}] {msg}")

    if any("CUDA Tree Learner was not enabled" in m for m in failures.values()):
        print("  → LightGBM was not built with CUDA. Run: source/setup_gpu.sh")
    elif any("No OpenCL device found" in m for m in failures.values()):
        print("  → No OpenCL runtime found. Run: source/setup_gpu.sh")

    return "cpu"


DEVICE = _probe_device()


# %%


def to_binary(y: np.ndarray) -> np.ndarray:
    """Convert 5-class labels to binary: N → 0, {L,R,V,A} → 1."""
    return np.where(y == "N", 0, 1).astype(int)


# %%


def make_objective(X_train: np.ndarray, y_train: np.ndarray):
    """
    Build the Optuna objective for binary classification.

    SMOTE is inside the Pipeline so it is applied independently to each
    training fold, preventing synthetic samples from leaking into validation.
    The objective returns mean anomaly-class F1 over N_CV_FOLDS folds.

    Extended search space vs 5-class train.py:
      min_child_samples — prevents leaves from fitting to <N samples;
                          larger values regularise more, helping DS1/DS2 gen.
      reg_alpha, reg_lambda — L1/L2 regularisation on leaf weights.
    """
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=SEED)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 500, 2000),
            "max_depth":         trial.suggest_int("max_depth", 5, 15),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 20, 50),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # Regularisation params new to the binary search:
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-8, 5.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-8, 5.0, log=True),
        }

        pipeline = Pipeline([
            ("smote", SMOTE(random_state=SEED)),
            ("clf", LGBMClassifier(
                **params,
                objective="binary",
                subsample_freq=1,
                device=DEVICE,
                random_state=SEED,
                n_jobs=1 if DEVICE != "cpu" else -1,
                verbose=-1,
            )),
        ])

        scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=ANOMALY_F1_SCORER,
            n_jobs=1,
        )
        return float(scores.mean())

    return objective


# %%


def get_or_create_study(split_name: str) -> optuna.Study:
    """Create or resume an Optuna study backed by SQLite."""
    db_path = CHECKPOINT_DIR / f"lgbm_binary_{split_name}_optuna.db"
    storage_url = f"sqlite:///{db_path.resolve()}"
    study_name  = f"lgbm_binary_{split_name}"

    print(f"\n  Optuna checkpoint: {db_path}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
    )

    # Mark any trials stuck in RUNNING state (from a previous crash) as FAIL
    # so they don't sit indefinitely in the checkpoint and confuse reporting.
    stuck = [t for t in study.trials
             if t.state == optuna.trial.TrialState.RUNNING]
    if stuck:
        print(f"  Cleaning up {len(stuck)} stuck RUNNING trial(s) → FAIL")
        for t in stuck:
            try:
                study._storage.set_trial_state_values(
                    t._trial_id, state=optuna.trial.TrialState.FAIL
                )
            except Exception:
                pass  # best-effort; not fatal if it fails

    return study


# %%


def train_and_save(split_name: str, split_path: Path) -> None:

    print(f"\n{'═'*62}")
    print(f"Binary split : {split_name}  ({split_path.name})")
    print(f"{'═'*62}")

    pkl_path  = MODEL_DIR / f"lgbm_binary_{split_name}.pkl"
    meta_path = MODEL_DIR / f"lgbm_binary_{split_name}_optuna.json"

    if pkl_path.exists() and meta_path.exists():
        print("\n  ✓ Final binary model already exists — skipping.")
        print(f"    Model : {pkl_path}")
        print(f"    Meta  : {meta_path}")
        return

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    data      = np.load(split_path, allow_pickle=True)
    X_train   = data["X_train"]
    y_train_5 = data["y_train"]
    y_train   = to_binary(y_train_5)

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    print(f"\n  Train  : {len(y_train):,} beats × {X_train.shape[1]} features")
    print(f"  Binary : N(0)={n_neg:,}  Anomaly(1)={n_pos:,}  ratio={n_neg/n_pos:.2f}:1")

    study = get_or_create_study(split_name)

    existing   = len(study.trials)
    completed  = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    print(f"\n  Existing trials : {existing}")
    print(f"  Completed trials: {completed}")
    print(f"  Target trials   : {N_TRIALS}")

    if existing == 0:
        print("\n  New study — enqueuing paper hyperparameters as trial 0.")
        study.enqueue_trial(PAPER_PARAMS_BINARY)
    else:
        print("\n  ✓ Existing checkpoint found — resuming Optuna study.")

    remaining = max(0, N_TRIALS - completed)

    if remaining > 0:
        print(f"\n  Running {remaining} remaining Optuna trials "
              f"({N_CV_FOLDS}-fold CV, anomaly-class F1) ...")
        print("  Each completed trial is saved to SQLite automatically.")
        print("  You can safely stop and rerun to resume.\n")

        try:
            study.optimize(
                make_objective(X_train, y_train),
                n_trials=remaining,
                show_progress_bar=True,
            )
        except KeyboardInterrupt:
            print("\n\n  Training interrupted. Completed trials are saved.")
            print("  Rerun to continue from checkpoint.")
            raise
        except Exception:
            print("\n\n  Training error. Completed trials are still saved.")
            raise
    else:
        print("\n  ✓ Target trials already completed — skipping search.")

    completed_trials = [t for t in study.trials
                        if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        raise RuntimeError("No Optuna trials completed successfully.")

    best_params = study.best_params
    best_score  = study.best_value

    print(f"\n  Best CV anomaly-F1 : {best_score:.4f}")
    print("  Best params vs paper:")
    for k in PAPER_PARAMS:
        print(f"    {k:<22}: {best_params[k]}   (paper: {PAPER_PARAMS[k]})")
    for k in ("min_child_samples", "reg_alpha", "reg_lambda"):
        print(f"    {k:<22}: {best_params[k]:.4g}   (new binary param)")

    # Save search checkpoint before final fit
    search_ckpt = CHECKPOINT_DIR / f"lgbm_binary_{split_name}_search_complete.json"
    with open(search_ckpt, "w") as f:
        json.dump({
            "split":             split_name,
            "best_params":       best_params,
            "cv_score":          best_score,
            "completed_trials":  len(completed_trials),
            "target_trials":     N_TRIALS,
            "n_cv_folds":        N_CV_FOLDS,
            "paper_params":      PAPER_PARAMS,
            "device":            DEVICE,
            "objective":         "binary_anomaly_f1",
        }, f, indent=2)
    print(f"\n  ✓ Search checkpoint → {search_ckpt}")

    # Final refit on all training data with SMOTE
    print("\n  Refitting on full training set with SMOTE ...")
    X_res, y_res = SMOTE(random_state=SEED).fit_resample(X_train, y_train)
    res_n = int((y_res == 0).sum()); res_p = int((y_res == 1).sum())
    print(f"  After SMOTE: {len(y_res):,} beats — N={res_n:,}  Anomaly={res_p:,}")

    final_model = LGBMClassifier(
        **best_params,
        objective="binary",
        subsample_freq=1,
        device=DEVICE,
        random_state=SEED,
        n_jobs=1 if DEVICE != "cpu" else -1,
        verbose=-1,
    )
    final_model.fit(X_res, y_res)

    with open(pkl_path, "wb") as f:
        pickle.dump({
            "model":       final_model,
            "best_params": best_params,
            "cv_score":    best_score,
            "split":       split_name,
            "task":        "binary",
            "label_map":   {"N": 0, "L": 1, "R": 1, "V": 1, "A": 1},
        }, f)

    with open(meta_path, "w") as f:
        json.dump({
            "split":            split_name,
            "task":             "binary",
            "best_params":      best_params,
            "cv_score":         best_score,
            "n_trials":         len(completed_trials),
            "n_cv_folds":       N_CV_FOLDS,
            "paper_params":     PAPER_PARAMS,
            "device":           DEVICE,
            "objective":        "binary_anomaly_f1",
        }, f, indent=2)

    print(f"\n  ✓ Model → {pkl_path}")
    print(f"  ✓ Meta  → {meta_path}")
    print(f"  ✓ DB    → {CHECKPOINT_DIR / f'lgbm_binary_{split_name}_optuna.db'}")


# %%


if __name__ == "__main__":
    print("\nMIT-BIH Binary LightGBM Training (N vs Anomaly)")
    print("=" * 62)
    print(f"Device        : {DEVICE}")
    print(f"Target trials : {N_TRIALS}")
    print(f"CV folds      : {N_CV_FOLDS}")
    print(f"Checkpoints   : {CHECKPOINT_DIR}")
    print(f"Objective     : anomaly-class F1 (pos_label=1)")

    for name in ("random", "ds1ds2"):
        train_and_save(name, PROC_DIR / f"split_{name}.npz")

    print("\n" + "═" * 62)
    print("Done.")
    print("═" * 62)


"""
does the train binary file run on ds1 ds2 split ?                                   

● Yes. Looking at the __main__ block at the bottom of train_binary.py:
                                                                
  for name in ("random", "ds1ds2"):
      train_and_save(name, PROC_DIR / f"split_{name}.npz")

  It runs both splits sequentially — random first, then DS1/DS2. Each gets its own
  Optuna study, checkpoint file, and saved model.

  If you only want to run DS1/DS2 (skip random), you can just change that line to:

  for name in ("ds1ds2",):
      train_and_save(name, PROC_DIR / f"split_{name}.npz")
"""