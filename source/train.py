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
# Savepoint / checkpoint support:
#   - Optuna studies are stored in SQLite
#   - Every completed trial is automatically persisted
#   - Interrupted runs resume from the last completed trial
#   - N_TRIALS means TOTAL desired trials, not additional trials
#   - Completed split models are detected and skipped on rerun
#
# Input:  datasets/processed/split_random.npz
#         datasets/processed/split_ds1ds2.npz
#
# Output: models/lgbm_random.pkl
#         models/lgbm_ds1ds2.pkl
#         models/lgbm_random_optuna.json
#         models/lgbm_ds1ds2_optuna.json
#
# Checkpoints:
#         models/checkpoints/lgbm_random_optuna.db
#         models/checkpoints/lgbm_ds1ds2_optuna.db

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

_HERE = Path(__file__).resolve().parent

PROC_DIR = _HERE.parent / "datasets" / "processed"
MODEL_DIR = _HERE.parent / "models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def _probe_device() -> str:
    """
    Return 'cuda', 'gpu', or 'cpu' — whichever LightGBM can actually use.

    Override by setting the environment variable LGBM_DEVICE=cuda|gpu|cpu
    (useful in CI or when the probe is slow).

    WSL2 note: the standard PyPI wheel is NOT compiled with CUDA support.
    Run `source/setup_gpu.sh` once to rebuild LightGBM with CUDA enabled.
    """
    import os

    forced = os.environ.get("LGBM_DEVICE", "").lower()

    if forced in ("cuda", "gpu", "cpu"):
        print(f"[device] LGBM_DEVICE override → '{forced}'")
        return forced

    try:
        subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            check=True,
        )
    except Exception:
        print("[device] No NVIDIA GPU detected — using CPU.")
        return "cpu"

    rng = np.random.default_rng(0)

    X_t = rng.random((200, 10)).astype(np.float32)
    y_t = (X_t[:, 0] > 0.5).astype(int)

    failures: dict[str, str] = {}

    for dev in ("cuda", "gpu"):
        try:
            LGBMClassifier(
                device=dev,
                n_estimators=5,
                verbose=-1,
            ).fit(X_t, y_t)

            print(f"[device] LightGBM will use device='{dev}'")
            return dev

        except Exception as exc:
            failures[dev] = str(exc).splitlines()[0]

    print("[device] GPU probe failed — falling back to CPU.")

    for dev, msg in failures.items():
        print(f"  [{dev}] {msg}")

    # WSL2: libcuda.so is present but the PyPI wheel may lack CUDA support.
    if any(
        "CUDA Tree Learner was not enabled" in m
        for m in failures.values()
    ):
        print(
            "  → LightGBM was not built with CUDA. "
            "Run: source/setup_gpu.sh"
        )

    elif any(
        "No OpenCL device found" in m
        for m in failures.values()
    ):
        print(
            "  → No OpenCL runtime found. "
            "Run: source/setup_gpu.sh"
        )

    return "cpu"


DEVICE = _probe_device()

N_TRIALS = 50
N_CV_FOLDS = 5
SEED = 42


# Paper's reported final hyperparameters
# (clarityBase.txt §6)
#
# Seeded as trial 0 so the paper's configuration is always evaluated
# even if Optuna finds better parameters.
PAPER_PARAMS = {
    "n_estimators": 1450,
    "max_depth": 11,
    "learning_rate": 0.05,
    "num_leaves": 38,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
}

# %%


def make_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    """
    Create the Optuna objective.

    SMOTE is kept INSIDE the pipeline so that during cross-validation
    it is fitted independently on each training fold.

    This prevents information leakage from validation folds.
    """

    cv = StratifiedKFold(
        n_splits=N_CV_FOLDS,
        shuffle=True,
        random_state=SEED,
    )

    def objective(trial: optuna.Trial) -> float:

        params = {
            "n_estimators": trial.suggest_int(
                "n_estimators",
                500,
                2000,
            ),

            "max_depth": trial.suggest_int(
                "max_depth",
                5,
                15,
            ),

            "learning_rate": trial.suggest_float(
                "learning_rate",
                0.01,
                0.2,
                log=True,
            ),

            "num_leaves": trial.suggest_int(
                "num_leaves",
                20,
                50,
            ),

            "subsample": trial.suggest_float(
                "subsample",
                0.6,
                1.0,
            ),

            "colsample_bytree": trial.suggest_float(
                "colsample_bytree",
                0.6,
                1.0,
            ),
        }

        pipeline = Pipeline([
            (
                "smote",
                SMOTE(
                    random_state=SEED,
                ),
            ),

            (
                "clf",
                LGBMClassifier(
                    **params,

                    # Required for subsample to actually take effect
                    subsample_freq=1,

                    device=DEVICE,
                    random_state=SEED,

                    # GPU handles its own parallelism.
                    n_jobs=1 if DEVICE != "cpu" else -1,

                    verbose=-1,
                ),
            ),
        ])

        # n_jobs=1 here avoids nested parallelism between
        # CV and LightGBM threads.
        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="f1_macro",
            n_jobs=1,
        )

        return float(scores.mean())

    return objective


# %%


def get_or_create_study(
    split_name: str,
) -> optuna.Study:
    """
    Create or resume an Optuna study.

    Optuna's SQLite storage acts as our savepoint system.

    Each completed trial is committed to disk automatically.
    Therefore, if training is interrupted, rerunning the script
    resumes from the completed trials.
    """

    db_path = (
        CHECKPOINT_DIR
        / f"lgbm_{split_name}_optuna.db"
    )

    # Absolute SQLite path.
    storage_url = f"sqlite:///{db_path.resolve()}"

    study_name = f"lgbm_{split_name}"

    print(f"\n  Optuna checkpoint:")
    print(f"    {db_path}")

    study = optuna.create_study(
        direction="maximize",

        sampler=optuna.samplers.TPESampler(
            seed=SEED,
        ),

        study_name=study_name,

        storage=storage_url,

        # This is the important part:
        # existing DB/study → resume
        # missing DB/study  → create
        load_if_exists=True,
    )

    return study


# %%


def train_and_save(
    split_name: str,
    split_path: Path,
) -> None:

    print(f"\n{'═' * 62}")
    print(
        f"Split : {split_name}  "
        f"({split_path.name})"
    )
    print(f"{'═' * 62}")

    # ----------------------------------------------------------
    # Output paths
    # ----------------------------------------------------------

    pkl_path = (
        MODEL_DIR
        / f"lgbm_{split_name}.pkl"
    )

    meta_path = (
        MODEL_DIR
        / f"lgbm_{split_name}_optuna.json"
    )

    # ----------------------------------------------------------
    # If final outputs already exist, this split is complete.
    # ----------------------------------------------------------

    if pkl_path.exists() and meta_path.exists():

        print("\n  ✓ Final model already exists.")
        print(f"    Model: {pkl_path}")
        print(f"    Meta : {meta_path}")
        print("  → Skipping this split.")

        return

    # ----------------------------------------------------------
    # Load training split
    # ----------------------------------------------------------

    if not split_path.exists():
        raise FileNotFoundError(
            f"Training split not found: {split_path}"
        )

    data = np.load(
        split_path,
        allow_pickle=True,
    )

    X_train = data["X_train"]
    y_train = data["y_train"]

    print(
        f"  Train  : "
        f"{X_train.shape[0]:,} beats × "
        f"{X_train.shape[1]} features"
    )

    dist = {
        cls: int((y_train == cls).sum())
        for cls in sorted(set(y_train))
    }

    print(f"  Classes: {dist}")

    # ----------------------------------------------------------
    # Create / resume Optuna study
    # ----------------------------------------------------------

    study = get_or_create_study(
        split_name
    )

    # ----------------------------------------------------------
    # Determine how many trials already exist
    # ----------------------------------------------------------

    existing_trials = len(study.trials)

    completed_trials = sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )

    print(
        f"\n  Existing trials : "
        f"{existing_trials}"
    )

    print(
        f"  Completed trials: "
        f"{completed_trials}"
    )

    print(
        f"  Target trials   : "
        f"{N_TRIALS}"
    )

    # ----------------------------------------------------------
    # Seed paper parameters ONLY for a brand-new study
    # ----------------------------------------------------------

    if existing_trials == 0:

        print(
            "\n  New study detected."
        )

        print(
            "  Enqueuing paper hyperparameters "
            "as trial 0."
        )

        study.enqueue_trial(
            PAPER_PARAMS
        )

    else:

        print(
            "\n  ✓ Existing checkpoint found."
        )

        print(
            "  Resuming Optuna study..."
        )

    # ----------------------------------------------------------
    # Calculate REMAINING trials
    #
    # Important:
    # N_TRIALS means the total desired number of trials.
    #
    # Example:
    #   target = 100
    #   completed = 37
    #
    #   rerun → only 63 more trials
    # ----------------------------------------------------------

    remaining_trials = max(
        0,
        N_TRIALS - completed_trials,
    )

    if remaining_trials > 0:

        print(
            f"\n  Running {remaining_trials} remaining "
            f"Optuna trials "
            f"({N_CV_FOLDS}-fold CV, macro-F1) ..."
        )

        print(
            "  Each completed trial is automatically "
            "saved to SQLite."
        )

        print(
            "  You can safely stop the script and "
            "rerun it later.\n"
        )

        try:

            study.optimize(
                make_objective(
                    X_train,
                    y_train,
                ),

                n_trials=remaining_trials,

                show_progress_bar=True,
            )

        except KeyboardInterrupt:

            print(
                "\n\n  Training interrupted by user."
            )

            print(
                "  ✓ Completed Optuna trials are saved."
            )

            print(
                "  Rerun this script to continue "
                "from the checkpoint."
            )

            raise

        except Exception:

            print(
                "\n\n  Training stopped because of an error."
            )

            print(
                "  ✓ Completed Optuna trials are still saved."
            )

            print(
                "  Fix the error and rerun this script "
                "to resume."
            )

            raise

    else:

        print(
            "\n  ✓ Target number of Optuna trials "
            "already completed."
        )

        print(
            "  Skipping hyperparameter search."
        )

    # ----------------------------------------------------------
    # Verify that at least one trial succeeded
    # ----------------------------------------------------------

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state
        == optuna.trial.TrialState.COMPLETE
    ]

    if not completed_trials:

        raise RuntimeError(
            "No Optuna trials completed successfully."
        )

    # ----------------------------------------------------------
    # Retrieve best result
    # ----------------------------------------------------------

    best_params = study.best_params
    best_score = study.best_value

    print(
        f"\n  Best CV macro-F1 : "
        f"{best_score:.4f}"
    )

    print(
        "  Best params vs paper:"
    )

    for k in PAPER_PARAMS:

        print(
            f"    {k:<20s}: "
            f"{best_params[k]}   "
            f"(paper: {PAPER_PARAMS[k]})"
        )

    # ----------------------------------------------------------
    # Save metadata BEFORE final refit
    #
    # This gives us an additional human-readable savepoint.
    # If final fitting fails, the Optuna result is still recorded.
    # ----------------------------------------------------------

    search_meta_path = (
        CHECKPOINT_DIR
        / f"lgbm_{split_name}_search_complete.json"
    )

    with open(
        search_meta_path,
        "w",
    ) as f:

        json.dump(
            {
                "split": split_name,
                "best_params": best_params,
                "cv_score": best_score,
                "completed_trials": len(
                    completed_trials
                ),
                "target_trials": N_TRIALS,
                "n_cv_folds": N_CV_FOLDS,
                "paper_params": PAPER_PARAMS,
                "device": DEVICE,
                "optuna_db": str(
                    CHECKPOINT_DIR
                    / f"lgbm_{split_name}_optuna.db"
                ),
            },
            f,
            indent=2,
        )

    print(
        f"\n  ✓ Search checkpoint saved → "
        f"{search_meta_path}"
    )

    # ----------------------------------------------------------
    # Final SMOTE
    # ----------------------------------------------------------

    print(
        "\n  Refitting on full training set "
        "with SMOTE ..."
    )

    X_res, y_res = SMOTE(
        random_state=SEED
    ).fit_resample(
        X_train,
        y_train,
    )

    res_dist = {
        cls: int((y_res == cls).sum())
        for cls in sorted(set(y_res))
    }

    print(
        f"  After SMOTE: "
        f"{X_res.shape[0]:,} beats  "
        f"{res_dist}"
    )

    # ----------------------------------------------------------
    # Final LightGBM model
    # ----------------------------------------------------------

    model = LGBMClassifier(
        **best_params,

        subsample_freq=1,

        device=DEVICE,

        random_state=SEED,

        n_jobs=(
            1
            if DEVICE != "cpu"
            else -1
        ),

        verbose=-1,
    )

    model.fit(
        X_res,
        y_res,
    )

    # ----------------------------------------------------------
    # Save final model
    # ----------------------------------------------------------

    with open(
        pkl_path,
        "wb",
    ) as f:

        pickle.dump(
            {
                "model": model,
                "best_params": best_params,
                "cv_score": best_score,
                "split": split_name,
            },
            f,
        )

    # ----------------------------------------------------------
    # Save final metadata
    # ----------------------------------------------------------

    with open(
        meta_path,
        "w",
    ) as f:

        json.dump(
            {
                "split": split_name,
                "best_params": best_params,
                "cv_score": best_score,
                "n_trials": len(
                    completed_trials
                ),
                "n_cv_folds": N_CV_FOLDS,
                "paper_params": PAPER_PARAMS,
                "device": DEVICE,
            },
            f,
            indent=2,
        )

    print(
        f"\n  ✓ Saved model → {pkl_path}"
    )

    print(
        f"  ✓ Saved meta  → {meta_path}"
    )

    print(
        f"  ✓ Optuna DB   → "
        f"{CHECKPOINT_DIR / f'lgbm_{split_name}_optuna.db'}"
    )


# %%


if __name__ == "__main__":

    print("\nMIT-BIH LightGBM Training")
    print("=" * 62)

    print(
        f"Device      : {DEVICE}"
    )

    print(
        f"Target trials: {N_TRIALS}"
    )

    print(
        f"CV folds      : {N_CV_FOLDS}"
    )

    print(
        f"Checkpoints   : {CHECKPOINT_DIR}"
    )

    for name in (
        "random",
        "ds1ds2",
    ):

        train_and_save(
            name,
            PROC_DIR
            / f"split_{name}.npz",
        )

    print(
        "\n"
        + "═" * 62
    )

    print(
        "Done."
    )

    print(
        "═" * 62
    )