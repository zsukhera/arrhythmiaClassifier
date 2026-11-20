"""
Experiment 4: Focal Loss
-------------------------
Problem: Categorical cross-entropy pushes the model to maximise accuracy on
         the dominant N class.  Class weights compensate but act multiplicatively
         on a linear loss signal.

Fix: Replace CCE with focal loss.  FL(p_t) = -(1-p_t)^gamma × log(p_t).
     The (1-p_t)^gamma term down-weights easy, well-classified examples
     (almost all N beats) and up-weights hard, poorly-classified ones (S beats).
     gamma=0 recovers plain CCE; gamma=2 is the standard choice from Lin et al. 2017.

Tried values: gamma ∈ {1, 2, 3}.  Best is selected by DS1-val macro-F1.
No class weights needed (focal loss handles imbalance directly).

Retrains from scratch.

Usage:
    python exp4_focal_loss.py --seeds 0 1 2
    python exp4_focal_loss.py --seeds 0 1 2 --gammas 1 2 3
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HasanML" / "paper3"))

from utils import load_config, load_data, load_kernel_mf, print_header, print_row, print_footer


# ---------------------------------------------------------------------------
# Focal loss implementation
# ---------------------------------------------------------------------------

def make_focal_loss(gamma: float = 2.0):
    """
    Multi-class focal loss for one-hot targets.
    Compatible with model.compile(loss=make_focal_loss(gamma=2)).
    """
    def focal_loss(y_true, y_pred):
        import tensorflow as tf
        eps   = 1e-7
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        ce     = -y_true * tf.math.log(y_pred)          # (batch, K)
        p_t    = tf.reduce_sum(y_true * y_pred, axis=-1, keepdims=True)  # (batch,1)
        weight = tf.pow(1.0 - p_t, gamma)               # (batch,1)
        loss   = weight * ce                             # (batch, K)
        return tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

    focal_loss.__name__ = f"focal_loss_gamma{gamma}"
    return focal_loss


# ---------------------------------------------------------------------------
# Training with focal loss
# ---------------------------------------------------------------------------

def _train_focal(cfg: dict, gamma: float, seeds: list[int],
                 train: dict, val: dict, test: dict,
                 kernel_mf: np.ndarray, verbose: bool = True) -> dict:
    import tensorflow as tf
    from tensorflow import keras

    from src.model import build_model, apply_mf_init, validate_param_count
    from src.train import _set_seeds, predict, MacroF1Callback
    from src.metrics import multiclass_metrics
    import time

    t_cfg = cfg["training"]
    nk        = int(cfg["input"]["nk"])
    n_classes = int(cfg["task"]["classes"])

    all_results = []

    for seed in seeds:
        _set_seeds(seed, deterministic=bool(cfg["compute"].get("deterministic", True)))

        model = build_model(
            nk=nk, n_channels=1, n_classes=n_classes,
            rr_dense_units=list(cfg["model"]["rr_dense_units"]),
            activation=str(cfg["model"]["activation"]),
            padding=str(cfg["model"]["padding"]),
            use_bias=bool(cfg["model"]["use_bias"]),
        )
        validate_param_count(model, nk, 1, n_classes)
        apply_mf_init(model, kernel_mf, frozen=False)

        # Compile with focal loss instead of CCE
        model.compile(
            optimizer=keras.optimizers.Adam(float(t_cfg["lr"])),
            loss=make_focal_loss(gamma),
            metrics=["accuracy"],
        )

        y_tr_oh  = keras.utils.to_categorical(train["y"], n_classes).astype(np.float32)
        y_val_oh = keras.utils.to_categorical(val["y"],   n_classes).astype(np.float32)

        # Use macro-F1 for early stopping (focal loss on val is not comparable to CCE)
        macro_f1_cb = MacroF1Callback(val["X_ecg"], val["X_rr"], val["y"], n_classes)

        callbacks = [
            macro_f1_cb,   # must be first so val_macro_f1 is in logs before EarlyStopping reads it
            keras.callbacks.EarlyStopping(
                monitor="val_macro_f1", mode="max",
                patience=int(t_cfg["early_stop_patience"]),
                restore_best_weights=bool(t_cfg["restore_best_weights"]),
                verbose=0,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(t_cfg["lr_reduce_factor"]),
                patience=int(t_cfg["lr_reduce_patience"]),
                min_lr=float(t_cfg["lr_min"]),
                verbose=0,
            ),
        ]

        t0 = time.perf_counter()
        history = model.fit(
            [train["X_ecg"], train["X_rr"]], y_tr_oh,
            validation_data=([val["X_ecg"], val["X_rr"]], y_val_oh),
            epochs=int(t_cfg["max_epochs"]),
            batch_size=int(t_cfg["batch_size"]),
            callbacks=callbacks,
            verbose=0,
        )
        elapsed = time.perf_counter() - t0
        n_ep    = len(history.history["loss"])

        y_pred_test, proba_test = predict(model, test["X_ecg"], test["X_rr"])
        m_test = multiclass_metrics(test["y"], y_pred_test, proba_test, k=n_classes)

        if verbose:
            print(f"  seed {seed}: {n_ep} epochs, {elapsed:.0f}s  "
                  f"Se_S={100*m_test.get('se_S',0):.1f}%  Se_V={100*m_test.get('se_V',0):.1f}%")

        all_results.append(m_test)

    mean_m = {
        k: float(np.mean([m[k] for m in all_results if k in m]))
        for k in all_results[0]
        if isinstance(all_results[0][k], float)
    }
    return {"per_seed": all_results, "mean": mean_m}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(seeds: list[int] | None = None, gammas: list[float] | None = None,
        verbose: bool = True) -> dict:
    if seeds  is None: seeds  = [0, 1, 2]
    if gammas is None: gammas = [1.0, 2.0, 3.0]

    cfg = load_config()
    cfg["training"]["class_weight"] = "none"   # focal loss handles imbalance

    train = load_data("ds1_train", cfg)
    val   = load_data("ds1_val",   cfg)
    test  = load_data("ds2",       cfg)

    kernel_mf = load_kernel_mf(cfg)

    gamma_results = {}
    if verbose:
        print_header()

    for gamma in gammas:
        if verbose:
            print(f"[exp4] Training with focal loss gamma={gamma}, seeds={seeds}…")
        res = _train_focal(cfg, gamma, seeds, train, val, test, kernel_mf, verbose)
        gamma_results[gamma] = res
        if verbose:
            print_row(f"Exp4: focal gamma={gamma} ({len(seeds)} seeds)", res["mean"])

    if verbose:
        print_footer()

    return gamma_results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Exp4: focal loss with varying gamma")
    ap.add_argument("--seeds",  nargs="+", type=int,   default=[0, 1, 2])
    ap.add_argument("--gammas", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    args = ap.parse_args()
    run(seeds=args.seeds, gammas=args.gammas, verbose=True)
