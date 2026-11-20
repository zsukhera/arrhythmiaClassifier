# %% [markdown]
# ## train.py — Training loop, multi-seed runs, class weights
#
# Reference: Farag (2023), Sections 5.3 and 6.1.
#
# Training details (paper Sec 5.3):
#   - Adam lr=1e-3, batch=512, max 500 epochs
#   - EarlyStopping on val_loss, patience=20 [assumption]
#   - ReduceLROnPlateau on val_loss, factor=0.5, patience=5 [assumption]
#   - 10 seeds per configuration; results reported as mean ± std
#   - Paper says "rarely exceeds 100 epochs or 2 min per run"
#
# DS2 is NEVER touched during training.  Templates, class weights, and early
# stopping statistics all come from DS1 only.

# %%
import logging
import os
import pickle
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# enable_op_determinism() is a one-time global flag in TF 2.9+.
# Track whether it has been activated so we only attempt it once per process.
_DETERMINISM_ENABLED: bool = False


def _set_seeds(seed: int, deterministic: bool = True) -> None:
    """
    Fix all random sources for reproducibility.

    deterministic=True calls enable_op_determinism() once per process.
    Set deterministic=False (or --no-deterministic) to skip it on GPU configs
    where certain ops have no deterministic kernel — the flag is optional and
    only improves cross-seed reproducibility at the cost of ~5–15% throughput.
    """
    global _DETERMINISM_ENABLED
    import tensorflow as tf
    from tensorflow import keras
    os.environ["PYTHONHASHSEED"] = str(seed)
    keras.utils.set_random_seed(seed)
    if deterministic and not _DETERMINISM_ENABLED:
        try:
            tf.config.experimental.enable_op_determinism()
            _DETERMINISM_ENABLED = True
        except Exception as exc:
            log.warning(
                "enable_op_determinism() not available on this device/driver "
                "(%s). Training will proceed — results may differ slightly "
                "across seeds. Use --no-deterministic to suppress this warning.",
                exc,
            )


# %%
def compute_class_weights(y: np.ndarray, n_classes: int) -> dict:
    """
    sklearn 'balanced' class weights: w_i = n / (n_classes * count_i).
    Applies to DS1-train ONLY [assumption: formula].
    Returns dict {class_index: weight} for use with model.fit(class_weight=...).
    """
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(n_classes)
    present = np.unique(y)
    if len(present) < n_classes:
        log.warning("Only %d of %d classes in DS1-train; missing: %s",
                    len(present), n_classes, set(range(n_classes)) - set(present))
    weights = compute_class_weight("balanced", classes=classes[np.isin(classes, present)],
                                   y=y[np.isin(y, present)])
    cw = dict(zip(present.tolist(), weights.tolist()))
    # Set absent classes to 1.0 (they won't appear so weight is irrelevant)
    for c in range(n_classes):
        cw.setdefault(c, 1.0)
    log.info("class weights: %s", {k: f"{v:.3f}" for k, v in sorted(cw.items())})
    return cw


def to_categorical(y: np.ndarray, n_classes: int) -> np.ndarray:
    """One-hot encode integer labels."""
    import tensorflow as tf
    return tf.keras.utils.to_categorical(y, num_classes=n_classes).astype(np.float32)


class MacroF1Callback:
    """
    Keras callback that computes macro-F1 on the validation set at each epoch
    and stores it in logs as 'val_macro_f1'.  Used to drive EarlyStopping when
    val_loss is a poor signal (e.g. heavily imbalanced data where N dominates).
    """
    def __new__(cls, X_ecg_val, X_rr_val, y_val, n_classes):
        # Delay importing keras so this module loads without TF installed.
        from tensorflow import keras

        class _CB(keras.callbacks.Callback):
            def __init__(self, X_ecg_val, X_rr_val, y_val, n_classes):
                super().__init__()
                self._Xe = X_ecg_val
                self._Xr = X_rr_val
                self._y  = y_val
                self._K  = n_classes

            def on_epoch_end(self, epoch, logs=None):
                y_pred = np.argmax(
                    self.model.predict([self._Xe, self._Xr], verbose=0), axis=1
                )
                f1s = []
                for c in range(self._K):
                    tp = np.sum((self._y == c) & (y_pred == c))
                    fp = np.sum((self._y != c) & (y_pred == c))
                    fn = np.sum((self._y == c) & (y_pred != c))
                    p  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    r  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0.0)
                if logs is not None:
                    logs["val_macro_f1"] = float(np.mean(f1s))

        return _CB(X_ecg_val, X_rr_val, y_val, n_classes)


# %%
def train_one(
    cfg:          dict,
    X_ecg_tr:     np.ndarray,  # (N, NK, C) float32
    X_rr_tr:      np.ndarray,  # (N, 4)     float32
    y_tr:         np.ndarray,  # (N,)       int
    X_ecg_val:    np.ndarray,
    X_rr_val:     np.ndarray,
    y_val:        np.ndarray,
    kernel_mf:    "np.ndarray | None",   # (NK, C, 13) if MF init, else None
    n_classes:    int,
    seed:         int,
    deterministic: bool = True,
) -> tuple:
    """
    Train the model for one seed.

    Returns (model, history, train_seconds).
    The returned model has the best validation weights restored.
    DS2 data must NOT appear in X_ecg_tr / X_rr_tr or their validation splits.
    """
    import tensorflow as tf
    from tensorflow import keras
    from .model import build_model, apply_mf_init, compile_model, validate_param_count

    t_cfg = cfg["training"]
    m_cfg = cfg["model"]
    i_cfg = cfg["input"]
    t_name = cfg["task"]["init"]  # mf_trainable | mf_frozen | default

    _set_seeds(seed, deterministic=deterministic)

    nk         = int(i_cfg["nk"])
    n_channels = X_ecg_tr.shape[-1]
    arch       = str(m_cfg.get("arch", "compact"))
    # Number of Conv1D filters follows the MF kernel when one is supplied
    # (13 by default; 11 for 3-class when templates.match_classes is true).
    n_filters  = int(kernel_mf.shape[-1]) if kernel_mf is not None else int(m_cfg.get("n_filters", 13))
    rr_units   = (list(m_cfg.get("table3_rr_units", [32, 16, 8])) if arch == "table3"
                  else list(m_cfg["rr_dense_units"]))
    model      = build_model(
        nk=nk,
        n_channels=n_channels,
        n_classes=n_classes,
        rr_dense_units=rr_units,
        activation=str(m_cfg["activation"]),
        padding=str(m_cfg["padding"]),
        use_bias=bool(m_cfg["use_bias"]),
        arch=arch,
        n_filters=n_filters,
        input_len=int(X_ecg_tr.shape[1]),
        n_rr=int(X_rr_tr.shape[1]),
    )
    if arch == "compact":
        validate_param_count(model, nk, n_channels, n_classes)
    else:
        # table3 counts are fixed by the paper (1267/1619 for 11 filters);
        # other filter counts (e.g. 3 AAMI templates) are legitimate ablations.
        log.info("params: total=%d (arch=%s, filters=%d, NK=%d, C=%d)",
                 model.count_params(), arch, n_filters, nk, n_channels)

    # Templates may be longer than the kernel when the ECG input is not
    # centre-cropped (input.crop_input: false): crop the template centre.
    if kernel_mf is not None and kernel_mf.shape[0] > nk:
        m0 = (kernel_mf.shape[0] - nk) // 2
        kernel_mf = kernel_mf[m0:m0 + nk]

    # Apply matched-filter initialisation if requested
    if t_name in ("mf_trainable", "mf_frozen") and kernel_mf is not None:
        apply_mf_init(model, kernel_mf, frozen=(t_name == "mf_frozen"))
    elif t_name in ("mf_trainable", "mf_frozen") and kernel_mf is None:
        log.warning("init=%s but no MF kernel supplied; using Glorot init", t_name)

    compile_model(model, lr=float(t_cfg["lr"]), n_classes=n_classes)

    # Class weights (applied during fit, not in compile)
    class_weight = None
    if str(t_cfg.get("class_weight", "none")) == "balanced":
        class_weight = compute_class_weights(y_tr, n_classes)
        # training.class_weight_power p (default 1 = sklearn 'balanced'):
        # w_c = balanced_c ** p, i.e. proportional to (1/freq_c)^p.
        _p = float(t_cfg.get("class_weight_power", 1.0))
        if _p != 1.0:
            class_weight = {c: float(w) ** _p for c, w in class_weight.items()}
            log.info("class weights ^%.2f: %s", _p,
                     {k: f"{v:.3f}" for k, v in sorted(class_weight.items())})

    y_tr_oh  = to_categorical(y_tr,  n_classes)
    y_val_oh = to_categorical(y_val, n_classes)

    # When class weights are used, val_loss is unweighted (Keras limitation) so
    # it stays dominated by N and triggers early stopping before minority classes
    # are learned.  Switch to macro-F1 on val set as the early stopping signal.
    use_balanced = str(t_cfg.get("class_weight", "none")) == "balanced"
    es_choice  = str(t_cfg.get("early_stop_monitor", "auto"))   # auto | val_loss | val_macro_f1
    if es_choice == "auto":
        es_monitor = "val_macro_f1" if use_balanced else "val_loss"
    else:
        es_monitor = es_choice
    es_mode    = "max" if es_monitor == "val_macro_f1" else "min"

    macro_f1_cb = (
        MacroF1Callback(X_ecg_val, X_rr_val, y_val, n_classes)
        if es_monitor == "val_macro_f1" else None
    )

    # MacroF1Callback must come FIRST so it populates val_macro_f1 in logs
    # before EarlyStopping reads it in the same on_epoch_end pass.
    callbacks = []
    if macro_f1_cb is not None:
        callbacks.append(macro_f1_cb)
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=es_monitor,
            mode=es_mode,
            patience=int(t_cfg["early_stop_patience"]),
            restore_best_weights=bool(t_cfg["restore_best_weights"]),
            verbose=0,
        ))
    # lr_schedule: plateau (default, ReduceLROnPlateau on val_loss) | none
    if str(t_cfg.get("lr_schedule", "plateau")) == "plateau":
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=float(t_cfg["lr_reduce_factor"]),
                patience=int(t_cfg["lr_reduce_patience"]),
                min_lr=float(t_cfg["lr_min"]),
                verbose=0,
            ))

    t0 = time.perf_counter()
    history = model.fit(
        [X_ecg_tr, X_rr_tr], y_tr_oh,
        validation_data=([X_ecg_val, X_rr_val], y_val_oh),
        epochs=int(t_cfg["max_epochs"]),
        batch_size=int(t_cfg["batch_size"]),
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=0,
    )
    elapsed = time.perf_counter() - t0
    n_epochs = len(history.history["loss"])
    log.info("seed %d: %d epochs in %.1f s (best val_loss=%.4f)",
             seed, n_epochs, elapsed,
             min(history.history.get("val_loss", [np.nan])))

    return model, history, elapsed


# %%
def predict(model, X_ecg: np.ndarray, X_rr: np.ndarray,
            batch_size: int = 1024) -> tuple:
    """
    Run inference and return (y_pred, proba).
    Batched to avoid OOM on large test sets.
    """
    proba  = model.predict([X_ecg, X_rr], batch_size=batch_size, verbose=0)
    y_pred = np.argmax(proba, axis=1)
    return y_pred, proba


# %%
def train_multi_seed(
    cfg:          dict,
    X_ecg_tr:     np.ndarray,
    X_rr_tr:      np.ndarray,
    y_tr:         np.ndarray,
    X_ecg_val:    np.ndarray,
    X_rr_val:     np.ndarray,
    y_val:        np.ndarray,
    X_ecg_test:   np.ndarray,
    X_rr_test:    np.ndarray,
    y_test:       np.ndarray,
    kernel_mf:    "np.ndarray | None",
    n_classes:    int,
    seeds:        list,
    cache_dir:    "Path | None" = None,
    run_name:     str = "run",
    deterministic: bool = True,
) -> list:
    """
    Run training for each seed and evaluate on the test set.
    Returns list of dicts: one per seed with 'metrics_val', 'metrics_test',
    'history', 'seed', 'elapsed'.

    If cache_dir is set, each seed's result is cached so interrupted runs can
    resume without re-training.
    """
    from .metrics import multiclass_metrics

    results = []
    for seed in seeds:
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"{run_name}_seed{seed}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as fh:
                    r = pickle.load(fh)
                log.info("seed %d: loaded from cache", seed)
                results.append(r)
                continue

        model, history, elapsed = train_one(
            cfg, X_ecg_tr, X_rr_tr, y_tr,
            X_ecg_val, X_rr_val, y_val,
            kernel_mf, n_classes, seed,
            deterministic=deterministic,
        )

        y_pred_val,  proba_val  = predict(model, X_ecg_val,  X_rr_val)
        y_pred_test, proba_test = predict(model, X_ecg_test, X_rr_test)

        m_val  = multiclass_metrics(y_val,  y_pred_val,  proba_val,  k=n_classes)
        m_test = multiclass_metrics(y_test, y_pred_test, proba_test, k=n_classes)

        r = {
            "seed":         seed,
            "elapsed":      elapsed,
            "n_epochs":     len(history.history["loss"]),
            "loss_history": history.history.get("loss", []),
            "val_loss_history": history.history.get("val_loss", []),
            "metrics_val":  m_val,
            "metrics_test": m_test,
            "proba_test":   proba_test,
            "proba_val":    proba_val,
            "y_pred_test":  y_pred_test,
            "y_pred_val":   y_pred_val,
        }
        results.append(r)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            # Save metrics cache (no model weights — keep pkl small)
            with open(cache_path, "wb") as fh:
                pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # Save model weights separately so evaluate.py can reload without retraining
            model_path = cache_path.with_suffix(".keras")
            model.save(str(model_path))
            log.info("seed %d: model saved to %s", seed, model_path)

        log.info("seed %d: test acc=%.2f%% ovr=%.2f%% macro_se=%.2f%% (S=%.1f%% V=%.1f%%)",
                 seed,
                 100 * r["metrics_test"].get("accuracy", 0),
                 100 * r["metrics_test"].get("ovr_accuracy", 0),
                 100 * r["metrics_test"].get("macro_se", 0),
                 100 * r["metrics_test"].get("se_S", 0),
                 100 * r["metrics_test"].get("se_V", 0))

    return results
