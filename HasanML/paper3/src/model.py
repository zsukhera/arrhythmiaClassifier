# %% [markdown]
# ## model.py — Farag (2023) tiny matched-filter 1D-CNN
#
# Reference: Farag (2023), Sections 4, Figure 3, Table 3.
#
# Architecture:
#   ECG branch:   ecg_in (NK, C) -> Conv1D(13, NK, valid, relu)
#                 -> BatchNorm -> GlobalMaxPool -> (13,)
#   RR branch:    rr_in (4,) -> Dense(16,relu) x3 -> (16,)
#   Merged:       Concatenate(13, 16) -> Dense(n_classes, softmax)
#
# Parameter counts (single-channel, 3-class):
#   NK=32: Conv1D 429 + BN 52 + RR 624 + head 90 = 1195 total (1169 trainable)
#   NK=64: Conv1D 845 + BN 52 + RR 624 + head 90 = 1611 total (1585 trainable)
# Both are within the paper's stated 1267–1619 range; see README.
#
# GlobalMaxPooling1D is used instead of GlobalAveragePooling1D because:
#   - With padding="valid" and kernel_size=NK the Conv1D output is (1, 13),
#     so max == mean anyway.  But max is more interpretable: it captures the
#     peak filter response anywhere in the window.
#
# ASSUMPTIONS: padding="valid"; activation="relu"; 3 x Dense(16) for RR branch.
# See configs/default.yaml for each assumption with justification.

# %%
import logging
import os

import numpy as np

log = logging.getLogger(__name__)

# Keras / TensorFlow import — required at module level for type annotations
# and to allow `from src.model import build_model`.
try:
    import tensorflow as tf
    from tensorflow import keras
    _TF_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TF_AVAILABLE = False
    log.error("TensorFlow not found; model.py functions will fail at runtime.")


# %%
def resolve_device(mode: str = "auto") -> str:
    """
    Return the effective device string ('cpu' or 'cuda') and configure TF.
    'auto' selects GPU if one is visible, else CPU.
    """
    if not _TF_AVAILABLE:
        return "cpu"
    if mode in ("cuda", "gpu"):
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            log.warning("device='%s' requested but no GPU found; falling back to CPU", mode)
            return "cpu"
        for g in gpus:
            tf.config.experimental.set_memory_growth(g, True)
        return "cuda"
    if mode == "auto":
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            return "cuda"
        return "cpu"
    return "cpu"  # mode == "cpu"


def configure_tf_threads(intra: int = 0, inter: int = 0) -> None:
    """Set TF inter/intra-op parallelism threads (0 = TF default)."""
    if _TF_AVAILABLE and (intra or inter):
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)


def enable_mixed_precision(device: str) -> bool:
    """
    Enable float16 mixed precision when running on GPU.

    RTX 4000 Ada / Ampere+ Tensor Cores give ~1.5-2× throughput with
    mixed_float16.  CPU always uses float32 (no Tensor Cores; float16 would
    be slower).  The final Dense(softmax) layer is kept at float32 by Keras
    automatically via the dtype policy.

    Returns True if mixed precision was activated, False otherwise.
    """
    if not _TF_AVAILABLE:
        return False
    if device not in ("cuda", "gpu"):
        return False
    try:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        log.info("Mixed precision activated: mixed_float16 "
                 "(compute=float16, weights=float32)")
        return True
    except Exception as exc:
        log.warning("Could not enable mixed precision: %s — using float32", exc)
        return False


# %%
def build_model(nk: int, n_channels: int, n_classes: int,
                rr_dense_units: list = None,
                activation: str = "relu",
                padding: str = "valid",
                use_bias: bool = True,
                arch: str = "compact",
                n_filters: int = 13,
                input_len: "int | None" = None,
                n_rr: int = 4) -> "keras.Model":
    """
    Build the Farag (2023) tiny 1D-CNN.

    Parameters
    ----------
    nk             : Conv1D kernel size (32 or 64)
    n_channels     : 1 (signal-only or derivative-only) or 2 (both)
    n_classes      : output units (2 for binary, 3/4/5 for multiclass)
    rr_dense_units : hidden units for the RR branch; default [16, 16, 16]
                     (compact) or [32, 16, 8] (table3)
    activation     : 'relu' or 'tanh' (paper says either works, Sec 4)
    padding        : 'valid' (default) or 'same' [assumption]
    use_bias       : Conv1D bias; True [assumption]
    arch           : 'compact' (default, original replication: ECG input of
                     length NK, Conv1D(act) -> BN -> GMP, RR Dense stack on the
                     4-vector) or 'table3' (paper Table 3 exactly: ECG input of
                     length input_len (64), Conv1D(linear, padding) -> BN ->
                     Tanh -> GMP; RR input reshaped to (n_rr, 1) and passed
                     through Dense(32)-Dense(16)-Dense(8) ReLU applied per RR
                     feature, then Flatten -> n_rr*8).  table3 params for
                     3 classes / 11 filters / NK=32 / same: 1267 (paper).
    n_filters      : number of Conv1D filters (= number of MF templates)
    input_len      : ECG input length; default nk (compact) / 64 (table3)
    n_rr           : number of RR input features (default 4)

    Architecture note (compact):
    With padding='valid' and kernel_size=nk, the Conv1D output has length 1
    regardless of input length.  GlobalMaxPooling1D of a (1, 13) tensor is
    identical to Flatten, but the name makes the design intent explicit.
    """
    if arch == "table3":
        return _build_model_table3(nk, n_channels, n_classes,
                                   rr_dense_units or [32, 16, 8],
                                   padding=padding, use_bias=use_bias,
                                   n_filters=n_filters,
                                   input_len=input_len or 64, n_rr=n_rr)
    if arch != "compact":
        raise ValueError(f"Unknown arch {arch!r}")
    if rr_dense_units is None:
        rr_dense_units = [16, 16, 16]

    # ECG branch
    ecg_in = keras.Input(shape=(input_len or nk, n_channels), name="ecg_in")
    x = keras.layers.Conv1D(
        filters=n_filters,
        kernel_size=nk,
        strides=1,
        padding=padding,
        use_bias=use_bias,
        activation=activation,
        name="conv1d",
    )(ecg_in)
    x = keras.layers.BatchNormalization(name="batch_norm")(x)
    x = keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)   # -> (batch, 13)

    # RR branch — three Dense(16) layers learn a non-linear function of
    # normalised heart rate context.  Three layers (not one) allow the branch
    # to represent more complex rhythm patterns; Table 3 [assumption].
    rr_in = keras.Input(shape=(n_rr,), name="rr_in")
    r = rr_in
    for i, units in enumerate(rr_dense_units):
        r = keras.layers.Dense(units, activation=activation,
                               name=f"rr_dense_{i}")(r)

    # Merge ECG features (morphology) and RR features (rhythm)
    merged = keras.layers.Concatenate(name="concat")([x, r])
    out    = keras.layers.Dense(n_classes, activation="softmax",
                                name="output")(merged)

    return keras.Model(inputs=[ecg_in, rr_in], outputs=out, name="farag2023")


def _build_model_table3(nk: int, n_channels: int, n_classes: int,
                        rr_dense_units: list, padding: str = "same",
                        use_bias: bool = True, n_filters: int = 11,
                        input_len: int = 64, n_rr: int = 4) -> "keras.Model":
    """
    Paper Table 3 architecture (Farag 2023, Sensors 23(3):1365):
      Signal Input (64, C) -> Conv1D(NF, NK) (64, NF) -> BatchNormalization
      -> Activation(Tanh) -> GlobalMaxPooling (NF)
      Interval Input (4, 1) -> Dense 32 ReLU (4, 32) -> Dense 16 ReLU (4, 16)
      -> Dense 8 ReLU (4, 8) -> Flatten (32)
      Concatenate (NF + 32) -> Dense(n_classes, softmax)
    With NF=11 (3 classes), NK=32: 363+44+64+528+136+132 = 1267 params.
    """
    ecg_in = keras.Input(shape=(input_len, n_channels), name="ecg_in")
    x = keras.layers.Conv1D(filters=n_filters, kernel_size=nk, strides=1,
                            padding=padding, use_bias=use_bias,
                            activation=None, name="conv1d")(ecg_in)
    x = keras.layers.BatchNormalization(name="batch_norm")(x)
    x = keras.layers.Activation("tanh", name="conv_tanh")(x)
    x = keras.layers.GlobalMaxPooling1D(name="global_max_pool")(x)

    rr_in = keras.Input(shape=(n_rr,), name="rr_in")
    r = keras.layers.Reshape((n_rr, 1), name="rr_reshape")(rr_in)
    for i, units in enumerate(rr_dense_units):
        r = keras.layers.Dense(units, activation="relu", name=f"rr_dense_{i}")(r)
    r = keras.layers.Flatten(name="rr_flatten")(r)

    merged = keras.layers.Concatenate(name="concat")([x, r])
    out = keras.layers.Dense(n_classes, activation="softmax", name="output")(merged)
    return keras.Model(inputs=[ecg_in, rr_in], outputs=out, name="farag2023_table3")


def validate_param_count(model: "keras.Model", nk: int, n_channels: int,
                         n_classes: int) -> None:
    """
    Guard: fail loudly if the total parameter count is outside 1,000–2,000
    for a 3-class single-channel model.  Adjust if Table 3 reveals a different
    architecture.
    """
    total = model.count_params()
    if n_classes == 3 and n_channels == 1:
        if not (1000 <= total <= 2000):
            raise RuntimeError(
                f"Parameter count {total} outside expected range 1000-2000 for "
                f"3-class single-channel model (NK={nk}).  Check the RR dense "
                f"stack against Table 3 and fix model.py."
            )
        log.info("param check OK: total=%d trainable=%d (3-class, NK=%d, 1-channel)",
                 total, _count_trainable(model), nk)
    else:
        log.info("params: total=%d trainable=%d (n_classes=%d, NK=%d, C=%d)",
                 total, _count_trainable(model), n_classes, nk, n_channels)


def _count_trainable(model: "keras.Model") -> int:
    return int(np.sum([np.prod(v.shape) for v in model.trainable_variables]))


def model_summary_lines(model: "keras.Model") -> list:
    """Return model.summary() as a list of strings for logging/writing."""
    lines = []
    model.summary(print_fn=lines.append)
    return lines


# %%
def apply_mf_init(model: "keras.Model", kernel: np.ndarray, frozen: bool = False) -> None:
    """
    Apply matched-filter kernel initialisation to the Conv1D layer.

    kernel : (NK, C, 13) float32 array from templates.templates_to_kernel()
    frozen : if True, set Conv1D trainable=False (mf_frozen variant)
             if False, MF init but trainable (mf_trainable variant, paper's best)

    After setting weights, we must recompile if frozen changes trainability.
    The caller is responsible for recompiling.
    """
    conv = model.get_layer("conv1d")
    current = conv.get_weights()
    if len(current) == 2:
        bias = current[1]
    else:
        bias = np.zeros(kernel.shape[-1], dtype=np.float32)
    conv.set_weights([kernel.astype(np.float32), bias])
    conv.trainable = not frozen
    log.info("MF init applied: frozen=%s, kernel shape=%s, L2 norms: %s",
             frozen, kernel.shape,
             [f"{np.linalg.norm(kernel[:, :, i]):.3f}" for i in range(kernel.shape[-1])])


def compile_model(model: "keras.Model", lr: float = 1e-3, n_classes: int = 3,
                  class_weight_arr: "np.ndarray | None" = None) -> None:
    """
    Compile with categorical cross-entropy and Adam.
    Class weights are applied during model.fit(), not here.
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
