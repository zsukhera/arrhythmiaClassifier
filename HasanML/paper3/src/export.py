# %% [markdown]
# ## export.py — TFLite export and ESP32 C header generation
#
# Reference: Farag (2023), Section 6.2.
#
# The paper targets the ESP32 microcontroller (240 MHz, ~300 KB SRAM).
# The model is exported as TFLite in four quantisation levels:
#   float32 : baseline; exact Keras accuracy
#   float16 : half-precision; minimal accuracy change
#   dynamic  : dynamic-range int8; weights quantised, activations float
#   int8_ptq : full-integer post-training quantisation (PTQ) with a
#              representative dataset of 500 DS1-train beats [assumption]
#   int8_qat : quantisation-aware training int8 (from TFMOT) [assumption: use
#              tfmot instead of QKeras which the paper used]
#   pruned   : 50 % magnitude pruning + int8_ptq [assumption: sparsity=0.5]
#
# Paper finding: int8_ptq caused >10% accuracy drop; QAT fixed it (Sec 6.2).
# We test whether the same gap appears with tfmot.
#
# The firmware must reproduce (documented in README Sec "Firmware checklist"):
#   128 Hz resampling, 64-sample window, mean removal, derivative,
#   RR ring buffer (400 entries), input quantisation scale+zero_point.

# %%
import logging
import struct
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

try:
    import tensorflow as tf
    _TF_OK = True
except ImportError:  # pragma: no cover
    _TF_OK = False

try:
    import tensorflow_model_optimization as tfmot
    _TFMOT_OK = True
except ImportError:
    _TFMOT_OK = False
    log.warning("tensorflow_model_optimization not installed; QAT/pruning export skipped.")


# %%
def _input_order(model) -> list:
    """
    Order of the inputs in the converted TFLite graph, as 'ecg'/'rr'.
    The converter does NOT keep the Keras [ecg_in, rr_in] order (it put rr_in
    first here), and the representative dataset must follow the TFLite order;
    feeding [ecg, rr] blindly made calibration crash ("CONV_2D input dims
    3 != 4" / "RESHAPE 4 != 32").
    """
    interp = tf.lite.Interpreter(model_content=convert_float32(model))
    return ["ecg" if len(d["shape"]) == 3 else "rr" for d in interp.get_input_details()]


def _representative_dataset_fn(X_ecg: np.ndarray, X_rr: np.ndarray,
                                n: int = 500, order=("ecg", "rr")):
    """
    Generator of representative inputs for full-int8 PTQ calibration.
    Draws `n` samples from DS1-train (never DS2).
    """
    idx = np.random.default_rng(0).choice(len(X_ecg), min(n, len(X_ecg)), replace=False)
    for i in idx:
        arrs = {"ecg": X_ecg[i:i+1].astype(np.float32), "rr": X_rr[i:i+1].astype(np.float32)}
        yield [arrs[k] for k in order]


def _converter(model):
    """
    TFLiteConverter on a batch-1 concrete function.  from_keras_model keeps a
    dynamic batch dimension, which turns every Reshape into SHAPE +
    STRIDED_SLICE + PACK + RESHAPE ops (useless work and extra op resolvers on
    the MCU).  The MCU always runs one beat per invoke, so fix batch=1.
    """
    specs = [tf.TensorSpec([1] + list(t.shape[1:]), tf.float32,
                           name=t.name.split(":")[0]) for t in model.inputs]
    fn = tf.function(lambda *xs: model(list(xs), training=False))
    cf = fn.get_concrete_function(*specs)
    return tf.lite.TFLiteConverter.from_concrete_functions([cf], model)


def _best_holdout_f1_callback(X_ecg_val, X_rr_val, y_val, n_classes, start_epoch=0):
    """
    Keeps the weights of the epoch with the best macro-F1 on the DS1 hold-out
    (only epochs >= start_epoch, e.g. after the pruning schedule has reached
    its final sparsity).  Call cb.restore() after fit().  Same selection
    signal as the float model's early stopping (src.train.MacroF1Callback).
    """
    from tensorflow import keras

    class _CB(keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best, self.best_w, self.best_epoch, self.hist = -1.0, None, -1, []

        def on_epoch_end(self, epoch, logs=None):
            pred = np.argmax(self.model.predict([X_ecg_val, X_rr_val], batch_size=4096,
                                                verbose=0), axis=1)
            f1s = []
            for c in range(n_classes):
                tp = np.sum((y_val == c) & (pred == c))
                fp = np.sum((y_val != c) & (pred == c))
                fn = np.sum((y_val == c) & (pred != c))
                f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
            f1 = float(np.mean(f1s))
            self.hist.append(f1)
            if epoch >= start_epoch and f1 > self.best:
                self.best, self.best_epoch = f1, epoch
                self.best_w = [w.copy() for w in self.model.get_weights()]

        def restore(self):
            if self.best_w is not None:
                self.model.set_weights(self.best_w)
            log.info("best hold-out macro-F1 %.4f at epoch %d (history %s)",
                     self.best, self.best_epoch, [round(x, 4) for x in self.hist])

    return _CB()


def convert_float32(model) -> bytes:
    """TFLite float32 — identical to Keras inference."""
    conv = _converter(model)
    return conv.convert()


def convert_float16(model) -> bytes:
    """TFLite float16 — weights and activations in fp16; minimal accuracy change."""
    conv = _converter(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def convert_dynamic(model) -> bytes:
    """Dynamic-range int8 quantisation — weights int8, activations float at run time."""
    conv = _converter(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def convert_int8_ptq(model, X_ecg_repr: np.ndarray, X_rr_repr: np.ndarray,
                     n_repr: int = 500) -> bytes:
    """
    Full-integer int8 PTQ with a representative dataset.
    Both weights and activations are quantised; no floating-point ops remain.
    The representative dataset should come from DS1-train ONLY.
    """
    order = _input_order(model)
    conv = _converter(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: _representative_dataset_fn(
        X_ecg_repr, X_rr_repr, n_repr, order
    )
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def convert_int8_qat(model, X_ecg_tr: np.ndarray, X_rr_tr: np.ndarray,
                     y_tr: np.ndarray, X_ecg_val: np.ndarray, X_rr_val: np.ndarray,
                     y_val: np.ndarray, n_classes: int,
                     X_ecg_repr: np.ndarray, X_rr_repr: np.ndarray,
                     cfg: dict, class_weight: Optional[dict] = None,
                     epochs: int = 10, lr: float = 1e-5,
                     rr_clip: Optional[float] = None) -> bytes:
    """
    Quantisation-aware training (QAT) int8 export using tensorflow_model_optimization.
    The paper used QKeras; here we use tfmot as a drop-in [assumption].

    QAT inserts fake-quant nodes during fine-tuning, minimising the accuracy gap
    that full-integer PTQ introduces by letting the model adapt to the reduced
    precision.

    tfmot only works with legacy tf.keras (tf_keras, TF_USE_LEGACY_KERAS=1);
    `model` must therefore be a tf_keras model (see 1DCNN/export/export_tflite.py,
    which rebuilds the Keras-3 model in a tf_keras venv).

    class_weight : pass the same weights the float model was trained with
                   (e.g. balanced); fine-tuning without them drifts toward N.
    """
    if not _TFMOT_OK:
        log.warning("tfmot not available; skipping QAT export")
        return b""

    from tensorflow import keras
    from .train import to_categorical

    t_cfg = cfg["training"]

    # Apply QAT to the entire model
    qat_model = tfmot.quantization.keras.quantize_model(model)
    qat_model.compile(
        optimizer=keras.optimizers.Adam(lr),   # low LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_tr_oh  = to_categorical(y_tr,  n_classes)
    y_val_oh = to_categorical(y_val, n_classes)

    if rr_clip is not None:
        # RR features have rare huge outliers (up to ~100 in DS1-train) that
        # blow up QAT's moving-average input range (scale 0.40 -> RR crushed
        # to a few levels).  Int8 inference saturates there anyway.
        X_rr_tr  = np.clip(X_rr_tr,  0.0, rr_clip)
        X_rr_val = np.clip(X_rr_val, 0.0, rr_clip)
        X_rr_repr = np.clip(X_rr_repr, 0.0, rr_clip)
    best_cb = _best_holdout_f1_callback(X_ecg_val, X_rr_val, y_val, n_classes)
    qat_model.fit(
        [X_ecg_tr, X_rr_tr], y_tr_oh,
        validation_data=([X_ecg_val, X_rr_val], y_val_oh),
        epochs=int(epochs),   # short fine-tuning [assumption]
        batch_size=int(t_cfg["batch_size"]),
        class_weight=class_weight,
        callbacks=[best_cb],
        verbose=0,
    )
    best_cb.restore()
    convert_int8_qat.last_history = best_cb.hist

    # Convert to int8
    order = _input_order(qat_model)
    conv = _converter(qat_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: _representative_dataset_fn(
        X_ecg_repr, X_rr_repr, 500, order
    )
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def convert_pruned_int8(model, X_ecg_tr: np.ndarray, X_rr_tr: np.ndarray,
                        y_tr: np.ndarray, X_ecg_val: np.ndarray, X_rr_val: np.ndarray,
                        y_val: np.ndarray, n_classes: int,
                        X_ecg_repr: np.ndarray, X_rr_repr: np.ndarray,
                        cfg: dict, class_weight: Optional[dict] = None,
                        epochs: int = 20, lr: float = 1e-4,
                        return_model: bool = False,
                        rr_clip: Optional[float] = None):
    """
    50 % magnitude pruning + int8 PTQ.
    Pruning removes small weights to increase sparsity; the remaining non-zero
    weights are quantised.  On microcontrollers with sparse-compute support
    this reduces both latency and RAM.
    """
    if not _TFMOT_OK:
        log.warning("tfmot not available; skipping pruning export")
        return b""

    from tensorflow import keras
    from .train import to_categorical

    t_cfg  = cfg["training"]
    e_cfg  = cfg["export"]
    sparsity  = float(e_cfg["pruning_sparsity"])
    end_step  = int(e_cfg["pruning_end_step"])
    freq      = int(e_cfg["pruning_frequency"])

    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0,
        final_sparsity=sparsity,
        begin_step=int(e_cfg["pruning_begin_step"]),
        end_step=end_step,
        frequency=freq,
    )
    pruned_model = tfmot.sparsity.keras.prune_low_magnitude(
        model, pruning_schedule=pruning_schedule
    )
    pruned_model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_tr_oh  = to_categorical(y_tr,  n_classes)
    y_val_oh = to_categorical(y_val, n_classes)

    if rr_clip is not None:
        X_rr_tr  = np.clip(X_rr_tr,  0.0, rr_clip)
        X_rr_val = np.clip(X_rr_val, 0.0, rr_clip)
    bs = int(t_cfg["batch_size"])
    steps_per_epoch = int(np.ceil(len(y_tr) / bs))
    # only keep epochs where the final sparsity has been reached
    start_epoch = int(np.ceil(end_step / steps_per_epoch))
    best_cb = _best_holdout_f1_callback(X_ecg_val, X_rr_val, y_val, n_classes,
                                        start_epoch=start_epoch)
    pruned_model.fit(
        [X_ecg_tr, X_rr_tr], y_tr_oh,
        validation_data=([X_ecg_val, X_rr_val], y_val_oh),
        epochs=int(epochs),
        batch_size=bs,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep(), best_cb],
        class_weight=class_weight,
        verbose=0,
    )
    best_cb.restore()
    convert_pruned_int8.last_history = best_cb.hist

    # Strip pruning wrappers and convert to int8
    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
    tfl = convert_int8_ptq(stripped, X_ecg_repr, X_rr_repr)
    return (tfl, stripped) if return_model else tfl


def weight_sparsity(model) -> dict:
    """Fraction of exactly-zero entries in each kernel (bias/BN excluded)."""
    out = {}
    for layer in model.layers:
        for w in layer.weights:
            if "kernel" in w.name:
                a = np.asarray(w.numpy())
                out[layer.name] = float((a == 0).mean())
    return out


# %%
def evaluate_tflite(tflite_bytes: bytes,
                    X_ecg: np.ndarray, X_rr: np.ndarray,
                    y_true: np.ndarray, return_proba: bool = False,
                    num_threads: int = 1) -> dict:
    """
    Run inference with the TFLite interpreter and compute accuracy + macro F1.
    Handles int8 input/output by applying the quantisation scale/zero_point.
    Inference is one beat per invoke() (batch 1), as on the MCU.
    return_proba=True adds 'proba' (n, K) (dequantised) and 'us_per_beat'
    (mean wall-clock invoke() time incl. set/get tensor, host CPU).
    """
    if not tflite_bytes:
        return {}

    interp = tf.lite.Interpreter(model_content=tflite_bytes, num_threads=num_threads)
    interp.allocate_tensors()
    inp_details  = interp.get_input_details()
    out_details  = interp.get_output_details()

    # Determine quantisation params for int8 models
    def _quantise(data, detail):
        if detail["dtype"] == np.int8:
            scale, zp = detail["quantization"]
            if scale == 0.0:
                scale = 1.0
            return np.round(data / scale + zp).clip(-128, 127).astype(np.int8)  # round, not truncate
        return data.astype(np.float32)

    def _dequantise(data, detail):
        if detail["dtype"] == np.int8:
            scale, zp = detail["quantization"]
            if scale == 0.0:
                scale = 1.0
            return (data.astype(np.float32) - zp) * scale
        return data.astype(np.float32)

    # Identify which input is ECG and which is RR by shape
    ecg_detail = rr_detail = None
    for d in inp_details:
        if len(d["shape"]) == 3:
            ecg_detail = d
        else:
            rr_detail  = d

    if ecg_detail is None or rr_detail is None:
        log.warning("Could not identify ECG/RR inputs; model inputs: %s",
                    [d["shape"] for d in inp_details])
        return {}

    import time as _time
    preds, all_proba = [], []
    t_start = _time.perf_counter()
    for i in range(len(y_true)):
        ecg_i = _quantise(X_ecg[i:i+1], ecg_detail)
        rr_i  = _quantise(X_rr[i:i+1],  rr_detail)
        interp.set_tensor(ecg_detail["index"], ecg_i)
        interp.set_tensor(rr_detail["index"],  rr_i)
        interp.invoke()
        out = _dequantise(interp.get_tensor(out_details[0]["index"]), out_details[0])
        all_proba.append(out[0])
        preds.append(int(np.argmax(out[0])))

    us_per_beat = 1e6 * (_time.perf_counter() - t_start) / max(1, len(y_true))
    preds     = np.array(preds)
    all_proba = np.array(all_proba)
    acc       = float((preds == y_true).mean())

    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    out = {"accuracy": acc, "macro_f1": f1, "size_bytes": len(tflite_bytes)}
    if return_proba:
        out["proba"] = all_proba
        out["us_per_beat"] = us_per_beat
    return out


def tflite_memory_estimate(tflite_bytes: bytes) -> dict:
    """
    Host-side estimate of TFLite-Micro RAM needs from the flatbuffer graph.

    activ_sum_bytes  : sum of all non-constant tensors (inputs, intermediates,
                       outputs) — upper bound with no buffer re-use.
    activ_peak_bytes : greedy liveness peak — at each op, bytes of every tensor
                       produced earlier (or a graph input) that is still needed
                       by this or a later op, plus this op's outputs.  This is
                       what TFLM's GreedyMemoryPlanner approaches (it can be a
                       little higher because of 16-byte alignment/fragmentation).
    The real TFLM tensor arena = activ_peak + per-op persistent buffers
    (quant multipliers etc.) + TfLiteTensor/NodeAndRegistration structs; add a
    few KB of head-room and confirm with MicroInterpreter::arena_used_bytes().
    """
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    tdet = {d["index"]: d for d in interp.get_tensor_details()}
    ops = interp._get_ops_details()
    const = set()
    try:
        # constant tensors have a buffer in the flatbuffer; detect by trying to read
        # them before any invoke: variable tensors are produced by an op
        produced = {t for op in ops for t in op["outputs"]}
        graph_in = {d["index"] for d in interp.get_input_details()}
        const = set(tdet) - produced - graph_in
    except Exception:  # pragma: no cover
        pass

    def nbytes(i):
        d = tdet[i]
        return int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize

    act = [i for i in tdet if i not in const]
    last_use = {}
    for k, op in enumerate(ops):
        for t in op["inputs"]:
            if t >= 0 and t not in const:
                last_use[t] = k
    out_idx = {d["index"] for d in interp.get_output_details()}
    for t in out_idx:
        last_use[t] = len(ops)
    first_def = {t: -1 for t in act}
    for k, op in enumerate(ops):
        for t in op["outputs"]:
            first_def[t] = k
    peak = 0
    for k in range(len(ops)):
        live = sum(nbytes(t) for t in act
                   if first_def.get(t, -1) <= k and last_use.get(t, -1) >= k)
        peak = max(peak, live)
    const_bytes = sum(nbytes(i) for i in const)
    return {
        "n_ops": len(ops),
        "op_types": sorted({op["op_name"] for op in ops}),
        "activ_sum_bytes": int(sum(nbytes(t) for t in act)),
        "activ_peak_bytes": int(peak),
        "const_tensor_bytes": int(const_bytes),
        "n_tensors": len(tdet),
    }


# %%
def generate_c_header(tflite_bytes: bytes, model_name: str,
                      out_path: Path) -> None:
    """
    Write a C header file containing the TFLite model as a byte array.
    Format matches what `xxd -i` produces, compatible with TFLite Micro.

    The firmware includes this header and passes the array to
    tflite::MicroInterpreter.
    """
    var_name = model_name.replace("-", "_").replace(".", "_")
    lines = [
        f"// Auto-generated by export.py — Farag (2023) tiny 1D-CNN",
        f"// Model: {model_name}",
        f"// Size: {len(tflite_bytes)} bytes",
        f"// Same byte layout as `xxd -i`; alignas(16) because TFLite Micro reads the",
        f"// flatbuffer in place and needs it aligned. `const` keeps it in flash.",
        f"",
        f"#ifndef {var_name.upper()}_H",
        f"#define {var_name.upper()}_H",
        f"",
        f"alignas(16) const unsigned char {var_name}[] = {{",
    ]
    hex_vals = [f"0x{b:02x}" for b in tflite_bytes]
    # 12 hex values per line (xxd -i layout)
    for i in range(0, len(hex_vals), 12):
        chunk = hex_vals[i:i+12]
        suffix = "," if i + 12 < len(hex_vals) else ""
        lines.append("  " + ", ".join(chunk) + suffix)
    lines += [
        f"}};",
        f"const unsigned int {var_name}_len = {len(tflite_bytes)};",
        f"",
        f"#endif // {var_name.upper()}_H",
    ]
    out_path.write_text("\n".join(lines) + "\n")
    log.info("C header written to %s (%d bytes)", out_path, len(tflite_bytes))


# %%
def export_all(model, model_name: str, results_dir: Path,
               X_ecg_tr: np.ndarray, X_rr_tr: np.ndarray, y_tr: np.ndarray,
               X_ecg_val: np.ndarray, X_rr_val: np.ndarray, y_val: np.ndarray,
               X_ecg_test: np.ndarray, X_rr_test: np.ndarray, y_test: np.ndarray,
               n_classes: int, cfg: dict) -> dict:
    """
    Export all TFLite variants and evaluate each on DS2.
    Returns dict mapping variant name -> eval metrics dict.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_repr = int(cfg["export"]["representative_beats"])
    rng    = np.random.default_rng(42)
    idx    = rng.choice(len(X_ecg_tr), min(n_repr, len(X_ecg_tr)), replace=False)
    X_ecg_repr, X_rr_repr = X_ecg_tr[idx], X_rr_tr[idx]

    variants = {}

    # --- float32 ---
    log.info("Exporting %s float32...", model_name)
    tfl = convert_float32(model)
    variants["float32"] = tfl

    # --- float16 ---
    log.info("Exporting %s float16...", model_name)
    variants["float16"] = convert_float16(model)

    # --- dynamic ---
    log.info("Exporting %s dynamic-range int8...", model_name)
    variants["dynamic"] = convert_dynamic(model)

    # --- int8_ptq ---
    log.info("Exporting %s int8 PTQ...", model_name)
    try:
        variants["int8_ptq"] = convert_int8_ptq(model, X_ecg_repr, X_rr_repr, n_repr)
    except Exception as e:
        log.warning("int8 PTQ failed: %s", e)
        variants["int8_ptq"] = b""

    # --- int8_qat ---
    log.info("Exporting %s int8 QAT (fine-tuning)...", model_name)
    try:
        variants["int8_qat"] = convert_int8_qat(
            model, X_ecg_tr, X_rr_tr, y_tr, X_ecg_val, X_rr_val, y_val,
            n_classes, X_ecg_repr, X_rr_repr, cfg
        )
    except Exception as e:
        log.warning("int8 QAT failed: %s", e)
        variants["int8_qat"] = b""

    # --- pruned + int8 ---
    log.info("Exporting %s pruned+int8...", model_name)
    try:
        variants["pruned_int8"] = convert_pruned_int8(
            model, X_ecg_tr, X_rr_tr, y_tr, X_ecg_val, X_rr_val, y_val,
            n_classes, X_ecg_repr, X_rr_repr, cfg
        )
    except Exception as e:
        log.warning("pruned+int8 failed: %s", e)
        variants["pruned_int8"] = b""

    # Evaluate each variant and write files
    eval_results = {}
    for name, tfl_bytes in variants.items():
        if not tfl_bytes:
            eval_results[name] = {}
            continue

        # Save .tflite file
        tfl_path = results_dir / f"model_{model_name}_{name}.tflite"
        tfl_path.write_bytes(tfl_bytes)

        # Evaluate on DS2
        ev = evaluate_tflite(tfl_bytes, X_ecg_test, X_rr_test, y_test)
        ev["size_bytes"] = len(tfl_bytes)
        eval_results[name] = ev

        # C header (float32 and int8 PTQ/QAT for firmware)
        if name in ("float32", "int8_ptq", "int8_qat"):
            h_path = results_dir / f"model_{model_name}_{name}.h"
            generate_c_header(tfl_bytes, f"{model_name}_{name}", h_path)

        log.info("%-15s size=%6d bytes  acc=%.2f%%  F1=%.2f%%",
                 name, len(tfl_bytes),
                 100 * ev.get("accuracy", 0), 100 * ev.get("macro_f1", 0))

    return eval_results
