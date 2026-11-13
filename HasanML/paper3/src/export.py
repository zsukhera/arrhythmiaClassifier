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
def _representative_dataset_fn(X_ecg: np.ndarray, X_rr: np.ndarray,
                                n: int = 500):
    """
    Generator of representative inputs for full-int8 PTQ calibration.
    Draws `n` samples from DS1-train (never DS2).
    """
    idx = np.random.default_rng(0).choice(len(X_ecg), min(n, len(X_ecg)), replace=False)
    for i in idx:
        yield [
            X_ecg[i:i+1].astype(np.float32),
            X_rr[i:i+1].astype(np.float32),
        ]


def convert_float32(model) -> bytes:
    """TFLite float32 — identical to Keras inference."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    return conv.convert()


def convert_float16(model) -> bytes:
    """TFLite float16 — weights and activations in fp16; minimal accuracy change."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.target_spec.supported_types = [tf.float16]
    return conv.convert()


def convert_dynamic(model) -> bytes:
    """Dynamic-range int8 quantisation — weights int8, activations float at run time."""
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    return conv.convert()


def convert_int8_ptq(model, X_ecg_repr: np.ndarray, X_rr_repr: np.ndarray,
                     n_repr: int = 500) -> bytes:
    """
    Full-integer int8 PTQ with a representative dataset.
    Both weights and activations are quantised; no floating-point ops remain.
    The representative dataset should come from DS1-train ONLY.
    """
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: _representative_dataset_fn(
        X_ecg_repr, X_rr_repr, n_repr
    )
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def convert_int8_qat(model, X_ecg_tr: np.ndarray, X_rr_tr: np.ndarray,
                     y_tr: np.ndarray, X_ecg_val: np.ndarray, X_rr_val: np.ndarray,
                     y_val: np.ndarray, n_classes: int,
                     X_ecg_repr: np.ndarray, X_rr_repr: np.ndarray,
                     cfg: dict) -> bytes:
    """
    Quantisation-aware training (QAT) int8 export using tensorflow_model_optimization.
    The paper used QKeras; here we use tfmot as a drop-in [assumption].

    QAT inserts fake-quant nodes during fine-tuning, minimising the accuracy gap
    that full-integer PTQ introduces by letting the model adapt to the reduced
    precision.
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
        optimizer=keras.optimizers.Adam(1e-5),   # low LR for fine-tuning
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_tr_oh  = to_categorical(y_tr,  n_classes)
    y_val_oh = to_categorical(y_val, n_classes)

    qat_model.fit(
        [X_ecg_tr, X_rr_tr], y_tr_oh,
        validation_data=([X_ecg_val, X_rr_val], y_val_oh),
        epochs=10,            # short fine-tuning [assumption]
        batch_size=int(t_cfg["batch_size"]),
        verbose=0,
    )

    # Convert to int8
    conv = tf.lite.TFLiteConverter.from_keras_model(qat_model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = lambda: _representative_dataset_fn(
        X_ecg_repr, X_rr_repr
    )
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type  = tf.int8
    conv.inference_output_type = tf.int8
    return conv.convert()


def convert_pruned_int8(model, X_ecg_tr: np.ndarray, X_rr_tr: np.ndarray,
                        y_tr: np.ndarray, X_ecg_val: np.ndarray, X_rr_val: np.ndarray,
                        y_val: np.ndarray, n_classes: int,
                        X_ecg_repr: np.ndarray, X_rr_repr: np.ndarray,
                        cfg: dict) -> bytes:
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
        optimizer=keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    y_tr_oh  = to_categorical(y_tr,  n_classes)
    y_val_oh = to_categorical(y_val, n_classes)

    pruned_model.fit(
        [X_ecg_tr, X_rr_tr], y_tr_oh,
        validation_data=([X_ecg_val, X_rr_val], y_val_oh),
        epochs=20,
        batch_size=int(t_cfg["batch_size"]),
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()],
        verbose=0,
    )

    # Strip pruning wrappers and convert to int8
    stripped = tfmot.sparsity.keras.strip_pruning(pruned_model)
    return convert_int8_ptq(stripped, X_ecg_repr, X_rr_repr)


# %%
def evaluate_tflite(tflite_bytes: bytes,
                    X_ecg: np.ndarray, X_rr: np.ndarray,
                    y_true: np.ndarray) -> dict:
    """
    Run inference with the TFLite interpreter and compute accuracy + macro F1.
    Handles int8 input/output by applying the quantisation scale/zero_point.
    """
    if not tflite_bytes:
        return {}

    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    inp_details  = interp.get_input_details()
    out_details  = interp.get_output_details()

    # Determine quantisation params for int8 models
    def _quantise(data, detail):
        if detail["dtype"] == np.int8:
            scale, zp = detail["quantization"]
            if scale == 0.0:
                scale = 1.0
            return (data / scale + zp).clip(-128, 127).astype(np.int8)
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

    preds, all_proba = [], []
    for i in range(len(y_true)):
        ecg_i = _quantise(X_ecg[i:i+1], ecg_detail)
        rr_i  = _quantise(X_rr[i:i+1],  rr_detail)
        interp.set_tensor(ecg_detail["index"], ecg_i)
        interp.set_tensor(rr_detail["index"],  rr_i)
        interp.invoke()
        out = _dequantise(interp.get_tensor(out_details[0]["index"]), out_details[0])
        all_proba.append(out[0])
        preds.append(int(np.argmax(out[0])))

    preds     = np.array(preds)
    all_proba = np.array(all_proba)
    acc       = float((preds == y_true).mean())

    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))

    return {"accuracy": acc, "macro_f1": f1, "size_bytes": len(tflite_bytes)}


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
        f"",
        f"#ifndef {var_name.upper()}_H",
        f"#define {var_name.upper()}_H",
        f"",
        f"const unsigned char {var_name}[] = {{",
    ]
    hex_vals = [f"  0x{b:02x}" for b in tflite_bytes]
    # 12 hex values per line
    for i in range(0, len(hex_vals), 12):
        chunk = hex_vals[i:i+12]
        suffix = "," if i + 12 < len(hex_vals) else ""
        lines.append(", ".join(chunk) + suffix)
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
