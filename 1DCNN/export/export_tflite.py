"""
Step B of the export pipeline — runs in $ROOT/.venv-export
(Python 3.12, TF 2.19, tf_keras 2.19, tfmot 0.8.0) with TF_USE_LEGACY_KERAS=1.

  1. Rebuild the Farag model with src.model.build_model (tf_keras) and load the
     weights dumped by dump_keras_weights.py; check it reproduces the Keras-3
     probabilities.
  2. Export float32 / float16 / dynamic-range / INT8 PTQ / INT8 QAT /
     50 %-pruned + INT8 with HasanML/paper3/src/export.py.
     PTQ representative data: 500 beats from the DS1 training portion.
     QAT / pruning fine-tune on the DS1 training portion, balanced class
     weights (as the float model), hold-out = official P1 DS1 hold-out.
  3. Evaluate each .tflite with the TFLite interpreter (batch 1) on the DS1
     hold-out and on DS2: plain acc, OvR acc, macro F1, per-class Se/PPV,
     argmax and calibrated S threshold (tau_S), drop vs Keras.
  4. Edge metrics: .tflite bytes, param count, activation peak/sum estimate,
     host-CPU latency per beat.
  5. C headers (xxd -i layout) for every variant + a model_meta header.

Usage (from $ROOT):
    TF_USE_LEGACY_KERAS=1 .venv-export/bin/python 1DCNN/export/export_tflite.py \
        --tag P1 --weights 1DCNN/export/weights/P1_fullDS1_seed6.npz \
        --data 1DCNN/export/.cache/data_p1.npz --n-classes 3 --tau-s 0.50
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parents[1]
PAPER3 = ROOT / "HasanML" / "paper3"
sys.path.insert(0, str(PAPER3))

import tensorflow as tf  # noqa: E402
import yaml  # noqa: E402

from src import export as ex  # noqa: E402
from src.model import build_model  # noqa: E402
from src.train import compute_class_weights  # noqa: E402

CLS3 = ["N", "S", "V"]


def metrics_mc(y, proba, k, tau_s=None):
    if tau_s is None:
        pred = proba.argmax(1)
    else:
        pred = proba.argmax(1).copy()
        pred[proba[:, 1] >= tau_s] = 1
    cm = np.zeros((k, k), int)
    np.add.at(cm, (y, pred), 1)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    se = tp / np.maximum(cm.sum(1), 1)
    ppv = np.where(cm.sum(0) > 0, tp / np.maximum(cm.sum(0), 1), 0.0)
    f1 = np.where(se + ppv > 0, 2 * se * ppv / np.maximum(se + ppv, 1e-12), 0.0)
    ovr = np.mean([(n - cm[c, :].sum() - cm[:, c].sum() + 2 * cm[c, c]) / n for c in range(k)])
    names = CLS3 if k == 3 else ["Nor", "Arr"]
    out = {"acc": tp.sum() / n, "ovr_acc": ovr, "macro_f1": f1.mean(), "cm": cm.tolist()}
    for c in range(k):
        out[f"se_{names[c]}"] = se[c]
        out[f"ppv_{names[c]}"] = ppv[c]
        out[f"f1_{names[c]}"] = f1[c]
    return out


def metrics_bin(y, p_arr, t):
    from sklearn.metrics import roc_auc_score
    pred = (p_arr >= t).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    se = tp / max(tp + fn, 1); ppv = tp / max(tp + fp, 1); sp = tn / max(tn + fp, 1)
    return {"acc": (tp + tn) / len(y), "se": se, "sp": sp, "ppv": ppv,
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "auc": float(roc_auc_score(y, p_arr)),
            "cm": [[tn, fp], [fn, tp]]}


def build_from_npz(wpath, n_classes, cfg):
    w = np.load(wpath)
    m = build_model(nk=32, n_channels=1, n_classes=n_classes,
                    rr_dense_units=list(cfg["model"]["rr_dense_units"]),
                    activation=cfg["model"]["activation"], padding=cfg["model"]["padding"],
                    use_bias=bool(cfg["model"]["use_bias"]))
    for layer in m.layers:
        ws = [w[f"{layer.name}::{i}"] for i in range(len(layer.get_weights()))]
        if ws:
            layer.set_weights(ws)
    m = to_conv2d_form(m, n_classes, cfg)
    return m, w


def to_conv2d_form(m1d, n_classes, cfg):
    """
    Numerically identical graph with Conv2D instead of Conv1D:
      ecg_in (32,1) -> Reshape(1,32,1) -> Conv2D(13,(1,32),valid,relu) -> BN
      -> Reshape(13)  [== GlobalMaxPool1D since conv output length is 1]
    Needed because (a) tfmot 0.8 has no QAT support for Conv1D and
    (b) TF 2.19's int8 calibrator fails on the Conv1D lowering
    ("CONV_2D input->dims->size != 4").  The float TFLite graph of the
    Conv1D model already lowers to CONV_2D, so the MCU op set is unchanged.
    """
    keras = tf.keras
    act = cfg["model"]["activation"]
    ecg_in = keras.Input(shape=(32, 1), name="ecg_in")
    x = keras.layers.Reshape((1, 32, 1), name="to_2d")(ecg_in)
    x = keras.layers.Conv2D(13, (1, 32), padding="valid", activation=act, name="conv2d")(x)
    # BatchNorm (after ReLU) is an affine per-channel map, so it is folded
    # exactly into the first 13 input rows of the output Dense (below).
    # tfmot cannot QAT a stand-alone BN layer, and the MCU graph gets simpler.
    x = keras.layers.Reshape((13,), name="flatten13")(x)
    rr_in = keras.Input(shape=(4,), name="rr_in")
    r = rr_in
    for i, u in enumerate(cfg["model"]["rr_dense_units"]):
        r = keras.layers.Dense(u, activation=act, name=f"rr_dense_{i}")(r)
    out = keras.layers.Dense(n_classes, activation="softmax", name="output")(
        keras.layers.Concatenate(name="concat")([x, r]))
    m2 = keras.Model([ecg_in, rr_in], out, name="farag2023_c2d")
    k, b = m1d.get_layer("conv1d").get_weights()
    m2.get_layer("conv2d").set_weights([k[None, ...], b])       # (32,1,13)->(1,32,1,13)
    for n in ["rr_dense_0", "rr_dense_1", "rr_dense_2"]:
        m2.get_layer(n).set_weights(m1d.get_layer(n).get_weights())
    bn = m1d.get_layer("batch_norm")
    gamma, beta, mean, var = bn.get_weights()
    a = gamma / np.sqrt(var + bn.epsilon)
    c = beta - mean * a
    W, bias = m1d.get_layer("output").get_weights()
    W = W.copy()
    bias = bias + c @ W[:13]
    W[:13] = W[:13] * a[:, None]
    m2.get_layer("output").set_weights([W, bias])
    return m2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--tau-s", type=float, default=0.50, help="S threshold (3-class)")
    ap.add_argument("--bin-t", type=float, default=0.50, help="arrhythmia threshold (binary)")
    ap.add_argument("--qat-epochs", type=int, default=10)
    ap.add_argument("--qat-lr", type=float, default=1e-5)
    ap.add_argument("--prune-epochs", type=int, default=20)
    ap.add_argument("--prune-lr", type=float, default=1e-4)
    ap.add_argument("--variants", nargs="+",
                    default=["float32", "float16", "dynamic", "int8_ptq", "int8_qat", "pruned_int8"])
    ap.add_argument("--latency-beats", type=int, default=5000)
    ap.add_argument("--no-rr-clip", dest="rr_clip", action="store_false")
    a = ap.parse_args()

    tf.keras.utils.set_random_seed(0)
    cfg = yaml.safe_load(open(PAPER3 / "configs" / "default.yaml"))
    out_dir = HERE / a.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    K = a.n_classes
    D = np.load(a.data)
    Xe_tr, Xr_tr, y_tr = D["X_ecg_train"], D["X_rr_train"], D["y_train"]
    Xe_h, Xr_h, y_h = D["X_ecg_hold"], D["X_rr_hold"], D["y_hold"]
    Xe_t, Xr_t, y_t = D["X_ecg_ds2"], D["X_rr_ds2"], D["y_ds2"]

    model, W = build_from_npz(a.weights, K, cfg)
    p_keras_h = model.predict([Xe_h, Xr_h], batch_size=4096, verbose=0)
    p_keras_t = model.predict([Xe_t, Xr_t], batch_size=4096, verbose=0)
    rebuild_diff = float(max(np.abs(p_keras_h - W["proba_ref_hold"]).max(),
                             np.abs(p_keras_t - W["proba_ref_ds2"]).max()))
    print(f"[{a.tag}] tf_keras rebuild vs Keras-3 max |dp| = {rebuild_diff:.2e}; "
          f"params={model.count_params()}", flush=True)

    # representative set: 500 DS1-train beats (rng 42, as export_all)
    idx = np.random.default_rng(42).choice(len(y_tr), 500, replace=False)
    Xe_rep, Xr_rep = Xe_tr[idx], Xr_tr[idx]
    cw = compute_class_weights(y_tr, K)
    # RR clip for QAT / pruning fine-tuning: 99.9th percentile of DS1-train RR
    rr_clip = float(np.percentile(Xr_tr, 99.9)) if a.rr_clip else None
    print(f"[{a.tag}] rr_clip={rr_clip}", flush=True)

    tfl, extra = {}, {}
    t0 = time.perf_counter()
    for v in a.variants:
        ts = time.perf_counter()
        try:
            if v == "float32":
                tfl[v] = ex.convert_float32(model)
            elif v == "float16":
                tfl[v] = ex.convert_float16(model)
            elif v == "dynamic":
                tfl[v] = ex.convert_dynamic(model)
            elif v == "int8_ptq":
                tfl[v] = ex.convert_int8_ptq(model, Xe_rep, Xr_rep, 500)
            elif v == "int8_qat":
                m2, _ = build_from_npz(a.weights, K, cfg)
                tfl[v] = ex.convert_int8_qat(m2, Xe_tr, Xr_tr, y_tr, Xe_h, Xr_h, y_h, K,
                                             Xe_rep, Xr_rep, cfg, class_weight=cw,
                                             epochs=a.qat_epochs, lr=a.qat_lr, rr_clip=rr_clip)
                extra["qat_holdout_f1_hist"] = getattr(ex.convert_int8_qat, "last_history", None)
            elif v == "pruned_int8":
                m3, _ = build_from_npz(a.weights, K, cfg)
                tfl[v], stripped = ex.convert_pruned_int8(
                    m3, Xe_tr, Xr_tr, y_tr, Xe_h, Xr_h, y_h, K, Xe_rep, Xr_rep, cfg,
                    class_weight=cw, epochs=a.prune_epochs, lr=a.prune_lr, return_model=True,
                    rr_clip=rr_clip)
                extra["prune_holdout_f1_hist"] = getattr(ex.convert_pruned_int8, "last_history", None)
                extra["pruned_sparsity"] = ex.weight_sparsity(stripped)
                extra["pruned_keras_ds2_proba"] = stripped.predict([Xe_t, Xr_t], batch_size=4096, verbose=0)
                extra["pruned_keras_hold_proba"] = stripped.predict([Xe_h, Xr_h], batch_size=4096, verbose=0)
            print(f"[{a.tag}] {v}: {len(tfl.get(v, b''))} bytes ({time.perf_counter()-ts:.0f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            print(f"[{a.tag}] {v}: FAILED: {e!r}", flush=True)
            tfl[v] = b""
    print(f"[{a.tag}] conversions done in {time.perf_counter()-t0:.0f}s", flush=True)

    def score(p, y, which):
        if K == 3:
            return {"argmax": metrics_mc(y, p, 3), f"tauS={a.tau_s:.2f}": metrics_mc(y, p, 3, a.tau_s)}
        return {"t=0.50": metrics_bin(y, p[:, 1], 0.5), f"t={a.bin_t:.2f}": metrics_bin(y, p[:, 1], a.bin_t)}

    report = {"tag": a.tag, "weights": a.weights, "params": int(model.count_params()),
              "rebuild_max_abs_diff": rebuild_diff, "tau_s": a.tau_s, "bin_t": a.bin_t,
              "rr_clip": rr_clip, "args": vars(a),
              "qat_holdout_f1_hist": extra.get("qat_holdout_f1_hist"),
              "prune_holdout_f1_hist": extra.get("prune_holdout_f1_hist"),
              "variants": {}}
    report["variants"]["keras"] = {"hold": score(p_keras_h, y_h, "h"), "ds2": score(p_keras_t, y_t, "t")}
    if "pruned_keras_ds2_proba" in extra:
        report["variants"]["keras_pruned"] = {
            "hold": score(extra["pruned_keras_hold_proba"], y_h, "h"),
            "ds2": score(extra["pruned_keras_ds2_proba"], y_t, "t"),
            "sparsity": extra["pruned_sparsity"]}

    for v, b in tfl.items():
        if not b:
            report["variants"][v] = {"failed": True}
            continue
        (out_dir / f"{a.tag}_{v}.tflite").write_bytes(b)
        ex.generate_c_header(b, f"{a.tag.lower()}_{v}", out_dir / f"{a.tag.lower()}_{v}.h")
        eh = ex.evaluate_tflite(b, Xe_h, Xr_h, y_h, return_proba=True)
        et = ex.evaluate_tflite(b, Xe_t, Xr_t, y_t, return_proba=True)
        # latency: dedicated timing loop, 1 thread, first N DS2 beats
        lat = latency_us(b, Xe_t[: a.latency_beats], Xr_t[: a.latency_beats])
        mem = ex.tflite_memory_estimate(b)
        interp = tf.lite.Interpreter(model_content=b)
        io = {"inputs": [(d["name"], d["shape"].tolist(), np.dtype(d["dtype"]).name,
                          [float(x) for x in d["quantization"]]) for d in interp.get_input_details()],
              "outputs": [(d["name"], d["shape"].tolist(), np.dtype(d["dtype"]).name,
                           [float(x) for x in d["quantization"]]) for d in interp.get_output_details()]}
        report["variants"][v] = {
            "size_bytes": len(b), "hold": score(eh["proba"], y_h, "h"), "ds2": score(et["proba"], y_t, "t"),
            "max_abs_dp_vs_keras_ds2": float(np.abs(et["proba"] - p_keras_t).max()),
            "agree_argmax_vs_keras_ds2": float((et["proba"].argmax(1) == p_keras_t.argmax(1)).mean()),
            "latency_us": lat, "memory": mem, "io": io}
        np.save(out_dir / f"proba_ds2_{v}.npy", et["proba"].astype(np.float32))
        print(f"[{a.tag}] {v}: evaluated; lat={lat['mean_us']:.1f}us arena~{mem['activ_peak_bytes']}B", flush=True)

    def _c(o):
        if isinstance(o, dict):
            return {str(k): _c(x) for k, x in o.items()}
        if isinstance(o, (list, tuple)):
            return [_c(x) for x in o]
        if isinstance(o, np.generic):
            return o.item()
        return o
    (out_dir / "report.json").write_text(json.dumps(_c(report), indent=1))
    print_tables(report, K)
    print(f"wrote {out_dir/'report.json'}")


def latency_us(b, Xe, Xr, reps=1):
    interp = tf.lite.Interpreter(model_content=b, num_threads=1)
    interp.allocate_tensors()
    ins = interp.get_input_details()
    e = [d for d in ins if len(d["shape"]) == 3][0]
    r = [d for d in ins if len(d["shape"]) == 2][0]

    def q(x, d):
        if d["dtype"] == np.int8:
            s, z = d["quantization"]
            return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        return x.astype(np.float32)
    qe, qr = q(Xe, e), q(Xr, r)
    for i in range(50):  # warm-up
        interp.set_tensor(e["index"], qe[i:i+1]); interp.set_tensor(r["index"], qr[i:i+1]); interp.invoke()
    inv = []
    for i in range(len(qe)):
        interp.set_tensor(e["index"], qe[i:i+1]); interp.set_tensor(r["index"], qr[i:i+1])
        t = time.perf_counter_ns(); interp.invoke(); inv.append(time.perf_counter_ns() - t)
    inv = np.array(inv) / 1e3
    return {"mean_us": float(inv.mean()), "median_us": float(np.median(inv)),
            "p99_us": float(np.percentile(inv, 99)), "n": int(len(inv))}


def print_tables(rep, K):
    print("\n=== DS2 ===")
    for v, r in rep["variants"].items():
        if r.get("failed"):
            print(f"{v:14s} FAILED"); continue
        for mode, m in r["ds2"].items():
            if K == 3:
                print(f"{v:14s} {mode:10s} acc={100*m['acc']:.2f} ovr={100*m['ovr_acc']:.2f} "
                      f"mF1={100*m['macro_f1']:.2f} SeN={100*m['se_N']:.2f} SeS={100*m['se_S']:.2f} "
                      f"SeV={100*m['se_V']:.2f} PPVS={100*m['ppv_S']:.2f} PPVV={100*m['ppv_V']:.2f} "
                      f"size={r.get('size_bytes','-')}")
            else:
                print(f"{v:14s} {mode:8s} acc={100*m['acc']:.2f} Se={100*m['se']:.2f} Sp={100*m['sp']:.2f} "
                      f"PPV={100*m['ppv']:.2f} F1={100*m['f1']:.2f} AUC={m['auc']:.4f} size={r.get('size_bytes','-')}")


if __name__ == "__main__":
    main()
