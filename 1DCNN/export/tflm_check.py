"""
Run every exported .tflite through the real TensorFlow Lite *Micro* runtime
(pip package `tflite-micro`, reference kernels, host build) to get:
  * the exact tensor-arena bytes the MicroInterpreter needs
    (RecordingMicroAllocator; host is 64-bit, so the pointer-holding structs
    are somewhat smaller on the 32-bit ESP32-S3 -> this is an upper bound),
  * DS2 metrics computed with TFLM itself (what the MCU will compute, modulo
    ESP-NN optimised kernels), and agreement with the TFLite interpreter.

Usage (from $ROOT):
    .venv-export/bin/python 1DCNN/export/tflm_check.py --tag P1 --data 1DCNN/export/.cache/data_p1.npz
Writes 1DCNN/export/<tag>/tflm_report.json
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def arena_bytes(path):
    code = (
        "from tflite_micro.python.tflite_micro import runtime as r\n"
        f"it=r.Interpreter.from_bytes(open({str(path)!r},'rb').read(), arena_size=65536)\n"
        "it.print_allocations()\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    txt = out.stdout + out.stderr
    m = re.search(r"Arena allocation total (\d+) bytes", txt)
    detail = [l.split("] ", 1)[1] for l in txt.splitlines() if "RecordingMicroAllocator" in l]
    return (int(m.group(1)) if m else None), detail, (None if m else txt[-2000:])


def run_tflm(path, Xe, Xr):
    from tflite_micro.python.tflite_micro import runtime as r
    it = r.Interpreter.from_bytes(open(path, "rb").read(), arena_size=65536)
    ins = [it.get_input_details(i) for i in range(2)]
    od = it.get_output_details(0)

    def q(x, d):
        if d["dtype"] == np.int8:
            s = d["quantization_parameters"]["scales"][0]; z = d["quantization_parameters"]["zero_points"][0]
            return np.clip(np.round(x / s + z), -128, 127).astype(np.int8)
        return x.astype(np.float32)
    order = ["ecg" if len(d["shape"]) == 3 else "rr" for d in ins]
    X = {"ecg": Xe, "rr": Xr}
    qx = [q(X[k], d) for k, d in zip(order, ins)]
    outs = []
    for i in range(len(Xe)):
        for j in range(2):
            it.set_input(qx[j][i:i + 1], j)
        it.invoke()
        o = it.get_output(0)
        if od["dtype"] == np.int8:
            s = od["quantization_parameters"]["scales"][0]; z = od["quantization_parameters"]["zero_points"][0]
            o = (o.astype(np.float32) - z) * s
        outs.append(o[0])
    return np.array(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--data", required=True)
    a = ap.parse_args()
    d = np.load(a.data)
    Xe, Xr, y = d["X_ecg_ds2"], d["X_rr_ds2"], d["y_ds2"]
    rep = {}
    for p in sorted((HERE / a.tag).glob("*.tflite")):
        v = p.stem.split("_", 1)[1]
        arena, detail, err = arena_bytes(p)
        r = {"tflite_bytes": p.stat().st_size, "tflm_arena_bytes_host64": arena,
             "arena_detail": detail}
        if err:
            r["error"] = err
            print(v, "TFLM FAILED:", err[-300:], flush=True)
        else:
            try:
                pr = run_tflm(str(p), Xe, Xr)
                np.save(HERE / a.tag / f"proba_ds2_tflm_{v}.npy", pr.astype(np.float32))
                ref = np.load(HERE / a.tag / f"proba_ds2_{v}.npy")
                r["tflm_vs_tflite_max_abs_dp"] = float(np.abs(pr - ref).max())
                r["tflm_vs_tflite_argmax_agree"] = float((pr.argmax(1) == ref.argmax(1)).mean())
                r["tflm_ds2_acc_argmax"] = float((pr.argmax(1) == y).mean())
            except Exception as e:  # noqa: BLE001
                r["error"] = repr(e)
            print(v, {k: r[k] for k in r if k != "arena_detail"}, flush=True)
        rep[v] = r
    (HERE / a.tag / "tflm_report.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
