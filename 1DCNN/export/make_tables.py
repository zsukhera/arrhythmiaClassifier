"""
Markdown tables from <tag>/report.json (+ tflm_report.json + TFLM probas).
Usage: .venv-export/bin/python 1DCNN/export/make_tables.py P1 [--data .cache/data_p1.npz]
Writes 1DCNN/export/<tag>/tables.md
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
tag = sys.argv[1]
data = sys.argv[2] if len(sys.argv) > 2 else str(HERE / ".cache" / f"data_{tag.lower()}.npz")
R = json.loads((HERE / tag / "report.json").read_text())
T = {}
if (HERE / tag / "tflm_report.json").exists():
    T = json.loads((HERE / tag / "tflm_report.json").read_text())
K = 3 if tag.startswith("P1") else 2
y = np.load(data)["y_ds2"]

from export_tflite import metrics_mc, metrics_bin  # noqa: E402

L = []
order = ["keras", "float32", "float16", "dynamic", "int8_ptq", "int8_qat", "keras_pruned", "pruned_int8"]
if K == 3:
    base = R["variants"]["keras"]["ds2"]
    for mode in base:
        kb = base[mode]
        L += [f"\n**DS2, {mode}** (TFLite interpreter, batch 1; Δ = vs Keras float)\n",
              "| Variant | Acc % | OvR acc % | Macro F1 % | Se N | Se S | Se V | PPV N | PPV S | PPV V | ΔAcc pp | ΔmF1 pp | hold-out mF1 % |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for v in order:
            r = R["variants"].get(v)
            if not r or r.get("failed"):
                continue
            m = r["ds2"][mode]; h = r["hold"][mode]
            L.append(f"| {v} | {100*m['acc']:.2f} | {100*m['ovr_acc']:.2f} | {100*m['macro_f1']:.2f} | "
                     f"{100*m['se_N']:.2f} | {100*m['se_S']:.2f} | {100*m['se_V']:.2f} | "
                     f"{100*m['ppv_N']:.2f} | {100*m['ppv_S']:.2f} | {100*m['ppv_V']:.2f} | "
                     f"{100*(m['acc']-kb['acc']):+.2f} | {100*(m['macro_f1']-kb['macro_f1']):+.2f} | "
                     f"{100*h['macro_f1']:.2f} |")
    # TFLM numerics
    rows = []
    for v in order:
        p = HERE / tag / f"proba_ds2_tflm_{v}.npy"
        if p.exists():
            pr = np.load(p)
            for mode, tau in (("argmax", None), (f"tauS={R['tau_s']:.2f}", R["tau_s"])):
                m = metrics_mc(y, pr, 3, tau)
                rows.append(f"| {v} | {mode} | {100*m['acc']:.2f} | {100*m['ovr_acc']:.2f} | {100*m['macro_f1']:.2f} | "
                            f"{100*m['se_N']:.2f} | {100*m['se_S']:.2f} | {100*m['se_V']:.2f} | "
                            f"{100*m['ppv_S']:.2f} | {100*m['ppv_V']:.2f} |")
    if rows:
        L += ["\n**DS2 computed by the TFLite-Micro runtime (reference kernels)**\n",
              "| Variant | Rule | Acc % | OvR % | Macro F1 % | Se N | Se S | Se V | PPV S | PPV V |",
              "|---|---|---|---|---|---|---|---|---|---|"] + rows
else:
    base = R["variants"]["keras"]["ds2"]
    for mode in base:
        kb = base[mode]
        L += [f"\n**DS2, {mode}** (TFLite interpreter, batch 1)\n",
              "| Variant | Acc % | Se % | Sp % | PPV % | F1 % | AUC | ΔAcc pp | ΔF1 pp | hold-out F1 % |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for v in order:
            r = R["variants"].get(v)
            if not r or r.get("failed"):
                continue
            m = r["ds2"][mode]; h = r["hold"][mode]
            L.append(f"| {v} | {100*m['acc']:.2f} | {100*m['se']:.2f} | {100*m['sp']:.2f} | {100*m['ppv']:.2f} | "
                     f"{100*m['f1']:.2f} | {m['auc']:.4f} | {100*(m['acc']-kb['acc']):+.2f} | "
                     f"{100*(m['f1']-kb['f1']):+.2f} | {100*h['f1']:.2f} |")
    rows = []
    for v in order:
        p = HERE / tag / f"proba_ds2_tflm_{v}.npy"
        if p.exists():
            pr = np.load(p)[:, 1]
            for t in sorted({0.5, R["bin_t"]}):
                m = metrics_bin(y, pr, t)
                rows.append(f"| {v} | t={t:.2f} | {100*m['acc']:.2f} | {100*m['se']:.2f} | {100*m['sp']:.2f} | "
                            f"{100*m['ppv']:.2f} | {100*m['f1']:.2f} | {m['auc']:.4f} |")
    if rows:
        L += ["\n**DS2 computed by the TFLite-Micro runtime**\n",
              "| Variant | Rule | Acc % | Se % | Sp % | PPV % | F1 % | AUC |", "|---|---|---|---|---|---|---|---|"] + rows

L += ["\n**Edge metrics** (params = float graph after BN folding; latency = TFLite interpreter, 1 thread, host CPU, mean/median of invoke() over 5000 DS2 beats)\n",
      "| Variant | .tflite bytes | TFLM arena bytes (host, 64-bit) | activation peak est. B | const tensor B | ops | latency mean / median µs | TFLM runs? |",
      "|---|---|---|---|---|---|---|---|"]
for v in order:
    r = R["variants"].get(v)
    if not r or r.get("failed") or "size_bytes" not in r:
        continue
    t = T.get(v, {})
    ops = [o for o in r["memory"]["op_types"] if o != "DELEGATE"]
    L.append(f"| {v} | {r['size_bytes']} | {t.get('tflm_arena_bytes_host64', '-')} | {r['memory']['activ_peak_bytes']} | "
             f"{r['memory']['const_tensor_bytes']} | {', '.join(ops)} | "
             f"{r['latency_us']['mean_us']:.2f} / {r['latency_us']['median_us']:.2f} | "
             f"{'no: ' + t['error'].strip().splitlines()[-1][:60] if t.get('error') else ('yes' if t else '-')} |")
L.append(f"\nParameters: {R['params']} (float graph after folding BN into the output Dense; the trained Keras model has +52 BN params, 26 of them non-trainable stats).")
if R["variants"].get("keras_pruned"):
    L.append(f"Pruned kernel sparsity: {R['variants']['keras_pruned']['sparsity']}")
(HERE / tag / "tables.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
