"""Aggregate LSTM runs -> LSTM/results/summary.md + summary.json.

Per experiment: per-seed metrics, mean +- std over seeds, and the seed-ensemble
(mean of softmax probabilities, argmax; no threshold tuning).
"""
import json, glob
from pathlib import Path
import numpy as np
from lstm_common import HERE, metrics

RUNS = HERE / "runs"; OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
pct = lambda v: f"{100*v:.2f}"
lines, summ = [], {}
for d in sorted(p for p in RUNS.iterdir() if p.is_dir()):
    ms = [json.load(open(f)) for f in sorted(glob.glob(str(d / "seed*/metrics.json")))]
    if not ms:
        continue
    cls = ms[0]["classes"]
    probs = [np.load(d / f"seed{m['seed']}/test_probs.npy") for m in ms]
    y = np.load(d / f"seed{ms[0]['seed']}/test_y.npy")
    ens = metrics(y, np.mean(probs, 0).argmax(1), cls)
    keys = ["acc", "ovr_acc", "macro_f1", "macro_prec", "macro_se"]
    agg = {k: (float(np.mean([m[k] for m in ms])), float(np.std([m[k] for m in ms]))) for k in keys}
    pc = {c: {k: (float(np.mean([m["per_class"][c][k] for m in ms])), float(np.std([m["per_class"][c][k] for m in ms])))
              for k in ["se", "ppv", "f1"]} for c in cls}
    summ[d.name] = dict(n_seeds=len(ms), params=ms[0]["params"], mean_std=agg, per_class=pc, ensemble=ens,
                        train_time_s=[m["train_time_s"] for m in ms], best_epochs=[m["best_epoch"] for m in ms],
                        per_seed=[{k: m[k] for k in keys + ["seed", "best_epoch", "best_val_macro_f1"]} for m in ms])
    L = lines
    L.append(f"\n## {d.name}  (seeds={len(ms)}, params={ms[0]['params']:,}, n_test={ens['n']:,}, "
             f"train time/seed={np.mean([m['train_time_s'] for m in ms]):.0f}s, epochs={[m['epochs_run'] for m in ms]}, "
             f"best epochs={[m['best_epoch'] for m in ms]})\n")
    L.append("| | Acc | OvR Acc | Macro F1 | Macro PREC | Macro Se | " + " | ".join(f"Se_{c} | PPV_{c} | F1_{c}" for c in cls) + " |")
    L.append("|" + "---|" * (6 + 3 * len(cls)))
    for m in ms:
        L.append(f"| seed {m['seed']} | " + " | ".join(pct(m[k]) for k in keys) + " | " +
                 " | ".join(f"{pct(m['per_class'][c]['se'])} | {pct(m['per_class'][c]['ppv'])} | {pct(m['per_class'][c]['f1'])}" for c in cls) + " |")
    L.append("| **mean ± std** | " + " | ".join(f"{pct(a)} ± {pct(s)}" for a, s in agg.values()) + " | " +
             " | ".join(" | ".join(f"{pct(pc[c][k][0])} ± {pct(pc[c][k][1])}" for k in ["se", "ppv", "f1"]) for c in cls) + " |")
    L.append("| **ensemble** | " + " | ".join(pct(ens[k]) for k in keys) + " | " +
             " | ".join(f"{pct(ens['per_class'][c]['se'])} | {pct(ens['per_class'][c]['ppv'])} | {pct(ens['per_class'][c]['f1'])}" for c in cls) + " |")
    L.append("\nEnsemble confusion matrix (rows = true, cols = predicted):\n")
    L.append("| true \\ pred | " + " | ".join(cls) + " | support |"); L.append("|" + "---|" * (len(cls) + 2))
    for c, row in zip(cls, ens["cm"]):
        L.append(f"| {c} | " + " | ".join(f"{v:,}" for v in row) + f" | {sum(row):,} |")
(OUT / "summary.md").write_text("# LSTM baseline — all runs\n" + "\n".join(lines) + "\n")
json.dump(summ, open(OUT / "summary.json", "w"), indent=1)
print((OUT / "summary.md").read_text())
