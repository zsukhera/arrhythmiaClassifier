import json, glob, os, numpy as np
rows = [("DEFAULT (full model)", "results/gpu_run/eval/eval_summary_DEFAULT.json")]
for d in ("ablations_a", "ablations_b"):
    for f in sorted(glob.glob(f"results/{d}/eval/eval_summary_A*.json")):
        rows.append((os.path.basename(f)[13:-5], f))
print("| run | seeds | DS1-val 5c macroF1 | DS2 acc | DS2 OvR | DS2 macroF1 | Se_N | Se_S | PPV_S | Se_V | PPV_V | Se_F | DS2 3c acc | 3c macroF1 |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for name, f in rows:
    d = json.load(open(f)); s = [r for r in d["results"] if r["name"].startswith("seed")]
    m = lambda key, k: (np.mean([100*r[key][k] for r in s]), np.std([100*r[key][k] for r in s]))
    def c(key, k):
        a, b = m(key, k); return f"{a:.1f}±{b:.1f}" if len(s) > 1 else f"{a:.1f}"
    print(f"| {name} | {len(s)} | {c('val_5c','macro_f1')} | {c('test_5c','accuracy')} | {c('test_5c','ovr_accuracy')} | {c('test_5c','macro_f1')} | {c('test_5c','se_N')} | {c('test_5c','se_S')} | {c('test_5c','ppv_S')} | {c('test_5c','se_V')} | {c('test_5c','ppv_V')} | {c('test_5c','se_F')} | {c('test_3c','accuracy')} | {c('test_3c','macro_f1')} |")
