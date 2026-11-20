"""Evaluate model bundles on DS2 with evaluate.py's metric set. Usage: eval_models.py name=path.pkl ..."""
import sys, json, pickle
from common import *
res = {}
for arg in sys.argv[1:]:
    name, path = arg.split("=", 1)
    b = pickle.load(open(path, "rb"))
    feats = (b.get("run_opts") or {}).get("features") or b.get("features")
    m = b["model"]; m.set_params(n_jobs=N_JOBS)
    r = eval_ds2(m, feats)
    r["cv_score"] = b.get("cv_score"); r["params"] = b.get("best_params"); r["run_opts"] = b.get("run_opts")
    res[name] = r
    print(fmt(r, name), f"| binP={r['bin_p']:.3f} binR={r['bin_r']:.3f} cv={r['cv_score']}", flush=True)
    for c in LABELS:
        pc = r["per_class"][c]; print(f"     {c}: P={pc['p']:.3f} R={pc['r']:.3f} F1={pc['f1']:.3f} n={pc['n']}")
out = OUT_DIR / "eval_ds2.json"
old = json.load(open(out)) if out.exists() else {}
old.update(res); json.dump(old, open(out, "w"), indent=1, default=str)
