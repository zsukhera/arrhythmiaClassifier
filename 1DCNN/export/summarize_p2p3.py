"""
Summarise the P2/P3 binary runs on DS2 (Se/Sp/PPV/F1/acc/AUC), mean ± std over
seeds and the probability-averaged ensemble, at t=0.5 and at a threshold chosen
on the run's own DS1 validation data (never DS2):

  paper-default : results/cache/{P2,P3}_dedicated_seed*.pkl  (val = DS1-val records)
  improved      : results/cache/{P2,P3}_fullDS1_seed*.pkl    (val = 15 % DS1 hold-out)

Usage (from $ROOT):  .venv/bin/python 1DCNN/export/summarize_p2p3.py
Writes 1DCNN/export/p2p3_summary.md and .json
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER3 = ROOT / "HasanML" / "paper3"
sys.path.insert(0, str(PAPER3))
sys.path.insert(0, str(HERE))

from src.metrics import binary_metrics, find_optimal_threshold  # noqa: E402

CACHE = HERE / ".cache"
KEYS = ["sensitivity", "specificity", "ppv", "f1_arr", "accuracy", "roc_auc"]


def labels_ds1val(b_def):
    p = CACHE / f"bin_{b_def}_ds1val.npz"
    if p.exists():
        return np.load(p)["y"]
    import yaml
    from src.data import DS1_VAL
    from src.run_experiment import build_dataset
    cfg = yaml.safe_load(open(PAPER3 / "configs" / "default.yaml"))
    d = build_dataset(DS1_VAL, cfg, "binary", 3, b_def)
    np.savez_compressed(p, y=d["y"])
    return d["y"]


def holdout_labels(b_def):
    from prepare_data import split
    y = np.load(CACHE / f"bin_{b_def}_ds1_all.npz")["y"]
    _, vi = split(len(y))
    return y[vi]


def summarise(run, b_def, y_val):
    y_te = np.load(CACHE / f"bin_{b_def}_ds2.npz")["y"]
    rs = [pickle.load(open(p, "rb")) for p in
          sorted((PAPER3 / "results" / "cache").glob(f"{run}_seed*.pkl"),
                 key=lambda p: int(p.stem.split("seed")[1]))]
    per = {"t05": [], "tval": []}
    for r in rs:
        pv, pt = r["proba_val"][:, 1], r["proba_test"][:, 1]
        assert len(pt) == len(y_te) and len(pv) == len(y_val)
        t, _ = find_optimal_threshold(pv, y_val)
        per["t05"].append(binary_metrics(y_te, (pt >= .5).astype(int), pt))
        per["tval"].append(binary_metrics(y_te, (pt >= t).astype(int), pt) | {"t": t})
    pv = np.mean([r["proba_val"][:, 1] for r in rs], 0)
    pt = np.mean([r["proba_test"][:, 1] for r in rs], 0)
    t, _ = find_optimal_threshold(pv, y_val)
    ens = {"t05": binary_metrics(y_te, (pt >= .5).astype(int), pt),
           "tval": binary_metrics(y_te, (pt >= t).astype(int), pt) | {"t": t}}
    out = {"n_seeds": len(rs), "n_ds2": int(len(y_te)), "n_arr_ds2": int(y_te.sum())}
    for mode in ("t05", "tval"):
        out[f"mean_{mode}"] = {k: float(np.nanmean([m[k] for m in per[mode]])) for k in KEYS}
        out[f"std_{mode}"] = {k: float(np.nanstd([m[k] for m in per[mode]])) for k in KEYS}
        out[f"ens_{mode}"] = {k: float(ens[mode][k]) for k in KEYS}
    out["ens_t"] = t
    out["per_seed_t05"] = [{k: float(m[k]) for k in KEYS} for m in per["t05"]]
    out["per_seed_tval_t"] = [m["t"] for m in per["tval"]]
    return out


def main():
    res = {}
    for name, b_def in (("P2", "aami"), ("P3", "raw")):
        if list((PAPER3 / "results" / "cache").glob(f"{name}_dedicated_seed*.pkl")):
            res[f"{name} paper-default"] = summarise(f"{name}_dedicated", b_def, labels_ds1val(b_def))
        if list((PAPER3 / "results" / "cache").glob(f"{name}_fullDS1_seed*.pkl")):
            res[f"{name} improved (fullDS1+balanced)"] = summarise(f"{name}_fullDS1", b_def,
                                                                     holdout_labels(b_def))
    (HERE / "p2p3_summary.json").write_text(json.dumps(res, indent=1))
    L = ["| Run | Threshold | Se % | Sp % | PPV % | F1 % | Acc % | AUC |", "|---|---|---|---|---|---|---|---|"]
    for name, r in res.items():
        for mode, lab in (("t05", "0.5"), ("tval", "val-opt")):
            m, s = r[f"mean_{mode}"], r[f"std_{mode}"]
            L.append(f"| {name} ({r['n_seeds']} seeds, mean±sd) | {lab} | "
                     + " | ".join(f"{100*m[k]:.2f}±{100*s[k]:.2f}" for k in KEYS[:5])
                     + f" | {m['roc_auc']:.4f}±{s['roc_auc']:.4f} |")
            e = r[f"ens_{mode}"]
            tl = lab if mode == "t05" else f"val-opt {r['ens_t']:.2f}"
            L.append(f"| {name} ensemble | {tl} | "
                     + " | ".join(f"{100*e[k]:.2f}" for k in KEYS[:5]) + f" | {e['roc_auc']:.4f} |")
    (HERE / "p2p3_summary.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
