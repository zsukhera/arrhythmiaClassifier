"""
P2 / P3 binary variants with the IMPROVED recipe used by the official P1 run
(1DCNN/1DCNN experiments/run_official_p1.py):

  * full DS1 (DS1_TRAIN + DS1_VAL) for training,
  * 15 % random beat hold-out of full DS1 (rng seed 42) for early stopping,
  * class_weight = balanced  -> src.train switches early stopping to val macro-F1,
  * MF kernel from DS1_TRAIN+DS1_VAL templates (same cached kernel as official P1),
  * 10 seeds; models cached as results/cache/{P2,P3}_fullDS1_seed{n}.keras.

Binary definitions:  P2 = AAMI (N vs S/V/F/Q),  P3 = raw ('N' symbol vs everything else).

Threshold on P(arrhythmia) is chosen on the DS1 hold-out only (max F1_arr).
DS2 is evaluated once at the end and never used for any choice.

Also reports the "collapsed" official P1 ensemble (P(arr) = 1 - P(N)) on the
same binary DS2 beats, threshold chosen on the same hold-out.

Usage (from $ROOT):
    .venv/bin/python 1DCNN/export/train_binary_fullds1.py --which P2 P3
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE   = Path(__file__).resolve().parent
ROOT   = HERE.parents[1]
PAPER3 = ROOT / "HasanML" / "paper3"
EXPS   = ROOT / "1DCNN" / "1DCNN experiments"
sys.path.insert(0, str(PAPER3))
sys.path.insert(0, str(EXPS))

from utils import load_config, load_kernel_mf  # noqa: E402

CACHE = HERE / ".cache"
OUT   = HERE / "p2p3_fullds1"


def split_train_val(n, val_frac=0.15, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_val = max(1, int(n * val_frac))
    return idx[n_val:], idx[:n_val]


def load_binary(split, b_def, cfg):
    from src.data import DS1_TRAIN, DS1_VAL, DS2
    from src.run_experiment import build_dataset
    p = CACHE / f"bin_{b_def}_{split}.npz"
    if p.exists():
        d = np.load(p)
        return {k: d[k] for k in ("X_ecg", "X_rr", "y")}
    recs = {"ds1_all": DS1_TRAIN + DS1_VAL, "ds2": DS2}[split]
    d = build_dataset(recs, cfg, task="binary", classes=3, binary_def=b_def)
    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(p, X_ecg=d["X_ecg"], X_rr=d["X_rr"], y=d["y"])
    return {k: d[k] for k in ("X_ecg", "X_rr", "y")}


def fmt(m):
    return (f"acc={100*m['accuracy']:.2f} Se={100*m['sensitivity']:.2f} "
            f"Sp={100*m['specificity']:.2f} PPV={100*m['ppv']:.2f} "
            f"F1={100*m['f1_arr']:.2f} AUC={m.get('roc_auc', float('nan')):.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", nargs="+", default=["P2", "P3"])
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    args = ap.parse_args()

    from src.train import train_multi_seed
    from src.metrics import binary_metrics, find_optimal_threshold
    import tensorflow as tf

    cfg = load_config()                      # P1 overrides + balanced
    kernel = load_kernel_mf(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {}

    for name in args.which:
        b_def = {"P2": "aami", "P3": "raw"}[name]
        cfg_b = {k: dict(v) if isinstance(v, dict) else v for k, v in cfg.items()}
        cfg_b["task"] = {**cfg["task"], "name": "binary", "binary_def": b_def,
                         "init": "mf_trainable"}
        cfg_b["training"] = {**cfg["training"], "class_weight": "balanced"}

        ds1 = load_binary("ds1_all", b_def, cfg_b)
        ds2 = load_binary("ds2", b_def, cfg_b)
        ti, vi = split_train_val(len(ds1["y"]))
        print(f"[{name}] full DS1 n={len(ds1['y'])} arr={int(ds1['y'].sum())}; "
              f"train={len(ti)} hold-out={len(vi)} (arr={int(ds1['y'][vi].sum())}); "
              f"DS2 n={len(ds2['y'])} arr={int(ds2['y'].sum())}", flush=True)

        t0 = time.perf_counter()
        res = train_multi_seed(
            cfg=cfg_b,
            X_ecg_tr=ds1["X_ecg"][ti], X_rr_tr=ds1["X_rr"][ti], y_tr=ds1["y"][ti],
            X_ecg_val=ds1["X_ecg"][vi], X_rr_val=ds1["X_rr"][vi], y_val=ds1["y"][vi],
            X_ecg_test=ds2["X_ecg"], X_rr_test=ds2["X_rr"], y_test=ds2["y"],
            kernel_mf=kernel, n_classes=2, seeds=args.seeds,
            cache_dir=PAPER3 / "results" / "cache", run_name=f"{name}_fullDS1",
            deterministic=True,
        )
        print(f"[{name}] trained {len(res)} seeds in {time.perf_counter()-t0:.0f}s", flush=True)

        y_val, y_te = ds1["y"][vi], ds2["y"]
        rows = []
        for r in res:
            pv, pt = r["proba_val"][:, 1], r["proba_test"][:, 1]
            t_opt, f1v = find_optimal_threshold(pv, y_val)
            m05 = binary_metrics(y_te, (pt >= 0.5).astype(int), pt)
            mopt = binary_metrics(y_te, (pt >= t_opt).astype(int), pt)
            mv05 = binary_metrics(y_val, (pv >= 0.5).astype(int), pv)
            rows.append({"seed": r["seed"], "epochs": r["n_epochs"], "t_opt": t_opt,
                         "holdout_f1_05": mv05["f1_arr"], "holdout_auc": mv05["roc_auc"],
                         "ds2_t05": m05, "ds2_topt": mopt})
            print(f"[{name}] seed {r['seed']} ep={r['n_epochs']} holdoutF1={100*mv05['f1_arr']:.2f} "
                  f"| t=0.5 {fmt(m05)} | t={t_opt:.2f} {fmt(mopt)}", flush=True)

        # ensemble
        pv = np.mean([r["proba_val"][:, 1] for r in res], axis=0)
        pt = np.mean([r["proba_test"][:, 1] for r in res], axis=0)
        t_ens, _ = find_optimal_threshold(pv, y_val)
        ens05 = binary_metrics(y_te, (pt >= 0.5).astype(int), pt)
        ensopt = binary_metrics(y_te, (pt >= t_ens).astype(int), pt)
        print(f"[{name}] ENSEMBLE t=0.5 {fmt(ens05)}", flush=True)
        print(f"[{name}] ENSEMBLE t={t_ens:.2f} {fmt(ensopt)}", flush=True)

        # collapsed official P1 ensemble on the same binary beats
        coll = {}
        p1_paths = sorted((PAPER3 / "results" / "cache").glob("P1_fullDS1_seed*.keras"))
        if p1_paths:
            pv3, pt3 = [], []
            for p in p1_paths:
                mdl = tf.keras.models.load_model(str(p), compile=False)
                pv3.append(1 - mdl.predict([ds1["X_ecg"][vi], ds1["X_rr"][vi]], batch_size=4096, verbose=0)[:, 0])
                pt3.append(1 - mdl.predict([ds2["X_ecg"], ds2["X_rr"]], batch_size=4096, verbose=0)[:, 0])
            pv3, pt3 = np.mean(pv3, 0), np.mean(pt3, 0)
            t_c, _ = find_optimal_threshold(pv3, y_val)
            coll = {"t_opt": t_c,
                    "ds2_t05": binary_metrics(y_te, (pt3 >= 0.5).astype(int), pt3),
                    "ds2_topt": binary_metrics(y_te, (pt3 >= t_c).astype(int), pt3)}
            print(f"[{name}] collapsed P1 ens (n={len(p1_paths)}) t=0.5 {fmt(coll['ds2_t05'])}", flush=True)
            print(f"[{name}] collapsed P1 ens t={t_c:.2f} {fmt(coll['ds2_topt'])}", flush=True)
            print("  NOTE: P1 hold-out beats overlap this hold-out only partially "
                  "(different label filtering); P1 saw most of these beats in training.", flush=True)

        summary[name] = {"per_seed": rows, "ensemble": {"t_opt": t_ens, "ds2_t05": ens05,
                         "ds2_topt": ensopt}, "collapsed_p1": coll}

    def _clean(o):
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items() if k != "cm"}
        if isinstance(o, list):
            return [_clean(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    (OUT / "summary.json").write_text(json.dumps(_clean(summary), indent=1))
    print(f"wrote {OUT/'summary.json'}")


if __name__ == "__main__":
    main()
