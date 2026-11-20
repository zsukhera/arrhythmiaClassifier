"""
farag-gap A2/A3/A4: full Table-4 metric set for the official P1_fullDS1 10-seed
ensemble, metric-definition fit against Farag (2023) Table 4, and test-set
composition check.

Run (on the box, from $ROOT):
    OMP_NUM_THREADS=2 .venv/bin/python 1DCNN/gap_analysis/gap_metrics.py

Outputs (all in 1DCNN/gap_analysis/):
    ensemble_probs.npz   DS1 + DS2 ensemble/per-seed probabilities with labels,
                         record ids, R-sample positions (360 Hz and 128 Hz), symbols
    gap_metrics.json     every metric computed here
    gap_metrics.log      (stdout when run via nohup)
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXPS = ROOT / "1DCNN" / "1DCNN experiments"
PAPER3 = ROOT / "HasanML" / "paper3"
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(PAPER3))
sys.path.insert(0, str(EXPS))

import tensorflow as tf  # noqa: E402
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

from utils import load_config, load_data, threshold_predict  # noqa: E402
from src.data import DS1_TRAIN, DS1_VAL, DS2, AAMI_MAP, BEAT_SYMBOLS, load_record, task_labels  # noqa: E402
from src.run_experiment import process_record, _resolve  # noqa: E402

CLS = ["N", "S", "V"]

# ----------------------------------------------------------------------------
# Paper Table 4, Model 6 (and Model 1) values, transcribed from the PMC full text
# (PMC9919183).  Testing results, %.
# ----------------------------------------------------------------------------
PAPER = {
    "model6": {"acc": 98.18,
               "N": {"prec": 99.00, "se": 99.10, "f1": 99.05},
               "S": {"prec": 82.68, "se": 81.60, "f1": 82.14},
               "V": {"prec": 95.63, "se": 95.00, "f1": 95.31},
               "avg": {"prec": 92.44, "se": 91.90, "f1": 92.17}},
    "model1": {"acc": 96.48,
               "N": {"prec": 99.38, "se": 96.96, "f1": 98.16},
               "S": {"prec": 58.06, "se": 85.30, "f1": 69.09},
               "V": {"prec": 90.20, "se": 96.34, "f1": 93.17},
               "avg": {"prec": 82.55, "se": 92.87, "f1": 86.81}},
}
PAPER_COUNTS = {"DS1": {"N": 45866, "S": 944, "V": 3788, "F": 415, "Q": 8, "total": 51021},
                "DS2": {"N": 44259, "S": 1837, "V": 3221, "F": 388, "Q": 7, "total": 49712}}


def full_metrics(y, yp):
    k = 3
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, yp), 1)
    n = cm.sum()
    tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp
    fp = cm.sum(0) - tp
    tn = n - tp - fn - fp
    sup = cm.sum(1).astype(float)
    se = tp / (tp + fn)
    ppv = np.where(tp + fp > 0, tp / np.maximum(tp + fp, 1), 0.0)
    sp = tn / (tn + fp)
    f1 = 2 * tp / (2 * tp + fp + fn)
    w = sup / n
    m = {
        "n": int(n),
        "acc_plain": tp.sum() / n,
        "acc_ovr_mean": float(np.mean((tp + tn) / n)),
        "acc_balanced": float(se.mean()),
        "macro_se": se.mean(), "macro_ppv": ppv.mean(), "macro_sp": sp.mean(), "macro_f1": f1.mean(),
        "macro_f1_of_macroPR": 2 * ppv.mean() * se.mean() / (ppv.mean() + se.mean()),
        "weighted_se": (w * se).sum(), "weighted_ppv": (w * ppv).sum(),
        "weighted_sp": (w * sp).sum(), "weighted_f1": (w * f1).sum(),
        "micro_f1": tp.sum() / n,
        "cm": cm.tolist(),
    }
    for i, c in enumerate(CLS):
        m[f"se_{c}"] = se[i]; m[f"ppv_{c}"] = ppv[i]; m[f"sp_{c}"] = sp[i]; m[f"f1_{c}"] = f1[i]
        m[f"acc_ovr_{c}"] = (tp[i] + tn[i]) / n
    return {k_: (float(v) if isinstance(v, (np.floating, float)) else v) for k_, v in m.items()}


def record_meta(records, cfg):
    """Per-beat record id, R positions, symbols for the 3-class dataset (same order as build_dataset)."""
    rec_ids, r360, r128, syms, X_rr = [], [], [], [], []
    excluded = Counter()
    data_cfg = cfg["data"]
    for name in records:
        r = process_record(name, cfg)
        rec = load_record(name, data_dir=_resolve(data_cfg["data_dir"]),
                          mirrors=[_resolve(m) for m in data_cfg["local_mirrors"]],
                          lead=data_cfg["lead"])
        beat_idx = np.where([str(s) in BEAT_SYMBOLS for s in rec.ann_symbols])[0]
        r_all360 = rec.ann_samples[beat_idx]
        yt = task_labels(r["y_aami"], r["symbols"], "multiclass", 3, "aami")
        mask = yt >= 0
        for s, a in zip(r["symbols"][~mask], r["y_aami"][~mask]):
            excluded[(str(s), int(a))] += 1
        rec_ids += [int(name)] * int(mask.sum())
        r360.append(r_all360[mask]); r128.append(r["r_samples"][mask])
        syms += r["symbols"][mask].tolist(); X_rr.append(r["rr_features"][mask])
        # all non-beat annotations in this record (for the exclusion audit)
        for s in rec.ann_symbols:
            if str(s) not in BEAT_SYMBOLS:
                excluded[("nonbeat:" + str(s), -9)] += 1
    return (np.array(rec_ids), np.concatenate(r360), np.concatenate(r128),
            np.array(syms), np.concatenate(X_rr), excluded)


def main():
    cfg = load_config()
    ds1 = load_data("ds1_all", cfg)
    ds2 = load_data("ds2", cfg)

    print("== A4: rebuild per-beat metadata and verify against cached arrays ==")
    meta = {}
    for split, recs, d in (("ds1", DS1_TRAIN + DS1_VAL, ds1), ("ds2", DS2, ds2)):
        rid, r360, r128, sym, xrr, excl = record_meta(recs, cfg)
        assert len(rid) == len(d["y"]), (split, len(rid), len(d["y"]))
        assert np.allclose(xrr, d["X_rr"]), f"{split}: RR mismatch vs cache"
        meta[split] = dict(rid=rid, r360=r360, r128=r128, sym=sym, excl=excl)
        cnt = Counter(d["y"].tolist())
        print(f"{split}: N={cnt[0]} S={cnt[1]} V={cnt[2]} total={len(d['y'])}")
        print(f"  symbols kept: {dict(Counter(sym.tolist()))}")
        print(f"  beats/annotations excluded: {dict(sorted(((f'{s}|aami={a}', c) for (s, a), c in excl.items())))}")

    # validation hold-out used by run_official_p1.py (random 15%, rng seed 42)
    n1 = len(ds1["y"])
    idx = np.random.default_rng(42).permutation(n1)
    val_order = idx[:max(1, int(n1 * 0.15))]  # order used for proba_val in the .pkl
    is_val = np.zeros(n1, bool); is_val[val_order] = True

    print("\n== A2: predict with the 10 P1_fullDS1 models ==")
    p1, p2 = [], []
    for s in range(10):
        mp = PAPER3 / "results" / "cache" / f"P1_fullDS1_seed{s}.keras"
        model = tf.keras.models.load_model(str(mp), compile=False)
        a = model.predict([ds1["X_ecg"], ds1["X_rr"]], batch_size=4096, verbose=0)
        b = model.predict([ds2["X_ecg"], ds2["X_rr"]], batch_size=4096, verbose=0)
        pk = pickle.load(open(mp.with_suffix(".pkl"), "rb"))
        dmax = float(np.abs(pk["proba_test"] - b).max())
        dval = float(np.abs(pk["proba_val"] - a[val_order]).max())
        print(f"seed {s}: max|p - cached proba_test| = {dmax:.2e}, val {dval:.2e}")
        p1.append(a.astype(np.float32)); p2.append(b.astype(np.float32))
        tf.keras.backend.clear_session()
    p1 = np.stack(p1); p2 = np.stack(p2)
    e1 = p1.mean(0); e2 = p2.mean(0)

    np.savez_compressed(
        OUT / "ensemble_probs.npz",
        ds2_proba=e2, ds2_proba_seeds=p2, ds2_y=ds2["y"].astype(np.int8),
        ds2_record=meta["ds2"]["rid"], ds2_r360=meta["ds2"]["r360"], ds2_r128=meta["ds2"]["r128"],
        ds2_symbol=meta["ds2"]["sym"],
        ds1_proba=e1, ds1_proba_seeds=p1, ds1_y=ds1["y"].astype(np.int8),
        ds1_record=meta["ds1"]["rid"], ds1_r360=meta["ds1"]["r360"], ds1_r128=meta["ds1"]["r128"],
        ds1_symbol=meta["ds1"]["sym"], ds1_is_val=is_val,
        classes=np.array(CLS), tau_S=np.float32(0.50),
        note=np.array("P1_fullDS1 seeds0-9 (full DS1, balanced CW). ds1_* are mostly TRAINING beats "
                      "(only ds1_is_val=True beats were held out, random 15%, rng 42). DS2 is test-only."),
    )
    print(f"saved {OUT / 'ensemble_probs.npz'}")

    y2 = ds2["y"].astype(int)
    res = {"paper": PAPER, "paper_counts": PAPER_COUNTS}
    res["ds2_ensemble_argmax"] = full_metrics(y2, e2.argmax(1))
    res["ds2_ensemble_tau050"] = full_metrics(y2, threshold_predict(e2, 0.50))
    res["tau050_differs_from_argmax_n"] = int((threshold_predict(e2, 0.50) != e2.argmax(1)).sum())
    seeds = [full_metrics(y2, p2[s].argmax(1)) for s in range(10)]
    res["ds2_seed_argmax"] = seeds
    keys = [k for k in seeds[0] if k not in ("cm", "n")]
    res["ds2_seedmean_argmax"] = {k: float(np.mean([m[k] for m in seeds])) for k in keys}
    res["ds2_seedstd_argmax"] = {k: float(np.std([m[k] for m in seeds])) for k in keys}
    y1 = ds1["y"].astype(int)
    res["ds1val_ensemble_argmax"] = full_metrics(y1[is_val], e1[is_val].argmax(1))
    res["ds1all_ensemble_argmax_insample"] = full_metrics(y1, e1.argmax(1))

    # A3: definition fit using the paper's own per-class numbers
    fit = {}
    for mid, P in PAPER.items():
        f = {}
        for met in ("prec", "se", "f1"):
            f[f"macro_{met}"] = np.mean([P[c][met] for c in CLS])
            nd2 = [44259, 1837, 3221]
            f[f"weighted_{met}"] = float(np.dot(nd2, [P[c][met] for c in CLS]) / sum(nd2))
        f["f1_of_avgPR"] = 2 * P["avg"]["prec"] * P["avg"]["se"] / (P["avg"]["prec"] + P["avg"]["se"])
        f["plain_acc_from_recalls"] = float(np.dot(nd2, [P[c]["se"] for c in CLS]) / sum(nd2))
        # OvR-mean accuracy implied by recalls+precisions (reconstruct confusion totals)
        tp = np.array([nd2[i] * P[c]["se"] / 100 for i, c in enumerate(CLS)])
        pred = np.array([tp[i] / (P[c]["prec"] / 100) for i, c in enumerate(CLS)])
        n = sum(nd2); fn = np.array(nd2) - tp; fp = pred - tp; tn = n - tp - fn - fp
        f["ovr_mean_acc_implied"] = float(np.mean((tp + tn) / n) * 100)
        f["plain_acc_if_FQ_counted_wrong"] = float(tp.sum() / (n + 388 + 7) * 100)
        f["implied_pred_counts"] = pred.round(0).tolist()
        f["implied_total_pred"] = float(pred.sum())
        fit[mid] = {k: (float(v) if not isinstance(v, list) else v) for k, v in f.items()}
    res["paper_definition_fit"] = fit

    with open(OUT / "gap_metrics.json", "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    def show(name, m):
        print(f"\n--- {name} ---")
        print(f"plain acc {100*m['acc_plain']:.2f} | OvR-mean acc {100*m['acc_ovr_mean']:.2f}")
        for c in CLS:
            print(f"  {c}: Se {100*m[f'se_{c}']:.2f} PPV {100*m[f'ppv_{c}']:.2f} "
                  f"Sp {100*m[f'sp_{c}']:.2f} F1 {100*m[f'f1_{c}']:.2f}")
        print(f"  macro    Se {100*m['macro_se']:.2f} PPV {100*m['macro_ppv']:.2f} "
              f"Sp {100*m['macro_sp']:.2f} F1 {100*m['macro_f1']:.2f} (F1(avgP,avgR) {100*m['macro_f1_of_macroPR']:.2f})")
        print(f"  weighted Se {100*m['weighted_se']:.2f} PPV {100*m['weighted_ppv']:.2f} "
              f"Sp {100*m['weighted_sp']:.2f} F1 {100*m['weighted_f1']:.2f}")
        if "cm" in m:
            print(f"  CM (rows=true N,S,V): {m['cm']}")

    show("DS2 ensemble argmax", res["ds2_ensemble_argmax"])
    show("DS2 ensemble tau_S=0.50", res["ds2_ensemble_tau050"])
    print(f"tau=0.50 vs argmax differing predictions: {res['tau050_differs_from_argmax_n']}")
    show("DS2 mean of 10 single seeds (paper protocol)", res["ds2_seedmean_argmax"])
    show("DS1 random-15% val (ensemble)", res["ds1val_ensemble_argmax"])
    print("\n--- A3 definition fit on paper numbers ---")
    print(json.dumps(fit, indent=1))


if __name__ == "__main__":
    main()
