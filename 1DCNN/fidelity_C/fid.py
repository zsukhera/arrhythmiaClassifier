"""
fidelity_C driver — pipeline-fidelity ablations for the Farag (2023) replication.

Runs named variants (config overrides on HasanML/paper3/configs/default.yaml)
for several seeds in parallel CPU worker processes (1 TF thread each), using the
SAME code path as `python -m src.run_experiment` (build_dataset, build_mf_kernel,
train.train_one, metrics.multiclass_metrics).

DS2 is used only for the final per-run evaluation.  Early stopping uses
DS1-val (records >= 200).  Nothing is tuned on DS2.

Usage (on the box, from 1DCNN/fidelity_C/):
    ../../.venv/bin/python fid.py --group ladder --seeds 0 1 2 --workers 6
    ../../.venv/bin/python fid.py --list
"""
import argparse
import copy
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
P3 = ROOT / "HasanML" / "paper3"
RUNS = HERE / "runs"
CACHE = HERE / "cache"

from variants import VARIANTS, GROUPS  # noqa: E402


# ---------------------------------------------------------------- helpers
def set_dotted(cfg, key, val):
    d = cfg
    parts = key.split(".")
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = val


def resolve_cfg(overrides):
    sys.path.insert(0, str(P3))
    from src.run_experiment import load_config
    cfg = load_config(P3 / "configs" / "default.yaml")
    cfg = copy.deepcopy(cfg)
    for k, v in overrides.items():
        if k.startswith("_"):
            continue
        set_dotted(cfg, k, v)
    cfg["output"]["plot_templates"] = False
    return cfg


def _hash(obj):
    return hashlib.md5(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def data_key(cfg):
    return _hash({"data": cfg["data"], "pre": cfg["preprocess"], "inp": cfg["input"],
                  "rr": cfg["rr"], "cls": cfg["task"]["classes"]})


def kernel_key(cfg, ov):
    return _hash({"data": cfg["data"], "pre": cfg["preprocess"], "inp": cfg["input"],
                  "tpl": cfg["templates"], "cls": cfg["task"]["classes"],
                  "aami": ov.get("_aami_templates", False),
                  "trrecs": ov.get("_train_recs")})


def get_data(cfg, ov=None):
    """Build (or load cached) DS1-train / DS1-val / DS2 arrays for cfg.
    ov['_train_recs'] / ov['_val_recs'] override the record lists (fold CV)."""
    ov = ov or {}
    sys.path.insert(0, str(P3))
    from src.run_experiment import build_dataset, train_records
    from src.data import DS1_VAL, DS2
    key = data_key(cfg)
    tag = cfg['data'].get('train_records', 'ds1_train')
    if ov.get("_train_recs"):
        tag = "custom" + _hash([ov["_train_recs"], ov.get("_val_recs")])
    path = CACHE / f"data_{key}_{tag}.npz"
    if path.exists():
        z = np.load(path)
        return {s: {"X_ecg": z[f"{s}_X_ecg"], "X_rr": z[f"{s}_X_rr"], "y": z[f"{s}_y"]}
                for s in ("tr", "val", "te")}
    classes = int(cfg["task"]["classes"])
    out = {}
    tr_recs = ov.get("_train_recs") or train_records(cfg)
    va_recs = ov.get("_val_recs") or DS1_VAL
    for s, recs in (("tr", tr_recs), ("val", va_recs), ("te", DS2)):
        d = build_dataset(recs, cfg, "multiclass", classes, "aami")
        out[s] = {k: d[k] for k in ("X_ecg", "X_rr", "y")}
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npz")
    np.savez(tmp, **{f"{s}_{k}": v for s, d in out.items() for k, v in d.items()})
    os.replace(tmp, path)
    return out


def get_kernel(cfg, ov):
    sys.path.insert(0, str(P3))
    from src.run_experiment import build_mf_kernel, compute_template_inputs
    key = kernel_key(cfg, ov)
    path = CACHE / f"kernel_{key}.npy"
    if path.exists():
        return np.load(path)
    if ov.get("_train_recs"):
        # fold CV: templates from the fold's training records only
        from src.run_experiment import build_template_data, _crop_template_segs
        from src.templates import compute_subclass_templates, templates_to_kernel
        from src.data import AAMI_MAP, AAMI_CLASSES
        nk = int(cfg["input"]["nk"])
        segs = build_template_data(list(ov["_train_recs"]), cfg)
        if not bool(cfg["input"].get("crop_input", True)):
            segs = _crop_template_segs(segs, nk)
        sub = list(cfg["templates"]["sub_classes"])
        if cfg["templates"].get("match_classes"):
            keep = set(AAMI_CLASSES[:int(cfg["task"]["classes"])])
            sub = [x for x in sub if AAMI_MAP.get(x) in keep]
        tpl = compute_subclass_templates(segs, nk, sub, min_beats=int(cfg["templates"]["min_beats"]),
                                         normalize_l2=bool(cfg["templates"]["normalize_l2"]),
                                         empty_fallback=str(cfg["templates"].get("empty_fallback", "zero")))
        n_ch = {"signal": 1, "derivative": 1, "both": 2}[cfg["input"]["mode"]]
        kern = templates_to_kernel(tpl, sub, n_ch)
    elif ov.get("_aami_templates", False):
        # 5 (or K) AAMI super-class templates instead of sub-class templates
        from src.templates import compute_aami_templates, templates_to_kernel
        from src.data import AAMI_CLASSES
        segs_p, _, _, nk = compute_template_inputs(cfg)
        tpl = compute_aami_templates(segs_p, nk,
                                     normalize_l2=bool(cfg["templates"]["normalize_l2"]))
        keep = AAMI_CLASSES[:int(cfg["task"]["classes"])] if cfg["templates"].get("match_classes") \
            else AAMI_CLASSES
        n_ch = {"signal": 1, "derivative": 1, "both": 2}[cfg["input"]["mode"]]
        kern = templates_to_kernel(tpl, list(keep), n_ch)
    else:
        kern = build_mf_kernel(cfg)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + f".{os.getpid()}.tmp.npy")
    np.save(tmp, kern)
    os.replace(tmp, path)
    return kern


# ---------------------------------------------------------------- worker
def run_job(job):
    name, seed = job
    ov = VARIANTS[name]
    out_json = RUNS / name / f"seed{seed}.json"
    if out_json.exists():
        return name, seed, "cached"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(1)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        pass
    sys.path.insert(0, str(P3))
    from src.train import train_one, predict
    from src.metrics import multiclass_metrics

    cfg = resolve_cfg(ov)
    d = get_data(cfg, ov)
    kern = get_kernel(cfg, ov)
    classes = int(cfg["task"]["classes"])
    t0 = time.time()
    model, hist, el = train_one(cfg, d["tr"]["X_ecg"], d["tr"]["X_rr"], d["tr"]["y"],
                                d["val"]["X_ecg"], d["val"]["X_rr"], d["val"]["y"],
                                kern, classes, seed, deterministic=True)
    yp_te, pr_te = predict(model, d["te"]["X_ecg"], d["te"]["X_rr"])
    yp_va, pr_va = predict(model, d["val"]["X_ecg"], d["val"]["X_rr"])
    m_te = multiclass_metrics(d["te"]["y"], yp_te, pr_te, k=classes)
    m_va = multiclass_metrics(d["val"]["y"], yp_va, pr_va, k=classes)
    h = hist.history
    rec = {"variant": name, "seed": seed, "overrides": ov,
           "n_epochs": len(h["loss"]), "train_s": el,
           "best_val_loss": float(np.min(h.get("val_loss", [np.nan]))),
           "best_epoch_f1": (int(np.argmax(h["val_macro_f1"])) + 1) if "val_macro_f1" in h else None,
           "best_epoch_loss": int(np.argmin(h["val_loss"])) + 1 if "val_loss" in h else None,
           "train_acc_last": float(h.get("accuracy", [np.nan])[-1]),
           "n_params": int(model.count_params()),
           "n_train": int(len(d["tr"]["y"])),
           "metrics_test": m_te, "metrics_val": m_va}
    (RUNS / name).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(RUNS / name / f"seed{seed}_proba.npz",
                        proba_test=pr_te.astype(np.float32), proba_val=pr_va.astype(np.float32),
                        y_test=d["te"]["y"], y_val=d["val"]["y"])
    tmp = out_json.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, default=float, indent=1))
    os.replace(tmp, out_json)
    return name, seed, f"done {time.time()-t0:.0f}s ep={rec['n_epochs']} " \
        f"acc={m_te['accuracy']*100:.2f} seS={m_te['se_S']*100:.1f} seV={m_te['se_V']*100:.1f}"


def prep_job(name):
    ov = VARIANTS[name]
    cfg = resolve_cfg(ov)
    get_data(cfg, ov)
    get_kernel(cfg, ov)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", nargs="+", default=[])
    ap.add_argument("--variants", nargs="+", default=[])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        for g, vs in GROUPS.items():
            print(g, vs)
        return
    names = list(a.variants)
    for g in a.group:
        names += [v for v in GROUPS[g] if v not in names]
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    # Phase 1: data + kernels (distinct keys only, in parallel)
    seen, uniq = set(), []
    for n in names:
        cfg = resolve_cfg(VARIANTS[n])
        k = (data_key(cfg), kernel_key(cfg, VARIANTS[n]), cfg["data"].get("train_records"),
             str(VARIANTS[n].get("_train_recs")), str(VARIANTS[n].get("_val_recs")))
        if k not in seen:
            seen.add(k)
            uniq.append(n)
    print(f"[prep] {len(uniq)} distinct data/kernel configs", flush=True)
    with ctx.Pool(min(a.workers, len(uniq))) as pool:
        for n in pool.imap_unordered(prep_job, uniq):
            print("[prep]", n, flush=True)
    jobs = [(n, s) for s in a.seeds for n in names]
    print(f"[train] {len(jobs)} jobs", flush=True)
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(run_job, jobs):
            print("[train]", *r, flush=True)


if __name__ == "__main__":
    main()
