"""
Train Farag Model-6 (NK=32, derivative, MF-trainable) on full DS1 with
class-weight power p and one of two validation schemes:

  rand : 15 % random-beat hold-out (rng seed 42, identical to run_official_p1.py)
  gkf  : patient-wise 5-fold split by record (cache/folds.json); one model per fold,
         validation fold used for early stopping + LR schedule; OOF probabilities kept.

Each job writes runs/<tag>.npz (val/OOF probs, DS2 probs, epochs) and runs/<tag>.keras.
Existing outputs are skipped, so reruns resume.  DS2 is only *predicted*, never
used for any choice.

Usage:
  python train_jobs.py --scheme gkf --p 0 0.25 0.5 0.75 1 --seeds 0 1 2 --workers 7
  python train_jobs.py --scheme rand --p 0 --seeds 0-9 --es loss
"""
import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np

from common import RUNS, CACHE, N_FOLDS, K, load_cfg, load_data, fold_assignment, class_weights

_D = None


def _init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    global _D
    _D = load_data()


def tag_of(scheme, p, seed, fold, es):
    t = f"{scheme}_p{p:g}_s{seed}"
    if fold >= 0:
        t += f"_f{fold}"
    if es != "f1":
        t += f"_es{es}"
    return t


def split(scheme, fold):
    y = _D["y1"]
    if scheme == "rand":
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(y)); n_val = int(len(y) * 0.15)
        return idx[n_val:], idx[:n_val], np.load(CACHE / "kernel_full.npy")
    folds = fold_assignment()["folds"]
    val_recs = np.array([int(r) for r in folds[fold]])
    vm = np.isin(_D["g1"], val_recs)
    return np.where(~vm)[0], np.where(vm)[0], np.load(CACHE / f"kernel_fold{fold}.npy")


def run_job(scheme, p, seed, fold, es):
    tag = tag_of(scheme, p, seed, fold, es)
    out = RUNS / f"{tag}.npz"
    if out.exists():
        return tag, "cached", 0.0
    import tensorflow as tf
    from tensorflow import keras
    from src.model import build_model, apply_mf_init, compile_model, validate_param_count
    from src.train import MacroF1Callback, _set_seeds

    cfg = load_cfg(); t = cfg["training"]
    Xe, Xr, y = _D["Xe1"], _D["Xr1"], _D["y1"]
    if scheme == "refit":
        return run_refit(tag, p, seed, cfg, Xe, Xr, y)
    tr, va, kernel = split(scheme, fold)
    _set_seeds(seed, deterministic=True)
    model = build_model(nk=32, n_channels=1, n_classes=K,
                        rr_dense_units=list(cfg["model"]["rr_dense_units"]),
                        activation=cfg["model"]["activation"], padding=cfg["model"]["padding"],
                        use_bias=bool(cfg["model"]["use_bias"]))
    validate_param_count(model, 32, 1, K)
    apply_mf_init(model, kernel, frozen=False)
    compile_model(model, lr=float(t["lr"]), n_classes=K)
    cw = class_weights(y[tr], p) if p > 0 else None
    oh = lambda a: keras.utils.to_categorical(a, K).astype(np.float32)
    cbs = []
    if es == "f1":
        cbs.append(MacroF1Callback(Xe[va], Xr[va], y[va], K))
        mon, mode = "val_macro_f1", "max"
    else:
        mon, mode = "val_loss", "min"
    cbs += [keras.callbacks.EarlyStopping(monitor=mon, mode=mode,
                                          patience=int(t["early_stop_patience"]),
                                          restore_best_weights=True, verbose=0),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=float(t["lr_reduce_factor"]),
                                              patience=int(t["lr_reduce_patience"]),
                                              min_lr=float(t["lr_min"]), verbose=0)]
    t0 = time.perf_counter()
    h = model.fit([Xe[tr], Xr[tr]], oh(y[tr]), validation_data=([Xe[va], Xr[va]], oh(y[va])),
                  epochs=int(t["max_epochs"]), batch_size=int(t["batch_size"]),
                  callbacks=cbs, class_weight=cw, verbose=0)
    el = time.perf_counter() - t0
    hist = h.history[mon]
    best_ep = int(np.argmax(hist) if mode == "max" else np.argmin(hist)) + 1
    pv = model.predict([Xe[va], Xr[va]], batch_size=4096, verbose=0)
    p2 = model.predict([_D["Xe2"], _D["Xr2"]], batch_size=4096, verbose=0)
    model.save(str(RUNS / f"{tag}.keras"))
    np.savez_compressed(out, val_idx=va, proba_val=pv.astype(np.float32),
                        proba_ds2=p2.astype(np.float32), best_epoch=best_ep,
                        n_epochs=len(hist), elapsed=el,
                        cw=np.array([cw[i] for i in range(K)] if cw else [1, 1, 1], float),
                        hist_mon=np.array(hist, float),
                        hist_val_loss=np.array(h.history["val_loss"], float))
    return tag, f"ep={best_ep}/{len(hist)}", el


def run_refit(tag, p, seed, cfg, Xe, Xr, y):
    """B4 refit: train on ALL of DS1 for the epoch count chosen by this seed's patient-wise
    CV (median best epoch of its 5 gkf fold models), constant LR, no validation data."""
    from tensorflow import keras
    from src.model import build_model, apply_mf_init, compile_model
    from src.train import _set_seeds
    eps = [int(np.load(RUNS / f"{tag_of('gkf', p, seed, f, 'f1')}.npz")["best_epoch"]) for f in range(N_FOLDS)]
    n_ep = int(round(float(np.median(eps))))
    t = cfg["training"]
    _set_seeds(seed, deterministic=True)
    model = build_model(nk=32, n_channels=1, n_classes=K,
                        rr_dense_units=list(cfg["model"]["rr_dense_units"]),
                        activation=cfg["model"]["activation"], padding=cfg["model"]["padding"],
                        use_bias=bool(cfg["model"]["use_bias"]))
    apply_mf_init(model, np.load(CACHE / "kernel_full.npy"), frozen=False)
    compile_model(model, lr=float(t["lr"]), n_classes=K)
    cw = class_weights(y, p) if p > 0 else None
    t0 = time.perf_counter()
    model.fit([Xe, Xr], keras.utils.to_categorical(y, K).astype(np.float32), epochs=n_ep,
              batch_size=int(t["batch_size"]), class_weight=cw, verbose=0)
    el = time.perf_counter() - t0
    p2 = model.predict([_D["Xe2"], _D["Xr2"]], batch_size=4096, verbose=0)
    model.save(str(RUNS / f"{tag}.keras"))
    np.savez_compressed(RUNS / f"{tag}.npz", proba_ds2=p2.astype(np.float32), best_epoch=n_ep,
                        n_epochs=n_ep, elapsed=el, fold_epochs=np.array(eps))
    return tag, f"refit ep={n_ep}", el


def parse_seeds(xs):
    out = []
    for x in xs:
        if "-" in x:
            a, b = x.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(x))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheme", choices=["rand", "gkf", "refit"], required=True)
    ap.add_argument("--p", nargs="+", type=float, required=True)
    ap.add_argument("--seeds", nargs="+", default=["0-2"])
    ap.add_argument("--es", choices=["f1", "loss"], default="f1")
    ap.add_argument("--workers", type=int, default=7)
    a = ap.parse_args()
    seeds = parse_seeds(a.seeds)
    folds = list(range(N_FOLDS)) if a.scheme == "gkf" else [-1]
    jobs = [(a.scheme, p, s, f, a.es) for s in seeds for p in a.p for f in folds]
    jobs = [j for j in jobs if not (RUNS / f"{tag_of(*j)}.npz").exists()]
    print(f"{len(jobs)} jobs to run", flush=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, mp_context=ctx, initializer=_init) as ex:
        futs = {ex.submit(run_job, *j): j for j in jobs}
        for i, f in enumerate(as_completed(futs), 1):
            try:
                tag, msg, el = f.result()
                print(f"[{i}/{len(jobs)} {time.time()-t0:.0f}s] {tag} {msg} {el:.0f}s", flush=True)
            except Exception as e:
                print(f"FAILED {futs[f]}: {e!r}", flush=True)
