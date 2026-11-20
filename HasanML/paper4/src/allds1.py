# %% [markdown]
# ## allds1.py — KAN-Former trained on ALL of DS1, settings chosen by
# ## patient-grouped CV on DS1 (kanformer2, 2026-11-19)
#
# Why: the Farag "<200" DS1-train split has only 149 S beats; the 1D-CNN
# only reached Se_S ~81 % once it was trained on all of DS1 (944 S beats).
# When all of DS1 is training data there is no DS1-val any more, so every
# choice (variant, epoch count, temperature) is made with 5-fold
# GroupKFold-by-record CV on DS1 (the S/V-balanced folds of
# 1DCNN/improve_B/cache/folds.json), then the model is refit on all of DS1.
# DS2 is used ONLY by `--mode refit` for the final report.
#
# Differences from train.py (same model/loss/EMA/schedule code):
#   * the whole dataset lives on the GPU and augmentation runs on the GPU
#     (train.py was DataLoader-bound at 12-20 s/epoch);
#   * no early stopping: a fixed cosine schedule of `epochs` epochs, the
#     held-out fold is scored after every epoch, and the epoch count E* is the
#     argmax of the pooled out-of-fold (OOF) N/S/V macro-F1;
#   * time-stretch is a real stretch about the R peak (augment.time_stretch
#     stretches and then un-stretches, i.e. it is almost a no-op).
#
# Usage (from HasanML/paper4):
#   PY=../../.venv-torch/bin/python
#   $PY -m src.allds1 --mode cache
#   $PY -m src.allds1 --mode cv     --variant V1_plain --seeds 0
#   $PY -m src.allds1 --mode cvsum  --variant V1_plain
#   $PY -m src.allds1 --mode refit  --variant V1_plain --seeds 0 1 2
#   $PY -m src.allds1 --mode table  --variant V0_fixB V1_plain ...

# %%
import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

try:
    from .run_experiment import load_config, _resolve, PROJECT_ROOT
    from .data import DS1, DS2, load_record, aami_labels, AAMI_CLASSES
    from .preprocess import bandpass_filter, extract_window
    from .rr import compute_rr_features_ext, RR_EXT_NAMES
    from .model import build_model, count_parameters
    from .train import EMA, sampler_target_proportions, cosine_schedule_with_warmup
    from .losses import compute_class_alpha, FocalLoss
    from .calibrate import TemperatureScaler, compute_ece
    from .metrics import multiclass_metrics, ovr_accuracy
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.run_experiment import load_config, _resolve, PROJECT_ROOT
    from src.data import DS1, DS2, load_record, aami_labels, AAMI_CLASSES
    from src.preprocess import bandpass_filter, extract_window
    from src.rr import compute_rr_features_ext, RR_EXT_NAMES
    from src.model import build_model, count_parameters
    from src.train import EMA, sampler_target_proportions, cosine_schedule_with_warmup
    from src.losses import compute_class_alpha, FocalLoss
    from src.calibrate import TemperatureScaler, compute_ece
    from src.metrics import multiclass_metrics, ovr_accuracy

log = logging.getLogger("p4.allds1")

P4 = PROJECT_ROOT
ROOT = P4.parents[1]
FOLDS_JSON = ROOT / "1DCNN" / "improve_B" / "cache" / "folds.json"
OUT = P4 / "results" / "allds1"
CACHE = OUT / "cache"

# %%
# Variant knobs.  BASE = Fix B regularisation of reports/kanformer.md (chosen
# there on DS1-val) with the plain recipe that made the 1D-CNN work:
# natural class distribution + unweighted CE.
BASE = dict(
    norm="zscore",          # zscore | none   (per-window, task a)
    sampler="natural",      # natural | balanced (capped inverse-frequency)
    loss="ce",              # ce (unweighted) | focal (sqrt-alpha, gamma 2)
    head="kan",             # kan | mlp
    context=True,           # 3 s context window branch
    rr="base6",             # base6 | ext11 (rr.RR_EXT_NAMES)
    rr_branch=False,        # dedicated RR MLP encoder
    rr_aux=False,           # + direct RR-only logits
    dropout=0.3, stoch_depth=0.2, head_dropout=0.4,
    wd=0.1, lr=1e-4, s_cap=0.15, v_cap=0.15, f_cap=0.05, q_cap=0.01,
    gamma=2.0, epochs=40, warmup=3, batch=256, ema=0.999, aug=True,
)
VARIANTS = {
    "V0_fixB":       dict(sampler="balanced", loss="focal"),   # previous best recipe
    "V1_plain":      dict(),
    "V1_noz":        dict(norm="none"),
    "V1_mlp":        dict(head="mlp"),
    "V1_noctx":      dict(context=False),
    "V1_mlp_noctx":  dict(head="mlp", context=False),
    "V1_rrext":      dict(rr="ext11"),
    "V1_rrbranch":   dict(rr_branch=True, rr_aux=True),
    "V1_rrext_branch": dict(rr="ext11", rr_branch=True, rr_aux=True),
    "V0_mlp_noctx":  dict(sampler="balanced", loss="focal", head="mlp", context=False),
}


def get_variant(name: str) -> dict:
    """VARIANTS entry, or composite 'A+B' (later keys win), over BASE."""
    v = dict(BASE)
    for part in name.split("+"):
        if part not in VARIANTS:
            raise KeyError(f"unknown variant {part!r}; known: {list(VARIANTS)}")
        v.update(VARIANTS[part])
    return v


def model_cfg(v: dict) -> dict:
    cfg = load_config(P4 / "configs" / "default.yaml")
    m = cfg["model"]
    m.update(head=v["head"], dropout=v["dropout"], stoch_depth=v["stoch_depth"],
             head_dropout=v["head_dropout"], rr_branch=v["rr_branch"],
             rr_aux_logits=v["rr_aux"])
    cfg["preprocess"]["context_len"] = 1080 if v["context"] else 0
    cfg["preprocess"]["context_half"] = 540 if v["context"] else 0
    cfg["rr"]["n_features"] = 6 if v["rr"] == "base6" else 11
    cfg["training"].update(lr=v["lr"], weight_decay=v["wd"], batch_size=v["batch"],
                           sampler=v["sampler"], loss=v["loss"], max_epochs=v["epochs"])
    cfg["kanformer2_variant"] = v
    return cfg


# %%
# ---------------------------------------------------------------- cache ---
def build_cache(cfg: dict) -> None:
    """Bandpassed (NOT normalised) beat/context windows + 11 RR features for
    every DS1 and DS2 beat.  Same filter/windows as preprocess.py."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = cfg["preprocess"]
    out = {k: [] for k in ("beat", "ctx", "rr", "y", "rid", "rs", "sym", "ds2")}
    for name in DS1 + DS2:
        rec = load_record(name, _resolve(cfg["data"]["data_dir"]),
                          [_resolve(m) for m in cfg["data"]["local_mirrors"]], "MLII")
        sig = bandpass_filter(rec.signal.astype(np.float64), rec.fs,
                              p["bandpass_low_hz"], p["bandpass_high_hz"], p["bandpass_order"])
        rr = compute_rr_features_ext(rec.ann_samples, rec.fs,
                                     cfg["rr"]["local_window"], cfg["rr"]["global_window"])
        y = aami_labels(rec.ann_symbols)
        keep = y >= 0
        b = np.stack([extract_window(sig, int(r), p["beat_before_r"], p["beat_after_r"], False)[0]
                      for r in rec.ann_samples[keep]])
        c = np.stack([extract_window(sig, int(r), 540, 540, False)[0]
                      for r in rec.ann_samples[keep]])
        out["beat"].append(b.astype(np.float32)); out["ctx"].append(c.astype(np.float16))
        out["rr"].append(rr[keep]); out["y"].append(y[keep])
        out["rid"] += [name] * int(keep.sum()); out["rs"].append(rec.ann_samples[keep])
        out["sym"] += list(rec.ann_symbols[keep]); out["ds2"] += [name in DS2] * int(keep.sum())
        log.info("record %s: %d beats", name, int(keep.sum()))
    np.save(CACHE / "beat_raw.npy", np.concatenate(out["beat"]))
    np.save(CACHE / "ctx_raw_f16.npy", np.concatenate(out["ctx"]))
    np.savez(CACHE / "meta.npz", rr=np.concatenate(out["rr"]), y=np.concatenate(out["y"]),
             record_ids=np.array(out["rid"]), r_samples=np.concatenate(out["rs"]),
             symbols=np.array(out["sym"]), is_ds2=np.array(out["ds2"]),
             rr_names=np.array(RR_EXT_NAMES))
    log.info("cache written to %s", CACHE)


def _zscore_rows(x: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    out = torch.empty_like(x)
    for i in range(0, len(x), chunk):
        xx = x[i:i + chunk].float()
        mu = xx.mean(1, keepdim=True)
        sd = xx.std(1, unbiased=False, keepdim=True)
        out[i:i + chunk] = ((xx - mu) / (sd + 1e-8)).to(x.dtype)
    return out


class GPUData:
    """All DS1+DS2 beats resident on the device, normalised per variant."""
    def __init__(self, v: dict, device):
        meta = np.load(CACHE / "meta.npz")
        self.y_np = meta["y"].astype(np.int64)
        self.rid = meta["record_ids"]; self.rs = meta["r_samples"]
        self.sym = meta["symbols"]; self.is_ds2 = meta["is_ds2"]
        rr = meta["rr"] if v["rr"] == "ext11" else meta["rr"][:, :6]
        self.beat = torch.from_numpy(np.load(CACHE / "beat_raw.npy")).to(device)
        self.ctx = (torch.from_numpy(np.load(CACHE / "ctx_raw_f16.npy")).to(device)
                    if v["context"] else None)
        if v["norm"] == "zscore":
            self.beat = _zscore_rows(self.beat)
            if self.ctx is not None:
                self.ctx = _zscore_rows(self.ctx)
        self.rr = torch.from_numpy(rr.astype(np.float32)).to(device)
        self.y = torch.from_numpy(self.y_np).to(device)
        self.device = device

    def idx_records(self, recs) -> np.ndarray:
        return np.flatnonzero(np.isin(self.rid, list(recs)))


# %%
# --------------------------------------------------------- augmentation ---
def gpu_augment(beat, ctx, rr, g, fs=360.0, r_idx=90):
    """Batch version of augment.augment_windows (each transform p=0.5)."""
    B, L = beat.shape
    dev = beat.device

    def U(lo, hi, shape=(B, 1)):
        return torch.rand(shape, device=dev, generator=g) * (hi - lo) + lo

    def M():
        return (torch.rand((B, 1), device=dev, generator=g) < 0.5).float()

    m = M(); s = 1 + m * (U(0.8, 1.2) - 1)                 # same amplitude both windows
    beat = beat * s
    if ctx is not None:
        ctx = ctx * s
    m = M(); snr = 10 ** (U(20, 35) / 10)                   # gaussian noise
    beat = beat + m * torch.randn(beat.shape, device=dev, generator=g) * \
        torch.sqrt(beat.pow(2).mean(1, keepdim=True) / snr)
    if ctx is not None:
        snr2 = 10 ** (U(20, 35) / 10)
        ctx = ctx + m * torch.randn(ctx.shape, device=dev, generator=g) * \
            torch.sqrt(ctx.pow(2).mean(1, keepdim=True) / snr2)

    def bw(x):                                             # baseline wander
        n = x.shape[1]
        t = torch.arange(n, device=dev).float()[None] / fs
        a = U(0, 0.1) * x.std(1, keepdim=True)
        return x + M() * a * torch.sin(2 * math.pi * U(0.1, 0.5) * t + U(0, 2 * math.pi))
    beat = bw(beat)
    if ctx is not None:
        ctx = bw(ctx)

    ar = torch.arange(L, device=dev)[None].float()
    m = M(); sh = torch.randint(-10, 11, (B, 1), device=dev, generator=g).float() * m
    src = (ar - sh).clamp(0, L - 1).long()                 # shift with edge padding
    beat = beat.gather(1, src)
    m = M(); fac = 1 + m * (U(0.9, 1.1) - 1)               # stretch about the R peak
    pos = (r_idx + (ar - r_idx) / fac).clamp(0, L - 1)
    lo = pos.floor().long(); hi = (lo + 1).clamp(max=L - 1); w = pos - lo.float()
    beat = beat.gather(1, lo) * (1 - w) + beat.gather(1, hi) * w

    m = M(); rr = rr + m * U(-0.03, 0.03, rr.shape)        # RR jitter
    return beat, ctx, rr


# %%
# -------------------------------------------------------------- metrics ---
def quick_metrics(y5: np.ndarray, L5: np.ndarray) -> dict:
    """3-class N/S/V (true F/Q dropped, argmax over N/S/V logits) + 5-class F1."""
    keep = y5 <= 2
    y = y5[keep]; pred = L5[keep][:, :3].argmax(1)
    out = {"acc3": float((y == pred).mean())}
    f1s = []
    for i, c in enumerate("NSV"):
        tp = int(((pred == i) & (y == i)).sum())
        se = tp / max(1, int((y == i).sum())); ppv = tp / max(1, int((pred == i).sum()))
        out[f"se_{c}"] = se; out[f"ppv_{c}"] = ppv
        f1s.append(2 * se * ppv / (se + ppv + 1e-12))
    out["f1_3"] = float(np.mean(f1s))
    p5 = L5.argmax(1); f15 = []
    for i in range(5):
        if (y5 == i).sum() == 0:
            continue
        tp = int(((p5 == i) & (y5 == i)).sum())
        se = tp / max(1, int((y5 == i).sum())); ppv = tp / max(1, int((p5 == i).sum()))
        f15.append(2 * se * ppv / (se + ppv + 1e-12))
    out["f1_5"] = float(np.mean(f15)); out["acc5"] = float((p5 == y5).mean())
    return out


def full_metrics(y5, L5):
    """5-class AAMI + 3-class N/S/V (non-strict) metric dicts."""
    m5 = multiclass_metrics(y5, L5.argmax(1), k=5)
    keep = y5 <= 2
    y3 = y5[keep]; p3 = L5[keep][:, :3].argmax(1)
    m3 = multiclass_metrics(y3, p3, k=3)
    m3["ovr_accuracy"] = ovr_accuracy(y3, p3, 3)
    return m5, m3


def fit_T(L, y):
    return float(TemperatureScaler(1.5).calibrate(np.asarray(L, np.float32), y, max_iter=200))


# %%
# ------------------------------------------------------------- training ---
@torch.no_grad()
def infer(model, data: GPUData, idx: np.ndarray, bs: int = 1024) -> np.ndarray:
    model.eval()
    out = []
    it = torch.from_numpy(idx).to(data.device)
    for i in range(0, len(it), bs):
        j = it[i:i + bs]
        ctx = data.ctx[j].float().unsqueeze(1) if data.ctx is not None else None
        with torch.autocast("cuda", dtype=torch.float16, enabled=data.device.type == "cuda"):
            lo = model(data.beat[j].unsqueeze(1), ctx, data.rr[j])
        out.append(lo.float().cpu())
    return torch.cat(out).numpy()


def train_model(v: dict, data: GPUData, tr_idx: np.ndarray, eval_sets: dict,
                seed: int, stop_epoch: int = None, tag: str = ""):
    """Train on tr_idx for min(stop_epoch, epochs) epochs of a cosine schedule
    of length v['epochs'].  After every epoch the EMA model scores every
    eval set; returns (model with EMA weights, {name: [E x n x 5 logits]}, hist)."""
    dev = data.device
    torch.manual_seed(seed); np.random.seed(seed)
    g = torch.Generator(device=dev); g.manual_seed(seed)
    cfg = model_cfg(v)
    model = build_model(cfg, n_classes=5).to(dev)
    y_tr = data.y_np[tr_idx]
    n_tr = len(tr_idx)
    tr_t = torch.from_numpy(tr_idx).to(dev)

    t_cfg = dict(s_cap=v["s_cap"], v_cap=v["v_cap"], f_cap=v["f_cap"], q_cap=v["q_cap"])
    if v["sampler"] == "balanced":
        target = sampler_target_proportions(y_tr, 5, t_cfg)
        cnt = np.bincount(y_tr, minlength=5).astype(np.float64)
        w = torch.tensor((target / np.maximum(cnt, 1))[y_tr], device=dev, dtype=torch.float64)
        eff = target
    else:
        w = None
        eff = np.bincount(y_tr, minlength=5) / n_tr
    if v["loss"] == "focal":
        y_eff = np.repeat(np.arange(5), np.round(eff * 100000).astype(int))
        loss_fn = FocalLoss(compute_class_alpha(y_eff, 5, "sqrt"), gamma=v["gamma"]).to(dev)
    else:
        loss_fn = nn.CrossEntropyLoss()

    bs = v["batch"]
    steps = n_tr // bs
    opt = torch.optim.AdamW(model.parameters(), lr=v["lr"], weight_decay=v["wd"])
    sched = cosine_schedule_with_warmup(opt, v["warmup"], v["epochs"], steps)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")
    ema = EMA(model, decay=v["ema"])
    n_ep = v["epochs"] if stop_epoch is None else min(stop_epoch, v["epochs"])
    hist = {"loss": [], "metrics": {k: [] for k in eval_sets}}
    logits = {k: [] for k in eval_sets}
    for ep in range(n_ep):
        t0 = time.perf_counter()
        model.train()
        order = (torch.multinomial(w, n_tr, replacement=True, generator=g) if w is not None
                 else torch.randperm(n_tr, device=dev, generator=g))
        tot = 0.0
        for s in range(steps):
            j = tr_t[order[s * bs:(s + 1) * bs]]
            beat = data.beat[j]; ctx = data.ctx[j].float() if data.ctx is not None else None
            rr = data.rr[j]; y = data.y[j]
            if v["aug"]:
                beat, ctx, rr = gpu_augment(beat, ctx, rr, g)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=dev.type == "cuda"):
                lo = model(beat.unsqueeze(1), ctx.unsqueeze(1) if ctx is not None else None, rr)
            loss = loss_fn(lo.float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            tot += loss.item() if s % 20 == 0 else 0.0
        hist["loss"].append(tot / max(1, math.ceil(steps / 20)))
        msg = ""
        if eval_sets:
            old = ema.apply_to(model)
            for k, idx in eval_sets.items():
                L = infer(model, data, idx)
                logits[k].append(L.astype(np.float16))
                qm = quick_metrics(data.y_np[idx], L)
                hist["metrics"][k].append(qm)
                msg += (f" | {k}: F1_3 {qm['f1_3']:.3f} SeS {qm['se_S']:.3f} "
                        f"PPVS {qm['ppv_S']:.3f} SeV {qm['se_V']:.3f}")
            ema.restore(model, old)
        log.info("%s ep %2d/%d loss %.4f (%.1fs)%s", tag, ep + 1, n_ep, hist["loss"][-1],
                 time.perf_counter() - t0, msg)
    ema.apply_to(model)          # keep EMA weights (they are what is evaluated)
    model.eval()
    return model, logits, hist


# %%
# ---------------------------------------------------------------- modes ---
def load_folds():
    return json.load(open(FOLDS_JSON))["folds"]


def run_cv(vname: str, seeds, device, folds_sel=None):
    v = get_variant(vname)
    d = OUT / "cv" / vname
    d.mkdir(parents=True, exist_ok=True)
    json.dump(v, open(d / "variant.json", "w"), indent=1)
    folds = load_folds()
    data = GPUData(v, device)
    ds1_idx = np.flatnonzero(~data.is_ds2)
    for seed in seeds:
        for k, recs in enumerate(folds):
            if folds_sel is not None and k not in folds_sel:
                continue
            f = d / f"fold{k}_seed{seed}.npz"
            if f.exists():
                log.info("%s fold %d seed %d: done, skipping", vname, k, seed)
                continue
            va = data.idx_records(recs)
            tr = np.setdiff1d(ds1_idx, va)
            assert not np.isin(data.rid[tr], recs).any() and not data.is_ds2[tr].any()
            _, L, h = train_model(v, data, tr, {"val": va}, seed, tag=f"{vname} f{k} s{seed}")
            tmp = d / f"fold{k}_seed{seed}.tmp.npz"
            np.savez_compressed(tmp, logits=np.stack(L["val"]), idx=va, y=data.y_np[va],
                                loss=np.array(h["loss"]))
            tmp.rename(f)
    del data
    torch.cuda.empty_cache()


def cv_summary(vname: str, smooth: int = 3) -> dict:
    """Pool OOF logits over folds; per-epoch curves; E* = argmax of the
    `smooth`-epoch moving average of the seed-mean pooled N/S/V macro-F1."""
    d = OUT / "cv" / vname
    v = get_variant(vname)
    files = sorted(d.glob("fold*_seed*.npz"))
    by_seed = {}
    for f in files:
        k, s = f.stem.replace("fold", "").split("_seed")
        if f.name.endswith(".tmp.npz"):
            continue
        by_seed.setdefault(int(s), {})[int(k)] = f
    nf = len(load_folds())
    seeds = sorted(s for s, fs in by_seed.items() if len(fs) == nf)
    if not seeds:
        raise RuntimeError(f"{vname}: no seed with all {nf} folds")
    pooled = {}
    for s in seeds:
        parts = [np.load(by_seed[s][k]) for k in range(nf)]
        idx = np.concatenate([p["idx"] for p in parts])
        y = np.concatenate([p["y"] for p in parts])
        L = np.concatenate([p["logits"].astype(np.float32) for p in parts], axis=1)  # E x n x 5
        pooled[s] = (idx, y, L)
    idx0, y0, _ = pooled[seeds[0]]
    E = pooled[seeds[0]][2].shape[0]
    curves = {s: [quick_metrics(y0, pooled[s][2][e]) for e in range(E)] for s in seeds}
    keys = curves[seeds[0]][0].keys()
    mean_curve = [{k: float(np.mean([curves[s][e][k] for s in seeds])) for k in keys}
                  for e in range(E)]
    f1 = np.array([c["f1_3"] for c in mean_curve])
    sm = np.convolve(np.pad(f1, (smooth // 2, smooth // 2), mode="edge"),
                     np.ones(smooth) / smooth, mode="valid")
    e_star = int(np.argmax(sm)) + 1
    Lens = np.mean([pooled[s][2][e_star - 1] for s in seeds], axis=0)
    ens = quick_metrics(y0, Lens)
    T_seed = [fit_T(pooled[s][2][e_star - 1], y0) for s in seeds]
    T_ens = fit_T(Lens, y0)
    p = torch.softmax(torch.from_numpy(Lens) / T_ens, 1).numpy()
    summ = {
        "variant": vname, "knobs": v, "seeds": seeds, "n_epochs_run": E,
        "E_star": e_star, "smooth": smooth,
        "at_E_star_seed_mean": mean_curve[e_star - 1],
        "at_E_star_seed_std": {k: float(np.std([curves[s][e_star - 1][k] for s in seeds]))
                               for k in keys},
        "at_E_star_ensemble": ens,
        "at_last_epoch_seed_mean": mean_curve[-1],
        "best_raw_epoch": int(np.argmax(f1)) + 1, "best_raw_f1_3": float(f1.max()),
        "T_seed": T_seed, "T_ensemble": T_ens,
        "ece_ensemble_before": float(compute_ece(torch.softmax(torch.from_numpy(Lens), 1).numpy(), y0)),
        "ece_ensemble_after": float(compute_ece(p, y0)),
        "curve_f1_3": [round(c["f1_3"], 4) for c in mean_curve],
        "curve_se_S": [round(c["se_S"], 4) for c in mean_curve],
        "curve_ppv_S": [round(c["ppv_S"], 4) for c in mean_curve],
        "curve_f1_5": [round(c["f1_5"], 4) for c in mean_curve],
    }
    json.dump(summ, open(d / "summary.json", "w"), indent=1)
    np.savez_compressed(d / "oof_Estar.npz", idx=idx0, y=y0,
                        logits=np.stack([pooled[s][2][e_star - 1] for s in seeds]),
                        seeds=np.array(seeds), E_star=e_star)
    return summ


def run_refit(vname: str, seeds, device, e_star: int = None):
    """Refit on all DS1 for E* epochs (E* from CV), score DS1 and DS2."""
    v = get_variant(vname)
    summ = json.load(open(OUT / "cv" / vname / "summary.json"))
    e_star = e_star or summ["E_star"]
    d = OUT / "refit" / vname
    d.mkdir(parents=True, exist_ok=True)
    data = GPUData(v, device)
    ds1 = np.flatnonzero(~data.is_ds2); ds2 = np.flatnonzero(data.is_ds2)
    for seed in seeds:
        f = d / f"seed{seed}.npz"
        if f.exists():
            log.info("%s refit seed %d: done, skipping", vname, seed)
            continue
        model, _, h = train_model(v, data, ds1, {}, seed, stop_epoch=e_star,
                                  tag=f"{vname} refit s{seed}")
        L1 = infer(model, data, ds1); L2 = infer(model, data, ds2)   # DS2: report only
        torch.save({"state_dict": model.state_dict(), "cfg": model_cfg(v), "E_star": e_star,
                    "seed": seed}, d / f"seed{seed}.pt")
        tmp = d / f"seed{seed}.tmp.npz"
        np.savez_compressed(tmp, logits_ds1=L1, logits_ds2=L2, idx_ds1=ds1, idx_ds2=ds2,
                            loss=np.array(h["loss"]))
        tmp.rename(f)
        log.info("%s refit seed %d DS2: %s", vname, seed,
                 {k: round(x, 4) for k, x in quick_metrics(data.y_np[ds2], L2).items()})
    del data
    torch.cuda.empty_cache()


def eval_refit(vname: str) -> dict:
    """DS2 5-class + 3-class metrics: per seed, mean +- std, logit ensemble."""
    d = OUT / "refit" / vname
    meta = np.load(CACHE / "meta.npz")
    y = meta["y"].astype(np.int64)
    files = sorted(d.glob("seed*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")]
    res = {"variant": vname, "seeds": [], "per_seed": []}
    Ls = []
    for f in files:
        z = np.load(f)
        y2 = y[z["idx_ds2"]]
        m5, m3 = full_metrics(y2, z["logits_ds2"])
        res["seeds"].append(int(f.stem.replace("seed", "")))
        res["per_seed"].append({"m5": m5, "m3": m3})
        Ls.append(z["logits_ds2"])
    m5, m3 = full_metrics(y2, np.mean(Ls, axis=0))
    res["ensemble"] = {"m5": m5, "m3": m3}
    keys5 = [k for k, x in res["per_seed"][0]["m5"].items() if isinstance(x, float)]
    keys3 = [k for k, x in res["per_seed"][0]["m3"].items() if isinstance(x, float)]
    res["mean"] = {"m5": {k: float(np.mean([p["m5"][k] for p in res["per_seed"]])) for k in keys5},
                   "m3": {k: float(np.mean([p["m3"][k] for p in res["per_seed"]])) for k in keys3}}
    res["std"] = {"m5": {k: float(np.std([p["m5"][k] for p in res["per_seed"]])) for k in keys5},
                  "m3": {k: float(np.std([p["m3"][k] for p in res["per_seed"]])) for k in keys3}}
    json.dump(res, open(d / "ds2_eval.json", "w"), indent=1)
    return res


def table(vnames):
    rows = ["| variant | seeds | E* | OOF acc3 | OOF F1 N/S/V | OOF Se_S | OOF PPV_S | OOF Se_V | "
            "OOF 5c F1 | ens OOF F1 | ens Se_S | ens PPV_S |", "|" + "---|" * 12]
    for vn in vnames:
        f = OUT / "cv" / vn / "summary.json"
        if not f.exists():
            continue
        s = json.load(open(f)); m = s["at_E_star_seed_mean"]; e = s["at_E_star_ensemble"]
        rows.append(f"| {vn} | {len(s['seeds'])} | {s['E_star']} | {100*m['acc3']:.2f} | "
                    f"{100*m['f1_3']:.2f} | {100*m['se_S']:.1f} | {100*m['ppv_S']:.1f} | "
                    f"{100*m['se_V']:.1f} | {100*m['f1_5']:.2f} | {100*e['f1_3']:.2f} | "
                    f"{100*e['se_S']:.1f} | {100*e['ppv_S']:.1f} |")
    return "\n".join(rows)


# %%
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["cache", "cv", "cvsum", "refit", "evalrefit", "table"])
    ap.add_argument("--variant", nargs="+", default=["V1_plain"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--folds", nargs="+", type=int, default=None)
    ap.add_argument("--e-star", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None, help="override BASE epochs (debug)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(OUT / f"{args.mode}.log")])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.epochs:
        BASE["epochs"] = args.epochs
    dev = torch.device(args.device)
    if args.mode == "cache":
        build_cache(load_config(P4 / "configs" / "default.yaml"))
        return
    for vn in args.variant:
        if args.mode == "cv":
            run_cv(vn, args.seeds, dev, args.folds)
        elif args.mode == "cvsum":
            s = cv_summary(vn)
            print(vn, "E*", s["E_star"], json.dumps(s["at_E_star_seed_mean"]),
                  "ens", json.dumps(s["at_E_star_ensemble"]), "T_ens", s["T_ensemble"])
        elif args.mode == "refit":
            run_refit(vn, args.seeds, dev, args.e_star)
        elif args.mode == "evalrefit":
            r = eval_refit(vn)
            print(vn, json.dumps({"mean3": r["mean"]["m3"], "ens3": {k: x for k, x in
                                  r["ensemble"]["m3"].items() if isinstance(x, float)}}))
    if args.mode == "table":
        print(table(args.variant))


if __name__ == "__main__":
    main()
