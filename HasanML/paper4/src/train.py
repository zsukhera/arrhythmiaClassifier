# %% [markdown]
# ## train.py — Training loop, EMA, balanced sampler, multi-seed runs
#
# Key design decisions:
#   - Early stopping on DS1-val MACRO-F1 (not val_loss): F1 directly reflects
#     S and V detection quality, which is what matters for arrhythmia detection.
#   - EMA (decay 0.999): exponential moving average of weights gives a smoother
#     model that generalises better; used for all validation and test inference.
#   - Cosine LR schedule with 5 warm-up epochs: matches the KAN-Former paper
#     and prevents the large initial updates from blowing up B-spline coefficients.
#   - Mixed precision (torch.amp): halves memory and speeds up GPU training.
#   - Gradient clipping (max_norm=1.0): B-spline KAN layers can produce large
#     gradients early in training when splines are far from their targets.
#   - Class-balanced sampler capped at 30% S and 30% V: prevents the network
#     from being dominated by N beats (90%) while avoiding over-correction.

# %%
import copy
import logging
import math
import os
import pickle
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

try:
    from .augment import augment_windows
    from .losses import build_loss, compute_class_alpha
    from .metrics import macro_f1_from_preds
except ImportError:
    from augment import augment_windows
    from losses import build_loss, compute_class_alpha
    from metrics import macro_f1_from_preds

log = logging.getLogger(__name__)


# %%
def _fix_seeds(seed: int, deterministic: bool = True) -> None:
    """Fix Python, NumPy, and PyTorch random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception as exc:
            log.warning("use_deterministic_algorithms not fully supported: %s", exc)
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_device(mode: str = "auto") -> torch.device:
    """
    Map 'auto' | 'cpu' | 'cuda' | 'gpu' to a torch.device.
    'auto' selects CUDA if available, else CPU.
    Sets memory growth (not needed in PyTorch, but log device info).
    """
    env_override = os.environ.get("P4_DEVICE", "").lower()
    req = (env_override or mode or "auto").lower().replace("gpu", "cuda")
    if req not in ("auto", "cpu", "cuda"):
        raise ValueError(f"Unknown device: {mode!r}")

    if req == "cpu":
        return torch.device("cpu")
    if req == "cuda" and not torch.cuda.is_available():
        log.warning("device=cuda requested but CUDA not available; falling back to CPU")
        return torch.device("cpu")
    if req == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda")


# %%
class BeatDataset(Dataset):
    """
    PyTorch Dataset wrapping beat windows, context windows, RR features, and labels.

    beat_segs    : (N, beat_len)
    context_segs : (N, ctx_len) or None
    rr_feats     : (N, 6)
    y            : (N,) int64
    augment      : whether to apply training augmentation
    cfg_aug      : augmentation config sub-dict
    """
    def __init__(self, beat_segs: np.ndarray, context_segs: Optional[np.ndarray],
                 rr_feats: np.ndarray, y: np.ndarray,
                 augment: bool = False, cfg_aug: Optional[dict] = None):
        self.beats   = beat_segs
        self.ctxs    = context_segs
        self.rr      = rr_feats
        self.y       = y
        self.augment = augment
        self.cfg_aug = cfg_aug or {}
        # Per-worker RNG seeded at worker init time
        self._rng = np.random.default_rng()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        beat = self.beats[idx].copy()
        ctx  = self.ctxs[idx].copy() if self.ctxs is not None else None
        rr   = self.rr[idx].copy()
        y    = int(self.y[idx])

        if self.augment and self.cfg_aug.get("enabled", False):
            beat, ctx, rr = augment_windows(beat, ctx, rr, self.cfg_aug, self._rng)

        beat_t = torch.from_numpy(beat).float().unsqueeze(0)   # (1, beat_len)
        rr_t   = torch.from_numpy(rr).float()                  # (6,)

        if ctx is not None:
            ctx_t = torch.from_numpy(ctx).float().unsqueeze(0)  # (1, ctx_len)
        else:
            ctx_t = None

        return beat_t, ctx_t, rr_t, torch.tensor(y, dtype=torch.long)


def _worker_init(worker_id):
    """
    Seed each DataLoader worker independently to avoid correlated augmentation.
    BUGFIX: the dataset's own numpy Generator (used by augment_windows) was
    created once in the parent process, so every worker inherited an identical
    copy and produced identical augmentation streams.  Re-seed it per worker
    from torch.initial_seed() (which differs per worker and per epoch and is
    derived from the DataLoader generator, so it is reproducible).
    """
    seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(seed % (2 ** 31))
    info = torch.utils.data.get_worker_info()
    if info is not None and hasattr(info.dataset, "_rng"):
        info.dataset._rng = np.random.default_rng(seed)


def sampler_target_proportions(y: np.ndarray, n_classes: int, cfg_tr: dict) -> np.ndarray:
    """
    Per-class target proportions of each training batch.

    Inverse-frequency targets with per-class caps, solved by water-filling:
    every minority class gets min(inverse-frequency share, cap) and the
    leftover mass goes to the uncapped classes (in practice N).

    BUGFIX (2026-11-18): the original code capped only S and V.  With the
    5-class DS1-train counts (N 23932, S 149, V 1234, F 13, Q 2) pure inverse
    frequency gave Q ~86 % and F ~13 % of every batch -- i.e. the network was
    trained almost exclusively on 2 Q beats and 13 F beats.  F and Q now have
    their own caps (f_cap, q_cap).
    """
    caps_cfg = {1: float(cfg_tr.get("s_cap", 0.30)), 2: float(cfg_tr.get("v_cap", 0.30)),
                3: float(cfg_tr.get("f_cap", 0.05)), 4: float(cfg_tr.get("q_cap", 0.01))}
    counts = np.array([(y == i).sum() for i in range(n_classes)], dtype=np.float64)
    present = counts > 0
    inv = np.where(present, 1.0 / np.maximum(counts, 1.0), 0.0)
    cap = np.array([caps_cfg.get(i, 1.0) for i in range(n_classes)])
    cap[~present] = 0.0
    target = np.zeros(n_classes)
    free = present.copy()
    mass = 1.0
    for _ in range(n_classes + 1):
        if not free.any() or mass <= 0:
            break
        share = np.where(free, inv, 0.0)
        share = share / share.sum() * mass
        over = free & (share > cap)
        if not over.any():
            target[free] = share[free]
            break
        target[over] = cap[over]
        mass -= cap[over].sum()
        free &= ~over
    return target / target.sum()


def build_sampler(y: np.ndarray, n_classes: int, cfg_tr: dict) -> Optional[WeightedRandomSampler]:
    """
    Build a WeightedRandomSampler that over-samples minority classes up to
    the capped target proportions of sampler_target_proportions().
    """
    if cfg_tr.get("sampler", "balanced") == "natural":
        return None   # natural class distribution

    counts = np.array([(y == i).sum() for i in range(n_classes)], dtype=np.float64)
    target = sampler_target_proportions(y, n_classes, cfg_tr)
    log.info("sampler target proportions: %s",
             {i: round(float(t), 4) for i, t in enumerate(target)})

    # Weight per sample = target proportion / class count
    weights_per_class = target / np.maximum(counts, 1.0)
    sample_weights    = torch.tensor(weights_per_class[y], dtype=torch.float64)

    return WeightedRandomSampler(sample_weights, num_samples=len(y), replacement=True)


def _collate_skip_none(batch):
    """Custom collate that handles ctx_t=None (context disabled)."""
    beats, ctxs, rrs, ys = zip(*batch)
    beats = torch.stack(beats)
    rrs   = torch.stack(rrs)
    ys    = torch.stack(ys)
    if ctxs[0] is None:
        return beats, None, rrs, ys
    return beats, torch.stack(ctxs), rrs, ys


# %%
class EMA:
    """
    Exponential Moving Average of model weights.
    After training, apply EMA weights for inference: they produce a smoother
    loss landscape and typically improve generalisation on DS2.
    decay=0.999 means each update moves the EMA by only 0.1%.
    """
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay  = decay
        self.n_updates = 0
        self.shadow = {n: p.data.clone().float() for n, p in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        # Warm-up: effective decay min(decay, (1+t)/(10+t)).  DS1-train gives
        # only ~100 steps/epoch, so a fixed 0.999 kept the EMA ~80 % at the
        # random init after 2 epochs (validation of a near-random model).
        self.n_updates += 1
        d = min(self.decay, (1.0 + self.n_updates) / (10.0 + self.n_updates))
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(d).add_(p.data.float(), alpha=1.0 - d)

    def apply_to(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        """Temporarily apply EMA weights; returns original weights for restore()."""
        old = {n: p.data.clone() for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.data.copy_(self.shadow[n].to(p.dtype))
        return old

    def restore(self, model: nn.Module, old: Dict[str, torch.Tensor]) -> None:
        for n, p in model.named_parameters():
            if n in old:
                p.data.copy_(old[n])


# %%
def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, max_epochs: int,
                                 steps_per_epoch: int):
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps  = max_epochs  * steps_per_epoch

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = (step - warmup_steps) / max(1.0, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_fn)


# %%
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader,
             device: torch.device) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run inference with EMA weights (caller applies EMA before calling)."""
    model.eval()
    all_logits, all_y = [], []
    for beats, ctxs, rrs, ys in loader:
        beats = beats.to(device)
        ctxs  = ctxs.to(device) if ctxs is not None else None
        rrs   = rrs.to(device)
        logits = model(beats, ctxs, rrs)
        all_logits.append(logits.cpu())
        all_y.append(ys)
    logits_all = torch.cat(all_logits).numpy()
    y_all      = torch.cat(all_y).numpy()
    preds      = logits_all.argmax(1)
    f1         = macro_f1_from_preds(y_all, preds)
    return logits_all, y_all, f1


# %%
def _select_f1(y: np.ndarray, pred: np.ndarray, classes=None) -> float:
    """Macro-F1 over `classes` (default: all classes present in y)."""
    if not classes:
        return macro_f1_from_preds(y, pred)
    f1s = []
    for c in classes:
        tp = int(((pred == c) & (y == c)).sum())
        fp = int(((pred == c) & (y != c)).sum())
        fn = int(((pred != c) & (y == c)).sum())
        p_, r_ = tp / (tp + fp + 1e-10), tp / (tp + fn + 1e-10)
        f1s.append(2 * p_ * r_ / (p_ + r_ + 1e-10))
    return float(np.mean(f1s))


def _val_summary(logits: np.ndarray, y: np.ndarray) -> dict:
    """Per-epoch DS1-val diagnostics (macro-F1 + S/V recall/PPV)."""
    pred = logits.argmax(1)
    out = {"f1": macro_f1_from_preds(y, pred),
           "f1_nsv": _select_f1(y, pred, [0, 1, 2])}
    for c, name in ((1, "S"), (2, "V")):
        tp = int(((pred == c) & (y == c)).sum())
        out[f"se_{name}"]  = tp / max(1, int((y == c).sum()))
        out[f"ppv_{name}"] = tp / max(1, int((pred == c).sum()))
    return out


def train_one(model: nn.Module, cfg: dict, ds_tr: dict, ds_val: dict,
              n_classes: int, seed: int,
              device: torch.device,
              ckpt_path: Optional[Path] = None) -> Tuple[nn.Module, EMA, dict]:
    """
    Train for one seed.  Returns (model_with_best_weights, ema, history).
    DS2 data must NOT appear in ds_tr or ds_val.

    If ckpt_path is given, the full training state (model, optimiser,
    scheduler, scaler, EMA, best-so-far, history) is written there after
    every epoch and training resumes from it if it already exists.
    """
    t_cfg  = cfg["training"]
    a_cfg  = cfg.get("augmentation", {})
    _fix_seeds(seed, deterministic=bool(cfg["compute"].get("deterministic", True)))

    # --- Dataset & DataLoader ---
    tr_dataset = BeatDataset(
        ds_tr["beat_segs"], ds_tr.get("context_segs"),
        ds_tr["rr_feats"],  ds_tr["y"],
        augment=bool(a_cfg.get("enabled", True)), cfg_aug=a_cfg,
    )
    tr_dataset._rng = np.random.default_rng(seed)   # used only if num_workers == 0
    val_dataset = BeatDataset(
        ds_val["beat_segs"], ds_val.get("context_segs"),
        ds_val["rr_feats"],  ds_val["y"],
        augment=False,
    )

    sampler    = build_sampler(ds_tr["y"], n_classes, t_cfg)
    gen        = torch.Generator().manual_seed(seed)
    if sampler is not None:
        sampler.generator = gen
    n_workers  = int(cfg["compute"].get("dataloader_workers", 4))
    batch_size = int(t_cfg["batch_size"])

    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_skip_none, worker_init_fn=_worker_init,
        generator=gen, persistent_workers=(n_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=0, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_skip_none,
    )

    # --- Loss & optimiser ---
    # BUGFIX (2026-11-18): alpha used to be computed from the RAW DS1-train
    # counts although the balanced sampler had already rebalanced the batches,
    # i.e. the imbalance was corrected twice (N weight 0.029 vs F 1.26 / Q 3.2
    # on top of F/Q oversampling) and the 2-epoch sanity run predicted F for
    # ~every beat.  Compute the sqrt weights from the class distribution the
    # loss actually sees (sampler targets when the balanced sampler is on).
    if t_cfg.get("sampler", "balanced") != "natural":
        eff = sampler_target_proportions(ds_tr["y"], n_classes, t_cfg)
        y_eff = np.repeat(np.arange(n_classes), np.round(eff * 100000).astype(int))
        alpha = compute_class_alpha(y_eff, n_classes, mode=t_cfg.get("alpha_mode", "sqrt"))
    else:
        alpha = compute_class_alpha(ds_tr["y"], n_classes, mode=t_cfg.get("alpha_mode", "sqrt"))
    loss_fn = build_loss(cfg, alpha).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(t_cfg["lr"]),
        weight_decay=float(t_cfg["weight_decay"]),
    )
    n_steps = len(tr_loader)
    scheduler = cosine_schedule_with_warmup(
        optimizer, int(t_cfg["warmup_epochs"]),
        int(t_cfg["max_epochs"]), n_steps,
    )

    use_amp   = bool(t_cfg.get("mixed_precision", True)) and (device.type == "cuda")
    scaler    = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema       = EMA(model, decay=float(t_cfg["ema_decay"]))
    patience  = int(t_cfg["early_stop_patience"])
    max_ep    = int(t_cfg["max_epochs"])
    clip_norm = float(t_cfg["grad_clip"])
    min_delta = float(t_cfg.get("early_stop_min_delta", 1e-5))

    best_f1   = -1.0
    best_wts  = None
    best_ema_shadow = None
    best_epoch = 0
    stale     = 0
    start_ep  = 0
    history   = {"train_loss": [], "val_f1": [], "val_diag": []}
    elapsed_prev = 0.0

    if ckpt_path is not None and Path(ckpt_path).exists():
        ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        ema.shadow = {k: v.to(device) for k, v in ck["ema_shadow"].items()}
        ema.n_updates = ck.get("ema_n_updates", 0)
        best_f1, best_epoch, stale = ck["best_f1"], ck["best_epoch"], ck["stale"]
        best_wts, best_ema_shadow = ck["best_wts"], ck["best_ema_shadow"]
        history, start_ep = ck["history"], ck["epoch"] + 1
        elapsed_prev = ck.get("elapsed", 0.0)
        # BUGFIX: map_location moved the RNG ByteTensors to CUDA -> must be CPU
        torch.set_rng_state(ck["torch_rng"].cpu())
        gen.set_state(ck["gen_state"].cpu())
        log.info("seed %d: resumed from %s at epoch %d", seed, ckpt_path, start_ep + 1)

    t0 = time.perf_counter()
    epoch = start_ep - 1
    done = (stale >= patience) if start_ep > 0 else False
    for epoch in range(start_ep, max_ep):
        if done:
            break
        te = time.perf_counter()
        # --- Training ---
        model.train()
        ep_loss = 0.0
        for beats, ctxs, rrs, ys in tr_loader:
            beats = beats.to(device, non_blocking=True)
            ctxs  = ctxs.to(device, non_blocking=True) if ctxs is not None else None
            rrs   = rrs.to(device, non_blocking=True)
            ys    = ys.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                 dtype=torch.float16, enabled=use_amp):
                logits = model(beats, ctxs, rrs)
            loss = loss_fn(logits.float(), ys)   # loss in fp32 for stability

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            ep_loss += loss.item()

        avg_loss = ep_loss / max(1, len(tr_loader))
        history["train_loss"].append(avg_loss)

        # --- Validation with EMA weights ---
        old_wts = ema.apply_to(model)
        val_logits, val_y, val_f1 = evaluate(model, val_loader, device)
        ema.restore(model, old_wts)
        diag = _val_summary(val_logits, val_y)
        sel_classes = t_cfg.get("select_classes")      # e.g. [0,1,2] = N/S/V macro-F1
        if sel_classes:
            val_f1 = _select_f1(val_y, val_logits.argmax(1), list(sel_classes))
        history["val_f1"].append(val_f1)
        history["val_diag"].append(diag)

        log.info("epoch %3d/%d  loss=%.4f  val_macro_f1=%.4f  val Se_S=%.3f PPV_S=%.3f "
                 "Se_V=%.3f PPV_V=%.3f  (%.1fs)",
                 epoch + 1, max_ep, avg_loss, val_f1, diag["se_S"], diag["ppv_S"],
                 diag["se_V"], diag["ppv_V"], time.perf_counter() - te)

        # --- Early stopping on macro-F1 ---
        if val_f1 > best_f1 + min_delta:
            best_f1  = val_f1
            best_epoch = epoch + 1
            best_wts = copy.deepcopy(model.state_dict())
            best_ema_shadow = copy.deepcopy(ema.shadow)
            stale    = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (patience=%d, best epoch %d)",
                         epoch + 1, patience, best_epoch)
                done = True

        if ckpt_path is not None:
            tmp = Path(str(ckpt_path) + ".tmp")
            torch.save({
                "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "ema_shadow": ema.shadow, "ema_n_updates": ema.n_updates,
                "best_f1": best_f1,
                "best_epoch": best_epoch, "stale": stale,
                "best_wts": best_wts, "best_ema_shadow": best_ema_shadow,
                "history": history, "epoch": epoch,
                "elapsed": elapsed_prev + time.perf_counter() - t0,
                "torch_rng": torch.get_rng_state(), "gen_state": gen.get_state(),
            }, str(tmp))
            os.replace(tmp, ckpt_path)

    elapsed = elapsed_prev + time.perf_counter() - t0

    # Restore best weights
    if best_wts is not None:
        model.load_state_dict(best_wts)
        ema.shadow = best_ema_shadow

    history["elapsed"] = elapsed
    history["best_val_f1"] = best_f1
    history["best_epoch"] = best_epoch
    history["n_epochs"] = len(history["train_loss"])
    log.info("seed %d done: %d epochs, %.1f s, best_val_f1=%.4f (epoch %d)",
             seed, history["n_epochs"], elapsed, best_f1, best_epoch)
    return model, ema, history


# %%
def train_multi_seed(cfg: dict, ds_tr: dict, ds_val: dict, ds_test: dict,
                     n_classes: int, seeds: List[int],
                     cache_dir: Optional[Path] = None,
                     run_name: str = "run",
                     device: Optional[torch.device] = None) -> List[dict]:
    """
    Run training for each seed. Each result is cached to disk.
    Returns list of per-seed result dicts.
    """
    from .model import build_model, model_summary
    from .metrics import multiclass_metrics

    if device is None:
        device = torch.device("cpu")

    results = []
    for seed in seeds:
        cache_path = None
        if cache_dir:
            cache_path = Path(cache_dir) / f"{run_name}_seed{seed}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as fh:
                    r = pickle.load(fh)
                log.info("seed %d: loaded from cache", seed)
                results.append(r)
                continue

        model = build_model(cfg, n_classes=n_classes).to(device)
        log.info("seed %d: %s", seed, model_summary(model))

        ckpt_path = (Path(cache_dir) / f"{run_name}_seed{seed}_ckpt.pt") if cache_dir else None
        model, ema, history = train_one(model, cfg, ds_tr, ds_val, n_classes, seed,
                                        device, ckpt_path=ckpt_path)

        # --- Evaluate with EMA weights ---
        old = ema.apply_to(model)
        logits_val,  y_val,  _ = _infer(model, ds_val,  device, int(cfg["training"]["batch_size"]))
        logits_test, y_test, _ = _infer(model, ds_test, device, int(cfg["training"]["batch_size"]))
        ema.restore(model, old)

        m_val  = multiclass_metrics(y_val,  logits_val.argmax(1),  logits_val,  k=n_classes)
        m_test = multiclass_metrics(y_test, logits_test.argmax(1), logits_test, k=n_classes)

        r = {
            "seed":          seed,
            "run_name":      run_name,
            "elapsed":       history["elapsed"],
            "n_epochs":      history["n_epochs"],
            "best_val_f1":   history["best_val_f1"],
            "best_epoch":    history.get("best_epoch"),
            "val_diag_hist": history.get("val_diag"),
            "train_loss":    history["train_loss"],
            "val_f1_hist":   history["val_f1"],
            "metrics_val":   m_val,
            "metrics_test":  m_test,
            "logits_val":    logits_val,
            "logits_test":   logits_test,
        }
        results.append(r)

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as fh:
                pickle.dump(r, fh, protocol=pickle.HIGHEST_PROTOCOL)
            # Save model weights separately for repredict and distillation
            torch.save({"state_dict": model.state_dict(),
                        "ema_shadow": ema.shadow,
                        "cfg": cfg}, cache_path.with_suffix(".pt"))
            log.info("seed %d: cached to %s", seed, cache_path)

        log.info("seed %d test: acc=%.2f%% macro_f1=%.3f se_S=%.1f%% se_V=%.1f%%",
                 seed,
                 100 * m_test.get("accuracy", 0),
                 m_test.get("macro_f1", 0),
                 100 * m_test.get("se_S", 0),
                 100 * m_test.get("se_V", 0))

    return results


@torch.no_grad()
def _infer(model, ds, device, batch_size=256):
    """Run forward pass over a full dataset dict."""
    dataset = BeatDataset(ds["beat_segs"], ds.get("context_segs"),
                          ds["rr_feats"], ds["y"], augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=_collate_skip_none)
    model.eval()
    all_logits, all_y = [], []
    for beats, ctxs, rrs, ys in loader:
        beats = beats.to(device)
        ctxs  = ctxs.to(device) if ctxs is not None else None
        rrs   = rrs.to(device)
        logits = model(beats, ctxs, rrs)
        all_logits.append(logits.cpu())
        all_y.append(ys)
    L = torch.cat(all_logits).numpy()
    Y = torch.cat(all_y).numpy()
    return L, Y, macro_f1_from_preds(Y, L.argmax(1))
