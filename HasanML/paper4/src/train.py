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
    """Seed each DataLoader worker independently to avoid correlated augmentation."""
    seed = torch.initial_seed() % (2 ** 31)
    np.random.seed(seed + worker_id)


def build_sampler(y: np.ndarray, n_classes: int, cfg_tr: dict) -> Optional[WeightedRandomSampler]:
    """
    Build a WeightedRandomSampler that over-samples minority classes,
    capping S and V at 30% of each batch (by weight proportions).
    """
    if cfg_tr.get("sampler", "balanced") == "natural":
        return None   # natural class distribution

    s_cap = float(cfg_tr.get("s_cap", 0.30))
    v_cap = float(cfg_tr.get("v_cap", 0.30))

    counts = np.array([(y == i).sum() for i in range(n_classes)], dtype=np.float64)
    total  = counts.sum()

    # Inverse-frequency target proportions, capped
    target = 1.0 / np.maximum(counts, 1.0)
    target /= target.sum()

    # Apply cap: AAMI S = class 1, V = class 2 (3/4/5-class all have N=0, S=1, V=2)
    if n_classes >= 2:
        target[1] = min(target[1], s_cap)
    if n_classes >= 3:
        target[2] = min(target[2], v_cap)
    target /= target.sum()

    # Weight per sample = target proportion / actual proportion
    weights_per_class = target / (counts / total + 1e-10)
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
        self.shadow = {n: p.data.clone().float() for n, p in model.named_parameters()}

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    self.shadow[n].mul_(self.decay).add_(
                        p.data.float(), alpha=1.0 - self.decay
                    )

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
def train_one(model: nn.Module, cfg: dict, ds_tr: dict, ds_val: dict,
              n_classes: int, seed: int,
              device: torch.device) -> Tuple[nn.Module, EMA, dict]:
    """
    Train for one seed.  Returns (model_with_ema_weights, ema, history).
    DS2 data must NOT appear in ds_tr or ds_val.
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
    val_dataset = BeatDataset(
        ds_val["beat_segs"], ds_val.get("context_segs"),
        ds_val["rr_feats"],  ds_val["y"],
        augment=False,
    )

    sampler    = build_sampler(ds_tr["y"], n_classes, t_cfg)
    n_workers  = int(cfg["compute"].get("dataloader_workers", 4))
    batch_size = int(t_cfg["batch_size"])

    tr_loader = DataLoader(
        tr_dataset, batch_size=batch_size,
        sampler=sampler, shuffle=(sampler is None),
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_skip_none, worker_init_fn=_worker_init,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size * 2, shuffle=False,
        num_workers=n_workers, pin_memory=(device.type == "cuda"),
        collate_fn=_collate_skip_none,
    )

    # --- Loss & optimiser ---
    alpha  = compute_class_alpha(ds_tr["y"], n_classes, mode="sqrt")
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
    scaler    = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema       = EMA(model, decay=float(t_cfg["ema_decay"]))
    patience  = int(t_cfg["early_stop_patience"])
    max_ep    = int(t_cfg["max_epochs"])
    clip_norm = float(t_cfg["grad_clip"])

    best_f1   = -1.0
    best_wts  = None
    best_ema_shadow = None
    stale     = 0
    history   = {"train_loss": [], "val_f1": []}

    t0 = time.perf_counter()
    for epoch in range(max_ep):
        # --- Training ---
        model.train()
        ep_loss = 0.0
        for beats, ctxs, rrs, ys in tr_loader:
            beats = beats.to(device)
            ctxs  = ctxs.to(device) if ctxs is not None else None
            rrs   = rrs.to(device)
            ys    = ys.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type,
                                 dtype=torch.float16, enabled=use_amp):
                logits = model(beats, ctxs, rrs)
                loss   = loss_fn(logits, ys)

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
        _, _, val_f1 = evaluate(model, val_loader, device)
        ema.restore(model, old_wts)
        history["val_f1"].append(val_f1)

        log.info("epoch %3d/%d  loss=%.4f  val_macro_f1=%.3f",
                 epoch + 1, max_ep, avg_loss, val_f1)

        # --- Early stopping on macro-F1 ---
        if val_f1 > best_f1 + 1e-5:
            best_f1  = val_f1
            best_wts = copy.deepcopy(model.state_dict())
            best_ema_shadow = copy.deepcopy(ema.shadow)
            stale    = 0
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d (patience=%d)", epoch + 1, patience)
                break

    elapsed = time.perf_counter() - t0

    # Restore best weights
    if best_wts is not None:
        model.load_state_dict(best_wts)
        ema.shadow = best_ema_shadow

    history["elapsed"] = elapsed
    history["best_val_f1"] = best_f1
    history["n_epochs"] = epoch + 1
    log.info("seed %d done: %d epochs, %.1f s, best_val_f1=%.4f",
             seed, epoch + 1, elapsed, best_f1)
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

        model, ema, history = train_one(model, cfg, ds_tr, ds_val, n_classes, seed, device)

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
