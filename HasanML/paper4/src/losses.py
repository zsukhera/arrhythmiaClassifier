# %% [markdown]
# ## losses.py — Focal loss, weighted CE, label smoothing
#
# Focal loss (Lin et al. 2017, RetinaNet) reduces the loss for well-classified
# easy examples so the model focuses on hard, misclassified ones:
#
#   FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
#
# For inter-patient ECG: N beats (90 %) are easy, S beats are hard.
# gamma=2 is the standard choice; (1-p_t)^2 reduces easy-N loss by 100x.
#
# Class weights alpha: sqrt(1/freq) instead of 1/freq.
#   - Full inverse-frequency would weight S at ~15x, destroying N/S PPV.
#   - Sqrt normalises the extremes while still giving minority classes a boost.
#   - Computed only on DS1-train so DS2 is never seen.
#
# References:
#   Lin et al. (2017), "Focal Loss for Dense Object Detection," ICCV.
#   Farag (2023): uses balanced class weights in training.

# %%
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# %%
def compute_class_alpha(y: np.ndarray, n_classes: int,
                        mode: str = "sqrt") -> torch.Tensor:
    """
    Compute per-class weight vector from DS1-train label array.

    mode="sqrt"  : alpha_i = sqrt(N / (n_classes * count_i)), normalised
    mode="inv"   : alpha_i = N / (n_classes * count_i), normalised
    mode="uniform": all ones

    Returns torch.Tensor of shape (n_classes,).
    """
    counts = np.array([(y == i).sum() for i in range(n_classes)], dtype=np.float64)
    total  = counts.sum()
    if mode == "uniform":
        return torch.ones(n_classes, dtype=torch.float32)

    raw = total / (n_classes * np.maximum(counts, 1.0))
    if mode == "sqrt":
        raw = np.sqrt(raw)
    # Normalise so the mean weight is 1 (keeps effective learning rate stable)
    raw = raw / (raw.mean() + 1e-8)
    alpha = torch.tensor(raw, dtype=torch.float32)
    log.info("class alpha weights: %s", {i: f"{v:.3f}" for i, v in enumerate(alpha.tolist())})
    return alpha


# %%
class FocalLoss(nn.Module):
    """
    Multi-class focal loss with per-class alpha weights.

    Uses log-softmax + (1 - softmax)^gamma to stay numerically stable:
    if we used cross-entropy then multiplied by (1-pt)^gamma we'd compute
    softmax twice.  Instead: pt = exp(log_softmax), FL = -alpha*(1-pt)^gamma*log_pt.
    """
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.reduction = reduction
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, n_classes) unnormalised
        targets : (B,) int64 class indices
        """
        log_p = F.log_softmax(logits, dim=1)             # (B, n_classes)
        p     = log_p.exp()                               # softmax probs
        # Gather the log-prob and prob for the true class
        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B,)
        pt     = p.gather(1, targets.unsqueeze(1)).squeeze(1)       # (B,)

        # Per-class alpha weight for each sample
        alpha_t = self.alpha.to(logits.device)[targets]   # (B,)

        loss = -alpha_t * (1.0 - pt) ** self.gamma * log_pt

        if self.reduction == "mean":
            # Weighted mean (same convention as F.cross_entropy(weight=...)),
            # so FocalLoss(gamma=0) == weighted CE exactly.
            return loss.sum() / alpha_t.sum().clamp_min(1e-12)
        if self.reduction == "sum":
            return loss.sum()
        return loss


class WeightedCE(nn.Module):
    """Standard weighted cross-entropy (ablation: gamma=0 focal)."""
    def __init__(self, alpha: torch.Tensor):
        super().__init__()
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets,
                               weight=self.alpha.to(logits.device))


class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing (Szegedy et al. 2016)."""
    def __init__(self, alpha: torch.Tensor, eps: float = 0.05):
        super().__init__()
        self.eps = eps
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets,
                               weight=self.alpha.to(logits.device),
                               label_smoothing=self.eps)


# %%
def build_loss(cfg, alpha: torch.Tensor = None, *, class_counts=None,
               gamma: float = 2.0, weight_mode: str = "sqrt",
               label_smooth_eps: float = 0.05) -> nn.Module:
    """
    Build loss function.

    Two call styles:
      build_loss(cfg_dict, alpha)                       -- from the YAML config
      build_loss("focal", class_counts=..., gamma=2.0,  -- standalone
                 weight_mode="sqrt")
    """
    if isinstance(cfg, str):
        if alpha is None:
            if class_counts is None:
                raise ValueError("build_loss(str, ...) needs alpha or class_counts")
            counts = np.asarray(class_counts)
            y = np.repeat(np.arange(len(counts)), counts.astype(int))
            alpha = compute_class_alpha(y, len(counts), mode=weight_mode)
        cfg = {"training": {"loss": cfg, "focal_gamma": gamma,
                            "label_smooth_eps": label_smooth_eps}}
    t = cfg["training"]
    loss_name = str(t.get("loss", "focal"))

    if loss_name == "focal":
        return FocalLoss(alpha, gamma=float(t.get("focal_gamma", 2.0)))
    elif loss_name == "weighted_ce":
        return WeightedCE(alpha)
    elif loss_name == "label_smooth":
        return LabelSmoothingCE(alpha, eps=float(t.get("label_smooth_eps", 0.05)))
    elif loss_name == "ce":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unknown loss: {loss_name}")
