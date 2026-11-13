# %% [markdown]
# ## calibrate.py — Temperature scaling for probability calibration
#
# A well-trained classifier may be over-confident (peak probabilities near 1)
# or under-confident.  For knowledge distillation, the SOFT probabilities at
# temperature T > 1 are more informative to the student than hard labels,
# because they encode the model's uncertainty about similar classes.
#
# Temperature scaling (Guo et al. 2017): divide logits by scalar T before softmax.
#   p_i(T) = softmax(logits / T)_i
#
# T = 1: original distribution.  T > 1: softer (more uniform). T < 1: sharper.
# Optimal T minimises NLL on DS1-val.
# ECE (Expected Calibration Error) measures calibration before and after.
#
# Reference:
#   Guo et al. (2017), "On Calibration of Modern Neural Networks," ICML.
#   Hinton et al. (2015), "Distilling the Knowledge in a Neural Network."

# %%
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# %%
class TemperatureScaler(nn.Module):
    """Single scalar temperature parameter, optimised on DS1-val NLL."""

    def __init__(self, T_init: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor([T_init]))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature.clamp(min=0.01)

    def calibrate(self, logits: np.ndarray, labels: np.ndarray,
                  max_iter: int = 200) -> float:
        """
        Fit temperature on (logits, labels) from DS1-val.
        Returns the optimal temperature value.
        """
        L = torch.from_numpy(logits).float()
        y = torch.from_numpy(labels).long()

        optimizer = torch.optim.LBFGS(self.parameters(), lr=0.01, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(self(L), y)
            loss.backward()
            return loss

        optimizer.step(closure)
        T = float(self.temperature.item())
        log.info("Temperature scaling: T = %.4f", T)
        return T


# %%
def compute_ece(probs: np.ndarray, labels: np.ndarray,
                n_bins: int = 15) -> float:
    """
    Expected Calibration Error: confidence-accuracy gap averaged over bins.
    ECE close to 0 means predicted confidence matches empirical accuracy.
    """
    preds  = probs.argmax(1)
    confs  = probs.max(1)
    corrects = (preds == labels).astype(float)

    ece = 0.0
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask   = (confs >= lo) & (confs < hi)
        if mask.sum() == 0:
            continue
        acc  = corrects[mask].mean()
        conf = confs[mask].mean()
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def calibrate_ensemble(logits_per_seed: list, labels: np.ndarray,
                       cfg_cal: dict) -> "TemperatureScaler":
    """
    Average logits across seeds and fit temperature on DS1-val.

    logits_per_seed : list of (N, K) arrays
    labels          : (N,) int64 from DS1-val
    """
    avg_logits = np.mean(logits_per_seed, axis=0)   # (N, K)

    T_init  = float(cfg_cal.get("T_init", 1.5))
    max_iter= int(cfg_cal.get("T_max_iter", 200))

    # Pre-calibration ECE
    probs_raw = F.softmax(torch.from_numpy(avg_logits).float(), dim=1).numpy()
    ece_before = compute_ece(probs_raw, labels)
    log.info("ECE before calibration: %.4f", ece_before)

    scaler = TemperatureScaler(T_init=T_init)
    T = scaler.calibrate(avg_logits, labels, max_iter=max_iter)

    probs_cal = F.softmax(
        torch.from_numpy(avg_logits).float() / T, dim=1
    ).numpy()
    ece_after = compute_ece(probs_cal, labels)
    log.info("ECE after calibration (T=%.3f): %.4f", T, ece_after)

    return scaler
