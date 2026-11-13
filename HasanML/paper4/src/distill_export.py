# %% [markdown]
# ## distill_export.py — Soft labels for distillation + ONNX/TorchScript export
#
# The KAN-Former is the TEACHER. Its job is to produce soft probability
# distributions over AAMI classes that the student (Farag 2023 Keras 1D-CNN
# in HasanML/paper3) uses instead of one-hot labels.
#
# Soft labels at temperature T:
#   p_i(T) = exp(logit_i / T) / sum_j exp(logit_j / T)
#
# Higher T → softer distribution → more information transferred from teacher.
# T=1: teacher's output probabilities.
# T=2 or T=4: more uniform, encodes relative similarity between classes.
#
# The student trains on:  L = alpha * CE(y_pred, hard_labels)
#                             + (1-alpha) * T^2 * KL(p_T_teacher || p_T_student)
# (multiply by T^2 to account for the 1/T^2 scale of the KL gradient)
#
# Reference: Hinton et al. (2015), "Distilling the Knowledge in a Neural Network."
# Output format: framework-agnostic .npy arrays so the Keras student can load
# them without installing PyTorch.

# %%
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)


# %%
@torch.no_grad()
def collect_logits(model, ds: dict, device: torch.device,
                   batch_size: int = 512) -> np.ndarray:
    """
    Run the model (with EMA weights already applied) over a full dataset dict.
    Returns logits (N, K) numpy float32.
    """
    from .train import BeatDataset, _collate_skip_none
    from torch.utils.data import DataLoader

    dataset = BeatDataset(ds["beat_segs"], ds.get("context_segs"),
                          ds["rr_feats"], ds["y"], augment=False)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                         collate_fn=_collate_skip_none)
    model.eval()
    all_logits = []
    for beats, ctxs, rrs, _ in loader:
        beats = beats.to(device)
        ctxs  = ctxs.to(device) if ctxs is not None else None
        rrs   = rrs.to(device)
        logits = model(beats, ctxs, rrs)
        all_logits.append(logits.cpu())
    return torch.cat(all_logits).float().numpy()


def ensemble_logits(logits_per_seed: List[np.ndarray]) -> np.ndarray:
    """
    Average logits across seeds (before softmax) for the ensemble.
    Averaging logits > averaging probabilities because logits are unconstrained;
    probability averaging is equivalent but requires all seeds to agree on scale.
    """
    return np.mean(logits_per_seed, axis=0)


# %%
def save_soft_labels(avg_logits: np.ndarray, ds: dict,
                     records_list: List[str],
                     out_dir: Path, temperatures: List[int],
                     split_name: str = "ds1",
                     temperature_scaler=None) -> None:
    """
    Save soft label arrays and beat index CSV.

    For each temperature T, saves:
      out_dir/soft_labels_{split_name}_T{T}.npy   — (N, K) float32, sums to 1 per row

    Also saves:
      out_dir/beat_index_{split_name}.csv          — record, r_sample, symbol, aami_label
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Apply calibrated temperature if provided, then save at each T
    cal_logits = avg_logits
    if temperature_scaler is not None:
        import torch
        L = torch.from_numpy(avg_logits).float()
        cal_logits = temperature_scaler(L).detach().numpy()

    for T in temperatures:
        probs = F.softmax(
            torch.from_numpy(cal_logits).float() / T, dim=1
        ).numpy().astype(np.float32)
        assert np.allclose(probs.sum(1), 1.0, atol=1e-5), \
            "soft labels must sum to 1"
        out_path = out_dir / f"soft_labels_{split_name}_T{T}.npy"
        np.save(str(out_path), probs)
        log.info("Saved soft labels (T=%d): %s  shape=%s", T, out_path, probs.shape)

    # Beat index CSV
    if "record_ids" in ds and "r_samples" in ds and "symbols" in ds and "y" in ds:
        df = pd.DataFrame({
            "record":     ds["record_ids"],
            "r_sample":   ds["r_samples"],
            "symbol":     ds["symbols"],
            "aami_label": ds["y"],
        })
        idx_path = out_dir / f"beat_index_{split_name}.csv"
        df.to_csv(idx_path, index=False)
        log.info("Saved beat index: %s  (%d beats)", idx_path, len(df))


# %%
def export_onnx(model, cfg: dict, out_path: Path,
                device: torch.device = torch.device("cpu")) -> None:
    """Export the model to ONNX for inference in any framework."""
    beat_len = int(cfg["preprocess"]["beat_len"])
    ctx_len  = int(cfg["preprocess"].get("context_len", 1080))
    n_rr     = int(cfg["rr"]["n_features"])

    model.eval()
    beat = torch.zeros(1, 1, beat_len,  device=device)
    ctx  = torch.zeros(1, 1, ctx_len,   device=device) if ctx_len > 0 else None
    rr   = torch.zeros(1, n_rr,         device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    input_names   = ["beat", "rr"] if ctx is None else ["beat", "context", "rr"]
    dynamic_axes  = {n: {0: "batch"} for n in input_names + ["logits"]}
    inputs        = (beat, ctx, rr) if ctx is not None else (beat, None, rr)

    # ONNX export requires a wrapper that handles None context
    class _Wrapper(torch.nn.Module):
        def __init__(self, m, has_ctx):
            super().__init__()
            self.m = m
            self.has_ctx = has_ctx
        def forward(self, beat, ctx, rr):
            return self.m(beat, ctx if self.has_ctx else None, rr)

    wrapper = _Wrapper(model, ctx_len > 0)
    ex_in = (beat, ctx if ctx is not None else torch.zeros(1, 1, ctx_len),  rr)

    torch.onnx.export(
        wrapper, ex_in, str(out_path),
        input_names=["beat", "context", "rr"],
        output_names=["logits"],
        dynamic_axes={n: {0: "batch"} for n in ["beat", "context", "rr", "logits"]},
        opset_version=14,
        do_constant_folding=True,
    )
    log.info("ONNX model saved: %s", out_path)


def export_torchscript(model, cfg: dict, out_path: Path,
                       device: torch.device = torch.device("cpu")) -> None:
    """Export to TorchScript for C++/mobile deployment."""
    beat_len = int(cfg["preprocess"]["beat_len"])
    ctx_len  = int(cfg["preprocess"].get("context_len", 1080))
    n_rr     = int(cfg["rr"]["n_features"])

    model.eval()
    beat = torch.zeros(2, 1, beat_len, device=device)
    ctx  = torch.zeros(2, 1, ctx_len,  device=device) if ctx_len > 0 else None
    rr   = torch.zeros(2, n_rr,        device=device)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        traced = torch.jit.trace(model, (beat, ctx, rr))
        traced.save(str(out_path))
        log.info("TorchScript saved: %s", out_path)
    except Exception as exc:
        log.warning("TorchScript export failed: %s — trying script()", exc)
        try:
            scripted = torch.jit.script(model)
            scripted.save(str(out_path))
            log.info("TorchScript (scripted) saved: %s", out_path)
        except Exception as exc2:
            log.error("TorchScript failed: %s", exc2)
