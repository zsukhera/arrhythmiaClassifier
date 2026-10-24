"""High-pass baseline removal and min-max normalisation, both free of test leakage."""

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt


def highpass_sos(order: int, cutoff_hz: float, fs: float) -> np.ndarray:
    return butter(order, cutoff_hz, btype="highpass", fs=fs, output="sos")


def filter_without_leakage(signal: np.ndarray, sos: np.ndarray, t_train: int, t_end: int,
                           zero_phase: bool = True) -> np.ndarray:
    """Filter `signal[:t_end]` so that no sample before `t_train` depends on later samples.

    A zero-phase filter run over the whole signal would let the test window
    leak backwards into the end of the training window. The training window
    is therefore filtered on its own; the test window is taken from a pass
    over [0, t_end), which only uses earlier (training) samples as context.
    """
    filt = sosfiltfilt if zero_phase else sosfilt
    train = filt(sos, signal[:t_train])
    full = filt(sos, signal[:t_end])
    return np.concatenate([train, full[t_train:]])


def minmax_normalize(x: np.ndarray, t_train: int, scope: str = "train") -> np.ndarray:
    """Scale to [0, 1] with min/max from the training window (scope='train') or all of x."""
    if scope not in ("train", "full"):
        raise ValueError(f"normalization_scope must be 'train' or 'full', got {scope!r}")
    ref = x[:t_train] if scope == "train" else x
    lo, hi = float(ref.min()), float(ref.max())
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)
