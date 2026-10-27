"""Baseline-wander removal and fixed-length beat windows."""

import logging

import numpy as np
from scipy import signal as sps

log = logging.getLogger(__name__)


def baseline_sos(order: int, cutoff_hz: float, rp_db: float, rs_db: float, fs: float) -> np.ndarray:
    """Elliptic low-pass that estimates the baseline (everything below `cutoff_hz`)."""
    return sps.ellip(order, rp_db, rs_db, cutoff_hz, btype="low", output="sos", fs=fs)


def remove_baseline(x: np.ndarray, sos: np.ndarray, zero_phase: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Return (x - baseline, baseline)."""
    base = sps.sosfiltfilt(sos, x) if zero_phase else sps.sosfilt(sos, x)
    return x - base, base


def beat_windows(x: np.ndarray, r: np.ndarray, before: int, after: int) -> tuple[np.ndarray, np.ndarray]:
    """Cut [r - before, r + after] around every R peak.

    Returns (windows of shape (n_kept, before + after + 1), boolean keep mask over `r`).
    Beats whose window runs past either end of the record are dropped.
    """
    r = np.asarray(r, dtype=np.int64)
    keep = (r - before >= 0) & (r + after < len(x))
    offsets = np.arange(-before, after + 1)
    return x[r[keep, None] + offsets[None, :]], keep
