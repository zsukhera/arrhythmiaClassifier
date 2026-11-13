# %% [markdown]
# ## preprocess.py — Signal filtering, beat/context windowing, normalisation
#
# Two aligned views per beat (both centred on R peak):
#   1. Beat window   (256 samples @ 360 Hz = 0.711 s): P-QRS-T morphology.
#      90 before R, 166 after — asymmetric because the T wave is longer than the P.
#   2. Context window (1080 samples @ 360 Hz = 3 s): rhythm context.
#      Centred (540 each side) so the attention model can see the preceding and
#      following beats.  S beats have early P waves before R; V beats have
#      compensatory pauses after R.  Ablation: 720 samples (2 s) or no context.
#
# Normalisation: Z-score per window (mean=0, std=1) on each beat independently.
# This removes inter-patient amplitude differences — the main generalisation
# failure mode for DS1→DS2.  It is NOT done per-record so that the teacher
# and student receive identically processed windows.
#
# Runs at 360 Hz (no resampling).  The student (paper3) resamples to 128 Hz,
# which makes TFLite input 32 samples; the teacher sees 256/1080 at 360 Hz.
# This is acceptable because the teacher is cloud-only.

# %%
import logging
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

log = logging.getLogger(__name__)


# %%
def bandpass_filter(signal: np.ndarray, fs: float,
                    low_hz: float = 0.5, high_hz: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth band-pass filter.
    Removes baseline wander (< 0.5 Hz) and powerline noise (> 40 Hz).
    Uses sosfiltfilt (second-order sections + forward-backward) for
    numerical stability and zero group delay.
    """
    nyq = fs / 2.0
    sos = butter(order, [low_hz / nyq, high_hz / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal).astype(np.float32)


# %%
def _zscore(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Z-score a 1D window; stable when std ≈ 0 (flat-line artefact)."""
    mu  = np.mean(x)
    std = np.std(x)
    return ((x - mu) / (std + eps)).astype(np.float32)


def extract_window(signal: np.ndarray, r_sample: int, before: int, after: int,
                   normalise: bool = True) -> Tuple[np.ndarray, bool]:
    """
    Extract a fixed-length window centred on r_sample.
    Edge padding uses the nearest signal value (edge mode), not zeros.
    Zero-padding would introduce an artificial discontinuity that shifts
    the window's mean and std, hurting normalisation.

    Returns (window, was_padded).
    """
    n = len(signal)
    start = r_sample - before
    end   = r_sample + after
    win_len = before + after

    was_padded = (start < 0) or (end > n)

    # np.pad with 'edge' replicates the boundary sample value.
    if was_padded:
        pad_l = max(0, -start)
        pad_r = max(0, end - n)
        clipped = signal[max(0, start): min(n, end)]
        window = np.pad(clipped, (pad_l, pad_r), mode="edge")
    else:
        window = signal[start:end].copy()

    # Ensure correct length (may differ by 1 if end==n exactly)
    if len(window) != win_len:
        window = np.resize(window, win_len)

    if normalise:
        window = _zscore(window)
    return window.astype(np.float32), was_padded


# %%
def preprocess_record(signal_raw: np.ndarray, r_samples: np.ndarray,
                      cfg: dict) -> dict:
    """
    Full preprocessing pipeline for one MIT-BIH record.

    Returns dict with arrays indexed over all beats:
      beat_segs    : (n_beats, beat_len)      float32 — morphology window
      context_segs : (n_beats, context_len)   float32 — rhythm context window
      was_padded   : (n_beats,)               bool    — any boundary beats
      r_samples    : (n_beats,)               int     — R-peak positions (unchanged)
    """
    p_cfg = cfg["preprocess"]
    fs    = float(cfg["data"]["fs"])

    # --- bandpass filter ---
    if p_cfg.get("filter", "none") == "bandpass":
        signal = bandpass_filter(
            signal_raw.astype(np.float64), fs,
            low_hz=float(p_cfg["bandpass_low_hz"]),
            high_hz=float(p_cfg["bandpass_high_hz"]),
            order=int(p_cfg["bandpass_order"]),
        )
    else:
        signal = signal_raw.astype(np.float32)

    normalise = p_cfg.get("normalisation", "zscore") == "zscore"

    beat_len     = int(p_cfg["beat_len"])
    beat_before  = int(p_cfg["beat_before_r"])
    beat_after   = int(p_cfg["beat_after_r"])
    ctx_len      = int(p_cfg.get("context_len", 1080))
    ctx_half     = int(p_cfg.get("context_half", 540))

    n = len(r_samples)
    beat_segs    = np.zeros((n, beat_len),   dtype=np.float32)
    context_segs = np.zeros((n, ctx_len),    dtype=np.float32) if ctx_len > 0 else None
    was_padded   = np.zeros(n, dtype=bool)

    for i, r in enumerate(r_samples):
        b, padded_b = extract_window(signal, int(r), beat_before, beat_after, normalise)
        beat_segs[i] = b
        was_padded[i] = padded_b

        if ctx_len > 0:
            c, padded_c = extract_window(signal, int(r), ctx_half, ctx_half, normalise)
            context_segs[i] = c
            was_padded[i] |= padded_c

    n_padded = int(was_padded.sum())
    if n_padded:
        log.debug("preprocess_record: %d/%d beats edge-padded", n_padded, n)

    return {
        "beat_segs":    beat_segs,
        "context_segs": context_segs,
        "was_padded":   was_padded,
        "r_samples":    r_samples,
    }
