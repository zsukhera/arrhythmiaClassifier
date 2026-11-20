# %% [markdown]
# ## preprocess.py — Resampling, segmentation, normalisation, derivative
#
# Reference: Farag (2023), Sections 3 and 5.2.
#
# Pipeline per record:
#   1. Resample 360 Hz -> 128 Hz with resample_poly (preserves peak timing)
#   2. Map R-peak indices from 360 Hz to 128 Hz
#   3. Optionally apply 0.5-40 Hz bandpass filter (ablation; default: none)
#   4. Segment 64 samples centred on each R-peak (32 before, 32 after) [assumption]
#   5. Edge-pad segments at record boundaries  [assumption: np.pad mode='edge']
#   6. Subtract per-segment mean [assumption: paper Sec 5.2 "mean" step]
#   7. If input=derivative: np.diff then edge-pad back to 64 samples [assumption]
#   8. If NK=32: crop to central 32 samples (drop 16 from each edge, Sec 5.2)

# %%
import logging
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

log = logging.getLogger(__name__)

# Resampling ratio: 360 Hz -> 128 Hz.
# resample_poly(x, up=16, down=45) because 16/45 = 128/360 exactly.
_UP, _DOWN = 16, 45


# %%
def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5, high: float = 40.0,
                    order: int = 4) -> np.ndarray:
    """
    Zero-phase 4th-order Butterworth bandpass (0.5–40 Hz, paper ablation).
    Zero-phase via sosfiltfilt eliminates group-delay distortion near the R-peak,
    which matters because the Conv1D kernel is only NK=32-64 samples wide.
    Default filter=none matches the paper's training protocol (Sec 3).
    """
    nyq = fs / 2.0
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfiltfilt(sos, signal)


def resample_signal(signal: np.ndarray, up: int = _UP, down: int = _DOWN) -> np.ndarray:
    """
    Resample from 360 Hz to 128 Hz using polyphase filtering.
    resample_poly is preferred over scipy.signal.resample (DFT-based) because
    it preserves causality in the filter and handles arbitrary-length signals.
    ASSUMPTION: the paper does not specify the resampling method.
    """
    return resample_poly(signal, up, down).astype(np.float64)


def map_r_peaks(r_samples_360: np.ndarray, ratio: float = 128.0 / 360.0) -> np.ndarray:
    """
    Map R-peak sample indices from 360 Hz to 128 Hz.
    Uses rounding (not floor/ceil) to minimise timing error (<= 0.5 sample).
    ASSUMPTION: the paper does not specify the mapping method.
    """
    return np.round(r_samples_360 * ratio).astype(np.int64)


# %%
def segment_beat(signal_128: np.ndarray, r_sample: int, before: int = 32,
                 after: int = 32) -> tuple:
    """
    Extract a window of `before + after` samples centred on r_sample.
    If the window extends beyond the signal, edge-pad (repeat boundary value).
    Returns (segment, was_padded).

    Edge-padding rather than dropping preserves the annotated beat count so
    that RR features remain consistent with the full record.
    ASSUMPTION: before=32, after=32 gives 64 samples (0.5 s at 128 Hz).
    """
    n = len(signal_128)
    start = r_sample - before
    end   = r_sample + after
    pad_left  = max(0, -start)
    pad_right = max(0,  end - n)
    was_padded = (pad_left > 0) or (pad_right > 0)
    s_clip = max(0, start)
    e_clip = min(n, end)
    seg = signal_128[s_clip:e_clip]
    if was_padded:
        seg = np.pad(seg, (pad_left, pad_right), mode="edge")
    return seg.astype(np.float64), was_padded


def normalise_segment(segment: np.ndarray, method: str = "mean") -> np.ndarray:
    """
    Normalise a beat segment.
    'mean'   : subtract the segment mean (paper Sec 5.2) [assumption]
    'z-score': subtract mean and divide by std (ablation)
    'none'   : no normalisation
    """
    if method == "mean":
        return segment - segment.mean()
    if method == "z-score":
        s = segment.std()
        return (segment - segment.mean()) / (s if s > 1e-9 else 1.0)
    if method == "none":
        return segment.copy()
    raise ValueError(f"Unknown normalisation method: {method!r}")


def compute_derivative(segment: np.ndarray) -> np.ndarray:
    """
    First-order discrete derivative of a 64-sample segment.
    np.diff reduces length by 1 (64 -> 63 samples); we edge-pad the right
    end to restore 64 samples.
    ASSUMPTION about edge-padding direction: append last diff value.
    Why derivative: shifts the focus from amplitude to morphology slope,
    making the matched-filter robust to baseline wander.
    """
    d = np.diff(segment)
    return np.pad(d, (0, 1), mode="edge")  # 63 -> 64 samples


def crop_to_nk(segment: np.ndarray, nk: int, full: int = 64) -> np.ndarray:
    """
    Crop the central `nk` samples from a `full`-sample segment.
    For NK=32: drop 16 samples from each edge (paper Sec 5.2).
    For NK=64: no-op (nk == full).
    The idea: a shorter kernel reduces parameters further while keeping the
    most informative morphology near the R-peak.
    """
    if nk == full:
        return segment
    margin = (full - nk) // 2
    return segment[margin: margin + nk]


# %%
def preprocess_record(raw_signal: np.ndarray, r_samples_360: np.ndarray,
                      cfg: dict) -> dict:
    """
    Full preprocessing pipeline for one record.

    Parameters
    ----------
    raw_signal    : 1-D signal at 360 Hz
    r_samples_360 : R-peak sample indices at 360 Hz (beat annotations only)
    cfg           : config dict (preprocess + input sections)

    Returns
    -------
    dict with keys:
      signal_128 : resampled signal (float64)
      r_samples  : R-peak sample indices in the 128 Hz signal
      segments   : (n_beats, nk, n_channels) float32 array — ready for the model
      was_padded : (n_beats,) bool array
    """
    p_cfg  = cfg["preprocess"]
    i_cfg  = cfg["input"]
    before = int(p_cfg["before_r"])
    after  = int(p_cfg["after_r"])
    full   = int(p_cfg["segment_samples"])  # 64
    nk     = int(i_cfg["nk"])
    norm   = p_cfg["normalization"]
    mode   = i_cfg["mode"]          # signal | derivative | both
    filt   = p_cfg["filter"]        # none | bandpass

    # Step 1 — bandpass (ablation; default none)
    if filt == "bandpass":
        raw_signal = bandpass_filter(raw_signal, cfg["data"]["fs_original"],
                                     float(p_cfg["bandpass_low_hz"]),
                                     float(p_cfg["bandpass_high_hz"]),
                                     int(p_cfg["bandpass_order"]))

    # Step 2 — resample 360 -> 128 Hz
    signal_128 = resample_signal(raw_signal)

    # Step 3 — map R-peak indices to the new sample rate
    r_samples = map_r_peaks(r_samples_360)

    n_beats  = len(r_samples)
    segments = []
    padded   = []

    for r in r_samples:
        # Step 4 — extract 64-sample window
        seg, wp = segment_beat(signal_128, int(r), before, after)
        padded.append(wp)

        # Step 5 — normalise
        seg = normalise_segment(seg, norm)

        # Step 6 — build channels
        if mode == "signal":
            channels = [crop_to_nk(seg, nk, full)]
        elif mode == "derivative":
            channels = [crop_to_nk(compute_derivative(seg), nk, full)]
        elif mode == "both":
            channels = [crop_to_nk(seg, nk, full),
                        crop_to_nk(compute_derivative(seg), nk, full)]
        else:
            raise ValueError(f"Unknown input mode: {mode!r}")

        # shape: (nk, n_channels)
        segments.append(np.stack(channels, axis=-1))

    segments_arr = np.array(segments, dtype=np.float32)  # (n_beats, nk, n_channels)
    padded_arr   = np.array(padded,   dtype=bool)

    n_padded = int(padded_arr.sum())
    if n_padded > 0:
        log.info("preprocess_record: %d/%d beats edge-padded", n_padded, n_beats)

    return {
        "signal_128": signal_128,
        "r_samples":  r_samples,
        "segments":   segments_arr,
        "was_padded": padded_arr,
    }
