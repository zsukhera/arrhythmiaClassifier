# %% [markdown]
# ## augment.py — On-the-fly training augmentation
#
# Augmentation simulates inter-patient variability to improve DS2 generalisation.
# Applied to the BEAT and CONTEXT windows independently (same transforms, same
# random parameters for the two aligned views of one beat).
#
# Each transform fires independently with probability p=0.5.  Applied in-place
# on numpy arrays during DataLoader collation on CPU, before moving to GPU.
#
# DO NOT use SMOTE or other synthetic-beat generation: interpolating beats
# across patients leaks morphology shapes and can create unphysiological waveforms.
#
# Reference: Mousavi & Afghah (2019) uses signal-level augmentation for
# inter-patient robustness; Xia et al. (2023) uses similar noise injection.

# %%
import logging

import numpy as np

log = logging.getLogger(__name__)


# %%
def _rng_seed(seed=None) -> np.random.Generator:
    return np.random.default_rng(seed)


def amplitude_scale(x: np.ndarray, rng: np.random.Generator,
                    lo: float = 0.8, hi: float = 1.2) -> np.ndarray:
    """Multiply signal by a random scalar in [lo, hi]."""
    scale = rng.uniform(lo, hi)
    return (x * scale).astype(np.float32)


def gaussian_noise(x: np.ndarray, rng: np.random.Generator,
                   snr_db_lo: float = 20.0, snr_db_hi: float = 35.0) -> np.ndarray:
    """Add white Gaussian noise at a random SNR between snr_db_lo and snr_db_hi."""
    snr_db  = rng.uniform(snr_db_lo, snr_db_hi)
    snr_lin = 10 ** (snr_db / 10.0)
    signal_power = np.mean(x ** 2) + 1e-10
    noise_power  = signal_power / snr_lin
    noise = rng.standard_normal(len(x)).astype(np.float32) * float(np.sqrt(noise_power))
    return (x + noise).astype(np.float32)


def baseline_wander(x: np.ndarray, rng: np.random.Generator,
                    fs: float = 360.0,
                    freq_lo: float = 0.1, freq_hi: float = 0.5,
                    amp_max: float = 0.1) -> np.ndarray:
    """
    Add a low-frequency sine wave baseline wander.
    Amplitude is at most amp_max * std(x), so the effect scales with signal level.
    Realistic BW occurs from respiration (0.15-0.5 Hz) and body motion (0.1-0.3 Hz).
    """
    freq    = rng.uniform(freq_lo, freq_hi)
    amp     = rng.uniform(0.0, amp_max) * (np.std(x) + 1e-8)
    phase   = rng.uniform(0.0, 2.0 * np.pi)
    t       = np.arange(len(x)) / fs
    bw      = (amp * np.sin(2.0 * np.pi * freq * t + phase)).astype(np.float32)
    return (x + bw).astype(np.float32)


def time_shift(x: np.ndarray, rng: np.random.Generator,
               max_shift: int = 10) -> np.ndarray:
    """
    Shift signal by a random integer in [-max_shift, max_shift] samples.
    Pads/trims with edge values to maintain window length.
    Simulates imprecise R-peak detection (± 10 samples at 360 Hz ≈ ± 28 ms).
    """
    shift = int(rng.integers(-max_shift, max_shift + 1))
    if shift == 0:
        return x
    n = len(x)
    out = np.empty(n, dtype=np.float32)
    if shift > 0:
        out[:shift] = x[0]
        out[shift:] = x[:n - shift]
    else:
        s = -shift
        out[n - s:] = x[-1]
        out[:n - s] = x[s:]
    return out


def time_stretch(x: np.ndarray, rng: np.random.Generator,
                 lo: float = 0.9, hi: float = 1.1) -> np.ndarray:
    """
    Resample the signal by a random ratio in [lo, hi] using linear interpolation,
    then resize back to the original length.
    Simulates small differences in heart rate between beats of the same class.
    """
    factor = rng.uniform(lo, hi)
    n      = len(x)
    n_new  = max(1, round(n * factor))
    xp     = np.linspace(0.0, n - 1.0, n_new)
    x_new  = np.interp(xp, np.arange(n), x.astype(np.float64)).astype(np.float32)
    # Resize back to n using linear interpolation
    out_x  = np.linspace(0.0, n_new - 1.0, n)
    return np.interp(out_x, np.arange(n_new), x_new).astype(np.float32)


# %%
def augment_windows(beat: np.ndarray, context: np.ndarray,
                    rr_feats: np.ndarray, cfg_aug: dict,
                    rng: np.random.Generator) -> tuple:
    """
    Apply all augmentations to a single beat's windows and RR features.

    beat    : (beat_len,) float32
    context : (context_len,) float32  or  None if context disabled
    rr_feats: (6,) float32
    cfg_aug : augmentation config sub-dict
    rng     : per-worker RNG (for reproducibility across DataLoader workers)

    Returns (beat_aug, context_aug, rr_aug).
    """
    p = 0.5   # probability for each transform

    fs = 360.0

    def _apply(x, fn):
        return fn(x, rng) if rng.random() < p else x

    beat_a = beat.copy()
    ctx_a  = context.copy() if context is not None else None

    amp_lo, amp_hi   = cfg_aug.get("amplitude_range", [0.8, 1.2])
    snr_lo, snr_hi   = cfg_aug.get("noise_snr_db", [20, 35])
    bw_lo,  bw_hi    = cfg_aug.get("bw_freq_range", [0.1, 0.5])
    bw_max           = cfg_aug.get("bw_amp_max", 0.1)
    max_shift        = cfg_aug.get("time_shift", 10)
    st_lo,  st_hi    = cfg_aug.get("stretch_range", [0.9, 1.1])
    rr_jitter        = cfg_aug.get("rr_jitter", 0.03)

    # Apply same amplitude scale and noise to both beat and context windows
    if rng.random() < p:
        s = float(rng.uniform(amp_lo, amp_hi))
        beat_a = (beat_a * s).astype(np.float32)
        if ctx_a is not None:
            ctx_a = (ctx_a * s).astype(np.float32)

    if rng.random() < p:
        beat_a = gaussian_noise(beat_a, rng, snr_lo, snr_hi)
        if ctx_a is not None:
            ctx_a = gaussian_noise(ctx_a, rng, snr_lo, snr_hi)

    # Baseline wander applied independently (simulates electrode-specific wander)
    beat_a = _apply(beat_a, lambda x, r: baseline_wander(x, r, fs, bw_lo, bw_hi, bw_max))
    if ctx_a is not None:
        ctx_a = _apply(ctx_a, lambda x, r: baseline_wander(x, r, fs, bw_lo, bw_hi, bw_max))

    # Time transforms applied only to beat window (context is long, shifts matter less)
    beat_a = _apply(beat_a, lambda x, r: time_shift(x, r, max_shift))
    beat_a = _apply(beat_a, lambda x, r: time_stretch(x, r, st_lo, st_hi))

    # RR jitter: perturb each ratio feature by ±rr_jitter
    rr_a = rr_feats.copy()
    if rng.random() < p:
        noise_rr = rng.uniform(-rr_jitter, rr_jitter, size=rr_a.shape).astype(np.float32)
        rr_a = rr_a + noise_rr

    return beat_a, ctx_a, rr_a
