# %% [markdown]

# # MIT-BIH — Step 5: 40-Feature Hybrid Extraction
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Extracts the 40-feature hybrid vector from clarityBase.txt §5.
# Exact feature list (gap §9.7 resolved — paper counts don't add up;
# this implementation defines its own documented 40):
#
#   Group A  Statistical       (6)  mean, std, ptp, skewness, kurtosis, RMS
#   Group B1 Fiducial-intervals(5)  RR_pre, RR_post, QRS_dur, QT_interval, P_dur
#   Group B2 Fiducial-morphology(8) R/Q/S/T/P amplitudes, R/S ratio, P/R ratio, ST elev
#   Group C  Wavelet DWT      (12)  db4 4-level: energy+entropy for cA4 & cD4-cD1,
#                                   total energy, cD1 relative energy
#   Group D1 HRV time+freq    (5)  SDNN, RMSSD, LF, HF, LF/HF
#   Group D2 Non-linear       (4)  Poincaré SD1, SD2, Sample Entropy, SD1/SD2 ratio
#
# Delineation: simple rule-based windowed extrema (no neurokit2 dependency).
# HRV window : last min(50, available) RR intervals per record (§9.9).
# SampEn params: m=2, r=0.2*SD (§9.10 defaults).
#
# Input:  datasets/processed/mit_bih_segments.npz
# Output: datasets/processed/mit_bih_features.npz
#   X            (n_beats, 40)  float32
#   y            (n_beats,)     <U1
#   record_ids   (n_beats,)     str
#   r_samples    (n_beats,)     int32
#   feature_names (40,)         str

# %%

import numpy as np
import pywt
import scipy.signal as sps
import scipy.stats as spstats
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%

_HERE       = Path(__file__).resolve().parent
PROC_DIR    = _HERE.parent / "datasets" / "processed"
INPUT_PATH  = PROC_DIR / "mit_bih_segments.npz"
OUTPUT_PATH = PROC_DIR / "mit_bih_features.npz"

FS          = 360     # MIT-BIH sampling rate (Hz)
R_IDX       = 100     # R-peak index within each 256-sample window
HRV_WINDOW  = 50      # recent RR intervals used for HRV / non-linear features
HRV_FS      = 4.0     # uniform resample frequency for Welch PSD (Hz)

# %%

FEATURE_NAMES = [
    # A — Statistical (6)
    "stat_mean", "stat_std", "stat_ptp", "stat_skewness", "stat_kurtosis", "stat_rms",
    # B1 — Fiducial intervals (5)
    "fid_rr_pre_ms", "fid_rr_post_ms", "fid_qrs_dur_ms", "fid_qt_interval_ms", "fid_p_dur_ms",
    # B2 — Fiducial morphology (8)
    "fid_r_amp", "fid_q_amp", "fid_s_amp", "fid_t_amp", "fid_p_amp",
    "fid_rs_ratio", "fid_pr_ratio", "fid_st_elev",
    # C — DWT (12): energies then entropies (cA4, cD4, cD3, cD2, cD1), total, cD1-rel
    "dwt_cA4_energy", "dwt_cD4_energy", "dwt_cD3_energy", "dwt_cD2_energy", "dwt_cD1_energy",
    "dwt_cA4_entropy", "dwt_cD4_entropy", "dwt_cD3_entropy", "dwt_cD2_entropy", "dwt_cD1_entropy",
    "dwt_total_energy", "dwt_cD1_rel_energy",
    # D1 — HRV time + frequency (5)
    "hrv_sdnn_ms", "hrv_rmssd_ms", "hrv_lf", "hrv_hf", "hrv_lf_hf",
    # D2 — Non-linear (4)
    "nl_sd1", "nl_sd2", "nl_sampen", "nl_sd1_sd2_ratio",
]

assert len(FEATURE_NAMES) == 40

# %%

# ── Group A: Statistical ──────────────────────────────────────────────────────

def extract_statistical(beat: np.ndarray) -> list:
    return [
        float(np.mean(beat)),
        float(np.std(beat, ddof=0)),
        float(np.ptp(beat)),
        float(spstats.skew(beat)),
        float(spstats.kurtosis(beat)),
        float(np.sqrt(np.mean(beat ** 2))),
    ]


# ── Group B: Fiducial ─────────────────────────────────────────────────────────

def _fwhm_samples(sig: np.ndarray, peak_idx: int) -> float:
    """Full-width at half-maximum in samples. Baseline = median of first 10 samples."""
    baseline  = float(np.median(sig[:10]))
    amplitude = float(sig[peak_idx]) - baseline
    if abs(amplitude) < 1e-4:
        return 10.0   # ~28 ms default

    half_level = baseline + 0.5 * amplitude
    above      = amplitude > 0

    left = peak_idx
    while left > 0 and ((above and sig[left - 1] > half_level) or
                        (not above and sig[left - 1] < half_level)):
        left -= 1

    right = peak_idx
    n     = len(sig)
    while right < n - 1 and ((above and sig[right + 1] > half_level) or
                              (not above and sig[right + 1] < half_level)):
        right += 1

    return float(max(1, right - left))


def extract_fiducial(beat: np.ndarray, rr_pre_ms: float, rr_post_ms: float) -> list:
    """
    Groups B1 + B2: 13 features.
    Windows chosen so the full 256-sample beat at 360 Hz is covered:
      Q:  samples [80, 100)   — 0–55 ms before R
      S:  samples [100, 140)  — 0–111 ms after R
      T:  s_idx+15 to min(s_idx+95, 251)
      P:  samples [p_end-60, p_end) where p_end = max(q_idx-5, 30)
    """
    n = len(beat)

    r_amp = float(beat[R_IDX])

    # Q — local min before R
    q_idx = 80 + int(np.argmin(beat[80:R_IDX]))
    q_amp = float(beat[q_idx])

    # S — local min after R
    s_rel = int(np.argmin(beat[R_IDX: R_IDX + 40]))
    s_idx = R_IDX + s_rel
    s_amp = float(beat[s_idx])

    # T peak — local max in plausible T-wave window
    t_start = min(s_idx + 15, n - 30)
    t_end   = min(t_start + 90, n - 5)
    t_idx   = t_start + int(np.argmax(beat[t_start:t_end]))
    t_amp   = float(beat[t_idx])

    # P peak — local max in plausible P-wave window
    p_end   = max(q_idx - 5, 30)
    p_start = max(p_end - 60, 5)
    p_idx   = p_start + int(np.argmax(beat[p_start:p_end]))
    p_amp   = float(beat[p_idx])

    # Intervals (ms)
    qrs_dur_ms     = (s_idx - q_idx)  / FS * 1000.0
    qt_interval_ms = (t_idx - q_idx)  / FS * 1000.0
    p_dur_ms       = _fwhm_samples(beat, p_idx) / FS * 1000.0

    # Morphology
    rs_ratio = r_amp       / (abs(s_amp) + 1e-6)
    pr_ratio = abs(p_amp)  / (abs(r_amp) + 1e-6)

    # ST elevation: mean of a 5–25 ms window after S
    st_s = min(s_idx + int(0.005 * FS), n - 2)
    st_e = min(s_idx + int(0.025 * FS) + 1, n)
    st_elev = float(np.mean(beat[st_s:st_e])) if st_e > st_s else 0.0

    return [
        rr_pre_ms, rr_post_ms, qrs_dur_ms, qt_interval_ms, p_dur_ms,
        r_amp, q_amp, abs(s_amp), t_amp, p_amp,
        rs_ratio, pr_ratio, st_elev,
    ]


# ── Group C: Wavelet DWT ──────────────────────────────────────────────────────

def extract_wavelet(beat: np.ndarray) -> list:
    """
    12 DWT features.
    pywt.wavedec returns [cA4, cD4, cD3, cD2, cD1] (level=4).
    Per subband: energy (Σ c²) + normalised Shannon entropy (-Σ p·ln p).
    Plus: total energy and cD1 relative energy.
    """
    coeffs    = pywt.wavedec(beat.astype(float), "db4", level=4)
    energies  = []
    entropies = []

    for c in coeffs:
        energy = float(np.sum(c ** 2))
        energies.append(energy)
        if energy > 0:
            p       = (c ** 2) / energy
            p       = p[p > 0]
            entropy = float(-np.sum(p * np.log(p + 1e-12)))
        else:
            entropy = 0.0
        entropies.append(entropy)

    total_energy   = sum(energies)
    cD1_rel        = energies[-1] / (total_energy + 1e-12)  # cD1 is last

    return energies + entropies + [total_energy, cD1_rel]


# ── Group D: HRV / Non-linear ─────────────────────────────────────────────────

def _lf_hf(rr_ms: np.ndarray) -> tuple:
    """
    LF (0.04–0.15 Hz) and HF (0.15–0.40 Hz) power via Welch's method on a
    uniformly resampled RR series.  Returns (nan, nan) if the window is
    too short (<10 s) or has fewer than 10 intervals.
    """
    if len(rr_ms) < 10:
        return np.nan, np.nan

    rr_s  = rr_ms / 1000.0
    t_cum = np.concatenate([[0.0], np.cumsum(rr_s[:-1])])
    if t_cum[-1] < 10.0:
        return np.nan, np.nan

    t_uni  = np.arange(0.0, t_cum[-1], 1.0 / HRV_FS)
    rr_uni = np.interp(t_uni, t_cum, rr_ms)

    nperseg     = min(len(rr_uni), max(32, len(rr_uni) // 4))
    freqs, psd  = sps.welch(rr_uni, fs=HRV_FS, nperseg=nperseg)
    df          = freqs[1] - freqs[0]

    lf_mask = (freqs >= 0.04) & (freqs < 0.15)
    hf_mask = (freqs >= 0.15) & (freqs < 0.40)

    lf = float(np.sum(psd[lf_mask]) * df) if lf_mask.any() else np.nan
    hf = float(np.sum(psd[hf_mask]) * df) if hf_mask.any() else np.nan
    return lf, hf


def _sample_entropy(ts: np.ndarray, m: int = 2, r_coef: float = 0.2) -> float:
    """
    Sample Entropy with tolerance r = r_coef * SD(ts).
    Vectorised over j (inner loop) to keep runtime < 1 ms for N ≤ 50.
    """
    N = len(ts)
    if N < m + 2:
        return np.nan
    r = r_coef * float(np.std(ts, ddof=1))
    if r == 0:
        return 0.0

    def _count(tlen: int) -> int:
        rows = N - tlen + 1
        if rows < 2:
            return 0
        # Build embedding matrix (rows, tlen)
        embedded = np.array([ts[i: i + tlen] for i in range(rows)])
        count = 0
        for i in range(rows - 1):
            diffs = np.max(np.abs(embedded[i] - embedded[i + 1:]), axis=1)
            count += int(np.sum(diffs <= r))
        return count

    A = _count(m + 1)
    B = _count(m)
    return float(-np.log((A + 1e-10) / (B + 1e-10)))


def extract_hrv(rr_window_ms: np.ndarray) -> list:
    """
    Groups D1 + D2: 9 features from a window of recent RR intervals (ms).
    SD2 formula: SD2² = 2·SDNN² − 0.5·Var(ΔRR)  (§9.11 standard form).
    """
    n   = len(rr_window_ms)
    drr = np.diff(rr_window_ms)

    # D1 — time domain
    sdnn  = float(np.std(rr_window_ms, ddof=1)) if n >= 2   else np.nan
    rmssd = float(np.sqrt(np.mean(drr ** 2)))    if len(drr) >= 1 else np.nan

    # D1 — frequency domain
    lf, hf = _lf_hf(rr_window_ms)
    lf_hf  = (lf / (hf + 1e-10)) if not (np.isnan(lf) or np.isnan(hf)) else np.nan

    # D2 — Poincaré
    if len(drr) >= 2:
        sd1    = float(np.sqrt(max(0.0, 0.5  * np.var(drr, ddof=1))))
        sd2_sq = max(0.0, 2.0 * np.var(rr_window_ms, ddof=1)
                          - 0.5 * np.var(drr, ddof=1))
        sd2    = float(np.sqrt(sd2_sq))
    else:
        sd1 = sd2 = np.nan

    sampen        = _sample_entropy(rr_window_ms) if n >= 10 else np.nan
    sd1_sd2_ratio = (sd1 / (sd2 + 1e-10)) if not (np.isnan(sd1) or np.isnan(sd2)) else np.nan

    return [sdnn, rmssd, lf, hf, lf_hf, sd1, sd2, sampen, sd1_sd2_ratio]


# ──────────────────────────────────────────────────────────────────────────────

def extract_beat_features(
    beat: np.ndarray,
    rr_pre_ms: float,
    rr_post_ms: float,
    rr_window_ms: np.ndarray,
) -> np.ndarray:
    """Assemble the (40,) feature vector for one beat."""
    feats = (
        extract_statistical(beat)
        + extract_fiducial(beat, rr_pre_ms, rr_post_ms)
        + extract_wavelet(beat)
        + extract_hrv(rr_window_ms)
    )
    return np.array(feats, dtype=np.float32)


def process_record(
    record_id: str,
    beat_indices: np.ndarray,
    segments: np.ndarray,
    r_samples: np.ndarray,
) -> np.ndarray:
    """
    Extract features for every beat in one record.
    Beats are processed in chronological (r_sample) order so the HRV window
    always refers to truly preceding beats.
    Returns (n_beats, 40) float32 in the same order as beat_indices.
    """
    n     = len(beat_indices)
    X_rec = np.full((n, 40), np.nan, dtype=np.float32)

    # Sort by r_sample position within the record
    order            = np.argsort(r_samples[beat_indices])   # local sort permutation
    sorted_global    = beat_indices[order]                    # global indices in time order
    sorted_rsamp     = r_samples[sorted_global]

    # RR intervals (ms) between consecutive beats
    rr_ms = np.diff(sorted_rsamp).astype(float) / FS * 1000.0

    for pos in range(n):
        beat  = segments[sorted_global[pos], 0, :]   # channel 0 = MLII

        rr_pre  = rr_ms[pos - 1] if pos > 0     else np.nan
        rr_post = rr_ms[pos]     if pos < n - 1 else np.nan

        # HRV window: up to HRV_WINDOW recent intervals ending at current beat
        win_s      = max(0, pos - HRV_WINDOW)
        rr_window  = rr_ms[win_s:pos]           # intervals before current beat

        # Fill boundary NaNs with window mean so fiducial features always have values
        win_mean = float(np.nanmean(rr_window)) if len(rr_window) > 0 else 800.0
        if np.isnan(rr_pre):
            rr_pre = win_mean
        if np.isnan(rr_post):
            rr_post = win_mean
        if len(rr_window) == 0:
            rr_window = np.array([win_mean])

        feat_vec          = extract_beat_features(beat, rr_pre, rr_post, rr_window)
        X_rec[order[pos]] = feat_vec   # store at original (un-sorted) position

    return X_rec


# %%

def build_feature_matrix(
    segments: np.ndarray,
    labels: np.ndarray,
    record_ids: np.ndarray,
    r_samples: np.ndarray,
    n_workers: int = 8,
) -> np.ndarray:
    N       = len(labels)
    X       = np.full((N, 40), np.nan, dtype=np.float32)
    records = np.unique(record_ids)

    print(f"Extracting features for {N:,} beats across {len(records)} records "
          f"({n_workers} workers)...\n")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(process_record, rec,
                        np.where(record_ids == rec)[0],
                        segments, r_samples): rec
            for rec in records
        }
        for fut in as_completed(futures):
            rec = futures[fut]
            idx = np.where(record_ids == rec)[0]
            try:
                X[idx] = fut.result()
                nan_rows = int(np.isnan(X[idx]).any(axis=1).sum())
                print(f"  [ok]  {rec:>3s}  →  {len(idx):5,} beats  "
                      f"({nan_rows} rows with NaN)")
            except Exception as exc:
                print(f"  [error] {rec}: {exc}")

    # Impute remaining NaNs with per-column medians
    nan_cols = np.where(np.isnan(X).any(axis=0))[0]
    if len(nan_cols):
        print(f"\nImputing {len(nan_cols)} columns with NaN values...")
        for col in nan_cols:
            med = float(np.nanmedian(X[:, col]))
            X[np.isnan(X[:, col]), col] = med

    return X


# %%

if __name__ == "__main__":
    print(f"Loading {INPUT_PATH.name} ...")
    data       = np.load(INPUT_PATH, allow_pickle=True)
    segments   = data["segments"]    # (N, 2, 256)
    labels     = data["labels"]      # (N,)
    record_ids = data["record_ids"]  # (N,)
    r_samples  = data["r_samples"]   # (N,)

    print(f"  {len(labels):,} beats  |  segments shape: {segments.shape}\n")

    X = build_feature_matrix(segments, labels, record_ids, r_samples, n_workers=8)

    print(f"\n{'─'*52}")
    print(f"Feature matrix  : {X.shape}  (beats × features)")
    print(f"NaN remaining   : {int(np.isnan(X).sum())}")
    print(f"Memory (float32): {X.nbytes / 1e6:.1f} MB")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_PATH,
        X=X,
        y=labels,
        record_ids=record_ids,
        r_samples=r_samples,
        feature_names=np.array(FEATURE_NAMES),
    )
    print(f"\nSaved → {OUTPUT_PATH}  ({OUTPUT_PATH.stat().st_size / 1e6:.1f} MB)")
