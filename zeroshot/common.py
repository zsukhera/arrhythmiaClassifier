"""
Zero-shot cross-dataset evaluation — shared helpers.

All MIT-BIH-trained models are used frozen.  Every external 10-s record AND every
MIT-BIH 10-s strip goes through exactly the same code path:

    lead-II/MLII signal at 360 Hz  ->  R-peak detection  ->  per-model features
    ->  beat predictions  ->  record aggregation (rule fixed on MIT-BIH DS1)

Re-uses (read-only imports) source/features.py and HasanML/paper3/src/{preprocess,rr}.py
so the feature code is byte-identical to what the models were trained with.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
    os.environ.setdefault(_v, "4")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import scipy.signal as sps

ROOT = Path(__file__).resolve().parents[1]
ZS = ROOT / "zeroshot"
OUT = ZS / "out"
EXT = ROOT / "datasets" / "external"
MITDB = ROOT / "datasets" / "mitbih" / "mitdb"
PAPER3 = ROOT / "HasanML" / "paper3"

sys.path.insert(0, str(ROOT / "source"))
sys.path.insert(0, str(PAPER3))

import features as lgbm_feats                                  # noqa: E402  (source/features.py)
from src.preprocess import (resample_signal, map_r_peaks,      # noqa: E402
                            segment_beat, normalise_segment,
                            compute_derivative, crop_to_nk)
from src.rr import compute_rr_features                         # noqa: E402

FS = 360
STRIP_SEC = 10
STRIP_LEN = FS * STRIP_SEC

DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
       "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
       "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
       "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
       "233", "234"]
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

LGBM_CLASSES = ["N", "L", "R", "V", "A"]
CNN_CLASSES = ["N", "S", "V"]
TAU_S = 0.50   # official P1 ensemble threshold (run_official_p1.py)

# Record-level conditions evaluated per model
LGBM_CONDS = ["L", "R", "V", "A"]   # A ~ supraventricular premature (PAC)
CNN_CONDS = ["S", "V"]


# ─────────────────────────────────────────────────────────── signal helpers ──
def resample_to_360(sig: np.ndarray, fs_in: float) -> np.ndarray:
    fs_in = int(round(fs_in))
    if fs_in == FS:
        return sig.astype(np.float64)
    g = np.gcd(FS, fs_in)
    return sps.resample_poly(sig, FS // g, fs_in // g).astype(np.float64)


_SOS_LGBM = lgbm_feats_sos = sps.butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
_SOS_DET = sps.butter(2, [5, 20], btype="band", fs=FS, output="sos")


def detect_rpeaks(sig360: np.ndarray, method: str = "xqrs", refine: int = 18) -> np.ndarray:
    """
    R-peak detection on a 360 Hz single-lead signal.
      xqrs : wfdb.processing.XQRS (default parameters)
      gqrs : wfdb.processing.gqrs_detect
    Detected QRS locations are then snapped to the extremum of the 0.5–40 Hz
    band-passed signal (largest |x - median|) within ±`refine` samples (50 ms)
    so the beat window is centred on the R wave like the .atr annotations.
    """
    import wfdb.processing as wp
    x = np.asarray(sig360, dtype=np.float64)
    if not np.isfinite(x).all() or np.ptp(x) < 1e-6:
        return np.zeros(0, dtype=np.int64)
    if method == "xqrs":
        xq = wp.XQRS(sig=x, fs=FS)
        xq.detect(verbose=False)
        q = np.asarray(xq.qrs_inds, dtype=np.int64)
    elif method == "gqrs":
        q = np.asarray(wp.gqrs_detect(sig=x, fs=FS), dtype=np.int64)
    else:
        raise ValueError(method)
    if len(q) == 0 or refine <= 0:
        return np.unique(q)
    bp = sps.sosfiltfilt(_SOS_LGBM, x)
    bp = bp - np.median(bp)
    out = []
    for p in q:
        a, b = max(0, p - refine), min(len(bp), p + refine + 1)
        out.append(a + int(np.argmax(np.abs(bp[a:b]))))
    out = np.unique(np.asarray(out, dtype=np.int64))
    # merge peaks closer than 200 ms (refinement can collapse two detections)
    if len(out) > 1:
        keep = [out[0]]
        for p in out[1:]:
            if p - keep[-1] < int(0.2 * FS):
                if abs(bp[p]) > abs(bp[keep[-1]]):
                    keep[-1] = p
            else:
                keep.append(p)
        out = np.asarray(keep, dtype=np.int64)
    return out


# ─────────────────────────────────────────────────── LightGBM 40 features ──
_LGBM_MEDIANS = None


def lgbm_train_medians() -> np.ndarray:
    """Column medians of the DS1 training feature matrix (NaN-imputation values)."""
    global _LGBM_MEDIANS
    if _LGBM_MEDIANS is None:
        d = np.load(ROOT / "datasets" / "processed" / "mit_bih_features.npz", allow_pickle=True)
        mask = np.isin(d["record_ids"].astype(str), DS1)
        _LGBM_MEDIANS = np.nanmedian(d["X"][mask], axis=0).astype(np.float32)
    return _LGBM_MEDIANS


def lgbm_strip_features(sig360: np.ndarray, peaks: np.ndarray):
    """
    Re-implementation of source/features.py::process_record for one 10-s strip
    (identical maths, identical extract_beat_features call):
      - 0.5–40 Hz 4th-order Butterworth, zero-phase (preprocess.py)
      - 256-sample window (100 pre / 156 post); beats whose window leaves the strip dropped
      - amplitude scale = median |R amplitude| of the strip's classified beats
      - RR from ALL detected peaks in the strip; rr_pre/rr_post divided by strip mean RR
      - HRV window = up to 50 preceding RR intervals *inside the strip* (<= ~12 here)
      - NaNs (LF/HF/LF-HF/SampEn need >=10 RR and >=10 s) imputed with DS1 training medians
    Returns (X (n,40) float32, kept_peak_positions (n,)) .
    """
    pre, post = lgbm_feats.R_IDX, 256 - lgbm_feats.R_IDX
    filt = sps.sosfiltfilt(_SOS_LGBM, np.asarray(sig360, dtype=np.float64)).astype(np.float32)
    n_sig = len(filt)
    peaks = np.sort(np.asarray(peaks, dtype=np.int64))
    kept = np.array([i for i, p in enumerate(peaks) if p - pre >= 0 and p + post <= n_sig], dtype=np.int64)
    if len(kept) == 0:
        return np.zeros((0, 40), np.float32), np.zeros(0, np.int64)
    rr_all = np.diff(peaks).astype(float) / FS * 1000.0           # ms
    mean_rr = float(np.mean(rr_all)) if len(rr_all) > 0 else 800.0
    beats = np.stack([filt[peaks[i] - pre: peaks[i] + post] for i in kept])
    scale = float(np.median(np.abs(beats[:, pre]))) + 1e-6
    X = np.full((len(kept), 40), np.nan, dtype=np.float32)
    for j, i in enumerate(kept):
        beat = beats[j] / scale
        if float(beat[pre]) < 0:
            beat = -beat
        rr_pre = rr_all[i - 1] if i > 0 else np.nan
        rr_post = rr_all[i] if i < len(peaks) - 1 else np.nan
        rr_window = rr_all[max(0, i - lgbm_feats.HRV_WINDOW):i]
        win_mean = float(np.nanmean(rr_window)) if len(rr_window) > 0 else 800.0
        if np.isnan(rr_pre):
            rr_pre = win_mean
        if np.isnan(rr_post):
            rr_post = win_mean
        if len(rr_window) == 0:
            rr_window = np.array([win_mean])
        with np.errstate(all="ignore"):
            X[j] = lgbm_feats.extract_beat_features(
                beat, rr_pre / (mean_rr + 1e-6), rr_post / (mean_rr + 1e-6), rr_window)
    med = lgbm_train_medians()
    bad = ~np.isfinite(X)
    X[bad] = np.take(med, np.where(bad)[1])
    return X, peaks[kept]


# ────────────────────────────────────────────────── Farag 1D-CNN inputs ──
def cnn_strip_inputs(sig360: np.ndarray, peaks: np.ndarray, rr_mode: str = "causal"):
    """
    paper3 pipeline (configs/default.yaml + utils.load_config overrides):
      filter=none, resample 360->128 (resample_poly 16/45), 64-sample window edge-padded,
      mean normalisation, first derivative, centre crop NK=32  -> (n, 32, 1)
    RR features (pre/local80, pre/global400, post/local80, post/global400):
      rr_mode='causal'   : paper3 compute_rr_features unchanged — causal mean over whatever
                           history exists (exactly what the code does at a record start)
      rr_mode='stripmean': local and global means both replaced by the mean RR of the whole strip
    """
    peaks = np.sort(np.asarray(peaks, dtype=np.int64))
    if len(peaks) == 0:
        return np.zeros((0, 32, 1), np.float32), np.zeros((0, 4), np.float32)
    s128 = resample_signal(np.asarray(sig360, dtype=np.float64))
    r128 = map_r_peaks(peaks)
    segs = []
    for r in r128:
        seg, _ = segment_beat(s128, int(r), 32, 32)
        seg = normalise_segment(seg, "mean")
        segs.append(crop_to_nk(compute_derivative(seg), 32, 64))
    X_ecg = np.asarray(segs, dtype=np.float32)[..., None]
    if rr_mode == "causal":
        X_rr = compute_rr_features(r128, fs=128.0, local_window=80, global_window=400, use_post_rr=True)
    elif rr_mode == "stripmean":
        if len(r128) >= 2:
            d = np.diff(r128.astype(np.float64))
            pre = np.concatenate([[d[0]], d])
            post = np.concatenate([d, [d[-1]]])
            m = float(np.mean(d))
        else:
            pre = post = np.array([128.0]); m = 128.0
        X_rr = np.column_stack([pre / m, pre / m, post / m, post / m]).astype(np.float32)
    else:
        raise ValueError(rr_mode)
    return X_ecg, X_rr


# ──────────────────────────────────────────────────────────────── models ──
_LGBM = None
_CNNS = None


def load_lgbm():
    global _LGBM
    if _LGBM is None:
        import pickle
        with open(ROOT / "models" / "lgbm_ds1ds2.pkl", "rb") as f:
            _LGBM = pickle.load(f)["model"]
        try:
            _LGBM.set_params(n_jobs=4, device="cpu")
        except Exception:
            pass
    return _LGBM


def lgbm_predict_proba(X: np.ndarray) -> np.ndarray:
    """Return probabilities with columns in LGBM_CLASSES order."""
    m = load_lgbm()
    if len(X) == 0:
        return np.zeros((0, 5))
    P = m.predict_proba(X)
    cls = [str(c) for c in m.classes_]
    return P[:, [cls.index(c) for c in LGBM_CLASSES]]


def load_cnns():
    global _CNNS
    if _CNNS is None:
        import tensorflow as tf
        tf.config.threading.set_intra_op_parallelism_threads(4)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        _CNNS = [tf.keras.models.load_model(str(PAPER3 / "results" / "cache" / f"P1_fullDS1_seed{s}.keras"),
                                            compile=False) for s in range(10)]
    return _CNNS


def cnn_predict_proba(X_ecg: np.ndarray, X_rr: np.ndarray) -> np.ndarray:
    if len(X_ecg) == 0:
        return np.zeros((0, 3))
    ps = [m.predict([X_ecg, X_rr], batch_size=4096, verbose=0) for m in load_cnns()]
    return np.mean(ps, axis=0)


def cnn_threshold_predict(P: np.ndarray, tau_s: float = TAU_S) -> np.ndarray:
    y = np.argmax(P, axis=1).copy()
    y[P[:, 1] >= tau_s] = 1
    return y


# ─────────────────────────────────────────────────────────── aggregation ──
def beat_counts(pred_labels: np.ndarray, classes: list[str]) -> dict:
    n = len(pred_labels)
    out = {"n_beats": n}
    for c in classes:
        k = int(np.sum(pred_labels == c))
        out[f"cnt_{c}"] = k
        out[f"frac_{c}"] = k / n if n else 0.0
    return out


def apply_rule(cnt: np.ndarray, frac: np.ndarray, k: int, p: float) -> np.ndarray:
    """Record positive for a condition iff count >= k AND fraction >= p."""
    return (cnt >= k) & (frac >= p)


def binary_metrics(y: np.ndarray, yhat: np.ndarray) -> dict:
    y = np.asarray(y, bool); yhat = np.asarray(yhat, bool)
    tp = int(np.sum(y & yhat)); fp = int(np.sum(~y & yhat))
    fn = int(np.sum(y & ~yhat)); tn = int(np.sum(~y & ~yhat))
    se = tp / (tp + fn) if tp + fn else float("nan")
    sp = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")
    return dict(n_pos=tp + fn, n_neg=tn + fp, TP=tp, FP=fp, FN=fn, TN=tn, Se=se, Sp=sp, PPV=ppv, F1=f1)
