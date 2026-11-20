"""Per-strip featurisation shared by MIT-BIH strips and external records (worker-safe, no TF)."""
import numpy as np

from common import detect_rpeaks, lgbm_strip_features, cnn_strip_inputs

DETECTOR = ("xqrs", 18)   # fixed after validate_detector.py (chosen on DS1)


def featurize(sig360, peaks=None):
    """Returns dict with detected peaks + LGBM features + CNN inputs (both RR fallbacks)."""
    if peaks is None:
        peaks = detect_rpeaks(sig360, *DETECTOR)
    peaks = np.sort(np.asarray(peaks, dtype=np.int64))
    Xl, lpk = lgbm_strip_features(sig360, peaks)
    Xe, Xr_c = cnn_strip_inputs(sig360, peaks, "causal")
    _, Xr_s = cnn_strip_inputs(sig360, peaks, "stripmean")
    return dict(peaks=peaks, X_lgbm=Xl, lgbm_peaks=lpk, X_ecg=Xe, X_rr_causal=Xr_c, X_rr_strip=Xr_s)
