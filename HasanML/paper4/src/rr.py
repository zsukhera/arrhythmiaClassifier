# %% [markdown]
# ## rr.py — Causal RR-interval ratio features and LightGBM sanity check
#
# Absolute RR values are patient-specific (heart rate varies widely across
# patients) and hurt inter-patient generalisation.  Ratios to local and global
# means remove the patient-level heart rate while preserving ectopic timing
# patterns, which are the defining cues for S and V beats:
#   - Premature beats (S, V) have short pre_RR and long post_RR.
#   - Local mean tracks short-term rate variability (anxiety, exercise).
#   - Global mean tracks the patient's baseline rate.
#   - pre/post directly captures the compensatory pause signature.
#   - (post - pre)/local captures the asymmetry (ectopic = very asymmetric).
#
# 6 features (more than paper3's 4; adds pre/post and (post-pre)/local):
#   pre/local, post/local, pre/global, post/global, pre/post, (post-pre)/local
#
# All features are computed from ALL annotated beats (including those dropped
# for the classification task) so intervals never jump over a filtered beat.
#
# References:
#   Farag (2023), Sec 3: 4-feature causal RR design.
#   Mousavi & Afghah (2019), ICASSP: context of neighbouring beats improves S.

# %%
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)

EPSILON = 1e-6   # prevent division by zero in ratio computation


# %%
def _causal_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Causal moving average: out[i] = mean(x[max(0, i-window+1) : i+1]).
    'At the start of a record, use whatever history exists.'
    O(n) via prefix sums.
    """
    n      = len(x)
    cs     = np.cumsum(x)
    starts = np.maximum(0, np.arange(n) - window + 1)
    counts = np.arange(1, n + 1) - starts
    sums   = cs - np.where(starts > 0, cs[starts - 1], 0.0)
    return sums / counts


def compute_rr_intervals(r_samples: np.ndarray) -> np.ndarray:
    """
    Convert R-peak sample positions -> inter-beat intervals (samples).
    pre_rr[0] uses the second interval as a proxy (no prior beat at record start).
    """
    n = len(r_samples)
    pre_rr = np.empty(n, dtype=np.float64)
    if n < 2:
        pre_rr[:] = float(360)   # fallback: 1 s at 360 Hz
        return pre_rr
    diffs   = np.diff(r_samples.astype(np.float64))
    pre_rr[1:] = diffs
    pre_rr[0]  = diffs[0]
    return pre_rr


# %%
def compute_rr_features(r_samples_all: np.ndarray, fs: float,
                        local_window: int = 10, global_window: int = 400) -> np.ndarray:
    """
    Compute 6 causal RR ratio features for every beat.

    Parameters
    ----------
    r_samples_all : R-peak sample positions of ALL beat annotations (360 Hz).
    fs            : sample rate (360 Hz for paper4; kept as parameter for clarity).
    local_window  : beats in local moving mean (last 10).
    global_window : beats in global moving mean (last 400).

    Returns
    -------
    features : (n_beats, 6) float32
      col 0: pre_RR  / local_mean
      col 1: post_RR / local_mean
      col 2: pre_RR  / global_mean
      col 3: post_RR / global_mean
      col 4: pre_RR  / post_RR       (< 1 for premature beats)
      col 5: (post_RR - pre_RR) / local_mean  (asymmetry; large for ectopics)
    """
    n      = len(r_samples_all)
    pre_rr = compute_rr_intervals(r_samples_all)

    # post_RR[i] = pre_RR[i+1] (one beat of look-ahead latency; acceptable for cloud)
    post_rr = np.empty(n, dtype=np.float64)
    post_rr[:-1] = pre_rr[1:]
    post_rr[-1]  = pre_rr[-1]  # last beat: use its own pre_RR as proxy

    local_mean  = _causal_mean(pre_rr, local_window)
    global_mean = _causal_mean(pre_rr, global_window)

    # Safe ratios: clamp denominators to epsilon
    lm = np.maximum(local_mean,  EPSILON)
    gm = np.maximum(global_mean, EPSILON)
    pr = np.maximum(pre_rr,      EPSILON)

    feats = np.stack([
        pre_rr  / lm,
        post_rr / lm,
        pre_rr  / gm,
        post_rr / gm,
        pre_rr  / (post_rr + EPSILON),
        (post_rr - pre_rr) / lm,
    ], axis=1)

    return feats.astype(np.float32)


# %%
def lightgbm_sanity_check(X_tr: np.ndarray, y_tr: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           n_classes: int = 5) -> dict:
    """
    Sanity check: LightGBM on RR features alone.
    Target: accuracy ~90-93%, S recall ~60% — if far off, the RR features are wrong.

    Uses DS1-train for fitting and DS1-val for evaluation, following the
    patient-wise split that the main model uses.
    """
    try:
        import lightgbm as lgb
    except ImportError:
        log.warning("lightgbm not installed; skipping RR sanity check")
        return {}

    from sklearn.metrics import classification_report, accuracy_score

    # Keep only beats with valid labels
    mask_tr  = y_tr  >= 0
    mask_val = y_val >= 0

    clf = lgb.LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        num_leaves=31, n_jobs=-1, verbose=-1,
        class_weight="balanced",
    )
    clf.fit(X_tr[mask_tr], y_tr[mask_tr])

    y_pred = clf.predict(X_val[mask_val])
    y_true = y_val[mask_val]

    acc    = accuracy_score(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    se_s   = report.get("1", {}).get("recall", float("nan"))   # AAMI S
    se_v   = report.get("2", {}).get("recall", float("nan"))   # AAMI V

    log.info("RR-only LightGBM: acc=%.1f%%  Se_S=%.1f%%  Se_V=%.1f%%",
             100*acc, 100*se_s, 100*se_v)
    log.info("Expected targets: acc ~90-93 %%, S recall ~60 %%")

    if acc < 0.85:
        log.warning("RR sanity check: acc=%.1f%% < 85%% — check RR feature computation", 100*acc)

    return {"accuracy": float(acc), "se_S": float(se_s), "se_V": float(se_v),
            "report": report}
