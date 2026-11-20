# %% [markdown]
# ## rr.py — Causal RR-interval features and XGBoost sanity baseline
#
# Reference: Farag (2023), Section 3.
#
# The paper feeds the model four dynamically-normalised RR features instead of
# raw RR values, because absolute RR (heart rate) degrades accuracy when patient
# heart rates differ — the network would overfit to record-specific rate.
#
# Key design decision — causal only: local and global means are computed from
# past beats (including the current one) so they are available in real time on
# the ESP32 without any look-ahead buffer.
#
# post_RR adds one beat of latency because the device needs to wait for the
# next beat to compute it.  The paper explicitly accepts this trade-off (Sec 3).
#
# RR features are computed from ALL annotated beats (including those dropped
# for 3-class classification), so the interval between consecutive beats is
# never inflated by a "missing" beat.

# %%
import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


# %%
def compute_rr_intervals(r_samples: np.ndarray) -> np.ndarray:
    """
    Convert R-peak sample indices to inter-beat intervals (in samples).
    r_samples[i] is the sample position of the i-th beat in the 128 Hz signal.

    Returns pre_rr of shape (n,):
      pre_rr[0] = r_samples[1] - r_samples[0]  (best proxy for a "typical" interval
                   at the record start, since there is no prior beat)
      pre_rr[i] = r_samples[i] - r_samples[i-1]  for i >= 1
    """
    n = len(r_samples)
    pre_rr = np.empty(n, dtype=np.float64)
    if n < 2:
        pre_rr[:] = 128.0  # fallback: 1 s at 128 Hz
        return pre_rr
    diffs = np.diff(r_samples.astype(np.float64))  # shape (n-1,)
    pre_rr[1:] = diffs
    pre_rr[0]  = diffs[0]  # first beat has no history; use the next interval
    return pre_rr


def _causal_mean(x: np.ndarray, window: int) -> np.ndarray:
    """
    Causal moving average: out[i] = mean(x[max(0,i-window+1):i+1]).
    'At the start of a record, average whatever history exists' (paper Sec 3).
    Uses cumulative sum for O(n) computation.
    """
    n = len(x)
    out = np.empty(n, dtype=np.float64)
    cumsum = np.cumsum(x)
    for i in range(n):
        start = max(0, i - window + 1)
        count = i - start + 1
        total = cumsum[i] - (cumsum[start - 1] if start > 0 else 0.0)
        out[i] = total / count
    return out


def compute_rr_features(r_samples_all: np.ndarray, fs: float,
                        local_window: int = 80, global_window: int = 400,
                        use_post_rr: bool = True) -> np.ndarray:
    """
    Compute the 4 normalised RR features for every beat in the record.

    Parameters
    ----------
    r_samples_all : R-peak sample indices at `fs` Hz, ALL annotated beats
                    (including those that will be dropped from classification).
                    This prevents RR intervals from jumping over filtered beats.
    fs            : sample rate (128 Hz)
    local_window  : moving-average window for local heart rate (80 beats, Sec 3)
    global_window : moving-average window for global heart rate (400 beats, Sec 3)
    use_post_rr   : whether to include post_RR features (adds 1-beat latency)

    Returns
    -------
    features : (n_beats, 4) float32
      col 0: pre_RR  / local_mean
      col 1: pre_RR  / global_mean
      col 2: post_RR / local_mean
      col 3: post_RR / global_mean
    If use_post_rr is False, returns (n_beats, 2) using only pre_RR.
    """
    n = len(r_samples_all)
    pre_rr = compute_rr_intervals(r_samples_all)   # samples

    # post_RR[i] = time to the NEXT beat (one-beat latency is accepted, Sec 3)
    if use_post_rr:
        post_rr = np.empty(n, dtype=np.float64)
        if n >= 2:
            post_rr[:-1] = np.diff(r_samples_all.astype(np.float64))
            post_rr[-1]  = post_rr[-2]  # last beat: no next beat, repeat previous
        else:
            post_rr[:] = pre_rr

    # Causal moving averages of pre_RR (local heart rate context)
    local_mean  = _causal_mean(pre_rr, local_window)   # shape (n,)
    global_mean = _causal_mean(pre_rr, global_window)  # shape (n,)

    # Avoid division by zero (should not happen with real ECG)
    local_mean  = np.where(local_mean  < 1e-6, 1.0, local_mean)
    global_mean = np.where(global_mean < 1e-6, 1.0, global_mean)

    if use_post_rr:
        features = np.column_stack([
            pre_rr  / local_mean,
            pre_rr  / global_mean,
            post_rr / local_mean,
            post_rr / global_mean,
        ])
    else:
        features = np.column_stack([
            pre_rr / local_mean,
            pre_rr / global_mean,
        ])

    return features.astype(np.float32)


# %%
def xgboost_sanity_check(
    rr_train: np.ndarray, y_train: np.ndarray,
    rr_val:   np.ndarray, y_val:   np.ndarray,
    rr_test:  np.ndarray, y_test:  np.ndarray,
    seed: int = 0,
) -> dict:
    """
    Train XGBoost on 4 RR features alone (3-class N/S/V).
    Paper Sec 5 reports: ~93 % accuracy, recall N/S/V = 97/60/58 %.
    If our numbers are far off, the RR feature computation has a bug.

    Returns a dict with accuracy, per-class recall, and confusion matrix.
    """
    try:
        import xgboost as xgb
    except ImportError:
        log.warning("xgboost not installed; skipping RR sanity check")
        return {}

    from sklearn.metrics import classification_report, confusion_matrix

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=seed,
        verbosity=0,
    )
    # Use DS1-train for fitting, DS1-val for early stopping, DS2 for reporting
    model.fit(
        rr_train, y_train,
        eval_set=[(rr_val, y_val)],
        verbose=False,
    )

    pred = model.predict(rr_test)
    acc  = float((pred == y_test).mean())
    cm   = confusion_matrix(y_test, pred, labels=[0, 1, 2])

    report = classification_report(y_test, pred, labels=[0, 1, 2],
                                   target_names=["N", "S", "V"],
                                   output_dict=True, zero_division=0)
    recalls = {c: report[c]["recall"] for c in ["N", "S", "V"]}

    log.info("RR-only XGBoost: acc=%.2f%%  N=%.1f%%  S=%.1f%%  V=%.1f%%",
             100 * acc, 100 * recalls["N"], 100 * recalls["S"], 100 * recalls["V"])
    log.info("Paper target:    acc=~93%%    N=~97%%    S=~60%%    V=~58%%")

    if acc < 0.85:
        log.warning("RR XGBoost accuracy %.1f%% is well below the paper's ~93%%; "
                    "check RR feature computation.", 100 * acc)

    return {
        "accuracy":     acc,
        "recall_N":     recalls["N"],
        "recall_S":     recalls["S"],
        "recall_V":     recalls["V"],
        "confusion":    cm.tolist(),
        "report":       report,
    }
