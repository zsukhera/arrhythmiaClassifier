"""Fiducial-point detection inside fixed sub-windows of each 256-sample beat (paper Fig. 3).

Everything is vectorised over a matrix of beats (one beat per row). Rules:

  R      local max within +/- `r_refine_samples` of the annotation (index `r_index`)
  P, T   P = max of x in the P window; T = max of |x - iso| in the T window (T may be inverted)
  Q, S   min of x in the Q window (before R) / S window (after R)
  P1/P2, T1/T2   walking out from the peak, the first sample with
                 |x - iso| < wave_amplitude_frac * |x[peak] - iso|
  Q1, S2         from the steepest sample of the Q downstroke (within `qrs_edge_ms` before Q)
                 walking left / of the S upstroke (within `qrs_edge_ms` after S) walking
                 right, the first sample with |dx| < qrs_slope_frac * max |dx| over the QRS
                 (Q window + S window). Starting at Q or S themselves would stop at once,
                 since the slope is ~0 at a turning point.

`iso` is the isoelectric level of the beat, the median of its window. Every search is
bounded by its neighbours (P2 <= Q1 <= Q, S <= S2 <= T1) so the points stay ordered.
"""

import logging

import numpy as np

log = logging.getLogger(__name__)

POINTS = ["P1", "P", "P2", "Q1", "Q", "R", "S", "S2", "T1", "T", "T2"]


def ms_to_samples(ms: float, fs: float) -> int:
    return int(round(ms * fs / 1000.0))


def _window_mask(n_cols: int, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Boolean (n, n_cols) mask for the inclusive column range [lo, hi] per row."""
    idx = np.arange(n_cols)[None, :]
    return (idx >= lo[:, None]) & (idx <= hi[:, None])


def masked_argmax(v: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Column of the maximum of each row of `v` within [lo, hi] (inclusive)."""
    hi = np.maximum(hi, lo)
    w = np.where(_window_mask(v.shape[1], lo, hi), v, -np.inf)
    return np.argmax(w, axis=1)


def walk_left(cond: np.ndarray, start: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Largest k in [limit, start) with cond[k]; `limit` if there is none."""
    n, L = cond.shape
    limit = np.minimum(limit, start)
    m = cond & _window_mask(L, limit, start - 1)
    found = m.any(axis=1)
    last = L - 1 - np.argmax(m[:, ::-1], axis=1)
    return np.where(found, last, limit)


def walk_right(cond: np.ndarray, start: np.ndarray, limit: np.ndarray) -> np.ndarray:
    """Smallest k in (start, limit] with cond[k]; `limit` if there is none."""
    n, L = cond.shape
    limit = np.maximum(limit, start)
    m = cond & _window_mask(L, start + 1, limit)
    found = m.any(axis=1)
    return np.where(found, np.argmax(m, axis=1), limit)


def detect(beats: np.ndarray, fs: float, r_index: int, cfg: dict) -> dict[str, np.ndarray]:
    """Return {point name: column index per beat} plus 'iso' (isoelectric level, mV)."""
    n, L = beats.shape
    rows = np.arange(n)
    ms = lambda v: ms_to_samples(v, fs)
    k = int(cfg["r_refine_samples"])

    r = r_index - k + np.argmax(beats[:, r_index - k:r_index + k + 1], axis=1)
    iso = np.median(beats, axis=1) if cfg.get("iso_level", "median") == "median" else beats[:, 0]
    dev = np.abs(beats - iso[:, None])

    def bounds(window_ms):
        lo = np.clip(r + ms(window_ms[0]), 0, L - 1)
        hi = np.clip(r + ms(window_ms[1]), 0, L - 1)
        return lo, hi

    p_lo, p_hi = bounds(cfg["p_window_ms"])
    q_lo, _ = bounds(cfg["q_window_ms"])
    _, s_hi = bounds(cfg["s_window_ms"])
    t_lo, t_hi = bounds(cfg["t_window_ms"])
    clipped = int(np.count_nonzero(r + ms(cfg["t_window_ms"][1]) > L - 1))
    if clipped:
        log.debug("T search window clipped to the beat-window end for %d/%d beats", clipped, n)

    p = masked_argmax(beats, p_lo, p_hi)
    q = masked_argmax(-beats, q_lo, np.maximum(r - 1, q_lo))
    s = masked_argmax(-beats, np.minimum(r + 1, s_hi), s_hi)
    t = masked_argmax(dev, t_lo, t_hi)

    slope = np.abs(np.diff(beats, axis=1, prepend=beats[:, :1]))
    slope[:, 0] = slope[:, 1]
    qrs_max = np.where(_window_mask(L, q_lo, s_hi), slope, 0.0).max(axis=1)
    flat = slope < cfg["qrs_slope_frac"] * qrs_max[:, None]
    edge = ms(cfg["qrs_edge_ms"])
    q_steep = masked_argmax(slope, np.maximum(q - edge, p + 1), q)
    s_steep = masked_argmax(slope, s + 1, np.minimum(s + edge, t - 1)) - 1   # slope[k] is k-1 -> k
    q1 = np.minimum(walk_left(flat, q_steep, p + 1), q)
    s2 = np.maximum(walk_right(flat, np.maximum(s_steep, s), t - 1), s)

    frac = cfg["wave_amplitude_frac"]
    p_small = dev < frac * dev[rows, p][:, None]
    t_small = dev < frac * dev[rows, t][:, None]
    p1 = walk_left(p_small, p, np.zeros(n, dtype=np.int64))
    p2 = walk_right(p_small, p, q1)
    t1 = walk_left(t_small, t, s2)
    t2 = walk_right(t_small, t, np.full(n, L - 1))

    pts = dict(P1=p1, P=p, P2=p2, Q1=q1, Q=q, R=r, S=s, S2=s2, T1=t1, T=t, T2=t2)
    pts = {name: pts[name].astype(np.int64) for name in POINTS}
    pts["iso"] = iso
    pts["t_clipped"] = clipped
    return pts
