"""Signal-to-spike encoding (delta modulation, thesis Sec. 2.2.2 / 4.3)."""

import numpy as np


def delta_modulation(x: np.ndarray, threshold: float) -> np.ndarray:
    """Return (T, 2) float32 spike trains: column 0 = UP, column 1 = DOWN.

    A spike is emitted at t when |x(t) - x(t_last_spike)| > threshold; the
    reference value starts at x(0) and is reset to x(t) at every spike.
    """
    x = np.asarray(x, dtype=np.float64)
    spikes = np.zeros((len(x), 2), dtype=np.float32)
    ref = x[0]
    for t in range(1, len(x)):
        diff = x[t] - ref
        if diff > threshold:
            spikes[t, 0] = 1.0
            ref = x[t]
        elif diff < -threshold:
            spikes[t, 1] = 1.0
            ref = x[t]
    return spikes


def build_input(x_norm: np.ndarray, mode: str, threshold: float) -> np.ndarray:
    """ESN input u(t) as a (T, K) float32 array for `mode` in {'spikes', 'raw'}."""
    if mode == "spikes":
        return delta_modulation(x_norm, threshold)
    if mode == "raw":
        return np.asarray(x_norm, dtype=np.float32)[:, None]
    raise ValueError(f"unknown input mode {mode!r}")
