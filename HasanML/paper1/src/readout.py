"""Ridge readout and CV selection of (input scaling, tau, lambda, threshold)."""

import numpy as np


def ridge_fit(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    """W_out = Y X^T (X X^T + lambda I)^-1 (thesis form; X is features x time).

    Include a row of ones in X for the bias; it is regularised like every other weight.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.atleast_2d(np.asarray(Y, dtype=np.float64))
    A = X @ X.T + lam * np.eye(X.shape[0])
    return np.linalg.solve(A, X @ Y.T).T


def ridge_path(xp, G, c, lambdas):
    """Solve (G + lambda I) w = c for every lambda (one batched LU solve per lambda).

    G: (..., n, n) Gram matrix X X^T, c: (..., n) cross term X y.
    Returns (..., n, n_lambda). For a handful of lambdas this is ~2x cheaper
    than one eigendecomposition shared by all of them.
    """
    n = G.shape[-1]
    eye = xp.eye(n, dtype=G.dtype)
    rhs = c[..., None]
    return xp.concatenate([xp.linalg.solve(G + float(lam) * eye, rhs) for lam in lambdas], axis=-1)


def class_weights(kind: str, n_all: np.ndarray, n_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-timestep sample weights (w_negative, w_positive) for weighted ridge."""
    n_all = np.asarray(n_all, dtype=np.float64)
    n_pos = np.asarray(n_pos, dtype=np.float64)
    ones = np.ones_like(n_all)
    if kind == "none":
        return ones, ones
    if kind == "balanced":
        n_neg = n_all - n_pos
        ok = (n_pos > 0) & (n_neg > 0)
        w_neg = np.where(ok, n_all / (2 * np.maximum(n_neg, 1)), 1.0)
        w_pos = np.where(ok, n_all / (2 * np.maximum(n_pos, 1)), 1.0)
        return w_neg, w_pos
    raise ValueError(f"unknown class weighting {kind!r}")


def threshold_grid(start: float, stop: float, step: float) -> np.ndarray:
    return np.round(np.arange(start, stop + step / 2, step), 4)


def select_operating_point(val_scores: np.ndarray, y_val: np.ndarray, thresholds: np.ndarray,
                           lambda_index: int | None = None) -> dict:
    """Pick (scaling, tau, lambda, threshold) maximising pooled cross-validated F1.

    val_scores: (n_scaling, n_tau, n_lambda, n_beats) out-of-fold beat scores.
    Ties are broken by accuracy, then by the threshold closest to 0.5, then by
    grid order. With `lambda_index`, lambda is fixed to that grid entry.
    """
    if lambda_index is not None:
        val_scores = val_scores[:, :, lambda_index:lambda_index + 1]
    y = np.asarray(y_val).astype(bool)
    pred = val_scores[..., None, :] > thresholds[:, None]
    tp = (pred & y).sum(-1)
    fp = (pred & ~y).sum(-1)
    fn = y.sum() - tp
    tn = (~y).sum() - fp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    acc = (tp + tn) / max(len(y), 1)
    closeness = np.broadcast_to(np.abs(thresholds - 0.5), f1.shape)
    best = np.lexsort((closeness.ravel(), -acc.ravel(), -f1.ravel()))[0]
    s, tau, lam, th = np.unravel_index(best, f1.shape)
    if lambda_index is not None:
        lam = lambda_index
    return {
        "scaling_index": int(s),
        "tau_index": int(tau),
        "lambda_index": int(lam),
        "threshold": float(thresholds[th]),
        "cv_f1": float(f1.ravel()[best]),
        "cv_accuracy": float(acc.ravel()[best]),
    }
