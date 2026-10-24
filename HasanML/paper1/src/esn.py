"""Leaky-integrator ESN (Jaeger et al. 2007) with streaming ridge-readout training.

    x(n+1) = (1 - a) x(n) + a tanh(W_in u(n+1) + W x(n) + b)
    z_tau(n) = d z_tau(n-1) + (1 - d) x(n),  d = exp(-1 / (tau fs))   (state smoothing)

A 768-unit state sampled at 360 Hz is ~1 GB per 15-minute run, so states are
never stored. One job simulates a batch of reservoirs that share the same
recurrent matrix W (same seed) in two passes over identical, deterministic
dynamics:

  pass 1  [0, t_train): accumulate per-CV-fold Gram matrices Z^T Z (all and
          PVC-target rows) for every tau and the mean state of every
          training beat window, then solve the ridge readout of each fold
          model and the full-training model for every lambda.
  pass 2  apply every readout and aggregate it over each beat window. The
          readout is linear, so a beat's mean output is the readout of its
          mean state: for "mean" aggregation pass 2 resumes from a state
          snapshot just before the first test beat. "max" replays from 0.

Pass 1 only reads inputs before t_train, so nothing from the test window
reaches the readout or the CV scores.
"""

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

from .backend import get_xp, to_host
from .readout import class_weights, ridge_path

F32 = np.float32
F64 = np.float64


def scale_spectral_radius(W: np.ndarray, rho: float) -> np.ndarray:
    radius = float(np.max(np.abs(np.linalg.eigvals(W))))
    return W * (rho / radius)


@lru_cache(maxsize=8)
def _recurrent_matrix(seed: int, n: int, connectivity: float, rho: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < connectivity
    W = np.where(mask, rng.standard_normal((n, n)), 0.0)
    return scale_spectral_radius(W, rho)


def make_reservoir(seed: int, n: int, n_inputs: int, connectivity: float, spectral_radius: float,
                   bias_scaling: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (W, W_in_unit, b). W_in_unit ~ U[-1, 1]; multiply by the input scaling s."""
    W = _recurrent_matrix(seed, n, connectivity, spectral_radius)
    W_in = np.random.default_rng([seed, 1]).uniform(-1.0, 1.0, (n, n_inputs))
    b = np.random.default_rng([seed, 2]).uniform(-bias_scaling, bias_scaling, n)
    return W, W_in, b


@dataclass
class JobSpec:
    """Settings shared by every reservoir in one job."""
    seed: int
    n_reservoir: int
    connectivity: float
    spectral_radius: float
    bias_scaling: float
    decays: tuple            # per-sample smoothing decay for each tau
    lambdas: tuple
    weightings: tuple        # e.g. ("none", "balanced")
    t_train: int
    washout: int
    n_folds: int
    window_len: int          # beat window length in samples
    chunk: int
    aggregations: tuple = ("mean",)  # per-beat readout aggregation(s): "mean", "max"
    device: str = "cpu"
    return_weights: bool = False


@dataclass
class Member:
    """One reservoir run: a record, an input encoding, a leak rate and an input scaling."""
    key: tuple
    u: np.ndarray            # (T, K) float32 input
    input_scaling: float
    leak: float
    target: np.ndarray       # (t_train,) bool: readout target y(t) on the training window
    beat_start: np.ndarray   # (n_beats,) first sample of each beat window, ascending
    beat_fold: np.ndarray    # (n_beats,) CV fold of training beats, -1 for test beats
    trace_start: int = -1    # optional segment whose readout trace is returned
    trace_len: int = 0
    meta: dict = field(default_factory=dict)


class _Reservoir:
    """Batched reservoir state that can be advanced chunk by chunk and snapshotted."""

    def __init__(self, xp, W, W_in, bias, U, leak, decay):
        self.xp, self.W, self.W_in, self.bias, self.U = xp, W, W_in, bias, U
        self.leak, self.keep = leak, 1 - leak
        self.decay = decay
        self.d = decay[:, None, None]
        self.g = 1 - self.d
        B, N = U.shape[0], bias.shape[0]
        self.x = xp.zeros((N, B), F32)
        self.z = xp.zeros((decay.shape[0], N, B), F32)
        self.t = 0

    def snapshot(self):
        return self.t, self.x.copy(), self.z.copy()

    def restore(self, snap):
        self.t, x, z = snap
        self.x, self.z = x.copy(), z.copy()

    def run(self, t_stop, chunk, snapshot_at=None, snapshots=None):
        """Yield (t0, Z), Z of shape (n_tau, B, C, N + 1): smoothed states for [t0, t0 + C),
        with the readout bias column of ones appended.

        Chunks start at multiples of `chunk`; if `snapshot_at` is one of those
        boundaries, the state at that time is stored in `snapshots`.
        """
        xp = self.xp
        n_tau, N, B = self.z.shape
        while self.t < t_stop:
            t0 = self.t
            if t0 == snapshot_at:
                snapshots.append(self.snapshot())
            t1 = min((t0 // chunk + 1) * chunk, t_stop)
            drive = xp.matmul(self.W_in, xp.swapaxes(self.U[:, t0:t1], 1, 2))   # (B, N, C)
            drive = xp.ascontiguousarray(drive.transpose(2, 1, 0)) + self.bias  # (C, N, B)
            out = xp.empty((n_tau, B, t1 - t0, N + 1), F32)
            out[..., N] = 1.0
            x, z = self.x.copy(), self.z.copy()
            gx = xp.empty_like(z)
            for i in range(t1 - t0):
                pre = self.W @ x
                pre += drive[i]
                xp.tanh(pre, out=pre)
                pre *= self.leak
                x *= self.keep
                x += pre
                xp.multiply(self.g, x, out=gx)
                z *= self.d
                z += gx
                out[:, :, i, :N] = xp.swapaxes(z, 1, 2)
            self.x, self.z = x, z
            self.t = t1
            yield t0, out


def _beats_in(starts: np.ndarray, L: int, t0: int, t1: int):
    """Yield (k, lo, hi): beats whose window [s, s+L) overlaps [t0, t1), as chunk offsets."""
    k0 = np.searchsorted(starts, t0 - L, side="right")
    k1 = np.searchsorted(starts, t1, side="left")
    for k in range(k0, k1):
        yield k, max(int(starts[k]), t0) - t0, min(int(starts[k]) + L, t1) - t0


def _accumulate_grams(xp, spec: JobSpec, members, states):
    """Per-fold Gram matrices and the summed state over every training-beat window."""
    B = len(members)
    n_tau = len(spec.decays)
    n1 = spec.n_reservoir + 1
    nf = spec.n_folds
    fold_len = spec.t_train // nf
    G = xp.zeros((nf, n_tau, B, n1, n1), F64)
    Gp = xp.zeros_like(G)
    cp = xp.zeros((nf, n_tau, B, n1), F64)
    n_all = np.zeros((nf, B))
    n_pos = np.zeros((nf, B))
    targets = np.stack([m.target for m in members])
    train_starts = [m.beat_start[m.beat_fold >= 0] for m in members]
    beat_sums = [xp.zeros((len(s), n_tau, n1), F32) for s in train_starts]
    for t0, Zaug in states:
        t1 = t0 + Zaug.shape[2]
        for f in range(nf):
            lo = max(t0, spec.washout, f * fold_len)
            hi = min(t1, spec.t_train if f == nf - 1 else (f + 1) * fold_len)
            if lo >= hi:
                continue
            rows = Zaug[:, :, lo - t0:hi - t0]                             # (n_tau, B, R, n1)
            G[f] += xp.matmul(xp.swapaxes(rows, -1, -2), rows)
            n_all[f] += hi - lo
            for b in range(B):
                idx = np.flatnonzero(targets[b, lo:hi])
                if idx.size == 0:
                    continue
                pos = rows[:, b, xp.asarray(idx)]                         # (n_tau, P, n1)
                Gp[f, :, b] += xp.matmul(xp.swapaxes(pos, -1, -2), pos)
                cp[f, :, b] += pos.sum(axis=1)
                n_pos[f, b] += idx.size
        for b in range(B):
            for k, lo, hi in _beats_in(train_starts[b], spec.window_len, t0, t1):
                beat_sums[b][k] += Zaug[:, b, lo:hi].sum(axis=1)
    return (G, Gp, cp, n_all, n_pos), beat_sums


def _solve_readouts(xp, spec: JobSpec, G, Gp, cp, n_all, n_pos):
    """Readout weights (n_tau, B, n1, n_models, n_weightings, n_lambdas).

    Model m < n_folds is trained on every fold except m; model n_folds on all folds.
    """
    nf = spec.n_folds
    lams = xp.asarray(spec.lambdas, F64)
    models = []
    for m in range(nf + 1):
        folds = [f for f in range(nf) if f != m]
        Ga, Gpa, ca = G[folds].sum(0), Gp[folds].sum(0), cp[folds].sum(0)
        per_weighting = []
        for kind in spec.weightings:
            w_neg, w_pos = class_weights(kind, n_all[folds].sum(0), n_pos[folds].sum(0))
            w_neg = xp.asarray(w_neg)[None, :, None, None]
            w_pos = xp.asarray(w_pos)[None, :, None, None]
            Gw = w_neg * (Ga - Gpa) + w_pos * Gpa
            cw = w_pos[..., 0] * ca
            per_weighting.append(ridge_path(xp, Gw, cw, lams))
        models.append(xp.stack(per_weighting, axis=-2))
    return xp.stack(models, axis=-3)


def _readout_pass(xp, spec: JobSpec, members, Wm, states):
    """Sum / max of every readout over each beat window and the trace segment."""
    n_tau, B, _, M = Wm.shape
    L = spec.window_len
    want_max = "max" in spec.aggregations
    sums = [xp.zeros((len(m.beat_start), n_tau, M), F32) for m in members]
    maxs = [xp.full((len(m.beat_start), n_tau, M), -np.inf, F32) if want_max else None
            for m in members]
    traces = [xp.zeros((n_tau, m.trace_len, M), F32) if m.trace_len else None for m in members]
    for t0, Z in states:
        t1 = t0 + Z.shape[2]
        Y = xp.matmul(Z, Wm)                                 # (n_tau, B, C, M)
        for b, m in enumerate(members):
            for k, lo, hi in _beats_in(m.beat_start, L, t0, t1):
                seg = Y[:, b, lo:hi]
                sums[b][k] += seg.sum(axis=1)
                if want_max:
                    maxs[b][k] = xp.maximum(maxs[b][k], seg.max(axis=1))
            if m.trace_len:
                lo = max(m.trace_start, t0)
                hi = min(m.trace_start + m.trace_len, t1)
                if lo < hi:
                    traces[b][:, lo - m.trace_start:hi - m.trace_start] = Y[:, b, lo - t0:hi - t0]
    return sums, maxs, traces


def run_job(spec: JobSpec, members: list[Member]) -> list[dict]:
    """Simulate a batch of reservoirs sharing one seed; return per-member beat scores.

    Every member must have the same input dimension and length. Returned per
    member, for each aggregation in spec.aggregations: val[agg] of shape
    (n_tau, n_w, n_lam, n_train_beats) with out-of-fold scores, test[agg] of
    shape (n_tau, n_w, n_lam, n_test_beats), and optionally a trace.
    """
    xp = get_xp(spec.device)
    K = members[0].u.shape[1]
    W, W_in_unit, b = make_reservoir(spec.seed, spec.n_reservoir, K, spec.connectivity,
                                     spec.spectral_radius, spec.bias_scaling)
    if spec.device == "cpu":
        W_op = sp.csr_array(W.astype(F32))
        W_op.indices = W_op.indices.astype(np.int32)
        W_op.indptr = W_op.indptr.astype(np.int32)
    else:
        W_op = xp.asarray(W, F32)
    res = _Reservoir(
        xp, W_op,
        xp.asarray(np.stack([W_in_unit * m.input_scaling for m in members]), F32),
        xp.asarray(b, F32)[:, None],
        xp.asarray(np.stack([m.u for m in members]), F32),
        xp.asarray([m.leak for m in members], F32),
        xp.asarray(spec.decays, F32),
    )
    T = members[0].u.shape[0]
    L = spec.window_len

    # Pass 2 for "mean" only needs states from just before the first test beat
    # (or trace) onwards, so snapshot the reservoir at that chunk boundary.
    first_needed = min(min([int(m.beat_start[m.beat_fold < 0].min(initial=T)) for m in members]
                           + [m.trace_start for m in members if m.trace_len]), spec.t_train)
    resume_at = (first_needed // spec.chunk) * spec.chunk
    snapshots = []
    grams, beat_sums = _accumulate_grams(xp, spec, members,
                                         res.run(spec.t_train, spec.chunk, resume_at, snapshots))
    if not snapshots and res.t == resume_at:
        snapshots.append(res.snapshot())
    W_out = _solve_readouts(xp, spec, *grams)
    del grams
    n_tau, B, n1 = W_out.shape[:3]
    readout_shape = W_out.shape[3:]
    Wm = W_out.reshape(n_tau, B, n1, -1).astype(F32)

    if "max" in spec.aggregations or not snapshots:
        res = _Reservoir(res.xp, res.W, res.W_in, res.bias, res.U, res.leak, res.decay)
    else:
        res.restore(snapshots[0])
    sums, maxs, traces = _readout_pass(xp, spec, members, Wm, res.run(T, spec.chunk))

    nf = spec.n_folds
    results = []
    for b, m in enumerate(members):
        train = np.flatnonzero(m.beat_fold >= 0)
        test = np.flatnonzero(m.beat_fold < 0)
        folds = m.beat_fold[train]
        shape = (len(m.beat_start), n_tau) + readout_shape
        out = {"val": {}, "test": {}}
        for agg in spec.aggregations:
            if agg == "mean":
                full = to_host(sums[b]) / L
                # training beats: readout of the beat's mean state (from pass 1)
                mean_states = xp.swapaxes(beat_sums[b], 0, 1) / L          # (n_tau, n_train, n1)
                full[train] = np.swapaxes(to_host(xp.matmul(mean_states, Wm[:, b])), 0, 1)
            elif agg == "max":
                full = to_host(maxs[b])
            else:
                raise ValueError(f"unknown aggregation {agg!r}")
            full = full.reshape(shape)                                     # (beats, tau, model, w, lam)
            # each training beat is scored by the model that did not see its fold
            out["val"][agg] = np.moveaxis(full[train, :, folds], 0, -1)
            out["test"][agg] = np.moveaxis(full[test, :, nf], 0, -1)       # (tau, w, lam, beats)
        if m.trace_len:
            tr = to_host(traces[b]).reshape((n_tau, m.trace_len) + readout_shape)
            out["trace"] = np.moveaxis(tr[:, :, nf], 1, -1)                # (tau, w, lam, P)
        if spec.return_weights:
            out["weights"] = to_host(W_out[:, b])
        results.append(out)
    return results
