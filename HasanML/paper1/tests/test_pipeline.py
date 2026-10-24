import dataclasses

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from src.data import extract_beats
from src.encode import build_input, delta_modulation
from src.esn import JobSpec, Member, make_reservoir, run_job, scale_spectral_radius
from src.preprocess import filter_without_leakage, highpass_sos, minmax_normalize
from src.readout import ridge_fit, ridge_path

FS = 360


def test_delta_modulation_up_down_and_reference_reset():
    x = np.array([0.0, 0.05, 0.11, 0.15, 0.30, 0.25, 0.10, 0.10, 0.21])
    spikes = delta_modulation(x, 0.1)
    # 0.11-0.0 > 0.1 -> UP (ref 0.11); 0.30-0.11 -> UP (ref 0.30);
    # 0.10-0.30 < -0.1 -> DOWN (ref 0.10); 0.21-0.10 > 0.1 -> UP
    np.testing.assert_array_equal(np.flatnonzero(spikes[:, 0]), [2, 4, 8])
    np.testing.assert_array_equal(np.flatnonzero(spikes[:, 1]), [6])
    assert spikes.shape == (9, 2)
    assert build_input(x, "raw", 0.1).shape == (9, 1)


def test_delta_modulation_needs_strictly_greater_than_threshold():
    spikes = delta_modulation(np.array([0.0, 0.1, 0.2]), 0.1)
    assert spikes[1].sum() == 0  # |0.1 - 0.0| is not > 0.1
    assert spikes[2, 0] == 1


@pytest.mark.parametrize("rho", [0.5, 0.9, 1.2])
def test_spectral_radius_scaling(rho):
    W = np.random.default_rng(0).standard_normal((200, 200))
    Ws = scale_spectral_radius(W, rho)
    assert np.isclose(np.max(np.abs(np.linalg.eigvals(Ws))), rho)


def test_reservoir_is_sparse_and_scaled():
    W, W_in, b = make_reservoir(3, 300, 2, 0.1, 0.9, 0.2)
    assert np.isclose(np.max(np.abs(np.linalg.eigvals(W))), 0.9)
    assert 0.08 < np.count_nonzero(W) / W.size < 0.12
    assert np.all(np.abs(W_in) <= 1) and np.all(np.abs(b) <= 0.2)


@pytest.mark.parametrize("lam", [1e-6, 1e-2, 1.0])
def test_ridge_matches_sklearn(lam):
    rng = np.random.default_rng(1)
    states = rng.standard_normal((500, 20))
    y = (rng.random(500) > 0.8).astype(float)
    X = np.vstack([states.T, np.ones(500)])  # features x time, bias row
    ref = Ridge(alpha=lam, fit_intercept=False).fit(X.T, y).coef_
    np.testing.assert_allclose(ridge_fit(X, y, lam)[0], ref, rtol=1e-8, atol=1e-10)
    path = ridge_path(np, X @ X.T, X @ y, np.array([lam]))[:, 0]
    np.testing.assert_allclose(path, ref, rtol=1e-6, atol=1e-8)


def test_label_extraction_v_is_pvc_and_non_beats_dropped():
    samples = [10, 20, 30, 40, 50, 60, 70, 80]
    symbols = ["N", "V", "+", "A", "~", "V", "|", "L"]
    s, sym, labels = extract_beats(samples, symbols)
    np.testing.assert_array_equal(s, [10, 20, 40, 60, 80])
    np.testing.assert_array_equal(sym, ["N", "V", "A", "V", "L"])
    np.testing.assert_array_equal(labels, [0, 1, 0, 1, 0])


# --- leakage: nothing from the test window may influence fitting or selection ---

T_TRAIN = 30 * FS          # 30 s training window for a fast synthetic run
T_END = T_TRAIN + 15 * FS
WIN = (-36, 54)


def _synthetic_ecg(rng):
    n = T_END + 200
    t = np.arange(n) / FS
    x = 0.3 * np.sin(2 * np.pi * 0.2 * t) + 0.02 * rng.standard_normal(n)
    r_peaks = np.arange(200, T_END, int(0.8 * FS))
    labels = (np.arange(len(r_peaks)) % 4 == 3).astype(np.int8)
    for r, lab in zip(r_peaks, labels):
        width = 12 if lab else 4
        x[r - 3 * width:r + 3 * width] += (2.0 if lab else 1.0) * np.exp(
            -0.5 * (np.arange(-3 * width, 3 * width) / width) ** 2)
    return x, r_peaks, labels


def _member(x_raw, r_peaks, labels, washout, nf):
    sos = highpass_sos(4, 0.5, FS)
    t_ext = T_END + WIN[1]
    x = minmax_normalize(filter_without_leakage(x_raw, sos, T_TRAIN, t_ext), T_TRAIN)
    u = build_input(x, "spikes", 0.1)
    train = (r_peaks < T_TRAIN) & (r_peaks + WIN[0] >= washout) & (r_peaks + WIN[1] <= T_TRAIN)
    test = (r_peaks >= T_TRAIN) & (r_peaks < T_END)
    keep = train | test
    fold = np.where(train, np.minimum(r_peaks // (T_TRAIN // nf), nf - 1), -1)
    target = np.zeros(T_TRAIN, bool)
    for r in r_peaks[(labels == 1) & (r_peaks < T_TRAIN)]:
        target[r + WIN[0]:min(r + WIN[1], T_TRAIN)] = True
    return Member(key=("syn",), u=u, input_scaling=0.5, leak=0.99, target=target,
                  beat_start=r_peaks[keep] + WIN[0], beat_fold=fold[keep],
                  trace_start=T_TRAIN, trace_len=2 * FS)


def _spec(n=40):
    return JobSpec(seed=0, n_reservoir=n, connectivity=0.2, spectral_radius=0.9, bias_scaling=0.2,
                   decays=(float(np.exp(-1000 / (10 * FS))), float(np.exp(-1000 / (50 * FS)))),
                   lambdas=(1e-6, 1e-2), weightings=("none", "balanced"), t_train=T_TRAIN,
                   washout=2 * FS, n_folds=5, window_len=WIN[1] - WIN[0], chunk=500,
                   aggregations=("mean", "max"), return_weights=True)


def test_no_test_window_sample_is_used_in_fitting():
    rng = np.random.default_rng(0)
    x, r_peaks, labels = _synthetic_ecg(rng)
    spec = _spec()
    base = run_job(spec, [_member(x, r_peaks, labels, spec.washout, spec.n_folds)])[0]

    # Corrupt everything from the start of the test window onwards, including its labels.
    x2 = x.copy()
    x2[T_TRAIN:] += 5 * rng.standard_normal(len(x) - T_TRAIN) + 3.0
    labels2 = labels.copy()
    labels2[r_peaks >= T_TRAIN] ^= 1
    changed = run_job(spec, [_member(x2, r_peaks, labels2, spec.washout, spec.n_folds)])[0]

    np.testing.assert_array_equal(base["weights"], changed["weights"])
    for agg in ("mean", "max"):
        np.testing.assert_array_equal(base["val"][agg], changed["val"][agg])
        assert not np.allclose(base["test"][agg], changed["test"][agg])


def test_batched_job_matches_individual_runs():
    rng = np.random.default_rng(1)
    x, r_peaks, labels = _synthetic_ecg(rng)
    spec = _spec()
    m = _member(x, r_peaks, labels, spec.washout, spec.n_folds)
    members = [dataclasses.replace(m, input_scaling=s, leak=a)
               for s, a in [(0.1, 0.99), (1.0, 0.01), (0.5, 0.99)]]
    batched = run_job(spec, members)
    for mem, res in zip(members, batched):
        single = run_job(spec, [mem])[0]
        for agg in ("mean", "max"):
            np.testing.assert_allclose(res["val"][agg], single["val"][agg], rtol=1e-4, atol=1e-5)
            np.testing.assert_allclose(res["test"][agg], single["test"][agg], rtol=1e-4, atol=1e-5)


def test_pvc_scores_separate_on_easy_synthetic_data():
    rng = np.random.default_rng(2)
    x, r_peaks, labels = _synthetic_ecg(rng)
    spec = dataclasses.replace(_spec(80), return_weights=False)
    m = _member(x, r_peaks, labels, spec.washout, spec.n_folds)
    res = run_job(spec, [m])[0]
    y_test = labels[(r_peaks >= T_TRAIN) & (r_peaks < T_END)]
    # tau 50 ms, unweighted, lambda 1e-2 (1e-6 overfits 30 s of synthetic training data)
    scores = res["test"]["mean"][1, 0, 1]
    assert scores[y_test == 1].min() > scores[y_test == 0].max()


def test_mean_from_snapshot_matches_full_replay():
    rng = np.random.default_rng(3)
    x, r_peaks, labels = _synthetic_ecg(rng)
    spec = _spec()
    m = _member(x, r_peaks, labels, spec.washout, spec.n_folds)
    full = run_job(spec, [m])[0]                                    # "max" forces a replay from 0
    resumed = run_job(dataclasses.replace(spec, aggregations=("mean",)), [m])[0]
    assert set(resumed["val"]) == {"mean"}
    for part in ("val", "test"):
        np.testing.assert_allclose(resumed[part]["mean"], full[part]["mean"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(resumed["trace"], full["trace"], rtol=1e-4, atol=1e-4)


def test_start_offset_shifts_split_windows():
    from src.run_experiment import load_config, prepare_record, PROJECT_ROOT
    cfg = load_config(PROJECT_ROOT / "configs" / "default.yaml")
    base = prepare_record("200", cfg)
    cfg["split"]["start_minutes_per_record"] = {"200": 5}
    shifted = prepare_record("200", cfg)
    fs, w_lo = base.fs, base.window[0]
    off = int(5 * 60 * fs)
    # test beats of the shifted split are exactly the beats with R in [15, 20) min of the record
    r_test = shifted.beat_start[shifted.beat_fold < 0] - w_lo + off
    assert r_test.min() >= 15 * 60 * fs and r_test.max() < 20 * 60 * fs
    assert (shifted.labels[shifted.beat_fold < 0] == 1).sum() == 170
    # the training target is 1 only around PVCs whose R peak is in [5, 15) min of the record
    t = np.flatnonzero(shifted.target) + off
    assert t.min() >= 5 * 60 * fs + w_lo and t.max() < 15 * 60 * fs
    assert shifted.x_norm.shape == base.x_norm.shape
