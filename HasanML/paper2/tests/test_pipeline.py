import numpy as np
import pytest

from src.data import (CLASSES, DS1, DS2, PACED_RECORDS, make_splits, map_labels,
                      split_inter_patient, split_intra_patient, used_records)
from src.features import FEATURE_NAMES, morphology_features, rr_features
from src.fiducials import POINTS, detect, walk_left, walk_right
from src.metrics import confusion, metrics_from_cm, paper_metrics_from_error
from src.preprocess import baseline_sos, beat_windows, remove_baseline

FS = 360
FID_CFG = {"r_refine_samples": 10, "p_window_ms": [-250, -80], "q_window_ms": [-80, 0],
           "s_window_ms": [0, 80], "t_window_ms": [100, 350], "wave_amplitude_frac": 0.2,
           "qrs_slope_frac": 0.1, "qrs_edge_ms": 40, "iso_level": "median"}


# ----------------------------------------------------------------------------- labels

def test_paper_label_map_follows_table1():
    syms = ["N", "L", "R", "e", "j", "A", "a", "J", "x", "V", "!", "E", "F", "Q", "S", "/", "f", "+"]
    got = map_labels(syms, "paper")
    want = [0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 3, 4, -1, -1, -1, -1]
    np.testing.assert_array_equal(got, want)


def test_aami_label_map():
    syms = ["N", "S", "x", "V", "!", "E", "F", "/", "f", "Q"]
    np.testing.assert_array_equal(map_labels(syms, "aami"), [0, 1, -1, 2, -1, 2, 3, 4, 4, 4])


def test_record_lists():
    recs = used_records(PACED_RECORDS)
    assert len(recs) == 44 and not set(recs) & set(PACED_RECORDS)
    assert len(DS1) == len(DS2) == 22 and sorted(DS1 + DS2) == recs


# ----------------------------------------------------------------------------- windows

def test_beat_window_extraction_and_edge_drops():
    x = np.arange(1000, dtype=float)
    r = np.array([100, 127, 500, 871, 872])
    w, keep = beat_windows(x, r, 127, 128)
    np.testing.assert_array_equal(keep, [False, True, True, True, False])
    assert w.shape == (3, 256)
    assert w[0, 127] == 127 and w[0, 0] == 0 and w[2, -1] == 999
    np.testing.assert_array_equal(w[1], np.arange(373, 629))


def test_baseline_removal_kills_slow_drift_keeps_qrs():
    t = np.arange(0, 30, 1 / FS)
    drift = 0.8 * np.sin(2 * np.pi * 0.2 * t)
    spikes = np.zeros_like(t)
    spikes[::FS] = 1.0
    clean, base = remove_baseline(drift + spikes, baseline_sos(4, 1.0, 0.1, 40.0, FS))
    mid = slice(5 * FS, 25 * FS)
    assert np.abs(base[mid] - drift[mid]).max() < 0.15
    assert clean[mid][spikes[mid] == 1].min() > 0.8


# ----------------------------------------------------------------------------- RR

def test_rr_features_on_synthetic_sequence():
    rr_ms = np.array([800, 800, 800, 400, 1200, 800, 800, 800, 800, 800, 800, 800], dtype=float)
    r = np.concatenate([[1000], 1000 + np.cumsum(rr_ms * FS / 1000)])
    f = rr_features(r, FS, 5)
    np.testing.assert_allclose(f["pre"][1:], rr_ms)
    np.testing.assert_allclose(f["post"][:-1], rr_ms)
    assert f["pre"][0] == rr_ms[0] and f["post"][-1] == rr_ms[-1]
    np.testing.assert_allclose(f["avg"], rr_ms.mean())
    # beat 5 (index into r): intervals rr[0..4] before and rr[5..9] after
    np.testing.assert_allclose(f["local"][5], rr_ms[0:10].mean())
    # beat 1: only one interval before it, five after -> rr[0..5]
    np.testing.assert_allclose(f["local"][1], rr_ms[0:6].mean())


# ----------------------------------------------------------------------------- fiducials

def _synthetic_beat():
    t = (np.arange(256) - 127) * 1000 / FS          # ms from R
    g = lambda mu, sd, a: a * np.exp(-0.5 * ((t - mu) / sd) ** 2)
    return g(-180, 25, 0.2) + g(-25, 6, -0.25) + g(0, 10, 1.5) + g(25, 6, -0.35) + g(260, 45, 0.35)


def test_fiducials_on_synthetic_beat_are_ordered_and_located():
    beat = _synthetic_beat()[None, :]
    pts = detect(beat, FS, 127, FID_CFG)
    cols = [int(pts[k][0]) for k in POINTS]
    assert cols == sorted(cols)
    ms = lambda c: (c - 127) * 1000 / FS
    assert abs(ms(pts["P"][0]) + 180) < 6
    assert abs(ms(pts["R"][0])) < 3
    assert abs(ms(pts["Q"][0]) + 25) < 8 and abs(ms(pts["S"][0]) - 25) < 8
    assert abs(ms(pts["T"][0]) - 260) < 6
    assert pts["Q1"][0] <= pts["Q"][0] - 2       # onset is before the Q minimum, not at it


def test_walks():
    cond = np.array([[1, 0, 0, 1, 0, 0, 0, 1]], dtype=bool)
    assert walk_left(cond, np.array([6]), np.array([0]))[0] == 3
    assert walk_left(cond, np.array([3]), np.array([1]))[0] == 1      # none in [1, 3) -> limit
    assert walk_right(cond, np.array([3]), np.array([7]))[0] == 7
    assert walk_right(cond, np.array([3]), np.array([6]))[0] == 6     # none in (3, 6] -> limit


def test_feature_matrix_shape():
    beats = np.repeat(_synthetic_beat()[None, :], 4, axis=0)
    X = morphology_features(beats, detect(beats, FS, 127, FID_CFG), FS)
    assert X.shape == (4, 57) == (4, len(FEATURE_NAMES))
    assert np.isnan(X[:, 30:35]).all() and not np.isnan(np.delete(X, range(30, 35), axis=1)).any()


# ----------------------------------------------------------------------------- metrics

def test_micro_averaging_formulas_on_known_confusion_matrix():
    cm = np.array([[90, 3, 1, 0, 0],
                   [4, 20, 0, 0, 0],
                   [1, 0, 30, 2, 0],
                   [0, 0, 1, 10, 0],
                   [0, 0, 0, 0, 3]])
    n, k = cm.sum(), 5
    e = 1 - np.trace(cm) / n
    m = metrics_from_cm(cm)
    assert m["accuracy"] == pytest.approx(1 - e)
    assert m["ovr_accuracy"] == pytest.approx(1 - 2 * e / k)
    assert m["micro_specificity"] == pytest.approx(1 - e / (k - 1))
    for key in ("micro_sensitivity", "micro_precision", "micro_f1"):
        assert m[key] == pytest.approx(1 - e)
    p = paper_metrics_from_error(e)
    assert p["accuracy"] == pytest.approx(m["ovr_accuracy"])
    assert p["specificity"] == pytest.approx(m["micro_specificity"])
    assert m["se_S"] == pytest.approx(20 / 24) and m["ppv_S"] == pytest.approx(20 / 23)
    assert m["majority_accuracy"] == pytest.approx(94 / n)


def test_paper_numbers_satisfy_micro_hypothesis():
    # LR, 20 features: Se = 99.416 -> Acc 99.766, Spec 99.854 (paper Table 4)
    p = paper_metrics_from_error(1 - 0.99416)
    assert round(100 * p["accuracy"], 3) == 99.766 and round(100 * p["specificity"], 3) == 99.854
    p = paper_metrics_from_error(1 - 0.99503)
    assert round(100 * p["accuracy"], 3) == 99.801 and round(100 * p["specificity"], 3) == 99.876


def test_confusion_counts():
    cm = confusion([0, 0, 1, 2, 4], [0, 1, 1, 2, 0])
    assert cm.sum() == 5 and cm[0, 1] == 1 and cm[4, 0] == 1


# ----------------------------------------------------------------------------- splits

def test_protocol_a_has_no_test_record_in_training():
    rng = np.random.default_rng(0)
    recs = np.array(used_records(PACED_RECORDS))
    record = rng.choice(recs, 5000)
    tr, te = split_inter_patient(record)
    assert not set(record[tr]) & set(record[te])
    assert set(record[tr]) <= set(DS1) and set(record[te]) <= set(DS2)
    assert len(tr) + len(te) == len(record)


def test_protocol_b_test_fraction_and_disjoint():
    y = np.repeat(np.arange(5), [90083, 2972, 7480, 802, 15])
    tr, te = split_intra_patient(y, 0)
    np.testing.assert_array_equal(np.bincount(y[te], minlength=5), [44238, 1972, 3220, 388, 7])
    assert not set(tr) & set(te) and len(tr) + len(te) == len(y)


def test_protocol_c_folds_partition_everything():
    y = np.repeat(np.arange(5), [500, 50, 60, 20, 10])
    folds = make_splits("C", y, np.zeros(len(y)), 0, 5)
    assert len(folds) == 5
    np.testing.assert_array_equal(np.sort(np.concatenate([te for _, te in folds])), np.arange(len(y)))
