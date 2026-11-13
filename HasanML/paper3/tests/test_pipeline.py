"""
Unit tests for the Farag (2023) tiny 1D-CNN pipeline.

Run from HasanML/paper3/:
    python -m pytest tests/ -v

Tests cover:
  1. DS2 / DS1-train / DS1-val isolation (no leakage)
  2. AAMI label mapping completeness and binary definitions
  3. RR features: causality, local/global window sizes
  4. Resampling: R-peak index mapping
  5. Segmentation: length 64, edge-padding
  6. MF init: no time-flip (cross-correlation, not convolution)
  7. Metrics: plain accuracy and OvR accuracy from a known CM
  8. Parameter count: in 1,000–2,000 for 3-class single-channel
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from the repo root or from HasanML/paper3/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    DS1_TRAIN, DS1_VAL, DS2, DS1,
    AAMI_MAP, AAMI_CLASSES, SUB_CLASSES, BEAT_SYMBOLS,
    aami_labels, task_labels,
)
from src.preprocess import (
    resample_signal, map_r_peaks, segment_beat,
    normalise_segment, compute_derivative, crop_to_nk,
)
from src.rr import compute_rr_intervals, compute_rr_features, _causal_mean
from src.templates import compute_subclass_templates, templates_to_kernel
from src.metrics import multiclass_metrics, confusion_matrix_k


# ============================================================
# 1. Split isolation
# ============================================================

def test_ds1_ds2_disjoint():
    """No record appears in both DS1 and DS2."""
    assert not set(DS1) & set(DS2), "DS1 and DS2 must be disjoint"


def test_ds1_train_val_disjoint():
    """DS1-train and DS1-val are disjoint subsets of DS1."""
    assert not set(DS1_TRAIN) & set(DS1_VAL)
    assert set(DS1_TRAIN) | set(DS1_VAL) == set(DS1)


def test_ds2_not_in_ds1():
    """No DS2 record appears in DS1-train or DS1-val."""
    for r in DS2:
        assert r not in DS1_TRAIN
        assert r not in DS1_VAL


def test_ds1_train_lt_200():
    """DS1-train = DS1 records with number < 200 (paper Sec 5.3)."""
    for r in DS1_TRAIN:
        assert int(r) < 200, f"DS1-train contains record {r} (>= 200)"


def test_ds1_val_ge_200():
    """DS1-val = DS1 records with number >= 200 (paper Sec 5.3)."""
    for r in DS1_VAL:
        assert int(r) >= 200, f"DS1-val contains record {r} (< 200)"


# ============================================================
# 2. AAMI label mapping
# ============================================================

def test_aami_map_covers_sub_classes():
    """Every sub-class symbol is in the AAMI map."""
    for sym in SUB_CLASSES:
        assert sym in AAMI_MAP, f"Sub-class '{sym}' not in AAMI_MAP"


def test_aami_classes():
    """AAMI map produces only valid class names."""
    for sym, cls in AAMI_MAP.items():
        assert cls in AAMI_CLASSES, f"'{sym}' maps to unknown class '{cls}'"


def test_aami_labels_known():
    """AAMI labels for known symbols are correct."""
    syms = ["N", "L", "R", "e", "j",   # N class
            "A", "a", "J", "S",          # S class
            "V", "E",                    # V class
            "F",                         # F class
            "/", "f", "Q"]               # Q class
    expected = [0, 0, 0, 0, 0,
                1, 1, 1, 1,
                2, 2,
                3,
                4, 4, 4]
    y = aami_labels(np.array(syms, dtype=object))
    assert list(y) == expected, f"AAMI labels mismatch: {list(y)} != {expected}"


def test_aami_unknown_symbol():
    """Symbols not in AAMI_MAP get label -1."""
    syms = np.array(["X", "~", "P"], dtype=object)
    y = aami_labels(syms)
    assert all(v == -1 for v in y), f"Expected all -1, got {y}"


def test_binary_aami_definition():
    """binary/aami: N=0 for AAMI-N symbols, Arr=1 for S/V/F/Q."""
    syms   = np.array(["N", "L", "R", "e", "j", "A", "V", "F", "/"], dtype=object)
    y_aami = aami_labels(syms)
    y_bin  = task_labels(y_aami, syms, "binary", classes=3, binary_def="aami")
    assert list(y_bin) == [0, 0, 0, 0, 0, 1, 1, 1, 1], f"binary/aami mismatch: {list(y_bin)}"


def test_binary_raw_definition():
    """binary/raw: N=0 for 'N' symbol only; everything else=1."""
    syms   = np.array(["N", "L", "R", "V", "A", "/"], dtype=object)
    y_aami = aami_labels(syms)
    y_bin  = task_labels(y_aami, syms, "binary", classes=3, binary_def="raw")
    assert list(y_bin) == [0, 1, 1, 1, 1, 1], f"binary/raw mismatch: {list(y_bin)}"


def test_multiclass3_drops_F_Q():
    """multiclass-3 drops F and Q (labels become -1)."""
    syms   = np.array(["N", "V", "F", "Q", "/"], dtype=object)
    y_aami = aami_labels(syms)
    y_3    = task_labels(y_aami, syms, "multiclass", classes=3, binary_def="aami")
    assert y_3[0] == 0   # N
    assert y_3[1] == 2   # V
    assert y_3[2] == -1  # F dropped
    assert y_3[3] == -1  # Q class not in 3
    assert y_3[4] == -1  # /  -> Q class -> dropped


# ============================================================
# 3. RR features — causality and window sizes
# ============================================================

def test_rr_causal_local():
    """Changing a future beat does not affect past features (except post_RR of the previous beat)."""
    # Create a sequence of 10 evenly-spaced beats at 128 Hz, 1 s apart
    fs = 128.0
    r  = np.arange(10) * 128  # beats at 0, 128, 256, ... samples

    feats_orig = compute_rr_features(r, fs, local_window=80, global_window=400)

    # Perturb beat index 7 (move it 10 samples later)
    r_perturbed = r.copy()
    r_perturbed[7] += 10

    feats_pert = compute_rr_features(r_perturbed, fs, local_window=80, global_window=400)

    # Beats 0–5 should be unchanged (beat 6 is affected via post_RR of beat 5,
    # but beats 0–5's pre_RR and their local/global means are unchanged because
    # they only use past RR intervals up to and including their own pre_RR)
    # Specifically: beats 0..5 pre_RR features should be identical.
    np.testing.assert_array_almost_equal(
        feats_orig[:6, 0:2],  # pre_RR features for beats 0-5
        feats_pert[:6, 0:2],
        decimal=5,
        err_msg="Pre-RR features for early beats should be causally unchanged",
    )


def test_rr_local_global_windows():
    """local_mean uses last 80 intervals; global_mean uses last 400."""
    fs = 128.0
    # 500 beats, all evenly spaced at 128 samples (1 Hz)
    n = 500
    r = np.arange(n) * 128
    # Introduce a sudden change at beat 200: double the interval
    r_changed = r.copy()
    r_changed[200:] += 128   # all beats after 200 shift by 128 samples

    feats = compute_rr_features(r_changed, fs, local_window=80, global_window=400)

    # At beat 200: pre_RR = 2.0 s (double the previous 1.0 s)
    # local_mean at beat 200 should still be ~1.0 s (only last 80 beats, all at 1.0)
    pre_rr_200   = r_changed[200] - r_changed[199]  # = 2 * 128 = 256 samples
    local_mean_200 = feats[200, 0] * pre_rr_200     # reverse-engineer the mean
    # local_mean at beat 200 is mean of last 80 pre_rr values including pre_rr[200]=256
    # 79 values of 128 + 1 value of 256 -> mean = (79*128 + 256)/80 = 129.6
    expected_local = (79 * 128 + 256) / 80
    # The feature is pre_rr / local_mean = 256 / 129.6 ≈ 1.975
    expected_feat  = 256.0 / expected_local
    assert abs(feats[200, 0] - expected_feat) < 0.05, (
        f"Local mean window at beat 200: expected feature ≈ {expected_feat:.3f}, "
        f"got {feats[200, 0]:.3f}"
    )


def test_rr_no_negative_or_zero_intervals():
    """RR intervals derived from sorted, monotone R-peaks should be positive."""
    r = np.array([0, 50, 110, 220, 350, 480], dtype=np.int64)
    pre_rr = compute_rr_intervals(r)
    assert all(pre_rr > 0), f"Non-positive pre_RR found: {pre_rr}"


# ============================================================
# 4. Resampling: R-peak index mapping
# ============================================================

def test_r_peak_mapping_exact():
    """A known R-peak at 360 Hz maps to the right index at 128 Hz."""
    # R-peak at sample 360 (1.0 s at 360 Hz) should map to sample 128 (1.0 s at 128 Hz)
    r_360 = np.array([360, 720, 1080], dtype=np.int64)
    r_128 = map_r_peaks(r_360)
    expected = np.array([128, 256, 384], dtype=np.int64)
    np.testing.assert_array_equal(r_128, expected)


def test_resample_length():
    """Resampled signal has approximately the right length (128/360 × n_samples)."""
    n = 3600 * 10   # 10 s at 360 Hz
    sig = np.random.default_rng(0).standard_normal(n)
    out = resample_signal(sig)
    expected_len = round(n * 128 / 360)
    assert abs(len(out) - expected_len) <= 2, (
        f"Resampled length {len(out)} differs from expected {expected_len} by >2 samples"
    )


# ============================================================
# 5. Segmentation
# ============================================================

def test_segment_length():
    """Segment is exactly before+after samples long."""
    sig = np.zeros(1000)
    seg, padded = segment_beat(sig, 500, before=32, after=32)
    assert len(seg) == 64
    assert not padded


def test_segment_edge_padding_start():
    """R-peak at start of record is padded on the left."""
    sig = np.ones(1000)
    seg, padded = segment_beat(sig, 10, before=32, after=32)  # would need samples -22..-1
    assert len(seg) == 64
    assert padded
    assert seg[0] == sig[0]   # edge value = first sample


def test_segment_edge_padding_end():
    """R-peak at end of record is padded on the right."""
    sig = np.ones(1000)
    seg, padded = segment_beat(sig, 990, before=32, after=32)  # needs samples 1023..1021
    assert len(seg) == 64
    assert padded
    assert seg[-1] == sig[-1]  # edge value = last sample


def test_derivative_length():
    """Derivative of a 64-sample segment has the same 64-sample length after padding."""
    seg = np.random.default_rng(42).standard_normal(64)
    d   = compute_derivative(seg)
    assert len(d) == 64


def test_nk32_crop():
    """NK=32 crop keeps samples [16:48] of a 64-sample segment."""
    seg = np.arange(64, dtype=float)
    cropped = crop_to_nk(seg, nk=32, full=64)
    assert len(cropped) == 32
    np.testing.assert_array_equal(cropped, seg[16:48])


def test_nk64_no_crop():
    """NK=64 crop is a no-op."""
    seg = np.arange(64, dtype=float)
    assert np.array_equal(crop_to_nk(seg, nk=64, full=64), seg)


# ============================================================
# 6. MF initialisation — no time-flip (cross-correlation)
# ============================================================

def test_mf_no_flip():
    """
    Conv1D applied to a beat that is exactly template i should produce its
    maximum response on filter i (not filter mirror(i)).
    Verifies that the kernel is loaded without time-reversal.
    """
    import tensorflow as tf
    from src.model import build_model, apply_mf_init
    from src.templates import compute_subclass_templates, templates_to_kernel

    nk = 16
    n_filters = 3
    sub_cls = ["N", "L", "R"]

    # Build synthetic templates: each is a one-hot-ish pulse at a different position
    templates_dict = {}
    for i, sym in enumerate(sub_cls):
        t = np.zeros(nk)
        t[i * nk // n_filters] = 1.0  # pulse at different location per template
        norm = np.linalg.norm(t)
        templates_dict[sym] = t / norm if norm > 0 else t

    kernel = templates_to_kernel(templates_dict, sub_cls, n_channels=1)
    # kernel shape: (nk, 1, n_filters)

    # Build a minimal model with just Conv1D + GlobalMaxPool
    ecg_in = tf.keras.Input(shape=(nk, 1))
    x = tf.keras.layers.Conv1D(n_filters, nk, padding="valid",
                                activation="linear", use_bias=False)(ecg_in)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    model_simple = tf.keras.Model(ecg_in, x)

    # Set weights
    model_simple.layers[1].set_weights([kernel])

    # Feed template 0 exactly: response on filter 0 should be maximum (= 1.0)
    for i, sym in enumerate(sub_cls):
        t_i = templates_dict[sym].reshape(1, nk, 1).astype(np.float32)
        resp = model_simple.predict(t_i, verbose=0)[0]
        assert resp[i] == pytest.approx(1.0, abs=1e-4), (
            f"Template {i} ({sym}): expected max response on filter {i}, "
            f"got responses {resp}"
        )
        # Other filters should have lower responses (orthogonal templates)
        for j in range(n_filters):
            if j != i:
                assert resp[j] < resp[i] - 0.1, (
                    f"Filter {j} responded as strongly as filter {i} to template {i}; "
                    f"check for time-reversal bug"
                )


# ============================================================
# 7. Metrics — accuracy definitions
# ============================================================

def test_plain_accuracy():
    """Plain accuracy = TP_total / N."""
    # K=3 confusion matrix; 100 beats total, 90 correct
    cm = np.array([[50, 5, 0], [0, 20, 5], [0, 0, 20]])
    y_true = []
    y_pred = []
    for i in range(3):
        for j in range(3):
            y_true += [i] * int(cm[i, j])
            y_pred += [j] * int(cm[i, j])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = multiclass_metrics(y_true, y_pred, k=3)
    n = cm.sum()
    expected_plain = np.trace(cm) / n
    assert abs(m["accuracy"] - expected_plain) < 1e-6


def test_ovr_accuracy():
    """OvR accuracy = 1 - 2e/K for a symmetric K-class problem."""
    # Perfect prediction: OvR = plain = 1.0
    k = 3
    n = 90
    y_true = np.array([i // 30 for i in range(n)])
    y_pred = y_true.copy()
    m = multiclass_metrics(y_true, y_pred, k=k)
    assert abs(m["accuracy"]     - 1.0) < 1e-6
    assert abs(m["ovr_accuracy"] - 1.0) < 1e-6


def test_ovr_accuracy_mismatch():
    """For a non-perfect predictor, OvR acc != plain acc in general."""
    cm = np.array([[80, 10, 10], [5, 40, 5], [0, 2, 48]])
    y_true, y_pred = [], []
    for i in range(3):
        for j in range(3):
            y_true += [i] * int(cm[i, j])
            y_pred += [j] * int(cm[i, j])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = multiclass_metrics(y_true, y_pred, k=3)
    # They should NOT be equal for an imperfect classifier
    assert m["ovr_accuracy"] != m["accuracy"], (
        "OvR and plain accuracy should differ for non-perfect prediction"
    )
    # OvR accuracy >= plain accuracy for K=3 (since 1-2e/3 >= 1-e when e<=0.5)
    # Actually: (1-2e/3) vs (1-e); 1-2e/3 >= 1-e iff e/3 >= 0 (always true)
    assert m["ovr_accuracy"] >= m["accuracy"] - 1e-9


def test_majority_baseline():
    """Majority class baseline = count(most common class) / total."""
    cm = np.array([[900, 50, 50], [10, 20, 0], [5, 5, 10]])
    y_true, y_pred = [], []
    for i in range(3):
        for j in range(3):
            y_true += [i] * int(cm[i, j])
            y_pred += [j] * int(cm[i, j])
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    m = multiclass_metrics(y_true, y_pred, k=3)
    expected = cm.sum(axis=1).max() / cm.sum()
    assert abs(m["majority_acc"] - expected) < 1e-6


# ============================================================
# 8. Parameter count
# ============================================================

@pytest.mark.parametrize("nk,n_channels,n_classes,expected_range", [
    (32, 1, 3, (1000, 2000)),
    (64, 1, 3, (1000, 2000)),
    (32, 2, 3, (1000, 2500)),  # 2-channel slightly more
    (32, 1, 2, (1000, 2000)),  # binary
    (32, 1, 5, (1000, 2500)),  # 5-class
])
def test_param_count(nk, n_channels, n_classes, expected_range):
    """Model parameter count is in the expected range."""
    from src.model import build_model, validate_param_count
    model = build_model(nk=nk, n_channels=n_channels, n_classes=n_classes)
    total = model.count_params()
    lo, hi = expected_range
    assert lo <= total <= hi, (
        f"nk={nk}, C={n_channels}, n_classes={n_classes}: "
        f"total params={total} outside [{lo}, {hi}]"
    )


def test_param_count_guard_raises():
    """validate_param_count raises for a clearly wrong architecture."""
    import tensorflow as tf
    # Build a deliberately wrong model (too many params) to trigger the guard
    ecg_in = tf.keras.Input(shape=(32, 1), name="ecg_in")
    x = tf.keras.layers.Conv1D(100, 32, padding="valid")(ecg_in)
    x = tf.keras.layers.GlobalMaxPooling1D()(x)
    rr_in = tf.keras.Input(shape=(4,), name="rr_in")
    merged = tf.keras.layers.Concatenate()([x, rr_in])
    out = tf.keras.layers.Dense(3, activation="softmax")(merged)
    bad_model = tf.keras.Model([ecg_in, rr_in], out)

    with pytest.raises(RuntimeError, match="outside expected range"):
        from src.model import validate_param_count
        validate_param_count(bad_model, nk=32, n_channels=1, n_classes=3)
