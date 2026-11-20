"""
Unit tests for the KAN-Former-1D (paper4) pipeline.

Run from HasanML/paper4/:
    python -m pytest tests/ -v

Tests cover:
  1. DS2 / DS1-train / DS1-val isolation (no patient leakage)
  2. AAMI label mapping completeness
  3. RR features: causality (no future information), 6-feature output
  4. Window centering: beat and context windows use correct offsets
  5. KANLinear: output shape, gradient flow, basis partition-of-unity, sin(x) fit
  6. Focal loss: gamma=0 reduces to weighted cross-entropy
  7. Metrics: confusion matrix, OvR accuracy, macro-F1 from known inputs
  8. Soft labels: sums to 1, shape preserved at all temperatures
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running from the repo root or from HasanML/paper4/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import (
    DS1_TRAIN, DS1_VAL, DS2, DS1,
    AAMI_MAP, AAMI_IDX, AAMI_CLASSES, BEAT_SYMBOLS,
    aami_labels, task_labels,
)
from src.rr import compute_rr_intervals, compute_rr_features, _causal_mean
from src.kan import KANLinear, b_splines
from src.losses import FocalLoss, WeightedCE, build_loss
from src.metrics import (
    ovr_accuracy, multiclass_metrics, macro_f1_from_preds,
)
from src.distill_export import ensemble_logits


# ============================================================
# 1. Split isolation
# ============================================================

def test_ds1_ds2_disjoint():
    assert not set(DS1) & set(DS2), "DS1 and DS2 must be disjoint"


def test_ds1_train_val_disjoint():
    assert not set(DS1_TRAIN) & set(DS1_VAL)
    assert set(DS1_TRAIN) | set(DS1_VAL) == set(DS1)


def test_ds2_not_in_ds1():
    for r in DS2:
        assert r not in DS1_TRAIN
        assert r not in DS1_VAL


def test_total_record_count():
    """de Chazal uses 44 records total (22 DS1 + 22 DS2), paced excluded."""
    assert len(DS1) == 22, f"DS1 should have 22 records, got {len(DS1)}"
    assert len(DS2) == 22, f"DS2 should have 22 records, got {len(DS2)}"


def test_ds1_split_sizes():
    """DS1-train ≈ 12, DS1-val ≈ 10 (by convention: <200 → train)."""
    assert 10 <= len(DS1_TRAIN) <= 14
    assert 8  <= len(DS1_VAL)   <= 12


# ============================================================
# 2. AAMI label mapping
# ============================================================

def test_aami_map_covers_beat_symbols():
    """Every known beat symbol maps to a valid AAMI class."""
    for sym in BEAT_SYMBOLS:
        assert sym in AAMI_MAP, f"Symbol '{sym}' missing from AAMI_MAP"
        assert AAMI_MAP[sym] in AAMI_CLASSES, (
            f"Symbol '{sym}' maps to unknown class '{AAMI_MAP[sym]}'"
        )


def test_aami_idx_bijection():
    """AAMI_IDX must be a bijection: each class maps to a unique 0-based int."""
    vals = list(AAMI_IDX.values())
    assert sorted(vals) == list(range(len(AAMI_CLASSES))), (
        "AAMI_IDX values must be 0 … K-1"
    )


def test_aami_labels_range():
    """aami_labels returns integers in [0, 4] for valid beat symbols."""
    syms = list(BEAT_SYMBOLS)
    labels = aami_labels(syms)
    assert labels.dtype in (np.int32, np.int64, np.intp)
    assert labels.min() >= 0
    assert labels.max() <= 4


def test_task_labels_binary():
    """Binary task returns only 0, 1 and -1 (dropped)."""
    syms = np.array(["N", "L", "V", "A", "F"])
    y_aami = aami_labels(syms)
    labs = task_labels(y_aami, syms, task="binary", binary_def="aami")
    valid = labs[labs >= 0]
    assert set(valid.tolist()).issubset({0, 1})
    # V → AAMI V (index 2) → arrhythmia (1 in binary)
    assert labs[2] == 1
    # N → AAMI N (index 0) → normal (0 in binary)
    assert labs[0] == 0


def test_task_labels_3class():
    """3-class task drops F and Q (returns -1 for them)."""
    syms = np.array(["N", "V", "A", "F", "/"])
    y_aami = aami_labels(syms)
    labs = task_labels(y_aami, syms, task="multiclass", classes=3)
    valid = labs[labs >= 0]
    assert set(valid.tolist()).issubset({0, 1, 2})
    # F and Q should be -1
    assert labs[3] == -1   # F → index 3 → dropped for 3-class
    assert labs[4] == -1   # / → Q → index 4 → dropped for 3-class


# ============================================================
# 3. RR features — causality
# ============================================================

def test_causal_mean_no_future():
    """_causal_mean at index i must not use values at indices > i."""
    rr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    mu = _causal_mean(rr, window=3)
    assert len(mu) == len(rr)
    # mu[0] uses only rr[0]
    np.testing.assert_allclose(mu[0], rr[0])
    # mu[2] uses rr[0:3]
    np.testing.assert_allclose(mu[2], np.mean(rr[:3]))
    # mu[4] uses rr[2:5]
    np.testing.assert_allclose(mu[4], np.mean(rr[2:5]))


def test_rr_features_shape():
    """compute_rr_features returns (n, 6) float32 given R-peak sample positions."""
    # 6 beats at ~360 Hz (roughly 1 beat per second)
    r_samples = np.array([360, 720, 1080, 1440, 1800, 2160], dtype=np.int64)
    feats = compute_rr_features(r_samples, fs=360.0, local_window=5, global_window=50)
    assert feats.shape == (6, 6), f"Expected (6, 6), got {feats.shape}"
    assert feats.dtype == np.float32


def test_rr_features_no_future_leakage():
    """
    Features for beat i may depend only on r_samples[0..i+1]: the post-RR
    features (cols 1,3,4,5) deliberately use ONE beat of look-ahead (the
    next R-peak), as documented in rr.py; everything else is causal.
    Appending an anomalous beat must therefore not change features of
    beats 0..n-2, nor the pre-RR-only columns (0, 2) of beat n-1.
    """
    r6 = np.array([360, 720, 1080, 1440, 1800, 2160], dtype=np.int64)
    # anomalous 7th beat very far away
    r7 = np.append(r6, 2160 + 36000)

    f6 = compute_rr_features(r6, fs=360.0, local_window=5, global_window=50)
    f7 = compute_rr_features(r7, fs=360.0, local_window=5, global_window=50)

    # Features for the first 6 beats must be identical
    np.testing.assert_allclose(f6[:5], f7[:5], rtol=1e-5,
                                err_msg="Appending a future beat changed earlier RR features")
    np.testing.assert_allclose(f6[5, [0, 2]], f7[5, [0, 2]], rtol=1e-5,
                                err_msg="pre-RR features of the last beat used look-ahead")


def test_rr_features_first_beat_finite():
    """First beat has no prior RR — features must be finite (not NaN/Inf)."""
    r_samples = np.array([360], dtype=np.int64)
    feats = compute_rr_features(r_samples, fs=360.0, local_window=5, global_window=50)
    assert np.all(np.isfinite(feats)), "RR features contain NaN or Inf for single beat"


# ============================================================
# 4. Window extraction
# ============================================================

def test_beat_window_length():
    """Beat window must have exactly beat_len = before+after samples."""
    from src.preprocess import extract_window
    sig = np.random.randn(1000)
    win, padded = extract_window(sig, r_sample=500, before=90, after=166)
    assert len(win) == 256, f"Expected 256 samples, got {len(win)}"
    assert not padded


def test_beat_window_edge_padded():
    """Near-boundary beats use edge padding (not zeros), so are still finite."""
    from src.preprocess import extract_window
    sig = np.random.randn(200)
    win, padded = extract_window(sig, r_sample=5, before=90, after=166)
    assert len(win) == 256
    assert padded
    assert np.all(np.isfinite(win)), "Padded beat window contains NaN/Inf"


def test_context_window_length():
    """Context window must have exactly context_len samples."""
    from src.preprocess import extract_window
    sig = np.random.randn(2000)
    win, _ = extract_window(sig, r_sample=1000, before=540, after=540)
    assert len(win) == 1080, f"Expected 1080 samples, got {len(win)}"


# ============================================================
# 5. KANLinear
# ============================================================

def test_kan_output_shape():
    """KANLinear(in=8, out=4) on batch produces (batch, 4)."""
    layer = KANLinear(in_features=8, out_features=4, grid_size=5, spline_order=3)
    x = torch.randn(16, 8)
    y = layer(x)
    assert y.shape == (16, 4), f"Expected (16, 4), got {y.shape}"


def test_kan_gradient_flows():
    """Gradients must flow back through the spline branch."""
    layer = KANLinear(in_features=4, out_features=2, grid_size=5, spline_order=3)
    x = torch.randn(8, 4, requires_grad=True)
    y = layer(x).sum()
    y.backward()
    assert x.grad is not None
    assert not torch.all(x.grad == 0), "All gradients are zero — spline branch broken"


def test_kan_spline_weight_grad():
    """Spline coefficients must receive gradients."""
    layer = KANLinear(in_features=4, out_features=2, grid_size=5, spline_order=3)
    x = torch.randn(8, 4)
    y = layer(x).sum()
    y.backward()
    assert layer.spline_weight.grad is not None
    assert not torch.all(layer.spline_weight.grad == 0)


def test_b_spline_partition_of_unity():
    """
    B-spline basis functions sum to 1 for inputs inside the grid range.
    This is a fundamental property: sum_i B_{i,k}(x) = 1 for x in [t_k, t_{n}].
    """
    grid_size = 5
    k = 3
    n_knots = grid_size + 2 * k + 1
    in_features = 1
    # Uniform grid on [-1, 1] with k repeated knots at both ends (clamped)
    inner = torch.linspace(-1.0, 1.0, grid_size + 1)
    left  = inner[0].expand(k)
    right = inner[-1].expand(k)
    grid  = torch.cat([left, inner, right]).unsqueeze(0)  # (1, n_knots)

    # Test points strictly inside the grid
    x = torch.linspace(-0.9, 0.9, 50).unsqueeze(1)  # (50, 1)
    bases = b_splines(x, grid, k)  # (50, 1, grid_size+k)
    sums = bases.squeeze(1).sum(dim=-1)
    np.testing.assert_allclose(sums.numpy(), np.ones(50), atol=1e-5,
                                err_msg="B-spline basis does not sum to 1")


def test_kan_learns_sin():
    """KANLinear(1→1) can approximate sin(x) on [-1,1] with <0.1 MSE after 300 steps."""
    torch.manual_seed(0)
    layer = KANLinear(in_features=1, out_features=1, grid_size=10, spline_order=3)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)

    x = torch.linspace(-1.0, 1.0, 200).unsqueeze(1)
    y = torch.sin(x * torch.pi)

    for _ in range(300):
        opt.zero_grad()
        loss = F.mse_loss(layer(x), y)
        loss.backward()
        opt.step()

    with torch.no_grad():
        mse = F.mse_loss(layer(x), y).item()
    assert mse < 0.1, f"KAN sin(x) fit MSE too high: {mse:.4f} (expected < 0.1)"


# ============================================================
# 6. Focal loss
# ============================================================

def test_focal_gamma0_equals_ce():
    """
    FocalLoss with gamma=0 and uniform weights must equal standard cross-entropy
    (up to floating point).
    """
    torch.manual_seed(42)
    logits = torch.randn(32, 5)
    labels = torch.randint(0, 5, (32,))
    alpha  = torch.ones(5) / 5

    fl  = FocalLoss(gamma=0.0, alpha=alpha)(logits, labels)
    ce  = F.cross_entropy(logits, labels, weight=alpha)

    np.testing.assert_allclose(fl.item(), ce.item(), rtol=1e-4,
                                err_msg="FocalLoss(gamma=0) != CE")


def test_focal_gamma2_smaller_than_gamma0():
    """
    FocalLoss(gamma=2) gives lower loss on well-classified examples because the
    modulating factor (1-p_t)^gamma down-weights easy examples.
    """
    logits = torch.zeros(10, 5)
    logits[torch.arange(10), torch.arange(10) % 5] = 10.0  # strong predictions
    labels = torch.arange(10) % 5
    alpha  = torch.ones(5) / 5

    fl0 = FocalLoss(gamma=0.0, alpha=alpha)(logits, labels).item()
    fl2 = FocalLoss(gamma=2.0, alpha=alpha)(logits, labels).item()
    assert fl2 < fl0, (
        f"FocalLoss(gamma=2) should be smaller than gamma=0 for easy examples "
        f"but got fl0={fl0:.4f}, fl2={fl2:.4f}"
    )


def test_build_loss_focal():
    """build_loss('focal', ...) returns a callable that accepts (logits, labels)."""
    class_counts = np.array([1000, 50, 200, 10, 30])
    loss_fn = build_loss("focal", class_counts=class_counts, gamma=2.0,
                         weight_mode="sqrt")
    logits = torch.randn(16, 5)
    labels = torch.randint(0, 5, (16,))
    loss = loss_fn(logits, labels)
    assert loss.ndim == 0, "Loss must be a scalar"
    assert loss.item() > 0


# ============================================================
# 7. Metrics
# ============================================================

def test_ovr_accuracy_known():
    """
    For K=5 classes, 10 samples, 2 wrong:
      ovr = 1 - 2*2 / (5*10) = 1 - 4/50 = 0.92
    """
    y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
    y_pred = np.array([0, 1, 1, 1, 2, 2, 3, 3, 4, 4])  # 1 wrong
    ovr = ovr_accuracy(y_true, y_pred, k=5)
    expected = 1.0 - 2 * 1 / (5 * 10)
    np.testing.assert_allclose(ovr, expected, atol=1e-8)


def test_macro_f1_all_correct():
    y = np.array([0, 1, 2, 3, 4])
    f1 = macro_f1_from_preds(y, y)
    np.testing.assert_allclose(f1, 1.0, atol=1e-6)


def test_multiclass_metrics_cm_shape():
    """multiclass_metrics returns a 5×5 confusion matrix for 5-class problem."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 3, 1, 1])
    m = multiclass_metrics(y_true, y_pred, k=5)
    cm = np.array(m["cm"])
    assert cm.shape == (5, 5), f"CM shape {cm.shape} != (5, 5)"


def test_multiclass_metrics_per_class_keys():
    """Per-class keys se_N, ppv_V, f1_S etc. must be present."""
    y_true = np.array([0, 1, 2, 3, 4, 0, 1])
    y_pred = np.array([0, 1, 2, 3, 3, 1, 1])
    m = multiclass_metrics(y_true, y_pred, k=5)
    for cls in ["N", "S", "V", "F", "Q"]:
        for prefix in ("se_", "ppv_", "f1_", "n_"):
            key = f"{prefix}{cls}"
            assert key in m, f"Missing key '{key}' in metrics dict"


def test_multiclass_metrics_accuracy_range():
    """Plain accuracy must be in [0, 1]."""
    y = np.random.randint(0, 5, 100)
    p = np.random.randint(0, 5, 100)
    m = multiclass_metrics(y, p, k=5)
    assert 0.0 <= m["accuracy"] <= 1.0


# ============================================================
# 8. Soft labels
# ============================================================

def test_soft_labels_sum_to_one():
    """Soft labels at any temperature must sum to 1 per row."""
    logits = np.random.randn(50, 5).astype(np.float32)
    for T in [1, 2, 4]:
        probs = F.softmax(
            torch.from_numpy(logits) / T, dim=1
        ).numpy()
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(50), atol=1e-5,
                                    err_msg=f"Soft labels at T={T} don't sum to 1")


def test_soft_labels_shape_preserved():
    """Soft labels have the same (N, K) shape as the logits."""
    logits = np.random.randn(123, 5).astype(np.float32)
    for T in [1, 2, 4]:
        probs = F.softmax(
            torch.from_numpy(logits) / T, dim=1
        ).numpy()
        assert probs.shape == logits.shape, (
            f"Soft labels shape {probs.shape} != logits shape {logits.shape}"
        )


def test_soft_labels_higher_T_more_uniform():
    """Higher temperature must produce a softer (more uniform) distribution."""
    logits = np.array([[3.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    p1 = F.softmax(torch.from_numpy(logits) / 1, dim=1).numpy()
    p4 = F.softmax(torch.from_numpy(logits) / 4, dim=1).numpy()
    # Entropy of p4 should be higher than p1
    ent = lambda p: -np.sum(p * np.log(p + 1e-10), axis=1)
    assert ent(p4) > ent(p1), "Higher T should produce higher entropy"


def test_ensemble_logits_shape():
    """ensemble_logits averages correctly across seeds."""
    seeds = [np.random.randn(100, 5).astype(np.float32) for _ in range(3)]
    avg = ensemble_logits(seeds)
    assert avg.shape == (100, 5)
    np.testing.assert_allclose(avg, np.mean(seeds, axis=0), atol=1e-6)


# ============================================================
# 9. Model forward pass (smoke test, CPU only)
# ============================================================

def test_kan_former_forward():
    """KANFormer forward pass produces (batch, 5) logits on CPU."""
    from src.model import KANFormer
    cfg = {
        "preprocess": {"beat_len": 256, "context_len": 1080},
        "model": {
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "ms_kernels": [3, 7, 15],
            "head": "kan",
            "kan_grid_size": 5,
            "dropout": 0.0,
            "stoch_depth": 0.0,
        },
        "rr": {"n_features": 6},
    }
    model = KANFormer(cfg, n_classes=5)
    model.eval()

    beat = torch.randn(2, 1, 256)
    ctx  = torch.randn(2, 1, 1080)
    rr   = torch.randn(2, 6)

    with torch.no_grad():
        logits = model(beat, ctx, rr)

    assert logits.shape == (2, 5), f"Expected (2, 5), got {logits.shape}"
    assert torch.all(torch.isfinite(logits)), "Logits contain NaN/Inf"


def test_kan_former_no_context():
    """KANFormer works when context is None (context_len=0)."""
    from src.model import KANFormer
    cfg = {
        "preprocess": {"beat_len": 256, "context_len": 0},
        "model": {
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "ms_kernels": [3, 7],
            "head": "kan",
            "kan_grid_size": 5,
            "dropout": 0.0,
            "stoch_depth": 0.0,
        },
        "rr": {"n_features": 6},
    }
    model = KANFormer(cfg, n_classes=5)
    model.eval()

    beat = torch.randn(2, 1, 256)
    rr   = torch.randn(2, 6)

    with torch.no_grad():
        logits = model(beat, None, rr)

    assert logits.shape == (2, 5)
    assert torch.all(torch.isfinite(logits))


def test_mlp_head_alternative():
    """KANFormer with head='mlp' produces valid logits (ablation A1)."""
    from src.model import KANFormer
    cfg = {
        "preprocess": {"beat_len": 256, "context_len": 1080},
        "model": {
            "d_model": 32,
            "n_heads": 2,
            "n_layers": 1,
            "ms_kernels": [3, 7],
            "head": "mlp",
            "kan_grid_size": 5,
            "dropout": 0.0,
            "stoch_depth": 0.0,
        },
        "rr": {"n_features": 6},
    }
    model = KANFormer(cfg, n_classes=5)
    model.eval()

    beat = torch.randn(2, 1, 256)
    ctx  = torch.randn(2, 1, 1080)
    rr   = torch.randn(2, 6)

    with torch.no_grad():
        logits = model(beat, ctx, rr)

    assert logits.shape == (2, 5)
    assert torch.all(torch.isfinite(logits))


# ============================================================
# 10. Balanced sampler (regression test for the Q/F domination bug)
# ============================================================

def test_sampler_caps_rare_classes():
    """With DS1-train-like counts no class may exceed its cap; N gets the rest."""
    from src.train import sampler_target_proportions
    y = np.repeat(np.arange(5), [23932, 149, 1234, 13, 2])
    cfg = {"s_cap": 0.30, "v_cap": 0.30, "f_cap": 0.05, "q_cap": 0.01}
    t = sampler_target_proportions(y, 5, cfg)
    np.testing.assert_allclose(t.sum(), 1.0, atol=1e-9)
    assert t[4] <= 0.01 + 1e-9 and t[3] <= 0.05 + 1e-9
    assert t[1] <= 0.30 + 1e-9 and t[2] <= 0.30 + 1e-9
    assert t[0] >= 0.30, f"N share too small: {t}"


def test_kan_former_no_rr():
    """Ablation A4: n_features=0 ignores the RR input entirely."""
    from src.model import KANFormer
    cfg = {"preprocess": {"beat_len": 256, "context_len": 0},
           "model": {"d_model": 32, "n_heads": 2, "n_layers": 1},
           "rr": {"n_features": 0}}
    model = KANFormer(cfg, n_classes=5).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 1, 256), None, torch.randn(2, 6))
    assert out.shape == (2, 5)


# ============================================================
# 11. kanformer2 additions: RR branch and extended RR features
# ============================================================

def test_kan_former_rr_branch():
    """model.rr_branch + rr_aux_logits builds, runs, and RR changes the output."""
    from src.model import KANFormer
    cfg = {"preprocess": {"beat_len": 256, "context_len": 0},
           "model": {"d_model": 32, "n_heads": 2, "n_layers": 1, "head": "mlp",
                     "rr_branch": True, "rr_aux_logits": True},
           "rr": {"n_features": 11}}
    model = KANFormer(cfg, n_classes=5).eval()
    beat = torch.randn(3, 1, 256)
    with torch.no_grad():
        a = model(beat, None, torch.ones(3, 11))
        b = model(beat, None, 2 * torch.ones(3, 11))
    assert a.shape == (3, 5) and not torch.allclose(a, b)


def test_rr_features_ext_causal_and_base_unchanged():
    """The 6 base columns equal compute_rr_features; the 5 extra columns of
    beat i use no R peak beyond i+1 (same one-beat look-ahead as post-RR)."""
    from src.rr import compute_rr_features_ext
    rng = np.random.default_rng(0)
    r = np.cumsum(rng.integers(250, 400, size=600))
    ext = compute_rr_features_ext(r, 360.0)
    assert ext.shape == (600, 11)
    np.testing.assert_allclose(ext[:, :6], compute_rr_features(r, 360.0))
    i = 450
    r2 = r.copy()
    r2[i + 2:] += 500          # change the future after beat i+1
    ext2 = compute_rr_features_ext(r2, 360.0)
    np.testing.assert_allclose(ext[i, 6:], ext2[i, 6:], rtol=1e-6)
