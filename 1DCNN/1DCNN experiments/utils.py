"""
Shared utilities for 1DCNN sensitivity-improvement experiments.
All experiments import from this module to avoid duplicating setup code.
"""

import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXPS    = Path(__file__).resolve().parent
PAPER3  = EXPS.parents[1] / "HasanML" / "paper3"
CACHE   = EXPS / ".cache"


def _ensure_paper3():
    if str(PAPER3) not in sys.path:
        sys.path.insert(0, str(PAPER3))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config() -> dict:
    _ensure_paper3()
    import yaml
    with open(PAPER3 / "configs" / "default.yaml") as fh:
        cfg = yaml.safe_load(fh)
    # Build the same P1 overrides as run_experiment.py
    cfg["input"]    = {**cfg["input"],    "mode": "derivative", "nk": 32}
    cfg["task"]     = {**cfg["task"],     "init": "mf_trainable", "classes": 3}
    cfg["training"] = {**cfg["training"], "class_weight": "balanced"}
    return cfg


# ---------------------------------------------------------------------------
# Data loading with npz cache
# ---------------------------------------------------------------------------

def _build_split(split_name: str, records: list, cfg: dict) -> dict:
    _ensure_paper3()
    from src.run_experiment import build_dataset
    return build_dataset(records, cfg, task="multiclass", classes=3, binary_def="aami")


def load_data(split_name: str, cfg: dict | None = None) -> dict:
    """
    Returns dict {X_ecg, X_rr, y} for 'ds1_train', 'ds1_val', 'ds1_all', or 'ds2'.
    Results are npz-cached under .cache/ so the 30-s preprocessing only runs once.
    """
    _ensure_paper3()
    if cfg is None:
        cfg = load_config()

    cache_path = CACHE / f"{split_name}.npz"
    if cache_path.exists():
        d = np.load(str(cache_path))
        return {"X_ecg": d["X_ecg"], "X_rr": d["X_rr"], "y": d["y"]}

    from src.data import DS1_TRAIN, DS1_VAL, DS2
    record_map = {
        "ds1_train": DS1_TRAIN,
        "ds1_val":   DS1_VAL,
        "ds1_all":   DS1_TRAIN + DS1_VAL,
        "ds2":       DS2,
    }
    if split_name not in record_map:
        raise ValueError(f"Unknown split: {split_name}")

    print(f"[utils] Building {split_name} data (first time — will cache)…")
    d = _build_split(split_name, record_map[split_name], cfg)

    CACHE.mkdir(exist_ok=True)
    np.savez_compressed(str(cache_path), X_ecg=d["X_ecg"], X_rr=d["X_rr"], y=d["y"])
    print(f"[utils] Cached to {cache_path}")
    return {"X_ecg": d["X_ecg"], "X_rr": d["X_rr"], "y": d["y"]}


# ---------------------------------------------------------------------------
# Matched-filter kernel
# ---------------------------------------------------------------------------

def load_kernel_mf(cfg: dict | None = None) -> np.ndarray:
    """
    Compute or load MF kernel from DS1-train+val templates.
    Cached to .cache/kernel_mf.npy.
    """
    _ensure_paper3()
    if cfg is None:
        cfg = load_config()

    cache_path = CACHE / "kernel_mf.npy"
    if cache_path.exists():
        return np.load(str(cache_path))

    from src.data import DS1_TRAIN, DS1_VAL
    from src.run_experiment import build_template_data
    from src.templates import compute_subclass_templates, templates_to_kernel

    print("[utils] Computing MF kernel (first time — will cache)…")
    segs = build_template_data(DS1_TRAIN + DS1_VAL, cfg)
    sub_classes = list(cfg["templates"]["sub_classes"])
    nk = int(cfg["input"]["nk"])
    templates = compute_subclass_templates(
        segs, nk, sub_classes,
        min_beats=int(cfg["templates"]["min_beats"]),
        normalize_l2=bool(cfg["templates"]["normalize_l2"]),
        fallback_segments=segs,
    )
    kernel = templates_to_kernel(templates, sub_classes, n_channels=1)

    CACHE.mkdir(exist_ok=True)
    np.save(str(cache_path), kernel)
    print(f"[utils] Kernel cached to {cache_path}")
    return kernel


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_p1_model(seed: int):
    """Load a saved P1 model from paper3/results/cache/P1_seed{seed}.keras."""
    import tensorflow as tf
    model_path = PAPER3 / "results" / "cache" / f"P1_seed{seed}.keras"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Run P1 training first:\n"
            f"  cd {PAPER3}\n"
            f"  python -m src.run_experiment --p1 --class-weight balanced"
        )
    return tf.keras.models.load_model(str(model_path))


def available_p1_seeds() -> list[int]:
    """Return seeds for which a saved P1 model exists."""
    cache = PAPER3 / "results" / "cache"
    seeds = []
    for p in sorted(cache.glob("P1_seed*.keras")):
        try:
            seeds.append(int(p.stem.split("seed")[1]))
        except (IndexError, ValueError):
            pass
    return seeds


# ---------------------------------------------------------------------------
# Threshold calibration helper
# ---------------------------------------------------------------------------

def threshold_predict(proba: np.ndarray, tau_S: float) -> np.ndarray:
    """
    Multiclass prediction with a boosted threshold for class S (index 1).
    Predict S whenever P(S) >= tau_S (overrides argmax).
    Otherwise argmax over [N, V].
    """
    y_pred = np.argmax(proba, axis=1).copy()
    boost  = proba[:, 1] >= tau_S
    y_pred[boost] = 1
    return y_pred


# ---------------------------------------------------------------------------
# Metrics pretty-print
# ---------------------------------------------------------------------------

HEADER = f"{'Method':<35}  {'Acc%':>6}  {'Se_N%':>6}  {'Se_S%':>6}  {'Se_V%':>6}  {'macF1%':>7}"
SEP    = "-" * len(HEADER)


def print_header():
    print(SEP)
    print(HEADER)
    print(SEP)


def print_row(label: str, m: dict):
    print(
        f"{label:<35}  "
        f"{100*m.get('accuracy', 0):6.2f}  "
        f"{100*m.get('se_N', 0):6.2f}  "
        f"{100*m.get('se_S', 0):6.2f}  "
        f"{100*m.get('se_V', 0):6.2f}  "
        f"{100*m.get('macro_f1', 0):7.3f}"
    )


def print_footer():
    print(SEP)
    print("Paper (Farag23 T4 Model 6, no CW)     98.18   99.10   81.60   95.00   92.170")
    print(SEP)
