"""Device selection: scikit-learn on CPU, cuML on CUDA GPUs (LR and SVM only).

Mirrors the LightGBM `_probe_device` in ../../source/train.py: an environment override,
then a tiny real fit on the device, falling back to CPU with the reason printed.
cuML has no LDA or MLP, so those always run on CPU.
"""

import logging
import os

import numpy as np

log = logging.getLogger(__name__)

GPU_MODELS = ("LR", "SVM")


def cuda_available() -> tuple[bool, str]:
    """Return (ok, reason) for whether cuML can fit on a CUDA device."""
    try:
        import cupy
        from cuml.linear_model import LogisticRegression
    except ImportError as exc:
        return False, f"cuML/CuPy not installed ({exc.name}); see README 'GPU'"
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False, "no CUDA devices visible"
        rng = np.random.default_rng(0)
        X = rng.random((200, 5)).astype(np.float32)
        LogisticRegression().fit(X, (X[:, 0] > 0.5).astype(np.int32))
    except Exception as exc:  # driver/runtime problems surface here
        return False, f"{type(exc).__name__}: {str(exc).splitlines()[0] if str(exc) else ''}"
    return True, ""


def resolve_device(requested: str) -> str:
    """Map 'auto' | 'cpu' | 'cuda' | 'gpu' to 'cpu' or 'cuda'.

    The environment variable P2_DEVICE=cpu|cuda|gpu overrides the config / CLI.
    """
    forced = os.environ.get("P2_DEVICE", "").lower()
    req = (forced or requested or "auto").lower()
    if forced:
        log.info("[device] P2_DEVICE override -> %r", forced)
    if req == "gpu":
        req = "cuda"
    if req not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device {requested!r}; use auto, cpu, cuda or gpu")
    if req == "cpu":
        return "cpu"
    ok, reason = cuda_available()
    if ok:
        log.info("[device] CUDA available: LR and SVM run on the GPU (cuML)")
        return "cuda"
    if req == "cuda":
        raise RuntimeError(f"device=cuda requested but unavailable: {reason}")
    log.info("[device] no usable CUDA GPU (%s) -> CPU", reason)
    return "cpu"
