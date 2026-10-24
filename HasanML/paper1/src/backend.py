"""Array backend selection: NumPy on CPU, CuPy on CUDA GPUs.

The reservoir and readout code is written against the NumPy API only, so the
same functions run on either backend (`xp` is `numpy` or `cupy`).
"""

import numpy as np


def cuda_available() -> tuple[bool, str]:
    """Return (ok, reason) for whether CuPy can use a CUDA device."""
    try:
        import cupy
    except ImportError:
        return False, "CuPy is not installed (pip install cupy-cuda12x)"
    try:
        if cupy.cuda.runtime.getDeviceCount() < 1:
            return False, "no CUDA devices visible"
        cupy.zeros(1).sum()
    except Exception as exc:  # driver/runtime problems surface here
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def resolve_device(requested: str) -> str:
    """Map 'auto' | 'cpu' | 'cuda' | 'gpu' to 'cpu' or 'cuda'."""
    req = (requested or "auto").lower()
    if req == "gpu":
        req = "cuda"
    if req not in ("auto", "cpu", "cuda"):
        raise ValueError(f"unknown device {requested!r}; use auto, cpu, cuda or gpu")
    if req == "cpu":
        return "cpu"
    ok, reason = cuda_available()
    if ok:
        return "cuda"
    if req == "cuda":
        raise RuntimeError(f"device=cuda requested but unavailable: {reason}")
    return "cpu"


def get_xp(device: str):
    if device == "cuda":
        import cupy
        return cupy
    return np


def to_host(a) -> np.ndarray:
    return a if isinstance(a, np.ndarray) else a.get()
