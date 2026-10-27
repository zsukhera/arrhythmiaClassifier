"""The 57 morphology / RR features of paper Table 2, per record and for the whole database."""

import hashlib
import json
import logging
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .data import BEAT_SYMBOLS, LABEL_MAPS, load_record, map_labels
from .fiducials import POINTS, detect
from .preprocess import baseline_sos, beat_windows, remove_baseline

log = logging.getLogger(__name__)

FEATURE_NAMES = [
    # 1-18: durations (ms)
    "P width", "PR", "P1R", "P2R", "PR interval", "PQ", "PS interval", "PT", "QR", "QS",
    "QT interval", "QRS interval", "RS", "RT1", "RT", "RT2", "ST interval", "T width",
    # 19-25: amplitudes (mV)
    "P amplitude", "PR amplitude", "QR amplitude", "R amplitude", "RS amplitude",
    "T amplitude", "RT amplitude",
    # 26-27: areas (mV*s); 28-30: angles (deg)
    "QR area", "RS area", "angle Q1-Q-R", "angle Q-R-S", "angle R-S-S2",
    # 31-35: RR (ms) and R time (s)
    "Pre-RR", "Post-RR", "RR average", "RR local average", "IR time information",
    # 36-47: fiducial amplitudes (mV)
    *[f"amp {p}" for p in POINTS], "amp baseline",
]
# 48-57: intervals (ms) between the 10 pairs of consecutive fiducial points
_CONSECUTIVE = list(zip(POINTS[:-1], POINTS[1:]))
FEATURE_NAMES += [f"int {a}-{b}" for a, b in _CONSECUTIVE]
assert len(FEATURE_NAMES) == 57, len(FEATURE_NAMES)
FEATURE35 = FEATURE_NAMES.index("IR time information")

# Bump when feature code changes, to invalidate cached feature files.
FEATURE_VERSION = 1


def rr_features(r: np.ndarray, fs: float, local_n: int) -> dict[str, np.ndarray]:
    """Pre-RR, Post-RR, record RR average and local RR average (ms) for every beat in `r`.

    The first beat's Pre-RR is its Post-RR and the last beat's Post-RR is its Pre-RR.
    The local average is the mean of the `local_n` RR intervals before the beat and the
    `local_n` after it (fewer at the record edges).
    """
    r = np.asarray(r, dtype=np.float64)
    rr = np.diff(r) * 1000.0 / fs           # rr[i] = R[i+1] - R[i]
    m = len(r)
    pre = np.empty(m)
    post = np.empty(m)
    pre[1:], post[:-1] = rr, rr
    pre[0], post[-1] = rr[0], rr[-1]
    i = np.arange(m)
    lo = np.clip(i - local_n, 0, len(rr))   # intervals rr[lo:hi]: 5 before (rr[i-5..i-1]) +
    hi = np.clip(i + local_n, 0, len(rr))   # 5 after (rr[i..i+4])
    cs = np.concatenate([[0.0], np.cumsum(rr)])
    local = (cs[hi] - cs[lo]) / np.maximum(hi - lo, 1)
    return {"pre": pre, "post": post, "avg": np.full(m, rr.mean()), "local": local}


def _angle(beats, rows, a, b, c, fs) -> np.ndarray:
    """Angle (deg) at vertex b between b->a and b->c in the (time s, amplitude mV) plane."""
    ux, uy = (a - b) / fs, beats[rows, a] - beats[rows, b]
    vx, vy = (c - b) / fs, beats[rows, c] - beats[rows, b]
    return np.degrees(np.arctan2(np.abs(ux * vy - uy * vx), ux * vx + uy * vy))


def morphology_features(beats: np.ndarray, pts: dict, fs: float) -> np.ndarray:
    """Features 1-30 and 36-57 (RR / R-time columns 31-35 are left as NaN)."""
    n = len(beats)
    rows = np.arange(n)
    P1, P, P2, Q1, Q, R, S, S2, T1, T, T2 = (pts[k] for k in POINTS)
    dt = lambda a, b: (b - a) * 1000.0 / fs           # ms from a to b
    amp = lambda k: beats[rows, k]
    cs = np.cumsum(beats, axis=1)
    area = lambda a, b: (cs[rows, b] - cs[rows, a] + beats[rows, a]) / fs   # sum over [a, b], mV*s

    cols = [
        dt(P1, P2), dt(P, R), dt(P1, R), dt(P2, R), dt(P1, Q1), dt(P, Q), dt(P, S), dt(P, T),
        dt(Q, R), dt(Q, S), dt(Q1, T2), dt(Q1, S2), dt(R, S), dt(R, T1), dt(R, T), dt(R, T2),
        dt(S2, T2), dt(T1, T2),
        amp(P), amp(R) - amp(P), amp(R) - amp(Q), amp(R), amp(R) - amp(S), amp(T), amp(R) - amp(T),
        area(Q, R), area(R, S),
        _angle(beats, rows, Q1, Q, R, fs), _angle(beats, rows, Q, R, S, fs),
        _angle(beats, rows, R, S, S2, fs),
        *[np.full(n, np.nan)] * 5,
        *[amp(pts[k]) for k in POINTS], beats[:, :5].mean(axis=1),
        *[dt(pts[a], pts[b]) for a, b in _CONSECUTIVE],
    ]
    return np.column_stack(cols)


@dataclass
class RecordFeatures:
    name: str
    X: np.ndarray            # (n, 57)
    y: np.ndarray            # class index into CLASSES
    symbol: np.ndarray
    sample: np.ndarray       # annotation sample of the R peak
    points: np.ndarray       # (n, 11) fiducial columns in the beat window, order POINTS
    drops: list = field(default_factory=list)   # (record, sample, symbol, reason)
    t_clipped: int = 0


def record_features(name: str, cfg: dict, project_root: Path, return_signal: bool = False):
    d, pp, fc = cfg["data"], cfg["preprocess"], cfg["features"]
    root = lambda p: Path(p) if Path(p).is_absolute() else project_root / p
    rec = load_record(name, root(d["data_dir"]), [root(m) for m in d.get("local_mirrors", [])], d["lead"])
    fs = rec.fs
    b = pp["baseline"]
    x, base = remove_baseline(rec.signal, baseline_sos(b["order"], b["cutoff_hz"], b["rp_db"],
                                                       b["rs_db"], fs), b["zero_phase"])

    labelled = set("".join(LABEL_MAPS[d["label_map"]].values()))
    in_rr = np.isin(rec.ann_symbols, list(BEAT_SYMBOLS | labelled))
    r_all, sym_all = rec.ann_samples[in_rr], rec.ann_symbols[in_rr]
    rr = rr_features(r_all, fs, fc["rr_local_n"])

    y_all = map_labels(sym_all, d["label_map"])
    drops = []
    unmapped = Counter(sym_all[y_all < 0].tolist())
    for s, c in sorted(unmapped.items()):
        drops.append((name, -1, s, f"symbol not in the '{d['label_map']}' label map (x{c})"))
    lab = y_all >= 0
    beats, keep = beat_windows(x, r_all[lab], pp["window_before"], pp["window_after"])
    for smp, s in zip(r_all[lab][~keep], sym_all[lab][~keep]):
        drops.append((name, int(smp), s, "256-sample window runs past the record edge"))

    sel = np.flatnonzero(lab)[keep]
    pts = detect(beats, fs, pp["window_before"], cfg["fiducials"])
    X = morphology_features(beats, pts, fs)
    X[:, 30], X[:, 31], X[:, 32], X[:, 33] = rr["pre"][sel], rr["post"][sel], rr["avg"][sel], rr["local"][sel]
    if fc["feature35"] != "r_time_s":
        raise ValueError(f"unknown feature35 option {fc['feature35']!r}")
    X[:, FEATURE35] = r_all[sel] / fs
    out = RecordFeatures(name=name, X=X, y=y_all[sel], symbol=sym_all[sel], sample=r_all[sel],
                         points=np.column_stack([pts[k] for k in POINTS]), drops=drops,
                         t_clipped=pts["t_clipped"])
    if return_signal:
        return out, rec.signal, base, x, beats
    return out


def feature_cache_key(cfg: dict) -> str:
    keys = {k: cfg[k] for k in ("preprocess", "fiducials", "features")}
    keys["data"] = {k: cfg["data"][k] for k in ("lead", "label_map")}
    keys["version"] = FEATURE_VERSION
    return hashlib.sha1(json.dumps(keys, sort_keys=True).encode()).hexdigest()[:12]


def _cached_record(name: str, cfg: dict, project_root: Path, cache_dir: Path) -> RecordFeatures:
    path = cache_dir / f"{name}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        drops = [tuple(json.loads(s)) for s in z["drops"]]
        return RecordFeatures(name=name, X=z["X"], y=z["y"], symbol=z["symbol"], sample=z["sample"],
                              points=z["points"], drops=drops, t_clipped=int(z["t_clipped"]))
    rf = record_features(name, cfg, project_root)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, X=rf.X, y=rf.y, symbol=rf.symbol, sample=rf.sample, points=rf.points,
             drops=np.array([json.dumps(d) for d in rf.drops], dtype=str), t_clipped=rf.t_clipped)
    tmp.replace(path)
    return rf


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    record: np.ndarray
    symbol: np.ndarray
    sample: np.ndarray
    points: np.ndarray
    drops: list
    t_clipped: int


def build_dataset(records: list[str], cfg: dict, project_root: Path, cache_root: Path,
                  workers: int = 1) -> Dataset:
    """Features for every record, one process per record (cached per config hash)."""
    cache_dir = cache_root / f"features_{feature_cache_key(cfg)}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            parts = list(ex.map(_cached_record, records, [cfg] * len(records),
                                [project_root] * len(records), [cache_dir] * len(records)))
    else:
        parts = [_cached_record(r, cfg, project_root, cache_dir) for r in records]
    return Dataset(
        X=np.concatenate([p.X for p in parts]),
        y=np.concatenate([p.y for p in parts]),
        record=np.concatenate([np.full(len(p.y), p.name) for p in parts]),
        symbol=np.concatenate([p.symbol for p in parts]),
        sample=np.concatenate([p.sample for p in parts]),
        points=np.concatenate([p.points for p in parts]),
        drops=[d for p in parts for d in p.drops],
        t_clipped=sum(p.t_clipped for p in parts),
    )
