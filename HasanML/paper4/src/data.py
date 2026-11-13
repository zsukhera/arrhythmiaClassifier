# %% [markdown]
# ## data.py — MIT-BIH loading, AAMI label maps, de Chazal DS1/DS2 split
#
# References:
#   de Chazal et al. (2004), "Automatic classification of heartbeats using
#   ECG morphology and heartbeat interval features," IEEE TBME 51(7):1196-1206.
#   Farag (2023), Sensors 23(3):1365 — identical split used so both models
#   are evaluated on the same patients.
#
# Split: DS1 (train patients) and DS2 (test patients) are fixed by de Chazal.
# DS1 is further divided patient-wise at record < 200 boundary (Farag 2023):
#   DS1-train: 12 records < 200  — model weights, focal loss weights
#   DS1-val:   10 records >= 200 — early stopping, calibration, threshold tuning
#   DS2:       22 records        — NEVER touched until frozen configuration
#
# Patient-wise validation is ESSENTIAL for inter-patient evaluation: a random
# beat split across DS1 would let the model see the same patient's morphology
# during training and validation, making early stopping over-optimistic and
# inflating DS2 numbers.

# %%
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import wfdb

log = logging.getLogger(__name__)

# %%
# Full MIT-BIH record list (48 records)
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Records with pacemaker artifacts produce large spikes that corrupt morphology
# features and bias the classifier.  Excluded from both DS1 and DS2.
PACED_RECORDS = ["102", "104", "107", "217"]

# de Chazal (2004) DS1/DS2 split — fixed for all inter-patient benchmarks.
DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "203", "205", "207", "208", "209", "215", "220",
    "223", "230",
]
DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202",
    "210", "212", "213", "214", "219", "221", "222", "228", "231", "232",
    "233", "234",
]

# Patient-wise validation split at record number < 200 (Farag 2023, Sec 5.3).
# Records 101-124 are physically different patients from 201-230 in DS1.
DS1_TRAIN = [r for r in DS1 if int(r) < 200]   # 12 records
DS1_VAL   = [r for r in DS1 if int(r) >= 200]  # 10 records

assert len(DS1_TRAIN) == 12, f"Expected 12 DS1-train records, got {len(DS1_TRAIN)}"
assert len(DS1_VAL)   == 10, f"Expected 10 DS1-val records, got {len(DS1_VAL)}"
assert len(DS2)       == 22, f"Expected 22 DS2 records, got {len(DS2)}"
assert not set(DS1_TRAIN) & set(DS1_VAL),  "DS1-train / DS1-val patient overlap"
assert not set(DS1_TRAIN) & set(DS2),      "DS1-train / DS2 patient overlap"
assert not set(DS1_VAL)   & set(DS2),      "DS1-val / DS2 patient overlap"

# %%
# ANSI/AAMI EC57 label mapping.
# Maps each MIT-BIH beat symbol -> AAMI super-class name.
# Beats not listed here are non-beat annotations (noise, rhythm changes, etc.)
# and are dropped from the classification task.
AAMI_MAP = {
    # Normal (N): sinus beats, left/right bundle branch block, escape beats
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic (S/SVEB): premature atrial, junctional
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic (V/VEB): PVC, ventricular escape
    "V": "V", "E": "V",
    # Fusion of normal and ventricular (F)
    "F": "F",
    # Unknown / paced / other (Q): paced beats, fusions of paced
    "/": "Q", "f": "Q", "Q": "Q",
}

# Integer encoding for AAMI classes (5-class is the DEFAULT)
AAMI_IDX = {"N": 0, "S": 1, "V": 2, "F": 3, "Q": 4}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# All MIT-BIH beat symbols (non-beat annotations excluded)
BEAT_SYMBOLS = set(AAMI_MAP.keys())


# %%
@dataclass
class RawRecord:
    """Loaded MIT-BIH record with signal and beat annotations."""
    name:        str
    signal:      np.ndarray   # (n_samples, n_channels) — full lead II signal
    fs:          float        # sample rate (360 Hz)
    ann_samples: np.ndarray   # (n_beats,) — R-peak positions of ALL beat annotations
    ann_symbols: np.ndarray   # (n_beats,) — symbol for each beat
    lead_idx:    int          # index of MLII channel (usually 0, record 114 swapped)


def _has_files(data_dir: Path, name: str) -> bool:
    return all((data_dir / f"{name}.{ext}").exists() for ext in ("dat", "hea", "atr"))


def ensure_record(name: str, data_dir: Path, mirrors: List[Path]) -> None:
    """Copy from local mirror or download from PhysioNet if not cached."""
    if _has_files(data_dir, name):
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    for mirror in mirrors:
        if _has_files(mirror, name):
            for ext in ("dat", "hea", "atr"):
                shutil.copy2(mirror / f"{name}.{ext}", data_dir / f"{name}.{ext}")
            log.info("record %s: copied from %s", name, mirror)
            return
    log.info("record %s: downloading from PhysioNet mitdb...", name)
    wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[name], annotators=["atr"])


def load_record(name: str, data_dir: Path, mirrors: List[Path],
                lead: str = "MLII") -> RawRecord:
    """Load a MIT-BIH record — MLII signal plus ALL beat annotations."""
    ensure_record(name, data_dir, mirrors)
    path = str(data_dir / name)
    rec  = wfdb.rdrecord(path)
    ann  = wfdb.rdann(path, "atr")

    # Select MLII by name because record 114 has its leads swapped.
    sig_names = [s.strip() for s in rec.sig_name]
    if lead in sig_names:
        lead_idx = sig_names.index(lead)
    else:
        lead_idx = 0
        log.warning("record %s: lead '%s' not found in %s; using index 0", name, lead, sig_names)

    signal = rec.p_signal[:, lead_idx]   # (n_samples,) float64 mV

    # Keep only beat annotations (drop rhythm markers, noise markers, etc.)
    beat_mask = np.array([str(sym) in BEAT_SYMBOLS for sym in ann.symbol])
    ann_samples = ann.sample[beat_mask]
    ann_symbols = np.array([str(sym) for sym in ann.symbol])[beat_mask]

    return RawRecord(
        name=name,
        signal=signal,
        fs=float(rec.fs),
        ann_samples=ann_samples,
        ann_symbols=ann_symbols,
        lead_idx=lead_idx,
    )


# %%
def aami_labels(symbols: np.ndarray) -> np.ndarray:
    """
    Convert MIT-BIH beat symbols -> AAMI integer labels.
    Returns -1 for symbols not in AAMI_MAP (should not happen after beat_mask filter).
    """
    out = np.full(len(symbols), -1, dtype=np.int64)
    for i, sym in enumerate(symbols):
        aami_cls = AAMI_MAP.get(str(sym))
        if aami_cls is not None:
            out[i] = AAMI_IDX[aami_cls]
    return out


def task_labels(y_aami: np.ndarray, symbols: np.ndarray,
                task: str = "multiclass", classes: int = 5,
                binary_def: str = "aami") -> np.ndarray:
    """
    Map 5-class AAMI labels to the task-specific label space.
    Returns -1 for beats that should be dropped (not in the active classes).

    task="multiclass":
      classes=5: N,S,V,F,Q → 0,1,2,3,4
      classes=4: N,S,V,F   → 0,1,2,3; Q → -1 (Q has ~15 beats in DS2)
      classes=3: N,S,V     → 0,1,2;   F,Q → -1 (for Farag-compatible comparison)

    task="binary":
      binary_def="aami": Normal=N(0), Arrhythmia=S,V,F,Q → 1
      binary_def="raw":  Normal=MIT-BIH 'N' only → 0, everything else → 1
    """
    if task == "multiclass":
        keep = list(range(classes))   # 0..classes-1 are always kept
        out = np.where(np.isin(y_aami, keep), y_aami, -1)
        return out.astype(np.int64)

    # binary
    if binary_def == "aami":
        # Normal = AAMI N (index 0); Arrhythmia = everything else with valid AAMI label
        valid = y_aami >= 0
        out = np.where(valid, (y_aami > 0).astype(np.int64), np.int64(-1))
        return out
    else:  # raw
        # Normal = MIT-BIH symbol 'N' exactly; everything else (including L,R,e,j) is anomaly
        is_n = np.array([str(s) == "N" for s in symbols])
        valid = y_aami >= 0
        out = np.where(valid, (~is_n).astype(np.int64), np.int64(-1))
        return out


# %%
def beat_mask_for_task(y_aami: np.ndarray, symbols: np.ndarray,
                       task: str, classes: int, binary_def: str) -> np.ndarray:
    """Return boolean mask of beats that have a valid label for this task."""
    yt = task_labels(y_aami, symbols, task, classes, binary_def)
    return yt >= 0


def print_beat_counts(split_labels: dict, task: str = "multiclass",
                      classes: int = 5) -> None:
    """Print a formatted beat-count table per split."""
    cls_names = AAMI_CLASSES[:classes] if task == "multiclass" else ["N", "Arr"]
    header = f"{'Split':<12}" + "".join(f"{c:>8}" for c in cls_names) + f"{'Total':>8}"
    print("\nBeat counts:")
    print(header)
    print("-" * len(header))
    for split_name, y in sorted(split_labels.items()):
        n_cls = classes if task == "multiclass" else 2
        counts = [int(np.sum(y == i)) for i in range(n_cls)]
        row = f"{split_name:<12}" + "".join(f"{c:>8,}" for c in counts) + f"{sum(counts):>8,}"
        print(row)
    print()


def get_records_for_split(split: str) -> List[str]:
    if split == "ds1_train": return DS1_TRAIN
    if split == "ds1_val":   return DS1_VAL
    if split == "ds1":       return DS1
    if split == "ds2":       return DS2
    raise ValueError(f"Unknown split: {split}")
