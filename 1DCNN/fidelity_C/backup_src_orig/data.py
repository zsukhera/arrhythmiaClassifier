# %% [markdown]
# ## data.py — MIT-BIH loading, AAMI label maps, de Chazal DS1/DS2 split
#
# Reference: M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient
# ECG Classification and Arrhythmia Detection at the Edge," Sensors 23(3):1365,
# 2023, doi: 10.3390/s23031365.
#
# Split follows de Chazal et al. (2004), confirmed against paper Table 2.
# DS1 is further split at record number 200 (paper Sec 5.3):
#   DS1-train = DS1 records < 200  (12 records) — used for templates, weights, stats
#   DS1-val   = DS1 records >= 200 (10 records) — used for early stopping / threshold
#   DS2                           (23 records) — NEVER touched until final eval

# %%
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

log = logging.getLogger(__name__)

# %%
# All 48 MIT-BIH Arrhythmia Database records (paced records NOT excluded here;
# they are excluded when building the dataset)
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]

# Records with pacemaker artifacts; excluded from both DS1 and DS2 per de Chazal.
PACED_RECORDS = ["102", "104", "107", "217"]

# de Chazal (2004) inter-patient split.  Paced records fall outside DS1/DS2.
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

# Paper Sec 5.3: "Record numbers less than 200 are used for model training,
# while the remaining are used for model validation."
DS1_TRAIN = [r for r in DS1 if int(r) < 200]   # 12 records; templates, weights
DS1_VAL   = [r for r in DS1 if int(r) >= 200]  # 10 records; early stopping, thresholds

assert len(DS1_TRAIN) == 12, f"Expected 12 DS1-train records, got {len(DS1_TRAIN)}"
assert len(DS1_VAL)   == 10, f"Expected 10 DS1-val records,   got {len(DS1_VAL)}"
assert len(DS2)       == 22, f"Expected 22 DS2 records,       got {len(DS2)}"
# No overlap between any of the three sets
assert not set(DS1_TRAIN) & set(DS1_VAL)
assert not set(DS1_TRAIN) & set(DS2)
assert not set(DS1_VAL)   & set(DS2)

# %%
# ANSI/AAMI EC57 label mapping (paper Table 1).
# Maps MIT-BIH annotation symbol -> AAMI super-class name.
# Beats not in any class map to None (and are dropped).
AAMI_MAP = {
    # Normal
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # Supraventricular ectopic (SVEB)
    "A": "S", "a": "S", "J": "S", "S": "S",
    # Ventricular ectopic (VEB)
    "V": "V", "E": "V",
    # Fusion
    "F": "F",
    # Unknown / paced / other
    "/": "Q", "f": "Q", "Q": "Q",
}

# The 5 AAMI super-classes
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# Which MIT-BIH symbols belong to each sub-class (for template computation).
# 13 sub-classes total (paper Sec 4.2, Figure 4) [assumption].
SUB_CLASSES = ["N", "L", "R", "e", "j", "A", "a", "J", "S", "V", "E", "F", "Q"]

# All annotation symbols that represent an actual beat (R-peak exists).
# Non-beat markers (rhythm annotations, noise labels, etc.) are excluded.
# These symbols define RR intervals; we keep them even if the beat is dropped
# from classification so that RR intervals are never computed across a gap.
BEAT_SYMBOLS = frozenset(
    "NLRBAaJSVrFejnE/fQ?"   # from wfdb PhysioNet documentation
)

RECORD_EXTENSIONS = ("dat", "hea", "atr")

# %%
# Paper Table 2 beat counts (DS1-train / DS1-val / DS2).
# Used for comparison only; actual counts are printed by the pipeline.
# ASSUMPTION: N/A — these come directly from the paper's table image.
# Totals confirmed against de Chazal (2004).
PAPER_TABLE2 = {
    # class: (DS1_train, DS1_val, DS2)
    "N": (None, None, None),  # fill from paper Table 2 image
    "S": (None, None, None),
    "V": (None, None, None),
    "F": (None, None, None),
    "Q": (None, None, None),
}


# %%
@dataclass
class RawRecord:
    """Single MIT-BIH record: signal for the chosen lead + annotations."""
    name: str
    signal: np.ndarray      # shape (n_samples,), physical units mV
    fs: float               # sample rate (360 Hz for MIT-BIH)
    channel: int            # which channel was used
    ann_samples: np.ndarray # sample index of every annotation
    ann_symbols: np.ndarray # annotation symbol strings


def _has_files(directory: Path, name: str) -> bool:
    return all((directory / f"{name}.{ext}").exists() for ext in RECORD_EXTENSIONS)


def ensure_record(name: str, data_dir: Path, mirrors: list) -> None:
    """Ensure data_dir/<name>.{dat,hea,atr} exist: local mirror first, then PhysioNet."""
    if _has_files(data_dir, name):
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    for mirror in mirrors:
        mirror = Path(mirror)
        if _has_files(mirror, name):
            for ext in RECORD_EXTENSIONS:
                shutil.copy2(mirror / f"{name}.{ext}", data_dir / f"{name}.{ext}")
            log.info("record %s: copied from local mirror %s", name, mirror)
            return
    log.info("record %s: downloading from PhysioNet mitdb", name)
    wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[name], annotators=["atr"])


def load_record(name: str, data_dir: Path, mirrors: list, lead: str = "MLII") -> RawRecord:
    """Load one MIT-BIH record and its beat annotations."""
    ensure_record(name, data_dir, mirrors)
    path = str(data_dir / name)
    rec = wfdb.rdrecord(path)
    if lead not in rec.sig_name:
        raise ValueError(f"record {name}: lead '{lead}' not in {rec.sig_name}")
    channel = rec.sig_name.index(lead)
    if channel != 0:
        log.info("record %s: '%s' is channel %d (leads: %s)", name, lead, channel, rec.sig_name)
    ann = wfdb.rdann(path, "atr")
    return RawRecord(
        name=name,
        signal=rec.p_signal[:, channel].astype(np.float64),
        fs=float(rec.fs),
        channel=channel,
        ann_samples=np.asarray(ann.sample, dtype=np.int64),
        ann_symbols=np.asarray(ann.symbol, dtype=object),
    )


# %%
def beat_mask(symbols: np.ndarray) -> np.ndarray:
    """Boolean mask: True where the annotation symbol is an actual beat (has an R-peak)."""
    return np.array([str(s) in BEAT_SYMBOLS for s in symbols], dtype=bool)


def aami_labels(symbols: np.ndarray) -> np.ndarray:
    """
    Map MIT-BIH symbols to AAMI class indices (N=0, S=1, V=2, F=3, Q=4).
    Beats not in AAMI_MAP (e.g. pacemaker 'P') get index -1.
    """
    return np.array(
        [AAMI_CLASSES.index(AAMI_MAP[str(s)]) if str(s) in AAMI_MAP else -1 for s in symbols],
        dtype=np.int64,
    )


def task_labels(aami: np.ndarray, symbols: np.ndarray, task: str, classes: int,
                binary_def: str) -> np.ndarray:
    """
    Convert 5-class AAMI labels to task-specific labels.
    Returns index array; beats dropped for the task get -1.

    For multiclass-3: keep N(0), S(1), V(2); drop F and Q.
    For multiclass-4: keep N(0), S(1), V(2), F(3); drop Q.
    For multiclass-5: keep all 5.
    For binary/aami:  N=0 (AAMI N), Arrhythmia=1 (S,V,F,Q).
    For binary/raw:   N=0 (MIT-BIH symbol 'N' only), Arrhythmia=1 (everything else).
    """
    if task == "multiclass":
        keep_classes = classes  # keep 0..keep_classes-1
        y = np.where(aami < keep_classes, aami, -1)
        return y
    # binary
    if binary_def == "aami":
        # Normal = AAMI N (index 0); Arrhythmia = any other AAMI class with valid label
        y = np.where(aami == 0, 0, np.where(aami > 0, 1, -1))
    elif binary_def == "raw":
        # Normal = MIT-BIH symbol 'N' exactly; Arrhythmia = any other beat symbol
        sym_arr = np.array([str(s) for s in symbols])
        # Only beats that are in BEAT_SYMBOLS get a label
        is_beat = np.array([s in BEAT_SYMBOLS for s in sym_arr], dtype=bool)
        y = np.where(~is_beat, -1, np.where(sym_arr == "N", 0, 1))
    else:
        raise ValueError(f"Unknown binary_def: {binary_def!r}")
    return y.astype(np.int64)


# %%
def get_records_for_split(split: str) -> list:
    """Return record list for a named split."""
    return {"ds1_train": DS1_TRAIN, "ds1_val": DS1_VAL, "ds1": DS1, "ds2": DS2}[split]


def print_beat_counts(datasets: dict, task: str = "multiclass", classes: int = 3) -> None:
    """
    Print a table of beat counts per class for each split.
    datasets: {'ds1_train': y_array, 'ds1_val': y_array, 'ds2': y_array}
    """
    n_classes = classes if task == "multiclass" else 2
    class_names = AAMI_CLASSES[:n_classes] if task == "multiclass" else ["N", "Arr"]
    header = f"{'Class':<8}" + "".join(f"{s:>14}" for s in datasets)
    print("\n" + header)
    print("-" * len(header))
    for i, name in enumerate(class_names):
        row = f"{name:<8}"
        for y in datasets.values():
            cnt = int((y == i).sum())
            row += f"{cnt:>14,}"
        print(row)
    print("-" * len(header))
    total_row = f"{'Total':<8}"
    for y in datasets.values():
        cnt = int((y >= 0).sum())
        total_row += f"{cnt:>14,}"
    print(total_row + "\n")
