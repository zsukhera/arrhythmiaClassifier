"""MIT-BIH loading, AAMI label maps and the three train/test protocols."""

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from sklearn.model_selection import StratifiedKFold

log = logging.getLogger(__name__)

ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
PACED_RECORDS = ["102", "104", "107", "217"]

# de Chazal et al. (2004) inter-patient split.
DS1 = ["101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
       "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230"]
DS2 = ["100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
       "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234"]

CLASSES = ["N", "S", "V", "F", "Q"]

# Paper Table 1 (S includes 'x', V includes '!', Q is 'Q' only) vs ANSI/AAMI EC57.
LABEL_MAPS = {
    "paper": {"N": "NLRej", "S": "AaJx", "V": "V!E", "F": "F", "Q": "Q"},
    "aami": {"N": "NLRej", "S": "AaJS", "V": "VE", "F": "F", "Q": "/fQ"},
}

# Annotations that mark a beat (R peak) and therefore define RR intervals. 'x' (non-conducted
# P wave) and '!' (flutter wave) are not beats in PhysioNet's sense but the paper labels them.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")

# Paper Table 1: class -> (total, train, test).
PAPER_COUNTS = {
    "N": (90083, 45845, 44238),
    "S": (2972, 1000, 1972),
    "V": (7480, 4260, 3220),
    "F": (802, 414, 388),
    "Q": (15, 8, 7),
}
RECORD_EXTENSIONS = ("dat", "hea", "atr")


def symbol_to_class(label_map: str) -> dict[str, int]:
    """MIT-BIH symbol -> class index into CLASSES."""
    return {s: CLASSES.index(c) for c, syms in LABEL_MAPS[label_map].items() for s in syms}


def map_labels(symbols, label_map: str) -> np.ndarray:
    """Class index per symbol, -1 for symbols outside the map."""
    table = symbol_to_class(label_map)
    return np.array([table.get(s, -1) for s in symbols], dtype=np.int64)


@dataclass
class Record:
    name: str
    signal: np.ndarray       # chosen lead, physical units (mV)
    fs: float
    channel: int
    ann_samples: np.ndarray  # every annotation sample
    ann_symbols: np.ndarray  # every annotation symbol


def _has_files(directory: Path, name: str) -> bool:
    return all((directory / f"{name}.{ext}").exists() for ext in RECORD_EXTENSIONS)


def ensure_record(name: str, data_dir: Path, mirrors: list[Path]) -> None:
    """Make `data_dir/<name>.{dat,hea,atr}` exist: local mirror first, then PhysioNet."""
    if _has_files(data_dir, name):
        return
    data_dir.mkdir(parents=True, exist_ok=True)
    for mirror in mirrors:
        if _has_files(mirror, name):
            for ext in RECORD_EXTENSIONS:
                shutil.copy2(mirror / f"{name}.{ext}", data_dir / f"{name}.{ext}")
            log.info("record %s: copied from local mirror %s", name, mirror)
            return
    log.info("record %s: downloading from PhysioNet", name)
    wfdb.dl_database("mitdb", dl_dir=str(data_dir), records=[name], annotators=["atr"])


def load_record(name: str, data_dir: Path, mirrors: list[Path], lead: str = "MLII") -> Record:
    ensure_record(name, data_dir, mirrors)
    path = str(data_dir / name)
    rec = wfdb.rdrecord(path)
    if lead not in rec.sig_name:
        raise ValueError(f"record {name}: lead {lead} not in {rec.sig_name}")
    channel = rec.sig_name.index(lead)
    if channel != 0:
        log.info("record %s: %s is channel %d of %s", name, lead, channel, rec.sig_name)
    ann = wfdb.rdann(path, "atr")
    return Record(name=name, signal=rec.p_signal[:, channel].astype(np.float64), fs=float(rec.fs),
                  channel=channel, ann_samples=np.asarray(ann.sample, dtype=np.int64),
                  ann_symbols=np.asarray(ann.symbol, dtype=str))


def used_records(exclude: list[str]) -> list[str]:
    return [r for r in ALL_RECORDS if r not in set(exclude)]


# --------------------------------------------------------------------------- splits

def split_inter_patient(record: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Protocol A: de Chazal DS1 (train) / DS2 (test)."""
    return np.flatnonzero(np.isin(record, DS1)), np.flatnonzero(np.isin(record, DS2))


def split_intra_patient(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Protocol B: per-class random split whose test fraction equals the paper's Table 1."""
    rng = np.random.default_rng(seed)
    train, test = [], []
    for c, name in enumerate(CLASSES):
        idx = rng.permutation(np.flatnonzero(y == c))
        total, _, n_test = PAPER_COUNTS[name]
        k = int(round(len(idx) * n_test / total))
        test.append(idx[:k])
        train.append(idx[k:])
    return np.sort(np.concatenate(train)), np.sort(np.concatenate(test))


def split_cv(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Protocol C: stratified k-fold over all beats."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def make_splits(protocol: str, y: np.ndarray, record: np.ndarray, seed: int,
                n_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if protocol == "A":
        return [split_inter_patient(record)]
    if protocol == "B":
        return [split_intra_patient(y, seed)]
    if protocol == "C":
        return split_cv(y, n_folds, seed)
    raise ValueError(f"unknown protocol {protocol!r}")
