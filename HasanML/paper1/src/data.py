"""MIT-BIH loading: MLII signal + per-beat PVC labels from the `atr` annotations."""

import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb

log = logging.getLogger(__name__)

# Beat annotation symbols (PhysioNet). Everything else ('+', '~', '|', '"',
# 'x', '[', ']', '!', ...) is a non-beat annotation and is ignored.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")
PVC_SYMBOL = "V"
RECORD_EXTENSIONS = ("dat", "hea", "atr")


@dataclass
class Record:
    name: str
    signal: np.ndarray        # MLII, physical units (mV), float64
    fs: float
    channel: int              # index of the lead used
    beat_samples: np.ndarray  # R-peak sample of every beat annotation, int64
    beat_symbols: np.ndarray  # annotation symbol per beat
    labels: np.ndarray        # 1 = PVC ('V'), 0 = any other beat, int8


def extract_beats(samples, symbols) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep beat annotations only; label 'V' as 1 and every other beat as 0."""
    samples = np.asarray(samples, dtype=np.int64)
    symbols = np.asarray(symbols, dtype=str)
    keep = np.isin(symbols, list(BEAT_SYMBOLS))
    symbols = symbols[keep]
    return samples[keep], symbols, (symbols == PVC_SYMBOL).astype(np.int8)


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
    ann = wfdb.rdann(path, "atr")
    samples, symbols, labels = extract_beats(ann.sample, ann.symbol)
    log.info("record %s: using channel %d (%s) of %s", name, channel, lead, rec.sig_name)
    counts = Counter(symbols.tolist())
    log.info("record %s: beats per symbol (whole record) %s | PVC=%d other=%d",
             name, dict(sorted(counts.items())), int(labels.sum()), int((labels == 0).sum()))
    return Record(
        name=name,
        signal=rec.p_signal[:, channel].astype(np.float64),
        fs=float(rec.fs),
        channel=channel,
        beat_samples=samples,
        beat_symbols=symbols,
        labels=labels,
    )
