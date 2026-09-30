# %% [markdown]

# # MIT-BIH — Steps 1–4: Load · Filter · Segment
#
# **Arrhythmia Classification — CLARITY-AI 2.0 Re-implementation**
# Zain Ul Arifeen Sukhera · 2026–27
#
# Pipeline steps implemented here (from clarityBase.txt §10):
#
# 1. **LOAD**    — read every MIT-BIH record with `wfdb`; pick channels by name
# 2. **FILTER**  — 4th-order Butterworth band-pass 0.5–40 Hz; zero-phase offline
# 3. **R-PEAKS** — use cardiologist `.atr` annotation positions as R-peak ground truth
# 4. **SEGMENT** — 256-sample window per beat (100 pre, 156 post); keep N/L/R/V/A only
#
# Output: `datasets/processed/mit_bih_segments.npz`
#   segments   (n_beats, 2, 256) float32  — channels-first; channel 0 = MLII
#   labels     (n_beats,)        <U1      — beat symbol
#   record_ids (n_beats,)        str      — source record name
#   r_samples  (n_beats,)        int32    — R-peak position within record (samples)

# %%

import numpy as np
import wfdb
import scipy.signal as sps
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# %%

# ── Paths ─────────────────────────────────────────────────────────────────────
# Anchor to the script's own location so the paths resolve correctly regardless
# of which directory the script is run from (project root, source/, or Colab).
_HERE      = Path(__file__).resolve().parent   # …/fyp/source/
DATA_DIR   = _HERE.parent / "datasets" / "mitbih" / "mitdb"
OUTPUT_DIR = _HERE.parent / "datasets" / "processed"

# ── Signal constants ──────────────────────────────────────────────────────────
FS = 360   # MIT-BIH sampling frequency (Hz) — digitised at 360 samples/second/channel

# Beat segmentation window — 256 samples at 360 Hz ≈ 711 ms total.
# The asymmetry (100 pre, 156 post) captures the P-wave before the R-peak and
# the full T-wave after it, which together span almost all diagnostic waveform features.
# clarityBase.txt §4 specifies this window; SKILL.md notes a 720-sample (2 s) window
# as the planned alternative for deep-learning models — change WIN_PRE/WIN_POST there.
WIN_PRE  = 100   # samples before R-peak
WIN_POST = 156   # samples after R-peak
WIN_LEN  = WIN_PRE + WIN_POST   # 256

# ── Class filter ──────────────────────────────────────────────────────────────
# The paper's 5-class scheme (clarityBase.txt §3 / §9.3):
#   N = Normal sinus, L = LBBB, R = RBBB, V = PVC, A = APC
# The remaining .atr symbols are rhythm markers ('+'), noise ('~'), flutter
# boundaries ('[' ']'), isolated artefacts ('|'), etc. — not heartbeat labels.
# Mixing them into the training set would corrupt the feature distributions.
KEEP_LABELS = {"N", "L", "R", "V", "A"}

# ── Paced record exclusion ────────────────────────────────────────────────────
# Records 102, 104, 107, 217 contain artificially paced beats (clarityBase.txt §9.4).
# Pacing spikes are sharper and higher-amplitude than natural QRS complexes;
# their QRS morphology and duration features would form a spurious cluster that
# the classifier might learn instead of arrhythmia-related patterns.
PACED_RECORDS = {"102", "104", "107", "217"}

# %%

def get_record_list() -> list[str]:
    """Return sorted record names, excluding paced records."""
    all_records = sorted({p.stem for p in DATA_DIR.glob("*.hea")})
    usable = [r for r in all_records if r not in PACED_RECORDS]
    print(f"Records found: {len(all_records)}  |  paced excluded: {len(PACED_RECORDS)}  |  usable: {len(usable)}")
    return usable

# %%

def make_bandpass_filter() -> np.ndarray:
    """
    Design a 4th-order Butterworth band-pass filter and return it as SOS.

    Band edges:
      0.5 Hz high-pass — removes DC offset and slow baseline wander due to
          breathing and electrode movement (the EDA showed record 100 MLII
          has a mean of –0.31 mV; this corrects it without distorting QRS shape).
      40 Hz low-pass  — removes high-frequency muscle (EMG) noise.
          QRS energy is concentrated below 40 Hz so all clinically relevant
          morphology is preserved.

    Order 4 matches the paper (clarityBase.txt §4).

    SOS (second-order sections) is numerically more stable than the (b, a)
    transfer-function form for steep filters — avoids coefficient overflow that
    would silently produce NaN segments at higher filter orders.

    sosfiltfilt (used in load_and_segment) applies the filter forward then
    backward → zero phase shift.  Zero phase is critical here because any
    time shift would displace Q, S, and T relative to the annotated R-peak,
    corrupting interval features like QRS duration and QT.
    """
    return sps.butter(4, [0.5, 40], btype="band", fs=FS, output="sos")

# %%

def load_and_segment(record_name: str, sos: np.ndarray) -> dict | None:
    """
    Steps 1–4 for a single MIT-BIH record.

    Returns a dict of arrays (n_beats entries each), or None if the record
    yields no usable beats after filtering and label selection.
    """
    path = str(DATA_DIR / record_name)

    # ── STEP 1: LOAD ──────────────────────────────────────────────────────────
    # rdrecord reads the .dat binary and converts raw ADC integers to mV using
    # the gain and baseline stored in the .hea (physical = (digital − baseline) / gain).
    # We load all channels — not just MLII — so the output can feed both the
    # single-channel LightGBM experiments and the two-channel deep-learning models.
    record = wfdb.rdrecord(path)
    signal = record.p_signal      # (sig_len, n_channels), float64, mV

    # rdann reads the binary .atr file and resolves the gap-encoded sample
    # positions into absolute sample numbers (ann.sample).
    ann = wfdb.rdann(path, "atr")

    # Select MLII by name — channel order is not fixed across records.
    # (e.g. record 114 has MLII on index 1, not 0)
    try:
        mlii_idx = record.sig_name.index("MLII")
    except ValueError:
        print(f"  [skip] {record_name}: MLII not found in {record.sig_name}")
        return None

    n_channels = signal.shape[1]
    # Second channel index (non-MLII) — None if the record is single-channel
    other_idx = next((i for i in range(n_channels) if i != mlii_idx), None)

    # ── STEP 2: FILTER ────────────────────────────────────────────────────────
    # sosfiltfilt: zero-phase, forward + backward pass.
    # axis=0 filters along the time axis independently for each channel.
    # Cast to float32 now to halve memory for the final .npz.
    filtered = sps.sosfiltfilt(sos, signal, axis=0).astype(np.float32)
    sig_len  = filtered.shape[0]

    # ── STEPS 3 & 4: R-PEAKS + SEGMENT ───────────────────────────────────────
    segments, labels, r_samples_out = [], [], []

    for sample, symbol in zip(ann.sample, ann.symbol):
        # Drop non-beat annotation types
        if symbol not in KEEP_LABELS:
            continue

        start = sample - WIN_PRE
        end   = sample + WIN_POST

        # Beats whose window extends past the recording boundary are dropped.
        # Zero-padding would introduce artificial signal at the window edges
        # that would distort amplitude, energy, and wavelet features.
        if start < 0 or end > sig_len:
            continue

        # Stack channels: MLII always at position 0, second channel at position 1.
        # Shape: (WIN_LEN, 2) then transposed to (2, WIN_LEN) — channels-first.
        if other_idx is not None:
            window = np.stack(
                [filtered[start:end, mlii_idx],
                 filtered[start:end, other_idx]],
                axis=1,
            )
        else:
            # Single-channel fallback (should not occur in standard MIT-BIH records)
            window = filtered[start:end, [mlii_idx]]

        segments.append(window.T)          # (n_channels, WIN_LEN)
        labels.append(symbol)
        r_samples_out.append(sample)

    if not segments:
        return None

    return {
        "segments":   np.stack(segments, axis=0),            # (n, ch, WIN_LEN)
        "labels":     np.array(labels),                       # (n,)  str
        "record_ids": np.array([record_name] * len(labels)),  # (n,)  str
        "r_samples":  np.array(r_samples_out, dtype=np.int32),
    }

# %%

def process_all_records(n_workers: int = 8) -> dict:
    """
    Run load_and_segment for all usable records and merge into one dataset.

    ThreadPoolExecutor is used because wfdb file reads are I/O-bound.
    scipy/numpy operations (filtering, stacking) release the GIL internally,
    so multiple threads can overlap them with file I/O effectively.
    For 44 records at ~30 min each the full run takes a few seconds.
    """
    records = get_record_list()
    sos     = make_bandpass_filter()

    results = []
    print(f"\nProcessing {len(records)} records ({n_workers} workers)...\n")

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(load_and_segment, r, sos): r for r in records}
        for future in as_completed(futures):
            rec = futures[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                    print(f"  [ok]    {rec:>3s}  →  {len(result['labels']):5,} beats")
            except Exception as exc:
                print(f"  [error] {rec}: {exc}")

    if not results:
        raise RuntimeError("No records processed — check DATA_DIR path.")

    # Concatenate per-record arrays along the beats axis
    dataset = {
        "segments":   np.concatenate([r["segments"]   for r in results], axis=0),
        "labels":     np.concatenate([r["labels"]     for r in results], axis=0),
        "record_ids": np.concatenate([r["record_ids"] for r in results], axis=0),
        "r_samples":  np.concatenate([r["r_samples"]  for r in results], axis=0),
    }
    return dataset

# %%

def save_dataset(dataset: dict, path: Path) -> None:
    """Write all arrays to a single compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        segments=dataset["segments"],
        labels=dataset["labels"],
        record_ids=dataset["record_ids"],
        r_samples=dataset["r_samples"],
    )
    size_mb = path.stat().st_size / 1e6
    print(f"\nSaved → {path}  ({size_mb:.1f} MB)")

# %%

def print_stats(dataset: dict) -> None:
    """Print a summary of the assembled dataset."""
    segs   = dataset["segments"]
    labels = dataset["labels"]
    recs   = dataset["record_ids"]

    label_names = {"N": "Normal", "L": "LBBB", "R": "RBBB", "V": "PVC", "A": "APC"}

    print(f"\n{'─'*52}")
    print(f"Segment tensor : {segs.shape}  (beats × channels × samples)")
    print(f"Total beats    : {len(labels):,}")
    print(f"Records        : {len(np.unique(recs))}")
    print(f"Memory (float32): {segs.nbytes / 1e6:.1f} MB")
    print(f"\n{'Cls':<4} {'Name':<8} {'Count':>8}  {'%':>6}")
    print("─"*32)
    for cls in sorted(np.unique(labels)):
        n   = int((labels == cls).sum())
        pct = 100 * n / len(labels)
        print(f"{cls:<4} {label_names.get(cls,''):8s} {n:>8,}  {pct:>5.1f}%")
    print("─"*32)

# %%

if __name__ == "__main__":
    dataset = process_all_records(n_workers=8)
    print_stats(dataset)
    save_dataset(dataset, OUTPUT_DIR / "mit_bih_segments.npz")
