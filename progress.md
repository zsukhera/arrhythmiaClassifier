# FYP Progress

**Zain Ul Arifeen Sukhera**
**Last updated: 2026-09-30**

---

## Done

### Dataset — MIT-BIH EDA (27–29 Sep 2026)

- Deep-dived the three file formats: `.hea` (header metadata), `.dat` (ECG samples, format 212), `.atr` (cardiologist annotations).
- Confirmed 48 records, 2 channels each (MLII + secondary lead), 360 Hz, ~30 min per record (~650,000 samples).
- Understood interleaved 3-byte packing of format 212; verified `(digital − baseline) / adc_gain` conversion to mV.
- Noted that channel order varies between records — MLII must always be selected by name, not hard-coded index.
- Catalogued all annotation symbol types; identified non-beat symbols (`+`, `~`, `|`, `"`, `x`, `[`, `]`, `!`) that must be filtered before training.
- Documented class imbalance (e.g. record 100: 2239 N, 33 A, 1 V) and DC offset / baseline wander issue requiring filtering.
- Compared MIT-BIH (beat-level, 2-channel) vs PTB-XL (record-level, 12-lead) — confirmed label spaces are incompatible; working decision is Lead II only for any cross-dataset work.

### Preprocessing pipeline — `source/preprocess.py` (29–30 Sep 2026)

Steps 1–4 of the CLARITY-AI 2.0 medical pipeline implemented and run:

| Step | What |
|------|------|
| 1. Load | `wfdb.rdrecord` + `wfdb.rdann`; MLII selected by name; two-channel output |
| 2. Filter | 4th-order Butterworth band-pass 0.5–40 Hz (SOS form); zero-phase via `sosfiltfilt` |
| 3. R-peaks | Cardiologist `.atr` annotation positions used directly as ground truth |
| 4. Segment | 256-sample window per beat (100 pre-R, 156 post-R); boundary beats dropped (no padding) |

- Paced records excluded (102, 104, 107, 217) — pacing spikes would corrupt QRS morphology features.
- 5-class filter applied: N, L, R, V, A only.
- Parallelised with `ThreadPoolExecutor` (8 workers) — I/O-bound, runs in a few seconds for all 44 records.
- Output: `datasets/processed/mit_bih_segments.npz`
  - `segments` — `(n_beats, 2, 256)` float32, channels-first
  - `labels` — `(n_beats,)` beat symbol
  - `record_ids` — `(n_beats,)` source record name
  - `r_samples` — `(n_beats,)` int32 R-peak sample position

---

## Left to do

### ML pipeline

- [ ] **Feature extraction** — 42 handcrafted features per channel per beat (RR interval stats, QRS morphology, R/Q/S amplitudes, P-wave, T-wave, interval durations, statistical moments). 84-feature vector for two-channel input.
- [ ] **Train/test split** — record-level (subject-disjoint); ~80% records train, ~20% test. No beat-level random split.
- [ ] **SMOTE** — oversample minority classes (L, R, V, A) on the training set only.
- [ ] **LightGBM experiments:**
  - Exp 1: 42 features, single-channel (MLII)
  - Exp 2: 84 features, two-channel concatenation
  - Exp 3: feature importance → test top 10/20/30 subsets
  - Tuning: Optuna Bayesian search, objective = CV macro-F1
- [ ] **1D-CNN baseline** — input shape `(N, 2, 720)`
- [ ] **LSTM baseline** — input shape `(N, 720, 2)`
- [ ] **KANformer (teacher)** — input shape `(N, 720, 2)`; train on ground-truth labels
- [ ] **Knowledge distillation** — freeze KANformer; train 1D-CNN student with `L = α·CE + (1−α)·KL`
- [ ] **Quantisation/pruning** — INT8 KD-CNN for ESP32 deployment
- [ ] **Evaluation** — macro-F1, balanced accuracy, per-class recall for all models; edge metrics (latency, RAM, flash, energy) for deployed model

### Zero-shot generalisation

- [ ] Download PTB-XL and Chapman-Shaoxing datasets (Lead II only)
- [ ] Run trained MIT-BIH model on PTB-XL and Chapman without retraining

### Hardware / system

- [ ] ECG acquisition firmware — ADS1292 + ESP32-S3, 500 SPS → resample to 360 Hz
- [ ] Fixed gateway pipeline — Raspberry Pi 4B, local multiclass inference
- [ ] Mobile relay — Android transport bridge (no decryption/inference)
- [ ] Cloud backend — inference on mobile path, IAM, data API
- [ ] IDS/IPS layer on fixed gateway
- [ ] Caregiver/patient dashboard — ECG display + alerts

### Milestones

| Milestone | Target |
|-----------|--------|
| FYP-1 Mid (Week 8) | Model baselines, ECG acquisition, gateway transport, dashboard prototype |
| FYP-1 Final (Week 16) | Full implementation — both routes, IDS/IPS, IAM, cloud, integrated validation |
| FYP-2 Mid (Week 8) | Model/policy refinement, hardware profiling, attack stress tests |
| FYP-2 Final (Week 16) | Held-out evaluation, final deployment, report, handover |
