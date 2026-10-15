# FYP Progress

**Zain Ul Arifeen Sukhera**
**Last updated: 2026-10-14**

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

### Feature extraction — `source/features.py` (early Oct 2026)

40-feature hybrid vector per beat, MLII channel only:

| Group | Features | Count |
|-------|----------|-------|
| A — Statistical | mean, std, ptp, skewness, kurtosis, RMS | 6 |
| B1 — Fiducial intervals | RR_pre, RR_post, QRS_dur, QT_interval, P_dur | 5 |
| B2 — Fiducial morphology | R/Q/S/T/P amplitudes, R/S ratio, P/R ratio, ST elev | 8 |
| C — DWT (db4, level 4) | energy + entropy per subband (cA4, cD4–cD1), total energy, cD1 rel energy | 12 |
| D1 — HRV time + freq | SDNN, RMSSD, LF, HF, LF/HF | 5 |
| D2 — Non-linear | Poincaré SD1, SD2, SampEn, SD1/SD2 | 4 |

- Rule-based windowed-extrema delineation for Q, S, T, P — no neurokit2 dependency.
- HRV window: last min(50, available) RR intervals per record.
- SampEn params: m=2, r=0.2·SD.
- Parallelised with `ThreadPoolExecutor` (8 workers) per record.
- Output: `datasets/processed/mit_bih_features.npz` — `X (N,40)`, `y`, `record_ids`, `r_samples`, `feature_names`.

### Train/test splits — `source/split.py` (early Oct 2026)

Two splits implemented and saved:

| Split | File | Train | Test |
|-------|------|-------|------|
| Random stratified 80/20 | `split_random.npz` | ~79,432 beats | 19,858 beats |
| Inter-patient DS1/DS2 | `split_ds1ds2.npz` | DS1 patients | 48,981 beats (DS2) |

DS1/DS2 follows the standard MIT-BIH inter-patient benchmark: no patient appears in both train and test.

### LightGBM training — `source/train.py` (early–mid Oct 2026)

- SMOTE inside imblearn Pipeline — applied per CV fold only, preventing leakage.
- Optuna TPE, objective = stratified 5-fold macro-F1, 10 trials.
- Paper's reported hyperparameters (`n_estimators=1450`, `max_depth=11`, `lr=0.05`, `num_leaves=38`, `subsample=0.85`, `colsample_bytree=0.75`) enqueued as trial 0.
- SQLite checkpointing — interrupted runs resume from last completed trial.
- Final model refit on full resampled (SMOTE) training data.
- Both splits trained: `lgbm_random.pkl`, `lgbm_ds1ds2.pkl`.

### Evaluation run 1 — `source/evaluate.py` (2026-10-13)

**Random split (19,858 test beats)**

| Metric | Ours | Paper |
|--------|------|-------|
| Accuracy | 0.9960 | 0.989 |
| Macro-F1 | 0.9875 | ~0.937 |
| Binary F1 | 0.9928 | 0.928 |
| OVR AUC | 0.9992 | 0.985 |

Exceeds the paper on every metric. Per-class F1: N=0.998, L=0.997, R=0.998, V=0.991, A=0.954.

**DS1/DS2 split (48,981 test beats)**

| Metric | Ours | Paper | Gap |
|--------|------|-------|-----|
| Accuracy | 0.7372 | 0.989 | −0.252 |
| Macro-F1 | 0.3132 | ~0.937 | −0.624 |
| Binary F1 | 0.3419 | 0.928 | −0.586 |
| OVR AUC | 0.7686 | 0.985 | −0.217 |

Per-class F1: N=0.838, L=**0.000**, R=**0.006**, V=0.655, A=0.067. LBBB completely missed; RBBB near-zero.

**Ablation A1→A4 (random split, paper params)**

| Config | Feats | Acc | AnomF1 | Paper Acc |
|--------|-------|-----|--------|-----------|
| A1 Statistical | 6 | 0.924 | 0.875 | 0.985 |
| A2 +Fiducial | 19 | 0.994 | 0.990 | 0.987 |
| A3 +Wavelet | 31 | 0.995 | 0.991 | 0.988 |
| A4 +HRV/NonLin | 40 | 0.996 | 0.993 | 0.989 |

A1→A2 jump is +0.070 accuracy in our model vs +0.002 in paper — fiducial amplitude features exploiting patient-identity on random split.

### Root cause analysis + fixes — `source/features.py`, `source/train.py` (2026-10-14)

**Root cause:** Inter-patient generalisation failure. Absolute amplitude features (`fid_r_amp`, `fid_q_amp`, etc.) encode electrode placement and cardiac axis — both patient-specific, not arrhythmia-specific. SMOTE reinforces DS1 patient distributions. CV F1 on DS1 training was 0.985 (no underfitting) — failure is entirely distribution shift, not capacity.

**Fixes implemented:**

1. **Per-record amplitude normalisation** (`features.py` — `process_record()`): each beat divided by `median(|R-amplitudes|)` across the record before feature extraction. All amplitude-derived features (B2 morphology, A statistical, C wavelet energies) become record-relative ratios.

2. **RR interval normalisation** (`features.py` — `process_record()`): `rr_pre` and `rr_post` divided by the record's mean RR before feature extraction — now dimensionless ratios isolating ectopic timing patterns. Feature names updated to `fid_rr_pre_norm` / `fid_rr_post_norm`.

3. **N_TRIALS 10 → 50** (`train.py`): more Optuna trials to explore from the paper-params seed after features change.

4. **DS1/DS2 checkpoints deleted**: old 10 trials trained on unnormalized features — would steer the new search incorrectly. Random split model untouched (already exceeds paper).

**Next:** rebuild features → splits → retrain → re-evaluate.

---

## Left to do

### ML pipeline

- [ ] **Re-run full pipeline** after normalisation fixes: `features.py` → `split.py` → `train.py` → `evaluate.py`
- [ ] **Verify DS1/DS2 recovery** — LBBB and RBBB F1 should recover from ~0 to meaningful values; target accuracy approaching paper's 0.989
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
