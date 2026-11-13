# FYP Progress

**Zain Ul Arifeen Sukhera**
**Last updated: 2026-11-13**

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

### Farag 2023 replication — `HasanML/paper3/` (2026-11-07)

Full end-to-end replication of M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification and Arrhythmia Detection at the Edge," *Sensors* 23(3):1365, 2023.  This is the **cloud 1D-CNN candidate teacher** and also the intended **ESP32 student model**.

**What was built** (all in `HasanML/paper3/`):

| Module | What it does |
|--------|-------------|
| `src/data.py` | MIT-BIH loading, DS1/DS2/DS1-train/DS1-val splits, full AAMI label map, binary (aami + raw) label definitions |
| `src/preprocess.py` | Resample 360→128 Hz (`resample_poly` up=16 down=45), 64-sample window (32+32 around R-peak), edge-pad, mean normalisation, first derivative, NK=32/64 centre-crop |
| `src/rr.py` | 4 causal RR features (pre/post × local-80/global-400), XGBoost sanity check targeting paper's ~93% acc / N=97%, S=60%, V=58% recall |
| `src/templates.py` | 13 sub-class matched-filter templates from DS1-train only, unit-L2 normalised, NO time-flip (Keras uses cross-correlation), template plots |
| `src/model.py` | Keras functional API two-input model (ECG branch + RR branch), MF init, param-count guard (1000–2000 for 3-class single-channel) |
| `src/train.py` | Multi-seed training (seeds 0–9), EarlyStopping + ReduceLROnPlateau, class weights, full caching so interrupted runs resume |
| `src/metrics.py` | Plain accuracy AND mean one-vs-rest accuracy (1−2e/K) reported side-by-side, majority-class baseline, per-class Se/PPV/Sp/F1, binary metrics, threshold sweep |
| `src/export.py` | TFLite float32 / float16 / dynamic / int8-PTQ / int8-QAT (tfmot) / pruned+int8, C header generation (`xxd -i` style) for TFLite Micro on ESP32 |
| `src/run_experiment.py` | Main CLI: `--p1` (Model 6 best), `--p2`/`--p3` (binary aami/raw), `--grid` (full Table 4 reproduction), `--ablations`, `--rr-baseline`, `--export` |
| `tests/test_pipeline.py` | **34/34 tests pass** — split isolation, label maps, RR causality, resampling, segmentation, MF no-flip, accuracy definitions, param counts |
| `configs/default.yaml` | Every parameter configurable; 17 ASSUMPTIONs documented with justification |
| `results/paper_table4.md` | Paper Table 4 reference (known values filled in; unknowns marked `—`) |
| `README.md` | Citation, architecture diagram, param counts, split table, full assumption log, firmware checklist |

**Architecture:**
```
ecg_in (NK, C)
  Conv1D(13, NK, valid, relu) -> BatchNorm -> GlobalMaxPool -> (13,)
rr_in (4,)
  Dense(16) x3 -> (16,)
Concatenate -> Dense(n_classes, softmax)
```
NK=32 → 1,195 total params; NK=64 → 1,611 params. Both in paper's stated range.

**Paper target** (DS2, Model 6): OvR accuracy 98.18%, avg F1 92.17%, SVEB sensitivity 85.3%, VEB sensitivity 96.34%.

**Key accuracy note:** "average accuracy" in this paper likely means **mean one-vs-rest accuracy** (1−2e/K), NOT plain multiclass accuracy. At K=3, 98.18% OvR = 97.27% plain. Pipeline reports both.

**Split:** DS1-train = 12 DS1 records with number < 200; DS1-val = 10 DS1 records ≥ 200 (matches Farag 2023). DS2 = 22 records. DS2 never used for templates, stats, or early stopping.

**Run:**
```bash
cd HasanML/paper3
python -m src.run_experiment --p1              # best model (Model 6), 10 seeds
python -m src.run_experiment --p1 --p2 --p3   # + binary extension
python -m src.run_experiment --grid            # full Table 4 grid
python -m src.run_experiment --p1 --export    # TFLite + C headers for ESP32
python -m pytest tests/ -v                    # 34/34 pass
```

**TF/package versions:** TensorFlow 2.21.0, Keras 3.15.1, tfmot 0.8.1, XGBoost 3.4.1 (all installed in `.venv`).

**DS2 count correction:** The de Chazal split has 22 DS2 records (not 23 as stated in some sources).

---

### Root cause analysis + fixes — `source/features.py`, `source/train.py` (2026-10-14)

**Root cause:** Inter-patient generalisation failure. Absolute amplitude features (`fid_r_amp`, `fid_q_amp`, etc.) encode electrode placement and cardiac axis — both patient-specific, not arrhythmia-specific. SMOTE reinforces DS1 patient distributions. CV F1 on DS1 training was 0.985 (no underfitting) — failure is entirely distribution shift, not capacity.

**Fixes implemented:**

1. **Per-record amplitude normalisation** (`features.py` — `process_record()`): each beat divided by `median(|R-amplitudes|)` across the record before feature extraction. All amplitude-derived features (B2 morphology, A statistical, C wavelet energies) become record-relative ratios.

2. **RR interval normalisation** (`features.py` — `process_record()`): `rr_pre` and `rr_post` divided by the record's mean RR before feature extraction — now dimensionless ratios isolating ectopic timing patterns. Feature names updated to `fid_rr_pre_norm` / `fid_rr_post_norm`.

3. **N_TRIALS 10 → 50** (`train.py`): more Optuna trials to explore from the paper-params seed after features change.

4. **DS1/DS2 checkpoints deleted**: old 10 trials trained on unnormalized features — would steer the new search incorrectly. Random split model untouched (already exceeds paper).

**Next:** rebuild features → splits → retrain → re-evaluate.

---

### KAN-Former-1D teacher model — `HasanML/paper4/` (2026-11-08)

Full PyTorch implementation of the KAN-Former-1D teacher model for 5-class AAMI ECG arrhythmia classification on MIT-BIH (de Chazal DS1/DS2 inter-patient split). Designed as a knowledge-distillation teacher for the paper3 Farag 2023 student.

**Architecture (KAN-Former):**
- **Multi-scale CNN stem (MS-CA):** three parallel depthwise-separable Conv1d branches (k=3,7,15) → SE channel attention → residual + MaxPool → 32 tokens (beat) / 135 tokens (context)
- **Two aligned views per beat:** Beat window (256 samples @ 360 Hz, 90+166 around R-peak) + Context window (1080 samples centred on R-peak, 3 s rhythm context)
- **6 causal RR ratio features:** pre/local, post/local, pre/global, post/global, pre/post, (post−pre)/local — ratios only (no absolute RR), so they generalise across patients
- **3× Transformer layers:** pre-norm, 4 heads, d_model=128, GLU-FFN(256), stochastic depth 0.1
- **KAN classification head:** B-spline KAN layers (Liu 2024, efficient-kan formulation) replace standard MLP; inputs squeezed to (−1,1) by LayerNorm + tanh before each layer
- **Focal loss** (γ=2, √(1/freq) class weights), **EMA** (decay=0.999), **cosine LR with warmup**, **temperature scaling** calibration on DS1-val
- **~1M parameters** (d_model=128, 3 transformer layers, KAN head 262→64→5)

**Files created (14 source, 1 test, support files):**

| File | Purpose |
|------|---------|
| `src/data.py` | MIT-BIH loading, AAMI map, DS1/DS2 split (identical to paper3) |
| `src/preprocess.py` | Bandpass filter, beat/context window extraction with edge padding |
| `src/rr.py` | 6 causal RR ratio features, LightGBM sanity check |
| `src/augment.py` | 5 augmentation strategies (amplitude, noise, baseline, time shift/stretch) |
| `src/kan.py` | B-spline KAN layer (Cox-de Boor recursion, efficient-kan formulation) |
| `src/model.py` | KANFormer full model: MS-CA CNN + Transformer + KAN head |
| `src/losses.py` | FocalLoss, WeightedCE, LabelSmoothing, build_loss() |
| `src/train.py` | EMA, cosine schedule, AMP (mixed precision), early stopping, BeatDataset |
| `src/calibrate.py` | Temperature scaling (Guo 2017) + ECE metric |
| `src/metrics.py` | OvR accuracy, macro F1, per-class Se/PPV/F1, confusion matrix |
| `src/distill_export.py` | Soft labels (.npy), ONNX export (opset 14), TorchScript |
| `src/run_experiment.py` | Full CLI: train, ablations (A1–A6b), distill, search, pretrain |
| `tests/test_pipeline.py` | 34+ unit tests (split isolation, AAMI, RR causality, KAN basis, focal loss, metrics) |
| `colab_paper4.ipynb` | 11-cell Colab notebook (Drive mount, GPU check, download, train, distill) |
| `configs/default.yaml` | Full hyperparameter config |
| `results/paper_targets.md` | Target metrics and ablation expectations |

**GPU/CPU/Colab compatibility:**
- Mixed precision via `torch.autocast` + `GradScaler` (`--mixed-precision` flag); TF32 enabled for Ampere/Ada
- `torch.use_deterministic_algorithms(True, warn_only=True)` for reproducibility; `--no-deterministic` for max throughput
- Colab notebook handles Drive mount, repo clone/pull, dep install, GPU detection
- `P4_DEVICE` env override for headless cluster runs
- Bash wrapper: `bash 1DCNN/run_paper4.sh {tests|sanity|train|gpu|ablations|distill}`

**Knowledge distillation pipeline:**
- Teacher (paper4) exports `soft_labels_{split}_T{1,2,4}.npy` for DS1 and DS2
- Student (paper3) trains with `L = α·CE(pred, hard) + (1−α)·T²·KL(teacher_soft ∥ student_soft)`
- Soft labels are framework-agnostic `.npy` so the Keras student needs no PyTorch

---

### Farag 2023 — P1 training run (2026-11-13)

Ran on university GPU lab (Precision 3680, CPU-only — CUDA drivers not installed). Python 3.14.4, TF 2.22.0rc0.

**Bugs found and fixed:**
- `run_experiment.py`: `DS1` undefined in template fallback — fixed to `DS1_TRAIN + DS1_VAL`
- `run_experiment.py`: `--class-weight` CLI flag ignored for P1 (hardcoded to `none`) — fixed to honour CLI
- `train.py`: EarlyStopping monitored `val_loss` (unweighted in Keras) — dominated by N class (90% of data), triggered at ~27 epochs before minority classes could be learned. Fixed by adding `MacroF1Callback` and switching to `val_macro_f1` when `class_weight=balanced`
- `requirements.txt`: added `pytest`, bumped `tensorflow>=2.22` for Python 3.14 compat

**Run 1 — no class weights (paper Model 6 as written):**

| Metric | Ours | Paper target |
|--------|------|-------------|
| OvR accuracy | 95.56% | 98.18% |
| Macro F1 | 54.32% | 92.17% |
| Se_S | 0.00% | 85.3% |
| Se_V | 57.32% | 96.34% |

Root cause: DS1-train has only 149 S beats (0.59% of training data). Without class weights the model never predicts S. Paper's claim of Se_S=85.3% with no class weights is not reproducible.

**Run 2 — balanced class weights + macro-F1 early stopping (in progress, 2026-11-13):**

Final results (10 seeds, all 500 epochs, ~285 s/seed, ~48 min total):

| Metric | Ours | Paper target |
|--------|------|-------------|
| OvR accuracy | 93.99 ± 0.36% | 98.18% |
| Plain accuracy | 90.98 ± 0.54% | — |
| Se_S | 35.01% | 85.3% |
| Se_V | 86.71% | 96.34% |

Se_S recovered from 0% → 35% (seed range 15%–52% — high variance due to only 149 S training beats). Gap from paper remains. The paper's claimed Se_S=85.3% with no class weights is not reproducible; achieving it requires balanced class weights and even then is limited by training data size.

---

## Left to do

### ML pipeline

- [ ] **Re-run full pipeline** after normalisation fixes: `features.py` → `split.py` → `train.py` → `evaluate.py`
- [ ] **Verify DS1/DS2 recovery** — LBBB and RBBB F1 should recover from ~0 to meaningful values; target accuracy approaching paper's 0.989
- [x] **Farag 2023 1D-CNN** (`HasanML/paper3/`) — full replication with TFLite/ESP32 export; 34 tests pass. P1 training run on university GPU (2026-11-13); see results below.
- [ ] **LSTM baseline** — input shape `(N, 720, 2)`
- [x] **KAN-Former (teacher)** (`HasanML/paper4/`) — PyTorch, 2-branch CNN + Transformer + KAN head; ~1M params; cloud-only. Full implementation complete: 14 source files, 34+ unit tests, Colab notebook, bash script, distillation export. **Run GPU training to get DS2 numbers.**
- [ ] **Knowledge distillation** — freeze KAN-Former (or paper3 1D-CNN if it's the better teacher); train Keras student in paper3 with `L = α·CE + (1−α)·KL`. Soft labels saved by teacher at T∈{1,2,4} as `results/teacher/soft_labels_T{t}.npy`.
- [ ] **Quantisation/pruning** — INT8 student via paper3's export.py (already implemented); QAT + 50% pruning. Test on-device accuracy.
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
