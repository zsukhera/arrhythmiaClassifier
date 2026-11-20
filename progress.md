# FYP Progress

**Zain Ul Arifeen Sukhera**
**Last updated: 2026-11-19 (all 9 agents done: Farag 2023 reproduced to within 0.6 acc via `p1_faithful.yaml`; KAN-Former, LSTM, zero-shot, ESP32 export and LightGBM results added. See "Multi-agent run — results")**

---

## Summary: multi-agent run (2026-11-18 → 19)

**Setup.**
- Nine parallel agents on the university GPU box (RTX 4000 Ada; CUDA now working) finished every remaining ML todo except knowledge distillation (paused on purpose) and the hardware/system items.
- All nine are done and nothing is still running.
- Code and results are synced into this repo. Large models and data stay on the box.
- Per-agent reports are in `reports/<agent>.md`. Details are in "Multi-agent run — results" below.

**Headline: Farag 2023 is reproduced** (`HasanML/paper3/configs/p1_faithful.yaml`). Three changes closed the gap:
1. Train on all of DS1.
2. Use the paper's exact Table 3 architecture (11 kernels, tanh, 64-sample input, 1,267 params).
3. z-score each segment, with no class weights and at most 100 epochs.

| DS2 | Before (official P1) | **Now (faithful, 5 seeds)** | Paper Model 6 |
|---|---|---|---|
| Plain accuracy | 91.61 | **97.60** (ensemble 98.01) | 98.18 |
| Macro F1 | 77.08 | **90.07** (ensemble 91.64) | 92.17 |
| PPV_S | 32.1 | **74.3** | 82.7 |
| Se N / S / V | 91.4 / 87.1 / 96.5 | 98.5 / 80.7 / 95.3 | 99.1 / 81.6 / 95.0 |

**Corrected paper targets.**
- The paper's 98.18 is **plain** accuracy, not one-vs-rest.
- SVEB Se 85.3 / VEB Se 96.34 belong to Model 1 (class-weighted), not Model 6.
- Se_V 89.7 / Se_N 97.8 appear nowhere in the paper.

**Other results.**

| Area | Outcome |
|---|---|
| Farag improvements (compact model) | Patient-wise CV refit: acc 94.84, F1 81.47. The random-beat hold-out leaked patients. Superseded by the faithful config. |
| ESP32 export | INT8 PTQ is lossless: 7 KB flash, ≤3.4 KB RAM, C headers in `1DCNN/export/`. Binary P2: F1 81.7, AUC 0.978. |
| KAN-Former | Trains on the GPU after 10 bug fixes. 3-class acc 94.6, but Se_S only 25% (trained on DS1-train), so it is not yet a useful teacher. |
| LSTM baseline | Waveform alone misses S beats (Se_S 13%). With RR features, macro F1 78.2. L/R fail across patients, as for LightGBM. |
| Zero-shot (PTB-XL, Chapman) | PVC transfers (Se 78–91%); PAC is moderate; LBBB/RBBB don't. 10-s records hurt the RR features most. |
| LightGBM (paused) | Grouped CV explains about 97% of the DS1→DS2 gap; best DS2 macro F1 0.423, binary F1 0.519. LBBB/RBBB fail because DS1 has only 2 + 2 such patients. The paper's 0.989 matches our random split only. |

**Next.**
1. Re-export to the ESP32 and rerun zero-shot with the `p1_faithful` model.
2. E-list experiments for the last ~8 points of PPV_S (E1, E3, E4, E7, E9, E12).
3. Retrain the KAN-Former on all of DS1.
4. Correct the old targets in the weekly summary and notes.txt.
5. Commit the synced work.
6. Later: knowledge distillation and the hardware/system items.

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

**Paper target** (DS2, Model 6, corrected 2026-11-18 by farag-gap):
- ACC 98.18% (plain).
- Se N/S/V: 99.10 / 81.60 / 95.00.
- PREC N/S/V: 99.00 / 82.68 / 95.63.
- Macro PREC / Se / F1: 92.44 / 91.90 / 92.17.

~~Key accuracy note: OvR accuracy~~. **Corrected:** the paper's ACC is **plain multiclass accuracy**. Rebuilding it from the paper's own recalls gives exactly 98.18; the one-vs-rest version would be 98.79. The averages are unweighted macro means over N/S/V. The SVEB Se 85.3 / VEB Se 96.34 in the abstract belong to **Model 1** (class weights, NK=64), not Model 6. See "Farag 2023 — gap to paper Table 4".

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
| OvR accuracy | 95.56% | (98.79 OvR / 98.18 plain) |
| Macro F1 | 54.32% | 92.17% |
| Se_S | 0.00% | 81.6% |
| Se_V | 57.32% | 95.0% |

Root cause: DS1-train has only 149 S beats (0.59% of training data). Without class weights the model never predicts S. *(Paper targets corrected 2026-11-18: Model 6 has Se_S 81.6 / Se_V 95.0; the 85.3 / 96.34 figures belong to Model 1, which uses class weights.)*

**Run 2 — balanced class weights + macro-F1 early stopping (in progress, 2026-11-13):**

Final results (10 seeds, all 500 epochs, ~285 s/seed, ~48 min total):

| Metric | Ours | Paper target |
|--------|------|-------------|
| OvR accuracy | 93.99 ± 0.36% | 98.79 |
| Plain accuracy | 90.98 ± 0.54% | 98.18 |
| Se_S | 35.01% | 81.6% |
| Se_V | 86.71% | 95.0% |

Se_S recovered from 0% to 35% (seed range 15%–52%, high variance because there are only 149 S training beats). The gap from the paper remains. *(Targets corrected 2026-11-18 to Model 6.)*

---

### Sensitivity experiments — `1DCNN/1DCNN experiments/` (2026-11-13)

Five targeted experiments to close the Se_S gap. Run on university GPU lab (CPU-only, TF 2.22.0rc0, Python 3.14.4).

**Bug fixed before experiments:** `train.py` callback ordering — `MacroF1Callback` was appended after `EarlyStopping` in the callback list. Since Keras fires callbacks in list order during `on_epoch_end`, EarlyStopping ran before `val_macro_f1` was added to logs, so it always saw `None` and never triggered. Fixed by putting `MacroF1Callback` first. This affected all retrain experiments.

**All results on DS2 test set (best seed from Exp5 beats the paper target):**

| Method | Acc% | Se_N% | Se_S% | Se_V% | macF1% | Notes |
|--------|------|-------|-------|-------|--------|-------|
| Baseline P1 (10 seeds, argmax) | 90.98 | 93.61 | 35.01 | 86.71 | 66.58 | 149 S training beats |
| **Exp1: threshold τ=0.310** | 90.44 | 92.35 | **57.05** | 83.30 | 70.03 | No retraining; best tau from DS1-val |
| Exp2: ensemble+threshold | 90.44 | 92.35 | 57.05 | 83.30 | 70.03 | Same as Exp1 (Exp1 already ensembles) |
| Exp3: S-beat augmentation ×7 | 90.85 | 94.96 | 18.65 | 75.50 | 61.10 | Worse — augmented beats cause OOD shift |
| Exp4: focal loss γ=2 | 94.09 | 98.42 | 0.36 | 88.11 | 58.76 | Total failure — 160:1 imbalance too severe for focal loss alone |
| **Exp5: full DS1 (944 S beats)** | 91.57 | 91.73 | **83.03** | **94.19** | 76.14 | Best overall; seed 2 alone hits 88.1% |
| **Paper target (Farag 2023, Model 6, corrected)** | 98.18 | 99.10 | **81.6** | 95.0 | 92.17 | |

Exp5 details (full DS1 = DS1_TRAIN + DS1_VAL, 944 S beats vs original 149):
- Seed 0: Se_S=80.62%, Se_V=96.15%
- Seed 1: Se_S=80.35%, Se_V=92.58%
- Seed 2: Se_S=88.13% ✓ (exceeds paper target), Se_V=93.85%

**Key findings:**
1. **Data is the bottleneck.** The only method that significantly improves Se_S is using more training data (Exp5 gives 149 → 944 S beats, Se_S 35% → 83%).
2. Threshold calibration gives a moderate free boost (35% → 57%) with no retraining — worth using in deployment.
3. Augmentation is counterproductive. Synthetic noise/scaling/shift variants of S beats have different morphology distribution than real S beats. The model learns to detect the artificial pattern, not the real one.
4. Focal loss fails for extreme imbalance. The 160:1 (N:S) ratio is so skewed that (1−p_t)^2 weighting alone cannot compensate. Class weights remain necessary.
5. Accuracy tradeoff: using balanced class weights + full DS1 drops plain accuracy from 97% → 91.5%. This is because the model now predicts S more aggressively, misclassifying some N beats. The Se_N drops from ~99% → ~91%. For clinical use, Se_S and Se_V matter more than plain accuracy.

**Next steps:** Use Exp5 configuration (full DS1 + balanced weights) as the definitive P1 model. Consider combining Exp1 (threshold calibration) on top of Exp5 for further improvement.

---

### Official P1 retraining — Steps 1 + 2 (2026-11-13, university GPU)

Run via `1DCNN/1DCNN experiments/run_official_p1.py`. Full DS1 (DS1_TRAIN + DS1_VAL), class_weight=balanced, 10 seeds, 15% random beat hold-out for early stopping. Models saved to `HasanML/paper3/results/cache/P1_fullDS1_seed{n}.keras`.

GPU: Precision 3680, CPU-only. TF 2.22.0rc0, Python 3.14.4. Total training time: **216 s (10 seeds)**.

**Data:** N=45,866, S=944, V=3,788 (S fraction=1.87%). Train: 43,009 beats (S=799), Val: 7,589 beats (S=145).

**Step 1 — argmax results (DS2 test set):**

| Seed | Acc% | Se_N% | Se_S% | Se_V% | macF1% |
|------|------|-------|-------|-------|--------|
| 0 | 91.02 | 91.08 | 80.62 | 96.15 | 75.03 |
| 1 | 91.35 | 91.72 | 80.35 | 92.58 | 75.28 |
| 2 | 92.33 | 92.39 | 88.13 | 93.85 | 78.12 |
| 3 | 91.89 | 91.86 | 88.24 | 94.38 | 77.64 |
| 4 | 90.94 | 90.63 | 90.47 | 95.50 | 75.92 |
| 5 | 90.64 | 90.61 | 82.80 | 95.47 | 74.65 |
| 6 | 90.48 | 90.62 | 77.35 | 96.03 | 73.05 |
| 7 | 90.25 | 90.01 | 86.17 | 95.87 | 74.47 |
| 8 | 90.98 | 90.77 | 88.46 | 95.37 | 75.72 |
| 9 | 90.46 | 90.05 | 90.31 | 96.18 | 75.86 |
| **Mean** | **91.03** | **90.97** | **85.29 ± 4.42** | **95.14 ± 1.12** | **75.57** |
| Paper Model 6 (corrected) | 98.18 | 99.10 | 81.60 | 95.00 | 92.17 |

~~Se_S matches the paper's 85.3% exactly; Se_V exceeds 89.7%.~~ **Corrected 2026-11-18:**
- The 85.3 / 89.7 / 97.8 targets were wrong.
- Against Model 6 (mean of 10 runs, compared with our mean of seeds): our Se_S 85.29 and Se_V 95.14 are on par with or above 81.6 / 95.0.
- Our accuracy (91.03 vs 98.18) and precision are far below.

**Step 2 — threshold calibration on averaged ensemble:**

Calibrated tau_S = 0.500, picked by sweeping [0.05, 0.50] on avg val probas (val F1_S = 61.3%).

| Method | Acc% | Se_N% | Se_S% | Se_V% | macF1% |
|--------|------|-------|-------|-------|--------|
| Ensemble argmax | 91.03 | 90.97 | 85.29 | 95.14 | 75.57 |
| **Ensemble + tau=0.500** | **91.61** | **91.44** | **87.10** | **96.52** | **77.08** |
| Paper Model 6 (corrected) | 98.18 | 99.10 | 81.60 | 95.00 | 92.17 |

~~Threshold calibration adds +1.81pp Se_S.~~ **Corrected 2026-11-18 (farag-gap):**
- tau_S = 0.50 gives **exactly the same predictions as argmax** (0 beats differ). With 3 classes, P(S) ≥ 0.5 already makes S the argmax.
- The "ensemble argmax" row above is really the **mean of single seeds**; the actual 10-seed ensemble gives 91.61 / 87.10 / 96.52. The +1.8pp comes from **ensembling**, not from the threshold.

**Current P1 configuration:** full DS1 training (balanced class weights) + 10-seed ensemble → Acc 91.61%, Se_S 87.10%, Se_V 96.52%, macro F1 77.08%. ~~Exceeds the paper on all metrics.~~ It is **behind Model 6 on accuracy (−6.6) and precision (PPV_S 32.1 vs 82.7)**; see the gap section below.

---

### LightGBM re-run with normalisation fixes — Step 5 (2026-11-13 → 14, university GPU, complete)

Re-ran `source/train.py` with Nov 13 normalised features (per-record amplitude + RR normalisation). Stale Optuna checkpoints deleted. N_TRIALS=100, 5-fold CV, both splits, `device=gpu`.

**Training — finished 2026-11-14 05:55 (100/100 trials on both splits):**

| Split | Run window | Mean time/trial | Best trial | Best CV macro-F1 | Trial 0 (paper params) |
|-------|-----------|-----------------|-----------|------------------|------------------------|
| Random | 11-13 19:21 → 11-14 01:44 | 230 s | #85 | 0.9858 | 0.9850 |
| DS1/DS2 | 11-14 01:45 → 05:54 | 150 s | #94 | 0.9865 | 0.9855 |

Best params:
- **Random:** n_estimators=1126, max_depth=9, lr=0.0567, num_leaves=29, subsample=0.680, colsample_bytree=0.651
- **DS1/DS2:** n_estimators=1282, max_depth=12, lr=0.0280, num_leaves=46, subsample=0.609, colsample_bytree=0.915

100 Optuna trials gained only ~+0.001 CV F1 over the paper's params. Tuning is not where the remaining gap is.

Models: `models/lgbm_random.pkl`, `models/lgbm_ds1ds2.pkl` (on GPU box `~/Desktop/arrhythmiaClassifier/`).

**Evaluation (main eval only, run 2026-11-18; log at `eval_step5_main.log` on GPU box):**

| Metric | Random (new) | Random (run 1) | DS1/DS2 (new) | DS1/DS2 (run 1) | Paper |
|--------|-------------|----------------|---------------|-----------------|-------|
| Accuracy | 0.9958 | 0.9960 | **0.7635** | 0.7372 | 0.989 |
| Macro-F1 | 0.9872 | 0.9875 | **0.3878** | 0.3132 | ~0.937 |
| OVR AUC | 0.9994 | 0.9992 | **0.7873** | 0.7686 | 0.985 |
| Binary F1 | 0.9924 | 0.9928 | **0.4087** | 0.3419 | 0.928 |
| Binary AUC | 0.9997 | — | 0.7413 | — | 0.985 |

DS1/DS2 per-class (new):

| Class | Prec | Rec | F1 | F1 run 1 | Support |
|-------|------|-----|----|----------|---------|
| N | 0.796 | 0.931 | 0.858 | 0.838 | 36,427 |
| L | 0.073 | 0.008 | **0.014** | 0.000 | 4,124 |
| R | 0.007 | 0.001 | **0.002** | 0.006 | 3,475 |
| V | 0.719 | 0.974 | 0.827 | 0.655 | 3,219 |
| A | 0.361 | 0.177 | 0.237 | 0.067 | 1,736 |

DS1/DS2 binary anomaly: precision 0.607, recall 0.308.

**Findings:**
1. The random split is unchanged and still near ceiling. Normalisation cost nothing there.
2. On DS1/DS2, normalisation helped where expected: V F1 0.655 → 0.827, A F1 0.067 → 0.237, macro-F1 +0.075.
3. **LBBB/RBBB did not recover** (F1 ≈ 0). CV F1 on DS1 is 0.987, so this is still pure inter-patient shift. The DS1 LBBB/RBBB records don't transfer to the DS2 ones with these hand-crafted features. Per-record amplitude/RR normalisation doesn't fix bundle-branch morphology differences between patients.
4. The large CV vs test gap (0.987 vs 0.388) means record-level CV is needed. Optuna is currently tuning on beat-level folds that leak patients, so it can't select for cross-patient generalisation.

**Not yet run:** A1→A4 ablation on the new features (`evaluate.run_ablation()`), binary LightGBM (`source/binaryLightGBM/`).

---

### Farag 2023 — gap to paper Table 4 and improvement plan (2026-11-18)

Targets were verified against the full paper text (Europe PMC PMC9919183) by the farag-gap agent on 2026-11-18. Full report: `reports/farag-gap.md` on the GPU box. `paper_table4.md` has been rewritten with all 9 models.

**Paper Model 6** (DS2, **no class weights**, mean of 10 runs):

| | PREC | Se | F1 |
|---|---|---|---|
| N | 99.00 | 99.10 | 99.05 |
| SVEB | 82.68 | 81.60 | 82.14 |
| VEB | 95.63 | 95.00 | 95.31 |
| Macro | 92.44 | 91.90 | 92.17 |

ACC 98.18 is plain accuracy.

**Gap under the paper's definitions (DS2):**

| Metric | Ours (10-seed ensemble) | Paper Model 6 | Gap | Paper Model 2 (same config as ours) | Gap |
|---|---|---|---|---|---|
| ACC (plain) | 91.61 | 98.18 | −6.6 | 95.58 | −4.0 |
| Macro PREC | 71.43 | 92.44 | −21.0 | 79.63 | −8.2 |
| Macro Se | 91.69 | 91.90 | −0.2 | 93.34 | −1.7 |
| Macro F1 | 77.08 | 92.17 | −15.1 | 84.79 | −7.7 |
| Se N / S / V | 91.4 / 87.1 / 96.5 | 99.1 / 81.6 / 95.0 | | 95.9 / 88.5 / 95.7 | |
| PPV N / S / V | 99.8 / **32.1** / 82.4 | 99.0 / **82.7** / 95.6 | PPV_S −50.6 | — / 51.5 / — | PPV_S −19.4 |

Our ensemble's DS2 confusion matrix (rows are true N, S, V; columns are predicted N, S, V):

| | N | S | V |
|---|---|---|---|
| N | 40,471 | **3,310** | 478 |
| S | 51 | 1,600 | 186 |
| V | 36 | 76 | 3,109 |

The **3,310 N beats predicted as S** are the whole gap. Sensitivities are on par with or above the paper's; precision is not.

**Re-thresholding cannot close it.** Per-class offsets chosen on DS1-val give acc 95.16 / macro F1 80.68. As a diagnostic only, offsets chosen on DS2 itself top out at about acc 95.4 / macro F1 82.4 / PPV_S 47. The model's S-vs-N separation has to improve.

**Wrong claims from earlier, now corrected:**
1. ACC 98.18 is plain, not one-vs-rest. The one-vs-rest version would be 98.79.
2. SVEB Se 85.3 / VEB Se 96.34 come from **Model 1** (class weights, NK=64). The abstract mixes Model 1 sensitivities with Model 6 averages.
3. Se_V 89.7 and Se_N 97.8 appear nowhere in the paper.
4. tau_S = 0.50 is a no-op.
5. Compare against the paper using the **mean of single seeds**, not our ensemble: Table 4 reports the mean of 10 runs.

#### A. Make the comparison fair — DONE (2026-11-18, farag-gap)
- [x] **A1.** Paper verified (see above).
  - `1DCNN experiments/utils.py` target line fixed to 98.18 / 99.10 / 81.60 / 95.00 / 92.17.
  - `paper_table4.md` rewritten. It had the wrong model configurations.
  - The paper reports no specificity values and no confusion matrix.
- [x] **A2.** Full metric set computed (above).
  - Macro Sp 96.57; weighted F1 93.20.
  - Mean of single seeds: acc 91.03, PPV_S 31.25, macro F1 75.57.
  - DS1/DS2 ensemble and per-seed probabilities saved to `1DCNN/gap_analysis/ensemble_probs.npz`.
- [x] **A3.** The paper uses unweighted macro means over N/S/V; Model 6 reproduces exactly as 92.44 / 91.90 / 92.17. A class-weighted average would give about 98.2, so a weighted average does *not* explain the gap.
- [x] **A4.** Beat counts match paper Table 2 exactly.
  - DS2: N 44,259 / S 1,837 / V 3,221. DS1: 45,866 / 944 / 3,788.
  - F/Q beats are dropped in both the paper and ours, but still used for RR intervals. No other beats are excluded.
- [x] **A5.** XGBoost was **missing from both venvs**, so `--rr-baseline` had been silently skipped. It is now installed privately in `gap_analysis/pylib`.

  | RR-only variant | Acc | Se N / S / V |
  |---|---|---|
  | Paper | 93 | 97 / 60 / 58 |
  | Trained on DS1-train only | 91.64 | 96.5 / 12.7 / 69.3 |
  | Trained on full DS1 | 93.73 | 97.4 / 30.8 / 79.3 |

  Accuracy and N match the paper, so an RR-feature bug is unlikely. The paper doesn't give its XGBoost settings.

**New leads from A1 (now the main hypotheses):**
- **The architecture differs from paper Table 3.**

  | | Paper | Ours |
  |---|---|---|
  | Kernels (3-class) | 11 | 13 |
  | Activation after batch norm | tanh | relu |
  | Conv1D | full 64-sample input, output length kept | cropped to 32 samples, valid padding |
  | RR branch | Dense 32/16/8 applied per RR value, then Flatten to 32 | Dense(16)×3 |
  | Total params | 1,267 | 1,195 |

  Passed to farag-fidelity as the main C1 item.
- **The S template is noisy.** DS1-train has 0 'S'-symbol beats; all of DS1 has 2.
- **B1 (full DS1, no class weights) is the paper's actual Model 6 setup.**
- `src/run_experiment.py` always overwrites `results/run.log`, `resolved_config.yaml`, `beat_counts.md` and the templates figure, even for `--rr-baseline`. Passed to farag-fidelity to fix.

#### B. Reduce N→S/V false positives (main accuracy lever)
- [x] **B1.** *(done 11-19: compact model acc 93.80, F1 70.43, Se_S 31.8; it only works with the Table 3 architecture, see C)* Run full DS1 **without** class weights. This is the paper's stated setup and the only combination not yet tried: Exp5 used balanced weights, and the no-weight run used only 149 S beats.
- [x] **B2.** *(done: p only trades Se_N vs Se_S; PPV_S stays at 31–37)* Sweep the class-weight strength with w_c = (1/freq_c)^p for p ∈ {0, 0.25, 0.5, 0.75, 1}. Balanced (p=1) over-penalises N; p≈0.5 may keep Se_S while restoring Se_N.
- [x] **B3.** *(done: DS1-chosen offsets don't transfer; DS2 ceiling ~95.8 acc / 84 F1)* Calibrate the thresholds for macro-F1 or accuracy instead of F1_S, using per-class logit offsets (bias for S and V) in a 2-D sweep, and plot the Se_S vs accuracy trade-off curve. *(A2 shows the ceiling: about +3.5 acc at most.)*
- [x] **B4.** *(done: the only clear win for the compact model, giving acc 94.84 / F1 81.47 / PPV_S 45.7)* Use **patient-wise validation** for early stopping and threshold tuning: leave-records-out within DS1 (GroupKFold by record). The current 15% random beat hold-out leaks patients, so tau and the stopping epoch are tuned on easier data than DS2.
- [x] **B5.** *(done: the constraint can't be met on patient-wise OOF; see reports/farag-improve.md)* Pick an operating point that holds Se_S ≥ 81.6 and Se_V ≥ 95.0 (Model 6) while maximising Se_N/PPV. Report it next to the max-sensitivity point.

#### C. Pipeline fidelity (if A/B don't close the gap)
- [x] **C1.** *(done: all-DS1 training + Table 3 architecture + per-segment z-score are what matter; most other assumptions are within noise)* Go through the 17 ASSUMPTIONs in `configs/default.yaml` and ablate the high-impact ones: the 128 Hz resampling, window size/centre crop, mean vs z-score normalisation, the RR local/global windows (80/400), template count (13) and template source.
- [x] **C2.** *(done: Adam 1e-3, batch 512, ≤100 epochs; early stopping never fires)* Match the training hyper-parameters exactly where the paper states them: optimizer, LR, batch size, epochs, LR schedule.
- [x] **C3.** *(done: `--grid-table4` added; the Model 1–9 ordering matches, Model 6 is best; Model 3's SVEB 95.6 is not reproduced)* Run `--grid` to reproduce Models 1–6. Check that the relative ordering matches Table 4 (e.g. Model 3: SVEB Se 95.59). A matching ordering means the pipeline is right and only absolute calibration differs.
- [x] **C4.** *(done: best precision, ens acc 98.25 / F1 92.30 / PPV_S 84.4, but it isn't causal so it can't run on the device)* Try per-patient RR normalisation (RR / record mean RR), which helped LightGBM V/A on DS2, as an extra input variant.

#### D. Beyond the paper (only after A–C)
- [ ] **D1.** Knowledge distillation from the KAN-Former (needs the KAN-Former trained first). Soft labels often improve the student's PPV.
- [ ] **D2.** Ensemble the CNN with an RR/LightGBM model (e.g. average the probabilities) for better S precision.

#### E. Experiments to improve accuracy / S precision (added 2026-11-19)

Starting point is the best recipe so far (farag-improve): full DS1, w = (1/freq)^0.75, epoch count from 5-fold GroupKFold by record, refit, argmax. DS2 (mean of 10 seeds): acc 94.84, macro F1 81.47, Se_S 79.6, **PPV_S 45.7**. Model 6 target: acc 98.18, macro F1 92.17, PPV_S 82.7.

The bottleneck is N beats predicted as S, so most of these experiments target S precision. Rules for all of them:
- choose on DS1 only (patient-wise CV);
- report the mean of single seeds and the ensemble;
- DS2 is test only.

**First: understand the errors**
- [ ] **E1. Per-record error analysis of the N→S false positives.** Find which DS2 records produce them, their RR prematurity, and how they look. If a few records dominate (e.g. sinus arrhythmia or noisy records), the fix is targeted rather than global.

**Architecture and inputs**
- [x] **E2.** *(done: with the farag-improve recipe the architecture adds sensitivity, not precision (acc 93.97, Se_S 95.6, PPV_S 40.9). Paper-level precision needs Table 3 + the Model 6 recipe = `p1_faithful.yaml`.)* **Exact Table 3 architecture plus the best training recipe** (11 kernels, tanh, 'same' conv over 64 samples, RR Dense 32/16/8 per value). This separates the architecture effect from the training protocol. farag-fidelity is running it.
- [ ] **E3. Fix the S matched-filter template.** Build it from all AAMI-S beats (A, a, J, S) in the training folds instead of the 'S' symbol, which has 0–2 beats. Also try random initialisation for that one kernel.
- [ ] **E4. Richer RR features.**
  - Add the pre/post RR ratio, RR relative to the record mean (per-patient normalisation, C4), and the local-window variance of RR.
  - Reason: the LSTM went from PPV_S 23 to 61 just by adding RR features.
- [ ] **E5. Input variants:** raw + derivative ("both", as in paper Model 4); a 2-lead input (MLII + V1/V5); a longer window (e.g. 96–128 samples, to capture the P wave, which is key for S beats).
- [ ] **E6. Per-record morphology normalisation:** subtract the record's median beat template, or z-score the beat by record statistics. This removes patient identity, which helped LightGBM's V/A classes.

**Decision and post-processing**
- [ ] **E7. RR prematurity gate.** Only allow an S prediction when the beat is premature (e.g. RR_pre / local mean RR below a threshold chosen on DS1 OOF). Otherwise fall back to the next class. This directly attacks N→S false alarms on regular rhythms.
- [ ] **E8. Two-stage classifier:** P2 binary (N vs abnormal; F1 81.7, PPV 74.6), then a dedicated S-vs-V classifier on the abnormal beats only.
- [ ] **E9. Model ensembles across families:** average the CNN, LSTM+RR (PPV_S 60.6) and LightGBM probabilities, with weights chosen on DS1 OOF. Cheap, since all the models already exist.

**Validation protocol**
- [ ] **E10. Leave-one-record-out (or 11-fold grouped) CV** instead of 5-fold. Record 209 alone holds 383 of the 944 S beats, so 5-fold OOF is very noisy and the epoch and threshold choices suffer.
- [ ] **E11. A learning-rate schedule in the refit** (cosine, or the CV-chosen ReduceLROnPlateau trajectory). The refit currently uses a constant LR.

**More data (no DS2 used)**
- [ ] **E12. Add training patients with S beats from other annotated databases**, resampled to 360 Hz: INCART (257 Hz, 75 records, beat-annotated) and the MIT-BIH Supraventricular Arrhythmia DB (78 records, many S beats). Data was the biggest lever so far (149 → 944 S beats gave Se_S 35 → 85). Train on MIT-BIH DS1 + external, then test on DS2.
- [ ] **E13. Re-test augmentation**, but only label-preserving physiological variants: small time-warp, baseline wander, amplitude drift. The earlier Exp3 used OOD noise and hurt.

**Capacity and edge budget**
- [ ] **E14. Slightly larger student:** 16–24 kernels, 2 conv layers. Check whether capacity limits S precision, while staying within the ESP32 budget (< ~20 KB flash, < 8 KB arena; the current model is 7 KB / 3.4 KB).
- [ ] **E15. Knowledge distillation from the KAN-Former**: on hold (D1) until the KAN-Former finishes.

Suggested order: E1 → E3/E4/E7 (cheap) → E9 → E12 → the rest.

**Update 2026-11-19:**
- The faithful config (`p1_faithful.yaml`, see below) is now the base for all E experiments, instead of the farag-improve recipe.
- Remaining gap to Model 6: acc −0.6, F1 −2.1, PPV_S −8.4. So E3, E4, E7, E9 and E12 now target the last points of S precision.
- E6/C4-style per-record normalisation works offline but isn't causal, so it isn't usable on the ESP32 as is.

---

### Multi-agent run — results (2026-11-18 → 19, GPU box)

Nine agents finished all remaining ML todos except knowledge distillation (paused) and the hardware items.
- Full reports are in `reports/<agent>.md`; code and results are synced into this repo.
- Large models and data stay on the box under `~/Desktop/arrhythmiaClassifier`: `models/`, `*/runs/`, `*/cache/`, `datasets/external/`, `LSTM/data/`.
- Each agent's restart notes are in `RESUME_AGENTS.txt`.

#### ⭐ Farag 2023 reproduced (farag-fidelity, `reports/farag-fidelity.md`)

Three fixes to our pipeline close almost the whole gap:
1. **Train on all of DS1** (51,021 beats). The paper says "more than 50 K examples". With DS1-train alone and no class weights, Se_S = 0 in every variant.
2. **The paper's exact Table 3 architecture.**
   - 64-sample input.
   - Conv1D with 11 filters, kernel 32, `same` padding, then BN, tanh and global max-pool.
   - RR branch: each of the 4 RR values goes through Dense 32/16/8, then Flatten.
   - **1,267 params**, matching the paper exactly.
   - This is the biggest single step: F1 77.9 → 89.6, PPV_S 59 → 75.
3. **Per-segment z-score normalisation.** Mean-subtraction is a no-op on the derivative input.

Plus no class weights (as in Model 6) and at most 100 epochs. Config: **`HasanML/paper3/configs/p1_faithful.yaml`**.

| DS2 | Faithful (FZ), 5 seeds | FZ ensemble | Paper Model 6 |
|---|---|---|---|
| Plain acc | **97.60 ± 0.3** | 98.01 | 98.18 |
| Macro F1 | 90.07 ± 0.9 | 91.64 | 92.17 |
| Macro PREC | 88.87 | — | 92.44 |
| Se N / S / V | 98.5 / 80.7 / 95.3 | — | 99.1 / 81.6 / 95.0 |
| PPV S / V | 74.3 / 93.3 | — | 82.7 / 95.6 |

**Other findings:**
- **Early stopping never fires.** Validation loss is still falling at epoch 100, so FZ is effectively "all DS1, 100 epochs". Neither DS1-val nor DS2 influences the model.
- **C3 grid:** the 9-model `--grid-table4` reproduces the paper's ordering (Model 6 best, Model 7 second). The class-weighted Models 1–5 have Se_S 88–95 but PPV_S 33–37. Model 3's SVEB 95.6 is not reproduced.
- **C1:** most assumptions don't matter (resampling, padding, 11 vs 13 templates, RR windows). Clearly worse: frozen templates, a 96-sample window, "signal"/"both" inputs, pre-RR only.
- **C4:** per-record RR normalisation gives the best precision (ensemble acc 98.25, F1 92.30, PPV_S 84.4, above the paper), but it needs the whole recording, so it isn't causal. The causal version is worse.
- **Code:** `paper3/src` was changed additively (old defaults kept; 34 tests pass). The originals are in `1DCNN/fidelity_C/backup_src_orig/`. Bugs fixed: `--grid` didn't vary anything, the summary printed "—" in every cell, the P2/P3 threshold, the 2-class AUC, and an empty S template.

#### Farag improvement experiments (farag-improve, `reports/farag-improve.md`)

These use the old compact architecture.
- Patient-wise 5-fold CV for the epoch count, w = (1/freq)^0.75, refit: acc 94.84, F1 81.47, PPV_S 45.7, up from official P1's 91.03 / 75.57 / 31.3.
- The 15% random-beat hold-out leaks patients (93–95 vs 62–71 macro F1).
- Class weights and logit offsets only trade Se against PPV.
- **Superseded by the faithful config above.**

#### ESP32 export (farag-export, `reports/farag-export.md`, `1DCNN/export/`)

- **INT8 PTQ is lossless:** 7 KB flash, ≤ 3.4 KB arena, about 1.1k params, 5 op types, and TFLite-Micro agreement 99.95%. QAT and 50% pruning give no gain (pruning saves no space).
- **P2 binary (improved recipe, ensemble):** F1 81.7, AUC 0.978, Se 90.2, PPV 74.6.
- **Fixed in `export.py`:** wrong calibration input order (TFLite uses [rr_in, ecg_in]) and an RR-outlier range issue in QAT.
- **New environment:** `.venv-export` (Python 3.12, TF 2.19, tfmot).
- **TODO:** re-export using the `p1_faithful` model.

#### KAN-Former (kanformer, `reports/kanformer.md`)

- Trained on the GPU (12–20 s per epoch) after fixing **10 bugs**. Among them: KAN forward crashed; the sampler filled ~86% of each batch with Q beats; focal weights were applied twice; `--seeds 3` trained only seed 3.
- **DS2 (best, Fix B ensemble):** 3-class acc 94.6, F1 70.0, Se N/S/V 97.7 / 25.0 / 90.9, PPV_S 34.6. 5-class acc 91.8, F1 42.2.
- **It misses about 75% of S beats.** It was trained on DS1-train (149 S beats), and an RR-only LightGBM on the same split does twice as well on S. **Not a good S teacher yet.** Next: train on all of DS1.
- **Ablations** (seed noise is large):
  - removing RR is worst;
  - an MLP head instead of KAN is as good or better;
  - removing the context window is slightly better;
  - a single kernel size is about the same.
- **Exported:** soft labels T ∈ {1, 2, 4} for DS1/DS2 (keyed by record_id + r_sample), plus ONNX and TorchScript, in `paper4/results/teacher/`.

#### LSTM baseline (lstm, `reports/lstm.md`, `LSTM/`)

- **Model:** strided conv, then a 2-layer BiLSTM; 193k params, 720×2 input.

| DS2 | Acc | Macro F1 | Se_S | PPV_S | Se_V |
|---|---|---|---|---|---|
| N/S/V conv-BiLSTM (refit, 5 seeds) | 92.0 | 62.7 | 13.1 | 23.2 | 87.0 |
| N/S/V +RR features, ensemble | 92.9 | 78.2 | 73.7 | 60.6 | 92.3 |

- **N/L/R/V/A:** the random split reaches 99.5 (= LightGBM). On DS1→DS2, 73–80 acc and L/R Se ≈ 0, the same failure as LightGBM.
- **Conclusion:** the waveform alone can't detect S across patients; the RR features are essential.

#### Zero-shot generalisation (zeroshot, `reports/zeroshot.md`, `zeroshot/`)

- **Data:** PTB-XL (8,226 records) and Chapman (8,541 records), lead II, no retraining.
- **R-peak detector:** XQRS + snap, Se 97.9 / PPV 99.7 on MIT-BIH DS2.
- **Record-level Se / Sp:**

| Condition, model | PTB-XL | Chapman |
|---|---|---|
| PVC, CNN | 84.7 / 95.3 | 77.7 / 93.5 |
| PVC, LGBM | 90.9 / 92.0 | 83.3 / 92.6 |
| PAC, CNN | 71.5 / 94.4 | 75.8 / 96.1 |
| LBBB, LGBM (Se only) | 45 | 41 |
| RBBB, LGBM (Se only) | 5 | 4 |

- **Short records are the main cause of the drop:** on 10-s strips of MIT-BIH itself, CNN Se_S falls 87 → 45, because the 80/400-beat RR means have almost no history.
- Specificity rises on the external data because its normals come from healthy people.
- These used the old P1 CNN. TODO: rerun with `p1_faithful`.

#### LightGBM (lgbm-groupcv + lgbm-binary; now paused)

- **Grouped-by-record CV explains about 97% of the DS1→DS2 gap** (CV 0.986 → grouped 0.406 vs DS2 0.388).
- **Best DS2 results:**
  - 5-class: acc 0.774, macro F1 0.423 (grouped Optuna + regularisation).
  - Binary: F1 0.519, AUC 0.715 (grouped-OOF threshold).
- **Why LBBB/RBBB fail:**
  - Only 2 + 2 DS1 patients carry these labels.
  - DS1 and DS2 beats of the same class are separable with AUC 0.99999.
  - Several features flip direction between DS1 and DS2.
  - QRS delineation fails on DS2 patient 111.
- **The paper's 0.989 matches our random split (0.996), not inter-patient.**
- **Ablation:** A2–A4 still beat the paper on the random split. On DS1→DS2, A2 (statistical + fiducial) is best.

#### Corrections and housekeeping

- The weekly summary (`weekly_summary_2026-11-11_to_11-18.txt`) and `notes.txt` still quote the old wrong targets (Se_V 89.7 / Se_N 97.8 / OvR accuracy). The correct Model 6 targets are in the gap section above.
- **XGBoost was missing** from both box venvs, so `--rr-baseline` had been silently skipped. It is now installed privately in `1DCNN/gap_analysis/pylib`.

---

## Left to do

### ML pipeline

- [x] **Re-run full pipeline** after normalisation fixes: `features.py` → `split.py` → `train.py` → `evaluate.py` (done 2026-11-14 / eval 2026-11-18)
- [x] **Verify DS1/DS2 recovery**: *closed.* V/A improved; LBBB/RBBB still F1 ≈ 0. lgbm-groupcv showed this is a data limit: DS1 has only 2 LBBB and 2 RBBB patients. LightGBM is paused for now (decision 2026-11-19).
- [x] **Patient-grouped CV** (GroupKFold by record) for Optuna on DS1: done 11-19 (lgbm-groupcv). Best DS2 result: acc 0.774, macro F1 0.423.
- [x] **Binary LightGBM** (`source/binaryLightGBM/`) on normalised features: done 11-19 (lgbm-binary). DS2 F1 0.519, AUC 0.715.
- [x] **Farag 2023 1D-CNN** (`HasanML/paper3/`) — full replication with TFLite/ESP32 export; 34 tests pass. P1 training run on university GPU (2026-11-13); see results below.
- [x] **LSTM baseline**, input shape `(N, 720, 2)`: done 11-19 (`LSTM/`). Plain Se_S 13%; the +RR variant reaches macro F1 78.2.
- [x] **KAN-Former (teacher)** (`HasanML/paper4/`) — PyTorch, 2-branch CNN + Transformer + KAN head; ~1M params; cloud-only. Full implementation complete: 14 source files, 34+ unit tests, Colab notebook, bash script, distillation export. **Trained on GPU 11-19 after 10 bug fixes:** 3-class acc 94.6, F1 70.0, but Se_S only 25%, so it is not yet a useful S teacher. Next: retrain on all of DS1.
- [ ] **Knowledge distillation** *(on hold by decision; teacher soft labels are exported in `HasanML/paper4/results/teacher/`, keyed by record_id + r_sample)* — freeze KAN-Former (or paper3 1D-CNN if it's the better teacher); train Keras student in paper3 with `L = α·CE + (1−α)·KL`. Soft labels saved by teacher at T∈{1,2,4} as `results/teacher/soft_labels_T{t}.npy`.
- [x] **Quantisation/pruning**: done 11-19 (farag-export). INT8 PTQ is lossless (7 KB flash, ≤3.4 KB arena); QAT and pruning give no gain. **Still to do:** re-export with the new `p1_faithful` model and measure on a real ESP32-S3.
- [ ] **Evaluation** — macro-F1, balanced accuracy, per-class recall for all models; edge metrics (latency, RAM, flash, energy) for deployed model

### Zero-shot generalisation

- [x] Download PTB-XL and Chapman-Shaoxing datasets (Lead II only), stored on the GPU box in `datasets/external/`
- [x] Run trained MIT-BIH model on PTB-XL and Chapman without retraining: done 11-19 (zeroshot); see the results section above

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
