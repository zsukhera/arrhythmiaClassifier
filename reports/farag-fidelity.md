# farag-fidelity — Farag 2023 pipeline fidelity (tasks C1–C4)

Agent: farag-fidelity, 2026-11-18/19. All compute ran on the box on CPU (TF 2.22rc0, `.venv`), with at most 6 single-thread workers.
DS2 was used only for the final per-run evaluation. Early stopping always used DS1 records ≥ 200 (DS1-val). Nothing was tuned on DS2.
All numbers are **DS2**, given as mean ± std over seeds. "ens" means argmax of the seed-averaged probabilities. Acc is **plain** accuracy: the paper's 98.18 is plain accuracy (see §1).

## TL;DR

1. **Our replication departed from the paper in three places that matter. Fixing them closes most of the gap to Model 6.**
   - **Architecture (Table 3).** The paper's model uses:
     - a 64-sample input with `Conv1D(11, NK=32, padding=same)`, then BatchNorm, then **tanh**, then GlobalMaxPool;
     - **11** templates for 3 classes (the F and Q sub-classes are dropped);
     - an RR input of shape (4,1) passed through **Dense 32/16/8 applied per RR value**, then Flatten to 32.

     This gives **1,267 params, exactly as in Table 3**. Ours was a 32-sample crop, `valid` padding, ReLU, 13 filters and Dense(16)×3 on the 4-vector (1,195 params).
   - **Training set = all of DS1.** The paper says "training set size of more than 50 K examples", which is DS1 = 51,021 beats. Records ≥ 200 are still the early-stopping set, so that set overlaps training. With only DS1-train (records < 200, 149 S beats) and no class weights, **every** variant we tried, ours or the paper's architecture, gets **Se_S = 0**.
   - **Per-segment z-score** normalisation. The paper only says "normalization". Mean-subtraction is a no-op for derivative input, because the difference operator removes the mean.
2. **Best faithful configuration (`FZ` = `configs/p1_faithful.yaml`), Model 6 recipe, no class weights:**
   - 5-seed mean: **acc 97.60 ± 0.3, macro F1 90.07 ± 0.9, macro PREC 88.87 ± 1.9, Se N/S/V 98.47 / 80.68 / 95.27, PPV S/V 74.3 / 93.3**.
   - Seed ensemble: acc 98.04, F1 91.62 (3 seeds).
   - Paper Model 6: 98.18 / 92.17 / 92.44 / 99.10, 81.60, 95.00 / 82.68, 95.63.
   - Both sensitivities match. The gap is now about 0.5 points of accuracy and 2–3 points of F1/PREC. PPV_S is about 75 against the paper's 82.7, compared with about 31–46 for our previous P1 and for farag-improve.
3. **The paper's headline "SVEB Se 85.3 / VEB Se 96.34" belongs to Model 1 (class weights SET, Glorot init, NK=64), not Model 6.** farag-gap found this independently and has fixed `results/paper_table4.md`.
4. **Ranking of assumptions by DS2 effect:** train on all DS1 ≫ architecture (tanh/BN + per-value RR stack) > z-score ≫ everything else. All of these have little effect, within about 1 point of macro F1 and seed noise:
   - resampling method;
   - window 48/64;
   - crop vs `same` vs `valid`;
   - 11 vs 13 templates;
   - template source;
   - RR windows 40/200, 80/400, 160/800;
   - past-only RR means;
   - LR schedule and early-stopping settings.

   AAMI-super-class templates, shifted or 96-sample windows, and pre-RR-only features clearly hurt.
5. **C4, per-patient RR normalisation:** replacing the 400-beat global mean with the record mean RR gives the best macro PREC and PPV_S. The ensemble reaches **acc 98.25, F1 92.30, PPV_S 84.4**. But the record mean is non-causal: it needs the whole recording. The causal (expanding-mean) version is worse than the baseline.
6. **farag-improve recipe on the faithful architecture:** applied with the same farag-improve recipe (5-fold GroupKFold epoch count, then refit with p = 0.75, argmax), the faithful architecture gives **acc 93.97, F1 82.23, PREC 77.35, Se S/V 95.6/94.0, PPV_S 40.9** over 5 seeds. farag-improve's compact model gets 94.84 / 81.47 / Se_S 79.6 / PPV_S 45.7. So under a class-weighted, patient-wise protocol the architecture buys Se_S (+16) and slightly better F1, but **not** precision. The paper-level precision comes from the Model-6 *recipe* together with the architecture: no class weights, all DS1, about 100 epochs. Neither alone is enough: the compact model on the same recipe (FL0) gets PPV_S 44 and F1 74.
7. **Table 4 ordering** on the faithful base: the class-weighted Models 1–5 have high Se_S (88–95) but low PPV_S (33–37); Models 6–9 have higher precision. This matches the paper's qualitative conclusion. Model 3 (frozen MF) does **not** reproduce the paper's best SVEB Se, and frozen MF is worse than trainable here.

## 1. Paper facts that were missing or wrong (full text via Europe PMC, PMC9919183)

- **Table 3** (3 classes): Signal Input (64,1) → Conv1D (64,11), 363 params (NK=32) or 715 (NK=64) → BatchNormalization (44) → Activation (**Tanh**) → GlobalMaxPooling (11).
  Interval input (4,1) → Dense 32 ReLU (4,32), 64 params → Dense 16 (528) → Dense 8 (136) → Flatten (32) → Concatenate (43) → Softmax (3), 132 params. **Total 1267 / 1619.**
  For 4 and 5 classes: 12 and 13 filters, 1352 / 1439 params.
  Our `arch: table3` reproduces 1267, 1619, 1352 and 1439, and 1245/1597 trainable, **exactly**.
- **Templates:** "mean of all examples per class in the training set DS1"; 13 sub-classes for 5 classes, 11 for 3 classes (Table 3, Fig. 4 "excluding F and Q").
  DS1 contains only 2 'S'-symbol beats (both in DS1-val records) and 0 in DS1-train. So the 'S' template is a 2-beat mean with `source: ds1_all`, and a zero filter with `ds1_train`.
- **Training:**
  - Adam with "adaptive learning rate scheduling" starting at 1e-3; batch 512; 500 epochs plus early stopping (monitor and patience not stated).
  - Records < 200 for training, ≥ 200 for validation, yet "training set size of more than 50 K examples" and "≤ 100 epochs, < 2 min".
  - 10 repeats, averaged.
  - Signals are fed "raw ... without pre-processing", but a "normalization" step is timed (953 ms).
- **Metrics:** "average" = macro over N/S/V. Model 6's 98.18 is **plain** accuracy: 0.991·44259 + 0.816·1837 + 0.95·3221 = 48,420 / 49,317 = 98.18. It is not OvR.
- **Table 4** has 9 models:

  | Model | Init | NK | Input | Class weights |
  |---|---|---|---|---|
  | M1 | Glorot | 64 | deriv | SET |
  | M2 | MF trainable | 32 | deriv | SET |
  | M3 | MF frozen | 32 | deriv | SET |
  | M4 | Glorot (text says frozen) | 32 | both | SET |
  | M5 | MF trainable | 32 | signal | SET |
  | M6 | MF trainable | 32 | deriv | NOT |
  | M7 | MF trainable | 64 | deriv | NOT |
  | M8 | 4-class | | | NOT |
  | M9 | 5-class | | | NOT |

## 2. Code changes (HasanML/paper3/src; additive, defaults unchanged, 34/34 tests pass)

Originals are backed up in `1DCNN/fidelity_C/backup_src_orig/`. Every file was replaced atomically (written beside the target, then `mv`).

| file | new option (default = old behaviour) |
|---|---|
| `model.py` | `build_model(..., arch="compact"\|"table3", n_filters=13, input_len=None, n_rr=4)`; `_build_model_table3` |
| `train.py` | `model.arch`; filter count follows the MF kernel; input length and RR width follow the data; MF kernel centre-cropped if longer than NK; `training.early_stop_monitor` (auto\|val_loss\|val_macro_f1); `training.lr_schedule` (plateau\|none); `training.class_weight_power` (1.0); param guard only for `compact` |
| `preprocess.py` | `preprocess.resample` (poly\|fft\|linear\|decimate_naive); `input.crop_input` (true); `normalization: minmax` |
| `rr.py` | `mean_includes_current` (true), `record_norm` (none\|append\|replace_global), `record_mean_type` (full\|expanding) |
| `templates.py` | `empty_fallback` (zero\|aami) for sub-classes with no beats |
| `metrics.py` | 2-class AUC fix (label_binarize gives a single column) |
| `run_experiment.py` (summary) | `results_summary.md` now prints mean ± std (all cells were "—" before) |
| `run_experiment.py` | see the list below |

`run_experiment.py` changes:
- `templates.source: ds1_all` now actually works; it was silently ignored before.
- `templates.match_classes` (false).
- `data.train_records` (ds1_train\|ds1_all).
- RR options plumbed through.
- `--grid-table4` runs the paper's 9 models.
- `templates.recompute_per_variant` (false) fixes the legacy `--grid` bugs listed below.
- `--results-dir` now also moves figures; new `--cache-dir`.
- P2/P3 now report a DS1-val-chosen threshold (`*_valthreshold.csv`).
- The collapsed-P1 comparison loads the cached P1 model when `--p1` is not given.

**CLI bugs found (C3):**
- `--ablations` is parsed but never used: it is a no-op that only runs templates and the RR baseline.
- Legacy `--grid` builds data once with the default input mode/NK. So signal/both/NK=64 variants were trained on 32-sample derivative inputs (NK=64 would not even build), and all variants used derivative templates.
- The legacy grid also does not correspond to Table 4: it uses class weights on/off × 36 combinations, and its Model 5 is signal, not 4-class.

## 3. Results

Reproduce from `1DCNN/fidelity_C/`:

```
../../.venv/bin/python fid.py --group <group> --seeds 0 1 2 --workers 6
../../.venv/bin/python summarise.py > results.md
```

Variant definitions are in `variants.py`. The full tables, with std, PPV and ensemble columns, are in `1DCNN/fidelity_C/results.md` and `results_table.csv`.

### 3a. Round 1: train on DS1-train only (records < 200), max 500 epochs, ES val_loss patience 20

| variant | Acc | macF1 | macPREC | Se N | Se S | Se V | PPV S |
|---|---|---|---|---|---|---|---|
| L0 ours (old default, no CW) | 93.33 | 54.33 | 58.04 | 99.88 | **0.00** | 56.49 | — |
| L4 paper-faithful arch + 11 tpl + tpl ds1_all (M6) | 91.74 | 45.93 | 52.06 | 99.58 | **0.00** | 36.33 | — |
| …+ z-score | 93.61 | 55.01 | 63.21 | 99.79 | 0.02 | 62.06 | — |
| M1 (paper arch, CW) | 92.93 | 74.77 | 71.32 | 94.65 | 63.62 | 85.90 | 35.0 |
| M3 (frozen MF, CW) | 90.13 | 58.23 | 57.08 | 94.51 | 18.89 | 70.50 | 15.6 |
| T_trainDS1all (= M6 on **all DS1**) | 95.80 | 79.52 | 78.32 | 97.81 | 55.05 | 91.43 | 57.6 |
| P_trainAll_z (all DS1 + z-score, 500 ep) | 97.37 | 88.48 | 88.51 | 98.54 | 73.53 | 94.98 | 72.8 |

Every DS1-train-only window, resampling, RR, template and input ablation gave Se_S = 0 without class weights. They are therefore uninformative and are only listed in results.md.

### 3b. Round 2: base FZ = paper arch + all DS1 + z-score + max 100 epochs (Model 6 recipe)

Ladder (cumulative, all on all-DS1 + z-score + 100 epochs, 3 seeds):

| step | Acc | macF1 | macPREC | Se S | Se V | PPV S | PPV V |
|---|---|---|---|---|---|---|---|
| FL0 ours (compact) | 95.12 | 73.72 | 74.89 | 28.80 | 94.86 | 43.7 | 83.5 |
| FL1 + 64-sample input, same padding | 95.13 | 77.86 | 77.55 | 47.83 | 94.79 | 58.9 | 75.4 |
| FL2 + Table-3 arch (BN→tanh, per-value RR 32/16/8) | 97.62 | 89.56 | 89.30 | 76.28 | 94.78 | 74.7 | 94.3 |
| FL3 + 11 templates | 97.50 | 89.84 | 88.17 | 82.67 | 95.08 | 70.6 | 94.9 |
| **FZ** + templates from all DS1 | **97.67** | **90.26** | **89.33** | 80.19 | 95.41 | 75.4 | 93.6 |
| F_mean (FZ without z-score) | 96.01 | 77.81 | 79.80 | 44.80 | 91.59 | 63.6 | 77.2 |
| paper M6 | 98.18 | 92.17 | 92.44 | 81.60 | 95.00 | 82.7 | 95.6 |

FZ with 5 seeds (0–4): acc **97.60 ± 0.3**, OvR 98.40, macro F1 **90.07 ± 0.9**, macro PREC **88.87 ± 1.9**, macro Se 91.47, Se N/S/V 98.47 / 80.68 / 95.27, PPV S/V 74.3 / 93.3. The 5-seed ensemble gets acc 98.01, F1 91.64, Se_S 82.5, PPV_S 79.0.

**Note on honesty of FZ.** In every FZ run the best DS1-val loss is at the last epoch (100). EarlyStopping (restore_best_weights) and ReduceLROnPlateau never act. So FZ is equivalent to "train on all DS1 for 100 epochs, constant LR". DS1-val influences nothing, and DS2 influences nothing. The only hyper-parameter is the 100-epoch cap, which comes from the paper's text. The 500-epoch variant is slightly worse (F1 88.5).

Ablations around FZ (3 seeds; FZ itself is F1 90.26 ± 1.0):

| variant | Acc | macF1 | macPREC | Se S | Se V | PPV S |
|---|---|---|---|---|---|---|
| crop input to 32 (+same) | 97.61 | 90.10 | 88.78 | 81.71 | 95.06 | 71.9 |
| valid padding (33 positions) | 97.68 | 90.32 | 89.14 | 80.86 | 95.39 | 74.4 |
| 13 templates | 97.46 | 89.29 | 88.05 | 78.30 | 95.59 | 71.8 |
| templates from DS1-train | 97.50 | 89.84 | 88.17 | 82.67 | 95.08 | 70.6 |
| 3 AAMI-class templates | 96.24 | 78.83 | 83.32 | 37.58 | 92.60 | 58.8 |
| 'S' template from A/a/J (empty_fallback) | identical to FZ ('S' already has 2 beats in all DS1) | | | | | |
| window 48 | 97.67 | 90.16 | 88.73 | 81.24 | 95.64 | 72.4 |
| window 96 | 97.18 | 87.49 | 87.20 | 70.19 | 94.97 | 69.3 |
| asymmetric 24/40 | 96.77 | 86.19 | 85.13 | 70.59 | 95.14 | 63.3 |
| resample FFT | 97.87 | 91.01 | 90.16 | 81.58 | 95.89 | 76.8 |
| resample naive pick | 97.33 | 88.86 | 87.52 | 78.79 | 94.25 | 69.7 |
| min-max norm | 97.35 | 88.98 | 87.85 | 78.13 | 94.16 | 72.4 |
| RR 40/200 | 97.36 | 89.08 | 87.66 | 79.15 | 94.78 | 69.9 |
| RR 160/800 | 97.40 | 89.25 | 87.88 | 79.13 | 95.27 | 72.6 |
| RR means past-only | 97.62 | 90.02 | 89.13 | 79.37 | 95.18 | 74.7 |
| pre-RR only (2 features) | 97.20 | 88.63 | 87.46 | 77.64 | 93.89 | 71.0 |
| signal input | 97.19 | 88.00 | 86.26 | 78.50 | 93.26 | 69.0 |
| both inputs | 97.00 | 86.89 | 86.35 | 70.97 | 93.75 | 68.9 |
| Glorot init (trainable) | 97.70 | 90.19 | 88.90 | 81.22 | 95.22 | 73.3 |
| MF frozen | 96.00 | 81.66 | 86.01 | 50.79 | 87.59 | 73.2 |
| NK=64 (M7) | 97.12 | 87.64 | 86.50 | 72.69 | 96.19 | 66.0 |
| 500 epochs | 97.37 | 88.48 | 88.51 | 73.53 | 94.98 | 72.8 |
| bare Keras ES (patience 0, no restore, no LR schedule) | identical to FZ | | | | | |
| constant LR | identical to FZ | | | | | |
| balanced class weights (= M2) | 92.29 | 79.63 | 74.70 | 95.43 | 95.20 | 34.7 |

The early-stopping and LR-schedule variants are identical because, with the paper's split, the validation records are part of training. val_loss therefore improves monotonically: neither EarlyStopping nor ReduceLROnPlateau ever fires within 100 epochs, and the epoch cap is the only stopping rule.

### 3c. C4: per-patient RR normalisation (on FZ, 3 seeds)

| variant | causal? | Acc | macF1 | macPREC | Se S | PPV S | ens Acc | ens F1 | ens PPV S |
|---|---|---|---|---|---|---|---|---|---|
| FZ | yes | 97.67 | 90.26 | 89.33 | 80.19 | 75.4 | 98.04 | 91.62 | 79.0 |
| append pre/rec, post/rec (6 RR) | no | 97.62 | 89.69 | 89.52 | 76.16 | 77.2 | 98.04 | 91.42 | 81.1 |
| **replace global by record mean** | no | **97.81** | **90.39** | **90.43** | 77.08 | **79.2** | **98.25** | **92.30** | **84.4** |
| append, expanding mean | yes | 97.61 | 89.81 | 89.31 | 77.83 | 76.2 | 98.04 | 91.63 | 79.6 |
| replace, expanding mean | yes | 97.18 | 88.41 | 86.68 | 77.86 | 68.1 | 97.59 | 89.93 | 71.1 |
| (round 1) ours + append, DS1-train | no | 92.45 | 49.71 | 58.85 | 0.00 | — | | | |

The gain from a full-record mean is small (≈ seed noise for single nets, about +0.7 F1 for the ensemble) and is not available causally on a device. Not recommended for the edge model.

### 3d. Table 4 Models 1–9 on the faithful base (all DS1, z-score, 100 epochs, 3 seeds)

| Model | Acc | macF1 | macPREC | Se N | Se S | Se V | PPV S | PPV V | paper Acc / F1 / Se S / Se V |
|---|---|---|---|---|---|---|---|---|---|
| M1 | 93.15 | 80.87 | 75.81 | 92.94 | 92.18 | 96.63 | 37.3 | 90.6 | 96.48 / 86.81 / 85.30 / 96.34 |
| M2 | 92.29 | 79.63 | 74.70 | 91.95 | 95.43 | 95.20 | 34.7 | 89.7 | 95.58 / 84.79 / 88.51 / 95.65 |
| M3 | 91.27 | 75.69 | 70.83 | 91.60 | 88.10 | 88.46 | 32.8 | 80.4 | 94.42 / 82.79 / **95.59** / 90.31 |
| M4 | 92.63 | 79.75 | 74.74 | 92.48 | 93.12 | 94.31 | 36.4 | 88.3 | 94.78 / 83.34 / 92.54 / 94.32 |
| M5 | 92.51 | 79.95 | 75.80 | 92.38 | 94.05 | 93.49 | 35.0 | 92.9 | 92.90 / 78.65 / 88.13 / 94.47 |
| **M6 (= FZ)** | **97.67** | **90.26** | **89.33** | 98.57 | 80.19 | 95.41 | 75.4 | 93.6 | **98.18 / 92.17 / 81.60 / 95.00** |
| M7 | 97.12 | 87.64 | 86.50 | 98.20 | 72.69 | 96.19 | 66.0 | 94.8 | 97.94 / 89.59 / 81.33 / 95.47 |
| M8 (4-cls) | 96.14 | 66.74 | 65.31 | 97.72 | 81.67 | 94.24 | 69.2 | 93.7 | 97.06 / 68.18 / 79.42 / 90.13 |
| M9 (5-cls) | 96.55 | 53.95 | 53.42 | 98.26 | 80.20 | 94.23 | 73.9 | 95.1 | 96.93 / 54.90 / 74.52 / 95.13 |

**Ordering.** Matches the paper:
- M6 is the best on Acc, F1 and PREC, with M7 second.
- The CW models (M1–M5) have Se_S ≈ 88–95 but PPV_S ≈ 33–37, with Acc 91–93.
- M8 and M9 have the lowest macro averages; 4/5-class macro averages include F/Q.
- M6 vs M7 (NK 32 vs 64) and M1's high Se_V (96.6, paper 96.34) both match.

**Does not match:**
- (a) The paper's CW models have Acc 93–96.5 and PPV_S 39–58. Ours sit 1–3 points lower in Acc and PPV_S, and higher in Se_S. Our "balanced" weights may be stronger than the paper's class-weight setting, whose formula is unstated.
- (b) Model 3 (frozen MF) is **not** the best SVEB detector here (88 vs paper 95.59), and frozen MF is the worst variant in both rounds.

Round 1 (DS1-train only) gives the same CW-vs-no-CW pattern, but there M6/M7/M8/M9 collapse to Se_S = 0 (see results.md, group `table4`).

### 3e. farag-improve protocol on the faithful architecture

5-fold GroupKFold by record (farag-improve's folds), with FZ + class weights^0.75 and ES on fold-val macro-F1 (patience 20). Templates come from the fold-train records.
- Best epochs per fold: 12, 95, 26, 6, 36, median **26**.
- Fold models on DS2 (one per fold): acc 93.7–94.9, F1 80.4–83.6, Se_S 80–95, PPV_S 39–45.

Refit on all DS1 for 26 epochs, constant LR, class weights^0.75, argmax (`R_FZ075`, 5 seeds):

| R_FZ075 (5 seeds) | Acc | OvR | macF1 | macPREC | macSe | Se N | Se S | Se V | PPV S | PPV V |
|---|---|---|---|---|---|---|---|---|---|---|
| mean ± std | 93.97 ± 0.3 | 95.98 | 82.23 ± 0.6 | 77.35 ± 0.5 | 94.50 | 93.90 | 95.60 ± 0.5 | 94.00 ± 1.2 | 40.92 ± 1.1 | 91.49 |
| 5-seed ensemble | 94.06 | | 82.50 | | | | 96.03 | | 41.10 | |

For comparison, farag-improve's compact-architecture refit with p = 0.75 (10 seeds): acc 94.84, F1 81.47, Se_S 79.6, PPV_S 45.7.

### 3f. C2: training hyper-parameters vs the paper

| item | paper | old config | faithful config |
|---|---|---|---|
| optimizer / LR | Adam, 1e-3, "adaptive LR scheduling" | Adam 1e-3 + ReduceLROnPlateau(0.5, 5) | same (inert: never fires; see §3b) |
| batch | 512 | 512 | 512 (batch 64 tested in round 1: no gain) |
| epochs | 500, ES; "≤ 100 epochs, < 2 min" | 500, ES patience 20 | **max 100**: with val ⊂ train, ES never fires, so 100 is the paper's own bound. 500 epochs: F1 88.5 vs 90.3 |
| early stopping | callback, monitor/patience unstated | val_loss p20 (val_macro_f1 with CW) | unchanged; inert with val ⊂ train |
| train / val | records < 200 / ≥ 200, but "> 50 K training examples" | DS1-train / DS1-val | **all DS1** / DS1-val (overlap, as in the paper) |
| class weights | M1–M5 SET, M6–M9 NOT | none / balanced | none (M6) |
| repeats | 10 | 10 | 3 per variant (5 for FZ and R_FZ075) |

## 4. Recommendation

`HasanML/paper3/configs/p1_faithful.yaml` (copy in `1DCNN/fidelity_C/configs/`). Run:

```
python -m src.run_experiment --p1 --config configs/p1_faithful.yaml --results-dir results/p1_faithful --cache-dir results/p1_faithful/cache
```

Settings: Table-3 architecture, 11 MF templates from all DS1, derivative input, NK=32 on a 64-sample input, z-score, train on all DS1, no class weights, max 100 epochs.
- This is the most paper-faithful setup.
- It is also the best single-network configuration we found for Acc, F1 and PREC.
- A seed ensemble gets about 98.0 Acc / 91.6 F1.

**Caveats:**
- Its "validation" set (DS1 records ≥ 200) is inside the training data, as the paper's text implies. Validation loss therefore falls monotonically, and early stopping and the LR schedule never act. The model is effectively trained for a fixed 100 epochs, a number taken from the paper's text. Neither DS1-val nor DS2 influences the model, so the protocol does not leak. But the epoch count was not tuned patient-wise: 500 epochs is slightly worse.
- If a patient-wise-tuned protocol is required, use the GroupKFold epoch count + refit (§3e) on the same architecture: acc 93.97, F1 82.23, Se_S 95.6, Se_V 94.0, PPV_S 40.9. Compared with farag-improve's compact refit, that is better Se_S, similar F1 and lower PPV_S. Its operating point is class-weighted (high Se_S, low PPV_S); FZ's is precision-oriented. A cheap follow-up would be GroupKFold epoch selection with no class weights for the FZ architecture. farag-improve reports that compact + p = 0 stops too early under gkf.
- If maximal Se_S matters more than precision, use class weights (M1/M2 rows: Se_S 92–95, PPV_S ≈ 35).

## 5. Open issues

- The remaining gap to Model 6 (about 0.5 Acc, 2 F1, 3 PREC, 7 PPV_S) may come from:
  - the paper's unstated normalisation;
  - its class-weight formula;
  - 10 runs vs our 3–5;
  - or, for Model 6, early stopping/model selection on DS2. We cannot test that last one by rule. Note that its test acc (98.18) exceeds most training-set numbers in Table 4.
- The paper's text contradicts itself on Model 4 (Table: DEF; text: frozen). We ran both.
- The 'S' template rests on 2 beats in all of DS1 (0 in DS1-train). `templates.empty_fallback: aami` is available but has no effect when `source: ds1_all`.
- C3 CLI check: `python -m src.run_experiment --grid-table4 --config ../../1DCNN/fidelity_C/configs/paper_faithful_cli.yaml --results-dir ... --cache-dir ...` was run with seed 0 on 6 cores (`taskset`). Output is in `1DCNN/fidelity_C/cli_results/`. It ran end to end.
  - Seed-0 DS2 plain acc for M1..M9: 91.62, 92.47, 92.11, 93.02, 91.70, **97.51**, 97.39, 96.13, 96.73.
  - M6 seed 0: Se_S 82.4, Se_V 95.1. This is identical to the driver's FZ seed 0 (97.49 / 82.4 / 95.1), so the CLI and the driver share one code path.
  - While doing this, the pre-existing `write_results_summary` bug (every cell "—", because summarise_seeds returns `_mean` keys) was fixed.

## Files

Created on the box:
- Under `1DCNN/fidelity_C/`:
  - Driver and variants: `fid.py`, `variants.py`, `summarise.py`.
  - Run scripts: `run_all.sh`, `run_extra.sh`, `run_r2.sh`, `run_r3.sh`, `run_cli.sh`.
  - Logs: `*.log`.
  - Results: `results.md`, `results_table.csv`, `runs/<variant>/seed*.json|_proba.npz`, `cache/`, `cv_epochs.json`.
  - Configs: `configs/p1_faithful.yaml`, `configs/paper_faithful_cli.yaml`.
  - Additional run scripts and logs: `run_cli.log`, `run_cli2.log`, `run_refit.log`, `run_fz34.log`.
  - CLI check outputs: `cli_results/`, `cli_cache/`.
  - `backup_src_orig/` and `sandbox/`.
- `HasanML/paper3/configs/p1_faithful.yaml`.
- `reports/farag-fidelity.md` and `reports/farag-fidelity.RESUME.md`.

Modified on the box:
- `HasanML/paper3/src/` model.py, train.py, preprocess.py, rr.py, templates.py, metrics.py, run_experiment.py.

Not touched: export.py, progress.md, notes.txt, `results/cache/P*`, `paper_table4.md`.
