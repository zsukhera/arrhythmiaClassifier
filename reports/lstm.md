# LSTM baseline (progress.md todo: "LSTM baseline — input shape (N, 720, 2)")

Agent: `lstm`, 2026-11-18/19. All code, data and results are in `$ROOT/LSTM/` (ROOT = `/home/i230510/Desktop/arrhythmiaClassifier`).
The full per-seed tables and every confusion matrix are in `LSTM/results/summary.md`; machine-readable versions are in `LSTM/results/summary.json`.

## TL;DR
- **Random 80/20 split (N/L/R/V/A):** Conv-BiLSTM gets acc **99.54 ± 0.06** and macro-F1 **98.66 ± 0.22** (5 seeds). The 5-seed ensemble gets 99.67 / 99.05. This is the same level as LightGBM (0.996 / 0.987). Patients leak between train and test in this split, so this result tells us little.
- **DS1/DS2 (N/L/R/V/A):**
  - The plain LSTM gets acc 73–77% and macro-F1 33–36%, about the same as or slightly below LightGBM (0.764 / 0.388).
  - **LBBB and RBBB fail completely (Se ≈ 0)**, the same failure LightGBM has.
  - N and V transfer well (Se_V ≈ 88–98%). APC is weak (Se_A 7–23%).
- **DS1/DS2 (AAMI N/S/V):**
  - The plain LSTM, which sees only the 2-s waveform, **cannot detect S beats across patients**: Se_S 7–13%. Its plain accuracy is 91–92% and macro-F1 60–63%.
  - An extra variant adds 4 causal RR-ratio features to the LSTM (+1.1k params). It raises Se_S to 44–67% (single seeds) or 52–74% (ensemble). Ensemble macro-F1 is 77–78%, about equal to our Farag 1D-CNN (77.08), with higher PPV_S (54–61 vs 32.1) but lower Se_S (52–74 vs 87.10).
  - Every LSTM variant stays far below Farag's paper numbers for Model 6: acc 98.18, Se_S 81.6, PPV_S 82.68, macro-F1 92.17.

## Setup
| Item | Value |
|---|---|
| Input | 720 samples (2 s @ 360 Hz) per beat: 360 before and 360 after the annotated R-peak. Ch0 = MLII (selected by name), ch1 = the other lead (V1 in 39/44 records; V2/V4/V5 in the rest) |
| Filtering | 4th-order Butterworth band-pass 0.5–40 Hz, SOS + `sosfiltfilt`, same as `source/preprocess.py` |
| Records | 44 (paced 102/104/107/217 excluded). Beats near record edges are **edge-padded** rather than dropped, so every beat in the LightGBM split files has a window |
| Normalisation | per-window, per-channel z-score |
| Labels (a) | N/L/R/V/A, using **exactly** the beats in `split_random.npz` (79,430 / 19,858) and `split_ds1ds2.npz` (50,307 / 48,981), matched on (record, R-sample) |
| Labels (b) | AAMI N/S/V (N,L,R,e,j / A,a,J,S / V,E; F and Q dropped). DS2 = N 44,259 / S 1,837 / V 3,221, the same as paper3 |
| Model "conv" (main) | Conv1d(2→32, k7, s2)-BN-ReLU → Conv1d(32→64, k5, s2)-BN-ReLU (720→180 steps) → 2-layer **BiLSTM** (hidden 64, dropout 0.2) → [mean ‖ max] pooling over time → Dropout 0.3 → Linear(256→64) → ReLU → Linear(→K). **193,411 params (3-class) / 193,541 (5-class)** |
| Ablation "pool" | AvgPool(4) on the raw 2-lead input (90 Hz, 180 steps) → same BiLSTM and head. A pure LSTM with no conv. 150,787 / 150,917 params |
| Extra "convrr" | "conv" plus 4 causal RR ratios (pre/local, post/local, pre/post, local/record-median; local = mean of the last ≤10 RR) → Linear(4→16)-ReLU, concatenated before the head. 194,515 / 194,645 params. **This is not a pure LSTM baseline**; it was added because the plain model missed S |
| Training | Class-weighted CE ("balanced" weights from the training counts), AdamW (lr 1e-3, wd 1e-2), batch 256, AMP fp16, grad clip 1.0, cosine LR over 40 epochs (floor 3%). 5 seeds (0–4) per configuration |
| Model selection | Early stopping on **validation macro-F1** (min 10 epochs, patience 15) |
| Validation, DS1/DS2 | **patient-held-out DS1 records {115, 207, 220, 223}** ("heldout"). Chosen in advance from DS1 class counts only, so that every class of both schemes appears in validation |
| Validation, random split | 10% stratified beat hold-out from the training part (that split is not patient-independent anyway) |
| "refit" | Retrain on **all of DS1** for a fixed number of epochs = median best epoch of the 5 heldout runs (conv: AAMI 24, N/L/R/V/A 14; pool: 21 / 21; convrr: 13 / 13). Same cosine schedule |
| DS2 | Test only. Scored once per run, after training. Never used for stopping, thresholds or model choice |
| Thresholds | none (argmax). The "ensemble" row = mean of the 5 seeds' softmax probabilities, then argmax |
| Hardware | RTX 4000 Ada (shared), < 0.5 GB GPU, 4 CPU threads |

Before seeing any DS2 numbers I named refit the primary DS1/DS2 result. Heldout is reported next to it; both are valid on DS2. On DS2, refit did **not** consistently beat heldout (see below).

## (b) AAMI N/S/V on DS1 → DS2 (49,317 beats)

Values are % (mean ± std over 5 seeds); "ens" = 5-seed probability-averaged ensemble.

| Model | Acc | OvR Acc | Macro F1 | Macro PREC | Se_N | Se_S | PPV_S | Se_V | PPV_V | Params | Train time/seed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **LSTM conv, refit (primary)** | 92.04 ± 1.67 | 94.69 ± 1.11 | 62.67 ± 3.63 | 62.35 ± 3.93 | 95.68 | 13.10 ± 4.05 | 23.24 | 86.99 ± 5.47 | 67.56 | 193,411 | 50 s |
| LSTM conv, refit, ens | 94.90 | 96.60 | 68.33 | 73.27 | 98.51 | 11.21 | 35.40 | 93.14 | 88.24 | | |
| LSTM conv, heldout | 91.28 ± 1.45 | 94.19 ± 0.97 | 60.28 ± 2.81 | 60.37 ± 5.21 | 94.91 | 7.19 ± 0.70 | 19.67 | 89.43 ± 4.80 | 65.96 | 193,411 | 88 s |
| LSTM conv, heldout, ens | 95.19 | 96.80 | 66.34 | 80.73 | 98.97 | 5.72 | 58.01 | 94.32 | 88.31 | | |
| LSTM pool (ablation), refit | 81.50 ± 2.56 | 87.67 ± 1.71 | 54.10 ± 7.03 | 49.69 ± 5.43 | 82.92 | 31.41 ± 26.70 | 14.65 | 90.52 ± 0.54 | 38.02 | 150,787 | 23 s |
| LSTM pool (ablation), heldout | 75.29 ± 5.63 | 83.53 ± 3.75 | 45.68 ± 3.30 | 43.89 ± 2.16 | 77.08 | 9.66 ± 2.00 | 4.78 | 88.03 ± 3.63 | 30.69 | 150,787 | 31 s |
| LSTM conv+RR, refit | 88.30 ± 3.26 | 92.20 ± 2.18 | 68.13 ± 5.96 | 62.09 ± 6.09 | 89.20 | 66.91 ± 14.72 | 36.61 | 88.18 ± 4.68 | 51.19 | 194,515 | 26 s |
| LSTM conv+RR, refit, ens | 92.86 | 95.24 | **78.21** | 72.75 | 93.70 | 73.71 | **60.55** | 92.30 | 59.07 | | |
| LSTM conv+RR, heldout | 92.34 ± 1.65 | 94.89 ± 1.10 | 70.75 ± 5.73 | 67.57 ± 5.27 | 94.35 | 44.12 ± 20.09 | 42.45 | 92.23 ± 2.13 | 62.10 | 194,515 | 49 s |
| LSTM conv+RR, heldout, ens | 94.43 | 96.28 | 77.37 | 74.78 | 96.20 | 52.42 | 53.62 | 94.07 | 72.51 | | |
| Farag 1D-CNN (ours), mean of seeds | 91.03 | – | 75.57 | – | 90.97 | 85.29 | – | 95.14 | – | ~1.2k | |
| Farag 1D-CNN (ours), ens + τ_S | 91.61 | – | 77.08 | – | 91.44 | **87.10** | 32.1 | **96.52** | – | | |
| Farag 2023 paper, **Model 6** | **98.18** (plain) | – | **92.17** | – | 99.10 | 81.60 | 82.68 | 95.00 | – | 1,195–1,611 | |

Notes on the Farag comparison:
- Paper targets are the corrected Model 6 values. The 85.3 / 96.34 figures used earlier belong to Model 1 (with class weights).
- Our Farag 1D-CNN was early-stopped on a 15% **random-beat** hold-out of DS1, which leaks patients. The LSTM numbers use patient-held-out validation, so the protocols are not identical.

Confusion matrices for the ensembles (rows = true, columns = predicted N/S/V):

| Model (ens) | N row | S row | V row |
|---|---|---|---|
| conv refit | 43,598 / 281 / 380 | 1,611 / 206 / 20 | 126 / 95 / 3,000 |
| conv heldout | 43,804 / 72 / 383 | 1,713 / 105 / 19 | 179 / 4 / 3,038 |
| conv+RR refit | 41,471 / 802 / 1,986 | 409 / 1,354 / 74 | 168 / 80 / 2,973 |
| conv+RR heldout | 42,575 / 783 / 901 | 626 / 963 / 248 | 141 / 50 / 3,030 |

## (a) N/L/R/V/A

### Random stratified 80/20 split (19,858 test beats, the same beats as LightGBM)
| Model | Acc | OvR Acc | Macro F1 | Macro PREC | F1 N / L / R / V / A | Train time/seed |
|---|---|---|---|---|---|---|
| **LSTM conv** (5 seeds) | 99.54 ± 0.06 | 99.81 ± 0.03 | 98.66 ± 0.22 | 98.65 ± 0.37 | 99.73 / 99.79 / 99.67 / 98.51 / 95.62 | 111 s |
| LSTM conv, ens | 99.67 | 99.87 | 99.05 | 99.27 | 99.81 / 99.88 / 99.72 / 98.92 / 96.92 | |
| LSTM pool (ablation) | 99.36 ± 0.08 | 99.74 ± 0.03 | 98.18 ± 0.22 | 97.99 ± 0.29 | 99.62 / 99.77 / 99.62 / 97.64 / 94.27 | 60 s |
| LSTM conv+RR | 99.59 ± 0.03 | 99.83 ± 0.01 | 98.80 ± 0.04 | 98.68 ± 0.11 | 99.77 / 99.79 / 99.72 / 98.53 / 96.17 | 107 s |
| LightGBM (step 5) | 99.58 | – | 98.72 | – | – | |

Ensemble confusion matrix for conv (rows = true N/L/R/V/A):

| true \ pred | N | L | R | V | A |
|---|---|---|---|---|---|
| N | 14,879 | 0 | 0 | 16 | 8 |
| L | 2 | 1,613 | 0 | 0 | 0 |
| R | 3 | 0 | 1,446 | 1 | 1 |
| V | 10 | 2 | 0 | 1,368 | 0 |
| A | 18 | 0 | 3 | 1 | 487 |

### DS1 → DS2 (48,981 test beats, the same beats as LightGBM)
| Model | Acc | OvR Acc | Macro F1 | Macro PREC | Se N / L / R / V / A | Train time/seed |
|---|---|---|---|---|---|---|
| **LSTM conv, refit (primary)** | 73.09 ± 3.73 | 89.24 ± 1.49 | 33.26 ± 5.81 | 37.90 ± 11.58 | 88.9 / 4.3 / 4.0 / 92.0 / 7.7 | 29 s |
| LSTM conv, refit, ens | 76.66 | 90.66 | 32.57 | 38.91 | 94.1 / 0.2 / 0.0 / 97.8 / 7.0 | |
| LSTM conv, heldout | 77.28 ± 1.28 | 90.91 ± 0.51 | 36.33 ± 3.53 | 40.05 ± 8.16 | 95.1 / 0.5 / 0.0 / 88.1 / 20.4 | 120 s |
| LSTM conv, heldout, ens | 79.74 | 91.90 | 37.26 | 40.81 | 98.6 / 0.0 / 0.0 / 93.9 / 6.5 | |
| LSTM pool, refit | 70.55 ± 4.08 | 88.22 ± 1.63 | 30.69 ± 2.03 | 33.43 ± 5.46 | 85.3 / 0.3 / 11.5 / 92.9 / 4.5 | 22 s |
| LSTM pool, heldout | 66.73 ± 2.93 | 86.69 ± 1.17 | 31.58 ± 3.85 | 36.29 ± 10.12 | 80.4 / 4.5 / 8.4 / 84.1 / 13.0 | 32 s |
| LSTM conv+RR, refit | 76.75 ± 2.20 | 90.70 ± 0.88 | 34.81 ± 3.97 | 33.88 ± 5.29 | 93.8 / 0.1 / 0.2 / 93.4 / 23.0 | 25 s |
| LSTM conv+RR, heldout | 77.45 ± 2.28 | 90.98 ± 0.91 | 35.78 ± 3.15 | 40.71 ± 5.03 | 95.0 / 0.0 / 1.9 / 91.7 / 18.1 | 47 s |
| LightGBM (step 5) | 76.35 | – | 38.78 | – | 93.1 / 0.8 / 0.1 / 97.4 / 17.7 | |

Ensemble confusion matrix for conv refit (rows = true N/L/R/V/A):

| true \ pred | N | L | R | V | A |
|---|---|---|---|---|---|
| N | 34,268 | 25 | 74 | 1,851 | 209 |
| L | 3,995 | 10 | 0 | 119 | 0 |
| R | 1,857 | 0 | 1 | 1,615 | 2 |
| V | 66 | 0 | 0 | 3,149 | 4 |
| A | 1,548 | 0 | 0 | 66 | 122 |

## Findings
1. **The random split is easy for any model.** All three LSTM variants reach 99.4–99.7% accuracy, matching LightGBM. Patients appear on both sides of the split, so this does not measure generalisation.
2. **The inter-patient N/L/R/V/A task is limited by the data, not by the model.** DS1 has LBBB from only 2 patients (109, 207) and RBBB from only 3 (118, 124, 207). A sequence model on raw waveforms fails on DS2 L/R exactly as LightGBM does on hand-made features. In the ensemble, L beats go to N and R beats are split between N and V. With these labels, macro-F1 has a ceiling of about 0.4 whatever the model.
3. **The plain waveform LSTM is a good V detector but a poor S detector across patients.** It gets Se_V 87–94% with PPV_V up to 88% (ensemble). S beats look like N in morphology; what tells them apart is timing. The 2-s window does contain neighbouring beats, but the LSTM does not learn to read RR timing from them well enough to transfer across patients. Adding explicit RR ratios raises Se_S from about 10% to 44–74%. With RR features, refit on all of DS1 (+274 S beats from the validation records) helps S further: ensemble Se_S 52 → 74%, at a cost in Se_N (96.2 → 93.7).
4. **The conv front-end matters.** The pure LSTM on 90 Hz input ("pool") is clearly worse on DS1/DS2 (AAMI acc 75–82% vs 91–92%) and makes many N→V errors.
5. **Comparison with the Farag 1D-CNN:**
   - The conv+RR LSTM ensemble roughly matches its macro-F1 (78.21 vs 77.08), with better PPV_S (60.6 vs 32.1) and Se_N (93.7 vs 91.4), but worse Se_S (73.7 vs 87.1) and Se_V (92.3 vs 96.5).
   - The plain LSTM (the requested baseline) is much worse on S.
   - None of these reach Farag's paper numbers for Model 6 (98.18 plain acc, macro-F1 92.17).
6. **Results vary a lot between seeds,** especially for S: conv+RR refit Se_S ranges 38–78%. That is expected with about 900 S training beats from a handful of patients. Validation macro-F1 on 4 held-out patients is also noisy, so the chosen epoch varied from 10 to 38.
7. **Refit vs heldout:** for the plain conv model, refit gave no reliable gain. It was slightly better on AAMI single seeds and worse on N/L/R/V/A, partly because its fixed epoch count came from the noisy validation. Both are reported; neither was chosen using DS2.

## Negative results and problems
- The first launch (constant LR, patience 10, no minimum epochs) early-stopped at epoch 3 on some seeds, because validation macro-F1 jumped between 0.25 and 0.58. I killed it, deleted those runs, and relaunched with a cosine schedule, min 10 epochs and patience 15. Only the relaunched runs are reported.
- No threshold or logit-offset calibration was done. A per-class bias tuned on the held-out DS1 records could trade Se_S against PPV_S, as Farag's τ_S did. Not done.
- No augmentation and no hyper-parameter search, by design (baseline).

## Reproduce
```bash
cd /home/i230510/Desktop/arrhythmiaClassifier/LSTM
../.venv-torch/bin/python build_data.py      # -> data/windows720.npz (100,733 x 720 x 2, float16)
../.venv-torch/bin/python rr_features.py     # -> data/rr.npz (only for the conv+RR variant)
./run_all.sh  > run_all.log 2>&1             # conv + pool: aami3/nlrva heldout, refit, nlrva random; 5 seeds each (~1.5 h)
./run_rr.sh   > run_rr.log  2>&1             # conv+RR variant (~40 min)
../.venv-torch/bin/python report.py          # -> results/summary.md, results/summary.json
# single run, e.g.:
../.venv-torch/bin/python train.py --scheme aami3 --split ds1ds2_heldout --arch conv [--rr] --seed 0
../.venv-torch/bin/python train.py --scheme aami3 --split ds1ds2_refit  --arch conv --epochs 24 --seed 0
```
The scripts can be re-run: finished runs (`runs/<tag>/seed<k>/metrics.json`) are skipped, and interrupted runs resume from `ckpt_last.pt`.

## Files created (all new; nothing outside `LSTM/` and `reports/lstm*.md` was touched)
- `LSTM/build_data.py`, `LSTM/rr_features.py`, `LSTM/lstm_common.py`, `LSTM/train.py`, `LSTM/report.py`, `LSTM/run_all.sh`, `LSTM/run_rr.sh`
- `LSTM/data/windows720.npz`, `LSTM/data/rr.npz`
- `LSTM/runs/<scheme>_<split>_<arch>[_e<E>]/seed{0..4}/` (`best.pt`, `metrics.json`, `test_probs.npy`, `test_y.npy`, `val_probs.npy`, `val_y.npy`). 15 configurations × 5 seeds
- `LSTM/results/summary.md`, `LSTM/results/summary.json`, `LSTM/run_all.log`, `LSTM/run_rr.log`
- `reports/lstm.md` (this file), `reports/lstm.RESUME.md`

## Open issues / next steps
- Validation-tuned logit offsets (S, V) for the conv+RR model, chosen on the held-out DS1 records, to place it on the Se_S/PPV_S trade-off curve.
- Repeat the heldout runs with other choices of validation records (record-level GroupKFold across DS1) to measure how much the 4-record choice affects results.
- MLII-only input: the second lead is not the same across records (V1/V2/V4/V5), which may hurt inter-patient transfer.
- L/R on DS2 will not improve without more bundle-branch-block patients in training (e.g. INCART or other databases) or a scheme that merges L/R into N, as AAMI does.
