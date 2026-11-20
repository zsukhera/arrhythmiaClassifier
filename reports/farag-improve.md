# farag-improve: Farag 2023 Model 6, B1–B5 (class weights, patient-wise validation, decision calibration)

Agent: farag-improve. Dates: 2026-11-18 to 2026-11-19. Everything ran on the box on CPU (TF 2.22rc0), with 7 worker processes of 1 thread each.
All files are in `$ROOT/1DCNN/improve_B/`. The full auto-generated tables are in `results/results.md`, and the raw numbers (including confusion matrices and chosen offsets) are in `results/summary.json`.

Targets follow the corrected values from `reports/farag-gap.md`. **Model 6** (no class weights, which is the B1 config) has plain ACC 98.18, Se N/S/V 99.10 / 81.60 / 95.00, PPV 99.00 / 82.68 / 95.63, and macro PREC / Se / F1 92.44 / 91.90 / 92.17. **Model 2** (class weights, the same config as our official P1) has ACC 95.58, macro F1 84.79, Se_S 88.51 and PPV_S 51.49. Our results are compared as **means of single seeds** (paper style). Ensembles are shown separately.

## TL;DR

1. **B1, the paper's stated setup (full DS1, no class weights), does not reproduce Model 6.** The 10-seed mean is acc 93.80, macro F1 70.43, Se_S 31.8, PPV_S 37.0. The paper reports 98.18 / 92.17 / 81.6 / 82.7. With paper-literal val_loss early stopping it is 93.46 / 70.94 / 36.8 / 37.1.
2. **B2, class-weight power:** lowering p trades Se_S for Se_N almost one-for-one. **PPV_S stays at 31–37 for every p** with the random hold-out. Class weighting is not what limits precision.
3. **B4, patient-wise validation, is the only lever that clearly helped.** With GroupKFold by record, early stopping fires after about 10–15 epochs instead of 30–200. The random-beat hold-out leaks patients, so it rewards fitting DS1 patients. We take the CV-chosen epoch count and refit on all of DS1 with **p = 0.75** (p chosen on DS1 OOF macro-F1). Single-seed mean on DS2 at argmax is **acc 94.84, macro F1 81.47, Se_S 79.6, PPV_S 45.7, Se_V 90.9**. The official P1 gets 91.03 / 75.57 / 85.3 / 31.3 / 95.1. That is +3.8 acc, +5.9 F1 and +14 PPV_S, paid for with 4 points of Se_V.
4. **B3, logit-offset calibration, is unreliable whichever DS1 calibration set it uses.** The random hold-out is optimistic: DS1 Se_S is 97 at argmax against 85 on DS2. Offsets tuned on it push S down, and DS2 Se_S falls to 55–62. Patient-wise OOF is pessimistic: OOF Se_S is 13–55 and Se_V 68–78. Offsets tuned on it are erratic, and maxAcc always collapses Se_S to below 25. On DS2, every config is capped at about **95.8 acc and about 83–84 macro F1** whatever the offsets (`results/tradeoff_SeS.png`). This matches farag-gap's DS2-oracle ceiling.
5. **B5:** the Model 6 sensitivity constraint (Se_S ≥ 81.6, Se_V ≥ 95.0) is **infeasible on patient-wise OOF data** for every config. It is feasible on the leaky random hold-out, but there it does not transfer (DS2 Se_S 54–59). DS1 alone cannot reliably place this operating point.
6. **The gap to the paper remains.** Our best paper-comparable single-seed result is 94.8 acc / 81.5 F1, against 95.58 / 84.79 for Model 2 and 98.18 / 92.17 for Model 6. Against the like-for-like Model 2 the gap shrinks from −4.6 acc / −9.2 F1 to −0.7 acc / −3.3 F1. PPV_S (≈ 46 against 51.5 and 82.7) is still the bottleneck. Model 6's PPV_S 82.7 with Se_S 81.6 lies outside every curve we can produce.

## Method

- **Data:** the same arrays as the official run, checked equal to `1DCNN/1DCNN experiments/.cache/ds1_all.npz` and `ds2.npz`. We added per-beat record ids. Full DS1 = 22 records (N 45,866 / S 944 / V 3,788). DS2 = 22 records (N 44,259 / S 1,837 / V 3,221).
- **Model / training:** identical to `src/train.py` `train_one`. We call `build_model`, `apply_mf_init` and `compile_model` from paper3. Adam lr 1e-3, batch 512, max 500 epochs. Early stopping on val macro-F1 with patience 20 and best weights restored. ReduceLROnPlateau on val_loss. Deterministic seeds. Our `rand_p1` reproduces the official Exp5 seeds exactly (acc 91.57, Se_S 83.03).
- **Class weights:** w_c ∝ (1/n_c)^p, normalised so that Σ n_c·w_c = n. p = 0 means no weights; p = 1 equals sklearn "balanced".
- **Validation schemes:**
  - `rand`: 15 % random-beat hold-out, rng 42, the same split as `run_official_p1.py`.
  - `gkf`: 5 patient-wise folds over the 22 DS1 records, greedily balanced for S and V beats (`cache/folds.json`):
    - fold 0: 209, 215, 203
    - fold 1: 201, 108, 101, 116, 119
    - fold 2: 207, 114, 112, 106, 115
    - fold 3: 118, 124, 205, 208, 122
    - fold 4: 220, 223, 109, 230

    Each fold model uses its held-out records for early stopping and LR scheduling. Its matched-filter templates come **only from its training records** (`cache/kernel_fold*.npy`). Out-of-fold (OOF) probabilities cover all of DS1.
  - `refit`: train on all of DS1 for round(median best epoch of that seed's 5 gkf fold models), with constant LR and no validation data. Calibration uses that seed's gkf OOF probabilities.
- **Decision rule (B3):** y = argmax(log P + [0, b_S, b_V]), searched on a 2-D grid over [-6, 6]. The step is 0.1 for ensembles and 0.2 for single seeds. Operating points are chosen **only on DS1 calibration data** (rand: the hold-out; gkf/refit: OOF):
  - `argmax`: b = 0.
  - `maxF1`: maximum macro F1.
  - `maxAcc`: maximum plain accuracy. This is identical to maximum OvR accuracy, because OvR = 1 − 2e/3.
  - `SeConstr` (B5-i): maximum accuracy subject to Se_S ≥ 81.6 and Se_V ≥ 95.0. If that is infeasible, it falls back to maximum macro Se (balanced accuracy).
- **"Seed-mean"** is the paper-comparable number. Each seed is calibrated on its own DS1 data and evaluated on DS2, then we average over seeds. For gkf, one "seed" is a 5-fold ensemble. The fully single-network numbers are the `refit` rows and the "single-model mean" table in `results/results.md`.
- DS2 was never used for training, early stopping, p choice, finalist choice or offsets. The only DS2-informed statement is the preference discussion in "Recommendation" below.

## Final table: DS2, mean of single seeds

Columns are in %. The ± values are std over seeds for Acc / F1 / Se_S. OvR Acc = 1 − 2e/3.

| Config (seeds) | point | Acc | OvR Acc | macro F1 | macro PREC | macro Se | Se_N | PPV_N | Se_S | PPV_S | Se_V | PPV_V | ± Acc/F1/Se_S |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Paper Model 6 (no CW)** | | **98.18** | 98.79 | **92.17** | 92.44 | 91.90 | 99.10 | 99.00 | **81.60** | **82.68** | **95.00** | 95.63 | |
| **Paper Model 2 (CW)** | | 95.58 | 97.05 | 84.79 | 79.63 | 93.34 | 95.86 | 99.47 | 88.51 | 51.49 | 95.65 | 87.93 | |
| Official P1: rand, p=1 (10) | argmax | 91.03 | 94.02 | 75.57 | 69.79 | 90.47 | 90.97 | 99.73 | 85.29 | 31.25 | 95.14 | 78.40 | 0.63/1.41/4.42 |
| Official P1 | maxF1 (hold-out) | 94.39 | 96.26 | 77.25 | 74.13 | 81.42 | 96.10 | 98.42 | 54.48 | 40.97 | 93.68 | 82.99 | 0.55/1.93/8.00 |
| Official P1 | SeConstr (hold-out) | 93.81 | 95.87 | 75.52 | 71.13 | 81.78 | 95.31 | 98.92 | 54.33 | 38.32 | 95.71 | 76.14 | 0.57/3.28/10.49 |
| **B1** rand, p=0, ES val-F1 (10) | argmax | 93.80 | 95.87 | 70.43 | 68.30 | 74.31 | 96.30 | 98.37 | 31.75 | 36.99 | 94.88 | 69.53 | 0.58/3.13/10.25 |
| **B1** rand, p=0, ES val_loss (10) | argmax | 93.46 | 95.64 | 70.94 | 67.81 | 75.84 | 95.70 | 98.56 | 36.80 | 37.08 | 95.02 | 67.79 | 0.68/3.64/11.58 |
| B1 rand, p=0, ES val_loss | maxF1 (hold-out) | 93.54 | 95.69 | 72.85 | 69.15 | 78.16 | 95.51 | 98.64 | 44.69 | 38.80 | 94.29 | 70.03 | 0.65/2.68/11.47 |
| B2 rand, p=0.25 (3) | argmax | 93.32 | 95.54 | 70.99 | 67.46 | 75.91 | 95.54 | 98.45 | 37.67 | 33.60 | 94.52 | 70.32 | 0.68/4.12/12.48 |
| B2 rand, p=0.5 (3) | argmax | 93.22 | 95.48 | 73.91 | 69.69 | 79.97 | 94.91 | 98.67 | 50.70 | 35.30 | 94.32 | 75.10 | 0.74/1.71/6.28 |
| B2 rand, p=0.75 (3) | argmax | 92.80 | 95.20 | 76.57 | 71.08 | 86.93 | 93.49 | 99.37 | 72.11 | 35.06 | 95.21 | 78.82 | 0.37/0.96/2.46 |
| B2 rand, p=1 (3) | argmax | 91.57 | 94.38 | 76.14 | 70.92 | 89.65 | 91.73 | 99.73 | 83.03 | 31.18 | 94.19 | 81.85 | 0.56/1.40/3.61 |
| B4 gkf, p=0 (10 × 5-fold ens) | argmax | 95.30 | 96.87 | 66.22 | 76.44 | 65.78 | 99.30 | 96.35 | 7.93 | 47.78 | 90.10 | 85.20 | 0.19/2.59/4.35 |
| B4 gkf, p=0 | maxF1 (OOF) | 94.85 | 96.57 | 78.00 | 75.00 | 83.03 | 96.38 | 98.81 | 57.69 | 44.99 | 95.04 | 81.21 | 0.52/4.12/19.03 |
| B4 gkf, p=0.25 (3) | maxF1 (OOF) | 95.40 | 96.93 | 78.11 | 77.32 | 79.61 | 97.48 | 98.15 | 47.03 | 50.23 | 94.32 | 83.60 | 0.25/2.46/11.96 |
| B4 gkf, p=0.5 (10) | argmax | 95.48 | 96.99 | 78.57 | 77.38 | 80.47 | 97.48 | 98.31 | 49.93 | 48.65 | 94.00 | 85.18 | 0.41/3.49/13.78 |
| B4 gkf, p=0.75 (10) | argmax | 95.07 | 96.71 | 80.87 | 77.09 | 86.96 | 96.16 | 99.10 | 70.59 | 45.17 | 94.13 | 87.00 | 0.48/2.50/12.95 |
| B4 gkf, p=0.75 | maxF1 (OOF) | 94.99 | 96.66 | 72.74 | 73.21 | 75.88 | 97.43 | 98.28 | 34.15 | 44.54 | 96.06 | 76.80 | 0.47/6.61/19.75 |
| B4 refit, p=0 (10 single nets) | maxF1 (OOF) | 94.65 | 96.44 | 77.90 | 74.64 | 83.12 | 96.21 | 98.89 | 60.23 | 49.37 | 92.92 | 75.67 | 0.77/3.98/17.32 |
| B4 refit, p=0.5 (10) | argmax | 95.09 | 96.73 | 79.80 | 76.80 | 83.81 | 96.67 | 98.71 | 63.29 | 50.02 | 91.47 | 81.68 | 0.74/2.63/9.12 |
| B4 refit, p=0.5 | SeConstr→maxMacroSe (OOF) | 91.56 | 94.37 | 74.51 | 67.96 | 87.85 | 91.94 | 99.71 | 77.08 | 36.93 | 94.52 | 67.25 | 1.78/3.13/8.79 |
| **B4 refit, p=0.75 (10 single nets)** | **argmax** | **94.84** | 96.56 | **81.47** | 77.71 | 88.73 | 95.76 | 99.17 | **79.56** | 45.66 | 90.86 | 88.28 | 0.78/1.72/9.72 |
| B4 refit, p=0.75 | maxF1 (OOF) | 94.86 | 96.57 | 77.64 | 75.41 | 81.89 | 96.59 | 98.73 | 55.83 | 48.25 | 93.24 | 79.24 | 0.42/2.52/17.38 |
| B4 refit, p=0.75 | maxAcc (OOF) | 95.00 | 96.67 | 69.44 | 74.14 | 70.51 | 98.30 | 97.35 | 21.68 | 47.48 | 91.55 | 77.59 | 0.20/5.48/17.70 |
| B4 refit, p=0.75 | SeConstr→maxMacroSe (OOF) | 92.05 | 94.70 | 74.30 | 68.47 | 86.16 | 92.71 | 99.68 | 70.45 | 38.41 | 95.32 | 67.31 | 1.34/3.84/16.50 |

### Ensembles (mean probabilities over the 10 seeds; deployment view, not comparable with the paper)

| Config | point | Acc | macro F1 | macro PREC | Se_N | Se_S | PPV_S | Se_V | PPV_V |
|---|---|---|---|---|---|---|---|---|---|
| Official P1 ens (as reported; tau_S = 0.50 ≡ argmax) | argmax | 91.61 | 77.08 | 71.43 | 91.44 | 87.10 | 32.09 | 96.52 | 82.40 |
| Official P1 ens | maxF1 (hold-out) | 95.16 | 80.36 | 77.24 | 96.55 | 61.89 | 44.89 | 95.00 | 88.16 |
| B1 rand p=0 ens | argmax | 94.20 | 71.53 | 69.47 | 96.61 | 32.01 | 38.58 | 96.52 | 71.39 |
| gkf p=0.75 ens (50 fold models) | argmax | 95.28 | 82.12 | 77.96 | 96.16 | 74.85 | 46.33 | 94.78 | 88.34 |
| refit p=0.5 ens | argmax | 95.69 | 82.24 | 78.95 | 96.96 | 68.05 | 52.17 | 94.07 | 85.79 |
| refit p=0.5 ens | SeConstr→maxMacroSe (OOF) | 92.38 | 77.84 | 71.63 | 92.38 | 89.33 | 36.72 | 94.13 | 78.41 |
| **refit p=0.75 ens** | **argmax** | **95.31** | **83.39** | 79.09 | 95.88 | **85.52** | 46.98 | 92.95 | 90.86 |
| refit p=0.75 ens | maxF1 (OOF) | 95.11 | 80.46 | 75.61 | 96.12 | 68.97 | 48.82 | 96.15 | 78.54 |
| refit p=0.75 ens | maxAcc (OOF) | 95.77 | 77.20 | 79.87 | 98.42 | 36.74 | 56.82 | 93.11 | 85.15 |
| refit p=0.75 ens | SeConstr→maxMacroSe (OOF) | 93.00 | 77.15 | 70.59 | 93.38 | 76.43 | 38.87 | 97.14 | 73.14 |

### DS1-only selection data (what the choices were based on)

| Config | best-epoch median | OOF / hold-out macro F1 @argmax | @maxF1 | SeConstr feasible? | DS1-cal Se_S / Se_V @argmax |
|---|---|---|---|---|---|
| gkf p=0 | 15 | 61.9 | 66.1 | no | 13.2 / 68.1 |
| gkf p=0.25 (3 seeds) | 17 | 64.2 | 65.4 | no | 23.2 / 77.8 |
| gkf p=0.5 | 14.5 | 68.0 | 69.1 | no | 40.4 / 78.0 |
| **gkf p=0.75** | 10.5 | **68.7** | **70.8** | no | 54.6 / 77.2 |
| gkf p=1 (3 seeds) | 11 | 63.8 | 68.7 | no | 62.4 / 74.6 |
| rand p=0 (val-F1 ES) | 92.5 | 93.1 | 93.7 | yes | 79.3 / 95.2 |
| rand p=0 (val_loss ES) | 198 (runs hit 500 epochs) | 94.1 | 94.9 | yes | 83.4 / 95.5 |
| rand p=1 / official | 31 | 82.0 / 83.2 | 87.3 / 88.6 | yes | 97.2 / 95.5 |

The rand hold-out scores 93–95 macro F1 and the patient-wise OOF scores 62–71 for the same models. DS2 lands in between (70–83). This is the patient leakage of the random-beat hold-out, measured directly.

## Findings per task

- **B1:** without class weights on full DS1, the model is N-biased (Se_N 96, Se_S 32–37), and PPV_S is still only about 37. The paper's Model 6 (Se_S 81.6, PPV_S 82.7) is not reproduced under any early-stopping choice. Paper-literal val_loss ES never stopped: all 10 seeds ran the full 500 epochs, with a median best epoch of 198. Under patient-wise ES (gkf p=0), Se_S drops to 8, because the OOF S signal is weak and stopping comes very early.
- **B2:** with the random hold-out, p moves the operating point along Se_N ↔ Se_S: p=0 gives 96.3 / 31.8 and p=1 gives 91.7 / 83.0. Macro F1 is flat to slightly better at high p (76–77 at p = 0.75–1), and PPV_S never exceeds 37. Under gkf, p=0.75 is best on DS1 OOF and is also best on DS2. p=0.5 is the runner-up.
- **B3:** see the TL;DR and `results/tradeoff_SeS.png`. In the plot, each curve sweeps b_S with b_V chosen on DS1. Solid lines are DS2 and dotted lines are DS1 calibration. Notes:
  - On DS2, all models trace almost the same envelope: maximum macro F1 about 83–84 at Se_S ≈ 80–87, and accuracy about 95–96 before Se_S collapses.
  - The DS1 curves (dotted) sit far above (rand) or far below (gkf OOF) the DS2 curves. That is why DS1-tuned offsets mostly fail to transfer.
  - Offsets help only when the base model is badly mis-calibrated towards S. For the official P1 ensemble they bring F1 from 77.1 to 80.4, but Se_S falls from 87 to 62.
- **B4:** patient-wise early stopping plus the refit is the main gain. The refit is slightly better than the fold ensemble on single-network numbers, because each net sees all 22 records.
- **B5:**
  - (i) The sensitivity-constrained point is infeasible on OOF for every config. On the leaky hold-out it is feasible, but DS2 then gives Se_S 54–59 for p=1. Nothing chosen from DS1 met Se_S ≥ 81.6 and Se_V ≥ 95.0 on DS2 in the seed mean. Ensembles come close:
    - refit p=0.5 ens fallback: Se_S 89.3, Se_V 94.1, acc 92.4.
    - refit p=0.75 ens argmax: Se_S 85.5, Se_V 93.0, acc 95.3.
  - (ii) Best macro-F1 point: the DS1-chosen maxF1 point was **worse** than plain argmax on DS2 for the best configs.

## Recommendation

**Configuration: full DS1, class-weight power p = 0.75, epoch count chosen by patient-wise 5-fold CV (median best epoch, 9 on average), refit on all of DS1, argmax decision with no threshold or offset.**

- Paper-comparable (mean of 10 single nets): acc 94.84 ± 0.78, macro F1 81.47 ± 1.72, macro PREC 77.71, Se_N / S / V 95.76 / 79.56 / 90.86, PPV_S 45.66.
- Deployed as a 10-seed ensemble: acc 95.31, macro F1 83.39, Se_S 85.52, Se_V 92.95, PPV_S 46.98.
- Models: `$ROOT/1DCNN/improve_B/runs/refit_p0.75_s{0..9}.keras`. Each `.npz` beside it holds the DS2 probabilities and the epoch count.
- Selection honesty:
  - p = 0.75 and the patient-wise scheme were chosen on DS1 OOF only.
  - Using **argmax rather than the OOF-tuned maxF1 offsets** is a judgment call. maxF1 was the DS1-only rule, and it gives seed-mean F1 77.64 / Se_S 55.8 on DS2. The tables show that DS1-tuned offsets do not transfer under either validation scheme, and argmax involves no tuning at all. Even so, report the maxF1 row next to it.
- The model trades Se_V (90.9 against 95.1 for the official P1) for much better precision. If V sensitivity is clinically required, use the refit p=0.75 ensemble at the OOF SeConstr fallback point (b_S = +0.7, b_V = +1.3): Se_V 97.1, Se_S 76.4, acc 93.0.

## Reproduce (on the box)

Run from `$ROOT/1DCNN/improve_B` with `PY=../../.venv/bin/python`:

```bash
$PY prep.py        # data.npz with record ids, folds.json, kernel_full/fold*.npy (checks vs official caches)
$PY train_jobs.py --scheme rand  --p 1 0 0.25 0.5 0.75 --seeds 0-2 --workers 7          # B2 (rand)
$PY train_jobs.py --scheme rand  --p 0 --seeds 0-9 --workers 7                          # B1 (val-F1 ES)
$PY train_jobs.py --scheme rand  --p 0 --seeds 0-9 --es loss --workers 7                # B1 (val_loss ES)
$PY train_jobs.py --scheme gkf   --p 0 0.25 0.5 0.75 1 --seeds 0-2 --workers 7          # B2/B4 sweep
$PY train_jobs.py --scheme gkf   --p 0.75 0.5 0 --seeds 0-9 --workers 7                 # finalists
$PY train_jobs.py --scheme refit --p 0.75 0.5 0 --seeds 0-9 --workers 7                 # B4 refit
OMP_NUM_THREADS=1 $PY analyze.py   # B3/B5: results/results.md, summary.json, tradeoff_SeS.png
```

The same steps are chained in `pipeline.sh`, `pipeline2.sh` and `pipeline3.sh`. All jobs are cached per run tag, so reruns resume. Wall time is about 1.5 h in total on 7 threads.

## Files created (all new; nothing outside the owned dir was modified)

All paths are under `$ROOT/1DCNN/improve_B/`:
- Scripts: `common.py`, `prep.py`, `train_jobs.py`, `analyze.py`, `pipeline.sh`, `pipeline2.sh`, `pipeline3.sh`, `RESUME.md`
- `cache/`: `data.npz`, `folds.json`, `kernel_full.npy`, `kernel_fold0-4.npy`
- `runs/`: 242 `.keras` models with matching `.npz` files (probabilities, epochs, histories), tagged `{rand|gkf|refit}_p{p}_s{seed}[_f{fold}][_esloss]`
- `results/`: `results.md`, `summary.json`, `tradeoff_SeS.png`
- `logs/`: all training and analysis logs

Also `$ROOT/reports/farag-improve.md` (this file) and `$ROOT/reports/farag-improve.RESUME.md`.

## Open issues

- **The paper's Model 6 is not reproduced.** No-weight training gives PPV_S ≈ 37–51 and Se_S ≤ 37 at argmax. A pipeline difference (preprocessing, RR features, templates; see C1–C4) is more likely than a training-recipe difference, because no class weighting, early stopping or calibration choice moves PPV_S above about 57 on DS2.
- The 5-fold patient-wise OOF is pessimistic. S beats are concentrated in a few records: 209 alone has 383 of 944. Each fold therefore loses a large share of the S morphology. Leave-one-record-out, or more folds, might give better-calibrated OOF probabilities for offset tuning. Not done.
- gkf p = 0.25 and p = 1 have only 3 seeds. Refit used a constant LR. ReduceLROnPlateau has no validation signal there, but with about 10 epochs it would rarely trigger.
- `rand` ES with val_loss never stopped within 500 epochs. Patience 20 with ReduceLROnPlateau on the same signal keeps it improving slowly.
- The OOF SeConstr fallback (max macro Se) is a substitute rule, because the target was infeasible on DS1.
