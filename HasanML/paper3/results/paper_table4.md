# Paper Table 4 — Farag (2023)

Reference: M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG
Classification and Arrhythmia Detection at the Edge," *Sensors* 23(3):1365, 2023,
doi: 10.3390/s23031365. Open access; all values below were transcribed from the full text
(Europe PMC PMC9919183) on 2026-11-18 by the farag-gap agent. Earlier versions of this file
had wrong model configurations and wrong per-model values; those were guesses, not readings.

## Metric definitions (paper Sec. 5.3, checked numerically)

- ACC is **plain multiclass accuracy** over the N/S/V beats of DS2 (49,317 beats; F and Q
  are dropped in the 3-class models). It is **not** the mean one-vs-rest accuracy. Check:
  Σ_c n_c·Se_c / Σ n_c with the DS2 counts gives 98.18 for Model 6 and 96.49 for Model 1, which
  matches the table. The OvR mean implied for Model 6 would be 98.79.
- PREC, SEN and F1 are per class. "Average Scores" are the **unweighted macro mean over N, S
  and V**. Model 6: (99.00+82.68+95.63)/3 = 92.44, (99.10+81.60+95.00)/3 = 91.90 and
  (99.05+82.14+95.31)/3 = 92.17. A class-weighted average would give about 98.2 for all three.
- Each reported score is the **average of 10 training runs** (Sec. 5.3). It is not an ensemble.
- Training uses Adam (lr 1e-3 with adaptive scheduling), categorical cross-entropy, 500 epochs,
  batch 512 and early stopping. Training uses DS1 records < 200 and validation uses DS1 records ≥ 200.

## Table 4 — DS2 testing results (%)

Abbreviations: DEF = default (Keras) init, trainable; TRUE = MF-initialised, trainable;
FALSE = MF-initialised, frozen; DREV = first derivative; SIG = raw signal; BOTH = 2 channels;
CW = class-weight parameter.

| Model | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|
| Classes | 3 | 3 | 3 | 3 | 3 | **3** | 3 | 4 | 5 |
| Conv1D trainable | DEF | TRUE | FALSE | DEF | TRUE | **TRUE** | TRUE | TRUE | TRUE |
| NK | 64 | 32 | 32 | 32 | 32 | **32** | 64 | 32 | 32 |
| Input | DREV | DREV | DREV | BOTH | SIG | **DREV** | DREV | DREV | DREV |
| Class weight | SET | SET | SET | SET | SET | **NOT** | NOT | NOT | NOT |
| Total params | 1619 | 1267 | 1267 | 1619 | 1267 | **1267** | 1619 | 1352 | 1420 |
| Trainable params | 1597 | 1245 | 879 | 1597 | 1245 | **1245** | 1597 | 1330 | 1389 |
| Train time (s) | 118.26 | 94.40 | 90.61 | 146.337 | 122.67 | **83.76** | 96.95 | 77.90 | 103.76 |
| *Train* ACC | 99.30 | 98.61 | 93.00 | 96.04 | 96.17 | **99.31** | 99.42 | 98.59 | 98.94 |
| *Train* avg F1 | 95.35 | 91.54 | 75.57 | 83.34 | 82.18 | **94.71** | 95.55 | 86.68 | 72.46 |
| **Test ACC** | 96.48 | 95.58 | 94.42 | 94.78 | 92.90 | **98.18** | 97.94 | 97.06 | 96.93 |
| N PREC | 99.38 | 99.47 | 99.39 | 99.57 | 99.67 | **99.00** | 99.06 | 97.73 | 97.96 |
| N SEN | 96.96 | 95.86 | 94.67 | 94.91 | 92.98 | **99.10** | 98.31 | 99.14 | 98.85 |
| N F1 | 98.16 | 97.63 | 96.98 | 97.18 | 96.21 | **99.05** | 98.68 | 98.43 | 98.40 |
| SVEB PREC | 58.06 | 51.49 | 43.85 | 56.09 | 39.19 | **82.68** | 69.72 | 84.29 | 90.78 |
| SVEB SEN | 85.30 | 88.51 | 95.59 | 92.54 | 88.13 | **81.60** | 81.33 | 79.42 | 74.52 |
| SVEB F1 | 69.09 | 65.11 | 60.12 | 69.84 | 54.26 | **82.14** | 75.08 | 81.78 | 81.85 |
| VEB PREC | 90.20 | 87.93 | 92.23 | 74.10 | 78.07 | **95.63** | 94.53 | 94.99 | 93.39 |
| VEB SEN | 96.34 | 95.65 | 90.31 | 94.32 | 94.47 | **95.00** | 95.47 | 90.13 | 95.13 |
| VEB F1 | 93.17 | 91.63 | 91.26 | 82.99 | 85.49 | **95.31** | 95.00 | 92.50 | 94.25 |
| Avg PREC | 82.55 | 79.63 | 78.49 | 76.59 | 72.31 | **92.44** | 78.77* | 69.25 | 56.42 |
| Avg SEN | 92.87 | 93.34 | 93.53 | 93.92 | 91.86 | **91.90** | 91.70 | 67.17 | 53.70 |
| Avg F1 | 86.81 | 84.79 | 82.79 | 83.34 | 78.65 | **92.17** | 89.59 | 68.18 | 54.90 |

\* The paper's Model 7 avg PREC of 78.77 is probably a typo. The mean of its per-class values is 87.77.

**Important:**
- The abstract's "SVEB/VEB sensitivity 85.3% / 96.34%" comes from **Model 1** (class weights SET,
  NK=64). It is not from Model 6, whose SVEB Se is 81.60 and VEB Se is 95.00. The abstract's 98.18 / 91.90 / 92.17 come
  from Model 6.
- The numbers "Se_V 89.7" and "Se_N 97.8" that were used in our progress.md, utils.py and the weekly
  summary **do not appear anywhere in the paper**. They are wrong.
- The closest paper configuration to our official P1 (MF-trainable, NK=32, derivative, **balanced CW**)
  is **Model 2**: ACC 95.58, avg F1 84.79.
- The paper reports no specificity values and no confusion matrix (Table 5 compares PREC/SEN only).

## Table 2 — beat counts (paper vs ours)

| Set | N | SVEB | VEB | F | Q | Total |
|---|---|---|---|---|---|---|
| DS1 (paper) | 45,866 | 944 | 3,788 | 415 | 8 | 51,021 |
| DS1 (ours, 3-class) | 45,866 | 944 | 3,788 | dropped (415) | dropped (8) | 50,598 |
| DS2 (paper) | 44,259 | 1,837 | 3,221 | 388 | 7 | 49,712 |
| DS2 (ours, 3-class) | 44,259 | 1,837 | 3,221 | dropped (388) | dropped (7) | 49,317 |

These match exactly. The paper's 3-class test set is 49,317 beats: the predicted totals implied by
the Model 6 PREC/SEN values add up to 49,316.

## Paper architecture (Table 3, 3-class): differences from our `src/model.py`

| Item | Paper Table 3 | Ours |
|---|---|---|
| Conv1D filters (3-class) | 11 (363 params at NK=32 → 32·11+11) | 13 |
| Conv1D output | (64, 11): full 64-sample input, length kept ('same') | NK-cropped input, 'valid' |
| Activation after BN | **tanh** | relu |
| RR branch | Dense 32→16→8 applied per RR value on (4,1), then Flatten → 32 | Dense(16)×3 on (4,) |
| Total params (NK=32) | 1,267 | 1,195 |

## RR-only XGBoost baseline (paper Sec. 3)

The paper reports acc 93%. Recall / precision / F1 are N 97/97/97, SVEB 60/36/45 and VEB 58/75/65.
(The paper also says "average recall and precision of 75% and 58%", which does not match these
per-class values.)

| Variant (DS2 test) | Acc | Se N/S/V | PPV N/S/V |
|---|---|---|---|
| Paper | 93 | 97 / 60 / 58 | 97 / 36 / 75 |
| Ours, CLI `--rr-baseline` (fit on DS1-train, 12 rec) | 91.64 | 96.5 / 12.7 / 69.3 | 97.3 / 17.3 / 55.0 |
| Ours, fit on full DS1, 300 trees, depth 6 | 92.74 | 96.5 / 33.2 / 74.5 | 97.1 / 54.3 / 57.4 |
| Ours, fit on full DS1, 100 trees, depth 3 | 93.73 | 97.4 / 30.8 / 79.3 | 97.1 / 60.8 / 63.9 |
| Ours, full DS1, balanced sample weights, depth 6 | 88.54 | 90.6 / 58.7 / 76.6 | 99.3 / 41.9 / 39.0 |

Source: `1DCNN/gap_analysis/rr_baseline/` (see `reports/farag-gap.md`).
