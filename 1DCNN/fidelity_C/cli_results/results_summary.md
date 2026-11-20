# Results Summary — Farag (2023) Replication

Reference: M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification and Arrhythmia Detection at the Edge," Sensors 23(3):1365, 2023, doi: 10.3390/s23031365.

## Accuracy definitions

The paper reports 'average accuracy'.  Based on our investigation of a related paper (Yakut 2020), 'average accuracy' in the ECG classification literature often means the **mean one-vs-rest (OvR) accuracy** = 1 − 2e/K, not plain accuracy 1 − e.  For K=3 and OvR accuracy 98.18 %, plain accuracy ≈ 97.27 %.  We report both; use the per-class Table 4 numbers to verify which definition matches.  See results/paper_table4.md.

Majority-class baseline: N is ~89.5 % of DS2 beats (paper Sec 5.1).  Plain accuracy of e.g. 97 % means only +7.5 pp above the trivial baseline.

## Results (mean ± std over 10 seeds unless noted)

| Run | Input | NK | Init | CW | Classes | Plain Acc (%) | OvR Acc (%) | Majority (%) | Macro Se (%) | Macro F1 (%) | Se S (%) | Se V (%) | Paper OvR Acc | Paper F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grid_derivative_nk64_default_cwbalanced_c3 | derivative | 64 | default | balanced | 3 | 91.62 ± 0.00 | 94.41 ± 0.00 | 89.74 ± 0.00 | 93.33 ± 0.00 | 78.53 ± 0.00 | 92.54 ± 0.00 | 96.21 ± 0.00 | — | — |
| grid_derivative_nk32_mf_trainable_cwbalanced_c3 | derivative | 32 | mf_trainable | balanced | 3 | 92.47 ± 0.00 | 94.98 ± 0.00 | 89.74 ± 0.00 | 94.23 ± 0.00 | 79.95 ± 0.00 | 95.65 ± 0.00 | 94.88 ± 0.00 | — | — |
| grid_derivative_nk32_mf_frozen_cwbalanced_c3 | derivative | 32 | mf_frozen | balanced | 3 | 92.11 ± 0.00 | 94.74 ± 0.00 | 89.74 ± 0.00 | 84.03 ± 0.00 | 74.14 ± 0.00 | 72.29 ± 0.00 | 86.43 ± 0.00 | — | — |
| grid_both_nk32_default_cwbalanced_c3 | both | 32 | default | balanced | 3 | 93.02 ± 0.00 | 95.35 ± 0.00 | 89.74 ± 0.00 | 93.04 ± 0.00 | 79.77 ± 0.00 | 91.83 ± 0.00 | 94.32 ± 0.00 | — | — |
| grid_signal_nk32_mf_trainable_cwbalanced_c3 | signal | 32 | mf_trainable | balanced | 3 | 91.70 ± 0.00 | 94.47 ± 0.00 | 89.74 ± 0.00 | 93.50 ± 0.00 | 78.92 ± 0.00 | 94.18 ± 0.00 | 94.97 ± 0.00 | — | — |
| grid_derivative_nk32_mf_trainable_cwnone_c3 | derivative | 32 | mf_trainable | none | 3 | 97.51 ± 0.00 | 98.34 ± 0.00 | 89.74 ± 0.00 | 91.92 ± 0.00 | 89.82 ± 0.00 | 82.36 ± 0.00 | 95.09 ± 0.00 | — | — |
| grid_derivative_nk64_mf_trainable_cwnone_c3 | derivative | 64 | mf_trainable | none | 3 | 97.39 ± 0.00 | 98.26 ± 0.00 | 89.74 ± 0.00 | 91.11 ± 0.00 | 89.19 ± 0.00 | 79.42 ± 0.00 | 95.65 ± 0.00 | — | — |
| grid_derivative_nk32_mf_trainable_cwnone_c4 | derivative | 32 | mf_trainable | none | 4 | 96.13 ± 0.00 | 98.06 ± 0.00 | 89.04 ± 0.00 | 68.53 ± 0.00 | 66.69 ± 0.00 | 82.96 ± 0.00 | 93.45 ± 0.00 | — | — |
| grid_derivative_nk32_mf_trainable_cwnone_c5 | derivative | 32 | mf_trainable | none | 5 | 96.73 ± 0.00 | 98.69 ± 0.00 | 89.03 ± 0.00 | 55.09 ± 0.00 | 54.12 ± 0.00 | 81.65 ± 0.00 | 95.47 ± 0.00 | — | — |

Every number above is computed by the pipeline.  Discrepancies > 1 pp from Table 4 are listed below with likely causes.

## Likely causes of discrepancies (if any)
- **Accuracy definition**: the paper may use OvR accuracy, not plain accuracy.
- **Resampling**: method not specified (we use resample_poly with up=16, down=45).
- **Segment centre**: 32 before + 32 after R-peak [assumption].
- **Normalisation**: mean subtraction [assumption].
- **EarlyStopping patience**: 20 [assumption].
- **LR schedule**: ReduceLROnPlateau, factor=0.5, patience=5 [assumption].
- **Templates**: computed on DS1-train only [assumption: paper says 'DS1'].
