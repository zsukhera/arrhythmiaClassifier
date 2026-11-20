# Results Summary — Farag (2023) Replication

Reference: M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification and Arrhythmia Detection at the Edge," Sensors 23(3):1365, 2023, doi: 10.3390/s23031365.

## Accuracy definitions

The paper reports 'average accuracy'.  Based on our investigation of a related paper (Yakut 2020), 'average accuracy' in the ECG classification literature often means the **mean one-vs-rest (OvR) accuracy** = 1 − 2e/K, not plain accuracy 1 − e.  For K=3 and OvR accuracy 98.18 %, plain accuracy ≈ 97.27 %.  We report both; use the per-class Table 4 numbers to verify which definition matches.  See results/paper_table4.md.

Majority-class baseline: N is ~89.5 % of DS2 beats (paper Sec 5.1).  Plain accuracy of e.g. 97 % means only +7.5 pp above the trivial baseline.

## Results (mean ± std over 10 seeds unless noted)

| Run | Input | NK | Init | CW | Classes | Plain Acc (%) | OvR Acc (%) | Majority (%) | Macro Se (%) | Macro F1 (%) | Se S (%) | Se V (%) | Paper OvR Acc | Paper F1 |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| P1 (Model 6) | derivative | 32 | mf_trainable | none | 3 | — | — | — | — | — | — | — | 98.18 | 92.17 |

Every number above is computed by the pipeline.  Discrepancies > 1 pp from Table 4 are listed below with likely causes.

## Likely causes of discrepancies (if any)
- **Accuracy definition**: the paper may use OvR accuracy, not plain accuracy.
- **Resampling**: method not specified (we use resample_poly with up=16, down=45).
- **Segment centre**: 32 before + 32 after R-peak [assumption].
- **Normalisation**: mean subtraction [assumption].
- **EarlyStopping patience**: 20 [assumption].
- **LR schedule**: ReduceLROnPlateau, factor=0.5, patience=5 [assumption].
- **Templates**: computed on DS1-train only [assumption: paper says 'DS1'].
