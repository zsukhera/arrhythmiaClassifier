# Paper Table 4 — Farag (2023) Table 4

Reference:
M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG
Classification and Arrhythmia Detection at the Edge," Sensors, vol. 23,
no. 3, art. 1365, 2023, doi: 10.3390/s23031365.

## Table 4: Results on DS2 (3-class N/S/V, confirmed from paper)

Values taken directly from Table 4. "—" means the paper's table image was not
legible at that cell.  Accuracy = average accuracy (see README for the
one-vs-rest vs plain-accuracy discussion).

| Model | Input       | NK | Init          | Class weight | Avg ACC (%) | Avg F1 (%) | Avg PREC (%) | SVEB Se (%) | VEB Se (%) | Avg Se (%) |
|-------|-------------|-----|---------------|--------------|-------------|------------|--------------|-------------|------------|------------|
| 1     | derivative  | 64  | default       | none         | —           | —          | —            | —           | 96.34      | —          |
| 2     | derivative  | 64  | mf_trainable  | none         | —           | —          | —            | —           | —          | —          |
| 3     | derivative  | 32  | mf_frozen     | none         | —           | —          | —            | 95.59       | —          | —          |
| 4     | both        | 32  | mf_frozen     | none         | —           | —          | —            | —           | —          | 93.92      |
| 5     | derivative  | 32  | mf_trainable  | none         | (lower)     | (lower)    | (lower)      | —           | —          | —          |
| **6** | **derivative** | **32** | **mf_trainable** | **none** | **98.18** | **92.17** | **92.44** | **85.30** | **96.34** | **91.90** |

Notes:
- Model 6 is the paper's best model; it uses 3 classes (N, S, V).
- Model 5 uses 4 classes (N, S, V, F); the last two models (4-class and 5-class)
  yield the lowest averages because F and Q beats are hard to classify.
- The abstract reports: best avg SEN 91.90 %, SVEB SEN 85.3 %, VEB SEN 96.34 %.
- NK = kernel size (and input length after NK=32 centre-crop).
- "Avg ACC" in the paper is the mean one-vs-rest (OvR) accuracy, NOT plain
  multiclass accuracy.  For K=3 and OvR accuracy A: plain accuracy ≈ (3A-1)/2.
  At A=98.18 %, plain accuracy ≈ 97.27 %.  See README.

## RR-only XGBoost baseline (paper Sec 5 sanity check)

| Metric       | Paper | Ours |
|--------------|-------|------|
| Accuracy (%) | ~93   | —    |
| Recall N (%) | ~97   | —    |
| Recall S (%) | ~60   | —    |
| Recall V (%) | ~58   | —    |

(Fill "Ours" column after running `python -m src.run_experiment --rr-baseline`.)
