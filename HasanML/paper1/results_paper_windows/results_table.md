# Results: Standard ESN, per-subject PVC detection (test window 10:00–15:00)

Our numbers are mean ± std over 5 reservoir seed(s), in percent. Δ = ours − paper (Liu 2019, Table 4.1, Standard ESN), in percentage points; **bold** means |Δ| > 2 pp. "All-normal acc" is the accuracy of predicting every beat as non-PVC.

## Paper (Table 4.1, Standard ESN)

| Record | Accuracy | Sensitivity | Precision | F1 |
|---|---|---|---|---|
| 106 | 98.75 | 97.22 | 97.22 | 97.22 |
| 119 | 99.70 | 100.00 | 99.10 | 99.55 |
| 200 | 99.07 | 98.24 | 99.40 | 98.82 |
| 201 | 99.24 | 100.00 | 97.18 | 98.57 |
| 203 | 98.14 | 100.00 | 90.32 | 94.92 |
| 223 | 99.07 | 99.05 | 98.11 | 98.58 |
| 233 | 99.78 | 100.00 | 99.21 | 99.60 |

## Variant overview (mean over records)

| Input | Leak a | Lambda | Weighting | Mean accuracy | Mean F1 | Mean ΔF1 | Records with \|ΔF1\| > 2 pp |
|---|---|---|---|---|---|---|---|
| spikes | 0.99 | cv | none | 97.90 | 97.21 | -1.97 | 1 / 2 |
| spikes | 0.99 | fixed | none | 93.68 | 91.25 | -7.94 | 2 / 2 |
| spikes | 0.99 | cv | balanced | 95.83 | 94.04 | -5.15 | 2 / 2 |
| spikes | 0.99 | fixed | balanced | 93.42 | 90.81 | -8.38 | 2 / 2 |
| raw | 0.99 | cv | none | 96.15 | 94.14 | -5.04 | 2 / 2 |
| raw | 0.99 | fixed | none | 98.08 | 97.45 | -1.73 | 1 / 2 |
| raw | 0.99 | cv | balanced | 98.46 | 97.98 | -1.20 | 1 / 2 |
| raw | 0.99 | fixed | balanced | 93.48 | 93.26 | -5.92 | 1 / 2 |

## input=spikes, leak a=0.99, lambda=cv, weighting=none (primary)

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 100.00 ± 0.00 | +0.30 | 100.00 ± 0.00 | +0.00 | 100.00 ± 0.00 | +0.90 | 100.00 ± 0.00 | +0.45 | 100.00 ± 0.00 |
| 200 | 170 / 259 | 60.37 | 95.80 ± 0.51 | **-3.27** | 89.76 ± 1.56 | **-8.48** | 99.62 ± 0.51 | +0.22 | 94.42 ± 0.72 | **-4.40** | 99.77 ± 0.31 |

## input=spikes, leak a=0.99, lambda=fixed, weighting=none

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 96.53 ± 6.20 | **-3.17** | 93.64 ± 11.00 | **-6.36** | 95.61 ± 8.31 | **-3.49** | 94.59 ± 9.72 | **-4.96** | 97.99 ± 3.79 |
| 200 | 170 / 259 | 60.37 | 90.82 ± 6.44 | **-8.25** | 83.88 ± 7.79 | **-14.36** | 92.42 ± 9.50 | **-6.98** | 87.91 ± 8.38 | **-10.91** | 95.37 ± 5.79 |

## input=spikes, leak a=0.99, lambda=cv, weighting=balanced

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 95.99 ± 5.35 | **-3.71** | 94.73 ± 10.55 | **-5.27** | 93.11 ± 6.44 | **-5.99** | 93.84 ± 8.42 | **-5.71** | 96.62 ± 2.90 |
| 200 | 170 / 259 | 60.37 | 95.66 ± 0.11 | **-3.41** | 89.41 ± 0.37 | **-8.83** | 99.61 ± 0.32 | +0.21 | 94.23 ± 0.16 | **-4.59** | 99.77 ± 0.19 |

## input=spikes, leak a=0.99, lambda=fixed, weighting=balanced

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 96.35 ± 3.52 | **-3.35** | 98.55 ± 2.12 | -1.45 | 91.74 ± 7.32 | **-7.36** | 94.92 ± 4.74 | **-4.63** | 95.25 ± 4.44 |
| 200 | 170 / 259 | 60.37 | 90.49 ± 3.89 | **-8.58** | 79.06 ± 6.91 | **-19.18** | 96.15 ± 4.19 | **-3.25** | 86.70 ± 5.68 | **-12.12** | 97.99 ± 2.16 |

## input=raw, leak a=0.99, lambda=cv, weighting=none

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 96.60 ± 6.36 | **-3.10** | 89.82 ± 19.01 | **-10.18** | 100.00 ± 0.00 | +0.90 | 93.38 ± 12.56 | **-6.17** | 100.00 ± 0.00 |
| 200 | 170 / 259 | 60.37 | 95.71 ± 4.16 | **-3.36** | 95.29 ± 1.34 | **-2.95** | 95.12 ± 9.14 | **-4.28** | 94.91 ± 4.47 | **-3.91** | 95.98 ± 7.65 |

## input=raw, leak a=0.99, lambda=fixed, weighting=none

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 99.76 ± 0.35 | +0.06 | 99.27 ± 1.06 | -0.73 | 100.00 ± 0.00 | +0.90 | 99.63 ± 0.54 | +0.08 | 100.00 ± 0.00 |
| 200 | 170 / 259 | 60.37 | 96.41 ± 1.06 | **-2.66** | 91.65 ± 2.67 | **-6.59** | 99.26 ± 1.20 | -0.14 | 95.27 ± 1.45 | **-3.55** | 99.54 ± 0.75 |

## input=raw, leak a=0.99, lambda=cv, weighting=balanced

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 100.00 ± 0.00 | +0.30 | 100.00 ± 0.00 | +0.00 | 100.00 ± 0.00 | +0.90 | 100.00 ± 0.00 | +0.45 | 100.00 ± 0.00 |
| 200 | 170 / 259 | 60.37 | 96.92 ± 0.58 | **-2.15** | 92.47 ± 1.41 | **-5.77** | 99.75 ± 0.31 | +0.35 | 95.97 ± 0.78 | **-2.85** | 99.85 ± 0.19 |

## input=raw, leak a=0.99, lambda=fixed, weighting=balanced

| Record | Test PVC / other | All-normal acc | Accuracy | Δ | Sensitivity | Δ | Precision | Δ | F1 | Δ | Specificity |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 119 | 110 / 219 | 66.57 | 99.27 ± 1.46 | -0.43 | 99.82 ± 0.36 | -0.18 | 98.17 ± 3.67 | -0.93 | 98.96 ± 2.09 | -0.59 | 99.00 ± 2.01 |
| 200 | 170 / 259 | 60.37 | 87.69 ± 15.70 | **-11.38** | 89.88 ± 1.64 | **-8.36** | 87.89 ± 20.53 | **-11.51** | 87.57 ± 13.25 | **-11.25** | 86.25 ± 25.06 |
