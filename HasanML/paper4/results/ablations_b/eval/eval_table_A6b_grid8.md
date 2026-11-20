# Eval A6b_grid8 (ablations_b)

## DS2 5-class AAMI (Se/PPV %)

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 85.78 | 94.31 | 36.93 | 89.5/95.8 | 15.4/6.3 | 84.6/79.0 | 1.5/1.4 | 0.0/0.0 |

## DS2 3-class N/S/V (F/Q beats dropped, F/Q columns dropped + renormalised)

| model | acc | OvR acc | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 87.32 | 91.55 | 61.44 | 90.5/96.7 | 15.4/6.3 | 84.7/79.2 |

## DS2 3-class strict (5-way argmax; N/S/V beat predicted F/Q = error)

| model | acc | OvR acc(K=3) | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 86.45 | 90.97 | 61.26 | 89.5/96.7 | 15.4/6.3 | 84.6/79.3 |

## DS1-val (selection set) 5-class

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 89.67 | 95.87 | 39.84 | 94.2/94.8 | 24.9/22.1 | 85.4/76.8 | 0.2/0.6 | 0.0/0.0 |

## Calibration (T fitted on DS1-val)

| model | T | ECE val before | after | ECE DS2 before | after |
|---|---|---|---|---|---|
| seed0 | 1.472 | 0.0543 | 0.0359 | 0.0934 | 0.0762 |

Selected by DS1-val 5-class macro-F1: **seed0**

Mean±std over single seeds (test_5c): acc 85.78±0.00, OvR 94.31±0.00, macroF1 36.93±0.00, Se_S 15.35±0.00, Se_V 84.57±0.00

Mean±std over single seeds (test_3c): acc 87.32±0.00, OvR 91.55±0.00, macroF1 61.44±0.00, Se_S 15.35±0.00, Se_V 84.66±0.00
