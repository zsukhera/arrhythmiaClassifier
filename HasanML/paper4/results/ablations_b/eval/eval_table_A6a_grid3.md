# Eval A6a_grid3 (ablations_b)

## DS2 5-class AAMI (Se/PPV %)

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 67.28 | 86.91 | 29.79 | 68.2/96.2 | 18.2/3.1 | 91.0/49.1 | 0.0/0.0 | 0.0/0.0 |

## DS2 3-class N/S/V (F/Q beats dropped, F/Q columns dropped + renormalised)

| model | acc | OvR acc | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 68.63 | 79.09 | 47.54 | 69.1/96.7 | 18.2/3.2 | 91.1/41.1 |

## DS2 3-class strict (5-way argmax; N/S/V beat predicted F/Q = error)

| model | acc | OvR acc(K=3) | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 67.82 | 78.55 | 50.11 | 68.2/96.7 | 18.2/3.2 | 91.0/50.5 |

## DS1-val (selection set) 5-class

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 81.53 | 92.61 | 37.86 | 82.5/97.6 | 58.0/21.0 | 93.3/54.5 | 0.2/0.2 | 0.0/0.0 |

## Calibration (T fitted on DS1-val)

| model | T | ECE val before | after | ECE DS2 before | after |
|---|---|---|---|---|---|
| seed0 | 1.263 | 0.0403 | 0.0416 | 0.2003 | 0.1440 |

Selected by DS1-val 5-class macro-F1: **seed0**

Mean±std over single seeds (test_5c): acc 67.28±0.00, OvR 86.91±0.00, macroF1 29.79±0.00, Se_S 18.18±0.00, Se_V 91.03±0.00

Mean±std over single seeds (test_3c): acc 68.63±0.00, OvR 79.09±0.00, macroF1 47.54±0.00, Se_S 18.18±0.00, Se_V 91.06±0.00
