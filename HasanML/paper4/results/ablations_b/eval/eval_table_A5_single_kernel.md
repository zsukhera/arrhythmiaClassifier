# Eval A5_single_kernel (ablations_b)

## DS2 5-class AAMI (Se/PPV %)

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 89.74 | 95.90 | 39.20 | 93.6/96.0 | 18.2/10.4 | 87.8/86.3 | 0.5/6.5 | 0.0/0.0 |

## DS2 3-class N/S/V (F/Q beats dropped, F/Q columns dropped + renormalised)

| model | acc | OvR acc | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 90.51 | 93.67 | 65.21 | 93.7/96.8 | 18.2/10.4 | 87.8/86.5 |

## DS2 3-class strict (5-way argmax; N/S/V beat predicted F/Q = error)

| model | acc | OvR acc(K=3) | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 90.45 | 93.64 | 65.20 | 93.6/96.8 | 18.2/10.4 | 87.8/86.5 |

## DS1-val (selection set) 5-class

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 86.61 | 94.65 | 38.71 | 90.2/95.5 | 37.1/25.3 | 85.0/60.6 | 0.0/0.0 | 0.0/0.0 |

## Calibration (T fitted on DS1-val)

| model | T | ECE val before | after | ECE DS2 before | after |
|---|---|---|---|---|---|
| seed0 | 1.500 | 0.0752 | 0.0421 | 0.0678 | 0.0452 |

Selected by DS1-val 5-class macro-F1: **seed0**

Mean±std over single seeds (test_5c): acc 89.74±0.00, OvR 95.90±0.00, macroF1 39.20±0.00, Se_S 18.18±0.00, Se_V 87.77±0.00

Mean±std over single seeds (test_3c): acc 90.51±0.00, OvR 93.67±0.00, macroF1 65.21±0.00, Se_S 18.18±0.00, Se_V 87.83±0.00
