# Eval DEFAULT (sanity_gpu)

## DS2 5-class AAMI (Se/PPV %)

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 75.02 | 90.01 | 28.76 | 77.0/96.6 | 11.9/3.9 | 93.3/35.5 | 0.8/0.9 | 0.0/0.0 |

## DS2 3-class N/S/V (F/Q beats dropped, F/Q columns dropped + renormalised)

| model | acc | OvR acc | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 76.06 | 84.04 | 47.93 | 77.4/97.3 | 11.9/3.9 | 93.7/35.7 |

## DS2 3-class strict (5-way argmax; N/S/V beat predicted F/Q = error)

| model | acc | OvR acc(K=3) | macro F1 | N | S | V |
|---|---|---|---|---|---|---|
| seed0 | 75.62 | 83.75 | 47.93 | 77.0/97.3 | 11.9/3.9 | 93.3/36.0 |

## DS1-val (selection set) 5-class

| model | acc | OvR acc | macro F1 | N | S | V | F | Q |
|---|---|---|---|---|---|---|---|---|
| seed0 | 84.96 | 93.98 | 35.75 | 88.4/95.4 | 18.6/12.7 | 89.8/59.6 | 0.2/0.3 | 0.0/0.0 |

## Calibration (T fitted on DS1-val)

| model | T | ECE val before | after | ECE DS2 before | after |
|---|---|---|---|---|---|
| seed0 | 1.090 | 0.0312 | 0.0348 | 0.0992 | 0.0848 |

Selected by DS1-val 5-class macro-F1: **seed0**

Mean±std over single seeds (test_5c): acc 75.02±0.00, OvR 90.01±0.00, macroF1 28.76±0.00, Se_S 11.92±0.00, Se_V 93.26±0.00

Mean±std over single seeds (test_3c): acc 76.06±0.00, OvR 84.04±0.00, macroF1 47.93±0.00, Se_S 11.92±0.00, Se_V 93.67±0.00
