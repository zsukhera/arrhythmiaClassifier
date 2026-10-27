# Beat counts vs. paper Table 1

Label map: `paper`; records: 44 (excluded: 102, 104, 107, 217).

## All beats

| Class | Paper total | Ours | Diff |
|---|---:|---:|---:|
| N | 90083 | 90087 | +4 |
| S | 2972 | 2972 | +0 |
| V | 7480 | 7480 | +0 |
| F | 802 | 802 | +0 |
| Q | 15 | 15 | +0 |
| All | 101352 | 101356 | +4 |

## Protocol A: inter-patient DS1/DS2

| Class | Paper train | Ours train | Diff | Paper test | Ours test | Diff |
|---|---:|---:|---:|---:|---:|---:|
| N | 45845 | 45846 | +1 | 44238 | 44241 | +3 |
| S | 1000 | 1000 | +0 | 1972 | 1972 | +0 |
| V | 4260 | 4260 | +0 | 3220 | 3220 | +0 |
| F | 414 | 414 | +0 | 388 | 388 | +0 |
| Q | 8 | 8 | +0 | 7 | 7 | +0 |
| All | 51527 | 51528 | +1 | 49825 | 49828 | +3 |

## Protocol B: intra-patient stratified split

| Class | Paper train | Ours train | Diff | Paper test | Ours test | Diff |
|---|---:|---:|---:|---:|---:|---:|
| N | 45845 | 45847 | +2 | 44238 | 44240 | +2 |
| S | 1000 | 1000 | +0 | 1972 | 1972 | +0 |
| V | 4260 | 4260 | +0 | 3220 | 3220 | +0 |
| F | 414 | 414 | +0 | 388 | 388 | +0 |
| Q | 8 | 8 | +0 | 7 | 7 | +0 |
| All | 51527 | 51529 | +2 | 49825 | 49827 | +2 |

## Protocol C: 5-fold stratified CV (fold 1 of 5 shown; folds differ by at most 1 beat per class)

| Class | Paper train | Ours train | Diff | Paper test | Ours test | Diff |
|---|---:|---:|---:|---:|---:|---:|
| N | 45845 | 72069 | +26224 | 44238 | 18018 | -26220 |
| S | 1000 | 2378 | +1378 | 1972 | 594 | -1378 |
| V | 4260 | 5984 | +1724 | 3220 | 1496 | -1724 |
| F | 414 | 641 | +227 | 388 | 161 | -227 |
| Q | 8 | 12 | +4 | 7 | 3 | -4 |
| All | 51527 | 81084 | +29557 | 49825 | 20272 | -29553 |

Dropped beats are listed in `dropped_beats.csv`.
