# farag-gap: Farag 2023 replication, gap to the paper's Table 4 (tasks A1–A5)

Agent: farag-gap · 2026-11-18 · all computation on the box, CPU only, 2 threads; DS2 was used for evaluation only.

## TL;DR
1. **Our targets were wrong.**
   - Model 6, the paper's best model, has **SVEB Se 81.60 and VEB Se 95.00**.
   - The abstract's 85.30 / 96.34 come from **Model 1**, which uses class weights.
   - "Se_V 89.7" and "Se_N 97.8" appear nowhere in the paper.
2. **The paper's "average accuracy" is plain accuracy, not OvR.** Its averages are unweighted macro means over N/S/V. The numbers reproduce exactly, so our 91.61 has to be compared with 98.18, not 94.41 with 98.18.
3. **Our test set matches the paper exactly**: DS2 N/S/V = 44,259 / 1,837 / 3,221, with F and Q dropped. DS1 matches too. The data is not the cause of the gap.
4. **The true gap is precision, mainly SVEB.**
   - Our PPV_S is 32.1 against 82.68 for Model 6 and 58.06 for Model 1.
   - Our sensitivities are at or above every paper model.
   - Re-thresholding cannot close the gap. Even with offsets tuned on DS2 itself (a diagnostic oracle), the best macro F1 is 82.4, against 92.17.
5. **Our official P1 uses balanced class weights, so it is not comparable to Model 6**, which has no class weights. The same-config paper row is Model 2: acc 95.58, avg F1 84.79, PPV_S 51.49.
6. **The RR-only XGBoost broadly reproduces the paper** when trained on full DS1: acc 93.7, Se_N 97.4, V F1 70.8. This makes an RR-feature bug unlikely. The SVEB recall depends heavily on the training set and weights (12.7–74.6).
7. **The paper's Table 3 architecture differs from ours.** It uses 11 kernels for 3 classes, BN followed by **tanh**, a 'same'-length conv over the full 64-sample input, and an RR branch of Dense 32/16/8 applied per value then Flattened, for 1,267 parameters. Ours uses 13 kernels, relu, a 32-sample valid crop and Dense(16)×3, for 1,195 parameters. This is the best lead for task C1.

## A1: Paper check
The source is the full text from Europe PMC (PMC9919183) via `curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9919183/fullTextXML"`. MDPI blocks automated fetches with a 403.

**Model 6, DS2:**

| | PREC | SEN | F1 |
|---|---|---|---|
| N | 99.00 | 99.10 | 99.05 |
| SVEB | 82.68 | 81.60 | 82.14 |
| VEB | 95.63 | 95.00 | 95.31 |
| Average | 92.44 | 91.90 | 92.17 |

ACC is 98.18. Model 6 is 3-class, trainable MF, NK=32, derivative input, **class weight NOT SET**.

**What checked out and what did not:**
- Avg ACC 98.18, avg F1 92.17, avg PREC 92.44 and avg Se 91.90 are all correct.
- "SVEB Se 85.30" and "VEB Se 96.34" are correct numbers but belong to **Model 1**, not Model 6.
- The paper reports no specificity values and no confusion matrix.
- The paper's beat-count table (Table 2) matches ours.
- Metric definitions are in Sec. 5.3: ACC = (TP+TN)/all, PREC = TP/(TP+FP), SEN = TP/(TP+FN), F1 = 2TP/(2TP+FP+FN). Per-class values are "averaged for the whole dataset". Each model's scores are the mean of 10 runs.
- The other training details are Adam with lr 1e-3, adaptive LR, 500 epochs, batch 512 and early stopping. It trains on DS1 records below 200 and validates on records 200 and above.

**Fixes made:**
- `utils.py` footer now prints `Paper (Farag23 T4 Model 6, no CW)  98.18  99.10  81.60  95.00  92.170`.
- `paper_table4.md` has been rewritten with the full Table 4 (all 9 models), Table 2, the Table 3 architecture differences and the RR baseline. The earlier version had wrong configs and values for Models 1–5.

## A2: Full metric set, official ensemble (P1_fullDS1 seeds 0–9, DS2)
With 3 classes, P(S) ≥ 0.5 already implies argmax = S. So **tau_S = 0.50 is identical to argmax**: 0 of 49,317 predictions differ. The "calibrated" result is just argmax.

| Metric | Ensemble argmax (= tau 0.50) | Mean of 10 single seeds (paper protocol) |
|---|---|---|
| Plain acc | 91.61 | 91.03 |
| OvR-mean acc | 94.41 | 94.02 |
| Se N / S / V | 91.44 / 87.10 / 96.52 | 90.97 / 85.29 / 95.14 |
| PPV N / S / V | 99.79 / 32.09 / 82.40 | 99.73 / 31.25 / 78.40 |
| Sp N / S / V | 98.28 / 92.87 / 98.56 | 97.84 / 92.74 / 98.13 |
| F1 N / S / V | 95.43 / 46.90 / 88.90 | 95.15 / 45.72 / 85.85 |
| Macro Se / PPV / Sp / F1 | 91.69 / 71.43 / 96.57 / 77.08 | 90.47 / 69.79 / 96.23 / 75.57 |
| Weighted Se / PPV / Sp / F1 | 91.61 / 96.13 / 98.10 / 93.20 | 91.03 / 95.79 / 97.67 / 92.70 |

Ensemble confusion matrix (rows are true N, S, V; columns are predicted):

|  | Pred N | Pred S | Pred V |
|---|---|---|---|
| True N | 40471 | 3310 | 478 |
| True S | 51 | 1600 | 186 |
| True V | 36 | 76 | 3109 |

**3,310 N→S errors are the whole story.**

On the DS1 random-15% hold-out (ensemble), plain acc is 96.24, Se_S 97.2 and PPV_S 44.8. This hold-out is patient-leaky, so it is much easier than DS2.

`ensemble_probs.npz` was verified against the cached `proba_test` and `proba_val`, with a maximum difference of 0.

## A3: Which averaging the paper uses
These values are computed from the paper's own per-class numbers. DS2 counts are used for the weighted and accuracy reconstructions.

| Definition | Model 6 | Paper value |
|---|---|---|
| Macro PREC / SEN / F1 | 92.44 / 91.90 / 92.17 | 92.44 / 91.90 / 92.17 ✓ |
| Weighted PREC / SEN / F1 | 98.17 / 98.18 / 98.18 | ✗ |
| F1 of (avg PREC, avg SEN) | 92.17 | also ✓ (indistinguishable here); for Model 1 it gives 87.41 vs 86.81 → per-class-F1 macro mean ✓ |
| Plain acc = Σ n_c·Se_c / N | 98.18 (Model 1: 96.49) | 98.18 / 96.48 ✓ |
| OvR-mean acc implied | 98.79 | ✗ |

**Conclusion:** ACC is plain accuracy on 49,317 N/S/V beats. Averages are unweighted macro means of the per-class metrics.

Our numbers under the paper's definitions: acc **91.61**, avg PREC **71.43**, avg SEN **91.69**, avg F1 **77.08**. Under a weighted average our F1 would be 93.2, but that is not what the paper uses.

## A4: Test-set composition
- **DS2:** N 44,259 / S 1,837 / V 3,221 / total 49,317. This equals paper Table 2's 49,712 minus F 388 and Q 7.
- **DS1:** 45,866 / 944 / 3,788, which matches the paper exactly.
- **F/Q handling:** F and Q are dropped from training and test sets, which is what the paper does ("Results of classes F and Q do not exist for classifiers with 3 output classes"). They are still used for RR intervals.
- **Edge beats:** no first or last beats are dropped. Edge beats are edge-padded, at most 2 per record.
- **Other annotations:** non-beat annotations (+ ~ | x ! " [ ]) are excluded. The kept symbols per split are in `gap_metrics.log`.

**Mismatch: 0 beats.**

## A5: RR-only XGBoost
`--rr-baseline` was run with a private config, `1DCNN/gap_analysis/rr_baseline_config.yaml`, so all outputs went to `gap_analysis/rr_baseline/`. I checked with md5 before and after that `HasanML/paper3/results` was unchanged.

xgboost was **not installed** in either venv, so the CLI had silently skipped the baseline. I installed it with `--no-deps --target 1DCNN/gap_analysis/pylib`; the shared venvs are untouched.

| Variant | Acc | Se N/S/V | PPV N/S/V |
|---|---|---|---|
| Paper | 93 | 97/60/58 | 97/36/75 |
| CLI (fit DS1-train 12 rec) | 91.64 | 96.5/12.7/69.3 | 97.3/17.3/55.0 |
| full DS1, 300×d6 | 92.74 | 96.5/33.2/74.5 | 97.1/54.3/57.4 |
| full DS1, 100×d3 | 93.73 | 97.4/30.8/79.3 | 97.1/60.8/63.9 |
| full DS1, balanced, 300×d6 | 88.54 | 90.6/58.7/76.6 | 99.3/41.9/39.0 |
| full DS1, balanced, 100×d3 | 86.46 | 88.5/74.6/65.3 | 99.4/27.7/42.4 |

Accuracy and N are reproduced, and V is in the right range. The paper's exact S/V trade-off (S 60/36, V 58/75) is not reproduced by any single variant. The paper's XGBoost settings are unspecified, so this is weak evidence either way. It does not indicate an RR-feature bug.

## Can thresholds close the gap? (diagnostic)
`oracle_offsets.py` adds log-prior offsets for S and V to the ensemble.

| Offsets chosen on | DS2 acc | DS2 macro F1 | Se N/S/V | PPV N/S/V |
|---|---|---|---|---|
| DS1-val, max macro F1 (legit) | 95.16 | 80.68 | 96.5/63.8/94.8 | 98.7/44.9/88.9 |
| DS1-val, max acc (legit) | 95.21 | 75.22 | 97.7/34.5/95.4 | 97.8/43.9/83.9 |
| **DS2 ORACLE**, max macro F1 | 95.10 | 82.37 | 96.2/75.8/90.8 | 98.6/45.3/95.3 |
| **DS2 ORACLE**, max acc | 95.40 | 78.76 | 97.7/47.2/91.9 | 97.7/47.2/90.9 |

The DS2-oracle rows are an upper bound only and must not be reported as results. Even so, they top out at about 95.4 acc and 82.4 F1, and PPV_S never goes above about 47. Model 6 has PPV_S 82.7 at Se_S 81.6. **The gap is in the model's S-vs-N discrimination, not in calibration.**

## Final table: ours vs paper under the paper's definitions (DS2)

| Metric | Ours ensemble | Ours seed-mean | Paper Model 6 (no CW) | Gap vs M6 | Paper Model 2 (same config as ours: CW) | Gap vs M2 |
|---|---|---|---|---|---|---|
| ACC (plain) | 91.61 | 91.03 | 98.18 | −6.6 | 95.58 | −4.0 |
| Avg PREC (macro) | 71.43 | 69.79 | 92.44 | −21.0 | 79.63 | −8.2 |
| Avg SEN (macro) | 91.69 | 90.47 | 91.90 | −0.2 | 93.34 | −1.7 |
| Avg F1 (macro) | 77.08 | 75.57 | 92.17 | −15.1 | 84.79 | −7.7 |
| Se N | 91.44 | 90.97 | 99.10 | −7.7 | 95.86 | −4.4 |
| Se S | 87.10 | 85.29 | 81.60 | +5.5 | 88.51 | −1.4 |
| Se V | 96.52 | 95.14 | 95.00 | +1.5 | 95.65 | +0.9 |
| PPV N | 99.79 | 99.73 | 99.00 | +0.8 | 99.47 | +0.3 |
| PPV S | 32.09 | 31.25 | 82.68 | **−50.6** | 51.49 | **−19.4** |
| PPV V | 82.40 | 78.40 | 95.63 | −13.2 | 87.93 | −5.5 |
| F1 S | 46.90 | 45.72 | 82.14 | −35.2 | 65.11 | −18.2 |
| F1 V | 88.90 | 85.85 | 95.31 | −6.4 | 91.63 | −2.7 |

**The true gap:**
- Against the like-for-like paper config (Model 2), we are about 4 points behind in accuracy and about 8 in macro F1. Almost all of it is Se_N / PPV_S, i.e. too many N→S false positives.
- Against Model 6 the gap is 6.6 points in accuracy and 15 in macro F1. Model 6 is a different operating point: it trades S sensitivity for precision and uses no class weights.
- Our earlier "we exceed the paper's Se_S / Se_V" claim is only true against Model 6's lower sensitivities. On sensitivity we are on par with the class-weighted paper models.

## Recommendations
These are for the B/C tasks owned by others.
1. **Run B1 (full DS1, no class weights).** This is the actual Model 6 config. Compare its result with Model 6, not Model 1.
2. **Run C1: match the Table 3 architecture.** That means 11 kernels for 3 classes (no F/Q templates), BN followed by tanh, 'same' conv over the 64-sample input, and the RR branch as Dense 32/16/8 on (4,1) then Flatten. This is the best remaining fidelity lead.
3. **Report the seed-mean, not the ensemble**, when comparing with the paper.
4. **Fix progress.md and the weekly summary.** Their targets are wrong: they use Se_V 89.7 / Se_N 97.8 and treat acc as OvR. I did not edit them because they are not my files.

## Reproduce (from $ROOT on the box)
```bash
OMP_NUM_THREADS=2 .venv/bin/python 1DCNN/gap_analysis/gap_metrics.py > 1DCNN/gap_analysis/gap_metrics.log 2>&1
OMP_NUM_THREADS=2 .venv/bin/python 1DCNN/gap_analysis/oracle_offsets.py
cd HasanML/paper3 && OMP_NUM_THREADS=2 PYTHONPATH=$ROOT/1DCNN/gap_analysis/pylib ../../.venv/bin/python -m src.run_experiment \
    --rr-baseline --config $ROOT/1DCNN/gap_analysis/rr_baseline_config.yaml --device cpu; cd ../..
PYTHONPATH=1DCNN/gap_analysis/pylib OMP_NUM_THREADS=2 .venv/bin/python 1DCNN/gap_analysis/rr_baseline_variants.py
```

## Files
**Created** (all under `$ROOT/1DCNN/gap_analysis/` unless noted):
- `gap_metrics.py`, `gap_metrics.json`, `gap_metrics.log`
- `ensemble_probs.npz`, which contains:
  - `ds{1,2}_proba` (ensemble) and `ds{1,2}_proba_seeds` (10×N×3)
  - `ds{1,2}_y`, `ds{1,2}_record`, `ds{1,2}_r360`, `ds{1,2}_r128` (R positions at 360 Hz and 128 Hz) and `ds{1,2}_symbol`
  - `ds1_is_val` (the random-15% hold-out; all other DS1 beats were training data)
  - `classes`, `tau_S`, `note`
- `oracle_offsets.py`, `oracle_offsets.json`, `oracle_offsets.log`
- `rr_baseline_variants.py`, `rr_baseline_config.yaml`
- `rr_baseline/`: CLI log, run.log, resolved_config, beat_counts, `rr_baseline_xgboost.csv`, `rr_baseline_variants.{json,log}`, and the md5 snapshot of paper3/results
- `pylib/` (xgboost 3.4.1)
- Backups: `utils.py.bak_before_farag_gap`, `paper_table4.md.bak_before_farag_gap`
- `$ROOT/reports/farag-gap.md` and `$ROOT/reports/farag-gap.RESUME.md`

**Modified:**
- `$ROOT/1DCNN/1DCNN experiments/utils.py` (only the paper-target print line)
- `$ROOT/HasanML/paper3/results/paper_table4.md` (rewritten)

## Open issues
- **Two unreported bugs in `src/run_experiment.py`** (not my file):
  - `--rr-baseline` silently skips when xgboost is missing.
  - It always writes `run.log`, `resolved_config.yaml`, `beat_counts.md` and the templates figure into `results/`, even for `--rr-baseline`. Use `--config` with other output dirs to avoid overwriting.
- **The S template has 0 beats in DS1-train** (only 2 'S' symbols exist in all of DS1, below min_beats=5), so it is noisy. This becomes moot if the paper's 11-kernel version is used.
- **The paper's XGBoost hyper-parameters and training set are unspecified.**
- **README.md and progress.md still claim the paper's accuracy is OvR.** This needs correcting by their owners.
