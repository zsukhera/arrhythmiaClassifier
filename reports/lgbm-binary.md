# lgbm-binary — feature-group ablation + binary LightGBM on the normalised features

Agent: lgbm-binary · 2026-11-18 → 11-19 · GPU box `~/Desktop/arrhythmiaClassifier` ($ROOT), CPU only, 6 threads.
Features: `datasets/processed/split_{random,ds1ds2}.npz` (per-record amplitude + RR normalised, 2026-11-13).
DS2 was used only for the final test numbers. It was never used for training, tuning, threshold choice or early stopping.

## TL;DR

- **Ablation, random split:** still at ceiling (A4 acc 0.995, anomaly F1 0.992, AUC 1.000), well above the paper. Only A1 changed: with the new normalisation, statistical features alone are much weaker (acc 0.924 → 0.886). The A1→A2 jump (+0.108 acc) is still far larger than the paper's (+0.002).
- **Ablation, DS1→DS2:** no feature group transfers LBBB/RBBB. L/R recall is about 0 in every config except "HRV/NonLin only", where L recall is 0.62 but V collapses. Adding groups past fiducial does not help cross-patient: A2 has the best 5-class anomaly F1 among A1–A4. Dropping the wavelet group is the best of all configs tried (binary F1 0.507, AUC 0.783).
- **Binary LightGBM, DS2** (grouped-CV-tuned, threshold from grouped OOF on DS1): **P 0.398, R 0.744, F1 0.519, AUC 0.715**. The 5-class model collapsed to binary gets F1 0.409 / AUC 0.741. The paper reports F1 0.928 / AUC 0.985. F1 improves by +0.11, but only because the operating point moves (recall up, N specificity 0.93 → 0.61). The ranking quality (AUC) is *worse* than the 5-class model's. At threshold 0.5 the binary model is worse than the 5-class model (F1 0.362).
- **Recall by original class on DS2** (binary @OOF threshold): L 0.683, R 0.457, V 0.999, A 0.987. The 5-class collapsed model gets L 0.015, R 0.070, V 0.979, A 0.240.
- **Patient leakage confirmed:** on the same DS1 data, beat-level 5-fold CV gives F1 0.994 / AUC 0.9999. Record-grouped CV gives F1 0.708 / AUC 0.867. DS2 then gives 0.52 / 0.72. Grouped CV is a much more honest estimate, though still optimistic.

- **Extra DS1/DS2 variants**, all with the threshold chosen on grouped OOF:

  | Variant | DS2 F1 | DS2 AUC |
  |---|---|---|
  | paper params | 0.513 | 0.767 |
  | AUC-objective study (60 trials) | 0.507 | 0.763 |
  | best-F1 study (main) | 0.519 | 0.715 |
  | beat-level stratified CV on DS1 (threshold stays 0.5) | 0.408 | 0.751 |

  Every grouped-threshold variant lands at F1 ≈ 0.51–0.52, so **the gain comes from choosing the threshold on patient-grouped OOF, not from the tuning.** The grouped-CV ranking of variants does not predict the DS2 ranking. The best grouped OOF F1 (0.708) gives the worst DS2 AUC. This matches lgbm-groupcv's finding that grouped CV is only a weak guide to DS2.

---

## 1. Ablation A1→A4

### 1a. Random split — `evaluate.run_ablation()` (unchanged code, paper params, 5-class + SMOTE)

Run through `source/ablation/ablation.py --mode random`. That script imports `evaluate` and only swaps `evaluate.LGBMClassifier` for a factory that forces `n_jobs=6`. Log: `source/ablation/results/ablation_random.log`.

| Config | #F | Acc (new) | AnomRec | AnomF1 (new) | AUC | Acc (old, progress.md) | AnomF1 (old) | Paper Acc / F1 / AUC |
|---|---|---|---|---|---|---|---|---|
| A1 Statistical | 6 | 0.886 | 0.928 | 0.819 | 0.970 | 0.924 | 0.875 | 0.985 / 0.860 / 0.971 |
| A2 +Fiducial | 19 | 0.994 | 0.985 | 0.988 | 0.999 | 0.994 | 0.990 | 0.987 / 0.901 / 0.979 |
| A3 +Wavelet | 31 | 0.994 | 0.988 | 0.990 | 1.000 | 0.995 | 0.991 | 0.988 / 0.921 / 0.983 |
| A4 +HRV/NonLin | 40 | 0.995 | 0.990 | 0.992 | 1.000 | 0.996 | 0.993 | 0.989 / 0.928 / 0.985 |

What this shows:
- A2–A4 are unchanged from the old features (within 0.002) and beat the paper on every column.
- A1 dropped (acc −0.038, F1 −0.056). Per-record amplitude normalisation strips the absolute-amplitude information (mean, std, ptp, rms) that made the statistical group a patient-identity proxy on the random split.
- The paper's ablation curve is flat (0.985 → 0.989). Ours is steep at A1→A2 (+0.108). The paper's A1 accuracy of 0.985 with 6 statistical features is not reproducible with our features.

### 1b. DS1 → DS2 — own script `source/ablation/ablation.py --mode ds1ds2 --extras`

Same paper params and SMOTE. For each config two models are trained on DS1 and tested on DS2: the 5-class model (as in 1a) and a binary-objective model at threshold 0.5. Results: `source/ablation/results/ablation_ds1ds2.{log,json}`.

| Config | #F | 5c Acc | 5c macro-F1 | F1 N | F1 L | F1 R | F1 V | F1 A | collapsed AnomF1 | collapsed AUC | bin P | bin R | bin F1 | bin AUC | rec L | rec R | rec V | rec A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 Statistical | 6 | 0.675 | 0.300 | 0.811 | 0.088 | 0.004 | 0.577 | 0.022 | 0.367 | 0.648 | 0.491 | 0.300 | 0.373 | 0.651 | 0.203 | 0.010 | 0.888 | 0.023 |
| A2 +Fiducial | 19 | **0.794** | 0.392 | 0.882 | 0.000 | 0.001 | **0.874** | 0.203 | 0.444 | 0.781 | **0.782** | 0.348 | 0.482 | 0.746 | 0.009 | 0.015 | 0.964 | 0.679 |
| A3 +Wavelet | 31 | 0.784 | **0.422** | 0.871 | 0.001 | 0.005 | 0.842 | **0.391** | 0.440 | 0.754 | 0.669 | 0.359 | 0.468 | 0.754 | 0.012 | 0.136 | 0.972 | 0.497 |
| A4 +HRV/NonLin (all 40) | 40 | 0.758 | 0.388 | 0.854 | 0.026 | 0.003 | 0.832 | 0.223 | 0.401 | 0.737 | 0.497 | 0.404 | 0.446 | 0.767 | 0.073 | 0.108 | 0.980 | 0.712 |
| only Fiducial | 13 | 0.778 | 0.334 | 0.879 | 0.001 | 0.001 | 0.718 | 0.071 | 0.471 | 0.776 | 0.716 | 0.357 | 0.477 | 0.727 | 0.006 | 0.014 | 0.955 | 0.769 |
| only Wavelet | 12 | 0.724 | 0.323 | 0.844 | 0.003 | 0.000 | 0.625 | 0.142 | 0.419 | 0.764 | 0.482 | 0.303 | 0.372 | 0.735 | 0.177 | 0.013 | 0.902 | 0.072 |
| only HRV/NonLin | 9 | 0.569 | 0.261 | 0.735 | **0.325** | 0.031 | 0.204 | 0.012 | 0.374 | 0.589 | 0.338 | 0.355 | 0.346 | 0.603 | **0.621** | 0.139 | 0.254 | 0.343 |
| all − Statistical | 34 | 0.745 | 0.363 | 0.844 | 0.012 | 0.002 | 0.801 | 0.157 | 0.372 | 0.733 | 0.480 | 0.378 | 0.423 | 0.768 | 0.052 | 0.105 | 0.978 | 0.588 |
| all − Fiducial | 27 | 0.709 | 0.317 | 0.827 | 0.052 | 0.002 | 0.696 | 0.007 | 0.384 | 0.726 | 0.439 | **0.445** | 0.442 | 0.767 | 0.425 | **0.235** | 0.912 | 0.044 |
| all − Wavelet | 28 | 0.789 | 0.396 | **0.893** | 0.056 | 0.010 | 0.831 | 0.189 | **0.553** | **0.791** | 0.615 | 0.431 | **0.507** | **0.783** | 0.093 | 0.087 | 0.975 | **0.914** |
| all − HRV/NonLin (= A3) | 31 | identical to A3 | | | | | | | | | | | | | | | | |

The paper reports 0.985–0.989 accuracy and 0.86–0.93 anomaly F1 for A1–A4 (inter-patient is implied but not reproducible).

What this shows:
1. With A4 and paper params, the collapsed anomaly F1 is 0.401. The Optuna-tuned 5-class model gets 0.409, so tuning is worth almost nothing.
2. LBBB/RBBB are not detected cross-patient by any group that contains fiducial or wavelet features. Per-record normalisation makes a whole-record bundle-branch-block patient look like its own "normal". Only HRV/NonLin (rhythm-level, per-window) gets some LBBB recall (0.62), but it loses V and A.
3. Wavelet features hurt cross-patient. Removing them gives the best collapsed F1 (0.553) and the best binary F1/AUC (0.507/0.783), mostly through A recall (0.914). This is a single DS1→DS2 run and it is selected *on DS2*, so treat it as a lead for grouped-CV feature selection, not a result.
4. Adding groups past A2 increases DS1 fit, but it does not increase DS2 accuracy.

---

## 2. Binary LightGBM (N vs {L,R,V,A})

### What changed in `source/binaryLightGBM/`
- **train_binary.py**
  - Rewritten with a CLI: `--split`, `--cv {stratified,grouped}`, `--n-trials`, `--objective {f1_05,f1_best,auc}`, `--threads`, `--tag-suffix`, `--force-refit`.
  - **Grouped CV** uses `StratifiedGroupKFold(5)` with records as groups, stratified on the 5-class label so the LBBB/RBBB records (109, 207 / 118, 124, 207) land in different folds. SMOTE is applied inside each training fold only. Grouped CV is the default for ds1ds2.
  - OOF probabilities are pooled across folds. The DS1/DS2 objective is the pooled-OOF best-threshold anomaly F1. F1@0.5, AUC and the best threshold are stored as Optuna user attributes.
  - The final threshold is the argmax-F1 threshold on the pooled grouped OOF probabilities of the best params, saved in the bundle as `threshold`.
  - Wider search space: 100–2000 trees, depth 3–15, leaves 7–63, min_child_samples 10–1000 (log), colsample 0.3–1, reg 1e-8–10. The balancing strategy is also searched: smote / scale_pos_weight / none.
  - Defaults to CPU (`LGBM_DEVICE` overrides). The old code probed the GPU at import time.
- **evaluate_binary.py**
  - Rewritten with `--tags`. It reports the 5-class model collapsed to binary (argmax, and sum P(anom) ≥ 0.5), and the binary model at three thresholds: 0.5, the OOF threshold, and the **ORACLE** test-optimal threshold. The oracle row is labelled as not a result.
  - Adds recall by original class (N column = specificity) and per-record FN/FP counts.
  - Writes JSON output.
  - Bug fixed: the old script reported the test-set-optimal threshold as its "optimal threshold" metric, which leaks DS2. It is now only a labelled oracle row.
- Stale DB `models/checkpoints/lgbm_binary_random_optuna.db` renamed to `.db.stale` (it was built on the old features).

### Grouped folds on DS1 (22 records)

| fold | records | contents |
|---|---|---|
| 0 | 201, 215, 223 | N 6847, V 835, A 105 |
| 1 | 114, 118, 119, 205, 230 | N 8187, R 2165, V 575, A 109 |
| 2 | 101, 115, 208, 209 | N 8016, V 993, A 386 |
| 3 | 108, 112, 203, 207 | N 6802, L 1457, R 85, V 566, A 113 |
| 4 | 106, 109, 116, 122, 124, 220 | N 8234, L 2491, R 1530, V 714, A 97 |

The folds holding RBBB (fold 1) and LBBB (fold 3) are where F1@0.5 collapses: 0.38–0.42 and 0.40–0.52 across trials. The other folds stay at 0.80–0.89.

### Training runs

| tag | CV | trials | objective | best trial | CV objective | OOF AUC | OOF thr | notes |
|---|---|---|---|---|---|---|---|---|
| random | stratified (beats) | 25 | F1@0.5 | #0 (paper params) | 0.9919 | 0.9998 | 0.361 | tuning gains nothing |
| ds1ds2 | **grouped (records)** | 80 | pooled OOF best-F1 | #45 | 0.7081 | 0.8674 | 0.182 | paper params (trial 0): 0.6787, AUC 0.8869, thr 0.0002 |
| ds1ds2_stratifiedcv | stratified (beats) — leaky comparison | 20 | F1@0.5 | #13 | 0.9936 | 0.9999 | 0.500 | patient leak: same data scores 0.99 vs 0.71 grouped |
| ds1ds2_paper | grouped | 1 (paper params only) | pooled OOF best-F1 | #0 | 0.6787 | 0.8869 | 0.0002 | |
| ds1ds2_auc | grouped | 60 | pooled OOF AUC | #58 | 0.8961 | 0.8961 | 0.0166 | OOF best-F1 0.698 |

Best ds1ds2 (grouped) params: n_estimators 214, max_depth 13, lr 0.0121, num_leaves 59, subsample 0.944, colsample 0.379, min_child_samples 104, reg_alpha ≈0, reg_lambda 0.285, balance smote.

SMOTE won the balancing search: 74/80 trials used it, and its best of 0.708 beats scale_pos_weight (0.670) and none (0.666).

The best-F1 objective prefers short, low-learning-rate models whose probabilities are less extreme. That gives a usable threshold (0.18) but a lower OOF AUC than the paper params (0.867 vs 0.887). With the paper params, the grouped OOF F1 optimum sits at a threshold of 0.0002, because L/R beats in held-out patients get near-zero probabilities.

### Results on test sets

**Random split** (19,858 test beats):

| Model | thr | Acc | P | R | F1 | AUC | spec N | rec L | rec R | rec V | rec A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper (§8.1) | – | 0.989 | 0.931 | 0.925 | 0.928 | 0.985 | | | | | |
| 5-class collapsed (argmax) | – | 0.9962 | 0.9947 | 0.9901 | 0.9924 | 0.9997 | 0.998 | 0.997 | 0.997 | 0.991 | 0.949 |
| Binary @0.5 | 0.5 | 0.9960 | 0.9945 | 0.9893 | 0.9919 | 0.9997 | 0.998 | 0.998 | 0.998 | 0.991 | 0.933 |
| Binary @OOF thr | 0.361 | 0.9959 | 0.9935 | 0.9901 | 0.9918 | 0.9997 | 0.998 | 0.998 | 0.998 | 0.992 | 0.939 |

The binary model is the same as the 5-class collapsed model on the random split (both at ceiling).

**DS2** (48,981 beats; N 36,427 / anomaly 12,554):

| Model | thr | Acc | P | R | F1 | AUC | spec N | rec L | rec R | rec V | rec A |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Paper (§8.1) | – | 0.989 | 0.931 | 0.925 | 0.928 | 0.985 | | | | | |
| 5-class collapsed (argmax), `lgbm_ds1ds2.pkl` | – | 0.7715 | 0.6067 | 0.3081 | 0.4087 | 0.7413 | 0.931 | 0.015 | 0.070 | 0.979 | 0.240 |
| **Binary, grouped CV best-F1 (`lgbm_binary_ds1ds2.pkl`) @OOF thr** | 0.182 | 0.6465 | 0.3984 | 0.7435 | **0.5188** | 0.7152 | 0.613 | 0.683 | 0.457 | 0.999 | 0.987 |
| same @0.5 | 0.5 | 0.6786 | 0.3682 | 0.3549 | 0.3615 | 0.7152 | 0.790 | 0.175 | 0.060 | 0.981 | 0.212 |
| Binary paper params (grouped OOF thr) | 0.0002 | 0.6730 | 0.4149 | 0.6731 | 0.5134 | **0.7671** | 0.673 | 0.406 | 0.544 | 0.997 | 0.968 |
| same @0.5 | 0.5 | 0.7428 | 0.4979 | 0.4074 | 0.4482 | 0.7671 | 0.858 | 0.076 | 0.110 | 0.982 | 0.725 |
| Binary AUC-objective (grouped OOF thr) | 0.017 | 0.6684 | 0.4097 | 0.6657 | 0.5072 | 0.7627 | 0.669 | 0.419 | 0.502 | 0.997 | 0.967 |
| Binary, beat-level CV on DS1 (OOF thr = 0.5) | 0.500 | 0.7373 | 0.4829 | 0.3526 | 0.4076 | 0.7505 | 0.870 | 0.016 | 0.083 | 0.975 | 0.537 |
| *ORACLE (DS2-chosen thr, not a result): main model* | 0.218 | 0.6670 | 0.4131 | 0.7112 | 0.5226 | 0.7152 | 0.652 | 0.633 | 0.411 | 0.999 | 0.964 |
| *ORACLE: paper params* | ~0 | 0.6100 | 0.3840 | 0.8639 | 0.5317 | 0.7671 | 0.522 | 0.709 | 0.859 | 0.999 | 0.990 |

"spec N" is the fraction of N beats predicted N. The "rec X" columns are the fraction of class-X beats flagged as anomaly.

What this shows:
- **Headline: DS2 binary F1 0.519 (P 0.398, R 0.744), AUC 0.715.** The 5-class collapsed model gets F1 0.409, AUC 0.741. The paper reports F1 0.928, AUC 0.985.
- The F1 gain over the 5-class model comes from moving the operating point, which the grouped OOF threshold found without seeing DS2. The OOF threshold (0.182) is close to the DS2 oracle (0.218, F1 0.523), so **the grouped-OOF threshold transfers well**. The beat-level threshold would not, because it stays at 0.5.
- The cost is N specificity of 0.61: 14k false alarms on DS2. Nearly all of them come from a few whole records: 105 (2358 FP), 233 (2218), 121 (1856), 213 (1578), 117 (1533), 222 (1489). Most missed anomalies are in RBBB record 212 (1584 FN) and LBBB record 214 (1208 FN). The errors are record-level, not beat-level: whether a record is called "abnormal" is mostly decided per patient.
- The binary objective does **not** improve ranking quality cross-patient much. The best AUC is 0.767 (paper-params binary) against 0.741 for the 5-class model. The best-F1-tuned model has the lowest AUC (0.715). Better F1 comes from threshold selection, not from a better model.
- The paper-params and AUC-objective variants pick extreme OOF thresholds (0.0002 and 0.017). These sit on the near-zero probability tail where held-out L/R patients lie, so the operating point is fragile, even though DS2 F1 comes out similar.
- The beat-level CV variant shows the leak directly. Beat-level OOF says F1 0.994 at threshold 0.5. On DS2 that gives F1 0.408, with LBBB records 111 and 214 and RBBB records 212, 231 and 232 missed entirely.

---

## 3. Comparison summary (DS2, binary anomaly)

| Source | P | R | F1 | AUC | L rec | R rec | V rec | A rec |
|---|---|---|---|---|---|---|---|---|
| Paper (claimed) | 0.931 | 0.925 | 0.928 | 0.985 | – | – | – | – |
| 5-class LGBM collapsed (tuned, step 5) | 0.607 | 0.308 | 0.409 | 0.741 | 0.015 | 0.070 | 0.979 | 0.240 |
| Binary LGBM, grouped-CV tuned + grouped-OOF threshold (**main result**) | 0.398 | 0.744 | **0.519** | 0.715 | 0.683 | 0.457 | 0.999 | 0.987 |
| Binary LGBM, paper params + grouped-OOF threshold | 0.415 | 0.673 | 0.513 | 0.767 | 0.406 | 0.544 | 0.997 | 0.968 |
| Binary LGBM, ablation "all − wavelet" @0.5 (selected on DS2, lead only) | 0.615 | 0.431 | 0.507 | 0.783 | 0.093 | 0.087 | 0.975 | 0.914 |

All honest inter-patient numbers stay below F1 0.53 and AUC 0.79. The paper's figures are only reached on the random (beat-level) split.

---

## Reproduce (from $ROOT on the box)

```bash
export OMP_NUM_THREADS=6 LGBM_DEVICE=cpu
# Task 1
.venv/bin/python -u source/ablation/ablation.py --mode random                 # ~6 min
.venv/bin/python -u source/ablation/ablation.py --mode ds1ds2 --extras        # ~20 min, cached in results/ablation_ds1ds2.json
# Task 2 (driver scripts, resumable via Optuna SQLite; refit skipped if pkl exists)
bash source/binaryLightGBM/run_pipeline.sh     # ds1ds2 grouped 80 trials -> random 25 trials -> evaluate (~45 min)
bash source/binaryLightGBM/run_extra.sh        # ds1ds2 stratified-CV 20, paper-only, AUC-objective 60 -> evaluate
# or individually:
.venv/bin/python source/binaryLightGBM/train_binary.py --split ds1ds2 --n-trials 80          # grouped CV
.venv/bin/python source/binaryLightGBM/train_binary.py --split random --n-trials 25
.venv/bin/python source/binaryLightGBM/evaluate_binary.py --tags random ds1ds2
```

## Files created or modified (box)

- Modified: `source/binaryLightGBM/train_binary.py`, `source/binaryLightGBM/evaluate_binary.py`
- Created:
  - Code: `source/binaryLightGBM/run_pipeline.sh`, `source/binaryLightGBM/run_extra.sh`, `source/ablation/ablation.py`
  - Logs and ablation results: `source/binaryLightGBM/logs/*.log`, `source/ablation/results/{ablation_random.log, ablation_ds1ds2.log, ablation_ds1ds2.json}`
  - Models: `models/lgbm_binary_{random,ds1ds2,ds1ds2_stratifiedcv,ds1ds2_paper,ds1ds2_auc}.pkl` with matching `_optuna.json`
  - Checkpoints: `models/checkpoints/lgbm_binary_*_optuna.db`, `models/checkpoints/lgbm_binary_*_oof.npz`
  - Evaluation output: `results/binary/binary_eval_*.json`, `results/binary/cm_*.png`, `results/binary/roc_*.png`
  - Reports: `reports/lgbm-binary.md`, `reports/lgbm-binary.RESUME.md`
- Renamed: `models/checkpoints/lgbm_binary_random_optuna.db` → `.db.stale`
- Not touched: `source/evaluate.py`, `source/train.py`, `models/lgbm_random.pkl`, `models/lgbm_ds1ds2.pkl`, their DBs, progress.md, notes.txt.

## Open issues / next steps

1. **LBBB/RBBB cross-patient failure is the whole gap.** V is already 0.98–1.0 recall on DS2. The cause is most likely the per-record normalisation: a patient whose every beat is a bundle-branch block gets normalised to their own baseline. Options:
   - add un-normalised morphology features (QRS width in ms is not amplitude-normalised but is apparently not enough on its own)
   - use global, not per-record, normalisation for a subset of features
   - add record-level context features
2. Feature-group selection under **grouped CV** on DS1. The "all − wavelet" lead from 1b must be confirmed without DS2. Per lgbm-groupcv, grouped CV correlates only weakly with DS2 (rank ρ ≈ 0.49), and the morphology group mostly encodes patient identity.
3. Only 22 DS1 records with 3 bundle-branch-block patients: grouped CV estimates are high-variance (fold F1 ranges 0.38–0.89).
4. The paper's binary F1 0.928 / AUC 0.985 is not reachable with these features under an honest inter-patient protocol. The paper's numbers are consistent with a beat-level (random) split, where we match or beat them.
5. Thread budget was exceeded once: the two ablations ran concurrently (6+6 threads) for about 15 minutes on 11-18.
