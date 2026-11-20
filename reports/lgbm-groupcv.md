# lgbm-groupcv: patient-grouped CV for the 5-class LightGBM (DS1/DS2)

Agent: lgbm-groupcv · 2026-11-18 → 19 · GPU box `~/Desktop/arrhythmiaClassifier` (= ROOT)

## TL;DR

1. **Grouped CV explains essentially the whole CV→DS2 gap.** The existing model's params score
   **0.986** macro-F1 under beat-level 5-fold CV but **0.406** under patient-grouped 5-fold CV on DS1.
   DS2 gives 0.388. The beat-level CV was measuring "recognise beats of patients you have already seen".
2. **Re-tuning against grouped CV gives small, real but modest DS2 gains:** macro-F1 0.388 → 0.419
   (paper search space, SMOTE) and 0.423 (wider space with regularisation, class weights). Most of the
   gain is in V and A. **L and R stay at F1 ≈ 0 on DS2 with every variant.**
3. **Grouped CV on 22 DS1 patients is a noisy selector.** Across 21 variants its rank correlation with
   DS2 macro-F1 is weak (Spearman ρ = 0.49 for macro-F1, 0.14 for binary F1; §5). The highest-gCV model (0.482)
   does *worse* on DS2 (0.416) than simpler variants: 40 Optuna trials are enough to overfit 5 folds of 22 patients.
4. **L/R failure is structural, not a tuning problem.** DS1 has 2 LBBB patients (109, 207) and 2 RBBB
   patients (118, 124), and DS2 has 2 and 3 different ones. A classifier separates DS1 L beats from
   DS2 L beats with AUC 0.99999 (R: 0.99999). Several key features change sign between the DS1
   and DS2 bundle-branch patients. Even inside DS1, a held-out L/R patient is classified ~0 %
   correctly (109 → mostly V, 118/124 → N). No feature subset chosen without DS2 carries over.

## 1. What was done

| Step | What | Output |
|---|---|---|
| train.py | Additive CLI: `--cv group` (StratifiedGroupKFold by record, objective = pooled out-of-fold macro-F1), `--sampler none`, `--class-weight balanced`, `--space extended` (+min_child_samples, reg_lambda, reg_alpha, wider colsample/leaves/depth), `--drop-features`, `--n-trials/--n-folds/--n-jobs`, `--tag` (required for any non-default option), `--splits`. No-argument run is unchanged. | `source/train.py` |
| helpers | Folds, grouped CV with SMOTE inside each training fold, DS2 metrics identical to `evaluate.run_main_eval` | `source/groupcv/common.py` |
| Task 1 | Old params under beat-level CV, grouped CV, and paper params under grouped CV, with per-record OOF breakdown | `task1_baseline.py`, `reports/lgbm-groupcv/task1_*` |
| Task 2 | Optuna 40 trials, paper search space, SMOTE, grouped-CV objective | `models/lgbm_ds1ds2_groupcv.pkl` |
| Task 3 | Imbalance handling, feature identity / permutation importance / group drops, regularisation (fixed configs + 2 Optuna studies of 40 trials each) | `task3_variants.py`, `models/lgbm_ds1ds2_groupcv_reg.pkl`, `models/lgbm_ds1ds2_groupcv_reg_noB2.pkl` |
| Task 4 | L/R shift analysis | `task4_lr_shift.py`, `reports/lgbm-groupcv/task4_*` |
| Wrap-up | LOGO (22-fold) robustness check, Spearman correlation between gCV and DS2, final best-by-gCV model | `task5_final.py`, `models/lgbm_ds1ds2_groupcv_best.pkl` |

**Grouped folds** (StratifiedGroupKFold(5), deterministic for this data):
f0 {201, 215, 223} · f1 {114, **118**R, 119, 205, 230} · f2 {101, 115, 208, 209} ·
f3 {108, 112, 203, **207**L+R} · f4 {106, **109**L, 116, 122, **124**R, 220}.
Every training fold still has at least one L and one R patient, and every L/R patient is validated
by a model that never saw them. The objective is macro-F1 on the pooled OOF predictions, because
per-fold macro-F1 is unstable when a fold has no L/R beats (mean-of-folds is also logged).

**Device:** LightGBM GPU (OpenCL) took 23.7 s per fit and CPU with 8 threads took 15.5 s
(156k-row SMOTE fold, old params), so everything ran on CPU with 8 threads (`LGBM_DEVICE=cpu`).

**Budget deviations:** Optuna ran 40 trials per study, not 100. The box was heavily shared
(load average ~30 on 28 cores) and a grouped CV with the old params took ~8.5 min. LOGO was
dropped from Task 1 and done later, only for the shortlisted models (§5).

## 2. Task 1: how much of the gap does grouped CV explain?

Same params (old `lgbm_ds1ds2.pkl`, beat-CV-tuned trial #94), SMOTE inside folds, pooled OOF on DS1:

| Evaluation | Acc | Macro-F1 | OVR AUC | Bin F1 | Bin AUC | N | L | R | V | A |
|---|---|---|---|---|---|---|---|---|---|---|
| Beat-level StratifiedKFold(5) | 0.9965 | **0.9862** | 0.9998 | 0.9936 | 0.9999 | .998 | .999 | .997 | .988 | .948 |
| **Grouped 5-fold (by record)** | 0.7608 | **0.4055** | 0.8708 | 0.6133 | 0.8874 | .888 | .009 | .001 | .517 | .612 |
| Paper params, grouped 5-fold | 0.7577 | 0.3972 | 0.8635 | 0.6181 | 0.8797 | .888 | .004 | .001 | .499 | .594 |
| *DS2 test (old model)* | 0.7635 | *0.3878* | 0.7873 | 0.4087 | 0.7413 | .858 | .014 | .002 | .827 | .237 |

Per-fold macro-F1 (grouped): 0.449, 0.521, 0.667, 0.186, 0.434. That is a large spread, so grouped
CV is noisy.

**Gap explained:** beat CV − DS2 = 0.598. Grouped CV − DS2 = 0.018, so **≈97 % of the gap is
patient leakage in the beat-level CV.** The per-class pattern also matches DS2: L/R ≈ 0, N high,
and V/A middling (V and A swap places, because DS1 and DS2 have different V/A patients).

Per-record grouped OOF accuracy (old params) shows which patients fail:
- 109 (L): 1.6 % (L beats → V 1588, N 936)
- 207 (L): 6.8 % (→ V 1560)
- 118 (R): 0.9 % (→ N 2164)
- 124 (R): 2.7 % (→ N 1529)
- 108: 17.7 % (noisy record: N → R/V)
- 203: 38 % (N → V)
- everyone else: >88 %

## 3. Task 2: Optuna on the grouped-CV objective

`models/lgbm_ds1ds2_groupcv.pkl`: 40 trials, paper search space, SMOTE. Trial 0 (paper params)
scored 0.3972, the median trial 0.4028 and the best (trial #26) **0.4064**. The best params were
n_estimators=541, max_depth=15, lr=0.144, num_leaves=49, subsample=0.715, colsample=0.810.
Within the paper's search space the grouped objective is almost flat (0.392–0.406), because
the hyperparameters in that space don't control the inter-patient error.

## 4. Task 3: variants, selected by grouped CV

Base = Task-2 best params. "gCV" = grouped 5-fold pooled OOF macro-F1 (the selection metric).
The DS2 columns are reported only and were never used for any choice.

| Variant | gCV mF1 | DS2 acc | DS2 mF1 | DS2 binF1 |
|---|---|---|---|---|
| a_smote (= Task 2 model) | 0.4064 | 0.7653 | 0.4194 | 0.4223 |
| **a_balanced** (class_weight, no SMOTE), chosen | **0.4151** | 0.7731 | 0.4086 | 0.4016 |
| a_none (no rebalancing) | 0.3962 | 0.7828 | 0.4429 | 0.4210 |
| drop A_stat (6) | 0.4067 | 0.7624 | 0.4081 | 0.3883 |
| drop B1_intervals (5) | 0.3964 | 0.6971 | 0.3151 | 0.3110 |
| **drop B2_morph (8)**, chosen | **0.4430** | 0.7741 | 0.4465 | 0.4357 |
| drop C_dwt (12) | 0.3830 | 0.7890 | 0.3978 | 0.5408 |
| drop D1_hrv (5) | 0.4185 | 0.7807 | 0.4486 | 0.4345 |
| drop D2_nonlin (4) | 0.4135 | 0.7843 | 0.3974 | 0.4078 |
| drop high-identity & low-importance (11) | 0.4241 | 0.7780 | 0.4212 | 0.4316 |
| drop permutation-importance ≤ 0 (6) | 0.4273 | 0.7671 | 0.3889 | 0.3859 |
| drop HRV+NL+identity | 0.4027 | 0.7870 | 0.4164 | 0.4356 |
| noB2 + min_child_samples 200 | 0.4213 | 0.7705 | 0.4540 | 0.4310 |
| **noB2 + min_child_samples 1000**, best fixed config | **0.4515** | 0.7596 | 0.4313 | 0.4254 |
| noB2 + λ=10, ff=0.5 | 0.4409 | 0.7684 | 0.4235 | 0.4216 |
| noB2 + shallow (d4, 15 leaves, 400 trees, λ10) | 0.4408 | 0.7640 | **0.4630** | 0.4620 |
| noB2 + shallow/slow (d3, 8 leaves, λ30) | 0.4421 | 0.7492 | 0.4591 | **0.4869** |
| Optuna extended space, balanced, all 40 feats (`_groupcv_reg`) | 0.4200 | **0.7744** | 0.4227 | 0.4113 |
| Optuna extended space, balanced, no B2 (`_groupcv_reg_noB2`) | **0.4819** | 0.7278 | 0.4156 | 0.4030 |

(The "noB2 + ..." rows also use balanced class weights.)

**Feature identity (eta² of record ID on DS1 N beats**, i.e. the share of a feature's variance
among normal beats that is explained by which patient they came from): stat_kurtosis 0.88,
fid_s_amp 0.85, stat_skewness 0.78, dwt_cD1_rel_energy 0.74, dwt_cA4_entropy 0.73,
fid_qrs_dur 0.72, then the DWT entropies, fid_st_elev and QT at ~0.64–0.70. **Most morphology
features are more patient-specific than class-specific.** Grouped permutation importance is
dominated by fid_rr_pre_norm (ΔF1 0.136), fid_rr_post_norm (0.036) and nl_sampen (0.034). Every
other feature is < 0.015, and fid_st_elev, dwt_cD3_entropy and nl_sd2 are negative.

**Regularisation:** larger min_child_samples and shallower trees raise gCV from 0.406 to ~0.44–0.45
and DS2 macro-F1 to 0.43–0.46. That direction is consistent. Within it, the gCV ranking does not
predict the DS2 ranking.

## 5. Is grouped CV a good selector? (LOGO check + rank correlation)

- **Spearman(gCV, DS2 macro-F1)** over all 21 variants: **ρ = 0.49** (p = 0.025).
  **Spearman(gCV, DS2 binary F1):** ρ = 0.14 (p = 0.55).
  Grouped CV ranks variants better than chance for macro-F1 but is far from reliable, and it does
  not rank them for binary F1. (Beat-level CV, by comparison, put everything at ~0.986 and could
  not rank anything.)
- **Leave-one-record-out (22 folds) for the 8 shortlisted models**, pooled OOF on DS1:

| Model | gCV5 | LOGO22 mF1 | LOGO binF1 | LOGO L F1 | DS2 mF1 |
|---|---|---|---|---|---|
| old_ds1ds2 (beat-CV tuned) | 0.4055 | 0.4095 | 0.651 | 0.004 | 0.3878 |
| groupcv (Task 2) | 0.4064 | 0.4028 | 0.637 | 0.002 | 0.4194 |
| groupcv_reg | 0.4200 | 0.4130 | 0.646 | 0.004 | 0.4227 |
| groupcv_reg_noB2 | **0.4819** | **0.4833** | 0.665 | **0.410** | 0.4156 |
| drop B2 (balanced) | 0.4430 | 0.4362 | 0.693 | 0.071 | 0.4465 |
| noB2 + mcs1000 | 0.4515 | 0.4153 | 0.701 | 0.051 | 0.4313 |
| noB2 + shallow | 0.4408 | 0.4266 | 0.704 | 0.105 | **0.4630** |
| noB2 + shallow/slow | 0.4421 | 0.4410 | 0.667 | 0.143 | 0.4591 |

  Spearman(gCV5, LOGO22) = 0.81: the two DS1 CV schemes agree with each other.
  Spearman(LOGO22, DS2) = 0.38: neither predicts DS2 well.
- **What the top gCV model actually learned:** `groupcv_reg_noB2` gets 0.41 L-F1 under DS1 LOGO,
  because held-out LBBB patient 109 is now predicted L for 1065 of 2491 beats (from ~1 before).
  On DS2 its L-F1 is still 0.023. The gCV gain comes from transferring 207 → 109 (two DS1 LBBB
  patients that resemble each other in these features). That does not extend to 111/214. This is
  the clearest illustration that **with 2 patients per class, no DS1-internal CV can estimate
  L/R generalisation**.
- **Final pick by the pre-declared criterion (highest gCV)** = `groupcv_reg_noB2`, refit and saved as
  `models/lgbm_ds1ds2_groupcv_best.pkl`: DS2 acc 0.728, macro-F1 0.416, binary F1 0.403. That is
  *worse* on DS2 accuracy than the Task-2 model. We report it as selected, not cherry-picked. The best
  DS2 numbers among the variants (noB2 + shallow: macro-F1 0.463, noB2 + shallow/slow: binary F1 0.487)
  are visible only through DS2 and **must not be claimed as the model's result**.


## 6. Task 4: why LBBB/RBBB fail

### 6.1 Who has L/R beats

| Class | DS1 records (beats) | DS2 records (beats) |
|---|---|---|
| L | 109 (2491), 207 (1457) | 111 (2123), 214 (2001) |
| R | 118 (2165), 124 (1530), 207 (85) | 212 (1825), 231 (1253), 232 (397) |

So 2 training patients per class, and the test set has entirely different ones. For L, 100 % of
the DS2 L beats come from 2 unseen patients.

### 6.2 Shift size

- **Domain classifier** (LightGBM, 3-fold, DS1-class-c vs DS2-class-c beats), AUC:
  N 0.9994, **L 0.99999**, **R 0.99999**, V 0.9976, A 0.9995. Every class is perfectly
  patient-separable. For L/R that is fatal, because the class is only 2 patients.
- **Per-feature shift for L** (SMD = standardised mean difference DS2 − DS1): dwt_cD4_entropy −1.71,
  dwt_cD3_energy +1.65, dwt_cD2_entropy −1.39, fid_t_amp +1.10, dwt_cD1_rel_energy +1.08.
  KS statistics are 0.5–0.85.
- **Per-feature shift for R:** fid_t_amp +3.61, dwt_cA4_entropy +3.33, dwt_cD3_entropy −3.16,
  dwt_cD3_energy +3.00, fid_q_amp −2.55, dwt_cD1_rel_energy +2.32, fid_p_amp −2.23.
  KS is 0.8–0.95. For these, the DS2 R median is often **closer to DS1 N than to DS1 R**.
- **Direction flips (R):** dwt_cD3_entropy, fid_q_amp, dwt_cD1_entropy, dwt_cA4_entropy, fid_t_amp,
  dwt_cD3_energy and dwt_cD1_rel_energy separate R from N in **opposite directions** in DS1 and DS2.
  For example, fid_t_amp R-vs-N AUC is 0.27 in DS1 and 0.87 in DS2. A model that learns DS1 R
  gets DS2 R backwards.
- **Per-record medians** show the heterogeneity:
  - DS1 RBBB 124 looks like a normal beat in MLII (QRS 81 ms, R/S 6.5, kurtosis 10.6).
  - DS1 RBBB 118 has R/S 1.1 and skewness −0.45.
  - DS2 RBBB 212, 231 and 232 (R/S 1.7–2.3, kurtosis 4.3–5.4, positive T) resemble DS1 **LBBB 109**
    more than either DS1 RBBB record.
  - The delineated QRS duration of DS2 LBBB 111 is 89 ms. LBBB is by definition ≥120 ms, so the
    rule-based delineation fails on that patient. 214 is 161 ms.
  - Record 232 has SDNN 586 ms (long pauses), far outside the DS1 range.

### 6.3 Does any feature subset carry over? (binary c-vs-N LightGBM, balanced, trained on DS1)

| Subset | L: DS2 AUC | L: DS1 leave-one-L-patient-out AUC (109 / 207) | R: DS2 AUC | R: DS1 LOPO AUC (118 / 124) |
|---|---|---|---|---|
| all 40 | 0.678 | 0.999 / 0.995 | 0.584 | 0.782 / 0.939 |
| A_stat (6) | **0.799** | 0.980 / 0.978 | 0.613 | 0.058 / 0.294 |
| B1_intervals (5) | 0.622 | 0.829 / 0.531 | **0.714** | 0.867 / 0.934 |
| B2_morph (8) | 0.477 | 0.275 / 0.292 | 0.281 | 0.438 / 0.368 |
| C_dwt (12) | 0.722 | 0.992 / 1.000 | 0.381 | 0.625 / 0.752 |
| D1_hrv (5) | 0.771 | 0.908 / 0.547 | 0.436 | 0.820 / 0.640 |
| D2_nonlin (4) | 0.743 | 0.845 / 0.915 | 0.554 | 0.780 / 0.751 |

- Some ranking signal survives for L: statistical features give AUC 0.80 on DS2. At the DS1-derived
  threshold, recall is only 0.24 (and 0.01 for all 40 features), so the 5-class model never fires L.
- For R nothing exceeds 0.71. The B2 morphology group is *anti*-predictive for both classes
  (AUC < 0.5, also within DS1). That agrees with dropping B2 being the best group-drop under grouped CV.
- The leave-one-L-patient-out AUC inside DS1 is high (0.98–1.0) while DS2 is 0.68–0.80, so even
  "unseen patient" checks with one held-out patient overstate transfer.
- A "consistent direction" subset (`carry_*` in the JSON) was built **using DS2 AUCs**. It is
  analysis only and was not used for any model. It still reaches only L 0.756 and R 0.713.

### 6.4 Where the DS2 L/R beats go (old model)

| Record | Class | Predicted N | Other predictions |
|---|---|---|---|
| 111 | L | 2082/2123 | L 32 |
| 214 | L | 1982/2001 | — |
| 212 | R | 1811/1825 | — |
| 231 | R | 1027/1253 | A 221 |
| 232 | R | 395/397 | — |

**Conclusion:** with 2 training patients per bundle-branch class and hand-crafted MLII features
whose values are dominated by patient identity (eta² up to 0.88), L/R cannot be learned in a
patient-independent way from DS1. This is a property of the DS1/DS2 split plus the feature set,
not of LightGBM tuning. Options that might help, none tried here:
- a global (not per-record) amplitude reference, or V1 lead features (RBBB is defined in V1)
- repaired QRS onset/offset delineation
- merging L/R into a "bundle-branch / wide-QRS" class
- reporting the AAMI N/S/V/F grouping, where L and R belong to N

## 7. Final table: DS2 test (48,981 beats)

| Model | Selection | Acc | Macro-F1 | OVR AUC | Bin P | Bin R | Bin F1 | Bin AUC | F1 N / L / R / V / A |
|---|---|---|---|---|---|---|---|---|---|
| Paper (Farag/CLARITY, random split) | — | 0.989 | ~0.937 | 0.985 | 0.931 | 0.925 | 0.928 | 0.985 | — |
| Old `lgbm_ds1ds2.pkl` | beat-level CV | 0.7635 | 0.3878 | 0.7873 | 0.607 | 0.308 | 0.4087 | 0.7413 | .858 / .014 / .002 / .827 / .237 |
| `lgbm_ds1ds2_groupcv.pkl` (Task 2) | gCV, paper space, SMOTE | 0.7653 | 0.4194 | 0.7890 | 0.606 | 0.324 | 0.4223 | 0.7461 | .859 / .013 / .003 / .845 / .378 |
| `lgbm_ds1ds2_groupcv_reg.pkl` | gCV, extended space, balanced | **0.7744** | **0.4227** | **0.7925** | 0.647 | 0.302 | 0.4113 | **0.7565** | .864 / .009 / .004 / .912 / .325 |
| `lgbm_ds1ds2_groupcv_best.pkl` (= reg_noB2) | highest gCV overall | 0.7278 | 0.4156 | 0.7859 | 0.478 | 0.348 | 0.4030 | 0.7131 | .830 / .023 / .001 / .800 / .424 |

Per-class P/R for every model is in `reports/lgbm-groupcv/eval_ds2.json` and `eval_groupcv*.log`.

**Bottom line:** grouped CV makes the DS1 validation number honest (0.41 instead of 0.99, matching
DS2's 0.39–0.42). Tuning against it gives +0.03 DS2 macro-F1 (mostly A and V) and +0.01 accuracy.
It cannot close the gap to the paper (−0.21 accuracy, −0.5 binary F1), because L and R
(7.6k of 49k test beats) are not learnable across patients from 2 training patients each with these
features. The paper's 0.989 / 0.928 / 0.985 is a random-split (patient-leaking) figure. Our own random
split reproduces it (acc 0.996, binary F1 0.992, OVR AUC 0.999; progress.md Step 5), so that is the like-for-like comparison.


## 8. Reproduce

```bash
cd ~/Desktop/arrhythmiaClassifier
PY=.venv/bin/python
# Task 1 / Task 4 (CWD must be source/groupcv)
(cd source/groupcv && GROUPCV_NJOBS=8 ../../$PY task1_baseline.py && ../../$PY task4_lr_shift.py)
# Task 2
LGBM_DEVICE=cpu $PY source/train.py --splits ds1ds2 --cv group --tag _groupcv --n-trials 40 --n-jobs 8
# Task 3
(cd source/groupcv && ../../$PY task3_variants.py ../../models/checkpoints/lgbm_ds1ds2_groupcv_search_complete.json)
LGBM_DEVICE=cpu $PY source/train.py --splits ds1ds2 --cv group --space extended --sampler none \
    --class-weight balanced --tag _groupcv_reg --n-trials 40 --n-jobs 8
LGBM_DEVICE=cpu $PY source/train.py --splits ds1ds2 --cv group --space extended --sampler none \
    --class-weight balanced --drop-features fid_r_amp fid_q_amp fid_s_amp fid_t_amp fid_p_amp \
    fid_rs_ratio fid_pr_ratio fid_st_elev --tag _groupcv_reg_noB2 --n-trials 40 --n-jobs 8
# DS2 evaluation + wrap-up
(cd source/groupcv && ../../$PY eval_models.py old_ds1ds2=../../models/lgbm_ds1ds2.pkl \
    groupcv=../../models/lgbm_ds1ds2_groupcv.pkl groupcv_reg=../../models/lgbm_ds1ds2_groupcv_reg.pkl \
    groupcv_reg_noB2=../../models/lgbm_ds1ds2_groupcv_reg_noB2.pkl && ../../$PY task5_final.py)
```

The chain scripts `source/groupcv/run_chain{1,2,3}.sh` ran exactly these steps.

## 9. Files

**Modified:** `source/train.py`. The changes are additive: `RUN_OPTS`, `_cv_score`, `_make_pipeline`,
`_suggest_extended` and the argparse CLI. The original is backed up at
`reports/lgbm-groupcv/train.py.orig_backup`. A no-argument run still uses beat-level CV, SMOTE,
the paper space, both splits and the same output names, and still skips the existing models.
Non-default options refuse to run without `--tag`.

**Created:**
- `source/groupcv/`: `__init__.py`, `common.py`, `task1_baseline.py`, `task3_variants.py`,
  `task4_lr_shift.py`, `task5_final.py`, `eval_models.py`, `run_chain1.sh`, `run_chain2.sh`, `run_chain3.sh`
- `models/`: `lgbm_ds1ds2_groupcv{,_reg,_reg_noB2,_best}.pkl` and the `*_optuna.json` files
- `models/checkpoints/`: `lgbm_ds1ds2_groupcv{,_reg,_reg_noB2}_optuna.db` and `*_search_complete.json`
- `reports/lgbm-groupcv/`: logs and JSON for every step, `task1_oof.npz`
- `reports/lgbm-groupcv.RESUME.md`, `reports/lgbm-groupcv.md`

**Untouched (md5 / mtime checked):** `models/lgbm_random.pkl`, `models/lgbm_ds1ds2.pkl` and their Optuna DBs.

## 10. Open issues / caveats

- The 22-patient grouped CV has high variance (per-fold macro-F1 0.19–0.67). Differences below
  ~0.03 gCV are not meaningful, and Optuna overfits it (the 0.482 study). A repeated or nested scheme
  would be better: e.g. several group partitions, or LOGO for the final selection.
- Optuna ran 40 trials per study (not 100) because of the shared-box load.
- The first Task-1 run printed OVR AUC = nan: sklearn needs sorted `labels`. This was fixed in
  `common.scores` before any reported number; all AUCs above come from the fixed code.
- `task4` `carry_*` subsets use DS2 information. They are analysis only and are excluded from model selection.
- No threshold tuning was done (it would need a DS1-only calibration scheme). Binary F1 on DS2 stays
  ~0.40–0.49, far from the paper's 0.928. The paper's number is almost certainly a beat-level
  (random-split) result; our random split gives 0.992.
