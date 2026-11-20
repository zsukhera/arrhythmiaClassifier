# lgbm-groupcv RESUME notes  (ROOT=/home/i230510/Desktop/arrhythmiaClassifier)

## Done
- `source/train.py` extended (additive CLI; no-arg run unchanged). Backup of original: `reports/lgbm-groupcv/train.py.orig_backup`.
  New flags: --splits --cv {stratified,group} --sampler {smote,none} --class-weight balanced --space {paper,extended}
  --drop-features NAMES --n-trials --n-folds --n-jobs --tag (required for non-default) --no-enqueue-paper.
  Group CV = StratifiedGroupKFold(5) by record, objective = pooled OOF macro-F1.
- `source/groupcv/common.py` helpers (folds, grouped_cv, DS2 metrics identical to evaluate.py).
- Device bench: CPU 8 threads 15.5 s/fit vs GPU (OpenCL) 23.7 s/fit -> use CPU (LGBM_DEVICE=cpu).
- Grouped folds (deterministic): f0 201 215 223 | f1 114 118 119 205 230 | f2 101 115 208 209 | f3 108 112 203 207 | f4 106 109 116 122 124 220

## Running (nohup)
- task1 first run KILLED after 2 of 4 results to save CPU (log kept: reports/lgbm-groupcv/task1_firstrun_partial.log).
  Results: old params beat-level 5-fold pooled mF1 0.9862; old params GROUPED 5-fold pooled mF1 0.4055 (acc 0.761, binF1 0.613;
  per-class N .888 L .009 R .001 V .517 A .612) vs DS2 0.388 -> grouped CV explains ~all of the gap.
  task1_baseline.py (LOGO removed) re-runs at the END of chain2 -> task1_rerun.log / task1_baseline.json (per-record OOF).
- run_chain1.sh PID 1204342 (waits for task1, then task4_lr_shift.py -> task4.log/task4_lr_shift.json,
  then Optuna: `LGBM_DEVICE=cpu .venv/bin/python source/train.py --splits ds1ds2 --cv group --tag _groupcv --n-trials 40 --n-jobs 8`  (40 not 100: box is loaded; extend later with same cmd + larger --n-trials)
  -> reports/lgbm-groupcv/task2_optuna.log, models/lgbm_ds1ds2_groupcv.pkl, models/checkpoints/lgbm_ds1ds2_groupcv_optuna.db).
  Progress: reports/lgbm-groupcv/chain1.status. Check `pgrep -af run_chain1`. If killed, re-run the same command (Optuna resumes).

- run_chain2.sh PID 1204343: waits for chain1, then eval_models.py (old + groupcv -> eval_groupcv.log, eval_ds2.json),
  task3_variants.py <groupcv search json> (-> task3.log, task3_variants.json, resumable),
  extended-space Optuna `--cv group --space extended --sampler none --class-weight balanced --tag _groupcv_reg --n-trials 40 --n-jobs 8`
  (-> task3_optuna_reg.log, models/lgbm_ds1ds2_groupcv_reg.pkl), eval -> eval_groupcv_reg.log. Status: chain2.status ("all done").
  If chain dies, re-run the remaining steps from run_chain2.sh by hand (each is resumable).

## Remaining (in order)
1-2. (automated in chains) Task 3 variants chosen by grouped CV (TODO script task3_variants.py): class_weight vs SMOTE; feature-identity
   scores (eta^2 by record on N beats) + grouped permutation importance + group-drop; extended regularisation Optuna
   (`--space extended --sampler none --class-weight balanced --tag _groupcv_reg`, ~60 trials).
3. Final table old vs groupcv vs paper (acc 0.989, binF1 0.928, AUC 0.985); write reports/lgbm-groupcv.md.

## Notes / decisions
- Box heavily loaded by other agents (load avg ~57); timings slower than bench.
- HRV/nonlinear features use a rolling 50-RR window in ms (not normalised) -> likely patient fingerprints.

4. After chains: maybe refit best task3 variant as models/lgbm_ds1ds2_groupcv_best.pkl; write report.

- BUG fixed 13:40: common.scores OVR AUC was nan (sklearn needs sorted labels); first task1 log AUC=nan is that bug.

## Milestone 13:42 — task4 done (reports/lgbm-groupcv/task4.log, task4_lr_shift.json)
Key: L DS1={109,207}, DS2={111,214}; R DS1={118,124,(207:85)}, DS2={212,231,232}. Domain AUC DS1-vs-DS2 same class ~1.0 for all classes.
Old model sends 98-99% of DS2 L and R beats to N (231 R -> 221 A). Per-record medians: QRS dur DS2-L 111 = 89 ms (delineation fails),
DS1-R 124 looks like N (QRS 81 ms, R/S 6.5). Single-group L-vs-N transfer: A_stat AUC .80 best; all40 .68. R: B1 intervals .71, all40 .58.
CAVEAT: "carry_*" subsets in task4 were picked using DS2 AUCs -> analysis only, not for model selection.
task2 Optuna started 13:41 (40 trials).

## Day 2 (2026-11-19 10:45) — chains 1+2 finished OK (see *.status). Results in eval_groupcv*.log, task3.log, task1_rerun.log.
- groupcv (paper space, SMOTE): gCV 0.4064 -> DS2 mF1 0.419 acc 0.765 binF1 0.422
- groupcv_reg (extended, balanced, no SMOTE): gCV 0.4200 -> DS2 mF1 0.423 acc 0.774 binF1 0.411
- task3 best by gCV: e_mcs1000 (drop B2 morph, balanced, mcs=1000) gCV 0.4515 -> DS2 0.431
- RUNNING: run_chain3.sh PID 3793488 -> extended Optuna without B2 (`--tag _groupcv_reg_noB2`, 40 trials,
  log task3_optuna_reg_noB2.log), eval, then task5_final.py (LOGO check, Spearman gCV vs DS2,
  saves models/lgbm_ds1ds2_groupcv_best.pkl = best by gCV). Status: chain3.status ("task5 done").
- Local /tmp copies were lost on laptop reboot; working copies now in session scratchpad; box copies are canonical.
- Remaining: write reports/lgbm-groupcv.md.

## Day 2 15:30 — ALL DONE. chain3 finished (task5 done 11:31). Final report: reports/lgbm-groupcv.md
