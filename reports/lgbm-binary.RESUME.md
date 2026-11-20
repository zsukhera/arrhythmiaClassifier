# lgbm-binary — RESUME notes (updated 2026-11-19 ~10:55)

ROOT=/home/i230510/Desktop/arrhythmiaClassifier. Thread budget 6. Never use DS2 for training/tuning/threshold.

## Done
- Stale `models/checkpoints/lgbm_binary_random_optuna.db` renamed `.db.stale`.
- Ablation (task 1) DONE: `source/ablation/results/ablation_random.log`, `ablation_ds1ds2.log`, `ablation_ds1ds2.json`.
- Main binary pipeline (task 2) DONE (`logs/eval.log` ends PIPELINE_DONE):
  - `models/lgbm_binary_ds1ds2.pkl` (grouped CV, 80 trials, objective pooled-OOF best-F1, best #45, OOF thr 0.1816)
  - `models/lgbm_binary_random.pkl` (stratified CV, 25 trials, best = trial 0 paper params, OOF thr 0.361)
  - eval JSON `results/binary/binary_eval_{random,ds1ds2}.json`
  - DS2 @OOF thr: P 0.398 R 0.744 F1 0.519 AUC 0.715; recall L .683 R .457 V .999 A .987, N spec .613
    @0.5: F1 0.362. 5-class collapsed: F1 0.409 AUC 0.741.

## Status: ALL DONE (2026-11-19). Final report: reports/lgbm-binary.md
- `source/binaryLightGBM/run_extra.sh <pid>` (driver PID 4048659) — waits for stratified-CV study PID 3714693, then
  ds1ds2_paper (1 trial), ds1ds2_auc (60 trials, objective auc), then evaluates the three tags.
  Finished when `source/binaryLightGBM/logs/eval_extra.log` ends with EXTRA_DONE.
  If dead: `nohup setsid bash source/binaryLightGBM/run_extra.sh > source/binaryLightGBM/run_extra.out 2>&1 &` (resumes).

## Remaining
1. Wait for EXTRA_DONE; read `logs/eval_extra.log`.
2. Write `reports/lgbm-binary.md`.

## Notes
- Never use `pgrep -f`/`pkill -f` with a string that also appears in the ssh command itself (self-match/self-kill). Use PIDs.
- Local /tmp staging was wiped by a reboot; local copies now in the agent scratchpad dir `lgbmbin/`. Box files are canonical.
- Pooled-OOF best-F1 objective favours small/low-lr models (best: 214 trees, lr 0.012) with lower AUC than paper params
  (OOF AUC 0.867 vs 0.887) — hence the extra AUC-objective study.
