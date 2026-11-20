# farag-fidelity RESUME (agent farag-fidelity, task section C)
ROOT=/home/i230510/Desktop/arrhythmiaClassifier ; my dir = $ROOT/1DCNN/fidelity_C

## KEY FINDINGS SO FAR (from paper full text, Europe PMC PMC9919183)
- Real Table 4 has 9 models; our results/paper_table4.md is WRONG in places:
  Model 6 (3cls, MF trainable, NK32, derivative, class weight NOT set): ACC 98.18 (PLAIN accuracy,
  verified from per-class Se and DS2 counts), N PREC/SEN 99.00/99.10, SVEB 82.68/81.60, VEB 95.63/95.00,
  avg PREC/SEN/F1 92.44/91.90/92.17 (macro over N,S,V).  SVEB Se 85.30 & VEB Se 96.34 are MODEL 1
  (default init, NK64, deriv, class weight SET), not Model 6.
  M1..M5 use class weights SET; M5 = SIGNAL input 3-class; M8 = 4-class, M9 = 5-class.
- Table 3 architecture differs from ours: input stays 64 samples for NK=32 (padding same, output (64,NF)),
  Conv1D linear -> BN -> Tanh -> GMP; NF=11 for 3 classes (12/13 for 4/5);
  RR (4,1) -> Dense32-16-8 ReLU per feature -> Flatten(32). Params 1267/1619 (we reproduce exactly).
- Templates: "mean of all examples per class in the training set DS1" (=all DS1).
- Training: Adam 1e-3 "adaptive LR scheduling", batch 512, 500 epochs, early stopping (unspecified),
  train = records <200, val = >=200; 10 repeats; "<=100 epochs, <2 min"; mentions ">50K training examples".
- Signals: "raw MLII ... without pre-processing"; a "normalization" step is timed but not defined.
- run_experiment --ablations is a NO-OP (flag parsed, never used). Legacy --grid is broken: data built once
  with default input mode/NK (input never varies; NK64 variants mismatch) and templates always from
  default (derivative) mode. Fixed behind templates.recompute_per_variant: true (default false).

## DONE
- src changes (additive, defaults unchanged, 34/34 tests pass), originals backed up in
  $ROOT/1DCNN/fidelity_C/backup_src_orig/: model.py (arch=table3, n_filters, input_len, n_rr),
  train.py (early_stop_monitor, lr_schedule, kernel crop), preprocess.py (resample method,
  input.crop_input, minmax norm), rr.py (mean_includes_current, record_norm, record_mean_type),
  run_experiment.py (templates.source ds1_all actually works, templates.match_classes,
  data.train_records, --grid-table4 flag, recompute_per_variant fix, rr options plumbed).
- Driver: $ROOT/1DCNN/fidelity_C/fid.py + variants.py (groups: ladder table4 window norm rr c4 train
  input resample table4_orig). Results: runs/<variant>/seed<k>.json (+_proba.npz). Cache: cache/.

## RUNNING
- run_all.sh (nohup), log $ROOT/1DCNN/fidelity_C/run_all.log ; finished when log has "=== ALL DONE".
  Check: pgrep -f run_all.sh ; resumable: re-running ./run_all.sh skips finished seed jsons.

- run_extra.sh (nohup) waits for run_all.sh then runs group 'extra'; log run_extra.log, done at "=== EXTRA DONE".
- Speed: table3 (64-sample, same padding) ~5 s/epoch single-thread on loaded box; ~3 min/run with
  val_loss ES; balanced-CW runs (macro-F1 ES) can run up to 500 epochs.
- Early result (ladder, DS1-train only, no CW): every variant gets Se_S = 0; paper arch even worse on V.
- Also done: run_experiment --results-dir now also moves figures; new --cache-dir flag (coordinator request).
- summarise.py written: `../../.venv/bin/python summarise.py > results.md` (also results_table.csv).

## REMAINING (in order)
1. Wait for run_all; fix any failing group and re-run (nohup ./run_all.sh > run_all2.log 2>&1 &).
2. Write summarise.py (aggregate runs/*/seed*.json -> results tables md: plain/OvR acc, macro F1/PREC,
   per-class Se/PPV mean±std) -> $ROOT/1DCNN/fidelity_C/results.md
3. Optional: CLI check `python -m src.run_experiment --grid-table4 --seeds 0 --config <fidelity config>
   --results-dir <my dir>` with output.cache_dir pointed into fidelity_C.
4. Write recommended config $ROOT/HasanML/paper3/configs/p1_faithful.yaml (+copy in fidelity_C).
5. Write $ROOT/reports/farag-fidelity.md ; final SubagentHandback.

## UPDATE 2026-11-19 10:55
- Round 1 (run_all.sh + run_extra.sh) FINISHED; results in 1DCNN/fidelity_C/results.md (summarise.py).
  Window group failed (param guard on 3-filter AAMI templates) -> guard now only for arch=compact (train.py).
- KEY: with DS1-train (<200) only, every no-class-weight variant has Se_S = 0 (ours and paper arch).
  Training on ALL DS1 (paper "more than 50K training examples") + paper arch + z-score:
  P_trainAll_z: acc 97.37, macF1 88.5, PREC 88.5, Se S/V 73.5/95.0, PPV S/V 72.8/94.0;
  10?-seed-ensemble of 3 seeds: acc 98.06, F1 91.36 (paper M6: 98.18 / 92.17).
- Round 2 RUNNING: run_r2.sh (nohup), log 1DCNN/fidelity_C/run_r2.log, done at "=== R2 DONE".
  Groups: window (old base), r2_core, r2_table4, r2_abl, r2_c4, r2_train (FZ base = PAPER + ds1_all +
  max 100 epochs + z-score). Resumable: rerun ./run_r2.sh.
- Next: summarise.py > results.md; write HasanML/paper3/configs/p1_faithful.yaml; optional CLI
  check --grid-table4 with 1DCNN/fidelity_C/configs/paper_faithful_cli.yaml; write reports/farag-fidelity.md.

## UPDATE 2026-11-19 11:20
- Fixed (coordinator/farag-export requests): P2/P3 val_f1 threshold (new csv *_valthreshold.csv),
  collapsed-P1 loads cached P1 model without --p1, 2-class AUC, templates.empty_fallback: aami
  ('S' template = mean of A/a/J siblings; default zero), training.class_weight_power (default 1).
  All deployed, 34/34 tests pass.
- First FZ results: FZ seed0 acc 97.49 SeS 82.4 SeV 95.1; seed1 acc 98.06 SeS 79.1 SeV 95.3 (~paper M6).
  F_mean (no z-score) much worse (acc 95.3, SeS 28) -> z-score matters a lot. Runs ~800 s each (box busy).
- run_r3.sh (nohup, log run_r3.log, done at "=== R3 DONE"): waits for run_r2, then 5-fold GroupKFold
  (farag-improve folds) epoch selection for FZ + cw^0.75, writes cv_epochs.json, refits R_FZ075 x3 seeds.

## UPDATE 2026-11-19 15:15
- run_r2 + CV done (results.md regenerated). CV best epochs [12,95,26,6,36] -> median 26.
- refit group missing from GROUPS on first try; fixed (appended to variants.py). RUNNING:
  run_refit.log (R_FZ075, seeds 0-4, 5 workers), run_fz34.log (FZ seeds 3,4).
  run_cli.sh (waits for fid.py to exit, then --grid-table4 seed 0, log run_cli.log, "=== CLI DONE").
- Wrote configs: HasanML/paper3/configs/p1_faithful.yaml (recommended), fidelity_C/configs/*.yaml.
- Next: summarise.py > results.md; write reports/farag-fidelity.md.

## UPDATE 2026-11-19 16:00 — COMPLETE
All runs finished; reports/farag-fidelity.md is final. Nothing running.
