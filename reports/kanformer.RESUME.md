# kanformer agent — RESUME notes (updated 2026-11-18 13:35)

ROOT=/home/i230510/Desktop/arrhythmiaClassifier ; P4=$ROOT/HasanML/paper4 ; PY=$ROOT/.venv-torch/bin/python
Box copy of HasanML/paper4 is the source of truth (local laptop scratch copy /tmp/kfwork is disposable).
Budget: <=6 CPU threads (OMP_NUM_THREADS=6, LightGBM n_jobs=6, dataloader_workers<=4), <=14 GB GPU.

## Done
1. Unit tests: 11 failures fixed -> `cd $P4 && $PY -m pytest tests/ -q` = 39 passed.
   Fixes: kan.py bitwise-or on float (crash in every KAN forward!); losses.py FocalLoss mean is now
   alpha-weighted mean (== weighted CE at gamma=0) and build_loss accepts ("focal", class_counts=...);
   model.py config defaults via .get + n_rr=0 support (ablation A4 would have crashed);
   test RR causality corrected (post-RR uses 1-beat look-ahead by design).
2. RR LightGBM sanity (DS1-train -> DS1-val): acc 87.1, Se_S 47.5, Se_V 52.3. Output results/sanity_rr/.
   rr.py LightGBM n_jobs -1 -> 6.
   Beat counts: DS1-train N23932 S149 V1234 F13 Q2 | DS1-val N21934 S795 V2554 F402 Q6 | DS2 N44259 S1837 V3221 F388 Q7
3. train.py bugs fixed: (a) sampler gave Q ~86% / F ~13% of every batch (only S/V capped) -> water-filling
   caps incl. f_cap 0.05, q_cap 0.01 (added to configs/default.yaml); (b) DataLoader workers shared one
   identical augmentation RNG -> reseeded per worker; (c) torch.cuda.amp.GradScaler -> torch.amp;
   (d) train_one now checkpoints every epoch to results/<run>/cache/<RUN>_seed<S>_ckpt.pt and resumes;
   logs val Se/PPV for S,V per epoch; loss computed in fp32.
   Known but not yet fixed: ONNX/TorchScript export in run_experiment --distill uses raw (non-EMA) weights.

   (e) alpha (focal class weights) were computed from RAW counts on top of the balanced sampler (double
   correction -> model predicted F everywhere) -> alpha now from sampler target proportions;
   (f) EMA warm-up (decay min(0.999,(1+t)/(10+t))), only ~100 steps/epoch.
   run_experiment: new flags --max-epochs, --set key.sub=val; --results-dir now also redirects cache/figures/teacher.
   data.py: ds1_split("farag"|"balanced"); config keys data.ds1_split, training.select_classes.
   New src/evaluate_full.py (5-class + 3-class N/S/V + temp scaling + ensemble) -> <run>/eval/eval_table_DEFAULT.md
4. GPU sanity: results/sanity_gpu (6 epochs, ~12-19 s/epoch, ~1.6 GB GPU).

## Status 2026-11-19 ~11:00 box time
DONE: results/gpu_run (default, 3 seeds) + eval -> results/gpu_run/eval/eval_table_DEFAULT.md
      DS2 5c: acc 84.9+-2.4, macroF1 34.1+-4.4, Se_S 13.0+-10.3, Se_V 90.3 (mean single seeds). Far below targets.
DONE: results/fix_balsplit (Fix A, balanced DS1 split) + eval: DS2 5c acc 87.9, macroF1 41.0, Se_S 23.2, Se_V 90.3;
      3c acc 91.45, macroF1 65.2. Val N/S/V macro-F1 0.63-0.68 (vs default ~0.35 5-class).
DONE: results/diag_rr.txt (RR-only LightGBM 3c on DS2 for diagnosis; DS2 S beats have post/local 0.81 -> SVT runs).
RUNNING:
 - ablations_a (A1-A4, seeds 0-2, default config): results/ablations_a/stdout.log; done when "Ablation summary" in log.
   resume: re-run same cmd (ckpt per epoch):
   OMP_NUM_THREADS=1 setsid nohup  $PY -m src.run_experiment --seeds 0 1 2 --ablations --ablation-seeds 0 1 2 --ablation-names A1_mlp_head A2_no_transformer A3_no_context A4_no_rr --device cuda --mixed-precision --results-dir results/ablations_a --set compute.dataloader_workers=3 > results/ablations_a/stdout.log 2>&1 < /dev/null &
 - Fix B (balanced split + stronger regularisation): bash results/launch_fixreg.sh -> results/fix_reg/stdout.log
PAUSED (to run Fix B within CPU budget): ablations_b (A5,A6a,A6b). Resume after fix_reg finishes:
   OMP_NUM_THREADS=1 setsid nohup  $PY -m src.run_experiment --seeds 0 1 2 --ablations --ablation-seeds 0 1 2 --ablation-names A5_single_kernel A6a_grid3 A6b_grid8 --device cuda --mixed-precision --results-dir results/ablations_b --set compute.dataloader_workers=2 > results/ablations_b/stdout.log 2>&1 < /dev/null &
 (ablation dirs have DEFAULT_seed*.pkl copied from gpu_run so DEFAULT is not retrained)
NEXT: evaluate_full on fix_reg; evaluate each ablation:  $PY -m src.evaluate_full --results-dir results/ablations_a --run-name A1_mlp_head (etc);
  pick teacher = best DS1-val among runs?? (NO: choose final config by DS1-val, then export_teacher from that run):
   $PY -m src.export_teacher --results-dir results/<chosen> --out-dir results/teacher ; copy README (teacher_README) ;
  write reports/kanformer.md. 3-class targets (Farag Model 6): acc 98.18, Se N/S/V 99.10/81.60/95.00, PPV_S 82.68, macroF1 92.17.
WARNING: never run pkill -f PATTERN in an ssh command containing the pattern (kills the ssh shell).

## Status 2026-11-19 ~15:20 box time
DONE: fix_reg (Fix B) + eval; teacher exported from fix_reg ensemble(3) -> results/teacher (README, npz, onnx, ts, export_check.json);
      ablations_a (A1-A4 x3 seeds) + eval; report written: reports/kanformer.md
RUNNING: ablations_b seed 0 only (A5, A6a, A6b) -> results/ablations_b/stdout.log (launched by results/launch_stage2.sh).
REMAINING: when "Ablation summary" appears in ablations_b log: bash results/eval_ablations.sh; $PY results/abl_table.py;
           paste A5/A6a/A6b rows into section 4 of reports/kanformer.md.

## COMPLETE (2026-11-19 evening)
All tasks are done. A5/A6a/A6b (seed 0) finished after the RNG resume fix (log: results/ablations_b/stdout2.log), were evaluated, and were added to reports/kanformer.md section 4.
Nothing is running.
