#!/bin/bash
cd ~/Desktop/arrhythmiaClassifier/HasanML/paper4
PY=../../.venv-torch/bin/python
# ablations_b: A5/A6a/A6b, seed 0 only (cut short for time), resumes from per-epoch ckpt
OMP_NUM_THREADS=1 setsid nohup $PY -m src.run_experiment --seeds 0 1 2 --ablations --ablation-seeds 0 \
  --ablation-names A5_single_kernel A6a_grid3 A6b_grid8 --device cuda --mixed-precision \
  --results-dir results/ablations_b --set compute.dataloader_workers=3 >> results/ablations_b/stdout.log 2>&1 < /dev/null &
echo "abl_b pid $!"
# evaluations (sequential): fix_reg then ablations_a
( OMP_NUM_THREADS=2 $PY -m src.evaluate_full --results-dir results/fix_reg > results/fix_reg/eval_stdout.log 2>&1;
  bash results/eval_ablations.sh > results/eval_ablations_a.log 2>&1; echo EVAL_DONE >> results/eval_ablations_a.log ) < /dev/null > /dev/null 2>&1 &
echo "eval pid $!"
