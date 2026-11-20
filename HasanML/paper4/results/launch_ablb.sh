#!/bin/bash
cd ~/Desktop/arrhythmiaClassifier/HasanML/paper4
OMP_NUM_THREADS=1 setsid nohup ../../.venv-torch/bin/python -m src.run_experiment --seeds 0 1 2 --ablations --ablation-seeds 0 \
  --ablation-names A5_single_kernel A6a_grid3 A6b_grid8 --device cuda --mixed-precision \
  --results-dir results/ablations_b --set compute.dataloader_workers=4 > results/ablations_b/stdout2.log 2>&1 < /dev/null &
echo $!
