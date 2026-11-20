#!/bin/bash
# Fix B: balanced DS1 split + stronger regularisation / milder oversampling
cd ~/Desktop/arrhythmiaClassifier/HasanML/paper4
mkdir -p results/fix_reg
OMP_NUM_THREADS=1 setsid nohup ../../.venv-torch/bin/python -m src.run_experiment --seeds 0 1 2 --device cuda --mixed-precision \
  --results-dir results/fix_reg --set data.ds1_split=balanced "training.select_classes=[0,1,2]" compute.dataloader_workers=2 \
  model.dropout=0.3 model.stoch_depth=0.2 model.head_dropout=0.4 training.weight_decay=0.1 training.lr=1.0e-4 \
  training.s_cap=0.15 training.v_cap=0.15 > results/fix_reg/stdout.log 2>&1 < /dev/null &
echo $!
