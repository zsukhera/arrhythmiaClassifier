#!/bin/bash
# lgbm-binary pipeline driver. Resumable: Optuna SQLite + skip-if-exists.
cd /home/i230510/Desktop/arrhythmiaClassifier
export OMP_NUM_THREADS=6 LGBM_DEVICE=cpu
LOG=source/binaryLightGBM/logs; mkdir -p $LOG
# wait for ablation jobs (6 threads each) so we stay within budget
while pgrep -f "source/ablation/ablation.py" >/dev/null; do sleep 20; done
.venv/bin/python -u source/binaryLightGBM/train_binary.py --split ds1ds2 --n-trials 80 --threads 6 >> $LOG/train_ds1ds2_grouped.log 2>&1
.venv/bin/python -u source/binaryLightGBM/train_binary.py --split random --n-trials 25 --threads 6 >> $LOG/train_random.log 2>&1
.venv/bin/python -u source/binaryLightGBM/evaluate_binary.py --tags random ds1ds2 >> $LOG/eval.log 2>&1
echo PIPELINE_DONE >> $LOG/eval.log
