#!/bin/bash
# Extra DS1/DS2 binary experiments (2026-11-19). Resumable (Optuna SQLite, refit skipped if pkl exists).
#   ds1ds2_stratifiedcv : beat-level StratifiedKFold on DS1 (leaky, for comparison), 20 trials
#   ds1ds2_paper        : paper params only, grouped CV OOF threshold
#   ds1ds2_auc          : grouped CV, objective = pooled OOF AUC, 60 trials
cd /home/i230510/Desktop/arrhythmiaClassifier
export OMP_NUM_THREADS=6 LGBM_DEVICE=cpu
LOG=source/binaryLightGBM/logs
# wait for an already-running job given as $1 (PID), to stay within 6 threads
if [ -n "$1" ]; then while kill -0 "$1" 2>/dev/null; do sleep 20; done; fi
PY=".venv/bin/python -u source/binaryLightGBM/train_binary.py --split ds1ds2 --threads 6"
[ -f models/lgbm_binary_ds1ds2_stratifiedcv.pkl ] || $PY --n-trials 20 --cv stratified >> $LOG/train_ds1ds2_stratifiedcv.log 2>&1
$PY --n-trials 1 --tag-suffix paper                   >> $LOG/train_ds1ds2_paper.log 2>&1
$PY --n-trials 60 --objective auc --tag-suffix auc    >> $LOG/train_ds1ds2_auc.log 2>&1
.venv/bin/python -u source/binaryLightGBM/evaluate_binary.py --tags ds1ds2_paper ds1ds2_auc ds1ds2_stratifiedcv > $LOG/eval_extra.log 2>&1
echo EXTRA_DONE >> $LOG/eval_extra.log
