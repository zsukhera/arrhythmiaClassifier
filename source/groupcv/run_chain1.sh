#!/bin/bash
# Chain: wait for task1 (PID $1), then task4 analysis, then Task-2 Optuna (grouped CV, paper space, 100 trials).
ROOT=/home/i230510/Desktop/arrhythmiaClassifier
PY=$ROOT/.venv/bin/python
L=$ROOT/reports/lgbm-groupcv
while kill -0 "$1" 2>/dev/null; do sleep 20; done
cd $ROOT/source/groupcv
[ -f $L/task4_lr_shift.json ] || GROUPCV_NJOBS=8 $PY task4_lr_shift.py > $L/task4.log 2>&1
echo "task4 done $(date)" >> $L/chain1.status
cd $ROOT
LGBM_DEVICE=cpu $PY source/train.py --splits ds1ds2 --cv group --tag _groupcv --n-trials 40 --n-jobs 8 > $L/task2_optuna.log 2>&1
echo "task2 done rc=$? $(date)" >> $L/chain1.status
