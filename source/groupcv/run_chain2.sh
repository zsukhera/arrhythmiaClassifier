#!/bin/bash
# Chain 2: wait for chain1 (PID $1), then DS2 eval of groupcv model, task3 variants,
# then extended-space (regularisation) Optuna without SMOTE, balanced class weights.
ROOT=/home/i230510/Desktop/arrhythmiaClassifier
PY=$ROOT/.venv/bin/python
L=$ROOT/reports/lgbm-groupcv
while kill -0 "$1" 2>/dev/null; do sleep 20; done
cd $ROOT/source/groupcv
$PY eval_models.py old_ds1ds2=$ROOT/models/lgbm_ds1ds2.pkl groupcv=$ROOT/models/lgbm_ds1ds2_groupcv.pkl > $L/eval_groupcv.log 2>&1
echo "eval done $(date)" >> $L/chain2.status
$PY task3_variants.py $ROOT/models/checkpoints/lgbm_ds1ds2_groupcv_search_complete.json > $L/task3.log 2>&1
echo "task3 done rc=$? $(date)" >> $L/chain2.status
cd $ROOT
LGBM_DEVICE=cpu $PY source/train.py --splits ds1ds2 --cv group --space extended --sampler none --class-weight balanced --tag _groupcv_reg --n-trials 40 --n-jobs 8 > $L/task3_optuna_reg.log 2>&1
echo "reg optuna done rc=$? $(date)" >> $L/chain2.status
cd $ROOT/source/groupcv
$PY eval_models.py groupcv_reg=$ROOT/models/lgbm_ds1ds2_groupcv_reg.pkl > $L/eval_groupcv_reg.log 2>&1
cd $ROOT/source/groupcv
[ -f $L/task1_baseline.json ] || $PY task1_baseline.py > $L/task1_rerun.log 2>&1
echo "all done $(date)" >> $L/chain2.status
