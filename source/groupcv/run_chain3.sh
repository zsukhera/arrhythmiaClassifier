#!/bin/bash
# Chain 3 (resume day 2): extended-space Optuna without the B2 morphology group (best feature set
# by grouped CV in task3), then task5_final.py (LOGO robustness check, best-variant refit, gCV-vs-DS2 correlation).
ROOT=/home/i230510/Desktop/arrhythmiaClassifier
PY=$ROOT/.venv/bin/python
L=$ROOT/reports/lgbm-groupcv
cd $ROOT
LGBM_DEVICE=cpu $PY source/train.py --splits ds1ds2 --cv group --space extended --sampler none --class-weight balanced \
  --drop-features fid_r_amp fid_q_amp fid_s_amp fid_t_amp fid_p_amp fid_rs_ratio fid_pr_ratio fid_st_elev \
  --tag _groupcv_reg_noB2 --n-trials 40 --n-jobs 8 > $L/task3_optuna_reg_noB2.log 2>&1
echo "reg_noB2 optuna done rc=$? $(date)" >> $L/chain3.status
cd $ROOT/source/groupcv
$PY eval_models.py groupcv_reg_noB2=$ROOT/models/lgbm_ds1ds2_groupcv_reg_noB2.pkl > $L/eval_groupcv_reg_noB2.log 2>&1
echo "eval noB2 done $(date)" >> $L/chain3.status
$PY task5_final.py > $L/task5.log 2>&1
echo "task5 done rc=$? $(date)" >> $L/chain3.status
