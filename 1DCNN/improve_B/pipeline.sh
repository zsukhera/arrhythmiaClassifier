#!/bin/bash
# Stage-1 pipeline for farag-improve (resumable: finished runs are skipped).
cd /home/i230510/Desktop/arrhythmiaClassifier/1DCNN/improve_B
PY=../../.venv/bin/python
while pgrep -f "train_jobs.py --scheme rand --p 1 0 0.25" >/dev/null; do sleep 20; done
$PY train_jobs.py --scheme rand --p 1 0 0.25 0.5 0.75 --seeds 0-2 --workers 7 >> logs/rand_sweep.log 2>&1
$PY train_jobs.py --scheme gkf --p 0 0.25 0.5 0.75 1 --seeds 0-2 --workers 7 > logs/gkf_sweep.log 2>&1
$PY train_jobs.py --scheme rand --p 0 --seeds 0-9 --workers 7 > logs/b1_rand_p0.log 2>&1
$PY train_jobs.py --scheme rand --p 0 --seeds 0-9 --es loss --workers 7 > logs/b1_rand_p0_esloss.log 2>&1
$PY analyze.py > logs/analyze_stage1.log 2>&1
echo STAGE1_DONE >> logs/pipeline.log
