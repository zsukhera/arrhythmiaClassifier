#!/bin/bash
# Stage-3 (B4 refit): wait for stage 2, then refit on full DS1 with CV-chosen epochs, then analyze.
cd /home/i230510/Desktop/arrhythmiaClassifier/1DCNN/improve_B
PY=../../.venv/bin/python
until grep -q STAGE2_DONE logs/pipeline2.log 2>/dev/null; do sleep 30; done
$PY train_jobs.py --scheme refit --p 0.75 0.5 0 --seeds 0-9 --workers 7 > logs/refit.log 2>&1
OMP_NUM_THREADS=1 $PY analyze.py > logs/analyze_stage3.log 2>&1
echo STAGE3_DONE >> logs/pipeline3.log
