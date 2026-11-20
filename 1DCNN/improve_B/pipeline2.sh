#!/bin/bash
# Stage-2: 10-seed finalists (chosen on DS1 OOF macro-F1: gkf p0.75 best, p0.5 next) + gkf p0 (B1 under patient-wise ES)
cd /home/i230510/Desktop/arrhythmiaClassifier/1DCNN/improve_B
PY=../../.venv/bin/python
$PY train_jobs.py --scheme gkf --p 0.75 0.5 0 --seeds 0-9 --workers 7 > logs/gkf_final.log 2>&1
OMP_NUM_THREADS=1 $PY analyze.py > logs/analyze_stage2.log 2>&1
echo STAGE2_DONE >> logs/pipeline2.log
