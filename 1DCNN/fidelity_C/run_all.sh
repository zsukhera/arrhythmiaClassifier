#!/bin/bash
# Runs all fidelity_C variant groups, 3 seeds, 6 single-thread workers.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
PY=../../.venv/bin/python
for g in ladder table4 window norm rr c4 train input resample table4_orig; do
  echo "=== group $g $(date)"
  $PY fid.py --group $g --seeds 0 1 2 --workers 6 || echo "!!! group $g failed"
done
echo "=== ALL DONE $(date)"
