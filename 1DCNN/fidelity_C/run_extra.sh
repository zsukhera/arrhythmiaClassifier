#!/bin/bash
# waits for run_all.sh then runs the 'extra' group
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=''
while pgrep -f run_all.sh >/dev/null; do sleep 60; done
../../.venv/bin/python fid.py --group extra --seeds 0 1 2 --workers 6
echo '=== EXTRA DONE'
