#!/bin/bash
# Round 2: ablations on the full-DS1 base (FZ). Resumable (skips finished seeds).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
PY=/home/i230510/Desktop/arrhythmiaClassifier/.venv/bin/python
$PY fid.py --group window --seeds 0 1 2 --workers 6 || echo "!!! group window failed"
for g in r2_core r2_table4 r2_abl r2_c4 r2_train; do
  echo "=== group $g $(date)"
  $PY fid.py --group $g --seeds 0 1 2 --workers 6 || echo "!!! group $g failed"
done
echo "=== R2 DONE $(date)"
