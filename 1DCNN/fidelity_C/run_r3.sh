#!/bin/bash
# Round 3: farag-improve protocol on the paper architecture (after run_r2.sh).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=""
PY=/home/i230510/Desktop/arrhythmiaClassifier/.venv/bin/python
while pgrep -f run_r2.sh >/dev/null; do sleep 60; done
echo "=== cv $(date)"
$PY fid.py --group cv_fz075 --seeds 0 --workers 5
$PY - <<'P'
import json, glob, numpy as np
e = [json.load(open(f))["best_epoch_f1"] for f in glob.glob("runs/CV_FZ075_f*/seed0.json")]
print("CV best epochs", e)
json.dump({"FZ075": int(round(float(np.median(e)))), "folds": e}, open("cv_epochs.json", "w"))
P
echo "=== refit $(date)"
$PY fid.py --group refit_fz075 --seeds 0 1 2 --workers 3
echo "=== R3 DONE $(date)"
