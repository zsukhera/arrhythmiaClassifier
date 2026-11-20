#!/bin/bash
# C3 CLI check: python -m src.run_experiment --grid-table4 (seed 0) on the p1_faithful base.
F=/home/i230510/Desktop/arrhythmiaClassifier/1DCNN/fidelity_C
while pgrep -f 'fid.py' >/dev/null; do sleep 60; done
cd /home/i230510/Desktop/arrhythmiaClassifier/HasanML/paper3
export OMP_NUM_THREADS=6 TF_CPP_MIN_LOG_LEVEL=2 CUDA_VISIBLE_DEVICES=''
taskset -c 0-5 ../../.venv/bin/python -m src.run_experiment --grid-table4   --config $F/configs/paper_faithful_cli.yaml --results-dir $F/cli_results --cache-dir $F/cli_cache
echo '=== CLI DONE' $(date)
