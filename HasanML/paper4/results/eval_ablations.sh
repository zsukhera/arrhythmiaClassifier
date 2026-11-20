#!/bin/bash
# Evaluate every ablation run (5-class + 3-class DS2, temp scaling) — run from anywhere
cd ~/Desktop/arrhythmiaClassifier/HasanML/paper4
PY=../../.venv-torch/bin/python
for d in ablations_a ablations_b; do
  for n in $(ls results/$d/cache 2>/dev/null | grep -E '^A.*_seed[0-9]\.pt$' | sed 's/_seed[0-9]\.pt//' | sort -u); do
    OMP_NUM_THREADS=2 $PY -m src.evaluate_full --results-dir results/$d --run-name $n > results/$d/eval_$n.log 2>&1
    echo "done $d $n"
  done
done
