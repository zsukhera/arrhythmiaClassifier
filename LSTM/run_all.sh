#!/bin/bash
# Full LSTM-baseline experiment grid (sequential, resumable: finished runs are skipped).
cd "$(dirname "$0")"
PY=../.venv-torch/bin/python
export OMP_NUM_THREADS=4
SEEDS="0 1 2 3 4"
for arch in conv pool; do
  for seed in $SEEDS; do $PY train.py --scheme aami3 --split ds1ds2_heldout --arch $arch --seed $seed; done
  for seed in $SEEDS; do $PY train.py --scheme nlrva --split ds1ds2_heldout --arch $arch --seed $seed; done
  # refit on all of DS1 for the median best epoch of the held-out-record runs (no DS2 involvement)
  for scheme in aami3 nlrva; do
    E=$($PY -c "import json,glob,numpy as np;print(int(np.median([json.load(open(f))['best_epoch'] for f in glob.glob('runs/${scheme}_ds1ds2_heldout_${arch}/seed*/metrics.json')])))")
    echo "refit $scheme $arch epochs=$E"
    for seed in $SEEDS; do $PY train.py --scheme $scheme --split ds1ds2_refit --arch $arch --epochs $E --seed $seed; done
  done
  for seed in $SEEDS; do $PY train.py --scheme nlrva --split random --arch $arch --seed $seed; done
done
echo ALL_DONE
