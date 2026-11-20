#!/bin/bash
# Extra variant: Conv-BiLSTM + 4 causal RR-ratio features (run after run_all.sh).
cd "$(dirname "$0")"
PY=../.venv-torch/bin/python
export OMP_NUM_THREADS=4
SEEDS="0 1 2 3 4"
[ -f data/rr.npz ] || $PY rr_features.py
arch=conv
for scheme in aami3 nlrva; do
  for seed in $SEEDS; do $PY train.py --scheme $scheme --split ds1ds2_heldout --arch $arch --rr --seed $seed; done
  E=$($PY -c "import json,glob,numpy as np;print(int(np.median([json.load(open(f))['best_epoch'] for f in glob.glob('runs/${scheme}_ds1ds2_heldout_${arch}rr/seed*/metrics.json')])))")
  echo "refit $scheme ${arch}rr epochs=$E"
  for seed in $SEEDS; do $PY train.py --scheme $scheme --split ds1ds2_refit --arch $arch --rr --epochs $E --seed $seed; done
done
for seed in $SEEDS; do $PY train.py --scheme nlrva --split random --arch $arch --rr --seed $seed; done
echo RR_DONE
