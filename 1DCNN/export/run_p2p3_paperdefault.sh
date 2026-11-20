#!/usr/bin/env bash
# Paper-default P2/P3: DS1-train for training, DS1-val for early stopping (val_loss),
# class_weight=none, 10 seeds.  Unchanged src/run_experiment.py CLI, outputs redirected.
cd /home/i230510/Desktop/arrhythmiaClassifier/HasanML/paper3
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3
/home/i230510/Desktop/arrhythmiaClassifier/.venv/bin/python -m src.run_experiment --p2 --p3 --config /home/i230510/Desktop/arrhythmiaClassifier/1DCNN/export/configs/p2p3_paperdefault.yaml
