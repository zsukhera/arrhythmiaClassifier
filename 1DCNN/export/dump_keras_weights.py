"""
Step A of the export pipeline (runs in $ROOT/.venv, Keras 3 / TF 2.22rc0).

The .keras files were written by Keras 3; tfmot (QAT / pruning) needs legacy
tf_keras, which cannot read the Keras-3 format.  So we dump every layer's
weights to an .npz (keyed by layer name) plus the Keras-3 reference
probabilities on the evaluation arrays, and rebuild the identical graph in
.venv-export (export_tflite.py) with src.model.build_model.

Usage (from $ROOT):
    .venv/bin/python 1DCNN/export/dump_keras_weights.py MODEL.keras OUT.npz \
        --eval 1DCNN/export/.cache/eval_p1.npz
"""
import argparse
import numpy as np
import tensorflow as tf

ap = argparse.ArgumentParser()
ap.add_argument("model")
ap.add_argument("out")
ap.add_argument("--eval", help="npz with X_ecg_*/X_rr_* arrays to compute reference probas on")
a = ap.parse_args()

m = tf.keras.models.load_model(a.model, compile=False)
d = {}
for layer in m.layers:
    for i, w in enumerate(layer.get_weights()):
        d[f"{layer.name}::{i}"] = w
if a.eval:
    e = np.load(a.eval)
    for split in ("hold", "ds2"):
        if f"X_ecg_{split}" in e:
            d[f"proba_ref_{split}"] = m.predict([e[f"X_ecg_{split}"], e[f"X_rr_{split}"]],
                                                batch_size=4096, verbose=0)
np.savez(a.out, **d)
print(f"saved {len(d)} arrays to {a.out}; params={m.count_params()}")
