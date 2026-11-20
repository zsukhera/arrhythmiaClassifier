"""Sanity: frozen models reproduce their documented MIT-BIH DS2 numbers (beat level, original pipelines)."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, lgbm_predict_proba, cnn_predict_proba, cnn_threshold_predict, LGBM_CLASSES
from sklearn.metrics import f1_score, accuracy_score, recall_score
d = np.load(ROOT / "1DCNN/1DCNN experiments/.cache/ds2.npz")
P = cnn_predict_proba(d["X_ecg"], d["X_rr"]); y = cnn_threshold_predict(P)
print("CNN DS2 acc %.4f Se(N,S,V)=%s" % (accuracy_score(d["y"], y), recall_score(d["y"], y, average=None)))
s = np.load(ROOT / "datasets/processed/split_ds1ds2.npz", allow_pickle=True)
yl = np.array(LGBM_CLASSES)[np.argmax(lgbm_predict_proba(s["X_test"]), 1)]
print("LGBM DS2 acc %.4f macroF1 %.4f" % (accuracy_score(s["y_test"], yl), f1_score(s["y_test"], yl, average="macro")))
