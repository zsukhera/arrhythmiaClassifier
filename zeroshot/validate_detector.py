"""
Step 2a — validate the R-peak detector on MIT-BIH (MLII) against .atr beat annotations.

Run on 10-s non-overlapping strips (the target condition) for all 44 non-paced records,
for XQRS and GQRS, with and without peak refinement.  Matching tolerance 150 ms
(ANSI/AAMI EC57).  Also records the timing offset detected - annotated for matched
beats (beat windows are centred on R, so offsets matter for the models).

Detector choice is made on DS1 only; DS2 numbers are reported.
Output: out/detector_validation.csv, out/detector_validation.txt
"""
import sys
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import MITDB, DS1, DS2, BEAT_SYMBOLS, STRIP_LEN, FS, OUT, detect_rpeaks  # noqa

TOL = int(0.150 * FS)
CONFIGS = [("xqrs", 18), ("xqrs", 0), ("gqrs", 18), ("gqrs", 0)]


def match(ref, det, tol=TOL):
    ref = np.sort(ref); det = np.sort(det)
    used = np.zeros(len(det), bool)
    tp, offs = 0, []
    for r in ref:
        if len(det) == 0:
            break
        j = np.searchsorted(det, r)
        best, bd = -1, tol + 1
        for jj in (j - 1, j):
            if 0 <= jj < len(det) and not used[jj] and abs(det[jj] - r) < bd:
                best, bd = jj, abs(det[jj] - r)
        if best >= 0 and bd <= tol:
            used[best] = True; tp += 1; offs.append(int(det[best] - r))
    return tp, len(det) - tp, len(ref) - tp, offs


def run_record(rec):
    r = wfdb.rdrecord(str(MITDB / rec))
    sig = r.p_signal[:, r.sig_name.index("MLII")]
    ann = wfdb.rdann(str(MITDB / rec), "atr")
    beats = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in BEAT_SYMBOLS])
    rows = []
    for m, ref in CONFIGS:
        TP = FP = FN = 0; offs = []
        for s0 in range(0, len(sig) - STRIP_LEN + 1, STRIP_LEN):
            det = detect_rpeaks(sig[s0:s0 + STRIP_LEN], m, ref) + s0
            rb = beats[(beats >= s0) & (beats < s0 + STRIP_LEN)]
            tp, fp, fn, o = match(rb, det)
            TP += tp; FP += fp; FN += fn; offs += o
        offs = np.array(offs) if offs else np.array([0])
        rows.append(dict(record=rec, split="DS1" if rec in DS1 else "DS2", method=m, refine=ref,
                         TP=TP, FP=FP, FN=FN, off_median=float(np.median(offs)),
                         off_mad=float(np.median(np.abs(offs - np.median(offs)))),
                         within_3=float(np.mean(np.abs(offs) <= 3)),
                         within_10=float(np.mean(np.abs(offs) <= 10))))
    return rows


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    recs = DS1 + DS2
    with ProcessPoolExecutor(4) as ex:
        rows = [r for rr in ex.map(run_record, recs) for r in rr]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "detector_validation.csv", index=False)
    lines = []
    for split in ("DS1", "DS2", "ALL"):
        d = df if split == "ALL" else df[df.split == split]
        g = d.groupby(["method", "refine"]).agg(TP=("TP", "sum"), FP=("FP", "sum"), FN=("FN", "sum"),
                                                off_median=("off_median", "median"),
                                                within_3=("within_3", "mean"),
                                                within_10=("within_10", "mean")).reset_index()
        g["Se"] = g.TP / (g.TP + g.FN); g["PPV"] = g.TP / (g.TP + g.FP)
        g["F1"] = 2 * g.TP / (2 * g.TP + g.FP + g.FN)
        lines.append(f"== {split} (10-s strips, tol 150 ms) ==\n" + g.to_string(index=False))
    # worst records for the chosen config
    txt = "\n\n".join(lines)
    print(txt)
    (OUT / "detector_validation.txt").write_text(txt + "\n")
