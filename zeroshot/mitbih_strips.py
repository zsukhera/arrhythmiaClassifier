"""
Step 4a — MIT-BIH DS1/DS2 cut into non-overlapping 10-s MLII strips, run through the
exact external-data pipeline (detected R-peaks), plus a diagnostic variant that uses
the .atr R-peak positions instead of the detector.

Outputs (out/):
  mitbih_strips.pkl   per-strip table (truth beat counts + per-model predicted counts)
  mitbih_beats.pkl    per-beat table (detected peaks matched to .atr, predictions)
"""
import sys
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import wfdb

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (MITDB, DS1, DS2, BEAT_SYMBOLS, STRIP_LEN, OUT, LGBM_CLASSES, CNN_CLASSES,  # noqa
                    lgbm_predict_proba, cnn_predict_proba, cnn_threshold_predict)
from featurize import featurize  # noqa

AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V", "F": "F", "/": "Q", "f": "Q", "Q": "Q"}
TOL = 54  # 150 ms


def nearest_sym(peaks, ref_s, ref_sym):
    out = []
    for p in peaks:
        if len(ref_s) == 0:
            out.append(""); continue
        j = np.searchsorted(ref_s, p)
        c = [jj for jj in (j - 1, j) if 0 <= jj < len(ref_s)]
        jj = min(c, key=lambda q: abs(ref_s[q] - p))
        out.append(ref_sym[jj] if abs(ref_s[jj] - p) <= TOL else "")
    return out


def do_record(rec):
    r = wfdb.rdrecord(str(MITDB / rec))
    sig = r.p_signal[:, r.sig_name.index("MLII")].astype(np.float64)
    ann = wfdb.rdann(str(MITDB / rec), "atr")
    bs = np.array([(s, y) for s, y in zip(ann.sample, ann.symbol) if y in BEAT_SYMBOLS], dtype=object)
    ref_s = bs[:, 0].astype(np.int64); ref_sym = bs[:, 1].astype(str)
    strips = []
    for k, s0 in enumerate(range(0, len(sig) - STRIP_LEN + 1, STRIP_LEN)):
        x = sig[s0:s0 + STRIP_LEN]
        m = (ref_s >= s0) & (ref_s < s0 + STRIP_LEN)
        rs, rsym = ref_s[m] - s0, ref_sym[m]
        det = featurize(x)
        atr = featurize(x, peaks=rs)
        strips.append(dict(record=rec, split="DS1" if rec in DS1 else "DS2", strip=k, start=s0,
                           ref_syms=rsym, ref_samples=rs, det=det, atr=atr,
                           det_sym_lgbm=nearest_sym(det["lgbm_peaks"], rs, rsym),
                           det_sym_cnn=nearest_sym(det["peaks"], rs, rsym),
                           atr_sym_lgbm=nearest_sym(atr["lgbm_peaks"], rs, rsym),
                           atr_sym_cnn=list(rsym)))
    return strips


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(4) as ex:
        allstrips = [s for ss in ex.map(do_record, DS1 + DS2) for s in ss]
    print(f"{len(allstrips)} strips featurised", flush=True)

    # batch predictions
    beat_rows, strip_rows = [], []
    for src in ("det", "atr"):
        Xl = np.concatenate([s[src]["X_lgbm"] for s in allstrips])
        Pl = lgbm_predict_proba(Xl)
        Xe = np.concatenate([s[src]["X_ecg"] for s in allstrips])
        Pc = {mode: cnn_predict_proba(Xe, np.concatenate([s[src][f"X_rr_{mode}"] for s in allstrips]))
              for mode in ("causal", "strip")}
        il = ic = 0
        for s in allstrips:
            nl, nc = len(s[src]["X_lgbm"]), len(s[src]["X_ecg"])
            s[f"{src}_P_lgbm"] = Pl[il:il + nl]; il += nl
            for mode in ("causal", "strip"):
                s[f"{src}_P_cnn_{mode}"] = Pc[mode][ic:ic + nc]
            ic += nc
    for s in allstrips:
        syms = s["ref_syms"]
        row = dict(record=s["record"], split=s["split"], strip=s["strip"], start=s["start"],
                   n_ref=len(syms), n_det=len(s["det"]["peaks"]))
        for c in "NLRVAaJSEFjef/Q":
            row[f"ref_{c}"] = int(np.sum(syms == c))
        for src in ("det", "atr"):
            yl = np.array(LGBM_CLASSES)[np.argmax(s[f"{src}_P_lgbm"], 1)] if len(s[f"{src}_P_lgbm"]) else np.array([])
            row[f"{src}_lgbm_n"] = len(yl)
            for c in LGBM_CLASSES:
                row[f"{src}_lgbm_cnt_{c}"] = int(np.sum(yl == c))
            for mode in ("causal", "strip"):
                P = s[f"{src}_P_cnn_{mode}"]
                yc = np.array(CNN_CLASSES)[cnn_threshold_predict(P)] if len(P) else np.array([])
                row[f"{src}_cnn_{mode}_n"] = len(yc)
                for c in CNN_CLASSES:
                    row[f"{src}_cnn_{mode}_cnt_{c}"] = int(np.sum(yc == c))
            # beat level
            for b, (sym, p) in enumerate(zip(s[f"{src}_sym_lgbm"], s[f"{src}_P_lgbm"])):
                beat_rows.append(dict(record=s["record"], split=s["split"], src=src, model="lgbm",
                                      true=sym, pred=LGBM_CLASSES[int(np.argmax(p))]))
            for mode in ("causal", "strip"):
                P = s[f"{src}_P_cnn_{mode}"]
                pr = cnn_threshold_predict(P) if len(P) else []
                for sym, p in zip(s[f"{src}_sym_cnn"], pr):
                    beat_rows.append(dict(record=s["record"], split=s["split"], src=src,
                                          model=f"cnn_{mode}", true=sym, pred=CNN_CLASSES[int(p)]))
        strip_rows.append(row)
    pd.DataFrame(strip_rows).to_pickle(OUT / "mitbih_strips.pkl")
    pd.DataFrame(beat_rows).to_pickle(OUT / "mitbih_beats.pkl")
    print("saved", flush=True)


if __name__ == "__main__":
    main()
