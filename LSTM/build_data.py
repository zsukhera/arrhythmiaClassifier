"""Build 720-sample (2 s @ 360 Hz) two-channel windows around every annotated beat.

Preprocessing mirrors source/preprocess.py: MLII selected by name (channel 0),
the other lead as channel 1, 4th-order Butterworth band-pass 0.5-40 Hz (SOS,
zero-phase sosfiltfilt). Paced records 102/104/107/217 excluded.

Differences from source/preprocess.py (deliberate):
  * window = 360 samples before + 360 after the R-peak (720 total);
  * beats whose window crosses the record boundary are EDGE-PADDED rather than
    dropped, so that every beat used by the LightGBM splits (which were built
    from 256-sample windows) also exists here and the test sets are identical;
  * all beat symbols that have an AAMI class are kept (needed for N/S/V), not
    only N/L/R/V/A.

Output: LSTM/data/windows720.npz
  X          (n, 720, 2) float16   raw filtered mV (normalised later, per window)
  symbols    (n,) <U1              MIT-BIH beat symbol
  record_ids (n,) <U3
  r_samples  (n,) int32
"""
from pathlib import Path
import numpy as np, wfdb, scipy.signal as sps

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "datasets" / "mitbih" / "mitdb"
OUT = Path(__file__).resolve().parent / "data" / "windows720.npz"
FS, PRE, POST = 360, 360, 360
PACED = {"102", "104", "107", "217"}
# symbols with an AAMI class (N,L,R,e,j | A,a,J,S | V,E | F | /,f,Q); '/' and 'f' only occur in paced recs
KEEP = set("NLReJjAaSVEFQ/f")


def main():
    sos = sps.butter(4, [0.5, 40], btype="band", fs=FS, output="sos")
    recs = sorted(p.stem for p in DATA_DIR.glob("*.hea") if p.stem not in PACED)
    Xs, syms, rids, rs = [], [], [], []
    for r in recs:
        rec = wfdb.rdrecord(str(DATA_DIR / r)); ann = wfdb.rdann(str(DATA_DIR / r), "atr")
        mi = rec.sig_name.index("MLII"); oi = [i for i in range(rec.p_signal.shape[1]) if i != mi][0]
        sig = sps.sosfiltfilt(sos, rec.p_signal, axis=0)[:, [mi, oi]].astype(np.float32)
        sigp = np.pad(sig, ((PRE, POST), (0, 0)), mode="edge")
        n0 = len(syms)
        for s, sym in zip(ann.sample, ann.symbol):
            if sym not in KEEP:
                continue
            Xs.append(sigp[s:s + PRE + POST])   # padded index s == original s-PRE
            syms.append(sym); rids.append(r); rs.append(s)
        print(f"{r}: {len(syms)-n0} beats  leads={rec.sig_name[mi]},{rec.sig_name[oi]}")
    X = np.stack(Xs).astype(np.float16)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, symbols=np.array(syms), record_ids=np.array(rids), r_samples=np.array(rs, np.int32))
    print("saved", OUT, X.shape)


if __name__ == "__main__":
    main()
