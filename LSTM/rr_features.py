"""Causal RR-ratio side features aligned with data/windows720.npz -> data/rr.npz (rr: (n,4) float32).

RR intervals are computed over ALL beat annotations of a record (incl. beats not classified),
so no interval spans a dropped beat.  Features (dimensionless, patient-normalised):
  pre/local, post/local, pre/post, local/record
with local = mean of the previous (up to) 10 pre-RRs incl. the current one (causal),
record = median RR of the whole record. 'post' uses the next beat (0.0-1 s look-ahead,
same information that is already visible in the 2-s window).
"""
import numpy as np, wfdb
from lstm_common import HERE, ROOT, load_windows

BEAT = set("NLRBAaJSVrFejnE/fQ?")
X, sym, rec, rs = load_windows(); del X
out = np.zeros((len(sym), 4), np.float32)
for r in np.unique(rec):
    ann = wfdb.rdann(str(ROOT / "datasets/mitbih/mitdb" / r), "atr")
    b = np.array([s for s, y in zip(ann.sample, ann.symbol) if y in BEAT])
    rr = np.diff(b) / 360.0
    pre = np.r_[rr[0], rr]; post = np.r_[rr, rr[-1]]
    local = np.array([pre[max(0, i - 9):i + 1].mean() for i in range(len(b))])
    med = np.median(rr)
    feat = np.c_[pre / local, post / local, pre / post, local / med]
    pos = {s: i for i, s in enumerate(b)}
    m = np.where(rec == r)[0]
    out[m] = feat[[pos[s] for s in rs[m]]]
np.savez(HERE / "data" / "rr.npz", rr=out)
print("saved", out.shape, out.mean(0), out.std(0))
