"""Shared pieces for the LSTM baseline: data/splits, model, metrics."""
from pathlib import Path
import numpy as np
import torch, torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PROC = ROOT / "datasets" / "processed"
WIN = HERE / "data" / "windows720.npz"

DS1 = "101 106 108 109 112 114 115 116 118 119 122 124 201 203 205 207 208 209 215 220 223 230".split()
DS2 = "100 103 105 111 113 117 121 123 200 202 210 212 213 214 219 221 222 228 231 232 233 234".split()
# Patient-held-out validation records (subset of DS1). Chosen a priori from DS1 class counts
# only (no DS2 information) so that every class of both schemes appears in validation:
#   115: N only | 207: L 1457, R 86, V 105(+E 105), A 107 | 220: N, A 94 | 223: N, V 473, A 72
VAL_RECORDS = ["115", "207", "220", "223"]
assert not set(VAL_RECORDS) & set(DS2) and set(VAL_RECORDS) <= set(DS1)

SCHEMES = {
    "nlrva": ["N", "L", "R", "V", "A"],
    "aami3": ["N", "S", "V"],
}
AAMI3 = {**{k: 0 for k in "NLRej"}, **{k: 1 for k in "AaJS"}, "V": 2, "E": 2}
NLRVA = {k: i for i, k in enumerate("NLRVA")}


def load_windows():
    d = np.load(WIN)
    return d["X"], d["symbols"], d["record_ids"], d["r_samples"]


def _map_feature_indices(rec, rs, idx_feat):
    """Map indices into mit_bih_features.npz to indices into windows720.npz via (record, R-sample)."""
    f = np.load(PROC / "mit_bih_features.npz", allow_pickle=True)
    key = {(a, int(b)): i for i, (a, b) in enumerate(zip(rec, rs))}
    out = np.array([key[(f["record_ids"][j], int(f["r_samples"][j]))] for j in idx_feat])
    return out, f["y"][idx_feat]


def get_split(scheme, split, symbols, rec, rs, seed_holdout=0):
    """Return dict of index arrays (into windows) + integer labels for train/val/test.

    split='random' : LightGBM split_random.npz beats; val = 10% stratified random beat hold-out of train
                     (that split is not patient-independent anyway).
    split='ds1ds2_heldout': train = DS1 minus VAL_RECORDS, val = VAL_RECORDS, test = DS2
    split='ds1ds2_refit'  : train = all DS1 (fixed epochs, no val), test = DS2
    """
    y_all = np.full(len(symbols), -1, np.int64)
    if scheme == "nlrva":
        sp = np.load(PROC / ("split_random.npz" if split == "random" else "split_ds1ds2.npz"), allow_pickle=True)
        tr, ytr_s = _map_feature_indices(rec, rs, sp["indices_train"])
        te, yte_s = _map_feature_indices(rec, rs, sp["indices_test"])
        for i, s in zip(np.r_[tr, te], np.r_[ytr_s, yte_s]):
            y_all[i] = NLRVA[s]
        assert (symbols[tr] == ytr_s).all() and (symbols[te] == yte_s).all()
    else:
        assert split != "random", "AAMI N/S/V only evaluated on DS1/DS2"
        y_all = np.array([AAMI3.get(s, -1) for s in symbols], np.int64)
        lab = y_all >= 0
        tr = np.where(lab & np.isin(rec, DS1))[0]
        te = np.where(lab & np.isin(rec, DS2))[0]
    if split == "random":
        from sklearn.model_selection import train_test_split
        tr, va = train_test_split(tr, test_size=0.10, stratify=y_all[tr], random_state=seed_holdout)
    elif split == "ds1ds2_heldout":
        m = np.isin(rec[tr], VAL_RECORDS)
        tr, va = tr[~m], tr[m]
    elif split == "ds1ds2_refit":
        va = np.array([], np.int64)
    else:
        raise ValueError(split)
    if split != "random":
        assert not np.isin(rec[np.r_[tr, va]], DS2).any() and np.isin(rec[te], DS2).all()
    return dict(train=tr, val=va, test=te, y=y_all)


class ConvBiLSTM(nn.Module):
    """Light strided-conv front-end (720 -> 180 steps) + 2-layer BiLSTM + mean/max pooling head.

    arch='conv'   : Conv(2->32,k7,s2)-BN-ReLU, Conv(32->64,k5,s2)-BN-ReLU  -> 180 x 64
    arch='pool'   : AvgPool(4) on raw 2-lead input (pure LSTM on 90 Hz signal) -> 180 x 2
    """
    def __init__(self, n_classes, arch="conv", hidden=64, layers=2, dropout=0.2, n_rr=0):
        super().__init__()
        self.n_rr = n_rr
        if arch == "conv":
            self.front = nn.Sequential(
                nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False), nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False), nn.BatchNorm1d(64), nn.ReLU())
            d_in = 64
        elif arch == "pool":
            self.front = nn.AvgPool1d(4)
            d_in = 2
        else:
            raise ValueError(arch)
        self.lstm = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True,
                            bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(4 * hidden + (16 if n_rr else 0), 64), nn.ReLU(),
                                  nn.Linear(64, n_classes))
        if n_rr:   # small MLP on log RR ratios, concatenated to the pooled LSTM features
            self.rr = nn.Sequential(nn.Linear(n_rr, 16), nn.ReLU())

    def forward(self, x, rr=None):   # x: (B, 720, 2); rr: (B, n_rr)
        h = self.front(x.transpose(1, 2)).transpose(1, 2)   # (B, T, C)
        o, _ = self.lstm(h)                                   # (B, T, 2H)
        z = [o.mean(1), o.amax(1)]
        if self.n_rr:
            z.append(self.rr(torch.log(rr.float().clamp_min(1e-2))))
        return self.head(torch.cat(z, 1))


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def normalise(xb):
    """Per-window, per-channel z-score (removes patient-specific gain/offset)."""
    xb = xb.float()
    return (xb - xb.mean(1, keepdim=True)) / (xb.std(1, keepdim=True) + 1e-3)


def metrics(y, p, classes):
    K = len(classes)
    cm = np.zeros((K, K), np.int64)
    np.add.at(cm, (y, p), 1)
    n = cm.sum(); tp = np.diag(cm).astype(float)
    fn = cm.sum(1) - tp; fp = cm.sum(0) - tp; tn = n - tp - fn - fp
    se = np.divide(tp, tp + fn, out=np.zeros(K), where=(tp + fn) > 0)
    ppv = np.divide(tp, tp + fp, out=np.zeros(K), where=(tp + fp) > 0)
    sp = np.divide(tn, tn + fp, out=np.zeros(K), where=(tn + fp) > 0)
    f1 = np.divide(2 * se * ppv, se + ppv, out=np.zeros(K), where=(se + ppv) > 0)
    return dict(
        acc=float(tp.sum() / n), ovr_acc=float(np.mean((tp + tn) / n)),
        macro_f1=float(f1.mean()), macro_prec=float(ppv.mean()), macro_se=float(se.mean()),
        per_class={c: dict(se=float(se[i]), ppv=float(ppv[i]), sp=float(sp[i]), f1=float(f1[i]),
                           support=int(cm[i].sum())) for i, c in enumerate(classes)},
        cm=cm.tolist(), classes=classes, n=int(n))


def macro_f1_present(y, p, K):
    """Macro F1 over classes present in y (used for validation monitoring)."""
    f = []
    for k in range(K):
        if (y == k).sum() == 0:
            continue
        tp = ((y == k) & (p == k)).sum(); fp = ((y != k) & (p == k)).sum(); fn = ((y == k) & (p != k)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))
