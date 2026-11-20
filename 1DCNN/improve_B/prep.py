"""
Build cache/data.npz (DS1 with record ids + DS2), the patient-wise fold
assignment (cache/folds.json) and per-fold MF kernels (cache/kernel_*.npy).

Checks: DS1 arrays equal the official ds1_all.npz cache; full-DS1 kernel equals
the official kernel_mf.npy used by run_official_p1.py.
"""
import json
import numpy as np

from common import CACHE, EXPS, N_FOLDS, load_cfg
from src.data import DS1_TRAIN, DS1_VAL, DS2, SUB_CLASSES, task_labels
from src.run_experiment import process_record
from src.templates import compute_subclass_templates, templates_to_kernel

cfg = load_cfg()
DS1 = DS1_TRAIN + DS1_VAL

per_rec = {}
for rec in DS1 + DS2:
    r = process_record(rec, cfg)
    yt = task_labels(r["y_aami"], r["symbols"], "multiclass", 3, "aami")
    per_rec[rec] = dict(seg=r["segments"], rr=r["rr_features"], y=yt, sym=r["symbols"])
    print(rec, len(yt), np.bincount(yt[yt >= 0], minlength=3))


def stack(recs):
    Xe, Xr, y, g = [], [], [], []
    for rec in recs:
        d = per_rec[rec]; m = d["y"] >= 0
        Xe.append(d["seg"][m]); Xr.append(d["rr"][m]); y.append(d["y"][m])
        g.append(np.full(m.sum(), int(rec)))
    return (np.concatenate(Xe).astype(np.float32), np.concatenate(Xr).astype(np.float32),
            np.concatenate(y).astype(np.int64), np.concatenate(g))


Xe1, Xr1, y1, g1 = stack(DS1)
Xe2, Xr2, y2, g2 = stack(DS2)

ref = np.load(EXPS / ".cache" / "ds1_all.npz")
assert np.array_equal(ref["y"], y1) and np.allclose(ref["X_ecg"], Xe1) and np.allclose(ref["X_rr"], Xr1)
ref2 = np.load(EXPS / ".cache" / "ds2.npz")
assert np.array_equal(ref2["y"], y2) and np.allclose(ref2["X_ecg"], Xe2)
print("DS1/DS2 arrays match official caches")
np.savez_compressed(CACHE / "data.npz", Xe1=Xe1, Xr1=Xr1, y1=y1, g1=g1,
                    Xe2=Xe2, Xr2=Xr2, y2=y2, g2=g2)

# ---------- patient-wise folds: greedy balance of S, then V, beats
cnt = {rec: np.bincount(per_rec[rec]["y"][per_rec[rec]["y"] >= 0], minlength=3) for rec in DS1}
order = sorted(DS1, key=lambda r: (-cnt[r][1], -cnt[r][2], r))
folds = [[] for _ in range(N_FOLDS)]
fS = np.zeros(N_FOLDS); fV = np.zeros(N_FOLDS); fn = np.zeros(N_FOLDS)
for rec in order:
    if cnt[rec][1] >= 20:
        k = int(np.lexsort((fn, fS))[0])
    elif cnt[rec][2] >= 50:
        k = int(np.lexsort((fn, fV))[0])
    else:
        k = int(np.lexsort((fS, fn))[0])
    folds[k].append(rec); fS[k] += cnt[rec][1]; fV[k] += cnt[rec][2]; fn[k] += cnt[rec].sum()
for k, f in enumerate(folds):
    tot = sum(cnt[r] for r in f)
    print(f"fold {k}: recs={f} N/S/V={tot.tolist()}")
json.dump({"folds": folds, "counts": {r: cnt[r].tolist() for r in DS1}},
          open(CACHE / "folds.json", "w"), indent=1)


# ---------- kernels
def kernel_for(recs):
    segs = {s: [] for s in SUB_CLASSES}
    for rec in recs:
        d = per_rec[rec]
        for i, s in enumerate(d["sym"]):
            if s in segs:
                segs[s].append(d["seg"][i][:, 0])
    t = compute_subclass_templates(segs, 32, list(cfg["templates"]["sub_classes"]),
                                   min_beats=int(cfg["templates"]["min_beats"]),
                                   normalize_l2=bool(cfg["templates"]["normalize_l2"]),
                                   fallback_segments=segs)
    return templates_to_kernel(t, list(cfg["templates"]["sub_classes"]), n_channels=1)


kfull = kernel_for(DS1)
kref = np.load(EXPS / ".cache" / "kernel_mf.npy")
print("full kernel max |diff| vs official:", float(np.abs(kfull - kref).max()))
np.save(CACHE / "kernel_full.npy", kfull)
for k, f in enumerate(folds):
    np.save(CACHE / f"kernel_fold{k}.npy", kernel_for([r for r in DS1 if r not in f]))
print("done")
