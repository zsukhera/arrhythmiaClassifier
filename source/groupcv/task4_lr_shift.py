"""
Task 4: why do LBBB (L) / RBBB (R) fail on DS1 -> DS2?

Outputs reports/lgbm-groupcv/task4_lr_shift.json + printed tables:
 1. which records hold L / R beats in DS1 and DS2
 2. per-feature shift between DS1 and DS2 for the same class:
      standardised mean difference (SMD, pooled SD) and KS statistic,
      and univariate AUC "DS1-L vs DS2-L" (0.5 = no shift)
 3. per-feature univariate separability of L-vs-N and R-vs-N computed
    separately in DS1 and DS2 -> does the direction/strength carry over?
 4. per-record medians of the most important bundle-branch features
 5. transfer of feature subsets: L-vs-N and R-vs-N LightGBM trained on DS1
    with one feature group (or the 'consistent' subset), AUC on DS2
    (one-vs-N on the DS2 beats of class c and N)
 6. where DS2 L/R beats go with the existing model (per record)
Uses at most GROUPCV_NJOBS threads.
"""
import json, pickle
import numpy as np
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score
from common import *

X1, y1, g1 = load_ds1()
X2, y2, g2 = load_ds2()
F = FEATURE_NAMES
out = {}

# 1 ── records
rec = {}
for c in ["L", "R"]:
    rec[c] = {
        "DS1": {str(r): int(((g1 == r) & (y1 == c)).sum()) for r in np.unique(g1[y1 == c])},
        "DS2": {str(r): int(((g2 == r) & (y2 == c)).sum()) for r in np.unique(g2[y2 == c])},
    }
    print(c, rec[c])
out["records"] = rec


def uauc(a, b):
    """univariate AUC of a feature separating sample b (label 1) from a (label 0)."""
    v = np.r_[a, b]; t = np.r_[np.zeros(len(a)), np.ones(len(b))]
    ok = np.isfinite(v)
    return float(roc_auc_score(t[ok], v[ok]))


# 2+3 ── per-feature shift and separability
rows = []
for j, f in enumerate(F):
    r = {"feature": f}
    for c in ["N", "L", "R", "V", "A"]:
        a, b = X1[y1 == c, j], X2[y2 == c, j]
        sd = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2) + 1e-12
        r[f"smd_{c}"] = float((np.nanmean(b) - np.nanmean(a)) / sd)
        r[f"ks_{c}"] = float(ks_2samp(a[np.isfinite(a)], b[np.isfinite(b)]).statistic)
    for c in ["L", "R"]:
        r[f"auc_{c}vsN_DS1"] = uauc(X1[y1 == "N", j], X1[y1 == c, j])
        r[f"auc_{c}vsN_DS2"] = uauc(X2[y2 == "N", j], X2[y2 == c, j])
        # how far DS2-c sits from DS1-c vs from DS1-N (in DS1-N SD units)
        sdN = np.nanstd(X1[y1 == "N", j]) + 1e-12
        r[f"{c}2_to_{c}1"] = float(abs(np.nanmedian(X2[y2 == c, j]) - np.nanmedian(X1[y1 == c, j])) / sdN)
        r[f"{c}2_to_N1"] = float(abs(np.nanmedian(X2[y2 == c, j]) - np.nanmedian(X1[y1 == "N", j])) / sdN)
    rows.append(r)
out["per_feature"] = rows

for c in ["L", "R"]:
    print(f"\n=== {c}: per-feature shift DS1->DS2 and L/R-vs-N separability (sorted by DS1 separability) ===")
    print(f"{'feature':<22}{'SMD_'+c:>8}{'KS_'+c:>7}{'SMD_N':>7}{'AUC1':>7}{'AUC2':>7}  carry  {c}2->{c}1 {c}2->N1")
    for r in sorted(rows, key=lambda r: -abs(r[f"auc_{c}vsN_DS1"] - 0.5)):
        a1, a2 = r[f"auc_{c}vsN_DS1"], r[f"auc_{c}vsN_DS2"]
        carry = "yes" if (a1 - .5) * (a2 - .5) > 0 and min(abs(a1 - .5), abs(a2 - .5)) > 0.15 else ("FLIP" if (a1 - .5) * (a2 - .5) < 0 and min(abs(a1-.5), abs(a2-.5)) > 0.1 else "")
        r[f"carry_{c}"] = carry
        print(f"{r['feature']:<22}{r['smd_'+c]:>8.2f}{r['ks_'+c]:>7.2f}{r['smd_N']:>7.2f}{a1:>7.3f}{a2:>7.3f}  {carry:<5}  {r[f'{c}2_to_{c}1']:>6.2f} {r[f'{c}2_to_N1']:>6.2f}")

# 4 ── per-record medians of key features
key = ["fid_qrs_dur_ms", "fid_rs_ratio", "fid_s_amp", "fid_q_amp", "fid_t_amp",
       "stat_skewness", "stat_kurtosis", "dwt_cD3_energy", "dwt_cA4_energy", "hrv_sdnn_ms", "nl_sampen"]
ki = [F.index(k) for k in key]
per_rec = {}
print("\n=== per-record medians (dominant-class beats) ===")
print(f"{'set':<4}{'rec':<5}{'cls':<4}{'n':>6} " + " ".join(f"{k[:12]:>12}" for k in key))
for ds, X, y, g in (("DS1", X1, y1, g1), ("DS2", X2, y2, g2)):
    for r in np.unique(g):
        for c in ["L", "R", "N"]:
            m = (g == r) & (y == c)
            if m.sum() < 200 or (c == "N" and not ((g == r) & np.isin(y, ["L", "R"])).any() and r not in ("100", "101", "103", "112", "115", "122")):
                continue
            med = np.nanmedian(X[m][:, ki], axis=0)
            per_rec[f"{ds}_{r}_{c}"] = dict(zip(key, map(float, med)))
            print(f"{ds:<4}{r:<5}{c:<4}{m.sum():>6} " + " ".join(f"{v:>12.3f}" for v in med))
out["per_record_medians"] = per_rec

# 5 ── subset transfer: c-vs-N binary LGBM trained on DS1, AUC on DS2 (and LORO within DS1)
P = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=50,
         subsample=0.8, colsample_bytree=0.8)
subsets = dict(FEATURE_GROUPS)
subsets["all40"] = list(range(40))
for c in ["L", "R"]:
    cons = [F.index(r["feature"]) for r in rows if r.get(f"carry_{c}") == "yes"]
    subsets[f"carry_{c}"] = cons
transfer = {}
print("\n=== subset transfer: binary c-vs-N, train DS1 -> AUC on DS2 (c vs N beats) ===")
for c in ["L", "R"]:
    m1 = np.isin(y1, ["N", c]); m2 = np.isin(y2, ["N", c])
    t1 = (y1[m1] == c).astype(int); t2 = (y2[m2] == c).astype(int)
    for name, idx in subsets.items():
        if not idx or (name.startswith("carry_") and name != f"carry_{c}"):
            continue
        clf = make_clf(P, class_weight="balanced")
        clf.fit(X1[m1][:, idx], t1)
        p2 = clf.predict_proba(X2[m2][:, idx])[:, 1]
        auc2 = float(roc_auc_score(t2, p2))
        # recall at the DS1-chosen 0.5 threshold, and per DS2 record recall
        rec2 = float((p2[t2 == 1] > 0.5).mean())
        fpr2 = float((p2[t2 == 0] > 0.5).mean())
        # leave-one-c-record-out within DS1 (train on the other c record(s))
        loro = []
        for r in rec[c]["DS1"]:
            if rec[c]["DS1"][r] < 200:
                continue
            tr = m1 & (g1 != r); te = m1 & (g1 == r) | (m1 & (y1 == "N") & np.isin(g1, ["101", "112", "115", "122"]))
            cl2 = make_clf(P, class_weight="balanced").fit(X1[tr][:, idx], (y1[tr] == c).astype(int))
            pt = cl2.predict_proba(X1[te][:, idx])[:, 1]
            loro.append((r, float(roc_auc_score((y1[te] == c).astype(int), pt))))
        transfer[f"{c}/{name}"] = {"n_feat": len(idx), "auc_ds2": auc2, "recall_ds2@0.5": rec2,
                                   "fpr_ds2@0.5": fpr2, "loro_ds1": loro}
        print(f"  {c} {name:<12} ({len(idx):>2} f)  DS2 AUC={auc2:.3f}  rec@.5={rec2:.3f}  FPR@.5={fpr2:.3f}  "
              f"DS1 leave-one-{c}-record-out AUC: " + ", ".join(f"{r}:{a:.3f}" for r, a in loro))
out["transfer"] = transfer
out["carry_features"] = {c: [F[i] for i in subsets[f"carry_{c}"]] for c in ["L", "R"]}

# domain classifier: can DS1-c be told apart from DS2-c?  (AUC≈1 = disjoint distributions)
dom = {}
for c in ["N", "L", "R", "V", "A"]:
    from sklearn.model_selection import cross_val_predict, StratifiedKFold
    Xc = np.r_[X1[y1 == c], X2[y2 == c]]; tc = np.r_[np.zeros((y1 == c).sum()), np.ones((y2 == c).sum())]
    p = cross_val_predict(make_clf(P), Xc, tc, cv=StratifiedKFold(3, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    dom[c] = float(roc_auc_score(tc, p))
print("\n=== domain classifier AUC (DS1-class vs DS2-class, 3-fold): 0.5 = same distribution ===", dom)
out["domain_auc"] = dom

# 6 ── where do DS2 L/R beats go with the existing model?
with open(MODEL_DIR / "lgbm_ds1ds2.pkl", "rb") as f:
    old = pickle.load(f)["model"]
old.set_params(n_jobs=N_JOBS)
pred2 = old.predict(X2)
where = {}
print("\n=== existing lgbm_ds1ds2.pkl: predictions for DS2 L/R beats, per record ===")
for c in ["L", "R"]:
    for r in rec[c]["DS2"]:
        m = (g2 == r) & (y2 == c)
        d = {k: int((pred2[m] == k).sum()) for k in LABELS}
        where[f"{c}_{r}"] = d
        print(f"  {c} rec {r} (n={m.sum()}): {d}")
out["old_model_ds2_LR_predictions"] = where
json.dump(out, open(OUT_DIR / "task4_lr_shift.json", "w"), indent=1)
print("saved")
