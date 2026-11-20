"""
Task 3: cheap variants, selected ONLY by grouped CV on DS1 (pooled OOF macro-F1).
Every variant is also refit on full DS1 and scored on DS2 for reporting (not selection).

Usage: task3_variants.py BASE_PARAMS_JSON   (json with "best_params", e.g. the
       models/checkpoints/lgbm_ds1ds2_groupcv_search_complete.json)
Steps
  a) imbalance handling: SMOTE / class_weight=balanced / nothing
  b) patient-identity score per feature: eta^2 of record id on DS1 N beats
     (share of variance explained by which patient the beat came from)
  c) grouped permutation importance (pooled OOF macro-F1 drop, 3 repeats)
  d) drop each feature group; drop features with high identity AND <=0 importance
  e) fixed stronger-regularisation configs
Results appended to reports/lgbm-groupcv/task3_variants.json (resumable: done keys skipped).
"""
import json, sys, pickle
import numpy as np
from common import *

base = json.load(open(sys.argv[1]))["best_params"]
X, y, g = load_ds1()
folds = group_folds(y, g)
F = FEATURE_NAMES
path = OUT_DIR / "task3_variants.json"
R = json.load(open(path)) if path.exists() else {}
R["base_params"] = base


def run(name, params, sampler="smote", cw=None, feats=None, ds2=True):
    if name in R and "cv" in R[name]:
        print("skip", name, flush=True); return R[name]
    r = grouped_cv(X, y, folds, params, sampler, cw, feats, return_models=False)
    entry = {"params": params, "sampler": sampler, "class_weight": cw,
             "features": None if feats is None else [F[i] for i in feats], "cv": strip(r)}
    if ds2:
        m = fit_model(X if feats is None else X[:, feats], y, params, sampler, cw)
        entry["ds2"] = eval_ds2(m, feats)
    R[name] = entry
    json.dump(R, open(path, "w"), indent=1)
    print(fmt(r, name + " [gCV]"), f"({r['seconds']:.0f}s)", flush=True)
    if ds2:
        print(fmt(entry["ds2"], name + " [DS2]"), flush=True)
    return entry

# a) imbalance
run("a_smote", base, "smote")
run("a_balanced", base, "none", "balanced")
run("a_none", base, "none")
best_imb = max(["a_smote", "a_balanced", "a_none"], key=lambda k: R[k]["cv"]["macro_f1"])
samp, cw = R[best_imb]["sampler"], R[best_imb]["class_weight"]
print("best imbalance handling by gCV:", best_imb, flush=True)

# b) identity score: eta^2 of record on N beats
if "identity" not in R:
    m = y == "N"; XN, gN = X[m], g[m]
    eta = {}
    for j, f in enumerate(F):
        v = XN[:, j]; ok = np.isfinite(v); v, gg = v[ok], gN[ok]
        tot = ((v - v.mean()) ** 2).sum() + 1e-12
        btw = sum((gg == r).sum() * (v[gg == r].mean() - v.mean()) ** 2 for r in np.unique(gg))
        eta[f] = float(btw / tot)
    R["identity"] = eta
    json.dump(R, open(path, "w"), indent=1)
print("eta^2 (record identity, N beats):", sorted(R["identity"].items(), key=lambda t: -t[1])[:15], flush=True)

# c) grouped permutation importance with the fold models of the chosen imbalance setting
if "perm" not in R:
    r = grouped_cv(X, y, folds, base, samp, cw, return_models=True)
    base_f1 = r["macro_f1"]; rng = np.random.default_rng(SEED); perm = {}
    for j, f in enumerate(F):
        drops = []
        for rep in range(3):
            pred = np.empty(len(y), dtype=object)
            for (tr, va), mdl in zip(folds, r["models"]):
                Xv = X[va].copy(); Xv[:, j] = rng.permutation(Xv[:, j])
                pred[va] = mdl.predict(Xv)
            drops.append(base_f1 - f1_score(y, pred.astype(str), average="macro"))
        perm[f] = float(np.mean(drops))
        print(f"  perm {f:<22} dF1={perm[f]:+.4f}", flush=True)
    R["perm"] = perm; R["perm_base_f1"] = base_f1
    json.dump(R, open(path, "w"), indent=1)

# d) feature-group drops and identity-feature drops
for gname, idx in FEATURE_GROUPS.items():
    keep = [i for i in range(40) if i not in idx]
    run(f"d_drop_{gname}", base, samp, cw, keep)
ident_sorted = sorted(F, key=lambda f: -R["identity"][f])
harm = [f for f in F if R["perm"][f] <= 0]
cand = [f for f in ident_sorted if R["identity"][f] >= 0.3 and R["perm"][f] <= 0.002]
R["drop_candidates"] = cand
for name, drop in [("d_drop_identity_lowimp", cand),
                   ("d_drop_negimp", harm),
                   ("d_drop_hrv_nl_and_identity", sorted(set(cand) | {F[i] for i in FEATURE_GROUPS["D1_hrv"] + FEATURE_GROUPS["D2_nonlin"]}))]:
    if drop:
        run(name, base, samp, cw, [i for i, f in enumerate(F) if f not in drop])

# e) stronger regularisation on the best feature set so far
d_keys = [k for k in R if k.startswith(("a_", "d_")) and isinstance(R[k], dict) and "cv" in R[k]]
best_d = max(d_keys, key=lambda k: R[k]["cv"]["macro_f1"])
fs = R[best_d]["features"]; feats = None if fs is None else [F.index(f) for f in fs]
s2, cw2 = R[best_d]["sampler"], R[best_d]["class_weight"]
print("best so far by gCV:", best_d, flush=True)
regs = {
    "e_mcs200": dict(min_child_samples=200),
    "e_mcs1000": dict(min_child_samples=1000),
    "e_lambda10_ff05": dict(reg_lambda=10.0, colsample_bytree=0.5),
    "e_shallow": dict(max_depth=4, num_leaves=15, n_estimators=400, min_child_samples=200, reg_lambda=10.0, colsample_bytree=0.5),
    "e_shallow_slow": dict(max_depth=3, num_leaves=8, n_estimators=800, learning_rate=0.03, min_child_samples=500, reg_lambda=30.0, colsample_bytree=0.4),
}
for name, upd in regs.items():
    run(name, {**base, **upd}, s2, cw2, feats)
all_keys = [k for k in R if isinstance(R[k], dict) and "cv" in R[k]]
best = max(all_keys, key=lambda k: R[k]["cv"]["macro_f1"])
R["best_by_gcv"] = best
json.dump(R, open(path, "w"), indent=1)
print("\nBEST variant by grouped CV:", best, R[best]["cv"]["macro_f1"], flush=True)
print("\n=== summary (gCV pooled macro-F1 | DS2 macro-F1, acc, binF1) ===")
for k in all_keys:
    c, d = R[k]["cv"], R[k].get("ds2", {})
    print(f"  {k:<30} gCV={c['macro_f1']:.4f}  DS2 mF1={d.get('macro_f1', float('nan')):.4f} acc={d.get('acc', float('nan')):.4f} binF1={d.get('bin_f1', float('nan')):.4f}")
