"""
Task 5 (wrap-up):
 1. Candidate set = Optuna models (groupcv, groupcv_reg, groupcv_reg_noB2) + task3 variants + old params.
 2. Robustness check of the grouped-CV ranking with leave-one-record-out (22 folds) on DS1 for the
    main candidates (pooled OOF metrics).
 3. Pick the final model by grouped 5-fold CV (the pre-declared criterion), refit on all DS1,
    save models/lgbm_ds1ds2_groupcv_best.pkl, score on DS2.
 4. How well does grouped CV rank variants for DS2?  Spearman(gCV, DS2 macro-F1) across all variants.
Writes reports/lgbm-groupcv/task5_final.json
"""
import json, pickle
import numpy as np
from scipy.stats import spearmanr
from common import *

X, y, g = load_ds1()
F = FEATURE_NAMES
T3 = json.load(open(OUT_DIR / "task3_variants.json"))
E = json.load(open(OUT_DIR / "eval_ds2.json"))
T1 = json.load(open(OUT_DIR / "task1_baseline.json"))

cands = {}
cands["old_ds1ds2"] = dict(params=OLD_DS1DS2_PARAMS, sampler="smote", cw=None, feats=None,
                           gcv=T1["old_params/group5"]["macro_f1"], ds2=E["old_ds1ds2"])
for tag, key in [("_groupcv", "groupcv"), ("_groupcv_reg", "groupcv_reg"), ("_groupcv_reg_noB2", "groupcv_reg_noB2")]:
    b = pickle.load(open(MODEL_DIR / f"lgbm_ds1ds2{tag}.pkl", "rb"))
    ro = b.get("run_opts", {})
    cands[key] = dict(params=b["best_params"], sampler=ro.get("sampler", "smote"), cw=ro.get("class_weight"),
                      feats=ro.get("features"), gcv=b["cv_score"], ds2=E.get(key))
for k, v in T3.items():
    if isinstance(v, dict) and "cv" in v:
        cands["t3_" + k] = dict(params=v["params"], sampler=v["sampler"], cw=v["class_weight"],
                                feats=None if v["features"] is None else [F.index(f) for f in v["features"]],
                                gcv=v["cv"]["macro_f1"], ds2=v["ds2"])

# 4. ranking agreement
names = [k for k in cands if cands[k]["ds2"] is not None]
gc = np.array([cands[k]["gcv"] for k in names]); d2 = np.array([cands[k]["ds2"]["macro_f1"] for k in names])
rho, p = spearmanr(gc, d2)
print(f"Spearman(gCV macro-F1, DS2 macro-F1) over {len(names)} variants: rho={rho:.3f} p={p:.3f}")
rho_b, p_b = spearmanr(gc, [cands[k]["ds2"]["bin_f1"] for k in names])
print(f"Spearman(gCV macro-F1, DS2 binary F1): rho={rho_b:.3f} p={p_b:.3f}")
for k in sorted(names, key=lambda k: -cands[k]["gcv"]):
    print(f"  {k:<32} gCV={cands[k]['gcv']:.4f}  DS2 mF1={cands[k]['ds2']['macro_f1']:.4f}")

# 2. LOGO for main candidates
logo = logo_folds(g)
main = ["old_ds1ds2", "groupcv", "groupcv_reg", "groupcv_reg_noB2", "t3_d_drop_B2_morph", "t3_e_mcs1000", "t3_e_shallow", "t3_e_shallow_slow"]
L = {}
for k in main:
    c = cands[k]
    r = grouped_cv(X, y, logo, c["params"], c["sampler"], c["cw"], c["feats"])
    L[k] = strip(r)
    print(fmt(r, k + " [LOGO22]"), f"({r['seconds']:.0f}s)", flush=True)
    for rec in ["109", "207", "118", "124"]:
        m = g == rec
        L[k][f"rec_{rec}"] = {c_: int((r["oof_pred"][m] == c_).sum()) for c_ in LABELS}
    print("     L/R records predicted as:", {rec: L[k][f"rec_{rec}"] for rec in ["109", "207", "118", "124"]}, flush=True)
lg = np.array([L[k]["macro_f1"] for k in main]); gm = np.array([cands[k]["gcv"] for k in main])
print(f"Spearman(gCV5, LOGO22) over main candidates: {spearmanr(gm, lg)[0]:.3f}")
print(f"Spearman(LOGO22, DS2) over main candidates: {spearmanr(lg, [cands[k]['ds2']['macro_f1'] for k in main])[0]:.3f}")

# 3. final pick by grouped 5-fold CV
best = max(cands, key=lambda k: cands[k]["gcv"])
c = cands[best]
print(f"\nFinal pick by grouped 5-fold CV: {best} (gCV={c['gcv']:.4f})")
m = fit_model(X if c["feats"] is None else X[:, c["feats"]], y, c["params"], c["sampler"], c["cw"])
ds2 = eval_ds2(m, c["feats"])
print(fmt(ds2, "groupcv_best [DS2]"))
with open(MODEL_DIR / "lgbm_ds1ds2_groupcv_best.pkl", "wb") as f:
    pickle.dump({"model": m, "best_params": c["params"], "cv_score": c["gcv"], "split": "ds1ds2",
                 "run_opts": {"cv": "group", "sampler": c["sampler"], "class_weight": c["cw"],
                              "features": c["feats"], "source_variant": best,
                              "feature_names": None if c["feats"] is None else [F[i] for i in c["feats"]]}}, f)
E["groupcv_best"] = ds2
json.dump(E, open(OUT_DIR / "eval_ds2.json", "w"), indent=1, default=str)
json.dump({"spearman_gcv_ds2": [rho, p], "spearman_gcv_ds2_binf1": [rho_b, p_b],
           "variants": {k: {"gcv": cands[k]["gcv"], "ds2_macro_f1": cands[k]["ds2"]["macro_f1"],
                            "ds2_acc": cands[k]["ds2"]["acc"], "ds2_bin_f1": cands[k]["ds2"]["bin_f1"]} for k in names},
           "logo": L, "best": best, "best_ds2": ds2}, open(OUT_DIR / "task5_final.json", "w"), indent=1, default=str)
print("saved")
