import numpy as np, yaml, lightgbm as lgb
from src.run_experiment import build_dataset, load_config, _resolve
from src.data import DS1, DS2, ds1_split
cfg = load_config(_resolve("configs/default.yaml"))
cfg["preprocess"]["context_len"]=0
d1 = build_dataset(DS1, cfg); d2 = build_dataset(DS2, cfg)
for name, recs in [("farag-train", ds1_split("farag")[0]), ("bal-train", ds1_split("balanced")[0]), ("DS1-all", DS1)]:
    m = np.isin(d1["record_ids"], recs)
    X, y = d1["rr_feats"][m], d1["y"][m]
    keep = y <= 2
    clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, num_leaves=31, n_jobs=6, verbose=-1, class_weight="balanced")
    clf.fit(X[keep], y[keep])
    k2 = d2["y"] <= 2
    p = clf.predict(d2["rr_feats"][k2]); t = d2["y"][k2]
    se = [((p==c)&(t==c)).sum()/(t==c).sum() for c in range(3)]
    ppv = [((p==c)&(t==c)).sum()/max(1,(p==c).sum()) for c in range(3)]
    print(f"RR-only LGBM 3c trained on {name}: DS2 acc {(p==t).mean():.4f} Se N/S/V {np.round(se,3)} PPV {np.round(ppv,3)}")
for nm, d in [("DS1", d1), ("DS2", d2)]:
    for c in (0, 1):
        print(nm, "class", c, "median RR feats", np.round(np.median(d["rr_feats"][d["y"]==c], 0), 3))
