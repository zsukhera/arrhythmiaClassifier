"""
Step 3/4 — zero-shot evaluation on an external dataset with rules from out/rules.json.

Usage: python -I run_external.py ptbxl|chapman
Outputs: out/<ds>_records.pkl (per-record counts), out/<ds>_report.md, out/<ds>_<model>_metrics.csv
"""
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, LGBM_CLASSES, CNN_CLASSES, lgbm_predict_proba, cnn_predict_proba, cnn_threshold_predict  # noqa
from featurize import featurize  # noqa
from recmetrics import predict_conditions, evaluate, confusion, fmt, md  # noqa

TRUTH = {c: f"t_{c}" for c in ["L", "R", "V", "S", "N"]}


def _feat(x):
    try:
        return featurize(np.asarray(x, dtype=np.float64))
    except Exception as e:  # noqa
        return dict(error=repr(e)[:200])


def main(ds):
    rules = json.loads((OUT / "rules.json").read_text())
    mode = rules["cnn_rr_mode"]
    man = pd.read_csv(OUT / f"{ds}_manifest.csv")
    X = np.load(OUT / f"{ds}_leadII_360.npz")["X"]
    with ProcessPoolExecutor(4) as ex:
        F = list(ex.map(_feat, list(X), chunksize=32))
    print("featurised", len(F), "errors", sum("error" in f for f in F), flush=True)
    empty = dict(peaks=np.zeros(0), X_lgbm=np.zeros((0, 40), np.float32), X_ecg=np.zeros((0, 32, 1), np.float32),
                 X_rr_causal=np.zeros((0, 4), np.float32), X_rr_strip=np.zeros((0, 4), np.float32))
    F = [empty if "error" in f else f for f in F]
    Pl = lgbm_predict_proba(np.concatenate([f["X_lgbm"] for f in F]))
    Xe = np.concatenate([f["X_ecg"] for f in F])
    Pc = {m: cnn_predict_proba(Xe, np.concatenate([f[f"X_rr_{m}"] for f in F])) for m in ("causal", "strip")}
    rows, il, ic = [], 0, 0
    for f in F:
        nl, nc = len(f["X_lgbm"]), len(f["X_ecg"])
        yl = np.array(LGBM_CLASSES)[np.argmax(Pl[il:il + nl], 1)] if nl else np.array([])
        r = dict(n_det=len(f["peaks"]),
                 hr=60 * 360 / np.mean(np.diff(f["peaks"])) if len(f["peaks"]) > 1 else np.nan,
                 det_lgbm_n=nl, **{f"det_lgbm_cnt_{c}": int(np.sum(yl == c)) for c in LGBM_CLASSES})
        for m in ("causal", "strip"):
            P = Pc[m][ic:ic + nc]
            yc = np.array(CNN_CLASSES)[cnn_threshold_predict(P)] if nc else np.array([])
            r[f"det_cnn_{m}_n"] = nc
            for c in CNN_CLASSES:
                r[f"det_cnn_{m}_cnt_{c}"] = int(np.sum(yc == c))
        il += nl; ic += nc
        rows.append(r)
    df = pd.concat([man.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    df.to_pickle(OUT / f"{ds}_records.pkl")

    lines = [f"# {ds} zero-shot (rules fixed on MIT-BIH DS1; CNN RR fallback = {mode})",
             f"Cohort records: {len(df)}; " + ", ".join(f"{c}={int(df[v].sum())}" for c, v in TRUTH.items()),
             f"Detected beats per record: median {df.n_det.median():.0f}, records with <3 beats: {int((df.n_det < 3).sum())}"]
    for model, pref in (("lgbm", "det_lgbm"), ("cnn", f"det_cnn_{mode}"), ("cnn", "det_cnn_strip" if mode == "causal" else "det_cnn_causal")):
        pr = predict_conditions(df, model, pref, rules[model])
        ev = evaluate(df, model, pr, TRUTH)
        ev.insert(0, "model", pref)
        ev.to_csv(OUT / f"{ds}_{pref}_metrics.csv", index=False)
        lines.append(f"## {pref}\n" + fmt(ev))
        lines.append("Confusion (truth x predicted set):\n" + md(confusion(df, model, pr, TRUTH), index=True))
    # beat-level predicted class distribution on normal records (false-alarm source)
    n = df[df.t_N]
    dist = {c: n[f"det_lgbm_cnt_{c}"].sum() / max(1, n.det_lgbm_n.sum()) for c in LGBM_CLASSES}
    distc = {c: n[f"det_cnn_{mode}_cnt_{c}"].sum() / max(1, n[f"det_cnn_{mode}_n"].sum()) for c in CNN_CLASSES}
    lines.append("Beat-class distribution predicted on NORMAL records: LGBM "
                 + ", ".join(f"{k}={100*v:.1f}%" for k, v in dist.items())
                 + " | CNN " + ", ".join(f"{k}={100*v:.1f}%" for k, v in distc.items()))
    txt = "\n\n".join(lines)
    (OUT / f"{ds}_report.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1])
