"""
Step 4b — fix everything on MIT-BIH DS1 strips, then report DS2 strips as in-domain reference.

  * CNN RR fallback (causal vs strip-mean): DS1 beat-level macro-F1 (detected peaks).
  * Aggregation rule per condition: positive iff count>=k AND fraction>=p, (k,p) maximising
    DS1 strip F1 (detected peaks).  Written to out/rules.json and never touched again.

Strip truth (from .atr beats inside the strip):
  L / R : >= 50 % of beats are L / R   (bundle-branch block is a persistent conduction state)
  V     : >= 1 beat in {V, E}
  S     : >= 1 beat in {A, a, J, S}  (AAMI S; matches PTB-XL PAC / Chapman APB)
  N     : every beat is 'N'
Cohort: strips that are N or carry at least one target condition.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, confusion_matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa
from recmetrics import md, choose_rule, predict_conditions, evaluate, confusion, fmt, MODELS, counts  # noqa

AAMI = {"N": "N", "L": "N", "R": "N", "e": "N", "j": "N", "A": "S", "a": "S", "J": "S", "S": "S",
        "V": "V", "E": "V"}


def strip_truth(df):
    n = df.n_ref.clip(lower=1)
    df["t_L"] = df.ref_L / n >= 0.5
    df["t_R"] = df.ref_R / n >= 0.5
    df["t_V"] = (df.ref_V + df.ref_E) >= 1
    df["t_S"] = (df.ref_A + df.ref_a + df.ref_J + df.ref_S) >= 1
    df["t_N"] = (df.ref_N == df.n_ref) & (df.n_ref > 0)
    df["cohort"] = df[["t_L", "t_R", "t_V", "t_S", "t_N"]].any(axis=1)
    return df


TRUTH = {c: f"t_{c}" for c in ["L", "R", "V", "S", "N"]}


def beat_report(b, model, labels):
    b = b[b.true.isin(labels)]
    y, p = b.true.to_numpy(), b.pred.to_numpy()
    se = recall_score(y, p, labels=labels, average=None, zero_division=0)
    ppv = precision_score(y, p, labels=labels, average=None, zero_division=0)
    return dict(n=len(y), acc=accuracy_score(y, p),
                macroF1=f1_score(y, p, labels=labels, average="macro", zero_division=0),
                **{f"Se_{c}": s for c, s in zip(labels, se)}, **{f"PPV_{c}": s for c, s in zip(labels, ppv)})


def main():
    S = strip_truth(pd.read_pickle(OUT / "mitbih_strips.pkl"))
    B = pd.read_pickle(OUT / "mitbih_beats.pkl")
    lines = []

    # ---- beat level -------------------------------------------------------
    rows = []
    for split in ("DS1", "DS2"):
        for src in ("atr", "det"):
            bl = B[(B.split == split) & (B.src == src) & (B.model == "lgbm")]
            rows.append(dict(split=split, src=src, model="lgbm", **beat_report(bl, "lgbm", ["N", "L", "R", "V", "A"])))
            for mode in ("causal", "strip"):
                bc = B[(B.split == split) & (B.src == src) & (B.model == f"cnn_{mode}")].copy()
                bc["true"] = bc.true.map(AAMI)
                rows.append(dict(split=split, src=src, model=f"cnn_{mode}", **beat_report(bc, "cnn", ["N", "S", "V"])))
    beat = pd.DataFrame(rows)
    beat.to_csv(OUT / "mitbih_beat_level.csv", index=False)
    lines.append("## Beat level on 10-s strips (matched beats only)\n" + md(beat.round(4)))

    d1 = beat[(beat.split == "DS1") & (beat.src == "det")]
    f_c = float(d1[d1.model == "cnn_causal"].macroF1.iloc[0]); f_s = float(d1[d1.model == "cnn_strip"].macroF1.iloc[0])
    rr_mode = "causal" if f_c >= f_s else "strip"
    lines.append(f"CNN RR fallback chosen on DS1 beat macro-F1: causal={f_c:.4f} strip={f_s:.4f} -> **{rr_mode}**")

    # ---- rules on DS1 strips ----------------------------------------------
    C = S[S.cohort]
    ds1, ds2 = C[C.split == "DS1"], C[C.split == "DS2"]
    prefixes = {"lgbm": "det_lgbm", "cnn": f"det_cnn_{rr_mode}"}
    rules = {"cnn_rr_mode": rr_mode}
    for model, pref in prefixes.items():
        rules[model] = {}
        for cond, cls in MODELS[model]["conds"].items():
            c, f = counts(ds1, pref, cls)
            rules[model][cond] = choose_rule(ds1[TRUTH[cond]].to_numpy(), c, f)
    (OUT / "rules.json").write_text(json.dumps(rules, indent=2))
    lines.append("## Rules (fixed on DS1 strips, detected peaks)\n```\n" + json.dumps(rules, indent=2) + "\n```")
    lines.append(f"Cohort strips: DS1={len(ds1)}, DS2={len(ds2)}; "
                 + ", ".join(f"{c}: DS1 {int(ds1[TRUTH[c]].sum())} / DS2 {int(ds2[TRUTH[c]].sum())}" for c in TRUTH))

    # ---- strip-level metrics ----------------------------------------------
    for split, d in (("DS1 (rule-fitting set, in-sample)", ds1), ("DS2 (in-domain reference)", ds2)):
        for model in ("lgbm", "cnn"):
            for src in ("det", "atr"):
                pref = f"{src}_lgbm" if model == "lgbm" else f"{src}_cnn_{rr_mode}"
                pr = predict_conditions(d, model, pref, rules[model])
                ev = evaluate(d, model, pr, TRUTH)
                ev.insert(0, "model", model); ev.insert(1, "src", src)
                ev.to_csv(OUT / f"mitbih_{split[:3]}_{model}_{src}_strip_metrics.csv", index=False)
                lines.append(f"### {split} — {model} — peaks={src}\n" + fmt(ev))
                if src == "det":
                    lines.append("Confusion (truth x predicted set):\n" + md(confusion(d, model, pr, TRUTH), index=True))
    txt = "\n\n".join(lines)
    (OUT / "mitbih_strip_report.md").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
