"""Record-level aggregation + metrics shared by MIT-BIH strips and external datasets."""
import numpy as np
import pandas as pd

from common import binary_metrics

K_GRID = [1, 2, 3, 4, 5]
P_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# model -> (prefix of count columns, conditions, predicted class label used for each condition)
MODELS = {
    "lgbm": dict(conds={"L": "L", "R": "R", "V": "V", "S": "A"}),
    "cnn": dict(conds={"V": "V", "S": "S"}),
}


def counts(df, model_prefix, cls):
    n = df[f"{model_prefix}_n"].to_numpy()
    c = df[f"{model_prefix}_cnt_{cls}"].to_numpy()
    return c, np.where(n > 0, c / np.maximum(n, 1), 0.0)


def choose_rule(truth, cnt, frac):
    best = None
    for k in K_GRID:
        for p in P_GRID:
            m = binary_metrics(truth, (cnt >= k) & (frac >= p))
            f1 = m["F1"] if np.isfinite(m["F1"]) else -1
            if best is None or f1 > best[0] + 1e-12:
                best = (f1, k, p, m)
    return dict(k=best[1], p=best[2], F1_DS1=best[0])


def predict_conditions(df, model, model_prefix, rules):
    out = {}
    for cond, cls in MODELS[model]["conds"].items():
        c, f = counts(df, model_prefix, cls)
        r = rules[cond]
        out[cond] = (c >= r["k"]) & (f >= r["p"])
    return pd.DataFrame(out, index=df.index)


def evaluate(df, model, preds, truth_cols):
    """preds: DataFrame of bool per condition. truth_cols: dict cond->column name (bool)."""
    rows = []
    conds = list(MODELS[model]["conds"])
    for cond in conds:
        m = binary_metrics(df[truth_cols[cond]].to_numpy(), preds[cond].to_numpy())
        rows.append(dict(condition=cond, **m))
    pred_N = ~preds[conds].any(axis=1).to_numpy()
    m = binary_metrics(df[truth_cols["N"]].to_numpy(), pred_N)
    rows.append(dict(condition="N (none fired)", **m))
    return pd.DataFrame(rows)


def confusion(df, model, preds, truth_cols):
    """Truth (records with exactly one target label, or N) x predicted set."""
    conds = list(MODELS[model]["conds"])
    allc = [c for c in ["L", "R", "V", "S"] if c in truth_cols]
    T = df[[truth_cols[c] for c in allc]].to_numpy().astype(int)
    single = T.sum(1) == 1
    tlab = np.where(df[truth_cols["N"]].to_numpy(), "N",
                    np.where(single, np.array(allc)[T.argmax(1)], "multi"))
    P = preds[conds].to_numpy().astype(int)
    plab = np.where(P.sum(1) == 0, "N", np.where(P.sum(1) == 1, np.array(conds)[P.argmax(1)], "multi"))
    rows = ["N"] + allc + ["multi"]
    cols = ["N"] + conds + ["multi"]
    tab = pd.crosstab(pd.Categorical(tlab, rows), pd.Categorical(plab, cols), dropna=False)
    tab.index.name = "truth"; tab.columns.name = "pred"
    return tab


def fmt(dfm):
    d = dfm.copy()
    for c in ["Se", "Sp", "PPV", "F1"]:
        d[c] = (100 * d[c]).map(lambda v: "—" if not np.isfinite(v) else f"{v:.1f}")
    return d[["condition", "n_pos", "n_neg", "TP", "FP", "FN", "TN", "Se", "Sp", "PPV", "F1"]].pipe(md)


def md(df, index=False):
    """Minimal markdown table (avoids a tabulate dependency)."""
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in d.iterrows():
        out.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(out)
