"""Aggregate fidelity_C runs into markdown tables (DS2 metrics, mean ± std over seeds).

Usage: ../../.venv/bin/python summarise.py > results.md
Also writes results_table.csv (one row per variant).
Ensemble column: argmax of the seed-averaged DS2 probabilities.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from variants import GROUPS, VARIANTS  # noqa: E402

PAPER = {  # Farag 2023 Table 4, DS2
    "M1": dict(acc=96.48, f1=86.81, prec=82.55, se=92.87, seN=96.96, seS=85.30, seV=96.34, ppvS=58.06, ppvV=90.20),
    "M2": dict(acc=95.58, f1=84.79, prec=79.63, se=93.34, seN=95.86, seS=88.51, seV=95.65, ppvS=51.49, ppvV=87.93),
    "M3": dict(acc=94.42, f1=82.79, prec=78.49, se=93.53, seN=94.67, seS=95.59, seV=90.31, ppvS=43.85, ppvV=92.23),
    "M4": dict(acc=94.78, f1=83.34, prec=76.59, se=93.92, seN=94.91, seS=92.54, seV=94.32, ppvS=56.09, ppvV=74.10),
    "M5": dict(acc=92.90, f1=78.65, prec=72.31, se=91.86, seN=92.98, seS=88.13, seV=94.47, ppvS=39.19, ppvV=78.07),
    "M6": dict(acc=98.18, f1=92.17, prec=92.44, se=91.90, seN=99.10, seS=81.60, seV=95.00, ppvS=82.68, ppvV=95.63),
    "M7": dict(acc=97.94, f1=89.59, prec=78.77, se=91.70, seN=98.31, seS=81.33, seV=95.47, ppvS=69.72, ppvV=94.53),
    "M8": dict(acc=97.06, f1=68.18, prec=69.25, se=67.17, seN=99.14, seS=79.42, seV=90.13, ppvS=84.29, ppvV=94.99),
    "M9": dict(acc=96.93, f1=54.90, prec=56.42, se=53.70, seN=98.85, seS=74.52, seV=95.13, ppvS=90.78, ppvV=93.39),
}
COLS = [("acc", "accuracy"), ("ovr", "ovr_accuracy"), ("f1", "macro_f1"), ("prec", "macro_ppv"),
        ("se", "macro_se"), ("seN", "se_N"), ("seS", "se_S"), ("seV", "se_V"),
        ("ppvS", "ppv_S"), ("ppvV", "ppv_V")]


def mf(y, p, k):
    sys.path.insert(0, str(HERE.parents[1] / "HasanML" / "paper3"))
    from src.metrics import multiclass_metrics
    return multiclass_metrics(y, p, k=k)


def load(name):
    recs = [json.loads(f.read_text()) for f in sorted((HERE / "runs" / name).glob("seed*.json"))]
    return recs


def row(name):
    recs = load(name)
    if not recs:
        return None
    out = {"n": len(recs), "ep": np.mean([r["n_epochs"] for r in recs])}
    for short, key in COLS:
        v = np.array([r["metrics_test"].get(key, np.nan) for r in recs]) * 100
        out[short] = (np.nanmean(v), np.nanstd(v))
    # macro over N,S,V only (for 4/5-class models, paper averages incl. F/Q; keep both)
    # ensemble
    try:
        probs = [np.load(HERE / "runs" / name / f"seed{r['seed']}_proba.npz") for r in recs]
        P = np.mean([z["proba_test"] for z in probs], axis=0)
        y = probs[0]["y_test"]
        m = mf(y, P.argmax(1), P.shape[1])
        out["ens_acc"], out["ens_f1"], out["ens_seS"], out["ens_ppvS"] = (
            m["accuracy"] * 100, m["macro_f1"] * 100, m["se_S"] * 100, m["ppv_S"] * 100)
    except Exception:
        pass
    return out


def fmt(v):
    m, s = v
    return f"{m:.2f}±{s:.1f}" if np.isfinite(m) else "—"


def main():
    hdr = ("| variant | n | ep | Acc | OvR | macF1 | macPREC | macSe | Se N | Se S | Se V | PPV S | PPV V "
           "| ens Acc | ens F1 | ens SeS | ens PPVS |")
    sep = "|" + "---|" * 18
    rows_csv = []
    print("# fidelity_C results — DS2 (mean ± std over seeds; 'ens' = seed-averaged probabilities)\n")
    for g, names in GROUPS.items():
        print(f"\n## group `{g}`\n")
        print(hdr)
        print(sep)
        for n in names:
            r = row(n)
            if r is None:
                print(f"| {n} | 0 | | " + "| " * 15)
                continue
            cells = [fmt(r[c]) for c, _ in COLS]
            ens = [f"{r.get(k, float('nan')):.2f}" for k in ("ens_acc", "ens_f1", "ens_seS", "ens_ppvS")]
            print(f"| {n} | {r['n']} | {r['ep']:.0f} | " + " | ".join(cells) + " | " + " | ".join(ens) + " |")
            rows_csv.append([n, g, r["n"], r["ep"]] + [r[c][0] for c, _ in COLS] + [r[c][1] for c, _ in COLS]
                            + [r.get(k, np.nan) for k in ("ens_acc", "ens_f1", "ens_seS", "ens_ppvS")])
            if n in PAPER and g == "table4":
                p = PAPER[n]
                print(f"| ↳ paper {n} | | | " + " | ".join(
                    f"{p[c]:.2f}" if c in p else "—" for c, _ in COLS) + " | | | | |")
    import csv
    with open(HERE / "results_table.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "group", "n_seeds", "epochs"] + [c for c, _ in COLS]
                   + [c + "_std" for c, _ in COLS] + ["ens_acc", "ens_f1", "ens_seS", "ens_ppvS"])
        w.writerows(rows_csv)


if __name__ == "__main__":
    main()
