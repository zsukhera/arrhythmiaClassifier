"""
B3/B5 analysis: ensemble DS2 probabilities, calibrate per-class logit offsets on
DS1-only data, evaluate chosen operating points on DS2, write tables/plots.

Calibration data per scheme
  rand : seed-averaged probabilities on the 15 % random-beat hold-out (leaks patients)
  gkf  : out-of-fold DS1 probabilities (each beat predicted by the fold model that did
         not see its record), averaged over seeds
DS2 ensemble: mean of DS2 probabilities over all models (seeds x folds).

Usage: python analyze.py   (scans runs/ and writes results/)
"""
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import (RUNS, RES, EXPS, PAPER3, K, GRID, load_data, evaluate, choose_points,
                    offset_grid_cms, grid_metrics, fmt_row, TABLE_HDR, TABLE_SEP, PAPER_ROW,
                    SE_S_TARGET, SE_V_TARGET, metrics_from_cm, confusion)

D = load_data()
y1, y2 = D["y1"], D["y2"]


def collect():
    groups = defaultdict(lambda: defaultdict(dict))  # cfg -> seed -> fold -> path
    for f in glob.glob(str(RUNS / "*.npz")):
        m = re.match(r"(rand|gkf|refit)_p([\d.]+)_s(\d+)(?:_f(\d))?(?:_es(\w+))?\.npz", Path(f).name)
        sch, p, s, fo, es = m.groups()
        cfg = f"{sch}_p{p}" + (f"_es{es}" if es else "")
        groups[cfg][int(s)][int(fo) if fo else -1] = f
    return groups


def assemble(cfg, seeds_map, seeds=None):
    """Return (calib_y, calib_proba, ds2_proba, per-seed ds2 probs, epochs) for complete seeds."""
    sch = cfg.split("_")[0]
    nf = 5 if sch == "gkf" else 1
    seeds = sorted(s for s, fm in seeds_map.items() if len(fm) == nf) if seeds is None else seeds
    cal, ds2, eps, folds2 = [], [], [], []
    if sch == "refit":   # calibration = OOF of the same seed's gkf fold models
        seeds = sorted(s for s in seeds_map if s in GKF.get(cfg.replace("refit", "gkf"), {})
                       and len(GKF[cfg.replace("refit", "gkf")][s]) == 5)
    if not seeds:
        return [], None, None, None, None, None, None, None
    for s in seeds:
        fm = seeds_map[s]
        if sch == "refit":
            oof = np.zeros((len(y1), K), np.float32)
            for f in range(5):
                z = np.load(GKF[cfg.replace("refit", "gkf")][s][f]); oof[z["val_idx"]] = z["proba_val"]
            z = np.load(fm[-1]); cy = y1
            cal.append(oof); ds2.append(z["proba_ds2"]); eps.append(int(z["best_epoch"])); folds2.append([z["proba_ds2"]])
        elif sch == "rand":
            z = np.load(fm[-1]); cy = y1[z["val_idx"]]
            cal.append(z["proba_val"]); ds2.append(z["proba_ds2"]); eps.append(int(z["best_epoch"]))
            folds2.append([z["proba_ds2"]])
        else:
            oof = np.zeros((len(y1), K), np.float32); p2 = []
            for f in range(5):
                z = np.load(fm[f]); oof[z["val_idx"]] = z["proba_val"]
                p2.append(z["proba_ds2"]); eps.append(int(z["best_epoch"]))
            cal.append(oof); ds2.append(np.mean(p2, 0)); cy = y1; folds2.append(p2)
    return seeds, cy, np.mean(cal, 0), np.mean(ds2, 0), ds2, cal, eps, folds2



def official():
    import pickle
    rng = np.random.default_rng(42); idx = rng.permutation(len(y1)); vi = idx[:int(len(y1) * .15)]
    pv, pt = [], []
    for s in range(10):
        r = pickle.load(open(PAPER3 / "results" / "cache" / f"P1_fullDS1_seed{s}.pkl", "rb"))
        pv.append(r["proba_val"]); pt.append(r["proba_test"])
    return y1[vi], np.mean(pv, 0), np.mean(pt, 0), pt


GRID_SEED = np.round(np.arange(-6.0, 6.0001, 0.2), 2)
POINTS = ("argmax", "maxF1", "maxAcc", "SeConstr")
KEYS = ("accuracy", "ovr_accuracy", "macro_f1", "macro_ppv", "macro_se",
        "se_N", "ppv_N", "se_S", "ppv_S", "se_V", "ppv_V")


def calibrate(cy, cp, grid):
    pts, _ = choose_points(cy, cp, grid)
    out = {"argmax": (0.0, 0.0)}
    for nm in POINTS[1:]:
        out[nm] = pts[nm]
    return out, pts["SeConstr_feasible"]


def mean_row(name, ms):
    mu = {k: float(np.mean([m[k] for m in ms])) for k in KEYS}
    sd = {k: float(np.std([m[k] for m in ms])) for k in KEYS}
    return fmt_row(name, mu, f" ±{100*sd['accuracy']:.2f} / ±{100*sd['macro_f1']:.2f} / ±{100*sd['se_S']:.2f} |"), mu, sd


GKF = {}


def main():
    groups = collect()
    GKF.update({k: v for k, v in groups.items() if k.startswith("gkf")})
    summary = {}
    entries = []
    cy, cp, d2, per = official()
    pv = []
    import pickle
    for s in range(10):
        pv.append(pickle.load(open(PAPER3 / "results" / "cache" / f"P1_fullDS1_seed{s}.pkl", "rb"))["proba_val"])
    entries.append(("official_P1 (rand,p=1)", cy, cp, d2, per, pv, list(range(10)), [], [[p] for p in per]))
    for cfg in sorted(groups):
        seeds, cy, cp, d2, per, cal, eps, f2 = assemble(cfg, groups[cfg])
        if seeds:
            entries.append((cfg, cy, cp, d2, per, cal, seeds, eps, f2))

    sel = ["## DS1-only model-selection table (calibration data = OOF for gkf, random hold-out for rand)", "",
           "| Config | n seeds | best epoch median | DS1-cal macro F1 @argmax | DS1-cal macro F1 @maxF1 | "
           "DS1-cal acc @maxAcc | SeConstr feasible on DS1 | DS1-cal Se_S/Se_V @argmax |", "|" + "---|" * 8]
    seed_rows, ens_rows, data_for_plots = [], [], {}
    single_rows = []
    for name, cy, cp, d2, per, cal, seeds, eps, f2 in entries:
        print("analysing", name, flush=True)
        res = {"seeds": seeds, "best_epochs": eps}
        # ---- ensemble: calibrate on seed-averaged calibration probs
        pts, feas = calibrate(cy, cp, GRID)
        res["ens_points"] = pts; res["ens_feasible"] = feas
        cal_m = {nm: evaluate(cy, cp, *pts[nm]) for nm in POINTS}
        sel.append(f"| {name} | {len(seeds)} | {np.median(eps) if eps else '—'} | "
                   f"{100*cal_m['argmax']['macro_f1']:.2f} | {100*cal_m['maxF1']['macro_f1']:.2f} | "
                   f"{100*cal_m['maxAcc']['accuracy']:.2f} | {feas} | {100*cal_m['argmax']['se_S']:.1f}/{100*cal_m['argmax']['se_V']:.1f} |")
        for nm in POINTS:
            md = evaluate(y2, d2, *pts[nm]); res[f"ens_{nm}"] = md; res[f"ens_cal_{nm}"] = cal_m[nm]
            ens_rows.append(fmt_row(f"{name} ens [{nm} {pts[nm][0]:+.1f},{pts[nm][1]:+.1f}]", md,
                                    f" {100*cal_m[nm]['se_S']:.1f}/{100*cal_m[nm]['se_V']:.1f}/{100*cal_m[nm]['macro_f1']:.1f} |"))
        # ---- per seed: each seed calibrated on its own calibration probs (paper-comparable seed mean)
        per_pt = {nm: [] for nm in POINTS}
        sing = {nm: [] for nm in POINTS}
        res["seed_points"] = []
        for i in range(len(seeds)):
            spts, _ = calibrate(cy, cal[i], GRID_SEED)
            res["seed_points"].append(spts)
            for nm in POINTS:
                per_pt[nm].append(evaluate(y2, per[i], *spts[nm]))
                sing[nm] += [evaluate(y2, q, *spts[nm]) for q in f2[i]]
        for nm in POINTS:
            row, mu, sd = mean_row(f"{name} seed-mean [{nm}]", per_pt[nm])
            seed_rows.append(row); res[f"seed_{nm}_mean"] = mu; res[f"seed_{nm}_std"] = sd
            row, mu, sd = mean_row(f"{name} single-model mean (n={len(sing[nm])}) [{nm}]", sing[nm])
            single_rows.append(row); res[f"single_{nm}_mean"] = mu; res[f"single_{nm}_std"] = sd
        summary[name] = res
        data_for_plots[name] = (cy, cp, d2)

    lines = ["# farag-improve results (auto-generated by analyze.py)", "",
             "Operating points (all chosen on DS1 only): argmax; maxF1 = max macro-F1; maxAcc = max plain acc "
             "(identical to max OvR acc); SeConstr = max acc s.t. Se_S >= "
             f"{100*SE_S_TARGET:.1f} and Se_V >= {100*SE_V_TARGET:.1f} on DS1 calibration data "
             "(Model 6 sensitivities; if infeasible on DS1, falls back to max macro Se = balanced accuracy).", "",
             "Decision rule: argmax(log P + [0, bS, bV]).", "", *sel, "",
             "## DS2, mean of single seeds (each seed calibrated on its own DS1 data) — paper-comparable", "",
             TABLE_HDR + " ± std Acc / F1 / Se_S |", TABLE_SEP + "---|", PAPER_ROW.replace("|\n", "| |\n") + " |",
             *seed_rows, "",
             "## DS2, mean over individual models (gkf: every fold model separately, with its seed's offsets)", "",
             "Closest analogue of the paper's 'mean of 10 runs' of single networks.", "",
             TABLE_HDR + " ± std Acc / F1 / Se_S |", TABLE_SEP + "---|", PAPER_ROW.replace("|\n", "| |\n") + " |",
             *single_rows, "",
             "## DS2, seed-ensemble (mean probabilities) at ensemble-calibrated points", "",
             "Last column: Se_S / Se_V / macro F1 of the same point on the DS1 calibration data.", "",
             TABLE_HDR + " DS1 cal |", TABLE_SEP + "---|", PAPER_ROW.replace("|\n", "| |\n") + " |", *ens_rows]
    (RES / "results.md").write_text("\n".join(lines) + "\n")
    json.dump(summary, open(RES / "summary.json", "w"), indent=1, default=float)
    print("\n".join(lines))
    plots(data_for_plots)


def plots(dp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    want = [n for n in dp if n.startswith("official") or n.startswith("gkf") or n.startswith("refit") or n == "rand_p0"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for n in want:
        cy, cp, d2 = dp[n]
        # trade-off curve: sweep bS; for each bS choose bV maximising DS1-calib accuracy
        gmc = grid_metrics(offset_grid_cms(cy, cp))
        gmd = grid_metrics(offset_grid_cms(y2, d2))
        jbest = np.argmax(gmc["acc"], axis=1)
        ii = np.arange(len(GRID))
        seS = gmd["se"][ii, jbest, 1]; acc = gmd["acc"][ii, jbest]; f1 = gmd["macro_f1"][ii, jbest]
        l, = axes[0].plot(100 * seS, 100 * acc, label=f"{n} DS2")
        axes[1].plot(100 * seS, 100 * f1, color=l.get_color(), label=f"{n} DS2")
        axes[0].plot(100 * gmc["se"][ii, jbest, 1], 100 * gmc["acc"][ii, jbest], ls=":", color=l.get_color(), lw=1)
        axes[1].plot(100 * gmc["se"][ii, jbest, 1], 100 * gmc["macro_f1"][ii, jbest], ls=":", color=l.get_color(), lw=1)
    for ax, yl in zip(axes, ["plain accuracy (%)", "macro F1 (%)"]):
        ax.axvline(81.6, color="k", ls="--", lw=.8); ax.set_xlabel("Se_S (%)"); ax.set_ylabel(yl)
        ax.grid(alpha=.3)
    axes[0].axhline(98.18, color="k", ls="--", lw=.8)   # Model 6 plain acc
    axes[1].axhline(92.17, color="k", ls="--", lw=.8)
    axes[1].legend(fontsize=6)
    fig.suptitle("Se_S trade-off: sweep S logit offset, V offset chosen on DS1 (solid=DS2, dotted=DS1 calib)")
    fig.tight_layout(); fig.savefig(RES / "tradeoff_SeS.png", dpi=130)


if __name__ == "__main__":
    main()
