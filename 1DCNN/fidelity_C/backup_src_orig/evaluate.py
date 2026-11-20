# %% [markdown]
# ## evaluate.py — Standalone metric reporter and result aggregator
#
# Reference: Farag (2023), Sections 5 and 6.
#
# This script loads cached training results (or re-runs inference from saved
# model weights if re-evaluation is requested) and prints a full metrics
# report without re-training.
#
# Usage:
#   # Print metrics for all cached runs:
#   python -m src.evaluate
#
#   # Print metrics for a specific run name:
#   python -m src.evaluate --run P1
#
#   # Re-run inference from saved .keras weights on DS2:
#   python -m src.evaluate --run P1 --repredict
#
#   # Dump a CSV of all metrics:
#   python -m src.evaluate --csv results/metrics_all.csv

# %%
import argparse
import glob
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()

log = logging.getLogger("p3.evaluate")

AAMI_CLASSES = ["N", "S", "V", "F", "Q"]


# %%
def load_cached_runs(cache_dir: Path, run_name: str = None) -> dict:
    """
    Load all .pkl result files from cache_dir.
    Returns dict: run_name -> list of per-seed result dicts.
    """
    pattern = str(cache_dir / "*.pkl")
    files   = sorted(glob.glob(pattern))
    if not files:
        return {}

    runs = {}
    for f in files:
        path = Path(f)
        # filename format: {run_name}_seed{n}.pkl
        stem = path.stem  # e.g. "P1_seed0"
        if "_seed" not in stem:
            continue
        rname, seed_str = stem.rsplit("_seed", 1)
        if run_name and rname != run_name:
            continue
        try:
            with open(path, "rb") as fh:
                r = pickle.load(fh)
            runs.setdefault(rname, []).append(r)
        except Exception as e:
            log.warning("Could not load %s: %s", path, e)

    # Sort each run's seed list by seed number
    for rname in runs:
        runs[rname].sort(key=lambda r: r.get("seed", 0))

    return runs


def aggregate_seeds(seed_results: list) -> dict:
    """
    Compute mean ± std of every float metric across seeds.
    Returns a flat dict with keys like 'accuracy_mean', 'accuracy_std', etc.
    """
    if not seed_results:
        return {}
    keys = [k for k, v in seed_results[0].get("metrics_test", {}).items()
            if isinstance(v, float) and not np.isnan(v)]
    out = {}
    for k in keys:
        vals = [r["metrics_test"][k] for r in seed_results
                if k in r.get("metrics_test", {}) and not np.isnan(r["metrics_test"][k])]
        if vals:
            out[f"{k}_mean"] = float(np.mean(vals))
            out[f"{k}_std"]  = float(np.std(vals, ddof=0))
    out["n_seeds"]       = len(seed_results)
    out["total_epochs"]  = sum(r.get("n_epochs", 0) for r in seed_results)
    out["total_time_s"]  = sum(r.get("elapsed", 0) for r in seed_results)
    return out


# %%
def print_metrics_table(run_name: str, agg: dict, n_classes: int = 3) -> None:
    """Print a formatted metrics table to stdout."""
    pct  = lambda k: f"{100*agg.get(k+'_mean', float('nan')):6.2f} ± {100*agg.get(k+'_std',0):4.2f}%"
    pct1 = lambda k: f"{100*agg.get(k+'_mean', float('nan')):6.2f}%"

    print(f"\n{'='*70}")
    print(f"  Run: {run_name}  ({agg.get('n_seeds', '?')} seeds)")
    print(f"{'='*70}")
    print(f"  {'Plain accuracy':<30} {pct('accuracy')}")
    print(f"  {'OvR accuracy (1-2e/K)':<30} {pct('ovr_accuracy')}")
    print(f"  {'Majority-class baseline':<30} {pct1('majority_acc')}")
    print(f"  {'-'*50}")
    print(f"  {'Macro sensitivity':<30} {pct('macro_se')}")
    print(f"  {'Macro precision (PPV)':<30} {pct('macro_ppv')}")
    print(f"  {'Macro F1':<30} {pct('macro_f1')}")
    if agg.get("auc_macro_mean") is not None:
        print(f"  {'Macro ROC-AUC':<30} {pct('auc_macro')}")
    print(f"  {'-'*50}")
    cls_names = AAMI_CLASSES[:n_classes]
    for c in cls_names:
        se  = agg.get(f"se_{c}_mean",  float("nan"))
        ppv = agg.get(f"ppv_{c}_mean", float("nan"))
        f1  = agg.get(f"f1_{c}_mean",  float("nan"))
        n   = int(agg.get(f"n_{c}_mean", 0))
        print(f"  Class {c:<4}  Se={100*se:5.1f}%  PPV={100*ppv:5.1f}%  F1={100*f1:5.1f}%  n={n:,}")
    print(f"  {'-'*50}")
    print(f"  Seeds:       {agg.get('n_seeds', '?')}")
    print(f"  Total time:  {agg.get('total_time_s', 0)/60:.1f} min")
    print(f"  Total epochs:{agg.get('total_epochs', '?')}")
    print(f"{'='*70}\n")


def print_per_seed_table(run_name: str, seed_results: list) -> None:
    """Print one row per seed for quick inspection."""
    print(f"\nPer-seed results — {run_name}")
    header = f"  {'Seed':>4}  {'Acc':>6}  {'OvR':>6}  {'MacF1':>6}  {'Se_S':>6}  {'Se_V':>6}  {'Epochs':>6}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in seed_results:
        m = r.get("metrics_test", {})
        print(f"  {r.get('seed', -1):>4}  "
              f"{100*m.get('accuracy',0):6.2f}  "
              f"{100*m.get('ovr_accuracy',0):6.2f}  "
              f"{100*m.get('macro_f1',0):6.2f}  "
              f"{100*m.get('se_S',0):6.1f}  "
              f"{100*m.get('se_V',0):6.1f}  "
              f"{r.get('n_epochs',0):>6}")


def print_confusion(run_name: str, seed_results: list, n_classes: int = 3) -> None:
    """Print summed confusion matrix across all seeds."""
    cls = AAMI_CLASSES[:n_classes]
    total_cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for r in seed_results:
        cm = r.get("metrics_test", {}).get("cm")
        if cm:
            total_cm += np.array(cm, dtype=np.int64)
    print(f"\nConfusion matrix (summed over {len(seed_results)} seeds) — {run_name}")
    header = f"  {'True\\Pred':<8}" + "".join(f"{c:>8}" for c in cls)
    print(header)
    for i, c in enumerate(cls):
        row = f"  {c:<8}" + "".join(f"{total_cm[i,j]:>8,}" for j in range(n_classes))
        print(row)
    print()


# %%
def repredict_from_weights(run_name: str, cache_dir: Path, cfg: dict) -> list:
    """
    Reload saved .keras model files and re-run inference on DS2.
    Use this if the pickle cache is stale or if you want fresh probabilities.
    """
    import tensorflow as tf
    from .run_experiment import build_dataset, _resolve, DS2
    from .data import AAMI_CLASSES

    task       = cfg["task"]["name"]
    classes    = int(cfg["task"]["classes"])
    binary_def = cfg["task"]["binary_def"]

    log.info("Loading DS2 for re-prediction...")
    ds_test = build_dataset(DS2, cfg, task, classes, binary_def)

    model_files = sorted(cache_dir.glob(f"{run_name}_seed*.keras"))
    if not model_files:
        log.error("No .keras files found for run '%s' in %s", run_name, cache_dir)
        return []

    from .metrics import multiclass_metrics
    results = []
    for mf in model_files:
        seed = int(mf.stem.rsplit("_seed", 1)[1])
        log.info("Re-predicting seed %d from %s", seed, mf)
        model = tf.keras.models.load_model(str(mf))
        proba = model.predict([ds_test["X_ecg"], ds_test["X_rr"]],
                               batch_size=1024, verbose=0)
        y_pred = np.argmax(proba, axis=1)
        m = multiclass_metrics(ds_test["y"], y_pred, proba, k=classes)
        results.append({"seed": seed, "metrics_test": m, "proba_test": proba,
                         "y_pred_test": y_pred})
        log.info("seed %d: acc=%.2f%%  macro_F1=%.2f%%  Se_S=%.1f%%  Se_V=%.1f%%",
                 seed, 100*m["accuracy"], 100*m["macro_f1"],
                 100*m.get("se_S",0), 100*m.get("se_V",0))
    return results


# %%
def save_metrics_csv(all_runs: dict, out_path: Path) -> None:
    """Save a flat CSV with one row per (run, seed) for further analysis."""
    rows = []
    for rname, seed_results in all_runs.items():
        for r in seed_results:
            m = r.get("metrics_test", {})
            row = {"run": rname, "seed": r.get("seed", -1),
                   "n_epochs": r.get("n_epochs", 0),
                   "elapsed_s": r.get("elapsed", 0)}
            row.update(m)
            rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_path, index=False)
        log.info("metrics CSV saved to %s", out_path)
    else:
        log.warning("No metrics to save.")


def print_comparison_table(all_runs: dict, paper_targets: dict = None) -> None:
    """
    Print a compact side-by-side comparison of all runs.
    paper_targets: dict run_name -> {'ovr_accuracy': 98.18, 'macro_f1': 92.17, ...}
    """
    if paper_targets is None:
        paper_targets = {
            "P1": {"ovr_accuracy": 98.18, "macro_f1": 92.17,
                   "se_S": 85.3, "se_V": 96.34},
        }

    print(f"\n{'Run':<30} {'OvR Acc':>8} {'PlnAcc':>8} {'MacF1':>8} "
          f"{'Se_S':>7} {'Se_V':>7} {'Seeds':>6}")
    print("-" * 82)
    for rname, seed_results in sorted(all_runs.items()):
        agg  = aggregate_seeds(seed_results)
        pr   = paper_targets.get(rname, {})

        def _f(k):
            v = agg.get(k + "_mean", float("nan"))
            return f"{100*v:7.2f}" if not np.isnan(v) else "     —"

        print(f"{rname:<30} {_f('ovr_accuracy')} {_f('accuracy')} {_f('macro_f1')} "
              f"{_f('se_S')} {_f('se_V')} {len(seed_results):>6}")

        if pr:
            print(f"  {'(paper target)':<28} "
                  f"{pr.get('ovr_accuracy','—'):>8} {'—':>8} "
                  f"{pr.get('macro_f1','—'):>8} "
                  f"{pr.get('se_S','—'):>7} {pr.get('se_V','—'):>7}")
    print()


# %%
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="Print metrics for trained paper3 models without re-training.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--run",        default=None,
                    help="Run name to report (default: all cached runs)")
    ap.add_argument("--cache-dir",  default=None,
                    help="Cache directory (default: results/cache/)")
    ap.add_argument("--repredict",  action="store_true",
                    help="Reload .keras weights and re-run inference on DS2")
    ap.add_argument("--config",     default="configs/default.yaml")
    ap.add_argument("--csv",        default=None,
                    help="Save all metrics to this CSV file")
    ap.add_argument("--per-seed",   action="store_true",
                    help="Print per-seed breakdown")
    ap.add_argument("--confusion",  action="store_true",
                    help="Print confusion matrices")
    ap.add_argument("--classes",    type=int, default=3,
                    help="Number of classes (for confusion matrix labels)")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    cache_dir = Path(args.cache_dir) if args.cache_dir else PROJECT_ROOT / "results" / "cache"

    if not cache_dir.exists():
        print(f"\nNo cache directory found at: {cache_dir}")
        print("No models have been trained yet. Run the following first:\n")
        print("  cd HasanML/paper3")
        print("  python -m src.run_experiment --p1\n")
        sys.exit(0)

    # Re-predict from saved weights if requested
    if args.repredict:
        import yaml
        with open(PROJECT_ROOT / args.config) as fh:
            cfg = yaml.safe_load(fh)
        repred = repredict_from_weights(args.run or "P1", cache_dir, cfg)
        if repred:
            agg = aggregate_seeds(repred)
            print_metrics_table(f"{args.run or 'P1'} (re-predicted)", agg, args.classes)
        return

    # Load cached results
    all_runs = load_cached_runs(cache_dir, args.run)
    if not all_runs:
        print(f"\nNo cached results found in {cache_dir}")
        print("Run `python -m src.run_experiment --p1` to train first.\n")
        sys.exit(0)

    # Overall comparison table
    print_comparison_table(all_runs)

    # Detailed per-run tables
    for rname, seed_results in sorted(all_runs.items()):
        agg = aggregate_seeds(seed_results)
        print_metrics_table(rname, agg, args.classes)
        if args.per_seed:
            print_per_seed_table(rname, seed_results)
        if args.confusion:
            print_confusion(rname, seed_results, args.classes)

    # Save CSV
    if args.csv:
        save_metrics_csv(all_runs, Path(args.csv))


if __name__ == "__main__":
    main()
