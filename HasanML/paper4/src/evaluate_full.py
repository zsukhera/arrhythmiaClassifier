# %% [markdown]
# ## evaluate_full.py — post-training evaluation of KAN-Former runs
#
# For every trained seed in <run_dir>/cache/<RUN>_seed<S>.pt (EMA weights):
#   * logits on DS1-train, DS1-val, DS2 (DS2 used ONLY for reporting)
#   * temperature T fitted on DS1-val NLL (Guo 2017); ECE before/after on
#     DS1-val and DS2
#   * 5-class AAMI metrics on DS2 (plain acc, OvR acc 1-2e/K, macro F1,
#     per-class Se/PPV)
#   * 3-class N/S/V metrics comparable with Farag 2023 / our 1D-CNN:
#       - beats whose TRUE label is F or Q are dropped,
#       - the prediction is argmax over the N/S/V probabilities
#         (= drop the F/Q columns and renormalise),
#       - OvR acc uses K=3.
#     A stricter variant ("3c_strict") keeps argmax over 5 classes, so an
#     N/S/V beat predicted as F/Q counts as an error.
#   * the same for the logit-average ensemble of all seeds.
# Model selection (which model becomes the teacher) uses DS1-val macro-F1 only.
#
# Usage:
#   python -m src.evaluate_full --results-dir results/gpu_run [--run-name DEFAULT]
#
# Writes <run_dir>/eval/eval_summary.json, eval_table.md and logits_*.npz.

# %%
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

try:
    from .run_experiment import build_dataset, _resolve, load_config
    from .data import DS2, ds1_split
    from .model import build_model
    from .train import EMA, _infer, resolve_device
    from .metrics import multiclass_metrics, ovr_accuracy
    from .calibrate import TemperatureScaler, compute_ece
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.run_experiment import build_dataset, _resolve, load_config
    from src.data import DS2, ds1_split
    from src.model import build_model
    from src.train import EMA, _infer, resolve_device
    from src.metrics import multiclass_metrics, ovr_accuracy
    from src.calibrate import TemperatureScaler, compute_ece

log = logging.getLogger("p4.eval")
NSV = ["N", "S", "V"]


# %%
def softmax_np(z, T=1.0):
    return F.softmax(torch.from_numpy(np.asarray(z)).float() / T, dim=1).numpy()


def probs3_from_probs5(p5: np.ndarray) -> np.ndarray:
    """Drop F/Q columns and renormalise over N/S/V."""
    p = p5[:, :3]
    return p / np.clip(p.sum(1, keepdims=True), 1e-12, None)


def metrics_3class(y5: np.ndarray, logits5: np.ndarray, strict: bool = False) -> dict:
    keep = y5 <= 2
    y = y5[keep]
    if strict:
        pred = logits5[keep].argmax(1)          # F/Q predictions count as errors
    else:
        pred = probs3_from_probs5(softmax_np(logits5[keep])).argmax(1)
    m = multiclass_metrics(y, pred, k=3) if not strict else _strict_metrics(y, pred)
    m["ovr_accuracy"] = ovr_accuracy(y, pred, 3)
    m["n_beats"] = int(keep.sum())
    return m


def _strict_metrics(y, pred):
    m = {"accuracy": float((y == pred).mean())}
    from sklearn.metrics import f1_score
    m["macro_f1"] = float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0))
    for i, c in enumerate(NSV):
        tp = int(((pred == i) & (y == i)).sum())
        m[f"se_{c}"] = tp / max(1, int((y == i).sum()))
        m[f"ppv_{c}"] = tp / max(1, int((pred == i).sum()))
    return m


def fit_temperature(logits_val, y_val, cfg_cal):
    sc = TemperatureScaler(T_init=float(cfg_cal.get("T_init", 1.5)))
    T = sc.calibrate(np.asarray(logits_val, dtype=np.float32), y_val,
                     max_iter=int(cfg_cal.get("T_max_iter", 200)))
    return float(T)


def load_seed_model(pt_path: Path, cfg: dict, n_classes: int, device):
    ck = torch.load(str(pt_path), map_location=device, weights_only=False)
    cfg_m = ck.get("cfg", cfg)
    model = build_model(cfg_m, n_classes=n_classes).to(device)
    model.load_state_dict(ck["state_dict"])
    ema = EMA(model)
    ema.shadow = ck["ema_shadow"]
    ema.apply_to(model)          # permanently use EMA weights
    model.eval()
    return model, cfg_m


def ensure_logits(run_dir: Path, run_name: str, cfg: dict, n_classes: int, device,
                  datasets: dict) -> dict:
    """Return {seed: {split: logits}} — cached in <run_dir>/eval/logits_<run>_seed<S>.npz."""
    out = {}
    eval_dir = run_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    for pt in sorted((run_dir / "cache").glob(f"{run_name}_seed*.pt")):
        if pt.name.endswith("_ckpt.pt"):
            continue
        seed = int(pt.stem.split("_seed")[-1])
        cache = eval_dir / f"logits_{run_name}_seed{seed}.npz"
        if cache.exists():
            d = np.load(cache)
            out[seed] = {k: d[k] for k in d.files}
            continue
        model, _ = load_seed_model(pt, cfg, n_classes, device)
        res = {}
        for split, ds in datasets.items():
            L, _, _ = _infer(model, ds, device, 512)
            res[split] = L.astype(np.float32)
        np.savez_compressed(cache, **res)
        out[seed] = res
        del model
        torch.cuda.empty_cache()
    return out


def evaluate_logits(name, Lval, Ltest, yval, ytest, cfg, n_classes):
    T = fit_temperature(Lval, yval, cfg.get("calibration", {}))
    r = {"name": name, "T": T}
    r["val_5c"] = multiclass_metrics(yval, Lval.argmax(1), softmax_np(Lval), k=n_classes)
    r["test_5c"] = multiclass_metrics(ytest, Ltest.argmax(1), softmax_np(Ltest), k=n_classes)
    r["val_3c"] = metrics_3class(yval, Lval)
    r["test_3c"] = metrics_3class(ytest, Ltest)
    r["test_3c_strict"] = metrics_3class(ytest, Ltest, strict=True)
    r["ece"] = {
        "val_before": compute_ece(softmax_np(Lval), yval),
        "val_after": compute_ece(softmax_np(Lval, T), yval),
        "test_before": compute_ece(softmax_np(Ltest), ytest),
        "test_after": compute_ece(softmax_np(Ltest, T), ytest),
    }
    for k in ("val_5c", "test_5c", "val_3c", "test_3c"):
        r[k].pop("cm", None) if k.startswith("val") else None
    return r


def fmt_row(r, key, cls):
    m = r[key]
    cells = [f"{100*m['accuracy']:.2f}", f"{100*m['ovr_accuracy']:.2f}", f"{100*m['macro_f1']:.2f}"]
    for c in cls:
        cells.append(f"{100*m[f'se_{c}']:.1f}/{100*m[f'ppv_{c}']:.1f}")
    return f"| {r['name']} | " + " | ".join(cells) + " |"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run-name", default="DEFAULT")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = _resolve(args.results_dir)
    cfg = load_config(run_dir / "resolved_config.yaml")
    n_classes = int(cfg["data"].get("classes", 5))
    device = resolve_device(args.device)

    tr_recs, val_recs = ds1_split(cfg["data"].get("ds1_split", "farag"))
    ds = {"ds1_train": build_dataset(tr_recs, cfg, "multiclass", n_classes),
          "ds1_val": build_dataset(val_recs, cfg, "multiclass", n_classes),
          "ds2": build_dataset(DS2, cfg, "multiclass", n_classes)}
    yval, ytest = ds["ds1_val"]["y"], ds["ds2"]["y"]

    logits = ensure_logits(run_dir, args.run_name, cfg, n_classes, device, ds)
    if not logits:
        raise SystemExit(f"no checkpoints in {run_dir/'cache'}")

    results = []
    for seed, L in sorted(logits.items()):
        results.append(evaluate_logits(f"seed{seed}", L["ds1_val"], L["ds2"], yval, ytest, cfg, n_classes))
    if len(logits) > 1:
        Lv = np.mean([L["ds1_val"] for L in logits.values()], axis=0)
        Lt = np.mean([L["ds2"] for L in logits.values()], axis=0)
        results.append(evaluate_logits(f"ensemble({len(logits)})", Lv, Lt, yval, ytest, cfg, n_classes))

    # mean ± std across single seeds
    singles = [r for r in results if r["name"].startswith("seed")]
    agg = {}
    for key in ("test_5c", "test_3c", "val_5c", "val_3c"):
        agg[key] = {}
        for k, v in singles[0][key].items():
            if isinstance(v, (int, float)):
                vals = [s[key][k] for s in singles]
                agg[key][k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    best = max(results, key=lambda r: r["val_5c"]["macro_f1"])
    summary = {"run_dir": str(run_dir), "run_name": args.run_name, "results": results,
               "agg_single_seeds": agg,
               "selected_by_ds1val_macro_f1": best["name"]}
    eval_dir = run_dir / "eval"
    with open(eval_dir / f"eval_summary_{args.run_name}.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)

    lines = [f"# Eval {args.run_name} ({run_dir.name})", "",
             "## DS2 5-class AAMI (Se/PPV %)", "",
             "| model | acc | OvR acc | macro F1 | N | S | V | F | Q |",
             "|---|---|---|---|---|---|---|---|---|"]
    lines += [fmt_row(r, "test_5c", "NSVFQ") for r in results]
    lines += ["", "## DS2 3-class N/S/V (F/Q beats dropped, F/Q columns dropped + renormalised)", "",
              "| model | acc | OvR acc | macro F1 | N | S | V |", "|---|---|---|---|---|---|---|"]
    lines += [fmt_row(r, "test_3c", NSV) for r in results]
    lines += ["", "## DS2 3-class strict (5-way argmax; N/S/V beat predicted F/Q = error)", "",
              "| model | acc | OvR acc(K=3) | macro F1 | N | S | V |", "|---|---|---|---|---|---|---|"]
    lines += [fmt_row(r, "test_3c_strict", NSV) for r in results]
    lines += ["", "## DS1-val (selection set) 5-class", "",
              "| model | acc | OvR acc | macro F1 | N | S | V | F | Q |",
              "|---|---|---|---|---|---|---|---|---|"]
    lines += [fmt_row(r, "val_5c", "NSVFQ") for r in results]
    lines += ["", "## Calibration (T fitted on DS1-val)", "",
              "| model | T | ECE val before | after | ECE DS2 before | after |", "|---|---|---|---|---|---|"]
    for r in results:
        e = r["ece"]
        lines.append(f"| {r['name']} | {r['T']:.3f} | {e['val_before']:.4f} | {e['val_after']:.4f} | "
                     f"{e['test_before']:.4f} | {e['test_after']:.4f} |")
    lines += ["", f"Selected by DS1-val 5-class macro-F1: **{best['name']}**"]
    for key in ("test_5c", "test_3c"):
        a = agg[key]
        lines.append(f"\nMean±std over single seeds ({key}): acc {100*a['accuracy']['mean']:.2f}±{100*a['accuracy']['std']:.2f}, "
                     f"OvR {100*a['ovr_accuracy']['mean']:.2f}±{100*a['ovr_accuracy']['std']:.2f}, "
                     f"macroF1 {100*a['macro_f1']['mean']:.2f}±{100*a['macro_f1']['std']:.2f}, "
                     f"Se_S {100*a['se_S']['mean']:.2f}±{100*a['se_S']['std']:.2f}, "
                     f"Se_V {100*a['se_V']['mean']:.2f}±{100*a['se_V']['std']:.2f}")
    (eval_dir / f"eval_table_{args.run_name}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
