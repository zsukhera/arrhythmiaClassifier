# %% [markdown]
# ## run_experiment.py — Main CLI for the KAN-Former experiments
#
# Usage:
#   # Paper DEFAULT (5-class, 5 seeds):
#   python -m src.run_experiment --classes 5
#
#   # Faster smoke test (1 seed):
#   python -m src.run_experiment --classes 3 --seeds 0
#
#   # RR sanity check only:
#   python -m src.run_experiment --rr-baseline
#
#   # GPU with mixed precision:
#   python -m src.run_experiment --device cuda --mixed-precision
#
#   # Ablation A1 (MLP head instead of KAN):
#   python -m src.run_experiment --head mlp
#
#   # Full ablation table:
#   python -m src.run_experiment --ablations
#
#   # Distillation soft labels (after training):
#   python -m src.run_experiment --distill
#
# DS2 is used ONLY for final evaluation after all configuration is frozen.

# %%
import argparse
import logging
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import yaml

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
except NameError:
    PROJECT_ROOT = Path.cwd()   # Colab fallback

try:
    from .data import (DS1_TRAIN, DS1_VAL, DS2, DS1,
                       load_record, aami_labels, task_labels,
                       print_beat_counts, BEAT_SYMBOLS, AAMI_CLASSES,
                       ds1_split)
    from .preprocess import preprocess_record
    from .rr import compute_rr_features, lightgbm_sanity_check
    from .model import build_model, model_summary, count_parameters
    from .train import resolve_device, train_multi_seed, _infer, EMA
    from .metrics import (multiclass_metrics, aggregate_seeds,
                          print_metrics_table, plot_confusion)
    from .calibrate import calibrate_ensemble, compute_ece
    from .distill_export import (collect_logits, ensemble_logits,
                                  save_soft_labels, export_onnx, export_torchscript)
    from .model import enable_mixed_precision
except ImportError:
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.data import (DS1_TRAIN, DS1_VAL, DS2, DS1,
                          load_record, aami_labels, task_labels,
                          print_beat_counts, BEAT_SYMBOLS, AAMI_CLASSES,
                       ds1_split)
    from src.preprocess import preprocess_record
    from src.rr import compute_rr_features, lightgbm_sanity_check
    from src.model import build_model, model_summary, count_parameters
    from src.train import resolve_device, train_multi_seed, _infer, EMA
    from src.metrics import (multiclass_metrics, aggregate_seeds,
                              print_metrics_table, plot_confusion)
    from src.calibrate import calibrate_ensemble, compute_ece
    from src.distill_export import (collect_logits, ensemble_logits,
                                     save_soft_labels, export_onnx, export_torchscript)

    def enable_mixed_precision(*a, **kw):
        pass


log = logging.getLogger("p4")


# %%
def load_config(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def _resolve(p: str) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PROJECT_ROOT / p


# %%
def process_record(name: str, cfg: dict) -> dict | None:
    """Load and fully preprocess one MIT-BIH record at 360 Hz."""
    data_cfg = cfg["data"]
    rr_cfg   = cfg["rr"]

    rec = load_record(
        name,
        data_dir=_resolve(data_cfg["data_dir"]),
        mirrors=[_resolve(m) for m in data_cfg.get("local_mirrors", [])],
        lead=data_cfg.get("lead", "MLII"),
    )

    n_beats = len(rec.ann_samples)
    if n_beats == 0:
        log.warning("record %s: no beat annotations", name)
        return None

    sym_all = rec.ann_symbols

    # Preprocess signal (bandpass + window extraction)
    prep = preprocess_record(rec.signal, rec.ann_samples, cfg)

    # RR features from ALL beats
    rr = compute_rr_features(
        rec.ann_samples,
        fs=rec.fs,
        local_window=int(rr_cfg["local_window"]),
        global_window=int(rr_cfg["global_window"]),
    )

    y_aami = aami_labels(sym_all)

    return {
        "beat_segs":    prep["beat_segs"],
        "context_segs": prep["context_segs"],
        "rr_feats":     rr,
        "y_aami":       y_aami,
        "symbols":      sym_all,
        "was_padded":   prep["was_padded"],
        "r_samples":    rec.ann_samples,
        "record_id":    name,
    }


def build_dataset(records: list, cfg: dict, task: str = "multiclass",
                  classes: int = 5, binary_def: str = "aami") -> dict:
    """Build the full (beat_segs, context_segs, rr_feats, y) dataset for a list of records."""
    beat_list, ctx_list, rr_list, y_list = [], [], [], []
    sym_list, rid_list, rs_list = [], [], []

    for name in records:
        r = process_record(name, cfg)
        if r is None:
            continue
        yt = task_labels(r["y_aami"], r["symbols"], task, classes, binary_def)
        mask = yt >= 0

        beat_list.append(r["beat_segs"][mask])
        if r["context_segs"] is not None:
            ctx_list.append(r["context_segs"][mask])
        rr_list.append(r["rr_feats"][mask])
        y_list.append(yt[mask])
        sym_list.extend(r["symbols"][mask].tolist())
        rid_list.extend([name] * int(mask.sum()))
        rs_list.extend(r["r_samples"][mask].tolist())

    if not beat_list:
        raise RuntimeError(f"No valid beats in records: {records}")

    ctx_arr = np.concatenate(ctx_list, axis=0) if ctx_list else None
    return {
        "beat_segs":    np.concatenate(beat_list, axis=0),
        "context_segs": ctx_arr,
        "rr_feats":     np.concatenate(rr_list, axis=0),
        "y":            np.concatenate(y_list, axis=0),
        "symbols":      np.array(sym_list),
        "record_ids":   np.array(rid_list),
        "r_samples":    np.array(rs_list),
    }


# %%
def write_counts(ds_tr, ds_val, ds_test, task, classes, out_dir):
    print_beat_counts(
        {"DS1-train": ds_tr["y"], "DS1-val": ds_val["y"], "DS2": ds_test["y"]},
        task=task, classes=classes,
    )
    with open(out_dir / "beat_counts.txt", "w") as fh:
        fh.write(f"DS1-train: {len(ds_tr['y'])} beats\n")
        fh.write(f"DS1-val:   {len(ds_val['y'])} beats\n")
        fh.write(f"DS2:       {len(ds_test['y'])} beats\n")


# %%
def run_ablation(name: str, cfg_override: dict, cfg_base: dict,
                 ds_tr: dict, ds_val: dict, ds_test: dict,
                 n_classes: int, seeds: list, cache_dir: Path,
                 device) -> list:
    """Run one ablation configuration."""
    cfg_a = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg_base.items()}
    for key, val in cfg_override.items():
        if isinstance(val, dict) and isinstance(cfg_a.get(key), dict):
            cfg_a[key] = {**cfg_a[key], **val}
        else:
            cfg_a[key] = val

    log.info("=== Ablation: %s ===", name)
    return train_multi_seed(
        cfg_a, ds_tr, ds_val, ds_test,
        n_classes=n_classes,
        seeds=seeds,
        cache_dir=cache_dir,
        run_name=name,
        device=device,
    )


# %%
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description="KAN-Former inter-patient ECG classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config",        default="configs/default.yaml")
    ap.add_argument("--classes",       type=int, choices=[3, 4, 5], default=None)
    ap.add_argument("--task",          choices=["multiclass", "binary"], default=None)
    ap.add_argument("--binary-def",    choices=["aami", "raw"], default=None)
    ap.add_argument("--head",          choices=["kan", "mlp"], default=None)
    ap.add_argument("--context",       type=int, choices=[0, 720, 1080], default=None)
    ap.add_argument("--no-rr",         action="store_true")
    ap.add_argument("--no-transformer",action="store_true")
    ap.add_argument("--stft",          action="store_true",
                    help="STFT spectrogram input (ablation A7; not for DS2 evaluation)")
    ap.add_argument("--seeds",         nargs="+", type=int, default=None)
    ap.add_argument("--device",        choices=["auto", "cpu", "cuda", "gpu"], default=None)
    ap.add_argument("--mixed-precision",   action="store_true", default=False)
    ap.add_argument("--no-deterministic",  action="store_true", default=False)
    ap.add_argument("--results-dir",   default=None)
    ap.add_argument("--rr-baseline",   action="store_true",
                    help="Run LightGBM RR sanity check and exit")
    ap.add_argument("--ablations",     action="store_true",
                    help="Run all A1-A6 ablations")
    ap.add_argument("--distill",       action="store_true",
                    help="Generate soft labels and export teacher")
    ap.add_argument("--search",        action="store_true",
                    help="Optuna hyper-parameter search")
    ap.add_argument("--pretrain",      action="store_true",
                    help="Self-supervised masked-signal pretraining before fine-tuning")
    ap.add_argument("--max-epochs",    type=int, default=None)
    ap.add_argument("--ablation-seeds", nargs="+", type=int, default=None,
                    help="Seeds for each ablation (default: first seed only)")
    ap.add_argument("--ablation-names", nargs="+", default=None,
                    help="Subset of ablations to run, e.g. A1_mlp_head A4_no_rr")
    ap.add_argument("--set",           nargs="*", default=[], metavar="KEY.SUB=VAL",
                    help="Generic YAML override, e.g. --set training.lr=1e-4 model.n_layers=2")
    args = ap.parse_args(argv)

    cfg = load_config(_resolve(args.config))

    # CLI overrides
    if args.classes:          cfg["data"]["classes"]                = args.classes
    if args.task:             cfg.setdefault("task", {})["name"]    = args.task
    if args.binary_def:       cfg.setdefault("task", {})["bin_def"] = args.binary_def
    if args.head:             cfg["model"]["head"]                  = args.head
    if args.context is not None:
        cfg["preprocess"]["context_len"]  = args.context
        cfg["preprocess"]["context_half"] = args.context // 2
    if args.seeds:            cfg["seeds"]                           = args.seeds
    if args.device:           cfg["compute"]["device"]              = args.device
    if args.mixed_precision:  cfg["compute"]["mixed_precision"]     = True
    if args.no_deterministic: cfg["compute"]["deterministic"]       = False
    if args.results_dir:
        # Keep cache/figures/teacher inside the run directory so separate runs
        # never read each other's cached seeds.
        cfg["output"]["results_dir"] = args.results_dir
        cfg["output"]["cache_dir"]   = str(Path(args.results_dir) / "cache")
        cfg["output"]["figures_dir"] = str(Path(args.results_dir) / "figures")
        cfg["output"]["teacher_dir"] = str(Path(args.results_dir) / "teacher")
    if args.max_epochs:       cfg["training"]["max_epochs"]         = args.max_epochs
    for kv in args.set:
        key, val = kv.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p_ in parts[:-1]:
            node = node.setdefault(p_, {})
        node[parts[-1]] = yaml.safe_load(val)

    out     = _resolve(cfg["output"]["results_dir"])
    fig_dir = _resolve(cfg["output"]["figures_dir"])
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out / "run.log", mode="w"),
        ],
    )
    log.info("KAN-Former — platform %s, Python %s", platform.platform(),
             platform.python_version())

    # Device setup
    device = resolve_device(cfg["compute"].get("device", "auto"))
    log.info("Device: %s", device)

    if cfg["compute"].get("mixed_precision", False) and device.type == "cuda":
        try:
            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            log.info("Mixed precision (AMP + TF32) enabled on CUDA")
        except Exception as exc:
            log.warning("Could not enable TF32: %s", exc)

    # Save resolved config
    with open(out / "resolved_config.yaml", "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)

    # --- Data loading ---
    classes    = int(cfg["data"].get("classes", 5))
    task       = cfg.get("task", {}).get("name", "multiclass")
    binary_def = cfg.get("task", {}).get("bin_def", "aami")
    seeds      = list(cfg["seeds"])
    cache_dir  = _resolve(cfg["output"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)

    log.info("=== Loading data: task=%s, classes=%d ===", task, classes)
    tr_recs, val_recs = ds1_split(cfg["data"].get("ds1_split", "farag"))
    log.info("DS1 split '%s': train=%s val=%s", cfg["data"].get("ds1_split", "farag"),
             tr_recs, val_recs)
    ds_tr   = build_dataset(tr_recs,  cfg, task, classes, binary_def)
    ds_val  = build_dataset(val_recs, cfg, task, classes, binary_def)
    ds_test = build_dataset(DS2,       cfg, task, classes, binary_def)
    write_counts(ds_tr, ds_val, ds_test, task, classes, out)

    # --- RR baseline sanity check ---
    if args.rr_baseline:
        log.info("=== RR-only LightGBM sanity check ===")
        y_tr_valid  = ds_tr["y"][ds_tr["y"]  >= 0]
        y_val_valid = ds_val["y"][ds_val["y"] >= 0]
        X_tr_valid  = ds_tr["rr_feats"][ds_tr["y"] >= 0]
        X_val_valid = ds_val["rr_feats"][ds_val["y"] >= 0]
        lightgbm_sanity_check(X_tr_valid, y_tr_valid,
                               X_val_valid, y_val_valid, n_classes=classes)
        return

    # --- Main training run ---
    log.info("=== Training DEFAULT (run name: DEFAULT) ===")
    results = train_multi_seed(
        cfg, ds_tr, ds_val, ds_test,
        n_classes=classes, seeds=seeds,
        cache_dir=cache_dir, run_name="DEFAULT",
        device=device,
    )

    agg = aggregate_seeds(results)
    print_metrics_table("DEFAULT", agg, n_classes=classes)

    # Confusion matrix for last seed
    if results:
        cm = np.array(results[-1]["metrics_test"]["cm"])
        plot_confusion(cm, AAMI_CLASSES[:classes], "DEFAULT DS2",
                       out_path=fig_dir / "confusion_DEFAULT.png")

    # --- Ablations ---
    if args.ablations:
        ablation_specs = [
            # A1: MLP head instead of KAN
            ("A1_mlp_head",     {"model": {"head": "mlp"}}),
            # A2: no Transformer (n_layers=0 → CNN+KAN only)
            ("A2_no_transformer",{"model": {"n_layers": 0}}),
            # A3: no context window
            ("A3_no_context",   {"preprocess": {"context_len": 0, "context_half": 0}}),
            # A4: no RR features (zero-out RR by using 0-length — handled via flag)
            ("A4_no_rr",        {"rr": {"n_features": 0}}),
            # A5: single kernel 7 (no multi-scale)
            ("A5_single_kernel",{"model": {"ms_kernels": [7, 7, 7]}}),
            # A6a: KAN grid_size 3
            ("A6a_grid3",       {"model": {"kan_grid_size": 3}}),
            # A6b: KAN grid_size 8
            ("A6b_grid8",       {"model": {"kan_grid_size": 8}}),
        ]
        abl_results = {}
        abl_seeds = args.ablation_seeds or [seeds[0]]
        for abl_name, override in ablation_specs:
            if args.ablation_names and abl_name not in args.ablation_names:
                continue
            r = run_ablation(abl_name, override, cfg,
                             ds_tr, ds_val, ds_test,
                             classes, abl_seeds, cache_dir, device)
            abl_results[abl_name] = r

        # Print ablation comparison
        log.info("=== Ablation summary ===")
        all_runs = {"DEFAULT": results}
        all_runs.update(abl_results)
        print(f"\n{'Run':<30} {'Acc':>7} {'OvR':>7} {'MacF1':>7} {'Se_S':>7} {'Se_V':>7}")
        print("-" * 65)
        for rname, rr in all_runs.items():
            if not rr: continue
            m = rr[0]["metrics_test"]
            print(f"{rname:<30} "
                  f"{100*m.get('accuracy',0):7.2f} "
                  f"{100*m.get('ovr_accuracy',0):7.2f} "
                  f"{100*m.get('macro_f1',0):7.2f} "
                  f"{100*m.get('se_S',0):7.1f} "
                  f"{100*m.get('se_V',0):7.1f}")

    # --- Distillation soft labels + export ---
    if args.distill:
        if not results:
            log.error("No trained models found; run training first")
            return

        log.info("=== Generating soft labels and exporting teacher ===")
        import torch

        teacher_dir = _resolve(cfg["output"]["teacher_dir"])
        teacher_dir.mkdir(parents=True, exist_ok=True)

        # Reload all seed models and collect logits on DS1 (train+val) and DS2
        logits_tr_per_seed, logits_val_per_seed, logits_test_per_seed = [], [], []

        for seed in seeds:
            pt_path = cache_dir / f"DEFAULT_seed{seed}.pt"
            if not pt_path.exists():
                log.warning("No .pt file for seed %d; skipping", seed)
                continue

            model = build_model(cfg, n_classes=classes).to(device)
            checkpoint = torch.load(str(pt_path), map_location=device, weights_only=False)
            model.load_state_dict(checkpoint["state_dict"])

            ema = EMA(model, decay=float(cfg["training"]["ema_decay"]))
            ema.shadow = checkpoint["ema_shadow"]
            old = ema.apply_to(model)

            logits_tr_per_seed.append(collect_logits(model, ds_tr, device))
            logits_val_per_seed.append(collect_logits(model, ds_val, device))
            logits_test_per_seed.append(collect_logits(model, ds_test, device))

            ema.restore(model, old)

        if not logits_tr_per_seed:
            log.error("No models loaded for distillation")
            return

        # Fit temperature on DS1-val averaged logits
        from .calibrate import calibrate_ensemble, TemperatureScaler
        avg_val_logits = ensemble_logits(logits_val_per_seed)
        scaler = None
        if cfg.get("calibration", {}).get("enabled", True):
            scaler = calibrate_ensemble(logits_val_per_seed, ds_val["y"],
                                        cfg.get("calibration", {}))

        # Build DS1 (train + val combined) arrays
        ds_ds1 = {
            "beat_segs":    np.concatenate([ds_tr["beat_segs"],    ds_val["beat_segs"]]),
            "context_segs": (np.concatenate([ds_tr["context_segs"], ds_val["context_segs"]])
                             if ds_tr.get("context_segs") is not None else None),
            "rr_feats":     np.concatenate([ds_tr["rr_feats"],     ds_val["rr_feats"]]),
            "y":            np.concatenate([ds_tr["y"],            ds_val["y"]]),
            "symbols":      np.concatenate([ds_tr["symbols"],      ds_val["symbols"]]),
            "record_ids":   np.concatenate([ds_tr["record_ids"],   ds_val["record_ids"]]),
            "r_samples":    np.concatenate([ds_tr["r_samples"],    ds_val["r_samples"]]),
        }
        avg_ds1_logits = ensemble_logits(
            [np.concatenate([lt, lv]) for lt, lv in zip(logits_tr_per_seed, logits_val_per_seed)]
        )

        temps = list(cfg.get("distillation", {}).get("temperatures", [1, 2, 4]))
        save_soft_labels(avg_ds1_logits, ds_ds1, DS1,
                         teacher_dir, temps, split_name="ds1",
                         temperature_scaler=scaler)

        if cfg.get("distillation", {}).get("save_ds2", True):
            avg_test_logits = ensemble_logits(logits_test_per_seed)
            save_soft_labels(avg_test_logits, ds_test, DS2,
                             teacher_dir, temps, split_name="ds2",
                             temperature_scaler=scaler)

        # ONNX + TorchScript export of the last-seed model
        model = build_model(cfg, n_classes=classes).to(device)
        pt_path = cache_dir / f"DEFAULT_seed{seeds[-1]}.pt"
        if pt_path.exists():
            ckpt = torch.load(str(pt_path), map_location=device, weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            # BUGFIX: export the EMA weights (the ones that were evaluated),
            # not the raw training weights.
            for n, p in model.named_parameters():
                if n in ckpt["ema_shadow"]:
                    p.data.copy_(ckpt["ema_shadow"][n].to(p.dtype))

        if cfg.get("export", {}).get("onnx", True):
            export_onnx(model, cfg, teacher_dir / "kanformer.onnx", device)
        if cfg.get("export", {}).get("torchscript", True):
            export_torchscript(model, cfg, teacher_dir / "kanformer.pt", device)


if __name__ == "__main__":
    main()
