# %% [markdown]
# ## export_teacher.py — teacher soft labels + ONNX / TorchScript export
#
# Exports the DS1-val-selected KAN-Former (single seed or logit-average
# ensemble, whichever has the best DS1-val 5-class macro-F1 in
# eval/eval_summary_<RUN>.json written by evaluate_full.py).
#
# NO distillation / student training happens here — export only.
#
# Output (default <P4>/results/teacher/):
#   soft_labels_{ds1,ds2}_T{1,2,4}.npz with per-beat keys
#     record_id   (str)    MIT-BIH record name, e.g. "101"
#     r_sample    (int64)  R-peak sample index in the original 360 Hz .atr annotation
#     symbol      (str)    original MIT-BIH beat symbol
#     aami_label  (int64)  0=N 1=S 2=V 3=F 4=Q
#     aami_name   (str)    "N","S","V","F","Q"
#     probs5      (float32, N x 5) softmax(logits / T)            [N,S,V,F,Q]
#     probs3      (float32, N x 3) probs5[:, :3] renormalised     [N,S,V]
#     is_FQ       (bool)   True if the TRUE label is F or Q (no valid 3-class target)
#     label3      (int64)  0/1/2 for N/S/V, -1 for F/Q beats
#     split_part  (str)    "ds1_train" / "ds1_val" / "ds2"
#     logits      (float32, N x 5) raw teacher logits (EMA weights)
#     T           (float)  softening temperature used for probs5/probs3
#     T_calibrated(float)  temperature fitted on DS1-val NLL (for calibrated probs: softmax(logits/T_calibrated))
#     teacher     (str)    which model produced the logits
#   kanformer_teacher.onnx, kanformer_teacher.ts (TorchScript), export_check.json, README.md

# %%
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

try:
    from .run_experiment import build_dataset, _resolve, load_config
    from .data import DS2, ds1_split, AAMI_CLASSES
    from .evaluate_full import (ensure_logits, load_seed_model, softmax_np,
                                probs3_from_probs5)
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.run_experiment import build_dataset, _resolve, load_config
    from src.data import DS2, ds1_split, AAMI_CLASSES
    from src.evaluate_full import (ensure_logits, load_seed_model, softmax_np,
                                   probs3_from_probs5)

log = logging.getLogger("p4.export")


class EnsembleTeacher(nn.Module):
    """Logit average of several KANFormer seeds (EMA weights)."""
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, beat, context, rr):
        out = self.models[0](beat, context, rr)
        for m in self.models[1:]:
            out = out + m(beat, context, rr)
        return out / len(self.models)


def save_split(out_dir, split, logits, ds, parts, T_list, T_cal, teacher_name):
    y = ds["y"].astype(np.int64)
    for T in T_list:
        p5 = softmax_np(logits, T).astype(np.float32)
        p3 = probs3_from_probs5(p5).astype(np.float32)
        np.savez_compressed(
            out_dir / f"soft_labels_{split}_T{T}.npz",
            record_id=ds["record_ids"].astype(str),
            r_sample=ds["r_samples"].astype(np.int64),
            symbol=ds["symbols"].astype(str),
            aami_label=y,
            aami_name=np.array([AAMI_CLASSES[i] for i in y]),
            probs5=p5, probs3=p3,
            is_FQ=(y >= 3),
            label3=np.where(y <= 2, y, -1).astype(np.int64),
            split_part=parts,
            logits=logits.astype(np.float32),
            T=np.float32(T), T_calibrated=np.float32(T_cal),
            teacher=np.array(teacher_name),
        )
        log.info("saved %s T=%s (%d beats)", split, T, len(y))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--run-name", default="DEFAULT")
    ap.add_argument("--out-dir", default="results/teacher")
    ap.add_argument("--teacher", default="auto",
                    help="auto (DS1-val selected) | seedN | ensemble")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    run_dir = _resolve(args.results_dir)
    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(run_dir / "resolved_config.yaml")
    n_classes = int(cfg["data"].get("classes", 5))
    device = torch.device(args.device)

    summ = json.load(open(run_dir / "eval" / f"eval_summary_{args.run_name}.json"))
    by_name = {r["name"]: r for r in summ["results"]}
    teacher = summ["selected_by_ds1val_macro_f1"] if args.teacher == "auto" else args.teacher
    if teacher == "ensemble":
        teacher = [n for n in by_name if n.startswith("ensemble")][0]
    T_cal = float(by_name[teacher]["T"])
    log.info("teacher = %s (DS1-val macro-F1 %.4f), T_cal = %.3f", teacher,
             by_name[teacher]["val_5c"]["macro_f1"], T_cal)

    tr_recs, val_recs = ds1_split(cfg["data"].get("ds1_split", "farag"))
    ds = {"ds1_train": build_dataset(tr_recs, cfg, "multiclass", n_classes),
          "ds1_val": build_dataset(val_recs, cfg, "multiclass", n_classes),
          "ds2": build_dataset(DS2, cfg, "multiclass", n_classes)}
    logits = ensure_logits(run_dir, args.run_name, cfg, n_classes, device, ds)
    seeds = sorted(logits)
    use_seeds = seeds if teacher.startswith("ensemble") else [int(teacher.replace("seed", ""))]
    L = {k: np.mean([logits[s][k] for s in use_seeds], axis=0) for k in ds}

    temps = list(cfg.get("distillation", {}).get("temperatures", [1, 2, 4]))
    # DS1 = train part + val part (concatenated in that order)
    ds1 = {k: np.concatenate([ds["ds1_train"][k], ds["ds1_val"][k]])
           for k in ("y", "record_ids", "r_samples", "symbols")}
    parts1 = np.array(["ds1_train"] * len(ds["ds1_train"]["y"]) + ["ds1_val"] * len(ds["ds1_val"]["y"]))
    L1 = np.concatenate([L["ds1_train"], L["ds1_val"]])
    save_split(out_dir, "ds1", L1, ds1, parts1, temps, T_cal, teacher)
    save_split(out_dir, "ds2", L["ds2"], ds["ds2"], np.array(["ds2"] * len(ds["ds2"]["y"])),
               temps, T_cal, teacher)

    # ---- ONNX + TorchScript (EMA weights, eval mode) ----
    models = []
    for s in use_seeds:
        m, cfg_m = load_seed_model(run_dir / "cache" / f"{args.run_name}_seed{s}.pt", cfg, n_classes, device)
        models.append(m)
    net = (EnsembleTeacher(models) if len(models) > 1 else models[0]).eval()
    beat_len = int(cfg["preprocess"]["beat_len"]); ctx_len = int(cfg["preprocess"].get("context_len", 1080))
    n_rr = int(cfg["rr"]["n_features"])
    idx = np.arange(0, len(ds["ds2"]["y"]), 997)[:32]
    xb = torch.from_numpy(ds["ds2"]["beat_segs"][idx]).unsqueeze(1)
    xc = torch.from_numpy(ds["ds2"]["context_segs"][idx]).unsqueeze(1)
    xr = torch.from_numpy(ds["ds2"]["rr_feats"][idx])
    with torch.no_grad():
        ref = net(xb, xc, xr).numpy()
    check = {"teacher": teacher, "n_check_beats": int(len(idx)),
             "max_abs_diff_vs_cached_logits": float(np.abs(ref - L["ds2"][idx]).max())}

    onnx_path = out_dir / "kanformer_teacher.onnx"
    try:
        torch.onnx.export(net, (xb[:2], xc[:2], xr[:2]), str(onnx_path),
                          input_names=["beat", "context", "rr"], output_names=["logits"],
                          dynamic_axes={n: {0: "batch"} for n in ["beat", "context", "rr", "logits"]},
                          opset_version=17, dynamo=False)
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
            o = sess.run(None, {"beat": xb.numpy(), "context": xc.numpy(), "rr": xr.numpy()})[0]
            check["onnx_max_abs_diff"] = float(np.abs(o - ref).max())
        except ImportError:
            check["onnx_max_abs_diff"] = "onnxruntime not installed"
        check["onnx"] = str(onnx_path)
    except Exception as exc:
        log.exception("ONNX export failed")
        check["onnx_error"] = repr(exc)

    ts_path = out_dir / "kanformer_teacher.ts"
    try:
        traced = torch.jit.trace(net, (xb[:2], xc[:2], xr[:2]))
        traced.save(str(ts_path))
        t2 = torch.jit.load(str(ts_path))
        with torch.no_grad():
            check["torchscript_max_abs_diff"] = float(np.abs(t2(xb, xc, xr).numpy() - ref).max())
        check["torchscript"] = str(ts_path)
    except Exception as exc:
        log.exception("TorchScript export failed")
        check["torchscript_error"] = repr(exc)

    check.update({"inputs": {"beat": [None, 1, beat_len], "context": [None, 1, ctx_len], "rr": [None, n_rr]},
                  "T_calibrated": T_cal, "temperatures": temps, "source_run": str(run_dir)})
    json.dump(check, open(out_dir / "export_check.json", "w"), indent=1)
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
