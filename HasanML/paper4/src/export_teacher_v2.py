# %% [markdown]
# ## export_teacher_v2.py — soft labels + ONNX/TorchScript for the all-DS1
# ## KAN-Former (kanformer2).  Export only: NO distillation / student training.
#
# Same per-beat keys as results/teacher/ (see export_teacher.py), plus:
#   logits_oof / probs5_oof / probs3_oof  (DS1 files only): out-of-fold logits
#       from the 5-fold grouped CV at E* (mean over CV seeds), i.e. each DS1
#       beat scored by models that never saw its patient.  The main `logits`
#       on DS1 come from the refit models, which trained on every DS1 beat.
#   split_part = "ds1_all" for every DS1 beat (all of DS1 is training data).
#
# Usage: $PY -m src.export_teacher_v2 --variant V1_xxx --out-dir results/teacher_v2

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
    from .allds1 import OUT, CACHE, GPUData, get_variant, full_metrics, infer
    from .model import build_model
    from .data import AAMI_CLASSES
    from .evaluate_full import softmax_np, probs3_from_probs5
    from .rr import RR_EXT_NAMES
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.allds1 import OUT, CACHE, GPUData, get_variant, full_metrics, infer
    from src.model import build_model
    from src.data import AAMI_CLASSES
    from src.evaluate_full import softmax_np, probs3_from_probs5
    from src.rr import RR_EXT_NAMES

log = logging.getLogger("p4.export2")


class Ensemble(nn.Module):
    """Logit average of seeds.  Signature (beat, context, rr) or (beat, rr)."""
    def __init__(self, models, has_ctx):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.has_ctx = has_ctx

    def _f(self, beat, ctx, rr):
        out = self.models[0](beat, ctx, rr)
        for m in self.models[1:]:
            out = out + m(beat, ctx, rr)
        return out / len(self.models)


class EnsCtx(Ensemble):
    def forward(self, beat, context, rr):
        return self._f(beat, context, rr)


class EnsNoCtx(Ensemble):
    def forward(self, beat, rr):
        return self._f(beat, None, rr)


@torch.no_grad()
def infer_fp32(model, data, idx, bs=1024):
    model.eval()
    out = []
    it = torch.from_numpy(idx).to(data.device)
    for i in range(0, len(it), bs):
        j = it[i:i + bs]
        ctx = data.ctx[j].float().unsqueeze(1) if data.ctx is not None else None
        out.append(model(data.beat[j].unsqueeze(1), ctx, data.rr[j]).float().cpu())
    return torch.cat(out).numpy()


def save(out_dir, split, logits, meta, idx, T_list, T_cal, teacher, part, oof=None):
    y = meta["y"][idx].astype(np.int64)
    for T in T_list:
        p5 = softmax_np(logits, T).astype(np.float32)
        extra = {}
        if oof is not None:
            q5 = softmax_np(oof, T).astype(np.float32)
            extra = dict(logits_oof=oof.astype(np.float32), probs5_oof=q5,
                         probs3_oof=probs3_from_probs5(q5).astype(np.float32))
        np.savez_compressed(
            out_dir / f"soft_labels_{split}_T{T}.npz",
            record_id=meta["record_ids"][idx].astype(str),
            r_sample=meta["r_samples"][idx].astype(np.int64),
            symbol=meta["symbols"][idx].astype(str),
            aami_label=y, aami_name=np.array([AAMI_CLASSES[i] for i in y]),
            probs5=p5, probs3=probs3_from_probs5(p5).astype(np.float32),
            is_FQ=(y >= 3), label3=np.where(y <= 2, y, -1).astype(np.int64),
            split_part=np.array([part] * len(y)),
            logits=logits.astype(np.float32),
            T=np.float32(T), T_calibrated=np.float32(T_cal), teacher=np.array(teacher),
            **extra)
        log.info("saved %s T=%s (%d beats)", split, T, len(y))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="results/teacher_v2")
    ap.add_argument("--temps", nargs="+", type=int, default=[1, 2, 4])
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    P4 = Path(__file__).resolve().parents[1]
    out_dir = P4 / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    v = get_variant(args.variant)
    summ = json.load(open(OUT / "cv" / args.variant / "summary.json"))
    T_cal = float(summ["T_ensemble"])
    meta = dict(np.load(CACHE / "meta.npz"))
    data = GPUData(v, dev)
    ds1 = np.flatnonzero(~data.is_ds2); ds2 = np.flatnonzero(data.is_ds2)

    models = []
    for s in args.seeds:
        ck = torch.load(OUT / "refit" / args.variant / f"seed{s}.pt", map_location=dev,
                        weights_only=False)
        m = build_model(ck["cfg"], n_classes=5).to(dev)
        m.load_state_dict(ck["state_dict"]); m.eval()
        models.append(m)
    L1 = np.mean([infer_fp32(m, data, ds1) for m in models], axis=0)
    L2 = np.mean([infer_fp32(m, data, ds2) for m in models], axis=0)

    # DS1 out-of-fold logits (grouped CV at E*, mean over CV seeds), reordered to ds1
    oz = np.load(OUT / "cv" / args.variant / "oof_Estar.npz")
    pos = {int(i): k for k, i in enumerate(oz["idx"])}
    oof = oz["logits"].mean(0)[[pos[int(i)] for i in ds1]]

    teacher = f"kanformer2 {args.variant} refit-on-all-DS1 ensemble seeds {args.seeds}, E*={summ['E_star']}"
    save(out_dir, "ds1", L1, meta, ds1, args.temps, T_cal, teacher, "ds1_all", oof=oof)
    save(out_dir, "ds2", L2, meta, ds2, args.temps, T_cal, teacher, "ds2")

    m5, m3 = full_metrics(meta["y"][ds2].astype(np.int64), L2)
    check = {"teacher": teacher, "variant_knobs": v, "T_calibrated": T_cal,
             "T_source": "fitted on pooled grouped-CV OOF logits of the CV seed ensemble at E* (DS1 only)",
             "ds2_fp32_3class": {k: m3[k] for k in ("accuracy", "macro_f1", "se_N", "se_S", "se_V", "ppv_S", "ppv_V")},
             "ds2_fp32_5class": {k: m5[k] for k in ("accuracy", "macro_f1", "se_S", "se_V")}}

    # ---- ONNX + TorchScript on CPU, fp32 ----
    cpu = torch.device("cpu")
    net = (EnsCtx if v["context"] else EnsNoCtx)([m.to(cpu) for m in models], v["context"]).eval()
    sel = ds2[np.arange(0, len(ds2), 997)[:32]]
    xb = data.beat[torch.from_numpy(sel).to(dev)].float().cpu().unsqueeze(1)
    xr = data.rr[torch.from_numpy(sel).to(dev)].float().cpu()
    xc = (data.ctx[torch.from_numpy(sel).to(dev)].float().cpu().unsqueeze(1) if v["context"] else None)
    inputs = (xb, xc, xr) if v["context"] else (xb, xr)
    names = ["beat", "context", "rr"] if v["context"] else ["beat", "rr"]
    with torch.no_grad():
        ref = net(*inputs).numpy()
    check["max_abs_diff_cpu_vs_gpu_logits"] = float(np.abs(ref - L2[np.arange(0, len(ds2), 997)[:32]]).max())
    onnx_path = out_dir / "kanformer_teacher_v2.onnx"
    try:
        torch.onnx.export(net, tuple(x[:2] for x in inputs), str(onnx_path),
                          input_names=names, output_names=["logits"],
                          dynamic_axes={n: {0: "batch"} for n in names + ["logits"]},
                          opset_version=17, dynamo=False)
        import onnx
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        o = sess.run(None, {n: x.numpy() for n, x in zip(names, inputs)})[0]
        check["onnx_max_abs_diff"] = float(np.abs(o - ref).max())
        check["onnx"] = onnx_path.name
    except Exception as exc:
        log.exception("ONNX export failed")
        check["onnx_error"] = repr(exc)
    ts_path = out_dir / "kanformer_teacher_v2.ts"
    try:
        traced = torch.jit.trace(net, tuple(x[:2] for x in inputs))
        traced.save(str(ts_path))
        t2 = torch.jit.load(str(ts_path))
        with torch.no_grad():
            check["torchscript_max_abs_diff"] = float(np.abs(t2(*inputs).numpy() - ref).max())
        check["torchscript"] = ts_path.name
    except Exception as exc:
        log.exception("TorchScript export failed")
        check["torchscript_error"] = repr(exc)
    check["inputs"] = {"beat": [None, 1, 256]}
    if v["context"]:
        check["inputs"]["context"] = [None, 1, 1080]
    n_rr = 6 if v["rr"] == "base6" else 11
    check["inputs"]["rr"] = [None, n_rr]
    check["rr_feature_names"] = RR_EXT_NAMES[:n_rr]
    check["temperatures"] = args.temps
    json.dump(check, open(out_dir / "export_check.json", "w"), indent=1)
    print(json.dumps(check, indent=1))


if __name__ == "__main__":
    main()
