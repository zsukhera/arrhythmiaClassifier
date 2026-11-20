# KAN-Former teacher export (`HasanML/paper4/results/teacher/`)

Made by `python -m src.export_teacher --results-dir <RUN_DIR> --out-dir results/teacher`
(run from `HasanML/paper4/`). This is an export only. No student or knowledge-distillation training was done.

**Quality warning:** this teacher is weak on S. Read `reports/kanformer.md` before you use it.
On DS2 its S sensitivity is about 20-25%, far below our 1D-CNN.

The teacher name and source run are in `export_check.json` (keys `teacher` and `source_run`).
The teacher is the model with the best **DS1-val** macro-F1 among the single seeds and the
logit-average ensemble of the chosen run. DS2 was not used for this choice. The weights are the
EMA weights from the best DS1-val epoch.

## Files

| file | content |
|---|---|
| `soft_labels_ds1_T{1,2,4}.npz` | every DS1 beat (DS1-train part first, then DS1-val part) |
| `soft_labels_ds2_T{1,2,4}.npz` | every DS2 beat (test patients: for analysis only, never for training or tuning) |
| `kanformer_teacher.onnx` | ONNX (opset 17). Inputs `beat (B,1,256)`, `context (B,1,1080)`, `rr (B,6)`. Output `logits (B,5)` |
| `kanformer_teacher.ts` | TorchScript (traced), same signature `model(beat, context, rr)` |
| `export_check.json` | which model, T_calibrated, and the ONNX/TorchScript max-abs-diff vs PyTorch on 32 DS2 beats |

Paced records 102, 104, 107 and 217 are excluded, as in training. Only beats with an AAMI label
(N, S, V, F, Q symbols) are included, in annotation order within each record.

## Keys in every `.npz` (`d = np.load(f)`)

| key | dtype / shape | meaning |
|---|---|---|
| `record_id` | str (N,) | MIT-BIH record, e.g. `"101"`. **Join key 1** |
| `r_sample` | int64 (N,) | R-peak sample index from the original 360 Hz `.atr` annotation (`ann.sample`). **Join key 2** |
| `symbol` | str (N,) | original MIT-BIH beat symbol |
| `aami_label` | int64 (N,) | 0=N 1=S 2=V 3=F 4=Q |
| `aami_name` | str (N,) | "N","S","V","F","Q" |
| `probs5` | float32 (N,5) | softmax(logits / T), columns [N,S,V,F,Q] |
| `probs3` | float32 (N,3) | N/S/V probabilities = `probs5[:, :3] / probs5[:, :3].sum(1)` (F and Q columns dropped, then renormalised) |
| `is_FQ` | bool (N,) | True when the TRUE label is F or Q, so there is no valid 3-class target. Filter these out for an N/S/V student |
| `label3` | int64 (N,) | 0/1/2 for N/S/V, -1 for F/Q beats |
| `split_part` | str (N,) | `ds1_train` / `ds1_val` / `ds2` (the patient-wise split the teacher run used) |
| `logits` | float32 (N,5) | raw teacher logits, so any other T can be computed |
| `T` | float32 scalar | softening temperature of this file |
| `T_calibrated` | float32 scalar | temperature fitted on DS1-val NLL. Calibrated probabilities = softmax(logits / T_calibrated) |
| `teacher` | str scalar | model name (`seedK` or `ensemble(n)`) |

The soft labels at T = 1, 2 and 4 are softmax(logits/T) of the **uncalibrated** logits (Hinton-style
softening). Use `logits` with `T_calibrated` if you want calibrated probabilities.

## Joining with another pipeline (e.g. the paper3 student at 128 Hz)

Join on (`record_id`, `r_sample`). `r_sample` counts 360 Hz samples of the original annotation.
A pipeline that resamples to 128 Hz must keep the original annotation index, or map back with
`round(r128 * 360 / 128)` and match to the nearest annotation. Beats that a pipeline drops (edge
beats, F/Q) simply have no partner.

```python
import numpy as np, pandas as pd
d = np.load("soft_labels_ds1_T2.npz")
df = pd.DataFrame({"record_id": d["record_id"], "r_sample": d["r_sample"],
                   "aami": d["aami_name"],
                   **{f"p3_{c}": d["probs3"][:, i] for i, c in enumerate("NSV")}})
df = df[~d["is_FQ"]]          # 3-class student: drop F/Q beats
```

## Inputs expected by the ONNX / TorchScript model

Per beat: 360 Hz MLII, band-pass 0.5–40 Hz (4th-order Butterworth, sosfiltfilt).
- `beat` = samples [R-90, R+166), z-scored.
- `context` = samples [R-540, R+540), z-scored, edge-padded at record boundaries.
- `rr` = the 6 RR-ratio features from `src/rr.py`: pre/local, post/local, pre/global, post/global,
  pre/post, (post-pre)/local. Local window = 10 beats, global window = 400 beats, causal means.
  The post-RR features look one beat ahead.
