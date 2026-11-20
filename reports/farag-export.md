# farag-export: P2/P3 binary variants and TFLite / TFLite-Micro export (Farag 2023 tiny 1D-CNN)

Agent: farag-export, 2026-11-18/19, run on the box (CPU only, 3 threads).
All artefacts are in `1DCNN/export/`. Firmware usage is in `1DCNN/export/README.md`.
DS2 was used only for the final evaluation. The seed, the thresholds, the PTQ
representative data, the QAT/pruning best epoch and the recommended variant
were all chosen on DS1 data.

## 1. P2 / P3 binary variants (DS2, arrhythmia = positive class)

P2 = AAMI binary (N vs S/V/F/Q, 49,712 DS2 beats, 5,453 positive).
P3 = raw binary (MIT-BIH 'N' vs every other beat symbol, 13,268 positive, so
L/R/e/j count as "arrhythmia").

* **Paper-default recipe.** The unchanged `src.run_experiment --p2 --p3` CLI:
  train on DS1-train, early-stop on val_loss over the DS1-val records, no class
  weights. Outputs were redirected with `--config 1DCNN/export/configs/p2p3_paperdefault.yaml`.
  Models are `results/cache/P{2,3}_dedicated_seed*.keras`.
* **Improved recipe.** Same as the official P1 run: full DS1, balanced class
  weights, macro-F1 early stopping on a 15 % random-beat DS1 hold-out (rng 42),
  matched-filter kernel from DS1-all. Models are `results/cache/P{2,3}_fullDS1_seed*.keras`.
* "val-opt" is the threshold on P(arr) that maximises F1 on that run's own
  DS1 validation data (DS1-val records or the DS1 hold-out).

| Run | Threshold | Se % | Sp % | PPV % | F1 % | Acc % | AUC |
|---|---|---|---|---|---|---|---|
| P2 paper-default (10 seeds, mean±sd) | 0.5 | 37.70±16.81 | 99.62±0.75 | 94.83±6.66 | 51.00±19.21 | 92.83±1.51 | 0.936±0.005 |
| P2 paper-default (mean±sd) | val-opt | 74.86±8.05 | 96.11±2.18 | 72.34±8.72 | 72.67±1.78 | 93.78±1.11 | 0.936 |
| P2 paper-default ensemble | val-opt 0.06 | 79.46 | 95.77 | 69.81 | 74.32 | 93.98 | 0.942 |
| **P2 improved (mean±sd)** | 0.5 | 94.89±1.52 | 89.92±1.84 | 54.20±5.26 | 68.80±3.46 | 90.47±1.49 | 0.972±0.010 |
| **P2 improved (mean±sd)** | val-opt | 90.56±1.49 | 94.89±0.90 | 68.82±3.55 | 78.12±1.81 | 94.42±0.66 | 0.972±0.010 |
| **P2 improved ensemble** | val-opt 0.82 | 90.21 | 96.22 | 74.64 | **81.69** | **95.56** | **0.978** |
| P2 from the collapsed official P1 ensemble (1−P(N)) | 0.5 | 93.69 | 91.13 | 56.56 | 70.54 | 91.41 | 0.966 |
| P3 paper-default (mean±sd) | 0.5 | 11.98±5.28 | 99.84±0.22 | 97.60±2.91 | 20.88±8.30 | 76.39±1.28 | 0.738±0.019 |
| P3 paper-default (mean±sd) | val-opt | 59.12±12.63 | 70.34±10.00 | 42.99±4.14 | 48.79±3.63 | 67.35±4.38 | 0.738 |
| P3 improved (mean±sd) | 0.5 | 66.45±2.66 | 95.38±0.89 | 84.02±2.81 | 74.18±2.37 | 87.66±1.09 | 0.877±0.020 |
| P3 improved ensemble | 0.5 | 66.76 | 96.35 | 86.95 | 75.53 | 88.46 | 0.891 |
| P3 improved ensemble | val-opt 0.58 | 64.96 | 97.12 | 89.16 | 75.16 | 88.54 | 0.891 |

Full tables, including the per-seed numbers, are in `1DCNN/export/p2p3_summary.md/.json`
and `1DCNN/export/logs/p2p3_fullds1.log`.

Findings:
- The paper-default binary models collapse toward "Normal" at t=0.5. One
  seed reaches Se=0.6 % (P2 seed 7). The improved recipe fixes this: P2 AUC
  goes from 0.936 to 0.972 and F1 at the val-opt threshold from 72.7 to 78.1.
- A dedicated P2 model is better than collapsing the 3-class P1 ensemble
  (AUC 0.978 vs 0.966 for the ensembles).
- P3 is much harder. Treating L/R/e/j as "arrhythmia" means the model must
  learn bundle-branch morphology, which does not transfer between patients
  (the same failure LightGBM showed). The best P3 AUC is 0.89. I did not
  export P3.
- Caveat for the improved recipe: the hold-out is a random-beat split, so it
  shares patients with training. Thresholds chosen on it are optimistic about
  generalisation (for example, hold-out F1 is about 88 % while DS2 F1 is about 78 %).

## 2. Exported P1 model, seed choice

The rule was declared before looking at DS2: pick the seed with the best
macro-F1 on the official P1 DS1 hold-out. I re-created that hold-out exactly
and checked it: max |Δp| = 0 against the stored `proba_val`.

| seed | 0 | 1 | 2 | 3 | 4 | 5 | **6** | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| hold-out macro-F1 % | 83.96 | 79.88 | 80.06 | 79.15 | 81.63 | 84.14 | **85.74** | 83.37 | 82.16 | 81.15 |

**Honest note:** seed 6 has the worst DS2 S sensitivity of the 10 seeds
(Se_S 77.35 %, against a mean of 85.29 %). A random-beat DS1 hold-out does
not predict inter-patient S performance. I kept the pre-declared choice and
did not swap seeds based on DS2. For comparison, the 10-seed ensemble with
tau gave acc 91.61 % and Se_S 87.10 %, but it cannot run as a single small
model on the MCU.

For P2, seed 3 had the best hold-out F1 (88.80 %). Its threshold of 0.77 is
that seed's hold-out optimum.

## 3. TFLite variants on DS2 (TFLite interpreter, batch 1)

### P1, 3-class N/S/V, seed 6, argmax
tau_S = 0.50 gives the same result as argmax for 3 classes, so only argmax is
shown. The Keras-3 model has 1,195 params; the exported graph has 1,143 after
folding BN.

| Variant | .tflite B | Acc % | OvR acc % | Macro F1 % | Se N | Se S | Se V | PPV N | PPV S | PPV V | ΔAcc pp vs Keras | ΔmF1 pp | DS1 hold-out mF1 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Keras float | – | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | 0 | 0 | 85.74 |
| float32 | 8,060 | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| float16 | 6,964 | 90.49 | 93.66 | 73.08 | 90.63 | 77.41 | 96.03 | 99.70 | 31.20 | 68.29 | +0.01 | +0.03 | 85.74 |
| dynamic-range | 8,060 | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| **INT8 PTQ** | 7,072 | 90.71 | 93.81 | 74.27 | 90.76 | 85.68 | 92.92 | 99.65 | 33.24 | 70.08 | +0.24 | +1.21 | **85.21** |
| INT8 QAT (tfmot) | 6,944 | 88.76 | 92.51 | 70.31 | 88.84 | 81.38 | 91.87 | 99.50 | 31.07 | 59.32 | −1.72 | −2.74 | 84.54 |
| Keras 50 %-pruned (float) | – | 88.76 | 92.50 | 71.05 | 88.50 | 86.66 | 93.48 | 99.80 | 31.44 | 60.14 | −1.72 | −2.01 | 82.08 |
| 50 %-pruned + INT8 | 7,072 | 89.31 | 92.87 | 71.82 | 89.15 | 86.34 | 93.23 | 99.78 | 32.35 | 61.64 | −1.17 | −1.24 | 82.89 |
| *Paper Model 6 target* | ~15 KB | 98.18 | – | 92.17 | 99.10 | 81.60 | 95.00 | – | – | – | | | |

The **TFLite-Micro runtime** (reference kernels, `tflite-micro` pip package,
full DS2) reproduces these numbers. INT8 PTQ gives acc 90.70, mF1 74.24,
Se N/S/V 90.75/85.74/92.89, and argmax agrees with the TFLite interpreter on
99.95 % of beats. float32 is bit-identical in argmax. QAT gives 88.77/70.34
and pruned+INT8 89.30/71.81. Full table: `1DCNN/export/P1/tables.md`.

Reading the table:
- **PTQ is essentially lossless.** It loses −0.5 pp macro-F1 on the DS1
  hold-out. On DS2 it moves Se_S from 77 % to 86 % and Se_V from 96 % to 93 %,
  which is decision-boundary jitter for the S class rather than a real gain.
  The paper reported a >10 % PTQ drop; we do not reproduce that drop once
  the PTQ input-order bug is fixed (see §5).
- **QAT does not help.** It is slightly worse than PTQ on the DS1 hold-out
  (84.5 vs 85.2) and on DS2. PTQ has no gap for QAT to close on a model this
  small.
- **50 % pruning costs about 3 pp of hold-out macro-F1.** Pruning a
  ~1.1k-parameter model is harsh. TFLite also stores the zeros densely, so
  pruned+INT8 is the same 7,072 B as PTQ. **Pruning saves no flash or RAM
  here** unless you add a sparse/compressed format.
- **float16 and dynamic-range give nothing.** Dynamic-range produces exactly
  the float32 file because every weight tensor has fewer than 1,024 elements,
  so the converter skips them. float16 adds DEQUANTIZE ops and TFLite-Micro
  cannot allocate it ("failed to allocate tensors").
- **Recommended for the ESP32: INT8 PTQ** (`p1_int8_ptq.h`), chosen on the DS1 hold-out.

### P2, binary AAMI, seed 3 (exported because it is the strongest binary model)

| Variant | B | t=0.5 Se / PPV / F1 / Acc % | t=0.77 (hold-out opt) Se / Sp / PPV / F1 / Acc % | AUC | hold-out F1 @0.77 |
|---|---|---|---|---|---|
| Keras / float32 / dynamic | 7,940 | 95.69 / 52.76 / 68.01 / 90.13 | 91.97 / 93.83 / 64.73 / 75.98 / 93.62 | 0.9749 | 91.27 |
| float16 | 6,900 | 95.71 / 52.73 / 68.00 / 90.12 | 92.00 / 93.83 / 64.74 / 76.00 / 93.63 | 0.9749 | 91.27 |
| INT8 PTQ | 7,024 | 94.55 / 55.58 / 70.01 / 91.11 | 88.89 / 94.59 / 66.92 / 76.35 / 93.96 | 0.9724 | 90.36 |
| **INT8 QAT** | 6,912 | 95.49 / 52.97 / 68.14 / 90.20 | 91.89 / 93.88 / 64.91 / 76.08 / 93.66 | 0.9733 | **91.27** |
| 50 %-pruned + INT8 | 7,024 | 92.08 / 54.75 / 68.67 / 90.78 | 83.07 / 94.95 / 66.97 / 74.16 / 93.65 | 0.9600 | 87.82 |

TFLite-Micro reproduces these as well (QAT at t=0.77: Se 92.06, F1 76.10).
For P2, QAT is the best int8 variant on the hold-out, so the P2 meta header
targets `p2_int8_qat`. P2's PTQ representative set contained an RR outlier,
which made the RR input scale coarse (0.048). It still worked, but this is
the reason QAT wins here.

## 4. Edge metrics (P1; P2 is nearly identical)

| Variant | Flash (.tflite) | TFLM tensor arena (measured, 64-bit host) | Activation peak (liveness est.) | Weights/const | Ops | Host latency / beat |
|---|---|---|---|---|---|---|
| float32 | 8,060 B | 3,056 B | 376 B | 4,596 B | CONV_2D, FULLY_CONNECTED×4, RESHAPE×2, CONCATENATION, SOFTMAX | ~1–4 µs |
| INT8 PTQ | 7,072 B | 3,376 B | 94 B | 1,359 B | same | ~1–2.5 µs |
| INT8 QAT | 6,944 B | 3,072 B | 94 B | 1,359 B | + QUANTIZE | ~1–2.5 µs |
| pruned+INT8 | 7,072 B | 3,376 B | 94 B | 1,359 B | same | ~1–2.5 µs |
| float16 | 6,964 B | does not run on TFLM | – | – | + DEQUANTIZE | – |

- Parameters: 1,143 (P1) and 1,113 (P2) in the exported graph, after folding
  BN into the output Dense. The Keras models have 1,195 and 1,165.
- The arena was measured with TFLM's RecordingMicroAllocator. Most of it is
  interpreter bookkeeping: NodeAndRegistration, TfLiteEvalTensor and
  persistent quantisation buffers. These structs hold pointers, so they
  shrink on the 32-bit ESP32-S3; the 3.4 KB figure is an upper bound.
  `P1_TENSOR_ARENA_BYTES` = 4,400 includes 1 KB of head-room.
- Most of the ~7 KB flash is flatbuffer metadata. The int8 weights are only
  about 1.4 KB.
- Latency is the TFLite interpreter on the box CPU, 1 thread, mean of
  `invoke()` over 5,000 DS2 beats. It ranged from 1 to 4 µs between runs
  because of load from other agents. This is **not** representative of the MCU.
- **Still to measure on a real ESP32-S3:**
  - `invoke()` latency with ESP-NN kernels; expect tens of µs.
  - Real `arena_used_bytes()`.
  - Energy per beat (current probe while invoking).
  - The latency of the full pipeline: resampling, R-peak detection, RR ring
    buffer, and the one-beat post-RR delay.
  - Whether int8 numerics with ESP-NN match the TFLM reference kernels.
    Tolerance: ≤0.05 % argmax disagreement, as measured here between TFLite and TFLM.

## 5. Bugs found and decisions

1. **PTQ calibration crashed** ("CONV_2D input dims 3 != 4", later
   "RESHAPE 4 != 32"). The TFLite graph orders its inputs `[rr_in, ecg_in]`,
   but `src/export.py` fed the representative data as `[ecg, rr]`.
   - Fixed in `src/export.py`: `_input_order()` reads the order from the converted model.
   - Firmware must use the `*_IN_*_INDEX` macros, not the Keras order.
   - This bug may be the real cause of the ">10 % PTQ drop" story. With
     correct calibration, PTQ loses nothing.
2. **int8 input quantisation in `evaluate_tflite` truncated instead of
   rounding** (`astype(int8)`). Fixed with `np.round`.
3. **tfmot 0.8 cannot QAT Conv1D or a standalone BatchNorm.** tfmot also
   needs legacy tf_keras, which does not exist for TF 2.22rc0 / Py 3.14.
   - Created `.venv-export` (Py 3.12, TF 2.19.1, tf-keras 2.19, tfmot 0.8.0,
     tflite-micro) with `1DCNN/export/setup_venv_export.sh`.
   - Do **not** install tensorflow-cpu next to tensorflow in that venv: it
     causes a protobuf FATAL.
   - Weights are moved across through an npz.
   - The model is rebuilt as an equivalent Conv2D(1×32) graph with BN folded
     exactly into the output Dense. Max |Δp| vs the Keras-3 model is 2e-6.
4. **Dynamic batch dimension added SHAPE/STRIDED_SLICE/PACK ops.** Conversion
   now uses a batch-1 concrete function (`_converter`), which leaves 5 op types.
5. **RR features have huge outliers** (up to ~100 in DS1-train, from long
   gaps). They made QAT's input scale 0.40, which crushed RR to a few levels
   and gave 79.7 % accuracy.
   - QAT and pruning fine-tuning now clip RR at the DS1-train 99.9th
     percentile (2.01).
   - On the device, int8 saturation does the same clipping.
6. QAT and pruning fine-tuning now use the float model's balanced class
   weights and keep the epoch with the best DS1 hold-out macro-F1. For
   pruning, only epochs after the target sparsity is reached are eligible.
   Pruning runs for 100 epochs: at 20–30 epochs hold-out F1 was still
   recovering. Without these changes, fine-tuning drifted (pruned hold-out
   mF1 was 0.74).
7. Not fixed (another agent owns these files):
   - `src/metrics.multiclass_metrics` warns "AUC computation failed: index 1
     is out of bounds" for k=2. This is cosmetic; my binary AUCs come from
     `binary_metrics`.
   - `run_experiment --p2/--p3` ignores `binary.threshold_strategy` and
     reports t=0.5 only.
   - The CLI only computes the collapsed-P1 comparison when `--p1` is also passed.
   - The P1 template for sub-class 'S' has L2 norm 0, because DS1-train has
     no 'S' beats.

## Reproduce
See the "Reproduce" section of `1DCNN/export/README.md`. It has the exact
commands: setup venv → run_p2p3_paperdefault.sh → train_binary_fullds1.py →
summarize_p2p3.py → prepare_data → dump_keras_weights → export_tflite →
tflm_check → make_tables → regen_headers → gen_meta_header.

## Files
Created on the box (`$ROOT`):

**Scripts, venv setup and inputs**
- `.venv-export/` — the export venv.
- `1DCNN/export/{setup_venv_export.sh, run_p2p3_paperdefault.sh, configs/p2p3_paperdefault.yaml}`
- `1DCNN/export/{train_binary_fullds1.py, summarize_p2p3.py, prepare_data.py, dump_keras_weights.py}`
- `1DCNN/export/{export_tflite.py, tflm_check.py, make_tables.py, regen_headers.py, gen_meta_header.py, README.md}`
- `1DCNN/export/weights/{P1_fullDS1_seed6,P2_fullDS1_seed3}.npz`
- `1DCNN/export/.cache/*.npz` — data caches.

**Results and exported models**
- `1DCNN/export/p2p3_paperdefault/`, `1DCNN/export/p2p3_fullds1/summary.json`, `1DCNN/export/p2p3_summary.{md,json}`, `1DCNN/export/logs/*`
- `1DCNN/export/P1/` and `1DCNN/export/P2/`, each with:
  - 6 `.tflite` files and 6 `.h` model headers,
  - `p{1,2}_model_meta.h`, `report.json`, `tflm_report.json`, `tables.md`, `proba_ds2_*.npy`.

**Trained models**
- `HasanML/paper3/results/cache/P2_dedicated_seed{0-9}.{keras,pkl}`, `P3_dedicated_*`, `P2_fullDS1_*`, `P3_fullDS1_*`

Modified: `HasanML/paper3/src/export.py`. The changes:
- Input-order fix and batch-1 converter.
- Rounding fix.
- class_weight, epochs, lr and rr_clip parameters plus best-hold-out-epoch selection for QAT and pruning.
- `weight_sparsity()`, `tflite_memory_estimate()`.
- `evaluate_tflite(return_proba)` and `alignas(16)` headers in xxd layout.

The old `export_all()` API still works.
