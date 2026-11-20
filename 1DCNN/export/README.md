# 1DCNN/export — TFLite / TFLite-Micro export of the Farag (2023) tiny 1D-CNN

These are the deployable artefacts for the ESP32-S3. Every number here comes
from DS2. DS2 was used only for the final evaluation: the seed, the
thresholds, the representative data, the QAT/pruning epoch and the
deploy-variant choice were all made on DS1 data.

| Dir | Model | Seed (picked on DS1) | Recommended firmware variant |
|---|---|---|---|
| `P1/` | 3-class N/S/V (official P1: full DS1, balanced weights) | `P1_fullDS1_seed6` (best macro-F1 on the DS1 hold-out, 85.74 %) | `p1_int8_ptq.h` + `p1_model_meta.h` |
| `P2/` | binary Normal vs Arrhythmia (AAMI), improved recipe | `P2_fullDS1_seed3` (best F1 on the DS1 hold-out, 88.80 %) | `p2_int8_qat.h` + `p2_model_meta.h` |

Each dir holds `<TAG>_<variant>.tflite` for every variant (float32, float16,
dynamic, int8_ptq, int8_qat, pruned_int8) and the matching C array
`<tag>_<variant>.h`. It also has `<tag>_model_meta.h` (quantisation
parameters, input order, thresholds, preprocessing constants),
`report.json` (all metrics), `tflm_report.json` (TFLite-Micro arena and
numerics), `tables.md` (human-readable tables) and
`proba_ds2_*.npy` (DS2 probabilities per variant, from both the TFLite and
the TFLM runtimes).

## Using the headers in firmware (ESP-IDF + esp-tflite-micro)

1. Add the component: `idf.py add-dependency "espressif/esp-tflite-micro"`.
   It pulls in TFLM with the ESP-NN optimised int8 kernels.
2. Copy `P1/p1_int8_ptq.h` and `P1/p1_model_meta.h` into `main/`. Only include
   the model header from one `.cpp`, because it defines the array. The array
   is `alignas(16) const` so it stays in flash and is read in place.
3. Code sketch:

```cpp
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "p1_int8_ptq.h"
#include "p1_model_meta.h"

static uint8_t arena[P1_TENSOR_ARENA_BYTES] __attribute__((aligned(16)));

const tflite::Model* model = tflite::GetModel(p1_int8_ptq);
static tflite::MicroMutableOpResolver<6> resolver;          // ops listed in the meta header
resolver.AddConv2D(); resolver.AddFullyConnected(); resolver.AddReshape();
resolver.AddConcatenation(); resolver.AddSoftmax();          // + AddQuantize() for *_int8_qat
static tflite::MicroInterpreter interp(model, resolver, arena, sizeof(arena));
interp.AllocateTensors();                                    // check == kTfLiteOk
// printf("%u\n", interp.arena_used_bytes());               // measure the real arena once

// per beat (after the NEXT R-peak is known, because post-RR needs it):
int8_t* ecg = interp.input(P1_IN_ECG_INDEX)->data.int8;      // NOTE: input 0 is RR, input 1 is ECG
int8_t* rr  = interp.input(P1_IN_RR_INDEX)->data.int8;
for (int k = 0; k < 32; ++k) ecg[k] = q8(d[16 + k], P1_IN_ECG_SCALE, P1_IN_ECG_ZP);
for (int j = 0; j < 4;  ++j) rr[j]  = q8(rr_feat[j], P1_IN_RR_SCALE,  P1_IN_RR_ZP);
interp.Invoke();
const int8_t* q = interp.output(0)->data.int8;               // q[0]=N, q[1]=S, q[2]=V
int cls = (q[1] >= P1_TAU_S_Q) ? 1 : argmax3(q);              // calibrated S rule
// q8(x,s,z) = clamp(lroundf(x / s) + z, -128, 127)
```

**The input order is not the Keras order.** The converter put `rr_in` first
(input 0) and `ecg_in` second (input 1). Always use the `*_IN_*_INDEX`
macros.

### Preprocessing the firmware must reproduce (see `src/preprocess.py`, `src/rr.py`)
1. ECG lead MLII in **mV**, resampled to **128 Hz** (training used
   `resample_poly(16,45)` from 360 Hz, so sampling at 128 Hz directly also works).
2. Take a 64-sample window `x = s[R-32 .. R+31]`, edge-padded at the start
   or end of a recording. Subtract its mean.
3. Derivative: `d[i] = x[i+1] - x[i]` for i < 63, and `d[63] = d[62]`.
   The model input is `d[16..47]` (32 samples).
4. RR features: pre-RR = R_i − R_{i−1}, post-RR = R_{i+1} − R_i (in samples).
   local = causal mean of the last 80 pre-RR values, global = causal mean of
   the last 400 (both include the current one).
   Features = `{pre/local, pre/global, post/local, post/global}`.
   This needs a 400-entry ring buffer and adds one beat of latency.
5. The int8 inputs saturate at the quantisation range. For P1 int8_ptq the
   RR range is about 0–2.02, which works as clipping. The training data has
   rare RR outliers up to ~100 and DS2 up to ~6.

## Reproduce

```bash
cd $ROOT   # /home/i230510/Desktop/arrhythmiaClassifier
1DCNN/export/setup_venv_export.sh                 # .venv-export: py3.12, TF 2.19.1, tf-keras 2.19, tfmot 0.8.0 (+ tflite-micro)
.venv-export/bin/pip install tflite-micro         # (was installed with uv: ~/.local/bin/uv pip install --python .venv-export/bin/python tflite-micro)
E=1DCNN/export; export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1
# P2/P3 training
$E/run_p2p3_paperdefault.sh                       # paper-default recipe (unchanged src CLI, outputs redirected)
.venv/bin/python $E/train_binary_fullds1.py       # improved recipe -> results/cache/P{2,3}_fullDS1_seed*.keras
.venv/bin/python $E/summarize_p2p3.py             # -> p2p3_summary.md/json
# P1 export
.venv/bin/python $E/prepare_data.py p1 p2
.venv/bin/python $E/dump_keras_weights.py HasanML/paper3/results/cache/P1_fullDS1_seed6.keras $E/weights/P1_fullDS1_seed6.npz --eval $E/.cache/data_p1.npz
TF_USE_LEGACY_KERAS=1 .venv-export/bin/python $E/export_tflite.py --tag P1 --weights $E/weights/P1_fullDS1_seed6.npz --data $E/.cache/data_p1.npz --n-classes 3 --tau-s 0.50 --prune-epochs 100
.venv-export/bin/python $E/tflm_check.py --tag P1 --data $E/.cache/data_p1.npz
.venv-export/bin/python $E/make_tables.py P1
.venv-export/bin/python $E/regen_headers.py P1 && .venv-export/bin/python $E/gen_meta_header.py P1 int8_ptq
# P2 export (same, with)
.venv/bin/python $E/dump_keras_weights.py HasanML/paper3/results/cache/P2_fullDS1_seed3.keras $E/weights/P2_fullDS1_seed3.npz --eval $E/.cache/data_p2.npz
TF_USE_LEGACY_KERAS=1 .venv-export/bin/python $E/export_tflite.py --tag P2 --weights $E/weights/P2_fullDS1_seed3.npz --data $E/.cache/data_p2.npz --n-classes 2 --bin-t 0.77 --prune-epochs 100
.venv-export/bin/python $E/tflm_check.py --tag P2 --data $E/.cache/data_p2.npz; .venv-export/bin/python $E/make_tables.py P2
.venv-export/bin/python $E/regen_headers.py P2 && .venv-export/bin/python $E/gen_meta_header.py P2 int8_qat
```

## Why there are two venvs
tfmot (QAT and pruning) only works with legacy `tf_keras`, and there is no
`tf_keras` for TF 2.22rc0 / Python 3.14 in `.venv`. The pipeline works around
this in two steps:
- `dump_keras_weights.py` (`.venv`, Keras 3) writes the weights to an `.npz`.
- `export_tflite.py` (`.venv-export`) rebuilds the network and checks it
  against Keras 3. The maximum probability difference was 2e-6.

The rebuilt graph is numerically the same as the trained one, but it uses
Conv2D(1×32) on a reshaped (1,32,1) input instead of Conv1D. The BatchNorm,
which sits after the ReLU and is a per-channel affine map, is folded exactly
into the first 13 rows of the output Dense. The float TFLite graph lowers
Conv1D to CONV_2D anyway, so the MCU op set does not change.
