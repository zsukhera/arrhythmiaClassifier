
**DS2, argmax** (TFLite interpreter, batch 1; Δ = vs Keras float)

| Variant | Acc % | OvR acc % | Macro F1 % | Se N | Se S | Se V | PPV N | PPV S | PPV V | ΔAcc pp | ΔmF1 pp | hold-out mF1 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keras | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| float32 | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| float16 | 90.49 | 93.66 | 73.08 | 90.63 | 77.41 | 96.03 | 99.70 | 31.20 | 68.29 | +0.01 | +0.03 | 85.74 |
| dynamic | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| int8_ptq | 90.71 | 93.81 | 74.27 | 90.76 | 85.68 | 92.92 | 99.65 | 33.24 | 70.08 | +0.24 | +1.21 | 85.21 |
| int8_qat | 88.76 | 92.51 | 70.31 | 88.84 | 81.38 | 91.87 | 99.50 | 31.07 | 59.32 | -1.72 | -2.74 | 84.54 |
| keras_pruned | 88.76 | 92.50 | 71.05 | 88.50 | 86.66 | 93.48 | 99.80 | 31.44 | 60.14 | -1.72 | -2.01 | 82.08 |
| pruned_int8 | 89.31 | 92.87 | 71.82 | 89.15 | 86.34 | 93.23 | 99.78 | 32.35 | 61.64 | -1.17 | -1.24 | 82.89 |

**DS2, tauS=0.50** (TFLite interpreter, batch 1; Δ = vs Keras float)

| Variant | Acc % | OvR acc % | Macro F1 % | Se N | Se S | Se V | PPV N | PPV S | PPV V | ΔAcc pp | ΔmF1 pp | hold-out mF1 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keras | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| float32 | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| float16 | 90.49 | 93.66 | 73.08 | 90.63 | 77.41 | 96.03 | 99.70 | 31.20 | 68.29 | +0.01 | +0.03 | 85.74 |
| dynamic | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 99.70 | 31.16 | 68.26 | +0.00 | +0.00 | 85.74 |
| int8_ptq | 90.64 | 93.76 | 74.17 | 90.68 | 85.74 | 92.92 | 99.65 | 33.01 | 70.08 | +0.17 | +1.12 | 84.89 |
| int8_qat | 88.70 | 92.46 | 70.23 | 88.77 | 81.38 | 91.87 | 99.50 | 30.87 | 59.32 | -1.78 | -2.83 | 84.44 |
| keras_pruned | 88.76 | 92.50 | 71.05 | 88.50 | 86.66 | 93.48 | 99.80 | 31.44 | 60.14 | -1.72 | -2.01 | 82.08 |
| pruned_int8 | 89.07 | 92.71 | 71.50 | 88.88 | 86.39 | 93.23 | 99.78 | 31.59 | 61.64 | -1.41 | -1.56 | 82.71 |

**DS2 computed by the TFLite-Micro runtime (reference kernels)**

| Variant | Rule | Acc % | OvR % | Macro F1 % | Se N | Se S | Se V | PPV S | PPV V |
|---|---|---|---|---|---|---|---|---|---|
| float32 | argmax | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 31.16 | 68.26 |
| float32 | tauS=0.50 | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 31.16 | 68.26 |
| dynamic | argmax | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 31.16 | 68.26 |
| dynamic | tauS=0.50 | 90.48 | 93.65 | 73.05 | 90.62 | 77.35 | 96.03 | 31.16 | 68.26 |
| int8_ptq | argmax | 90.70 | 93.80 | 74.24 | 90.75 | 85.74 | 92.89 | 33.23 | 70.02 |
| int8_ptq | tauS=0.50 | 90.63 | 93.76 | 74.16 | 90.67 | 85.79 | 92.89 | 33.01 | 70.02 |
| int8_qat | argmax | 88.77 | 92.51 | 70.34 | 88.86 | 81.44 | 91.65 | 30.82 | 59.88 |
| int8_qat | tauS=0.50 | 88.70 | 92.47 | 70.25 | 88.79 | 81.44 | 91.65 | 30.61 | 59.88 |
| pruned_int8 | argmax | 89.30 | 92.87 | 71.81 | 89.13 | 86.39 | 93.23 | 32.32 | 61.65 |
| pruned_int8 | tauS=0.50 | 89.06 | 92.71 | 71.50 | 88.87 | 86.45 | 93.23 | 31.58 | 61.65 |

**Edge metrics** (params = float graph after BN folding; latency = TFLite interpreter, 1 thread, host CPU, mean/median of invoke() over 5000 DS2 beats)

| Variant | .tflite bytes | TFLM arena bytes (host, 64-bit) | activation peak est. B | const tensor B | ops | latency mean / median µs | TFLM runs? |
|---|---|---|---|---|---|---|---|
| float32 | 8060 | 3056 | 376 | 4596 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 0.93 / 0.82 | yes |
| float16 | 6964 | None | 4896 | 2310 | CONCATENATION, CONV_2D, DEQUANTIZE, FULLY_CONNECTED, RESHAPE, SOFTMAX | 1.08 / 0.87 | no: RuntimeError: TFLM failed to allocate tensors |
| dynamic | 8060 | 3056 | 376 | 4596 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 0.87 / 0.81 | yes |
| int8_ptq | 7072 | 3376 | 94 | 1359 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 1.27 / 1.35 | yes |
| int8_qat | 6944 | 3072 | 94 | 1359 | CONCATENATION, CONV_2D, FULLY_CONNECTED, QUANTIZE, RESHAPE, SOFTMAX | 1.21 / 0.98 | yes |
| pruned_int8 | 7072 | 3376 | 94 | 1359 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 1.20 / 0.92 | yes |

Parameters: 1143 (float graph after folding BN into the output Dense; the trained Keras model has +52 BN params, 26 of them non-trainable stats).
Pruned kernel sparsity: {'rr_dense_0': 0.5, 'conv2d': 0.5, 'rr_dense_1': 0.5, 'rr_dense_2': 0.5, 'output': 0.4942528735632184}
