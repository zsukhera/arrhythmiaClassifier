
**DS2, t=0.50** (TFLite interpreter, batch 1)

| Variant | Acc % | Se % | Sp % | PPV % | F1 % | AUC | ΔAcc pp | ΔF1 pp | hold-out F1 % |
|---|---|---|---|---|---|---|---|---|---|
| keras | 90.13 | 95.69 | 89.44 | 52.76 | 68.01 | 0.9749 | +0.00 | +0.00 | 88.80 |
| float32 | 90.13 | 95.69 | 89.44 | 52.76 | 68.01 | 0.9749 | +0.00 | +0.00 | 88.80 |
| float16 | 90.12 | 95.71 | 89.43 | 52.73 | 68.00 | 0.9749 | -0.01 | -0.01 | 88.80 |
| dynamic | 90.13 | 95.69 | 89.44 | 52.76 | 68.01 | 0.9749 | +0.00 | +0.00 | 88.80 |
| int8_ptq | 91.11 | 94.55 | 90.69 | 55.58 | 70.01 | 0.9724 | +0.99 | +2.00 | 88.31 |
| int8_qat | 90.20 | 95.49 | 89.55 | 52.97 | 68.14 | 0.9733 | +0.08 | +0.12 | 87.77 |
| keras_pruned | 90.91 | 92.17 | 90.76 | 55.13 | 69.00 | 0.9617 | +0.79 | +0.98 | 86.76 |
| pruned_int8 | 90.78 | 92.08 | 90.62 | 54.75 | 68.67 | 0.9600 | +0.66 | +0.65 | 86.44 |

**DS2, t=0.77** (TFLite interpreter, batch 1)

| Variant | Acc % | Se % | Sp % | PPV % | F1 % | AUC | ΔAcc pp | ΔF1 pp | hold-out F1 % |
|---|---|---|---|---|---|---|---|---|---|
| keras | 93.62 | 91.97 | 93.83 | 64.73 | 75.98 | 0.9749 | +0.00 | +0.00 | 91.27 |
| float32 | 93.62 | 91.97 | 93.83 | 64.73 | 75.98 | 0.9749 | +0.00 | +0.00 | 91.27 |
| float16 | 93.63 | 92.00 | 93.83 | 64.74 | 76.00 | 0.9749 | +0.00 | +0.02 | 91.27 |
| dynamic | 93.62 | 91.97 | 93.83 | 64.73 | 75.98 | 0.9749 | +0.00 | +0.00 | 91.27 |
| int8_ptq | 93.96 | 88.89 | 94.59 | 66.92 | 76.35 | 0.9724 | +0.34 | +0.37 | 90.36 |
| int8_qat | 93.66 | 91.89 | 93.88 | 64.91 | 76.08 | 0.9733 | +0.04 | +0.10 | 91.27 |
| keras_pruned | 93.55 | 82.51 | 94.92 | 66.66 | 73.74 | 0.9617 | -0.07 | -2.24 | 89.22 |
| pruned_int8 | 93.65 | 83.07 | 94.95 | 66.97 | 74.16 | 0.9600 | +0.03 | -1.83 | 87.82 |

**DS2 computed by the TFLite-Micro runtime**

| Variant | Rule | Acc % | Se % | Sp % | PPV % | F1 % | AUC |
|---|---|---|---|---|---|---|---|
| float32 | t=0.50 | 90.13 | 95.69 | 89.44 | 52.76 | 68.01 | 0.9749 |
| float32 | t=0.77 | 93.62 | 91.97 | 93.83 | 64.73 | 75.98 | 0.9749 |
| dynamic | t=0.50 | 90.13 | 95.69 | 89.44 | 52.76 | 68.01 | 0.9749 |
| dynamic | t=0.77 | 93.62 | 91.97 | 93.83 | 64.73 | 75.98 | 0.9749 |
| int8_ptq | t=0.50 | 91.10 | 94.57 | 90.67 | 55.53 | 69.97 | 0.9724 |
| int8_ptq | t=0.77 | 93.92 | 88.96 | 94.53 | 66.73 | 76.26 | 0.9724 |
| int8_qat | t=0.50 | 90.19 | 95.56 | 89.53 | 52.93 | 68.12 | 0.9735 |
| int8_qat | t=0.77 | 93.66 | 92.06 | 93.85 | 64.86 | 76.10 | 0.9735 |
| pruned_int8 | t=0.50 | 90.80 | 92.06 | 90.64 | 54.79 | 68.70 | 0.9599 |
| pruned_int8 | t=0.77 | 93.62 | 83.02 | 94.93 | 66.86 | 74.07 | 0.9599 |

**Edge metrics** (params = float graph after BN folding; latency = TFLite interpreter, 1 thread, host CPU, mean/median of invoke() over 5000 DS2 beats)

| Variant | .tflite bytes | TFLM arena bytes (host, 64-bit) | activation peak est. B | const tensor B | ops | latency mean / median µs | TFLM runs? |
|---|---|---|---|---|---|---|---|
| float32 | 7940 | 3056 | 376 | 4476 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 2.41 / 2.39 | yes |
| float16 | 6900 | None | 4776 | 2250 | CONCATENATION, CONV_2D, DEQUANTIZE, FULLY_CONNECTED, RESHAPE, SOFTMAX | 2.44 / 2.42 | no: RuntimeError: TFLM failed to allocate tensors |
| dynamic | 7940 | 3056 | 376 | 4476 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 2.43 / 2.41 | yes |
| int8_ptq | 7024 | 3376 | 94 | 1326 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 2.53 / 2.50 | yes |
| int8_qat | 6912 | 3072 | 94 | 1326 | CONCATENATION, CONV_2D, FULLY_CONNECTED, QUANTIZE, RESHAPE, SOFTMAX | 2.58 / 2.55 | yes |
| pruned_int8 | 7024 | 3376 | 94 | 1326 | CONCATENATION, CONV_2D, FULLY_CONNECTED, RESHAPE, SOFTMAX | 2.44 / 2.41 | yes |

Parameters: 1113 (float graph after folding BN into the output Dense; the trained Keras model has +52 BN params, 26 of them non-trainable stats).
Pruned kernel sparsity: {'rr_dense_0': 0.5, 'conv2d': 0.5, 'rr_dense_1': 0.5, 'rr_dense_2': 0.5, 'output': 0.5}
