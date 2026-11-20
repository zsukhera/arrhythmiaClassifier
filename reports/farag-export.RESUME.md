# farag-export RESUME notes (updated 2026-11-18 13:35)

Owned: 1DCNN/export/, HasanML/paper3/src/export.py, results/cache/P2_*, P3_*. Envs: .venv (Keras3, training), .venv-export (py3.12 TF2.19 tf_keras tfmot 0.8; created by 1DCNN/export/setup_venv_export.sh — NOTE install tensorflow (not tensorflow-cpu): both together crash with protobuf FATAL).
Thread env for every job: CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=3 TF_NUM_INTRAOP_THREADS=3 TF_NUM_INTEROP_THREADS=1

## Done
- P2/P3 paper-default (DS1-train/DS1-val, cw none, val_loss ES): 1DCNN/export/run_p2p3_paperdefault.sh -> log 1DCNN/export/logs/p2p3_paperdefault.log, outputs 1DCNN/export/p2p3_paperdefault/, models results/cache/P2_dedicated_seed*.keras, P3_dedicated_*. (Collapsed-P1 comparison not run there: CLI only does it with --p1.)
- Seed selection for P1 export (DS1 only): max hold-out macro-F1 from P1_fullDS1_seed*.pkl metrics_val -> seed 6 (0.8574). (seed 6 happens to be worst DS2 Se_S 77.35% — report honestly.)
- 1DCNN/export/prepare_data.py p1 -> .cache/data_p1.npz (exact official hold-out, verified maxdiff 0 vs pkl proba_val)
- dump_keras_weights.py -> weights/P1_fullDS1_seed6.npz (Keras3 -> npz, for tf_keras rebuild)
- src/export.py edited: class_weight/epochs/lr params for QAT+pruning, weight_sparsity(), evaluate_tflite return_proba+latency, int8 input ROUNDING fix (was truncation), tflite_memory_estimate(), alignas(16) in headers.

## Running (13:50)
- PID 44519: train_binary_fullds1.py (P2 then P3, ~1.5 min/seed), log 1DCNN/export/logs/p2p3_fullds1.log; done when log has "wrote .../p2p3_fullds1/summary.json".
- P1 export DONE: 1DCNN/export/P1/ (tflite, .h, report.json, tflm_report.json, tables.md via make_tables.py P1). Pruning used --prune-epochs 100.

## Findings / bugs
- First P1 export attempt: float32/16/dynamic OK (9532/8576/9532 B; dynamic == float32 size because all weight tensors < 1024 elems so converter does not quantize them). int8_ptq FAILED (TF2.19 calibrator: CONV_2D input dims 3!=4 with Conv1D). QAT FAILED (tfmot does not support Conv1D).
- FIXED (root causes): (1) PTQ crash = representative dataset fed [ecg,rr] but TFLite input order is [rr_in, ecg_in] -> export.py _input_order(). (2) QAT: tfmot lacks Conv1D and standalone BN -> export_tflite.py uses Conv2D form + BN folded exactly into output Dense (params 1195->1143). (3) dynamic batch -> SHAPE/STRIDED_SLICE/PACK ops; export.py _converter() uses batch-1 concrete fn. (4) QAT RR input scale 0.40 due to RR outliers (max ~100) -> rr_clip = DS1-train 99.9 pct (2.01) for QAT/pruning fine-tune. (5) QAT/pruning now keep best hold-out macro-F1 epoch.
- Results so far (DS2 argmax, seed6): keras 90.48 acc/mF1 73.05/SeS 77.35; int8_ptq 90.71/74.27/85.68 (hold-out mF1 .852 vs .857); QAT 88.76/70.31; pruned(30ep) 87.70/68.32.
- Old note: export_tflite.py builds an equivalent Conv2D graph (Reshape(1,32,1) -> Conv2D(13,(1,32)) -> BN -> Reshape(13)) and copies weights.

- P2 export running: PID 1206510 (export+tflm+tables), log 1DCNN/export/logs/export_P2.log, seed 3 (best hold-out F1 88.80), bin-t 0.77 (seed-3 hold-out opt). Done when 1DCNN/export/P2/tables.md exists.
- P2 improved recipe results in logs/p2p3_fullds1.log (ens t=0.82: Se 90.2 PPV 74.6 F1 81.7 AUC .978).

- P2 export DONE (1DCNN/export/P2/, tables.md). Headers regenerated+verified (regen_headers.py). Meta headers: P1 -> int8_ptq, P2 -> int8_qat (each = best DS1 hold-out among int8 variants).

## Remaining steps
1. (running) rerun:
   TF_USE_LEGACY_KERAS=1 nohup .venv-export/bin/python 1DCNN/export/export_tflite.py --tag P1 --weights 1DCNN/export/weights/P1_fullDS1_seed6.npz --data 1DCNN/export/.cache/data_p1.npz --n-classes 3 --tau-s 0.50 > 1DCNN/export/logs/export_P1.log 2>&1 &
   -> 1DCNN/export/P1/{*.tflite,*.h,report.json}
2. (DONE) When P2 fullDS1 done: pick P2 seed by hold-out (summary.json per_seed holdout_f1_05), .venv/bin/python 1DCNN/export/prepare_data.py p2, dump weights, export with --tag P2 --n-classes 2 --bin-t <hold-out threshold>.
3. README 1DCNN/export/README.md (firmware usage), reports/farag-export.md.

4. After PID 44519 finishes (P3): .venv/bin/python 1DCNN/export/summarize_p2p3.py -> p2p3_summary.md

## STATUS 2026-11-19: ALL DONE. Final report: reports/farag-export.md
