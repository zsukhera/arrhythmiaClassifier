# paper4 — KAN-Former-1D: ECG Arrhythmia Teacher Model

**Reference implementation** of a Kolmogorov–Arnold Network (KAN) Transformer for 5-class AAMI arrhythmia classification on the MIT-BIH dataset, using the de Chazal DS1/DS2 inter-patient split.

This model is the **teacher** in a knowledge-distillation pipeline. It produces soft probability labels at temperatures T = {1, 2, 4} that the student (Farag 2023 1D-CNN, `paper3`) uses instead of one-hot labels.

---

## Architecture

```
Beat (256 @ 360 Hz)   Context (1080 @ 360 Hz)   6 RR ratio features
       │                       │                        │
   MS-CA CNN              MS-CA CNN                     │
   (k=3,7,15)             (k=3,7,15)                    │
       │                       │                     RR proj
   32 tokens             135 tokens                   1 token
       └───────────────────────┴────────────────────────┘
                                │
              [CLS] + beat(32) + ctx(135) + RR(1)  =  169 tokens
                                │
                   3× Transformer Layer
                   (pre-norm, 4 heads, GLU-FFN)
                                │
              cat(CLS_out, mean_pool, RR_direct)  →  262-dim
                                │
                     LayerNorm → tanh
                                │
                     KAN Head (262→64→5)
                                │
                          5-class logits
```

**Key design choices:**
- **Multi-scale CNN stem (MS-CA):** three parallel depthwise-separable branches (kernel 3, 7, 15) + SE channel attention. Captures QRS spikes (narrow) and T/P waves (wide) simultaneously.
- **Two aligned views:** Beat window (0.71 s, morphology) + Context window (3.0 s, rhythm). Attention over context tokens lets the model see the compensatory pause after a PVC.
- **6 causal RR ratio features:** pre/local, post/local, pre/global, post/global, pre/post, (post−pre)/local. Ratios only (no absolute RR), so they generalise across patients with different heart rates.
- **KAN classification head:** B-spline KAN layers instead of linear layers. Smooth non-linear activation learned per-input-dimension from gradient descent.
- **Focal loss (γ=2):** Down-weights easy N-beat examples; uses √(1/freq) class weights.
- **EMA (decay=0.999):** All validation and test inference uses EMA weights.

**Parameter count:** ~1.0 M (d_model=128, 3 transformer layers).

---

## Evaluation target (DS2 — inter-patient)

| Metric | Target |
|--------|--------|
| Accuracy | ≥ 95% |
| OvR accuracy | ≥ 97% |
| Se_N | ≥ 98% |
| Se_S | ≥ 70% |
| Se_V | ≥ 90% |
| Macro F1 | ≥ 80% |

See `results/paper_targets.md` for detailed per-class targets and ablation expectations.

---

## Quick start

### Setup

```bash
# 1. Create venv
python -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch (GPU — CUDA 12.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 3. Install other dependencies
pip install -r requirements.txt

# 4. Run tests
python -m pytest tests/ -v
```

### Training (5-class AAMI)

```bash
# CPU (slow)
python -m src.run_experiment --classes 5 --seeds 3

# GPU (RTX / Ampere / Ada)
python -m src.run_experiment --classes 5 --seeds 3 --device cuda --mixed-precision

# Save results to custom directory
python -m src.run_experiment --classes 5 --seeds 3 --device cuda --results-dir results/run1
```

### Distillation export

```bash
# After training, generate soft labels for paper3 student
python -m src.run_experiment --distill --results-dir results/run1
# Output: results/run1/distill/soft_labels_ds2_T{1,2,4}.npy
#         results/run1/distill/soft_labels_ds1_T{1,2,4}.npy
#         results/run1/distill/beat_index_ds2.csv
```

### Ablations

```bash
python -m src.run_experiment --ablations --seeds 3 --device cuda
```

### RR-only baseline

```bash
python -m src.run_experiment --rr-baseline
```

---

## Bash wrapper (from repo root)

```bash
bash 1DCNN/run_paper4.sh tests          # unit tests
bash 1DCNN/run_paper4.sh sanity         # RR LightGBM baseline
bash 1DCNN/run_paper4.sh train          # 3-seed CPU training
bash 1DCNN/run_paper4.sh gpu            # 3-seed GPU training + distillation
bash 1DCNN/run_paper4.sh ablations      # all 7 ablations
bash 1DCNN/run_paper4.sh distill        # export soft labels (requires trained model)
```

---

## File structure

```
paper4/
├── configs/
│   └── default.yaml          # all hyperparameters
├── src/
│   ├── data.py               # MIT-BIH loading, AAMI map, DS1/DS2 split
│   ├── preprocess.py         # bandpass filter, beat/context window extraction
│   ├── rr.py                 # 6 causal RR ratio features
│   ├── augment.py            # amplitude scale, noise, baseline wander, time shift/stretch
│   ├── kan.py                # B-spline KAN layer (efficient-kan formulation)
│   ├── model.py              # KANFormer: MS-CA CNN + Transformer + KAN head
│   ├── losses.py             # FocalLoss, WeightedCE, LabelSmoothing
│   ├── train.py              # EMA, cosine LR, AMP, early stopping, train loop
│   ├── calibrate.py          # temperature scaling (Guo et al. 2017) + ECE
│   ├── metrics.py            # accuracy, OvR, macro F1, per-class Se/PPV/F1
│   ├── distill_export.py     # soft labels, ONNX export, TorchScript export
│   └── run_experiment.py     # CLI entry point
├── tests/
│   └── test_pipeline.py      # 34 unit tests
├── results/
│   └── paper_targets.md      # target metrics and ablation expectations
├── colab_paper4.ipynb        # end-to-end Colab notebook (CPU + GPU)
└── requirements.txt
```

---

## Google Colab

Open `colab_paper4.ipynb`. It handles:
1. Detecting Colab and mounting Google Drive
2. Installing PyTorch extras and other deps
3. Enabling GPU + mixed precision if available
4. Downloading MIT-BIH data via `wfdb`
5. Running tests, sanity check, and training
6. Generating distillation soft labels

---

## Knowledge distillation pipeline

```
paper4 teacher (KAN-Former)
   └── soft_labels_{split}_T{1,2,4}.npy
           ↓
paper3 student (Farag 2023 1D-CNN)
   └── train with: L = α·CE(pred, hard) + (1-α)·T²·KL(teacher_soft || student_soft)
```

The soft labels are framework-agnostic `.npy` files, so the Keras student does not need PyTorch installed.

---

## References

1. Z. Liu et al., "KAN: Kolmogorov–Arnold Networks," arXiv:2404.19756, 2024.
2. Wang et al., "KAN-Former-1D: Knowledge-Augmented Network for ECG," 2026 (adapted).
3. G. Hinton et al., "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015.
4. P. de Chazal et al., "Automatic classification of heartbeats using ECG morphology and heartbeat interval features," IEEE TBME, 2004.
5. M. M. Farag, "A Tiny Matched Filter-Based CNN for Inter-Patient ECG Classification," Sensors 2023.
6. C. Guo et al., "On Calibration of Modern Neural Networks," ICML 2017.
