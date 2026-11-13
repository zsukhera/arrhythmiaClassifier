# 1DCNN Sensitivity Experiments

Five targeted experiments to improve Se_S (sensitivity for SVEB / class S) of the
Farag 2023 1D-CNN replication (paper3).

**Baseline:** Se_S ≈ 35%, Se_V ≈ 87%  
**Paper target:** Se_S = 85.3%, Se_V = 89.7%

---

## Prerequisites

1. **Paper3 environment must be set up** on the machine where you run these experiments.
   The paper3 package lives at `HasanML/paper3/` relative to the FYP root.

2. **Data must be available.** The first run of any experiment preprocesses the MIT-BIH
   data (~30 s) and caches results to `.cache/` in this directory. Subsequent runs are fast.

3. For **Exp1 and Exp2**, you need trained P1 models from a previous run:
   ```
   # From HasanML/paper3/
   python -m src.run_experiment --p1 --class-weight balanced
   ```
   This saves `P1_seed{0..9}.keras` to `results/cache/`.

4. **Exp3-5 retrain from scratch** — no pre-trained models needed.

---

## Running from the GPU lab machine

All commands below should be run from `HasanML/paper3/` on the GPU machine:

```bash
# SSH to GPU lab
ssh i230510@100.124.204.76

cd ~/Desktop/arrhythmiaClassifier

# Activate virtual environment
source .venv/bin/activate   # or: conda activate <env>
```

> **Path note:** The experiments directory is at `1DCNN/1DCNN experiments/`.
> The scripts add `HasanML/paper3` to `sys.path` automatically, so they can be
> run directly with `python`:
>
> ```bash
> cd "1DCNN/1DCNN experiments"
> python exp1_threshold.py
> ```

---

## Experiment Descriptions

### Exp 1 — Threshold Calibration (`exp1_threshold.py`)
**No retraining.** Requires P1 models.

The model's argmax decision implicitly uses τ_S = 0.33.  
Since DS1-train has only 149 S beats, P(S) is systematically under-confident.  
This experiment sweeps τ_S ∈ [0.05, 0.50], picks the value maximising F1_S on
DS1-val, then locks it in and evaluates on DS2.

```bash
python exp1_threshold.py
python exp1_threshold.py --seeds 0 1 2 3 4   # use only 5 of the 10 models
```

Expected gain: Se_S may improve from ~35% to ~50-60% at some cost to Sp_N.

---

### Exp 2 — Model Ensemble (`exp2_ensemble.py`)
**No retraining.** Requires P1 models.

Average softmax probabilities across all available seeds (soft voting), then
apply threshold calibration. Combines the variance-reduction of ensembling with
the bias correction of Exp1.

```bash
python exp2_ensemble.py
python exp2_ensemble.py --seeds 0 1 2 3 4
```

Expected gain: 2-5 pp Se_S improvement over Exp1 alone.

---

### Exp 3 — Data Augmentation (`exp3_augmentation.py`)
**Retrains from scratch.** ~15 min per seed on CPU; ~3 min on GPU.

Synthetically expands S beats (149 → ~1,043) using:
- Gaussian noise addition
- Amplitude scaling ×0.85 and ×1.15
- Time shifts ±3 samples
- Noise + scale combination

Retrains with `class_weight=balanced`.

```bash
python exp3_augmentation.py --seeds 0 1 2
python exp3_augmentation.py --seeds 0 1 2 3 4   # more seeds on GPU
```

Expected gain: moderate Se_S improvement (5-15 pp) due to more S training signal.

---

### Exp 4 — Focal Loss (`exp4_focal_loss.py`)
**Retrains from scratch.** ~15 min per seed/gamma on CPU.

Replaces categorical cross-entropy with focal loss FL(p_t) = -(1-p_t)^γ log(p_t).
Tests γ ∈ {1, 2, 3}. The (1-p_t)^γ term down-weights easy N examples and forces
the model to focus on hard-to-classify S beats.

```bash
python exp4_focal_loss.py --seeds 0 1 2
python exp4_focal_loss.py --seeds 0 1 2 --gammas 2.0   # just gamma=2
```

Expected gain: moderate improvement in Se_S, potentially better calibrated
confidence scores than class weights.

---

### Exp 5 — Full DS1 Training (`exp5_full_ds1.py`)
**Retrains from scratch.** Most impactful experiment.

Combines DS1_TRAIN + DS1_VAL (944 S beats total, +534% vs 149).
Uses a 15% random beat-level hold-out from the combined set for early stopping.
DS2 is never touched.

```bash
python exp5_full_ds1.py --seeds 0 1 2
python exp5_full_ds1.py --seeds 0 1 2 --val-frac 0.10   # smaller val hold-out
```

Expected gain: largest Se_S improvement of all experiments (target: >70%).

---

## Run All Experiments

```bash
# All 5 experiments, 3 seeds for exp3-5:
python run_all.py

# Smoke test (1 seed only):
python run_all.py --seeds 0

# Only run specific experiments:
python run_all.py --only 5         # just full DS1
python run_all.py --only 1 2       # just threshold + ensemble (no retraining)
python run_all.py --only 3 4 5     # just the retrain experiments

# Skip exp1/exp2 if no P1 models exist:
python run_all.py --skip-no-model
```

The final output is a comparison table:

```
----------------------------------------------------------------------
Method                              Acc%  Se_N%  Se_S%  Se_V%  macF1%
----------------------------------------------------------------------
Baseline P1 (mean 10 seeds)        97.11  99.14  35.01  86.71  73.620
Exp1: threshold calibration        96.xx  xx.xx  xx.xx  xx.xx  xx.xxx
Exp2: soft ensemble + threshold    96.xx  xx.xx  xx.xx  xx.xx  xx.xxx
Exp3: augmentation (3 seeds)       96.xx  xx.xx  xx.xx  xx.xx  xx.xxx
Exp4: focal loss gamma=2 (3 seeds) 96.xx  xx.xx  xx.xx  xx.xx  xx.xxx
Exp5: full DS1 (3 seeds)           96.xx  xx.xx  xx.xx  xx.xx  xx.xxx
----------------------------------------------------------------------
Target (Farag 2023, Table 4):       97.8   n/a   85.3   89.7    n/a
----------------------------------------------------------------------
```

---

## Running on the GPU (recommended for exp3-5)

```bash
# On GPU machine, in HasanML/paper3/ with venv activated:
cd "1DCNN/1DCNN experiments"
nohup python exp5_full_ds1.py --seeds 0 1 2 3 4 > exp5.log 2>&1 &
tail -f exp5.log
```

Or add GPU flags by passing them through the paper3 config:

```python
# In each exp script you can override cfg before training:
cfg["compute"]["device"] = "cuda"
cfg["compute"]["mixed_precision"] = True
```

---

## File Structure

```
1DCNN experiments/
├── README.md           ← this file
├── utils.py            ← shared: data loading, model loading, metrics printing
├── exp1_threshold.py   ← threshold calibration (no retraining)
├── exp2_ensemble.py    ← model ensemble + threshold (no retraining)
├── exp3_augmentation.py← S-beat augmentation + retrain
├── exp4_focal_loss.py  ← focal loss γ=1,2,3 + retrain
├── exp5_full_ds1.py    ← full DS1 training + retrain
├── run_all.py          ← master runner + comparison table
└── .cache/             ← auto-created: preprocessed data and MF kernel npz files
```
