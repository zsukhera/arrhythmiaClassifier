# KAN-Former-1D — Target Metrics (paper4)

**Evaluation protocol:** de Chazal DS1/DS2 inter-patient split. DS2 is the held-out test set (22 patients, never seen during training). All numbers are on DS2.

---

## Primary 5-class AAMI targets

| Metric | Target | Majority baseline (predict-all-N) | Notes |
|--------|--------|-----------------------------------|-------|
| Plain accuracy | ≥ 95.0% | ~90% | High because N = majority |
| OvR accuracy (1-2e/K) | ≥ 97.0% | ~96% | More informative than plain |
| Macro sensitivity | ≥ 75.0% | 20% | Captures minority classes |
| Macro F1 | ≥ 80.0% | — | Harmonic mean Se/PPV |
| Macro AUC | ≥ 97.0% | — | OvR |

### Per-class targets

| Class | n (DS2 est.) | Se target | PPV target | Notes |
|-------|-------------|-----------|------------|-------|
| N (normal) | ~40,000 | ≥ 98% | ≥ 97% | Very common; must stay high |
| S (SVEB) | ~700 | ≥ 70% | ≥ 60% | Key difficulty: morphologically ambiguous |
| V (VEB) | ~3,500 | ≥ 90% | ≥ 85% | High clinical importance |
| F (fusion) | ~200 | ≥ 50% | ≥ 40% | Very rare; difficult even for cardiologists |
| Q (unknown/paced) | ~15 | ≥ 60% | — | Very rare; not critical |

**Key benchmark comparisons:**
- Farag (2023) paper3: Se_S ≈ 60-70%, Se_V ≈ 85-90% (much smaller 1D-CNN)
- de Chazal (2004): Se_S ≈ 75%, Se_V ≈ 77% (morphology + LDA, 2004 baseline)
- KAN-Former target: Se_S ≥ 70%, Se_V ≥ 90% using context + RR features

---

## Ablation expected outcomes

Ablations are labelled A1–A6b. Each is trained with 3 seeds. Expected direction of change relative to full model:

| ID | Config | Expected Δ accuracy | Expected Δ Se_S | Notes |
|----|--------|---------------------|-----------------|-------|
| A1 | MLP head (replace KAN) | ≈ 0 | -2 to +1% | KAN contribution is head; similar accuracy |
| A2 | No Transformer | -1 to -3% | -5 to -10% | Attention is key for rhythm context |
| A3 | No context window | -1 to -2% | -5 to -8% | Context helps S detection (rhythm cue) |
| A4 | No RR features | -0.5 to -2% | -3 to -6% | RR important for S/V; context partially compensates |
| A5 | Single kernel k=7 only | -1 to -2% | -2 to -5% | Multi-scale captures P wave better |
| A6a | KAN grid_size=3 | ≈ 0 to -1% | -1 to -3% | Coarser spline grid |
| A6b | KAN grid_size=8 | ≈ 0 to +0.5% | ≈ 0% | Finer grid; marginal at added cost |

---

## Training targets

| Metric | Expected |
|--------|----------|
| Final DS1-val macro F1 | ≥ 75% (early stopping criterion) |
| Convergence (epochs) | 60–120 (cosine schedule, 50-epoch warmup) |
| Best seed vs. worst seed | < 2% spread in Se_V |
| Temperature T* (calibration) | 1.2 – 2.5 |
| ECE before calibration | < 0.10 |
| ECE after calibration | < 0.05 |

---

## Distillation soft label quality targets

For the soft labels to be useful to the paper3 student:

| Metric | Target |
|--------|--------|
| Entropy at T=1 (teacher conf.) | Mean H > 0.3 nats for minority classes |
| Rank correlation Se_S: teacher→student | Positive after distillation training |
| Student Se_S improvement vs. hard labels | ≥ +3% absolute |

---

## Comparison table (to be filled after training)

| Run | Seeds | Se_N | Se_S | Se_V | Se_F | Acc | OvR | Macro F1 |
|-----|-------|------|------|------|------|-----|-----|----------|
| Full model (default) | 3 | — | — | — | — | — | — | — |
| A1 MLP head | 3 | — | — | — | — | — | — | — |
| A2 no transformer | 3 | — | — | — | — | — | — | — |
| A3 no context | 3 | — | — | — | — | — | — | — |
| A4 no RR | 3 | — | — | — | — | — | — | — |
| A5 single kernel | 3 | — | — | — | — | — | — | — |
| A6a grid=3 | 3 | — | — | — | — | — | — | — |
| A6b grid=8 | 3 | — | — | — | — | — | — | — |
