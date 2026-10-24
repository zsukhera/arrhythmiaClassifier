# Paper 1: ESN + ridge readout for per-subject PVC detection (MIT-BIH)

Replication of the software **"Standard ESN"** baseline from

> X. He, T. Liu, F. Hadaeghi, H. Jaeger, "Reservoir transfer on analog neuromorphic
> hardware," *Proc. 9th IEEE/EMBS NER*, 2019, pp. 1234–1238, doi:10.1109/NER.2019.8716891

with the experimental details from T. Liu, *Harnessing Slow Dynamics in Neuromorphic
Computation*, MSc thesis, Jacobs University Bremen, 2019, arXiv:1905.12116
(Ch. 4.3, Table 4.1). Where the two disagree, the thesis is followed.

Each record gets its own (patient-specific) **binary** classifier, PVC (`V`, label 1)
vs. every other beat (label 0). The classifier is a ridge-regression readout on
leaky-ESN states. There are no handcrafted features.

## Install

```bash
cd HasanML/paper1
source ../../.venv/bin/activate          # or any Python >= 3.10 environment
pip install -r requirements.txt
# optional, for the GPU path (choose the wheel matching your CUDA version):
pip install cupy-cuda12x
```

## Run

```bash
python -m src.run_experiment --config configs/default.yaml   # full run, all 7 records
python -m pytest                                             # unit tests

# quick end-to-end check on one record
python -m src.run_experiment --records 119 --seeds 0 --input spikes --leak 0.99 --results-dir results_119
```

Useful flags: `--input {spikes,raw,both}`, `--leak 0.99 0.01`, `--seeds 0 1 2`,
`--device {auto,cpu,cuda,gpu}`, `--workers N`, `--batch-size N`, `--n-reservoir N`,
`--start-minutes [REC=]MIN` (shift the 10 + 5 min interval, globally or per record).
Every hyperparameter and assumption lives in `configs/default.yaml`.

Records are read from `data/mitdb/`. A missing record is copied from the
repository's `datasets/mitbih/mitdb/` if present, and otherwise downloaded from
PhysioNet with `wfdb`.

### CPU, threads and GPU

| Device | How work is split |
|---|---|
| CPU (`device: cpu`, or `auto` with no CUDA) | Each reservoir run is a job in a **process pool** (`cpu_workers`, default 6; `auto` sizes the pool from core count and free RAM). Each process is pinned to one BLAS thread. The per-time-step recurrence is a Python loop that holds the GIL, so processes scale where threads would not. |
| GPU (`device: cuda` or `gpu`, needs CuPy) | Up to `gpu_batch_size` reservoirs that share a seed are simulated **together** as one batched state matrix. A **thread pool** (`gpu_workers`) feeds the device, so host-side work overlaps GPU kernels. |
| Both | Records are loaded and filtered in a **thread pool**. |

`device: auto` uses CUDA when CuPy can see a device and falls back to CPU
otherwise. The same NumPy-API code runs on both backends.

A full CPU run is 420 reservoir simulations (7 records × 2 inputs × 2 leak rates ×
5 seeds × 3 input scalings). It took **157 min** with 6 processes
on an Intel Core Ultra 7 155H (WSL2, 7.6 GB RAM). The GPU path was not used
there: the machine only has Intel Arc integrated graphics, which CuPy (CUDA) cannot
drive.

Performance notes from profiling on that machine (one full-size run, one core):
about 15 s goes to the per-fold Gram matrices (float32 GEMM, accumulated in
float64), about 29 s to the sparse recurrence (latency-bound sparse
matrix-vector products) and about 4 s to the ridge solves. Aggregate throughput saturates at
about 2.8× one core however many processes run, so 6 workers are faster than 12;
12 workers also pushed the 7.6 GB machine into swap. Batching several reservoirs
per process did not lower the per-run cost on this CPU, so `cpu_batch_size` is 1.
A numba-compiled recurrence was also tried and was slower than SciPy's CSR
kernel.

Every finished reservoir run is saved to `results/cache/` under a hash of its
full specification (hyper-parameters, seed, input signal and labels), so an
interrupted run resumes where it stopped (`--no-cache` disables this).

### Outputs (`results/`)

| File | Content |
|---|---|
| `results_table.md` / `results_table.csv` | Per record and variant: mean ± std over seeds, the paper's Table 4.1 values, and the difference |
| `per_seed.csv` | Every seed: selected hyperparameters, CV F1, test metrics, confusion matrix, beat counts, all-normal baseline |
| `figures/record_<rec>_<input>_leak<a>.png` | 10 s of the test window: ECG, readout output, threshold, true and predicted PVCs |
| `run.log`, `resolved_config.yaml` | Log (channel used, beat counts per class) and the exact configuration |

## Method

1. **Data.** Records 106, 119, 200, 201, 203, 223 and 233. The MLII lead is selected
   by name (channel 0 for all seven; logged). Beat annotations: `V` → 1, other
   beat symbols → 0. Non-beat annotations (`+ ~ | " x [ ] !` …) are dropped.
2. **Split.** Train = the first 10 min, test = 10:00–15:00 (shiftable per record with
   `split.start_minutes_per_record`; the record is cropped before preprocessing). A beat belongs to the
   window that contains its R peak. CV uses 5 contiguous 2-minute folds inside the
   training window, unshuffled.
3. **Preprocessing.** 4th-order Butterworth high-pass at 0.5 Hz, zero-phase,
   then min-max scaling to [0, 1] using the training window's min/max (the test
   window is clipped to [0, 1]).
4. **Encoding.** Delta modulation with threshold 0.1 gives UP and DOWN spike
   channels (`spikes`). The other option feeds the normalised ECG directly (`raw`).
5. **ESN.** N = 768, x(n+1) = (1−a)x(n) + a·tanh(W_in u(n+1) + W x(n) + b).
   W is 10 % dense with normal entries, rescaled to spectral radius 0.9. W_in ~ U[−s, s].
   A 2 s washout is discarded. The reservoir runs continuously through training and test.
6. **Smoothing.** Exponential kernel z(n) = d·z(n−1) + (1−d)·x(n), d = exp(−1/(τ·fs)).
7. **Readout.** Ridge regression to a target that is 1 inside [R−100 ms, R+150 ms)
   of each PVC, with a bias column: W_out = Y Xᵀ(XXᵀ + λI)⁻¹.
8. **Beat decision.** Mean of the readout over each beat window, compared with a
   threshold θ.
9. **Model selection.** (s, τ, λ, θ) are chosen together to maximise the pooled
   out-of-fold F1 of the training beats. Ties go to higher accuracy, then θ closest
   to 0.5. Grids: s ∈ {0.1, 0.5, 1}, τ ∈ {10, 25, 50, 100, 200} ms,
   λ ∈ {1e-8, 1e-6, 1e-4, 1e-2, 1}, θ ∈ {0.10, 0.15, …, 0.90}.
10. **Evaluation.** Per beat, test window only: accuracy, sensitivity, precision, F1,
    specificity, confusion matrix, test beat counts, and the all-normal baseline.
    Results are mean ± std over 5 reservoir seeds.

**Variants reported:** input {spikes, raw} × leak a {0.99, 0.01} × λ {CV-selected,
fixed 1e-6 (thesis value)} × readout weighting {none, class-balanced}. The
**primary** variant is spikes, a = 0.99, λ by CV, no weighting.

### How states are handled

A 768-unit state at 360 Hz is about 1 GB per 15-minute run, so states are never
stored. Pass 1 streams over the training window and accumulates, per CV fold
and per τ, the Gram matrices `ZᵀZ` (all rows and PVC-target rows) and `Zᵀy`.
One eigendecomposition per fold model then gives the ridge solution for every λ,
and the split by class gives the weighted-ridge ablation at no extra cost.
Pass 2 replays the deterministic reservoir, applies every readout, and keeps
only the per-beat mean and max of the output.

### No test leakage

- The high-pass filter runs on the training window **on its own**. The test window
  comes from a separate pass over [0, 15 min), so no training sample depends on a
  later one (a single zero-phase pass would leak backwards across 10:00).
- Normalisation min/max, readout fits, and all hyperparameter selection use
  training samples only.
- The training target only contains PVCs whose R peak is before 10:00.
- Training beats whose window crosses 10:00 or the washout are dropped from CV.
- `tests/test_pipeline.py::test_no_test_window_sample_is_used_in_fitting` corrupts
  every test-window sample and flips every test label. It checks that the readout
  weights and all CV scores stay bit-identical while the test scores change.

## Assumptions where the paper/thesis is unspecified

| Item | Choice here | Note |
|---|---|---|
| Position of the 10 + 5 min interval | Train 0–10 min, test 10–15 min | The thesis only says "an interval of 10 minutes … and the next 5 minutes". The paper's counts suggest other windows; see "Are our test sets the paper's?". |
| High-pass filter | Butterworth, order 4, 0.5 Hz, zero-phase | Configurable (`preprocess.*`). |
| Normalisation scope | Min/max from the training window only; test clipped to [0, 1] | The thesis may have normalised the whole signal. `normalization_scope: full` reproduces that, but it leaks. |
| ESN input (spikes vs raw) | Both are run and reported | The thesis does not say which input the Standard ESN received. |
| Leak-rate convention | a = 0.99 in x ← (1−a)x + a·tanh(·) (primary); a = 0.01 as ablation | "Leakage rate = 0.99" can be read either way. |
| W_in | Uniform [−s, s], s ∈ {0.1, 0.5, 1.0} chosen by CV | |
| Bias b | Uniform [−0.2, 0.2] | Not described in the thesis. |
| Sparsity of W | 10 % non-zero, normal entries, spectral radius 0.9 | |
| Reservoir size | 768 | Matches the hardware reservoir. |
| Washout | First 2 s | |
| Smoothing kernel | Causal exponential, τ ∈ {10, 25, 50, 100, 200} ms by CV | |
| Readout target / beat window | 1 in [R−100 ms, R+150 ms) of each PVC | Also used as the per-beat aggregation window. |
| Per-beat aggregation | Mean of the readout (max is also computed; `readout.aggregation`) | |
| Ridge bias | Bias column is regularised like the other weights | Literal reading of W_out = YXᵀ(XXᵀ+λI)⁻¹. |
| λ | CV over {1e-8 … 1}; fixed 1e-6 also reported | |
| θ | CV over 0.10–0.90, step 0.05 | |
| CV criterion | Pooled out-of-fold F1 over training beats | |
| Seeds | 5 seeds for W, W_in, b | W is shared across records for a given seed. |

## Differences from the original

- **Software ESN only.** Nothing runs on the Dynap-se neuromorphic chip, so only
  the "Standard ESN" rows of Table 4.1 are replicated. The optional spiking-reservoir
  stretch goal (Brian2 LIF + reservoir transfer) is **not** implemented.
- The thesis's exact preprocessing, input encoding of the software baseline, beat
  window, CV criterion and random reservoir are unknown (see the table above).
  Agreement can therefore only be approximate.
- The test-leak precautions above (separate filtering of the training window,
  training-only normalisation) may differ from what the authors did.

## Results

Full tables (every variant, mean ± std over 5 seeds, Δ against the paper):
`results/results_table.md` / `.csv`; per-seed rows with the CV-selected
hyper-parameters: `results/per_seed.csv`. Every number below was computed by
`python -m src.run_experiment --config configs/default.yaml` (157 min, 420
reservoir runs). Nothing was tuned on the test window.

### Primary variant (spike input, a = 0.99, λ by CV, unweighted ridge), test F1 %

| Record | Paper F1 | Ours F1 (mean ± std, 5 seeds) | Δ F1 (pp) | CV F1 (train) | All-normal acc |
|---|---|---|---|---|---|
| 106 | 97.22 | 92.18 ± 1.96 | **−5.04** | 84.3 | 67.47 |
| 119 | 99.55 | 99.64 ± 0.71 | +0.09 | 100.0 | 82.67 |
| 200 | 98.82 | 93.75 ± 1.38 | **−5.07** | 94.9 | 77.95 |
| 201 | 98.57 | 100.00 ± 0.00 | +1.43 | 97.6 | 72.41 |
| 203 | 94.92 | 67.59 ± 0.91 | **−27.33** | 73.1 | 84.40 |
| 223 | 98.58 | 86.56 ± 1.22 | **−12.02** | 69.8 | 78.44 |
| 233 | 99.60 | 99.58 ± 0.14 | −0.02 | 98.9 | 71.96 |

Mean over records: accuracy 95.99 %, F1 91.33 % (paper: 99.11 % / 98.18 %).
The raw-ECG input with a = 0.99 is equivalent on average (mean F1 91.40 %): it is
better on 106, 119 and 203, and worse on 200, 201, 223 and 233.

### Which reading of the ambiguous settings fits the paper better

- **Leak rate:** a = 0.99 (fast, nearly memory-less units, as in our update
  equation) is clearly better than a = 0.01 for both inputs. Mean F1 is 91.3 vs 75.2 %
  (spikes) and 91.4 vs 85.4 % (raw), so a = 0.99 is the reading that matches the paper.
- **Input:** spikes and raw ECG are tied on average. Neither reproduces the paper on
  all records.
- **λ = 1e-6 (thesis value):** much worse than λ chosen by CV (spikes, a = 0.99:
  mean F1 72.0 vs 91.3 %), with large seed-to-seed variance. On our states, 1e-6
  overfits ten minutes of 768-dimensional data. CV picked λ = 1e-2 or 1 for every
  record.
- **Class-balanced (weighted) ridge:** no gain (mean F1 87.3 vs 91.3 %).

### Are our test sets the paper's?

Table 4.1 reports percentages with two decimals, so each row only fits a few whole-number
confusion matrices. Working back from the Standard ESN row gives the number of test PVCs
(P) and test beats (N) that the paper must have used. These can be checked against the
annotations (beat counts from `atr`, R peak inside the window):

| Record | Paper implies P / N | Ours, 10:00–15:00 | 5-min windows (10 s steps) that fit |
|---|---|---|---|
| 106 | 36 / ~160, 72 / ~320 or 108 / ~480 | 121 / 372 | none |
| 119 | 109–111 / 318–350 | **57** / 329 | 18:30–20:00 starts, e.g. 20:00–25:00 (110 / 329) |
| 200 | 170–171 / 424–437 | **101** / 458 | **only 15:00–20:00 (170 / 429)** |
| 201 | 69 / 259–268 | 72 / 261 | 9:20, 9:30, 14:00 starts (close to ours) |
| 203 | 28 / ~161, 56 / ~323 or 84 / ~484 | 85 / 545 | none |
| 223 | 105 / 318–327 | 97 / 450 | none |
| 233 | 124–128 / 426–487 | 143 / 510 | none |

So the paper did **not** test 119 and 200 on 10:00–15:00 (57 vs ~110 PVCs, 101 vs 170),
and for four records no 5-minute window fits at all. The paper's test sets were probably
picked per subject, or not exactly 5 minutes long, or some beats were left out. Also, the
Standard ESN and Dynap-se rows do not fit the same counts for 201, 203, 223 and 233. Per-record
gaps are therefore partly a comparison on different beats, not only a model difference.
(Only the paper's published numbers and the annotations were used for this, never our
test results.)

**Rerun on the windows the counts point to** (`results_paper_windows/`):

```bash
python -m src.run_experiment --records 119 200 --start-minutes 119=10 200=5 \
    --leak 0.99 --input both --results-dir results_paper_windows
```

Record 200: train 5–15 min, test 15–20 min (170 PVC / 259 other). Record 119: train
10–20 min, test 20–25 min (110 / 219). Leak 0.99, λ by CV, unweighted, 5 seeds:

| Record | Input | Accuracy | Sensitivity | Precision | F1 | Paper F1 | Δ F1 (pp) | Δ F1 on 10:00–15:00 |
|---|---|---|---|---|---|---|---|---|
| 119 | spikes | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 | 99.55 | +0.45 | +0.09 |
| 119 | raw | 96.60 ± 7.11 | 89.82 ± 21.26 | 100.00 ± 0.00 | 93.38 ± 14.04 | 99.55 | **−6.17** | +0.45 |
| 200 | spikes | 95.80 ± 0.57 | 89.76 ± 1.74 | 99.62 ± 0.57 | 94.42 ± 0.81 | 98.82 | **−4.40** | **−5.07** |
| 200 | raw | 95.71 ± 4.65 | 95.29 ± 1.50 | 95.12 ± 10.22 | 94.91 ± 5.00 | 98.82 | **−3.91** | **−6.98** |

- **119** matches the paper on the paper's own test beats. With spike input every seed
  gets all 110 PVCs and no false positives (the paper had one). Raw input has one bad
  seed that misses about half the PVCs.
- **200** stays about 4–5 pp below the paper on the matching window. Spike input still
  misses about 17 of 170 PVCs (sensitivity 89.8 vs 98.2 %). So for 200 the gap is a
  model or preprocessing difference, not the choice of window.
- λ = 1e-6 is again worse than λ by CV here (spikes F1: 94.6 vs 100 on 119, 87.9 vs 94.4
  on 200).

### Records that differ from the paper by more than 2 pp, and likely causes

Four of the seven records (106, 200, 203, 223) are more than 2 pp below the paper's F1
in the primary variant. 119, 201 and 233 match within 1.5 pp. The cross-validated
F1 on the training window is already low for exactly these records (84, 95, 73 and
70 %), so these are not overfitting or test-set surprises. The model misses
the same patterns on training data. Most likely causes, in order:

1. **Unknown preprocessing and target definition.** The filter, the
   normalisation scope, the beat window [R − 100 ms, R + 150 ms] and the mean
   aggregation are assumptions. Record 203 (multiform PVCs, heavy noise and
   baseline artefacts) fails on precision: 51.6 % at 97.9 % sensitivity, so
   noisy normal beats are scored as PVCs. `results/figures/record_203_*` shows
   one mechanism. With the CV-selected τ = 100–200 ms, the smoothed readout
   decays slowly after each PVC. The low CV-selected threshold (0.15) is then
   still crossed in the window of the next, normal beat. 223 fails on sensitivity (77.7 %):
   with only 60 training PVCs, PVCs whose morphology differs from those in the
   training window are missed.
2. **The CV grids hit their edges.** The selected thresholds are often 0.10
   (the lowest value in the grid), and λ is usually 1e-2 or 1, the top of the
   grid. The selection would prefer values outside the specified grids. They
   were left unchanged, because widening them after seeing test numbers would be tuning
   toward the paper.
3. **Few training PVCs.** 201 has 20 PVCs in its first 10 minutes and 223 has 60, so the CV
   estimates and the readout are noisy.
4. **Different random reservoir and smoothing.** The paper's reservoir (and any
   per-subject hand tuning of τ) cannot be reproduced. Seed std is small for the
   primary variant (≤ 2 pp F1), so this explains the spread less than (1) to (3).
5. **Possible differences in evaluation.** The paper's exact per-beat decision
   rule and its handling of beats at the window borders are not stated.
6. **Different test beats.** The paper's test sets have different beat counts from ours
   (see "Are our test sets the paper's?"). For 106, 203 and 223 we cannot tell how much of
   the gap this explains, because no 5-minute window fits the paper's counts. For 200,
   rerunning on the matching window leaves the gap almost unchanged.

Figures for every record × input × leak (test segment, readout, threshold, true
and predicted PVCs) are in `results/figures/`.
