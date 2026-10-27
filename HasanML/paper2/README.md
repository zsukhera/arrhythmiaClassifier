# Paper 2: 57 ECG morphology features + Logistic Regression, 5-class AAMI (MIT-BIH)

Replication of

> Ö. Yakut and E. Doğru Bolat, "An efficient arrhythmic heartbeat classification method
> using ECG morphology based features," *Euroasia J. Math., Eng., Nat. Med. Sci.*, vol. 7,
> no. 13, pp. 200–212, 2020, doi:10.38065/euroasiaorg.403.

From each beat, 57 morphology and RR features are extracted (paper Table 2). They are
optionally reduced to 20 with `SelectKBest(f_classif)`. Four scikit-learn classifiers
with default hyperparameters then label each beat N / S / V / F / Q: Logistic
Regression (the proposed method), LDA, SVM and MLP. Many details are only in the first
author's PhD thesis (Kocaeli University, 2018), which we do not have. Each gap is filled
with a configurable assumption, listed [below](#assumptions).

## Install

```bash
cd HasanML/paper2
source ../../.venv/bin/activate          # or any Python >= 3.10 environment
pip install -r requirements.txt
```

## Run

```bash
python -m src.run_experiment --config configs/default.yaml   # full run: 3 protocols x {57,20} features x 4 models + ablations
python -m pytest                                             # unit tests

# quick check: inter-patient split, linear models only, no ablations
python -m src.run_experiment --protocols A --models LR LDA --no-ablations --results-dir results_quick
# standard ANSI/AAMI EC57 label mapping instead of the paper's Table 1
python -m src.run_experiment --label-map aami --results-dir results_aami
```

Other flags: `--nfeat 57 20`, `--mlp-seeds 0 1 2`, `--device {auto,cpu,cuda,gpu}`,
`--workers N`, `--blas-threads N`, `--no-figures`. Always run it as a module
(`python -m src...`). `src/select.py` would shadow the standard-library `select`
module if `src/` were put on `sys.path`.

Every parameter and assumption is in `configs/default.yaml`. Records are read from
`data/mitdb/`. A missing record is copied from the repository's
`datasets/mitbih/mitdb/` if present; otherwise it is downloaded from PhysioNet with
`wfdb`. Features are cached per configuration hash in `results/cache/`. Each finished
model fit is cached too, so an interrupted run resumes where it stopped and re-running
only rebuilds the reports.

### CPU, threads and GPU

The design follows the LightGBM pipeline in `../../source/train.py`: a device probe
with an environment override and fallback, then the same code path on either device.

| Stage | How work is split |
|---|---|
| Feature extraction | One **process** per record (`feature_workers`, default 8). Filtering and fiducial detection are vectorised NumPy over a record's beat matrix. |
| Model fits | Every fit is an independent job: protocol × fold × feature set × ablation × model × seed. Jobs run in a **process pool** (`cpu_workers`, default 8), and each process is capped at `blas_threads` (default 2) BLAS/OpenMP threads, so the pool does not oversubscribe the cores. The slowest jobs (SVM, then MLP, then CV folds) are scheduled first. |
| GPU (`device: cuda`/`gpu`, or `auto` when available) | LR and SVM jobs run through **cuML** on the GPU. A **thread pool** (`gpu_workers`) feeds the device while the CPU pool runs LDA and MLP, which cuML lacks. |

`device: auto` probes for cuML and a CUDA device by doing a tiny real fit, and falls
back to CPU, logging the reason. The `P2_DEVICE=cpu|cuda|gpu` environment variable
overrides the config, like `LGBM_DEVICE` in the LightGBM pipeline. Note that cuML's
defaults (QN solver for LR, its own Platt scaling for SVC) are close to scikit-learn's
but not identical. GPU rows are labelled with `device=cuda` in the CSVs.

**On this machine the GPU path was not used.** The machine is an Intel Core Ultra 7 155H
under WSL2 (22 threads, 7.6 GB RAM) with Intel Arc integrated graphics. WSL exposes no
CUDA driver here (no `nvidia-smi`, no `libcuda`), so the probe falls back to CPU. All
reported numbers are scikit-learn on CPU.

## Pipeline

1. **Data.** MIT-BIH Arrhythmia Database, lead MLII (selected by `sig_name`, which
   handles record 114's swapped leads), 44 records (paced 102/104/107/217 excluded).
   R positions come from the `atr` annotations.
2. **Labels.** Paper Table 1 mapping (default): N = {N, L, R, e, j}, S = {A, a, J, x},
   V = {V, !, E}, F = {F}, Q = {Q}. The standard EC57 mapping (`--label-map aami`):
   S = {A, a, J, S}, V = {V, E}, Q = {/, f, Q}.
3. **Baseline wander.** Estimate the baseline with an elliptic low-pass (order 4,
   1 Hz, 0.1 dB ripple, 40 dB stopband, zero-phase `sosfiltfilt`), then subtract it
   (`results/figures/baseline_101.png`, paper Fig. 2).
4. **Beat window.** 127 samples before R + R + 128 after = 256 samples. Beats whose
   window runs past the record edge are dropped and logged (`results/dropped_beats.csv`).
5. **Fiducials** (`src/fiducials.py`). P, Q, R, S and T, plus the onsets/offsets
   P1/P2, Q1, S2 and T1/T2, are found in fixed sub-windows around R. See
   `results/figures/fiducials_119.png` and `fiducials_by_class.png`.
6. **57 features** (`src/features.py`, names in `FEATURE_NAMES`): 18 durations,
   7 amplitudes, 2 areas, 3 angles, 4 RR features, R time, 12 fiducial amplitudes and
   10 consecutive-point intervals.
7. **Min-max scaling** to [0, 1], fit on the training set.
8. **SelectKBest(f_classif, k=20)**, fit on the training set (20-feature runs).
9. **Classifiers**, scikit-learn defaults: `LogisticRegression()`,
   `LinearDiscriminantAnalysis()`, `SVC(probability=True)` and `MLPClassifier()`, with
   `random_state` fixed. MLP is repeated over 5 seeds. LR does not converge in the
   default 100 iterations, so it is also reported with `max_iter=5000`.

### Split protocols

| | Protocol | Train / test |
|---|---|---|
| A | inter-patient, de Chazal et al. (2004) DS1 / DS2 (default) | 22 / 22 records |
| B | intra-patient: per-class random split reproducing Table 1's test fraction (seed 0) | beats pooled from all records |
| C | 5-fold stratified cross-validation over all beats; out-of-fold predictions pooled | beats pooled from all records |

## Assumptions

Every item is a key in `configs/default.yaml`.

| # | Item | Paper says | Our choice |
|---|---|---|---|
| 1 | Paced records | Fig. 1: 102, 104, 107, 217 excluded | excluded (confirmed by Fig. 1; our counts then match Table 1) |
| 2 | Baseline filter | "lowpass elliptic IIR … does not pass noise below 1 Hz" | baseline = ellip(order 4, 0.1 dB, 40 dB, 1 Hz) low-pass, zero-phase; signal − baseline |
| 3 | Fiducial search windows | Fig. 3, not legible | relative to R: P [−250, −80] ms, Q [−80, 0), S (0, 80], T [100, 350] ms, clipped to the window end (R + 353 ms); clipping happens for the ~2 % of beats whose refined R lies right of the annotation |
| 4 | R | annotation | local max within ±10 samples of the annotation |
| 5 | P / T peak | unspecified | P = max of x; T = max of \|x − iso\| (allows inverted T); iso = median of the beat window |
| 6 | P1/P2, T1/T2 | unspecified | walking out from the peak, the first sample with \|x − iso\| < 20 % of the peak's \|x − iso\| |
| 7 | Q1 (QRS onset), S2 (QRS offset) | unspecified | from the steepest point within 40 ms before Q (after S), walking out, the first sample with \|dx\| < 10 % of the max \|dx\| in the QRS |
| 8 | Feature definitions 1–30 | names only | PR = R − P peak; PR interval = Q1 − P1; PS = S − P; QT = T2 − Q1; QRS = S2 − Q1; ST interval = T2 − S2; T width = T2 − T1; amplitudes are signed values of the baseline-corrected signal (mV), differences as named (e.g. RS amplitude = x[R] − x[S]); areas = Σ x·Δt over [Q, R] and [R, S] (mV·s) |
| 9 | Angles 28–30 | names only | the angle at the middle point between the two segments, in the (time **s**, amplitude **mV**) plane; a degenerate segment (two points coinciding) gives 0° |
| 10 | RR features | de Chazal 2004 | Pre-/Post-RR to the previous / next annotated beat (first/last beat copies its other side); RR average = record mean; local = mean of 5 RR before + 5 after. The RR sequence uses every beat annotation plus the label-map symbols (`x`, `!`) |
| 11 | Feature 35 "IR time information" | unspecified | time of the R peak in the record (s); `no_feature35` ablation drops it |
| 12 | Features 36–47 | "amplitudes of the 12 fiducial points" | x at P1, P, P2, Q1, Q, R, S, S2, T1, T, T2, plus the baseline level = mean of the first 5 window samples |
| 13 | Features 48–57 | "intervals" | durations between consecutive points P1→P→P2→Q1→Q→R→S→S2→T1→T→T2 (ms) |
| 14 | Normalisation fit | implied: all data before the split | training set (default); `scaler_all` ablation = paper-faithful |
| 15 | Feature selection fit | unspecified | training set (default); `select_all` ablation = all beats |
| 16 | Split | Fig. 1: the 44 records are divided equally into training and test sets (record level) | protocol A (DS1/DS2) is the faithful one; B and C are run for comparison |
| 17 | Class order for metrics | – | N, S, V, F, Q; K = 5 in every averaged metric |

Some of the 57 features duplicate others exactly (for example QR = int Q→R and
RS = int R→S). The paper's list implies this; the duplicates are kept.

Known limitations of the simple fiducial rules: a baseline dip larger than the T wave
can be taken as T (iso = window median), and Q-class beats (noisy) get erratic points.
The rules are not tuned on test data.

## Results

Full tables are in `results/results_summary.md`. `results/results_<protocol>_<nfeat>.csv`
has every averaging scheme, per-class Se/PPV and the 5×5 confusion matrix for every
model, seed and ablation. The full run took 271 min on CPU (8 processes × 2 BLAS
threads); 49 SVM fits with Platt-scaling CV dominate the time.

### Beat counts (Table 1): match

With the paper's label map and the 4 paced records excluded, S, V, F and Q match
Table 1 **exactly** (2972 / 7480 / 802 / 15). N has 90 087 beats against 90 083
(+4); the paper probably dropped a few more edge beats. 40 beats were dropped because
their window runs past a record edge, and 1 `S` beat (not in the paper's map); all are
listed in `results/dropped_beats.csv`. Under protocol **A (DS1/DS2)** the per-class
train/test counts also match the paper: S 1000 / 1972, V 4260 / 3220, F 414 / 388 and
Q 8 / 7 all exact, with N 45 846 / 44 241 against 45 845 / 44 238. Protocol B matches
by construction, so the counts cannot tell A and B apart. However, an *exact*
per-class match under a fixed record split (S = 1000 train beats exactly) would be a
striking coincidence for a random split. See `results/counts_vs_paper.md`.

### Metric definitions: checked against the paper's own tables

Checked against the full paper text (journal PDF), which includes training rows
for every model in Tables 3 and 4 (16 rows in total).

- **Test rows (8/8): the micro-averaging hypothesis holds exactly.** With
  e = 1 − Sensitivity, the paper's "Accuracy" = 1 − 2e/5 (mean one-vs-rest accuracy)
  and "Specificity" = 1 − e/4 (micro specificity), to all 3 decimals. So
  **"Sensitivity" is plain multiclass accuracy**: LR's true 5-class test accuracy is
  99.503 % (57 features) and 99.416 % (20 features), not 99.80 / 99.77 %.
- **Training rows (0/8): it does not hold** (e.g. LR-57 train: Acc 99.300 vs
  1 − 2e/5 = 99.428). The training rows ("five-fold cross-validation") use another,
  unstated definition, so they cannot be compared with the test rows.
- **"F1" is not an F1 score in 13 of 16 rows.** It equals the harmonic mean of the
  paper's Sensitivity and **Specificity** (precision does not enter), e.g. SVM-57 test
  2·97.015·99.254 / (97.015 + 99.254) = 98.122. In the other 3 rows (LR-57 test,
  LR-20 test, LDA-20 test) F1 = Sensitivity. Those three are the paper's best results.
- The Discussion repeats the tables with two discrepancies: LDA-20 is given as
  Se 98.243 / Sp 99.026 / F1 98.982 (table: 99.243 / 99.811 / 99.243), and SVM-20 F1
  as 98.298 (table: 99.298).

`tests/test_pipeline.py` checks the formulas on a known confusion matrix and on the
paper's own LR numbers.

### Headline comparison (paper-style accuracy = mean OvR accuracy; plain accuracy in brackets)

LR is shown with `max_iter=5000`: the default 100 iterations never converge (the
default-LR numbers are within 0.5 pp; see the summary). MLP is the mean of 5 seeds.
MLP's default 200 iterations also do not converge; this is reported, not changed.

| Model | Paper, 57 F | A, 57 F | B, 57 F | C, 57 F | Paper, 20 F | A, 20 F | B, 20 F | C, 20 F |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LR  | 99.801 (99.50) | 97.77 (94.42) | 98.15 (95.38) | 98.48 (96.21) | 99.766 (99.42) | 97.41 (93.53) | 97.94 (94.85) | 98.30 (95.75) |
| LDA | 99.339 (98.35) | 96.44 (91.10) | 97.99 (94.97) | 98.07 (95.17) | 99.697 (99.24) | 95.51 (88.77) | 97.74 (94.35) | 97.77 (94.41) |
| SVM | 98.806 (97.02) | 95.60 (89.01) | 98.94 (97.36) | 99.22 (98.05) | 99.552 (98.88) | 97.08 (92.71) | 98.74 (96.85) | 98.98 (97.44) |
| MLP | 98.623 (96.56) | 96.62 (91.55) | 99.40 (98.49) | 99.52 (98.79) | 98.776 (96.94) | 97.85 (94.63) | 99.16 (97.90) | 99.35 (98.38) |

Majority-class ("all N") plain accuracy is 88.79 % for A/B and 88.88 % for C.

S-class sensitivity, LR (5000 it.) / MLP: **A 6.5 % / 7.1 %**, B 32.5 % / 83.0 %,
C 45.3 % / 87.0 % (57 features). Our AUCs: the micro AUC (98.0–99.9) is in the paper's
range; the macro OvR AUC (80–96) is much lower, because the S, F and Q classes are poorly
separated.

### Which protocol reproduces the paper?

- **The paper used a record-level (inter-patient) split.** Its Figure 1 (an image in
  the PDF, extracted and read directly) shows: 48 records → the 4 paced records
  excluded → "the remaining 44 records … equally into the data sets" → training
  51 527 / testing 49 825 beats. That is 22 / 22 records, and DS1/DS2 reproduces the
  per-class counts exactly, so **protocol A is the faithful replication**. (DS1/DS2 is
  not strictly patient-disjoint: records 201 and 202 come from the same patient and sit
  on opposite sides.)
- **Counts confirm A**: the match is exact per class.
- **Results point to B/C** (intra-patient). Under A every model is 2.3–10.5 pp below the
  paper in plain accuracy, and S sensitivity is < 15 %. That is the typical
  inter-patient failure on S beats and in line with published inter-patient results.
  Under B and C, SVM and MLP reach or exceed the paper's numbers (e.g. C, 57 F:
  MLP 98.79 % plain vs the paper's 96.56 %; SVM 98.05 % vs 97.02 %).
- **LR does not reproduce under any protocol.** It is the paper's best model at
  99.50 % plain accuracy, but in our pipeline it reaches only
  94.4 % (A), 95.4 % (B), 96.2 % (C) with 57 features. The 3.3–5.9 pp gap (both feature
  sets) is far beyond 1 pp. The
  paper's model ranking (LR > LDA > SVM > MLP) is the reverse of ours under B/C
  (MLP > SVM > LR ≈ LDA), which is the usual ranking for min-max-scaled handcrafted
  features with default hyperparameters.

So the paper's results are only approached under beat-level splits, which the paper
did not use. Under its own record-level split our LR reaches 94.4 % (57 F) / 93.5 %
(20 F) against 99.50 % / 99.42 %. A linear model at 99.4–99.5 % 5-class accuracy on an
inter-patient split would far exceed published inter-patient results. We cannot explain the paper's LR numbers with the
information in the paper.

### Ablations (`results/results_summary.md`, last table)

- **Paper-faithful scaler (fit on all data)** changes little: ≤ 1.4 pp plain accuracy,
  mostly *worse* for LR under A and B. LDA is unchanged, because it is invariant to
  per-feature affine scaling.
- **SelectKBest on all data** changes LR/SVM/MLP under A by +0.3 / −1.3 / −3.3 pp. It
  makes no difference under B/C, where the training and full-data F-rankings pick the
  same 20.
- **Without feature 35 (R time)**: ≤ 0.2 pp for LR/LDA/SVM. MLP moves by up to
  ±2.6 pp (A, 57 F), within seed noise for that unconverged model (seed std 2.0 pp).
  With 20 features nothing changes, because R time is never selected. Note that
  "RR average" is also a per-record constant, so under B/C both features could encode
  record identity.

### Feature selection

Only 4–14 of our 57 features have F > 1000 on a training set (6 under A, 14 under C),
and 15 on all beats. The paper's criterion "F > 1000" gave it 20, so its features
separate the classes much better than ours. This is the most likely root of the LR gap
(see below). The kept features and all F-scores are in `results/selected_features.md`
and `results/fscores_*.png` (paper Fig. 5).

## Differences from the original

- **Fiducial detection and feature definitions are ours** (assumptions 3–13). The
  paper's Fig. 3 windows and the thesis's exact rules for on-/offsets, angles, areas,
  feature 35 and features 36–57 are unavailable. Our features are clearly less
  discriminative (see Feature selection), which alone could explain a several-pp gap
  for the linear models, which cannot compensate for weak features the way SVM/MLP can.
- **Split.** Fig. 1 shows a record-level 22/22 split, and DS1/DS2 matches its counts
  exactly. Our assignment of records to the two sets is inferred from the counts; the
  paper does not list them. Under it, every model falls 2.3–10.5 pp short of the paper.
- **Figure 3 (fiducial schematic)** confirms the definitions of P width, PR interval
  (P1→Q1), QRS (Q1→S2), QT (Q1→T2), T width, P1R/P2R/PR and RT1/RT/RT2. It labels
  S2→T1 as the "ST segment", while our feature 17 ("ST interval") is S2→T2. The figure
  gives nominal durations only, not the search-window boundaries.
- **Figure 5 (selected features)** lists the paper's top 20 by number: 34, 50, 53, 12,
  33, 13, 28, 27, 48, 29, 47, 56, 55, 54, 52, 49, 31, 9, 11, 32, with F ≈ 1000–2620.
  Its top two are RR local average (≈ 2620) and an inter-fiducial interval. Its
  RR average, a per-record constant, scores ≈ 1900. Our F-scores for the same features
  are lower (e.g. RR local average 758, RR average 847 on all beats), and our ranking
  differs (our top: Pre-RR, amp S2, angle R-S-S2).
- **Toolchain.** The paper filtered the ECG and extracted features in MATLAB R2015b, and
  ran scikit-learn (version not given) on Google Colab in late 2020. Our filter and
  features are a Python re-implementation, and the LR default (`lbfgs`, `max_iter=100`)
  does not converge on our features.
- **Order of operations.** The paper normalises all beats and runs SelectKBest *before*
  dividing them into training and test sets (Section 3). We fit both on the training
  set by default and run each paper-faithful variant as an ablation; both change
  results by little.
- **"F1" column** in the paper is mostly the harmonic mean of Se and Sp, not F1 (see above).
- **N count +4** relative to Table 1 (edge-beat handling).
- **AUC averaging** is unspecified; we report macro OvR (main) and micro.
- Every difference > 1 pp is flagged ⚠ in `results/results_summary.md`. Likely causes,
  in order: (1) feature definitions / fiducial rules, (2) split protocol, (3) LR solver
  and multi-class strategy, (4) unknown preprocessing details (filter order/ripple,
  window alignment).

## Files

```
configs/default.yaml        every parameter and assumption
src/data.py                 records, label maps, splits A/B/C
src/preprocess.py           elliptic baseline removal, beat windows
src/fiducials.py            vectorised fiducial-point detection
src/features.py             57 features, RR features, parallel per-record extraction + cache
src/select.py               min-max scaling, SelectKBest(f_classif)
src/models.py               LR / LDA / SVM / MLP (scikit-learn defaults; cuML on GPU)
src/backend.py              device probe (auto | cpu | cuda | gpu)
src/metrics.py              all averaging schemes, AUC, ROC
src/plots.py                figures
src/run_experiment.py       CLI, job scheduling (processes / GPU threads), reports
tests/test_pipeline.py      label maps, windows, RR, fiducials, micro formulas, split leakage
results/                    counts_vs_paper.md, results_summary.md, results_*_*.csv,
                            selected_features.md, confusion_*.png, roc_*.png,
                            fscores_*.png, figures/, run.log, dropped_beats.csv
```
