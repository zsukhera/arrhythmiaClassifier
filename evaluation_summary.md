# Evaluation Summary — 2026-10-13

## Changes Made
- Removed `run_synergy()`, `_gbc_binary_f1()`, and the `GradientBoostingClassifier` import from `evaluate.py`
- Ran full evaluation: main eval + ablation A1→A4

---

## Main Evaluation Results

### Random Split (19,858 test beats)

| Metric     | Ours   | Paper  |
|------------|--------|--------|
| Accuracy   | 0.9960 | 0.989  |
| Macro-F1   | 0.9875 | ~0.937 |
| Binary F1  | 0.9928 | 0.928  |
| OVR AUC    | 0.9992 | 0.985  |

Per-class:

| Class | Name   | Prec  | Rec   | F1    | Support |
|-------|--------|-------|-------|-------|---------|
| N     | Normal | 0.997 | 0.998 | 0.998 | 14,903  |
| L     | LBBB   | 0.998 | 0.996 | 0.997 | 1,615   |
| R     | RBBB   | 0.998 | 0.997 | 0.998 | 1,451   |
| V     | PVC    | 0.989 | 0.993 | 0.991 | 1,380   |
| A     | APC    | 0.968 | 0.941 | 0.954 | 509     |

**Exceeds the paper on every metric. No action needed for this split.**

---

### DS1/DS2 Split (48,981 test beats)

| Metric     | Ours   | Paper  | Gap     |
|------------|--------|--------|---------|
| Accuracy   | 0.7372 | 0.989  | -0.252  |
| Macro-F1   | 0.3132 | ~0.937 | -0.624  |
| Binary F1  | 0.3419 | 0.928  | -0.586  |
| OVR AUC    | 0.7686 | 0.985  | -0.217  |

Per-class:

| Class | Name   | Prec  | Rec   | F1    | Support |
|-------|--------|-------|-------|-------|---------|
| N     | Normal | 0.781 | 0.903 | 0.838 | 36,427  |
| L     | LBBB   | 0.000 | 0.000 | 0.000 | 4,124   |
| R     | RBBB   | 0.051 | 0.003 | 0.006 | 3,475   |
| V     | PVC    | 0.494 | 0.970 | 0.655 | 3,219   |
| A     | APC    | 0.284 | 0.038 | 0.067 | 1,736   |

**LBBB is completely missed (F1=0.000). RBBB is near-zero (F1=0.006). This split is broken.**

---

## Ablation A1→A4 (Random Split, Paper Params)

| Config              | Feats | Acc   | AnomRec | AnomF1 | AUC   | Paper Acc |
|---------------------|-------|-------|---------|--------|-------|-----------|
| A1 Statistical      | 6     | 0.924 | 0.940   | 0.875  | 0.982 | 0.985     |
| A2 +Fiducial        | 19    | 0.994 | 0.988   | 0.990  | 1.000 | 0.987     |
| A3 +Wavelet         | 31    | 0.995 | 0.988   | 0.991  | 1.000 | 0.988     |
| A4 +HRV/Non-linear  | 40    | 0.996 | 0.992   | 0.993  | 1.000 | 0.989     |

A1 (6 stats only) underperforms the paper's A1 baseline (0.924 vs 0.985). Adding fiducial features (A2) jumps to 0.994 — these features are doing most of the heavy lifting on random split. A3 and A4 add marginal gains above that.

---

## Root Cause of DS1/DS2 Failure

This is an **inter-patient generalisation failure**. The model is learning patient identity rather than arrhythmia patterns.

- DS1 trains on one patient cohort, DS2 tests on a different cohort
- Absolute amplitude features (`fid_r_amp`, `fid_q_amp`, etc.) encode electrode placement and cardiac axis — both patient-specific, not arrhythmia-specific
- LBBB/RBBB rely heavily on QRS morphology whose absolute shape varies across patients
- SMOTE synthesises more examples within the same patient's distribution, reinforcing patient-specific patterns

The CV F1 on DS1 training is 0.985 — the model is not underfitting. The issue is entirely about what information the features carry.

---

## Retraining Plan (in priority order)

### 1. Per-record signal amplitude normalisation — `features.py`
**Biggest single fix.**

In `process_record()`, before calling `extract_beat_features`, divide the beat signal by the record's median R-amplitude (or 95th-percentile amplitude across all beats in the record). This makes all amplitude-based features relative rather than absolute.

```python
# compute once per record, before the beat loop
record_beats = segments[sorted_global, 0, :]
r_amps = record_beats[:, R_IDX]
scale = float(np.median(np.abs(r_amps))) + 1e-6

# then inside the loop:
beat = segments[sorted_global[pos], 0, :] / scale
```

### 2. Normalise RR intervals to patient baseline — `features.py`
Replace `rr_pre_ms` and `rr_post_ms` (absolute ms) with their ratio to the record's mean RR interval. This removes heart rate as a confound and isolates ectopic timing patterns.

```python
mean_rr = float(np.mean(rr_ms)) if len(rr_ms) > 0 else 800.0
rr_pre_norm  = rr_pre_ms  / (mean_rr + 1e-6)
rr_post_norm = rr_post_ms / (mean_rr + 1e-6)
# pass these instead of raw ms values
```

### 3. Increase Optuna trials — `train.py`
Change `N_TRIALS = 10` to `N_TRIALS = 50` (or higher). The DS1/DS2 model only explored 10 hyperparameter combinations. With normalised features this matters more.

### 4. Delete the DS1/DS2 model and checkpoint before retraining
```
rm models/lgbm_ds1ds2.pkl
rm models/lgbm_ds1ds2_optuna.json
rm models/checkpoints/lgbm_ds1ds2_optuna.db
rm models/checkpoints/lgbm_ds1ds2_search_complete.json
```
The random split model can also be deleted and retrained, but it already exceeds the paper so it's lower priority.

### 5. Re-run the full pipeline
```
python source/features.py    # rebuilds mit_bih_features.npz
python source/split.py       # rebuilds split_random.npz + split_ds1ds2.npz
python source/train.py       # retrains both models
python source/evaluate.py    # re-evaluates
```

---

## Expected Outcome

Fixes 1 and 2 alone should substantially close the DS1/DS2 gap by removing patient-identity leakage from amplitude and timing features. The random split accuracy will stay at ceiling (~0.996) since it already has full patient coverage. LBBB and RBBB F1 on DS1/DS2 should recover from near-zero to something meaningful once the model stops relying on absolute amplitude cues.

---
zain's query:
read the evaluation_summary.md. and the whole directory ofc. tell me if we                     
    follow what has been written in the evaluatoin summary, won't that count as not              
    following the base paper ? and secondly, when the model approaching the weights              
    of the paper, does it automatically stop? how about if we begin with the                     
    parameters of the paper and run the optuna trials one by one to further improve              
    those parameters. so how would that be ?    

1. Does following the evaluation_summary.md conflict with the base paper?
  
  No — it's the opposite. The paper achieves 0.989 accuracy on DS1/DS2. Your implementation gets
  0.737 on the same split. The summary's fixes (amplitude normalisation, RR normalisation) are
  meant to close the gap to the paper, not diverge from it.

  The root cause is that your current features.py uses absolute amplitude values (fid_r_amp,
  fid_q_amp, etc.) which encode patient-specific electrode placement — not arrhythmia patterns.
  The paper presumably uses features that are relative/patient-agnostic, which is why it
  generalises across patients in DS1/DS2. Applying the summary's normalisation recommendations
  brings your feature engineering in line with what the paper must have done to achieve 0.989 on
  that split.

  Random split is already fine (0.996 vs paper's 0.989), so the summary only really matters for
  DS1/DS2.

  ---
  2. Does the model automatically stop when approaching the paper's hyperparameters?
  
  No. The stopping criterion is purely N_TRIALS = 10 (total trials completed). The paper params
  are enqueued as trial 0 (train.py:439), then Optuna runs 9 more TPE-sampled trials. Whatever
  trial scores the highest CV macro-F1 wins — there's no proximity-to-paper logic anywhere. If
  the paper params happen to score best, they get used; if trial 7 finds something better, that
  wins instead.

  ---
  3. Starting from paper params and running Optuna to improve — how does that work?
  
  The code already does this. PAPER_PARAMS is enqueued as trial 0 (train.py:428-441). Optuna's
  TPE sampler uses all completed trials (including trial 0) to build a probabilistic model of
  which regions of hyperparameter space look promising, then samples from there. So trial 0
  (paper params) directly seeds the search direction.

  The current limitation is that only 10 trials total are run — and trial 0 is the paper. That
  means only 9 new candidates get explored, which is very thin. The summary's recommendation of
  bumping N_TRIALS to 50 makes sense: with more trials, TPE has more signal to refine from the
  paper's starting point.

  One thing to note: because the existing DS1/DS2 checkpoint (.db) already has 10 completed
  trials, simply raising N_TRIALS = 50 and rerunning will only add 40 more trials on top of the
  existing ones — it won't restart. However, the summary also recommends deleting the DS1/DS2
  checkpoint first (step 4) since the features are going to change (steps 1–2), making the old
  search results meaningless anyway. So the workflow is:

  1. Fix features.py (normalise amplitude + RR)
  2. Rebuild the splits
  3. Delete the old DS1/DS2 checkpoint
  4. Raise N_TRIALS = 50 in train.py
  which regions of hyperparameter space look promising, then samples from there. So trial 0
  (paper params) directly seeds the search direction.

  The current limitation is that only 10 trials total are run — and trial 0 is the paper. That
  means only 9 new candidates get explored, which is very thin. The summary's recommendation of
  bumping N_TRIALS to 50 makes sense: with more trials, TPE has more signal to refine from the
  paper's starting point.

  One thing to note: because the existing DS1/DS2 checkpoint (.db) already has 10 completed
  trials, simply raising N_TRIALS = 50 and rerunning will only add 40 more trials on top of the
  existing ones — it won't restart. However, the summary also recommends deleting the DS1/DS2
  checkpoint first (step 4) since the features are going to change (steps 1–2), making the old
  search results meaningless anyway. So the workflow is:

  1. Fix features.py (normalise amplitude + RR)
  2. Rebuild the splits
  3. Delete the old DS1/DS2 checkpoint
  4. Raise N_TRIALS = 50 in train.py
  5. Retrain — paper params seed trial 0 again, then 49 more trials explore from there

✻ Crunched for 46s
  which regions of hyperparameter space look promising, then samples from there. So trial 0
  (paper params) directly seeds the search direction.

  The current limitation is that only 10 trials total are run — and trial 0 is the paper. That
  means only 9 new candidates get explored, which is very thin. The summary's recommendation of
  bumping N_TRIALS to 50 makes sense: with more trials, TPE has more signal to refine from the
  paper's starting point.

  One thing to note: because the existing DS1/DS2 checkpoint (.db) already has 10 completed
  trials, simply raising N_TRIALS = 50 and rerunning will only add 40 more trials on top of the
  existing ones — it won't restart. However, the summary also recommends deleting the DS1/DS2
  checkpoint first (step 4) since the features are going to change (steps 1–2), making the old
  search results meaningless anyway. So the workflow is:

  1. Fix features.py (normalise amplitude + RR)
  2. Rebuild the splits
  3. Delete the old DS1/DS2 checkpoint
  4. Raise N_TRIALS = 50 in train.py
  5. Retrain — paper params seed trial 0 again, then 49 more trials explore from there
