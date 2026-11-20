# Zero-shot generalisation: MIT-BIH models on PTB-XL and Chapman-Shaoxing

Agent: **zeroshot**, 2026-11-18. Everything ran on the GPU box, CPU only, with 4 workers.
`ROOT=/home/i230510/Desktop/arrhythmiaClassifier`

## TL;DR

Two frozen MIT-BIH models were run on PTB-XL (8,226 records) and Chapman-Shaoxing (8,541 records): the LightGBM 5-class model (`models/lgbm_ds1ds2.pkl`) and the Farag 1D-CNN P1 10-seed ensemble (tau_S = 0.50). Neither model was retrained. Only lead II was used, records are 10 s long, and labels are per record. Every decision (R-peak detector, the CNN's RR fallback, and the "≥ k beats" aggregation rules) was fixed on MIT-BIH DS1 10-s strips before any target data was evaluated.

Se and Sp do not depend on prevalence, so they are the fair numbers to compare across datasets. PPV and F1 change with how common each label is.

| Condition / model | MIT-BIH DS2 strips Se / Sp | PTB-XL Se / Sp | Chapman Se / Sp |
|---|---|---|---|
| PVC (V), CNN | 94.5 / 90.8 | 84.7 / 95.3 | 77.7 / 93.5 |
| PVC (V), LGBM | 93.1 / 88.4 | 90.9 / 92.0 | 83.3 / 92.6 |
| PAC (S), CNN | 84.9 / 75.0 | 71.5 / 94.4 | 75.8 / 96.1 |
| PAC (S), LGBM | 78.3 / 73.1 | 68.7 / 83.3 | 74.3 / 89.0 |
| LBBB (L), LGBM | 6.7 / 98.8 | 45.0 / 98.7 | 41.2 / 99.6 |
| RBBB (R), LGBM | 0.0 / 99.3 | 4.7 / 99.3 | 4.4 / 99.6 |

F1 for the same rows:

| | DS2 strips | PTB-XL | Chapman |
|---|---|---|---|
| CNN V | 86.9 | 76.6 | 59.2 |
| LGBM V | 83.9 | 71.8 | 59.5 |
| CNN S | 41.8 | 50.4 | 75.7 |
| LGBM S | 37.6 | 27.4 | 61.2 |
| LGBM L | 11.3 | 54.0 | 52.2 |
| LGBM R | 0.0 | 8.0 | 8.2 |

**Main findings**

1. **Ectopic-beat detection transfers reasonably well.**
   - V sensitivity falls by about 10 pp on PTB-XL and 11–17 pp on Chapman, compared with the in-domain 10-s strips.
   - S sensitivity falls by 3–13 pp.
   - Specificity is *higher* on the external sets. Their normal records are clean 10-s ECGs from healthy people. The MIT-BIH "normal" strips come from arrhythmia patients' Holter tapes, so they are noisier.
2. **Bundle-branch block does not transfer, and was already broken in-domain.**
   - LGBM finds RBBB in about 4–5 % of records.
   - LGBM finds LBBB in about 41–45 % of external records, but only 7 % of DS2 strips. DS2's LBBB patients (records 111 and 214) seem atypical rather than the external data being easier.
   - The CNN cannot output L or R at all, because AAMI maps LBBB and RBBB to N.
3. **For the CNN, short records cost more than the dataset change.** On MIT-BIH DS2 the CNN's beat-level S sensitivity is:
   - 87.1 % with full 30-min records (the original pipeline);
   - 45.5 % on 10-s strips with the true .atr R-peaks;
   - 44.6 % on 10-s strips with detected R-peaks.

   The global-400 / local-80 RR normalisers have almost no history in a 10-s strip. The detector costs very little (about 1 pp).

## What was done

### 1. Data (`datasets/external/`)

| Dataset | Source | Size | Extracted |
|---|---|---|---|
| PTB-XL v1.0.3 | `physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip` (PhysioNet get-zip redirect) | 1.84 GB zip | `ptbxl/ptb-xl-.../` with records500, csvs; records100 skipped |
| Chapman-Shaoxing (ecg-arrhythmia 1.0.0) | `physionet-open.s3.amazonaws.com/ecg-arrhythmia/ecg-arrhythmia-1.0.0.zip` | 2.50 GB zip | `chapman/<dir>/WFDBRecords/`, 45,152 records |

The archives were only unzipped and read with `wfdb`. No code shipped with the data (e.g. `example_physionet.py`) was run.

### 2. Signal path (identical for MIT-BIH strips and external records)

1. Take lead II (MLII for MIT-BIH).
2. Resample 500 → 360 Hz with `resample_poly(18/25)`, then take the first 3,600 samples.
3. Detect R-peaks on the 360 Hz signal.
4. Run each model's own pipeline:
   - **LightGBM:** 0.5–40 Hz 4th-order Butterworth `sosfiltfilt`, then the 256-sample window (100/156). Beat features are byte-identical to `source/features.py`: `extract_beat_features` is imported, and `process_record` is re-implemented per strip.
   - **Farag CNN:** no filter (the config's `filter: none`), then the paper3 functions as they are: `resample_poly` 360 → 128, 64-sample edge-padded window, mean normalisation, derivative, NK = 32 crop, and `compute_rr_features`.

**R-peak detector** (`validate_detector.py`, `out/detector_validation.txt`)

- Tested on MIT-BIH MLII cut into 10-s strips, against the .atr beats with a 150 ms tolerance.
- Four configurations were compared on DS1: XQRS or GQRS, each with or without snapping the peak to the band-passed extremum within ±50 ms.
- Chosen: **XQRS + snap** (best DS1 F1).

| Split | Se | PPV | F1 | \|offset\| ≤ 3 samples |
|---|---|---|---|---|
| DS1 | 96.59 | 98.69 | 97.63 | 95.0 % |
| DS2 | 97.94 | 99.66 | 98.79 | 96.0 % |

- GQRS had Se of only 90 %.
- Without snapping, XQRS peaks are only 94 % within 3 samples, and GQRS peaks are offset by a median of −12 samples. Snapping matters because both models centre their windows on the R peak.
- The missed beats were not broken down by position. I expect most are at strip edges, during XQRS's learning phase, but this was not checked.

### 3. Feature fallbacks for 10-s records (all documented, decided a priori or on DS1)

**LightGBM**

- Per-record normalisation becomes per-strip:
  - amplitude scale = median |R| of the strip's beats;
  - mean RR = mean of the strip's RR intervals.
- RR is computed from *all* detected peaks. Only beats whose 256-sample window fits inside the strip are classified, which drops beats in the first 0.28 s and the last 0.43 s.
- The HRV window is the ≤ 50 preceding RR intervals, but a strip only has about 10–12, so HRV is affected:
  - SDNN, RMSSD, SD1 and SD2 are computed from those few intervals (noisy).
  - LF, HF, LF/HF and SampEn need ≥ 10 intervals and ≥ 10 s, so they are almost always NaN.
  - NaNs are imputed with **DS1 training column medians**. This mirrors `features.py`, which imputed with global column medians.
- In practice, 4 of the 40 features are constant and 5 are noisy on 10-s inputs.

**Farag CNN**

- The global-400 and local-80 RR means cannot be filled. Two fallbacks were compared on DS1 strips by beat-level macro-F1:
  - **causal (chosen):** the paper3 code unchanged, i.e. a causal mean over whatever history exists, which is what the code already does at a record start;
  - **stripmean:** both means replaced by the mean RR of the whole strip.
- Results: causal 0.7174 vs stripmean 0.7076, so **causal** was used.
- For reference, stripmean gives better S sensitivity on DS2 and on both targets. Its rows are reported as a secondary result; it was not selected on target data.

### 4. Labels, cohorts and aggregation

**MIT-BIH strips.** All 44 non-paced records were cut into non-overlapping 10-s strips: 7,920 strips; 3,930 DS1 and 3,911 DS2 strips in the cohort. Strip truth comes from the .atr beats in the strip:

| Label | Rule |
|---|---|
| L / R | ≥ 50 % of beats are L / R |
| V | ≥ 1 beat in {V, E} |
| S | ≥ 1 beat in {A, a, J, S} |
| N | every beat is N |

The cohort is N ∪ any target label.

**PTB-XL** (`scp_codes`)

| Label | Codes |
|---|---|
| N | keys ⊆ {NORM, SR}, NORM = 100, SR present |
| L | CLBBB |
| R | CRBBB |
| V | PVC |
| S | PAC |

- Excluded: AFIB, AFLT, SVTAC, PSVT, PACE, and anything in the pacemaker column.
- IRBBB and ILBBB are not positives.
- BIGU and SVARR are not used because their origin is ambiguous.
- Cohort: 8,226 records (N 6,140, L 464, R 446, V 931, S 393).

**Chapman** (SNOMED codes in `#Dx`)

| Label | Codes |
|---|---|
| N | Dx == {426783006 SR} |
| L | 164909002, 733534002 |
| R | 59118001, 713427006 |
| V | 17338001 VPB, 427172004 PVC, 11157007 VB |
| S | 284470004 APB, 251173003, 251164006, 63593006 |

- Excluded: AF/AFL (164889003, 164890007), SVT, AT, AVNRT, AVRT, VT, wandering pacemaker, and 10370003 (pacing rhythm).
- Note: 713427006 (complete RBBB), 733534002 (complete LBBB) and 10370003 appear in the headers but **not** in `ConditionNames_SNOMED-CT.csv`. I added them after counting the codes, before any model was evaluated on Chapman. Without them, R had only 313 records instead of 864.
- Cohort: 8,541 records (N 5,908, L 187, R 864, V 610, S 1,180).

**Aggregation rule.** A record is positive for condition c if (#beats predicted c ≥ k) AND (fraction ≥ p). (k, p) was grid-searched for maximum F1 on DS1 strips with detected peaks (`out/rules.json`):

| Model | Rule |
|---|---|
| LGBM | L: k = 4; R: k = 4; V: k = 1; S (LGBM class A): k = 1; p = 0 for all |
| CNN | V: k = 1; S: k = 1 |

- "N" means no condition fired.
- Caveat: DS1 is the models' **training** set, so these rules are fitted on in-sample predictions. They are likely too permissive for out-of-sample data; k = 1 for S produces the S false alarms.

## Results

### MIT-BIH DS2 10-s strips (in-domain reference, `out/mitbih_strip_report.md`)

**Beat level, DS2, matched beats**

| Pipeline | LGBM acc / macro-F1 | CNN acc / Se_S / Se_V |
|---|---|---|
| Original full-record pipeline (sanity check, reproduces progress.md exactly) | 0.7635 / 0.3878 | 0.9161 / 87.1 / 96.5 |
| 10-s strips, .atr peaks | 0.7628 / 0.4182 | 0.9275 / 45.5 / 95.1 |
| 10-s strips, detected peaks | 0.7625 / 0.4336 | 0.9238 / 44.6 / 94.7 |
| 10-s strips, detected peaks, CNN stripmean RR | — | 0.9218 / 59.9 / 94.1 |

**Strip level, DS2, detected peaks (Se / Sp / PPV / F1)**

| | LGBM | CNN |
|---|---|---|
| L | 6.7 / 98.8 / 36.4 / 11.3 | n/a |
| R | 0.0 / 99.3 / 0.0 / 0.0 | n/a |
| V | 93.1 / 88.4 / 76.3 / 83.9 | 94.5 / 90.8 / 80.4 / 86.9 |
| S | 78.3 / 73.1 / 24.7 / 37.6 | 84.9 / 75.0 / 27.7 / 41.8 |
| N (none fired) | 57.7 / 79.4 / 74.6 / 65.1 | 73.0 / 75.8 / 75.9 / 74.4 |

Using .atr peaks instead of detected peaks changes strip-level F1 by ≤ 4 pp (see the report).

DS1 strips are in-sample: e.g. LGBM L F1 96.6, R 92.2, V 90.6. This shows how large the inter-patient gap already is within MIT-BIH.

### PTB-XL (`out/ptbxl_report.md`), 8,226 records, median 11 detected beats per record

| | LGBM Se / Sp / PPV / F1 | CNN (causal) Se / Sp / PPV / F1 | CNN (stripmean) Se / Sp / PPV / F1 |
|---|---|---|---|
| L (464) | 45.0 / 98.7 / 67.4 / 54.0 | n/a | n/a |
| R (446) | 4.7 / 99.3 / 27.3 / 8.0 | n/a | n/a |
| V (931) | 90.9 / 92.0 / 59.3 / 71.8 | 84.7 / 95.3 / 69.9 / 76.6 | 82.3 / 95.5 / 69.8 / 75.5 |
| S (393) | 68.7 / 83.3 / 17.1 / 27.4 | 71.5 / 94.4 / 38.9 / 50.4 | 77.1 / 92.6 / 34.2 / 47.4 |
| N (6,140) | 80.3 / 84.5 / 93.8 / 86.6 | 97.8 / 67.6* / 89.9 / 93.7 | 97.2 / 67.9* / 89.9 / 93.4 |

\* CNN N-specificity is low partly by construction: L and R records count as "not normal", but the CNN maps BBB beats to N.

**Confusion, LGBM** (truth × predicted set)

| truth | N | L | R | V | S | multi |
|---|---|---|---|---|---|---|
| N | 4933 | 3 | 4 | 129 | 1054 | 17 |
| L | 96 | 98 | 8 | 90 | 6 | 116 |
| R | 174 | 35 | 12 | 74 | 56 | 53 |
| V | 32 | 2 | 0 | 599 | 41 | 135 |
| S | 20 | 0 | 0 | 60 | 188 | 50 |
| multi | 2 | 0 | 0 | 82 | 12 | 45 |

**Confusion, CNN causal**

| truth | N | V | S | multi |
|---|---|---|---|---|
| N | 6003 | 20 | 109 | 8 |
| L | 329 | 63 | 7 | 15 |
| R | 302 | 83 | 6 | 13 |
| V | 35 | 505 | 92 | 177 |
| S | 8 | 68 | 189 | 53 |
| multi | 2 | 86 | 15 | 38 |

### Chapman-Shaoxing (`out/chapman_report.md`), 8,541 records, median 12 detected beats per record

| | LGBM Se / Sp / PPV / F1 | CNN (causal) Se / Sp / PPV / F1 | CNN (stripmean) Se / Sp / PPV / F1 |
|---|---|---|---|
| L (187) | 41.2 / 99.6 / 71.3 / 52.2 | n/a | n/a |
| R (864) | 4.4 / 99.6 / 55.9 / 8.2 | n/a | n/a |
| V (610) | 83.3 / 92.6 / 46.3 / 59.5 | 77.7 / 93.5 / 47.8 / 59.2 | 75.7 / 94.0 / 49.2 / 59.7 |
| S (1,180) | 74.3 / 89.0 / 52.1 / 61.2 | 75.8 / 96.1 / 75.6 / 75.7 | 80.8 / 95.0 / 72.1 / 76.2 |
| N (5,908) | 93.2 / 81.6 / 91.9 / 92.6 | 98.3 / 66.0* / 86.6 / 92.1 | 98.0 / 66.4* / 86.7 / 92.0 |

**Confusion, LGBM**

| truth | N | L | R | V | S | multi |
|---|---|---|---|---|---|---|
| N | 5509 | 0 | 10 | 59 | 322 | 8 |
| L | 39 | 29 | 4 | 25 | 12 | 31 |
| R | 305 | 12 | 12 | 113 | 278 | 49 |
| V | 40 | 0 | 0 | 297 | 43 | 117 |
| S | 93 | 0 | 1 | 138 | 649 | 147 |
| multi | 8 | 3 | 0 | 83 | 50 | 55 |

**Confusion, CNN causal**

| truth | N | V | S | multi |
|---|---|---|---|---|
| N | 5806 | 33 | 48 | 21 |
| L | 119 | 20 | 0 | 1 |
| R | 687 | 50 | 25 | 7 |
| V | 32 | 287 | 81 | 97 |
| S | 44 | 176 | 653 | 155 |
| multi | 14 | 90 | 40 | 55 |

Share of beats predicted as each class on NORMAL records:

| Dataset | LGBM | CNN |
|---|---|---|
| PTB-XL | N 97.7 %, A 1.6 % | N 99.7 % |
| Chapman | N 99.1 %, A 0.5 % | N 99.8 % |

## How much performance drops, and why

The table below is reference → PTB-XL → Chapman. Se is prevalence-independent; F1/PPV move with label prevalence (e.g. S is 14 % of the Chapman cohort but 5 % of PTB-XL's), so read F1 with care.

| | V Se (CNN / LGBM) | S Se (CNN / LGBM) | L Se (LGBM) | R Se (LGBM) |
|---|---|---|---|---|
| DS2 10-s strips | 94.5 / 93.1 | 84.9 / 78.3 | 6.7 | 0.0 |
| PTB-XL | 84.7 / 90.9 (−10 / −2) | 71.5 / 68.7 (−13 / −10) | 45.0 | 4.7 |
| Chapman | 77.7 / 83.3 (−17 / −10) | 75.8 / 74.3 (−9 / −4) | 41.2 | 4.4 |

Specificity for V and S goes **up** externally (CNN S Sp 75 → 94–96 %). As a result, CNN S F1 is higher on the external sets than on DS2.

**Why performance drops**

1. **Record length (10 s vs 30 min).** This is the largest single factor for the CNN.
   - Its RR inputs are normalised by 80- and 400-beat running means. With about 11 beats, those means barely exist.
   - On MIT-BIH itself, cutting to 10 s halves the CNN's beat-level S sensitivity (87 → 45 %). The model was not retrained and nothing else changed.
   - LGBM's per-record amplitude/RR normalisation is computed from about 10 beats, and 4–5 HRV/non-linear features are undefined or noisy (imputed). Its beat-level DS2 numbers barely moved, but it was already weak on DS2.
   - Aggregation over about 11 beats with a k = 1 rule makes one false beat enough for a false record alarm: 1.6 % of LGBM beats are called A on PTB-XL normals, which gives 17 % S false alarms.
2. **Label granularity and taxonomy.**
   - The external labels are per record and come from 12-lead reports, while the models were trained on per-beat MIT-BIH labels.
   - BBB is a 12-lead diagnosis made mostly from QRS width and V1/V6 morphology. Lead II alone carries little RBBB information, so lead II at 10 s barely detects RBBB (about 4 %).
   - A PTB-XL "PVC" or Chapman "VPB" label does not guarantee that a PVC falls inside the analysed lead-II window or is visible there.
   - In AAMI the CNN folds L and R into N, so it cannot flag BBB by design.
   - Many PVC-positive records fire *both* V and S ("multi" column). The prematurity RR feature pushes compensatory-pause neighbours or the PVC itself toward S.
3. **Lead and electrode differences.**
   - MIT-BIH MLII is a modified ambulatory chest-placement lead; PTB-XL and Chapman use standard limb lead II from 12-lead resting carts.
   - QRS axis, amplitude and T-wave morphology differ. This mainly hurts morphology-based classes: V Se drops more for the CNN, which depends on matched-filter templates, than for LGBM, which uses amplitude-normalised features.
   - The Farag pipeline has no band-pass, so cart-ECG baseline wander and mains noise reach the derivative input unfiltered. Chapman is noisier, and its V Se drops most there.
4. **Sampling rate.** 500 → 360 Hz polyphase resampling is benign: it loses no information below 180 Hz and both models band-limit far lower. It was not a measurable factor; the detector also works on the resampled signal.
5. **Population.**
   - The external normals are healthy resting ECGs, which explains the higher specificity.
   - MIT-BIH DS2 has only a few LBBB/RBBB patients (111, 214 / 212, 231). The 0–7 % in-domain BBB sensitivity is an inter-patient failure of the LGBM features (see progress.md), not something specific to cross-dataset use.
6. **R-peak detection** is a minor factor: XQRS + snap reaches Se 97–98 % and PPV 99 %+ on MIT-BIH strips. Using .atr peaks changes strip metrics by ≤ 4 pp. On the external sets only 56 (PTB-XL) and 33 (Chapman) records had < 3 detected beats. This was not validated against external beat annotations, because none exist.

## Reproduce

```bash
cd $ROOT/zeroshot ; PY=$ROOT/.venv/bin/python   # TF 2.22rc0 CPU + LightGBM + wfdb 4.3.1
# data
wget -O ../datasets/external/ptbxl/ptb-xl-1.0.3.zip https://physionet-open.s3.amazonaws.com/ptb-xl/ptb-xl-1.0.3.zip
wget -O ../datasets/external/chapman/ecg-arrhythmia-1.0.0.zip https://physionet-open.s3.amazonaws.com/ecg-arrhythmia/ecg-arrhythmia-1.0.0.zip
(cd ../datasets/external/ptbxl && unzip -q ptb-xl-1.0.3.zip -x "*/records100/*")
(cd ../datasets/external/chapman && unzip -q ecg-arrhythmia-1.0.0.zip)
$PY -I sanity_models.py            # frozen models reproduce DS2 numbers (~1 min)
$PY -I validate_detector.py        # detector Se/PPV on MIT-BIH strips (~8 min)
$PY -I mitbih_strips.py            # MIT-BIH DS1/DS2 10-s strips, both models (~10 min)
$PY -I choose_rules.py             # rules.json + DS2 reference report (seconds)
$PY -I prepare_external.py ptbxl ; $PY -I prepare_external.py chapman   # ~1 min each
$PY -I run_external.py ptbxl ; $PY -I run_external.py chapman            # ~6-8 min each
```

## Files created (box)

The detector configuration is set in `featurize.py` as `DETECTOR`.

- **Scripts in `zeroshot/`:**
  - `common.py`
  - `featurize.py`
  - `validate_detector.py`
  - `sanity_models.py`
  - `mitbih_strips.py`
  - `recmetrics.py`
  - `choose_rules.py`
  - `prepare_external.py`
  - `run_external.py`
- **Logs in `zeroshot/`:** `*.log`
- **Outputs in `zeroshot/out/`:**
  - `detector_validation.{csv,txt}`
  - `mitbih_strips.pkl`, `mitbih_beats.pkl`, `mitbih_beat_level.csv`, `mitbih_DS{1,2}_*_strip_metrics.csv`, `mitbih_strip_report.md`
  - `rules.json`
  - `{ptbxl,chapman}_manifest.csv`, `{ptbxl,chapman}_leadII_360.npz`, `{ptbxl,chapman}_records.pkl`, `{ptbxl,chapman}_det_*_metrics.csv`, `{ptbxl,chapman}_report.md`
- **Data:**
  - `datasets/external/ptbxl/`: zip plus extracted, 4.4 GB
  - `datasets/external/chapman/`: zip plus extracted, 7.7 GB. The zips can be deleted to save about 4.3 GB.
- **Reports:** `reports/zeroshot.md` (this file) and `reports/zeroshot.RESUME.md`.
- No existing files were modified. `source/features.py` and `HasanML/paper3/src/{preprocess,rr}.py` are only imported.

## Open issues / caveats

- **Rules come from in-sample predictions.** The aggregation rules (k = 1 for V/S) were fitted on DS1, which is the models' training data. A held-out rule-fitting set, e.g. leave-records-out models, would probably choose k ≥ 2 for S and cut false alarms. Rules must not be tuned on PTB-XL or Chapman.
- **Metrics are per record, not per patient.** PTB-XL has several records for some patients. No confidence intervals were computed; with n_pos = 187–1,180, the 95 % CI on Se is about ±3–7 pp.
- **Short-record RR features are the main lever.** A short-record-aware RR feature, e.g. normalising by the strip median or training with 10-s context, could recover much of the CNN's S sensitivity. That would need retraining, so it is out of scope here.
- **BBB detection needs more than lead II.** It needs multi-lead or QRS-width features, or a model that does not depend on per-patient morphology. Lead II at 10 s is close to uninformative for RBBB.
- **Chapman code mapping.** Some codes (713427006, 733534002, 10370003) are undocumented in the dataset CSV and were mapped from SNOMED-CT. Chapman's 164890007, nominally atrial flutter (8 k records), is used as AF in that dataset; it is excluded either way.
- **Detector not checked on target data.** The detector was not validated on the external data, which has no beat annotations. Records with < 3 detected beats (56 PTB-XL, 33 Chapman) mostly end up as "N".
