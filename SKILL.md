
## System architecture

```
Wearable (ECG acquisition, TinyDL screening, on-demand ECG)
   │
   ├── Fixed path (trusted) ──► Fixed gateway: multiclass ECG classification + IDS/IPS
   │                                 │
   └── Mobile path (untrusted) ──► Mobile relay: transport bridge only; no device identity or decryption keys
                                        │
                                        ▼
                              Cloud backend: multiclass ECG (mobile path only),
                              IAM / data / API
                                        │
                                        ▼
                          Caregiver / Patient dashboard: ECG + alerts
```

- **Wearable hardware:** AD8232 + ESP32-S3
- **Fixed gateway hardware:** Raspberry Pi 4B (trusted — may decrypt and classify)
- **Mobile relay:** Android phone acting only as an untrusted communication bridge. It is not a MedMusketeer device/security principal, is not separately enrolled in the device registry, and receives no MedMusketeer client certificate or payload decryption keys. Wearable identity/authenticity remains end-to-end to the cloud.
- **Fixed path:** authenticated links; trusted gateway does local multiclass classification.
- **Mobile path:** cloud performs the multiclass classification since the relay cannot decrypt. Cloud inference is mandatory on this path.
- Authenticated on-demand commands return over the corresponding protected route.

## Evaluation plan (proposed measures — nothing below is a measured result yet)

- **ECG:** Macro-F1, balanced accuracy, recall
- **Edge:** latency, RAM, flash, energy
- **Security:** detection rate, false positives, response time
- **System:** alert latency, reliability, recovery



**Risks/dependencies:** limited/imbalanced ECG data, wearable resource constraints, hardware/dataset/network/library dependencies.

**GenAI boundary:** GenAI (LLMs) may assist with coding/debugging; final code, design decisions and results must be human-reviewed and validated by the team — never presented as unverified LLM output.

## Datasets referenced

- **MIT-BIH Arrhythmia Database** — annotated beat classification; subject-disjoint splits.
- **PTB-XL** — 12-lead, record-level rhythm-label study with multilabel structure; five diagnostic superclasses ≠ arrhythmia classes. Keep label spaces separate from MIT-BIH; do not conflate the two tasks.


## Timeline / milestones

- **FYP-1 Mid (Week 8):** model baselines, ECG acquisition, gateway transport, dashboard prototype, requirements/design.
- **FYP-1 Final (Week 16):** **complete implementation** — both routes, IDS/IPS, IAM, cloud; integrated baseline validation.
- **FYP-2 Mid (Week 8):** model/policy refinement, hardware profiling, attack stress tests, failure recovery, deployment trials.
- **FYP-2 Final (Week 16):** held-out evaluation, final deployment, installation guide, report, handover.

(Weeks 8/16 and the detailed deliverables under each milestone are planning assumptions, not confirmed handbook dates — flag this if asked to treat them as fixed.)



1. **Never state or imply measured results.** All performance/security figures in the proposal are *proposed acceptance targets*. No experiments have been run yet. Always phrase future outputs as "target"/"planned"/"to be evaluated," never as achieved results, unless the user explicitly supplies new measured data.
2. **Don't expand scope silently.** If a request implies work outside "Committed" 
4. **Keep MIT-BIH and PTB-XL label spaces separate** — don't treat PTB-XL diagnostic superclasses as arrhythmia classes interchangeable with MIT-BIH beat labels.
6. **This file is the scope reference, not the progress tracker.** A separate skill will track what's actually been completed against this plan — don't try to infer completion status from this file.

---

## ECG acquisition hardware configuration (ADS1292 + ESP32-S3)

| Parameter            | Recommended setting                                      | Why                                                            |
| -------------------- | -------------------------------------------------------: | -------------------------------------------------------------- |
| ECG channels         | 1 primary ECG channel                                    | Keep model/input simple                                        |
| Acquisition rate     | **500 SPS**                                              | Native ADS1292 rate                                            |
| Model rate           | **360 Hz**                                               | Match MIT-BIH                                                  |
| PGA gain             | **6** initially                                          | Good starting point; verify against electrode signal amplitude |
| Input                | Differential ECG                                         | Required for ECG measurement                                   |
| Lead configuration   | **Lead II-equivalent**                                   | Closest practical target for MIT-BIH's MLII channel            |
| RLD                  | **Enabled**                                              | Improves common-mode rejection                                 |
| Lead-off detection   | Optional during development; disable for clean ML signal | Avoid injecting lead-off behaviour into training signal        |
| Respiration          | **Disabled**                                             | Not needed for arrhythmia classifier                           |
| Internal test signal | Disabled during normal operation                         | Only use for hardware validation                               |
| ADC resolution       | 24-bit                                                   | Native ADS1292                                                 |
| Digital interface    | SPI                                                      | Native interface                                               |
| Filtering            | External/software preprocessing                          | Make identical to MIT-BIH preprocessing                        |
| Output to model      | 360-Hz normalised ECG                                    | Same representation as training                                |

### 500 SPS → 360 Hz resampling pipeline

ADS1292 outputs at 500 SPS; MIT-BIH was digitised at 360 Hz. The rates cannot be aligned natively — resample in software:

```
ADS1292 (500 SPS)
      │
  anti-alias / low-pass filter
      │
  resample to 360 Hz
      │
  ECG classifier
```

- MIT-BIH was digitised at 11-bit resolution over a 10 mV range; ADS1292 is 24-bit — no deliberate bit-dropping needed.
- The conversion formula is: `physical (mV) = (digital − baseline) / adc_gain`  (e.g. gain = 200, baseline = 1024 for record 100).

---

## ML pipeline — model inputs, feature engineering, and training

### ECG windowing

- **Standard input:** 2-second window at 360 Hz = **720 samples per channel** (the working default; kept as an experimental value).
- **Alternative tested:** 1-second centred on R-peak = 360 samples per channel.
- Two-channel input shape used throughout: `[2 × 720]`.

### LightGBM — handcrafted features

42 features extracted **per channel**, concatenated to give an 84-feature vector for two-channel input:

```
RR interval stats (10):   RR_prev, RR_current, RR_next, mean_RR, median_RR, sd_RR, min_RR, max_RR, RR_range, RR_CV
QRS morphology (6):       QRS_duration, QRS_amplitude, QRS_area, QRS_energy, QRS_max_slope, QRS_min_slope
R/Q/S amplitudes (5):     R_amplitude, R_prominence, Q_amplitude, S_amplitude, R_S_ratio, Q_R_ratio  [note: 6 values]
P wave (3):               P_amplitude, P_duration, P_area
T wave (4):               T_amplitude, T_duration, T_area, T_R_ratio
Interval durations (5):   PR, QRS_duration, QT, QTc, ST
Statistical (8):          mean, std, RMS, max, min, range, skewness, kurtosis
```

Experiments planned:
- Experiment 1: LightGBM, 42 features (single-channel).
- Experiment 2: LightGBM, 84 features (two-channel concatenation).
- Experiment 3: LightGBM feature importance → top 10/20/30 features to determine whether full 84 features are needed.

### Deep-learning model input shapes

| Model     | Input shape   | Notes                                 |
| --------- | ------------- | ------------------------------------- |
| CNN       | (N, 2, 720)   | channels-first                        |
| LSTM      | (N, 720, 2)   | time-steps first                      |
| KANformer | (N, 720, 2)   | same as LSTM                          |

Deep models learn QRS morphology and temporal structure directly from the waveform — explicit feature extraction is not needed for them.

### Knowledge Distillation (KD) pipeline

**Teacher:** KANformer (high-capacity, trained on two-channel MIT-BIH, 5-class arrhythmia labels).  
**Student:** 1D-CNN (lightweight, primary KD experiment). LSTM student is an optional second experiment.

Training procedure:
1. Train KANformer on ground-truth labels (normal supervised training).
2. Train standalone 1D-CNN and LSTM as baselines (no KD).
3. Freeze KANformer; train 1D-CNN student using KD loss = Cross-Entropy (hard labels) + KL Divergence (soft teacher predictions).
4. Quantise/prune the KD-CNN → INT8 model → deploy on ESP32 + ADS1292.

**KD Loss:**  `L = α · CE(student, true_labels) + (1−α) · KL(student_soft, teacher_soft)`

### Train / test split — patient-aware (record-level)

Do **not** randomly split individual beats — beats from the same patient would contaminate both sets. Instead:

```
MIT-BIH records
      │
  Record/patient-level split  (e.g. 80 % records → train, 20 % → test)
      │
  Preprocessing
      │
  Beat extraction
      │
  Feature extraction
      │
  Training / testing
```

Subject-disjoint splits are required for honest generalisation estimates on MIT-BIH.

### Model comparison summary

| Model     | Role                     | Input           | Notes                          |
| --------- | ------------------------ | --------------- | ------------------------------ |
| LightGBM  | Handcrafted-feature baseline | 42 or 84 features | Fastest to train, interpretable |
| 1D-CNN    | Lightweight DL baseline  | (N, 2, 720)     | —                              |
| LSTM      | Lightweight DL baseline  | (N, 720, 2)     | —                              |
| KANformer | High-capacity teacher    | (N, 720, 2)     | Not deployed to edge           |
| KD-CNN    | Distilled edge model     | (N, 2, 720)     | Target for ESP32 deployment    |
| KD-LSTM   | Optional distilled model | (N, 720, 2)     | Optional second experiment     |

Primary comparison axis: **Macro-F1 / sensitivity vs deployment cost** (parameters, RAM, flash, latency, energy) — not raw accuracy alone.
