---
name: medmusketeer-fyp
description: Locked project context for "MedMusketeer" — a Final Year Project (FYP) titled "Design and Development of a Secure Edge-Intelligent Cardiac Telemetry System for Remote Patient Monitoring." Use this skill whenever the user asks for help on this FYP, MedMusketeer, arrhythmia/ECG classification work, the IDS/IPS or gateway security components, the dashboard, timeline/milestones, or work division among the team — even if they don't name the project explicitly and just say things like "my FYP". Always consult this skill before writing code, slides, reports, or planning documents for this project, so scope, hardware, ownership and terminology stay consistent with the approved proposal. Do NOT let scope silently expand beyond what's listed here as "Committed" without the user explicitly asking to change scope.
---

# MedMusketeer — FYP Project Context

This skill is a **scope-lock / context memory** for one Final Year Project. It exists so that any LLM session working on this project (writing code, slides, reports, evaluation scripts, etc.) starts from the same facts instead of re-deriving or drifting from them. A separate skill (to be created later) will track *progress* against this scope — this skill only defines *what the project is*.

**Source of truth**: the FYP-1 proposal defence deck (`medmusketeer_proposal_defence.tex`) and the user's explicit clarifications, which override both underlying proposal documents. If anything in a future conversation conflicts with this file, ask the user rather than assuming — and if the user confirms a change, update this file.

## Project identity

- **Title:** Design and Development of a Secure Edge-Intelligent Cardiac Telemetry System for Remote Patient Monitoring
- **Short name:** MedMusketeer
- **Stage:** FYP-1 (Stream A: research-derived, proposed)
- **Institution:** FAST National University of Computer and Emerging Sciences, Islamabad
- **Supervisor:** Mr. Shams Farooq — **Co-supervisor:** Dr. Muhammad Asim
- **Team & ownership:**
  - **Zain Ul Arifeen Sukhera** — Medical: ECG pipeline, edge models, medical evaluation
  - **Hasan Adil** — IDS/Cyber: traffic features, IDS models, IPS, attack evaluation 
  - **Hamza Tariq** — Platform: IAM, security, hardware, cloud, dashboard

## Core problem

Reliable ECG (electrocardiogram) screening on resource-constrained wearable/edge devices is hard because of limited compute, memory and energy. Network attacks can additionally disrupt secure communication and delay or compromise delivery of critical health alerts. The project must jointly solve a **model problem** (lightweight arrhythmia classification) and a **security problem** (protecting telemetry and detecting attacks) without treating them independently.

- **Deployment setting:** old-age homes, hospital wards, remote patients
- **Stakeholders:** patients, doctors/caregivers, platform admin (admin manages device records/status and IDS/IPS event visibility)

## Research orientation / publication goal

- The project's **core research contribution** is experimenting with and benchmarking **recent, IoT-friendly, resource-light secure communication and authentication protocols** for the wearable↔gateway and gateway↔cloud links (e.g. lightweight/certificateless auth schemes, MQTT+TLS vs alternatives such as CoAP+DTLS). This benchmarking axis is **transport/auth protocols only** — it does NOT extend to comparing IDS/IPS approaches; IDS/IPS remains an implementation deliverable, not a benchmarking/publication axis.
- Publication is an explicit goal — findings, benchmarks, and observed issues/limitations/future directions around protocol choice (including the mobile-gateway relay approach) should be tracked with an eye toward a paper, not just a working demo.
- **The mobile-relay-vs-Tethys distinction is a design justification, not a standalone research contribution.** Tethys used smartphone-as-gateway for a different purpose; this project's relay-only, no-decryption-key design should be framed as "we adopted X because Y, distinct from Tethys's use case," not oversold as a novel Tethys alternative in slides/report.
- Because the transport/auth layer is itself the thing being benchmarked, cloud-side ingestion should stay **protocol-agnostic/swappable** rather than hardwiring one broker/protocol as if it were a settled choice.

**Additional benchmarking axes (latency-focused):**
- **Path:** fixed gateway vs mobile relay.
- **Transmission mode:** continuous telemetry vs alert-only-on-detection (feasible because the wearable already runs on-device TinyDL screening per the architecture — alert-only mode uses that screening result as the send trigger instead of shipping raw ECG continuously).
- **FYP-1 scoping decision:** fix on **one** protocol/auth candidate and vary **path × mode** (2×2) for FYP-1. Full factorial across multiple protocol candidates as well is out of scope for FYP-1 — flag if a future request assumes the full cross was run.
- **Fixed protocol for the FYP-1 path×mode benchmark: MQTT+TLS.** Other protocol/auth candidates are compared against this baseline as a separate axis, not folded into the path×mode matrix for FYP-1.
- **Implementation detail (checkpoint timestamps, clock sync, the continuous-vs-alert-only confound) lives in the companion `medmusketeer-stack` file** — this file only needs the axes and scoping decision.

## Locked scope

**Committed (must be delivered):**
- Arrhythmia classification (lightweight, multiclass)
- Both communication paths (fixed-gateway and mobile-relay — see architecture below)
- IDS/IPS (intrusion detection/prevention at the gateway)
- IAM (identity and access management)
- Cloud platform (data/API/dashboard backend), including:
  - Device registry + live status for registered wearables/fixed gateways, visible to admin (the mobile relay is not a registered device)
  - Ingestion of IDS/IPS events/records from the gateway for admin visibility on the platform (the IDS/IPS detection logic itself stays gateway-side — cloud only stores/surfaces the records)

**Explicitly excluded (out of scope — do not implicitly add):**
- Physiological spoofing
- Adversarial ECG manipulation
- Firmware integrity
- Gateway/platform host integrity beyond what network-based IDS/IPS can detect (rootkit persistence, tampered OS image, a locally-exploited vulnerability with no anomalous network signature, physical/insider access to the RPi). Same category as firmware integrity above — IDS/IPS as scoped operates on network traffic and structurally cannot see host-level compromise. The gateway's trustworthiness as a node is an assumed boundary, not something this project verifies.

**User clarifications that override both source proposals:**
- Replay attack handling **is included** in scope.
- The mobile relay is **relay-only** — it must never hold or use payload decryption keys. The Android phone is **not a MedMusketeer device/security principal**: it has no separate device-registry identity or MedMusketeer client certificate merely for relaying traffic; the wearable remains the authenticated principal end-to-end on the mobile path.
- "FYP-1 Final" targets a **complete, integrated implementation** of both routes, IDS/IPS, IAM and cloud (not just a partial core).
- **User roles/groups are tentative** (patient, doctor, admin identified so far — more may be added/renamed). IAM design must keep roles/policies as data (configurable), not hardcoded, until finalized.

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

## POC / current evidence

- Dashboard (proof of concept, deployment/interaction only): https://med-muskeeter.vercel.app/
- **Important:** the POC only demonstrates dashboard interaction and deployment. Model performance, hardware performance, and security validation are all still proposed/unperformed — never imply the POC validates them. 
- **The new dashboard will also be hosted on this same URL


**Key related research consulted:** (new research will be added in the Progress md file)
- Knowledge-distillation lightweight arrhythmia classification (teacher/student)
- KAN-Former (multi-scale conv + attention + KAN, pruned/INT8, Cortex-M4)
- RNN arrhythmia classification on STM32 (MIT-BIH, AAMI protocol)
- CMAP — certificateless MQTT authentication for medical IoT (ECC, Raspberry Pi)
- Ultra-lightweight anonymous auth for telehealth (Ascon AEAD)
- Tethys — untrusted smartphone-as-gateway sensor collection model (basis for the mobile relay-only design)
- MQTT broker end-to-end security models (double-encryption/broker re-encryption overheads)

This defines *how* the already-committed IDS/IPS deliverable is meant to work; it does not expand scope. Keep distinct from the protocol/auth benchmarking axis (Research orientation section) — this is the IDS/IPS implementation methodology, not a benchmarking comparison. **Full IDS/IPS methodology and the attack-testing infrastructure now live in the companion `medmusketeer-stack` file** — consult that for implementation detail; this file only needs you to know these components exist and are committed scope.



1. **Never state or imply measured results.** All performance/security figures in the proposal are *proposed acceptance targets*. No experiments have been run yet. Always phrase future outputs as "target"/"planned"/"to be evaluated," never as achieved results, unless the user explicitly supplies new measured data.
2. **Don't expand scope silently.** If a request implies work outside "Committed" (e.g., adversarial ECG defenses, firmware integrity, spoofing detection), flag that it's out of the locked scope before proceeding.
3. **Keep the mobile relay transport-only.** Any design/code touching the mobile path must not give it payload decryption capability or silently turn the Android phone into a separately enrolled MedMusketeer device/security principal. The wearable remains the authenticated identity on the mobile path.
4. **Keep MIT-BIH and PTB-XL label spaces separate** — don't treat PTB-XL diagnostic superclasses as arrhythmia classes interchangeable with MIT-BIH beat labels.
5. **Keep the protocol/auth benchmarking axis scoped to transport + authentication only.** Don't drift it into an IDS/IPS-approach comparison — that stays an implementation deliverable. Don't oversell the mobile-relay-vs-Tethys distinction as a standalone publishable contribution; it's a design justification.
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
