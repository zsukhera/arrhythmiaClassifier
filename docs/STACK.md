
## Wearable hardware — ECG acquisition (ADS1292 + ESP32-S3) [Tentative - will be finalised once experiments are run]

### Chip overview

| Component | Part | Role |
|---|---|---|
| MCU | ESP32-S3 | BLE + WiFi, runs TinyDL inference, drives SPI bus |
| AFE / ADC | ADS1292 (Texas Instruments) | Analog front-end — two 24-bit ECG channels, on-chip PGA and RLD driver |
| Interface | SPI | ADS1292 is a pure SPI peripheral; one CS pin per chip |

The ADS1292 is a dedicated medical-grade ECG AFE — it is not a generic ADC. It integrates a differential amplifier, programmable gain, right-leg drive (RLD) output, lead-off detection, and the ADC in one chip. **Do not treat it like a raw ADC attached to electrodes — the gain, RLD, and filter settings must be configured explicitly over SPI before acquisition.**

### Key electrical characteristics

| Parameter | Value | Notes |
|---|---|---|
| ADC resolution | 24-bit | Effective noise-limited bits are lower (typically ~18–20 ENOB at ECG bandwidths) |
| Native output rate | **500 SPS** | The rate used for acquisition |
| MIT-BIH target rate | **360 Hz** | Mismatch — software resample required (see below) |
| Input type | Differential | Electrodes connect between IN1P/IN1N (and IN2P/IN2N for channel 2) |
| PGA gain | **6** (initial) | Configurable: 1, 2, 3, 4, 6, 8, 12. Verify against actual electrode signal amplitude before finalising |
| RLD | **Enabled** | Reduces common-mode noise; driven by the ADS1292's internal RLD amplifier |
| Lead-off detection | **Disabled during normal ML acquisition** | Injects a small AC current that contaminates the training signal — enable only for hardware validation or explicit lead-status monitoring |
| Respiration channel | **Disabled** | Second channel used for lead II-equivalent ECG only; respiration not needed for the arrhythmia classifier |
| Internal test signal | **Disabled during normal operation** | Use only for initial hardware bring-up / self-test |
| Digital interface voltage | 3.3 V (matches ESP32-S3 IO) | No level shifting required |

### Lead configuration

- **Lead II equivalent** — closest practical target to MIT-BIH's MLII channel (the reference lead in nearly all MIT-BIH records).
- Use differential input: IN1P = LA electrode, IN1N = RA electrode, RLD electrode on RL. This mimics the standard Lead II limb placement.
- **Channel 2** is used for a second lead (e.g. V1, V5) in the two-channel MIT-BIH-matched setup — the exact placement mirrors the second channel used in each MIT-BIH record being replicated.

### 500 SPS → 360 Hz resampling pipeline

ADS1292 cannot be clocked to exactly 360 Hz. Resample in software on the ESP32-S3 (or on the gateway for heavier filters) before feeding the classifier:

```
ADS1292 output (500 SPS, 24-bit, mV)
         │
     Anti-alias / low-pass filter
     (cutoff < 180 Hz to satisfy Nyquist for 360 Hz output)
         │
     scipy.signal.resample_poly or polyphase FIR
     (500 → 360 via integer ratio 9:5 or equivalent)
         │
     360 Hz ECG (matches MIT-BIH training data rate)
         │
     Classifier / feature extraction
```

- MIT-BIH was digitised at 11-bit effective resolution over a 10 mV range (ADC gain = 200 units/mV, baseline = 1024, usable range ≈ 2048 levels). ADS1292 delivers 24-bit — **do not artificially truncate bits** to match MIT-BIH; the higher resolution helps upstream signal quality before resampling and filtering normalise the representation.
- After resampling, normalise the mV signal to match MIT-BIH's statistical distribution (per-record z-score or band-pass + baseline-wander removal) so the model receives a consistent input regardless of electrode contact variation.

### SPI communication — ESP32-S3 integration notes

- ADS1292 requires a specific SPI initialisation sequence: power-on reset → SDATAC command → register writes (gain, data rate, RLD, lead-off, respiration) → RDATAC command → START pin high to begin conversion.
- Data-ready signalled by DRDY pin going low; read 3 bytes × 2 channels per sample in continuous-data mode.
- SPI clock: up to 16 MHz rated; use ≤ 4 MHz for reliable long-wire connections in a breadboard/prototype context.
- All SPI writes to ADS1292 registers must be verified by a subsequent readback — the chip does not acknowledge incorrect writes at the protocol level.

### Signal quality / preprocessing checklist (hardware validation)

1. Verify `init_value` and `checksum` equivalents from the first received frame against expected values (confirms SPI wiring and register config are correct before recording real ECG).
2. Confirm mV range is within ±5 mV and not clipping (ADS1292 full-scale at gain=6 is ≈ ±0.44 V before clipping, well within biological ECG amplitudes).
3. Check that the mean signal is close to 0 mV after baseline-wander removal — a persistent DC offset after filtering suggests RLD is not connected or PGA gain is misconfigured.
4. Disable lead-off detection before any ML-intended recording session.

## Open next steps (not yet decided)
- Concrete Postgres schema (`devices`, scheme-neutral `device_credentials`, IDS/IPS event log, attack log, clinical/application relationship tables)
- Ingestion API's normalized message envelope (fields, where hop-timestamps live)
- MQTT topic structure for both paths (telemetry vs alert-only, fixed-classified vs cloud-classified results)
- Concrete transport/auth adapter interface and normalized authenticated-device envelope
- Mobile-path envelope format details: header fields, signature/encryption scheme, how `device_id` is carried (decision to draft this out is pending — envelope crypto *approach*, i.e. Option B, is decided; the wire format is not)
- CN/SAN pinning implementation specifics per side (ESP32-S3/mbedTLS validation code; cloud-side and gateway-side equivalent checks) — note this now explicitly includes the **gateway's outbound check of the cloud's identity**, not just the wearable's checks, to avoid the same chain-trust-only gap on that leg
- Heartbeat envelope interval/timeout thresholds for mobile-path liveness inference, and how `status`/`last_seen` are represented in the device registry schema to capture the fixed-path vs mobile-path distinction
- Cloud cert rotation is deferred to future scope (see "Cloud cert rotation" section) — nothing to build for FYP-1; revisit only if a future deployment needs it
