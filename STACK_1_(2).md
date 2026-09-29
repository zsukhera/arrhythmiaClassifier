---
name: medmusketeer-stack
description: Cloud/platform stack and architecture decisions for MedMusketeer. Companion to medmusketeer-fyp (scope-lock) — that file defines *what* the project is; this file defines *how* the cloud/platform layer is built. Consult before writing cloud backend code, infra config, onboarding flows, or the attack-testing module.
---

# MedMusketeer — Cloud/Platform Stack & Architecture

This is a **technical decision record**, not the scope-lock file. If a decision here ever conflicts with `medmusketeer-fyp`, that file wins on scope; this file wins on implementation detail. Update this file as stack decisions evolve — it's expected to change more often than the scope-lock file.

## Core design principle: protocol-agnostic ingestion

Because transport/auth protocol benchmarking is a research goal (see scope-lock file), the cloud must not hardwire one protocol or one device-credential lifecycle as if settled. A **transport/auth adapter layer** sits between the wire and everything else:

```
[Wearable / fixed gateway / mobile-relay transport] → transport/auth adapter (MQTT+TLS today | future: CoAP+DTLS | certificateless-X)
                            │  (per-adapter: handshake time, auth overhead, msg size logged)
                            │
                            ├── device identity/auth layer
                            │   canonical device_id stays stable; credential lifecycle is scheme-specific
                            ▼
                    Ingestion API (single internal contract)
                            │
              ┌─────────────┼─────────────┐
              ▼             ▼              ▼
       Time-series DB   Relational DB   Notification bus
       (ECG)            (device/IDS     (dashboard)
                          registry/logs)
```

Each adapter speaks one protocol/authentication scheme, resolves the authenticated registered device to a stable canonical `device_id`, then hands a normalized message to the ingestion API. DB writes, classification triggers, IDS/IPS event storage, and notifications never depend on whether the device authenticated with X.509/mTLS, DTLS credentials, or a future certificateless scheme. Swapping/adding a protocol/auth candidate = new adapter/credential-lifecycle implementation, not a platform rearchitecture. The Android mobile relay is transport-only and is not resolved as a MedMusketeer device identity.

## Recommended stack

| Layer | Choice | Why |
|---|---|---|
| MQTT broker | **EMQX** (self-hosted, open source) | Native TLS for the FYP-1 baseline; auth-hook/ACL plugin system means swapping in a certificateless scheme later doesn't require replacing the broker; exposes per-client connect/disconnect events useful for latency checkpointing |
| Transport/auth adapter layer | Thin Python service(s) between the wire/broker and the ingestion API | Keeps protocol and device-authentication swap-in/out cheap; emits the same normalized authenticated-device contract downstream |
| Ingestion API | **FastAPI** | Single internal contract regardless of which adapter delivered the message |
| Time-series DB (ECG) | **TimescaleDB** (Postgres extension) | Hypertables for waveform chunks; same engine as relational data — one DB to run |
| Relational DB (device registry, credential references, IDS/IPS event log, attack log, application relationships/policies) | Same **Postgres** instance, separate schema | Stores MedMusketeer domain/device data; human authentication and coarse human roles stay in ZITADEL rather than being duplicated as a second IAM source of truth |
| Human IAM | **ZITADEL** | API-first OIDC/OAuth IAM for admins/doctors/patients and organizations; FastAPI integrates its management APIs so routine IAM administration stays inside the MedMusketeer platform UI |
| Notifications (web dash) | WebSockets via FastAPI, backed by Postgres LISTEN/NOTIFY or Redis pub/sub | Real-time alert push without a heavyweight message queue |
| Hosting | Self-managed VM — DigitalOcean Droplet or free-tier Oracle Cloud ARM instance | Full control over broker/adapter config; no platform-imposed protocol assumptions |

**Explicitly ruled out:**
- **AWS IoT Core** — provisioning model assumes X.509-per-device identity as the only path, which fights a future certificateless-auth comparison. Wrong fit given the protocol-benchmarking goal.
- **Managed MQTT (HiveMQ Cloud)** — works for the MQTT+TLS baseline itself, but won't allow custom auth hooks for latency checkpointing or future protocol swaps without leaving the managed tier.

## Benchmark design (current scope)

- **Fixed for FYP-1: MQTT+TLS.** This validates the path×mode harness; other protocol/auth candidates get compared against this baseline as a separate axis later, not folded into the path×mode matrix now.
- **Axes varied for FYP-1:** path (fixed gateway vs mobile relay) × transmission mode (continuous telemetry vs alert-only-on-detection, using the wearable's existing TinyDL screening as the trigger) — a 2×2, not a full factorial across protocols.
- **Methodology requirement:** checkpoint timestamps at each hop — wearable send/detect → gateway receive/classify (fixed path) → cloud arrive/classify (mobile path) → dashboard render — with clock sync (NTP) across devices. Naive wall-clock deltas across hops are invalid without this.
- **Known confound to report, not normalize away:** continuous mode's alert latency depends on where detection happens (gateway or cloud); alert-only mode collapses detection to the wearable. These are structurally different pipelines, not just faster/slower versions of one pipeline.

## Device onboarding

| Device/component | Connectivity | Onboarding approach (FYP-1) | Future upgrade |
|---|---|---|---|
| Wearable, fixed path | WiFi (client mode, joins same network as RPi gateway) | Manual pre-provisioning: identity credential/key generated at build time and registered by hand in the device registry | BLE-based WiFi provisioning (ESP-IDF `wifi_provisioning` component) or SoftAP+captive portal |
| Wearable, mobile path | BLE to phone relay | Same canonical wearable identity as the fixed path; BLE pairing/provisioning only establishes the local relay link. The wearable, not the phone, remains the authenticated security principal for patient telemetry. | More automated wearable provisioning if needed |
| Fixed gateway (RPi 4B) | Ethernet primary, WiFi fallback | Manual pre-provisioning as a registered infrastructure device | — |
| Mobile relay app (Android phone) | BLE to wearable + phone cellular/WiFi to cloud | **No MedMusketeer device enrollment, CSR, client certificate, or device-registry identity.** The app is an untrusted communication bridge that forwards the wearable's opaque protected envelope. It may use ordinary server-authenticated TLS/app transport to reach the cloud, but this transport must not grant it wearable identity or payload-decryption capability. | Relay-session hardening/rate limiting if required |
| Doctors/patients/admin (human) | N/A | ZITADEL-backed onboarding/login through MedMusketeer; FastAPI uses ZITADEL management APIs for routine user/role/organization administration | — |

**Mobile-relay rule:** the Android phone is **not a MedMusketeer device/security principal**. It must not be issued a device certificate, added as `mobile_gateway` in the device registry, or given payload decryption keys. The cloud authenticates the **wearable identity carried by the protected message/envelope**, while the phone only transports it.

**Hard rule:** compromise or replacement of the phone must not require rotating the wearable's long-term identity/payload keys merely because the relay changed. Relay transport credentials, if any are introduced later for abuse control, remain separate from MedMusketeer device identity and from payload decryption keys.

### Device provisioning / CSR-signing workflow

**CA custody:** one private MedMusketeer CA (`ca.key`/`ca.crt`), held by the platform/IAM owner as a passphrase-encrypted file on a single machine — never in the repo, never on any device. `ca.key` is only ever touched at signing time (below); it is not reachable at runtime by any wearable, gateway, or the cloud. Flagged explicitly as an FYP-1 shortcut: not HSM-backed, no offline signing ceremony — acceptable for project scope, would need hardening for real deployment.

**Provisioning sequence (one script, one custodian, per new device):**
1. Mint the canonical `device_id` in the registry first (`devices` row, `status: pending`) — the cert is issued *for* an identity that already exists, not the reverse.
2. Generate the device's own keypair **on the build/provisioning machine, not on-chip** (`openssl genrsa`). On-device key generation is deferred until on-device private-key storage hardening is decided — generating "securely" into plain flash gains nothing.
3. Build the CSR from that keypair, with CN/SAN set to the `device_id` from step 1 — this is where CN/SAN pinning's identity actually gets bound, not added later.
4. Sign the CSR against `ca.key`/`ca.crt` (the only step that touches the CA key) → `device.crt`.
5. Flash `device.key` + `device.crt` + `ca.crt` onto the device over serial.
6. Insert the `device_credentials` row (`auth_method: x509_mtls`, `credential_reference`, `status: active`), flip the `devices` row to active.

No network-exposed signing service for FYP-1 — local/manual only, given low device volume.

## Attack-testing / ground-truth infrastructure

- **Cloud: admin-only "Attack Test" module**, separate from the production patient/doctor data path.
  - `POST /attack-test/start` — `{attack_type, target_device_id, params}` → `attack_id`, logs `start_ts`
  - `POST /attack-test/stop` — `{attack_id}` → logs `end_ts`
  - Attack log table: `attack_id, type, target_device_id, start_ts, end_ts, triggered_by`
  - **Device registry must flag testbed vs production devices** — attack triggers should be impossible against a production-flagged device.
  - Correlates each attack window against the IDS/IPS event log to compute detection latency, false negatives (no alert inside window), false positives (alerts outside any window) — this is the ground-truth source for the evaluation plan's IDS metrics.
- **Phone module:** lives in a **dev/test build of the mobile relay app, stripped from production builds.**
  - Feasible without root (application-layer): MQTT flood, replay (duplicate a captured relayed payload — natural fit since the relay already handles encrypted payloads without decrypting them), malformed/fuzzed MQTT packets, TCP connection-flood/port scan.
  - Not feasible from an unrooted phone (network-layer, needs raw sockets/packet injection): ARP spoofing/MITM, SYN flood — run these from a separate testbed machine (laptop/RPi on the same test LAN), reporting into the same cloud Attack Test service.

## IDS/IPS methodology (Hasan's component — not yet measured, target/goal framing)

**IDS methodology:**
1. Train/develop model on public IoMT attack data
2. Freeze the model
3. Zero-shot test on the team's own physical hardware testbed (no retraining at this step)
4. Measure the performance drop between public-data training and the physical testbed (planned measurement)
5. Apply a small amount of local adaptation
6. Measure recovery after adaptation (planned measurement)
- **Target/goal:** adaptation to zero-day attacks on the physical testbed.

**IPS methodology — when the gateway is detected under attack, it should:**
Reject/throttle the malicious source → preserve ECG acquisition → preserve TinyDL screening → preserve the high-priority alarm channel → buffer unsent measurements → reconnect/recover → upload the backlog once clear.
- **Target/goal:** resilience when under attack — graceful degradation with recovery, not detection alone.

**Practical zero-shot testing procedure:**
1. **Match the feature space** between the public IoMT dataset and your own testbed's traffic-capture pipeline first — feature mismatch (not real domain shift) is usually the biggest cause of a misleading "performance drop."
2. Stand up the physical testbed generating genuine benign traffic.
3. Reproduce the public dataset's attack categories against the testbed — you can only zero-shot test attack types the model actually saw in training.
4. Run the frozen model in inference-only mode (no weight updates) on captured testbed traffic.
5. Compare detection rate/false-positive rate here vs. the model's known numbers on the public dataset's own test split — the gap is the performance drop.
6. Only then apply local adaptation and re-measure for recovery.

## Attack-testing / ground-truth infrastructure (supports IDS/IPS evaluation)

- **Cloud: admin-only "Attack Test" module**, separate from the production data path. Start/stop endpoints trigger labeled attack windows against **testbed devices only** (device registry distinguishes testbed vs production). Logs `attack_id, type, target_device_id, start_ts, end_ts, triggered_by`.
- Correlates each attack window against the IDS/IPS event log to compute detection latency, false negatives, false positives — the ground-truth source for the evaluation plan's IDS metrics.
- **Phone module: dev/test build of the mobile relay app only, stripped from production.** Covers root-free application-layer attacks (MQTT flood, replay, malformed/fuzzed packets, connection-flood/port scan). Network-layer attacks (ARP spoofing/MITM, SYN flood) run from a separate testbed machine, reporting into the same cloud service.


## Human IAM vs device identity/authentication

**Human IAM is separate from device authentication.** ZITADEL handles **people** (admins, doctors, patients), including login, OIDC/OAuth tokens, MFA, organizations, and coarse human roles. Routine IAM operations should be exposed through the MedMusketeer admin UI via FastAPI calling ZITADEL's management APIs; platform admins should not need to open the ZITADEL console during normal operation. MedMusketeer/Postgres stores application-domain relationships such as doctor↔patient, patient↔wearable, organization/ward membership, and monitoring-station assignments.

**Device identity is protocol/authentication-neutral.** The registered MedMusketeer device security principals are the **wearable** and the **fixed gateway**. Each has a stable canonical `device_id` that is independent of whichever credential/authentication scheme is currently under test. The Android mobile relay is deliberately excluded from the device-identity model.

The transport/auth adapter must normalize successful device authentication into a common internal result, conceptually:

```text
{
  device_id: <canonical MedMusketeer device UUID>,
  device_type: wearable | fixed_gateway,
  auth_method: x509_mtls | dtls_<scheme> | certificateless_<scheme>,
  authenticated: true,
  auth_metadata: {...}    # benchmark/debug metadata only; downstream logic must not depend on its scheme-specific shape
}
```

**Permanent device registry fields:** `device_id` (UUID, canonical), `device_type` (`wearable` or `fixed_gateway`), `status`, `testbed_flag`, `owner_ref`/assignment relationship, and `last_seen`. Do **not** make certificate serial/expiry mandatory columns of the core device identity record.

**Scheme-specific credential records:** keep credential/authentication data in a separate structure such as `device_credentials`: `credential_id`, `device_id`, `auth_method`, `credential_reference`, `status`, `created_at`, `expires_at` (nullable), and scheme-specific `metadata`. This lets one physical device retain the same `device_id` while its authentication method changes during experiments.

**Credential lifecycle is replaceable, not permanent platform logic.** Expose generic lifecycle operations such as `enroll`, `authenticate`, `rotate_or_update_credential`, and `revoke_or_disable`. Their implementation depends on the selected candidate:
- **MQTT+TLS/X.509 baseline:** device keypair/CSR or pre-provisioned key material → CA certificate issuance → mTLS authentication → certificate renewal/rotation → certificate revocation/check.
- **Future DTLS/lightweight/certificateless candidate:** use that scheme's own registration, authentication, key-update, and revocation/disable procedure. It must not be forced into certificate terminology if the scheme has no certificates.

For FYP-1, manual/pre-provisioned credential enrollment is acceptable for the baseline; automation of certificate renewal is not a permanent architectural requirement because later candidates may use a different lifecycle entirely.

**Registered device handling:**
- **Wearable:** generate/register the candidate-specific identity credential at build/onboarding time. The same canonical wearable `device_id` is used on both the fixed and mobile paths.
- **Fixed gateway:** register as an infrastructure device and provision whichever candidate-specific credential the current experiment requires.
- **Mobile relay:** **no MedMusketeer device credential is issued.** It does not receive a device certificate or registry identity merely for forwarding patient telemetry.

**Critical separation:** ZITADEL human IAM must never become the device-authentication mechanism, and the mobile relay must remain cryptographically unable to impersonate the wearable or decrypt its payload. The cloud must validate the wearable's identity/authenticity from the protected end-to-end message/envelope after it traverses the phone. Any ordinary phone↔cloud app/session protection used for transport hardening or abuse control is separate from MedMusketeer device identity.

## Mobile-path envelope security (wearable ↔ cloud, via untrusted relay)

**Decision: Option B — application-layer envelope crypto, not a live end-to-end TLS/DTLS session through the phone.**

The wearable encrypts/signs the payload itself, addressed to the cloud, *before* handing it to the phone. The phone ships that opaque blob to the cloud over its own ordinary session-level TLS (authenticates the phone app only, not the wearable). Chosen over a live TLS/DTLS session relayed transparently through BLE because that approach is fragile against BLE link drops and heavier on the ESP32-S3 — application-layer envelope crypto survives an intermittent relay link and matches the "opaque envelope" framing already used elsewhere in this file.

**CA structure: single private MedMusketeer CA, reused for both gateway and cloud trust anchors.** The wearable flashes one `ca.crt` used to verify both the fixed gateway's TLS cert and the cloud endpoint's cert/public key for envelope encryption. Rationale: no dependency on a real public domain for the testbed; one trust anchor to reason about and document. (A public CA/Let's Encrypt is still the right call for the dashboard's own browser-facing HTTPS — that's a separate cert, separate audience, not part of device/envelope trust.)

**CN/SAN identity pinning is a non-negotiable requirement, not optional hardening — implement alongside the shared CA, not after.** Chain-of-trust validation alone (any cert signed by `ca.crt` is accepted) is insufficient once multiple gateways share a root with the cloud: it lets a compromised gateway's cert be used to impersonate the cloud (or another gateway) to a wearable, and the exposure scales with gateway count. Every TLS/envelope verification the wearable performs must check the peer cert's CN/SAN against the *specific* expected identity for that connection (e.g. this gateway's `device_id`, or the fixed cloud identity such as `cloud.medmusketeer.local`) — not just chain validity. This applies on both paths: wearable→gateway (fixed) and wearable→cloud (mobile).

**Implementation note — do not use `crt_bundle_attach`.** ESP-IDF's certificate-bundle API verifies against a bundle of many public root CAs (correct for talking to arbitrary standard TLS servers, e.g. OTA hosts) and is the wrong tool here — MedMusketeer's model is the opposite: one private CA, plus explicit per-connection identity pinning. Use `cacert_buf` (the single `ca.crt`) together with `common_name` + `skip_common_name = false` set to the *specific* expected peer identity per connection. **Open item to verify during implementation:** `common_name` is documented primarily as an SNI/hostname mechanism; confirm it does a literal CN string match against device-UUID-style CNs (not DNS-style hostnames) before relying on it — fall back to a custom `mbedtls_ssl_conf_verify` callback if it doesn't.

**Documented future hardening (not required for FYP-1 baseline, flag if skipped without reason):** split the shared root into a Device Intermediate CA (signs wearable/gateway certs) and a Server Intermediate CA (signs the cloud endpoint cert), both rolling up to one offline root. Contains blast radius if the device-issuance signing key is compromised. CN/SAN pinning alone is sufficient to close the known impersonation flaw for FYP-1; this is defense-in-depth on top of it, not a substitute.

## Cloud cert rotation — mobile-path key distribution (FUTURE SCOPE, not FYP-1)

**Decision: out of scope for FYP-1.** Rotating the cloud's leaf keypair the wearable encrypts envelopes to on the mobile path is deferred, not built. FYP-1 uses a long-dated cloud cert that comfortably outlasts the project timeline (same move already made elsewhere in this file for on-device key generation and the offline-signing ceremony) — rotation during the project is treated as a non-event by design, not solved in software.

## Gateway/wearable credential revocation (fixed path) — PENDING, not yet confirmed

**Status: candidate design only — user has not confirmed this yet. Do not treat as a locked decision the way the cloud-rotation section above is.**

**The gap this would address:** on the fixed path, wearable↔gateway is a genuine live mTLS session (ESP makes a direct connection to the gateway), so a rotated cert is a non-issue — chain/CN-SAN gets re-checked fresh on every handshake automatically. What live TLS does *not* give you for free is **revocation before natural expiry** — if a device's credential is revoked (compromised gateway, stolen device) before its cert expires, a plain chain+CN/SAN check still passes. Same gap applies to gateway↔cloud, but that leg is trivial to close (the broker's auth-hook just queries `device_credentials.status` in Postgres live, on every connect — no caching needed since it already sits in front of the registry).

**Why wearable↔gateway is the harder case:** per the IPS scope, the gateway must keep doing local ECG acquisition/classification/alerting even when it can't reach the cloud — so a live "phone home to check revocation" call on every wearable handshake would work against that graceful-degradation requirement.

**Candidate mechanism (not yet approved):**
- Gateway periodically pulls a small CA-signed revocation list from the cloud whenever it *can* reach it (CRL-style: revoked `device_id`s + a version/timestamp), verifies the signature against `ca.crt`, and caches it locally.
- Rollback guard: reject any pulled list whose version is older than what's already cached, so a replayed stale "nothing revoked" list can't un-revoke a device.
- On each wearable↔gateway handshake, check the peer's `device_id` against this local cache in addition to the normal chain/CN-SAN check — no network round-trip required, so it degrades gracefully if the cloud link is down. Worst case is working off a slightly stale list, which would need to be an explicitly flagged tradeoff, not a silent gap.

**Open, unresolved even at the design level:**
- Whether this is worth building for FYP-1 at all, given the low device count and short project timeline — may be a case for the same "document as future hardening, don't build now" call made elsewhere in this file, rather than actual implementation.
- If built: revocation-list wire format, pull cadence, and how the gateway surfaces "operating on a stale revocation list" to the cloud/admin dashboard once reconnected.
- Whether the broker-side check (gateway↔cloud) and this local cache (wearable↔gateway) should read from one shared source of revoked IDs or can stay independent.

## Device presence/liveness detection (online/offline status)

**Decision: no application-level "ping the cloud" polling loop on either path — mechanism differs per path because the wearable's relationship to the cloud connection differs per path.**

**Fixed path (wearable, fixed gateway → EMQX as direct MQTT clients):** use native MQTT liveness machinery, not custom heartbeat traffic.
- **Keepalive (PINGREQ/PINGRESP):** client/broker agree an interval at connect time; broker considers the connection dead if it hears nothing within 1.5× that interval. No app code required.
- **Last Will and Testament (LWT):** client registers a will message (`device_id`, `status: offline`) at connect time; broker auto-publishes it only on an *ungraceful* disconnect (crash/power loss/network failure). A clean disconnect should publish an explicit offline message instead, not rely on the will.
- These ride the already-authenticated mTLS session (same CN/SAN-pinned cert used for telemetry), so presence detection adds no new auth surface.
- EMQX's per-client connect/disconnect events (already noted above as a benchmarking checkpoint source) double as the presence signal — feed directly into the device registry's `status`/`last_seen`, no separate heartbeat topic needed.

**Mobile path (wearable → phone relay → cloud):** broker/session-level liveness is insufficient here and must not be treated as a proxy for wearable presence. The wearable is never itself an MQTT client to the cloud broker on this path (per the Option B envelope design above) — only the phone holds that session. Phone-session-alive does **not** imply wearable-present (BLE out of range, wearable powered off, etc., are all invisible to the phone's own transport-level liveness).
- **Mechanism:** a wearable-originated heartbeat envelope — same auth/encryption scheme as the telemetry envelope (Option B, CN/SAN-pinned), minimal payload — relayed by the phone like any other envelope.
- **Cloud inference:** wearable considered offline based on **envelope-arrival gap** (no heartbeat or telemetry envelope within the expected interval), tracked independently from the phone's own transport-session state.
- **Why keep these separate signals distinct:** "phone alive, wearable silent" and "both offline" are diagnostically different states and should surface differently in the device registry / admin dashboard, consistent with the existing asymmetric trust model (wearable = real security principal, phone = transport-only, no separate identity).

**Device registry implication (not yet schema'd):** `status`/`last_seen` semantics differ by path — fixed-path status derives from broker connection events; mobile-path status derives from envelope-arrival timing against the phone's session state as a secondary signal, not the primary one. This needs to land in the eventual Postgres schema design (see Open next steps).

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
