# KAN-Former (paper4) — first GPU training, evaluation, ablations and teacher export

Agent: kanformer. Dates: 2026-11-18/19. Compute: GPU box, RTX 4000 Ada, `$ROOT/.venv-torch` (torch 2.15.0+cu132, Python 3.14).
`$ROOT=/home/i230510/Desktop/arrhythmiaClassifier`, `$P4=$ROOT/HasanML/paper4`, `PY=$ROOT/.venv-torch/bin/python`.

## Summary

- **The KAN-Former trains and runs now, but on DS2 it is far below every target, mainly on S.**
  - Default config, 3 seeds, 5-class DS2: plain acc 84.9±2.4, OvR acc 94.0±1.0, macro F1 34.1±4.4, Se_S 13.0±10.3, Se_V 90.3±4.3.
  - Best configuration (Fix B; the 3-seed ensemble is the exported teacher), 3-class N/S/V on DS2: acc 94.6, Se N/S/V 97.7/25.0/90.9, PPV_S 34.6, macro F1 70.0.
  - Corrected Farag Model 6 targets: acc 98.18, Se N/S/V 99.1/81.6/95.0, PPV_S 82.7, macro F1 92.17.
  - Our 1D-CNN: acc 91.61, Se_S 87.10, Se_V 96.52.
  - **The KAN-Former is not a better teacher than the 1D-CNN on S. Do not use it for distillation without further work.**
- **The code had serious bugs.** As written it would have crashed (KAN forward) or trained on garbage (sampler and loss weighting). Ten bugs are fixed; all 39 unit tests pass.
- **Root cause of the S failure: the model memorises the few S beats it trains on and does not generalise across patients.**
  - The Farag "<200" DS1-train split has only 149 S beats.
  - Training loss reaches ~1e-3 within 15 epochs.
  - The model ignores the RR timing cue. A 6-feature RR-only LightGBM gets higher Se_S than the network on both DS1-val and DS2.
  - DS1-val-justified fixes helped a little (DS2 Se_S 13 → 27, 3-class macro F1 57 → 69) but nowhere near the targets.
- **Teacher export is done:**
  - Soft labels at T ∈ {1,2,4} for DS1 and DS2 as `.npz`, with join keys and 5-class plus 3-class probabilities.
  - ONNX and TorchScript exports, checked against PyTorch (max diff 2e-6 / 0).
- **No knowledge distillation and no student training was done.**

## 1. Tests and sanity run

`cd $P4 && $PY -m pytest tests -q` → **39 passed**. Before my changes, 11 of 37 tests failed. I added 2 regression tests.

Bugs fixed (all in `HasanML/paper4/src`):

| # | file | bug | effect |
|---|---|---|---|
| 1 | kan.py | `bases[..., -1] \| at_end` (bitwise-or on a float tensor) | **every KAN forward crashed** |
| 2 | kan.py | `grid` buffer was a non-contiguous `expand` view | checkpoints could not be loaded on CPU |
| 3 | train.py | balanced sampler capped only S and V | with DS1-train counts (F=13, Q=2), **Q was ~86% and F ~13% of every batch** |
| 4 | train.py | focal α computed from raw counts *on top of* the balanced sampler (double correction: N 0.029 vs Q 3.2) | the model predicted F for almost every beat |
| 5 | train.py | every DataLoader worker shared one copy of the augmentation RNG | identical augmentation streams |
| 6 | train.py | EMA decay fixed at 0.999 with only ~100 steps/epoch | early validation used ~80%-random-init weights; added EMA warm-up |
| 7 | train.py | deprecated `torch.cuda.amp.GradScaler`; no checkpoint/resume | per-epoch checkpoint + resume added; loss now computed in fp32 |
| 8 | model.py | A4 (`n_features=0`) crashed; no config defaults | RR token/shortcut now dropped cleanly; `.get` defaults |
| 9 | losses.py | focal "mean" was not α-weighted; test API mismatch | now equals weighted CE at γ=0; `build_loss("focal", class_counts=…)` works |
| 10 | run_experiment.py | `--distill` exported ONNX from raw (non-EMA) weights; `--results-dir` shared one global cache | EMA export; cache, figures and teacher now live inside the run dir |

Other fixes:
- `run_paper4.sh` passed `--seeds 3`, which trains *seed value 3 only*, not 3 seeds. Fixed.
- `rr.py`: LightGBM used `n_jobs=-1` (all cores); now 6.
- One test claimed the RR features are fully causal. They are not: post-RR uses a one-beat look-ahead, as documented. The test now checks that.

Sanity run:
- RR-only LightGBM (DS1-train → DS1-val): acc 87.1, Se_S 47.5, Se_V 52.3 (`results/sanity_rr`).
- GPU sanity: `results/sanity_gpu`. 820,740 params; **~12–20 s/epoch** (data-loader bound; ~1.6 GB GPU). A full seed with early stopping takes 8–18 min.

Beat counts:

| split | N | S | V | F | Q |
|---|---|---|---|---|---|
| DS1-train (Farag) | 23,932 | 149 | 1,234 | 13 | 2 |
| DS1-val (Farag) | 21,934 | 795 | 2,554 | 402 | 6 |
| DS2 | 44,259 | 1,837 | 3,221 | 388 | 7 |

## 2. Main results on DS2 (3 seeds each; temperature fitted on DS1-val)

Runs:
- **DEFAULT** = `configs/default.yaml` as shipped (focal loss, EMA, cosine LR, AMP; Farag DS1 split; early stopping on DS1-val 5-class macro F1).
- **Fix A** = a balanced patient-wise DS1 split (14 train / 8 val records, chosen from DS1 label counts only) plus selection on DS1-val N/S/V macro F1.
- **Fix B** = Fix A plus stronger regularisation and milder oversampling:
  - dropout 0.3, stochastic depth 0.2, head dropout 0.4;
  - weight decay 0.1, lr 1e-4;
  - S/V caps 0.15.

### 5-class AAMI (Se/PPV in %)

| run | model | acc | OvR acc | macro F1 | N Se/PPV | S Se/PPV | V Se/PPV | F Se/PPV | Q Se |
|---|---|---|---|---|---|---|---|---|---|
| DEFAULT | mean±std of seeds | 84.9±2.4 | 94.0±1.0 | 34.1±4.4 | 88.3/95.7 | 13.0±10.3 / 6.0 | 90.3±4.3 / 58.9 | 0.8 / 1.0 | 0 |
| DEFAULT | ensemble(3) | 87.4 | 95.0 | 37.8 | 90.7/96.2 | 14.4/6.6 | 94.4/76.5 | 1.0/17.4 | 0 |
| Fix A | mean±std | 87.9±3.6 | 95.2±1.4 | 41.0±0.7 | 90.9/97.6 | 23.2±0.8 / 18.6 | 90.3±2.7 / 69.9 | 29.9/8.9 | 0 |
| Fix A | ensemble(3) | 91.0 | 96.4 | 42.1 | 94.5/97.5 | 21.1/19.9 | 91.4/70.8 | 19.8/11.0 | 0 |
| Fix B | mean±std | 90.0±2.2 | 96.0±0.9 | 42.1±1.7 | 93.4/97.4 | 27.4±5.2 / 32.4 | 88.7±2.0 / 74.1 | 9.9/2.0 | 4.8 |
| **Fix B** | **ensemble(3) = teacher** | **91.8** | **96.7** | **42.2** | **95.5/97.3** | **25.0/34.6** | **89.4/76.1** | **6.2/2.1** | 0 |
| target (paper_targets.md) | | ≥95 | ≥97 | ≥80 | ≥98/≥97 | ≥70/≥60 | ≥90/≥85 | ≥50/≥40 | ≥60 |

### 3-class N/S/V

Comparable with Farag 2023 Model 6 and our 1D-CNN.
- **Protocol:** beats whose true label is F or Q are dropped. The prediction is the argmax over N/S/V probabilities (F/Q columns dropped, then renormalised). OvR acc uses K=3.
- A "strict" variant (5-way argmax, so an N/S/V beat predicted as F/Q counts as an error) is in each `eval_table_DEFAULT.md`. It is 0.3–4 pp lower in acc.

| model | plain acc | OvR acc | macro F1 | Se N | Se S | PPV S | Se V |
|---|---|---|---|---|---|---|---|
| DEFAULT mean±std | 86.0±2.8 | 90.7±1.9 | 56.7±7.2 | 88.7 | 13.0 | 6.0 | 90.5 |
| DEFAULT ensemble | 88.1 | 92.1 | 62.6 | 90.7 | 14.4 | 6.7 | 94.6 |
| Fix A mean±std | 91.5±1.6 | 94.3±1.1 | 65.2±1.9 | 94.2 | 23.2 | 18.6 | 93.4 |
| Fix A ensemble | 92.6 | 95.1 | 65.9 | 95.5 | 21.1 | 19.9 | 93.9 |
| Fix B mean±std | 93.8±1.3 | 95.9±0.8 | 69.1±3.4 | 96.8 | 27.4 | 32.4 | 90.6 |
| **Fix B ensemble (teacher)** | **94.6** | **96.4** | **70.0** | **97.7** | **25.0** | **34.6** | **90.9** |
| Farag 2023 Model 6 (corrected target) | 98.18 | — | 92.17 | 99.10 | 81.60 | 82.68 | 95.00 |
| our 1D-CNN (10-seed ens., full DS1) | 91.61 | — | 77.08 | 91.4 | 87.10 | 32.1 | 96.52 |

The KAN-Former has better N sensitivity and plain accuracy than the 1D-CNN. But it misses about 75% of S beats, against the 1D-CNN's 13%. It also falls 1.4–6 pp short on V.

### Calibration

ECE values are on DS1-val, before → after temperature scaling.
- DEFAULT: T = 1.56 / 1.57 / 0.73 for seeds 0/1/2. ECE 0.12→0.08, 0.07→0.05 and 0.46→0.39 (seed 2 is a bad model, best epoch 1).
- Fix B: T = 1.46 / 1.57 / 1.39 / ensemble 1.26.
  - Ensemble ECE: DS1-val 0.018→0.038; DS2 0.016→0.018.
  - Temperature scaling does not reduce ECE for the already well-calibrated Fix B ensemble. It does help the individual DEFAULT seeds.
  - The `paper_targets.md` "ECE after < 0.05" target is met only by Fix A/B, not by DEFAULT.

### Model selection (DS1-val only)

- Fix A and Fix B share the same DS1-val records, so they can be compared directly.
  - N/S/V macro F1 of the ensembles: Fix A 0.633, **Fix B 0.687**.
  - 5-class macro F1 of the ensembles: Fix A 0.382, **Fix B 0.416**.
- Within Fix B, the ensemble beats every single seed on DS1-val. It is therefore the teacher.
- DEFAULT's DS1-val (Farag records ≥200) is a different patient set, so its numbers cannot be compared directly.
- I prefer the fixes over DEFAULT for DS1 reasons, not DS2:
  - DEFAULT's DS1-train has 149 S beats.
  - DEFAULT's early stopping was unstable: seed 2's best epoch was 1, and its DS1-val ECE was 0.46.
- Caveat: the DS2 ranking agrees with the DS1-val ranking, but I did not use DS2 for any choice.

## 3. Diagnosis of the S failure (task 5)

1. **S-starved and highly memorisable training data.**
   - The Farag DS1-train has 149 S beats. 96 of them are from record 118, and 6 of the 12 records have 4 or fewer S beats.
   - With 30% S oversampling, each S beat is seen ~50 times per epoch. Training loss goes to ~3e-4 while DS1-val Se_S stays at 20–40%.
2. **The network does not use the RR timing cue well enough.**
   - A 3-class LightGBM on only the 6 RR-ratio features (`results/diag_rr.txt`) gets these DS2 Se_S values: 15.5 when trained on Farag DS1-train, 53.8 on the balanced split, 41.6 on all of DS1.
   - So with the same balanced training records, an RR-only tree model detects S twice as well as the KAN-Former (25–27%).
   - Ablation A4 (no RR) drops Se_S further, to 7%. The RR input matters, but the morphology/context branches dominate and overfit.
3. **The S distribution shifts between DS1 and DS2.**
   - DS1 S beats have median post-RR/local 1.06 and pre/post 0.82, i.e. an isolated premature beat followed by a pause.
   - DS2 S beats have post/local 0.81 and pre/post 0.96, i.e. mostly *runs* of S (SVT, e.g. record 232).
   - A morphology-memorising model trained on isolated DS1 PACs does not transfer.
4. **Fixes tried** (2, as allowed; both justified on DS1 only):
   - Fix A, balanced split: helped stability and S/V PPV (DS2 macro F1 34→41). Se_S +10 pp.
   - Fix B, regularisation: a further +4 pp Se_S, PPV_S 19→32, 3-class acc 94.6. Still far from 81.6.
5. **Not tried (suggested next steps):**
   - Train on all of DS1 with the epoch count chosen by patient-grouped CV, as our 1D-CNN does. It sees all 944 DS1 S beats.
   - Feed patient-normalised RR (e.g. relative to a longer record median) or a "run of S" feature.
   - Strongly reduce morphology capacity (e.g. drop the context branch: A3 was *better* than DEFAULT).
   - Stack the KAN-Former with the RR LightGBM.

## 4. Ablations (DEFAULT config base, Farag split)

All ablations use full schedules with early stopping.
- A1–A4: 3 seeds each.
- A5, A6a, A6b: cut to **seed 0 only** for time. Compare them with DEFAULT seed 0: DS2 acc 87.5, macro F1 39.9, Se_S 25.6, Se_V 93.0.

Values are mean±std over seeds. DS2 is 5-class unless marked 3c.

| run | seeds | DS1-val 5c macroF1 | DS2 acc | OvR | macroF1 | Se_N | Se_S | PPV_S | Se_V | PPV_V | Se_F | 3c acc | 3c macroF1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DEFAULT (full model) | 3 | 36.3±1.5 | 84.9±2.4 | 94.0±1.0 | 34.1±4.4 | 88.3±2.7 | 13.0±10.3 | 6.0±4.4 | 90.3±4.3 | 58.9±19.2 | 0.8±0.6 | 86.0±2.8 | 56.7±7.2 |
| A1 MLP head | 3 | 39.8±0.1 | 85.1±1.3 | 94.0±0.5 | 37.3±0.4 | 88.3±1.3 | 18.6±0.6 | 7.6±0.8 | 88.1±2.0 | 78.1±7.2 | 1.6±1.8 | 87.5±0.5 | 61.8±1.5 |
| A2 no Transformer | 3 | 39.1±0.2 | 82.0±4.9 | 92.8±2.0 | 33.7±2.5 | 85.5±5.8 | 14.7±4.1 | 5.8±1.4 | 82.4±5.8 | 60.5±12.2 | 1.5±0.9 | 84.6±2.8 | 55.8±4.2 |
| A3 no context | 3 | 38.4±1.9 | 85.9±1.2 | 94.4±0.5 | 38.9±0.9 | 88.8±1.4 | 21.5±3.0 | 9.7±1.7 | 93.4±0.8 | 83.5±3.9 | 1.7±1.0 | 88.4±1.6 | 64.8±2.0 |
| A4 no RR | 3 | 35.8±0.9 | 80.2±0.6 | 92.1±0.3 | 30.5±3.1 | 83.8±1.2 | 7.3±4.6 | 2.9±0.8 | 81.6±5.4 | 47.9±17.1 | 0.4±0.3 | 81.2±0.8 | 51.1±5.4 |
| A5 single kernel k=7 | 1 (seed 0) | 38.7 | 89.7 | 95.9 | 39.2 | 93.6 | 18.2 | 10.4 | 87.8 | 86.3 | 0.5 | 90.5 | 65.2 |
| A6a KAN grid 3 | 1 (seed 0) | 37.9 | 67.3 | 86.9 | 29.8 | 68.2 | 18.2 | 3.1 | 91.0 | 49.1 | 0.0 | 68.6 | 47.5 |
| A6b KAN grid 8 | 1 (seed 0) | 39.8 | 85.8 | 94.3 | 36.9 | 89.5 | 15.4 | 6.3 | 84.6 | 79.0 | 1.5 | 87.3 | 61.4 |

Observations (noisy; seed std is often larger than the differences):
- **RR features matter most.** A4 is the worst on everything.
- **The Transformer helps V.** A2 has Se_V 82 vs 90.
- **The KAN head gives no benefit.** The MLP head (A1) is equal or better (macro F1 37.3 vs 34.1, lower variance).
- **A5 single kernel (seed 0) is on par with DEFAULT seed 0.** It has higher acc but lower Se_S. Multi-scale gives no clear gain.
- **KAN grid size:** grid 3 (A6a) collapsed on N (Se_N 68%, a single seed). Grid 8 (A6b) is on par with the default grid 5. With 1 seed each, neither is conclusive.
- **The 3 s context window hurts** in this data regime. A3 is better than DEFAULT on every DS2 metric and has higher DS1-val F1. This is the opposite of the expected −5 to −8 pp Se_S in `paper_targets.md`.
- Because DEFAULT is so far below target, these ablations say little about the architecture's value in a working regime.

## 5. Teacher export (task 4) — `$P4/results/teacher/`

Command: `$PY -m src.export_teacher --results-dir results/fix_reg --out-dir results/teacher`

- **Teacher:** Fix B ensemble(3) (logit average of 3 seeds, EMA weights), selected on DS1-val. T_calibrated = 1.260.
- **Soft-label files:** `soft_labels_ds1_T{1,2,4}.npz` (51,021 beats) and `soft_labels_ds2_T{1,2,4}.npz` (49,712 beats).
- **Keys:**
  - `record_id`, `r_sample` (R-peak index in the original 360 Hz `.atr`), `symbol`, `aami_label`/`aami_name`;
  - `probs5` = softmax(logits/T) over [N,S,V,F,Q];
  - `probs3` = the N/S/V columns of probs5, renormalised (F/Q columns dropped);
  - `is_FQ` (true label is F or Q) and `label3` (−1 for F/Q);
  - `split_part`, `logits`, `T`, `T_calibrated`, `teacher`.
- **Checks:**
  - (record_id, r_sample) is unique; each probs5 row sums to 1.
  - The DS2 3-class metrics recomputed from the file match the table above (acc 94.58, Se_S 25.0).
  - The DS1 3-class acc is 97.4 (Se_S 87.0). This is mostly training data, so it is inflated.
- **Models:** `kanformer_teacher.onnx` (opset 17, dynamic batch) and `kanformer_teacher.ts` (traced). Max abs diff vs PyTorch on 32 DS2 beats: ONNX 2.0e-6, TorchScript 0.
- `README.md` describes all files, keys, the 3-class mapping, join instructions and the input preprocessing.

## Reproduce (from `$P4`)

```bash
PY=../../.venv-torch/bin/python
$PY -m pytest tests -q
$PY -m src.run_experiment --rr-baseline --results-dir results/sanity_rr
# main runs (resume automatically from per-epoch checkpoints if interrupted)
$PY -m src.run_experiment --seeds 0 1 2 --device cuda --mixed-precision --results-dir results/gpu_run
$PY -m src.run_experiment --seeds 0 1 2 --device cuda --mixed-precision --results-dir results/fix_balsplit \
    --set data.ds1_split=balanced "training.select_classes=[0,1,2]"
bash results/launch_fixreg.sh          # Fix B (same as above + regularisation overrides; see script)
$PY -m src.evaluate_full --results-dir results/gpu_run      # likewise fix_balsplit, fix_reg
# ablations
$PY -m src.run_experiment --seeds 0 1 2 --ablations --ablation-seeds 0 1 2 \
    --ablation-names A1_mlp_head A2_no_transformer A3_no_context A4_no_rr \
    --device cuda --mixed-precision --results-dir results/ablations_a
bash results/eval_ablations.sh; $PY results/abl_table.py
# teacher export
$PY -m src.export_teacher --results-dir results/fix_reg --out-dir results/teacher
```

Wrapper: `bash 1DCNN/run_paper4.sh {tests|sanity|gpu|fix|eval_full RUN|ablations|export RUN}`.

## Files created or modified on the box

- **Modified:**
  - `HasanML/paper4/src/{kan,losses,model,train,run_experiment,rr,data}.py`
  - `HasanML/paper4/tests/test_pipeline.py`
  - `HasanML/paper4/configs/default.yaml` (added `f_cap`, `q_cap`)
  - `1DCNN/run_paper4.sh`
- **Created:**
  - `HasanML/paper4/src/evaluate_full.py`, `HasanML/paper4/src/export_teacher.py`
  - `HasanML/paper4/results/{sanity_rr,sanity_gpu,gpu_run,fix_balsplit,fix_reg,ablations_a,ablations_b,teacher}/`
  - `results/diag_rr.{py,txt}`, `results/abl_table.py`, `results/eval_ablations.sh`, `results/launch_fixreg.sh`, `results/launch_stage2.sh`, `results/launch_ablb.sh`
  - `reports/kanformer.md`, `reports/kanformer.RESUME.md`
- **New config keys:**
  - `data.ds1_split` (`farag`|`balanced`)
  - `training.select_classes`, `training.f_cap`, `training.q_cap`, `training.alpha_mode`, `training.early_stop_min_delta`
- **New CLI flags:** `--max-epochs`, `--set k.sub=v`, `--ablation-seeds`, `--ablation-names`.
- **Packages:** `onnxscript` and `onnxruntime` installed into `.venv-torch`. Torch is unchanged.

## Open issues

- The KAN-Former is far below every target. **Its value as a distillation teacher for S is doubtful:** on S it is worse than the 1D-CNN student it was meant to teach. See section 3 for next steps.
- Ablations A5, A6a and A6b have 1 seed only. All ablations use the DEFAULT base, not Fix B.
- Resume bug fixed late: `torch.load(map_location=cuda)` moved the saved RNG state to CUDA, so resuming crashed with "RNG state must be a torch.ByteTensor". `train.py` now calls `.cpu()` on it. A5 seed 0 was resumed at epoch 15 after this fix.
- Flash-attention backward is non-deterministic (warning). Seeds are not bit-reproducible.
- Another agent runs paper4's `src.run_experiment --rr-baseline` with its own config. The CLI stays backward-compatible, but `--results-dir` now also relocates the cache.
- `colab_paper4.ipynb` and `README.md` in paper4 were not updated for the new flags.
