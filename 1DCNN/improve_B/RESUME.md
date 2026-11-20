# farag-improve RESUME (agent farag-improve, tasks B1-B5)

Owned dir: $ROOT/1DCNN/improve_B  ($ROOT=/home/i230510/Desktop/arrhythmiaClassifier). Python: $ROOT/.venv/bin/python (run from improve_B dir).

## Files
- common.py  : paths, weights w_c=(1/freq)^p normalised (p=0 none, p=1 = sklearn balanced), metrics, logit-offset decision rule argmax(logP+[0,bS,bV]), grid search choose_points (maxF1, maxAcc(=max OvR acc), SeConstr: Se_S>=85.3 & Se_V>=96.3 on DS1 calib, max acc).
- prep.py    : DONE. cache/data.npz (DS1 w/ record ids g1, DS2), cache/folds.json (5 patient-wise folds, greedy S/V-balanced), cache/kernel_full.npy (== official kernel_mf), cache/kernel_fold{k}.npy (templates from training-fold records only). Verified arrays equal official .cache npz.
- train_jobs.py : one job per (scheme rand|gkf, p, seed, fold, es f1|loss) -> runs/<tag>.npz + .keras; skips existing (resumable). 7 workers x 1 thread.
   rand = 15% random-beat hold-out (rng 42, same as run_official_p1.py); gkf = GroupKFold by record, OOF probs saved.
   ES on val macro-F1 (patience 20), ReduceLROnPlateau on val_loss — same as official train_one. `--es loss` = val_loss ES (paper-literal, for B1).
- analyze.py : collects runs/, official P1 ensemble reference (from HasanML/paper3/results/cache/P1_fullDS1_seed*.pkl), calibrates offsets on DS1 (rand: hold-out; gkf: OOF), evaluates DS2; writes results/results.md, results/summary.json, results/tradeoff_SeS.png.
- pipeline.sh : stage-1 chain (nohup'd): rand sweep p{0,.25,.5,.75,1} seeds0-2 -> gkf sweep same p seeds0-2 (75 jobs) -> rand p0 seeds0-9 (B1) -> rand p0 es=loss seeds0-9 -> analyze. Writes "STAGE1_DONE" to logs/pipeline.log at the end.

## Running (started 2026-11-18 ~13:25 box time)
- pipeline.sh PID 392377 (log logs/pipeline.log; per-stage logs logs/rand_sweep.log, gkf_sweep.log, b1_rand_p0.log, b1_rand_p0_esloss.log, analyze_stage1.log).
- Finished when `grep STAGE1_DONE logs/pipeline.log` hits / `pgrep -f pipeline.sh` empty. If killed: just rerun `nohup ./pipeline.sh > logs/pipeline.log 2>&1 < /dev/null &` (done jobs skipped).
- Speed: ~1.5 s/epoch/job under load, ~2-3 min per job, 7 parallel.

## Remaining steps
1. After STAGE1_DONE: read results/results.md. Pick finalists (best gkf p by DS1 OOF macro-F1 — NOT DS2) e.g. 2 values of p.
2. Run 10 seeds for finalists: `nohup ../../.venv/bin/python train_jobs.py --scheme gkf --p <p1> <p2> --seeds 0-9 --workers 7 > logs/gkf_final.log 2>&1 < /dev/null &` (also rand for same p if useful), then `python analyze.py`.
3. Select recommended config + operating point using DS1-only info (OOF macro-F1 / SeConstr feasibility); report DS2.
4. Write $ROOT/reports/farag-improve.md (what/how/tables/files/open issues), final summary.

## Decisions / notes
- OvR accuracy = 1-2e/3 is monotone in plain accuracy, so "max OvR acc" == "max plain acc".
- ES uses val macro-F1 for all p (consistent); B1 additionally has paper-literal val_loss ES variant.
- DS2 never used for any choice; trade-off plot shows DS2 curves descriptively only.

## Milestone 1 (rand sweep done, 15 jobs)
- rand_p1 seeds0-2 reproduces official Exp5 exactly (Acc 91.57, Se_S 83.03) -> pipeline faithful.
- Interim results in results/results.md. Finding: random-beat hold-out is leaky (DS1 hold-out Se_S ~97% at argmax vs 85% on DS2), so offsets tuned on it push S down and DS2 Se_S collapses (maxF1: Se_S 61.9, acc 95.2, F1 80.4 for official ensemble). Lower p raises Se_N but kills Se_S (p0: Se_S 25, acc 94.1).
- gkf sweep running (logs/gkf_sweep.log), then B1 10-seed runs, then analyze.

## Milestone 2 (coordinator update applied, ~14:15)
- Targets corrected per reports/farag-gap.md: Model 6 (no CW) ACC 98.18 is PLAIN acc; Se N/S/V 99.10/81.60/95.00, PPV 99.00/82.68/95.63; macro PREC 92.44 Se 91.90 F1 92.17. Model 2 (CW, = our official config) acc 95.58, F1 84.79, PPV_S 51.49. Compare with SEED MEANS, not ensembles. B5 constraint now Se_S>=81.6, Se_V>=95.0 (common.py SE_*_TARGET).
- analyze.py now reports: DS1-only selection table; DS2 seed-mean (each seed calibrated on its own DS1 calib data); DS2 single-model mean (each gkf fold model separately = paper-comparable); DS2 ensemble. SeConstr falls back to max macro-Se when infeasible on DS1 (it is infeasible on gkf OOF: OOF Se_V only ~71-80%).
- Interim finding: gkf (patient-wise ES) models stop early (median best epoch ~10) and are far less N->S/V-happy: gkf_p0.5 fold-ensemble argmax acc 95.7, F1 81.6, PPV_S 51.7 vs rand_p1 92.0/77.3/31.5.

## Milestone 3 (2026-11-19 ~10:50) stage 1 DONE (107 runs). Stage 2 running
- Finalists picked on DS1 OOF macro-F1@maxF1 (gkf): p0.75 (70.30) > p0.5 = p1 (68.71) > p0 (66.55) > p0.25. Running 10 seeds gkf for p0.75, p0.5, p0 (p0 = B1 under patient-wise ES).
- `pipeline2.sh` (setsid nohup), log logs/gkf_final.log, done when logs/pipeline2.log contains STAGE2_DONE (it re-runs analyze.py). Rerun the same script if killed (resumable).
- B1: rand_p0 (ES on val macro-F1) 10 seeds; rand_p0_esloss (ES on val_loss, paper-literal) 10 seeds -> ran all 500 epochs (best epoch median 198).
- Remaining after stage 2: write reports/farag-improve.md from results/results.md.
- Added scheme `refit` (train_jobs.py run_refit): full DS1, epochs = median best epoch of the seed's 5 gkf folds, constant LR, no val; calibrated with that seed's gkf OOF. pipeline3.sh waits for STAGE2_DONE then runs refit p0.75/0.5/0 seeds0-9 + analyze; done when logs/pipeline3.log has STAGE3_DONE.

## COMPLETE (2026-11-19 ~11:20): all stages done; final report at reports/farag-improve.md
