# farag-gap — RESUME notes (updated 2026-11-18 ~13:35)

ROOT=/home/i230510/Desktop/arrhythmiaClassifier ; G=$ROOT/1DCNN/gap_analysis

## Paper source
Full text fetched from Europe PMC (PMC9919183): `curl -s "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC9919183/fullTextXML"` (MDPI returns 403).
KEY FACTS (verified from the text and Tables 2–6):
- Model 6 (3-class, Conv1D trainable, NK=32, derivative, class weight NOT SET), DS2 test: ACC 98.18;
  N PREC 99.00 / SEN 99.10 / F1 99.05; SVEB 82.68 / 81.60 / 82.14; VEB 95.63 / 95.00 / 95.31; avg PREC 92.44, SEN 91.90, F1 92.17.
- SVEB Se 85.30 and VEB Se 96.34 come from **Model 1** (NK=64, default init, derivative, class weight SET; acc 96.48, avg F1 86.81). The abstract mixes Model 6 averages with Model 1 sensitivities.
- 89.7 and 97.8 appear nowhere in the paper → wrong targets.
- Averages = unweighted macro over N/S/V; ACC = plain accuracy over 49,317 N/S/V beats (F/Q dropped). NOT OvR. Verified numerically.
- Table 2 DS1 45866/944/3788/415/8; DS2 44259/1837/3221/388/7 → our counts match exactly.
- Model 2 (NK=32, trainable MF, DREV, CW SET) = closest config to our balanced-CW P1: acc 95.58, avg F1 84.79.
- Table 3 architecture differs from ours: 11 kernels for 3-class, BN + **tanh**, 'same'-padded output (64,11) on 64-sample input, RR branch Dense 32/16/8 applied on (4,1) then Flatten(32); 1267 params.
- Paper model 7 avg PREC 78.77 is a typo (per-class mean = 87.77).

## Done
- A2/A3/A4: `$G/gap_metrics.py` → `$G/gap_metrics.json`, `$G/gap_metrics.log`, `$G/ensemble_probs.npz` (verified identical to cached proba_test / proba_val).
  Ensemble DS2 argmax == tau 0.50 (0 differing preds): acc 91.61, OvR 94.41, Se N/S/V 91.44/87.10/96.52, PPV 99.79/32.09/82.40, macro F1 77.08, macro PPV 71.43.
  Seed-mean (paper protocol): acc 91.03, macro F1 75.57, Se_S 85.29, Se_V 95.14.
- A5: CLI run with a private config (`$G/rr_baseline_config.yaml`, outputs to `$G/rr_baseline/`; paper3/results verified unchanged by md5).
  xgboost was missing in both venvs → installed with `--no-deps --target $G/pylib` (use PYTHONPATH=$G/pylib). CLI: acc 91.64, Se 96.5/12.7/69.3.
  Variants `$G/rr_baseline_variants.py` → `$G/rr_baseline/rr_baseline_variants.{json,log}`: full DS1 100x d3: acc 93.73, Se 97.4/30.8/79.3.

## Running
- Nothing is running. All tasks (A1–A5) are done. The final report is reports/farag-gap.md; the utils.py footer and paper_table4.md have been updated.
