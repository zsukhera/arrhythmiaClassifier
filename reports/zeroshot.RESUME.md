# zeroshot agent — RESUME notes (updated 2026-11-18 ~14:15)

ROOT=/home/i230510/Desktop/arrhythmiaClassifier ; PY=$ROOT/.venv/bin/python (run with `-I` from $ROOT/zeroshot)

## Done (outputs in $ROOT/zeroshot/out/)
- Downloads + unzip: datasets/external/ptbxl/ptb-xl-1.0.3/ (records500 only extracted), datasets/external/chapman/<dir>/WFDBRecords.
- sanity_models.py: frozen models reproduce DS2 numbers exactly (CNN acc .9161; LGBM acc .7635/macroF1 .3878).
- validate_detector.py -> detector_validation.{csv,txt}: XQRS+50ms refine chosen on DS1 (Se 96.6 / PPV 98.7; DS2 97.9 / 99.7).
- mitbih_strips.py -> mitbih_strips.pkl, mitbih_beats.pkl (7920 strips).
- choose_rules.py -> rules.json (LGBM L,R: >=4 beats; V,S: >=1 beat; CNN V,S >=1 beat; CNN RR fallback = causal), mitbih_strip_report.md.
- prepare_external.py ptbxl|chapman -> <ds>_manifest.csv, <ds>_leadII_360.npz (PTB-XL cohort 8226, Chapman 8541 after adding
  713427006/733534002 complete RBBB/LBBB and excluding 10370003 pacing).

## Running (nohup)
- run_external.py ptbxl (pid in run_ptbxl.pid, log run_ptbxl.log) -> out/ptbxl_report.md
- queued after it: run_external.py chapman (log run_chapman.log) -> out/chapman_report.md
  Finished when out/chapman_report.md exists. If chain died: `nohup $PY -I run_external.py chapman > run_chapman.log 2>&1 &`

## Remaining
1. Write $ROOT/reports/zeroshot.md from out/mitbih_strip_report.md, out/ptbxl_report.md, out/chapman_report.md, detector_validation.txt.
ALL STEPS DONE — final report at reports/zeroshot.md
