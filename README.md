# Arrhythmia Classification — MIT-BIH Replication

**Zain Ul Arifeen Sukhera · 2026–27**

Replication of the CLARITY-AI 2.0 LightGBM arrhythmia classifier on the MIT-BIH Arrhythmia Database. On a random 80/20 split (S1) the pipeline exceeds the paper on every metric (accuracy 0.996 vs 0.989, binary F1 0.993 vs 0.928). The inter-patient evaluation (S3) — training on DS1 patients, testing on a completely disjoint DS2 cohort — drops to accuracy 0.760, revealing an inter-patient generalisation gap driven by absolute-amplitude features that encode electrode placement rather than arrhythmia morphology.

## Results

| Setup | Accuracy | Macro-F1 | Binary F1 | LBBB F1 | RBBB F1 |
|---|---|---|---|---|---|
| **S1** Random split (replication) | 0.996 | 0.987 | 0.993 | 0.998 | 0.997 |
| **S2** Random model tested on DS2 only | 0.999 | 0.998 | 0.998 | 1.000 | 1.000 |
| **S3** DS1 train → DS2 test (honest inter-patient) | 0.760 | 0.381 | 0.401 | 0.025 | 0.004 |
| Paper (random split) | 0.989 | ~0.937 | 0.928 | 0.915 | 0.921 |

**S3** is the clinically honest evaluation: the model trains on DS1 (one cohort of 22 patients) and is tested on DS2 (a completely different cohort of 22 patients), with no patient appearing in both — the standard inter-patient benchmark for MIT-BIH. S2 looks artificially perfect because the random-split training set already contained 80 % of DS2 beats.

**Caveats:** All results are from a single random seed. DS1 contains very few LBBB and RBBB patients, so per-class F1 for those two arrhythmias on S3 is unreliable regardless of the model's true capability. S4 (feature-ablation on DS1→DS2) is in [`evaluation_output.txt`](evaluation_output.txt).

## Dataset

This project uses the **MIT-BIH Arrhythmia Database** from PhysioNet — 48 half-hour two-channel ECG recordings from 47 subjects, annotated by cardiologists at 360 Hz.

### Downloading

Make sure `download_mitdb.sh` is in the project directory, then run:

```bash
./download_mitdb.sh
```

Downloads `.dat` (signal), `.hea` (header), and `.atr` (annotations) for all 48 records (144 files total). Requires `wget`:

```bash
sudo apt update && sudo apt install wget
```

## Running the pipeline

```bash
python source/preprocess.py   # segment beats → datasets/processed/mit_bih_segments.npz
python source/features.py     # extract 40 features → datasets/processed/mit_bih_features.npz
python source/split.py        # build random + DS1/DS2 splits
python source/train.py        # train LightGBM (Optuna, paper params as seed)
python source/evaluate.py     # reproduce the S1–S4 results above
```

## Project structure

```
.
├── datasets/
│   ├── mitdb/          # raw .dat/.hea/.atr files
│   └── processed/      # mit_bih_segments.npz, mit_bih_features.npz, split_*.npz
├── models/             # trained .pkl models + Optuna JSON logs
├── results/            # confusion-matrix PNGs (S1–S3)
├── source/
│   ├── preprocess.py
│   ├── features.py
│   ├── split.py
│   ├── train.py
│   ├── evaluate.py
│   └── binaryLightGBM/ # N vs anomaly binary classifier
├── evaluation_output.txt
├── evaluation_summary.md
├── progress.md
└── requirements.txt
```

## Requirements

```bash
pip install -r requirements.txt
```

Key dependencies: `wfdb`, `lightgbm`, `imbalanced-learn`, `optuna`, `scipy`, `scikit-learn`.

## Source

MIT-BIH Arrhythmia Database — PhysioNet: https://www.physionet.org/content/mitdb/1.0.0/
