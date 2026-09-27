# Arrhythmia Classification

**Zain Ul Arifeen Sukhera**
**2026–27**

## Dataset

This project uses the **MIT-BIH Arrhythmia Database** from PhysioNet.

The database contains 48 half-hour, two-channel ECG recordings obtained from 47 subjects. The downloaded dataset includes the ECG signals, headers, and expert annotations.

## Downloading the Dataset

To download the MIT-BIH dataset, make sure `download_mitdb.sh` is in the project directory and run:

```bash
./download_mitdb.sh
```

The script downloads the following file types:

* `.dat` — ECG signal data
* `.hea` — Recording header and metadata
* `.atr` — Expert ECG annotations

The dataset contains **48 recordings**, resulting in **144 files** when downloading all three file types.

## Requirements

The download script requires `wget`.

On Ubuntu/WSL, install it with:

```bash
sudo apt update
sudo apt install wget
```

## Project Structure

After downloading the dataset, the project may look like:

```text
.
├── README.md
├── download_mitdb.sh
└── datasets/
    └── mitdb/
        ├── 100.dat
        ├── 100.hea
        ├── 100.atr
        ├── 101.dat
        ├── 101.hea
        ├── 101.atr
        └── ...
```

## Source

MIT-BIH Arrhythmia Database — PhysioNet:

https://www.physionet.org/content/mitdb/1.0.0/
