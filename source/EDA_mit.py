# %% [markdown]

# # MIT-BIH Arrhythmia Database — Exploratory Data Analysis

#

# **Arrhythmia Classification**

#

# Zain Ul Arifeen Sukhera

# 2026–27

# %%

#!pip install wfdb
#remove the comment from the above line if you want to use it in colab

# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wfdb

# %%

RECORD = "100"
#uncomment when using collab - data will be fetched from the internet 
"""
record = wfdb.rdrecord(
RECORD,
pn_dir="mitdb"
)
"""

#this version should be sufficient when one has the data available locally
record = wfdb.rdrecord(
    "../datasets/mitbih/mitdb/100"
)

print("Sampling frequency:", record.fs)
print("Signal length:", record.sig_len)
print("Channels:", record.sig_name)

#printing all the information contained in the hea format
print()
print("=======================================================")
print("Printing all the information contained in the hea format")
header = wfdb.rdheader("../datasets/mitbih/mitdb/100")
for key, value in vars(header).items():
    print(f"{key}: {value}")

print("=======================================================", end = '\n')
print()
# %% 
#now we shall explore the file format of atr
#rdann turns the .atr into an Annotation object, one entry per annotation in each list/array.
#long arrays are cut to their first 10 values so the output stays readable
print("=======================================================")
print("Printing all the information contained in the atr format")
atr = wfdb.rdann(
    "../datasets/mitbih/mitdb/100",
    "atr"
)
for key, value in vars(atr).items():
    if isinstance(value, (list, np.ndarray)) and len(value) > 10:
        print(f"{key}: {type(value).__name__} of length {len(value)}, first 10 -> {list(value[:10])}")
    else:
        print(f"{key}: {value}")

#one row per annotation. sample / fs converts the position to seconds.
#aux_note is stored as a C string, so strip the trailing null byte
atr_df = pd.DataFrame({
    "sample": atr.sample,
    "time_s": atr.sample / atr.fs,
    "symbol": atr.symbol,
    "subtype": atr.subtype,
    "chan": atr.chan,
    "num": atr.num,
    "aux_note": [n.strip("\x00") for n in atr.aux_note],
})
print()
print(atr_df.head(10).to_string())

#which fields actually carry information in this record
print()
for col in ["subtype", "chan", "num"]:
    print(f"{col}: unique values -> {sorted(atr_df[col].unique().tolist())}")

#what each symbol means, and how often it appears
print()
label_desc = dict(zip(wfdb.io.annotation.ann_label_table["symbol"],
                      wfdb.io.annotation.ann_label_table["description"]))
for symbol, count in atr_df["symbol"].value_counts().items():
    print(f"  {symbol}: {count} -> {label_desc.get(symbol, '')}")

#'+' annotations are rhythm changes, the rhythm name is in aux_note
print()
print("rhythm changes:")
print(atr_df[atr_df["symbol"] == "+"][["sample", "time_s", "aux_note"]].to_string())

print("=======================================================", end = '\n')
print()

# %%
#now we shall explore the file format .dat
#the .dat has no structure of its own, it is just the samples. rdheader leaves the signal arrays as None,
#rdrecord fills them in. physical=False gives the raw integers exactly as stored in the .dat
print("=======================================================")
print("Printing all the information contained in the dat format")
record_digital = wfdb.rdrecord(
    "../datasets/mitbih/mitdb/100",
    physical=False
)
d_signal = record_digital.d_signal
p_signal = record.p_signal

print("d_signal (raw ADC integers) shape:", d_signal.shape, "dtype:", d_signal.dtype)
print("p_signal (millivolts) shape:", p_signal.shape, "dtype:", p_signal.dtype)
print("first 5 frames, digital:", d_signal[:5].tolist())
print("first 5 frames, mV:     ", p_signal[:5].tolist())

#physical (mV) = (digital - baseline) / adc_gain
converted = (d_signal - np.array(header.baseline)) / np.array(header.adc_gain)
print("(digital - baseline) / adc_gain matches p_signal:", np.allclose(converted, p_signal))

#per channel checks against what the .hea promised
for ch, name in enumerate(header.sig_name):
    d = d_signal[:, ch]
    #checksum = sum of all samples, kept as a signed 16-bit number
    checksum = ((int(d.astype(np.int64).sum()) + 2**15) % 2**16) - 2**15
    print()
    print(f"{name}:")
    print(f"  init_value: {d[0]} (hea says {header.init_value[ch]})")
    print(f"  checksum: {checksum} (hea says {header.checksum[ch]})")
    print(f"  digital range: {d.min()} to {d.max()} (ADC allows 0 to {2**header.adc_res[ch] - 1})")
    print(f"  mV range: {p_signal[:, ch].min():.3f} to {p_signal[:, ch].max():.3f}")
    print(f"  mean: {p_signal[:, ch].mean():.3f} mV, std: {p_signal[:, ch].std():.3f} mV")

print("=======================================================", end = '\n')
print()

# %%

signal = record.p_signal

print("Signal shape:", signal.shape)

# %%

plt.figure(figsize=(15, 4))

plt.plot(signal[:, 0])

plt.title(f"MIT-BIH Record {RECORD}")
plt.xlabel("Sample")
plt.ylabel("Amplitude")

plt.show()

# %%
#uncomment when you need to get annotations from physionet instead of making use of local data
"""
annotation = wfdb.rdann(
RECORD,
"atr",
pn_dir="mitdb"
)
"""
#local
annotation = wfdb.rdann(
    "../datasets/mitbih/mitdb/100",
    "atr"
)
#number of annotations is of paramount importance to our ML model
print("Number of annotations:", len(annotation.sample))
print("First annotations:")
print(annotation.symbol[:55])

# %%
#this is a plot of the ECG, and is not a plot of the annotation
plt.figure(figsize=(15, 4))

plt.plot(signal[:, 0])

plt.title(f"ECG Record {RECORD} with Annotations")

plt.xlim(0, 5000)

plt.show()
