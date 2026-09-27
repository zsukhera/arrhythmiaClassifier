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
