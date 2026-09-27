#!/bin/bash

BASE_URL="https://physionet.org/files/mitdb/1.0.0"
OUTPUT_DIR="$HOME/datasets/mitdb"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR" || exit 1

# MIT-BIH Arrhythmia Database record numbers
records=(
    100 101 102 103 104 105 106 107 108 109
    111 112 113 114 115 116 117 118 119 121
    122 123 124 200 201 202 203 205 207 208 209
    210 212 213 214 215 217 219 220 221 222
    223 228 230 231 232 233 234
)

for record in "${records[@]}"; do
    echo "===== Downloading record $record ====="

    wget -c "$BASE_URL/$record.dat"
    wget -c "$BASE_URL/$record.hea"
    wget -c "$BASE_URL/$record.atr"
done

echo
echo "======================================"
echo "MIT-BIH download complete."
echo "Location: $OUTPUT_DIR"
echo "======================================"