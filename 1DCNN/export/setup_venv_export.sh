#!/usr/bin/env bash
# Creates $ROOT/.venv-export: Python 3.12 + TF 2.19 + tf-keras 2.19 + tfmot 0.8.0.
# Needed because tensorflow-model-optimization (QAT / pruning) only works with
# legacy tf.keras (tf_keras), which does not exist for TF 2.22rc0 / Python 3.14.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
UV=~/.local/bin/uv
$UV venv --python 3.12 "$ROOT/.venv-export"
$UV pip install --python "$ROOT/.venv-export/bin/python" \
    "tensorflow==2.19.1" "tf-keras==2.19.0" "tensorflow-model-optimization==0.8.0" \
    "numpy<2.2" scipy scikit-learn pandas pyyaml wfdb matplotlib
