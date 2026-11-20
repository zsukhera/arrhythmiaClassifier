#!/usr/bin/env bash
# =============================================================================
# run_paper4.sh — KAN-Former-1D (paper4) wrapper
#
# Usage:  bash 1DCNN/run_paper4.sh <mode>
#
# Modes:
#   setup      — create .venv in HasanML/paper4/ and install PyTorch + deps
#   tests      — run unit tests (CPU, ~30 s)
#   sanity     — RR-only LightGBM sanity check (~2 min)
#   train      — 3-seed CPU training, 5-class AAMI
#   gpu        — 3-seed GPU training + distillation export
#   ablations  — 7 ablations A1-A6b (3 seeds each)
#   distill    — export soft labels + ONNX (requires prior training)
#   evaluate   — print metrics table from saved results
#   eval_full  — 5-class + 3-class N/S/V DS2 eval + temperature scaling  [RUN_DIR]
#   fix        — 3-seed GPU training with the balanced DS1 split (S-starvation fix)
#   export     — teacher soft labels (.npz) + ONNX + TorchScript       [RUN_DIR]
#
# Environment: PY=/path/to/python overrides the interpreter; on the lab box
# the shared $REPO_ROOT/.venv-torch (CUDA torch) is used automatically.
#
# This script can be run from any directory inside the fyp repo.
# =============================================================================

set -euo pipefail

# ── Locate repo root (parent of this script's directory) ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PAPER4_DIR="$REPO_ROOT/HasanML/paper4"
VENV_DIR="$PAPER4_DIR/.venv"

echo "=== run_paper4.sh ==="
echo "Repo root : $REPO_ROOT"
echo "Paper4 dir: $PAPER4_DIR"

if [[ ! -d "$PAPER4_DIR" ]]; then
    echo "ERROR: $PAPER4_DIR does not exist" >&2
    exit 1
fi

# ── Resolve Python interpreter ────────────────────────────────────────────────
if [[ -n "${PY:-}" ]]; then
    :
elif [[ -f "$REPO_ROOT/.venv-torch/bin/python" ]]; then
    PY="$REPO_ROOT/.venv-torch/bin/python"
elif [[ -f "$VENV_DIR/bin/python" ]]; then
    PY="$VENV_DIR/bin/python"
elif command -v python3 &>/dev/null; then
    PY="python3"
elif command -v python &>/dev/null; then
    PY="python"
else
    echo "ERROR: No Python found. Run:  bash 1DCNN/run_paper4.sh setup" >&2
    exit 1
fi

run_py() { cd "$PAPER4_DIR" && "$PY" "$@"; }

MODE="${1:-help}"

case "$MODE" in

# ── setup ─────────────────────────────────────────────────────────────────────
setup)
    echo "--- Setting up paper4 virtual environment ---"
    cd "$PAPER4_DIR"
    python3 -m venv "$VENV_DIR"
    "$VENV_DIR/bin/pip" install --upgrade pip

    # PyTorch — CPU build (safe default; switch to cu121 for GPU)
    echo "Installing PyTorch (CPU). For GPU, edit this script to use --index-url https://download.pytorch.org/whl/cu121"
    "$VENV_DIR/bin/pip" install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

    "$VENV_DIR/bin/pip" install -r "$PAPER4_DIR/requirements.txt"
    echo ""
    echo "Setup complete. Activate with:"
    echo "  source HasanML/paper4/.venv/bin/activate"
    echo "Or run modes through this script (it uses the venv automatically)."
    ;;

# ── tests ─────────────────────────────────────────────────────────────────────
tests)
    echo "--- Running unit tests (paper4) ---"
    run_py -m pytest tests/ -v --tb=short
    ;;

# ── sanity ────────────────────────────────────────────────────────────────────
sanity)
    echo "--- RR-only LightGBM sanity check ---"
    echo "Expected: accuracy ~90-93%, Se_S ~60%"
    run_py -m src.run_experiment --rr-baseline
    ;;

# ── train ─────────────────────────────────────────────────────────────────────
train)
    echo "--- Training: 5-class AAMI, 3 seeds, CPU ---"
    echo "Note: CPU training is slow (~3-4 h). Use 'gpu' mode for faster training."
    run_py -m src.run_experiment \
        --classes 5 \
        --seeds 0 1 2 \
        --device cpu \
        --results-dir results/cpu_run
    ;;

# ── gpu ───────────────────────────────────────────────────────────────────────
gpu)
    echo "--- Training: 5-class AAMI, 3 seeds, GPU + mixed precision + distillation ---"
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "WARNING: CUDA not available — falling back to CPU"
        DEVICE="cpu"
        EXTRA_FLAGS=""
    else
        DEVICE="cuda"
        EXTRA_FLAGS="--mixed-precision"
        GPU_NAME=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "unknown")
        echo "GPU: $GPU_NAME"
    fi

    # NOTE: --seeds takes seed VALUES (the old "--seeds 3" trained only seed 3)
    run_py -m src.run_experiment \
        --classes 5 \
        --seeds 0 1 2 \
        --device "$DEVICE" \
        --results-dir results/gpu_run \
        $EXTRA_FLAGS

    echo ""
    echo "--- DS2 evaluation (5-class + 3-class) ---"
    run_py -m src.evaluate_full --results-dir results/gpu_run
    ;;

# ── fix: balanced DS1 patient split + N/S/V macro-F1 model selection ───────────
fix)
    run_py -m src.run_experiment --seeds 0 1 2 --device cuda --mixed-precision \
        --results-dir results/fix_balsplit \
        --set data.ds1_split=balanced "training.select_classes=[0,1,2]"
    run_py -m src.evaluate_full --results-dir results/fix_balsplit
    ;;

# ── eval_full ────────────────────────────────────────────────────────────────
eval_full)
    run_py -m src.evaluate_full --results-dir "${2:-results/gpu_run}"
    ;;

# ── export: teacher soft labels + ONNX/TorchScript (no distillation) ─────────
export)
    run_py -m src.export_teacher --results-dir "${2:-results/gpu_run}" --out-dir results/teacher
    ;;

# ── ablations ────────────────────────────────────────────────────────────────
ablations)
    echo "--- Ablation study: A1-A6b (3 seeds each) ---"
    echo "This takes ~7x training time. Consider running on GPU overnight."
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        DEVICE="cpu"
        EXTRA_FLAGS=""
    else
        DEVICE="cuda"
        EXTRA_FLAGS="--mixed-precision"
    fi

    run_py -m src.run_experiment \
        --ablations \
        --seeds 0 \
        --ablation-seeds 0 1 2 \
        --device "$DEVICE" \
        --results-dir results/ablations \
        $EXTRA_FLAGS
    ;;

# ── distill ───────────────────────────────────────────────────────────────────
distill)
    # Teacher EXPORT only (soft labels .npz + ONNX + TorchScript, EMA weights).
    # The old "run_experiment --distill" path exported non-EMA weights for ONNX.
    echo "--- Teacher export (requires prior training + eval_full in RUN_DIR) ---"
    RESULTS_DIR="${2:-results/gpu_run}"
    run_py -m src.export_teacher --results-dir "$RESULTS_DIR" --out-dir results/teacher
    ;;

# ── evaluate ─────────────────────────────────────────────────────────────────
evaluate)
    echo "--- Printing metrics table from saved results ---"
    RESULTS_DIR="${2:-results/gpu_run}"
    run_py -c "
import json, sys
from pathlib import Path
from src.metrics import print_metrics_table, aggregate_seeds

results_dir = Path('$RESULTS_DIR')
seed_results = []
import pickle
for f in sorted((results_dir / 'cache').glob('DEFAULT_seed*.pkl')):
    with open(f, 'rb') as fp:
        seed_results.append(pickle.load(fp))

if not seed_results:
    print('No results found in', results_dir)
    sys.exit(1)

agg = aggregate_seeds(seed_results)
print_metrics_table('KAN-Former (DS2)', agg, n_classes=5)
"
    ;;

# ── help ──────────────────────────────────────────────────────────────────────
help|*)
    echo ""
    echo "Usage: bash 1DCNN/run_paper4.sh <mode>"
    echo ""
    echo "Modes:"
    echo "  setup      — create .venv in HasanML/paper4/ and install deps"
    echo "  tests      — run unit tests (CPU, ~30 s)"
    echo "  sanity     — RR-only LightGBM sanity check (~2 min)"
    echo "  train      — 3-seed CPU training, 5-class AAMI"
    echo "  gpu        — 3-seed GPU training + mixed precision + distillation"
    echo "  ablations  — 7 ablations A1-A6b (3 seeds each; slow)"
    echo "  distill    — export soft labels + ONNX (requires prior training)"
    echo "  evaluate   — print metrics table from saved results"
    echo "  eval_full  — DS2 5-class + 3-class N/S/V eval, temperature scaling [RUN_DIR]"
    echo "  fix        — 3-seed training with balanced DS1 split (S fix)"
    echo "  export     — teacher soft labels .npz + ONNX + TorchScript [RUN_DIR]"
    echo ""
    echo "See HasanML/paper4/README.md and 1DCNN/how_to_run_paper4.txt for details."
    ;;
esac
