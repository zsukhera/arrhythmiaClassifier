#!/usr/bin/env bash
# run_paper3.sh — Convenience wrapper for HasanML/paper3 experiments
#
# Usage (from anywhere inside the fyp repo):
#   bash 1DCNN/run_paper3.sh [mode] [extra flags]
#
# Modes:
#   setup       Install pip dependencies
#   tests       Run 34-test suite (always do this first)
#   sanity      RR-only XGBoost + template plots (~2 min)
#   train       Run P1 (Model 6, 10 seeds, ~20–60 min CPU / ~5 min GPU)
#   quick       Run P1 with 1 seed only (fast smoke test)
#   gpu         Run P1 on GPU with mixed precision (RTX/Ampere/Ada)
#   binary      Run P1 + P2 (binary AAMI) + P3 (binary raw)
#   grid        Full Table 4 grid — 36 configs × 10 seeds (~6–12 h CPU)
#   export      TFLite export after P1 is trained
#   evaluate    Print metrics for all cached runs (no retraining)
#
# Any extra flags after the mode are forwarded to run_experiment.py.
# Examples:
#   bash 1DCNN/run_paper3.sh train --device cpu
#   bash 1DCNN/run_paper3.sh evaluate --run P1 --per-seed
#   bash 1DCNN/run_paper3.sh evaluate --csv results/all.csv

set -euo pipefail

# ── Locate repo root (parent of this script's directory) ─────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
PAPER3_DIR="$REPO_ROOT/HasanML/paper3"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"

# ── Validate environment ──────────────────────────────────────────────────────
if [[ ! -d "$PAPER3_DIR" ]]; then
    echo "ERROR: HasanML/paper3/ not found at $PAPER3_DIR"
    exit 1
fi

if [[ ! -x "$VENV_PYTHON" ]]; then
    echo "ERROR: venv not found at $VENV_PYTHON"
    echo "       Create it with: python3 -m venv $REPO_ROOT/.venv"
    exit 1
fi

# ── Parse mode ────────────────────────────────────────────────────────────────
MODE="${1:-help}"
shift || true   # remaining args forwarded to Python

cd "$PAPER3_DIR"

run_py() {
    "$VENV_PYTHON" "$@"
}

case "$MODE" in

    setup)
        echo "==> Installing paper3 dependencies..."
        "$VENV_PYTHON" -m pip install -r requirements.txt
        echo "Done."
        ;;

    tests)
        echo "==> Running test suite..."
        run_py -m pytest tests/ -v "$@"
        ;;

    sanity)
        echo "==> RR-only XGBoost sanity check + template plots..."
        run_py -m src.run_experiment --rr-baseline "$@"
        echo ""
        echo "==> Plotting matched-filter templates..."
        run_py -m src.run_experiment --templates-only "$@"
        echo ""
        echo "Results written to:"
        echo "  results/rr_baseline_xgboost.csv"
        echo "  results/figures/templates.png"
        ;;

    train)
        echo "==> Training P1 (Model 6, 10 seeds)..."
        run_py -m src.run_experiment --p1 "$@"
        echo ""
        echo "==> Summary of results:"
        run_py -m src.evaluate --run P1
        ;;

    quick)
        echo "==> Quick smoke test — P1, 1 seed only..."
        run_py -m src.run_experiment --p1 --seeds 0 "$@"
        echo ""
        run_py -m src.evaluate --run P1
        ;;

    gpu)
        echo "==> GPU training — P1 with mixed precision (RTX/Ampere/Ada)..."
        run_py -m src.run_experiment --p1 --device cuda --mixed-precision "$@"
        echo ""
        run_py -m src.evaluate --run P1
        ;;

    binary)
        echo "==> Training P1 + P2 (binary AAMI) + P3 (binary raw)..."
        run_py -m src.run_experiment --p1 --p2 --p3 "$@"
        echo ""
        run_py -m src.evaluate
        ;;

    grid)
        echo "==> Full Table 4 grid (36 configs × 10 seeds)..."
        echo "    Expected: ~6–12 hours on CPU, ~1–2 hours on GPU."
        echo "    Run is resumable — Ctrl-C and re-run to continue."
        run_py -m src.run_experiment --grid "$@"
        echo ""
        run_py -m src.evaluate
        ;;

    export)
        echo "==> TFLite export (requires P1 to have been trained)..."
        run_py -m src.run_experiment --p1 --export "$@"
        echo ""
        echo "Exported files in results/export/:"
        ls -lh results/export/ 2>/dev/null || echo "  (directory not yet created)"
        ;;

    evaluate)
        echo "==> Evaluating cached results..."
        run_py -m src.evaluate "$@"
        ;;

    help|--help|-h)
        grep '^#' "$SCRIPT_DIR/run_paper3.sh" | head -20 | sed 's/^# \{0,1\}//'
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Run:  bash 1DCNN/run_paper3.sh help"
        exit 1
        ;;
esac
