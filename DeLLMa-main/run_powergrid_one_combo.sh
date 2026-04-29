#!/usr/bin/env bash
# Powergrid smoke test: first choice set only (same order as get_combinations), zero-shot + DeLLMa rank.
#
# Usage (from anywhere):
#   ./run_powergrid_one_combo.sh
#   ./run_powergrid_one_combo.sh --export-prompts-only
#   RESULTS=/tmp/out PYTHON=python ./run_powergrid_one_combo.sh
#
# Conda: `conda activate dellma` before running, or set PYTHON to that interpreter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS="${RESULTS:-$SCRIPT_DIR/results}"
PY="${PYTHON:-python3}"
MAX="${MAX_COMBINATIONS:-1}"

for m in zero-shot rank; do
  echo "=== powergrid mode=${m} max_combinations=${MAX} ==="
  "$PY" main.py \
    --agent_name powergrid \
    --dellma_mode "$m" \
    --max_combinations "$MAX" \
    --results_path "$RESULTS" \
    "$@"
done

echo "Done."
echo "Evaluate zero-shot:  python evaluate_dellma.py --agent_name powergrid --pref_enum_mode zero-shot --max_combinations ${MAX} --results_path ${RESULTS}"
echo "Evaluate rank:       python evaluate_dellma.py --agent_name powergrid --pref_enum_mode rank --max_combinations ${MAX} --results_path ${RESULTS}"
