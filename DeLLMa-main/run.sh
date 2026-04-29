#!/usr/bin/env bash
# Run all five methods (baselines + DeLLMa) via main.py for one agent:
#   zero-shot, self-consistency, cot, rank, rank-minibatch
#
# Usage:
#   ./run.sh farmer [YEAR] [extra args passed to main.py...]
#   ./run.sh trader [extra args passed to main.py...]
#   ./run.sh powergrid [extra args passed to main.py...]
#
# Examples:
#   ./run.sh powergrid
#   ./run.sh farmer 2021
#   ./run.sh trader --export-prompts-only
#   RESULTS=/path/to/out ./run.sh powergrid
#   ./run.sh powergrid --sample_size 16   # writes under ./results by default

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Output under DeLLMa-main/results (override with RESULTS=/path or extra --results_path after)
RESULTS="${RESULTS:-$SCRIPT_DIR/results}"
PY="${PYTHON:-python3}"

usage() {
  echo "Usage: $0 <farmer|trader|powergrid> [farmer_year|extra main.py args...]" >&2
  echo "  farmer:   optional year as second arg (default 2021), then any main.py flags." >&2
  echo "  trader|powergrid: all args after the agent name go to main.py." >&2
  exit 1
}

[[ "${1:-}" ]] || usage
AGENT="$1"
shift

case "$AGENT" in
  farmer|trader|powergrid) ;;
  *) usage ;;
esac

METHODS=(zero-shot self-consistency cot rank rank-minibatch)

if [[ "$AGENT" == "farmer" ]]; then
  if [[ "${1:-}" =~ ^[0-9]{4}$ ]]; then
    YEAR="$1"
    shift
  else
    YEAR="2021"
  fi
  for m in "${METHODS[@]}"; do
    echo "=== ${AGENT} year=${YEAR} mode=${m} ==="
    "$PY" main.py --agent_name "$AGENT" --year "$YEAR" --dellma_mode "$m" --results_path "$RESULTS" "$@"
  done
else
  for m in "${METHODS[@]}"; do
    echo "=== ${AGENT} mode=${m} ==="
    "$PY" main.py --agent_name "$AGENT" --dellma_mode "$m" --results_path "$RESULTS" "$@"
  done
fi

echo "Done. Evaluate with evaluate_dellma.py using matching --pref_enum_mode and rank hyperparameters."
