#!/usr/bin/env bash
# Batch evaluation (evaluate_dellma.py) for agriculture + stocks.
# Reads from DeLLMa-main/results by default (override with RESULTS=/path).

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RESULTS="${RESULTS:-$SCRIPT_DIR/results}"
PY="${PYTHON:-python3}"

echo "Using --results_path $RESULTS"

echo "Running DeLLMa-Pairs on Agriculture"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --pref_enum_mode rank-minibatch --sample_size 64 --alpha 2e-3

echo "Running DeLLMa-Top1 on Agriculture"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --pref_enum_mode rank-minibatch --sample_size 64 --mode top1 --alpha 1e-3

echo "Running DeLLMa-Naive on Agriculture"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --pref_enum_mode rank --sample_size 50 --alpha 1e-3

echo "Running DeLLMa-Pairs on Stocks"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --agent_name trader --softmax_mode action --overlap_pct 0.5 --pref_enum_mode rank-minibatch --sample_size 64 --alpha 2e-8 --temperature 0.1

echo "Running DeLLMa-Top1 on Stocks"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --agent_name trader --softmax_mode action --overlap_pct 0.5 --pref_enum_mode rank-minibatch --sample_size 64 --mode top1 --alpha 2e-8 --temperature 0.1

echo "Running DeLLMa-Naive on Stocks"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --agent_name trader --softmax_mode action --pref_enum_mode rank --sample_size 50 --alpha 2e-8

echo "Baselines..."

echo "Running Zero-Shot on Agriculture"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --pref_enum_mode zero-shot

echo "Running CoT on Agriculture"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --pref_enum_mode cot

echo "Running Self-Consistency on Agriculture"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --pref_enum_mode self-consistency

echo "Running Zero-Shot on Stocks"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --agent_name trader --pref_enum_mode zero-shot

echo "Running CoT on Stocks"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --agent_name trader --pref_enum_mode cot

echo "Running Self-Consistency on Stocks"
"$PY" evaluate_dellma.py --results_path "$RESULTS" --agent_name trader --pref_enum_mode self-consistency
