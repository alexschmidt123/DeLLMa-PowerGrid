#!/usr/bin/env bash
# Unified runner:
# - choose app: farmer/trader/powergrid/seir
# - choose backend: openai/vllm
# - choose model (llama/mistral/gemma or HF model id for vLLM)
# - auto-start local vLLM and auto-download models into ./local_models when needed
#
# Backward compatible examples:
#   ./run.sh powergrid
#   ./run.sh seir
#   ./run.sh farmer 2021
#
# Unified examples:
#   ./run.sh --agent seir --backend vllm --model llama --max-choices 3
#   ./run.sh --agent powergrid --backend vllm --model llama
#   ./run.sh --agent trader --backend vllm --model mistral --port 8001
#   ./run.sh --agent farmer --year 2021 --backend openai

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_FILE="${SCRIPT_DIR}/.env"
if [[ -f "${ENV_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
  set +a
fi

RESULTS="${RESULTS:-$SCRIPT_DIR/results}"
PY="${PYTHON:-python3}"
ALL_METHODS=(zero-shot self-consistency cot rank rank-minibatch)
LOG_DIR="${SCRIPT_DIR}/logs/vllm"
mkdir -p "${LOG_DIR}"

usage() {
  cat <<'EOF' >&2
Usage:
  ./run.sh <farmer|trader|powergrid|seir> [farmer_year] [extra main.py args...]
  ./run.sh --agent <farmer|trader|powergrid|seir> [options] [-- extra main.py args...]

Options:
  --agent, -a <name>            Application: farmer|trader|powergrid|seir
  --year, -y <YYYY>             Farmer year (default: 2021)
  --backend <openai|vllm>       LLM backend (default: openai; auto vllm if --model set)
  --model, -m <key|hf-id>       vLLM model: llama|mistral|gemma or full HF model id
  --method <name>               Run one method (repeatable): zero-shot|self-consistency|cot|rank|rank-minibatch
  --port <N>                    vLLM port (default: 8000)
  --vllm-host <host>            vLLM server host bind (default: 127.0.0.1)
  --gpu-util <0-1>              vLLM gpu-memory-utilization (default: 0.9)
  --max-model-len <N>           vLLM max model len (default: 16384)
  --no-vllm-start               Do not auto-start vLLM; assume server is already running
  --max-choices <X>             Use first X datasets in the agent list (passed to main.py)
  --help, -h                    Show this help

Notes:
  - For vLLM, model files are stored under ./local_models by default.
  - If --method is omitted, run all methods.
  - Without --max-choices, all datasets are used and all choice-set combinations are run.
  - Trader: --max-choices 5 matches legacy stocks_5 (AMD, BILI, DIS, GE, GME).
  - Extra args are passed to main.py unchanged.
EOF
  exit 1
}

map_model_key() {
  local key="$1"
  case "$key" in
    llama) echo "meta-llama/Llama-3.1-8B-Instruct" ;;
    mistral) echo "mistralai/Mistral-7B-Instruct-v0.3" ;;
    gemma) echo "google/gemma-2-9b-it" ;;
    *) echo "$key" ;;
  esac
}

ensure_vllm_server() {
  local model_id="$1"
  local host="$2"
  local port="$3"
  local api_key="$4"
  local gpu_util="$5"
  local max_len="$6"

  local local_model_dir="${LOCAL_MODEL_DIR:-$SCRIPT_DIR/local_models}"
  local hf_home="${HF_HOME:-$local_model_dir/.hf_cache}"
  mkdir -p "$local_model_dir" "$hf_home"
  export HF_HOME

  local base_url="http://localhost:${port}/v1"
  # On lab Ubuntu nodes without nvcc, flashinfer JIT can fail at startup.
  # Prefer xformers backend by default to avoid requiring local CUDA toolkit.
  export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
  export LLM_BACKEND="vllm"
  export DEFAULT_MODEL="$model_id"
  export VLLM_API_KEY="$api_key"
  export VLLM_BASE_URL="$base_url"

  if curl -fsS "${base_url}/models" -H "Authorization: Bearer ${api_key}" >/dev/null 2>&1; then
    echo "[run.sh] vLLM already running at ${base_url}; reusing it."
    echo "[run.sh] model dir: ${local_model_dir}"
    echo "[run.sh] HF cache:  ${hf_home}"
    return
  fi

  local log_file="${LOG_DIR}/vllm_${port}.log"
  local pid_file="${LOG_DIR}/vllm_${port}.pid"

  echo "[run.sh] starting vLLM model=${model_id} on ${host}:${port}"
  echo "[run.sh] model dir: ${local_model_dir}"
  echo "[run.sh] HF cache:  ${hf_home}"
  echo "[run.sh] attention backend: ${VLLM_ATTENTION_BACKEND}"
  nohup vllm serve "$model_id" \
    --host "$host" \
    --port "$port" \
    --api-key "$api_key" \
    --download-dir "$local_model_dir" \
    --gpu-memory-utilization "$gpu_util" \
    --max-model-len "$max_len" \
    > "$log_file" 2>&1 &
  local server_pid="$!"
  echo "$server_pid" > "$pid_file"

  echo "[run.sh] waiting for vLLM to become ready..."
  local i
  for i in {1..120}; do
    if curl -fsS "${base_url}/models" -H "Authorization: Bearer ${api_key}" >/dev/null 2>&1; then
      echo "[run.sh] vLLM is ready at ${base_url}"
      return
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      echo "[run.sh] ERROR: vLLM process exited during startup." >&2
      echo "[run.sh] Check log: ${log_file}" >&2
      echo "---------- vLLM log tail ----------" >&2
      tail -n 60 "$log_file" >&2 || true
      echo "-----------------------------------" >&2
      exit 1
    fi
    sleep 2
  done

  echo "[run.sh] ERROR: vLLM did not become ready in time." >&2
  echo "[run.sh] Check log: ${log_file}" >&2
  echo "---------- vLLM log tail ----------" >&2
  tail -n 60 "$log_file" >&2 || true
  echo "-----------------------------------" >&2
  exit 1
}

AGENT=""
YEAR="2021"
BACKEND="${LLM_BACKEND:-openai}"
MODEL_KEY="${DEFAULT_MODEL:-}"
PORT="${PORT:-8000}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
GPU_UTIL="${GPU_UTIL:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
AUTO_START_VLLM=1
RUN_METHODS=()
EXTRA_ARGS=()

# Backward compatibility: first positional arg can be AGENT.
if [[ "${1:-}" =~ ^(farmer|trader|powergrid|seir)$ ]]; then
  AGENT="$1"
  shift
  if [[ "$AGENT" == "farmer" && "${1:-}" =~ ^[0-9]{4}$ ]]; then
    YEAR="$1"
    shift
  fi
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent|-a) AGENT="${2:-}"; shift 2 ;;
    --year|-y) YEAR="${2:-}"; shift 2 ;;
    --backend) BACKEND="${2:-}"; shift 2 ;;
    --model|-m) MODEL_KEY="${2:-}"; shift 2 ;;
    --method) RUN_METHODS+=("${2:-}"); shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --vllm-host) VLLM_HOST="${2:-}"; shift 2 ;;
    --gpu-util) GPU_UTIL="${2:-}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:-}"; shift 2 ;;
    --no-vllm-start) AUTO_START_VLLM=0; shift ;;
    --max-choices) EXTRA_ARGS+=("--max_choices" "${2:-}"); shift 2 ;;
    --help|-h) usage ;;
    --) shift; EXTRA_ARGS+=("$@"); break ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$AGENT" in
  farmer|trader|powergrid|seir) ;;
  *) usage ;;
esac

if [[ "${#RUN_METHODS[@]}" -eq 0 ]]; then
  RUN_METHODS=("${ALL_METHODS[@]}")
fi

for m in "${RUN_METHODS[@]}"; do
  case "$m" in
    zero-shot|self-consistency|cot|rank|rank-minibatch) ;;
    *)
      echo "[run.sh] Unknown --method: $m" >&2
      usage
      ;;
  esac
done

# If user selected a model and did not explicitly set backend, assume vLLM.
if [[ -n "$MODEL_KEY" && "${BACKEND}" == "openai" ]]; then
  BACKEND="vllm"
fi

if [[ "$BACKEND" == "vllm" ]]; then
  MODEL_ID="$(map_model_key "${MODEL_KEY:-llama}")"
  API_KEY="${VLLM_API_KEY:-token-abc123}"
  if [[ -z "${MAX_MODEL_LEN}" ]]; then
    # Local open models benefit from a larger context window for rank prompts.
    # Llama-3.1-8B supports much larger contexts; keep a conservative default.
    MAX_MODEL_LEN="16384"
  fi
  if [[ "$AUTO_START_VLLM" -eq 1 ]]; then
    ensure_vllm_server "$MODEL_ID" "$VLLM_HOST" "$PORT" "$API_KEY" "$GPU_UTIL" "$MAX_MODEL_LEN"
  else
    export LLM_BACKEND="vllm"
    export DEFAULT_MODEL="$MODEL_ID"
    export VLLM_API_KEY="$API_KEY"
    export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:${PORT}/v1}"
  fi
else
  export LLM_BACKEND="openai"
fi

if [[ "$AGENT" == "farmer" ]]; then
  for m in "${RUN_METHODS[@]}"; do
    echo "=== ${AGENT} year=${YEAR} mode=${m} backend=${BACKEND} model=${MODEL_KEY:-default} ==="
    "$PY" main.py --agent_name "$AGENT" --year "$YEAR" --dellma_mode "$m" --results_path "$RESULTS" "${EXTRA_ARGS[@]}"
  done
else
  for m in "${RUN_METHODS[@]}"; do
    echo "=== ${AGENT} mode=${m} backend=${BACKEND} model=${MODEL_KEY:-default} ==="
    "$PY" main.py --agent_name "$AGENT" --dellma_mode "$m" --results_path "$RESULTS" "${EXTRA_ARGS[@]}"
  done
fi

echo "Done. Evaluate with evaluate_dellma.py using matching --pref_enum_mode and rank hyperparameters."
