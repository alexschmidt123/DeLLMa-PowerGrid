#!/usr/bin/env bash
# scripts/start_vllm_gemma.sh
# Start vLLM server with google/gemma-2-9b-it
#
# Usage:
#   bash scripts/start_vllm_gemma.sh
#
# Environment variables:
#   LOCAL_MODEL_DIR  path to downloaded weights (default: local_models/)
#   VLLM_API_KEY     API key for the server     (default: token-abc123)
#   VLLM_PORT        server port                (default: 8000)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${LOCAL_MODEL_DIR:-$SCRIPT_DIR/../local_models}"
MODEL_DIR="$(realpath "$MODEL_DIR")"

MODEL_ID="google/gemma-2-9b-it"
MODEL_PATH="$MODEL_DIR/gemma-2-9b-it"
API_KEY="${VLLM_API_KEY:-token-abc123}"
PORT="${VLLM_PORT:-8000}"

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model not found at $MODEL_PATH"
    echo "Run: huggingface-cli download $MODEL_ID --local-dir $MODEL_PATH"
    exit 1
fi

echo "[vLLM] Starting $MODEL_ID on port $PORT ..."
echo "[vLLM] Set LLM_BACKEND=vllm and DEFAULT_MODEL=$MODEL_ID in your .env"

# Gemma 2 uses a sliding window attention that requires flash-attn or eager mode
vllm serve "$MODEL_PATH" \
    --served-model-name "$MODEL_ID" \
    --api-key "$API_KEY" \
    --port "$PORT" \
    --enforce-eager
