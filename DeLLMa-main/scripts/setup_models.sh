#!/usr/bin/env bash
# scripts/setup_models.sh
# Check whether the three supported local models are already available.
# Does NOT download anything automatically.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${LOCAL_MODEL_DIR:-$SCRIPT_DIR/../local_models}"
MODEL_DIR="$(realpath "$MODEL_DIR")"

MODELS=(
    "meta-llama/Llama-3.1-8B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-2-9b-it"
)

echo "Checking model availability in: $MODEL_DIR"
echo ""

all_found=true
for model in "${MODELS[@]}"; do
    short="${model##*/}"
    local_path="$MODEL_DIR/$short"
    if [ -d "$local_path" ] && [ "$(ls -A "$local_path" 2>/dev/null)" ]; then
        echo "[OK]     $model -> $local_path"
    else
        echo "[MISSING] $model"
        echo "          Download with:"
        echo "          huggingface-cli download $model --local-dir $local_path"
        all_found=false
    fi
    echo ""
done

if $all_found; then
    echo "All models found. You can start vLLM with scripts/start_vllm_*.sh"
else
    echo "Some models are missing. See instructions above."
    echo "Set LOCAL_MODEL_DIR or HF_HOME to change the storage location."
fi
