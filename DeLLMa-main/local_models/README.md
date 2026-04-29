# local_models/

This directory is the recommended storage location for downloaded open-weight model weights.

## Default path

Set `LOCAL_MODEL_DIR` or `HF_HOME` to control where weights are stored:

```bash
export LOCAL_MODEL_DIR=/path/to/local_models   # this folder
export HF_HOME=$LOCAL_MODEL_DIR/.hf_cache       # Hugging Face cache root
```

## Supported models

| Model | HF identifier |
|-------|--------------|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Gemma 2 9B IT | `google/gemma-2-9b-it` |

## Downloading weights

Do NOT run these commands automatically — they download many GB of weights.
Run them manually when you are ready:

```bash
# Llama 3.1 8B (requires Meta access agreement on HF)
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir $LOCAL_MODEL_DIR/Llama-3.1-8B-Instruct

# Mistral 7B
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 \
    --local-dir $LOCAL_MODEL_DIR/Mistral-7B-Instruct-v0.3

# Gemma 2 9B (requires Google access agreement on HF)
huggingface-cli download google/gemma-2-9b-it \
    --local-dir $LOCAL_MODEL_DIR/gemma-2-9b-it
```

## Checking if models are available

```bash
bash ../scripts/setup_models.sh
```
