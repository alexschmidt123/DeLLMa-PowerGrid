# local_models/

This directory is the recommended storage location for downloaded open-weight model weights.

## Default path

`run.sh` uses this directory by default and stores cache under:

```bash
local_models/.hf_cache
```

You can still override using `LOCAL_MODEL_DIR` and `HF_HOME` if needed.

## Supported models

| Model | HF identifier |
|-------|--------------|
| Llama 3.1 8B Instruct | `meta-llama/Llama-3.1-8B-Instruct` |
| Mistral 7B Instruct v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` |
| Gemma 2 9B IT | `google/gemma-2-9b-it` |

## Downloading weights

When you run:

```bash
./run.sh --agent powergrid --backend vllm --model llama
```

`vllm` will auto-download missing model files into this folder.
