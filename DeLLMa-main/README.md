# DeLLMa-PowerGrid

**DeLLMa** (Decision-making with Large Language Models agents) applied to **power-grid operating regime selection**.

> See `CODEBASE_RESEARCH_SUMMARY.md` for a full code-level analysis of this project.

---

## Research Purpose

Given 7 microgrid operating clusters (derived from a 4-node smart-grid simulation dataset), the agent must **select the most unstable cluster** under operational uncertainty so a grid operator can prioritize intervention with a limited protection budget.

Each cluster is described by historical `tau` (response time constants), `p` (power injection/load), and `g` (control gain) statistics. The ground-truth stability margin `stab` is withheld from the LLM during inference and used only for evaluation.

**Metrics:**
- **Accuracy** — fraction of choice sets where the LLM picks the same cluster as the oracle (highest mean `stab`)
- **Regret** — `oracle_mean_stab − predicted_mean_stab` (lower is better; 0 is perfect)

---

## Project Structure

```
DeLLMa-main/
├── main.py                     # Main runner (prompt generation + API calls)
├── evaluate_dellma.py          # Offline evaluation (Acc, Regret)
├── agent/
│   ├── agent.py                # Base DeLLMaAgent: prompting, state/action logic
│   ├── powergridagent.py       # Power-grid agent (unstable-first objective)
│   ├── farmagent.py            # Agriculture agent
│   └── tradeagent.py          # Stock-trading agent
├── utils/
│   ├── llm_client.py           # Backend-agnostic LLM wrapper (NEW)
│   ├── prompt_utils.py         # inference(), CoT, majority-vote helpers
│   └── data_utils.py           # Cluster/stock/fruit constants + combo generator
├── data/powergrid/
│   ├── Data_for_UCI_named.csv  # Raw 10k-row simulation dataset
│   ├── cluster_00..06.csv      # Per-regime files (k-means, k=7)
│   └── cluster_split.py        # Clustering script
├── cache/                      # Cached LLM state-belief JSONs
├── results/                    # Generated prompts + LLM responses
├── local_models/               # Recommended weight storage
│   └── README.md
├── CODEBASE_RESEARCH_SUMMARY.md
└── requirements.txt
```

---

## LLM Backend Layer

All LLM calls pass through `utils/llm_client.py`. The backend is selected by environment variables — no code changes needed to switch models.

### Environment Variables

| Variable | Values | Default | Description |
|----------|--------|---------|-------------|
| `LLM_BACKEND` | `openai` / `vllm` | `openai` | Select backend |
| `OPENAI_API_KEY` | your key | — | Required for OpenAI backend |
| `OPENAI_MODEL` | model id | `gpt-4o` | OpenAI model name |
| `OPENAI_BASE_URL` | URL | SDK default | Override OpenAI base URL |
| `VLLM_BASE_URL` | URL | `http://localhost:8000/v1` | vLLM server endpoint |
| `VLLM_API_KEY` | token | `token-abc123` | vLLM auth token |
| `DEFAULT_MODEL` | model id | — | Override model for any backend |
| `LLM_JSON_MODE` | `true` / `false` | `true` | Enable JSON response format |
| `LOCAL_MODEL_DIR` | path | `local_models/` | Weight storage path |
| `HF_HOME` | path | — | Hugging Face cache root |

Set variables in `.env` (already auto-loaded) or export in your shell.

---

## Running DeLLMa

### 1. With OpenAI (hosted)

```bash
# .env
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o       # or gpt-4o-mini, gpt-4-turbo, etc.

conda activate dellma

# Generate prompts + call API for all 120 choice-set combinations
python main.py --agent_name powergrid --dellma_mode zero-shot --results_path results
python main.py --agent_name powergrid --dellma_mode rank      --results_path results
```

### 2. With local vLLM (Llama / Mistral / Gemma)

Use the unified runner. It can choose app + model and auto-start local vLLM.
If weights are missing, they are downloaded automatically to `local_models/`.

```bash
conda activate dellma

# Powergrid with local Llama
./run.sh --agent powergrid --backend vllm --model llama

# Trader with local Mistral
./run.sh --agent trader --backend vllm --model mistral

# Farmer (2021) with local Gemma
./run.sh --agent farmer --year 2021 --backend vllm --model gemma
```

> **Note on JSON mode:** Some local models may not support `response_format=json_object`.
> If you see parsing errors, add `LLM_JSON_MODE=false` to your `.env`.

---

## Evaluating Results

```bash
# Zero-shot accuracy and regret:
conda run -n dellma python evaluate_dellma.py \
    --agent_name powergrid --pref_enum_mode zero-shot --results_path results

# DeLLMa rank accuracy and regret:
conda run -n dellma python evaluate_dellma.py \
    --agent_name powergrid --pref_enum_mode rank --results_path results
```

---

## Running by Application

```bash
# OpenAI backend
./run.sh --agent powergrid --backend openai

# Local vLLM backend
./run.sh --agent powergrid --backend vllm --model llama
./run.sh --agent trader    --backend vllm --model mistral
./run.sh --agent farmer --year 2021 --backend vllm --model gemma
```

---

## Modes

| `--dellma_mode` | Description |
|----------------|-------------|
| `zero-shot` | Single direct prompt |
| `self-consistency` | 5× samples, majority vote |
| `cot` | Chain-of-thought (multi-turn) |
| `rank` | Rank sampled state-action pairs (full batch) |
| `rank-minibatch` | Rank in overlapping minibatches |

---

## Key Flags

```bash
python main.py --agent_name powergrid \
    --dellma_mode zero-shot \
    --results_path results \
    --max_choices 5 \                # optional: first X datasets (trader: legacy stocks_5)
    --powergrid-reveal-stability-stats  # give stab stats to the model (easy mode)
    --export-prompts-only            # generate prompts without calling API
```
