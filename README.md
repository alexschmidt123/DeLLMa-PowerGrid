# DeLLMa: Decision Making Under Uncertainty with Large Language Models

[Project Website](https://DeLLMa.github.io) | [Implementation](https://github.com/DeLLMa/DeLLMa/)

Setup the environment by first downloading this repository, then install dependencies (for example inside a Conda env you create yourself—do not use `python -m venv`):

```
pip install -r requirements.txt
```

## Domains (pick one to test or run)

DeLLMa uses the same entrypoint for every domain. Choose which scenario you want with `--agent_name`:

| `--agent_name` | Setting |
|----------------|---------|
| `farmer` | Agriculture / crop choice (USDA-style context) |
| `trader` | Stocks |
| `grid` | Power grid (IEEE-14–style probing) |

There is no separate workflow for power grid: it is a third option alongside agriculture and stocks. Swap `agent_name` to run baselines, rank-based DeLLMa, or other modes on whichever domain you care about.

## Baselines

* Query GPT-4 for baseline methods. `[AGENT]` is one of `{farmer, trader, grid}`. `[BASELINE]` is one of `{zero-shot, cot, self-consistency}`.

```
python main.py --agent_name [AGENT] --dellma_mode [BASELINE] --results_path PATH/TO/RESULT
```

## DeLLMa Agents

* Query GPT-4 for DeLLMa-Naive. Here, `[SIZE]` denotes the **total samples size** we use for DeLLMa-Naive (i.e. distributed across all actions). We use 50 in our paper.

```
python main.py --agent_name [AGENT] --dellma_mode rank --sample_size [SIZE] --results_path PATH/TO/RESULT
```

* Query GPT-4 for DeLLMa-{Pairs, Top1}. Here, `[SIZE]` denotes the **per action sample size**. We use 64 for our best performing agent and ablate from 4 to 64 in our ablation studies. `[PCT]` denotes the proportions shared between minibatches. For example, we use overlap `0.25` for `farmer` and `0.5` for `trader` in the paper; you can set `--overlap_pct` for any agent.

```
python main.py --agent_name [AGENT] --dellma_mode rank-minibatch --sample_size [SIZE] --overlap_pct [PCT] --results_path PATH/TO/RESULT
```
