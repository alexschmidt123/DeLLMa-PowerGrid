import os
import json
import argparse
from typing import Dict

from tqdm import tqdm
from utils.prompt_utils import (
    inference,
    majority_voting_inference,
    chain_of_thought_inference,
    set_export_prompts_only,
    format_chatgpt_request_text,
    is_api_enabled,
)

from agent.farmagent import FarmAgent
from agent.tradeagent import TradeAgent
from agent.powergridagent import PowerGridAgent
from agent.agent import StateConfig, ActionConfig, PreferenceConfig
from utils.data_utils import get_combinations, FRUITS, STOCKS
from utils.llm_client import get_model_name
from functools import partial


def parse_baseline_response(response: Dict[str, str]) -> int:
    try:
        decision = int(response["decision"].split(".")[0].split()[1]) - 1
    except ValueError:
        decision = -1
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent_name",
        type=str,
        default="farmer",
        choices=["farmer", "trader", "powergrid"],
    )
    parser.add_argument("--year", type=str, default="2021")
    parser.add_argument(
        "--sample_size", type=int, default=16, help="number of beliefs to sample"
    )
    parser.add_argument(
        "--minibatch_size",
        type=int,
        default=32,
        help="minibatch size for DeLLMa prompt",
    )
    parser.add_argument(
        "--overlap_pct",
        type=float,
        default=0.25,
        help="overlap percentage for DeLLMa prompt",
    )
    parser.add_argument(
        "--sc-samples",
        type=int,
        default=5,
        help="number of samples for self-consistency",
    )
    parser.add_argument(
        "--results_path", type=str, default="results", help="path to data folder"
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default=None,
        help=(
            "Short name used as a subfolder under results_path to separate results by model. "
            "Defaults to the last component of the model name (e.g. 'gpt-4o', "
            "'Mistral-7B-Instruct-v0.3'). Set to '' to disable the subfolder."
        ),
    )
    parser.add_argument(
        "--export-prompts-only",
        action="store_true",
        help=(
            "Force offline mode: never call the API even if OPENAI_API_KEY is set. "
            "If this flag is omitted and no key is found, offline mode is used automatically."
        ),
    )

    # Method
    parser.add_argument(
        "--dellma_mode",
        type=str,
        default="zero-shot",
        choices=["zero-shot", "self-consistency", "cot", "rank", "rank-minibatch"],
    )
    parser.add_argument(
        "--powergrid-reveal-stability-stats",
        action="store_true",
        help=(
            "(powergrid only) Put per-regime mean stab, frac unstable, and stab/stabf in "
            "example rows. Matches the CSV oracle for instability triage trivially; default is OFF so the model "
            "must infer from tau/p/g only (fair benchmark vs farmer/trader)."
        ),
    )
    parser.add_argument(
        "--max_combinations",
        type=int,
        default=None,
        metavar="N",
        help=(
            "If set, only run the first N choice-set combinations (order from "
            "get_combinations). Useful for quick tests; default runs all."
        ),
    )

    args = parser.parse_args()
    set_export_prompts_only(args.export_prompts_only)

    # Resolve model tag: use explicit flag, else derive from model name
    if args.model_tag is None:
        raw_model = get_model_name()
        args.model_tag = raw_model.split("/")[-1]  # e.g. "gpt-4o", "Mistral-7B-Instruct-v0.3"
    # Build effective results root: results/{model_tag}/  (or results/ if tag is "")
    effective_results_root = (
        os.path.join(args.results_path, args.model_tag)
        if args.model_tag
        else args.results_path
    )
    print(f"[DeLLMa] results root: {effective_results_root}")
    if is_api_enabled():
        print(
            "DeLLMa: OpenAI API enabled (OPENAI_API_KEY detected). "
            "Calling the model; responses are written to response/."
        )
    else:
        print(
            "DeLLMa: offline mode — no API calls. "
            "Set OPENAI_API_KEY (e.g. in DeLLMa-main/.env) to enable the API, "
            "or use --export-prompts-only to force offline even with a key. "
            "Writing prompts (+ *_chatgpt.txt when applicable); response JSON files are stubs."
        )

    if args.agent_name == "farmer":
        products = FRUITS[args.year]
        domain = "agriculture"
        agent_init_fct = partial(
            FarmAgent,
            raw_context_fname=f"fruit-sept-{args.year}.txt",
        )
        budget = 10
    elif args.agent_name == "trader":
        products = STOCKS
        domain = "stocks"
        agent_init_fct = partial(
            TradeAgent,
            history_length=24,
        )
        budget = 10000
        args.year = ""
    elif args.agent_name == "powergrid":
        domain = "powergrid"
        agent_init_fct = partial(
            PowerGridAgent,
            summary_sample_rows=5,
            reveal_stability_stats=args.powergrid_reveal_stability_stats,
        )
        budget = 10000
        args.year = ""
    else:
        raise ValueError(f"Unknown agent_name: {args.agent_name}")

    action_config = ActionConfig(budget=budget)

    if args.dellma_mode.startswith("rank"):
        result_folder = (
            f"{effective_results_root}/{domain}/{args.year}/dellma/{args.dellma_mode}"
        )
        if args.dellma_mode == "rank-minibatch":
            result_folder = f"{result_folder}/sample_size={args.sample_size}_minibatch_size={args.minibatch_size}_overlap_pct={int(args.overlap_pct*100)}"
        state_enum_mode = "sequential"
        preference_config = PreferenceConfig(
            pref_enum_mode=args.dellma_mode,
            sample_size=args.sample_size,
            # if dellma_mode is rank, then all below are ignored
            minibatch_size=args.minibatch_size,
            overlap_pct=args.overlap_pct,
        )
    elif args.dellma_mode in ["zero-shot", "self-consistency", "cot"]:
        result_folder = f"{effective_results_root}/{domain}/{args.year}/{args.dellma_mode}"
        state_enum_mode = "base"
        preference_config = PreferenceConfig()
    else:
        raise ValueError(f"Unknown dellma mode: {args.dellma_mode}")

    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    combs = get_combinations(args.agent_name, source_year=args.year)
    if args.max_combinations is not None:
        combs = combs[: args.max_combinations]
    pbar = tqdm(combs)
    for choices in pbar:
        pbar.set_description(f"Processing {choices}")
        agent = agent_init_fct(
            choices=choices,
            state_config=StateConfig(state_enum_mode),
            action_config=action_config,
            preference_config=preference_config,
        )
        if args.dellma_mode == "cot":
            prompts = agent.prepare_chain_of_thought_prompt()
        else:
            prompts = agent.prepare_dellma_prompt()
        if type(prompts) == str:
            prompts = [prompts]
        if args.dellma_mode == "cot":
            inference_fct = partial(
                chain_of_thought_inference,
                system_content=agent.system_content,
            )
        elif args.dellma_mode == "self-consistency":
            inference_fct = partial(
                majority_voting_inference,
                system_content=agent.system_content,
                num_samples=args.sc_samples,
            )
        else:
            inference_fct = partial(
                inference,
                system_content=agent.system_content,
            )

        path = f"{result_folder}/{'-'.join(choices)}"
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + "/prompt"):
            os.makedirs(path + "/prompt")
        if not os.path.exists(path + "/response"):
            os.makedirs(path + "/response")

        if args.dellma_mode == "cot":
            output = inference_fct(chain=prompts)
            response = output["response"]
            prompt = output["query"]
            decision = response["decision"]
            with open(f"{path}/prompt/prompt.json", "w") as f:
                json.dump(prompt, f, indent=4)
            if not is_api_enabled():
                cot_dump = []
                for step in prompt:
                    cot_dump.append(
                        format_chatgpt_request_text(
                            agent.system_content,
                            step.get("prompt", "") + "<json>",
                        )
                    )
                cot_txt = "\n\n" + ("=" * 40) + "\n\n".join(cot_dump)
                with open(f"{path}/prompt/cot_all_steps_chatgpt.txt", "w", encoding="utf-8") as f:
                    f.write(cot_txt)
            with open(f"{path}/response/response.json", "w") as f:
                json.dump(response, f, indent=4)
        else:
            for i, prompt in enumerate(prompts):
                # save dellma prompt
                with open(f"{path}/prompt/prompt_{i}.txt", "w") as f:
                    f.write(prompt)
                if not is_api_enabled():
                    with open(
                        f"{path}/prompt/prompt_{i}_chatgpt.txt",
                        "w",
                        encoding="utf-8",
                    ) as f:
                        f.write(
                            format_chatgpt_request_text(
                                agent.system_content, prompt + "<json>"
                            )
                        )
                response = inference_fct(prompt)
                # save dellma response
                with open(f"{path}/response/response_{i}.json", "w") as f:
                    json.dump(response, f, indent=4)
