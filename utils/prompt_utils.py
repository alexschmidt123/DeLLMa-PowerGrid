from __future__ import annotations

import os
import json
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional
from time import sleep
import pandas as pd

import openai
from openai import OpenAI

if TYPE_CHECKING:
    from utils.data_utils import PosteriorSummary

api_key = os.environ.get("OPENAI_API_KEY")
CLIENT = None


def _get_client():
    """Lazy-init OpenAI client so evaluation can run without API key."""
    global CLIENT
    if CLIENT is None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for running main.py (GPT-4 calls).")
        CLIENT = OpenAI(api_key=api_key)
    return CLIENT

SUMMARY_PROMPT = (
    "You are a helpful agricultural expert studying a report published by the USDA"
)
ANALYST_PROMPT = "You are a helpful agricultural expert helping farmers decide what produce to plant next year."


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object.",
):
    return f"{query}\n{format_instruction}"


def inference(
    query: str,
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.0,
):
    success = False
    while not success:
        try:
            response = _get_client().chat.completions.create(
                model="gpt-4-1106-preview",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": query + "<json>"},
                ],
                temperature=temperature,
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        response = response.choices[0].message.content
    except Exception as e:
        print(e)
        response = ""

    try:
        response = json.loads(response.lower())
    except:
        response = response

    return response


def majority_voting_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.7,
    num_samples: int = 5,
    use_chain_of_thought: bool = False,
):
    responses = []
    for _ in range(num_samples):
        if use_chain_of_thought:
            response = chain_of_thought_inference(
                chain=query, system_content=system_content, temperature=temperature
            )["response"]
        else:
            response = inference(query, system_content, temperature)
        responses.append(response)

    decisions = [r["decision"] for r in responses]
    majority_decision = max(set(decisions), key=decisions.count)
    response = {
        "decision": majority_decision,
        "explanation": responses,
    }
    return response


def chain_of_thought_inference(
    chain: List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.5,
):
    history = {}
    for query in chain:
        if isinstance(query, str):
            response = inference(query, system_content, temperature)
        else:
            previous_results = [history[k] for k in history.keys()]
            query = query(*previous_results)
            response = inference(query, system_content, temperature)
        history[query] = response

    return {
        "query": [{"prompt": q, "response": r} for q, r in history.items()],
        "response": response,
    }


def summarize(
    fname: str, products: List[str], temperature: float = 0.0
) -> Dict[str, str]:
    products = sorted(p.lower() for p in products)
    summary_fname = fname.split(".")[0] + "-" + "-".join(products) + ".json"
    if os.path.exists(summary_fname):
        # print(f"Summary file {summary_fname} already exists.")
        return json.load(open(summary_fname))

    report = open(fname).read()
    query = f"Below is an agriculture report published by the USDA:\n\n{report}\n\n"

    format_instruction = f"""Please write a detailed summary of the report.

You should format your response as a JSON object. The JSON object should contain the following keys:
    overview: a string that describes, in detail, the overview of the report. Your summary should focus on factors that affect the overall furuit and nut market.
    """
    for p in products:
        format_instruction += f"""
    {p}: a string that describes, in detail, information pertaining to {p} in the report. You should include information on {p} prices and production, as well as factors that affect them. 
        """
    query = format_query(query, format_instruction)
    response = inference(query, SUMMARY_PROMPT, temperature)
    try:
        response = json.loads(response.lower())
    except:
        response = response
    with open(summary_fname, "w") as f:
        json.dump(response, f, indent=4)
    return response


# =============================================================================
# Grid domain (IEEE-14): format solver / belief outputs into LLM context strings.
# Imports from data_utils are lazy inside these functions so farmer/trader code paths
# that only use format_query / inference above do not load the grid stack.
# =============================================================================


def format_case_summary(case_config: Dict[str, Any]) -> str:
    from utils.data_utils import F0_HZ

    meta = case_config.get("pandapower") or {}
    lines = [
        f"Case: {case_config.get('case_name', 'IEEE-14')} at {case_config.get('frequency_hz', F0_HZ)} Hz, "
        f"Sbase={case_config.get('base_mva', 100)} MVA.",
        f"Finite synchronous machines in reduced model: {case_config.get('finite_machines', 'n/a')}.",
    ]
    if meta.get("converged") is not None:
        lines.append(
            f"PandaPower load-flow (optional): converged={meta.get('converged')}, "
            f"buses={meta.get('n_bus', '?')}, branches={meta.get('n_branch', '?')}."
        )
    if meta.get("note"):
        lines.append(str(meta["note"]))
    csv_case = case_config.get("csv_case") or {}
    if csv_case:
        tl = csv_case.get("total_load_p_mw")
        nbus = csv_case.get("n_buses")
        nld = csv_case.get("n_loads")
        nl = csv_case.get("n_lines")
        tl_s = f"{tl:.2f}" if tl is not None else "n/a"
        lines.append(
            f"On-disk case CSVs ({csv_case.get('source_dir', 'data/powergrid/simulation/pandapower/ieee14')}): "
            f"buses={nbus}, loads={nld}, lines={nl}, "
            f"total nominal P_load≈{tl_s} MW (from loads.csv)."
        )
    return " ".join(lines)


def format_posterior_summary(p: PosteriorSummary) -> str:
    parts = [
        f"Inertia level (belief): {p.inertia_level}.",
        f"Coupling level (belief): {p.coupling_level}.",
        f"Response regime (belief): {p.response_regime}.",
    ]
    if p.entropy_bits is not None:
        parts.append(f"Discrete belief entropy (approx. bits): {p.entropy_bits:.3f}.")
    return " ".join(parts)


def format_probe_history(entries: List[Dict[str, Any]]) -> str:
    """Each entry: action_id, features (optional), valid flag."""
    if not entries:
        return "No previous probes recorded in this session."
    lines = []
    for i, e in enumerate(entries, start=1):
        aid = e.get("action_id", "?")
        v = e.get("valid", True)
        mf = e.get("max_abs_freq_dev_hz")
        mr = e.get("max_abs_rocof_hz_s")
        lines.append(
            f"Probe {i}: {aid}, valid={v}, "
            f"max|Δf|≈{mf if mf is not None else 'n/a'} Hz, "
            f"max|ROCOF|≈{mr if mr is not None else 'n/a'} Hz/s."
        )
    return "\n".join(lines)


def build_grid_context(
    *,
    case_config: Optional[Dict[str, Any]] = None,
    posterior: Optional[PosteriorSummary] = None,
    probe_history: Optional[List[Dict[str, Any]]] = None,
    current_probe: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Assemble short readable context. Numbers come from solver outputs passed in
    (or default case_config from load_ieee14_case_config).
    """
    from utils.data_utils import load_ieee14_case_config

    case_config = case_config or load_ieee14_case_config()
    blocks = []
    blocks.append("=== IEEE-14 reduced swing model (solver-grounded) ===")
    blocks.append(format_case_summary(case_config))
    if posterior:
        blocks.append(format_posterior_summary(posterior))
    if probe_history:
        blocks.append("=== Previous probes ===")
        blocks.append(format_probe_history(probe_history))
    if current_probe:
        blocks.append("=== Current candidate / last run summary ===")
        cp = current_probe
        blocks.append(
            f"Action: {cp.get('action_id')}, bus={cp.get('bus')}, "
            f"amplitude={cp.get('amplitude_pu')} pu, duration={cp.get('duration_s')} s."
        )
        if "max_abs_freq_dev_hz" in cp:
            blocks.append(
                f"Features: max|Δf|={cp.get('max_abs_freq_dev_hz')} Hz, "
                f"max|ROCOF|={cp.get('max_abs_rocof_hz_s')} Hz/s, "
                f"stability={cp.get('stability_flag')}."
            )
    return "\n".join(blocks)
