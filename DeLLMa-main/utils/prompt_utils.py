import os
import json
from typing import List, Dict, Callable, Optional, Any
from time import sleep

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_ENV_PATH = os.path.join(_PROJECT_ROOT, ".env")


def _load_env_manual(path: str = _ENV_PATH) -> None:
    """Parse simple KEY=value lines from .env without python-dotenv (fallback)."""
    if not os.path.isfile(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if not key:
                    continue
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                    value = value[1:-1]
                # Do not override a non-empty env var (same idea as python-dotenv)
                if not os.environ.get(key, "").strip():
                    os.environ[key] = value
    except OSError:
        pass


# Load DeLLMa-main/.env (python-dotenv if installed, else manual parse)
try:
    from dotenv import load_dotenv

    load_dotenv(_ENV_PATH)
except ImportError:
    pass
_load_env_manual(_ENV_PATH)

# ------------------------------------------------------------------
# Backend-agnostic LLM client (replaces direct openai.OpenAI usage)
# ------------------------------------------------------------------
from utils.llm_client import (
    get_llm_client,
    chat_completion,
    is_client_available,
    get_model_name,
)

# Set True from main.py when --export-prompts-only (skip API even if key exists)
_EXPORT_PROMPTS_ONLY: bool = False


def set_export_prompts_only(flag: bool) -> None:
    """If True, never call the API (use placeholders); still write prompt files from main."""
    global _EXPORT_PROMPTS_ONLY
    _EXPORT_PROMPTS_ONLY = bool(flag)


def _api_enabled() -> bool:
    return is_client_available() and not _EXPORT_PROMPTS_ONLY


def is_api_enabled() -> bool:
    """True when a backend is reachable and export-prompts-only mode is off."""
    return _api_enabled()


SUMMARY_PROMPT = (
    "You are a helpful agricultural expert studying a report published by the USDA"
)
ANALYST_PROMPT = "You are a helpful agricultural expert helping farmers decide what produce to plant next year."


def format_chatgpt_request_text(system_content: str, user_content: str) -> str:
    """Plain text you can paste into any chat LLM to reproduce the API call."""
    return (
        "=== ROLE: system ===\n"
        f"{system_content}\n\n"
        "=== ROLE: user ===\n"
        f"{user_content}\n"
    )


def format_query(
    query: str,
    format_instruction: str = "You should format your response as a JSON object.",
):
    return f"{query}\n{format_instruction}"


def openai_chat_model() -> str:
    """Chat Completions model id (kept for backward compatibility)."""
    return get_model_name()


def _offline_generic_response() -> Dict[str, Any]:
    return {
        "decision": (
            "Action 1. offline_placeholder: no API call — "
            "paste prompt_*_chatgpt.txt into your LLM and replace this file."
        ),
        "explanation": (
            "No LLM backend configured (or --export-prompts-only). "
            "Response JSON is a stub; use saved *_chatgpt.txt prompts."
        ),
    }


def inference(
    query: str,
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.0,
):
    if not _api_enabled():
        return _offline_generic_response()

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query + "<json>"},
    ]

    success = False
    response_text = ""
    while not success:
        try:
            response_text = chat_completion(
                messages=messages,
                temperature=temperature,
            )
            success = True
        except Exception as e:
            print(e)
            sleep(10)

    try:
        response = json.loads(response_text.lower())
    except Exception:
        response = response_text

    return response


def majority_voting_inference(
    query: str | List[str | Callable],
    system_content: str = ANALYST_PROMPT,
    temperature: float = 0.7,
    num_samples: int = 5,
    use_chain_of_thought: bool = False,
):
    if not _api_enabled():
        base = _offline_generic_response()
        return {
            "decision": base["decision"],
            "explanation": [base] * num_samples,
        }

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

    if not _api_enabled():
        chat_txt = format_chatgpt_request_text(SUMMARY_PROMPT, query + "<json>")
        chat_path = summary_fname.replace(".json", "_chatgpt.txt")
        with open(chat_path, "w", encoding="utf-8") as f:
            f.write(chat_txt)
        placeholder = {
            "overview": "[offline — no API; paste companion *_chatgpt.txt into an LLM]",
        }
        for p in products:
            placeholder[p] = "[offline placeholder]"
        with open(summary_fname, "w") as f:
            json.dump(placeholder, f, indent=4)
        return placeholder

    response = inference(query, SUMMARY_PROMPT, temperature)
    try:
        response = json.loads(response.lower())
    except Exception:
        response = response
    with open(summary_fname, "w") as f:
        json.dump(response, f, indent=4)
    return response
