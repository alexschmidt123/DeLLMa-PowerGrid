import os
import json
import re
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

    def _is_context_length_error(err_text: str) -> bool:
        e = err_text.lower()
        return (
            "maximum context length" in e
            or "requested" in e and "tokens" in e and "maximum" in e
            or "context length" in e
        )

    def _extract_token_counts(err_text: str) -> tuple[Optional[int], Optional[int]]:
        # Matches patterns like:
        # "maximum context length is 4096 tokens ... requested 4884 tokens"
        max_match = re.search(r"maximum context length is (\d+)", err_text, re.IGNORECASE)
        req_match = re.search(r"requested (\d+) tokens", err_text, re.IGNORECASE)
        max_tokens = int(max_match.group(1)) if max_match else None
        req_tokens = int(req_match.group(1)) if req_match else None
        return max_tokens, req_tokens

    def _truncate_middle(text: str, keep_chars: int) -> str:
        if len(text) <= keep_chars:
            return text
        if keep_chars < 80:
            return text[:keep_chars]
        head = int(keep_chars * 0.6)
        tail = keep_chars - head
        return (
            text[:head]
            + "\n\n[... truncated automatically to fit model context window ...]\n\n"
            + text[-tail:]
        )

    working_query = query
    response_text = ""
    parsed_response = None
    max_attempts = 8
    # Keep generation budget configurable so local models with tighter context
    # windows can trade response length for higher success rate.
    max_output_tokens = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "512"))
    for attempt in range(1, max_attempts + 1):
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": working_query + "<json>"},
        ]
        try:
            response_text = chat_completion(
                messages=messages,
                temperature=temperature,
                # Keep outputs bounded so JSON replies are less likely to be cut.
                max_tokens=max_output_tokens,
            )
            try:
                parsed_response = json.loads(response_text)
                if isinstance(parsed_response, dict):
                    break
                print(
                    f"[inference] non-object JSON on attempt {attempt}/{max_attempts}; retrying."
                )
            except Exception as parse_err:
                print(
                    f"[inference] invalid JSON on attempt {attempt}/{max_attempts}: {parse_err}"
                )
            if attempt < max_attempts:
                continue
        except Exception as e:
            err_text = str(e)
            if _is_context_length_error(err_text):
                max_tokens, req_tokens = _extract_token_counts(err_text)
                if max_tokens and req_tokens and req_tokens > 0:
                    # Reserve room for completion tokens + small buffer.
                    ratio = max(0.25, (max_tokens - max_output_tokens - 128) / req_tokens)
                    target_chars = int(len(working_query) * ratio)
                else:
                    target_chars = int(len(working_query) * 0.7)
                target_chars = max(300, target_chars)
                next_query = _truncate_middle(working_query, target_chars)
                if next_query == working_query:
                    print(f"[inference] context overflow and cannot shrink further: {e}")
                    return {
                        "decision": "Action 1. context_overflow_fallback",
                        "explanation": f"Prompt exceeded model context window and could not be reduced further. Error: {err_text}",
                    }
                print(
                    f"[inference] context overflow on attempt {attempt}/{max_attempts}; "
                    f"shrinking prompt from {len(working_query)} to {len(next_query)} chars."
                )
                working_query = next_query
                continue

            print(e)
            sleep(10)
    else:
        return {
            "decision": "Action 1. inference_failed_fallback",
            "explanation": "Inference failed after max retries.",
        }
    if parsed_response is None:
        return {
            "decision": "Action 1. malformed_json_fallback",
            "explanation": "Model returned malformed JSON repeatedly.",
        }
    return parsed_response


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
