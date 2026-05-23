"""
utils/llm_client.py
-------------------
Thin backend-agnostic LLM client for DeLLMa.

Supports:
  - OpenAI hosted API  (LLM_BACKEND=openai, default)
  - vLLM local server  (LLM_BACKEND=vllm)  — uses the OpenAI-compatible endpoint

Configuration (environment variables / .env):
  LLM_BACKEND          openai | vllm                 (default: openai)
  OPENAI_API_KEY       your OpenAI key               (required for openai backend)
  OPENAI_MODEL         model name for OpenAI         (default: gpt-4o)
  OPENAI_BASE_URL      override OpenAI base URL      (optional)
  VLLM_BASE_URL        vLLM server URL               (default: http://localhost:8000/v1)
  VLLM_API_KEY         vLLM auth token               (default: token-abc123)
  DEFAULT_MODEL        override model for any backend (optional)
  LLM_JSON_MODE        true | false — use response_format=json_object (default: true)

Supported local models (set DEFAULT_MODEL or pass model= to chat_completion):
  meta-llama/Llama-3.1-8B-Instruct
  mistralai/Mistral-7B-Instruct-v0.3
  google/gemma-2-9b-it
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from openai import OpenAI

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_client: Optional[OpenAI] = None
_logged: bool = False  # print backend info only once per process


def _env(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _backend() -> str:
    return _env("LLM_BACKEND", "openai").lower()


def get_model_name() -> str:
    """Return the model name that will be used for chat completions."""
    override = _env("DEFAULT_MODEL")
    if override:
        return override
    if _backend() == "vllm":
        # sensible default local model; user should override via DEFAULT_MODEL
        return "meta-llama/Llama-3.1-8B-Instruct"
    return _env("OPENAI_MODEL", "gpt-4o") or "gpt-4o"


def json_mode_enabled() -> bool:
    """Return True if response_format=json_object should be requested."""
    val = _env("LLM_JSON_MODE", "true").lower()
    return val not in ("0", "false", "no", "off")


def get_llm_client() -> Optional[OpenAI]:
    """
    Return a configured OpenAI SDK client pointed at the selected backend.
    Returns None when neither OpenAI key nor vLLM backend is configured.
    """
    global _client, _logged

    backend = _backend()

    if backend == "vllm":
        base_url = _env("VLLM_BASE_URL", "http://localhost:8000/v1")
        api_key = _env("VLLM_API_KEY", "token-abc123")
        model = get_model_name()
        if _client is None:
            _client = OpenAI(base_url=base_url, api_key=api_key)
        if not _logged:
            logger.info(
                "[LLM] backend=vllm  model=%s  base_url=%s", model, base_url
            )
            print(f"[LLM] backend=vllm | model={model} | base_url={base_url}")
            _logged = True
        return _client

    # default: openai
    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        return None
    base_url = _env("OPENAI_BASE_URL") or None  # None → SDK default
    model = get_model_name()
    if _client is None:
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        _client = OpenAI(**kwargs)
    if not _logged:
        url_display = base_url or "https://api.openai.com/v1"
        logger.info(
            "[LLM] backend=openai  model=%s  base_url=%s", model, url_display
        )
        print(f"[LLM] backend=openai | model={model} | base_url={url_display}")
        _logged = True
    return _client


def _vllm_server_reachable() -> bool:
    """Probe the vLLM OpenAI-compatible /models endpoint."""
    base_url = _env("VLLM_BASE_URL", "http://localhost:8000/v1").rstrip("/")
    api_key = _env("VLLM_API_KEY", "token-abc123")
    req = Request(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (URLError, OSError, ValueError):
        return False


def is_client_available() -> bool:
    """True when a usable LLM client can be constructed."""
    if _backend() == "vllm":
        return _vllm_server_reachable()
    return bool(_env("OPENAI_API_KEY"))


def chat_completion(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    use_json_mode: Optional[bool] = None,
    **kwargs: Any,
) -> str:
    """
    Send a chat completion request and return the raw response string.

    Args:
        messages:       List of {"role": ..., "content": ...} dicts.
        model:          Override the model name (defaults to get_model_name()).
        temperature:    Sampling temperature.
        max_tokens:     Max tokens to generate (None = backend default).
        use_json_mode:  Force or suppress JSON mode regardless of env setting.
        **kwargs:       Additional kwargs forwarded to chat.completions.create.

    Returns:
        The response content string (or "" on failure).

    Raises:
        RuntimeError if no client is available.
    """
    client = get_llm_client()
    if client is None:
        raise RuntimeError(
            "No LLM client available. Set OPENAI_API_KEY or LLM_BACKEND=vllm."
        )

    _model = model or get_model_name()
    _json_mode = json_mode_enabled() if use_json_mode is None else use_json_mode

    create_kwargs: Dict[str, Any] = {
        "model": _model,
        "messages": messages,
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        create_kwargs["max_tokens"] = max_tokens
    if _json_mode:
        create_kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**create_kwargs)
    return response.choices[0].message.content or ""
