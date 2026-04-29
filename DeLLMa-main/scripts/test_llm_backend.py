#!/usr/bin/env python3
"""
scripts/test_llm_backend.py
---------------------------
Smoke test: send one minimal request to the configured LLM backend
and print the response.

Usage (from DeLLMa-main/):
    # Test OpenAI (default):
    python scripts/test_llm_backend.py

    # Test vLLM with Llama:
    LLM_BACKEND=vllm DEFAULT_MODEL=meta-llama/Llama-3.1-8B-Instruct \\
        VLLM_BASE_URL=http://localhost:8000/v1 \\
        python scripts/test_llm_backend.py

    # Test vLLM with Mistral:
    LLM_BACKEND=vllm DEFAULT_MODEL=mistralai/Mistral-7B-Instruct-v0.3 \\
        python scripts/test_llm_backend.py

    # Test vLLM with Gemma:
    LLM_BACKEND=vllm DEFAULT_MODEL=google/gemma-2-9b-it \\
        python scripts/test_llm_backend.py
"""

import os
import sys
import json

# Make sure the project root is on the path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _ROOT)

# Load .env
from utils.prompt_utils import _load_env_manual, _ENV_PATH
_load_env_manual(_ENV_PATH)
try:
    from dotenv import load_dotenv
    load_dotenv(_ENV_PATH)
except ImportError:
    pass

from utils.llm_client import get_llm_client, get_model_name, chat_completion, is_client_available

print("=" * 60)
print("DeLLMa LLM Backend Smoke Test")
print("=" * 60)

if not is_client_available():
    print("[ERROR] No LLM backend is configured.")
    print("  For OpenAI: set OPENAI_API_KEY in .env or environment.")
    print("  For vLLM:   set LLM_BACKEND=vllm and start the server.")
    sys.exit(1)

client = get_llm_client()
model = get_model_name()

print(f"\nSending one minimal request ...")
print(f"  Model: {model}\n")

messages = [
    {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON."},
    {"role": "user", "content": 'Respond with a JSON object with one key "answer" whose value is the string "ok".'},
]

try:
    response_text = chat_completion(messages=messages, temperature=0.0)
    print("Raw response:")
    print(response_text)
    try:
        parsed = json.loads(response_text)
        print("\nParsed JSON:")
        print(json.dumps(parsed, indent=2))
        print("\n[PASS] Backend is working correctly.")
    except json.JSONDecodeError:
        print("\n[WARN] Response is not valid JSON, but a response was received.")
        print("       This may happen with models that do not support JSON mode.")
        print("       Set LLM_JSON_MODE=false in your .env if needed.")
except Exception as e:
    print(f"\n[FAIL] Request failed: {e}")
    sys.exit(1)
