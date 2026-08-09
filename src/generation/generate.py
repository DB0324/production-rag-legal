"""
LLM call wrapper — provider-agnostic generation.
Supports: Google Gemini (default), OpenAI, or local Ollama.

Set provider via environment variable:
    LLM_PROVIDER=gemini     (default) — needs GOOGLE_API_KEY
    LLM_PROVIDER=openai     — needs OPENAI_API_KEY
    LLM_PROVIDER=ollama     — needs Ollama running locally

Set model name (optional):
    LLM_MODEL=gemini-2.0-flash          (default for gemini)
    LLM_MODEL=gpt-4o-mini               (default for openai)
    LLM_MODEL=llama3.1                   (default for ollama)
"""
import os
import time
import json
import re

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "gemini")
LLM_MODEL = os.environ.get("LLM_MODEL", None)

# Default models per provider
DEFAULT_MODELS = {
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
    "ollama": "llama3.1",
}


def _get_model_name():
    return LLM_MODEL or DEFAULT_MODELS.get(LLM_PROVIDER, "gemini-2.0-flash")


def _call_gemini(prompt: str, temperature: float = 0.1) -> dict:
    """Call Google Gemini API."""
    # pyrefly: ignore [missing-import]
    import google.generativeai as genai

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set. "
                         "Set it or switch to a different provider via LLM_PROVIDER.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_get_model_name())

    start = time.time()
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=1024,
        ),
    )
    latency = time.time() - start

    text = response.text
    # Estimate token counts (Gemini provides usage_metadata)
    usage = getattr(response, "usage_metadata", None)
    tokens_in = getattr(usage, "prompt_token_count", len(prompt.split()) * 1.3) if usage else len(prompt.split()) * 1.3
    tokens_out = getattr(usage, "candidates_token_count", len(text.split()) * 1.3) if usage else len(text.split()) * 1.3

    return {
        "text": text,
        "tokens_in": int(tokens_in),
        "tokens_out": int(tokens_out),
        "latency_s": round(latency, 3),
        "model": _get_model_name(),
        "provider": "gemini",
    }


def _call_openai(prompt: str, temperature: float = 0.1) -> dict:
    """Call OpenAI API."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    client = OpenAI(api_key=api_key)
    model = _get_model_name()

    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=1024,
    )
    latency = time.time() - start

    text = response.choices[0].message.content
    usage = response.usage

    return {
        "text": text,
        "tokens_in": usage.prompt_tokens,
        "tokens_out": usage.completion_tokens,
        "latency_s": round(latency, 3),
        "model": model,
        "provider": "openai",
    }


def _call_ollama(prompt: str, temperature: float = 0.1) -> dict:
    """Call local Ollama server."""
    import requests

    model = _get_model_name()
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")

    start = time.time()
    response = requests.post(
        f"{url}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
    )
    latency = time.time() - start

    data = response.json()
    text = data.get("response", "")

    return {
        "text": text,
        "tokens_in": data.get("prompt_eval_count", 0),
        "tokens_out": data.get("eval_count", 0),
        "latency_s": round(latency, 3),
        "model": model,
        "provider": "ollama",
    }


# Provider dispatch
_PROVIDERS = {
    "gemini": _call_gemini,
    "openai": _call_openai,
    "ollama": _call_ollama,
}


def generate(prompt: str, temperature: float = 0.1) -> dict:
    """
    Generate text from the configured LLM provider.

    Returns:
        {
            "text": str,           # Generated answer
            "tokens_in": int,      # Input tokens
            "tokens_out": int,     # Output tokens
            "latency_s": float,    # LLM call latency in seconds
            "model": str,          # Model name
            "provider": str,       # Provider name
        }
    """
    provider_fn = _PROVIDERS.get(LLM_PROVIDER)
    if provider_fn is None:
        raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. "
                         f"Choose from: {list(_PROVIDERS.keys())}")
    return provider_fn(prompt, temperature)


def extract_citations(answer_text: str) -> list:
    """
    Extract citation references from a generated answer.
    Looks for patterns like [Case Title, Doc ID] or [Doc ID].
    """
    # Match [anything inside brackets]
    citations = re.findall(r'\[([^\]]+)\]', answer_text)
    # Filter out common non-citation bracket content
    filtered = [c for c in citations if not c.isdigit() and len(c) > 3]
    return filtered
