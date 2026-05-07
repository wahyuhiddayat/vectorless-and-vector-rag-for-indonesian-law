"""Multi-vendor LLM client + token accounting for indexing and retrieval.

Routes by model name prefix:
  gpt-*, o1, o3, o4   -> OpenAI Chat Completions
  claude-*            -> Anthropic Messages API
  gemini-*            -> Google GenAI on Vertex AI
  deepseek-*          -> DeepSeek (OpenAI-compatible) Chat Completions

Public surface (unchanged from prior single-vendor impl):
  client(), call(prompt, model, ...), get_stats(), reset_counters(),
  snapshot_counters(), step_metrics(t_start, snap_before)

JSON-mode is enforced where the backend supports it. For Anthropic the
prompt itself must request JSON output; the retry loop catches parse
failures and retries.
"""

import json
import os
import random
import sys
import threading
import time

from dotenv import load_dotenv

from .models import RETRIEVAL_MODEL

load_dotenv()

MODEL = RETRIEVAL_MODEL

_openai_cache = None
_anthropic_cache = None
_vertex_cache = None
_deepseek_cache = None
_input_tokens = 0
_output_tokens = 0
_calls = 0
_lock = threading.Lock()


def _openai_client():
    """Lazy-init OpenAI client."""
    global _openai_cache
    if _openai_cache is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(1)
        _openai_cache = OpenAI(api_key=api_key, timeout=300.0, max_retries=0)
    return _openai_cache


def _anthropic_client():
    """Lazy-init Anthropic client."""
    global _anthropic_cache
    if _anthropic_cache is None:
        from anthropic import Anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: ANTHROPIC_API_KEY not set.")
            sys.exit(1)
        _anthropic_cache = Anthropic(api_key=api_key, timeout=300.0, max_retries=0)
    return _anthropic_cache


def _deepseek_client():
    """Lazy-init DeepSeek client (OpenAI-compatible)."""
    global _deepseek_cache
    if _deepseek_cache is None:
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            print("ERROR: DEEPSEEK_API_KEY not set.")
            sys.exit(1)
        _deepseek_cache = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            timeout=300.0,
            max_retries=0,
        )
    return _deepseek_cache


def _vertex_client():
    """Lazy-init Google GenAI client on Vertex AI (uses ADC)."""
    global _vertex_cache
    if _vertex_cache is None:
        from google import genai
        from google.genai import types as gtypes
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            print("ERROR: GOOGLE_CLOUD_PROJECT not set.")
            sys.exit(1)
        _vertex_cache = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=gtypes.HttpOptions(api_version="v1", timeout=300_000),
        )
    return _vertex_cache


def client():
    """Return the OpenAI client.

    Kept for backward compatibility with callers that need a raw client
    handle (e.g. preflight smoke tests). Prefer call() for normal use.
    """
    return _openai_client()


def _backend(model: str) -> str:
    """Return the backend identifier for a given model name."""
    if model.startswith("gpt-") or model.startswith(("o1", "o3", "o4")):
        return "openai"
    if model.startswith("claude-"):
        return "anthropic"
    if model.startswith("gemini-"):
        return "vertex"
    if model.startswith("deepseek-"):
        return "deepseek"
    raise ValueError(f"Unknown model family: {model!r}")


def _supports_openai_reasoning(model: str) -> bool:
    """OpenAI reasoning models accept reasoning_effort instead of temperature."""
    return model.startswith("gpt-5") or model.startswith(("o1", "o3", "o4"))


def _track(input_tokens: int, output_tokens: int) -> None:
    global _input_tokens, _output_tokens, _calls
    with _lock:
        _input_tokens += input_tokens or 0
        _output_tokens += output_tokens or 0
        _calls += 1


def reset_counters() -> None:
    global _input_tokens, _output_tokens, _calls
    with _lock:
        _input_tokens = 0
        _output_tokens = 0
        _calls = 0


def get_stats() -> dict:
    return {
        "llm_calls": _calls,
        "input_tokens": _input_tokens,
        "output_tokens": _output_tokens,
        "total_tokens": _input_tokens + _output_tokens,
    }


def snapshot_counters() -> dict:
    return {
        "llm_calls": _calls,
        "input_tokens": _input_tokens,
        "output_tokens": _output_tokens,
    }


def step_metrics(t_start: float, snap_before: dict) -> dict:
    snap_after = snapshot_counters()
    return {
        "elapsed_s": round(time.time() - t_start, 3),
        "llm_calls": snap_after["llm_calls"] - snap_before["llm_calls"],
        "input_tokens": snap_after["input_tokens"] - snap_before["input_tokens"],
        "output_tokens": snap_after["output_tokens"] - snap_before["output_tokens"],
    }


_RETRYABLE_TOKENS = (
    "rate", "429", "503", "500", "502", "504",
    "quota", "resource_exhausted", "insufficient_quota",
    "rate_limit_exceeded", "server_error", "service unavailable",
    "overloaded", "timeout", "timed out", "connection",
    "apiconnectionerror", "apitimeouterror", "internalservererror",
    "deadline_exceeded", "unavailable",
)


def _call_openai(prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """One OpenAI Chat Completions call. Returns (text, input_tokens, output_tokens)."""
    cli = _openai_client()
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_tokens,
    }
    if _supports_openai_reasoning(model):
        kwargs["reasoning_effort"] = "minimal"
    else:
        kwargs["temperature"] = 0.0

    resp = cli.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
    out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
    return text, in_tok, out_tok


def _call_anthropic(prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """One Anthropic Messages call. Returns (text, input_tokens, output_tokens).

    Anthropic has no native JSON-mode; rely on prompt to request JSON and
    on the outer retry loop to recover from parse failures.
    """
    cli = _anthropic_client()
    resp = cli.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    text = (resp.content[0].text or "").strip() if resp.content else ""
    in_tok = getattr(resp.usage, "input_tokens", 0) or 0
    out_tok = getattr(resp.usage, "output_tokens", 0) or 0
    return text, in_tok, out_tok


def _call_deepseek(prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """One DeepSeek Chat Completions call. Returns (text, input_tokens, output_tokens).

    Uses OpenAI-compatible endpoint at api.deepseek.com. JSON output mode is
    supported for both v4-flash and v4-pro. Thinking is disabled here to
    mirror gpt-5's `reasoning_effort="minimal"` for structured-extraction
    callers (parser, judge): empirically the V4 default thinking=on bloats
    output ~5x without quality lift on prompt-driven JSON tasks.
    """
    cli = _deepseek_client()
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "extra_body": {"thinking": {"type": "disabled"}},
    }
    resp = cli.chat.completions.create(**kwargs)
    text = (resp.choices[0].message.content or "").strip()
    in_tok = getattr(resp.usage, "prompt_tokens", 0) or 0
    out_tok = getattr(resp.usage, "completion_tokens", 0) or 0
    return text, in_tok, out_tok


def _call_vertex(prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """One Vertex AI Gemini call. Returns (text, input_tokens, output_tokens)."""
    cli = _vertex_client()
    from google.genai import types as gtypes
    cfg_kwargs: dict = {
        "temperature": 0.0,
        "max_output_tokens": max_tokens,
        "response_mime_type": "application/json",
    }
    if model.startswith("gemini-2.5-flash"):
        cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)

    resp = cli.models.generate_content(
        model=model,
        contents=prompt,
        config=gtypes.GenerateContentConfig(**cfg_kwargs),
    )
    text = (getattr(resp, "text", "") or "").strip()
    meta = getattr(resp, "usage_metadata", None)
    in_tok = getattr(meta, "prompt_token_count", 0) or 0 if meta else 0
    out_tok = getattr(meta, "candidates_token_count", 0) or 0 if meta else 0
    return text, in_tok, out_tok


def _call_backend(prompt: str, model: str, max_tokens: int) -> tuple[str, int, int]:
    """Dispatch to the right backend. Returns (text, input_tokens, output_tokens)."""
    backend = _backend(model)
    if backend == "openai":
        return _call_openai(prompt, model, max_tokens)
    if backend == "anthropic":
        return _call_anthropic(prompt, model, max_tokens)
    if backend == "vertex":
        return _call_vertex(prompt, model, max_tokens)
    if backend == "deepseek":
        return _call_deepseek(prompt, model, max_tokens)
    raise ValueError(f"Unknown backend: {backend}")


def call(prompt: str, *, model: str = MODEL, max_retries: int = 8,
         max_completion_tokens: int = 16384,
         return_usage: bool = False) -> dict | tuple[dict, dict]:
    """Send prompt to the configured LLM, return parsed JSON.

    Routes by model prefix to OpenAI, Anthropic, or Vertex Gemini. Retries
    transient errors and non-JSON responses up to max_retries times with
    exponential backoff.

    With return_usage=True, returns (json_dict, usage_dict) for caller-local
    accounting.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            text, in_tok, out_tok = _call_backend(prompt, model, max_completion_tokens)
            _track(in_tok, out_tok)

            # Strip markdown fences in case the model wraps despite JSON mode.
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

            try:
                parsed = json.loads(text)
            except json.JSONDecodeError as json_err:
                last_exc = json_err
                if attempt < max_retries - 1:
                    wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 5)
                    sys.stderr.write(
                        f"  LLM returned non-JSON (attempt {attempt+1}, model={model}), retrying in {wait:.0f}s...\n"
                    )
                    time.sleep(wait)
                    continue
                raise

            if return_usage:
                return parsed, {
                    "input_tokens": in_tok,
                    "output_tokens": out_tok,
                    "calls": 1,
                }
            return parsed

        except json.JSONDecodeError:
            raise
        except Exception as e:
            last_exc = e
            err = str(e).lower()
            if any(tok in err for tok in _RETRYABLE_TOKENS) and attempt < max_retries - 1:
                wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 5)
                sys.stderr.write(
                    f"  LLM error (attempt {attempt+1}, model={model}): {e!r} - retrying in {wait:.0f}s...\n"
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"call failed after {max_retries} attempts (model={model})") from last_exc
