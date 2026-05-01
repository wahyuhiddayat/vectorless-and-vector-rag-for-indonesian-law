"""Shared OpenAI client + token accounting for indexing and retrieval."""

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

_client = None
_input_tokens = 0
_output_tokens = 0
_calls = 0
_lock = threading.Lock()


def client():
    """Lazy-init OpenAI client with a 5min HTTP deadline."""
    global _client
    if _client is None:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: OPENAI_API_KEY not set.")
            sys.exit(1)
        _client = OpenAI(api_key=api_key, timeout=300.0, max_retries=0)
    return _client


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
)


def _supports_reasoning(model: str) -> bool:
    """Return True if the model accepts reasoning_effort parameter."""
    return model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")


def call(prompt: str, *, model: str = MODEL, max_retries: int = 8,
         max_completion_tokens: int = 16384,
         return_usage: bool = False) -> dict | tuple[dict, dict]:
    """Send prompt to OpenAI, return parsed JSON. Retries transient errors and non-JSON responses.

    With return_usage=True, returns (json_dict, usage_dict) for caller-local accounting.
    Uses response_format={'type':'json_object'} so the response is always JSON.
    """
    cli = client()

    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": max_completion_tokens,
    }
    # gpt-5 family supports reasoning_effort; "minimal" is the cheapest tier
    # (analog of Gemini thinking_budget=0). gpt-4.1 family does not accept it.
    if _supports_reasoning(model):
        kwargs["reasoning_effort"] = "minimal"
    else:
        # gpt-4.1 family supports temperature; reasoning models reject it.
        kwargs["temperature"] = 0.0

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = cli.chat.completions.create(**kwargs)

            usage = resp.usage
            input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            output_tokens = getattr(usage, "completion_tokens", 0) or 0
            _track(input_tokens, output_tokens)

            text = (resp.choices[0].message.content or "").strip()
            # Strip markdown fences in case the model wraps despite json_object mode.
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
                        f"  OpenAI returned non-JSON (attempt {attempt+1}), retrying in {wait:.0f}s...\n"
                    )
                    time.sleep(wait)
                    continue
                raise

            if return_usage:
                return parsed, {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
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
                    f"  OpenAI error (attempt {attempt+1}): {e!r} - retrying in {wait:.0f}s...\n"
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"call failed after {max_retries} attempts") from last_exc
