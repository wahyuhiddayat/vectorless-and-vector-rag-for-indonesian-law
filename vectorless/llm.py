"""Shared Gemini client + token accounting for indexing and retrieval."""

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
    """Lazy-init Gemini client with a 120s HTTP deadline."""
    global _client
    if _client is None:
        from google import genai
        from google.genai import types as gtypes
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set.")
            sys.exit(1)
        _client = genai.Client(
            api_key=api_key,
            http_options=gtypes.HttpOptions(timeout=300_000),
        )
    return _client


def _track(usage) -> None:
    global _input_tokens, _output_tokens, _calls
    with _lock:
        _input_tokens += usage.prompt_token_count or 0
        _output_tokens += usage.candidates_token_count or 0
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
    "rate", "429", "503", "500", "quota", "resource_exhausted",
    "deadline_exceeded", "service unavailable", "overloaded",
    "timeout", "timed out", "connection",
)


def call(prompt: str, *, model: str = MODEL, max_retries: int = 3,
         return_usage: bool = False) -> dict | tuple[dict, dict]:
    """Send prompt; return parsed JSON. Retries transient errors and non-JSON responses.

    With return_usage=True, returns (json_dict, usage_dict) for caller-local accounting.
    """
    from google.genai import types as gtypes
    cli = client()

    cfg_kwargs: dict = {}
    # gemini-2.5.x supports thinking_budget=0 to skip the thinking phase; 3.x rejects it.
    if model.startswith("gemini-2.5"):
        cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = cli.models.generate_content(
                model=model,
                contents=prompt,
                config=gtypes.GenerateContentConfig(**cfg_kwargs),
            )
            _track(resp.usage_metadata)

            text = resp.text.strip()
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
                        f"  Gemini returned non-JSON (attempt {attempt+1}), retrying in {wait:.0f}s...\n"
                    )
                    time.sleep(wait)
                    continue
                raise

            if return_usage:
                u = resp.usage_metadata
                return parsed, {
                    "input_tokens": u.prompt_token_count or 0,
                    "output_tokens": u.candidates_token_count or 0,
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
                    f"  Gemini error (attempt {attempt+1}): {e!r} — retrying in {wait:.0f}s...\n"
                )
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"call failed after {max_retries} attempts") from last_exc
