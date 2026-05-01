"""Batch-judge (doc, type) items via OpenAI or Anthropic API.

Replaces the manual paste step from build_validate -> Judge LLM -> apply
chain. Loops over gt_allocation.json, for each annotated item assembles the
Judge prompt (Layer 1 + Layer 2 already enforced inside assemble_prompt),
calls the chosen API, then runs the cleaned array through the struct gate.

Cross-family rule (design v3). Annotator and Judge must be from different
model families relative to the Gemini retrieval backbone. Default annotator
is Anthropic Claude Sonnet 4.6, default judge is OpenAI gpt-5.

Per design v3 (3-type stratified): supported query types are factual,
paraphrased, multihop.

Usage:
    python scripts/gt/auto_judge.py --category UU --dry-run
    python scripts/gt/auto_judge.py --category UU
    python scripts/gt/auto_judge.py --doc-id uu-13-2025 --type multihop
    python scripts/gt/auto_judge.py --category UU --force
"""

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from scripts.gt.build_validate import assemble_prompt
from scripts.gt.apply_validation import apply_cleaned, extract_cleaned_array, raw_path_for
from scripts.gt.run_allocation import iter_plan, _state

ALLOCATION_FILE = Path("data/gt_allocation.json")
LOG_FILE = Path("data/gt_judge_log.json")
LOG_DIR = Path("data/gt_judge_logs")
QUERY_TYPES = ("factual", "paraphrased", "multihop")
DEFAULT_MAX_COST = 1.00
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42

PROVIDER_DEFAULTS = {
    "openai":    {"model": "gpt-5"},
    "anthropic": {"model": "claude-sonnet-4-6"},
}

# Default judge provider is OpenAI gpt-5 because the GT annotator defaults to
# Anthropic Claude Sonnet 4.6; cross-family judge stays per design v2.
DEFAULT_PROVIDER = "openai"

PRICING = {
    "gpt-5.5":              {"input":  5.00, "cached": 0.50,  "output": 30.00},
    "gpt-5":                {"input":  1.25, "cached": 0.125, "output": 10.00},
    "gpt-5-mini":           {"input":  0.50, "cached": 0.05,  "output":  2.00},
    "gpt-4.1":              {"input":  2.00, "cached": 0.50,  "output":  8.00},
    "gpt-4.1-mini":         {"input":  0.40, "cached": 0.10,  "output":  1.60},
    "gpt-4.1-nano":         {"input":  0.10, "cached": 0.025, "output":  0.40},
    "gpt-4o":               {"input":  2.50, "cached": 1.25,  "output": 10.00},
    "gpt-4o-mini":          {"input":  0.15, "cached": 0.075, "output":  0.60},
    "claude-sonnet-4-5":    {"input":  3.00, "cached": 0.30,  "output": 15.00},
    "claude-sonnet-4-6":    {"input":  3.00, "cached": 0.30,  "output": 15.00},
    "claude-opus-4-7":      {"input":  5.00, "cached": 0.50,  "output": 25.00},
    "claude-haiku-4-5":     {"input":  1.00, "cached": 0.10,  "output":  5.00},
}


def lookup_pricing(model: str) -> dict[str, float]:
    """Return per-1M-token pricing for a model alias or pinned snapshot."""
    if model in PRICING:
        return PRICING[model]
    for alias, price in PRICING.items():
        if model.startswith(alias + "-") or model.startswith(alias):
            return price
    raise SystemExit(f"Unknown model '{model}'. Add it to PRICING in auto_judge.py.")


def estimate_cost(prompt: str, n_items: int, pricing: dict[str, float]) -> float:
    """Estimate one judge call cost, ~600 tokens output per item."""
    in_tok = max(1, len(prompt) // 4)
    out_tok = n_items * 600 + 400
    return (in_tok / 1_000_000) * pricing["input"] + (out_tok / 1_000_000) * pricing["output"]


def actual_cost_openai(usage: dict, pricing: dict[str, float]) -> float:
    """Cost from OpenAI usage object."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    fresh = max(0, prompt_tokens - cached)
    completion = usage.get("completion_tokens", 0)
    return (
        (fresh / 1_000_000) * pricing["input"]
        + (cached / 1_000_000) * pricing["cached"]
        + (completion / 1_000_000) * pricing["output"]
    )


def actual_cost_anthropic(usage: dict, pricing: dict[str, float]) -> float:
    """Cost from Anthropic usage object (input_tokens, output_tokens, cache_*)."""
    fresh = usage.get("input_tokens", 0)
    cached_read = usage.get("cache_read_input_tokens", 0)
    cached_write = usage.get("cache_creation_input_tokens", 0)
    output = usage.get("output_tokens", 0)
    return (
        (fresh / 1_000_000) * pricing["input"]
        + (cached_read / 1_000_000) * pricing["cached"]
        + (cached_write / 1_000_000) * pricing["input"] * 1.25
        + (output / 1_000_000) * pricing["output"]
    )


def call_openai(client, model: str, prompt: str, temperature: float, seed: int) -> tuple[str, dict]:
    """Single OpenAI Chat Completions call."""
    kwargs: dict = {"model": model, "messages": [{"role": "user", "content": prompt}]}
    try:
        resp = client.chat.completions.create(temperature=temperature, seed=seed, **kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "temperature" in msg or "seed" in msg or "unsupported" in msg:
            resp = client.chat.completions.create(**kwargs)
        else:
            raise
    text = resp.choices[0].message.content or ""
    usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    return text, usage


def call_anthropic(client, model: str, prompt: str, temperature: float) -> tuple[str, dict]:
    """Single Anthropic Messages call."""
    resp = client.messages.create(
        model=model,
        max_tokens=8000,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = []
    for block in resp.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    text = "".join(parts)
    usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    return text, usage


def append_log(entry: dict) -> None:
    """Append one entry to the master judge log."""
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    if LOG_FILE.exists():
        try:
            existing = json.loads(LOG_FILE.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except json.JSONDecodeError:
            existing = []
    else:
        existing = []
    existing.append(entry)
    LOG_FILE.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_call_detail(stamp: str, doc_id: str, query_type: str, payload: dict) -> Path:
    """Write per-call full payload for replay/debug."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{stamp}_{doc_id}__{query_type}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def judge_one(client, provider: str, model: str, pricing: dict, cat: str, doc_id: str,
              query_type: str, dry_run: bool, temperature: float, seed: int) -> tuple[float, int, str]:
    """Build judge prompt, call API, apply gate. Return (cost, n_items, status)."""
    out_path, items = assemble_prompt(doc_id, 600, query_type=query_type, skip_layer2=False)
    prompt = out_path.read_text(encoding="utf-8")
    n_items_in = len(items)

    if dry_run:
        est = estimate_cost(prompt, n_items_in, pricing)
        return est, 0, f"dry-run estimate ${est:.4f}"

    started = dt.datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    if provider == "openai":
        text, usage = call_openai(client, model, prompt, temperature, seed)
        cost = actual_cost_openai(usage, pricing)
    else:
        text, usage = call_anthropic(client, model, prompt, temperature)
        cost = actual_cost_anthropic(usage, pricing)
    elapsed = time.time() - t0

    base_log = {
        "doc_id": doc_id,
        "query_type": query_type,
        "category": cat,
        "provider": provider,
        "model": model,
        "n_items_in": n_items_in,
        "started_at": started.isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 2),
        "temperature": temperature,
        "seed": seed if provider == "openai" else None,
        "usage": usage,
        "cost_usd": round(cost, 6),
    }

    try:
        cleaned = extract_cleaned_array(text)
    except SystemExit as e:
        detail = write_call_detail(stamp, doc_id, query_type, {**base_log, "status": "extract_error", "error": str(e), "prompt": prompt, "response": text})
        append_log({**base_log, "status": "extract_error", "error": str(e), "n_items_out": 0, "detail_path": str(detail)})
        return cost, 0, f"extract failed ({e}), full payload at {detail}"

    try:
        apply_cleaned(doc_id, cleaned, dry_run=False, query_type=query_type)
    except SystemExit as e:
        detail = write_call_detail(stamp, doc_id, query_type, {**base_log, "status": "apply_error", "error": str(e), "prompt": prompt, "response": text, "cleaned": cleaned})
        append_log({**base_log, "status": "apply_error", "error": str(e), "n_items_out": len(cleaned), "detail_path": str(detail)})
        return cost, 0, f"apply gate rejected, full payload at {detail}"

    detail = write_call_detail(stamp, doc_id, query_type, {**base_log, "status": "ok", "n_items_out": len(cleaned), "prompt": prompt, "response": text, "cleaned": cleaned})
    append_log({**base_log, "status": "ok", "n_items_out": len(cleaned), "detail_path": str(detail)})
    target = raw_path_for(doc_id, query_type)
    return cost, len(cleaned), f"wrote {len(cleaned)} item(s) -> {target}"


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--doc-id", type=str, default=None,
                    help="Single doc_id, requires --type")
    ap.add_argument("--type", "-t", type=str, default=None,
                    choices=list(QUERY_TYPES))
    ap.add_argument("--provider", type=str, default=DEFAULT_PROVIDER, choices=["openai", "anthropic"],
                    help=f"LLM provider (default {DEFAULT_PROVIDER})")
    ap.add_argument("--model", type=str, default=None,
                    help="Model id (defaults to provider's recommended judge model)")
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST,
                    help=f"Hard stop if estimated batch cost exceeds this (USD, default {DEFAULT_MAX_COST})")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"Sampling temperature (default {DEFAULT_TEMPERATURE})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"OpenAI seed (default {DEFAULT_SEED}, ignored by Anthropic)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview cost only, do not call API")
    ap.add_argument("--force", action="store_true",
                    help="Re-judge items already in 'applied' state")
    args = ap.parse_args()

    if args.doc_id and not args.type:
        ap.error("--doc-id requires --type")

    if not args.model:
        args.model = PROVIDER_DEFAULTS[args.provider]["model"]

    if not ALLOCATION_FILE.exists():
        raise SystemExit(f"{ALLOCATION_FILE} not found")
    allocation = json.loads(ALLOCATION_FILE.read_text(encoding="utf-8"))

    pricing = lookup_pricing(args.model)
    print(f"Provider    : {args.provider}")
    print(f"Model       : {args.model}")
    print(f"Pricing     : ${pricing['input']}/${pricing['cached']}/${pricing['output']} per 1M (input/cached/output)")
    print(f"Temperature : {args.temperature}")
    print(f"Max cost    : ${args.max_cost}")
    print(f"Log file    : {LOG_FILE}")
    print(f"Dry run     : {args.dry_run}")
    print()

    # Build work list. 'annotated' and 'built' both mean raw is bare JSON
    # ready for Judge. 'applied' only re-runs with --force. Everything else
    # (not-annotated, judged) is skipped.
    valid_states = {"annotated", "built"}
    if args.force:
        valid_states.add("applied")
    work = []
    for cat, doc_id, qt, count in iter_plan(allocation, args.category, args.type):
        if args.doc_id and doc_id != args.doc_id:
            continue
        st = _state(doc_id, qt)
        if st not in valid_states:
            print(f"  skip  {doc_id}  type={qt}  (state={st})")
            continue
        work.append((cat, doc_id, qt, count))

    if not work:
        print("Nothing to do.")
        return

    print(f"\nEstimating cost for {len(work)} item(s)...")
    estimates: list[float] = []
    for cat, doc_id, qt, n in work:
        try:
            out_path, items = assemble_prompt(doc_id, 600, query_type=qt, skip_layer2=False)
            prompt = out_path.read_text(encoding="utf-8")
            est = estimate_cost(prompt, len(items), pricing)
            estimates.append(est)
        except SystemExit as e:
            print(f"  SKIP  {doc_id}  type={qt}  ({e})")
            estimates.append(0.0)
    total_est = sum(estimates)
    print(f"Estimated total, ${total_est:.4f}")

    if total_est > args.max_cost:
        raise SystemExit(
            f"\nEstimated cost ${total_est:.2f} exceeds --max-cost ${args.max_cost}."
        )

    if args.dry_run:
        print("\n[dry-run] not calling API")
        return

    if args.provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY not set in env")
        try:
            from openai import OpenAI
        except ImportError:
            raise SystemExit("openai not installed, pip install 'openai>=1.50'")
        client = OpenAI()
    else:
        if not os.getenv("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set in env")
        try:
            from anthropic import Anthropic
        except ImportError:
            raise SystemExit("anthropic not installed, pip install 'anthropic>=0.40'")
        client = Anthropic()

    print(f"\nRunning {len(work)} judge call(s)...")
    total_cost = 0.0
    n_ok = n_fail = 0
    for cat, doc_id, qt, _ in work:
        try:
            cost, n_items, status = judge_one(
                client, args.provider, args.model, pricing, cat, doc_id, qt,
                dry_run=False, temperature=args.temperature, seed=args.seed,
            )
            total_cost += cost
            if n_items > 0:
                n_ok += 1
                print(f"  ok    {doc_id}  type={qt}  ${cost:.4f}  {status}")
            else:
                n_fail += 1
                print(f"  FAIL  {doc_id}  type={qt}  ${cost:.4f}  {status}")
        except SystemExit as e:
            n_fail += 1
            print(f"  FAIL  {doc_id}  type={qt}  ({e})")
        except Exception as e:
            n_fail += 1
            print(f"  FAIL  {doc_id}  type={qt}  ({type(e).__name__}, {e})")

    print(f"\nJudged {n_ok}, failed {n_fail}, total cost ${total_cost:.4f}")
    if n_ok:
        print()
        print("Next.")
        print(f"  1. python scripts/gt/log_review.py <doc> --type <type>  (per item, manual)")
        print(f"  2. python scripts/gt/collect.py && python scripts/gt/finalize.py")


if __name__ == "__main__":
    main()
