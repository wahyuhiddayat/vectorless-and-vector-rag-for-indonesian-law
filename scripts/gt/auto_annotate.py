"""Batch-annotate (doc, type) items via OpenAI API.

Replaces the manual paste step from prompt.py for the annotator stage. Loops
over gt_allocation.json, calls the OpenAI Chat Completions API per item with
the same prompt that prompt.py emits, and writes the JSON array straight to
the raw GT file. Judge step stays manual via Copilot/IDE for cross-family
diversity per design v2.

Resume-aware. Skips items whose raw file is already non-empty. State after
this script runs equals "annotated" in run_allocation, so build_validate.py
takes over from there.

Usage:
    python scripts/gt/auto_annotate.py --category UU --dry-run
    python scripts/gt/auto_annotate.py --category UU
    python scripts/gt/auto_annotate.py --doc-id uu-3-2025 --type factual
    python scripts/gt/auto_annotate.py --category UU --model gpt-5.5-2026-04-23
    python scripts/gt/auto_annotate.py --category UU --max-cost 5
"""

import argparse
import datetime as dt
import json
import os
import re
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

from vectorless.ids import doc_category
from scripts.gt.prompt import (
    build_prompt_parts,
    collect_leaf_nodes,
    compute_adaptive_n,
    filter_preamble,
    find_doc,
    raw_filename,
    write_meta_sidecar,
)
from scripts.gt.run_allocation import iter_plan, _state, _raw_path
from scripts.gt.apply_validation import _normalize_schema

ALLOCATION_FILE = Path("data/gt_allocation.json")
RAW_DIR = Path("data/ground_truth_raw")
LOG_FILE = Path("data/gt_annotate_log.json")
LOG_DIR = Path("data/gt_annotate_logs")
DEFAULT_MODEL = "gpt-5.5"
DEFAULT_MAX_COST = 1.00
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = 42
# API context windows are far larger than the manual paste budget (45K).
# 500K chars ~ 125K tokens, fits all corpus docs in a single call including
# crossdoc pairs. Bump higher only if a doc exceeds this.
DEFAULT_CHAR_BUDGET = 500_000

# Pricing per 1M tokens, USD. Update if OpenAI repricing.
PRICING = {
    "gpt-5.5":      {"input": 5.00, "cached": 0.50, "output": 30.00},
    "gpt-5":        {"input": 2.50, "cached": 0.25, "output": 10.00},
    "gpt-5-mini":   {"input": 0.50, "cached": 0.05, "output":  2.00},
    "gpt-4o":       {"input": 2.50, "cached": 1.25, "output": 10.00},
    "gpt-4o-mini":  {"input": 0.15, "cached": 0.075, "output": 0.60},
}


def lookup_pricing(model: str) -> dict[str, float]:
    """Return per-1M-token pricing for a model alias or pinned snapshot.

    Falls back to gpt-5.5 family pricing when the snapshot starts with the
    alias (e.g. gpt-5.5-2026-04-23).
    """
    if model in PRICING:
        return PRICING[model]
    for alias, price in PRICING.items():
        if model.startswith(alias + "-"):
            return price
    raise SystemExit(f"Unknown model '{model}'. Add it to PRICING in auto_annotate.py.")


def estimate_input_tokens(prompt: str) -> int:
    """Rough token estimate, 4 chars per token (English+Indonesian mix)."""
    return max(1, len(prompt) // 4)


def estimate_output_tokens(n_questions: int) -> int:
    """Rough output token budget per item, 600 tokens covers JSON + answer_hint."""
    return n_questions * 600


def estimate_cost(prompt: str, n_questions: int, pricing: dict[str, float]) -> float:
    """Estimate cost for one annotator call, no caching assumed."""
    in_tok = estimate_input_tokens(prompt)
    out_tok = estimate_output_tokens(n_questions)
    return (in_tok / 1_000_000) * pricing["input"] + (out_tok / 1_000_000) * pricing["output"]


def actual_cost(usage: dict, pricing: dict[str, float]) -> float:
    """Compute cost from real OpenAI usage object, accounting for cached prefix."""
    prompt_tokens = usage.get("prompt_tokens", 0)
    cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
    fresh = max(0, prompt_tokens - cached)
    completion = usage.get("completion_tokens", 0)
    return (
        (fresh / 1_000_000) * pricing["input"]
        + (cached / 1_000_000) * pricing["cached"]
        + (completion / 1_000_000) * pricing["output"]
    )


def extract_json_array(text: str) -> list[dict]:
    """Pull the first JSON array out of an LLM response, tolerating fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        end = next((i for i in range(1, len(lines)) if lines[i].strip().startswith("```")), len(lines))
        text = "\n".join(lines[1:end])
    match = re.search(r"\[\s*\{", text)
    if not match:
        raise ValueError("No JSON array found in response")
    start = match.start()
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(text[start:], start=start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                payload = text[start:i + 1]
                data = json.loads(payload)
                if not isinstance(data, list):
                    raise ValueError("Top-level JSON is not an array")
                return data
    raise ValueError("Unbalanced JSON array")


def build_one_prompt(doc_id: str, query_type: str, n_questions: int,
                     paired_doc_id: str | None,
                     char_budget: int = DEFAULT_CHAR_BUDGET) -> tuple[str, int]:
    """Resolve doc, build the annotator prompt, return (prompt_text, n_used).

    Raises SystemExit with a clear message on conditions that block API
    annotation, e.g. doc not found, doc too small, prompt would be multipart.
    """
    doc_path = find_doc(doc_id)
    if not doc_path:
        raise SystemExit(f"doc '{doc_id}' not found in data/index_rincian")
    doc = json.loads(doc_path.read_text(encoding="utf-8"))

    secondary_doc = None
    if paired_doc_id:
        sec_path = find_doc(paired_doc_id)
        if not sec_path:
            raise SystemExit(f"paired doc '{paired_doc_id}' not found")
        secondary_doc = json.loads(sec_path.read_text(encoding="utf-8"))

    leaves = filter_preamble(collect_leaf_nodes(doc["structure"]))
    n_used = min(compute_adaptive_n(len(leaves)) or 0, n_questions)
    if n_used <= 0:
        raise SystemExit(f"doc '{doc_id}' has too few leaves ({len(leaves)}) for GT")

    parts, _ = build_prompt_parts(
        doc, n_questions=n_used, char_budget=char_budget,
        query_type=query_type, secondary_doc=secondary_doc,
    )
    if len(parts) > 1:
        raise SystemExit(
            f"doc '{doc_id}' still produced {len(parts)} prompt parts even at "
            f"char_budget={char_budget}. Run prompt.py manually for this one, "
            f"or raise --char-budget."
        )
    return parts[0]["prompt"], n_used


def crossdoc_pair(allocation: dict, doc_id: str) -> str | None:
    """Look up the crossdoc paired-doc for a given primary doc_id."""
    for payload in allocation.values():
        for entry in payload.get("per_doc_allocation", []):
            if entry["doc_id"] == doc_id:
                return entry.get("crossdoc_paired_with")
    return None


def call_openai(client, model: str, prompt: str, temperature: float, seed: int) -> tuple[str, dict]:
    """Single Chat Completions call, return (text, usage_dict).

    Sets temperature and seed for reproducibility. Falls back to defaults if
    the model rejects either parameter (some reasoning models lock them).
    """
    kwargs: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
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


def append_log(entry: dict) -> None:
    """Append one entry to the master log array, creating the file if needed."""
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
    """Write the full per-call payload (prompt, response, usage) to its own file."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    path = LOG_DIR / f"{stamp}_{doc_id}__{query_type}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def annotate_one(client, model: str, pricing: dict, cat: str, doc_id: str,
                 query_type: str, n_questions: int, paired_doc_id: str | None,
                 dry_run: bool, temperature: float, seed: int,
                 char_budget: int) -> tuple[float, int, str]:
    """Build prompt, call API, write raw file. Return (cost, n_items, status)."""
    prompt, n_used = build_one_prompt(doc_id, query_type, n_questions, paired_doc_id, char_budget)
    if dry_run:
        est = estimate_cost(prompt, n_used, pricing)
        return est, 0, f"dry-run estimate ${est:.4f}"

    started = dt.datetime.now()
    stamp = started.strftime("%Y%m%d_%H%M%S")
    t0 = time.time()
    text, usage = call_openai(client, model, prompt, temperature=temperature, seed=seed)
    elapsed = time.time() - t0
    cost = actual_cost(usage, pricing)

    base_log = {
        "doc_id": doc_id,
        "query_type": query_type,
        "category": cat,
        "model": model,
        "paired_doc_id": paired_doc_id,
        "n_questions_target": n_used,
        "started_at": started.isoformat(timespec="seconds"),
        "elapsed_s": round(elapsed, 2),
        "temperature": temperature,
        "seed": seed,
        "usage": usage,
        "cost_usd": round(cost, 6),
    }

    try:
        items = extract_json_array(text)
    except ValueError as e:
        detail = write_call_detail(stamp, doc_id, query_type, {**base_log, "status": "parse_error", "error": str(e), "prompt": prompt, "response": text})
        append_log({**base_log, "status": "parse_error", "error": str(e), "n_items_returned": 0, "detail_path": str(detail)})
        return cost, 0, f"parse error ({e}), full payload at {detail}"

    items, fixups = _normalize_schema(items)

    raw_dir = RAW_DIR / cat
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / raw_filename(doc_id, query_type)
    raw_path.write_text(json.dumps(items, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    meta_path = write_meta_sidecar(cat, doc_id, n_used, total_parts=1, query_type=query_type)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["annotator_model"] = model
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    detail = write_call_detail(stamp, doc_id, query_type, {**base_log, "status": "ok", "n_items_returned": len(items), "schema_fixups": fixups, "prompt": prompt, "response": text, "items": items})
    append_log({**base_log, "status": "ok", "n_items_returned": len(items), "schema_fixups_count": len(fixups), "raw_path": str(raw_path), "detail_path": str(detail)})

    return cost, len(items), f"wrote {len(items)} item(s) -> {raw_path}"


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--category", type=str, default=None)
    ap.add_argument("--doc-id", type=str, default=None,
                    help="Single doc_id, requires --type")
    ap.add_argument("--type", "-t", type=str, default=None,
                    choices=["factual", "paraphrased", "multihop", "crossdoc", "adversarial"])
    ap.add_argument("--model", type=str, default=DEFAULT_MODEL,
                    help=f"OpenAI model id or pinned snapshot (default {DEFAULT_MODEL})")
    ap.add_argument("--max-cost", type=float, default=DEFAULT_MAX_COST,
                    help=f"Hard stop if estimated batch cost exceeds this (USD, default {DEFAULT_MAX_COST})")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"Sampling temperature (default {DEFAULT_TEMPERATURE} for determinism)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Random seed for reproducibility (default {DEFAULT_SEED})")
    ap.add_argument("--char-budget", type=int, default=DEFAULT_CHAR_BUDGET,
                    help=f"Per-prompt char budget (default {DEFAULT_CHAR_BUDGET}). "
                         f"Bigger than the manual paste flow because API context windows are larger.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview cost only, do not call API")
    ap.add_argument("--force", action="store_true",
                    help="Re-annotate items even if already annotated")
    args = ap.parse_args()

    if args.doc_id and not args.type:
        ap.error("--doc-id requires --type")

    if not ALLOCATION_FILE.exists():
        raise SystemExit(f"{ALLOCATION_FILE} not found, run allocate_quotas.py first")
    allocation = json.loads(ALLOCATION_FILE.read_text(encoding="utf-8"))

    pricing = lookup_pricing(args.model)
    print(f"Model       : {args.model}")
    print(f"Pricing     : ${pricing['input']}/${pricing['cached']}/${pricing['output']} per 1M (input/cached/output)")
    print(f"Temperature : {args.temperature}")
    print(f"Seed        : {args.seed}")
    print(f"Max cost    : ${args.max_cost}")
    print(f"Log file    : {LOG_FILE}")
    print(f"Dry run     : {args.dry_run}")
    print()

    # Build work list, skipping already-annotated unless --force
    work = []
    for cat, doc_id, qt, count in iter_plan(allocation, args.category, args.type):
        if args.doc_id and doc_id != args.doc_id:
            continue
        st = _state(doc_id, qt)
        if not args.force and st not in ("not-annotated",):
            print(f"  skip  {doc_id}  type={qt}  (state={st})")
            continue
        work.append((cat, doc_id, qt, count))

    if not work:
        print("Nothing to do.")
        return

    # Cost preview by estimating each prompt
    print(f"\nEstimating cost for {len(work)} item(s)...")
    estimates: list[float] = []
    for cat, doc_id, qt, n in work:
        paired = crossdoc_pair(allocation, doc_id) if qt == "crossdoc" else None
        try:
            prompt, n_used = build_one_prompt(doc_id, qt, n, paired, args.char_budget)
            est = estimate_cost(prompt, n_used, pricing)
            estimates.append(est)
        except SystemExit as e:
            print(f"  SKIP  {doc_id}  type={qt}  ({e})")
            estimates.append(0.0)
    total_est = sum(estimates)
    print(f"Estimated total, ${total_est:.4f} (no caching assumed, real cost likely lower)")

    if total_est > args.max_cost:
        raise SystemExit(
            f"\nEstimated cost ${total_est:.2f} exceeds --max-cost ${args.max_cost}. "
            f"Raise --max-cost or scope down with --category / --type / --doc-id."
        )

    if args.dry_run:
        print("\n[dry-run] not calling API")
        return

    # Real API calls
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY not set in env, add to .env")
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("openai package not installed, run, pip install 'openai>=1.50'")
    client = OpenAI()

    print(f"\nRunning {len(work)} call(s)...")
    total_cost = 0.0
    n_ok = n_fail = 0
    for cat, doc_id, qt, n in work:
        paired = crossdoc_pair(allocation, doc_id) if qt == "crossdoc" else None
        try:
            cost, n_items, status = annotate_one(
                client, args.model, pricing, cat, doc_id, qt, n, paired,
                dry_run=False, temperature=args.temperature, seed=args.seed,
                char_budget=args.char_budget,
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

    print(f"\nAnnotated {n_ok}, failed {n_fail}, total cost ${total_cost:.4f}")
    if n_ok:
        print()
        print("Next.")
        print(f"  1. python scripts/gt/run_allocation.py --build --category {args.category or '<cat>'}")
        print(f"  2. (Judge step manual via Copilot Sonnet 4.6 over each tmp/validate_*.txt,")
        print(f"     paste full response with ---CLEANED--- framing over the matching raw GT file)")
        print(f"  3. python scripts/gt/run_allocation.py --apply --category {args.category or '<cat>'}")


if __name__ == "__main__":
    main()
