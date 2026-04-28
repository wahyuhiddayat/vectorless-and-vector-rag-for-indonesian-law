"""
Subprocess worker for vectorless evaluation.

Runs exactly one vectorless retrieval call in a fresh Python process so the
active DATA_INDEX granularity is isolated per invocation.

Usage:
    python scripts/eval/vectorless_worker.py --system bm25-flat --granularity ayat --query "..."
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


GRANULARITY_TO_INDEX = {
    "pasal": "data/index_pasal",
    "ayat": "data/index_ayat",
    "rincian": "data/index_rincian",
}


def run_retrieval(system: str, query: str, top_k: int) -> dict:
    """Dispatch to the requested retrieval module in this fresh subprocess.

    The module-level `save_log = lambda _: None` patch disables the on-disk
    retrieval log each module would normally write. We mute it so concurrent
    eval combos do not race on the same log file. Real retrieval logs from
    the eval harness live in records/<system>__<granularity>.jsonl instead.
    """
    if system == "bm25-flat":
        from vectorless.retrieval.bm25 import flat as module

        module.save_log = lambda _result: None
        return module.retrieve(query, top_k=top_k, verbose=False)

    if system == "bm25-tree":
        from vectorless.retrieval.bm25 import tree as module

        module.save_log = lambda _result: None
        return module.retrieve(query, verbose=False)

    if system == "hybrid-flat":
        from vectorless.retrieval.hybrid import flat as module

        module.save_log = lambda _result: None
        return module.retrieve(query, bm25_top_k=max(top_k, 20), verbose=False)

    if system == "hybrid-tree":
        from vectorless.retrieval.hybrid import tree as module

        module.save_log = lambda _result: None
        return module.retrieve(query, bm25_top_k=max(top_k, 10), verbose=False)

    if system == "llm-flat":
        from vectorless.retrieval.llm import flat as module

        module.save_log = lambda _result: None
        # Disable random sampling. With a 70-doc corpus the entire flat list
        # fits inside Gemini Flash 1M context (rincian peaks at ~760K tokens).
        # Sampling 100 of >1900 leaves makes recall near-zero by construction.
        return module.retrieve(query, max_candidates=10**9, verbose=False)

    if system == "llm-tree":
        from vectorless.retrieval.llm import tree as module

        module.save_log = lambda _result: None
        return module.retrieve(query, verbose=False)

    raise ValueError(f"Unsupported system: {system}")


def _llm_model_constant() -> str | None:
    """Capture the LLM model name used by the retrieval modules, if any."""
    try:
        from vectorless.llm import MODEL
        return str(MODEL)
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one vectorless retrieval call in a fresh process.")
    ap.add_argument("--system", required=True, choices=[
        "bm25-flat", "bm25-tree",
        "hybrid-flat", "hybrid-tree",
        "llm-flat", "llm-tree",
    ])
    ap.add_argument("--granularity", required=True, choices=["pasal", "ayat", "rincian"])
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    args = ap.parse_args()

    os.environ["DATA_INDEX"] = GRANULARITY_TO_INDEX[args.granularity]

    try:
        result = run_retrieval(args.system, args.query, args.top_k)
        payload = {
            "ok": True,
            "system": args.system,
            "granularity": args.granularity,
            "llm_model": _llm_model_constant(),
            "result": result,
        }
    except Exception as exc:  # pragma: no cover - operational fallback
        payload = {
            "ok": False,
            "system": args.system,
            "granularity": args.granularity,
            "llm_model": _llm_model_constant(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
