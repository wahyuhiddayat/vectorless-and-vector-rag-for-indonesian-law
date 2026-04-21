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
    "full_split": "data/index_full_split",
}


def run_retrieval(system: str, query: str, top_k: int) -> dict:
    if system == "bm25-flat":
        from vectorless.retrieval.bm25 import flat as module

        module.save_log = lambda _result: None
        return module.retrieve(query, top_k=top_k, verbose=False)

    if system == "hybrid":
        from vectorless.retrieval.hybrid import search as module

        module.save_log = lambda _result: None
        return module.retrieve(query, bm25_top_k=max(top_k, 10), verbose=False)

    if system == "llm-stepwise":
        from vectorless.retrieval.llm import search as module

        module.save_log = lambda _result: None
        return module.retrieve(query, strategy="stepwise", verbose=False)

    if system == "llm-full":
        from vectorless.retrieval.llm import search as module

        module.save_log = lambda _result: None
        return module.retrieve(query, strategy="full", verbose=False)

    if system == "hybrid-flat":
        from vectorless.retrieval.hybrid_flat import search as module

        module.save_log = lambda _result: None
        return module.retrieve(query, bm25_top_k=max(top_k, 20), verbose=False)

    raise ValueError(f"Unsupported system: {system}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run one vectorless retrieval call in a fresh process.")
    ap.add_argument("--system", required=True, choices=["bm25-flat", "hybrid", "hybrid-flat", "llm-stepwise", "llm-full"])
    ap.add_argument("--granularity", required=True, choices=["pasal", "ayat", "full_split"])
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
            "result": result,
        }
    except Exception as exc:  # pragma: no cover - operational fallback
        payload = {
            "ok": False,
            "system": args.system,
            "granularity": args.granularity,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
