"""
Subprocess worker for vector RAG evaluation.

Runs exactly one vector retrieval call in a fresh Python process so env vars
(VECTOR_EMBEDDING_MODEL, VECTOR_COLLECTION, QDRANT_PATH) are isolated per invocation.

Usage:
    python scripts/eval/vector_worker.py \\
        --system vector-dense \\
        --granularity pasal \\
        --embedding-model gemini-embedding-001 \\
        --query "Apa syarat penyadapan?" \\
        --qdrant-path ./qdrant_local
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

# Collection naming convention: law-{granularity}-{model_short}
_MODEL_SHORT = {
    "bge-m3": "bgem3",
    "all-indobert-base-v4": "indobert",
    "multilingual-e5-large-instruct": "e5",
}


def run_retrieval(system: str, query: str, top_k: int) -> dict:
    if system == "vector-dense":
        from vector.retrieve_vector import retrieve
        return retrieve(query, top_k=top_k, verbose=False)

    if system == "vector-hybrid":
        from vector.retrieve_vector_hybrid import retrieve
        return retrieve(query, top_k=top_k, verbose=False)

    raise ValueError(f"Unsupported system: {system!r}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run one vector retrieval call in a fresh process."
    )
    ap.add_argument(
        "--system", required=True,
        choices=["vector-dense", "vector-hybrid"],
    )
    ap.add_argument(
        "--granularity", required=True,
        choices=["pasal", "ayat", "full_split"],
    )
    ap.add_argument(
        "--embedding-model", required=True,
        choices=list(_MODEL_SHORT),
        help="Embedding model: bge-m3 | all-indobert-base-v4 | multilingual-e5-large-instruct",
    )
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument(
        "--qdrant-path", default=None,
        help="Path to local Qdrant storage directory",
    )
    args = ap.parse_args()

    # Set env vars BEFORE importing any vector modules (they read at import time)
    model_short = _MODEL_SHORT[args.embedding_model]
    collection = f"law-{args.granularity}-{model_short}"

    os.environ["VECTOR_EMBEDDING_MODEL"] = args.embedding_model
    os.environ["VECTOR_COLLECTION"] = collection
    os.environ["VECTOR_GRANULARITY"] = args.granularity
    if args.qdrant_path:
        os.environ["QDRANT_PATH"] = args.qdrant_path

    try:
        result = run_retrieval(args.system, args.query, args.top_k)
        payload = {
            "ok": True,
            "system": args.system,
            "granularity": args.granularity,
            "embedding_model": args.embedding_model,
            "collection": collection,
            "result": result,
        }
    except Exception as exc:  # pragma: no cover - operational fallback
        payload = {
            "ok": False,
            "system": args.system,
            "granularity": args.granularity,
            "embedding_model": args.embedding_model,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(payload, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
