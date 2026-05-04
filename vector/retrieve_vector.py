"""
Pure vector (dense) retrieval for Indonesian legal documents.

Two-stage when reranker is configured. Embed query, Qdrant cosine to fetch
top-N=RERANKER_TOP_N (50) candidates, then optional reranker reorders to top-k.
When reranker is "none", returns Qdrant top-k directly. Retrieval-only pipeline,
no answer generation.

Configuration via env vars (see common.py):
    VECTOR_EMBEDDING_MODEL, VECTOR_COLLECTION, QDRANT_PATH / QDRANT_URL,
    VECTOR_GRANULARITY, VECTOR_RERANKER

Usage:
    python -m vector.retrieve_vector "Apa syarat penyadapan?"
    python -m vector.retrieve_vector "Apa syarat penyadapan?" --top_k 10
    python -m vector.retrieve_vector "Apa syarat penyadapan?" --reranker bge-reranker-v2-m3
"""

import argparse
import os
import time

from . import common as _common
from .common import embed_query, save_log, get_qdrant_client
from .rerank import rerank as run_rerank


def vector_search(query: str, top_k: int, verbose: bool = True) -> dict:
    """Embed the query and return the top Qdrant matches."""
    query_vec = embed_query(query)

    qdrant = get_qdrant_client()
    response = qdrant.query_points(
        collection_name=_common.COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
    )

    rankings = []
    for hit in response.points:
        p = hit.payload
        rankings.append({
            "score": hit.score,
            "score_type": "cosine",
            "doc_id": p["doc_id"],
            "doc_title": p["doc_title"],
            "node_id": p["node_id"],
            "title": p["title"],
            "navigation_path": p["navigation_path"],
            "text": p["text"],
        })

    if verbose:
        print(f"\n[Vector Search] Top {len(rankings)} results:")
        for r in rankings:
            print(f"  {r['node_id']} {r['title']} (cosine: {r['score']:.4f})")
            print(f"    doc: {r['doc_id']}  path: {r['navigation_path']}")

    return {"rankings": rankings}


def retrieve(query: str, top_k: int = 10, verbose: bool = True) -> dict:
    """Run dense retrieval with optional reranker. Returns retrieved chunks only."""
    t_start = time.time()
    reranker = _common.RERANKER
    use_rerank = reranker != "none"
    first_stage_n = _common.RERANKER_TOP_N if use_rerank else top_k

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: vector-dense (top_k={top_k}, reranker={reranker})")
        print(f"Model: {_common.EMBEDDING_MODEL}  Collection: {_common.COLLECTION_NAME}")
        if use_rerank:
            print(f"First-stage top_n: {first_stage_n}")
        print(f"{'='*60}")

    search_result = vector_search(query, first_stage_n, verbose)
    rankings = search_result["rankings"]

    if not rankings:
        return {"query": query, "strategy": "vector-dense", "error": "No results found"}

    if use_rerank:
        t_rerank = time.time()
        rerank_input = [
            {**r, "cosine_score": r["score"]}
            for r in rankings
        ]
        reranked = run_rerank(query, rerank_input, reranker, top_k=top_k)
        rerank_elapsed = time.time() - t_rerank
        rankings = reranked
        if verbose:
            print(f"\n[Reranker {reranker}] Reordered to top {len(rankings)} "
                  f"in {rerank_elapsed:.2f}s")
            for r in rankings:
                print(f"  {r['node_id']} {r['title']} (rerank: {r['rerank_score']:.4f})")

    sources = []
    for r in rankings:
        src = {
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "cosine_score": r.get("cosine_score", r.get("score")),
        }
        if use_rerank:
            src["rerank_score"] = r.get("rerank_score")
        sources.append(src)

    elapsed = time.time() - t_start

    result = {
        "query": query,
        "strategy": "vector-dense",
        "chunking": _common.GRANULARITY,
        "embedding_model": _common.EMBEDDING_MODEL,
        "reranker": reranker,
        "first_stage_top_n": first_stage_n,
        "collection": _common.COLLECTION_NAME,
        "vector_search": search_result,
        "sources": sources,
        "metrics": {
            "llm_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "elapsed_s": round(elapsed, 2),
        },
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  0 LLM calls (retrieval-only)")
        print(f"{'='*60}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Vector (dense) retrieval for Indonesian legal docs")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k", type=int, default=10, help="Final top-k (default: 10)")
    ap.add_argument("--reranker", default=None,
                    help="Override VECTOR_RERANKER env. none, bge-reranker-v2-m3, qwen3-reranker-0.6b")
    args = ap.parse_args()

    if args.reranker is not None:
        os.environ["VECTOR_RERANKER"] = args.reranker
        _common.RERANKER = args.reranker

    result = retrieve(args.query, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        score_label = f"rerank: {src['rerank_score']:.4f}" if src.get("rerank_score") is not None else f"cosine: {src.get('cosine_score', 0):.4f}"
        print(f"  > {src['navigation_path']} ({score_label})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
