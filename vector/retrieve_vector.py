"""
Pure vector (dense) retrieval for Indonesian legal documents.

Embed query, then Qdrant cosine similarity. Retrieval-only pipeline,
no answer generation.

Configuration via env vars (see common.py):
    VECTOR_EMBEDDING_MODEL, VECTOR_COLLECTION, QDRANT_PATH / QDRANT_URL, VECTOR_GRANULARITY

Usage:
    python -m vector.retrieve_vector "Apa syarat penyadapan?"
    python -m vector.retrieve_vector "Apa syarat penyadapan?" --top_k 10
"""

import argparse
import time

from .common import (
    embed_query, save_log,
    COLLECTION_NAME, GRANULARITY, EMBEDDING_MODEL,
    get_qdrant_client,
)


def vector_search(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """Embed the query and return the top Qdrant matches."""
    query_vec = embed_query(query)

    qdrant = get_qdrant_client()
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
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


def retrieve(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """Run dense retrieval. Returns retrieved chunks only, no answer generation."""
    t_start = time.time()

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: vector-dense (top_k={top_k})")
        print(f"Model: {EMBEDDING_MODEL}  Collection: {COLLECTION_NAME}")
        print(f"{'='*60}")

    search_result = vector_search(query, top_k, verbose)
    rankings = search_result["rankings"]

    if not rankings:
        return {"query": query, "strategy": "vector-dense", "error": "No results found"}

    sources = [
        {
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "cosine_score": r["score"],
        }
        for r in rankings
    ]

    elapsed = time.time() - t_start

    result = {
        "query": query,
        "strategy": "vector-dense",
        "chunking": GRANULARITY,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
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
    ap.add_argument("--top_k", type=int, default=5, help="Number of results (default: 5)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > {src['navigation_path']} (cosine: {src.get('cosine_score', 'N/A'):.4f})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
