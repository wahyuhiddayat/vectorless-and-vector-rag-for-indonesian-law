"""
Pure vector (dense) retrieval for Indonesian legal documents.

Embed query with gemini-embedding-001, cosine similarity search in Qdrant,
LLM generates answer from retrieved chunks.

Usage:
    python -m vector.retrieve_vector "Apa syarat penyadapan?"
    python -m vector.retrieve_vector "Apa syarat penyadapan?" --top_k 10
    python -m vector.retrieve_vector "Apa syarat penyadapan?" --top_k 10
"""

import argparse
import time

from qdrant_client import QdrantClient

from .retrieve_common import (
    embed_query, reset_token_counters, get_token_stats,
    generate_answer, save_log, COLLECTION_NAME, QDRANT_URL,
)


# ============================================================
# RETRIEVAL
# ============================================================

def vector_search(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """Dense vector search: embed query -> Qdrant cosine similarity -> top-K."""
    query_vec = embed_query(query)

    qdrant = QdrantClient(url=QDRANT_URL)
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


# ============================================================
# MAIN PIPELINE
# ============================================================

def retrieve(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """Full dense retrieval pipeline: vector search -> answer generation."""
    reset_token_counters()
    t_start = time.time()

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: vector-dense (top_k={top_k})")
        print(f"{'='*60}")

    # Step 1: Vector search
    search_result = vector_search(query, top_k, verbose)
    rankings = search_result["rankings"]

    if not rankings:
        return {"query": query, "strategy": "vector-dense",
                "error": "No results found"}

    # Step 2: Generate answer
    answer_result = generate_answer(query, rankings, verbose=verbose)

    # Build sources
    sources = []
    for r in rankings:
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "cosine_score": r["score"],
        })

    elapsed = time.time() - t_start
    stats = get_token_stats()

    result = {
        "query": query,
        "strategy": "vector-dense",
        "chunking": "pasal",
        "vector_search": search_result,
        "answer": answer_result.get("answer", ""),
        "cited_pasals": answer_result.get("cited_pasals", []),
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2)},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="Vector (dense) retrieval for Indonesian legal docs")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k", type=int, default=5, help="Number of results (default: 5)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > {src['navigation_path']} (cosine: {src.get('cosine_score', 'N/A'):.4f})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
