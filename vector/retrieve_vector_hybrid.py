"""
Hybrid vector retrieval: BM25 (sparse) + Qdrant (dense) for Indonesian legal documents.

Sparse: BM25 keyword search on all chunk texts loaded from Qdrant
Dense:  configured embedding model + Qdrant cosine similarity
Merge:  dense first (semantic priority), then sparse (keyword), deduplicated

Configuration via env vars (see common.py):
    VECTOR_EMBEDDING_MODEL, VECTOR_COLLECTION, QDRANT_PATH / QDRANT_URL, VECTOR_GRANULARITY

Usage:
    python -m vector.retrieve_vector_hybrid "Apa syarat penyadapan?"
    python -m vector.retrieve_vector_hybrid "Apa syarat penyadapan?" --top_k 10
"""

import argparse
import re
import time

from rank_bm25 import BM25Okapi

from .common import (
    embed_query, reset_token_counters, get_token_stats,
    generate_answer, save_log,
    COLLECTION_NAME, GRANULARITY, EMBEDDING_MODEL,
    get_qdrant_client,
)


# Indonesian stopwords (same set as vectorless-rag)
_STOPWORDS = {
    "yang", "dan", "di", "dari", "untuk", "dengan", "pada", "adalah",
    "ini", "itu", "atau", "dalam", "ke", "se", "oleh", "sebagai",
    "tidak", "akan", "juga", "sudah", "telah", "serta", "bahwa",
    "tersebut", "dapat", "lebih", "antara", "tentang", "setiap",
    "atas", "secara", "terhadap", "kepada", "suatu", "sesuai",
    "berdasarkan", "melalui", "mengenai", "apabila", "sampai",
    "dimaksud", "sebagaimana", "ayat", "huruf", "pasal",
    "apa", "berapa", "bagaimana", "siapa",
}


def tokenize(text: str) -> list[str]:
    """Simple Indonesian tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


# ============================================================
# DATA LOADING (for BM25 — needs all chunk texts from Qdrant)
# ============================================================

def load_all_chunks() -> list[dict]:
    """Load all chunks from Qdrant for BM25 corpus building."""
    qdrant = get_qdrant_client()
    all_points = []
    offset = None

    while True:
        result = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points, next_offset = result
        all_points.extend(points)
        if next_offset is None:
            break
        offset = next_offset

    chunks = []
    for p in all_points:
        chunks.append({
            "id": p.id,
            "doc_id": p.payload["doc_id"],
            "doc_title": p.payload["doc_title"],
            "node_id": p.payload["node_id"],
            "title": p.payload["title"],
            "navigation_path": p.payload["navigation_path"],
            "text": p.payload["text"],
        })

    return chunks


# ============================================================
# BM25 SPARSE SEARCH
# ============================================================

def bm25_search(query: str, chunks: list[dict], top_k: int = 5,
                verbose: bool = True) -> list[dict]:
    """BM25 sparse search over chunk texts."""
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)

    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] <= 0:
            continue
        c = chunks[idx]
        results.append({
            "score": float(scores[idx]),
            "score_type": "bm25",
            "doc_id": c["doc_id"],
            "doc_title": c["doc_title"],
            "node_id": c["node_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "text": c["text"],
        })

    if verbose:
        print(f"\n[Sparse Search - BM25] Top {len(results)}:")
        for r in results:
            print(f"  {r['node_id']} {r['title']} (BM25: {r['score']:.4f})")
            print(f"    doc: {r['doc_id']}  path: {r['navigation_path']}")

    return results


# ============================================================
# QDRANT DENSE SEARCH
# ============================================================

def dense_search(query: str, top_k: int = 5, verbose: bool = True) -> list[dict]:
    """Dense vector search: embed query -> Qdrant cosine similarity -> top-K."""
    query_vec = embed_query(query)

    qdrant = get_qdrant_client()
    response = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vec,
        limit=top_k,
    )

    results = []
    for hit in response.points:
        p = hit.payload
        results.append({
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
        print(f"\n[Dense Search - Vector] Top {len(results)}:")
        for r in results:
            print(f"  {r['node_id']} {r['title']} (cosine: {r['score']:.4f})")
            print(f"    doc: {r['doc_id']}  path: {r['navigation_path']}")

    return results


# ============================================================
# MERGE
# ============================================================

def merge_results(sparse: list[dict], dense: list[dict],
                  verbose: bool = True) -> list[dict]:
    """Merge dense + sparse results by concatenation with deduplication.

    Dense first (semantic priority), then sparse (keyword priority).
    """
    seen: set = set()
    merged = []

    for r in dense:
        key = (r["doc_id"], r["node_id"])
        if key not in seen:
            seen.add(key)
            merged.append(r)

    for r in sparse:
        key = (r["doc_id"], r["node_id"])
        if key not in seen:
            seen.add(key)
            merged.append(r)

    if verbose:
        print(f"\n[Merge] {len(dense)} dense + {len(sparse)} sparse -> {len(merged)} unique")

    return merged


# ============================================================
# MAIN PIPELINE
# ============================================================

def retrieve(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """Full hybrid retrieval: BM25 sparse + Qdrant dense -> merge -> answer."""
    reset_token_counters()
    t_start = time.time()

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: vector-hybrid (top_k={top_k})")
        print(f"Model: {EMBEDDING_MODEL}  Collection: {COLLECTION_NAME}")
        print(f"{'='*60}")

    # Load all chunks for BM25
    chunks = load_all_chunks()

    # Sparse search (BM25)
    sparse_results = bm25_search(query, chunks, top_k, verbose)

    # Dense search (vector)
    dense_results = dense_search(query, top_k, verbose)

    # Merge
    merged = merge_results(sparse_results, dense_results, verbose)

    if not merged:
        return {"query": query, "strategy": "vector-hybrid", "error": "No results found"}

    # Answer generation
    answer_result = generate_answer(query, merged, verbose)

    # Build sources
    sources = []
    for r in merged:
        src = {
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "score_type": r["score_type"],
        }
        if r["score_type"] == "cosine":
            src["cosine_score"] = r["score"]
        else:
            src["bm25_score"] = r["score"]
        sources.append(src)

    elapsed = time.time() - t_start
    stats = get_token_stats()

    result = {
        "query": query,
        "strategy": "vector-hybrid",
        "chunking": GRANULARITY,
        "embedding_model": EMBEDDING_MODEL,
        "collection": COLLECTION_NAME,
        "sparse_search": {"rankings": sparse_results},
        "dense_search": {"rankings": dense_results},
        "merged_count": len(merged),
        "answer": answer_result.get("answer", ""),
        "citations": answer_result.get("citations", []),
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
    ap = argparse.ArgumentParser(description="Hybrid vector (BM25 + dense) retrieval")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k", type=int, default=5, help="Top-K per method (default: 5)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        score_key = "cosine_score" if src["score_type"] == "cosine" else "bm25_score"
        print(f"  > {src['navigation_path']} ({src['score_type']}: {src.get(score_key, 'N/A'):.4f})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
