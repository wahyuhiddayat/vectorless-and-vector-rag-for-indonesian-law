"""
BM25 flat (single-stage) retrieval for Indonesian legal QA.

Instead of 2-stage (doc search → node search), this searches ALL leaf nodes
from ALL documents at once. Each leaf node's text is enriched with document
title and navigation path for better keyword matching.

This is the IR-standard way to use BM25 (flat corpus search), avoiding the
cascading failure problem where doc-level metadata mismatch blocks retrieval.

Usage:
    python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?" --top_k 5
    python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?" --top_k 10
"""

import argparse
import time

from rank_bm25 import BM25Okapi

from ..common import (
    tokenize, reset_token_counters, get_token_stats,
    load_all_leaf_nodes, generate_answer_multi_doc, save_log,
)


# ============================================================
# FLAT SEARCH (single-stage across all documents)
# ============================================================

def flat_search(query: str, leaves: list[dict], top_k: int = 5,
                verbose: bool = True) -> list[dict]:
    """BM25 search across ALL leaf nodes from ALL documents.

    Each leaf's text is enriched with doc title and navigation path
    for better keyword coverage (metadata enrichment).
    """
    # Build enriched corpus
    corpus = []
    for leaf in leaves:
        enriched = leaf["doc_title"] + " " + leaf["navigation_path"] + " " + leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            enriched += " " + leaf["penjelasan"]
        corpus.append(tokenize(enriched))

    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Rank and filter
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue
        leaf = leaves[idx]
        results.append({
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "node_id": leaf["node_id"],
            "title": leaf["title"],
            "navigation_path": leaf["navigation_path"],
            "text": leaf["text"],
            "penjelasan": leaf.get("penjelasan"),
            "score": round(float(score), 4),
        })

    if verbose:
        print(f"\n[Flat BM25 Search] Top {len(results)} results:")
        for r in results:
            print(f"  {r['node_id']} {r['title']} (BM25: {r['score']:.4f})")
            print(f"    doc: {r['doc_id']}  path: {r['navigation_path']}")

    return results


# ============================================================
# MAIN PIPELINE
# ============================================================

def retrieve(query: str, top_k: int = 5, verbose: bool = True) -> dict:
    """Full flat BM25 retrieval: load all leaves → search → answer.

    Args:
        query: Legal question in Indonesian
        top_k: Max results to retrieve
        verbose: Print progress
    """
    reset_token_counters()
    t_start = time.time()

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: bm25-flat (top_k={top_k})")
        print(f"{'='*60}")

    # Step 1: Load all leaf nodes from all documents
    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    # Step 2: Flat BM25 search
    results = flat_search(query, leaves, top_k=top_k, verbose=verbose)

    if not results:
        return {"query": query, "strategy": "bm25-flat",
                "error": "No results found"}

    # Step 3: Generate answer (multi-doc)
    answer_result = generate_answer_multi_doc(query, results, verbose=verbose)

    # Build sources
    sources = []
    for r in results:
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "bm25_score": r["score"],
        })

    elapsed = time.time() - t_start
    stats = get_token_stats()

    result = {
        "query": query,
        "strategy": "bm25-flat",
        "corpus_size": len(leaves),
        "search": {"rankings": results},
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
    ap = argparse.ArgumentParser(
        description="BM25 flat (single-stage) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k", type=int, default=5, help="Number of results (default: 5)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']} (BM25: {src['bm25_score']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
