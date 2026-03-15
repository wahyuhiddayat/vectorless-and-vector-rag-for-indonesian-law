"""
BM25 retrieval strategy for Indonesian legal QA — NO STOPWORD REMOVAL variant.

Same as bm25_2stage.py but without stopword removal in tokenizer.
Based on Faisal et al. (2024) finding that stopword removal hurts BM25
on formal Indonesian legal documents.

Usage:
    python -m vectorless.retrieval.bm25_no_sw "Apa syarat penyadapan?"
    python -m vectorless.retrieval.bm25_no_sw "Apa syarat penyadapan?" --top_k 5
    python -m vectorless.retrieval.bm25_no_sw --interactive
"""

import argparse
import json
import re
import time

from rank_bm25 import BM25Okapi

from .common import (
    reset_token_counters, get_token_stats,
    load_catalog, load_doc, extract_nodes,
    generate_answer, save_log, DATA_INDEX,
)


# ============================================================
# TOKENIZER (no stopword removal)
# ============================================================

def tokenize(text: str) -> list[str]:
    """Simple Indonesian tokenizer: lowercase, split on non-alphanumeric. No stopword removal."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return [t for t in tokens if len(t) > 1]


# ============================================================
# DOC SEARCH (BM25 on catalog metadata)
# ============================================================

def doc_search(query: str, catalog: list[dict], top_k: int = 3,
               verbose: bool = True) -> dict:
    """Select relevant doc_ids from catalog using BM25 on metadata fields.

    Indexes a combined text of judul + bidang + subjek + materi_pokok per doc.
    """
    # Build corpus from catalog metadata
    corpus = []
    for doc in catalog:
        combined = " ".join([
            doc.get("judul", ""),
            doc.get("bidang", ""),
            doc.get("subjek", ""),
            doc.get("materi_pokok", ""),
        ])
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Rank and filter
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            results.append({
                "doc_id": catalog[idx]["doc_id"],
                "judul": catalog[idx]["judul"],
                "score": round(float(score), 4),
            })

    doc_ids = [r["doc_id"] for r in results]

    if verbose:
        print(f"\n[Doc Search - BM25] Top {len(results)} docs:")
        for r in results:
            print(f"  {r['doc_id']} (score: {r['score']:.4f}) — {r['judul'][:80]}")

    return {"doc_ids": doc_ids, "rankings": results}


# ============================================================
# NODE SEARCH (BM25 on leaf node texts)
# ============================================================

def _collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    """Recursively collect all leaf nodes (nodes with text, no children) from tree."""
    leaves = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            leaves.extend(_collect_leaf_nodes(node["nodes"]))
        elif node.get("text"):
            leaves.append({
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "text": node["text"],
                "navigation_path": node.get("navigation_path", ""),
                "penjelasan": node.get("penjelasan"),
            })
    return leaves


def node_search(query: str, doc: dict, top_k: int = 5,
                verbose: bool = True) -> dict:
    """Find relevant Pasal nodes using BM25 on leaf node texts.

    Indexes the full text of every leaf node (Pasal) in the document tree,
    including penjelasan (official explanation) if available.
    """
    leaves = _collect_leaf_nodes(doc["structure"])

    if not leaves:
        return {"node_ids": [], "rankings": []}

    # Build corpus: pasal text + penjelasan + navigation_path for context
    corpus = []
    for leaf in leaves:
        combined = leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            combined += " " + leaf["penjelasan"]
        combined += " " + leaf.get("navigation_path", "")
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    query_tokens = tokenize(query)
    scores = bm25.get_scores(query_tokens)

    # Rank and filter
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            results.append({
                "node_id": leaves[idx]["node_id"],
                "title": leaves[idx]["title"],
                "navigation_path": leaves[idx]["navigation_path"],
                "score": round(float(score), 4),
            })

    node_ids = [r["node_id"] for r in results]

    if verbose:
        print(f"\n[Node Search - BM25] Top {len(results)} nodes:")
        for r in results:
            print(f"  {r['node_id']} {r['title']} (score: {r['score']:.4f})")
            print(f"    path: {r['navigation_path']}")

    return {"node_ids": node_ids, "rankings": results}


# ============================================================
# MAIN PIPELINE
# ============================================================

def retrieve(query: str, top_k_docs: int = 3, top_k_nodes: int = 5,
             verbose: bool = True) -> dict:
    """Full BM25 retrieval pipeline: doc search → node search → answer.

    Args:
        query: Legal question in Indonesian
        top_k_docs: Max docs to consider from doc search
        top_k_nodes: Max Pasal nodes to retrieve per doc
        verbose: Print progress
    """
    reset_token_counters()
    t_start = time.time()

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: bm25-no-sw (top_k_docs={top_k_docs}, top_k_nodes={top_k_nodes})")
        print(f"{'='*60}")

    # Step 1: Doc search via BM25
    catalog = load_catalog()
    doc_result = doc_search(query, catalog, top_k=top_k_docs, verbose=verbose)
    doc_ids = doc_result.get("doc_ids", [])

    if not doc_ids:
        return {"query": query, "strategy": "bm25-no-sw", "error": "No relevant documents found"}

    # Step 2: Node search via BM25 (on first relevant doc)
    doc_id = doc_ids[0]
    doc = load_doc(doc_id)

    node_result = node_search(query, doc, top_k=top_k_nodes, verbose=verbose)
    node_ids = node_result.get("node_ids", [])

    if not node_ids:
        return {"query": query, "strategy": "bm25-no-sw", "doc_ids": doc_ids,
                "error": "No relevant nodes found"}

    # Step 3: Extract text and generate answer (LLM only here)
    nodes = extract_nodes(doc, node_ids)

    if not nodes:
        return {"query": query, "strategy": "bm25-no-sw", "doc_ids": doc_ids,
                "node_ids": node_ids, "error": "Selected nodes not found in tree"}

    doc_meta = {"doc_id": doc_id, "judul": doc.get("judul", "")}
    answer_result = generate_answer(query, nodes, doc_meta, verbose=verbose)

    # Build sources
    sources = []
    for r in node_result.get("rankings", []):
        if r["node_id"] in node_ids:
            sources.append({
                "doc_id": doc_id,
                "node_id": r["node_id"],
                "title": r["title"],
                "navigation_path": r["navigation_path"],
                "bm25_score": r["score"],
            })

    elapsed = time.time() - t_start
    stats = get_token_stats()

    result = {
        "query": query,
        "strategy": "bm25-no-sw",
        "doc_search": doc_result,
        "node_search": node_result,
        "answer": answer_result.get("answer", ""),
        "citations": answer_result.get("citations", []),
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2)},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls (answer only)  |  "
              f"{stats['total_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(description="BM25 retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k_docs", type=int, default=3, help="Max docs from doc search")
    ap.add_argument("--top_k_nodes", type=int, default=5, help="Max nodes from node search")
    args = ap.parse_args()

    result = retrieve(args.query, top_k_docs=args.top_k_docs,
                      top_k_nodes=args.top_k_nodes)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > {src['navigation_path']} (BM25: {src.get('bm25_score', 'N/A')})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
