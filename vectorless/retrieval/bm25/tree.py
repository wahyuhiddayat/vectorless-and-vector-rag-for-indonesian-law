"""BM25 hierarchical tree navigation for Indonesian legal QA.

Navigates the document tree level-by-level using BM25 scoring at each level.
At each level, nodes are scored by their title and summary, and top-k are
selected for drilling down. This is expected to perform poorly because
node titles and summaries have limited keyword coverage compared to full text.

This module exists to empirically prove that BM25 cannot effectively use
tree structure for navigation, unlike LLM-based approaches.

Usage:
    python -m vectorless.retrieval.bm25.tree "Apa syarat penyadapan?"
    python -m vectorless.retrieval.bm25.tree "Apa syarat penyadapan?" --top_k_per_level 3
"""

import argparse
import time

from rank_bm25 import BM25Okapi

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_catalog, load_doc, extract_nodes, save_log,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = 3) -> list[dict]:
    """Rank catalog entries with BM25 over the metadata fields.

    Args:
        query: Legal question in Indonesian.
        catalog: List of document metadata dicts.
        top_k: Number of top documents to return.

    Returns:
        List of dicts with doc_id, judul, and bm25_score.
    """
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
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            results.append({
                "doc_id": catalog[idx]["doc_id"],
                "judul": catalog[idx]["judul"],
                "bm25_score": round(float(score), 4),
            })
    return results


def _bm25_level_search(query: str, nodes: list[dict], top_k: int = 3) -> list[dict]:
    """Score nodes at one level using BM25 on title and summary.

    Args:
        query: Legal question in Indonesian.
        nodes: List of nodes at the current tree level.
        top_k: Number of top nodes to select.

    Returns:
        List of selected nodes with their BM25 scores.
    """
    if not nodes:
        return []

    corpus = []
    for node in nodes:
        combined = node.get("title", "") + " " + node.get("summary", "")
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        node = nodes[idx]
        results.append({
            "node_id": node["node_id"],
            "title": node.get("title", ""),
            "summary": node.get("summary", ""),
            "bm25_score": round(float(score), 4),
            "has_children": "nodes" in node and bool(node.get("nodes")),
            "_node_ref": node,
        })

    return results


def tree_search(query: str, doc: dict, top_k_per_level: int = 3,
                verbose: bool = True) -> dict:
    """Navigate the document tree level-by-level using BM25.

    At each level, BM25 scores nodes by title+summary and selects top-k.
    Continues drilling down until reaching leaf nodes.

    Args:
        query: Legal question in Indonesian.
        doc: Loaded document dict with structure field.
        top_k_per_level: Max nodes to select at each level.
        verbose: Print progress.

    Returns:
        Dict with steps (navigation trace) and node_ids (final leaf nodes).
    """
    structure = doc["structure"]
    doc_title = doc.get("judul", "")
    steps = []
    max_rounds = 8

    current_nodes = structure
    current_ids = []
    round_num = 1

    while round_num <= max_rounds:
        selected = _bm25_level_search(query, current_nodes, top_k=top_k_per_level)

        if not selected:
            break

        steps.append({
            "round": round_num,
            "level": f"level-{round_num}",
            "options_shown": [n.get("title", "") for n in current_nodes],
            "selected": [s["node_id"] for s in selected],
            "scores": {s["node_id"]: s["bm25_score"] for s in selected},
        })

        if verbose:
            print(f"\n[BM25 Tree - Round {round_num}] Selected:")
            for s in selected:
                print(f"  {s['node_id']} {s['title']} (BM25: {s['bm25_score']:.4f})")

        final_ids = []
        need_drill = []

        for s in selected:
            node_ref = s["_node_ref"]
            if s["has_children"]:
                need_drill.extend(node_ref.get("nodes", []))
            else:
                final_ids.append(s["node_id"])

        if not need_drill:
            current_ids = final_ids
            break

        current_nodes = need_drill
        current_ids = final_ids
        round_num += 1

    if not current_ids and selected:
        current_ids = [s["node_id"] for s in selected]

    return {"steps": steps, "node_ids": current_ids}


def retrieve(query: str, top_k_per_level: int = 3, verbose: bool = True) -> dict:
    """Full BM25 tree retrieval pipeline.

    1. BM25 doc search to select document.
    2. BM25 tree navigation level-by-level.
    3. Answer generation from retrieved leaf nodes.

    Args:
        query: Legal question in Indonesian.
        top_k_per_level: Max nodes to select at each tree level.
        verbose: Print progress.

    Returns:
        Dict with query, strategy, search results, answer, sources, and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: bm25-tree (top_k_per_level={top_k_per_level})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_results = _bm25_doc_search(query, catalog)
    steps["doc_search"] = step_metrics(t_step, snap)

    if not doc_results:
        return {"query": query, "strategy": "bm25-tree",
                "error": "No relevant documents found"}

    if verbose:
        print(f"\n[Doc Search - BM25] Selected: {[r['doc_id'] for r in doc_results]}")
        for r in doc_results:
            print(f"  {r['doc_id']} (BM25: {r['bm25_score']:.4f})")

    snap = snapshot_counters()
    t_step = time.time()

    doc_id = doc_results[0]["doc_id"]
    doc = load_doc(doc_id)

    tree_result = tree_search(query, doc, top_k_per_level=top_k_per_level,
                              verbose=verbose)
    node_ids = tree_result.get("node_ids", [])
    steps["tree_search"] = step_metrics(t_step, snap)

    if not node_ids:
        return {"query": query, "strategy": "bm25-tree", "doc_ids": [doc_id],
                "error": "No relevant nodes found"}

    nodes = extract_nodes(doc, node_ids)

    if not nodes:
        return {"query": query, "strategy": "bm25-tree", "doc_ids": [doc_id],
                "node_ids": node_ids, "error": "Selected nodes not found in tree"}

    sources = []
    for node in nodes:
        sources.append({
            "doc_id": doc_id,
            "node_id": node["node_id"],
            "title": node["title"],
            "navigation_path": node["navigation_path"],
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "bm25-tree",
        "doc_search": {"rankings": doc_results},
        "tree_search": tree_result,
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": steps},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


def main():
    """CLI entry point for BM25 tree retrieval."""
    ap = argparse.ArgumentParser(
        description="BM25 tree (hierarchical) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--top_k_per_level", type=int, default=3,
                    help="Max nodes to select at each tree level (default: 3)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k_per_level=args.top_k_per_level)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > {src['navigation_path']}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
