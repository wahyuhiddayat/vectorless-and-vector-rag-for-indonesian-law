"""BM25 hierarchical tree navigation for Indonesian legal QA.

Navigates the document tree level-by-level using BM25 scoring at each level.
At each level, nodes are scored by their title and summary, and top-k are
selected for drilling down. This is expected to perform poorly because
node titles and summaries have limited keyword coverage compared to full text.

The beam search uses title and summary for node scoring at intermediate
levels, then re-ranks all collected leaves with full-text BM25 using the
same enrichment as bm25-flat (doc_title + navigation_path + text + penjelasan).
Summaries are LLM-generated at indexing time, not per query (0 LLM calls
per query).

Usage:
    python -m vectorless.retrieval.bm25.tree "Apa syarat penyadapan?"
    python -m vectorless.retrieval.bm25.tree "Apa syarat penyadapan?" --top_k_per_level 3 --top_k 10
"""

import argparse
import time

from rank_bm25 import BM25Okapi

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_catalog, load_doc, extract_nodes, save_log,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = 1) -> list[dict]:
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
            doc.get("judul") or "",
            doc.get("bidang") or "",
            doc.get("subjek") or "",
            doc.get("materi_pokok") or "",
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
    """Score non-leaf nodes at one tree level using BM25 on title and summary.

    Used for navigation through aggregator levels (Bab, Bagian, Pasal-with-ayat).
    These nodes do not have direct text content, so descriptors (title and
    LLM-generated summary) are the natural representation for pruning the
    subtree search.

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


def _bm25_leaf_search(query: str, leaves: list[dict], doc_title: str,
                      top_k: int = 3) -> list[dict]:
    """Score leaf nodes using BM25 on the same fields as bm25-flat.

    Leaf nodes are the decision-point of bm25-tree (the final ranking output).
    To preserve decision-point fairness with bm25-flat, leaves are scored over
    `doc_title + navigation_path + text + penjelasan`, the same fields that
    bm25-flat uses per leaf. Summary is excluded by design (already consumed
    in the beam navigation stage).

    Args:
        query: Legal question in Indonesian.
        leaves: List of leaf nodes (terminal nodes without children).
        doc_title: Document title (judul) from the parent doc context.
        top_k: Number of top leaves to select.

    Returns:
        List of selected leaves with their BM25 scores.
    """
    if not leaves:
        return []

    corpus = []
    for leaf in leaves:
        combined = doc_title + " " + leaf.get("navigation_path", "") + " " + leaf.get("text", "")
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            combined += " " + leaf["penjelasan"]
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    results = []
    for idx, score in ranked[:top_k]:
        leaf = leaves[idx]
        results.append({
            "node_id": leaf["node_id"],
            "title": leaf.get("title", ""),
            "summary": leaf.get("summary", ""),
            "bm25_score": round(float(score), 4),
            "has_children": False,
            "_node_ref": leaf,
        })

    return results


def tree_search(query: str, doc: dict, top_k_per_level: int = 3,
                top_k: int = 10, verbose: bool = True) -> dict:
    """Navigate the document tree using BM25 beam search; rank leaves at the end.

    Two-phase design:
      1. Beam navigation. At each non-leaf level, BM25 scores nodes by
         title+summary and selects top `top_k_per_level` to drill into.
         Any leaf encountered at a beam level is added to the candidate
         pool (preserved across iterations, not discarded when the loop
         continues to deeper levels).
      2. Final ranking. After the beam exits (all reached, no children
         left, or max_rounds hit), BM25 re-scores the full candidate pool
         on doc_title+navigation_path+text+penjelasan (identical to
         bm25-flat for decision-point fairness) and returns top `top_k`.

    The pool accumulator avoids the earlier bug where leaves selected
    at mixed-depth intermediate levels were dropped when the iteration
    continued. With heterogeneous legal-document trees (some Bab skip
    Bagian directly to Pasal), this preserved every leaf the beam
    deemed promising.

    Args:
        query: Legal question in Indonesian.
        doc: Loaded document dict with structure field.
        top_k_per_level: Beam width during traversal.
        top_k: Final number of leaves to return after pool re-ranking.
        verbose: Print progress.

    Returns:
        Dict with steps (navigation trace), node_ids (final ranked leaves),
        and pool_size (number of leaves the beam reached, for diagnostics).
    """
    structure = doc["structure"]
    doc_title = doc.get("judul", "")
    steps = []
    max_rounds = 8

    candidate_pool: list[dict] = []
    seen_leaf_ids: set[str] = set()

    def _add_to_pool(leaves_to_add):
        for leaf in leaves_to_add:
            lid = leaf.get("node_id", "")
            if lid and lid not in seen_leaf_ids:
                candidate_pool.append(leaf)
                seen_leaf_ids.add(lid)

    current_nodes = structure
    round_num = 1

    while round_num <= max_rounds:
        all_leaves = all(not (n.get("nodes")) for n in current_nodes)

        if all_leaves:
            _add_to_pool(current_nodes)
            steps.append({
                "round": round_num,
                "level": f"level-{round_num}",
                "all_leaves": True,
                "options_shown": [n.get("title", "") for n in current_nodes],
                "added_to_pool": [n.get("node_id", "") for n in current_nodes],
            })
            if verbose:
                print(f"\n[BM25 Tree - Round {round_num}] All-leaves level reached; "
                      f"added {len(current_nodes)} leaves to candidate pool.")
            break

        selected = _bm25_level_search(query, current_nodes,
                                      top_k=top_k_per_level)
        if not selected:
            break

        steps.append({
            "round": round_num,
            "level": f"level-{round_num}",
            "all_leaves": False,
            "options_shown": [n.get("title", "") for n in current_nodes],
            "selected": [s["node_id"] for s in selected],
            "scores": {s["node_id"]: s["bm25_score"] for s in selected},
        })

        if verbose:
            print(f"\n[BM25 Tree - Round {round_num}] Beam selected:")
            for s in selected:
                print(f"  {s['node_id']} {s['title']} (BM25: {s['bm25_score']:.4f})")

        need_drill = []
        for s in selected:
            node_ref = s["_node_ref"]
            if s["has_children"]:
                need_drill.extend(node_ref.get("nodes", []))
            else:
                _add_to_pool([node_ref])

        if not need_drill:
            break

        current_nodes = need_drill
        round_num += 1

    if not candidate_pool:
        return {"steps": steps, "node_ids": [], "pool_size": 0}

    final_ranked = _bm25_leaf_search(query, candidate_pool, doc_title,
                                     top_k=top_k)
    final_ids = [s["node_id"] for s in final_ranked]

    if verbose:
        print(f"\n[BM25 Tree - Final Ranking] Re-ranked pool of "
              f"{len(candidate_pool)} leaves, returned top-{len(final_ids)}.")

    return {
        "steps": steps,
        "node_ids": final_ids,
        "pool_size": len(candidate_pool),
    }


def retrieve(query: str, top_k_per_level: int = 3, top_k: int = 10,
             verbose: bool = True) -> dict:
    """Full BM25 tree retrieval pipeline.

    1. BM25 doc search to select document.
    2. BM25 tree navigation level-by-level.
    3. Answer generation from retrieved leaf nodes.

    Args:
        query: Legal question in Indonesian.
        top_k_per_level: Beam width during traversal (paradigm choice).
        top_k: Final number of leaves to return (matches eval cutoff).
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
        print(f"Strategy: bm25-tree (top_k_per_level={top_k_per_level}, top_k={top_k})")
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
                              top_k=top_k, verbose=verbose)
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
                    help="Beam width during traversal (default: 3)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final number of leaves returned (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query, top_k_per_level=args.top_k_per_level,
                      top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > {src['navigation_path']}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
