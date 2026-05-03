"""LLM flat retrieval for Indonesian legal QA.

LLM selects relevant nodes from a flat list of all leaf nodes across all
documents. No tree navigation is performed. The LLM sees node metadata
(node_id, title, doc_title, navigation_path, summary) but not full text,
allowing more candidates to fit in the context window.

This provides a baseline to compare against LLM-Tree and measure whether
tree-based navigation adds value for LLM-driven retrieval.

Usage:
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?" --max_candidates 100
"""

import argparse
import json
import random
import time

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_all_leaf_nodes, save_log,
)


def flat_search(query: str, leaves: list[dict], max_candidates: int = 100,
                verbose: bool = True) -> dict:
    """Have the LLM select relevant nodes from a flat list.

    Shows only metadata (no full text) to fit more candidates in context.
    If corpus exceeds max_candidates, random samples are taken.

    Args:
        query: Legal question in Indonesian.
        leaves: All leaf nodes from all documents.
        max_candidates: Max nodes to show the LLM.
        verbose: Print progress.

    Returns:
        Dict with thinking, selected_ids, and candidates_shown.
    """
    if len(leaves) > max_candidates:
        sampled = random.sample(leaves, max_candidates)
        if verbose:
            print(f"\n[LLM Flat] Sampled {max_candidates} from {len(leaves)} leaves")
    else:
        sampled = leaves
        if verbose:
            print(f"\n[LLM Flat] Showing all {len(leaves)} leaves")

    candidates_for_prompt = []
    for leaf in sampled:
        candidates_for_prompt.append({
            "node_id": leaf["node_id"],
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "title": leaf.get("title", ""),
            "navigation_path": leaf.get("navigation_path", ""),
            "summary": leaf.get("summary", ""),
        })

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan daftar Pasal dari berbagai Undang-Undang Indonesia.
Setiap kandidat memiliki ringkasan isi (summary) dan lokasi dalam dokumen.

Pertanyaan: {query}

Daftar Pasal kandidat:
{candidates_text}

Pilih Pasal yang paling relevan untuk menjawab pertanyaan.

Balas dalam format JSON:
{{
  "thinking": "<penalaran mengapa Pasal ini relevan berdasarkan summary dan konteks UU>",
  "selected_ids": ["node_id1", "node_id2"]
}}

Aturan:
- Pilih berdasarkan ISI summary, bukan hanya judul
- Pilih Pasal yang benar-benar menjawab pertanyaan (biasanya 1-5 Pasal)
- Perhatikan sumber UU (doc_title) dan navigation_path untuk konteks
- Jika beberapa Pasal saling melengkapi, pilih semuanya
- Kembalikan HANYA JSON, tanpa teks lain
"""

    result = llm_call(prompt)

    valid_ids = {c["node_id"] for c in candidates_for_prompt}
    selected_ids = [nid for nid in result.get("selected_ids", []) if nid in valid_ids]
    result["selected_ids"] = selected_ids
    result["candidates_shown"] = len(sampled)

    if verbose:
        print(f"\n[LLM Flat] Selected: {selected_ids}")
        if result.get("thinking"):
            print(f"  Reasoning: {result['thinking'][:200]}")

    return result


def retrieve(query: str, max_candidates: int = 100, verbose: bool = True) -> dict:
    """Full LLM flat retrieval pipeline.

    1. Load all leaf nodes.
    2. LLM selects from flat list (with sampling if too large).
    3. Answer generation from selected nodes.

    Args:
        query: Legal question in Indonesian.
        max_candidates: Max candidates to show the LLM.
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
        print(f"Strategy: llm-flat (max_candidates={max_candidates})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    search_result = flat_search(query, leaves, max_candidates=max_candidates,
                                verbose=verbose)
    selected_ids = search_result.get("selected_ids", [])
    steps["flat_search"] = step_metrics(t_step, snap)

    if not selected_ids:
        return {"query": query, "strategy": "llm-flat",
                "error": "No relevant nodes selected"}

    leaf_map = {leaf["node_id"]: leaf for leaf in leaves}
    selected_results = [leaf_map[nid] for nid in selected_ids if nid in leaf_map]

    if not selected_results:
        return {"query": query, "strategy": "llm-flat",
                "node_ids": selected_ids, "error": "Selected nodes not found in corpus"}

    sources = []
    for r in selected_results:
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r.get("title", ""),
            "navigation_path": r.get("navigation_path", ""),
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "llm-flat",
        "corpus_size": len(leaves),
        "flat_search": search_result,
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
    """CLI entry point for LLM flat retrieval."""
    ap = argparse.ArgumentParser(
        description="LLM flat (no tree) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--max_candidates", type=int, default=100,
                    help="Max candidates to show LLM (default: 100)")
    args = ap.parse_args()

    result = retrieve(args.query, max_candidates=args.max_candidates)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
