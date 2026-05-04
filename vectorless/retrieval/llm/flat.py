"""LLM flat retrieval for Indonesian legal QA.

LLM ranks all leaf nodes across all documents in a single prompt. No tree
navigation. The LLM sees node metadata (node_id, title, doc_title,
navigation_path, summary) but not full text, so more candidates fit in
context. Output is a full RankGPT-style permutation of the input set.

Tests the hypothesis that for LLM retrieval, flat scaling fails as the
candidate count grows past the model's effective ranking ability. Compared
in the RQ1 matrix against `llm-agentic-doc` (hierarchical agentic) to
isolate the flat-vs-hierarchical axis.

When the LLM truncates or hallucinates, `validate_llm_ranking` drops bad
ids and appends missing candidates in input order so Recall@k semantics
remain well defined.

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
    load_all_leaf_nodes, save_log, validate_llm_ranking,
)


def flat_search(query: str, leaves: list[dict], max_candidates: int = 100,
                verbose: bool = True) -> dict:
    """Ask the LLM to rank a flat list of leaf nodes.

    Shows only metadata (no full text) so more candidates fit in context.
    If the corpus exceeds max_candidates, a random sample is shown and the
    rest are discarded for this call. With max_candidates large enough to
    cover all leaves (set by the eval harness), this becomes a pure
    full-permutation rerank over the entire leaf corpus.

    Args:
        query: Legal question in Indonesian.
        leaves: All leaf nodes from all documents.
        max_candidates: Max nodes shown to the LLM.
        verbose: Print progress.

    Returns:
        Dict with thinking, raw `ranking`, validated `ranked_node_ids`,
        candidates_shown, llm_ranking_length, validated_ranking_length.
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
    n_candidates = len(candidates_for_prompt)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan {n_candidates} Pasal kandidat dari berbagai Undang-Undang Indonesia.
Setiap kandidat memiliki ringkasan isi (summary) dan lokasi dalam dokumen.

Pertanyaan: {query}

Daftar Pasal kandidat:
{candidates_text}

Tugas: Urutkan SELURUH {n_candidates} kandidat dari paling relevan ke paling tidak relevan
untuk menjawab pertanyaan. Output harus berisi SEMUA {n_candidates} node_ids dari input,
tanpa duplikat dan tanpa node_id yang tidak ada di input.

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat tentang kriteria ranking>",
  "ranking": ["node_id_paling_relevan", "node_id_kedua", "...", "node_id_paling_tidak_relevan"]
}}

Aturan:
- "ranking" HARUS berisi tepat {n_candidates} node_ids
- "ranking" tidak boleh ada duplikat
- Setiap node_id harus muncul di input (tidak boleh hallucinate)
- Urutan menentukan ranking (index 0 = paling relevan)
- Pertimbangkan ISI summary, sumber UU (doc_title), dan navigation_path
- Kembalikan HANYA JSON, tanpa teks lain
"""

    result = llm_call(prompt)

    raw_ranking = result.get("ranking", [])
    validated = validate_llm_ranking(raw_ranking, candidates_for_prompt)
    result["ranked_node_ids"] = validated
    result["candidates_shown"] = len(sampled)
    result["llm_ranking_length"] = len(raw_ranking)
    result["validated_ranking_length"] = len(validated)

    if verbose:
        print(f"\n[LLM Flat] Ranked {len(validated)} candidates "
              f"(LLM returned {len(raw_ranking)} of {n_candidates})")
        if result.get("thinking"):
            print(f"  Reasoning: {result['thinking'][:200]}")

    return result


def retrieve(query: str, max_candidates: int = 100, verbose: bool = True) -> dict:
    """Full LLM flat retrieval pipeline.

    1. Load all leaf nodes.
    2. LLM full-ranks the flat list (sampling only if corpus > max_candidates).
    3. Validate ranking, append missing ids in input order, build sources.

    Args:
        query: Legal question in Indonesian.
        max_candidates: Cap on candidates shown. Eval harness passes a very
            large value so the entire corpus is shown.
        verbose: Print progress.

    Returns:
        Dict with query, strategy, search results, sources, and metrics.
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
    ranked_ids = search_result.get("ranked_node_ids", [])
    steps["flat_search"] = step_metrics(t_step, snap)

    if not ranked_ids:
        return {"query": query, "strategy": "llm-flat",
                "error": "LLM returned no ranking"}

    leaf_map = {leaf["node_id"]: leaf for leaf in leaves}
    ranked_results = [leaf_map[nid] for nid in ranked_ids if nid in leaf_map]

    if not ranked_results:
        return {"query": query, "strategy": "llm-flat",
                "node_ids": ranked_ids, "error": "Ranked nodes not found in corpus"}

    sources = []
    for pos, r in enumerate(ranked_results):
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r.get("title", ""),
            "navigation_path": r.get("navigation_path", ""),
            "rerank_position": pos,
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
