"""LLM flat retrieval for Indonesian legal QA via chunked listwise reranking.

LLM ranks all leaf nodes across all documents using RankGPT-style listwise
reranking (Sun et al. 2023 EMNLP) generalized to handle full-corpus ranking
via chunked elimination. Each leaf is seen by the LLM at least once. After
each round, the top `LISTWISE_SURVIVORS` from each chunk advance to the
next round; rounds repeat until at most `LISTWISE_WINDOW` candidates remain
for a final ranking pass that produces the top-k output.

The LLM sees node metadata (node_id, title, doc_title, navigation_path,
summary) but not full text, so more candidates fit per prompt. Output is
RankGPT-style permutation per chunk; the validate_llm_ranking helper
drops hallucinations and appends missing IDs in input order so Recall@k
semantics remain well defined per chunk.

This replaces the earlier single-prompt design which silently failed at
rincian granularity because the full corpus (~38K leaves, ~2M+ tokens)
exceeded Gemini's 1M context window.

Usage:
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.llm.flat "Apa syarat penyadapan?" --window_size 200
"""

import argparse
import json
import time

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_all_leaf_nodes, save_log, validate_llm_ranking,
)


LISTWISE_WINDOW = 200      # candidates per LLM ranking call
LISTWISE_SURVIVORS = 20    # top-K kept from each chunk to advance to next round


def _build_rank_prompt(query: str, candidates: list[dict]) -> str:
    """Build the listwise ranking prompt for one chunk of candidates."""
    n = len(candidates)
    candidates_text = json.dumps(candidates, ensure_ascii=False, indent=2)
    return f"""\
Kamu diberi pertanyaan hukum dan {n} Pasal kandidat dari berbagai Undang-Undang Indonesia.
Setiap kandidat memiliki ringkasan isi (summary) dan lokasi dalam dokumen.

Pertanyaan: {query}

Daftar Pasal kandidat:
{candidates_text}

Tugas: Urutkan SELURUH {n} kandidat dari paling relevan ke paling tidak relevan
untuk menjawab pertanyaan. Output harus berisi SEMUA {n} node_ids dari input,
tanpa duplikat dan tanpa node_id yang tidak ada di input.

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat tentang kriteria ranking>",
  "ranking": ["node_id_paling_relevan", "node_id_kedua", "...", "node_id_paling_tidak_relevan"]
}}

Aturan:
- "ranking" HARUS berisi tepat {n} node_ids
- "ranking" tidak boleh ada duplikat
- Setiap node_id harus muncul di input (tidak boleh hallucinate)
- Urutan menentukan ranking (index 0 = paling relevan)
- Pertimbangkan ISI summary, sumber UU (doc_title), dan navigation_path
- Kembalikan HANYA JSON, tanpa teks lain
"""


def _rank_chunk(query: str, chunk: list[dict]) -> list[dict]:
    """LLM-rank a single chunk of leaves and return them reordered.

    Missing or hallucinated ids are handled by `validate_llm_ranking`, which
    appends missing candidates in input order. The returned list always has
    the same length and members as the input chunk, only the order changes.
    """
    candidates_for_prompt = [
        {
            "node_id": leaf["node_id"],
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "title": leaf.get("title", ""),
            "navigation_path": leaf.get("navigation_path", ""),
            "summary": leaf.get("summary", ""),
        }
        for leaf in chunk
    ]

    prompt = _build_rank_prompt(query, candidates_for_prompt)
    result = llm_call(prompt)

    raw_ranking = result.get("ranking", []) if isinstance(result, dict) else []
    validated_ids = validate_llm_ranking(raw_ranking, candidates_for_prompt)

    leaf_map = {leaf["node_id"]: leaf for leaf in chunk}
    return [leaf_map[nid] for nid in validated_ids if nid in leaf_map]


def flat_search(query: str, leaves: list[dict],
                window_size: int = LISTWISE_WINDOW,
                survivors_per_chunk: int = LISTWISE_SURVIVORS,
                top_k: int = 10, verbose: bool = True) -> dict:
    """Chunked listwise LLM reranking over the full leaf corpus.

    Algorithm:
      1. Split candidates into chunks of `window_size`.
      2. LLM ranks each chunk; the top `survivors_per_chunk` advance.
      3. Repeat until len(candidates) <= window_size.
      4. Final pass: LLM ranks the remaining set; return top-k.

    For small corpora (len(leaves) <= window_size), this collapses to a
    single LLM call, matching the original llm-flat behavior.

    Args:
        query: Legal question in Indonesian.
        leaves: All leaf nodes from all documents.
        window_size: Max candidates per LLM call (default 200).
        survivors_per_chunk: Top-K kept per chunk per round (default 20).
        top_k: Final number of leaves to return (default 10).
        verbose: Print progress per round.

    Returns:
        Dict with ranked_node_ids (top-k), candidates_shown (input size),
        validated_ranking_length, rounds (per-round chunk count + survivors),
        and total_llm_calls (sum across rounds).
    """
    candidates = list(leaves)
    rounds_info: list[dict] = []
    total_calls = 0

    if verbose:
        print(f"\n[LLM Flat] Starting with {len(candidates)} leaves")

    while len(candidates) > window_size:
        chunks = [candidates[i:i + window_size]
                  for i in range(0, len(candidates), window_size)]
        new_candidates: list[dict] = []
        for chunk in chunks:
            ranked = _rank_chunk(query, chunk)
            new_candidates.extend(ranked[:survivors_per_chunk])
            total_calls += 1

        rounds_info.append({
            "round": len(rounds_info) + 1,
            "chunks": len(chunks),
            "input_size": sum(len(c) for c in chunks),
            "survivors": len(new_candidates),
            "calls": len(chunks),
        })
        if verbose:
            print(f"  Round {len(rounds_info)}: "
                  f"{len(chunks)} chunks of {window_size} -> "
                  f"{len(new_candidates)} survivors")

        candidates = new_candidates

    # Final pass when remaining set fits in one window
    if len(candidates) > 1:
        candidates = _rank_chunk(query, candidates)
        total_calls += 1
        rounds_info.append({
            "round": len(rounds_info) + 1,
            "chunks": 1,
            "input_size": len(candidates),
            "survivors": len(candidates),
            "calls": 1,
            "final": True,
        })
        if verbose:
            print(f"  Final round: {len(candidates)} candidates -> top-{top_k}")

    final = candidates[:top_k]

    return {
        "ranked_node_ids": [leaf["node_id"] for leaf in final],
        "candidates_shown": len(leaves),
        "validated_ranking_length": len(final),
        "rounds": rounds_info,
        "total_llm_calls": total_calls,
    }


def retrieve(query: str, window_size: int = LISTWISE_WINDOW,
             survivors_per_chunk: int = LISTWISE_SURVIVORS,
             top_k: int = 10, verbose: bool = True) -> dict:
    """Full LLM flat retrieval pipeline.

    1. Load all leaf nodes.
    2. Chunked listwise LLM reranking via tournament elimination.
    3. Build sources from final top-k ranking.

    Args:
        query: Legal question in Indonesian.
        window_size: Max candidates per LLM call.
        survivors_per_chunk: Top-K kept per chunk per round.
        top_k: Final number of leaves to return.
        verbose: Print progress.

    Returns:
        Dict with query, strategy, search results, sources, and metrics.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'=' * 60}")
        print(f"Query: {query}")
        print(f"Strategy: llm-flat "
              f"(window_size={window_size}, survivors={survivors_per_chunk}, "
              f"top_k={top_k})")
        print(f"{'=' * 60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    search_result = flat_search(query, leaves,
                                window_size=window_size,
                                survivors_per_chunk=survivors_per_chunk,
                                top_k=top_k, verbose=verbose)
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
        print(f"\n{'=' * 60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        print(f"{'=' * 60}")

    return result


def main():
    """CLI entry point for LLM flat retrieval."""
    ap = argparse.ArgumentParser(
        description="LLM flat (chunked listwise) retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--window_size", type=int, default=LISTWISE_WINDOW,
                    help=f"Max candidates per LLM call (default: {LISTWISE_WINDOW})")
    ap.add_argument("--survivors_per_chunk", type=int, default=LISTWISE_SURVIVORS,
                    help=f"Top-K kept per chunk per round (default: {LISTWISE_SURVIVORS})")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Number of final results to return (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query,
                      window_size=args.window_size,
                      survivors_per_chunk=args.survivors_per_chunk,
                      top_k=args.top_k)
    print(f"\n{'-' * 60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}")
    print(f"{'-' * 60}")


if __name__ == "__main__":
    main()
