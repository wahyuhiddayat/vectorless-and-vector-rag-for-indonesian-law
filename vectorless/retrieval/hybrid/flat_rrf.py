"""Hybrid flat retrieval with RRF fusion of BM25 and LLM rankings.

Stage 1: BM25 global search across all leaf nodes (top-N candidates).
Stage 2: LLM listwise reranks the same N candidates.
Stage 3: Reciprocal Rank Fusion (Cormack et al. 2009) of BM25 rank
         and LLM rank. Final order = sorted by RRF score.

Difference vs `hybrid-flat`:
  - hybrid-flat: LLM rerank consumes BM25 ranking (LLM ordering = final)
  - hybrid-flat-rrf: BM25 ranking AND LLM ranking are independent signals,
                    fused via score sum 1/(k+rank_bm25) + 1/(k+rank_llm)

This variant represents the classic academic IR hybrid paradigm
(score fusion) versus the modern production RAG cascade rerank paradigm.

Usage:
    python -m vectorless.retrieval.hybrid.flat_rrf "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.flat_rrf "Apa syarat penyadapan?" --bm25_top_k 20 --k_rrf 60
"""

import argparse
import random
import time

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import load_all_leaf_nodes, save_log, validate_llm_ranking
from .flat import flat_bm25_candidates, llm_rerank


def rrf_fuse(rankings_a: list[str], rankings_b: list[str],
             k_rrf: int = 60, top_k: int = 10) -> tuple[list[str], dict[str, float]]:
    """Reciprocal Rank Fusion of two ranked id lists.

    Each id receives score 1/(k_rrf + rank) from each ranking it appears in.
    Items present in only one ranking still receive that ranking's contribution.
    Ties are broken by appearance order in `rankings_a` so the output is
    deterministic.

    Args:
        rankings_a: first ranked id list, rank 1 = best.
        rankings_b: second ranked id list, rank 1 = best.
        k_rrf: RRF dampening constant. 60 is the original Cormack et al. value
            and is the de-facto standard in IR.
        top_k: final cut.

    Returns:
        (fused_top_k_ids, score_map). score_map covers every id in either input.
    """
    scores: dict[str, float] = {}
    for rank, nid in enumerate(rankings_a, start=1):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rank)
    for rank, nid in enumerate(rankings_b, start=1):
        scores[nid] = scores.get(nid, 0.0) + 1.0 / (k_rrf + rank)
    order_a = {nid: i for i, nid in enumerate(rankings_a)}
    ordered = sorted(
        scores.keys(),
        key=lambda x: (-scores[x], order_a.get(x, len(rankings_a))),
    )
    return ordered[:top_k], scores


def retrieve(query: str, bm25_top_k: int = 20, k_rrf: int = 60,
             top_k: int = 10, verbose: bool = True) -> dict:
    """Run the BM25 + LLM rerank + RRF fusion pipeline for one query."""
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid-flat-rrf "
              f"(bm25_top_k={bm25_top_k}, k_rrf={k_rrf}, top_k={top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    candidates = flat_bm25_candidates(query, leaves, top_k=bm25_top_k, verbose=verbose)
    steps["bm25_search"] = step_metrics(t_step, snap)

    if not candidates:
        return {"query": query, "strategy": "hybrid-flat-rrf",
                "error": "No BM25 candidates found"}

    # Compound ref "doc_id/node_id" is the unique key. Plain node_id is NOT
    # unique across docs at pasal granularity (e.g. many docs have pasal_1),
    # so RRF over plain node_id would conflate distinct items and sum scores.
    def _ref(c: dict) -> str:
        return f"{c['doc_id']}/{c['node_id']}"

    # BM25 ranking preserved from BM25 score order (already sorted descending).
    bm25_ranking_refs = [_ref(c) for c in candidates]

    snap = snapshot_counters()
    t_step = time.time()

    # Shuffle before LLM to mitigate BM25-order anchor bias on the LLM rerank.
    shuffled = list(candidates)
    random.shuffle(shuffled)

    rerank_result = llm_rerank(query, shuffled)

    raw_ranking = rerank_result.get("ranking", [])
    # LLM returns node_ids. Map back to refs via shuffle order; ambiguous
    # node_ids (duplicate across docs in the chunk) resolve to the candidate
    # they appeared in within the prompt order. validate_llm_ranking dedupes
    # at node_id level which is acceptable here because the LLM only saw one
    # text per node_id occurrence and its semantic intent is the chosen ref.
    valid_ids = {c["node_id"] for c in candidates}
    n_hallucinated = sum(1 for nid in raw_ranking if nid not in valid_ids)
    llm_node_ranking = validate_llm_ranking(raw_ranking, candidates)

    # Resolve LLM node-id ranking to refs by walking candidates in prompt order.
    # If a node_id maps to multiple refs (cross-doc duplicate), pick the first
    # unvisited ref. This is rare for the small bm25_top_k=20 pool.
    node_to_refs: dict[str, list[str]] = {}
    for c in candidates:
        node_to_refs.setdefault(c["node_id"], []).append(_ref(c))
    used_refs: set[str] = set()
    llm_ranking_refs: list[str] = []
    for nid in llm_node_ranking:
        for ref in node_to_refs.get(nid, []):
            if ref not in used_refs:
                llm_ranking_refs.append(ref)
                used_refs.add(ref)
                break
    # Pad missing refs in BM25 order so both rankings span the same candidate set.
    for ref in bm25_ranking_refs:
        if ref not in used_refs:
            llm_ranking_refs.append(ref)
            used_refs.add(ref)

    rerank_result["validated_ranking"] = llm_ranking_refs
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(llm_ranking_refs)
    rerank_result["n_hallucinated"] = n_hallucinated

    steps["rerank"] = step_metrics(t_step, snap)

    # Stage 3 RRF fusion over compound refs.
    fused_refs, rrf_scores = rrf_fuse(
        bm25_ranking_refs, llm_ranking_refs, k_rrf=k_rrf, top_k=top_k,
    )

    if verbose:
        print(f"\n[RRF Fusion] k_rrf={k_rrf}, top_k={top_k}")
        for pos, ref in enumerate(fused_refs[:top_k]):
            print(f"  rank {pos+1}: {ref}  rrf_score={rrf_scores.get(ref, 0):.6f}")

    candidate_map = {_ref(c): c for c in candidates}
    bm25_rank_map = {ref: i + 1 for i, ref in enumerate(bm25_ranking_refs)}
    llm_rank_map = {ref: i + 1 for i, ref in enumerate(llm_ranking_refs)}

    sources = []
    for pos, ref in enumerate(fused_refs):
        c = candidate_map.get(ref)
        if not c:
            continue
        sources.append({
            "doc_id": c["doc_id"],
            "node_id": c["node_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "bm25_score": c["bm25_score"],
            "bm25_rank": bm25_rank_map.get(ref),
            "llm_rank": llm_rank_map.get(ref),
            "rrf_score": round(rrf_scores.get(ref, 0.0), 6),
            "rerank_position": pos,
        })

    if not sources:
        return {"query": query, "strategy": "hybrid-flat-rrf",
                "error": "RRF fusion produced no sources"}

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid-flat-rrf",
        "corpus_size": len(leaves),
        "bm25_candidates": [{
            "node_id": c["node_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "bm25_score": c["bm25_score"],
        } for c in candidates],
        "rerank_result": rerank_result,
        "rrf": {
            "k_rrf": k_rrf,
            "top_k": top_k,
            "fused_refs": fused_refs,
            "score_map": {nid: round(s, 6) for nid, s in rrf_scores.items()},
        },
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": steps},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        for step_name, sm in steps.items():
            print(f"  {step_name}: {sm['elapsed_s']:.1f}s, {sm['llm_calls']} calls, "
                  f"{sm['input_tokens']+sm['output_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


def main():
    """CLI entry point for hybrid-flat-rrf retrieval."""
    ap = argparse.ArgumentParser(
        description="Hybrid-flat retrieval with RRF fusion of BM25 and LLM rankings.")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates for LLM reranking (default: 20)")
    ap.add_argument("--k_rrf", type=int, default=60,
                    help="RRF dampening constant (default: 60, Cormack et al.)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final number of leaves to return (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k,
                      k_rrf=args.k_rrf, top_k=args.top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}  "
              f"(rrf={src['rrf_score']}, bm25_rank={src['bm25_rank']}, llm_rank={src['llm_rank']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
