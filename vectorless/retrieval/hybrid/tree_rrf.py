"""Hybrid tree-based retrieval with RRF fusion at the cross-doc rerank stage.

Stage 1: doc-pick (LLM + BM25 merge, identical to hybrid-tree).
Stage 2: per-doc BM25 candidates concatenated cross-doc (identical to hybrid-tree).
Stage 3: BM25 cross-doc ranking by raw bm25_score AND LLM listwise rerank,
         fused via Reciprocal Rank Fusion (Cormack et al. 2009) instead of
         taking the LLM ordering as final.

Difference vs `hybrid-tree`:
  - hybrid-tree:     LLM rerank ordering = final
  - hybrid-tree-rrf: RRF(BM25 cross-doc rank, LLM rerank rank) = final

This is the tree analogue of `hybrid-flat-rrf`. Tests whether the
cascade-vs-fusion tradeoff observed at flat granularity (R@10 slight
gain, H@1 substantial drop) replicates in the tree paradigm.

Usage:
    python -m vectorless.retrieval.hybrid.tree_rrf "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.tree_rrf "Apa syarat penyadapan?" --bm25_top_k 20 --k_rrf 60
"""

import argparse
import random
import time

from ...llm import reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_catalog, load_doc, save_log, validate_llm_ranking,
    DOC_PICK_TOP_K,
)
from .tree import doc_search, _bm25_node_candidates, _llm_rerank_multidoc
from .flat_rrf import rrf_fuse


def retrieve(query: str, bm25_top_k: int = 20, k_rrf: int = 60,
             top_k: int = 10, top_k_docs: int = DOC_PICK_TOP_K,
             verbose: bool = True) -> dict:
    """Run hybrid-tree-rrf pipeline. Stage 1 and Stage 2 identical to hybrid-tree.

    Stage 3 fuses two ranked lists via RRF instead of letting LLM cascade win:
      - bm25_ranking: cross-doc candidates sorted by raw per-doc bm25_score
      - llm_ranking: LLM listwise output over shuffled candidates
      - fused = rrf_fuse(bm25_ranking, llm_ranking, k_rrf=60)

    LLM call count: 2 per query (doc-pick + cross-doc rerank), identical
    to hybrid-tree.
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid-tree-rrf "
              f"(top_k_docs={top_k_docs}, bm25_top_k={bm25_top_k}, k_rrf={k_rrf})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_result = doc_search(query, catalog, top_k=top_k_docs, verbose=verbose)
    doc_ids = doc_result.get("doc_ids", [])
    steps["doc_search"] = step_metrics(t_step, snap)

    if not doc_ids:
        return {"query": query, "strategy": "hybrid-tree-rrf",
                "picked_doc_ids": [], "error": "No relevant documents found"}

    snap = snapshot_counters()
    t_step = time.time()

    all_candidates: list[dict] = []
    per_doc_candidate_counts: dict[str, int] = {}
    for did in doc_ids:
        doc = load_doc(did)
        cands = _bm25_node_candidates(query, doc, top_k=bm25_top_k)
        per_doc_candidate_counts[did] = len(cands)
        for c in cands:
            tagged = dict(c)
            tagged["doc_id"] = did
            tagged["doc_title"] = doc.get("judul", "")
            all_candidates.append(tagged)

    if verbose:
        print(f"\n[Node Search - BM25 Candidates] Multi-doc total {len(all_candidates)}:")
        for did, cnt in per_doc_candidate_counts.items():
            print(f"  {did}: {cnt} candidates")

    if not all_candidates:
        return {"query": query, "strategy": "hybrid-tree-rrf",
                "picked_doc_ids": doc_ids, "error": "No relevant nodes found"}

    # Stage 3a: BM25 cross-doc ranking by raw per-doc bm25_score. Per-doc
    # scores are comparable cross-doc because tokenization, query terms,
    # and enrichment fields (doc_title + nav_path + text + penjelasan) are
    # uniform across docs.
    bm25_sorted = sorted(
        all_candidates,
        key=lambda c: c.get("bm25_score", 0.0),
        reverse=True,
    )
    bm25_ranking_refs = [f"{c['doc_id']}/{c['node_id']}" for c in bm25_sorted]

    # Stage 3b: LLM listwise rerank, identical pattern to hybrid-tree.
    shuffled = list(all_candidates)
    random.shuffle(shuffled)
    rerank_result = _llm_rerank_multidoc(query, shuffled)
    raw_ranking = rerank_result.get("ranking", [])
    valid_refs = {f"{c['doc_id']}/{c['node_id']}" for c in all_candidates}
    n_hallucinated = sum(1 for r in raw_ranking if r not in valid_refs)

    pseudo_candidates = [
        {"node_id": f"{c['doc_id']}/{c['node_id']}"} for c in all_candidates
    ]
    llm_ranking_refs = validate_llm_ranking(raw_ranking, pseudo_candidates)
    rerank_result["validated_ranking"] = llm_ranking_refs
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(llm_ranking_refs)
    rerank_result["n_hallucinated"] = n_hallucinated
    steps["node_rerank"] = step_metrics(t_step, snap)

    if verbose:
        print(f"\n[LLM Rerank] Ranked {len(llm_ranking_refs)} candidates")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    # Stage 3c: RRF fusion of BM25 cross-doc and LLM rerank rankings.
    fused_refs, rrf_scores = rrf_fuse(
        bm25_ranking_refs, llm_ranking_refs, k_rrf=k_rrf, top_k=top_k,
    )

    if verbose:
        print(f"\n[RRF Fusion] k_rrf={k_rrf}, top_k={top_k}")
        for pos, ref in enumerate(fused_refs[:top_k]):
            print(f"  rank {pos+1}: {ref}  rrf_score={rrf_scores.get(ref, 0):.6f}")

    candidate_by_ref = {f"{c['doc_id']}/{c['node_id']}": c for c in all_candidates}
    bm25_rank_map = {ref: i + 1 for i, ref in enumerate(bm25_ranking_refs)}
    llm_rank_map = {ref: i + 1 for i, ref in enumerate(llm_ranking_refs)}

    sources = []
    for pos, ref in enumerate(fused_refs):
        c = candidate_by_ref.get(ref)
        if not c:
            continue
        sources.append({
            "doc_id": c["doc_id"],
            "node_id": c["node_id"],
            "title": c.get("title", ""),
            "navigation_path": c.get("navigation_path", ""),
            "bm25_score": c.get("bm25_score"),
            "bm25_rank": bm25_rank_map.get(ref),
            "llm_rank": llm_rank_map.get(ref),
            "rrf_score": round(rrf_scores.get(ref, 0.0), 6),
            "rerank_position": pos,
        })

    if not sources:
        return {"query": query, "strategy": "hybrid-tree-rrf",
                "picked_doc_ids": doc_ids,
                "error": "RRF fusion produced no sources"}

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid-tree-rrf",
        "picked_doc_ids": doc_ids,
        "doc_search": doc_result,
        "per_doc_candidate_counts": per_doc_candidate_counts,
        "merged_candidate_count": len(all_candidates),
        "llm_rerank": rerank_result,
        "rrf": {
            "k_rrf": k_rrf,
            "top_k": top_k,
            "fused_refs": fused_refs,
            "score_map": {ref: round(s, 6) for ref, s in rrf_scores.items()},
        },
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
    """CLI entry point for hybrid-tree-rrf retrieval."""
    ap = argparse.ArgumentParser(
        description="Hybrid-tree retrieval with RRF fusion of BM25 and LLM rankings.")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates per doc (default: 20)")
    ap.add_argument("--k_rrf", type=int, default=60,
                    help="RRF dampening constant (default: 60, Cormack 2009)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final number of leaves to return (default: 10)")
    ap.add_argument("--top_k_docs", type=int, default=DOC_PICK_TOP_K,
                    help=f"Number of docs picked at stage 1 (default: {DOC_PICK_TOP_K})")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k, k_rrf=args.k_rrf,
                      top_k=args.top_k, top_k_docs=args.top_k_docs)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']}  "
              f"(rrf={src['rrf_score']}, bm25_rank={src['bm25_rank']}, llm_rank={src['llm_rank']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
