"""Hybrid flat retrieval for Indonesian legal QA.

BM25 global search across all leaf nodes, followed by LLM reranking.
No tree structure is used. This is the flat variant of hybrid retrieval.

Stage 1: BM25 search across ALL leaf nodes (same corpus as bm25-flat).
Stage 2: LLM reranks top-K BM25 candidates using full leaf text (capped at 5000 chars).

Unlike the tree-based hybrid strategy that navigates within a selected doc,
this variant searches the full leaf node corpus directly, eliminating the
doc selection bottleneck where a wrong doc pick causes total miss.

Usage:
    python -m vectorless.retrieval.hybrid.flat "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.flat "Apa syarat penyadapan?" --bm25_top_k 20
"""

import argparse
import json
import time

from rank_bm25 import BM25Okapi

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_all_leaf_nodes, save_log,
    validate_llm_ranking,
)


def flat_bm25_candidates(query: str, leaves: list[dict], top_k: int = 20,
                         verbose: bool = True) -> list[dict]:
    """Return BM25-ranked leaf candidates."""
    corpus = []
    for leaf in leaves:
        enriched = leaf["doc_title"] + " " + leaf["navigation_path"] + " " + leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            enriched += " " + leaf["penjelasan"]
        corpus.append(tokenize(enriched))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    candidates = []
    for idx, score in ranked[:top_k]:
        if score <= 0:
            continue
        leaf = leaves[idx]
        candidates.append({
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "node_id": leaf["node_id"],
            "title": leaf["title"],
            "navigation_path": leaf["navigation_path"],
            "text": leaf["text"],
            "penjelasan": leaf.get("penjelasan"),
            "summary": leaf.get("summary", ""),
            "bm25_score": round(float(score), 4),
        })

    if verbose:
        print(f"\n[Hybrid-Flat BM25] Top {len(candidates)} candidates:")
        for c in candidates:
            print(f"  {c['node_id']} {c['title']} (BM25: {c['bm25_score']:.4f})")
            print(f"    doc: {c['doc_id']}  path: {c['navigation_path']}")

    return candidates


CANDIDATE_TEXT_CAP = 5000


def llm_rerank(query: str, candidates: list[dict]) -> dict:
    """Ask the LLM to rank all candidates from most to least relevant.

    Uses RankGPT-style full permutation generation (Sun et al. 2023, EMNLP) so
    the output is a complete ordering over the input set. Each candidate
    exposes summary plus full text and penjelasan (capped at 5000 chars to
    match agentic read budget) so the reranker has access to the same content
    that BM25 stage 1 scored over, ensuring fair stage-1 vs stage-2 comparison.
    The caller validates via `validate_llm_ranking` to drop hallucinations
    and append missing IDs.
    """
    candidates_for_prompt = []
    for c in candidates:
        entry = {
            "node_id": c["node_id"],
            "doc_title": c["doc_title"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "summary": c.get("summary", ""),
            "text": (c.get("text") or "")[:CANDIDATE_TEXT_CAP],
        }
        penjelasan = c.get("penjelasan")
        if penjelasan and penjelasan != "Cukup jelas.":
            entry["penjelasan"] = penjelasan[:CANDIDATE_TEXT_CAP]
        candidates_for_prompt.append(entry)

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)
    n_candidates = len(candidates)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan {n_candidates} Pasal kandidat dari berbagai Undang-Undang.
Setiap kandidat memiliki ringkasan (summary), isi teks (text), dan penjelasan resmi (jika ada).

Pertanyaan: {query}

Kandidat Pasal (diurutkan berdasarkan kecocokan kata kunci awal):
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
- Pertimbangkan ISI text dan penjelasan, summary, sumber UU (doc_title), dan navigation_path
- Kembalikan HANYA JSON
"""

    return llm_call(prompt)


def retrieve(query: str, bm25_top_k: int = 20, verbose: bool = True) -> dict:
    """Run the global BM25 plus LLM-rerank pipeline for one query."""
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid-flat (bm25_top_k={bm25_top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    candidates = flat_bm25_candidates(query, leaves, top_k=bm25_top_k, verbose=verbose)
    steps["bm25_search"] = step_metrics(t_step, snap)

    if not candidates:
        return {"query": query, "strategy": "hybrid-flat",
                "error": "No BM25 candidates found"}

    snap = snapshot_counters()
    t_step = time.time()

    rerank_result = llm_rerank(query, candidates)
    raw_ranking = rerank_result.get("ranking", [])
    ranked_ids = validate_llm_ranking(raw_ranking, candidates)
    rerank_result["validated_ranking"] = ranked_ids
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(ranked_ids)

    steps["rerank"] = step_metrics(t_step, snap)

    if verbose:
        print(f"\n[Hybrid-Flat LLM Rerank] Ranked {len(ranked_ids)} candidates")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    candidate_map = {c["node_id"]: c for c in candidates}
    ranked_results = [candidate_map[nid] for nid in ranked_ids if nid in candidate_map]

    sources = []
    for pos, r in enumerate(ranked_results):
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "bm25_score": r["bm25_score"],
            "rerank_position": pos,
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid-flat",
        "corpus_size": len(leaves),
        "bm25_candidates": [{
            "node_id": c["node_id"],
            "doc_id": c["doc_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "bm25_score": c["bm25_score"],
        } for c in candidates],
        "rerank_result": rerank_result,
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
    ap = argparse.ArgumentParser(
        description="Hybrid-flat retrieval (BM25 global + LLM rerank) for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates for LLM reranking (default: 20)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']} (BM25: {src['bm25_score']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
