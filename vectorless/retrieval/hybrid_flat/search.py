"""
Hybrid-flat retrieval: BM25 global search + LLM rerank for Indonesian legal QA.

Stage 1: BM25 search across ALL leaf nodes (same corpus as bm25-flat)
Stage 2: LLM reranks top-K BM25 candidates using KWIC text snippets
Stage 3: Answer generation from reranked results (multi-doc)

Unlike the catalog-based "hybrid" strategy that searches only 1 doc,
this variant searches the full leaf node corpus directly — eliminating
the doc selection bottleneck where a wrong doc pick causes total miss.

Usage:
    python -m vectorless.retrieval.hybrid_flat.search "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid_flat.search "query" --bm25_top_k 20
"""

import argparse
import json
import time

from rank_bm25 import BM25Okapi

from ..common import (
    tokenize, llm_call, reset_token_counters, get_token_stats,
    snapshot_token_counters, compute_step_metrics,
    load_all_leaf_nodes, extract_kwic_snippet,
    generate_answer_multi_doc, save_log,
)


# ============================================================
# BM25 GLOBAL CANDIDATES
# ============================================================

def flat_bm25_candidates(query: str, leaves: list[dict], top_k: int = 20,
                         verbose: bool = True) -> list[dict]:
    """BM25 search across ALL leaf nodes, returning candidates with KWIC snippets.

    Same corpus-building as bm25/flat.py but returns enriched candidates
    (with snippets) for LLM reranking instead of final results.
    """
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
        snippet = extract_kwic_snippet(leaf["text"], query)
        candidates.append({
            "doc_id": leaf["doc_id"],
            "doc_title": leaf["doc_title"],
            "node_id": leaf["node_id"],
            "title": leaf["title"],
            "navigation_path": leaf["navigation_path"],
            "text": leaf["text"],
            "penjelasan": leaf.get("penjelasan"),
            "bm25_score": round(float(score), 4),
            "snippet": snippet,
        })

    if verbose:
        print(f"\n[Hybrid-Flat BM25] Top {len(candidates)} candidates:")
        for c in candidates:
            print(f"  {c['node_id']} {c['title']} (BM25: {c['bm25_score']:.4f})")
            print(f"    doc: {c['doc_id']}  path: {c['navigation_path']}")

    return candidates


# ============================================================
# LLM RERANK (multi-doc aware)
# ============================================================

def llm_rerank(query: str, candidates: list[dict]) -> dict:
    """LLM reranks BM25 candidates from multiple documents using KWIC snippets.

    Unlike hybrid's _llm_rerank which works within a single doc, this version
    shows doc_title per candidate so the LLM can reason across documents.
    """
    candidates_for_prompt = []
    for c in candidates:
        candidates_for_prompt.append({
            "node_id": c["node_id"],
            "doc_title": c["doc_title"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "snippet": c["snippet"],
        })

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan daftar Pasal kandidat dari berbagai Undang-Undang.
Setiap kandidat memiliki cuplikan teks (snippet) dari isinya.

Pertanyaan: {query}

Kandidat Pasal (diurutkan berdasarkan kecocokan kata kunci):
{candidates_text}

Pilih Pasal yang paling relevan untuk menjawab pertanyaan.

Balas dalam format JSON:
{{
  "thinking": "<penalaran mengapa Pasal ini paling relevan berdasarkan snippet>",
  "selected_ids": ["node_id1", "node_id2"]
}}

Aturan:
- Pilih berdasarkan ISI snippet, bukan hanya judul
- Pilih Pasal yang benar-benar menjawab pertanyaan (biasanya 1-3 Pasal)
- Jika beberapa Pasal saling melengkapi, pilih semuanya
- Perhatikan sumber UU (doc_title) — pilih dari UU yang paling relevan
- Kembalikan HANYA JSON
"""

    return llm_call(prompt)


# ============================================================
# MAIN PIPELINE
# ============================================================

def retrieve(query: str, bm25_top_k: int = 20, verbose: bool = True) -> dict:
    """Full hybrid-flat pipeline: BM25 global → LLM rerank → answer.

    Args:
        query: Legal question in Indonesian
        bm25_top_k: Max BM25 candidates for LLM reranking (default: 20)
        verbose: Print progress
    """
    reset_token_counters()
    t_start = time.time()
    step_metrics = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid-flat (bm25_top_k={bm25_top_k})")
        print(f"{'='*60}")

    # Step 1: BM25 global search
    snap = snapshot_token_counters()
    t_step = time.time()

    leaves = load_all_leaf_nodes()
    if verbose:
        print(f"\nCorpus: {len(leaves)} leaf nodes from all documents")

    candidates = flat_bm25_candidates(query, leaves, top_k=bm25_top_k, verbose=verbose)
    step_metrics["bm25_search"] = compute_step_metrics(t_step, snap)

    if not candidates:
        return {"query": query, "strategy": "hybrid-flat",
                "error": "No BM25 candidates found"}

    # Step 2: LLM rerank
    snap = snapshot_token_counters()
    t_step = time.time()

    rerank_result = llm_rerank(query, candidates)
    selected_ids = rerank_result.get("selected_ids", [])

    # Guard: only keep IDs that exist in candidates
    valid_ids = {c["node_id"] for c in candidates}
    selected_ids = [nid for nid in selected_ids if nid in valid_ids]

    # Fallback: if LLM returns empty/invalid, use top BM25 result
    if not selected_ids:
        selected_ids = [candidates[0]["node_id"]]

    step_metrics["rerank"] = compute_step_metrics(t_step, snap)

    if verbose:
        print(f"\n[Hybrid-Flat LLM Rerank] Selected: {selected_ids}")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    # Filter candidates to selected ones (preserve BM25 order for selected)
    selected_map = {c["node_id"]: c for c in candidates}
    selected_results = [selected_map[nid] for nid in selected_ids if nid in selected_map]

    # Step 3: Generate answer (multi-doc)
    snap = snapshot_token_counters()
    t_step = time.time()

    answer_result = generate_answer_multi_doc(query, selected_results, verbose=verbose)
    step_metrics["answer_gen"] = compute_step_metrics(t_step, snap)

    # Build sources
    sources = []
    for r in selected_results:
        sources.append({
            "doc_id": r["doc_id"],
            "node_id": r["node_id"],
            "title": r["title"],
            "navigation_path": r["navigation_path"],
            "bm25_score": r["bm25_score"],
        })

    elapsed = time.time() - t_start
    stats = get_token_stats()

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
        "answer": answer_result.get("answer", ""),
        "citations": answer_result.get("citations", []),
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": step_metrics},
    }

    save_log(result)

    if verbose:
        print(f"\n{'='*60}")
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens")
        for step_name, sm in step_metrics.items():
            print(f"  {step_name}: {sm['elapsed_s']:.1f}s, {sm['llm_calls']} calls, "
                  f"{sm['input_tokens']+sm['output_tokens']:,} tokens")
        print(f"{'='*60}")

    return result


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Hybrid-flat retrieval (BM25 global + LLM rerank) for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates for LLM reranking (default: 20)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > [{src['doc_id']}] {src['navigation_path']} (BM25: {src['bm25_score']})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
