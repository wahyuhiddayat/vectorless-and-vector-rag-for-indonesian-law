"""Hybrid tree-based retrieval for Indonesian legal QA.

Combines keyword matching (BM25) with LLM semantic understanding using the
document tree structure. This is the tree variant of hybrid retrieval.

Pipeline:
  1. Doc search  - union of BM25 metadata match + LLM semantic selection
  2. Node search - BM25 retrieves candidate Pasal within selected doc, LLM reranks

This addresses weaknesses of both pure approaches:
  - Pure BM25 fails on vocabulary mismatch (query term not in metadata)
  - Pure LLM fails on blind navigation (generic titles like "Pasal 3")
  - Hybrid: BM25 finds keyword-relevant content, LLM adds semantic understanding

Usage:
    python -m vectorless.retrieval.hybrid.tree "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.tree "Apa syarat penyadapan?" --bm25_top_k 20
"""

import argparse
import json
import time

from rank_bm25 import BM25Okapi

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_catalog, load_doc, find_node, extract_nodes,
    extract_kwic_snippet, save_log, validate_llm_ranking, DATA_INDEX,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = 3) -> list[dict]:
    """Rank catalog entries with BM25 over the metadata fields."""
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


def _llm_doc_search(query: str, catalog: list[dict]) -> list[dict]:
    """Ask the LLM to pick relevant documents from the catalog."""
    docs_text = json.dumps(catalog, ensure_ascii=False, indent=2)

    prompt = f"""\
Kamu diberi daftar Undang-Undang Indonesia beserta metadata-nya.
Pilih UU yang relevan untuk menjawab pertanyaan hukum berikut.

Pertanyaan: {query}

Daftar UU:
{docs_text}

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat mengapa UU tersebut relevan>",
  "doc_ids": ["doc_id_1", "doc_id_2"]
}}

Aturan:
- Pilih hanya UU yang benar-benar relevan (biasanya 1-2 saja)
- Jika tidak ada yang relevan, kembalikan doc_ids kosong: []
- Perhatikan bidang, subjek, dan materi_pokok untuk menentukan relevansi
- Kembalikan HANYA JSON, tanpa teks lain
"""
    return llm_call(prompt)


def doc_search(query: str, catalog: list[dict], verbose: bool = True) -> dict:
    """Merge BM25 catalog hits with the LLM's document picks."""
    bm25_results = _bm25_doc_search(query, catalog)
    llm_result = _llm_doc_search(query, catalog)

    bm25_ids = [r["doc_id"] for r in bm25_results]
    llm_ids = llm_result.get("doc_ids", [])

    valid_ids = {d["doc_id"] for d in catalog}
    llm_ids = [doc_id for doc_id in llm_ids if doc_id in valid_ids]

    seen = set()
    merged_ids = []
    for doc_id in llm_ids + bm25_ids:
        if doc_id not in seen:
            seen.add(doc_id)
            merged_ids.append(doc_id)

    bm25_scores = {r["doc_id"]: r["bm25_score"] for r in bm25_results}

    if verbose:
        print(f"\n[Doc Search - Hybrid]")
        print(f"  BM25 hits: {bm25_ids}")
        print(f"  LLM picks: {llm_ids}")
        print(f"  Merged: {merged_ids}")
        if llm_result.get("thinking"):
            print(f"  LLM reasoning: {llm_result['thinking'][:200]}")

    return {
        "doc_ids": merged_ids,
        "bm25_results": bm25_results,
        "llm_result": llm_result,
    }


def _collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    """Collect leaf nodes that carry text."""
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


def _bm25_node_candidates(query: str, doc: dict, top_k: int = 20) -> list[dict]:
    """Return the top leaf candidates scored by BM25."""
    leaves = _collect_leaf_nodes(doc["structure"])
    if not leaves:
        return []

    corpus = []
    for leaf in leaves:
        combined = leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            combined += " " + leaf["penjelasan"]
        combined += " " + leaf.get("navigation_path", "")
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    candidates = []
    for idx, score in ranked[:top_k]:
        if score > 0:
            leaf = leaves[idx]
            snippet = extract_kwic_snippet(leaf["text"], query)
            candidates.append({
                "node_id": leaf["node_id"],
                "title": leaf["title"],
                "navigation_path": leaf["navigation_path"],
                "bm25_score": round(float(score), 4),
                "snippet": snippet,
            })
    return candidates


def _llm_rerank(query: str, candidates: list[dict], doc_title: str) -> dict:
    """Ask the LLM to rank all candidates from most to least relevant.

    RankGPT-style full permutation generation (Sun et al. 2023, EMNLP). Caller
    validates via `validate_llm_ranking` to drop hallucinations and append
    missing IDs, so the final ranking always covers the full candidate set.
    """
    candidates_for_prompt = []
    for c in candidates:
        candidates_for_prompt.append({
            "node_id": c["node_id"],
            "title": c["title"],
            "navigation_path": c["navigation_path"],
            "snippet": c["snippet"],
        })

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)
    n_candidates = len(candidates)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan {n_candidates} Pasal kandidat dari "{doc_title}".
Setiap kandidat memiliki cuplikan teks (snippet) dari isinya.

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
- Pertimbangkan ISI snippet dan navigation_path
- Kembalikan HANYA JSON
"""

    return llm_call(prompt)


def node_search(query: str, doc: dict, bm25_top_k: int = 20,
                verbose: bool = True) -> dict:
    """Run BM25 candidate search and LLM reranking within one document."""
    candidates = _bm25_node_candidates(query, doc, top_k=bm25_top_k)

    if not candidates:
        return {"node_ids": [], "bm25_candidates": [], "llm_rerank": {}}

    if verbose:
        print(f"\n[Node Search - BM25 Candidates] Top {len(candidates)}:")
        for c in candidates:
            print(f"  {c['node_id']} {c['title']} (BM25: {c['bm25_score']:.4f})")
            print(f"    path: {c['navigation_path']}")

    rerank_result = _llm_rerank(query, candidates, doc.get("judul", ""))
    raw_ranking = rerank_result.get("ranking", [])
    ranked_ids = validate_llm_ranking(raw_ranking, candidates)
    rerank_result["validated_ranking"] = ranked_ids
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(ranked_ids)

    if verbose:
        print(f"\n[Node Search - LLM Rerank] Ranked {len(ranked_ids)} candidates")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    return {
        "node_ids": ranked_ids,
        "bm25_candidates": candidates,
        "llm_rerank": rerank_result,
    }


def retrieve(query: str, bm25_top_k: int = 20, verbose: bool = True) -> dict:
    """Run the catalog-first hybrid retrieval pipeline for one query."""
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid (bm25_top_k={bm25_top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_result = doc_search(query, catalog, verbose=verbose)
    doc_ids = doc_result.get("doc_ids", [])
    steps["doc_search"] = step_metrics(t_step, snap)

    if not doc_ids:
        return {"query": query, "strategy": "hybrid", "error": "No relevant documents found"}

    snap = snapshot_counters()
    t_step = time.time()

    doc_id = doc_ids[0]
    doc = load_doc(doc_id)

    node_result = node_search(query, doc, bm25_top_k=bm25_top_k, verbose=verbose)
    node_ids = node_result.get("node_ids", [])
    steps["node_search"] = step_metrics(t_step, snap)

    if not node_ids:
        return {"query": query, "strategy": "hybrid", "doc_ids": doc_ids,
                "error": "No relevant nodes found"}

    nodes = extract_nodes(doc, node_ids)

    if not nodes:
        return {"query": query, "strategy": "hybrid", "doc_ids": doc_ids,
                "node_ids": node_ids, "error": "Selected nodes not found in tree"}

    bm25_scores = {c["node_id"]: c["bm25_score"] for c in node_result.get("bm25_candidates", [])}
    sources = []
    for pos, node in enumerate(nodes):
        sources.append({
            "doc_id": doc_id,
            "node_id": node["node_id"],
            "title": node["title"],
            "navigation_path": node["navigation_path"],
            "bm25_score": bm25_scores.get(node["node_id"]),
            "rerank_position": pos,
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid",
        "doc_search": doc_result,
        "node_search": node_result,
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
    ap = argparse.ArgumentParser(description="Hybrid BM25+LLM retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=20,
                    help="Max BM25 candidates for LLM reranking (default: 20)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        score = src.get("bm25_score", "N/A")
        print(f"  > {src['navigation_path']} (BM25: {score})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
