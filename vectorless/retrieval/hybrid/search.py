"""
Hybrid BM25 + LLM retrieval strategy for Indonesian legal QA.

Combines keyword matching (BM25) with LLM semantic understanding:
  1. Doc search  — union of BM25 metadata match + LLM semantic selection
  2. Node search — BM25 retrieves candidate Pasal, LLM reranks with text context
  3. Answer gen  — LLM generates grounded answer (same as other strategies)

This addresses weaknesses of both pure approaches:
  - Pure BM25 fails on vocabulary mismatch (query term not in metadata)
  - Pure LLM fails on blind navigation (generic titles like "Pasal 3")
  - Hybrid: BM25 finds keyword-relevant content, LLM adds semantic understanding

Usage:
    python -m vectorless.retrieval.hybrid.search "Apa syarat penyadapan?"
    python -m vectorless.retrieval.hybrid.search "Apa definisi penyadapan?" --bm25_top_k 10
    python -m vectorless.retrieval.hybrid.search "Apa definisi penyadapan?" --bm25_top_k 15
"""

import argparse
import json
import time

from rank_bm25 import BM25Okapi

from ..common import (
    tokenize, llm_call, reset_token_counters, get_token_stats,
    snapshot_token_counters, compute_step_metrics,
    load_catalog, load_doc, find_node, extract_nodes,
    extract_kwic_snippet, generate_answer, save_log, DATA_INDEX,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = 3) -> list[dict]:
    """BM25 scoring on catalog metadata. Returns ranked docs with scores."""
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
    """LLM semantic selection from catalog. Returns selected docs."""
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
    """Hybrid doc search: union of BM25 keyword matches and LLM semantic picks.

    BM25 catches exact keyword matches in metadata.
    LLM catches semantic matches (e.g., "penyadapan" → hukum acara pidana).
    Union ensures neither blind spot causes a miss.
    """
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
    """Recursively collect all leaf nodes (nodes with text, no children) from tree."""
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


def _bm25_node_candidates(query: str, doc: dict, top_k: int = 10) -> list[dict]:
    """BM25 retrieval on leaf node texts. Returns top-K candidates with scores."""
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
    """LLM reranks BM25 candidates based on text snippets.

    Unlike pure LLM tree search (which only sees titles), the LLM here sees
    actual text snippets from each candidate Pasal, enabling informed selection.
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

    prompt = f"""\
Kamu diberi pertanyaan hukum dan daftar Pasal kandidat dari "{doc_title}".
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
- Kembalikan HANYA JSON
"""

    return llm_call(prompt)


def node_search(query: str, doc: dict, bm25_top_k: int = 10,
                verbose: bool = True) -> dict:
    """Hybrid node search: BM25 retrieves candidates, LLM reranks with snippets.

    BM25 ensures keyword-relevant Pasal are found (not missed by blind navigation).
    LLM reranking ensures semantic relevance (understands context beyond keywords).
    """
    candidates = _bm25_node_candidates(query, doc, top_k=bm25_top_k)

    if not candidates:
        return {"node_ids": [], "bm25_candidates": [], "llm_rerank": {}}

    if verbose:
        print(f"\n[Node Search - BM25 Candidates] Top {len(candidates)}:")
        for c in candidates:
            print(f"  {c['node_id']} {c['title']} (BM25: {c['bm25_score']:.4f})")
            print(f"    path: {c['navigation_path']}")

    rerank_result = _llm_rerank(query, candidates, doc.get("judul", ""))
    selected_ids = rerank_result.get("selected_ids", [])

    valid_candidate_ids = {c["node_id"] for c in candidates}
    selected_ids = [nid for nid in selected_ids if nid in valid_candidate_ids]
    if not selected_ids:
        selected_ids = [candidates[0]["node_id"]]

    if verbose:
        print(f"\n[Node Search - LLM Rerank] Selected: {selected_ids}")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    return {
        "node_ids": selected_ids,
        "bm25_candidates": candidates,
        "llm_rerank": rerank_result,
    }


def retrieve(query: str, bm25_top_k: int = 10, verbose: bool = True) -> dict:
    """Full hybrid retrieval pipeline: hybrid doc search → BM25+LLM node search → answer.

    Args:
        query: Legal question in Indonesian
        bm25_top_k: Max BM25 candidates for LLM reranking
        verbose: Print progress
    """
    reset_token_counters()
    t_start = time.time()
    step_metrics = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid (bm25_top_k={bm25_top_k})")
        print(f"{'='*60}")

    snap = snapshot_token_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_result = doc_search(query, catalog, verbose=verbose)
    doc_ids = doc_result.get("doc_ids", [])
    step_metrics["doc_search"] = compute_step_metrics(t_step, snap)

    if not doc_ids:
        return {"query": query, "strategy": "hybrid", "error": "No relevant documents found"}

    snap = snapshot_token_counters()
    t_step = time.time()

    doc_id = doc_ids[0]
    doc = load_doc(doc_id)

    node_result = node_search(query, doc, bm25_top_k=bm25_top_k, verbose=verbose)
    node_ids = node_result.get("node_ids", [])
    step_metrics["node_search"] = compute_step_metrics(t_step, snap)

    if not node_ids:
        return {"query": query, "strategy": "hybrid", "doc_ids": doc_ids,
                "error": "No relevant nodes found"}

    snap = snapshot_token_counters()
    t_step = time.time()

    nodes = extract_nodes(doc, node_ids)

    if not nodes:
        return {"query": query, "strategy": "hybrid", "doc_ids": doc_ids,
                "node_ids": node_ids, "error": "Selected nodes not found in tree"}

    doc_meta = {"doc_id": doc_id, "judul": doc.get("judul", "")}
    answer_result = generate_answer(query, nodes, doc_meta, verbose=verbose)
    step_metrics["answer_gen"] = compute_step_metrics(t_step, snap)

    bm25_scores = {c["node_id"]: c["bm25_score"] for c in node_result.get("bm25_candidates", [])}
    sources = []
    for node in nodes:
        sources.append({
            "doc_id": doc_id,
            "node_id": node["node_id"],
            "title": node["title"],
            "navigation_path": node["navigation_path"],
            "bm25_score": bm25_scores.get(node["node_id"]),
        })

    elapsed = time.time() - t_start
    stats = get_token_stats()

    result = {
        "query": query,
        "strategy": "hybrid",
        "doc_search": doc_result,
        "node_search": node_result,
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
        print(f"{'='*60}")

    return result


def main():
    ap = argparse.ArgumentParser(description="Hybrid BM25+LLM retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--bm25_top_k", type=int, default=10,
                    help="Max BM25 candidates for LLM reranking (default: 10)")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        score = src.get("bm25_score", "N/A")
        print(f"  > {src['navigation_path']} (BM25: {score})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
