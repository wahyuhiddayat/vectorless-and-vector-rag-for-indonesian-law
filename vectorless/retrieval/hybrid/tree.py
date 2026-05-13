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
import random
import time

from rank_bm25 import BM25Okapi

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    tokenize, load_catalog, load_doc, find_node, extract_nodes,
    save_log, validate_llm_ranking, DATA_INDEX,
    doc_corpus_string, catalog_for_llm_prompt,
    DOC_PICK_TOP_K,
)


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K) -> list[dict]:
    """Rank catalog entries with BM25 over the doc corpus string.

    Corpus per doc comes from `doc_corpus_string` (metadata + the aggregated
    `doc_summary_text` from indexing.build) so doc-level signal is rich
    enough to actually rank docs by topical match instead of the 15-20
    token metadata baseline. Returns up to `top_k` docs above zero score.
    """
    corpus = [tokenize(doc_corpus_string(doc)) for doc in catalog]

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
    """Ask the LLM to pick relevant documents from the catalog.

    The catalog is projected through `catalog_for_llm_prompt` to truncate
    each doc's aggregated summary to a manageable per-doc budget. Full
    summaries would inflate the prompt past a comfortable budget at 308
    docs in scope.
    """
    slim_catalog = catalog_for_llm_prompt(catalog)
    docs_text = json.dumps(slim_catalog, ensure_ascii=False, indent=2)

    prompt = f"""\
Kamu diberi daftar Undang-Undang Indonesia beserta metadata dan ringkasan isi-nya.
Pilih UU yang paling mungkin mengandung jawaban untuk pertanyaan hukum berikut.

Pertanyaan: {query}

Daftar UU:
{docs_text}

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat mengapa UU tersebut kemungkinan relevan>",
  "doc_ids": ["doc_id_1", "doc_id_2"]
}}

Aturan:
- Pilih 1 sampai 3 UU yang paling mungkin mengandung jawaban (recall-oriented).
- Lebih baik over-include sedikit daripada miss UU yang relevan.
- Pertimbangkan judul, bidang, subjek, materi_pokok, dan doc_summary_text (kalau tersedia).
- Hanya kembalikan doc_ids kosong [] jika benar-benar tidak ada satupun yang dekat dengan topik pertanyaan.
- Kembalikan HANYA JSON, tanpa teks lain.
"""
    return llm_call(prompt)


def doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K,
               verbose: bool = True) -> dict:
    """Merge BM25 catalog hits with the LLM's document picks, cap at top_k.

    LLM picks take precedence; BM25 picks fill remaining slots up to
    `top_k`. Both stages are recall-oriented (1-3 docs each), so the merge
    rarely produces fewer than `top_k` docs unless the catalog is genuinely
    empty of topical overlap.
    """
    bm25_results = _bm25_doc_search(query, catalog, top_k=top_k)
    llm_result = _llm_doc_search(query, catalog)

    bm25_ids = [r["doc_id"] for r in bm25_results]
    llm_ids = llm_result.get("doc_ids", [])

    valid_ids = {d["doc_id"] for d in catalog}
    llm_ids = [doc_id for doc_id in llm_ids if doc_id in valid_ids]

    # Interleave LLM and BM25 picks so neither signal dominates. LLM gets
    # priority at each position (semantic precedence), but BM25 always has
    # a slot if both lists have entries at that rank. This matters when
    # LLM picks semantically-related-but-keyword-mismatched docs while BM25
    # picks the lexically-matching doc that actually contains the answer.
    seen: set[str] = set()
    merged_ids: list[str] = []
    for i in range(max(len(llm_ids), len(bm25_ids))):
        for source in (llm_ids, bm25_ids):
            if i < len(source):
                doc_id = source[i]
                if doc_id not in seen:
                    seen.add(doc_id)
                    merged_ids.append(doc_id)
                    if len(merged_ids) >= top_k:
                        break
        if len(merged_ids) >= top_k:
            break

    if verbose:
        print(f"\n[Doc Search - Hybrid]")
        print(f"  BM25 hits: {bm25_ids}")
        print(f"  LLM picks: {llm_ids}")
        print(f"  Merged (top-{top_k}): {merged_ids}")
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
                "summary": node.get("summary", ""),
            })
    return leaves


def _bm25_node_candidates(query: str, doc: dict, top_k: int = 20) -> list[dict]:
    """Return the top leaf candidates scored by BM25.

    Corpus enrichment mirrors `vectorless/retrieval/bm25/flat.py`
    (doc_title + navigation_path + text + penjelasan) so within-doc BM25
    ranking is comparable across flat and hybrid-tree variants.
    """
    leaves = _collect_leaf_nodes(doc["structure"])
    if not leaves:
        return []

    doc_title = doc.get("judul", "")
    corpus = []
    for leaf in leaves:
        combined = doc_title + " " + leaf.get("navigation_path", "") + " " + leaf["text"]
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            combined += " " + leaf["penjelasan"]
        corpus.append(tokenize(combined))

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    target = max(top_k, 10)
    candidates = []
    for idx, score in ranked[:target]:
        leaf = leaves[idx]
        candidates.append({
            "node_id": leaf["node_id"],
            "title": leaf["title"],
            "navigation_path": leaf["navigation_path"],
            "text": leaf["text"],
            "penjelasan": leaf.get("penjelasan"),
            "summary": leaf.get("summary", ""),
            "bm25_score": round(float(score), 4),
        })
    return candidates


def _llm_rerank_multidoc(query: str, candidates: list[dict]) -> dict:
    """Ask the LLM to rank candidates from multiple docs in a single call.

    Listwise reranking across docs: each candidate carries its `doc_id` and
    `doc_title` so the reranker can disambiguate when nodes share titles or
    navigation paths across documents. Candidates are shuffled before this
    call to mitigate anchor bias from the BM25 ordering, and the caller
    validates via `validate_llm_ranking` to drop hallucinations and append
    missing ids.

    The candidate key (the unique id the LLM ranks) is `doc_id/node_id` so
    cross-doc nodes never collide. `validate_llm_ranking` operates on this
    string id; the caller resolves back to (doc_id, node_id) tuples.
    """
    candidates_for_prompt = []
    for c in candidates:
        key = f"{c['doc_id']}/{c['node_id']}"
        entry = {
            "ref": key,
            "doc_id": c["doc_id"],
            "doc_title": c.get("doc_title", ""),
            "title": c.get("title", ""),
            "navigation_path": c.get("navigation_path", ""),
            "text": c.get("text") or "",
        }
        penjelasan = c.get("penjelasan")
        if penjelasan and penjelasan != "Cukup jelas.":
            entry["penjelasan"] = penjelasan
        candidates_for_prompt.append(entry)

    candidates_text = json.dumps(candidates_for_prompt, ensure_ascii=False, indent=2)
    n_candidates = len(candidates)

    prompt = f"""\
Kamu diberi pertanyaan hukum dan {n_candidates} Pasal kandidat dari beberapa UU dalam katalog.
Setiap kandidat punya `ref` (format "doc_id/node_id"), doc_id, doc_title, navigation_path, text, dan penjelasan resmi (jika ada).

Pertanyaan: {query}

Kandidat Pasal (lintas UU):
{candidates_text}

Tugas: Urutkan SELURUH {n_candidates} kandidat dari paling relevan ke paling tidak relevan
untuk menjawab pertanyaan. Output harus berisi SEMUA {n_candidates} ref dari input,
tanpa duplikat dan tanpa ref yang tidak ada di input.

Balas dalam format JSON:
{{
  "thinking": "<penalaran singkat tentang kriteria ranking lintas UU>",
  "ranking": ["doc_id_1/node_id_1", "doc_id_2/node_id_2", "..."]
}}

Aturan:
- "ranking" HARUS berisi tepat {n_candidates} ref
- "ranking" tidak boleh ada duplikat
- Setiap ref harus muncul di input (tidak boleh hallucinate)
- Urutan menentukan ranking (index 0 = paling relevan)
- Pertimbangkan isi text, penjelasan, navigation_path, dan doc_title untuk konteks lintas UU
- Kembalikan HANYA JSON
"""

    return llm_call(prompt)


def retrieve(query: str, bm25_top_k: int = 20, top_k: int = 10,
             top_k_docs: int = DOC_PICK_TOP_K, verbose: bool = True) -> dict:
    """Run the multi-doc hybrid retrieval pipeline for one query.

    Stage 1: BM25 + LLM merge doc-pick (top-K=3).
    Stage 2: BM25 leaf candidates per picked doc, concatenated.
    Stage 3: single LLM listwise rerank across all cross-doc candidates.

    LLM call count: 2 (doc-pick + cross-doc rerank), independent of K. The
    rerank prompt grows linearly with K but fits comfortably in Gemini's
    1M context at K=3 (about 30K tokens of candidate text).
    """
    reset_counters()
    t_start = time.time()
    steps: dict = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: hybrid (top_k_docs={top_k_docs}, bm25_top_k={bm25_top_k})")
        print(f"{'='*60}")

    snap = snapshot_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_result = doc_search(query, catalog, top_k=top_k_docs, verbose=verbose)
    doc_ids = doc_result.get("doc_ids", [])
    steps["doc_search"] = step_metrics(t_step, snap)

    if not doc_ids:
        return {"query": query, "strategy": "hybrid", "picked_doc_ids": [],
                "error": "No relevant documents found"}

    snap = snapshot_counters()
    t_step = time.time()

    all_candidates: list[dict] = []
    per_doc_candidate_counts: dict[str, int] = {}
    loaded_docs: dict[str, dict] = {}
    for did in doc_ids:
        doc = load_doc(did)
        loaded_docs[did] = doc
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
        return {"query": query, "strategy": "hybrid", "picked_doc_ids": doc_ids,
                "error": "No relevant nodes found"}

    # Shuffle to mitigate anchor bias from BM25 order before LLM rerank.
    shuffled = list(all_candidates)
    random.shuffle(shuffled)

    rerank_result = _llm_rerank_multidoc(query, shuffled)
    raw_ranking = rerank_result.get("ranking", [])
    valid_refs = {f"{c['doc_id']}/{c['node_id']}" for c in all_candidates}
    n_hallucinated = sum(1 for r in raw_ranking if r not in valid_refs)

    pseudo_candidates = [
        {"node_id": f"{c['doc_id']}/{c['node_id']}"} for c in all_candidates
    ]
    validated_refs = validate_llm_ranking(raw_ranking, pseudo_candidates)
    rerank_result["validated_ranking"] = validated_refs
    rerank_result["llm_ranking_length"] = len(raw_ranking)
    rerank_result["validated_ranking_length"] = len(validated_refs)
    rerank_result["n_hallucinated"] = n_hallucinated
    steps["node_rerank"] = step_metrics(t_step, snap)

    if verbose:
        print(f"\n[Node Search - LLM Rerank Multi-doc] Ranked {len(validated_refs)} candidates")
        if rerank_result.get("thinking"):
            print(f"  Reasoning: {rerank_result['thinking'][:200]}")

    candidate_by_ref = {f"{c['doc_id']}/{c['node_id']}": c for c in all_candidates}
    sources = []
    for pos, ref in enumerate(validated_refs[:top_k]):
        c = candidate_by_ref.get(ref)
        if not c:
            continue
        sources.append({
            "doc_id": c["doc_id"],
            "node_id": c["node_id"],
            "title": c.get("title", ""),
            "navigation_path": c.get("navigation_path", ""),
            "bm25_score": c.get("bm25_score"),
            "rerank_position": pos,
        })

    if not sources:
        return {"query": query, "strategy": "hybrid", "picked_doc_ids": doc_ids,
                "error": "Reranked refs empty after validation"}

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "hybrid",
        "picked_doc_ids": doc_ids,
        "doc_search": doc_result,
        "per_doc_candidate_counts": per_doc_candidate_counts,
        "merged_candidate_count": len(all_candidates),
        "llm_rerank": rerank_result,
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
                    help="Max BM25 candidates per doc for LLM reranking (default: 20)")
    ap.add_argument("--top_k", type=int, default=10,
                    help="Final number of leaves returned (default: 10)")
    ap.add_argument("--top_k_docs", type=int, default=DOC_PICK_TOP_K,
                    help=f"Number of docs picked at stage 1 (default: {DOC_PICK_TOP_K})")
    args = ap.parse_args()

    result = retrieve(args.query, bm25_top_k=args.bm25_top_k,
                      top_k=args.top_k, top_k_docs=args.top_k_docs)
    print(f"\n{'-'*60}")
    print(f"DASAR HUKUM:")
    for src in result.get("sources", []):
        score = src.get("bm25_score", "N/A")
        print(f"  > [{src['doc_id']}] {src['navigation_path']} (BM25: {score})")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
