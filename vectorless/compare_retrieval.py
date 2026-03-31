"""
Side-by-side retrieval comparison across index granularities.

Supports all 3 retrieval strategies: bm25, llm (full/stepwise), hybrid.
Loads the same document from pasal-level, ayat-level, and full-split index
directories and runs the node-level search with identical queries.

Usage:
    python -m vectorless.compare_retrieval                    # BM25 (default, no API needed)
    python -m vectorless.compare_retrieval --strategy llm     # LLM stepwise tree search
    python -m vectorless.compare_retrieval --strategy hybrid  # BM25 candidates + LLM rerank
"""

import argparse
import json
from pathlib import Path

from .retrieval.bm25.two_stage import node_search as bm25_node_search
from .retrieval.llm.search import tree_search_stepwise as llm_tree_search
from .retrieval.hybrid.search import node_search as hybrid_node_search

# Paths to the three index variants
INDEX_PASAL = Path("data/index_pasal")
INDEX_AYAT = Path("data/index_ayat")
INDEX_FULL_SPLIT = Path("data/index_full_split")

DOC_ID = "perpu-1-2016"
DOC_PATH = f"PERPU/{DOC_ID}.json"


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _run_search(strategy: str, query: str, doc: dict) -> list[str]:
    """Run node search with the given strategy. Returns list of node_ids."""
    if strategy == "bm25":
        result = bm25_node_search(query, doc, top_k=3, verbose=False)
        return result.get("node_ids", [])
    elif strategy == "llm":
        result = llm_tree_search(query, doc, verbose=False)
        return result.get("node_ids", [])
    elif strategy == "hybrid":
        result = hybrid_node_search(query, doc, bm25_top_k=10, verbose=False)
        return result.get("node_ids", [])
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def _format_results(strategy: str, query: str, doc: dict) -> list[dict]:
    """Run search and return formatted results with node_id + title."""
    if strategy == "bm25":
        result = bm25_node_search(query, doc, top_k=3, verbose=False)
        return result.get("rankings", [])
    elif strategy == "llm":
        result = llm_tree_search(query, doc, verbose=False)
        # LLM returns node_ids without scores — format consistently
        node_ids = result.get("node_ids", [])
        return [{"node_id": nid, "title": "", "score": None} for nid in node_ids]
    elif strategy == "hybrid":
        result = hybrid_node_search(query, doc, bm25_top_k=10, verbose=False)
        # Hybrid returns node_ids after LLM rerank
        node_ids = result.get("node_ids", [])
        bm25_map = {c["node_id"]: c for c in result.get("bm25_candidates", [])}
        return [
            {
                "node_id": nid,
                "title": bm25_map.get(nid, {}).get("title", ""),
                "score": bm25_map.get(nid, {}).get("bm25_score"),
            }
            for nid in node_ids
        ]
    return []


# Test questions targeting specific provisions in Perpu 1/2016.
# Each has a "gold" answer description for manual evaluation.
QUESTIONS = [
    {
        "query": "Berapa lama pidana penjara untuk pelaku kekerasan seksual terhadap anak?",
        "gold": "Pasal 81 Ayat (1): paling singkat 5 tahun, paling lama 15 tahun",
    },
    {
        "query": "Apa hukuman tambahan bagi pelaku kekerasan seksual terhadap anak?",
        "gold": "Pasal 81 Ayat (6): pengumuman identitas pelaku",
    },
    {
        "query": "Siapa saja yang mendapat penambahan sepertiga hukuman?",
        "gold": "Pasal 81 Ayat (3): orang tua, wali, keluarga, pengasuh, pendidik, aparat",
    },
    {
        "query": "Apa itu tindakan kebiri kimia dan kapan diterapkan?",
        "gold": "Pasal 81 Ayat (7): kebiri kimia + alat pendeteksi elektronik untuk residivis/korban banyak; Pasal 81A Ayat (1): paling lama 2 tahun setelah pidana pokok",
    },
    {
        "query": "Kapan pelaku bisa dihukum mati?",
        "gold": "Pasal 81 Ayat (5): korban >1, luka berat, gangguan jiwa, penyakit menular, hilang fungsi reproduksi, atau korban meninggal",
    },
    {
        "query": "Apakah anak yang menjadi pelaku dikenai pidana tambahan?",
        "gold": "Pasal 81 Ayat (9): pidana tambahan dan tindakan dikecualikan bagi pelaku Anak",
    },
    {
        "query": "Bagaimana pelaksanaan rehabilitasi bagi pelaku?",
        "gold": "Pasal 81A Ayat (3): kebiri kimia disertai rehabilitasi; Ayat (4): tata cara diatur PP",
    },
    {
        "query": "Apa hukuman untuk pencabulan terhadap anak?",
        "gold": "Pasal 82 Ayat (1): penjara 5-15 tahun dan denda 5 miliar",
    },
]


def run_comparison(strategy: str):
    doc_pasal = load_json(INDEX_PASAL / DOC_PATH)
    doc_full = load_json(INDEX_FULL_SPLIT / DOC_PATH)

    label = {"bm25": "BM25", "llm": "LLM Stepwise", "hybrid": "Hybrid (BM25+LLM)"}
    print(f"{'='*70}")
    print(f"Comparing retrieval: {DOC_ID}")
    print(f"  Strategy:           {label.get(strategy, strategy)}")
    print(f"  Pasal-level index:  Pasal-level leaves")
    print(f"  Full-split index:   Ayat/Huruf/Angka-level leaves")
    print(f"{'='*70}\n")

    for i, q in enumerate(QUESTIONS, 1):
        query = q["query"]
        gold = q["gold"]

        print(f"--- Q{i}: {query}")
        print(f"    Gold: {gold}\n")

        # Pasal-level
        results_pasal = _format_results(strategy, query, doc_pasal)
        print(f"  [Pasal-level] Top results:")
        for r in results_pasal:
            score_str = f" score={r['score']:.4f}" if r["score"] is not None else ""
            print(f"    {r['node_id']:12s} {r.get('title', ''):40s}{score_str}")
        if not results_pasal:
            print(f"    (no results)")

        # Full-split
        results_full = _format_results(strategy, query, doc_full)
        print(f"  [Full-split] Top results:")
        for r in results_full:
            score_str = f" score={r['score']:.4f}" if r["score"] is not None else ""
            print(f"    {r['node_id']:12s} {r.get('title', ''):40s}{score_str}")
        if not results_full:
            print(f"    (no results)")

        print()

    print(f"{'='*70}")
    print("Done. Compare which variant retrieves the most precise (granular) answer.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare retrieval across index granularities")
    parser.add_argument(
        "--strategy", choices=["bm25", "llm", "hybrid"], default="bm25",
        help="Retrieval strategy: bm25 (default, no API), llm (needs Gemini), hybrid (needs Gemini)"
    )
    args = parser.parse_args()
    run_comparison(args.strategy)
