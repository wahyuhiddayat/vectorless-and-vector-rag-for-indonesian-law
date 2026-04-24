"""LLM-only retrieval strategies for the vectorless legal index.

All retrieval decisions (doc selection, tree navigation) are made by LLM prompting.
No algorithmic scoring — the LLM is the sole searcher.

Two tree search modes:
  - "full"     — show entire tree skeleton, LLM picks nodes in one shot
  - "stepwise" — navigate level-by-level (BAB → Bagian → Pasal) with reasoning

"""

import argparse
import json
import time

from ..common import (
    llm_call, reset_token_counters, get_token_stats,
    snapshot_token_counters, compute_step_metrics,
    load_catalog, load_doc, find_node, extract_nodes,
    generate_answer, save_log,
)


def doc_search(query: str, catalog: list[dict], verbose: bool = True) -> dict:
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

    result = llm_call(prompt)

    valid_ids = {d["doc_id"] for d in catalog}
    result["doc_ids"] = [doc_id for doc_id in result.get("doc_ids", []) if doc_id in valid_ids]

    if verbose:
        print(f"\n[Doc Search] Selected: {result.get('doc_ids', [])}")
        print(f"  Reasoning: {result.get('thinking', '')[:200]}")

    return result


def _build_tree_skeleton(nodes: list[dict], depth: int = 0) -> str:
    """Render the tree as titles and node IDs only."""
    lines = []
    for node in nodes:
        indent = "  " * depth
        lines.append(f"{indent}{node['node_id']} {node['title']}")
        if "nodes" in node:
            lines.append(_build_tree_skeleton(node["nodes"], depth + 1))
    return "\n".join(lines)


def tree_search_full(query: str, doc: dict, verbose: bool = True) -> dict:
    """Pick leaf nodes from the full tree in one LLM call."""
    skeleton = _build_tree_skeleton(doc["structure"])

    prompt = f"""\
Kamu diberi pertanyaan hukum dan struktur hierarkis sebuah Undang-Undang Indonesia.
Navigasi tree untuk menemukan Pasal yang paling relevan.

Pertanyaan: {query}

Undang-Undang: {doc['judul']}

Struktur:
{skeleton}

Balas dalam format JSON:
{{
  "thinking": "<penalaran hierarkis: BAB mana yang relevan, lalu Bagian/Pasal mana>",
  "node_ids": ["node_id1", "node_id2"]
}}

Aturan:
- Pilih node LEAF (Pasal) yang paling relevan, bukan BAB/Bagian
- Jika ada beberapa Pasal yang relevan, masukkan semuanya
- Jika query terlalu umum, pilih Pasal definisi (biasanya Pasal 1)
- Kembalikan HANYA JSON, tanpa teks lain
"""

    result = llm_call(prompt)

    if verbose:
        print(f"\n[Tree Search - Full] Selected: {result.get('node_ids', [])}")
        print(f"  Reasoning: {result.get('thinking', '')[:200]}")

    return result


def _get_top_level_nodes(structure: list[dict]) -> list[dict]:
    """Return the nodes shown in the first navigation round."""
    return [{"node_id": n["node_id"], "title": n["title"]} for n in structure]


def _get_children_summary(node: dict) -> list[dict]:
    """Summarize one node's children for the next navigation step."""
    if "nodes" not in node:
        return []
    result = []
    for c in node["nodes"]:
        entry: dict = {"node_id": c["node_id"], "title": c["title"]}
        if c.get("navigation_path"):
            entry["navigation_path"] = c["navigation_path"]
        if not c.get("nodes") and c.get("text"):
            entry["text_preview"] = c["text"][:150].rstrip()
        result.append(entry)
    return result


def tree_search_stepwise(query: str, doc: dict, verbose: bool = True) -> dict:
    """Walk the tree level by level until the LLM reaches leaf nodes."""
    steps = []
    structure = doc["structure"]
    doc_title = doc["judul"]
    max_rounds = 8

    top_nodes = _get_top_level_nodes(structure)
    top_text = json.dumps(top_nodes, ensure_ascii=False, indent=2)

    prompt = f"""\
Kamu sedang menavigasi struktur hierarkis UU "{doc_title}" untuk menjawab pertanyaan hukum.

Pertanyaan: {query}

Ini adalah bagian-bagian utama dalam UU tersebut:
{top_text}

Pilih 1-2 bagian yang paling relevan untuk menjawab pertanyaan.

Balas dalam format JSON:
{{
  "thinking": "<penalaran mengapa bagian ini relevan>",
  "selected_ids": ["node_id1"]
}}

Kembalikan HANYA JSON.
"""

    r1 = llm_call(prompt)
    steps.append({
        "round": 1, "level": "top",
        "options_shown": [n["title"] for n in top_nodes],
        "selected": r1.get("selected_ids", []),
        "thinking": r1.get("thinking", ""),
    })

    if verbose:
        print(f"\n[Tree Search - Round 1] Selected: {r1.get('selected_ids', [])}")
        print(f"  Reasoning: {r1.get('thinking', '')[:200]}")

    current_ids = r1.get("selected_ids", [])
    round_num = 2

    while round_num <= max_rounds:
        final_ids = []
        need_drill = []
        for nid in current_ids:
            node = find_node(structure, nid)
            if node and "nodes" in node and node["nodes"]:
                need_drill.append(node)
            else:
                final_ids.append(nid)

        if not need_drill:
            return {"steps": steps, "node_ids": final_ids}

        drill_candidates = []
        for node in need_drill:
            drill_candidates.extend(_get_children_summary(node))

        if not drill_candidates:
            final_ids.extend([n["node_id"] for n in need_drill])
            return {"steps": steps, "node_ids": final_ids}

        drill_text = json.dumps(drill_candidates, ensure_ascii=False, indent=2)

        prompt = f"""\
Kamu sedang menavigasi UU "{doc_title}" ke level lebih dalam.

Pertanyaan: {query}

Ini adalah sub-bagian di dalam bagian yang kamu pilih:
{drill_text}

Pilih bagian yang paling relevan untuk menjawab pertanyaan.

Balas dalam format JSON:
{{
  "thinking": "<penalaran mengapa bagian ini relevan>",
  "selected_ids": ["node_id1", "node_id2"]
}}

Aturan:
- Jika ada yang sudah spesifik (Ayat/Huruf), pilih langsung
- Jika masih umum (Bagian/Paragraf), pilih yang relevan (kita akan drill down lagi)
- Kembalikan HANYA JSON
"""

        result = llm_call(prompt)
        steps.append({
            "round": round_num,
            "level": f"drill-{round_num}",
            "options_shown": [n["title"] for n in drill_candidates],
            "selected": result.get("selected_ids", []),
            "thinking": result.get("thinking", ""),
        })

        if verbose:
            print(f"\n[Tree Search - Round {round_num}] Selected: {result.get('selected_ids', [])}")
            print(f"  Reasoning: {result.get('thinking', '')[:200]}")

        new_ids = result.get("selected_ids", [])
        if not new_ids:
            final_ids.extend([n["node_id"] for n in need_drill])
            return {"steps": steps, "node_ids": final_ids}

        current_ids = final_ids + new_ids
        round_num += 1

    return {"steps": steps, "node_ids": current_ids}


def retrieve(query: str, strategy: str = "stepwise", verbose: bool = True) -> dict:
    """Run the LLM-only retrieval pipeline for one query."""
    reset_token_counters()
    t_start = time.time()
    step_metrics = {}

    if verbose:
        print(f"{'='*60}")
        print(f"Query: {query}")
        print(f"Strategy: llm-{strategy}")
        print(f"{'='*60}")

    snap = snapshot_token_counters()
    t_step = time.time()

    catalog = load_catalog()
    doc_result = doc_search(query, catalog, verbose=verbose)
    doc_ids = doc_result.get("doc_ids", [])
    step_metrics["doc_search"] = compute_step_metrics(t_step, snap)

    if not doc_ids:
        return {"query": query, "strategy": f"llm-{strategy}", "error": "No relevant documents found"}

    snap = snapshot_token_counters()
    t_step = time.time()

    doc_id = doc_ids[0]
    doc = load_doc(doc_id)

    if strategy == "full":
        tree_result = tree_search_full(query, doc, verbose=verbose)
    else:
        tree_result = tree_search_stepwise(query, doc, verbose=verbose)

    node_ids = tree_result.get("node_ids", [])
    step_metrics["tree_search"] = compute_step_metrics(t_step, snap)

    if not node_ids:
        return {"query": query, "strategy": f"llm-{strategy}", "doc_ids": doc_ids,
                "error": "No relevant nodes found"}

    snap = snapshot_token_counters()
    t_step = time.time()

    nodes = extract_nodes(doc, node_ids)

    if not nodes:
        return {"query": query, "strategy": f"llm-{strategy}", "doc_ids": doc_ids,
                "node_ids": node_ids, "error": "Selected nodes not found in tree"}

    doc_meta = {"doc_id": doc_id, "judul": doc.get("judul", "")}
    answer_result = generate_answer(query, nodes, doc_meta, verbose=verbose)
    step_metrics["answer_gen"] = compute_step_metrics(t_step, snap)

    sources = []
    for node in nodes:
        sources.append({
            "doc_id": doc_id,
            "node_id": node["node_id"],
            "title": node["title"],
            "navigation_path": node["navigation_path"],
        })

    elapsed = time.time() - t_start
    stats = get_token_stats()

    result = {
        "query": query,
        "strategy": f"llm-{strategy}",
        "doc_search": doc_result,
        "tree_search": tree_result,
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
    ap = argparse.ArgumentParser(description="Pure LLM retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--strategy", choices=["full", "stepwise"], default="stepwise",
                    help="Tree search strategy (default: stepwise)")
    args = ap.parse_args()

    result = retrieve(args.query, strategy=args.strategy)
    print(f"\n{'-'*60}")
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print(f"\nDASAR HUKUM:")
    for src in result.get("sources", []):
        print(f"  > {src['navigation_path']}")
    print(f"{'-'*60}")


if __name__ == "__main__":
    main()
