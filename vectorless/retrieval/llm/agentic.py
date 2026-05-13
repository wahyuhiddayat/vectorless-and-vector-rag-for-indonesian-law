"""Agentic LLM retrieval for Indonesian legal QA, PageIndex-inspired.

The LLM acts as an agent. The full document tree (titles + summaries)
is exposed from the start — identical to PageIndex's
get_document_structure() which returns all nodes. The agent uses
expand() to re-focus on a subtree (visibility hint), read() to inspect
leaf text, and submit() to finalize.

doc_search picks one document from the catalog first (1 LLM call).
After the agent loop, a three-layer fallback fills remaining top-k
slots: agent submit → visited nodes → BM25 (scoped to the primary doc).
Unlike PageIndex, BM25 fallback exists because the legal-eval pipeline
requires fixed-width top-k output.

Usage:
    python -m vectorless.retrieval.llm.agentic "Apa syarat penyadapan?"
"""

from __future__ import annotations

import argparse
import json
import time

from rank_bm25 import BM25Okapi

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_catalog, load_doc, find_node, save_log,
    agentic_finalize, tokenize,
    doc_corpus_string, catalog_for_llm_prompt,
)


DOC_PICK_TOP_K = 3


def _bm25_doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K) -> list[str]:
    """Rank catalog entries with BM25 over the doc corpus string, return doc_ids.

    Uses `doc_corpus_string` from common so the BM25 fallback in doc_search
    benefits from the same aggregated-summary corpus used by bm25-tree and
    hybrid-tree. Returns the top-K doc_ids that score above zero.
    """
    corpus = [tokenize(doc_corpus_string(doc)) for doc in catalog]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [catalog[idx]["doc_id"] for idx, score in ranked[:top_k] if score > 0]


def doc_search(query: str, catalog: list[dict], top_k: int = DOC_PICK_TOP_K,
               verbose: bool = True) -> dict:
    """Pick up to `top_k` relevant documents via recall-oriented LLM plus BM25 fallback.

    The LLM is asked to lean over-inclusive (1-3 docs) rather than refuse,
    because the previous strict prompt caused 2/7 queries to hard-fail in
    the 2026-05-13 pilot. BM25 picks are always merged in as a safety net,
    so single-doc cascade failures (the tree-paradigm Achilles heel) are
    softened by giving the agent loop multiple doc candidates.

    Returns:
        Dict with `doc_ids` (merged, capped at top_k), `llm_doc_ids`,
        `bm25_doc_ids`, `thinking`, and `merge_source` per doc for telemetry.
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
  "doc_ids": ["doc_id_1", "doc_id_2", "doc_id_3"]
}}

Aturan:
- Pilih 1 sampai {top_k} UU yang paling mungkin mengandung jawaban (recall-oriented).
- Lebih baik over-include sedikit daripada miss UU yang relevan.
- Pertimbangkan judul, bidang, subjek, materi_pokok, dan doc_summary_text (kalau tersedia).
- Hanya kembalikan doc_ids kosong [] jika benar-benar tidak ada satupun yang dekat dengan topik pertanyaan.
- Kembalikan HANYA JSON, tanpa teks lain.
"""

    llm_result = llm_call(prompt)

    valid_ids = {d["doc_id"] for d in catalog}
    llm_doc_ids = [doc_id for doc_id in llm_result.get("doc_ids", []) if doc_id in valid_ids]
    bm25_doc_ids = _bm25_doc_search(query, catalog, top_k=top_k)

    merged: list[str] = []
    merge_source: dict[str, str] = {}
    for doc_id in llm_doc_ids:
        if doc_id not in merge_source:
            merged.append(doc_id)
            merge_source[doc_id] = "llm"
        if len(merged) >= top_k:
            break
    for doc_id in bm25_doc_ids:
        if len(merged) >= top_k:
            break
        if doc_id not in merge_source:
            merged.append(doc_id)
            merge_source[doc_id] = "bm25"

    result = {
        "doc_ids": merged,
        "llm_doc_ids": llm_doc_ids,
        "bm25_doc_ids": bm25_doc_ids,
        "merge_source": merge_source,
        "thinking": llm_result.get("thinking", ""),
    }

    if verbose:
        print(f"\n[Doc Search] LLM picks: {llm_doc_ids}")
        print(f"             BM25 picks: {bm25_doc_ids}")
        print(f"             Merged: {merged}")
        if llm_result.get("thinking"):
            print(f"  Reasoning: {llm_result['thinking'][:200]}")

    return result


MAX_ACTIONS = 20
MAX_READS = 15
OBSERVATION_RENDER_CAP = 1800
SCRATCHPAD_RECENT_FULL = 2
DEFAULT_TOP_K = 10
VALID_ACTIONS = ("inspect_doc", "expand", "read", "submit")


def _node_view(node: dict) -> dict:
    """Render one node and its descendants, titles and summaries only."""
    out = {
        "node_id": node.get("node_id", ""),
        "title": node.get("title", ""),
    }
    if node.get("summary"):
        out["summary"] = node["summary"]
    children = node.get("nodes") or []
    if children:
        out["nodes"] = [_node_view(c) for c in children]
    return out


def _tool_inspect_doc(doc: dict) -> dict:
    """Full recursive tree structure of a document, no leaf text."""
    return {
        "doc_id": doc.get("doc_id", ""),
        "judul": doc.get("judul", ""),
        "structure": [_node_view(n) for n in doc.get("structure", [])],
    }


def _render_tree_outline(nodes: list[dict], depth: int = 0) -> str:
    """Render the full document tree as an indented outline for the prompt.

    Returns all nodes at all levels — identical to PageIndex's
    get_document_structure(). The agent sees the complete tree structure
    from the start, exactly like PageIndex.
    """
    lines: list[str] = []
    indent = "  " * depth
    for n in nodes:
        node_id = n.get("node_id", "")
        title = (n.get("title") or "").strip()
        summary = (n.get("summary") or "").strip()
        head = f"{indent}- [{node_id}] {title}" if title else f"{indent}- [{node_id}]"
        if summary:
            head += f" :: {summary}"
        lines.append(head)
        children = n.get("nodes") or []
        if children:
            lines.append(_render_tree_outline(children, depth + 1))
    return "\n".join(lines)


def _tool_expand(doc: dict, node_id: str) -> dict:
    """Children of one internal node. Helps the agent focus on a subtree."""
    node = find_node(doc.get("structure", []), node_id)
    if node is None:
        return {"error": f"node_id '{node_id}' not found in doc '{doc.get('doc_id')}'"}
    children = node.get("nodes") or []
    if not children:
        return {
            "error": f"node_id '{node_id}' has no children, it is already a leaf",
            "title": node.get("title", ""),
        }
    return {
        "doc_id": doc.get("doc_id", ""),
        "parent": {"node_id": node_id, "title": node.get("title", ""),
                   "navigation_path": node.get("navigation_path", "")},
        "children": [_node_view(c) for c in children],
    }


def _tool_read(doc: dict, node_id: str) -> dict:
    """Full text of one leaf node, plus penjelasan if any."""
    node = find_node(doc.get("structure", []), node_id)
    if node is None:
        return {"error": f"node_id '{node_id}' not found in doc '{doc.get('doc_id')}'"}
    out = {
        "doc_id": doc.get("doc_id", ""),
        "node_id": node_id,
        "title": node.get("title", ""),
        "navigation_path": node.get("navigation_path", ""),
        "text": node.get("text") or "",
    }
    penjelasan = node.get("penjelasan")
    if penjelasan and penjelasan != "Cukup jelas.":
        out["penjelasan"] = penjelasan
    return out


def _parse_node_ref(ref, default_doc_id: str) -> tuple[str | None, str | None]:
    """Accept dict, 'doc_id/node_id' string, or bare node_id, scoped to default_doc_id."""
    if isinstance(ref, dict):
        return ref.get("doc_id") or default_doc_id, ref.get("node_id")
    if isinstance(ref, str):
        if "/" in ref:
            doc_id, node_id = ref.split("/", 1)
            return doc_id, node_id
        return default_doc_id, ref
    return None, None


def _render_scratchpad(scratchpad: list[dict]) -> str:
    """Render the scratchpad for the next prompt, recent steps full, older truncated."""
    lines = []
    n = len(scratchpad)
    for i, entry in enumerate(scratchpad):
        recent = (n - i) <= SCRATCHPAD_RECENT_FULL
        cap = OBSERVATION_RENDER_CAP if recent else 600
        action = entry.get("action", "?")
        args_str = json.dumps(entry.get("args") or {}, ensure_ascii=False)[:200]
        obs = entry.get("observation")
        obs_str = json.dumps(obs, ensure_ascii=False)
        if len(obs_str) > cap:
            obs_str = obs_str[:cap] + "...[truncated]"
        lines.append(f"[{entry.get('step', i)}] {action}({args_str}) -> {obs_str}")
        if entry.get("thinking") and recent:
            lines.append(f"     thinking: {entry['thinking'][:300]}")
    return "\n".join(lines) if lines else "(belum ada tindakan)"


def _build_prompt(query: str, scratchpad: list[dict],
                  actions_left: int, reads_left: int,
                  primary_doc_id: str, primary_doc_title: str,
                  tree_outline: str) -> str:
    """Build the next-step prompt for the agent.

    PageIndex-style: the full document structure is exposed upfront.
    The agent inspects it, uses expand() to focus on a subtree,
    read() to verify leaf content, and submit() to finalize.
    """
    return (
        "Kamu adalah agen retrieval dokumen hukum.\n"
        "Tugas: temukan node paling relevan (pasal/ayat/rincian) untuk menjawab pertanyaan.\n\n"
        f"Pertanyaan: {query}\n"
        f"Dokumen: {primary_doc_id} — {primary_doc_title}\n\n"
        "── STRUKTUR DOKUMEN ──\n"
        f"{tree_outline}\n\n"
        "── TOOLS ──\n"
        "- inspect_doc()       lihat struktur lengkap (title + ringkasan)\n"
        "- expand(node_id)     lihat anak-anak node, fokus pada satu cabang\n"
"- read(node_id)       baca teks LENGKAP satu node (pasal/ayat/rincian)\n"
"- submit(node_ids)    finalisasi ranking [paling relevan, ..., kurang relevan] (max 10)\n\n"
"-- ALUR KERJA --\n"
"1. Scan struktur → identifikasi cabang yang tematis relevan\n"
"2. expand() → lihat sub-node dalam cabang itu\n"
"3. read() → baca teks lengkap untuk verifikasi\n"
        "4. submit() → kirimkan node_ids terurut (paling relevan pertama)\n\n"
        f"── SISA ANGGARAN ──\n"
        f"Action: {actions_left} | Read: {reads_left}\n"
        "Gunakan read() secara selektif — hanya untuk verifikasi.\n\n"
        "── RIWAYAT TINDAKAN ──\n"
        f"{_render_scratchpad(scratchpad)}\n\n"
        "── FORMAT ──\n"
        "{\n"
        '  "thinking": "...",\n'
        '  "action": "inspect_doc" | "expand" | "read" | "submit",\n'
        '  "args": { ... }\n'
        "}\n\n"
        "Apa langkah selanjutnya?\n"
    )


def _siblings_hint(doc: dict, missing_id: str, limit: int = 5) -> list[str]:
    """Best-effort list of node_ids near the missing id, to help the agent self-correct."""
    all_ids: list[str] = []

    def _walk(nodes):
        for n in nodes:
            nid = n.get("node_id")
            if nid:
                all_ids.append(nid)
            if n.get("nodes"):
                _walk(n["nodes"])

    _walk(doc.get("structure", []))
    prefix = missing_id.split("_")[0] if missing_id else ""
    near = [nid for nid in all_ids if prefix and nid.startswith(prefix)]
    return near[:limit] if near else all_ids[:limit]


def _collect_visited_node_ids(scratchpad: list[dict]) -> list[str]:
    """Collect node_ids the agent visited via read or expand, by visit recency.

    Most recently visited first. Read targets and expanded children are both
    treated as visited. Used as the second layer of the RAPTOR-style fallback
    when the agent's submit list does not fill the top_k slots.
    """
    visited: list[str] = []
    seen: set[str] = set()
    for entry in reversed(scratchpad):
        action = entry.get("action")
        obs = entry.get("observation") or {}
        if "error" in obs:
            continue
        if action == "read":
            nid = obs.get("node_id")
            if nid and nid not in seen:
                visited.append(nid)
                seen.add(nid)
        elif action == "expand":
            for child in obs.get("children") or []:
                nid = child.get("node_id")
                if nid and nid not in seen:
                    visited.append(nid)
                    seen.add(nid)
    return visited


def _collect_doc_leaf_ids(doc: dict) -> list[dict]:
    """Flatten doc tree to leaf nodes carrying text, preserving traversal order."""
    leaves: list[dict] = []

    def _walk(nodes):
        for n in nodes:
            children = n.get("nodes") or []
            if children:
                _walk(children)
            elif n.get("text"):
                leaves.append({
                    "node_id": n.get("node_id", ""),
                    "title": n.get("title", ""),
                    "text": n.get("text", ""),
                    "navigation_path": n.get("navigation_path", ""),
                    "penjelasan": n.get("penjelasan"),
                })

    _walk(doc.get("structure") or [])
    return leaves


def _bm25_rank_leaves(query: str, leaves: list[dict], doc_title: str = "") -> list[str]:
    """BM25-rank a doc's leaves against the query. Returns leaf node_ids.

    Uses the same enriched corpus as bm25-flat (doc_title + navigation_path +
    text + penjelasan) so the RAPTOR fallback produces rankings comparable to
    bm25-flat. Without this alignment, agentic's fallback would be a weaker
    BM25 than bm25-flat and bias the comparison.
    """
    if not leaves:
        return []
    corpus = []
    for leaf in leaves:
        enriched = doc_title + " " + leaf.get("navigation_path", "") + " " + leaf.get("text", "")
        if leaf.get("penjelasan") and leaf["penjelasan"] != "Cukup jelas.":
            enriched += " " + leaf["penjelasan"]
        corpus.append(tokenize(enriched))
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [leaves[i]["node_id"] for i, _ in ranked]


def retrieve(query: str,
             max_actions: int = MAX_ACTIONS, max_reads: int = MAX_READS,
             top_k: int = DEFAULT_TOP_K, verbose: bool = True) -> dict:
    """Run the agentic LLM retrieval pipeline for one query.

    Args:
        query: Indonesian legal question.
        max_actions: hard cap on agent steps before forced fallback.
        max_reads: hard cap on read tool calls (other tools are uncapped).
        verbose: print step traces during the loop.

    Returns:
        Result dict compatible with the existing eval schema. The
        sources field is what the eval harness reads for Recall and MRR.
    """
    reset_counters()
    t_start = time.time()
    timings: dict = {}

    if verbose:
        print("=" * 60)
        print(f"Query: {query}")
        print("Strategy: llm-agentic-doc")
        print("=" * 60)

    catalog = load_catalog()
    doc_cache: dict[str, dict] = {}

    def _get_doc(doc_id: str) -> dict:
        """Lazy-load and memoize a document by id."""
        if doc_id not in doc_cache:
            doc_cache[doc_id] = load_doc(doc_id)
        return doc_cache[doc_id]

    scratchpad: list[dict] = []

    snap = snapshot_counters()
    t_step = time.time()
    doc_result = doc_search(query, catalog, verbose=verbose)
    timings["doc_search"] = step_metrics(t_step, snap)

    doc_ids = doc_result.get("doc_ids", [])
    if not doc_ids:
        return {
            "query": query,
            "strategy": "llm-agentic-doc",
            "picked_doc_ids": [],
            "doc_search": doc_result,
            "error": "No relevant documents found",
            "metrics": {**get_stats(), "elapsed_s": round(time.time() - t_start, 2),
                        "step_metrics": timings},
        }
    primary_doc_id = doc_ids[0]
    primary_doc = _get_doc(primary_doc_id)
    primary_doc_title = primary_doc.get("judul", "")
    tree_outline = _render_tree_outline(primary_doc.get("structure", []))
    scratchpad.append({
        "step": 0,
        "action": "doc_search",
        "args": {},
        "observation": {"doc_id": primary_doc_id, "judul": primary_doc_title},
    })

    snap = snapshot_counters()
    t_step = time.time()

    actions_used = 0
    reads_used = 0
    submitted = False
    selected: list[dict] = []
    parse_failures = 0

    while actions_used < max_actions and not submitted:
        prompt = _build_prompt(
            query, scratchpad,
            actions_left=max_actions - actions_used,
            reads_left=max_reads - reads_used,
            primary_doc_id=primary_doc_id,
            primary_doc_title=primary_doc_title,
            tree_outline=tree_outline,
        )

        try:
            response = llm_call(prompt)
            parse_failures = 0
        except json.JSONDecodeError:
            parse_failures += 1
            scratchpad.append({
                "step": len(scratchpad),
                "action": "(invalid_json)",
                "args": {},
                "observation": {"error": "Response was not valid JSON, try again with valid JSON."},
            })
            actions_used += 1
            if parse_failures >= 3:
                break
            continue

        action = response.get("action") or ""
        args = response.get("args") or {}
        thinking = response.get("thinking", "")

        observation: dict = {}

        if action == "inspect_doc":
            observation = _tool_inspect_doc(primary_doc)

        elif action == "expand":
            node_id = args.get("node_id")
            if not node_id:
                observation = {"error": "expand requires node_id."}
            else:
                obs = _tool_expand(primary_doc, node_id)
                if "error" in obs:
                    obs["hint_nearby"] = _siblings_hint(primary_doc, node_id)
                observation = obs

        elif action == "read":
            if reads_used >= max_reads:
                observation = {"error": f"Read budget exhausted ({max_reads}). Submit soon."}
            else:
                node_id = args.get("node_id")
                if not node_id:
                    observation = {"error": "read requires node_id."}
                else:
                    obs = _tool_read(primary_doc, node_id)
                    if "error" in obs:
                        obs["hint_nearby"] = _siblings_hint(primary_doc, node_id)
                    observation = obs
                    if "error" not in obs:
                        reads_used += 1

        elif action == "submit":
            refs = args.get("node_ids") or []
            reasoning = args.get("reasoning", "")
            resolved: list[dict] = []
            invalid: list[str] = []
            for ref in refs:
                doc_id, node_id = _parse_node_ref(ref, default_doc_id=primary_doc_id)
                if not (doc_id and node_id):
                    invalid.append(str(ref))
                    continue
                if doc_id != primary_doc_id:
                    invalid.append(f"{doc_id}/{node_id} (out of scope)")
                    continue
                if find_node(primary_doc.get("structure", []), node_id) is None:
                    invalid.append(f"{doc_id}/{node_id} (not found)")
                    continue
                key = (doc_id, node_id)
                if key not in {(s["doc_id"], s["node_id"]) for s in resolved}:
                    resolved.append({"doc_id": doc_id, "node_id": node_id})
            if resolved:
                selected = resolved
                submitted = True
                observation = {
                    "submitted": True,
                    "count": len(resolved),
                    "node_ids": [f"{s['doc_id']}/{s['node_id']}" for s in resolved],
                    "reasoning": reasoning[:300],
                }
                if invalid:
                    observation["dropped"] = invalid
            else:
                observation = {
                    "error": "submit produced no valid node_ids, refine and try again.",
                    "invalid": invalid,
                }

        else:
            observation = {
                "error": f"Unknown action '{action}'.",
                "valid_actions": list(VALID_ACTIONS),
            }

        scratchpad.append({
            "step": len(scratchpad),
            "thinking": thinking[:300],
            "action": action or "(empty)",
            "args": args,
            "observation": observation,
        })
        actions_used += 1

        if verbose:
            obs_preview = json.dumps(observation, ensure_ascii=False)[:200]
            print(f"\n[Step {actions_used}] {action} -> {obs_preview}")
            if thinking:
                print(f"  thinking: {thinking[:200]}")

    timings["agent_loop"] = step_metrics(t_step, snap)

    submitted_ids = [s["node_id"] for s in selected]
    visited_ids = _collect_visited_node_ids(scratchpad)
    doc_leaves = _collect_doc_leaf_ids(primary_doc)
    bm25_fallback_ids = _bm25_rank_leaves(query, doc_leaves, doc_title=primary_doc_title)
    final_ids, slot_labels = agentic_finalize(
        submitted_ids=submitted_ids,
        visited_ids=visited_ids,
        fallback_ids=bm25_fallback_ids,
        top_k=top_k,
    )

    if not final_ids:
        return {
            "query": query,
            "strategy": "llm-agentic-doc",
            "picked_doc_ids": doc_ids,
            "doc_search": doc_result,
            "agent": {
                "actions_used": actions_used,
                "reads_used": reads_used,
                "submitted": submitted,
                "scratchpad": scratchpad,
            },
            "error": (
                f"Agent submitted no valid node and BM25 fallback returned no leaves "
                f"(doc_leaves={len(doc_leaves)}, submitted={len(submitted_ids)}, "
                f"visited={len(visited_ids)})"
            ),
            "metrics": {**get_stats(), "elapsed_s": round(time.time() - t_start, 2),
                        "step_metrics": timings},
        }

    sources: list[dict] = []
    for pos, (nid, src_label) in enumerate(zip(final_ids, slot_labels)):
        node = find_node(primary_doc.get("structure", []), nid) or {}
        sources.append({
            "doc_id": primary_doc_id,
            "node_id": nid,
            "title": node.get("title", ""),
            "navigation_path": node.get("navigation_path", ""),
            "rerank_position": pos,
            "submission_source": src_label,
        })

    submission_source_counts = {
        "agent_submit": slot_labels.count("agent_submit"),
        "visited_unsubmitted": slot_labels.count("visited_unsubmitted"),
        "bm25_fallback": slot_labels.count("bm25_fallback"),
    }

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": "llm-agentic-doc",
        "picked_doc_ids": doc_ids,
        "doc_search": doc_result,
        "agent": {
            "actions_used": actions_used,
            "reads_used": reads_used,
            "submitted": submitted,
            "submitted_count": len(submitted_ids),
            "visited_count": len(visited_ids),
            "submission_source_counts": submission_source_counts,
            "scratchpad": scratchpad,
        },
        "node_ids": final_ids,
        "sources": sources,
        "metrics": {**stats, "elapsed_s": round(elapsed, 2), "step_metrics": timings},
    }

    save_log(result)

    if verbose:
        print("\n" + "=" * 60)
        print(f"Done in {elapsed:.1f}s  |  {stats['llm_calls']} LLM calls  |  "
              f"{stats['total_tokens']:,} tokens  |  submitted={submitted}")
        print("=" * 60)

    return result


def main() -> None:
    """CLI entry point for the agentic retrieval module."""
    ap = argparse.ArgumentParser(description="Agentic LLM retrieval for Indonesian legal QA")
    ap.add_argument("query", help="Legal question in Indonesian")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS,
                    help=f"Hard cap on agent steps (default {MAX_ACTIONS})")
    ap.add_argument("--max-reads", type=int, default=MAX_READS,
                    help=f"Hard cap on read tool calls (default {MAX_READS})")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Final ranked output length after RAPTOR fallback (default {DEFAULT_TOP_K})")
    args = ap.parse_args()

    result = retrieve(args.query,
                      max_actions=args.max_actions, max_reads=args.max_reads,
                      top_k=args.top_k)

    print("\n" + "-" * 60)
    print("DASAR HUKUM:")
    for src in result.get("sources", []):
        path = src.get("navigation_path") or src.get("node_id", "")
        print(f"  > {src.get('doc_id', '')} :: {path}")
    print("-" * 60)


if __name__ == "__main__":
    main()
