"""Agentic LLM retrieval for Indonesian legal QA, PageIndex-style.

The LLM acts as an agent. Each step it picks one of inspect_doc, expand,
read, or submit, observes the result, then plans the next step. This
contrasts with llm-tree which forces a fixed level-by-level drill.

Two modes.
  - "doc"     run doc_search first, then the agent navigates one document
  - "corpus"  the agent starts from the catalog and may pick nodes from
              one or more documents in a single submit

Usage:
    python -m vectorless.retrieval.llm.agentic "Apa syarat penyadapan?" --mode doc
    python -m vectorless.retrieval.llm.agentic "Bandingkan sanksi X dan Y" --mode corpus
"""

from __future__ import annotations

import argparse
import json
import time

from ...llm import call as llm_call, reset_counters, get_stats, snapshot_counters, step_metrics
from ..common import (
    load_catalog, load_doc, find_node, extract_nodes,
    generate_answer, generate_answer_multi_doc, save_log,
)
from .tree import doc_search


MAX_ACTIONS = 15
MAX_READS = 8
READ_TEXT_CAP = 1500
OBSERVATION_RENDER_CAP = 1800
SCRATCHPAD_RECENT_FULL = 2
VALID_ACTIONS = ("list_docs", "inspect_doc", "expand", "read", "submit")


def _compact_catalog(catalog: list[dict]) -> list[dict]:
    """Strip catalog rows down to the fields the agent actually uses."""
    keep = ("doc_id", "judul", "bidang", "subjek", "materi_pokok")
    return [{k: row[k] for k in keep if k in row} for row in catalog]


def _node_view(node: dict) -> dict:
    """Render one node as a compact dict, no text body."""
    out = {
        "node_id": node.get("node_id", ""),
        "title": node.get("title", ""),
        "has_children": bool(node.get("nodes")),
    }
    if node.get("summary"):
        out["summary"] = node["summary"]
    return out


def _tool_inspect_doc(doc: dict) -> dict:
    """Top-level structure of a document, no leaf text."""
    return {
        "doc_id": doc.get("doc_id", ""),
        "judul": doc.get("judul", ""),
        "top_level": [_node_view(n) for n in doc.get("structure", [])],
    }


def _tool_expand(doc: dict, node_id: str) -> dict:
    """Children of one internal node."""
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
    """Full text of one leaf node, capped, plus penjelasan if any."""
    node = find_node(doc.get("structure", []), node_id)
    if node is None:
        return {"error": f"node_id '{node_id}' not found in doc '{doc.get('doc_id')}'"}
    text = (node.get("text") or "")[:READ_TEXT_CAP]
    out = {
        "doc_id": doc.get("doc_id", ""),
        "node_id": node_id,
        "title": node.get("title", ""),
        "navigation_path": node.get("navigation_path", ""),
        "text": text,
    }
    penjelasan = node.get("penjelasan")
    if penjelasan and penjelasan != "Cukup jelas.":
        out["penjelasan"] = penjelasan[:READ_TEXT_CAP]
    return out


def _parse_node_ref(ref, default_doc_id: str | None) -> tuple[str | None, str | None]:
    """Accept dict, 'doc_id/node_id' string, or bare node_id with default_doc_id."""
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


def _build_prompt(query: str, mode: str, scratchpad: list[dict],
                  actions_left: int, reads_left: int, allow_list_docs: bool,
                  primary_doc_id: str | None) -> str:
    """Build the next-step prompt for the agent."""
    tools = []
    if allow_list_docs:
        tools.append("- list_docs()                              daftar dokumen di katalog dengan metadata")
    if mode == "doc":
        tools.append("- inspect_doc()                            struktur top-level dokumen yang aktif")
        tools.append("- expand(node_id)                          anak satu node internal")
        tools.append("- read(node_id)                            teks lengkap satu leaf, hemat anggaran")
        tools.append('- submit(node_ids, reasoning)              finalisasi pilihan, contoh node_ids ["pasal_3", "pasal_5_ayat_1"]')
        scope_line = f'Dokumen aktif: {primary_doc_id}. Semua tool memakai dokumen ini secara implisit.'
    else:
        tools.append("- inspect_doc(doc_id)                      struktur top-level satu dokumen")
        tools.append("- expand(doc_id, node_id)                  anak satu node internal")
        tools.append("- read(doc_id, node_id)                    teks lengkap satu leaf, hemat anggaran")
        tools.append('- submit(node_ids, reasoning)              finalisasi pilihan, contoh node_ids ["uu-3-2025/pasal_4", "uu-16-2025/pasal_8"]')
        scope_line = "Mode corpus. node_ids di submit harus pakai format 'doc_id/node_id'."

    rules = [
        "- Setiap balasan WAJIB JSON valid berisi field thinking, action, args.",
        "- Pakai inspect_doc dan expand untuk menavigasi, gunakan read hanya saat perlu cek isi.",
        f"- Sisa anggaran action = {actions_left}, sisa anggaran read = {reads_left}.",
        "- Submit segera setelah node yang dipilih cukup untuk menjawab pertanyaan.",
        "- Jangan ulangi action persis sama dengan langkah sebelumnya.",
        "- Kembalikan HANYA JSON, tanpa teks lain.",
    ]

    return (
        "Kamu adalah agen retrieval dokumen hukum Indonesia. Pilih node yang paling "
        "relevan untuk menjawab pertanyaan dari katalog UU.\n\n"
        f"Pertanyaan: {query}\n\n"
        f"{scope_line}\n\n"
        "Tools.\n" + "\n".join(tools) + "\n\n"
        "Aturan.\n" + "\n".join(rules) + "\n\n"
        "Riwayat tindakan.\n" + _render_scratchpad(scratchpad) + "\n\n"
        "Format balasan.\n"
        "{\n"
        '  "thinking": "<satu kalimat alasan>",\n'
        '  "action": "list_docs" | "inspect_doc" | "expand" | "read" | "submit",\n'
        '  "args": { ... }\n'
        "}\n\n"
        "Tindakan berikutnya?\n"
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


def _resolve_doc_id(catalog: list[dict], doc_id: str) -> bool:
    """True iff doc_id is in the catalog."""
    return any(row.get("doc_id") == doc_id for row in catalog)


def _fallback_select(scratchpad: list[dict], primary_doc_id: str | None) -> list[dict]:
    """If the agent never submitted, recover the most recent committed nodes."""
    selected: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for entry in reversed(scratchpad):
        if entry.get("action") != "read":
            continue
        obs = entry.get("observation") or {}
        if "error" in obs:
            continue
        doc_id = obs.get("doc_id") or primary_doc_id
        node_id = obs.get("node_id")
        if doc_id and node_id and (doc_id, node_id) not in seen:
            seen.add((doc_id, node_id))
            selected.append({"doc_id": doc_id, "node_id": node_id})
        if len(selected) >= 3:
            break

    if selected:
        return selected

    for entry in reversed(scratchpad):
        if entry.get("action") != "expand":
            continue
        obs = entry.get("observation") or {}
        if "error" in obs:
            continue
        doc_id = obs.get("doc_id") or primary_doc_id
        for child in obs.get("children", []) or []:
            if not child.get("has_children"):
                key = (doc_id, child.get("node_id"))
                if key[0] and key[1] and key not in seen:
                    seen.add(key)
                    selected.append({"doc_id": key[0], "node_id": key[1]})
            if len(selected) >= 3:
                break
        if selected:
            break

    return selected


def retrieve(query: str, mode: str = "corpus",
             max_actions: int = MAX_ACTIONS, max_reads: int = MAX_READS,
             verbose: bool = True) -> dict:
    """Run the agentic LLM retrieval pipeline for one query.

    Args:
        query: Indonesian legal question.
        mode: 'doc' to run doc_search first then navigate inside one doc,
              'corpus' for full agentic over the catalog.
        max_actions: hard cap on agent steps before forced fallback.
        max_reads: hard cap on read tool calls (other tools are uncapped).
        verbose: print step traces during the loop.

    Returns:
        Result dict compatible with the existing eval schema. The
        sources field is what the eval harness reads for Recall and MRR.
    """
    if mode not in ("doc", "corpus"):
        raise ValueError(f"mode must be 'doc' or 'corpus', got {mode!r}")

    reset_counters()
    t_start = time.time()
    timings: dict = {}

    if verbose:
        print("=" * 60)
        print(f"Query: {query}")
        print(f"Strategy: llm-agentic-{mode}")
        print("=" * 60)

    catalog = load_catalog()
    doc_cache: dict[str, dict] = {}

    def _get_doc(doc_id: str) -> dict:
        """Lazy-load and memoize a document by id."""
        if doc_id not in doc_cache:
            doc_cache[doc_id] = load_doc(doc_id)
        return doc_cache[doc_id]

    scratchpad: list[dict] = []
    doc_result: dict | None = None
    primary_doc_id: str | None = None

    if mode == "doc":
        snap = snapshot_counters()
        t_step = time.time()
        doc_result = doc_search(query, catalog, verbose=verbose)
        timings["doc_search"] = step_metrics(t_step, snap)

        doc_ids = doc_result.get("doc_ids", [])
        if not doc_ids:
            return {
                "query": query,
                "strategy": "llm-agentic-doc",
                "doc_search": doc_result,
                "error": "No relevant documents found",
                "metrics": {**get_stats(), "elapsed_s": round(time.time() - t_start, 2),
                            "step_metrics": timings},
            }
        primary_doc_id = doc_ids[0]
        primary_doc = _get_doc(primary_doc_id)
        scratchpad.append({
            "step": 0,
            "action": "doc_search",
            "args": {},
            "observation": {"doc_id": primary_doc_id, "judul": primary_doc.get("judul", "")},
        })
        scratchpad.append({
            "step": 1,
            "action": "inspect_doc",
            "args": {"doc_id": primary_doc_id},
            "observation": _tool_inspect_doc(primary_doc),
        })
        allow_list_docs = False
    else:
        scratchpad.append({
            "step": 0,
            "action": "list_docs",
            "args": {},
            "observation": {"docs": _compact_catalog(catalog)},
        })
        allow_list_docs = True

    snap = snapshot_counters()
    t_step = time.time()

    actions_used = 0
    reads_used = 0
    submitted = False
    selected: list[dict] = []
    parse_failures = 0

    while actions_used < max_actions and not submitted:
        prompt = _build_prompt(
            query, mode, scratchpad,
            actions_left=max_actions - actions_used,
            reads_left=max_reads - reads_used,
            allow_list_docs=allow_list_docs,
            primary_doc_id=primary_doc_id,
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

        if action == "list_docs":
            if allow_list_docs:
                observation = {"docs": _compact_catalog(catalog)}
            else:
                observation = {"error": "list_docs is not available in mode 'doc'."}

        elif action == "inspect_doc":
            doc_id = args.get("doc_id") if mode == "corpus" else primary_doc_id
            if not doc_id:
                observation = {"error": "inspect_doc requires doc_id."}
            elif not _resolve_doc_id(catalog, doc_id):
                observation = {"error": f"Unknown doc_id '{doc_id}'."}
            elif mode == "doc" and doc_id != primary_doc_id:
                observation = {"error": f"Mode doc, only '{primary_doc_id}' is allowed."}
            else:
                observation = _tool_inspect_doc(_get_doc(doc_id))

        elif action == "expand":
            doc_id = args.get("doc_id") if mode == "corpus" else primary_doc_id
            node_id = args.get("node_id")
            if not (doc_id and node_id):
                observation = {"error": "expand requires doc_id and node_id."}
            elif not _resolve_doc_id(catalog, doc_id):
                observation = {"error": f"Unknown doc_id '{doc_id}'."}
            elif mode == "doc" and doc_id != primary_doc_id:
                observation = {"error": f"Mode doc, only '{primary_doc_id}' is allowed."}
            else:
                doc = _get_doc(doc_id)
                obs = _tool_expand(doc, node_id)
                if "error" in obs:
                    obs["hint_nearby"] = _siblings_hint(doc, node_id)
                observation = obs

        elif action == "read":
            if reads_used >= max_reads:
                observation = {"error": f"Read budget exhausted ({max_reads}). Submit soon."}
            else:
                doc_id = args.get("doc_id") if mode == "corpus" else primary_doc_id
                node_id = args.get("node_id")
                if not (doc_id and node_id):
                    observation = {"error": "read requires doc_id and node_id."}
                elif not _resolve_doc_id(catalog, doc_id):
                    observation = {"error": f"Unknown doc_id '{doc_id}'."}
                elif mode == "doc" and doc_id != primary_doc_id:
                    observation = {"error": f"Mode doc, only '{primary_doc_id}' is allowed."}
                else:
                    doc = _get_doc(doc_id)
                    obs = _tool_read(doc, node_id)
                    if "error" in obs:
                        obs["hint_nearby"] = _siblings_hint(doc, node_id)
                    observation = obs
                    if "error" not in obs:
                        reads_used += 1

        elif action == "submit":
            refs = args.get("node_ids") or []
            reasoning = args.get("reasoning", "")
            resolved: list[dict] = []
            invalid: list[str] = []
            for ref in refs:
                doc_id, node_id = _parse_node_ref(
                    ref, default_doc_id=primary_doc_id if mode == "doc" else None
                )
                if not (doc_id and node_id):
                    invalid.append(str(ref))
                    continue
                if not _resolve_doc_id(catalog, doc_id):
                    invalid.append(f"{doc_id}/{node_id} (unknown doc)")
                    continue
                if mode == "doc" and doc_id != primary_doc_id:
                    invalid.append(f"{doc_id}/{node_id} (out of scope)")
                    continue
                doc = _get_doc(doc_id)
                if find_node(doc.get("structure", []), node_id) is None:
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

    if not submitted:
        selected = _fallback_select(scratchpad, primary_doc_id)

    if not selected:
        return {
            "query": query,
            "strategy": f"llm-agentic-{mode}",
            "doc_search": doc_result,
            "agent": {
                "mode": mode,
                "actions_used": actions_used,
                "reads_used": reads_used,
                "submitted": submitted,
                "scratchpad": scratchpad,
            },
            "error": "Agent did not select any valid node",
            "metrics": {**get_stats(), "elapsed_s": round(time.time() - t_start, 2),
                        "step_metrics": timings},
        }

    snap = snapshot_counters()
    t_step = time.time()

    if mode == "doc":
        primary_doc = _get_doc(primary_doc_id)
        nodes = extract_nodes(primary_doc, [s["node_id"] for s in selected])
        doc_meta = {"doc_id": primary_doc_id, "judul": primary_doc.get("judul", "")}
        answer_result = generate_answer(query, nodes, doc_meta, verbose=verbose)
    else:
        results: list[dict] = []
        for s in selected:
            doc = _get_doc(s["doc_id"])
            for node in extract_nodes(doc, [s["node_id"]]):
                results.append({
                    **node,
                    "doc_id": s["doc_id"],
                    "doc_title": doc.get("judul", ""),
                })
        answer_result = generate_answer_multi_doc(query, results, verbose=verbose)

    timings["answer_gen"] = step_metrics(t_step, snap)

    sources: list[dict] = []
    for s in selected:
        doc = _get_doc(s["doc_id"])
        node = find_node(doc.get("structure", []), s["node_id"]) or {}
        sources.append({
            "doc_id": s["doc_id"],
            "node_id": s["node_id"],
            "title": node.get("title", ""),
            "navigation_path": node.get("navigation_path", ""),
        })

    elapsed = time.time() - t_start
    stats = get_stats()

    result = {
        "query": query,
        "strategy": f"llm-agentic-{mode}",
        "doc_search": doc_result,
        "agent": {
            "mode": mode,
            "actions_used": actions_used,
            "reads_used": reads_used,
            "submitted": submitted,
            "scratchpad": scratchpad,
        },
        "node_ids": [s["node_id"] for s in selected],
        "answer": answer_result.get("answer", ""),
        "citations": answer_result.get("citations", []),
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
    ap.add_argument("--mode", choices=["doc", "corpus"], default="corpus",
                    help="Agent scope, single doc after doc_search or full catalog")
    ap.add_argument("--max-actions", type=int, default=MAX_ACTIONS,
                    help=f"Hard cap on agent steps (default {MAX_ACTIONS})")
    ap.add_argument("--max-reads", type=int, default=MAX_READS,
                    help=f"Hard cap on read tool calls (default {MAX_READS})")
    args = ap.parse_args()

    result = retrieve(args.query, mode=args.mode,
                      max_actions=args.max_actions, max_reads=args.max_reads)

    print("\n" + "-" * 60)
    print(f"JAWABAN:\n{result.get('answer', 'No answer generated')}")
    print("\nDASAR HUKUM:")
    for src in result.get("sources", []):
        path = src.get("navigation_path") or src.get("node_id", "")
        print(f"  > {src.get('doc_id', '')} :: {path}")
    print("-" * 60)


if __name__ == "__main__":
    main()
