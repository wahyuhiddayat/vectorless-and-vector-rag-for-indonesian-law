"""Shared helpers for the vectorless retrieval pipelines."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ..ids import doc_category

load_dotenv()

DATA_INDEX = Path(os.environ.get("DATA_INDEX", "data/index_pasal"))
LOG_DIR = Path("data/retrieval_logs")

DOC_PICK_TOP_K = 3
"""Standard top-K doc-pick for the multi-doc tree paradigm.

Tree variants (bm25-tree, hybrid-tree, llm-agentic-doc) pick up to K=3 docs at
stage 1, then navigate each hierarchy independently and merge. K=3 sits in the
center of the IR multi-stage-retrieval default range (LangChain K=4, LlamaIndex
K=5, RAG production K=3-10). K=1 was the original single-doc PageIndex setting
which assumes single-doc input, not applicable to a 308-doc corpus.
"""


def tokenize(text: str) -> list[str]:
    """Lowercase, regex split on [a-z0-9]+, drop length-1 tokens.

    No stopword removal, no stemming. Aligns with Faisal et al. (2024)
    Table 5: both preprocessing techniques decrease BM25 EM on Indonesian
    legal QA. BM25 IDF naturally downweights common terms.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1 or t.isdigit()]


def load_catalog() -> list[dict]:
    """Load the document catalog at the active DATA_INDEX."""
    with open(DATA_INDEX / "catalog.json", encoding="utf-8") as f:
        return json.load(f)


def doc_corpus_string(doc_meta: dict) -> str:
    """Build the BM25 doc-level corpus string for one catalog entry.

    Prefers `doc_summary_text` (aggregated leaf summaries added by the
    2026-05-13 catalog enrichment in `indexing/build.py`). Falls back to
    the metadata fields when the summary field is absent so the helper
    works both before and after the catalog rebuild.
    """
    parts = [
        doc_meta.get("judul") or "",
        doc_meta.get("bidang") or "",
        doc_meta.get("subjek") or "",
        doc_meta.get("materi_pokok") or "",
    ]
    summary_text = doc_meta.get("doc_summary_text") or ""
    if summary_text:
        parts.append(summary_text)
    return " ".join(parts)


def catalog_for_llm_prompt(catalog: list[dict], summary_cap: int = 600) -> list[dict]:
    """Return a slim catalog projection suitable for an LLM doc-pick prompt.

    The full `doc_summary_text` per doc can be several thousand characters,
    so dumping the entire catalog inflates the prompt past a comfortable
    budget. This helper keeps the metadata fields intact and truncates the
    aggregated summary to `summary_cap` characters per doc, preserving the
    leading signal (top-level pasals first) which is the most relevant for
    doc-level topical match.
    """
    slim = []
    for doc in catalog:
        entry = {
            "doc_id": doc.get("doc_id"),
            "judul": doc.get("judul"),
            "bidang": doc.get("bidang"),
            "subjek": doc.get("subjek"),
            "materi_pokok": doc.get("materi_pokok"),
        }
        summary_text = doc.get("doc_summary_text") or ""
        if summary_text:
            entry["doc_summary_text"] = summary_text[:summary_cap]
        slim.append(entry)
    return slim


def load_doc(doc_id: str) -> dict:
    """Load one indexed document."""
    path = DATA_INDEX / doc_category(doc_id) / f"{doc_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    leaves = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            leaves.extend(_collect_leaf_nodes(node["nodes"]))
        elif node.get("text"):
            leaves.append(node)
    return leaves


def load_all_leaf_nodes() -> list[dict]:
    """Flat list of every leaf node across all docs in the active index."""
    catalog = load_catalog()
    all_leaves = []
    for doc_meta in catalog:
        doc_id = doc_meta["doc_id"]
        path = DATA_INDEX / doc_category(doc_id) / f"{doc_id}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        for node in _collect_leaf_nodes(doc.get("structure", [])):
            all_leaves.append({
                "doc_id": doc_id,
                "doc_title": doc_meta["judul"],
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "navigation_path": node.get("navigation_path", ""),
                "text": node.get("text", ""),
                "penjelasan": node.get("penjelasan"),
                "summary": node.get("summary", ""),
            })
    return all_leaves


def find_node(nodes: list[dict], node_id: str) -> dict | None:
    """Locate one node in the tree by node_id."""
    for node in nodes:
        if node.get("node_id") == node_id:
            return node
        if "nodes" in node:
            found = find_node(node["nodes"], node_id)
            if found:
                return found
    return None


def extract_nodes(doc: dict, node_ids: list[str]) -> list[dict]:
    """Resolve node_ids in doc.structure to compact dicts with text + penjelasan."""
    out = []
    for nid in node_ids:
        node = find_node(doc["structure"], nid)
        if node:
            out.append({
                "node_id": nid,
                "title": node.get("title", ""),
                "navigation_path": node.get("navigation_path", ""),
                "text": node.get("text", ""),
                "penjelasan": node.get("penjelasan"),
            })
    return out


def agentic_finalize(submitted_ids: list[str],
                     top_k: int) -> tuple[list[str], list[str]]:
    """Finalize an agentic retrieval ranking from the agent's submitted ids.

    Returns the agent's submitted ids deduplicated and truncated to top_k.
    The output may be shorter than top_k when the agent chooses to submit
    fewer candidates. This is intentional. The method is the agent, padding
    the output with BM25 or visited nodes would conflate paradigms and the
    7q audit showed both layers added negligible recall while inflating the
    BM25 share of slots to roughly 70 percent.

    Args:
        submitted_ids: ordered ids submitted by the agent, most relevant first.
        top_k: maximum output length.

    Returns:
        Tuple `(final_ranking, sources_per_slot)`. `final_ranking` has length
        min(top_k, unique submitted ids). `sources_per_slot` is a parallel
        list labelling each slot as `agent_submit`.
    """
    seen: set[str] = set()
    final: list[str] = []
    labels: list[str] = []
    for nid in submitted_ids:
        if nid and nid not in seen:
            final.append(nid)
            labels.append("agent_submit")
            seen.add(nid)
            if len(final) >= top_k:
                break
    return final, labels


def validate_llm_ranking(llm_ranking: list[str], candidates: list[dict]) -> list[str]:
    """Validate and complete an LLM-generated ranking over candidate node_ids.

    Drops hallucinated and duplicate IDs, then appends any missing candidate IDs
    in original first-stage order so the output length always matches the input
    candidate count. Used by hybrid-flat and hybrid-tree LLM rerank stages.

    Args:
        llm_ranking: list of node_ids returned by the LLM in descending relevance
            order. May contain hallucinations, duplicates, or be shorter than
            len(candidates).
        candidates: list of candidate dicts (each with `node_id`), in first-stage
            (BM25) order. Used for the valid-id set and the deterministic
            tiebreak fallback.

    Returns:
        List of node_ids of length len(candidates), unique, all present in the
        candidate set, with the LLM ranking honoured first and any missing
        candidates appended in first-stage order.
    """
    valid_order = [c["node_id"] for c in candidates]
    valid_set = set(valid_order)
    seen: set[str] = set()
    cleaned: list[str] = []
    for nid in llm_ranking:
        if nid in valid_set and nid not in seen:
            cleaned.append(nid)
            seen.add(nid)
    for nid in valid_order:
        if nid not in seen:
            cleaned.append(nid)
            seen.add(nid)
    return cleaned


def save_log(result: dict) -> None:
    """Persist a retrieval result under data/retrieval_logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
