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


# Indonesian stopwords. Scope: KWIC anchor selection only (see _content_tokens
# below). NOT applied to BM25 tokenization. Faisal et al. (2024, IJAIN 10(3))
# Table 5 reports stopword removal decreases BM25 EM on Indonesian legal QA
# from 29.73 to 26.49 percent. BM25 IDF naturally downweights common terms.
STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
    "ayat", "huruf", "angka",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, regex split on [a-z0-9]+, drop length-1 tokens.

    No stopword removal, no stemming. Aligns with Faisal et al. (2024)
    Table 5: both preprocessing techniques decrease BM25 EM on Indonesian
    legal QA. BM25 IDF naturally downweights common terms.
    """
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 1]


def _content_tokens(text: str) -> list[str]:
    """Tokenize and drop STOPWORDS. Used by KWIC snippet anchor selection.

    Stopword removal helps here because we need a content word to anchor
    the snippet window, and anchoring on a particle (e.g. "yang") would
    almost always match position 0 of the leaf text.
    """
    return [t for t in tokenize(text) if t not in STOPWORDS]


def load_catalog() -> list[dict]:
    """Load the document catalog at the active DATA_INDEX."""
    with open(DATA_INDEX / "catalog.json", encoding="utf-8") as f:
        return json.load(f)


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
            })
    return all_leaves


def extract_kwic_snippet(text: str, query: str, window: int = 200) -> str:
    """Window of text around the first matching query token; falls back to text head."""
    text_lower = text.lower()
    best_pos = -1
    for token in _content_tokens(query):
        pos = text_lower.find(token)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        return text[: window * 2]

    start = max(0, best_pos - window)
    end = min(len(text), best_pos + window)
    snippet = text[start:end]
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."
    return snippet


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


def raptor_finalize(submitted_ids: list[str],
                    visited_ids: list[str],
                    fallback_ids: list[str],
                    top_k: int) -> tuple[list[str], list[str]]:
    """Finalize an agentic retrieval ranking with RAPTOR-style fallback.

    Builds the final ordered list of `top_k` node_ids by stacking three layers
    in order of preference, deduplicating across layers:
      1. agent_submit       node_ids the agent explicitly submitted
      2. visited_unsubmitted node_ids visited during navigation but not submitted
      3. bm25_fallback      ranked fallback (typically BM25 over doc leaves)

    Mirrors the tree-retrieval evaluation convention used by Sarthi et al.
    (RAPTOR, ICLR 2024) where the final retrieved set is a fixed-cardinality
    list capped at top_k.

    Args:
        submitted_ids: ordered ids submitted by the agent, most relevant first.
        visited_ids:   ordered ids the agent inspected but did not submit, in
            descending visit recency.
        fallback_ids:  deterministic fallback ordering (e.g. BM25 leaves of the
            primary doc) used to fill remaining slots.
        top_k: target output length.

    Returns:
        Tuple `(final_ranking, sources_per_slot)`. `final_ranking` has length
        min(top_k, total unique ids). `sources_per_slot` is a parallel list
        labelling each slot as `agent_submit`, `visited_unsubmitted`, or
        `bm25_fallback`, used for telemetry.
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
                return final, labels
    for nid in visited_ids:
        if nid and nid not in seen:
            final.append(nid)
            labels.append("visited_unsubmitted")
            seen.add(nid)
            if len(final) >= top_k:
                return final, labels
    for nid in fallback_ids:
        if nid and nid not in seen:
            final.append(nid)
            labels.append("bm25_fallback")
            seen.add(nid)
            if len(final) >= top_k:
                return final, labels
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
