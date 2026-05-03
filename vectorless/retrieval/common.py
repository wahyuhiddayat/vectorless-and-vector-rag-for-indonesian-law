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


STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
    "ayat", "huruf", "angka",
}


def tokenize(text: str) -> list[str]:
    """Lowercase, split, drop common Indonesian stopwords."""
    text = text.lower()
    tokens = re.findall(r"[a-z0-9]+", text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


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
    for token in tokenize(query):
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


def save_log(result: dict) -> None:
    """Persist a retrieval result under data/retrieval_logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
