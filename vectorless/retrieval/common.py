"""Shared helpers for the vectorless retrieval pipelines."""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from ..ids import doc_category
from ..llm import call as llm_call

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


def _format_source(label: str, node: dict, include_doc: bool = False) -> str:
    parts = [f"### [{label}] {node['title']}", f"Lokasi: {node['navigation_path']}"]
    if include_doc and node.get("doc_title"):
        parts.append(f"Sumber: {node['doc_title']}")
    parts.append("")
    parts.append(f"Isi:\n{node['text']}")
    if node.get("penjelasan") and node["penjelasan"] != "Cukup jelas.":
        parts.append(f"\nPenjelasan Resmi:\n{node['penjelasan']}")
    return "\n".join(parts) + "\n"


def _build_answer_prompt(query: str, header: str, context: str, labels: list[str]) -> str:
    labels_list = ", ".join(f"[{l}]" for l in labels)
    return f"""\
Kamu adalah asisten hukum Indonesia. Jawab pertanyaan berdasarkan HANYA teks Pasal yang diberikan.

Pertanyaan: {query}

{header}
{context}

Balas dalam format JSON:
{{
  "answer": "<jawaban yang jelas dan lengkap, sisipkan [R1], [R2], dll. di kalimat yang mengacu ke sumber>",
  "cited": ["R1", "R2"]
}}

Aturan:
- Jawab berdasarkan teks Pasal saja, JANGAN mengarang
- Jika ada Penjelasan Resmi, gunakan untuk menginterpretasi
- Label sumber yang tersedia: {labels_list}
- Sisipkan label [R1], [R2], dll. langsung di dalam teks jawaban, tepat setelah kalimat/fakta yang mengacu ke sumber tersebut
- "cited" berisi daftar label yang dipakai (tanpa kurung siku)
- Jawab dalam Bahasa Indonesia
- Kembalikan HANYA JSON
"""


def generate_answer(query: str, nodes: list[dict], doc_meta: dict,
                    verbose: bool = True) -> dict:
    """Answer a query grounded in nodes from a single document."""
    label_map = {}
    parts = []
    for i, node in enumerate(nodes, 1):
        label = f"R{i}"
        label_map[label] = {
            "node_id": node["node_id"],
            "title": node.get("title", ""),
            "navigation_path": node.get("navigation_path", ""),
        }
        parts.append(_format_source(label, node))

    prompt = _build_answer_prompt(
        query,
        header=f"Sumber hukum ({doc_meta.get('judul', '')}):",
        context="\n---\n".join(parts),
        labels=list(label_map),
    )
    result = llm_call(prompt)
    result["citations"] = [
        {**label_map[l], "label": l} for l in result.get("cited", []) if l in label_map
    ]
    result["label_map"] = label_map

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        for c in result["citations"]:
            print(f"  [{c['label']}] {c['node_id']} — {c['title']}")
    return result


def generate_answer_multi_doc(query: str, results: list[dict],
                              verbose: bool = True) -> dict:
    """Answer a query grounded in nodes drawn from multiple documents."""
    if not results:
        return {"answer": "No answer generated", "cited": [], "citations": [], "label_map": {}}

    label_map = {}
    parts = []
    for i, r in enumerate(results, 1):
        label = f"R{i}"
        label_map[label] = {
            "node_id": r["node_id"],
            "doc_id": r["doc_id"],
            "title": r.get("title", ""),
            "navigation_path": r.get("navigation_path", ""),
            "doc_title": r.get("doc_title", ""),
        }
        parts.append(_format_source(label, r, include_doc=True))

    prompt = _build_answer_prompt(
        query,
        header="Sumber hukum:",
        context="\n---\n".join(parts),
        labels=list(label_map),
    )
    result = llm_call(prompt)
    result["citations"] = [
        {**label_map[l], "label": l} for l in result.get("cited", []) if l in label_map
    ]
    result["label_map"] = label_map

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        for c in result["citations"]:
            print(f"  [{c['label']}] {c['node_id']} — {c['title']}")
    return result


def save_log(result: dict) -> None:
    """Persist a retrieval result under data/retrieval_logs."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(LOG_DIR / f"{timestamp}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
