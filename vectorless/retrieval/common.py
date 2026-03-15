"""
Shared utilities for retrieval pipelines.

Contains: tokenizer, LLM client, data loading, tree helpers, answer generation, logging.
These are used by all retrieval strategies (bm25_flat, llm, hybrid, ablations).
"""

import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DATA_INDEX = Path(os.environ.get("DATA_INDEX", "data/index_pasal"))
LOG_DIR = Path("data/retrieval_logs")


# ============================================================
# TOKENIZER
# ============================================================

STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
    "ayat", "huruf", "angka",
}


def tokenize(text: str) -> list[str]:
    """Indonesian tokenizer: lowercase, split on non-alphanumeric, remove stopwords."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def tokenize_no_sw(text: str) -> list[str]:
    """Indonesian tokenizer without stopword removal (ablation variant)."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return [t for t in tokens if len(t) > 1]


# ============================================================
# LLM CLIENT
# ============================================================

_client = None
_total_input_tokens = 0
_total_output_tokens = 0
_total_calls = 0


def _get_client():
    """Lazy-init Gemini client."""
    global _client
    if _client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set.")
            sys.exit(1)
        _client = genai.Client(api_key=api_key)
    return _client


def llm_call(prompt: str, max_retries: int = 3) -> dict:
    """Send prompt to Gemini 2.5 Flash, return parsed JSON."""
    global _total_input_tokens, _total_output_tokens, _total_calls
    client = _get_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            break
        except Exception as e:
            err = str(e).lower()
            if ("rate" in err or "429" in err) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    # Track tokens
    usage = response.usage_metadata
    _total_input_tokens += usage.prompt_token_count or 0
    _total_output_tokens += usage.candidates_token_count or 0
    _total_calls += 1

    # Parse JSON from response
    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    return json.loads(text)


def reset_token_counters():
    """Reset per-query token counters."""
    global _total_input_tokens, _total_output_tokens, _total_calls
    _total_input_tokens = 0
    _total_output_tokens = 0
    _total_calls = 0


def get_token_stats() -> dict:
    """Return current token usage stats."""
    return {
        "llm_calls": _total_calls,
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
        "total_tokens": _total_input_tokens + _total_output_tokens,
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_catalog() -> list[dict]:
    """Load catalog.json for doc-level search."""
    with open(DATA_INDEX / "catalog.json", encoding="utf-8") as f:
        return json.load(f)


def _doc_category(doc_id: str) -> str:
    """Derive category subfolder from doc_id: 'uu-1-2026' → 'UU'."""
    return doc_id.split("-")[0].upper()


def load_doc(doc_id: str) -> dict:
    """Load a full index document by doc_id."""
    path = DATA_INDEX / _doc_category(doc_id) / f"{doc_id}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _collect_leaf_nodes(nodes: list[dict]) -> list[dict]:
    """Recursively collect all leaf nodes (nodes with text, no children) from tree."""
    leaves = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            leaves.extend(_collect_leaf_nodes(node["nodes"]))
        elif node.get("text"):
            leaves.append(node)
    return leaves


def load_all_leaf_nodes() -> list[dict]:
    """Load ALL leaf nodes from ALL documents in the index.

    Returns a flat list of dicts, each with doc-level metadata attached:
      doc_id, doc_title, node_id, title, navigation_path, text, penjelasan
    """
    catalog = load_catalog()
    all_leaves = []

    for doc_meta in catalog:
        doc_id = doc_meta["doc_id"]
        path = DATA_INDEX / _doc_category(doc_id) / f"{doc_id}.json"
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


# ============================================================
# TREE HELPERS
# ============================================================

def find_node(nodes: list[dict], node_id: str) -> dict | None:
    """Find a node by node_id in a tree structure (recursive)."""
    for node in nodes:
        if node.get("node_id") == node_id:
            return node
        if "nodes" in node:
            found = find_node(node["nodes"], node_id)
            if found:
                return found
    return None


def extract_nodes(doc: dict, node_ids: list[str]) -> list[dict]:
    """Look up node_ids in tree, return node info with text + penjelasan."""
    results = []
    for nid in node_ids:
        node = find_node(doc["structure"], nid)
        if node:
            results.append({
                "node_id": nid,
                "title": node.get("title", ""),
                "navigation_path": node.get("navigation_path", ""),
                "text": node.get("text", ""),
                "penjelasan": node.get("penjelasan"),
            })
    return results


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query: str, nodes: list[dict], doc_meta: dict,
                    verbose: bool = True) -> dict:
    """Generate a grounded answer from selected Pasal texts.

    Uses label-based citations [R1], [R2], etc. to prevent hallucinated references.
    Each retrieved chunk gets a label, and the LLM must cite using those labels.
    The response maps labels back to actual node_ids for traceability.
    """
    # Build label mapping: R1 -> node info
    label_map = {}
    context_parts = []
    for i, node in enumerate(nodes, 1):
        label = f"R{i}"
        label_map[label] = {
            "node_id": node["node_id"],
            "title": node.get("title", ""),
            "navigation_path": node.get("navigation_path", ""),
        }
        part = f"### [{label}] {node['title']}\n"
        part += f"Lokasi: {node['navigation_path']}\n\n"
        part += f"Isi:\n{node['text']}\n"
        if node.get("penjelasan") and node["penjelasan"] != "Cukup jelas.":
            part += f"\nPenjelasan Resmi:\n{node['penjelasan']}\n"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)
    labels_list = ", ".join(f"[{l}]" for l in label_map)

    prompt = f"""\
Kamu adalah asisten hukum Indonesia. Jawab pertanyaan berdasarkan HANYA teks Pasal yang diberikan.

Pertanyaan: {query}

Sumber hukum ({doc_meta.get('judul', '')}):
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

    result = llm_call(prompt)

    # Map labels back to node_ids
    cited_labels = result.get("cited", [])
    result["citations"] = [
        {**label_map[l], "label": l}
        for l in cited_labels if l in label_map
    ]
    result["label_map"] = label_map

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        for c in result.get("citations", []):
            print(f"  [{c['label']}] {c['node_id']} — {c['title']}")

    return result


def generate_answer_multi_doc(query: str, results: list[dict],
                              verbose: bool = True) -> dict:
    """Generate a grounded answer from results spanning multiple documents.

    Same label-based citation as generate_answer, but each chunk shows its own
    doc source instead of a single doc_meta header. Used by flat BM25 and hybrid flat.
    """
    if not results:
        return {"answer": "No answer generated", "cited": [], "citations": [], "label_map": {}}

    # Build label mapping: R1 -> node info
    label_map = {}
    context_parts = []
    for i, r in enumerate(results, 1):
        label = f"R{i}"
        label_map[label] = {
            "node_id": r["node_id"],
            "doc_id": r["doc_id"],
            "title": r.get("title", ""),
            "navigation_path": r.get("navigation_path", ""),
            "doc_title": r.get("doc_title", ""),
        }
        part = f"### [{label}] {r['title']}\n"
        part += f"Lokasi: {r['navigation_path']}\n"
        part += f"Sumber: {r['doc_title']}\n\n"
        part += f"Isi:\n{r['text']}\n"
        if r.get("penjelasan") and r["penjelasan"] != "Cukup jelas.":
            part += f"\nPenjelasan Resmi:\n{r['penjelasan']}\n"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)
    labels_list = ", ".join(f"[{l}]" for l in label_map)

    prompt = f"""\
Kamu adalah asisten hukum Indonesia. Jawab pertanyaan berdasarkan HANYA teks Pasal yang diberikan.

Pertanyaan: {query}

Sumber hukum:
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

    result = llm_call(prompt)

    # Map labels back to node_ids
    cited_labels = result.get("cited", [])
    result["citations"] = [
        {**label_map[l], "label": l}
        for l in cited_labels if l in label_map
    ]
    result["label_map"] = label_map

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        for c in result.get("citations", []):
            print(f"  [{c['label']}] {c['node_id']} — {c['title']}")

    return result


# ============================================================
# LOGGING
# ============================================================

def save_log(result: dict):
    """Save retrieval result to data/retrieval_logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
