"""Shared helpers for the vectorless retrieval pipelines."""

import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

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
    """Lowercase, split, and drop common Indonesian stopwords."""
    text = text.lower()
    tokens = re.findall(r'[a-z0-9]+', text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


_client = None
_total_input_tokens = 0
_total_output_tokens = 0
_total_calls = 0


def _get_client():
    """Lazy-init Gemini client with 120s hard HTTP timeout.

    The timeout is set at client level via HttpOptions (timeout in ms) so that
    hung API calls cannot stall the subprocess indefinitely — a known failure
    mode on Windows where subprocess.run(timeout=...) + capture_output=True
    does not reliably kill a child process that is blocked in a network read.
    """
    global _client
    if _client is None:
        from google import genai
        from google.genai import types as _genai_types
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set.")
            sys.exit(1)
        _client = genai.Client(
            api_key=api_key,
            http_options=_genai_types.HttpOptions(timeout=120_000),
        )
    return _client


def llm_call(prompt: str, max_retries: int = 3) -> dict:
    """Send prompt to Gemini 2.5 Flash, return parsed JSON.

    max_retries is 3 (down from 5) so that worst-case accumulated wait time
    (90s call + 5s + 90s + 10s + 90s = ~285s) stays well under the 600s
    subprocess timeout used by the evaluation harness.
    """
    global _total_input_tokens, _total_output_tokens, _total_calls
    client = _get_client()

    _RETRYABLE = ("rate", "429", "503", "500", "quota", "resource_exhausted",
                  "deadline_exceeded", "service unavailable", "overloaded",
                  "timeout", "timed out", "connection")

    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            # gemini-2.5-flash with thinking_budget=0 disables the thinking phase,
            # giving fast responses (~5-15s) comparable to 2.0-flash while staying
            # on the latest model. (gemini-2.0-flash is deprecated for new users.)
            from google.genai import types as _genai_types
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=_genai_types.GenerateContentConfig(
                    thinking_config=_genai_types.ThinkingConfig(thinking_budget=0)
                ),
            )
            # Track tokens
            usage = response.usage_metadata
            _total_input_tokens += usage.prompt_token_count or 0
            _total_output_tokens += usage.candidates_token_count or 0
            _total_calls += 1

            # Parse JSON from response (retry if Gemini returns non-JSON)
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                text = "\n".join(lines[1:-1])

            try:
                return json.loads(text)
            except json.JSONDecodeError as json_err:
                last_exc = json_err
                if attempt < max_retries - 1:
                    # Start at 5s (was 15s) so 3 retries fit within 600s subprocess budget.
                    wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 5)
                    sys.stderr.write(f"  Gemini returned non-JSON (attempt {attempt+1}), retrying in {wait:.0f}s...\n")
                    time.sleep(wait)
                    continue
                raise

        except json.JSONDecodeError:
            raise
        except Exception as e:
            last_exc = e
            err = str(e).lower()
            if any(tok in err for tok in _RETRYABLE) and attempt < max_retries - 1:
                wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 5)
                sys.stderr.write(f"  Gemini error (attempt {attempt+1}): {e!r} — retrying in {wait:.0f}s...\n")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError(f"llm_call failed after {max_retries} attempts") from last_exc


def reset_token_counters():
    """Reset the in-process token counters."""
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


def snapshot_token_counters() -> dict:
    """Snapshot current token counters for delta computation.

    Use with compute_step_metrics() to measure per-step LLM usage:
        snap = snapshot_token_counters()
        t = time.time()
        # ... do work ...
        step = compute_step_metrics(t, snap)
    """
    return {
        "llm_calls": _total_calls,
        "input_tokens": _total_input_tokens,
        "output_tokens": _total_output_tokens,
    }


def compute_step_metrics(t_start: float, snap_before: dict) -> dict:
    """Compute elapsed time and token deltas since snapshot."""
    snap_after = snapshot_token_counters()
    return {
        "elapsed_s": round(time.time() - t_start, 3),
        "llm_calls": snap_after["llm_calls"] - snap_before["llm_calls"],
        "input_tokens": snap_after["input_tokens"] - snap_before["input_tokens"],
        "output_tokens": snap_after["output_tokens"] - snap_before["output_tokens"],
    }


def load_catalog() -> list[dict]:
    """Load the document catalog."""
    with open(DATA_INDEX / "catalog.json", encoding="utf-8") as f:
        return json.load(f)


_MULTI_WORD_PREFIXES = {
    "peraturan-bssn": "PERATURAN_BSSN",
    "peraturan-ojk": "PERATURAN_OJK",
}


def _doc_category(doc_id: str) -> str:
    """Derive category subfolder from doc_id: 'uu-1-2026' -> 'UU', 'peraturan-bssn-1-2025' -> 'PERATURAN_BSSN'."""
    low = doc_id.lower()
    for prefix, folder in _MULTI_WORD_PREFIXES.items():
        if low.startswith(prefix + "-"):
            return folder
    return doc_id.split("-")[0].upper()


def load_doc(doc_id: str) -> dict:
    """Load one indexed document."""
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


def extract_kwic_snippet(text: str, query: str, window: int = 200) -> str:
    """Extract a snippet around the first matching query token."""
    text_lower = text.lower()
    query_tokens = tokenize(query)

    # Find the position of the best-matching query term
    best_pos = -1
    for token in query_tokens:
        pos = text_lower.find(token)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos

    if best_pos == -1:
        # No keyword found — fall back to start of text
        snippet = text[:window * 2]
    else:
        # Extract window around the match
        start = max(0, best_pos - window)
        end = min(len(text), best_pos + window)
        snippet = text[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(text):
            snippet += "..."

    return snippet


def find_node(nodes: list[dict], node_id: str) -> dict | None:
    """Find one node in the tree by `node_id`."""
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


def generate_answer(query: str, nodes: list[dict], doc_meta: dict,
                    verbose: bool = True) -> dict:
    """Generate an answer grounded in one document's selected nodes."""
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

    cited_labels = result.get("cited", [])
    result["citations"] = [
        {**label_map[l], "label": l}
        for l in cited_labels if l in label_map
    ]
    result["label_map"] = label_map

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        for c in result.get("citations", []):
            print(f"  [{c['label']}] {c['node_id']} â€” {c['title']}")

    return result


def generate_answer_multi_doc(query: str, results: list[dict],
                              verbose: bool = True) -> dict:
    """Generate an answer grounded in nodes from multiple documents."""
    if not results:
        return {"answer": "No answer generated", "cited": [], "citations": [], "label_map": {}}

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

    cited_labels = result.get("cited", [])
    result["citations"] = [
        {**label_map[l], "label": l}
        for l in cited_labels if l in label_map
    ]
    result["label_map"] = label_map

    if verbose:
        print(f"\n[Answer] {result.get('answer', '')[:300]}")
        for c in result.get("citations", []):
            print(f"  [{c['label']}] {c['node_id']} â€” {c['title']}")

    return result


def save_log(result: dict):
    """Persist a retrieval result under `data/retrieval_logs`."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{timestamp}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
