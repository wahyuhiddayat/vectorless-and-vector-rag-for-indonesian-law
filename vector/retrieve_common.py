"""
Shared utilities for vector RAG retrieval pipelines.

Contains: LLM client, embedding, token tracking, answer generation, logging.
Used by retrieve_vector.py and retrieve_vector_hybrid.py.

All key settings are configurable via environment variables:
    VECTOR_EMBEDDING_MODEL  e.g. gemini-embedding-001 (default)
    VECTOR_COLLECTION       e.g. law-pasal-gemini (default)
    QDRANT_URL              e.g. http://localhost:6333 (default)
    QDRANT_PATH             e.g. ./qdrant_local  (local mode, takes priority over URL)
    VECTOR_GRANULARITY      e.g. pasal (default) — stored in result dict only
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION (all overridable via env vars)
# ============================================================

EMBEDDING_MODEL = os.environ.get("VECTOR_EMBEDDING_MODEL", "bge-m3")
COLLECTION_NAME = os.environ.get("VECTOR_COLLECTION", "law-pasal-bgem3")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.environ.get("QDRANT_PATH", None)  # local mode (takes priority over URL)
GRANULARITY = os.environ.get("VECTOR_GRANULARITY", "pasal")
LOG_DIR = Path("data/retrieval_logs")


# ============================================================
# EMBEDDING MODEL CONFIG
# ============================================================

_EMBEDDING_MODEL_MAP: dict[str, dict] = {
    # Hypothesis A: MIRACL SOTA, 8K context, handles long pasal
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "backend": "sentence_transformers",
    },
    # Hypothesis B: Indonesian-specific IndoBERT, 128-token limit (finding, not flaw)
    "all-indobert-base-v4": {
        "model_id": "LazarusNLP/all-indobert-base-v4",
        "dim": 768,
        "backend": "sentence_transformers",
    },
    # Hypothesis C: MMTEB best public multilingual, instruction-tuned, 512-token context
    "multilingual-e5-large-instruct": {
        "model_id": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "backend": "sentence_transformers",
        "query_instruction": (
            "Given a legal question in Indonesian, retrieve relevant legal "
            "document sections that answer the question"
        ),
    },
}


def get_embedding_dim() -> int:
    """Return embedding dimension for the currently configured model."""
    cfg = _EMBEDDING_MODEL_MAP.get(EMBEDDING_MODEL)
    if not cfg:
        raise ValueError(f"Unknown embedding model: {EMBEDDING_MODEL!r}")
    return cfg["dim"]


# ============================================================
# QDRANT CLIENT
# ============================================================

def get_qdrant_client():
    """Return Qdrant client: local path mode if QDRANT_PATH is set, else URL mode."""
    from qdrant_client import QdrantClient
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(url=QDRANT_URL)


# ============================================================
# CLIENTS / TOKEN COUNTERS
# ============================================================

_genai_client = None
_total_input_tokens = 0
_total_output_tokens = 0
_total_calls = 0


def _get_genai_client():
    """Lazy-init Google GenAI client."""
    global _genai_client
    if _genai_client is None:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            print("ERROR: GEMINI_API_KEY not set.")
            sys.exit(1)
        _genai_client = genai.Client(api_key=api_key)
    return _genai_client


_st_model_cache: dict = {}  # model_id -> SentenceTransformer instance


def _get_st_model(model_id: str):
    """Lazy-init SentenceTransformer model (cached by model_id)."""
    if model_id not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        _st_model_cache[model_id] = SentenceTransformer(model_id)
    return _st_model_cache[model_id]


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
# EMBEDDING
# ============================================================

def embed_query(query: str) -> list[float]:
    """Embed a single query using the configured SentenceTransformer model.

    For multilingual-e5-large-instruct the query is wrapped with the task instruction.
    For bge-m3 and all-indobert-base-v4 no prefix is needed.
    """
    cfg = _EMBEDDING_MODEL_MAP.get(EMBEDDING_MODEL)
    if not cfg:
        raise ValueError(f"Unknown embedding model: {EMBEDDING_MODEL!r}")

    st = _get_st_model(cfg["model_id"])
    instruction = cfg.get("query_instruction")
    text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
    vec = st.encode(text, normalize_embeddings=True)
    return [float(x) for x in vec]


# ============================================================
# LLM
# ============================================================

def llm_call(prompt: str, max_retries: int = 3) -> dict:
    """Send prompt to Gemini 2.5 Flash, return parsed JSON."""
    global _total_input_tokens, _total_output_tokens, _total_calls
    client = _get_genai_client()

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

    usage = response.usage_metadata
    _total_input_tokens += usage.prompt_token_count or 0
    _total_output_tokens += usage.candidates_token_count or 0
    _total_calls += 1

    text = response.text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1])

    return json.loads(text)


# ============================================================
# ANSWER GENERATION
# ============================================================

def generate_answer(query: str, results: list[dict],
                    verbose: bool = True) -> dict:
    """Generate a grounded answer with label-based citations [R1], [R2].

    Citation format matches vectorless-rag for fair evaluation comparison.

    Returns:
        {
            "answer": "...[R1]...",
            "citations": [{"label": "[R1]", "node_id": "...", "doc_id": "...", ...}]
        }
    """
    if not results:
        return {"answer": "No answer generated", "citations": []}

    # Build label map: R1, R2, ... for each retrieved chunk
    label_map: dict[str, dict] = {}
    context_parts: list[str] = []
    for i, r in enumerate(results, 1):
        label = f"R{i}"
        label_map[label] = {
            "node_id": r.get("node_id", ""),
            "doc_id": r.get("doc_id", ""),
            "title": r.get("title", ""),
            "navigation_path": r.get("navigation_path", ""),
        }
        part = f"[{label}] {r['title']}\n"
        part += f"Lokasi: {r['navigation_path']}\n"
        part += f"Sumber: {r['doc_title']}\n\n"
        part += f"Isi:\n{r['text']}\n"
        context_parts.append(part)

    context = "\n---\n".join(context_parts)

    prompt = f"""\
Kamu adalah asisten hukum Indonesia. Jawab pertanyaan berdasarkan HANYA teks Pasal yang diberikan.

Pertanyaan: {query}

Sumber hukum:
{context}

Balas dalam format JSON:
{{
  "answer": "<jawaban yang jelas dan lengkap, gunakan label [R1], [R2] dst untuk sitasi>",
  "cited_labels": ["R1", "R2"]
}}

Aturan:
- Jawab berdasarkan teks Pasal saja, JANGAN mengarang
- Jika ada Penjelasan Resmi, gunakan untuk menginterpretasi
- Gunakan label [R1], [R2] dst di dalam teks jawaban untuk sitasi inline
- Di "cited_labels", hanya sertakan label yang benar-benar dikutip dalam jawaban
- Jawab dalam Bahasa Indonesia
- Kembalikan HANYA JSON
"""

    raw = llm_call(prompt)

    cited_labels = [lbl for lbl in (raw.get("cited_labels") or []) if lbl in label_map]
    citations = [{"label": f"[{lbl}]", **label_map[lbl]} for lbl in cited_labels]

    if verbose:
        print(f"\n[Answer] {raw.get('answer', '')[:300]}")
        print(f"  Citations: {[c['label'] for c in citations]}")

    return {
        "answer": raw.get("answer", ""),
        "citations": citations,
    }


# ============================================================
# LOGGING
# ============================================================

def save_log(result: dict):
    """Save retrieval result to data/retrieval_logs/."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = result.get("strategy", "unknown").replace(" ", "_")
    log_path = LOG_DIR / f"{timestamp}_{strategy}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path.name}")
