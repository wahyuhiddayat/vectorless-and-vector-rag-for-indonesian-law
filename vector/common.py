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

EMBEDDING_MODEL = os.environ.get("VECTOR_EMBEDDING_MODEL", "bge-m3")
COLLECTION_NAME = os.environ.get("VECTOR_COLLECTION", "law-pasal-bgem3")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH = os.environ.get("QDRANT_PATH", None)
GRANULARITY = os.environ.get("VECTOR_GRANULARITY", "pasal")
LOG_DIR = Path("data/retrieval_logs")


_EMBEDDING_MODEL_MAP: dict[str, dict] = {
    "bge-m3": {
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "backend": "sentence_transformers",
    },
    "all-indobert-base-v4": {
        "model_id": "LazarusNLP/all-indobert-base-v4",
        "dim": 768,
        "backend": "sentence_transformers",
    },
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


def get_qdrant_client():
    """Return a Qdrant client for local-path or server mode."""
    from qdrant_client import QdrantClient
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(url=QDRANT_URL)


_genai_client = None
_total_input_tokens = 0
_total_output_tokens = 0
_total_calls = 0


def _get_genai_client():
    """Create the Google GenAI client on Vertex AI on first use.

    DEPRECATED. The indexing and answer-generation pipelines now use OpenAI
    via vectorless.llm. This factory is retained only for the legacy Gemini
    embedding path (gemini-embedding-001) which is not used in the current
    eval matrix. Vector RAG uses local sentence-transformer models per
    CLAUDE.md (BGE-M3, IndoBERT, multilingual-e5).
    """
    global _genai_client
    if _genai_client is None:
        from google import genai
        from google.genai import types as gtypes
        project = os.environ.get("GOOGLE_CLOUD_PROJECT", "skripsi-gavin")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        _genai_client = genai.Client(
            vertexai=True,
            project=project,
            location=location,
            http_options=gtypes.HttpOptions(api_version="v1"),
        )
    return _genai_client


_st_model_cache: dict = {}


def _get_st_model(model_id: str):
    """Create and cache a SentenceTransformer model."""
    if model_id not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        _st_model_cache[model_id] = SentenceTransformer(model_id)
    return _st_model_cache[model_id]


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


def embed_query(query: str) -> list[float]:
    """Embed a query with the configured SentenceTransformer model."""
    cfg = _EMBEDDING_MODEL_MAP.get(EMBEDDING_MODEL)
    if not cfg:
        raise ValueError(f"Unknown embedding model: {EMBEDDING_MODEL!r}")

    st = _get_st_model(cfg["model_id"])
    instruction = cfg.get("query_instruction")
    text = f"Instruct: {instruction}\nQuery: {query}" if instruction else query
    vec = st.encode(text, normalize_embeddings=True)
    return [float(x) for x in vec]


def llm_call(prompt: str, max_retries: int = 3) -> dict:
    """Send a prompt to Gemini and parse the JSON response."""
    global _total_input_tokens, _total_output_tokens, _total_calls
    from google.genai import types as gtypes
    from vectorless.llm import MODEL
    client = _get_genai_client()

    cfg_kwargs: dict = {"temperature": 0.0}
    if MODEL.startswith("gemini-2.5"):
        cfg_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=gtypes.GenerateContentConfig(**cfg_kwargs),
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


def generate_answer(query: str, results: list[dict],
                    verbose: bool = True) -> dict:
    """Generate an answer grounded in the retrieved chunks."""
    if not results:
        return {"answer": "No answer generated", "citations": []}

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


def save_log(result: dict):
    """Persist a retrieval result under `data/retrieval_logs`."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = result.get("strategy", "unknown").replace(" ", "_")
    log_path = LOG_DIR / f"{timestamp}_{strategy}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path.name}")
