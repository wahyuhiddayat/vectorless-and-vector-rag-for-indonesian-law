"""
Shared utilities for vector RAG retrieval pipelines.

Contains: embedding loader, query embedding, log persistence.
Used by retrieve_vector.py.

All key settings are configurable via environment variables:
    VECTOR_EMBEDDING_MODEL  e.g. bge-m3 (default)
    VECTOR_COLLECTION       e.g. law-pasal-bgem3 (default)
    QDRANT_URL              e.g. http://localhost:6333 (default)
    QDRANT_PATH             e.g. ./qdrant_local  (local mode, takes priority over URL)
    VECTOR_GRANULARITY      e.g. pasal (default), stored in result dict only
"""

import json
import os
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


_st_model_cache: dict = {}


def _get_st_model(model_id: str):
    """Create and cache a SentenceTransformer model."""
    if model_id not in _st_model_cache:
        from sentence_transformers import SentenceTransformer
        _st_model_cache[model_id] = SentenceTransformer(model_id)
    return _st_model_cache[model_id]


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


def save_log(result: dict):
    """Persist a retrieval result under `data/retrieval_logs`."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    strategy = result.get("strategy", "unknown").replace(" ", "_")
    log_path = LOG_DIR / f"{timestamp}_{strategy}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"  Log saved: {log_path.name}")
