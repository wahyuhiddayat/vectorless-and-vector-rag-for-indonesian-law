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
    VECTOR_RERANKER         e.g. none (default), bge-reranker-v2-m3, qwen3-reranker-0.6b
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
RERANKER = os.environ.get("VECTOR_RERANKER", "none")
LOG_DIR = Path("data/retrieval_logs")

# First-stage candidate count fed to the reranker. Final top-k=10 returned to caller.
# 50 chosen per Notes/06-decisions/2026-05-04-rq2-reranker-iv.md, balances recall ceiling
# with reranker latency budget. See ADR for rationale and source citations.
RERANKER_TOP_N = 50


# RQ2 axis. Indonesian specialization gradient (breadth vs depth of training data).
# All three are XLM-R or BERT-family encoders, sentence-transformers compatible,
# Qdrant cosine, dense-only. See Notes/06-decisions/2026-05-03-embedding-model-axis.md.
_EMBEDDING_MODEL_MAP: dict[str, dict] = {
    "bge-m3": {
        # Tier 1. Broad multilingual contrastive. XLM-R-large, MIT, native 8K ctx.
        # Dense-only in RQ2. Sparse and ColBERT capabilities intentionally unused.
        "model_id": "BAAI/bge-m3",
        "dim": 1024,
        "backend": "sentence_transformers",
    },
    "multilingual-e5-large-instruct": {
        # Tier 2. Multilingual plus instruction-tuned. XLM-R-large with GPT-4
        # synthetic data, MIT, 512 ctx.
        "model_id": "intfloat/multilingual-e5-large-instruct",
        "dim": 1024,
        "backend": "sentence_transformers",
        "query_instruction": (
            "Given a legal question in Indonesian, retrieve relevant legal "
            "document sections that answer the question"
        ),
    },
    "all-nusabert-large-v4": {
        # Tier 3. Indonesian and Nusantara supervised. NusaBERT-large, Apache-2.0, 512 ctx.
        "model_id": "LazarusNLP/all-nusabert-large-v4",
        "dim": 1024,
        "backend": "sentence_transformers",
    },
}


# RQ2 reranker IV. Scoring-paradigm tier (no-rerank, encoder cross-attention, LLM pointwise).
# See Notes/06-decisions/2026-05-04-rq2-reranker-iv.md for axis justification and model selection.
_RERANKER_REGISTRY: dict[str, dict] = {
    "none": {
        # R0. No reranker. First-stage Qdrant top-k=10 returned directly.
        "model_id": None,
        "backend": "none",
    },
    "bge-reranker-v2-m3": {
        # R1. Encoder cross-encoder. XLM-R-large seq-cls, Apache-2.0, 8K ctx.
        # Indonesian seen via MIRACL-id training mixture.
        "model_id": "BAAI/bge-reranker-v2-m3",
        "backend": "cross_encoder",
    },
    "qwen3-reranker-0.6b": {
        # R2. Decoder LLM pointwise yes/no logit. Qwen3-0.6B, Apache-2.0, 32K ctx.
        # Use the seq-cls conversion checkpoint for sentence-transformers CrossEncoder API.
        "model_id": "tomaarsen/Qwen3-Reranker-0.6B-seq-cls",
        "backend": "cross_encoder",
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
