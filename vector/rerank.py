"""Reranker stage for vector RAG. Scores first-stage candidates and reorders.

Two backends supported via sentence-transformers `CrossEncoder` API. Both load
HuggingFace models locally, no API call.

Backends:
    cross_encoder    Encoder cross-attention (e.g. BAAI/bge-reranker-v2-m3).
                     Single forward pass per (query, doc) pair, output relevance logit.
    cross_encoder    Decoder-LLM pointwise yes/no logit via tomaarsen seq-cls
                     checkpoint conversion (e.g. tomaarsen/Qwen3-Reranker-0.6B-seq-cls).
                     Same API surface, different scoring paradigm.

The "none" reranker is a no-op pass-through, used for the R0 baseline.

See Notes/06-decisions/vector-reranker-axis.md for model selection rationale.
"""

from .common import _RERANKER_REGISTRY


_ce_model_cache: dict = {}


def _get_cross_encoder(model_id: str):
    """Load and cache a sentence-transformers CrossEncoder model."""
    if model_id not in _ce_model_cache:
        from sentence_transformers import CrossEncoder
        _ce_model_cache[model_id] = CrossEncoder(model_id)
    return _ce_model_cache[model_id]


def rerank(query: str, candidates: list[dict], reranker_name: str,
           top_k: int = 10) -> list[dict]:
    """Rerank candidates with the configured reranker, return top_k by descending score.

    Args:
        query: Indonesian legal question.
        candidates: list of dicts each containing at least a `text` key. Order preserved
            from first-stage retrieval. Other keys (doc_id, node_id, etc.) propagate.
        reranker_name: registry key in `_RERANKER_REGISTRY`. "none" returns the first
            top_k candidates unchanged.
        top_k: number of candidates to return after reranking.

    Returns:
        Reranked list of candidate dicts, length up to top_k. Each dict gets a new key
        `rerank_score` (None for "none" backend).
    """
    cfg = _RERANKER_REGISTRY.get(reranker_name)
    if cfg is None:
        raise ValueError(f"Unknown reranker: {reranker_name!r}")

    if cfg["backend"] == "none":
        out = []
        for c in candidates[:top_k]:
            c_copy = dict(c)
            c_copy["rerank_score"] = None
            out.append(c_copy)
        return out

    if cfg["backend"] == "cross_encoder":
        ce = _get_cross_encoder(cfg["model_id"])
        pairs = [(query, c["text"]) for c in candidates]
        scores = ce.predict(pairs, show_progress_bar=False)
        scored = [(float(s), c) for s, c in zip(scores, candidates)]
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for s, c in scored[:top_k]:
            c_copy = dict(c)
            c_copy["rerank_score"] = s
            out.append(c_copy)
        return out

    raise ValueError(f"Unsupported reranker backend: {cfg['backend']!r}")
