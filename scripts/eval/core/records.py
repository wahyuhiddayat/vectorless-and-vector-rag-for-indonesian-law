"""Per-query record building.

Converts raw worker output into a flat record suitable for JSONL storage
and aggregation.
"""

from __future__ import annotations

from .metrics import (
    GOLD_KEY_BY_GRANULARITY,
    compute_doc_pick_diagnostics,
    score_ranked_retrieval,
    unique_preserve_order,
)


def normalize_worker_payload(payload: dict | None) -> dict:
    """Flatten and validate the worker JSON output for downstream scoring."""
    if not payload:
        return {
            "worker_ok": False,
            "error": "Missing worker payload",
            "retrieved_sources": [],
            "retrieved_node_ids": [],
            "picked_doc_ids": [],
            "metrics": {},
            "llm_model": None,
        }

    if not payload.get("ok"):
        return {
            "worker_ok": False,
            "error": payload.get("error", "Worker error"),
            "traceback": payload.get("traceback", ""),
            "retrieved_sources": [],
            "retrieved_node_ids": [],
            "picked_doc_ids": [],
            "metrics": {},
            "llm_model": payload.get("llm_model"),
        }

    raw = payload.get("result", {})
    sources = raw.get("sources", []) or []
    picked_doc_ids = raw.get("picked_doc_ids") or []

    return {
        "worker_ok": True,
        "error": raw.get("error", ""),
        "retrieved_sources": sources,
        "retrieved_node_ids": unique_preserve_order([src.get("node_id", "") for src in sources]),
        "picked_doc_ids": list(picked_doc_ids),
        "metrics": raw.get("metrics", {}) or {},
        "raw_strategy": raw.get("strategy", payload.get("system", "")),
        "llm_model": payload.get("llm_model"),
    }


def build_per_query_record(
    qid: str,
    item: dict,
    system: str,
    granularity: str,
    cutoffs: list[int],
    normalized: dict,
    worker_stdout: str,
    worker_stderr: str,
    retry_count: int = 0,
    error_category: str = "",
) -> dict:
    gold_key = GOLD_KEY_BY_GRANULARITY[granularity]
    relevant_ids = set(item.get(gold_key, set()))
    retrieval_metrics = score_ranked_retrieval(
        normalized["retrieved_node_ids"], relevant_ids, cutoffs
    )

    gold_doc_ids = list(item.get("gold_doc_ids", [item.get("gold_doc_id", "")]))
    gold_doc_ids = [d for d in gold_doc_ids if d]
    picked_doc_ids = normalized.get("picked_doc_ids", [])
    diag_metrics = compute_doc_pick_diagnostics(
        retrieved_sources=normalized.get("retrieved_sources", []),
        picked_doc_ids=picked_doc_ids,
        gold_doc_ids=gold_doc_ids,
        relevant_ids=relevant_ids,
        cutoffs=cutoffs,
    )

    metrics = normalized.get("metrics", {})
    record = {
        "query_id": qid,
        "query": item.get("query", ""),
        "system": system,
        "eval_granularity": granularity,
        "gold_doc_id": item.get("gold_doc_id", ""),
        "gold_doc_ids": gold_doc_ids,
        "query_type": item.get("query_type", "factual"),
        "query_style": item.get("query_style", ""),
        "reference_mode": item.get("reference_mode", ""),
        "gold_anchor_node_id": item.get("gold_anchor_node_id", ""),
        "gold_anchor_node_ids": list(item.get("gold_anchor_node_ids", [item.get("gold_anchor_node_id", "")])),
        "navigation_path": item.get("navigation_path", ""),
        "relevant_node_ids": sorted(relevant_ids),
        "retrieved_node_ids": normalized.get("retrieved_node_ids", []),
        "retrieved_sources": normalized.get("retrieved_sources", []),
        "picked_doc_ids": picked_doc_ids,
        "worker_ok": normalized.get("worker_ok", False),
        "error": normalized.get("error", ""),
        "error_category": error_category,
        "retry_count": retry_count,
        "worker_stderr": worker_stderr,
        "llm_calls": metrics.get("llm_calls", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "elapsed_s": metrics.get("elapsed_s", 0.0),
        "step_metrics": metrics.get("step_metrics", {}),
        "llm_model": normalized.get("llm_model"),
    }
    record.update(retrieval_metrics)
    record.update(diag_metrics)
    if normalized.get("worker_ok") is False and not record["error"]:
        record["error"] = "Worker failed"
    if record["error"] and not worker_stderr and worker_stdout:
        record["worker_stderr"] = worker_stdout
    return record
