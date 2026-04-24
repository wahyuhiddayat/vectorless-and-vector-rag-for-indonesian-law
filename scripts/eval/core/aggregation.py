"""Summary computation across per-query records."""

from __future__ import annotations

from .metrics import SLICE_FIELDS, safe_mean


def aggregate_records(records: list[dict], cutoffs: list[int]) -> dict:
    summary = {
        "num_queries": len(records),
        "error_count": sum(1 for row in records if row.get("error")),
        "answer_count": sum(1 for row in records if row.get("answer_nonempty")),
        "citation_count_total": sum(int(row.get("num_citations", 0)) for row in records),
        "avg_num_retrieved": safe_mean([row.get("num_retrieved", 0) for row in records]),
        "avg_num_relevant": safe_mean([row.get("num_relevant", 0) for row in records]),
        "avg_elapsed_s": safe_mean([float(row.get("elapsed_s", 0.0)) for row in records]),
        "total_elapsed_s": float(sum(float(row.get("elapsed_s", 0.0)) for row in records)),
        "avg_llm_calls": safe_mean([float(row.get("llm_calls", 0)) for row in records]),
        "avg_input_tokens": safe_mean([float(row.get("input_tokens", 0)) for row in records]),
        "avg_output_tokens": safe_mean([float(row.get("output_tokens", 0)) for row in records]),
        "avg_total_tokens": safe_mean([float(row.get("total_tokens", 0)) for row in records]),
        "total_input_tokens": float(sum(float(row.get("input_tokens", 0)) for row in records)),
        "total_output_tokens": float(sum(float(row.get("output_tokens", 0)) for row in records)),
        "total_tokens": float(sum(float(row.get("total_tokens", 0)) for row in records)),
        "mean_first_relevant_rank_on_hit": safe_mean(
            [float(row["first_relevant_rank"]) for row in records if row.get("first_relevant_rank")]
        ),
        "answer_nonempty_rate": safe_mean([row.get("answer_nonempty", 0.0) for row in records]),
        "citation_nonempty_rate": safe_mean([row.get("citation_nonempty", 0.0) for row in records]),
        "avg_citations_per_answer": (
            float(sum(int(row.get("num_citations", 0)) for row in records if row.get("answer_nonempty")))
            / max(1, sum(1 for row in records if row.get("answer_nonempty")))
        ),
        "citation_precision": safe_mean([row.get("citation_precision", 0.0) for row in records]),
        "citation_recall": safe_mean([row.get("citation_recall", 0.0) for row in records]),
        "citation_hit_rate": safe_mean([row.get("citation_hit", 0.0) for row in records]),
        "fully_grounded_citation_rate": safe_mean([row.get("fully_grounded_citations", 0.0) for row in records]),
        "supported_answer_rate": safe_mean([row.get("supported_answer", 0.0) for row in records]),
        "unsupported_answer_rate": safe_mean([row.get("unsupported_answer", 0.0) for row in records]),
        "uncited_answer_rate": safe_mean([row.get("uncited_answer", 0.0) for row in records]),
        "answer_hint_token_recall": safe_mean([row.get("answer_hint_token_recall", 0.0) for row in records]),
        "answer_hint_token_f1": safe_mean([row.get("answer_hint_token_f1", 0.0) for row in records]),
    }

    for k in cutoffs:
        summary[f"hit@{k}"] = safe_mean([row.get(f"hit@{k}", 0.0) for row in records])
        summary[f"recall@{k}"] = safe_mean([row.get(f"recall@{k}", 0.0) for row in records])
        summary[f"ndcg@{k}"] = safe_mean([row.get(f"ndcg@{k}", 0.0) for row in records])
        summary[f"map@{k}"] = safe_mean([row.get(f"map@{k}", 0.0) for row in records])

    max_k = max(cutoffs)
    summary[f"mrr@{max_k}"] = safe_mean([row.get(f"mrr@{max_k}", 0.0) for row in records])
    summary["exact_top1_hit_rate"] = safe_mean([float(row.get("exact_top1_hit", False)) for row in records])
    return summary


def compute_combo_summaries(
    records: list[dict],
    systems: list[str],
    granularities: list[str],
    cutoffs: list[int],
) -> list[dict]:
    """One summary row per (system, granularity) combo."""
    summary_rows: list[dict] = []
    for system in systems:
        for granularity in granularities:
            rows = [
                row for row in records
                if row["system"] == system and row["eval_granularity"] == granularity
            ]
            if not rows:
                continue
            summary = {"system": system, "eval_granularity": granularity}
            summary.update(aggregate_records(rows, cutoffs))
            summary_rows.append(summary)
    return summary_rows


def compute_slice_summaries(
    records: list[dict],
    systems: list[str],
    granularities: list[str],
    cutoffs: list[int],
) -> list[dict]:
    """One summary row per (system, granularity, slice_field, slice_value)."""
    slice_rows: list[dict] = []
    for system in systems:
        for granularity in granularities:
            base_rows = [
                row for row in records
                if row["system"] == system and row["eval_granularity"] == granularity
            ]
            for slice_field in SLICE_FIELDS:
                values = sorted({str(row.get(slice_field, "")) for row in base_rows})
                for value in values:
                    rows = [row for row in base_rows if str(row.get(slice_field, "")) == value]
                    if not rows:
                        continue
                    summary = {
                        "slice_type": slice_field,
                        "slice_value": value,
                        "system": system,
                        "eval_granularity": granularity,
                    }
                    summary.update(aggregate_records(rows, cutoffs))
                    slice_rows.append(summary)
    return slice_rows
