"""Summary computation across per-query records."""

from __future__ import annotations

import random

from .metrics import SLICE_FIELDS, rank_distribution_stats, safe_mean


# Bootstrap configuration. 1000 resamples, percentile interval, seed-locked.
BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 42
BOOTSTRAP_CI = 0.95


def _bootstrap_ci(values: list[float], resamples: int, seed: int, ci: float) -> dict:
    """Percentile bootstrap confidence interval of the mean.

    Hit/miss is Bernoulli. A paired t-test would assume normality which does
    not hold, so we use percentile bootstrap. Cheap (1000 resamples on ~150
    queries is sub-second) and well understood by reviewers.
    """
    n = len(values)
    if n < 2:
        return {"mean": safe_mean(values), "ci_low": 0.0, "ci_high": 0.0, "n": n}
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    alpha = (1.0 - ci) / 2.0
    low_idx = int(alpha * resamples)
    high_idx = int((1.0 - alpha) * resamples) - 1
    high_idx = max(low_idx, min(high_idx, resamples - 1))
    return {
        "mean": safe_mean(values),
        "ci_low": float(means[low_idx]),
        "ci_high": float(means[high_idx]),
        "n": n,
    }


def aggregate_records(records: list[dict], cutoffs: list[int]) -> dict:
    """Aggregate one bucket of records into a flat summary dict."""
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
        "full_reciprocal_rank": safe_mean([row.get("full_reciprocal_rank", 0.0) for row in records]),
    }

    for k in cutoffs:
        summary[f"hit@{k}"] = safe_mean([row.get(f"hit@{k}", 0.0) for row in records])
        summary[f"recall@{k}"] = safe_mean([row.get(f"recall@{k}", 0.0) for row in records])
        summary[f"ndcg@{k}"] = safe_mean([row.get(f"ndcg@{k}", 0.0) for row in records])
        summary[f"map@{k}"] = safe_mean([row.get(f"map@{k}", 0.0) for row in records])
        summary[f"mrr@{k}"] = safe_mean([row.get(f"mrr@{k}", 0.0) for row in records])

    summary["exact_top1_hit_rate"] = safe_mean([float(row.get("exact_top1_hit", False)) for row in records])

    rank_stats = rank_distribution_stats(records)
    summary["mean_rank_on_hit"] = rank_stats["mean_rank_on_hit"]
    summary["median_rank_on_hit"] = rank_stats["median_rank_on_hit"]
    summary["max_rank_on_hit"] = rank_stats["max_rank_on_hit"]
    summary["hits_anywhere"] = rank_stats["n_hits_anywhere"]

    # Backward-compat name kept for older notebooks.
    summary["mean_first_relevant_rank_on_hit"] = rank_stats["mean_rank_on_hit"]
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


def compute_combo_confidence_intervals(
    records: list[dict],
    systems: list[str],
    granularities: list[str],
    cutoffs: list[int],
) -> list[dict]:
    """Bootstrap CIs for the headline metrics per (system, granularity) combo.

    Headline metrics, recall@k for each k in cutoffs plus mrr@max_k. These
    are the metrics RQ1, RQ2, and RQ3 are reported on. Other metrics can be
    bootstrapped on demand from the per-query JSONL files.
    """
    out: list[dict] = []
    headline_metrics = [f"recall@{k}" for k in cutoffs] + [f"mrr@{max(cutoffs)}"]
    for system in systems:
        for granularity in granularities:
            rows = [
                row for row in records
                if row["system"] == system and row["eval_granularity"] == granularity
            ]
            if not rows:
                continue
            row = {
                "system": system,
                "eval_granularity": granularity,
                "n": len(rows),
                "method": "percentile-bootstrap",
                "resamples": BOOTSTRAP_RESAMPLES,
                "ci_level": BOOTSTRAP_CI,
                "seed": BOOTSTRAP_SEED,
            }
            for metric in headline_metrics:
                values = [float(r.get(metric, 0.0)) for r in rows]
                row[metric] = _bootstrap_ci(values, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, BOOTSTRAP_CI)
            out.append(row)
    return out


def compute_reference_mode_breakdown(
    records: list[dict],
    systems: list[str],
    granularities: list[str],
    cutoffs: list[int],
) -> list[dict]:
    """Per-reference_mode breakdown of headline metrics.

    Surfaces whether the winner system shifts across reference_mode buckets
    (none, legal_ref, doc_only, both). Important for RQ3 narrative.
    """
    modes = ["none", "legal_ref", "doc_only", "both"]
    out: list[dict] = []
    for system in systems:
        for granularity in granularities:
            base = [
                row for row in records
                if row["system"] == system and row["eval_granularity"] == granularity
            ]
            for mode in modes:
                rows = [r for r in base if r.get("reference_mode") == mode]
                if not rows:
                    continue
                summary = {
                    "system": system,
                    "eval_granularity": granularity,
                    "reference_mode": mode,
                }
                summary.update(aggregate_records(rows, cutoffs))
                out.append(summary)
    return out
