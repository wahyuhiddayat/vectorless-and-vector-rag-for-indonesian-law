"""Summary computation across per-query records."""

from __future__ import annotations

import random

from .metrics import SLICE_FIELDS, rank_distribution_stats, safe_mean


def _pct(values: list[float], p: float) -> float:
    """Percentile via linear interpolation between order statistics.

    Reports tail behaviour (p95, p99) that the mean hides. Useful for cost
    metrics where a few outlier queries dominate the worst case, e.g. an
    agentic retriever that occasionally burns its full action budget on a
    single query.
    """
    if not values:
        return 0.0
    s = sorted(values)
    idx = (p / 100.0) * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] * (1 - frac) + s[hi] * frac)


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
        "full_reciprocal_rank": safe_mean([row.get("full_reciprocal_rank", 0.0) for row in records]),
    }

    # Headline metrics for the thesis: hit@k, recall@k, mrr@k.
    # Completeness: precision@k, f1@k, ndcg@k, dcg@k, map@k.
    # Diagnostic: sibling_hit@k (failure analysis, see metrics module N4).
    for k in cutoffs:
        summary[f"hit@{k}"] = safe_mean([row.get(f"hit@{k}", 0.0) for row in records])
        summary[f"recall@{k}"] = safe_mean([row.get(f"recall@{k}", 0.0) for row in records])
        summary[f"precision@{k}"] = safe_mean([row.get(f"precision@{k}", 0.0) for row in records])
        summary[f"f1@{k}"] = safe_mean([row.get(f"f1@{k}", 0.0) for row in records])
        summary[f"mrr@{k}"] = safe_mean([row.get(f"mrr@{k}", 0.0) for row in records])
        summary[f"ndcg@{k}"] = safe_mean([row.get(f"ndcg@{k}", 0.0) for row in records])
        summary[f"dcg@{k}"] = safe_mean([row.get(f"dcg@{k}", 0.0) for row in records])
        summary[f"map@{k}"] = safe_mean([row.get(f"map@{k}", 0.0) for row in records])
        summary[f"sibling_hit@{k}"] = safe_mean(
            [row.get(f"sibling_hit@{k}", 0.0) for row in records]
        )

    # Cost/latency distribution percentiles. avg + total alone hide outliers;
    # p50/p95/p99 give the typical query, headroom, and worst-case tail.
    elapsed_vals = [float(row.get("elapsed_s", 0.0)) for row in records]
    token_vals = [float(row.get("total_tokens", 0)) for row in records]
    input_token_vals = [float(row.get("input_tokens", 0)) for row in records]
    output_token_vals = [float(row.get("output_tokens", 0)) for row in records]
    llm_call_vals = [float(row.get("llm_calls", 0)) for row in records]
    for p in (50, 95, 99):
        summary[f"p{p}_elapsed_s"] = _pct(elapsed_vals, p)
        summary[f"p{p}_total_tokens"] = _pct(token_vals, p)
        summary[f"p{p}_input_tokens"] = _pct(input_token_vals, p)
        summary[f"p{p}_output_tokens"] = _pct(output_token_vals, p)
        summary[f"p{p}_llm_calls"] = _pct(llm_call_vals, p)

    summary["exact_top1_hit_rate"] = safe_mean([float(row.get("exact_top1_hit", False)) for row in records])

    # Tree-paradigm stage-1 vs stage-2 attribution. For flat methods these
    # fields are vacuously zero (no doc-pick stage), for tree methods they
    # let us distinguish doc-pick failure from within-doc nav failure.
    summary["doc_pick_hit_rate"] = safe_mean(
        [float(row.get("doc_pick_hit", 0.0)) for row in records]
    )
    summary["avg_doc_pick_count"] = safe_mean(
        [float(row.get("doc_pick_count", 0)) for row in records]
    )
    for k in cutoffs:
        summary[f"within_doc_hit@{k}"] = safe_mean(
            [row.get(f"within_doc_hit@{k}", 0.0) for row in records]
        )
        summary[f"within_doc_recall@{k}"] = safe_mean(
            [row.get(f"within_doc_recall@{k}", 0.0) for row in records]
        )
        summary[f"within_doc_mrr@{k}"] = safe_mean(
            [row.get(f"within_doc_mrr@{k}", 0.0) for row in records]
        )
        # Oracle-conditional: average over rows where doc-pick actually hit,
        # so this isolates within-doc retrieval quality from stage-1 failure.
        oracle_rows = [
            row.get(f"within_doc_recall@{k}", 0.0)
            for row in records
            if float(row.get("doc_pick_hit", 0.0)) > 0.0
        ]
        summary[f"oracle_within_doc_recall@{k}"] = safe_mean(oracle_rows)

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
