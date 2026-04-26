"""Pure retrieval + answer scoring functions.

Zero I/O, zero side effects. Used by both the vectorless and vector eval
harnesses so RQ1, RQ2, and RQ3 numbers are computed identically.

Methodology notes for thesis writeup.

  N1. Single-gold GT collapses several IR metrics. With exactly one gold node
      per query per granularity, recall@k equals hit@k, and map@k equals
      mrr@k. NDCG@k is monotone-equivalent to mrr@k. We still report all of
      them so reader can cross-reference the name they prefer.

  N2. Retrieval may return fewer than k items (LLM-stepwise can stop early).
      Metrics are computed over the actual list. Recall@k for a 3-item list
      with the gold at rank 2 is hit. We do not pad to length k.

  N3. full_reciprocal_rank ignores the cutoff. Useful when gold appears at
      rank 12 with k=10, where mrr@10=0 hides the signal but the system
      still found the answer. Report alongside mrr@10 for failure analysis.
"""

from __future__ import annotations

import math
import re


# ----------------------------------------------------------------------
# Constants shared across the harness
# ----------------------------------------------------------------------

DEFAULT_CUTOFFS = [1, 3, 5, 10]

GOLD_KEY_BY_GRANULARITY = {
    "pasal": "gold_pasal_node_ids",
    "ayat": "gold_ayat_node_ids",
    "rincian": "gold_rincian_node_ids",
}

SLICE_FIELDS = ["reference_mode", "query_style", "difficulty", "gold_doc_id"]

STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
    "ayat", "huruf", "angka",
}


# ----------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------

def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def strip_citation_labels(text: str) -> str:
    return re.sub(r"\[R\d+\]", " ", text or "")


def normalize_tokens(text: str) -> list[str]:
    lowered = strip_citation_labels(text).lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


# ----------------------------------------------------------------------
# Retrieval scoring
# ----------------------------------------------------------------------

def score_ranked_retrieval(
    ranked_ids: list[str],
    relevant_ids: set[str],
    cutoffs: list[int],
) -> dict:
    """Score one ranked retrieval result against the gold set.

    Outputs cover every cutoff plus a few rank-distribution descriptive stats.
    For single-gold GT (the thesis design) recall@k collapses to hit@k and
    map@k collapses to mrr@k. We emit both names so reviewers from different
    sub-fields can reference the metric they recognise.
    """
    ranked = unique_preserve_order(ranked_ids)
    relevant = set(relevant_ids)
    hit_positions = [idx for idx, node_id in enumerate(ranked, start=1) if node_id in relevant]
    first_rank = hit_positions[0] if hit_positions else None

    out = {
        "num_retrieved": len(ranked),
        "num_relevant": len(relevant),
        "first_relevant_rank": first_rank,
        "exact_top1_hit": bool(ranked) and ranked[0] in relevant,
        # Reciprocal rank without a cutoff. Useful when gold lands beyond k
        # (mrr@k = 0 hides the rank, full_rr keeps the signal).
        "full_reciprocal_rank": (1.0 / first_rank) if first_rank else 0.0,
    }

    for k in cutoffs:
        top_k = ranked[:k]
        retrieved_relevant = [node_id for node_id in top_k if node_id in relevant]
        hit = bool(retrieved_relevant)
        recall = (len(set(retrieved_relevant)) / len(relevant)) if relevant else 0.0

        dcg = 0.0
        for rank, node_id in enumerate(top_k, start=1):
            if node_id in relevant:
                dcg += 1.0 / math.log2(rank + 1)
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        num_hits = 0
        precision_sum = 0.0
        for rank, node_id in enumerate(top_k, start=1):
            if node_id in relevant:
                num_hits += 1
                precision_sum += num_hits / rank
        ap = precision_sum / len(relevant) if relevant else 0.0

        # Per-cutoff MRR. For single-gold GT this equals map@k, but both names
        # appear in the literature so we keep both for completeness.
        mrr_k = (1.0 / first_rank) if first_rank and first_rank <= k else 0.0

        out[f"hit@{k}"] = float(hit)
        out[f"recall@{k}"] = recall
        out[f"ndcg@{k}"] = ndcg
        out[f"map@{k}"] = ap
        out[f"mrr@{k}"] = mrr_k

    return out


# ----------------------------------------------------------------------
# Rank-distribution descriptive stats over a population of records
# ----------------------------------------------------------------------

def rank_distribution_stats(records: list[dict]) -> dict:
    """Mean and median rank across records that hit, plus hit-rate context.

    MRR weighs top ranks heavily so the average can be misleading. Reporting
    mean and median rank on the hit-only subset gives a complementary view of
    "where does the gold land when we find it".
    """
    ranks = [
        int(row["first_relevant_rank"])
        for row in records
        if row.get("first_relevant_rank")
    ]
    if not ranks:
        return {
            "n_hits_anywhere": 0,
            "n_total": len(records),
            "mean_rank_on_hit": 0.0,
            "median_rank_on_hit": 0.0,
            "max_rank_on_hit": 0,
        }
    sorted_ranks = sorted(ranks)
    n = len(sorted_ranks)
    median = (
        sorted_ranks[n // 2]
        if n % 2 == 1
        else 0.5 * (sorted_ranks[n // 2 - 1] + sorted_ranks[n // 2])
    )
    return {
        "n_hits_anywhere": n,
        "n_total": len(records),
        "mean_rank_on_hit": float(sum(sorted_ranks) / n),
        "median_rank_on_hit": float(median),
        "max_rank_on_hit": int(sorted_ranks[-1]),
    }


# ----------------------------------------------------------------------
# Answer + citation scoring
# ----------------------------------------------------------------------

def tokenize_overlap(answer: str, answer_hint: str) -> tuple[float, float]:
    answer_tokens = normalize_tokens(answer)
    hint_tokens = normalize_tokens(answer_hint)
    if not hint_tokens or not answer_tokens:
        return 0.0, 0.0

    answer_set = set(answer_tokens)
    hint_set = set(hint_tokens)
    overlap = len(answer_set & hint_set)
    precision = overlap / len(answer_set) if answer_set else 0.0
    recall = overlap / len(hint_set) if hint_set else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return recall, f1


def score_citations_and_answer(
    answer: str,
    cited_ids: list[str],
    relevant_ids: set[str],
    answer_hint: str,
) -> dict:
    cited = unique_preserve_order(cited_ids)
    cited_set = set(cited)
    relevant = set(relevant_ids)
    answer_nonempty = bool((answer or "").strip())
    citation_nonempty = bool(cited)
    citation_gold = cited_set & relevant

    citation_precision = (len(citation_gold) / len(cited_set)) if cited_set else 0.0
    citation_recall = (len(citation_gold) / len(relevant)) if relevant else 0.0
    citation_hit = bool(citation_gold)
    fully_grounded = bool(cited_set) and cited_set.issubset(relevant)
    supported_answer = answer_nonempty and citation_hit
    unsupported_answer = answer_nonempty and not citation_hit
    uncited_answer = answer_nonempty and not citation_nonempty
    hint_recall, hint_f1 = tokenize_overlap(answer or "", answer_hint or "")

    return {
        "answer_nonempty": float(answer_nonempty),
        "citation_nonempty": float(citation_nonempty),
        "num_citations": len(cited),
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "citation_hit": float(citation_hit),
        "fully_grounded_citations": float(fully_grounded),
        "supported_answer": float(supported_answer),
        "unsupported_answer": float(unsupported_answer),
        "uncited_answer": float(uncited_answer),
        "answer_hint_token_recall": hint_recall,
        "answer_hint_token_f1": hint_f1,
    }


# ----------------------------------------------------------------------
# Self-test (called from CLI via --self-test-metrics)
# ----------------------------------------------------------------------

def run_self_test() -> None:
    def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
        if abs(actual - expected) > tol:
            raise AssertionError(f"expected {expected}, got {actual}")

    cutoffs = [1, 3, 5, 10]

    row = score_ranked_retrieval(["a", "b"], {"a"}, cutoffs)
    assert_close(row["hit@1"], 1.0)
    assert_close(row["recall@1"], 1.0)
    assert_close(row["mrr@1"], 1.0)
    assert_close(row["mrr@10"], 1.0)
    assert_close(row["map@10"], 1.0)
    assert_close(row["ndcg@10"], 1.0)
    assert_close(row["full_reciprocal_rank"], 1.0)

    row = score_ranked_retrieval(["a", "b", "c"], {"a", "c"}, cutoffs)
    assert_close(row["hit@1"], 1.0)
    assert_close(row["recall@1"], 0.5)
    assert_close(row["recall@3"], 1.0)
    assert_close(row["mrr@10"], 1.0)
    assert_close(row["map@10"], (1.0 + (2 / 3)) / 2)

    row = score_ranked_retrieval(["x", "y", "z"], {"a"}, cutoffs)
    assert_close(row["hit@10"], 0.0)
    assert_close(row["recall@10"], 0.0)
    assert_close(row["mrr@10"], 0.0)
    assert_close(row["map@10"], 0.0)
    assert_close(row["ndcg@10"], 0.0)
    assert_close(row["full_reciprocal_rank"], 0.0)

    row = score_ranked_retrieval(["x", "y", "a"], {"a"}, cutoffs)
    assert_close(row["hit@1"], 0.0)
    assert_close(row["hit@3"], 1.0)
    assert_close(row["mrr@1"], 0.0)
    assert_close(row["mrr@3"], 1 / 3)
    assert_close(row["mrr@10"], 1 / 3)
    assert_close(row["full_reciprocal_rank"], 1 / 3)

    # full_reciprocal_rank > 0 even when gold is past cutoff k=10
    long_list = [f"x{i}" for i in range(11)] + ["a"]
    row = score_ranked_retrieval(long_list, {"a"}, cutoffs)
    assert_close(row["mrr@10"], 0.0)
    assert_close(row["full_reciprocal_rank"], 1 / 12)

    # Rank distribution stats over a small population
    fake_records = [
        {"first_relevant_rank": 1},
        {"first_relevant_rank": 3},
        {"first_relevant_rank": 5},
        {"first_relevant_rank": None},
    ]
    rstat = rank_distribution_stats(fake_records)
    assert_close(rstat["mean_rank_on_hit"], 3.0)
    assert_close(rstat["median_rank_on_hit"], 3.0)
    if rstat["n_hits_anywhere"] != 3 or rstat["n_total"] != 4:
        raise AssertionError("rank_distribution_stats hit count mismatch")
