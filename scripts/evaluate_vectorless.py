"""
Vectorless evaluation harness.

Evaluates current vectorless retrieval systems against data/validated_testset.pkl
across pasal / ayat / full_split and writes reproducible run artifacts.

Usage:
    python scripts/evaluate_vectorless.py
    python scripts/evaluate_vectorless.py --doc-id permenaker-1-2026 --query-limit 5
    python scripts/evaluate_vectorless.py --systems bm25-flat,hybrid --granularities ayat
    python scripts/evaluate_vectorless.py --self-test-metrics
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TESTSET_FILE = REPO_ROOT / "data/validated_testset.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/eval_runs"
WORKER_SCRIPT = REPO_ROOT / "scripts/eval_vectorless_worker.py"

SYSTEMS = ["bm25-flat", "hybrid", "llm-stepwise", "llm-full"]
GRANULARITIES = ["pasal", "ayat", "full_split"]
DEFAULT_CUTOFFS = [1, 3, 5, 10]
# Seconds to sleep between queries for systems that make Gemini calls during retrieval
LLM_INTER_QUERY_DELAY_S = 3
LLM_SYSTEMS = {"hybrid", "llm-stepwise", "llm-full"}
STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
    "ayat", "huruf", "angka",
}
GOLD_KEY_BY_GRANULARITY = {
    "pasal": "gold_pasal_node_ids",
    "ayat": "gold_ayat_node_ids",
    "full_split": "gold_full_split_node_ids",
}
SLICE_FIELDS = ["reference_mode", "query_style", "difficulty", "gold_doc_id"]
PROCESS_TIMEOUT_S = 600


def parse_csv_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values or list(default)


def sanitize_label(label: str | None) -> str:
    if not label:
        return "vectorless_eval"
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return cleaned.strip("-") or "vectorless_eval"


def strip_citation_labels(text: str) -> str:
    return re.sub(r"\[R\d+\]", " ", text or "")


def normalize_tokens(text: str) -> list[str]:
    lowered = strip_citation_labels(text).lower()
    tokens = re.findall(r"[a-z0-9]+", lowered)
    return [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            output.append(value)
    return output


def safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def tokenize_overlap(answer: str, answer_hint: str) -> tuple[float, float]:
    answer_tokens = normalize_tokens(answer)
    hint_tokens = normalize_tokens(answer_hint)
    if not hint_tokens:
        return 0.0, 0.0
    if not answer_tokens:
        return 0.0, 0.0

    answer_set = set(answer_tokens)
    hint_set = set(hint_tokens)
    overlap = len(answer_set & hint_set)
    precision = overlap / len(answer_set) if answer_set else 0.0
    recall = overlap / len(hint_set) if hint_set else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return recall, f1


def score_ranked_retrieval(ranked_ids: list[str], relevant_ids: set[str], cutoffs: list[int]) -> dict:
    ranked = unique_preserve_order(ranked_ids)
    relevant = set(relevant_ids)
    hit_positions = [idx for idx, node_id in enumerate(ranked, start=1) if node_id in relevant]
    first_rank = hit_positions[0] if hit_positions else None

    out = {
        "num_retrieved": len(ranked),
        "num_relevant": len(relevant),
        "first_relevant_rank": first_rank,
        "exact_top1_hit": bool(ranked) and ranked[0] in relevant,
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

        out[f"hit@{k}"] = float(hit)
        out[f"recall@{k}"] = recall
        out[f"ndcg@{k}"] = ndcg
        out[f"map@{k}"] = ap

    max_k = max(cutoffs)
    out[f"mrr@{max_k}"] = (1.0 / first_rank) if first_rank and first_rank <= max_k else 0.0
    return out


def score_citations_and_answer(answer: str, cited_ids: list[str], relevant_ids: set[str], answer_hint: str) -> dict:
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


def load_testset(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"validated testset not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def select_queries(testset: dict[str, dict], doc_id: str | None, query_limit: int | None) -> list[tuple[str, dict]]:
    items = sorted(testset.items(), key=lambda kv: kv[0])
    if doc_id:
        items = [(qid, item) for qid, item in items if item.get("gold_doc_id") == doc_id]
    if query_limit is not None:
        items = items[:query_limit]
    return items


def run_worker(system: str, granularity: str, query: str, top_k: int, timeout_s: int) -> tuple[dict | None, str, str]:
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--system", system,
        "--granularity", granularity,
        "--query", query,
        "--top-k", str(top_k),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        payload = {
            "ok": False,
            "system": system,
            "granularity": granularity,
            "error": f"Worker timed out after {timeout_s}s",
        }
        stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        return payload, stdout, stderr

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()

    if not stdout:
        return None, stdout, stderr or f"Worker exited with code {proc.returncode}"

    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return None, stdout, stderr or "Worker output was not valid JSON"

    return payload, stdout, stderr


def normalize_worker_payload(payload: dict | None) -> dict:
    if not payload:
        return {
            "worker_ok": False,
            "error": "Missing worker payload",
            "answer": "",
            "citations": [],
            "retrieved_sources": [],
            "retrieved_node_ids": [],
            "metrics": {},
        }

    if not payload.get("ok"):
        return {
            "worker_ok": False,
            "error": payload.get("error", "Worker error"),
            "traceback": payload.get("traceback", ""),
            "answer": "",
            "citations": [],
            "retrieved_sources": [],
            "retrieved_node_ids": [],
            "metrics": {},
        }

    raw = payload.get("result", {})
    sources = raw.get("sources", []) or []
    citations = raw.get("citations", []) or []

    return {
        "worker_ok": True,
        "error": raw.get("error", ""),
        "answer": (raw.get("answer") or "").strip(),
        "citations": citations,
        "retrieved_sources": sources,
        "retrieved_node_ids": unique_preserve_order([src.get("node_id", "") for src in sources]),
        "citation_node_ids": unique_preserve_order([c.get("node_id", "") for c in citations]),
        "metrics": raw.get("metrics", {}) or {},
        "raw_strategy": raw.get("strategy", payload.get("system", "")),
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
) -> dict:
    gold_key = GOLD_KEY_BY_GRANULARITY[granularity]
    relevant_ids = set(item.get(gold_key, set()))
    retrieval_metrics = score_ranked_retrieval(normalized["retrieved_node_ids"], relevant_ids, cutoffs)
    answer_metrics = score_citations_and_answer(
        normalized.get("answer", ""),
        normalized.get("citation_node_ids", []),
        relevant_ids,
        item.get("answer_hint", ""),
    )

    metrics = normalized.get("metrics", {})
    record = {
        "query_id": qid,
        "query": item.get("query", ""),
        "system": system,
        "eval_granularity": granularity,
        "gold_doc_id": item.get("gold_doc_id", ""),
        "query_style": item.get("query_style", ""),
        "difficulty": item.get("difficulty", ""),
        "reference_mode": item.get("reference_mode", ""),
        "answer_hint": item.get("answer_hint", ""),
        "gold_anchor_node_id": item.get("gold_anchor_node_id", ""),
        "navigation_path": item.get("navigation_path", ""),
        "relevant_node_ids": sorted(relevant_ids),
        "retrieved_node_ids": normalized.get("retrieved_node_ids", []),
        "retrieved_sources": normalized.get("retrieved_sources", []),
        "citation_node_ids": normalized.get("citation_node_ids", []),
        "citations": normalized.get("citations", []),
        "answer": normalized.get("answer", ""),
        "worker_ok": normalized.get("worker_ok", False),
        "error": normalized.get("error", ""),
        "worker_stderr": worker_stderr,
        "llm_calls": metrics.get("llm_calls", 0),
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "elapsed_s": metrics.get("elapsed_s", 0.0),
    }
    record.update(retrieval_metrics)
    record.update(answer_metrics)
    if normalized.get("worker_ok") is False and not record["error"]:
        record["error"] = "Worker failed"
    if record["error"] and not worker_stderr and worker_stdout:
        record["worker_stderr"] = worker_stdout
    return record


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


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def run_metric_self_test() -> None:
    def assert_close(actual: float, expected: float, tol: float = 1e-6) -> None:
        if abs(actual - expected) > tol:
            raise AssertionError(f"expected {expected}, got {actual}")

    cutoffs = [1, 3, 5, 10]

    row = score_ranked_retrieval(["a", "b"], {"a"}, cutoffs)
    assert_close(row["hit@1"], 1.0)
    assert_close(row["recall@1"], 1.0)
    assert_close(row["mrr@10"], 1.0)
    assert_close(row["map@10"], 1.0)
    assert_close(row["ndcg@10"], 1.0)

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

    row = score_ranked_retrieval(["x", "y", "a"], {"a"}, cutoffs)
    assert_close(row["hit@1"], 0.0)
    assert_close(row["hit@3"], 1.0)
    assert_close(row["mrr@10"], 1 / 3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate vectorless retrieval + answer pipelines on validated GT.")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="Comma-separated systems")
    ap.add_argument("--granularities", default=",".join(GRANULARITIES), help="Comma-separated granularities")
    ap.add_argument("--top-k", type=int, default=10, help="Maximum cutoff K for retrieval metrics")
    ap.add_argument("--query-limit", type=int, default=None, help="Only evaluate the first N queries after filtering")
    ap.add_argument("--doc-id", type=str, default=None, help="Restrict evaluation to one gold_doc_id")
    ap.add_argument("--label", type=str, default=None, help="Optional label appended to the run directory name")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Base directory for eval runs")
    ap.add_argument("--worker-timeout-s", type=int, default=PROCESS_TIMEOUT_S, help="Per-query worker timeout in seconds")
    ap.add_argument("--verbose", action="store_true", help="Print per-query progress")
    ap.add_argument("--save-errors", action="store_true", help="Write errors.jsonl when worker failures occur")
    ap.add_argument("--self-test-metrics", action="store_true", help="Run synthetic metric checks and exit")
    args = ap.parse_args()

    if args.self_test_metrics:
        run_metric_self_test()
        print("Metric self-test passed.")
        return 0

    systems = parse_csv_list(args.systems, SYSTEMS)
    granularities = parse_csv_list(args.granularities, GRANULARITIES)

    unknown_systems = [s for s in systems if s not in SYSTEMS]
    unknown_grans = [g for g in granularities if g not in GRANULARITIES]
    if unknown_systems:
        raise SystemExit(f"Unknown systems: {unknown_systems}")
    if unknown_grans:
        raise SystemExit(f"Unknown granularities: {unknown_grans}")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    cutoffs = [k for k in DEFAULT_CUTOFFS if k <= args.top_k]
    if args.top_k not in cutoffs:
        cutoffs.append(args.top_k)
    cutoffs = sorted(set(cutoffs))

    testset = load_testset(TESTSET_FILE)
    selected_queries = select_queries(testset, args.doc_id, args.query_limit)
    if not selected_queries:
        raise SystemExit("No queries matched the requested filters.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = sanitize_label(args.label)
    run_dir = Path(args.output_dir) / f"{timestamp}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "testset_file": str(TESTSET_FILE.relative_to(REPO_ROOT)),
        "systems": systems,
        "granularities": granularities,
        "top_k": args.top_k,
        "cutoffs": cutoffs,
        "doc_id": args.doc_id,
        "query_limit": args.query_limit,
        "label": run_label,
        "num_queries": len(selected_queries),
        "worker_script": str(WORKER_SCRIPT.relative_to(REPO_ROOT)),
        "notes": {
            "vectorless_only": True,
            "llm_judge_used": False,
            "answer_eval": "citation-grounding + weak lexical overlap vs answer_hint",
        },
        "worker_timeout_s": args.worker_timeout_s,
    }
    write_json(run_dir / "config.json", config)

    per_query_records: list[dict] = []
    error_records: list[dict] = []

    total_runs = len(systems) * len(granularities) * len(selected_queries)
    run_index = 0

    for system in systems:
        for granularity in granularities:
            for qid, item in selected_queries:
                run_index += 1
                if args.verbose:
                    print(f"[{run_index}/{total_runs}] {system} | {granularity} | {qid}")

                payload, worker_stdout, worker_stderr = run_worker(
                    system,
                    granularity,
                    item["query"],
                    args.top_k,
                    args.worker_timeout_s,
                )
                normalized = normalize_worker_payload(payload)
                record = build_per_query_record(
                    qid=qid,
                    item=item,
                    system=system,
                    granularity=granularity,
                    cutoffs=cutoffs,
                    normalized=normalized,
                    worker_stdout=worker_stdout,
                    worker_stderr=worker_stderr,
                )
                per_query_records.append(record)

                if system in LLM_SYSTEMS:
                    time.sleep(LLM_INTER_QUERY_DELAY_S)

                if record.get("error"):
                    error_records.append({
                        "query_id": qid,
                        "system": system,
                        "eval_granularity": granularity,
                        "query": item["query"],
                        "error": record["error"],
                        "worker_stderr": worker_stderr,
                    })

    write_jsonl(run_dir / "per_query.jsonl", per_query_records)

    summary_rows: list[dict] = []
    for system in systems:
        for granularity in granularities:
            rows = [
                row for row in per_query_records
                if row["system"] == system and row["eval_granularity"] == granularity
            ]
            summary = {
                "system": system,
                "eval_granularity": granularity,
            }
            summary.update(aggregate_records(rows, cutoffs))
            summary_rows.append(summary)

    write_csv(run_dir / "summary_by_system_granularity.csv", summary_rows)

    slice_rows: list[dict] = []
    for system in systems:
        for granularity in granularities:
            base_rows = [
                row for row in per_query_records
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

    write_csv(run_dir / "summary_by_slice.csv", slice_rows)

    overall_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "overall": aggregate_records(per_query_records, cutoffs),
        "by_system_granularity": summary_rows,
    }
    write_json(run_dir / "summary_overall.json", overall_summary)

    if args.save_errors and error_records:
        write_jsonl(run_dir / "errors.jsonl", error_records)

    print(f"Saved evaluation run to: {run_dir}")
    print(f"Queries evaluated: {len(selected_queries)}")
    print(f"Records written: {len(per_query_records)}")
    print(f"Summary rows: {len(summary_rows)} system/granularity, {len(slice_rows)} slice rows")
    if error_records:
        print(f"Worker errors: {len(error_records)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
