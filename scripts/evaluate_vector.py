"""
Vector RAG evaluation harness.

Evaluates vector retrieval systems (vector-dense, vector-hybrid) across
granularities (pasal / ayat / full_split) and embedding models
(gemini-embedding-001 / multilingual-e5-large-instruct / indo-sentence-bert-base).

Reuses all scoring logic from evaluate_vectorless.py for a fair, consistent comparison.

Output files (in data/eval_runs/<timestamp>_<label>/):
    config.json                        Run configuration
    per_query.jsonl                    One record per (system, granularity, embedding_model, query)
    summary_by_system_gran_model.csv   Aggregate metrics per combination
    summary_by_slice.csv               Aggregate metrics per query-level slice
    summary_overall.json               Full summary JSON

Usage:
    python scripts/evaluate_vector.py --qdrant-path ./qdrant_local
    python scripts/evaluate_vector.py --systems vector-dense --granularities pasal --embedding-models gemini-embedding-001 --qdrant-path ./qdrant_local
    python scripts/evaluate_vector.py --doc-id uu-13-2025 --query-limit 5 --qdrant-path ./qdrant_local
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
TESTSET_FILE = REPO_ROOT / "data/validated_testset.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/eval_runs"
WORKER_SCRIPT = SCRIPTS_DIR / "eval_vector_worker.py"

# Add scripts/ to path so we can import from evaluate_vectorless
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from evaluate_vectorless import (  # noqa: E402
    score_ranked_retrieval,
    score_citations_and_answer,
    normalize_worker_payload,
    build_per_query_record,
    aggregate_records,
    load_testset,
    select_queries,
    write_json,
    write_jsonl,
    write_csv,
    parse_csv_list,
    sanitize_label,
    GOLD_KEY_BY_GRANULARITY,
    DEFAULT_CUTOFFS,
    SLICE_FIELDS,
    PROCESS_TIMEOUT_S,
)

SYSTEMS = ["vector-dense", "vector-hybrid"]
GRANULARITIES = ["pasal", "ayat", "full_split"]
EMBEDDING_MODELS = [
    "bge-m3",                         # Hypothesis A: MIRACL SOTA, 8K context
    "all-indobert-base-v4",            # Hypothesis B: Indonesian-specific, 128-token
    "multilingual-e5-large-instruct",  # Hypothesis C: MMTEB best public multilingual
]

# All vector systems call Gemini for answer generation (and optionally embedding)
LLM_INTER_QUERY_DELAY_S = 3


# ============================================================
# WORKER RUNNER
# ============================================================

def run_worker(
    system: str,
    granularity: str,
    embedding_model: str,
    query: str,
    top_k: int,
    timeout_s: int,
    qdrant_path: str | None = None,
) -> tuple[dict | None, str, str]:
    cmd = [
        sys.executable,
        str(WORKER_SCRIPT),
        "--system", system,
        "--granularity", granularity,
        "--embedding-model", embedding_model,
        "--query", query,
        "--top-k", str(top_k),
    ]
    if qdrant_path:
        cmd += ["--qdrant-path", qdrant_path]

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
            "embedding_model": embedding_model,
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


# ============================================================
# MAIN
# ============================================================

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Evaluate vector RAG retrieval + answer pipelines on validated GT."
    )
    ap.add_argument("--systems", default=",".join(SYSTEMS),
                    help="Comma-separated systems (default: all)")
    ap.add_argument("--granularities", default=",".join(GRANULARITIES),
                    help="Comma-separated granularities (default: all)")
    ap.add_argument("--embedding-models", default=",".join(EMBEDDING_MODELS),
                    help="Comma-separated embedding models (default: all)")
    ap.add_argument("--top-k", type=int, default=10,
                    help="Maximum cutoff K for retrieval metrics")
    ap.add_argument("--query-limit", type=int, default=None,
                    help="Only evaluate the first N queries after filtering")
    ap.add_argument("--doc-id", type=str, default=None,
                    help="Restrict evaluation to one gold_doc_id")
    ap.add_argument("--label", type=str, default=None,
                    help="Optional label appended to the run directory name")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                    help="Base directory for eval runs")
    ap.add_argument("--worker-timeout-s", type=int, default=PROCESS_TIMEOUT_S,
                    help="Per-query worker timeout in seconds")
    ap.add_argument("--qdrant-path", type=str, default=None,
                    help="Path to local Qdrant storage directory (passed to worker)")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-query progress")
    ap.add_argument("--save-errors", action="store_true",
                    help="Write errors.jsonl when worker failures occur")
    args = ap.parse_args()

    systems = parse_csv_list(args.systems, SYSTEMS)
    granularities = parse_csv_list(args.granularities, GRANULARITIES)
    embedding_models = parse_csv_list(args.embedding_models, EMBEDDING_MODELS)

    unknown_systems = [s for s in systems if s not in SYSTEMS]
    unknown_grans = [g for g in granularities if g not in GRANULARITIES]
    unknown_models = [m for m in embedding_models if m not in EMBEDDING_MODELS]
    if unknown_systems:
        raise SystemExit(f"Unknown systems: {unknown_systems}")
    if unknown_grans:
        raise SystemExit(f"Unknown granularities: {unknown_grans}")
    if unknown_models:
        raise SystemExit(f"Unknown embedding models: {unknown_models}")
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
    run_label = sanitize_label(args.label or "vector_eval")
    run_dir = Path(args.output_dir) / f"{timestamp}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=False)

    config = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "testset_file": str(TESTSET_FILE.relative_to(REPO_ROOT)),
        "systems": systems,
        "granularities": granularities,
        "embedding_models": embedding_models,
        "top_k": args.top_k,
        "cutoffs": cutoffs,
        "doc_id": args.doc_id,
        "query_limit": args.query_limit,
        "label": run_label,
        "num_queries": len(selected_queries),
        "worker_script": str(WORKER_SCRIPT.relative_to(REPO_ROOT)),
        "qdrant_path": args.qdrant_path,
        "notes": {
            "vector_only": True,
            "llm_judge_used": False,
            "answer_eval": "citation-grounding + weak lexical overlap vs answer_hint",
        },
        "worker_timeout_s": args.worker_timeout_s,
    }
    write_json(run_dir / "config.json", config)

    per_query_records: list[dict] = []
    error_records: list[dict] = []

    total_runs = len(systems) * len(granularities) * len(embedding_models) * len(selected_queries)
    run_index = 0

    for system in systems:
        for granularity in granularities:
            for embedding_model in embedding_models:
                for qid, item in selected_queries:
                    run_index += 1
                    if args.verbose:
                        print(
                            f"[{run_index}/{total_runs}] "
                            f"{system} | {granularity} | {embedding_model} | {qid}"
                        )

                    payload, worker_stdout, worker_stderr = run_worker(
                        system=system,
                        granularity=granularity,
                        embedding_model=embedding_model,
                        query=item["query"],
                        top_k=args.top_k,
                        timeout_s=args.worker_timeout_s,
                        qdrant_path=args.qdrant_path,
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
                    record["embedding_model"] = embedding_model
                    per_query_records.append(record)

                    # Delay to avoid Gemini API rate limiting (all systems use Gemini for answer gen)
                    time.sleep(LLM_INTER_QUERY_DELAY_S)

                    if record.get("error"):
                        error_records.append({
                            "query_id": qid,
                            "system": system,
                            "eval_granularity": granularity,
                            "embedding_model": embedding_model,
                            "query": item["query"],
                            "error": record["error"],
                            "worker_stderr": worker_stderr,
                        })

    write_jsonl(run_dir / "per_query.jsonl", per_query_records)

    # Aggregate by (system, granularity, embedding_model)
    summary_rows: list[dict] = []
    for system in systems:
        for granularity in granularities:
            for embedding_model in embedding_models:
                rows = [
                    row for row in per_query_records
                    if (row["system"] == system
                        and row["eval_granularity"] == granularity
                        and row["embedding_model"] == embedding_model)
                ]
                if not rows:
                    continue
                summary = {
                    "system": system,
                    "eval_granularity": granularity,
                    "embedding_model": embedding_model,
                }
                summary.update(aggregate_records(rows, cutoffs))
                summary_rows.append(summary)

    write_csv(run_dir / "summary_by_system_gran_model.csv", summary_rows)

    # Aggregate by query-level slices
    slice_rows: list[dict] = []
    for system in systems:
        for granularity in granularities:
            for embedding_model in embedding_models:
                base_rows = [
                    row for row in per_query_records
                    if (row["system"] == system
                        and row["eval_granularity"] == granularity
                        and row["embedding_model"] == embedding_model)
                ]
                for slice_field in SLICE_FIELDS:
                    values = sorted({str(row.get(slice_field, "")) for row in base_rows})
                    for value in values:
                        rows = [
                            row for row in base_rows
                            if str(row.get(slice_field, "")) == value
                        ]
                        if not rows:
                            continue
                        summary = {
                            "slice_type": slice_field,
                            "slice_value": value,
                            "system": system,
                            "eval_granularity": granularity,
                            "embedding_model": embedding_model,
                        }
                        summary.update(aggregate_records(rows, cutoffs))
                        slice_rows.append(summary)

    write_csv(run_dir / "summary_by_slice.csv", slice_rows)

    overall_summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "overall": aggregate_records(per_query_records, cutoffs),
        "by_system_gran_model": summary_rows,
    }
    write_json(run_dir / "summary_overall.json", overall_summary)

    if args.save_errors and error_records:
        write_jsonl(run_dir / "errors.jsonl", error_records)

    print(f"Saved evaluation run to: {run_dir}")
    print(f"Queries evaluated: {len(selected_queries)}")
    print(f"Records written: {len(per_query_records)}")
    print(f"Summary rows: {len(summary_rows)} system/gran/model, {len(slice_rows)} slice rows")
    if error_records:
        print(f"Worker errors: {len(error_records)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
