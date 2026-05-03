"""Vector RAG evaluation harness.

Evaluates the vector-dense retrieval system across granularities (pasal,
ayat, rincian) and embedding models, then writes the same artifact set as
vectorless.py for an apples-to-apples comparison.

Reuses every scoring function from scripts.eval.core, so RQ1 and RQ2 numbers
are computed by identical code paths.

Usage:
    python scripts/eval/vector.py --label main_rq2_140q --qdrant-path ./qdrant_local
    python scripts/eval/vector.py --label main_rq2_140q --resume --qdrant-path ./qdrant_local
    python scripts/eval/vector.py --label debug --systems vector-dense --embedding-models bge-m3 --query-limit 5 --qdrant-path ./qdrant_local

Artifacts per run mirror the vectorless harness, with embedding_model as an
extra dimension in records and summaries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core import io as eval_io  # noqa: E402
from scripts.eval.core.aggregation import (  # noqa: E402
    aggregate_records,
    compute_combo_confidence_intervals,
    compute_combo_summaries,
    compute_reference_mode_breakdown,
    compute_slice_summaries,
)
from scripts.eval.core.logger import ProgressLogger  # noqa: E402
from scripts.eval.core.metrics import DEFAULT_CUTOFFS  # noqa: E402
from scripts.eval.core.preflight import (  # noqa: E402
    check_corpus_consistency,
    check_index_coverage,
    check_qdrant_reachable,
    gemini_model_name,
    gt_fingerprint,
    query_distribution,
)
from scripts.eval.core.metrics import sibling_failure_stats  # noqa: E402
from scripts.eval.core.records import build_per_query_record, normalize_worker_payload  # noqa: E402
from scripts.eval.core.runner import (  # noqa: E402
    categorise_error,
    is_retryable,
    retry_backoff,
)


TESTSET_FILE = REPO_ROOT / "data/validated_testset.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/eval_runs"
WORKER_SCRIPT = REPO_ROOT / "scripts/eval/vector_worker.py"

SYSTEMS = ["vector-dense"]
GRANULARITIES = ["pasal", "ayat", "rincian"]
EMBEDDING_MODELS = [
    "bge-m3",
    "all-indobert-base-v4",
    "multilingual-e5-large-instruct",
]
LLM_INTER_QUERY_DELAY_S = 3.0
PROCESS_TIMEOUT_S = 900
DEFAULT_MAX_RETRIES = 2


def combo_key(system: str, granularity: str, embedding_model: str) -> str:
    """File-safe combo identifier used for record JSONL filenames."""
    return f"{eval_io.sanitize_label(system, 'system')}__{eval_io.sanitize_label(granularity, 'gran')}__{eval_io.sanitize_label(embedding_model, 'model')}"


def invoke_vector_worker(
    repo_root: Path,
    system: str,
    granularity: str,
    embedding_model: str,
    query: str,
    top_k: int,
    timeout_s: int,
    qdrant_path: str | None,
) -> tuple[dict | None, str, str]:
    """Invoke vector_worker.py in a fresh subprocess with isolated env vars."""
    import subprocess

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
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "").strip() if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "").strip() if isinstance(exc.stderr, str) else ""
        return (
            {"ok": False, "system": system, "granularity": granularity,
             "embedding_model": embedding_model,
             "error": f"Worker timed out after {timeout_s}s"},
            stdout, stderr,
        )

    stdout = (proc.stdout or "").strip()
    stderr = (proc.stderr or "").strip()
    if not stdout:
        return None, stdout, stderr or f"Worker exited with code {proc.returncode}"
    try:
        payload = json.loads(stdout.splitlines()[-1])
    except json.JSONDecodeError:
        return None, stdout, stderr or "Worker output was not valid JSON"
    return payload, stdout, stderr


def run_one_query_with_retry(
    *, repo_root: Path, system: str, granularity: str, embedding_model: str,
    qid: str, item: dict, top_k: int, cutoffs: list[int],
    worker_timeout_s: int, max_retries: int, qdrant_path: str | None,
    logger: ProgressLogger,
) -> tuple[dict, float]:
    """Execute one vector retrieval call, retrying transient errors."""
    retry_count = 0
    error_category = ""
    total_t0 = time.time()
    while True:
        payload, stdout_text, stderr_text = invoke_vector_worker(
            repo_root, system, granularity, embedding_model,
            item["query"], top_k, worker_timeout_s, qdrant_path,
        )
        normalized = normalize_worker_payload(payload)
        err = normalized.get("error", "")
        if not err and normalized.get("worker_ok"):
            error_category = ""
            break
        error_category = categorise_error(err or stderr_text or "")
        if retry_count >= max_retries or not is_retryable(error_category):
            break
        retry_count += 1
        wait = retry_backoff(error_category, retry_count)
        logger.info(
            f"  .. {qid} transient error ({error_category}), retry {retry_count}/"
            f"{max_retries} after {wait:.0f}s"
        )
        time.sleep(wait)

    elapsed = time.time() - total_t0
    record = build_per_query_record(
        qid=qid, item=item, system=system, granularity=granularity,
        cutoffs=cutoffs, normalized=normalized,
        worker_stdout=stdout_text, worker_stderr=stderr_text,
        retry_count=retry_count, error_category=error_category,
    )
    record["embedding_model"] = embedding_model
    return record, elapsed


def build_config(args, label: str, selected_queries: list, cutoffs: list[int]) -> dict:
    """Snapshot the run configuration for write + resume diff."""
    return {
        "run_kind": "vector",
        "label": label,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "testset_file": str(TESTSET_FILE.relative_to(REPO_ROOT)),
        "testset_fingerprint": gt_fingerprint(TESTSET_FILE),
        "systems": args.systems_list,
        "granularities": args.granularities_list,
        "embedding_models": args.embedding_models_list,
        "top_k": args.top_k,
        "cutoffs": cutoffs,
        "doc_id": args.doc_id,
        "query_limit": args.query_limit,
        "random_seed": args.random_seed,
        "num_queries": len(selected_queries),
        "worker_script": str(WORKER_SCRIPT.relative_to(REPO_ROOT)),
        "worker_timeout_s": args.worker_timeout_s,
        "inter_query_delay_s": LLM_INTER_QUERY_DELAY_S,
        "max_retries": args.max_retries,
        "resume": args.resume,
        "qdrant_path": args.qdrant_path,
        "qdrant_url": None,
        "llm_model": gemini_model_name(),
        "notes": {
            "single_gold_gt": True,
        },
    }


_RESUME_SIGNATURE_KEYS = (
    "run_kind", "testset_file", "systems", "granularities",
    "embedding_models", "top_k", "cutoffs", "doc_id", "query_limit",
    "random_seed",
)


def validate_resume_config(run_dir: Path, current: dict, logger: ProgressLogger) -> None:
    """Refuse to resume into a directory whose config disagrees with this run."""
    existing_path = run_dir / "config.json"
    if not existing_path.exists():
        return
    try:
        with open(existing_path, encoding="utf-8") as f:
            existing = json.load(f)
    except (json.JSONDecodeError, OSError):
        logger.warn(f"Existing config.json at {existing_path} is unreadable, leaving in place.")
        return

    cur = {k: current.get(k) for k in _RESUME_SIGNATURE_KEYS}
    old = {k: existing.get(k) for k in _RESUME_SIGNATURE_KEYS}
    if cur != old:
        logger.warn("Resume config mismatch, refusing to mix configs:")
        for k in _RESUME_SIGNATURE_KEYS:
            if cur[k] != old[k]:
                logger.warn(f"  - {k}: existing={old[k]!r} new={cur[k]!r}")
        raise SystemExit(2)

    old_fp = (existing.get("testset_fingerprint") or {}).get("sha256_16")
    new_fp = (current.get("testset_fingerprint") or {}).get("sha256_16")
    if old_fp and new_fp and old_fp != new_fp:
        logger.warn(
            f"Testset fingerprint changed since last run ({old_fp} -> {new_fp}). "
            f"Resumed records may reference older GT. Use --overwrite to start fresh."
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct the CLI argument parser."""
    ap = argparse.ArgumentParser(description="Evaluate vector RAG pipelines on validated GT.")
    ap.add_argument("--label", type=str, default=None,
                    help="Run folder name under data/eval_runs/. Required.")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="Comma-separated systems")
    ap.add_argument("--granularities", default=",".join(GRANULARITIES), help="Comma-separated granularities")
    ap.add_argument("--embedding-models", default=",".join(EMBEDDING_MODELS),
                    help="Comma-separated embedding models")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--query-limit", type=int, default=None)
    ap.add_argument("--random-seed", type=int, default=None)
    ap.add_argument("--doc-id", type=str, default=None)
    ap.add_argument("--query-types", type=str, default=None,
                    help="Comma-separated query types (factual, paraphrased, multihop)")
    ap.add_argument("--per-type-limit", type=int, default=None,
                    help="Stratified sample, pick N items per query_type")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    ap.add_argument("--worker-timeout-s", type=int, default=PROCESS_TIMEOUT_S)
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    ap.add_argument("--qdrant-path", type=str, default=None,
                    help="Path to local Qdrant storage directory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--strict", action="store_true",
                    help="Abort on pre-flight failures (missing index, Qdrant unreachable)")
    return ap


def main() -> int:  # noqa: C901
    """CLI entrypoint. Mirrors vectorless.py lifecycle but adds the embedding_model dim."""
    ap = build_arg_parser()
    args = ap.parse_args()

    if not args.label:
        ap.error("--label is required")
    if args.resume and args.overwrite:
        ap.error("--resume and --overwrite are mutually exclusive")

    args.systems_list = eval_io.parse_csv_list(args.systems, SYSTEMS)
    args.granularities_list = eval_io.parse_csv_list(args.granularities, GRANULARITIES)
    args.embedding_models_list = eval_io.parse_csv_list(args.embedding_models, EMBEDDING_MODELS)

    unknown = (
        [s for s in args.systems_list if s not in SYSTEMS]
        + [g for g in args.granularities_list if g not in GRANULARITIES]
        + [m for m in args.embedding_models_list if m not in EMBEDDING_MODELS]
    )
    if unknown:
        raise SystemExit(f"Unknown system, granularity, or model: {unknown}")
    if args.top_k < 1:
        raise SystemExit("--top-k must be >= 1")

    cutoffs = [k for k in DEFAULT_CUTOFFS if k <= args.top_k]
    if args.top_k not in cutoffs:
        cutoffs.append(args.top_k)
    cutoffs = sorted(set(cutoffs))

    testset = eval_io.load_testset(TESTSET_FILE)
    qtypes = [t.strip() for t in args.query_types.split(",")] if args.query_types else None
    selected_queries = eval_io.select_queries(
        testset, args.doc_id, args.query_limit, args.random_seed,
        query_types=qtypes, per_type_limit=args.per_type_limit,
    )
    if not selected_queries:
        raise SystemExit("No queries matched the requested filters.")

    label = eval_io.sanitize_label(args.label)
    run_dir = Path(args.output_dir) / label
    eval_io.prepare_run_dir(run_dir, resume=args.resume, overwrite=args.overwrite)
    records_dir = run_dir / "records"

    logger = ProgressLogger(run_dir / "progress.log")
    started_at = datetime.now()

    config = build_config(args, label, selected_queries, cutoffs)
    config["started_at"] = started_at.isoformat(timespec="seconds")
    logger.header(config)

    # Pre-flight
    logger.preflight_header()
    qids = [q for q, _ in selected_queries]
    dist = query_distribution(testset, qids)
    logger.preflight_testset(dist)

    missing = check_index_coverage(REPO_ROOT, dict(selected_queries), args.granularities_list)
    if missing:
        for gran, docs in missing.items():
            logger.preflight_missing_index(gran, docs)
        if args.strict:
            logger.warn("--strict set, aborting.")
            return 2
    else:
        logger.ok("Index coverage complete for all requested granularities.")

    offenders = check_corpus_consistency(REPO_ROOT, dict(selected_queries), args.granularities_list)
    if offenders:
        logger.warn(f"{len(offenders)} doc(s) violate the leaf-count invariant.")
        if args.strict:
            return 2
    else:
        logger.ok("Corpus leaf-count invariant holds across granularities.")

    ok, msg = check_qdrant_reachable(args.qdrant_path, qdrant_url=None)
    if ok:
        logger.ok(f"Qdrant reachable, {msg}")
    else:
        logger.warn(f"Qdrant check failed, {msg}")
        if args.strict:
            return 2

    if args.resume:
        validate_resume_config(run_dir, config, logger)

    eval_io.write_json(run_dir / "config.json", config)

    # Execute, triple-nested loop with embedding_model as the inner dim.
    error_categories: dict[str, int] = {}
    total_combos = (
        len(args.systems_list)
        * len(args.granularities_list)
        * len(args.embedding_models_list)
    )
    combo_idx = 0
    for system in args.systems_list:
        for granularity in args.granularities_list:
            for embedding_model in args.embedding_models_list:
                combo_idx += 1
                logger.combo_start(combo_idx, total_combos, system,
                                   f"{granularity} | {embedding_model}")
                key = combo_key(system, granularity, embedding_model)
                combo_path = records_dir / f"{key}.jsonl"

                completed = (
                    {r["query_id"] for r in eval_io.read_records_file(combo_path, validate=True) if r.get("query_id")}
                    if args.resume else set()
                )
                invalid = getattr(eval_io.read_records_file, "last_invalid_count", 0)
                if invalid:
                    logger.warn(
                        f"  resume: skipped {invalid} truncated/invalid record(s) in {key}, "
                        f"those queries will be re-run"
                    )
                    eval_io.read_records_file.last_invalid_count = 0  # type: ignore[attr-defined]
                if completed:
                    logger.info(f"  resume: {len(completed)}/{len(selected_queries)} already done")

                mode = "a" if (args.resume and completed) else "w"
                fh = combo_path.open(mode, encoding="utf-8")
                combo_t0 = time.time()
                combo_records: list[dict] = []
                if completed:
                    combo_records.extend(eval_io.read_records_file(combo_path, validate=True))
                try:
                    for qid, item in selected_queries:
                        if qid in completed:
                            continue
                        record, elapsed = run_one_query_with_retry(
                            repo_root=REPO_ROOT, system=system, granularity=granularity,
                            embedding_model=embedding_model, qid=qid, item=item,
                            top_k=args.top_k, cutoffs=cutoffs,
                            worker_timeout_s=args.worker_timeout_s,
                            max_retries=args.max_retries, qdrant_path=args.qdrant_path,
                            logger=logger,
                        )
                        fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        fh.flush()
                        combo_records.append(record)
                        logger.query_line(record, elapsed)
                        if record.get("error"):
                            cat = record.get("error_category", "other") or "other"
                            error_categories[cat] = error_categories.get(cat, 0) + 1
                        time.sleep(LLM_INTER_QUERY_DELAY_S)
                finally:
                    fh.close()

                logger.combo_summary(system, f"{granularity}|{embedding_model}",
                                     combo_records, time.time() - combo_t0)

    # Finalize, aggregate by (system, granularity, embedding_model).
    completed_at = datetime.now()
    wall_s = (completed_at - started_at).total_seconds()
    all_records = eval_io.read_all_records(records_dir)

    # Synthetic system label "<system>:<embedding_model>" so the existing
    # aggregator (which groups by system x granularity) treats each embedding
    # model as its own system. Mirrors how RQ2 reports compare across models.
    for r in all_records:
        if r.get("system") and r.get("embedding_model"):
            r["system_full"] = f"{r['system']}:{r['embedding_model']}"

    synthetic_systems = sorted({r["system_full"] for r in all_records if r.get("system_full")})

    # Re-key the system field so groupings line up.
    for r in all_records:
        if r.get("system_full"):
            r["system"] = r["system_full"]

    combo_summaries = compute_combo_summaries(
        all_records, synthetic_systems, args.granularities_list, cutoffs
    )
    slice_rows = compute_slice_summaries(
        all_records, synthetic_systems, args.granularities_list, cutoffs
    )
    ref_mode_rows = compute_reference_mode_breakdown(
        all_records, synthetic_systems, args.granularities_list, cutoffs
    )
    bootstrap_ci = compute_combo_confidence_intervals(
        all_records, synthetic_systems, args.granularities_list, cutoffs
    )

    eval_io.write_csv(run_dir / "summary_by_system_granularity.csv", combo_summaries)
    eval_io.write_csv(run_dir / "summary_by_slice.csv", slice_rows)
    eval_io.write_csv(run_dir / "summary_by_reference_mode.csv", ref_mode_rows)

    # Diagnostic only failure analysis, see metrics module N4.
    failure_analysis: dict[str, dict] = {}
    for system in synthetic_systems:
        for granularity in args.granularities_list:
            rows = [
                r for r in all_records
                if r["system"] == system and r["eval_granularity"] == granularity
            ]
            if not rows:
                continue
            failure_analysis[f"{system}__{granularity}"] = sibling_failure_stats(rows, cutoffs)

    overall = {
        "generated_at": completed_at.isoformat(timespec="seconds"),
        "started_at": started_at.isoformat(timespec="seconds"),
        "completed_at": completed_at.isoformat(timespec="seconds"),
        "wall_elapsed_s": round(wall_s, 2),
        "config": config,
        "overall": aggregate_records(all_records, cutoffs),
        "by_system_granularity": combo_summaries,
        "by_reference_mode": ref_mode_rows,
        "bootstrap_ci": bootstrap_ci,
        "failure_analysis": failure_analysis,
        "error_categories": error_categories,
    }
    eval_io.write_json(run_dir / "summary_overall.json", overall)

    error_records = [r for r in all_records if r.get("error")]
    if error_records:
        eval_io.write_jsonl(run_dir / "errors.jsonl", error_records)

    artifact_files = [
        "config.json",
        f"records/  ({len(list(records_dir.glob('*.jsonl')))} files)",
        "summary_overall.json",
        "summary_by_system_granularity.csv",
        "summary_by_slice.csv",
        "summary_by_reference_mode.csv",
        "progress.log",
    ]
    if error_records:
        artifact_files.append("errors.jsonl")

    logger.run_footer(
        combo_summaries=combo_summaries,
        total_records=len(all_records),
        error_categories=error_categories,
        started_at=started_at.strftime("%Y-%m-%d %H:%M:%S"),
        completed_at=completed_at.strftime("%Y-%m-%d %H:%M:%S"),
        wall_s=wall_s,
        run_dir=run_dir,
        artifact_files=artifact_files,
    )
    logger.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
