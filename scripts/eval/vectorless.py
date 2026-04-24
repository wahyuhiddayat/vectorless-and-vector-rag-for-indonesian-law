"""Vectorless evaluation harness.

Evaluates current vectorless retrieval systems against
data/validated_testset.pkl across pasal / ayat / rincian and writes
reproducible run artifacts under data/eval_runs/<label>/.

Usage:
    python scripts/eval/vectorless.py --label main_primary_140q
    python scripts/eval/vectorless.py --label pilot_primary_10q --query-limit 10 --random-seed 42
    python scripts/eval/vectorless.py --label main_primary_140q --resume
    python scripts/eval/vectorless.py --label main_primary_140q --overwrite
    python scripts/eval/vectorless.py --label debug --doc-id pmk-4-2026 --query-limit 3
    python scripts/eval/vectorless.py --self-test-metrics

Artifacts per run:
    data/eval_runs/<label>/
      config.json
      records/<system>__<granularity>.jsonl
      summary_overall.json
      summary_by_system_granularity.csv
      summary_by_slice.csv
      errors.jsonl            # only if any errors occurred
      progress.log            # tee'd terminal output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core import io as eval_io  # noqa: E402
from scripts.eval.core.metrics import DEFAULT_CUTOFFS  # noqa: E402
from scripts.eval.core.runner import EvalRunner  # noqa: E402


TESTSET_FILE = REPO_ROOT / "data/validated_testset.pkl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/eval_runs"
WORKER_SCRIPT = REPO_ROOT / "scripts/eval/vectorless_worker.py"

SYSTEMS = ["bm25", "hybrid", "hybrid-tree", "llm", "llm-full"]
GRANULARITIES = ["pasal", "ayat", "rincian"]
LLM_SYSTEMS = {"hybrid", "hybrid-tree", "llm", "llm-full"}
LLM_INTER_QUERY_DELAY_S = 3.0
PROCESS_TIMEOUT_S = 900
DEFAULT_MAX_RETRIES = 2


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Evaluate vectorless retrieval pipelines on validated GT.")
    ap.add_argument("--label", type=str, default=None,
                    help="Run folder name under data/eval_runs/. Required unless --self-test-metrics.")
    ap.add_argument("--systems", default=",".join(SYSTEMS), help="Comma-separated systems")
    ap.add_argument("--granularities", default=",".join(GRANULARITIES), help="Comma-separated granularities")
    ap.add_argument("--top-k", type=int, default=10, help="Maximum cutoff K for retrieval metrics")
    ap.add_argument("--query-limit", type=int, default=None,
                    help="Only evaluate the first N queries after filtering")
    ap.add_argument("--random-seed", type=int, default=None,
                    help="With --query-limit, random-sample N queries using this seed")
    ap.add_argument("--doc-id", type=str, default=None, help="Restrict evaluation to one gold_doc_id")
    ap.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                    help="Base directory for eval runs")
    ap.add_argument("--worker-timeout-s", type=int, default=PROCESS_TIMEOUT_S,
                    help="Per-query worker timeout in seconds")
    ap.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                    help="Number of auto-retries for transient errors (network, 5xx, worker crash)")
    ap.add_argument("--resume", action="store_true",
                    help="Continue a previous run: skip queries already recorded")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete existing run directory contents before starting")
    ap.add_argument("--strict", action="store_true",
                    help="Abort on pre-flight failures (missing index, Gemini unreachable)")
    ap.add_argument("--self-test-metrics", action="store_true",
                    help="Run synthetic metric checks and exit")
    return ap


def main() -> int:
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.self_test_metrics:
        from scripts.eval.core.metrics import run_self_test
        run_self_test()
        print("Metric self-test passed.")
        return 0

    if not args.label:
        ap.error("--label is required (unless --self-test-metrics)")

    if args.resume and args.overwrite:
        ap.error("--resume and --overwrite are mutually exclusive")

    systems = eval_io.parse_csv_list(args.systems, SYSTEMS)
    granularities = eval_io.parse_csv_list(args.granularities, GRANULARITIES)

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

    testset = eval_io.load_testset(TESTSET_FILE)
    selected_queries = eval_io.select_queries(
        testset, args.doc_id, args.query_limit, args.random_seed
    )
    if not selected_queries:
        raise SystemExit("No queries matched the requested filters.")

    label = eval_io.sanitize_label(args.label)
    run_dir = Path(args.output_dir) / label

    eval_io.prepare_run_dir(run_dir, resume=args.resume, overwrite=args.overwrite)

    runner = EvalRunner(
        repo_root=REPO_ROOT,
        worker_script=WORKER_SCRIPT,
        run_dir=run_dir,
        testset_file=TESTSET_FILE,
        systems=systems,
        granularities=granularities,
        cutoffs=cutoffs,
        top_k=args.top_k,
        selected_queries=selected_queries,
        testset=testset,
        worker_timeout_s=args.worker_timeout_s,
        inter_query_delay_s=LLM_INTER_QUERY_DELAY_S,
        llm_systems=LLM_SYSTEMS,
        resume=args.resume,
        strict=args.strict,
        max_retries=args.max_retries,
        random_seed=args.random_seed,
        doc_id=args.doc_id,
        query_limit=args.query_limit,
        label=label,
    )

    runner.preflight()
    runner.execute()
    runner.finalize()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
