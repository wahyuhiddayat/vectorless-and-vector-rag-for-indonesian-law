"""Evaluation run orchestrator.

Wraps a single run: pre-flight -> per-combo loop -> summary.
Handles resume, auto-retry on transient errors, and incremental record
persistence so any crash preserves completed work.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import io as eval_io
from .aggregation import (
    aggregate_records,
    compute_combo_summaries,
    compute_slice_summaries,
)
from .logger import ProgressLogger
from .metrics import DEFAULT_CUTOFFS
from .preflight import (
    check_gemini_reachable,
    check_index_coverage,
    query_distribution,
)
from .records import build_per_query_record, normalize_worker_payload


# ----------------------------------------------------------------------
# Error categorisation
# ----------------------------------------------------------------------

_NETWORK_PATTERNS = (
    "getaddrinfo", "connection", "timed out", "timeout",
    "dns", "unreachable", "reset by peer", "eof occurred",
)
_RATE_LIMIT_PATTERNS = (
    "429", "quota", "resource_exhausted", "rate limit", "too many requests",
)
_FILE_NOT_FOUND_PATTERNS = (
    "errno 2", "no such file", "filenotfound",
)
_LLM_EMPTY_PATTERNS = (
    "no relevant documents", "no relevant nodes", "no relevant",
)
_5XX_PATTERNS = ("500", "502", "503", "504", "service unavailable", "overloaded")


def categorise_error(message: str) -> str:
    if not message:
        return ""
    low = message.lower()
    if any(p in low for p in _FILE_NOT_FOUND_PATTERNS):
        return "missing-index"
    if any(p in low for p in _LLM_EMPTY_PATTERNS):
        return "llm-empty"
    if any(p in low for p in _RATE_LIMIT_PATTERNS):
        return "rate-limit"
    if any(p in low for p in _NETWORK_PATTERNS) or any(p in low for p in _5XX_PATTERNS):
        return "network"
    if "json" in low and ("decode" in low or "invalid" in low):
        return "worker-crash"
    if "non-json" in low or "worker exited" in low:
        return "worker-crash"
    return "other"


def is_retryable(category: str) -> bool:
    return category in {"network", "rate-limit", "worker-crash"}


def retry_backoff(category: str, attempt: int) -> float:
    """Seconds to sleep before retry attempt (1-indexed)."""
    if category == "rate-limit":
        return [30, 90][min(attempt - 1, 1)]
    # network / worker-crash
    return [15, 45][min(attempt - 1, 1)]


# ----------------------------------------------------------------------
# Worker invocation
# ----------------------------------------------------------------------

def invoke_worker(
    worker_script: Path,
    repo_root: Path,
    system: str,
    granularity: str,
    query: str,
    top_k: int,
    timeout_s: int,
) -> tuple[dict | None, str, str]:
    cmd = [
        sys.executable,
        str(worker_script),
        "--system", system,
        "--granularity", granularity,
        "--query", query,
        "--top-k", str(top_k),
    ]
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


# ----------------------------------------------------------------------
# Runner
# ----------------------------------------------------------------------

class EvalRunner:
    """Owns one evaluation run lifecycle."""

    def __init__(
        self,
        *,
        repo_root: Path,
        worker_script: Path,
        run_dir: Path,
        testset_file: Path,
        systems: list[str],
        granularities: list[str],
        cutoffs: list[int],
        top_k: int,
        selected_queries: list[tuple[str, dict]],
        testset: dict,
        worker_timeout_s: int,
        inter_query_delay_s: float,
        llm_systems: set[str],
        resume: bool,
        strict: bool,
        max_retries: int,
        random_seed: int | None,
        doc_id: str | None,
        query_limit: int | None,
        label: str,
    ):
        self.repo_root = repo_root
        self.worker_script = worker_script
        self.run_dir = run_dir
        self.records_dir = run_dir / "records"
        self.testset_file = testset_file
        self.systems = systems
        self.granularities = granularities
        self.cutoffs = cutoffs
        self.top_k = top_k
        self.selected_queries = selected_queries
        self.testset = testset
        self.worker_timeout_s = worker_timeout_s
        self.inter_query_delay_s = inter_query_delay_s
        self.llm_systems = llm_systems
        self.resume = resume
        self.strict = strict
        self.max_retries = max_retries
        self.random_seed = random_seed
        self.doc_id = doc_id
        self.query_limit = query_limit
        self.label = label

        self.logger = ProgressLogger(run_dir / "progress.log")
        self.started_at = datetime.now()
        self.error_categories: dict[str, int] = {}

    # ------------------------------------------------------------------

    def _build_config(self) -> dict:
        return {
            "label": self.label,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "testset_file": str(self.testset_file.relative_to(self.repo_root)),
            "systems": self.systems,
            "granularities": self.granularities,
            "top_k": self.top_k,
            "cutoffs": self.cutoffs,
            "doc_id": self.doc_id,
            "query_limit": self.query_limit,
            "random_seed": self.random_seed,
            "num_queries": len(self.selected_queries),
            "worker_script": str(self.worker_script.relative_to(self.repo_root)),
            "worker_timeout_s": self.worker_timeout_s,
            "inter_query_delay_s": self.inter_query_delay_s,
            "max_retries": self.max_retries,
            "resume": self.resume,
            "notes": {
                "vectorless_only": True,
                "llm_judge_used": False,
                "answer_eval": "citation-grounding + weak lexical overlap vs answer_hint",
            },
        }

    # ------------------------------------------------------------------

    def preflight(self) -> None:
        config = self._build_config()
        self.logger.header(config)

        self.logger.preflight_header()
        qids = [q for q, _ in self.selected_queries]
        dist = query_distribution(self.testset, qids)
        self.logger.preflight_testset(dist)

        missing_by_gran = check_index_coverage(
            self.repo_root, dict(self.selected_queries), self.granularities
        )
        if missing_by_gran:
            for gran, docs in missing_by_gran.items():
                self.logger.preflight_missing_index(gran, docs)
            related_q = sum(
                1 for _, item in self.selected_queries
                if item.get("gold_doc_id") in {
                    d for docs in missing_by_gran.values() for d in docs
                }
            )
            self.logger.warn(
                f"Affected queries will error with missing-index (non-retryable): "
                f"~{related_q} queries across affected granularities."
            )
            if self.strict:
                self.logger.warn("--strict set, aborting.")
                raise SystemExit(2)
        else:
            self.logger.ok("Index coverage complete for all requested granularities.")

        if any(sys in self.llm_systems for sys in self.systems):
            ok, msg = check_gemini_reachable(timeout_s=15.0)
            self.logger.preflight_gemini(ok, msg)
            if not ok and self.strict:
                self.logger.warn("--strict set and Gemini unreachable, aborting.")
                raise SystemExit(2)

        eval_io.write_json(self.run_dir / "config.json", config)

    # ------------------------------------------------------------------

    def _run_one_query_with_retry(
        self, system: str, granularity: str, qid: str, item: dict,
    ) -> tuple[dict, float]:
        """Execute one query, retrying transient errors. Returns (record, elapsed_s)."""
        retry_count = 0
        error_category = ""
        total_t0 = time.time()

        while True:
            payload, worker_stdout, worker_stderr = invoke_worker(
                self.worker_script,
                self.repo_root,
                system,
                granularity,
                item["query"],
                self.top_k,
                self.worker_timeout_s,
            )
            normalized = normalize_worker_payload(payload)
            err = normalized.get("error", "")
            if not err and normalized.get("worker_ok"):
                error_category = ""
                break
            error_category = categorise_error(err or worker_stderr or "")
            if retry_count >= self.max_retries or not is_retryable(error_category):
                break
            retry_count += 1
            wait = retry_backoff(error_category, retry_count)
            self.logger.info(
                f"  .. {qid} transient error ({error_category}), retry {retry_count}/"
                f"{self.max_retries} after {wait:.0f}s"
            )
            time.sleep(wait)

        elapsed = time.time() - total_t0
        record = build_per_query_record(
            qid=qid,
            item=item,
            system=system,
            granularity=granularity,
            cutoffs=self.cutoffs,
            normalized=normalized,
            worker_stdout=worker_stdout,
            worker_stderr=worker_stderr,
            retry_count=retry_count,
            error_category=error_category,
        )
        return record, elapsed

    # ------------------------------------------------------------------

    def execute(self) -> None:
        total_combos = len(self.systems) * len(self.granularities)
        combo_idx = 0

        for system in self.systems:
            for granularity in self.granularities:
                combo_idx += 1
                self.logger.combo_start(combo_idx, total_combos, system, granularity)

                combo_path = self.records_dir / eval_io.combo_filename(system, granularity)
                completed = (
                    eval_io.completed_qids_for_combo(self.records_dir, system, granularity)
                    if self.resume else set()
                )
                if completed:
                    self.logger.info(
                        f"  resume: {len(completed)}/{len(self.selected_queries)} "
                        f"already done, skipping"
                    )

                mode = "a" if (self.resume and completed) else "w"
                combo_fh = combo_path.open(mode, encoding="utf-8")
                combo_t0 = time.time()
                combo_records: list[dict] = []
                try:
                    # Include already-completed rows in combo-summary aggregation
                    if completed:
                        combo_records.extend(eval_io.read_records_file(combo_path))

                    for qid, item in self.selected_queries:
                        if qid in completed:
                            continue
                        record, elapsed = self._run_one_query_with_retry(
                            system, granularity, qid, item
                        )
                        combo_fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                        combo_fh.flush()
                        combo_records.append(record)
                        self.logger.query_line(record, elapsed)

                        if record.get("error"):
                            cat = record.get("error_category", "other") or "other"
                            self.error_categories[cat] = self.error_categories.get(cat, 0) + 1

                        if system in self.llm_systems and self.inter_query_delay_s > 0:
                            time.sleep(self.inter_query_delay_s)
                finally:
                    combo_fh.close()

                combo_elapsed = time.time() - combo_t0
                self.logger.combo_summary(system, granularity, combo_records, combo_elapsed)

    # ------------------------------------------------------------------

    def finalize(self) -> None:
        completed_at = datetime.now()
        wall_s = (completed_at - self.started_at).total_seconds()

        all_records = eval_io.read_all_records(self.records_dir)

        combo_summaries = compute_combo_summaries(
            all_records, self.systems, self.granularities, self.cutoffs
        )
        slice_rows = compute_slice_summaries(
            all_records, self.systems, self.granularities, self.cutoffs
        )

        eval_io.write_csv(self.run_dir / "summary_by_system_granularity.csv", combo_summaries)
        eval_io.write_csv(self.run_dir / "summary_by_slice.csv", slice_rows)

        overall_summary = {
            "generated_at": completed_at.isoformat(timespec="seconds"),
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "completed_at": completed_at.isoformat(timespec="seconds"),
            "wall_elapsed_s": round(wall_s, 2),
            "config": self._build_config(),
            "overall": aggregate_records(all_records, self.cutoffs),
            "by_system_granularity": combo_summaries,
            "error_categories": self.error_categories,
        }
        eval_io.write_json(self.run_dir / "summary_overall.json", overall_summary)

        error_records = [r for r in all_records if r.get("error")]
        if error_records:
            eval_io.write_jsonl(self.run_dir / "errors.jsonl", error_records)

        artifact_files = [
            "config.json",
            "records/  (" + str(len(list(self.records_dir.glob('*.jsonl')))) + " files)",
            "summary_overall.json",
            "summary_by_system_granularity.csv",
            "summary_by_slice.csv",
            "progress.log",
        ]
        if error_records:
            artifact_files.append("errors.jsonl")

        self.logger.run_footer(
            combo_summaries=combo_summaries,
            total_records=len(all_records),
            error_categories=self.error_categories,
            started_at=self.started_at.strftime("%Y-%m-%d %H:%M:%S"),
            completed_at=completed_at.strftime("%Y-%m-%d %H:%M:%S"),
            wall_s=wall_s,
            run_dir=self.run_dir,
            artifact_files=artifact_files,
        )
        self.logger.close()


# ----------------------------------------------------------------------
# Default constants exported for CLI
# ----------------------------------------------------------------------

__all__ = [
    "EvalRunner",
    "DEFAULT_CUTOFFS",
    "categorise_error",
    "is_retryable",
    "invoke_worker",
]
