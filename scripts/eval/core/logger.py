"""Progress logger: stdout + persisted progress.log file.

Every line printed to the terminal is also tee'd into a log file so runs
can be reviewed after the fact without redirecting stdout manually.
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TextIO


def _fmt_time(seconds: float) -> str:
    seconds = float(seconds)
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


class ProgressLogger:
    """Tee writer that mirrors everything to stdout and progress.log.

    Writes are line-buffered and flushed after each call so partial progress
    survives a crash or Ctrl+C.
    """

    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = open(log_path, "a", encoding="utf-8")

    # Core writer
    def _emit(self, text: str, *, to_stdout: bool = True) -> None:
        if to_stdout:
            print(text)
            sys.stdout.flush()
        self._fh.write(text + "\n")
        self._fh.flush()

    def info(self, text: str = "") -> None:
        self._emit(text)

    def warn(self, text: str) -> None:
        self._emit(f"  ! {text}")

    def ok(self, text: str) -> None:
        self._emit(f"  + {text}")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # High-level section formatters
    # ------------------------------------------------------------------

    def rule(self, char: str = "=", width: int = 64) -> None:
        self._emit(char * width)

    def header(self, config: dict, estimated_wall_s: float | None = None) -> None:
        self.rule("=")
        self._emit(f"Run: {config['label']}")
        self._emit(f"Started: {config['started_at']}")
        self._emit(f"Systems: {', '.join(config['systems'])} ({len(config['systems'])})")
        self._emit(f"Granularities: {', '.join(config['granularities'])} ({len(config['granularities'])})")
        n_q = config.get("num_queries", 0)
        combos = len(config["systems"]) * len(config["granularities"])
        self._emit(f"Queries: {n_q} (seed={config.get('random_seed')}, source={config.get('testset_file')})")
        self._emit(f"Combinations: {combos}  ->  {combos * n_q} total calls")
        if estimated_wall_s:
            self._emit(f"Estimated wall time: ~{_fmt_time(estimated_wall_s)}")
        self.rule("=")
        self._emit("")

    def combo_start(self, combo_idx: int, total_combos: int, system: str, granularity: str) -> None:
        now = datetime.now().strftime("%H:%M:%S")
        title = f"[{combo_idx}/{total_combos}] {system} x {granularity}"
        dashes = "-" * max(4, 52 - len(title))
        self._emit("")
        self._emit(f"--- {title} {dashes} {now}")
        header = f"  {'qid':<8}  {'result':<6}  {'hit@1':>5}  {'hit@10':>6}  {'R@10':>5}  {'MRR':>5}  {'time':>6}  note"
        self._emit(header)

    def query_line(self, record: dict, elapsed: float) -> None:
        qid = record.get("query_id", "?")
        hit1 = record.get("hit@1", 0.0)
        hit10 = record.get("hit@10", 0.0)
        r10 = record.get("recall@10", 0.0)
        mrr = record.get("mrr@10", 0.0)

        if record.get("error"):
            result_label = "ERR"
            note = record["error"][:60]
        elif hit1:
            result_label = "HIT"
            note = ""
        elif hit10:
            result_label = "TOP"
            note = f"rank={record.get('first_relevant_rank', '?')}"
        else:
            result_label = "MISS"
            note = f"got={record.get('retrieved_node_ids', [])[:2]}"

        retry_count = record.get("retry_count", 0)
        if retry_count and not record.get("error"):
            note = (note + f"  retried={retry_count}").strip()

        self._emit(
            f"  {qid:<8}  {result_label:<6}  {hit1:>5.0f}  {hit10:>6.0f}  "
            f"{r10:>5.2f}  {mrr:>5.2f}  {elapsed:>5.1f}s  {note}"
        )

    def combo_summary(self, system: str, granularity: str, records: list[dict], elapsed_s: float) -> None:
        n = len(records)
        if n == 0:
            self._emit(f"  === {system} x {granularity}: no records")
            return
        errors = sum(1 for r in records if r.get("error"))
        hit1 = sum(r.get("hit@1", 0.0) for r in records) / n
        r10 = sum(r.get("recall@10", 0.0) for r in records) / n
        mrr = sum(r.get("mrr@10", 0.0) for r in records) / n
        avg_tok = sum(r.get("total_tokens", 0) for r in records) / n
        self._emit(
            f"  === combo summary ===  Hit@1={hit1:.3f}  R@10={r10:.3f}  MRR={mrr:.3f}"
            f"  errors={errors}/{n}  avg_tokens={avg_tok:.0f}  wall={_fmt_time(elapsed_s)}"
        )

    def preflight_header(self) -> None:
        self._emit("Pre-flight:")

    def preflight_testset(self, distribution: dict) -> None:
        self.ok(
            f"Testset loaded: {distribution['num_queries']} queries  |  "
            f"{distribution['num_docs']} unique docs  |  "
            f"{len(distribution['category'])} categories"
        )
        ref = distribution.get("reference_mode", {})
        cat = distribution.get("category", {})
        if ref:
            self._emit("    reference_mode: " + "  ".join(f"{k}={v}" for k, v in sorted(ref.items())))
        if cat:
            self._emit("    category:       " + "  ".join(f"{k}={v}" for k, v in sorted(cat.items())))

    def preflight_missing_index(self, gran: str, missing: list[str]) -> None:
        self.warn(f"{len(missing)} docs missing from data/index_{gran}:")
        # Show up to 10, summarise the rest
        preview = missing[:10]
        self._emit("      " + ", ".join(preview) + ("  ..." if len(missing) > 10 else ""))

    def preflight_gemini(self, ok: bool, message: str) -> None:
        if ok:
            self.ok(f"Gemini API reachable ({message})")
        else:
            self.warn(f"Gemini API check failed: {message}")

    def run_footer(
        self,
        combo_summaries: list[dict],
        total_records: int,
        error_categories: dict,
        started_at: str,
        completed_at: str,
        wall_s: float,
        run_dir: Path,
        artifact_files: list[str],
    ) -> None:
        self._emit("")
        self.rule("=")
        self._emit(f"Run completed: {completed_at}  (wall={_fmt_time(wall_s)})")
        self._emit(f"Started:       {started_at}")
        self._emit("")
        self._emit("System/Granularity summary:")
        for row in combo_summaries:
            self._emit(
                f"  {row['system']:<12} x {row['eval_granularity']:<8}  "
                f"Hit@1={row.get('hit@1', 0):.3f}  R@10={row.get('recall@10', 0):.3f}  "
                f"MRR={row.get('mrr@10', 0):.3f}  "
                f"n={row.get('num_queries', 0)}  errors={row.get('error_count', 0)}"
            )
        total_errors = sum(error_categories.values())
        if total_errors:
            self._emit("")
            self._emit(f"Errors breakdown ({total_errors} of {total_records}):")
            for cat, count in sorted(error_categories.items(), key=lambda kv: -kv[1]):
                self._emit(f"  {cat:<18} : {count}")
        self._emit("")
        self._emit(f"Artifacts: {run_dir}")
        for name in artifact_files:
            self._emit(f"  - {name}")
        self.rule("=")
