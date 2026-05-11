"""Aggregate per-query retrieval cost from eval_runs JSONL records.

Walks data/eval_runs/<label>/records/<system>__<granularity>.jsonl
(produced by scripts/eval/{vectorless,vector}.py) and rolls up LLM
calls, tokens, and wall-clock time per (run, system, granularity).
Per-step breakdowns are derived from each record's `step_metrics`
field, and per-query-style slices are derived from `query_style`.

The canonical retrieval-cost source is `data/eval_runs/<label>/records/`,
not `data/retrieval_logs/`. The latter holds ad-hoc per-call dumps
from interactive runs and is sparse. This script reads only the
former.

Errored queries (`worker_ok=false`) are counted but excluded from
cost rollups, since their token and elapsed counts are unreliable.

Usage:
    python scripts/aggregation/retrieval_cost.py
    python scripts/aggregation/retrieval_cost.py --run run04_vectorless_25q
    python scripts/aggregation/retrieval_cost.py --json-only
"""
import argparse
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_EVAL_DIR = Path("data/eval_runs")
DEFAULT_OUTPUT = Path("data/retrieval_cost_summary.json")


def _zero_step() -> dict:
    """Per-step accumulator initialiser."""
    return {"calls_total": 0, "input_tokens_total": 0, "output_tokens_total": 0,
            "tokens_total": 0, "elapsed_s_total": 0.0, "occurrences": 0}


def _zero_style() -> dict:
    """Per-query-style accumulator initialiser."""
    return {"count": 0, "tokens_total": 0, "elapsed_s_total": 0.0,
            "llm_calls_total": 0}


def _percentile(values: list[float], pct: float) -> float:
    """Return the requested percentile (0-100) using linear interpolation."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    sorted_v = sorted(values)
    rank = (pct / 100.0) * (len(sorted_v) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(sorted_v) - 1)
    frac = rank - lo
    return sorted_v[lo] + (sorted_v[hi] - sorted_v[lo]) * frac


def _round(x: float, n: int = 2) -> float:
    """Round helper that keeps zero as int 0 to avoid 0.0 noise in JSON."""
    return round(float(x), n)


def parse_combo_filename(fname: str) -> tuple[str, str]:
    """Split `<system>__<granularity>.jsonl` into (system, granularity)."""
    stem = fname[:-len(".jsonl")] if fname.endswith(".jsonl") else fname
    if "__" not in stem:
        return stem, ""
    system, _, granularity = stem.partition("__")
    return system, granularity


def aggregate_combo(records: list[dict]) -> dict:
    """Reduce a per-combo JSONL list into a cost summary dict."""
    successful: list[dict] = []
    errored = 0
    for rec in records:
        if rec.get("worker_ok") is False:
            errored += 1
        else:
            successful.append(rec)

    if not successful:
        return {
            "query_count": len(records),
            "successful_count": 0,
            "errored_count": errored,
            "skipped": True,
        }

    tokens = [int(r.get("total_tokens", 0) or 0) for r in successful]
    inputs = [int(r.get("input_tokens", 0) or 0) for r in successful]
    outputs = [int(r.get("output_tokens", 0) or 0) for r in successful]
    elapsed = [float(r.get("elapsed_s", 0.0) or 0.0) for r in successful]
    calls = [int(r.get("llm_calls", 0) or 0) for r in successful]

    per_step: dict = defaultdict(_zero_step)
    per_style: dict = defaultdict(_zero_style)

    for r in successful:
        steps = r.get("step_metrics") or {}
        if not steps:
            bucket = per_step["total"]
            bucket["calls_total"] += int(r.get("llm_calls", 0) or 0)
            bucket["input_tokens_total"] += int(r.get("input_tokens", 0) or 0)
            bucket["output_tokens_total"] += int(r.get("output_tokens", 0) or 0)
            bucket["tokens_total"] += int(r.get("total_tokens", 0) or 0)
            bucket["elapsed_s_total"] += float(r.get("elapsed_s", 0.0) or 0.0)
            bucket["occurrences"] += 1
        else:
            for step_name, sm in steps.items():
                bucket = per_step[step_name]
                bucket["calls_total"] += int(sm.get("llm_calls", 0) or 0)
                bucket["input_tokens_total"] += int(sm.get("input_tokens", 0) or 0)
                bucket["output_tokens_total"] += int(sm.get("output_tokens", 0) or 0)
                bucket["tokens_total"] += (
                    int(sm.get("input_tokens", 0) or 0)
                    + int(sm.get("output_tokens", 0) or 0)
                )
                bucket["elapsed_s_total"] += float(sm.get("elapsed_s", 0.0) or 0.0)
                bucket["occurrences"] += 1

        style = r.get("query_style") or "unknown"
        s = per_style[style]
        s["count"] += 1
        s["tokens_total"] += int(r.get("total_tokens", 0) or 0)
        s["elapsed_s_total"] += float(r.get("elapsed_s", 0.0) or 0.0)
        s["llm_calls_total"] += int(r.get("llm_calls", 0) or 0)

    n = len(successful)
    summary = {
        "query_count": len(records),
        "successful_count": n,
        "errored_count": errored,
        "llm_calls_total": sum(calls),
        "input_tokens_total": sum(inputs),
        "output_tokens_total": sum(outputs),
        "total_tokens_total": sum(tokens),
        "elapsed_s_total": _round(sum(elapsed)),
        "mean_llm_calls": _round(sum(calls) / n),
        "mean_total_tokens": _round(sum(tokens) / n, 1),
        "mean_elapsed_s": _round(sum(elapsed) / n),
        "p50_total_tokens": int(_percentile(tokens, 50)),
        "p95_total_tokens": int(_percentile(tokens, 95)),
        "p50_elapsed_s": _round(_percentile(elapsed, 50)),
        "p95_elapsed_s": _round(_percentile(elapsed, 95)),
        "per_step": OrderedDict(),
        "per_query_style": OrderedDict(),
    }

    for step_name in sorted(per_step):
        s = per_step[step_name]
        occ = s["occurrences"] or 1
        summary["per_step"][step_name] = {
            "occurrences": s["occurrences"],
            "calls_total": s["calls_total"],
            "tokens_total": s["tokens_total"],
            "elapsed_s_total": _round(s["elapsed_s_total"]),
            "mean_calls": _round(s["calls_total"] / occ),
            "mean_tokens": _round(s["tokens_total"] / occ, 1),
            "mean_elapsed_s": _round(s["elapsed_s_total"] / occ),
        }

    for style in sorted(per_style):
        s = per_style[style]
        c = s["count"] or 1
        summary["per_query_style"][style] = {
            "count": s["count"],
            "llm_calls_total": s["llm_calls_total"],
            "tokens_total": s["tokens_total"],
            "elapsed_s_total": _round(s["elapsed_s_total"]),
            "mean_llm_calls": _round(s["llm_calls_total"] / c),
            "mean_tokens": _round(s["tokens_total"] / c, 1),
            "mean_elapsed_s": _round(s["elapsed_s_total"] / c),
        }

    return summary


def aggregate_run(run_dir: Path) -> dict:
    """Reduce one eval run directory into a per-run summary."""
    cfg_path = run_dir / "config.json"
    cfg = None
    if cfg_path.exists():
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
        except json.JSONDecodeError:
            cfg = None

    records_dir = run_dir / "records"
    if not records_dir.exists():
        return {
            "config": _trim_config(cfg),
            "empty": True,
            "per_combo": {},
            "totals": {},
        }

    per_combo: dict = {}
    systems_present: set = set()
    granularities_present: set = set()
    models_present: set = set()
    all_records: list[dict] = []

    for jl_path in sorted(records_dir.glob("*.jsonl")):
        system, granularity = parse_combo_filename(jl_path.name)
        records: list[dict] = []
        with open(jl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records.append(rec)
                if rec.get("llm_model"):
                    models_present.add(rec["llm_model"])
        if not records:
            continue
        systems_present.add(system)
        granularities_present.add(granularity)
        per_combo[f"{system}__{granularity}"] = aggregate_combo(records)
        all_records.extend(records)

    totals = aggregate_combo(all_records) if all_records else {}
    if totals:
        totals.pop("per_step", None)
        totals.pop("per_query_style", None)

    return {
        "config": _trim_config(cfg),
        "systems_present": sorted(systems_present),
        "granularities_present": sorted(granularities_present),
        "models_present": sorted(models_present),
        "totals": totals,
        "per_combo": per_combo,
    }


def _trim_config(cfg: dict | None) -> dict | None:
    """Keep only the user-visible config fields; drop noisy internals."""
    if not cfg:
        return None
    keep = ("label", "started_at", "ended_at", "split", "split_fingerprint",
            "query_limit", "systems", "granularities", "git_commit",
            "gt_fingerprint")
    return {k: cfg[k] for k in keep if k in cfg}


def combine_runs(per_run: dict) -> dict:
    """Sum per-run totals into a corpus-wide rollup."""
    grand = {
        "query_count": 0, "successful_count": 0, "errored_count": 0,
        "llm_calls_total": 0, "input_tokens_total": 0,
        "output_tokens_total": 0, "total_tokens_total": 0,
        "elapsed_s_total": 0.0, "runs_with_data": 0,
    }
    for run in per_run.values():
        t = run.get("totals") or {}
        if not t or t.get("skipped"):
            continue
        grand["runs_with_data"] += 1
        for key in ("query_count", "successful_count", "errored_count",
                    "llm_calls_total", "input_tokens_total",
                    "output_tokens_total", "total_tokens_total"):
            grand[key] += int(t.get(key, 0) or 0)
        grand["elapsed_s_total"] += float(t.get("elapsed_s_total", 0.0) or 0.0)
    grand["elapsed_s_total"] = _round(grand["elapsed_s_total"])
    return grand


def print_summary(report: dict) -> None:
    """Print one section per run, then the cross-run totals."""
    for label, run in report["runs"].items():
        if run.get("empty"):
            print(f"\n[{label}] (no records)")
            continue
        t = run["totals"]
        print(f"\n[{label}]  q={t.get('query_count', 0)}, "
              f"succ={t.get('successful_count', 0)}, "
              f"err={t.get('errored_count', 0)}, "
              f"models={','.join(run.get('models_present', []))}")
        print(f"{'combo':<38} {'q':>4} {'succ':>5} {'calls':>7} "
              f"{'tokens':>10} {'mean_t':>9} {'mean_s':>8} {'p95_s':>7}")
        print("-" * 95)
        for combo, c in run["per_combo"].items():
            if c.get("skipped"):
                print(f"{combo:<38} {c['query_count']:>4} {0:>5} "
                      f"{'-':>7} {'-':>10} {'-':>9} {'-':>8} {'-':>7}  (all errored)")
                continue
            print(
                f"{combo:<38} {c['query_count']:>4} {c['successful_count']:>5} "
                f"{c['llm_calls_total']:>7} {c['total_tokens_total']:>10} "
                f"{c['mean_total_tokens']:>9.1f} {c['mean_elapsed_s']:>8.2f} "
                f"{c['p95_elapsed_s']:>7.2f}"
            )

    g = report["totals_across_runs"]
    print("\n[totals across all runs]")
    print(f"  runs_with_data={g['runs_with_data']}")
    print(f"  q={g['query_count']}, succ={g['successful_count']}, err={g['errored_count']}")
    print(f"  llm_calls={g['llm_calls_total']}, tokens={g['total_tokens_total']}, "
          f"elapsed_s={g['elapsed_s_total']}")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-runs-dir", type=Path, default=DEFAULT_EVAL_DIR,
                    help=f"Directory containing run<NN> dirs (default {DEFAULT_EVAL_DIR}).")
    ap.add_argument("--run", action="append", default=None,
                    help="Restrict to one or more run labels. Repeatable. Default scans all.")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"Path to write JSON report (default {DEFAULT_OUTPUT}).")
    ap.add_argument("--json-only", action="store_true",
                    help="Skip the printed summary and only write the JSON output.")
    args = ap.parse_args()

    if not args.eval_runs_dir.exists():
        raise SystemExit(f"eval runs dir not found: {args.eval_runs_dir}")

    candidate_dirs = sorted(p for p in args.eval_runs_dir.iterdir() if p.is_dir())
    if args.run:
        wanted = set(args.run)
        candidate_dirs = [p for p in candidate_dirs if p.name in wanted]

    if not candidate_dirs:
        raise SystemExit("no matching run directories found")

    per_run: dict = OrderedDict()
    for run_dir in candidate_dirs:
        per_run[run_dir.name] = aggregate_run(run_dir)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "eval_runs_dir": str(args.eval_runs_dir),
        "runs": per_run,
        "totals_across_runs": combine_runs(per_run),
    }

    if not args.json_only:
        print_summary(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
