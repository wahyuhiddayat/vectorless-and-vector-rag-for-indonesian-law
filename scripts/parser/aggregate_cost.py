"""Aggregate per-doc indexing cost logs into per-stage summaries.

Reads data/indexing_logs/cost_pasal.json, cost_ayat.json, and
cost_rincian.json (written incrementally by vectorless.indexing.build),
and tallies time, token, and call totals per stage and per category.

Stages reported.
  - parse        OpenAI gpt-5 LLM parser (pasal log only)
  - ocr_clean    Vertex Gemini flash-lite OCR repair (pasal log only)
  - resplit      Deterministic splitter, no LLM (ayat and rincian logs)
  - summary      Vertex Gemini flash-lite per-node annotator (all three)

Output schema mirrors the input granularities. Each granularity holds
per-stage aggregates plus per-category breakdowns and a per-doc mean.
A top-level "totals" block sums across all three granularities so the
overall corpus indexing cost can be quoted directly.

Usage:
    python scripts/parser/aggregate_cost.py
    python scripts/parser/aggregate_cost.py --granularity pasal
    python scripts/parser/aggregate_cost.py --json-only
"""
import argparse
import json
import statistics
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GRANULARITIES = ("pasal", "ayat", "rincian")
DEFAULT_LOG_DIR = Path("data/indexing_logs")
DEFAULT_OUTPUT = Path("data/indexing_cost_summary.json")

# Field-to-stage mapping. Each tuple is (stage, field-prefix) so the
# aggregator can extract uniform (time_s, tokens, calls) triples from
# the heterogeneous per-doc records.
_PASAL_STAGES = (
    ("parse",     {"time": "llm_time_s",        "tokens": "llm_total_tokens",  "calls": "llm_calls"}),
    ("ocr_clean", {"time": "ocr_clean_time_s",  "tokens": "ocr_clean_tokens",  "calls": "ocr_clean_calls"}),
    ("summary",   {"time": "summary_time_s",    "tokens": "summary_tokens",    "calls": "summary_calls"}),
)
_DERIVED_STAGES = (
    ("resplit", {"time": "resplit_time_s",  "tokens": None,             "calls": None}),
    ("summary", {"time": "summary_time_s",  "tokens": "summary_tokens", "calls": "summary_calls"}),
)


def _zero_stage() -> dict:
    """Return an empty per-stage accumulator."""
    return {"docs": 0, "time_s": 0.0, "tokens": 0, "calls": 0}


def _accumulate(record: dict, stage_map) -> dict:
    """Extract one stage's (time, tokens, calls) triple from a doc record."""
    out = {}
    for stage, fields in stage_map:
        triple = {
            "time_s": float(record.get(fields["time"], 0.0) or 0.0),
            "tokens": int(record.get(fields["tokens"], 0) or 0) if fields["tokens"] else 0,
            "calls":  int(record.get(fields["calls"],  0) or 0) if fields["calls"]  else 0,
        }
        out[stage] = triple
    return out


def _add(target: dict, triple: dict) -> None:
    """Fold a single triple into a per-stage accumulator."""
    target["docs"]   += 1
    target["time_s"] += triple["time_s"]
    target["tokens"] += triple["tokens"]
    target["calls"]  += triple["calls"]


def _round_stage(stage: dict) -> dict:
    """Return a JSON-friendly rounded copy of a per-stage accumulator."""
    docs = stage["docs"]
    return {
        "docs":            docs,
        "time_s":          round(stage["time_s"], 2),
        "tokens":          stage["tokens"],
        "calls":           stage["calls"],
        "mean_time_s":     round(stage["time_s"] / docs, 2) if docs else 0.0,
        "mean_tokens":     round(stage["tokens"] / docs, 1) if docs else 0.0,
        "mean_calls":      round(stage["calls"]  / docs, 2) if docs else 0.0,
    }


def aggregate_log(log_path: Path, granularity: str) -> dict:
    """Reduce one cost log file into per-stage and per-category summaries."""
    with open(log_path, encoding="utf-8") as f:
        records = json.load(f)

    stage_map = _PASAL_STAGES if granularity == "pasal" else _DERIVED_STAGES

    overall: dict = {stage: _zero_stage() for stage, _ in stage_map}
    by_category: dict = defaultdict(lambda: {stage: _zero_stage() for stage, _ in stage_map})

    for doc_id, rec in records.items():
        cat = rec.get("category", "UNKNOWN")
        triples = _accumulate(rec, stage_map)
        for stage, triple in triples.items():
            _add(overall[stage], triple)
            _add(by_category[cat][stage], triple)

    return {
        "doc_count": len(records),
        "per_stage": {stage: _round_stage(acc) for stage, acc in overall.items()},
        "per_category": OrderedDict(
            (cat, {stage: _round_stage(acc) for stage, acc in stages.items()})
            for cat, stages in sorted(by_category.items())
        ),
    }


def combined_totals(report: dict) -> dict:
    """Sum stages across granularities to produce a corpus-wide rollup.

    Deduplicates the `resplit` stage. vectorless.indexing.build measures
    one combined re-split timer that produces both ayat and rincian leaves
    in one pass, then writes that same value to both granularity cost logs.
    Summing them would double-count, so resplit is taken from the first
    granularity that reports it.
    """
    grand: dict = {}
    seen_dedup: set = set()
    dedup_stages = {"resplit"}
    for gran in GRANULARITIES:
        gr = report.get(gran)
        if not gr:
            continue
        for stage, stats in gr["per_stage"].items():
            if stage in dedup_stages and stage in seen_dedup:
                continue
            bucket = grand.setdefault(
                stage,
                {"docs": 0, "time_s": 0.0, "tokens": 0, "calls": 0},
            )
            bucket["docs"]   += stats["docs"]
            bucket["time_s"] += stats["time_s"]
            bucket["tokens"] += stats["tokens"]
            bucket["calls"]  += stats["calls"]
            if stage in dedup_stages:
                seen_dedup.add(stage)
    return {
        stage: {
            "docs":   data["docs"],
            "time_s": round(data["time_s"], 2),
            "tokens": data["tokens"],
            "calls":  data["calls"],
        }
        for stage, data in grand.items()
    }


def print_summary(report: dict) -> None:
    """Print per-granularity per-stage tables, then the combined totals."""
    for gran in GRANULARITIES:
        gr = report.get(gran)
        if not gr:
            continue
        print(f"\n[{gran}] docs={gr['doc_count']}")
        print(f"{'stage':<10} {'docs':>5} {'time_s':>10} {'tokens':>12} {'calls':>7} "
              f"{'mean_t':>8} {'mean_tok':>10}")
        print("-" * 70)
        for stage, stats in gr["per_stage"].items():
            print(
                f"{stage:<10} {stats['docs']:>5} {stats['time_s']:>10.2f} "
                f"{stats['tokens']:>12} {stats['calls']:>7} "
                f"{stats['mean_time_s']:>8.2f} {stats['mean_tokens']:>10.1f}"
            )

    totals = report.get("totals")
    if totals:
        print("\n[corpus totals across pasal + ayat + rincian]")
        print(f"{'stage':<10} {'docs':>5} {'time_s':>10} {'tokens':>12} {'calls':>7}")
        print("-" * 50)
        for stage, stats in totals.items():
            print(
                f"{stage:<10} {stats['docs']:>5} {stats['time_s']:>10.2f} "
                f"{stats['tokens']:>12} {stats['calls']:>7}"
            )


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help=f"Directory containing cost_<granularity>.json files (default {DEFAULT_LOG_DIR}).",
    )
    ap.add_argument(
        "--granularity",
        choices=GRANULARITIES,
        default=None,
        help="Restrict to one granularity. Default scans all three.",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write JSON report (default {DEFAULT_OUTPUT}).",
    )
    ap.add_argument(
        "--json-only",
        action="store_true",
        help="Skip the printed summary and only write the JSON output.",
    )
    args = ap.parse_args()

    targets = [args.granularity] if args.granularity else list(GRANULARITIES)
    report: dict = {}
    for gran in targets:
        log_path = args.log_dir / f"cost_{gran}.json"
        if not log_path.exists():
            print(f"skip, missing log {log_path}")
            continue
        report[gran] = aggregate_log(log_path, gran)

    if not report:
        raise SystemExit("no cost logs found")

    report["totals"] = combined_totals(report)

    if not args.json_only:
        print_summary(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
