"""On-demand paired significance test between two eval combos.

Designed for RQ3 head-to-head, given the winners of RQ1 and RQ2, compute
paired-randomization p-value, paired t-test p-value, and Cohen's d on the
shared query set. Uses the existing per-query records.jsonl files, no
re-running of queries needed.

Usage:
    # Compare hybrid x ayat (vectorless run) vs vector-dense:bge-m3 x pasal
    python scripts/eval/significance.py \\
        --run-a data/eval_runs/main_rq1_140q --system-a hybrid --gran-a ayat \\
        --run-b data/eval_runs/main_rq2_140q --system-b vector-dense:bge-m3 --gran-b pasal \\
        --metrics recall@10,mrr@10,recall@5,recall@1

    # Compare two vectorless combos within the same run
    python scripts/eval/significance.py \\
        --run-a data/eval_runs/main_rq1_140q --system-a hybrid --gran-a ayat \\
        --run-b data/eval_runs/main_rq1_140q --system-b bm25 --gran-b ayat

    # Self-test
    python scripts/eval/significance.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.eval.core import io as eval_io  # noqa: E402
from scripts.eval.core.significance import (  # noqa: E402
    DEFAULT_RANDOMIZATION_B,
    DEFAULT_SEED,
    compare_paired,
    run_self_test,
)


DEFAULT_METRICS = ["recall@10", "mrr@10", "recall@5", "recall@1"]


def load_combo_records(run_dir: Path, system: str, granularity: str) -> dict[str, dict]:
    """Return {query_id: record} for one (system, granularity) combo.

    Reads every JSONL file under run_dir/records/ and filters to the requested
    combo. The system field can be a synthetic "vector-dense:bge-m3" form
    used by the vector harness, in which case records carry that exact
    string after finalize re-keying.
    """
    records_dir = run_dir / "records"
    if not records_dir.exists():
        raise SystemExit(f"records directory not found, {records_dir}")

    matched: dict[str, dict] = {}
    for path in records_dir.glob("*.jsonl"):
        for row in eval_io.read_records_file(path, validate=True):
            if row.get("system") == system and row.get("eval_granularity") == granularity:
                qid = row.get("query_id")
                if qid:
                    matched[qid] = row
    return matched


def align_pairs(
    a_records: dict[str, dict],
    b_records: dict[str, dict],
    metrics: list[str],
) -> tuple[list[str], dict[str, list[float]], dict[str, list[float]], dict[str, int]]:
    """Align records on shared query_ids and return per-metric paired arrays.

    Drops queries where either side is missing the metric or has an error.
    Returns (sorted_qids, a_values, b_values, dropped_counts).
    """
    shared = sorted(set(a_records) & set(b_records))
    a_only = set(a_records) - set(b_records)
    b_only = set(b_records) - set(a_records)

    a_values: dict[str, list[float]] = {m: [] for m in metrics}
    b_values: dict[str, list[float]] = {m: [] for m in metrics}
    kept_qids: list[str] = []
    dropped_error = 0
    dropped_missing_metric = 0

    for qid in shared:
        ra = a_records[qid]
        rb = b_records[qid]
        if ra.get("error") or rb.get("error"):
            dropped_error += 1
            continue
        ok = True
        for m in metrics:
            if m not in ra or m not in rb:
                ok = False
                break
        if not ok:
            dropped_missing_metric += 1
            continue
        for m in metrics:
            a_values[m].append(float(ra[m]))
            b_values[m].append(float(rb[m]))
        kept_qids.append(qid)

    return kept_qids, a_values, b_values, {
        "n_shared": len(shared),
        "n_a_only": len(a_only),
        "n_b_only": len(b_only),
        "dropped_error": dropped_error,
        "dropped_missing_metric": dropped_missing_metric,
    }


def main() -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--run-a", type=str, help="Path to first eval run directory")
    ap.add_argument("--system-a", type=str, help="System label of side A")
    ap.add_argument("--gran-a", type=str, help="Granularity of side A")
    ap.add_argument("--run-b", type=str, help="Path to second eval run directory")
    ap.add_argument("--system-b", type=str, help="System label of side B")
    ap.add_argument("--gran-b", type=str, help="Granularity of side B")
    ap.add_argument("--metrics", type=str, default=",".join(DEFAULT_METRICS),
                    help=f"Comma-separated metric fields (default {','.join(DEFAULT_METRICS)})")
    ap.add_argument("--B", type=int, default=DEFAULT_RANDOMIZATION_B,
                    help=f"Randomization permutations (default {DEFAULT_RANDOMIZATION_B})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--output", type=str, default=None,
                    help="Write result JSON to this path (default stdout)")
    ap.add_argument("--self-test", action="store_true",
                    help="Run synthetic checks on the significance module and exit")
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        print("Significance self-test passed.")
        return 0

    required = [args.run_a, args.system_a, args.gran_a, args.run_b, args.system_b, args.gran_b]
    if any(v is None for v in required):
        ap.error("--run-a/--system-a/--gran-a/--run-b/--system-b/--gran-b are all required")

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if not metrics:
        ap.error("--metrics is empty")

    run_a_dir = Path(args.run_a).resolve()
    run_b_dir = Path(args.run_b).resolve()

    a_records = load_combo_records(run_a_dir, args.system_a, args.gran_a)
    b_records = load_combo_records(run_b_dir, args.system_b, args.gran_b)
    if not a_records:
        raise SystemExit(f"no records for ({args.system_a}, {args.gran_a}) in {run_a_dir}")
    if not b_records:
        raise SystemExit(f"no records for ({args.system_b}, {args.gran_b}) in {run_b_dir}")

    qids, a_vals, b_vals, alignment = align_pairs(a_records, b_records, metrics)
    if not qids:
        raise SystemExit("no overlapping queries between the two runs after filtering")

    per_metric: dict[str, dict] = {}
    for m in metrics:
        per_metric[m] = compare_paired(a_vals[m], b_vals[m], B=args.B, seed=args.seed)

    output = {
        "side_a": {
            "run_dir": str(run_a_dir),
            "system": args.system_a,
            "granularity": args.gran_a,
            "n_records": len(a_records),
        },
        "side_b": {
            "run_dir": str(run_b_dir),
            "system": args.system_b,
            "granularity": args.gran_b,
            "n_records": len(b_records),
        },
        "alignment": alignment,
        "n_paired": len(qids),
        "metrics": per_metric,
        "config": {"B": args.B, "seed": args.seed},
    }

    text = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
        print(f"Wrote {args.output}")
    else:
        sys.stdout.write(text)

    # Brief stdout summary for terminal users.
    print()
    print(f"Side A: {args.system_a} x {args.gran_a}, n={len(a_records)}")
    print(f"Side B: {args.system_b} x {args.gran_b}, n={len(b_records)}")
    print(f"Paired queries: {len(qids)}  (shared {alignment['n_shared']}, "
          f"dropped error={alignment['dropped_error']}, "
          f"dropped_missing={alignment['dropped_missing_metric']})")
    for m, res in per_metric.items():
        rand = res["paired_randomization"]
        t = res["paired_t_test"]
        eff = res["cohens_d"]
        print(
            f"  {m:>14s}  diff={res['mean_diff']:+.4f}  "
            f"p_rand={rand['p_value']:.4f}  p_t={t['p_value']:.4f}  "
            f"d={eff['d']:+.3f} ({eff['label']})  "
            f"converge={'yes' if res['tests_converge'] else 'no'}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
