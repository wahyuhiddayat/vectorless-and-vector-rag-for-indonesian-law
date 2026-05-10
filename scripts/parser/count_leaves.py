"""Count leaf nodes in the indexed corpus per granularity.

Walks data/index_pasal, data/index_ayat, and data/index_rincian, counts
text-bearing leaves per document, and reports per-granularity totals
plus per-category breakdown. Mirrors the leaf definition used by
scripts/gt/select_gt_docs.py:count_leaves so numbers stay comparable
across the GT pipeline and corpus reporting.

Usage:
    python scripts/parser/count_leaves.py
    python scripts/parser/count_leaves.py --granularity pasal
    python scripts/parser/count_leaves.py --json-only
"""
import argparse
import json
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GRANULARITIES = ("pasal", "ayat", "rincian")
DEFAULT_INDEX_ROOT = Path("data")
DEFAULT_OUTPUT = Path("data/leaf_counts.json")


def count_leaves(structure: list[dict]) -> int:
    """Count text-bearing leaf nodes in a parsed index structure.

    Identical to scripts/gt/select_gt_docs.py:count_leaves so the totals
    here line up with the leaf counts the GT selection pipeline uses
    for stratification.
    """
    total = 0
    for node in structure:
        if node.get("nodes"):
            total += count_leaves(node["nodes"])
        elif node.get("text"):
            total += 1
    return total


def scan_granularity(granularity_dir: Path) -> "OrderedDict[str, dict]":
    """Walk one granularity directory and tally leaves per category."""
    by_category: OrderedDict[str, dict] = OrderedDict()
    for cat_dir in sorted(p for p in granularity_dir.iterdir() if p.is_dir()):
        per_doc: list[tuple[str, int]] = []
        for doc_path in sorted(cat_dir.glob("*.json")):
            if doc_path.name == "catalog.json":
                continue
            with open(doc_path, encoding="utf-8") as f:
                doc = json.load(f)
            per_doc.append((doc_path.stem, count_leaves(doc.get("structure", []))))
        if not per_doc:
            continue
        leaves = [n for _, n in per_doc]
        by_category[cat_dir.name] = {
            "docs": len(per_doc),
            "total_leaves": sum(leaves),
            "min_leaves": min(leaves),
            "max_leaves": max(leaves),
            "mean_leaves": round(statistics.mean(leaves), 2),
            "median_leaves": int(statistics.median(leaves)),
        }
    return by_category


def aggregate(by_category: "OrderedDict[str, dict]") -> dict:
    """Combine per-category stats into one granularity-wide summary."""
    total_docs = sum(c["docs"] for c in by_category.values())
    total_leaves = sum(c["total_leaves"] for c in by_category.values())
    means = [c["mean_leaves"] for c in by_category.values()]
    return {
        "categories": len(by_category),
        "total_docs": total_docs,
        "total_leaves": total_leaves,
        "leaves_per_doc_overall": round(total_leaves / total_docs, 2) if total_docs else 0,
        "category_mean_of_means": round(statistics.mean(means), 2) if means else 0,
    }


def print_summary(report: dict) -> None:
    """Print the per-granularity aggregate plus per-category leaf totals."""
    for granularity in GRANULARITIES:
        gr = report.get(granularity)
        if not gr:
            continue
        agg = gr["aggregate"]
        print(
            f"\n[{granularity}] docs={agg['total_docs']}, "
            f"leaves={agg['total_leaves']}, "
            f"leaves_per_doc={agg['leaves_per_doc_overall']}, "
            f"categories={agg['categories']}"
        )
        print(
            f"{'category':<22} {'docs':>5} {'leaves':>8} "
            f"{'mean':>7} {'median':>7} {'min':>5} {'max':>5}"
        )
        print("-" * 64)
        for cat, stats in gr["per_category"].items():
            print(
                f"{cat:<22} {stats['docs']:>5} {stats['total_leaves']:>8} "
                f"{stats['mean_leaves']:>7} {stats['median_leaves']:>7} "
                f"{stats['min_leaves']:>5} {stats['max_leaves']:>5}"
            )


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--index-root",
        type=Path,
        default=DEFAULT_INDEX_ROOT,
        help=f"Directory containing index_<granularity> dirs (default {DEFAULT_INDEX_ROOT}).",
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
    for granularity in targets:
        gdir = args.index_root / f"index_{granularity}"
        if not gdir.exists():
            print(f"skip, missing directory {gdir}")
            continue
        per_category = scan_granularity(gdir)
        report[granularity] = {
            "per_category": per_category,
            "aggregate": aggregate(per_category),
        }

    if not report:
        raise SystemExit("no index directories found")

    if not args.json_only:
        print_summary(report)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
