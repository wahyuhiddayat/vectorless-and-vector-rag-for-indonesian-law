"""CLI: annotate an indexed document tree with per-node `summary` fields.

For each target doc, calls `annotate_doc` at the chosen granularity and
prints per-doc stats. Aggregates totals across the batch.

Usage:
    python scripts/parser/add_node_summary.py --doc-id uu-3-2025
    python scripts/parser/add_node_summary.py --category UU --granularity ayat --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing import GRANULARITY_INDEX_MAP  # noqa: E402
from vectorless.indexing.summary import annotate_doc  # noqa: E402
from vectorless.indexing.targets import resolve_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Annotate per-node summary fields")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[],
                    help="Doc to annotate (repeatable)")
    ap.add_argument("--category", help="Annotate every doc in this jenis_folder")
    ap.add_argument("--granularity", choices=list(GRANULARITY_INDEX_MAP), default="pasal")
    ap.add_argument("--force", action="store_true",
                    help="Re-summarise nodes that already have a summary")
    args = ap.parse_args()

    targets = resolve_targets(list(args.doc_ids), args.category)
    print(f"Annotating {len(targets)} doc(s) at granularity={args.granularity}")

    totals = {"elapsed_s": 0.0, "llm_calls": 0, "total_tokens": 0, "failed": 0}
    for did in targets:
        try:
            stats = annotate_doc(did, granularity=args.granularity, force=args.force)
        except FileNotFoundError as e:
            print(f"  SKIP missing: {e}")
            continue
        for k in totals:
            totals[k] += stats[k]
        print()

    suffix = f" failed={totals['failed']}" if totals["failed"] else ""
    print(f"Total: {totals['elapsed_s']:.0f}s, {totals['llm_calls']} calls, "
          f"{totals['total_tokens']:,} tokens{suffix}")


if __name__ == "__main__":
    main()
