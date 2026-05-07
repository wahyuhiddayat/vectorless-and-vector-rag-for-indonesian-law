"""CLI: repair OCR garbles in pasal-level leaf text.

For each target doc, calls `clean_doc` and prints per-doc stats.
Aggregates totals across the batch.

Usage:
    python scripts/parser/clean_ocr.py --doc-id uu-3-2025
    python scripts/parser/clean_ocr.py --category UU --force
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.ocr import clean_doc  # noqa: E402
from vectorless.indexing.targets import resolve_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Clean OCR garbles in pasal leaves")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[])
    ap.add_argument("--category", help="Process every doc in this jenis_folder")
    ap.add_argument("--force", action="store_true",
                    help="Re-clean leaves already marked ocr_cleaned")
    args = ap.parse_args()

    targets = resolve_targets(list(args.doc_ids), args.category)
    print(f"Cleaning OCR for {len(targets)} doc(s)")

    totals = {"elapsed_s": 0.0, "llm_calls": 0, "total_tokens": 0, "fixes_total": 0, "rejected": 0}
    for did in targets:
        try:
            stats = clean_doc(did, force=args.force)
        except FileNotFoundError as e:
            print(f"  SKIP missing: {e}")
            continue
        for k in totals:
            totals[k] += stats[k]
        print()

    print(f"Total: {totals['elapsed_s']:.0f}s, {totals['llm_calls']} calls, "
          f"{totals['total_tokens']:,} tokens, fixes={totals['fixes_total']}, "
          f"rejected={totals['rejected']}")


if __name__ == "__main__":
    main()
