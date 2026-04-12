"""
Generate catalog_gt.json for each index granularity directory.

Derives GT-eligible document IDs directly from ground_truth.json (the single
source of truth), then filters each granularity's catalog.json to only those
documents.

Run this script once after ground_truth.json changes (e.g., new docs added).

Usage:
    python scripts/build_gt_catalog.py
    python scripts/build_gt_catalog.py --gt data/ground_truth.json
    python scripts/build_gt_catalog.py --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

GRANULARITY_DIRS = [
    REPO_ROOT / "data" / "index_pasal",
    REPO_ROOT / "data" / "index_ayat",
    REPO_ROOT / "data" / "index_full_split",
]

DEFAULT_GT_PATH = REPO_ROOT / "data" / "ground_truth.json"


def load_gt_doc_ids(gt_path: Path) -> set[str]:
    """Extract unique gold_doc_id values from ground_truth.json."""
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    doc_ids = {entry["gold_doc_id"] for entry in gt.values() if "gold_doc_id" in entry}
    return doc_ids


def build_gt_catalog(
    gt_path: Path = DEFAULT_GT_PATH,
    dry_run: bool = False,
) -> None:
    if not gt_path.exists():
        raise FileNotFoundError(f"ground_truth.json not found: {gt_path}")

    gt_doc_ids = load_gt_doc_ids(gt_path)
    print(f"GT doc IDs from {gt_path.name}: {len(gt_doc_ids)} unique documents")

    for gran_dir in GRANULARITY_DIRS:
        catalog_path = gran_dir / "catalog.json"
        out_path = gran_dir / "catalog_gt.json"

        if not catalog_path.exists():
            print(f"  SKIP {gran_dir.name}: catalog.json not found")
            continue

        with open(catalog_path, encoding="utf-8") as f:
            catalog = json.load(f)

        filtered = [d for d in catalog if d["doc_id"] in gt_doc_ids]
        missing = gt_doc_ids - {d["doc_id"] for d in filtered}

        label = "[dry-run] " if dry_run else ""
        print(f"  {label}{gran_dir.name}: {len(catalog)} -> {len(filtered)} docs", end="")
        if missing:
            print(f"  (WARNING: {len(missing)} GT doc(s) not in catalog: {sorted(missing)})")
        else:
            print()

        if not dry_run:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(filtered, f, ensure_ascii=False, indent=2)
            print(f"    Written: {out_path.relative_to(REPO_ROOT)}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate catalog_gt.json for each granularity from ground_truth.json"
    )
    ap.add_argument(
        "--gt", type=Path, default=DEFAULT_GT_PATH,
        help=f"Path to ground_truth.json (default: {DEFAULT_GT_PATH.relative_to(REPO_ROOT)})",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be written without actually writing files",
    )
    args = ap.parse_args()
    build_gt_catalog(gt_path=args.gt, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
