"""Merge multipart raw GT outputs into one final raw GT file.

Expected input layout.
    data/ground_truth_parts/<CAT>/<doc_id>__<type>/part01.json
    data/ground_truth_parts/<CAT>/<doc_id>__<type>/part02.json
    ...

Legacy flat layout (data/ground_truth_parts/<doc_id>__<type>/partNN.json) is
also supported for backward compatibility.

Output, data/ground_truth_raw/<CAT>/<doc_id>__<type>.json.

Usage:
    python scripts/gt/merge_parts.py permenaker-13-2025 --type factual
    python scripts/gt/merge_parts.py permenaker-13-2025 --type multihop --pretty
"""

import argparse
import json
import re
from pathlib import Path

DEFAULT_PARTS_DIR = Path("data/ground_truth_parts")
DEFAULT_RAW_DIR = Path("data/ground_truth_raw")
DATA_INDEX = Path("data/index_ayat")
PART_FILENAME_RE = re.compile(r"part(\d+)\.json$", re.IGNORECASE)


def find_category(doc_id: str) -> str | None:
    """Look up the index category (e.g. 'PERMENAKER') for a given doc_id."""
    for path in DATA_INDEX.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path.parent.name
    return None


def part_sort_key(path: Path) -> tuple[int, str]:
    """Sort part files by numeric suffix."""
    match = PART_FILENAME_RE.search(path.name)
    if match:
        return int(match.group(1)), path.name
    return 10**9, path.name


def load_json_array(path: Path) -> list[dict]:
    """Load one JSON file and require a top-level array."""
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path}: top-level JSON value must be an array")
    return data


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge multipart GT raw files into one document-level raw JSON")
    ap.add_argument("doc_id", type=str, help="Document ID, e.g. permenaker-13-2025")
    ap.add_argument("--type", "-t", type=str, default="factual",
                    choices=["factual", "paraphrased", "multihop", "crossdoc", "adversarial"],
                    help="Query type (default factual)")
    ap.add_argument("--parts-dir", type=str, default=str(DEFAULT_PARTS_DIR), help="Base directory for multipart raw outputs")
    ap.add_argument("--out", type=str, default=None, help="Output file path (default data/ground_truth_raw/<CAT>/<doc_id>__<type>.json)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print merged JSON instead of compact JSON")
    args = ap.parse_args()

    parts_root = Path(args.parts_dir)
    category = find_category(args.doc_id)
    basename = f"{args.doc_id}__{args.type}"
    candidates = []
    if category:
        candidates.append(parts_root / category / basename)
    candidates.append(parts_root / basename)
    doc_parts_dir = next((c for c in candidates if c.exists()), None)
    if doc_parts_dir is None:
        raise SystemExit(f"Parts directory not found (checked: {', '.join(str(c) for c in candidates)})")

    part_files = sorted(doc_parts_dir.glob("part*.json"), key=part_sort_key)
    if not part_files:
        raise SystemExit(f"No part*.json files found in: {doc_parts_dir}")

    merged: list[dict] = []
    for part_file in part_files:
        merged.extend(load_json_array(part_file))

    if args.out:
        output_path = Path(args.out)
    else:
        category = find_category(args.doc_id)
        if category:
            output_path = DEFAULT_RAW_DIR / category / f"{basename}.json"
        else:
            output_path = DEFAULT_RAW_DIR / f"{basename}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    text = json.dumps(merged, ensure_ascii=False, indent=indent)
    if indent is not None:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")

    print(f"\nMerged.")
    print(f"  {len(part_files)} part(s) -> {output_path}  ({len(merged)} items)")
    print()
    print("Next.")
    print(f"  1. python scripts/gt/build_validate.py --doc-id {args.doc_id} --type {args.type}")


if __name__ == "__main__":
    main()
