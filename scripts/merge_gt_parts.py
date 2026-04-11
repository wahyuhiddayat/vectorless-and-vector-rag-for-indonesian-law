"""
Merge multipart raw GT outputs into one final raw GT file.

Expected input layout:
    data/ground_truth_parts/<doc_id>/part01.json
    data/ground_truth_parts/<doc_id>/part02.json
    ...

Output:
    data/ground_truth_raw/<KATEGORI>/<doc_id>.json

Examples:
    python scripts/merge_gt_parts.py permenaker-13-2025
    python scripts/merge_gt_parts.py permenaker-13-2025 --pretty
    python scripts/merge_gt_parts.py permenaker-13-2025 --parts-dir data/ground_truth_parts
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
    ap.add_argument("--parts-dir", type=str, default=str(DEFAULT_PARTS_DIR), help="Base directory for multipart raw outputs")
    ap.add_argument("--out", type=str, default=None, help="Output file path (default: data/ground_truth_raw/<KATEGORI>/<doc_id>.json)")
    ap.add_argument("--pretty", action="store_true", help="Pretty-print merged JSON instead of compact JSON")
    args = ap.parse_args()

    parts_root = Path(args.parts_dir)
    doc_parts_dir = parts_root / args.doc_id
    if not doc_parts_dir.exists():
        raise SystemExit(f"Parts directory not found: {doc_parts_dir}")

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
            output_path = DEFAULT_RAW_DIR / category / f"{args.doc_id}.json"
        else:
            output_path = DEFAULT_RAW_DIR / f"{args.doc_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    indent = 2 if args.pretty else None
    text = json.dumps(merged, ensure_ascii=False, indent=indent)
    if indent is not None:
        text += "\n"
    output_path.write_text(text, encoding="utf-8")

    print(f"Merged {len(part_files)} part file(s) into: {output_path}")
    print(f"Total GT items: {len(merged)}")


if __name__ == "__main__":
    main()
