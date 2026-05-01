"""Apply Judge LLM cleaned output to the raw GT file with a struct-validate gate.

The Judge produces a JSON report followed by a `---CLEANED---` separator and a
JSON array of cleaned items. This script extracts the cleaned array, runs the
same hard structural validation as collect.py, and only overwrites the raw GT
file when the cleaned array passes. A backup of the previous raw file is kept
under data/ground_truth_raw/<CAT>/.bak/.

Three input modes, in order of friction.

  default     Read directly from the raw GT file. Paste the entire Judge
              response (including ---CLEANED--- framing and any prose summary)
              over the raw file, then run with just --doc-id (and --type if
              not factual). The script extracts the cleaned array and rewrites
              the raw file as pure JSON. If the raw file is already pure JSON
              (no framing), it is validated as-is.
  --judge-file  Read from a separate file the Judge wrote to.
  --stdin     Read from a pipe.

Usage:
    python scripts/gt/apply_validation.py --doc-id uu-13-2025
    python scripts/gt/apply_validation.py --doc-id uu-13-2025 --type multihop
    python scripts/gt/apply_validation.py --doc-id perma-2-2022 --judge-file tmp/judge_perma-2-2022.txt
    python scripts/gt/apply_validation.py --doc-id perma-2-2022 --stdin < paste.txt
    python scripts/gt/apply_validation.py --doc-id perma-2-2022 --dry-run
"""

import argparse
import datetime as dt
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vectorless.ids import doc_category
from scripts.gt.collect import validate_raw_file

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/ground_truth_raw")
SEPARATOR = "---CLEANED---"


def _basename(doc_id: str, query_type: str) -> str:
    return f"{doc_id}__{query_type}"


def raw_path_for(doc_id: str, query_type: str = "factual") -> Path:
    """Return the path to the raw GT file for a doc_id and query type."""
    return RAW_DIR / doc_category(doc_id) / f"{_basename(doc_id, query_type)}.json"


def backup_path_for(doc_id: str, query_type: str = "factual") -> Path:
    """Return a timestamped backup path for the previous raw GT file."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RAW_DIR / doc_category(doc_id) / ".bak" / f"{_basename(doc_id, query_type)}.{stamp}.json"


def extract_cleaned_array(text: str) -> list[dict]:
    """Pull the JSON array that follows the `---CLEANED---` separator.

    If the text has no separator but is already a bare JSON array, return it
    as-is. This makes the function idempotent so re-applying an
    already-cleaned raw file is a no-op.
    """
    idx = text.find(SEPARATOR)
    if idx < 0:
        stripped = text.strip()
        if stripped.startswith("["):
            data = json.loads(stripped)
            if not isinstance(data, list):
                raise SystemExit("Input parsed but is not a JSON array")
            return data
        raise SystemExit(f"Judge output is missing the '{SEPARATOR}' separator")

    tail = text[idx + len(SEPARATOR):].strip()
    if tail.startswith("```"):
        lines = tail.splitlines()
        end = next(
            (i for i in range(1, len(lines)) if lines[i].strip().startswith("```")),
            None,
        )
        if end is not None:
            tail = "\n".join(lines[1:end])
        else:
            tail = "\n".join(lines[1:])
    start = tail.find("[")
    if start < 0:
        raise SystemExit("Could not find a JSON array after the separator")

    depth = 0
    end_idx = -1
    in_str = False
    esc = False
    for i, ch in enumerate(tail[start:], start=start):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end_idx = i + 1
                break
    if end_idx < 0:
        raise SystemExit("JSON array after separator is not balanced")

    payload = tail[start:end_idx]
    data = json.loads(payload)
    if not isinstance(data, list):
        raise SystemExit("Cleaned section did not parse as a JSON array")
    return data


def _normalize_schema(items: list[dict]) -> tuple[list[dict], list[str]]:
    """Re-assert fixed schema fields so Judge 'corrections' do not break Layer 1.

    The Judge sometimes rewrites convention fields (e.g. changes
    `gold_anchor_granularity` from "rincian" to "pasal" because the leaf
    happens to be a pasal-level node). This pass restores the canonical
    values and back-fills the singular/jamak mirror fields so the array
    survives the struct gate. Returns (items, log_lines).
    """
    log: list[str] = []
    for i, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue
        if item.get("gold_anchor_granularity") != "rincian":
            log.append(f"item {i}, gold_anchor_granularity {item.get('gold_anchor_granularity')!r} -> 'rincian'")
            item["gold_anchor_granularity"] = "rincian"
        anchor_id = item.get("gold_anchor_node_id") or item.get("gold_node_id")
        if anchor_id and item.get("gold_node_id") != anchor_id:
            item["gold_node_id"] = anchor_id
        if anchor_id and not item.get("gold_anchor_node_id"):
            item["gold_anchor_node_id"] = anchor_id
        anchors = item.get("gold_anchor_node_ids")
        if (not isinstance(anchors, list) or not anchors) and anchor_id:
            item["gold_anchor_node_ids"] = [anchor_id]
        primary_doc = item.get("gold_doc_id")
        doc_ids = item.get("gold_doc_ids")
        if (not isinstance(doc_ids, list) or not doc_ids) and primary_doc:
            item["gold_doc_ids"] = [primary_doc]
        elif isinstance(doc_ids, list) and doc_ids and not primary_doc:
            item["gold_doc_id"] = doc_ids[0]
    return items, log


def apply_cleaned(doc_id: str, cleaned: list[dict], dry_run: bool, query_type: str = "factual") -> None:
    """Validate the cleaned array and overwrite the raw GT file on success."""
    raw_path = raw_path_for(doc_id, query_type)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    cleaned, fixups = _normalize_schema(cleaned)
    if fixups:
        print("Schema normalize,")
        for line in fixups:
            print(f"  {line}")

    fd, tmp_str = tempfile.mkstemp(prefix=f"{_basename(doc_id, query_type)}-", suffix=".json")
    os.close(fd)
    tmp_path = Path(tmp_str)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(cleaned, f, ensure_ascii=False, indent=2)
            f.write("\n")
        valid_items, errors = validate_raw_file(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    if errors:
        print(f"Cleaned array failed structural validation, {len(errors)} error(s),")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    print(f"Validation passed, {len(valid_items)} item(s) ready")

    if dry_run:
        print("[dry-run] not writing to disk")
        return

    print("\nApplied.")
    if raw_path.exists():
        bak = backup_path_for(doc_id, query_type)
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path, bak)
        print(f"  Backup -> {bak}")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"  Wrote  -> {raw_path}")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Apply Judge LLM output to raw GT with a struct gate.")
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--type", "-t", type=str, default="factual",
                    choices=["factual", "paraphrased", "multihop"],
                    help="Query type to apply (default: factual)")
    src = ap.add_mutually_exclusive_group(required=False)
    src.add_argument("--judge-file", type=str, default=None,
                     help="Path to a text file containing the Judge LLM output")
    src.add_argument("--stdin", action="store_true",
                     help="Read Judge LLM output from stdin")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate only, do not overwrite the raw GT file")
    args = ap.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    elif args.judge_file:
        text = Path(args.judge_file).read_text(encoding="utf-8")
    else:
        raw_path = raw_path_for(args.doc_id, args.type)
        if not raw_path.exists():
            raise SystemExit(f"raw GT not found, {raw_path}")
        text = raw_path.read_text(encoding="utf-8")

    cleaned = extract_cleaned_array(text)
    apply_cleaned(args.doc_id, cleaned, dry_run=args.dry_run, query_type=args.type)

    if not args.dry_run:
        print()
        print("Next.")
        print(f"  1. python scripts/gt/log_review.py {args.doc_id} --type {args.type}")
        print(f"  (or proceed to the next allocation item)")


if __name__ == "__main__":
    main()
