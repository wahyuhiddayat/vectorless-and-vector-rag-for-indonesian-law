"""Apply Judge LLM cleaned output to the raw GT file with a struct-validate gate.

The Judge produces a JSON report followed by a `---CLEANED---` separator and a
JSON array of cleaned items. This script extracts the cleaned array, runs the
same hard structural validation as collect.py, and only overwrites the raw GT
file when the cleaned array passes. A backup of the previous raw file is kept
under data/ground_truth_raw/<CAT>/.bak/.

Usage:
    python scripts/gt/apply_validation.py --doc-id perma-2-2022 --judge-file tmp/judge_perma-2-2022.txt
    python scripts/gt/apply_validation.py --doc-id perma-2-2022 --stdin < paste.txt
    python scripts/gt/apply_validation.py --doc-id perma-2-2022 --judge-file tmp/x.txt --dry-run
"""

import argparse
import datetime as dt
import json
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


def raw_path_for(doc_id: str) -> Path:
    """Return the path to the raw GT file for a doc_id."""
    return RAW_DIR / doc_category(doc_id) / f"{doc_id}.json"


def backup_path_for(doc_id: str) -> Path:
    """Return a timestamped backup path for the previous raw GT file."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RAW_DIR / doc_category(doc_id) / ".bak" / f"{doc_id}.{stamp}.json"


def extract_cleaned_array(text: str) -> list[dict]:
    """Pull the JSON array that follows the `---CLEANED---` separator."""
    idx = text.find(SEPARATOR)
    if idx < 0:
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


def apply_cleaned(doc_id: str, cleaned: list[dict], dry_run: bool) -> None:
    """Validate the cleaned array and overwrite the raw GT file on success."""
    raw_path = raw_path_for(doc_id)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = Path(tempfile.mkstemp(prefix=f"{doc_id}-", suffix=".json")[1])
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

    if raw_path.exists():
        bak = backup_path_for(doc_id)
        bak.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_path, bak)
        print(f"Backup saved to {bak}")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {raw_path}")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Apply Judge LLM output to raw GT with a struct gate.")
    ap.add_argument("--doc-id", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--judge-file", type=str, default=None,
                     help="Path to a text file containing the Judge LLM output")
    src.add_argument("--stdin", action="store_true",
                     help="Read Judge LLM output from stdin")
    ap.add_argument("--dry-run", action="store_true",
                    help="Validate only, do not overwrite the raw GT file")
    args = ap.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    else:
        text = Path(args.judge_file).read_text(encoding="utf-8")

    cleaned = extract_cleaned_array(text)
    apply_cleaned(args.doc_id, cleaned, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
