"""Batch orchestrator over the gt_allocation.json plan.

Runs the GT pipeline phases in bulk so an IDE-based Judge (Copilot, Codex,
Claude Code) can process many (doc, query_type) prompts in one workspace pass
instead of one CLI invocation per item. The orchestrator never calls any
external API. It builds prompt files, waits for the Judge agent to write
judge files into tmp/, then applies them through the existing gate.

Phases.
  status   default. Print the state matrix for every (doc, type) in the plan.
  build    Run Layer 1 + Layer 2 + emit Judge prompt for every (doc, type)
           that has a raw GT file. Continues past per-item failures.
  apply    Apply judge_<doc>(__<type>).txt through apply_validation for every
           (doc, type) where the judge file exists.

State derived from filesystem.
  not-annotated  raw GT file missing or placeholder.
  built          tmp/validate_<doc>(__<type>).txt exists, judge file missing.
  judged         tmp/judge_<doc>(__<type>).txt exists, raw not yet overwritten
                 since judge file was written.
  applied        raw GT mtime > judge file mtime (gate ran successfully).
  failed         build or apply raised an error in the current run.

Usage:
    python scripts/gt/run_allocation.py
    python scripts/gt/run_allocation.py --build --category UU
    python scripts/gt/run_allocation.py --apply --category UU
    python scripts/gt/run_allocation.py --build --type multihop
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from vectorless.ids import doc_category
from scripts.gt.build_validate import assemble_prompt
from scripts.gt.apply_validation import (
    apply_cleaned,
    extract_cleaned_array,
    raw_path_for as raw_path_for_apply,
)

ALLOCATION_FILE = Path("data/gt_allocation.json")
RAW_DIR = Path("data/ground_truth_raw")
TMP_DIR = Path("tmp")
QUERY_TYPES = ("factual", "paraphrased", "multihop")


def _basename(doc_id: str, query_type: str) -> str:
    return f"{doc_id}__{query_type}"


def _raw_path(doc_id: str, query_type: str) -> Path:
    return RAW_DIR / doc_category(doc_id) / f"{_basename(doc_id, query_type)}.json"


def _validate_path(doc_id: str, query_type: str) -> Path:
    return TMP_DIR / f"validate_{_basename(doc_id, query_type)}.txt"


def _read_raw_text(raw_path: Path) -> str | None:
    """Return raw file content as text, or None if missing."""
    if not raw_path.exists():
        return None
    return raw_path.read_text(encoding="utf-8")


def _has_real_items(raw_path: Path) -> bool:
    """True when the raw GT file is a non-empty JSON array (not the placeholder)."""
    text = _read_raw_text(raw_path)
    if text is None:
        return False
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return False
    return isinstance(data, list) and len(data) > 0


def _state(doc_id: str, query_type: str) -> str:
    """Derive pipeline state for one (doc_id, query_type) from raw content + tmp.

    States.
      not-annotated  raw missing or empty placeholder
      annotated      raw is bare JSON array, no validate prompt yet
      built          raw is bare JSON array AND tmp/validate_*.txt exists
      judged         raw contains the ---CLEANED--- separator (Judge response
                     pasted, awaiting apply gate)
      applied        raw is bare JSON array AND tmp/validate_*.txt exists
                     AND raw mtime > validate mtime (apply ran after build)
    """
    raw = _raw_path(doc_id, query_type)
    text = _read_raw_text(raw)
    if text is None:
        return "not-annotated"
    if "---CLEANED---" in text:
        return "judged"
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return "not-annotated"
    if not isinstance(data, list) or not data:
        return "not-annotated"
    val = _validate_path(doc_id, query_type)
    if not val.exists():
        return "annotated"
    if raw.stat().st_mtime > val.stat().st_mtime:
        return "applied"
    return "built"


def iter_plan(allocation: dict, category: str | None, query_type: str | None):
    """Yield (category, doc_id, query_type, count) entries from the plan."""
    for cat, payload in allocation.items():
        if category and cat != category:
            continue
        for entry in payload.get("per_doc_allocation", []):
            doc_id = entry["doc_id"]
            for qt in QUERY_TYPES:
                count = entry.get(qt, 0)
                if count <= 0:
                    continue
                if query_type and qt != query_type:
                    continue
                yield cat, doc_id, qt, count


def cmd_status(allocation: dict, category: str | None, query_type: str | None) -> None:
    """Print the per-(doc, type) state matrix and a short summary."""
    counts: dict[str, int] = {}
    rows: list[tuple[str, str, str, int, str]] = []
    for cat, doc_id, qt, count in iter_plan(allocation, category, query_type):
        st = _state(doc_id, qt)
        counts[st] = counts.get(st, 0) + 1
        rows.append((cat, doc_id, qt, count, st))

    if not rows:
        print("No matching items in plan.")
        return

    print(f"{'category':10s}  {'doc_id':25s}  {'type':12s}  {'n':>3s}  state")
    print("-" * 70)
    for cat, doc_id, qt, count, st in rows:
        print(f"{cat:10s}  {doc_id:25s}  {qt:12s}  {count:>3d}  {st}")
    print("-" * 70)
    summary = "  ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Total {len(rows)} item(s),  {summary}")


def cmd_build(allocation: dict, category: str | None, query_type: str | None,
              skip_layer2: bool) -> None:
    """Run build_validate for every annotated (doc, type) in the plan."""
    n_built = n_skipped = n_failed = 0
    for cat, doc_id, qt, _ in iter_plan(allocation, category, query_type):
        raw = _raw_path(doc_id, qt)
        if not _has_real_items(raw):
            print(f"  skip  {doc_id}  type={qt}  (raw not annotated)")
            n_skipped += 1
            continue
        try:
            out_path, items = assemble_prompt(doc_id, 600, query_type=qt, skip_layer2=skip_layer2)
            print(f"  ok    {doc_id}  type={qt}  -> {out_path}  ({len(items)} item)")
            n_built += 1
        except SystemExit:
            print(f"  FAIL  {doc_id}  type={qt}  (Layer 1/2 gate)")
            n_failed += 1
        except Exception as e:
            print(f"  FAIL  {doc_id}  type={qt}  ({type(e).__name__}, {e})")
            n_failed += 1

    print(f"\nBuilt {n_built}, skipped {n_skipped}, failed {n_failed}")
    if n_built:
        print()
        print("Next.")
        print("  1. Ask the IDE Judge to process each tmp/validate_*.txt and paste the")
        print("     full response (with ---CLEANED--- framing) over the matching raw GT file.")
        print("  2. python scripts/gt/run_allocation.py --apply --category <cat>")


def cmd_apply(allocation: dict, category: str | None, query_type: str | None) -> None:
    """Apply each Judge response (from raw file) in the plan through the struct gate.

    Reads the raw GT file directly. The Judge is expected to have pasted its
    full response (with ---CLEANED--- framing) over the raw file. Items where
    the raw is still bare JSON (no framing) are treated as already-applied
    when a tmp/validate prompt exists, otherwise skipped.
    """
    n_applied = n_skipped = n_failed = 0
    for cat, doc_id, qt, _ in iter_plan(allocation, category, query_type):
        st = _state(doc_id, qt)
        if st in ("not-annotated", "annotated", "built", "applied"):
            n_skipped += 1
            print(f"  skip  {doc_id}  type={qt}  (state={st})")
            continue
        # state == "judged": raw contains ---CLEANED---
        raw = _raw_path(doc_id, qt)
        try:
            text = raw.read_text(encoding="utf-8")
            cleaned = extract_cleaned_array(text)
            apply_cleaned(doc_id, cleaned, dry_run=False, query_type=qt)
            target = raw_path_for_apply(doc_id, qt)
            print(f"  ok    {doc_id}  type={qt}  -> {target}")
            n_applied += 1
        except SystemExit:
            print(f"  FAIL  {doc_id}  type={qt}  (struct gate rejected cleaned array)")
            n_failed += 1
        except Exception as e:
            print(f"  FAIL  {doc_id}  type={qt}  ({type(e).__name__}, {e})")
            n_failed += 1

    print(f"\nApplied {n_applied}, skipped {n_skipped}, failed {n_failed}")
    if n_applied:
        print()
        print("Next.")
        print("  1. Run author spot-check per (doc, type),")
        print("     python scripts/gt/log_review.py <doc> --type <type>")
        print("  2. After all items reviewed, python scripts/gt/collect.py && python scripts/gt/finalize.py")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Batch orchestrator over gt_allocation.json")
    ap.add_argument("--build", action="store_true", help="Run Layer 1+2 and emit Judge prompts")
    ap.add_argument("--apply", action="store_true", help="Apply tmp/judge_*.txt through the gate")
    ap.add_argument("--category", type=str, default=None, help="Limit to one category")
    ap.add_argument("--type", "-t", type=str, default=None, choices=QUERY_TYPES,
                    help="Limit to one query type")
    ap.add_argument("--skip-layer2", action="store_true",
                    help="Skip per-type deterministic gate during --build (diagnostic only)")
    args = ap.parse_args()

    if not ALLOCATION_FILE.exists():
        print(f"ERROR: {ALLOCATION_FILE} not found. Run scripts/gt/allocate_quotas.py first.")
        sys.exit(1)

    with open(ALLOCATION_FILE, encoding="utf-8") as f:
        allocation = json.load(f)

    if args.build and args.apply:
        print("ERROR: pick one phase, --build or --apply, not both.")
        sys.exit(2)

    if args.build:
        cmd_build(allocation, args.category, args.type, args.skip_layer2)
    elif args.apply:
        cmd_apply(allocation, args.category, args.type)
    else:
        cmd_status(allocation, args.category, args.type)


if __name__ == "__main__":
    main()
