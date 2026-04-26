"""Disk I/O and resume-support helpers."""

from __future__ import annotations

import csv
import json
import pickle
import re
import shutil
from pathlib import Path


# ----------------------------------------------------------------------
# Label & path helpers
# ----------------------------------------------------------------------

def sanitize_label(label: str | None, fallback: str = "eval") -> str:
    if not label:
        return fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", label.strip())
    return cleaned.strip("-") or fallback


def parse_csv_list(raw: str | None, default: list[str]) -> list[str]:
    if not raw:
        return list(default)
    values = [v.strip() for v in raw.split(",") if v.strip()]
    return values or list(default)


def combo_filename(system: str, granularity: str) -> str:
    return f"{sanitize_label(system, 'system')}__{sanitize_label(granularity, 'gran')}.jsonl"


# ----------------------------------------------------------------------
# Testset loading
# ----------------------------------------------------------------------

def load_testset(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise FileNotFoundError(f"validated testset not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def select_queries(
    testset: dict[str, dict],
    doc_id: str | None,
    query_limit: int | None,
    random_seed: int | None = None,
) -> list[tuple[str, dict]]:
    items = sorted(testset.items(), key=lambda kv: kv[0])
    if doc_id:
        items = [(qid, item) for qid, item in items if item.get("gold_doc_id") == doc_id]
    if random_seed is not None and query_limit is not None:
        import random
        rng = random.Random(random_seed)
        items = rng.sample(items, min(query_limit, len(items)))
        items.sort(key=lambda kv: kv[0])
    elif query_limit is not None:
        items = items[:query_limit]
    return items


# ----------------------------------------------------------------------
# Writers
# ----------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in records:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


# ----------------------------------------------------------------------
# Records directory readers (resume support)
# ----------------------------------------------------------------------

REQUIRED_RECORD_FIELDS = {
    "query_id", "system", "eval_granularity", "gold_doc_id",
    "recall@10", "mrr@10",
}


def read_records_file(path: Path, *, validate: bool = False) -> list[dict]:
    """Load all records from one JSONL file. Tolerates empty / missing file.

    With validate=True, drops records that are missing any required field.
    Returns the dropped count alongside the records via the side-channel
    last_invalid_records[]. Used on resume to skip rows that may have been
    written before a crash mid-flush.
    """
    if not path.exists():
        return []
    records: list[dict] = []
    invalid_count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                invalid_count += 1
                continue
            if validate and not REQUIRED_RECORD_FIELDS.issubset(row.keys()):
                invalid_count += 1
                continue
            records.append(row)
    if invalid_count:
        # Stash count on the function for the runner to surface in logs.
        read_records_file.last_invalid_count = invalid_count  # type: ignore[attr-defined]
    return records


def read_all_records(records_dir: Path) -> list[dict]:
    """Concatenate records from every jsonl file in the records directory."""
    if not records_dir.exists():
        return []
    all_records: list[dict] = []
    for path in sorted(records_dir.glob("*.jsonl")):
        all_records.extend(read_records_file(path))
    return all_records


def completed_qids_for_combo(records_dir: Path, system: str, granularity: str) -> set[str]:
    """Set of query_ids already recorded for this (system, granularity).

    Validates each record on read. Truncated rows or rows missing a required
    metric are not counted as completed and will be re-run on resume.
    """
    path = records_dir / combo_filename(system, granularity)
    return {
        r["query_id"]
        for r in read_records_file(path, validate=True)
        if r.get("query_id")
    }


# ----------------------------------------------------------------------
# Run directory lifecycle
# ----------------------------------------------------------------------

def prepare_run_dir(run_dir: Path, *, resume: bool, overwrite: bool) -> None:
    """Create or validate the run directory according to the chosen mode.

    - default:   fail if folder exists
    - resume:    reuse folder, keep records/
    - overwrite: delete folder contents, start fresh
    """
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        elif not resume:
            raise SystemExit(
                f"Run directory already exists: {run_dir}\n"
                f"Use --resume to continue, --overwrite to replace, "
                f"or pick a different --label."
            )
    run_dir.mkdir(parents=True, exist_ok=resume)
    (run_dir / "records").mkdir(exist_ok=True)
