"""Disk I/O and resume-support helpers."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import pickle
import re
import shutil
from pathlib import Path


SPLITS_DIR = Path("data/splits")
TEST_SEAL_ENV = "EVAL_ALLOW_TEST"


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


def load_split_qids(split: str, splits_dir: Path | None = None) -> list[str]:
    """Read the qid list for one of dev, val, test from data/splits/."""
    base = splits_dir if splits_dir is not None else SPLITS_DIR
    path = base / f"{split}_qids.json"
    if not path.exists():
        raise SystemExit(
            f"Split file not found, {path}. Run scripts/gt/split_dataset.py first."
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def split_fingerprint(split: str, splits_dir: Path | None = None) -> str:
    """Sha256 of the qid list, mirroring the manifest format."""
    qids = load_split_qids(split, splits_dir)
    payload = "\n".join(sorted(qids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_queries(
    testset: dict[str, dict],
    doc_id: str | None,
    query_limit: int | None,
    random_seed: int | None = None,
    query_types: list[str] | None = None,
    per_type_limit: int | None = None,
    split: str | None = None,
    splits_dir: Path | None = None,
) -> list[tuple[str, dict]]:
    """Select a query subset for evaluation.

    Filtering precedence, split -> doc_id -> query_types -> per_type_limit -> query_limit.
    The split filter reads data/splits/<split>_qids.json. Test split is sealed,
    callers must set EVAL_ALLOW_TEST=1 to opt in. per_type_limit picks N items
    per query_type (stratified). When combined with random_seed the selection
    within each type is sampled rather than head-of-list.
    """
    items = sorted(testset.items(), key=lambda kv: kv[0])
    if split:
        if split not in {"dev", "val", "test"}:
            raise SystemExit(f"Unknown split, {split}. Choose dev, val, or test.")
        if split == "test" and os.environ.get(TEST_SEAL_ENV) != "1":
            raise SystemExit(
                "Test split is sealed. Set EVAL_ALLOW_TEST=1 to confirm "
                "intentional final-report usage."
            )
        keep = set(load_split_qids(split, splits_dir))
        items = [(qid, item) for qid, item in items if qid in keep]
    if doc_id:
        items = [(qid, item) for qid, item in items if item.get("gold_doc_id") == doc_id]
    if query_types:
        types_set = set(query_types)
        items = [(qid, item) for qid, item in items if item.get("query_type", "factual") in types_set]
    if per_type_limit is not None:
        import random
        buckets: dict[str, list[tuple[str, dict]]] = {}
        for qid, item in items:
            buckets.setdefault(item.get("query_type", "factual"), []).append((qid, item))
        rng = random.Random(random_seed) if random_seed is not None else None
        picked: list[tuple[str, dict]] = []
        for bucket in buckets.values():
            sample = bucket if len(bucket) <= per_type_limit else (
                rng.sample(bucket, per_type_limit) if rng else bucket[:per_type_limit]
            )
            picked.extend(sample)
        items = sorted(picked, key=lambda kv: kv[0])
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
