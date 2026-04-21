"""Detect amendment (perubahan) documents by title.

Scans all indexed docs and flags those whose title matches the amendment
pattern "Perubahan [Ke-N] atas <UU|Peraturan> ...". Useful for splitting
the re-index backlog into "light" (non-amendment) vs "complex" (amendment)
batches — amendment docs need Option A LLM-first generation, while
non-amendment docs can go through the standard llm_fix.py flow.

Output: writes `data/amendment_docs.json` with split lists, plus prints
summary to stdout.

Usage:
    python scripts/parser/detect_perubahan.py
    python scripts/parser/detect_perubahan.py --non-ok-only   # intersect with non-OK
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"
QUALITY_REPORT = REPO_ROOT / "data" / "parser_quality_report.json"
OUTPUT_PATH = REPO_ROOT / "data" / "amendment_docs.json"

# Title pattern: "Perubahan [Ke-N] atas <Undang-Undang|Peraturan> ...".
# Matches "Perubahan atas", "Perubahan Kedua atas", "Perubahan Keempat Atas", etc.
AMENDMENT_TITLE_RE = re.compile(
    r"\bPerubahan(?:\s+Ke\w+)?\s+[Aa]tas\s+(?:Undang|Peraturan)",
    re.IGNORECASE,
)


def is_amendment_title(judul: str) -> bool:
    """Return True if the title matches an amendment pattern."""
    return bool(AMENDMENT_TITLE_RE.search(judul or ""))


def load_quality_report() -> dict[str, str]:
    """Return dict of doc_id -> status (OK/PARTIAL/FAIL) from quality report."""
    if not QUALITY_REPORT.exists():
        return {}
    data = json.load(open(QUALITY_REPORT, encoding="utf-8"))
    return {d["doc_id"]: d.get("status", "UNKNOWN") for d in data.get("docs", [])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--non-ok-only",
        action="store_true",
        help="Only show docs that are also non-OK in quality report",
    )
    args = ap.parse_args()

    status_by_id = load_quality_report()

    amendment: list[dict] = []
    non_amendment: list[dict] = []
    for p in sorted(glob.glob(str(INDEX_PASAL / "*/*.json"))):
        if Path(p).name == "catalog.json":
            continue
        d = json.load(open(p, encoding="utf-8"))
        doc_id = d.get("doc_id")
        judul = d.get("judul", "")
        status = status_by_id.get(doc_id, "UNKNOWN")
        entry = {"doc_id": doc_id, "status": status, "judul": judul}
        if is_amendment_title(judul):
            amendment.append(entry)
        else:
            non_amendment.append(entry)

    if args.non_ok_only:
        amendment = [e for e in amendment if e["status"] != "OK"]
        non_amendment = [e for e in non_amendment if e["status"] != "OK"]

    total = len(amendment) + len(non_amendment)
    print(f"Total docs: {total}")
    print(f"Amendment (is_perubahan=True): {len(amendment)}")
    print(f"Non-amendment: {len(non_amendment)}")
    if args.non_ok_only:
        print("(filtered to non-OK only)")
    print()

    def _summary(bucket: list[dict], label: str) -> None:
        if not bucket:
            return
        counts: dict[str, int] = {}
        for e in bucket:
            counts[e["status"]] = counts.get(e["status"], 0) + 1
        print(f"{label} by status:")
        for st, n in sorted(counts.items()):
            print(f"  {st}: {n}")
        print()

    _summary(amendment, "Amendment")
    _summary(non_amendment, "Non-amendment")

    print(f"Amendment doc IDs ({len(amendment)}):")
    for e in amendment:
        print(f"  [{e['status']}] {e['doc_id']}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "amendment": amendment,
                "non_amendment": non_amendment,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
