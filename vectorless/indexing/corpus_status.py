"""Single source of truth for corpus state. Audit and reconcile artifacts.

Cross-references raw PDFs, registry, all 3 index granularities, and judge
verdicts. Writes a per-doc status into `data/corpus_status.json`. Decides
GT eligibility from the judge verdict.

Eligibility policy.
  KEEP    judge verdict in {OK, MINOR}
  DROP    judge verdict in {MAJOR, FAIL, ERROR}
  KEEP    no judge entry yet (treated as not-yet-judged, not as failure)

Public entries: `build_status() -> dict`, `write_status(status)`, and
`reconcile(status, dry_run=False)`. CLI access (with --reconcile/--dry-run/--json
flags and a printed summary) lives at `scripts/parser/corpus_status.py`.
This module is library-only.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from ..ids import doc_category

ROOT = Path(__file__).resolve().parents[2]

REGISTRY_PATH = ROOT / "data" / "raw" / "registry.json"
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIRS = {
    "pasal": ROOT / "data" / "index_pasal",
    "ayat": ROOT / "data" / "index_ayat",
    "rincian": ROOT / "data" / "index_rincian",
}
JUDGE_PATH = ROOT / "data" / "judge_report.json"
STATUS_PATH = ROOT / "data" / "corpus_status.json"
DROPPED_LOG_PATH = ROOT / "data" / "dropped_docs.json"

KEEP_VERDICTS = {"OK", "MINOR"}
DROP_VERDICTS = {"MAJOR", "FAIL", "ERROR"}


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _load_judge() -> dict:
    """Map doc_id to its latest judge entry."""
    if not JUDGE_PATH.exists():
        return {}
    with open(JUDGE_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return {d["doc_id"]: d for d in data.get("docs", [])}


def _index_files(doc_id: str, category: str) -> dict[str, bool]:
    return {
        gran: (path / category / f"{doc_id}.json").exists()
        for gran, path in INDEX_DIRS.items()
    }


def _raw_pdf_path(category: str, doc_id: str) -> Path:
    return RAW_DIR / category / "pdfs" / f"{doc_id}.pdf"


def _raw_metadata_paths(category: str, doc_id: str) -> list[Path]:
    md_dir = RAW_DIR / category / "metadata"
    if not md_dir.exists():
        return []
    return [p for p in md_dir.iterdir() if p.stem.startswith(f"{doc_id}__") or p.stem == doc_id]


def _eligible(verdict: str | None, raw_pdf: bool, fully_indexed: bool) -> tuple[bool, str | None]:
    """Return (eligible_for_gt, skip_reason).

    None verdict means not yet judged. Such a doc is eligible only if its
    artifacts are coherent (has raw PDF and is fully indexed). A registry
    entry with no raw and no index is an orphan, treat as ineligible.
    """
    if verdict in KEEP_VERDICTS:
        return True, None
    if verdict in DROP_VERDICTS:
        return False, f"judge_verdict={verdict}"
    if verdict is None:
        if not raw_pdf and not fully_indexed:
            return False, "orphan: registry entry with no raw and no index"
        return True, None
    return True, None


def build_status() -> dict:
    """Scan filesystem and judge report. Return canonical status dict."""
    registry = _load_registry()
    judge = _load_judge()

    all_doc_ids: set[str] = set(registry) | set(judge)
    for gran, base in INDEX_DIRS.items():
        if not base.exists():
            continue
        for cat_dir in base.iterdir():
            if not cat_dir.is_dir():
                continue
            for f in cat_dir.glob("*.json"):
                all_doc_ids.add(f.stem)

    docs: list[dict] = []
    for doc_id in sorted(all_doc_ids):
        if doc_id == "catalog":
            continue
        category = doc_category(doc_id)
        reg_entry = registry.get(doc_id)
        judge_entry = judge.get(doc_id)
        verdict = judge_entry.get("verdict") if judge_entry else None
        idx = _index_files(doc_id, category)
        pdf = _raw_pdf_path(category, doc_id)
        eligible, skip_reason = _eligible(verdict, pdf.exists(), all(idx.values()))

        docs.append({
            "doc_id": doc_id,
            "category": category,
            "raw_pdf": pdf.exists(),
            "in_registry": reg_entry is not None,
            "indexed": idx,
            "fully_indexed": all(idx.values()),
            "judge_verdict": verdict,
            "judge_score": judge_entry.get("overall_score") if judge_entry else None,
            "judge_missing_pasals": len((judge_entry or {}).get("coverage", {}).get("missing", [])),
            "eligible_for_gt": eligible,
            "skip_reason": skip_reason,
        })

    by_cat: dict[str, dict[str, int]] = {}
    for d in docs:
        c = by_cat.setdefault(d["category"], {
            "total": 0, "raw": 0, "registry": 0, "indexed": 0,
            "judged": 0, "eligible_gt": 0,
            "ok": 0, "minor": 0, "major": 0, "fail": 0, "error": 0,
        })
        c["total"] += 1
        c["raw"] += d["raw_pdf"]
        c["registry"] += d["in_registry"]
        c["indexed"] += d["fully_indexed"]
        c["judged"] += d["judge_verdict"] is not None
        c["eligible_gt"] += d["eligible_for_gt"]
        v = (d["judge_verdict"] or "").lower()
        if v in c:
            c[v] += 1

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "policy": {
            "keep_verdicts": sorted(KEEP_VERDICTS),
            "drop_verdicts": sorted(DROP_VERDICTS),
            "no_judge_entry": "treated as eligible (not yet judged)",
        },
        "by_category": by_cat,
        "docs": docs,
    }


def write_status(status: dict) -> None:
    STATUS_PATH.write_text(
        json.dumps(status, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_dropped_log() -> dict:
    """Load the persistent log of previously-dropped doc_ids."""
    if not DROPPED_LOG_PATH.exists():
        return {"docs": []}
    try:
        with open(DROPPED_LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"docs": []}


def reconcile(status: dict, dry_run: bool = False) -> dict:
    """Drop ineligible doc artifacts so raw, registry, and index stay in sync.

    Appends an entry to data/dropped_docs.json for each doc dropped, so
    future scrapes or expansions can skip known-bad docs without losing
    the reason they were rejected.
    """
    registry = _load_registry()
    judge_data: dict | None
    if JUDGE_PATH.exists():
        with open(JUDGE_PATH, encoding="utf-8") as f:
            judge_data = json.load(f)
    else:
        judge_data = None
    dropped_log = _load_dropped_log()
    log_seen = {d["doc_id"] for d in dropped_log["docs"]}

    drops = [d for d in status["docs"] if not d["eligible_for_gt"]]
    actions: dict[str, list] = {"docs_dropped": [], "files_removed": []}

    for d in drops:
        doc_id = d["doc_id"]
        category = d["category"]
        removed: list[str] = []

        judge_entry = next(
            (j for j in (judge_data or {}).get("docs", []) if j["doc_id"] == doc_id),
            None,
        )
        if doc_id not in log_seen:
            dropped_log["docs"].append({
                "doc_id": doc_id,
                "category": category,
                "dropped_at": datetime.now(timezone.utc).isoformat(),
                "reason": d["skip_reason"],
                "verdict": d["judge_verdict"],
                "score": d["judge_score"],
                "missing_pasals": d["judge_missing_pasals"],
                "notes": (judge_entry or {}).get("notes"),
            })

        for gran, base in INDEX_DIRS.items():
            f = base / category / f"{doc_id}.json"
            if f.exists():
                if not dry_run:
                    f.unlink()
                removed.append(str(f.relative_to(ROOT)))

        pdf = _raw_pdf_path(category, doc_id)
        if pdf.exists():
            if not dry_run:
                pdf.unlink()
            removed.append(str(pdf.relative_to(ROOT)))

        for md in _raw_metadata_paths(category, doc_id):
            if not dry_run:
                md.unlink()
            removed.append(str(md.relative_to(ROOT)))

        if doc_id in registry:
            if not dry_run:
                registry.pop(doc_id)
            removed.append(f"registry[{doc_id}]")

        if judge_data is not None:
            before = len(judge_data["docs"])
            judge_data["docs"] = [j for j in judge_data["docs"] if j["doc_id"] != doc_id]
            if len(judge_data["docs"]) < before:
                removed.append(f"judge_report[{doc_id}]")

        actions["docs_dropped"].append({"doc_id": doc_id, "reason": d["skip_reason"]})
        actions["files_removed"].extend(removed)

    if not dry_run:
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        if judge_data is not None:
            JUDGE_PATH.write_text(
                json.dumps(judge_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        DROPPED_LOG_PATH.write_text(
            json.dumps(dropped_log, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return actions


