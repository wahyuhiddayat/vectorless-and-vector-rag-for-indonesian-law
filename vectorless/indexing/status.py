import argparse
import json
from datetime import datetime, UTC
from pathlib import Path

DATA_RAW = Path("data/raw")
REGISTRY_PATH = DATA_RAW / "registry.json"
STATUS_PATH = Path("data/index_status.json")

GRANULARITY_INDEX_MAP = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "full_split": Path("data/index_full_split"),
}

SCHEMA_VERSION = 1
PARSER_VERSION = "2026-04-02"
LLM_CLEANUP_VERSION = "2026-04-02"


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def normalize_categories(category: str | None) -> set[str]:
    if not category:
        return set()
    return {part.strip().upper() for part in category.split(",") if part.strip()}


def _empty_verify_status() -> dict:
    return {gran: "MISSING" for gran in GRANULARITY_INDEX_MAP}


def _empty_warning_count() -> dict:
    return {gran: 0 for gran in GRANULARITY_INDEX_MAP}


def load_status_manifest(current_parser_version: str, current_llm_cleanup_version: str) -> dict:
    if STATUS_PATH.exists():
        with open(STATUS_PATH, encoding="utf-8") as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest.setdefault("schema_version", SCHEMA_VERSION)
    manifest["generated_at"] = now_iso()
    manifest["current_parser_version"] = current_parser_version
    manifest["current_llm_cleanup_version"] = current_llm_cleanup_version
    manifest.setdefault("docs", {})
    return manifest


def prune_orphan_manifest_entries(manifest: dict, registry: dict) -> None:
    """Drop stale manifest docs that are no longer in the registry and have no files."""
    docs = manifest.get("docs", {})
    to_remove = []
    for doc_id in docs:
        if doc_id in registry:
            continue
        has_any_index = any(_index_path(granularity, doc_id).exists() for granularity in GRANULARITY_INDEX_MAP)
        if not has_any_index:
            to_remove.append(doc_id)
    for doc_id in to_remove:
        docs.pop(doc_id, None)


def write_status_manifest(manifest: dict):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    manifest["generated_at"] = now_iso()
    with open(STATUS_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def ensure_doc_entry(manifest: dict, doc_id: str, registry_entry: dict | None = None) -> dict:
    docs = manifest.setdefault("docs", {})
    entry = docs.setdefault(doc_id, {
        "doc_id": doc_id,
        "jenis_folder": None,
        "pasal_exists": False,
        "ayat_exists": False,
        "full_split_exists": False,
        "llm_cleaned": False,
        "parser_version": None,
        "llm_cleanup_version": None,
        "parse_updated_at": None,
        "pasal_updated_at": None,
        "llm_cleaned_at": None,
        "verify_status": _empty_verify_status(),
        "warning_count": _empty_warning_count(),
        "stale_parse": False,
        "stale_derived": False,
        "last_error": None,
    })

    entry.setdefault("verify_status", _empty_verify_status())
    entry.setdefault("warning_count", _empty_warning_count())

    if registry_entry:
        entry["jenis_folder"] = registry_entry.get("jenis_folder", entry.get("jenis_folder"))

    return entry


def _index_path(granularity: str, doc_id: str) -> Path:
    category = doc_id.split("-")[0].upper()
    return GRANULARITY_INDEX_MAP[granularity] / category / f"{doc_id}.json"


def _load_doc_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def is_cleanup_stale(entry: dict, current_llm_cleanup_version: str) -> bool:
    if not entry.get("pasal_exists"):
        return False
    if not entry.get("llm_cleaned"):
        return True
    return entry.get("llm_cleanup_version") != current_llm_cleanup_version


def sync_doc_from_indexes(
    manifest: dict,
    doc_id: str,
    registry_entry: dict | None,
    current_parser_version: str,
    current_llm_cleanup_version: str,
):
    entry = ensure_doc_entry(manifest, doc_id, registry_entry)

    docs_by_granularity = {}
    paths_by_granularity = {}
    for granularity in GRANULARITY_INDEX_MAP:
        path = _index_path(granularity, doc_id)
        paths_by_granularity[granularity] = path
        doc = _load_doc_json(path)
        docs_by_granularity[granularity] = doc
        entry[f"{granularity}_exists"] = doc is not None
        if doc is None:
            entry["verify_status"][granularity] = "MISSING"
            entry["warning_count"][granularity] = 0
        else:
            entry["warning_count"][granularity] = len(doc.get("warnings", []))

    pasal_doc = docs_by_granularity["pasal"]
    pasal_path = paths_by_granularity["pasal"]

    if pasal_doc is None:
        entry["llm_cleaned"] = False
        entry["parser_version"] = None
        entry["llm_cleanup_version"] = None
        entry["parse_updated_at"] = None
        entry["pasal_updated_at"] = None
        entry["llm_cleaned_at"] = None
        entry["stale_parse"] = False
        entry["stale_derived"] = False
        return

    entry["llm_cleaned"] = bool(pasal_doc.get("llm_cleaned", False))
    entry["parser_version"] = pasal_doc.get("parser_version") or entry.get("parser_version") or current_parser_version
    entry["llm_cleanup_version"] = (
        pasal_doc.get("llm_cleanup_version")
        or entry.get("llm_cleanup_version")
        or (current_llm_cleanup_version if entry["llm_cleaned"] else None)
    )
    entry["parse_updated_at"] = pasal_doc.get("parse_updated_at") or entry.get("parse_updated_at") or mtime_iso(pasal_path)
    entry["pasal_updated_at"] = pasal_doc.get("pasal_updated_at") or mtime_iso(pasal_path)
    entry["llm_cleaned_at"] = (
        pasal_doc.get("llm_cleaned_at")
        or entry.get("llm_cleaned_at")
        or (mtime_iso(pasal_path) if entry["llm_cleaned"] else None)
    )
    entry["stale_parse"] = entry["parser_version"] != current_parser_version

    stale_derived = False
    for granularity in ("ayat", "full_split"):
        derived_doc = docs_by_granularity[granularity]
        if derived_doc is None:
            stale_derived = True
            continue
        source_parser_version = (
            derived_doc.get("source_pasal_parser_version")
            or derived_doc.get("parser_version")
            or current_parser_version
        )
        source_parse_updated_at = (
            derived_doc.get("source_pasal_parse_updated_at")
            or derived_doc.get("parse_updated_at")
            or mtime_iso(paths_by_granularity[granularity])
        )
        source_pasal_updated_at = (
            derived_doc.get("source_pasal_updated_at")
            or source_parse_updated_at
        )
        derived_updated_at = derived_doc.get("derived_updated_at")
        if source_parser_version != entry["parser_version"]:
            stale_derived = True
        if source_parse_updated_at != entry["parse_updated_at"]:
            if not (derived_updated_at and entry["pasal_updated_at"] and derived_updated_at >= entry["pasal_updated_at"]):
                stale_derived = True
        if source_pasal_updated_at != entry["pasal_updated_at"]:
            if not (derived_updated_at and entry["pasal_updated_at"] and derived_updated_at >= entry["pasal_updated_at"]):
                stale_derived = True
    entry["stale_derived"] = stale_derived


def sync_manifest_from_indexes(
    manifest: dict,
    registry: dict,
    current_parser_version: str,
    current_llm_cleanup_version: str,
    doc_ids: list[str] | None = None,
):
    prune_orphan_manifest_entries(manifest, registry)
    target_doc_ids = doc_ids or sorted(doc_id for doc_id, entry in registry.items() if entry.get("has_pdf"))
    for doc_id in target_doc_ids:
        sync_doc_from_indexes(
            manifest,
            doc_id,
            registry.get(doc_id),
            current_parser_version,
            current_llm_cleanup_version,
        )


def set_doc_error(manifest: dict, doc_id: str, message: str, registry_entry: dict | None = None):
    entry = ensure_doc_entry(manifest, doc_id, registry_entry)
    entry["last_error"] = message


def clear_doc_error(manifest: dict, doc_id: str, registry_entry: dict | None = None):
    entry = ensure_doc_entry(manifest, doc_id, registry_entry)
    entry["last_error"] = None


def apply_verify_results(manifest: dict, granularity: str, results: list[dict], registry: dict | None = None):
    for result in results:
        doc_id = result["doc_id"]
        entry = ensure_doc_entry(manifest, doc_id, registry.get(doc_id) if registry else None)
        entry["verify_status"][granularity] = result["status"]
        entry["warning_count"][granularity] = result["checks"]["warnings"]["count"]


def _summarize_status(manifest: dict, doc_ids: list[str] | None = None) -> tuple[dict, dict]:
    docs = manifest.get("docs", {})
    selected = [
        docs[doc_id]
        for doc_id in sorted(doc_ids or docs.keys())
        if doc_id in docs
    ]

    summary = {
        "total_docs": len(selected),
        "pasal_exists": sum(1 for d in selected if d.get("pasal_exists")),
        "ayat_exists": sum(1 for d in selected if d.get("ayat_exists")),
        "full_split_exists": sum(1 for d in selected if d.get("full_split_exists")),
        "llm_cleaned": sum(
            1 for d in selected
            if d.get("pasal_exists") and not is_cleanup_stale(d, manifest.get("current_llm_cleanup_version"))
        ),
        "uncleaned_or_cleanup_stale": sum(
            1 for d in selected
            if d.get("pasal_exists") and is_cleanup_stale(d, manifest.get("current_llm_cleanup_version"))
        ),
        "stale_parse": sum(1 for d in selected if d.get("stale_parse")),
        "stale_derived": sum(1 for d in selected if d.get("stale_derived")),
        "gt_candidates": sum(
            1 for d in selected
            if d.get("pasal_exists") and d.get("verify_status", {}).get("pasal") in {"OK", "WARN"}
        ),
        "clean_retrieval_candidates": sum(
            1 for d in selected
            if d.get("pasal_exists")
            and not d.get("stale_parse")
            and not d.get("stale_derived")
            and not is_cleanup_stale(d, manifest.get("current_llm_cleanup_version"))
            and d.get("verify_status", {}).get("pasal") in {"OK", "WARN"}
        ),
    }

    verify_summary = {}
    for granularity in GRANULARITY_INDEX_MAP:
        counts = {"OK": 0, "WARN": 0, "FAIL": 0, "MISSING": 0}
        for entry in selected:
            counts[entry.get("verify_status", {}).get(granularity, "MISSING")] += 1
        verify_summary[granularity] = counts

    return summary, verify_summary


def _refresh_verify_into_manifest(manifest: dict, doc_id: str | None = None):
    from .verify import verify_index

    registry = load_registry()
    for granularity, index_dir in GRANULARITY_INDEX_MAP.items():
        if not index_dir.exists():
            continue
        results = verify_index(index_dir, doc_id)
        apply_verify_results(manifest, granularity, results, registry=registry)


def main():
    ap = argparse.ArgumentParser(description="Show indexing progress/status manifest")
    ap.add_argument("--doc-id", type=str, help="Show status for one document")
    ap.add_argument("--category", type=str, help="Show only selected categories, e.g. UU,PP,PMK")
    ap.add_argument("--json", action="store_true", help="Output status summary as JSON")
    ap.add_argument("--refresh-verify", action="store_true",
                    help="Refresh verify_status in the manifest before printing")
    ap.add_argument("--parser-version", type=str, default=PARSER_VERSION)
    ap.add_argument("--llm-cleanup-version", type=str, default=LLM_CLEANUP_VERSION)
    args = ap.parse_args()

    registry = load_registry()
    manifest = load_status_manifest(args.parser_version, args.llm_cleanup_version)
    categories = normalize_categories(args.category)
    if args.doc_id:
        doc_ids = [args.doc_id]
    elif categories:
        doc_ids = [
            doc_id for doc_id, entry in registry.items()
            if entry.get("has_pdf") and (entry.get("jenis_folder") or doc_id.split("-")[0]).upper() in categories
        ]
    else:
        doc_ids = None
    sync_manifest_from_indexes(manifest, registry, args.parser_version, args.llm_cleanup_version, doc_ids=doc_ids)

    if args.refresh_verify:
        if doc_ids:
            for did in doc_ids:
                _refresh_verify_into_manifest(manifest, did)
        else:
            _refresh_verify_into_manifest(manifest, None)

    write_status_manifest(manifest)

    summary, verify_summary = _summarize_status(manifest, doc_ids=doc_ids)

    if args.json:
        payload = {
            "summary": summary,
            "verify_status": verify_summary,
            "docs": manifest["docs"] if not doc_ids else {doc_ids[0]: manifest["docs"].get(doc_ids[0])},
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("Index Status")
    print(f"Manifest: {STATUS_PATH}")
    if args.category:
        print(f"Category filter: {args.category}")
    print(f"Total docs: {summary['total_docs']}")
    print(f"Pasal: {summary['pasal_exists']}  |  Ayat: {summary['ayat_exists']}  |  Full split: {summary['full_split_exists']}")
    print(f"LLM cleaned current: {summary['llm_cleaned']}")
    print(f"Uncleaned / cleanup-stale: {summary['uncleaned_or_cleanup_stale']}")
    print(f"Stale parse: {summary['stale_parse']}  |  Stale derived: {summary['stale_derived']}")
    print(f"GT candidates: {summary['gt_candidates']}")
    print(f"Clean retrieval candidates: {summary['clean_retrieval_candidates']}")
    print("")
    for granularity in GRANULARITY_INDEX_MAP:
        counts = verify_summary[granularity]
        print(
            f"{granularity:10s}  OK={counts['OK']}  WARN={counts['WARN']}  "
            f"FAIL={counts['FAIL']}  MISSING={counts['MISSING']}"
        )


if __name__ == "__main__":
    main()
