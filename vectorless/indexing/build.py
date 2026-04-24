"""LLM-first indexing: LLM parse + deterministic re-split.

For each target doc:
  1. LLM-parse → data/index_pasal/<CAT>/<doc_id>.json
  2. Re-split ayat → data/index_ayat/<CAT>/<doc_id>.json
  3. Re-split rincian → data/index_rincian/<CAT>/<doc_id>.json

Usage:
    python -m vectorless.indexing.build --category UU
    python -m vectorless.indexing.build --doc-id uu-3-2025
    python -m vectorless.indexing.build --category UU --resplit-only
    python -m vectorless.indexing.build --category UU --dry-run
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

from .parser import (  # noqa: E402
    ayat_split_leaves,
    deep_split_leaves,
    iter_leaves,
    strip_ocr_headers,
)

REGISTRY_PATH = Path("data/raw/registry.json")
INDEXING_LOGS_DIR = Path("data/indexing_logs")

GRANULARITY_INDEX_MAP = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "rincian": Path("data/index_rincian"),
}

CATALOG_FIELDS = (
    "doc_id", "judul", "nomor", "tahun", "bentuk_singkat", "status",
    "tanggal_penetapan", "bidang", "subjek", "materi_pokok", "relasi",
    "total_pages", "jenis_folder",
)


def add_navigation_paths(nodes: list[dict], ancestors: list[str] | None = None) -> None:
    """Fill `navigation_path` from the node ancestry."""
    if ancestors is None:
        ancestors = []
    for node in nodes:
        path = ancestors + [node["title"]]
        node["navigation_path"] = " > ".join(path)
        if "nodes" in node:
            add_navigation_paths(node["nodes"], path)


def resplit_one(pasal_doc: dict, granularity: str) -> dict:
    """Derive ayat or rincian doc from a pasal-granularity doc."""
    split_fn = ayat_split_leaves if granularity == "ayat" else deep_split_leaves
    doc = copy.deepcopy(pasal_doc)
    doc["structure"] = split_fn(doc["structure"])
    strip_ocr_headers(doc["structure"])
    add_navigation_paths(doc["structure"])
    return doc


def build_catalog(index_dir: Path) -> list[dict]:
    """Build the compact catalog stored beside each index."""
    catalog = []
    for path in sorted(index_dir.rglob("*.json")):
        if path.name == "catalog.json":
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "doc_id" not in doc:
            continue
        catalog.append({f: doc.get(f) for f in CATALOG_FIELDS})
    return catalog


def _load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"registry not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _resolve_targets(doc_ids: list[str], category: str | None) -> list[str]:
    if doc_ids:
        return doc_ids
    if not category:
        raise SystemExit("must pass --doc-id(s) or --category")
    reg = _load_registry()
    target = category.upper()
    return sorted(
        did for did, entry in reg.items()
        if (entry.get("jenis_folder") or "").upper() == target
    )


def _update_cost_log(granularity: str, doc_id: str, entry: dict) -> None:
    """Replace one document entry in the per-granularity cost log."""
    INDEXING_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEXING_LOGS_DIR / f"cost_{granularity}.json"
    data = {}
    if path.exists():
        try:
            data = json.load(open(path, encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[doc_id] = entry
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resplit_derived(pasal_path: Path, jenis_folder: str, doc_id: str) -> dict[str, int]:
    """Re-split a pasal document into its derived granularities."""
    with open(pasal_path, encoding="utf-8") as f:
        pasal_doc = json.load(f)

    counts: dict[str, int] = {}
    for gran in ("ayat", "rincian"):
        out_path = GRANULARITY_INDEX_MAP[gran] / jenis_folder / f"{doc_id}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        resplit_doc = resplit_one(pasal_doc, gran)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(resplit_doc, f, ensure_ascii=False, indent=2)
        counts[gran] = sum(1 for _ in iter_leaves(resplit_doc["structure"]))
    return counts


def index_doc(doc_id: str, dry_run: bool = False, resplit_only: bool = False) -> dict:
    """Index one document and return a short status summary."""
    from scripts.parser.llm_parse import parse_doc as llm_parse_doc, _append_audit

    t0 = time.time()
    summary: dict = {"doc_id": doc_id}

    if resplit_only:
        pasal_path = None
        for p in GRANULARITY_INDEX_MAP["pasal"].glob(f"*/{doc_id}.json"):
            pasal_path = p
            break
        if not pasal_path:
            summary["status"] = "error"
            summary["error"] = "no existing pasal index to re-split from"
            return summary
        with open(pasal_path, encoding="utf-8") as f:
            jenis_folder = json.load(f).get("jenis_folder") or pasal_path.parent.name
        t_resplit = time.time()
        derived = _resplit_derived(pasal_path, jenis_folder, doc_id)
        resplit_elapsed = round(time.time() - t_resplit, 3)
        summary.update({"status": "resplit_ok", "derived_counts": derived, "elapsed_s": round(time.time() - t0, 2)})
        for gran in ("ayat", "rincian"):
            _update_cost_log(gran, doc_id, {
                "category": jenis_folder.upper(),
                "updated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
                "resplit_time_s": resplit_elapsed,
                "leaf_count": derived[gran],
            })
        return summary

    audit = llm_parse_doc(doc_id, dry_run=dry_run)
    _append_audit(audit)
    summary["llm_parse_status"] = audit.get("status")
    summary["pasal_count"] = audit.get("pasal_count")
    summary["validation_errors"] = audit.get("validation_errors") or []

    if audit.get("status") != "parsed" or dry_run:
        summary["status"] = audit.get("status", "error")
        if "error" in audit:
            summary["error"] = audit["error"]
        return summary

    pasal_path = Path(audit["index_path"])
    jenis_folder = pasal_path.parent.name
    t_resplit = time.time()
    derived = _resplit_derived(pasal_path, jenis_folder, doc_id)
    resplit_elapsed = round(time.time() - t_resplit, 3)
    usage = audit.get("usage") or {}
    now_iso = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")

    _update_cost_log("pasal", doc_id, {
        "category": jenis_folder.upper(),
        "updated_at": now_iso,
        "llm_time_s": usage.get("elapsed_s"),
        "llm_input_tokens": usage.get("input_tokens"),
        "llm_output_tokens": usage.get("output_tokens"),
        "llm_total_tokens": usage.get("total_tokens"),
        "llm_calls": usage.get("calls"),
        "mode": audit.get("mode"),
        "pasal_count": audit.get("pasal_count"),
        "pdf_pasal_regex_count": audit.get("pdf_pasal_regex_count"),
        "pdf_chars": audit.get("pdf_chars"),
        "body_pages": audit.get("body_pages"),
    })
    for gran in ("ayat", "rincian"):
        _update_cost_log(gran, doc_id, {
            "category": jenis_folder.upper(),
            "updated_at": now_iso,
            "resplit_time_s": resplit_elapsed,
            "leaf_count": derived[gran],
        })

    summary.update({
        "status": "indexed",
        "derived_counts": derived,
        "elapsed_s": round(time.time() - t0, 2),
    })
    return summary


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description="LLM-first indexing pipeline")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[],
                    help="Doc to index (repeatable)")
    ap.add_argument("--doc-ids", dest="doc_ids_csv", default="",
                    help="Comma-separated doc_ids")
    ap.add_argument("--category",
                    help="Index every doc in this jenis_folder (e.g. UU, OJK)")
    ap.add_argument("--resplit-only", action="store_true",
                    help="Skip LLM-parse; only re-derive ayat + rincian from existing pasal index")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview LLM-parse only; do not overwrite any index file")
    args = ap.parse_args()

    doc_ids = list(args.doc_ids)
    if args.doc_ids_csv:
        doc_ids.extend([x.strip() for x in args.doc_ids_csv.split(",") if x.strip()])

    targets = _resolve_targets(doc_ids, args.category)
    log.info(f"indexing {len(targets)} docs (resplit_only={args.resplit_only}, dry_run={args.dry_run})")

    ok = 0
    t_total = time.time()
    for i, did in enumerate(targets, 1):
        log.info(f"[{i}/{len(targets)}] {did}")
        try:
            result = index_doc(did, dry_run=args.dry_run, resplit_only=args.resplit_only)
        except Exception as exc:
            log.exception(f"  failed: {exc}")
            continue
        status = result.get("status")
        msg = f"  {status}"
        if "pasal_count" in result and result["pasal_count"] is not None:
            msg += f"  pasals={result['pasal_count']}"
        if "derived_counts" in result:
            dc = result["derived_counts"]
            msg += f"  ayat={dc.get('ayat')} rincian={dc.get('rincian')}"
        if result.get("validation_errors"):
            msg += f"  val_errs={len(result['validation_errors'])}"
        if result.get("error"):
            msg += f"  error={result['error']}"
        log.info(msg)
        if status in ("indexed", "resplit_ok", "dry_run_ok"):
            ok += 1

    elapsed = time.time() - t_total
    log.info(f"done  {ok}/{len(targets)} succeeded  {elapsed:.1f}s")

    if args.dry_run:
        return

    for gran, idx_dir in GRANULARITY_INDEX_MAP.items():
        if not idx_dir.exists():
            continue
        catalog = build_catalog(idx_dir)
        with open(idx_dir / "catalog.json", "w", encoding="utf-8") as f:
            json.dump(catalog, f, ensure_ascii=False, indent=2)
        log.info(f"catalog ({gran})  {len(catalog)} docs  {idx_dir / 'catalog.json'}")


if __name__ == "__main__":
    main()
