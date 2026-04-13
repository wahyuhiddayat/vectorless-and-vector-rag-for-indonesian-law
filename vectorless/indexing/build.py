"""
Build structural indices from scraped Indonesian legal PDFs.

Granularity outputs:
- pasal: data/index_pasal/
- ayat: data/index_ayat/
- full_split: data/index_full_split/

Pipeline:
- Pass 1 (parse): PDF -> tree (offline)
- Pass 2 (LLM): Gemini cleanup -> llm_cleaned=true

Use --help for CLI examples.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger(__name__)

from .parser import (parse_legal_pdf, parse_penjelasan, attach_penjelasan,
                     extract_pages, clean_page_text, find_penjelasan_page,
                     iter_leaves, ayat_split_leaves, deep_split_leaves,
                     strip_ocr_headers, apply_llm_cleanup)
from .status import (
    apply_verify_results,
    clear_doc_error,
    is_cleanup_stale,
    load_status_manifest,
    set_doc_error,
    sync_manifest_from_indexes,
    write_status_manifest,
)

DATA_RAW = Path("data/raw")
REGISTRY_PATH = DATA_RAW / "registry.json"
PARSER_VERSION = "2026-04-02"
LLM_CLEANUP_VERSION = "2026-04-02"


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def load_registry() -> dict:
    """Load registry.json produced by the scraper."""
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_metadata(doc_id: str, detail_id: str, jenis_folder: str) -> dict | None:
    """Load per-document metadata; return None when metadata file is missing."""
    meta_dir = DATA_RAW / jenis_folder / "metadata"
    meta_path = meta_dir / f"{doc_id}__{detail_id}.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def pick_main_pdf(metadata: dict) -> str | None:
    """Return the main law PDF filename, excluding Lampiran appendix files.

    Filters out Lampiran PDFs (tables, maps, org charts) that would break the
    parser, then picks the shortest remaining filename as a heuristic for the
    primary law text. Returns None if no pdf_files are listed in metadata.
    """
    pdf_files = metadata.get("pdf_files", [])
    if not pdf_files:
        return None
    if len(pdf_files) == 1:
        return pdf_files[0]["filename"]

    # Filter out lampiran files
    candidates = [
        p["filename"] for p in pdf_files
        if "Lampiran" not in p["filename"]
    ]
    if not candidates:
        candidates = [p["filename"] for p in pdf_files]

    return min(candidates, key=len)


def pick_penjelasan_pdf(metadata: dict) -> str | None:
    """Return the separate Penjelasan PDF filename if present."""
    pdf_files = metadata.get("pdf_files", [])
    for p in pdf_files:
        if "Penjelasan" in p["filename"] or "penjelasan" in p["filename"]:
            return p["filename"]
    return None


def add_navigation_paths(nodes: list[dict], ancestors: list[str] | None = None):
    """Populate navigation_path for each node recursively."""
    if ancestors is None:
        ancestors = []
    for node in nodes:
        path = ancestors + [node["title"]]
        node["navigation_path"] = " > ".join(path)
        if "nodes" in node:
            add_navigation_paths(node["nodes"], path)


def enrich_doc(parse_result: dict, registry_entry: dict, metadata: dict | None) -> dict:
    """Merge parser output with registry and metadata for index storage."""
    doc = {
        "doc_id": registry_entry["doc_id"],
        "judul": registry_entry["judul"],
        "nomor": registry_entry.get("nomor"),
        "tahun": registry_entry.get("tahun"),
        "bentuk_singkat": registry_entry.get("bentuk_singkat"),
        "status": registry_entry.get("status"),
        "tanggal_penetapan": registry_entry.get("tanggal_penetapan"),
        "jenis_folder": registry_entry.get("jenis_folder"),
    }

    # Prefer richer fields from metadata when available.
    if metadata:
        doc["bidang"] = metadata.get("bidang")
        doc["subjek"] = metadata.get("subjek")
        doc["materi_pokok"] = metadata.get("materi_pokok")
        # Metadata relasi includes richer details than registry relasi.
        doc["relasi"] = metadata.get("relasi", registry_entry.get("relasi", []))
    else:
        doc["relasi"] = registry_entry.get("relasi", [])

    # Parser-derived fields.
    doc["total_pages"] = parse_result["total_pages"]
    doc["body_pages"] = parse_result["body_pages"]
    doc["penjelasan_page"] = parse_result["penjelasan_page"]
    doc["penjelasan_umum"] = parse_result.get("penjelasan_umum")
    doc["penjelasan_pasal_demi_pasal"] = parse_result.get("penjelasan_pasal_demi_pasal")
    doc["element_counts"] = parse_result["element_counts"]
    doc["warnings"] = parse_result["warnings"]

    # Add navigation_path to every node.
    structure = parse_result["structure"]
    add_navigation_paths(structure)
    doc["structure"] = structure

    return doc


def build_catalog(index_dir: Path) -> list[dict]:
    """Build catalog.json summary from all indexed document JSON files."""
    catalog = []
    for path in sorted(index_dir.rglob("*.json")):
        if path.name.startswith("catalog"):
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "doc_id" not in doc:
            log.warning(f"skip non-document json while building catalog: {path}")
            continue
        catalog.append({
            "doc_id": doc["doc_id"],
            "judul": doc["judul"],
            "nomor": doc.get("nomor"),
            "tahun": doc.get("tahun"),
            "bentuk_singkat": doc.get("bentuk_singkat"),
            "status": doc.get("status"),
            "tanggal_penetapan": doc.get("tanggal_penetapan"),
            "bidang": doc.get("bidang"),
            "subjek": doc.get("subjek"),
            "materi_pokok": doc.get("materi_pokok"),
            "relasi": doc.get("relasi", []),
            "total_pages": doc.get("total_pages"),
            "element_counts": doc.get("element_counts"),
            "jenis_folder": doc.get("jenis_folder"),
        })
    return catalog


GRANULARITY_INDEX_MAP = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "full_split": Path("data/index_full_split"),
}


def _resplit_from_pasal(granularity: str, doc_id: str | None, rebuild: str | None,
                        manifest: dict, registry: dict, category: str | None = None):
    """Re-split pasal index to ayat/full_split without PDF parse or LLM calls."""
    import copy

    pasal_dir = GRANULARITY_INDEX_MAP["pasal"]
    target_dir = GRANULARITY_INDEX_MAP[granularity]
    split_fn = ayat_split_leaves if granularity == "ayat" else deep_split_leaves
    categories = _normalize_categories(category)

    log.info(f"re-split from pasal to {granularity}  output {target_dir}  rebuild {rebuild or 'missing'}")
    if category:
        log.info(f"category {category}")

    pasal_files = sorted(pasal_dir.rglob("*.json"))
    pasal_files = [f for f in pasal_files if not f.name.startswith("catalog")]

    if not pasal_files:
        log.error(f"no pasal index files found in {pasal_dir}")
        sys.exit(1)

    success, skipped = 0, 0
    t_start = time.time()

    for pf in pasal_files:
        with open(pf, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "doc_id" not in doc:
            log.warning(f"skip non-document json: {pf}")
            skipped += 1
            continue

        did = doc["doc_id"]
        if doc_id and did != doc_id:
            continue
        if categories:
            doc_category = (doc.get("jenis_folder") or did.split("-")[0]).upper()
            if doc_category not in categories:
                continue

        # Preserve pasal directory layout in target index.
        rel = pf.relative_to(pasal_dir)
        out_path = target_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)

        status_entry = manifest.get("docs", {}).get(did, {})
        if status_entry.get("stale_parse"):
            log.info(f"  skip  {did:30s}  pasal parse is stale, rebuild pasal first")
            skipped += 1
            continue

        if out_path.exists() and not _should_resplit_doc(out_path, rebuild, did, status_entry):
            skipped += 1
            continue

        # Re-split copied structure and normalize residual OCR headers.
        structure = copy.deepcopy(doc["structure"])
        structure = split_fn(structure)
        strip_ocr_headers(structure)
        add_navigation_paths(structure)

        doc["structure"] = structure
        doc["parser_version"] = doc.get("parser_version") or PARSER_VERSION
        doc["source_pasal_parser_version"] = doc["parser_version"]
        doc["source_pasal_parse_updated_at"] = doc.get("parse_updated_at") or now_iso()
        doc["source_pasal_updated_at"] = doc.get("pasal_updated_at") or now_iso()
        doc["derived_updated_at"] = now_iso()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        leaves = sum(1 for _ in iter_leaves(structure))
        log.info(f"  {did:30s}  {leaves:5d} leaves  {out_path}")
        clear_doc_error(manifest, did, registry.get(did))
        sync_manifest_from_indexes(
            manifest,
            registry,
            PARSER_VERSION,
            LLM_CLEANUP_VERSION,
            doc_ids=[did],
        )
        write_status_manifest(manifest)
        success += 1

    # Rebuild target catalog after re-splitting.
    log.info("building catalog")
    catalog = build_catalog(target_dir)
    catalog_path = target_dir / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    log.info(f"catalog  {len(catalog)} documents  {catalog_path}")

    elapsed = time.time() - t_start
    log.info(f"re-split done  {success} re-split, {skipped} skipped  {elapsed:.1f}s (no LLM)")


# Shared helpers for parse and LLM passes.

def _normalize_categories(category: str | None) -> set[str]:
    if not category:
        return set()
    return {part.strip().upper() for part in category.split(",") if part.strip()}


def _resolve_docs(registry: dict, rebuild: str | None, doc_id: str | None,
                  category: str | None = None) -> dict:
    """Return the filtered docs dict based on --rebuild, --doc-id, and --category flags."""
    docs = {k: v for k, v in registry.items() if v.get("has_pdf")}
    categories = _normalize_categories(category)

    if doc_id:
        if doc_id not in docs:
            log.error(f"doc_id '{doc_id}' not found in registry (or has no PDF)")
            sys.exit(1)
        return {doc_id: docs[doc_id]}

    if categories:
        docs = {
            k: v for k, v in docs.items()
            if (v.get("jenis_folder") or k.split("-")[0]).upper() in categories
        }

    return docs


def _output_path(data_index: Path, doc_id: str) -> Path:
    """Return the expected output JSON path for a given doc_id."""
    category = doc_id.split("-")[0].upper()
    return data_index / category / f"{doc_id}.json"


def _targeted_doc_ids(rebuild: str | None) -> set[str]:
    if rebuild and rebuild not in ("all", "uncleaned", "stale"):
        return {doc_id.strip() for doc_id in rebuild.split(",") if doc_id.strip()}
    return set()


def _should_parse_doc(output_path: Path, rebuild: str | None, doc_id: str, status_entry: dict | None) -> bool:
    """Return True if a parse pass should run for this document."""
    if rebuild == "all":
        return True
    if rebuild == "stale":
        return (status_entry or {}).get("stale_parse", False) or not output_path.exists()
    if rebuild == "uncleaned":
        return not output_path.exists()
    if rebuild:
        return doc_id in _targeted_doc_ids(rebuild)
    return not output_path.exists()


def _should_force_llm(rebuild: str | None, doc_id: str, status_entry: dict | None) -> bool:
    """Return True if LLM cleanup should re-run even on already-clean docs."""
    entry = status_entry or {}
    if rebuild == "all":
        return True
    if rebuild == "stale":
        return bool(
            entry.get("pasal_exists")
            and entry.get("llm_cleaned")
            and not entry.get("stale_parse")
            and entry.get("llm_cleanup_version") != LLM_CLEANUP_VERSION
        )
    if rebuild == "uncleaned":
        return bool(
            entry.get("pasal_exists")
            and entry.get("llm_cleaned")
            and entry.get("llm_cleanup_version") != LLM_CLEANUP_VERSION
            and not entry.get("stale_parse")
        )
    if rebuild:
        return doc_id in _targeted_doc_ids(rebuild)
    return False


def _should_resplit_doc(output_path: Path, rebuild: str | None, doc_id: str, status_entry: dict | None) -> bool:
    """Return True if ayat/full_split should be regenerated from pasal."""
    if rebuild == "all":
        return True
    if rebuild == "stale":
        return (status_entry or {}).get("stale_derived", False) or not output_path.exists()
    if rebuild == "uncleaned":
        return not output_path.exists()
    if rebuild:
        return doc_id in _targeted_doc_ids(rebuild)
    return not output_path.exists()


def _load_pdf_for_doc(entry: dict, metadata: dict | None,
                      data_index: Path) -> tuple[Path, str | None]:
    """Return (pdf_path, error_message). error_message is None on success."""
    jenis_folder = entry.get("jenis_folder", entry["doc_id"].split("-")[0].upper())
    if metadata:
        pdf_filename = pick_main_pdf(metadata)
    else:
        bentuk = entry.get("bentuk_singkat", "UU")
        nomor = entry.get("nomor", "")
        tahun = entry.get("tahun", "")
        pdf_filename = f"{bentuk} Nomor {nomor} Tahun {tahun}.pdf"

    pdf_path = DATA_RAW / jenis_folder / "pdfs" / pdf_filename
    if not pdf_path.exists():
        return pdf_path, f"PDF not found: {pdf_path}"
    return pdf_path, None


# Pass 1 parses PDFs and saves llm_cleaned as false.

def _parse_pass(data_index: Path, docs: dict, granularity: str,
                rebuild: str | None, manifest: dict) -> tuple[int, int, int]:
    """Parse PDFs into tree JSON and save with llm_cleaned=false.

    Fully offline (no network calls). Skips existing outputs unless targeted by --rebuild.
    Returns (success, skipped, failed).
    """
    log.info("pass 1  parse PDFs (offline, no LLM)")

    success, skipped, failed = 0, 0, 0
    t_total = time.time()

    for i, (doc_id, entry) in enumerate(docs.items(), 1):
        output_path = _output_path(data_index, doc_id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        status_entry = manifest.get("docs", {}).get(doc_id, {})

        # Skip existing files unless they are selected for rebuild.
        if output_path.exists() and not _should_parse_doc(output_path, rebuild, doc_id, status_entry):
            log.info(f"[{i}/{len(docs)}] skip  {doc_id}")
            skipped += 1
            continue

        judul = entry.get("judul", "")
        log.info(f"[{i}/{len(docs)}] {doc_id}  {judul}")

        detail_id = entry.get("detail_id", "")
        jenis_folder = entry.get("jenis_folder", doc_id.split("-")[0].upper())
        metadata = load_metadata(doc_id, detail_id, jenis_folder)
        if metadata is None:
            log.warning("no metadata file, using registry only")

        pdf_path, err = _load_pdf_for_doc(entry, metadata, data_index)
        log.info(f"pdf  {pdf_path.name}")
        if err:
            log.error(f"{err}")
            set_doc_error(manifest, doc_id, err, entry)
            write_status_manifest(manifest)
            failed += 1
            continue

        t0 = time.time()
        try:
            parse_result = parse_legal_pdf(str(pdf_path), verbose=False,
                                           use_llm_cleanup=False,
                                           granularity=granularity)
        except Exception as e:
            log.error(f"parse failed: {e}")
            set_doc_error(manifest, doc_id, f"parse failed: {e}", entry)
            write_status_manifest(manifest)
            failed += 1
            continue

        # Parse separate Penjelasan PDF when it is split from the main PDF.
        if not parse_result["penjelasan_page"] and metadata:
            penjelasan_filename = pick_penjelasan_pdf(metadata)
            if penjelasan_filename:
                penjelasan_path = DATA_RAW / jenis_folder / "pdfs" / penjelasan_filename
                if penjelasan_path.exists():
                    log.info(f"penjelasan  {penjelasan_filename}")
                    try:
                        penj_pages = extract_pages(str(penjelasan_path))
                        for p in penj_pages:
                            p["clean_text"] = clean_page_text(p["raw_text"])
                        penj_start = find_penjelasan_page(penj_pages) or 1
                        penj_data = parse_penjelasan(penj_pages, penj_start, len(penj_pages))
                        parse_result["penjelasan_umum"] = penj_data.get("umum")
                        attach_penjelasan(parse_result["structure"], penj_data["pasal"])
                        matched = sum(1 for n in iter_leaves(parse_result["structure"])
                                      if n.get("penjelasan"))
                        log.info(f"{len(penj_data['pasal'])} pasal parsed, {matched} matched")
                    except Exception as e:
                        log.warning(f"failed to parse separate penjelasan: {e}")

        counts = parse_result["element_counts"]
        parts = [f"{parse_result['total_pages']} pages"]
        for key in ("bab", "bagian", "paragraf", "pasal"):
            if counts.get(key):
                parts.append(f"{counts[key]} {key}")
        elapsed = time.time() - t0
        log.info(f"parse  {', '.join(parts)}  ({elapsed:.1f}s)")

        if parse_result["warnings"]:
            log.warning(f"{len(parse_result['warnings'])} OCR warning(s)")

        enriched = enrich_doc(parse_result, entry, metadata)
        enriched["llm_cleaned"] = False
        enriched["parser_version"] = PARSER_VERSION
        enriched["llm_cleanup_version"] = None
        enriched["parse_updated_at"] = now_iso()
        enriched["pasal_updated_at"] = enriched["parse_updated_at"]
        enriched["llm_cleaned_at"] = None

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        log.info("saved (pending LLM cleanup)")
        clear_doc_error(manifest, doc_id, entry)
        sync_manifest_from_indexes(
            manifest,
            docs,
            PARSER_VERSION,
            LLM_CLEANUP_VERSION,
            doc_ids=[doc_id],
        )
        write_status_manifest(manifest)
        success += 1

    total_elapsed = time.time() - t_total
    log.info(f"pass 1 done  {success} parsed, {skipped} skipped, {failed} failed  {total_elapsed:.1f}s")
    return success, skipped, failed


# Pass 2 runs LLM cleanup and updates llm_cleaned to true.

def _llm_pass(data_index: Path, docs: dict, rebuild: str | None,
              manifest: dict) -> tuple[int, int, int]:
    """Run Gemini LLM cleanup on parsed docs with llm_cleaned=false.

    Returns (success, skipped, failed).
    """
    import os

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        log.error("GEMINI_API_KEY is not set")
        sys.exit(1)

    log.info("pass 2  LLM cleanup (Gemini 2.5 Flash)")

    success, skipped, failed = 0, 0, 0
    t_total = time.time()

    for i, (doc_id, entry) in enumerate(docs.items(), 1):
        output_path = _output_path(data_index, doc_id)
        status_entry = manifest.get("docs", {}).get(doc_id, {})

        if not output_path.exists():
            log.warning(f"[{i}/{len(docs)}] missing {doc_id}, run parse pass first")
            set_doc_error(manifest, doc_id, "missing parsed output, run parse pass first", entry)
            write_status_manifest(manifest)
            failed += 1
            continue

        with open(output_path, encoding="utf-8") as f:
            doc = json.load(f)

        already_cleaned = doc.get("llm_cleaned", False)
        if status_entry.get("stale_parse"):
            log.warning(f"[{i}/{len(docs)}] skip  {doc_id} (parse stale, rerun parse pass first)")
            skipped += 1
            continue

        # Re-run cleanup when requested by rebuild targeting.
        force_this = _should_force_llm(rebuild, doc_id, status_entry)

        if already_cleaned and not force_this:
            log.info(f"[{i}/{len(docs)}] skip  {doc_id} (already cleaned)")
            skipped += 1
            continue

        judul = entry.get("judul", "")
        log.info(f"[{i}/{len(docs)}] {doc_id}  {judul}")

        structure = doc["structure"]
        penjelasan_umum = doc.get("penjelasan_umum")
        # Mutable proxy so apply_llm_cleanup can write the cleaned penjelasan_umum back.
        penjelasan_proxy = {"umum": penjelasan_umum} if penjelasan_umum else None

        t0 = time.time()
        try:
            llm_failures = apply_llm_cleanup(structure, penjelasan_proxy, verbose=True)
        except Exception as e:
            # Keep processing other docs if one doc fails unexpectedly.
            log.error(f"LLM cleanup failed: {e.__class__.__name__}: {e}")
            log.warning("doc kept with llm_cleaned=false, retry with --llm-only")
            set_doc_error(manifest, doc_id, f"LLM cleanup failed: {e.__class__.__name__}: {e}", entry)
            write_status_manifest(manifest)
            failed += 1
            continue

        elapsed = time.time() - t0
        n_failures = len(llm_failures) if llm_failures else 0

        # Persist cleaned content and cleanup status.
        doc["structure"] = structure
        if penjelasan_proxy:
            doc["penjelasan_umum"] = penjelasan_proxy["umum"]
        doc["llm_cleaned"] = True
        doc["llm_cleanup_version"] = LLM_CLEANUP_VERSION
        doc["pasal_updated_at"] = now_iso()
        doc["llm_cleaned_at"] = now_iso()
        if llm_failures:
            existing = doc.get("warnings", [])
            doc["warnings"] = existing + llm_failures

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        if n_failures:
            log.warning(f"LLM done ({elapsed:.1f}s)  {n_failures} batch(es) kept as raw OCR")
        else:
            log.info(f"LLM done ({elapsed:.1f}s)  all batches cleaned")
        clear_doc_error(manifest, doc_id, entry)
        sync_manifest_from_indexes(
            manifest,
            docs,
            PARSER_VERSION,
            LLM_CLEANUP_VERSION,
            doc_ids=[doc_id],
        )
        write_status_manifest(manifest)
        success += 1

    total_elapsed = time.time() - t_total
    log.info(f"pass 2 done  {success} cleaned, {skipped} skipped, {failed} failed  {total_elapsed:.1f}s")
    return success, skipped, failed


# CLI entry point.

def main():
    """Entry point. Parses CLI args and runs the indexing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description="Build structural index from scraped legal PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Full pipeline (parse + LLM):
    python -m vectorless.indexing.build --granularity pasal

  Parse only (offline, no API cost):
    python -m vectorless.indexing.build --granularity pasal --parse-only

  LLM cleanup only (resume after sleep/network failure):
    python -m vectorless.indexing.build --granularity pasal --llm-only

  Rebuild specific docs:
    python -m vectorless.indexing.build --granularity pasal --rebuild uu-14-2025,pp-3-2026

  Rebuild all:
    python -m vectorless.indexing.build --granularity pasal --rebuild all

  Rebuild only uncleaned docs:
    python -m vectorless.indexing.build --granularity pasal --rebuild uncleaned

  Rebuild only stale docs after a parser-version bump:
    python -m vectorless.indexing.build --granularity pasal --rebuild stale

  Full pipeline across all granularities + verify:
    python -m vectorless.indexing.build --granularity pasal --full-pipeline
""")
    ap.add_argument("--granularity", choices=["pasal", "ayat", "full_split"], required=True,
                    help="Leaf node granularity: pasal (coarsest), ayat (mid), full_split (finest)")
    ap.add_argument("--doc-id", type=str,
                    help="Operate on a single document by doc_id (overrides --rebuild)")
    ap.add_argument("--category", type=str,
                    help="Filter docs by kategori/folder, e.g. UU,PP,PMK,PERMENAKER")
    ap.add_argument("--parse-only", action="store_true",
                    help="Run only Pass 1 (PDF parsing, no LLM). Good for iterating parser fixes.")
    ap.add_argument("--llm-only", action="store_true",
                    help="Run only Pass 2 (LLM cleanup on parsed docs). Resumes after sleep/network failure.")
    ap.add_argument("--rebuild", type=str, default=None, metavar="WHAT",
                    help=(
                        "What to rebuild. Options: "
                        "'all' = force-rebuild everything; "
                        "'uncleaned' = redo docs with llm_cleaned=false; "
                        "'stale' = rebuild docs whose parser/derived versions are stale; "
                        "'doc1,doc2,...' = specific doc_ids. "
                        "Default: only build missing files."
                    ))
    ap.add_argument("--from-pasal", action="store_true",
                    help="Re-split from existing pasal index (no PDF parsing, no LLM). Only for ayat/full_split.")
    ap.add_argument("--full-pipeline", action="store_true",
                    help="Run complete pipeline: pasal parse+LLM → ayat resplit → full_split resplit → verify.")
    # Backward compatibility aliases.
    ap.add_argument("--no-llm", action="store_true",
                    help="(legacy) Alias for --parse-only.")
    ap.add_argument("--force", action="store_true",
                    help="(legacy) Alias for --rebuild all.")
    args = ap.parse_args()

    # Normalize compatibility flags to current options.
    if args.no_llm:
        args.parse_only = True
    if args.force and not args.rebuild:
        args.rebuild = "all"

    granularity = args.granularity
    registry = load_registry()
    manifest = load_status_manifest(PARSER_VERSION, LLM_CLEANUP_VERSION)
    sync_manifest_from_indexes(manifest, registry, PARSER_VERSION, LLM_CLEANUP_VERSION)
    write_status_manifest(manifest)

    # Fast re-split mode for ayat/full_split from existing pasal outputs.
    if args.from_pasal:
        if granularity == "pasal":
            log.error("--from-pasal only works with ayat or full_split")
            sys.exit(1)
        _resplit_from_pasal(granularity, args.doc_id, args.rebuild, manifest, registry, args.category)
        return

    # Full pipeline mode across all granularities, then verification.
    if args.full_pipeline:
        if granularity != "pasal":
            log.error("--full-pipeline must be started with --granularity pasal")
            sys.exit(1)
        _run_full_pipeline(args, registry, manifest)
        return

    data_index = GRANULARITY_INDEX_MAP[granularity]
    data_index.mkdir(parents=True, exist_ok=True)

    docs = _resolve_docs(registry, args.rebuild, args.doc_id, args.category)
    log.info(f"granularity {granularity}  output {data_index}  docs {len(docs)}")
    if args.rebuild:
        log.info(f"rebuild {args.rebuild}")
    if args.category:
        log.info(f"category {args.category}")

    # Determine active passes.
    run_parse = not args.llm_only
    run_llm   = not args.parse_only

    if run_parse:
        _parse_pass(data_index, docs, granularity, rebuild=args.rebuild, manifest=manifest)

    if run_llm:
        _llm_pass(data_index, docs, rebuild=args.rebuild, manifest=manifest)

    # Rebuild catalog after pass execution.
    log.info("building catalog")
    catalog = build_catalog(data_index)
    catalog_path = data_index / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    log.info(f"catalog  {len(catalog)} documents  {catalog_path}")
    sync_manifest_from_indexes(
        manifest,
        registry,
        PARSER_VERSION,
        LLM_CLEANUP_VERSION,
        doc_ids=list(docs.keys()),
    )
    write_status_manifest(manifest)


def _run_full_pipeline(args, registry: dict, manifest: dict):
    """Run pasal parse+LLM, then ayat/full_split resplit, then verify."""
    from .verify import verify_index, print_report

    docs = _resolve_docs(registry, args.rebuild, args.doc_id, args.category)
    rebuild = args.rebuild

    # Pasal phase runs parse and LLM cleanup.
    pasal_index = GRANULARITY_INDEX_MAP["pasal"]
    pasal_index.mkdir(parents=True, exist_ok=True)

    _parse_pass(pasal_index, docs, "pasal", rebuild=rebuild, manifest=manifest)
    _llm_pass(pasal_index, docs, rebuild=rebuild, manifest=manifest)

    catalog = build_catalog(pasal_index)
    with open(pasal_index / "catalog.json", "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    log.info(f"catalog (pasal)  {len(catalog)} docs")

    # Ayat and full_split phases resplit from pasal outputs.
    for gran in ("ayat", "full_split"):
        _resplit_from_pasal(gran, args.doc_id, rebuild, manifest, registry, args.category)

    # Verify all granularities.
    log.info("verify all granularities")
    all_ok = True
    for gran in ("pasal", "ayat", "full_split"):
        idx = GRANULARITY_INDEX_MAP[gran]
        results = verify_index(idx, args.doc_id)
        if results:
            print_report(results, idx)
            apply_verify_results(manifest, gran, results, registry=registry)
            if any(r["status"] != "OK" for r in results):
                all_ok = False
    sync_manifest_from_indexes(
        manifest,
        registry,
        PARSER_VERSION,
        LLM_CLEANUP_VERSION,
        doc_ids=list(docs.keys()),
    )
    write_status_manifest(manifest)

    log.info(f"pipeline complete  {'all checks passed' if all_ok else 'some issues found, check output above'}")


if __name__ == "__main__":
    main()
