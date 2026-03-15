"""
build_ayat.py — Build ayat-level structural index from scraped Indonesian legal PDFs.

Identical to build_pasal.py but uses parser_ayat instead of parser.
The ayat-level parser splits Pasal leaf nodes into Ayat sub-nodes only,
without further recursion into Huruf/Angka. This produces a mid-granularity
tree index for comparison against Pasal-level and deep-leaf indices.

Output goes to data/index_ayat/ (not data/index_pasal/) to avoid overwriting
the pasal-level index.

Usage:
    python -m vectorless.indexing.build_ayat                          # index all documents (with LLM cleanup)
    python -m vectorless.indexing.build_ayat --no-llm                 # skip LLM cleanup (dev/testing only)
    python -m vectorless.indexing.build_ayat --doc-id uu-20-2025      # index single document
    python -m vectorless.indexing.build_ayat --force                  # re-index even if exists
"""

import argparse
import json
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .parser import (parse_legal_pdf, parse_penjelasan, _attach_penjelasan,
                     extract_pages, clean_page_text, find_penjelasan_page,
                     _iter_leaves)

DATA_RAW = Path("data/raw")
DATA_INDEX = Path("data/index_ayat")
REGISTRY_PATH = DATA_RAW / "registry.json"

def load_registry() -> dict:
    """Load data/raw/registry.json produced by the scraper.

    Returns dict of doc_id -> registry entry, e.g.:
      {"uu-20-2025": {"judul": "...", "has_pdf": True, "detail_id": "337302", ...}}
    """
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)

def load_metadata(doc_id: str, detail_id: str, jenis_folder: str) -> dict | None:
    """Load detailed metadata for a document from scraper output.

    Metadata files are named: {doc_id}__{detail_id}.json
    jenis_folder comes from registry (e.g. "UU", "UU Darurat", "Peraturan BPK").
    Returns None if file not found.
    """
    meta_dir = DATA_RAW / jenis_folder / "metadata"
    meta_path = meta_dir / f"{doc_id}__{detail_id}.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)

def pick_main_pdf(metadata: dict) -> str | None:
    """Select the main PDF filename from metadata['pdf_files'].

    Some documents on BPK JDIH are split into multiple PDFs:
    the main law text + separate Lampiran (appendix) files for tables/attachments.
    We only want to parse the main law text — Lampiran files contain non-normative
    content (data tables, maps, org charts) that don't follow BAB/Pasal structure
    and would break the parser.

    After filtering Lampiran out, picks the shortest filename as a heuristic —
    the main document always has the simplest name (e.g. "UU Nomor 15 Tahun 2025.pdf")
    while variants like "Salinan Naskah Resmi" are longer.

    Returns None if no pdf_files listed in metadata.
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
    """Find a separate Penjelasan PDF if one exists.

    Some older documents (e.g. Perpu 2014) split the Penjelasan into a
    separate PDF file. Returns the filename if found, None otherwise.
    """
    pdf_files = metadata.get("pdf_files", [])
    for p in pdf_files:
        if "Penjelasan" in p["filename"] or "penjelasan" in p["filename"]:
            return p["filename"]
    return None

def add_navigation_paths(nodes: list[dict], ancestors: list[str] | None = None):
    """Add 'navigation_path' field to each node in-place, recursively.

    A navigation_path is the full path from the document root down to a node,
    joined by " > ". It records exactly where in the document a Pasal lives,
    and becomes the reasoning path shown to the user after retrieval.

    Example: "BAB V - UPAYA PAKSA > Bagian Kelima - Penangkapan > Pasal 113"
    """
    if ancestors is None:
        ancestors = []
    for node in nodes:
        path = ancestors + [node["title"]]
        node["navigation_path"] = " > ".join(path)
        if "nodes" in node:
            add_navigation_paths(node["nodes"], path)

def enrich_doc(parse_result: dict, registry_entry: dict, metadata: dict | None) -> dict:
    """Combine parser output with registry/metadata into the final index document.

    The final document is what gets saved to data/index_ayat/{doc_id}.json and contains:
    - Document metadata: doc_id, judul, bidang, subjek, materi_pokok, relasi, etc.
    - Parser stats: total_pages, element_counts, warnings
    - Tree structure: BAB > Bagian > Paragraf > Pasal, with navigation_path on each node
    """
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

    # Add metadata-only fields if available (richer than registry)
    if metadata:
        doc["bidang"] = metadata.get("bidang")
        doc["subjek"] = metadata.get("subjek")
        doc["materi_pokok"] = metadata.get("materi_pokok")
        # Use relasi from metadata — has keterangan + ref_slug detail
        doc["relasi"] = metadata.get("relasi", registry_entry.get("relasi", []))
    else:
        doc["relasi"] = registry_entry.get("relasi", [])

    # Add parser results
    doc["total_pages"] = parse_result["total_pages"]
    doc["body_pages"] = parse_result["body_pages"]
    doc["penjelasan_page"] = parse_result["penjelasan_page"]
    doc["penjelasan_umum"] = parse_result.get("penjelasan_umum")
    doc["penjelasan_pasal_demi_pasal"] = parse_result.get("penjelasan_pasal_demi_pasal")
    doc["element_counts"] = parse_result["element_counts"]
    doc["warnings"] = parse_result["warnings"]

    # Add navigation_paths to structure
    structure = parse_result["structure"]
    add_navigation_paths(structure)
    doc["structure"] = structure

    return doc

def build_catalog(index_dir: Path) -> list[dict]:
    """Scan all indexed docs and produce catalog.json — a lightweight summary for document discovery.

    The catalog contains only metadata fields (no tree structure), so the retrieval agent
    can quickly scan all documents to decide which one(s) are relevant to a query,
    without loading full tree JSONs into memory.

    Scans category subfolders (UU/, PP/, Perpres/, Perpu/) for index JSONs.
    """
    catalog = []
    for path in sorted(index_dir.rglob("*.json")):
        if path.name == "catalog.json":
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
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

def main():
    """Entry point. Reads registry, parses each PDF, and saves enriched index JSONs.

    Flow per document:
      registry.json -> load metadata -> pick main PDF -> parse (+ LLM cleanup)
      -> add navigation_paths -> merge metadata -> save data/index_ayat/{doc_id}.json

    After all documents: rebuild catalog.json from all saved index files.
    """
    ap = argparse.ArgumentParser(description="Build structural index from scraped legal PDFs")
    ap.add_argument("--no-llm", action="store_true", help="Skip Gemini LLM cleanup (dev/testing only, reduces index quality)")
    ap.add_argument("--doc-id", type=str, help="Index a single document by doc_id")
    ap.add_argument("--force", action="store_true", help="Re-index even if output exists")
    args = ap.parse_args()

    use_llm = not args.no_llm
    print(f"Mode: LLM cleanup {'ON' if use_llm else 'OFF (--no-llm)'}  |  Force: {'ON' if args.force else 'OFF'}")

    # Load registry
    registry = load_registry()
    docs = {k: v for k, v in registry.items() if v.get("has_pdf")}
    print(f"Registry: {len(docs)} documents with PDFs\n")

    # Filter to single doc if specified
    if args.doc_id:
        if args.doc_id not in docs:
            print(f"ERROR: doc_id '{args.doc_id}' not found in registry (or has no PDF)")
            sys.exit(1)
        docs = {args.doc_id: docs[args.doc_id]}

    DATA_INDEX.mkdir(parents=True, exist_ok=True)

    success, skipped, failed = 0, 0, 0
    t_total = time.time()

    for i, (doc_id, entry) in enumerate(docs.items(), 1):
        category = doc_id.split("-")[0].upper()
        category_dir = DATA_INDEX / category
        category_dir.mkdir(parents=True, exist_ok=True)
        output_path = category_dir / f"{doc_id}.json"

        # Skip if already indexed
        if output_path.exists() and not args.force:
            print(f"[{i}/{len(docs)}] SKIP {doc_id} (already indexed)")
            skipped += 1
            continue

        judul = entry.get("judul", "")
        print(f"\n[{i}/{len(docs)}] {doc_id} — {judul}")

        # Load metadata
        detail_id = entry.get("detail_id", "")
        jenis_folder = entry.get("jenis_folder", doc_id.split("-")[0].upper())
        metadata = load_metadata(doc_id, detail_id, jenis_folder)
        if metadata is None:
            print(f"  WARN  No metadata file, using registry only")

        # Pick main PDF
        if metadata:
            pdf_filename = pick_main_pdf(metadata)
        else:
            bentuk = entry.get("bentuk_singkat", "UU")
            nomor = entry.get("nomor", "")
            tahun = entry.get("tahun", "")
            pdf_filename = f"{bentuk} Nomor {nomor} Tahun {tahun}.pdf"

        pdf_path = DATA_RAW / jenis_folder / "pdfs" / pdf_filename
        print(f"  PDF   {pdf_filename}")

        if not pdf_path.exists():
            print(f"  ERROR PDF not found: {pdf_path}")
            failed += 1
            continue

        # Parse
        t0 = time.time()
        try:
            parse_result = parse_legal_pdf(str(pdf_path), verbose=False, use_llm_cleanup=use_llm,
                                               granularity="ayat")
        except Exception as e:
            print(f"  ERROR Parse failed: {e}")
            failed += 1
            continue
        elapsed = time.time() - t0

        counts = parse_result["element_counts"]
        parts = [f"{parse_result['total_pages']} pages"]
        for key in ("bab", "bagian", "paragraf", "pasal"):
            if counts.get(key):
                parts.append(f"{counts[key]} {key}")
        print(f"  Parse {', '.join(parts)} ({elapsed:.1f}s)")

        if parse_result["warnings"]:
            print(f"  WARN  {len(parse_result['warnings'])} OCR warning(s)")

        # Handle separate Penjelasan PDF (some older docs split it out)
        if not parse_result["penjelasan_page"] and metadata:
            penjelasan_filename = pick_penjelasan_pdf(metadata)
            if penjelasan_filename:
                penjelasan_path = DATA_RAW / jenis_folder / "pdfs" / penjelasan_filename
                if penjelasan_path.exists():
                    print(f"  PENJ  Separate PDF: {penjelasan_filename}")
                    try:
                        penj_pages = extract_pages(str(penjelasan_path))
                        for p in penj_pages:
                            p["clean_text"] = clean_page_text(p["raw_text"])
                        penj_start = find_penjelasan_page(penj_pages) or 1
                        penj_data = parse_penjelasan(penj_pages, penj_start, len(penj_pages))
                        parse_result["penjelasan_umum"] = penj_data.get("umum")
                        _attach_penjelasan(parse_result["structure"], penj_data["pasal"])
                        matched = sum(1 for n in _iter_leaves(parse_result["structure"])
                                      if n.get("penjelasan"))
                        print(f"         {len(penj_data['pasal'])} pasal parsed, {matched} matched")
                    except Exception as e:
                        print(f"  WARN  Failed to parse separate Penjelasan: {e}")

        # Enrich and save
        enriched = enrich_doc(parse_result, entry, metadata)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)
        print(f"  Saved {output_path}")
        success += 1

    # Rebuild catalog
    print(f"\nBuilding catalog...")
    catalog = build_catalog(DATA_INDEX)
    catalog_path = DATA_INDEX / "catalog.json"
    with open(catalog_path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    print(f"Catalog: {len(catalog)} documents -> {catalog_path}")

    total_elapsed = time.time() - t_total
    print(f"\nDone: {success} indexed, {skipped} skipped, {failed} failed  |  {total_elapsed:.1f}s total")

if __name__ == "__main__":
    main()