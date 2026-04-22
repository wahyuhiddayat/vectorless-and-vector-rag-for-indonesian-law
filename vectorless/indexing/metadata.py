"""Build document metadata from scrape + PDF preamble (no regex structure).

The output dict is the non-structure portion of a pasal-index JSON. The LLM
structure tree is attached separately by scripts/parser/llm_parse.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from vectorless.indexing.parser import (
    clean_page_text,
    detect_perubahan,
    extract_pages,
    find_closing_page,
    find_penjelasan_page,
    parse_penjelasan,
)

DATA_RAW = Path("data/raw")

# "Perubahan [Ke-N] atas Undang|Peraturan ..." — matches amendment titles only.
_AMENDMENT_TITLE_RE = re.compile(
    r"\bPerubahan(?:\s+Ke\w+)?\s+[Aa]tas\s+(?:Undang|Peraturan)",
    re.IGNORECASE,
)


def _load_registry() -> dict:
    path = DATA_RAW / "registry.json"
    if not path.exists():
        raise FileNotFoundError(f"registry not found at {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_detail_metadata(doc_id: str, detail_id: str, jenis_folder: str) -> dict | None:
    meta_path = DATA_RAW / jenis_folder / "metadata" / f"{doc_id}__{detail_id}.json"
    if not meta_path.exists():
        return None
    with open(meta_path, encoding="utf-8") as f:
        return json.load(f)


def _pick_penjelasan_pdf(detail: dict | None) -> str | None:
    """Return filename of a separate Penjelasan PDF if one was attached."""
    if not detail:
        return None
    for p in detail.get("pdf_files", []):
        if "penjelasan" in p["filename"].lower():
            return p["filename"]
    return None


def build_metadata(doc_id: str) -> dict:
    """Build the non-structure metadata portion of a pasal-index document.

    Sources:
      - registry.json: judul, nomor, tahun, scrape fields
      - detail metadata JSON: bidang, subjek, materi_pokok, relasi
      - PDF preamble parse: total_pages, body_pages, penjelasan_page,
        penjelasan_umum, penjelasan_pasal_demi_pasal
      - Heuristic: is_perubahan
    """
    registry = _load_registry()
    entry = registry.get(doc_id)
    if not entry:
        raise KeyError(f"doc_id {doc_id!r} not in registry")

    jenis_folder = entry["jenis_folder"]
    detail = _load_detail_metadata(doc_id, entry["detail_id"], jenis_folder)

    pdf_path = DATA_RAW / entry["pdf_path"]
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = extract_pages(str(pdf_path))
    for p in pages:
        p["clean_text"] = clean_page_text(p["raw_text"])
    total_pages = len(pages)

    closing_page = find_closing_page(pages)
    penjelasan_page = find_penjelasan_page(pages)
    boundaries = [p for p in (closing_page, penjelasan_page) if p]
    body_end = min(boundaries) - 1 if boundaries else total_pages

    # Title regex first, fallback to PDF-text scan (catches OCR-dropped "PERUBAHAN").
    judul = entry["judul"]
    is_perubahan = bool(_AMENDMENT_TITLE_RE.search(judul or "")) or detect_perubahan(pages)

    penjelasan_umum: str | None = None
    penjelasan_pasal: dict | None = None
    if penjelasan_page:
        penj = parse_penjelasan(pages, penjelasan_page, total_pages)
        penjelasan_umum = penj.get("umum")
        penjelasan_pasal = penj.get("pasal") or None

    # Some older UUs ship penjelasan as a separate PDF listed in detail metadata.
    if penjelasan_umum is None:
        penj_filename = _pick_penjelasan_pdf(detail)
        if penj_filename:
            penj_path = DATA_RAW / jenis_folder / "pdfs" / penj_filename
            if penj_path.exists():
                penj_pages = extract_pages(str(penj_path))
                for p in penj_pages:
                    p["clean_text"] = clean_page_text(p["raw_text"])
                penj_start = find_penjelasan_page(penj_pages) or 1
                penj = parse_penjelasan(penj_pages, penj_start, len(penj_pages))
                penjelasan_umum = penj.get("umum")
                penjelasan_pasal = penj.get("pasal") or penjelasan_pasal

    meta: dict = {
        "doc_id": doc_id,
        "judul": judul,
        "nomor": entry.get("nomor"),
        "tahun": entry.get("tahun"),
        "bentuk_singkat": entry.get("bentuk_singkat"),
        "status": entry.get("status"),
        "tanggal_penetapan": entry.get("tanggal_penetapan"),
        "jenis_folder": jenis_folder,
        "pdf_path": entry["pdf_path"],
        "total_pages": total_pages,
        "body_pages": body_end,
        "penjelasan_page": penjelasan_page,
        "penjelasan_umum": penjelasan_umum,
        "penjelasan_pasal_demi_pasal": penjelasan_pasal,
        "is_perubahan": is_perubahan,
    }

    if detail:
        meta["bidang"] = detail.get("bidang")
        meta["subjek"] = detail.get("subjek")
        meta["materi_pokok"] = detail.get("materi_pokok")
        meta["relasi"] = detail.get("relasi") or entry.get("relasi") or []
    else:
        meta["bidang"] = None
        meta["subjek"] = None
        meta["materi_pokok"] = None
        meta["relasi"] = entry.get("relasi") or []

    return meta
