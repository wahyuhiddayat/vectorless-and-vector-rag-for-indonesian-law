"""Shared utilities for scripts/ — path resolution, registry, catalog loading.

Extracted from duplicated code across gt/, eval/, and parser/ scripts.
Import via:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _shared import load_registry, load_catalog, find_pdf_path
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data" / "raw" / "registry.json"
DATA_DIR = REPO_ROOT / "data"

GRANULARITIES = ("pasal", "ayat", "full_split")


def load_registry() -> dict:
    """Load data/raw/registry.json and return the parsed dict.

    Returns empty dict if registry missing (caller decides error handling).
    """
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def load_catalog(granularity: str = "pasal") -> list[dict]:
    """Load data/index_<granularity>/catalog.json and return list of entries."""
    if granularity not in GRANULARITIES:
        raise ValueError(f"granularity must be one of {GRANULARITIES}")
    path = DATA_DIR / f"index_{granularity}" / "catalog.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_pdf_path(doc_id: str) -> Path | None:
    """Locate the main PDF for a doc_id via registry metadata.

    Preferred: return `entry["pdf_path"]` (deterministic mapping written by
    the scraper). Fallback: pattern-match filenames under the jenis folder
    for legacy registries that lack pdf_path.
    """
    registry = load_registry()
    entry = registry.get(doc_id)
    if not entry:
        return None
    # Preferred: explicit path recorded by scraper.
    explicit = entry.get("pdf_path")
    if explicit:
        p = DATA_DIR / "raw" / explicit
        if p.exists():
            return p
    # Fallback: filename pattern match (legacy pre-fix registry).
    jenis = entry.get("jenis_folder", "")
    nomor = entry.get("nomor", "")
    tahun = entry.get("tahun", "")
    if not jenis:
        return None
    pdf_dir = DATA_DIR / "raw" / jenis / "pdfs"
    if not pdf_dir.exists():
        return None
    candidates = []
    # Use word-boundary regex so "5" does not false-match inside "2025" etc.
    # Common filename patterns seen in corpus:
    #   peraturan-bi-no-5-tahun-2025.pdf           (slug style)
    #   PBI_102024.pdf                              (concat: NN + YYYY)
    #   Lamp_Batang_Tubuh_2025PBI010.pdf           (concat: YYYY + PBI + NNN pad)
    #   3 tahun 2026.pdf                            (loose)
    # Strategy: accept if ANY pattern with nomor + tahun AS STANDALONE tokens
    # (digit boundary, not substring) matches.
    nom_re = re.escape(str(nomor))
    thn_re = re.escape(str(tahun))
    patterns = [
        # "nomor 5 tahun 2025" / "no 5 tahun 2025" / "no. 5 tahun 2025"
        rf"(?:nomor|no\.?)\s+{nom_re}\s+tahun\s+{thn_re}",
        # "5-tahun-2025" / "5 tahun 2025"
        rf"(?<!\d){nom_re}[-\s]tahun[-\s]{thn_re}(?!\d)",
        # "PBI_NN YYYY" or "PBI_NNNYYYY" (concat, zero-padded up to 3)
        rf"(?<!\d){nom_re.zfill(1)}{thn_re}(?!\d)",
        rf"(?<!\d)0{nom_re}{thn_re}(?!\d)",
        rf"(?<!\d)00{nom_re}{thn_re}(?!\d)",
        # "YYYYPBI_NN" concat
        rf"{thn_re}PBI_?0*{nom_re}(?!\d)",
        rf"{thn_re}[A-Za-z]+0*{nom_re}(?!\d)",
    ]
    combined = re.compile("|".join(patterns), re.IGNORECASE)
    for pdf in pdf_dir.glob("*.pdf"):
        if "Lampiran" in pdf.name:
            continue
        if combined.search(pdf.name):
            candidates.append(pdf)
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.name))


__all__ = ["load_registry", "load_catalog", "find_pdf_path", "REPO_ROOT", "DATA_DIR"]
