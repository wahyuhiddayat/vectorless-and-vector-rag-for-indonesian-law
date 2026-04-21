"""Shared utilities for scripts/ — path resolution, registry, catalog loading.

Extracted from duplicated code across gt/, eval/, and parser/ scripts.
Import via:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from _shared import load_registry, load_catalog, find_pdf_path
"""

import json
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

    Returns the shortest non-Lampiran PDF filename under
    data/raw/<jenis_folder>/pdfs/ that matches the registry's nomor+tahun.
    Returns None if not found.
    """
    registry = load_registry()
    entry = registry.get(doc_id)
    if not entry:
        return None
    jenis = entry.get("jenis_folder", "")
    nomor = entry.get("nomor", "")
    tahun = entry.get("tahun", "")
    if not jenis:
        return None
    pdf_dir = DATA_DIR / "raw" / jenis / "pdfs"
    if not pdf_dir.exists():
        return None
    candidates = []
    for pdf in pdf_dir.glob("*.pdf"):
        if "Lampiran" in pdf.name:
            continue
        name_lower = pdf.name.lower()
        if nomor and tahun and (
            f"nomor {nomor} tahun {tahun}" in name_lower
            or f"no. {nomor} tahun {tahun}" in name_lower
            or f"no {nomor} tahun {tahun}" in name_lower
            or (nomor in pdf.name and tahun in pdf.name)
        ):
            candidates.append(pdf)
    if not candidates:
        return None
    return min(candidates, key=lambda p: len(p.name))


__all__ = ["load_registry", "load_catalog", "find_pdf_path", "REPO_ROOT", "DATA_DIR"]
