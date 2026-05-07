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

GRANULARITIES = ("pasal", "ayat", "rincian")


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


# find_pdf_path moved to vectorless/indexing/pdf.py (single source of truth).
# Re-exported here so scripts that already import from `scripts._shared`
# continue to work without modification.
from vectorless.indexing.pdf import find_pdf_path  # noqa: E402,F401


__all__ = ["load_registry", "load_catalog", "find_pdf_path", "REPO_ROOT", "DATA_DIR"]
