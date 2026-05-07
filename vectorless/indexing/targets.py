"""Resolve a list of doc_ids from CLI arguments and the registry.

Every CLI entry point in the indexing pipeline takes either explicit
`--doc-id X` flags or a `--category Y` filter. This module owns that
resolution so the four CLIs (`vectorless.indexing.build`, plus the
parser, judge, and granularity-check scripts) share one implementation.
"""
from __future__ import annotations

import json
from pathlib import Path

REGISTRY_PATH = Path("data/raw/registry.json")


def resolve_targets(doc_ids: list[str], category: str | None) -> list[str]:
    """Resolve doc_id targets from explicit IDs or a category filter.

    Explicit doc_ids win when present. Otherwise the registry is loaded
    and filtered by `jenis_folder == category` (case-insensitive).

    Args:
        doc_ids: doc_ids passed via repeatable `--doc-id` flags.
        category: jenis_folder filter from `--category`.

    Returns:
        Sorted list of doc_ids. Explicit input is returned in order.

    Raises:
        SystemExit: when neither doc_ids nor category is provided.
        FileNotFoundError: when category filter is requested but the
            registry file is missing.
    """
    if doc_ids:
        return list(doc_ids)
    if not category:
        raise SystemExit("must pass --doc-id(s) or --category")
    if not REGISTRY_PATH.exists():
        raise FileNotFoundError(f"registry not found at {REGISTRY_PATH}")
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    target = category.upper()
    return sorted(
        doc_id
        for doc_id, entry in registry.items()
        if (entry.get("jenis_folder") or "").upper() == target
    )
