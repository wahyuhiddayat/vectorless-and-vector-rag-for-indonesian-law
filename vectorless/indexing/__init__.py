"""Indexing pipeline package.

Public surface for shared constants. Per-stage modules (llm_parse, ocr,
summary, judge, corpus_status, status, verify, build) import the canonical
values from here so we never have two diverging copies of the same map or
version string.
"""
from pathlib import Path

GRANULARITY_INDEX_MAP: dict[str, Path] = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "rincian": Path("data/index_rincian"),
}

GRANULARITIES: tuple[str, ...] = tuple(GRANULARITY_INDEX_MAP.keys())

# Bumped manually when a parser change invalidates prior outputs.
PARSER_VERSION = "2026-04-02"
LLM_CLEANUP_VERSION = "2026-04-02"
