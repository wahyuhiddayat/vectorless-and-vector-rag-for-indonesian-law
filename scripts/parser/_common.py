"""Shared utilities for parser scripts (LLM calls, PDF loading, JSON parsing).

Extracted from scripts/parser/llm_fix.py (archived) so scripts/parser/llm_parse.py
can import without depending on the legacy llm_fix module.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vectorless.indexing.parser import extract_pages  # noqa: E402
from scripts._shared import find_pdf_path  # noqa: E402


_PASAL_TITLE_RE = re.compile(r"^Pasal\s+\d+[A-Z]?$")


def load_pdf_text(doc_id: str) -> str:
    """Read the document's main PDF and concatenate all pages (flat text)."""
    pdf_path = find_pdf_path(doc_id)
    if not pdf_path:
        raise FileNotFoundError(f"PDF not found for {doc_id}")
    pages = extract_pages(str(pdf_path))
    return "\n\n".join(p.get("raw_text", "") for p in pages)


def load_pdf_pages(doc_id: str) -> list[dict]:
    """Read the PDF and return list of {page_num, blocks} per page.

    Each block: {x0, y0, text}. Blocks sorted (y, x) for reading order.
    Use format_pdf_pages() to render a page range as prompt text.
    """
    import pymupdf

    pdf_path = find_pdf_path(doc_id)
    if not pdf_path:
        raise FileNotFoundError(f"PDF not found for {doc_id}")

    pages: list[dict] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page_i, page in enumerate(doc, 1):
            blocks = []
            for b in page.get_text("dict").get("blocks", []):
                if b.get("type") != 0:
                    continue
                text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        text += span["text"]
                    text += "\n"
                text = text.strip()
                if not text:
                    continue
                bbox = b["bbox"]
                blocks.append({
                    "x0": round(bbox[0]),
                    "y0": round(bbox[1]),
                    "text": text,
                })
            blocks.sort(key=lambda b: (b["y0"], b["x0"]))
            pages.append({"page_num": page_i, "blocks": blocks})
    return pages


def format_pdf_pages(
    pages: list[dict],
    start_page: int | None = None,
    end_page: int | None = None,
) -> str:
    """Render pages as '=== PAGE N ===' sections with [x=y=] tagged blocks.

    If start_page/end_page given, only include pages in that inclusive range
    (1-indexed). Used to scope PDF context per chunk for efficiency.
    """
    out: list[str] = []
    for p in pages:
        n = p["page_num"]
        if start_page is not None and n < start_page:
            continue
        if end_page is not None and n > end_page:
            continue
        out.append(f"=== PAGE {n} ===")
        for b in p["blocks"]:
            out.append(f"[x={b['x0']} y={b['y0']}] {b['text']}")
        out.append("")
    return "\n".join(out)


def parse_llm_json(raw: str) -> dict:
    """Strip markdown fences and parse as JSON, normalizing key aliases."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    obj = json.loads(cleaned)
    _normalize_keys(obj)
    return obj


def _normalize_keys(obj) -> None:
    """Rename 'name' -> 'title' recursively (LLM sometimes uses 'name')."""
    if isinstance(obj, dict):
        if "name" in obj and "title" not in obj:
            obj["title"] = obj.pop("name")
        for v in obj.values():
            _normalize_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            _normalize_keys(item)


def count_pasals_in_tree(structure: list[dict]) -> int:
    """Count nodes whose title is exactly a Pasal heading (e.g. 'Pasal 12', 'Pasal 5A').

    Uses a strict regex to avoid false positives from titles like 'Pasal 1 1.'.
    """
    count = 0
    for node in structure:
        title = (node.get("title") or "").strip()
        if _PASAL_TITLE_RE.match(title):
            count += 1
        if "nodes" in node:
            count += count_pasals_in_tree(node["nodes"])
    return count
