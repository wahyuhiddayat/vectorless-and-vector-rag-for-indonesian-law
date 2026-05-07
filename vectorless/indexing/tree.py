"""Tree-normalization helpers for parsed legal documents.

Functions in this module operate on the LLM parser's output tree
(`structure: list[dict]` with nested `nodes` children). They post-process
the structure into the canonical form that downstream consumers (re-split,
summary, retrieval) expect:

  - rebuild canonical pasal-family titles from node_id (handles OCR drift)
  - re-derive node_id from title when the LLM produces malformed ids
  - set navigation_path on every node
  - backfill start_index/end_index page numbers by matching titles to PDF
    page text

These helpers were originally embedded in `scripts/parser/llm_parse.py`.
Moved here so the parser orchestrator stays focused on LLM dispatch and
chunking, and so other entry points can reuse the helpers without
importing from a CLI script.
"""
from __future__ import annotations

import re
from typing import Iterable

_PASAL_TITLE_RE = re.compile(r"^Pasal\s+\d+[A-Z]?$")


def iter_nodes(structure: list[dict]) -> Iterable[dict]:
    """Yield every node in the tree (depth-first)."""
    for n in structure:
        yield n
        if n.get("nodes"):
            yield from iter_nodes(n["nodes"])


def collect_pasal_numbers(structure: list[dict]) -> list[str]:
    """Return list of Pasal numbers (arabic only) in tree order."""
    out: list[str] = []
    for n in iter_nodes(structure):
        title = (n.get("title") or "").strip()
        m = re.match(r"^Pasal\s+(\d+[A-Z]?)$", title)
        if m:
            out.append(m.group(1))
    return out


def _sanitize_node_id(s: str) -> str:
    """Lowercase + underscores only; preserve Roman pasal (I, II, III)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_")


def _canonical_title_from_node_id(node_id: str, original_title: str) -> str:
    """Rebuild canonical pasal-family title from node_id.

    node_id is LLM-consistent and structural; the title may carry verbatim OCR
    artifacts. For nested amendment ids (pasal_I_angka_N_pasal_3A), the deepest
    `pasal` segment wins, so the title reflects the nested Pasal, not the Roman
    container. BAB/Bagian/Paragraf titles pass through unchanged.
    """
    parts = node_id.split("_")
    last_pasal_idx = -1
    for i, seg in enumerate(parts):
        if seg == "pasal" and i + 1 < len(parts):
            last_pasal_idx = i
    if last_pasal_idx == -1:
        return original_title
    tail = parts[last_pasal_idx:]
    out: list[str] = []
    i = 0
    while i < len(tail):
        seg = tail[i]
        if seg == "pasal":
            if i + 1 < len(tail):
                num = tail[i + 1]
                if re.match(r"^[IVX]+$", num):
                    out.append(f"Pasal {num}")
                elif re.match(r"^\d+[A-Z]?$", num, re.IGNORECASE):
                    m = re.match(r"^(\d+)([A-Za-z]?)$", num)
                    if m:
                        n, suf = m.group(1), m.group(2).upper()
                        out.append(f"Pasal {n}{suf}")
                    else:
                        out.append(f"Pasal {num}")
                else:
                    out.append(f"Pasal {num}")
                i += 2
                continue
        elif seg == "ayat":
            if i + 1 < len(tail):
                out.append(f"Ayat ({tail[i + 1]})")
                i += 2
                continue
        elif seg == "huruf":
            if i + 1 < len(tail):
                out.append(f"Huruf {tail[i + 1]}")
                i += 2
                continue
        elif seg == "angka":
            if i + 1 < len(tail):
                out.append(f"Angka {tail[i + 1]}")
                i += 2
                continue
        i += 1
    canonical = " ".join(out)
    return canonical if canonical else original_title


def normalize_pasal_titles_in_tree(structure: list[dict]) -> int:
    """Rebuild pasal-family titles from node_id; return count modified."""
    count = 0
    for node in structure:
        title = node.get("title", "")
        nid = node.get("node_id", "")
        if nid and title.startswith("Pasal "):
            new = _canonical_title_from_node_id(nid, title)
            if new != title:
                node["title"] = new
                count += 1
        if node.get("nodes"):
            count += normalize_pasal_titles_in_tree(node["nodes"])
    return count


def _derive_node_id_from_title(title: str, ancestor_id: str) -> str:
    """Fallback derivation when LLM output is missing/malformed node_id."""
    t = title.lower()
    m = re.match(r"^pasal\s+([ivx]+)(?:\s+angka\s+(\d+))?(?:\s+pasal\s+(\d+[a-z]?))?", t)
    if m:
        parts = [f"pasal_{m.group(1).upper()}"]
        if m.group(2):
            parts.append(f"angka_{m.group(2)}")
        if m.group(3):
            parts.append(f"pasal_{m.group(3).upper()}")
        return "_".join(parts)
    m = re.match(r"^bab\s+([ivxlc]+)", t)
    if m:
        from .parser import roman_to_int
        n = roman_to_int(m.group(1).upper())
        return f"bab_{n if n else m.group(1)}"
    m = re.match(r"^bagian\s+(\w+)", t)
    if m:
        return f"{ancestor_id}_bagian_{m.group(1).lower()}" if ancestor_id else f"bagian_{m.group(1).lower()}"
    m = re.match(r"^paragraf\s+(\d+)", t)
    if m:
        return f"{ancestor_id}_paragraf_{m.group(1)}" if ancestor_id else f"paragraf_{m.group(1)}"
    m = re.match(
        r"^pasal\s+(\d+[a-z]?)(?:\s+ayat\s+\((\d+)\))?(?:\s+huruf\s+([a-z]+))?(?:\s+angka\s+(\d+))?",
        t,
    )
    if m:
        parts = [f"pasal_{m.group(1).upper() if m.group(1) and not m.group(1).isdigit() else m.group(1)}"]
        if m.group(2):
            parts.append(f"ayat_{m.group(2)}")
        if m.group(3):
            parts.append(f"huruf_{m.group(3)}")
        if m.group(4):
            parts.append(f"angka_{m.group(4)}")
        return "_".join(parts)
    return _sanitize_node_id(title)[:60].lower() or "node"


def assign_readable_node_ids(structure: list[dict], ancestor_id: str = "") -> None:
    """Keep conforming node_ids; re-derive from title + ancestor when missing or malformed."""
    for node in structure:
        title = (node.get("title") or "").strip()
        nid = (node.get("node_id") or "").strip()
        if not nid or not re.match(r"^[a-zA-Z0-9_]+$", nid):
            nid = _derive_node_id_from_title(title, ancestor_id)
        node["node_id"] = nid
        if node.get("nodes"):
            assign_readable_node_ids(node["nodes"], ancestor_id=nid)


def build_navigation_paths(structure: list[dict], ancestors: list[str] | None = None) -> None:
    """Set navigation_path on every node using title ancestry."""
    ancestors = ancestors or []
    for node in structure:
        title = (node.get("title") or "").strip()
        path = ancestors + [title]
        node["navigation_path"] = " > ".join(path)
        if node.get("nodes"):
            build_navigation_paths(node["nodes"], path)


def backfill_page_indices(
    structure: list[dict], pages: list[dict], doc_total_pages: int
) -> None:
    """Derive start_index / end_index per node by matching title or text against
    PDF page raw_text. Missing matches default to the surrounding sibling range
    (or 1..doc_total_pages for the outermost frame)."""
    page_text = {p["page_num"]: p.get("raw_text", "") for p in pages}

    def _find_title_page(title: str) -> int | None:
        if not title:
            return None
        needle_norm = re.sub(r"\s+", " ", title.strip())
        for n in sorted(page_text):
            page_norm = re.sub(r"\s+", " ", page_text[n])
            if needle_norm and needle_norm[:40] in page_norm:
                return n
        return None

    def _assign(node: dict, default_start: int = 1, default_end: int = doc_total_pages) -> None:
        title = node.get("title", "")
        start = _find_title_page(title) or default_start
        node["start_index"] = start
        children = node.get("nodes", []) or []
        if children:
            for i, child in enumerate(children):
                next_sib = children[i + 1] if i + 1 < len(children) else None
                next_start = _find_title_page(next_sib.get("title", "")) if next_sib else None
                child_default_end = (next_start - 1) if next_start else default_end
                _assign(child, default_start=start, default_end=child_default_end)
            node["end_index"] = children[-1].get("end_index", default_end)
        else:
            node["end_index"] = default_end

    for i, top in enumerate(structure):
        next_top = structure[i + 1] if i + 1 < len(structure) else None
        next_start = _find_title_page(next_top.get("title", "")) if next_top else None
        default_end = (next_start - 1) if next_start else doc_total_pages
        _assign(top, default_start=1, default_end=default_end)
