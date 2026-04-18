"""
Compare the current pasal index against a fresh re-parse (no LLM).

Usage:
    python scripts/diff_index.py --doc-id pp-3-2026
    python scripts/diff_index.py --category PP
    python scripts/diff_index.py --all
    python scripts/diff_index.py --doc-id pp-3-2026 --verbose

Output: for each doc, prints element_count diffs, added/removed/changed nodes.
Only prints docs that have differences (unless --verbose).
Exit code 0 = no diffs, 1 = diffs found.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Project root is one level up from scripts/
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_RAW = ROOT / "data" / "raw"
DATA_INDEX_PASAL = ROOT / "data" / "index_pasal"

from vectorless.indexing.build import _load_pdf_for_doc, pick_main_pdf
from vectorless.indexing.parser import parse_legal_pdf


# ---------------------------------------------------------------------------
# Tree utilities
# ---------------------------------------------------------------------------

def flatten_tree(nodes: list, parent_id: str = "", depth: int = 0) -> dict:
    """Flatten a nested node tree into {node_id: info_dict}.

    Each info_dict has: title, parent_id, depth, n_children, text_len.
    """
    result = {}
    for node in nodes:
        nid = node.get("node_id", "?")
        result[nid] = {
            "title": node.get("title", ""),
            "parent_id": parent_id,
            "depth": depth,
            "n_children": len(node.get("nodes", [])),
            "text_len": len(node.get("text", "")),
        }
        result.update(flatten_tree(node.get("nodes", []), parent_id=nid, depth=depth + 1))
    return result


def diff_structures(old: dict, new: dict) -> dict:
    """Compute structural diff between two flattened trees.

    Returns dict with keys: added, removed, changed.
    Each is a list of (node_id, info) or (node_id, old_info, new_info).
    """
    old_ids = set(old)
    new_ids = set(new)

    added = [(nid, new[nid]) for nid in sorted(new_ids - old_ids)]
    removed = [(nid, old[nid]) for nid in sorted(old_ids - new_ids)]

    changed = []
    for nid in sorted(old_ids & new_ids):
        o, n = old[nid], new[nid]
        changes = {}
        if o["title"] != n["title"]:
            changes["title"] = (o["title"], n["title"])
        if o["parent_id"] != n["parent_id"]:
            changes["parent_id"] = (o["parent_id"], n["parent_id"])
        if o["n_children"] != n["n_children"]:
            changes["n_children"] = (o["n_children"], n["n_children"])
        # Text length change > 20% is worth flagging (LLM vs raw OCR differ slightly;
        # large changes suggest content was gained or lost, not just cleaned).
        if o["text_len"] > 0 and n["text_len"] > 0:
            ratio = abs(o["text_len"] - n["text_len"]) / max(o["text_len"], n["text_len"])
            if ratio > 0.2:
                changes["text_len"] = (o["text_len"], n["text_len"])
        elif o["text_len"] == 0 and n["text_len"] > 100:
            changes["text_len"] = (0, n["text_len"])
        elif n["text_len"] == 0 and o["text_len"] > 100:
            changes["text_len"] = (o["text_len"], 0)
        if changes:
            changed.append((nid, o, n, changes))

    return {"added": added, "removed": removed, "changed": changed}


def diff_element_counts(old_counts: dict, new_counts: dict) -> dict:
    """Return keys whose values differ between old and new element_counts."""
    all_keys = set(old_counts) | set(new_counts)
    return {k: (old_counts.get(k, 0), new_counts.get(k, 0))
            for k in sorted(all_keys)
            if old_counts.get(k, 0) != new_counts.get(k, 0)}


# ---------------------------------------------------------------------------
# Per-doc diff
# ---------------------------------------------------------------------------

def diff_doc(doc_id: str, registry: dict, verbose: bool = False) -> bool:
    """Re-parse one doc and compare against current index. Returns True if diff found."""
    entry = registry.get(doc_id)
    if not entry:
        print(f"[SKIP] {doc_id}: not in registry")
        return False

    # Load existing index
    jenis_folder = entry.get("jenis_folder", doc_id.split("-")[0].upper())
    index_path = DATA_INDEX_PASAL / jenis_folder / f"{doc_id}.json"
    if not index_path.exists():
        print(f"[SKIP] {doc_id}: not in index ({index_path})")
        return False

    with open(index_path, encoding="utf-8") as f:
        old_doc = json.load(f)

    # Find the PDF
    metadata_path = DATA_RAW / jenis_folder / "metadata" / f"{doc_id}.json"
    metadata = None
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)

    pdf_path, err = _load_pdf_for_doc(entry, metadata, DATA_INDEX_PASAL)
    if err:
        # Fallback: glob for PDFs matching nomor+tahun.
        # Match nomor as exact last integer in the stem (handles zero-padding like "06" for nomor "6")
        # and tahun as substring. Simple substring on nomor is wrong — "6" matches "2026", "16", etc.
        nomor = entry.get("nomor", "")
        tahun = entry.get("tahun", "")
        pdfs_dir = DATA_RAW / jenis_folder / "pdfs"
        candidates = []
        for p in pdfs_dir.glob("*.pdf"):
            if "Penjelasan" in p.name or "Lampiran" in p.name:
                continue
            if tahun not in p.name:
                continue
            # Nomor must match as the last integer in the stem (possibly zero-padded).
            m = re.search(r'(\d+)$', p.stem)
            if m and int(m.group(1)) == int(nomor):
                candidates.append(p)
                continue
            # Also accept explicit word-boundary match (e.g. "Nomor 6 Tahun").
            if re.search(rf'(?<!\d){re.escape(nomor)}(?!\d)', p.name):
                candidates.append(p)
        if candidates:
            pdf_path = candidates[0]
        else:
            print(f"[SKIP] {doc_id}: {err}")
            return False

    # Re-parse with current parser (no LLM, no disk write)
    try:
        new_doc = parse_legal_pdf(str(pdf_path), verbose=False, use_llm_cleanup=False)
    except Exception as e:
        print(f"[ERROR] {doc_id}: parse failed — {e}")
        return False

    # Compare element_counts
    old_counts = old_doc.get("element_counts", {})
    new_counts = new_doc.get("element_counts", {})
    count_diff = diff_element_counts(old_counts, new_counts)

    # Compare structure
    old_flat = flatten_tree(old_doc.get("structure", []))
    new_flat = flatten_tree(new_doc.get("structure", []))
    struct_diff = diff_structures(old_flat, new_flat)

    has_diff = bool(count_diff or struct_diff["added"] or
                    struct_diff["removed"] or struct_diff["changed"])

    if not has_diff:
        if verbose:
            print(f"[OK]   {doc_id}: no structural changes")
        return False

    # Print diffs
    print(f"\n{'='*60}")
    print(f"DIFF: {doc_id}  ({jenis_folder})")
    print(f"{'='*60}")

    if count_diff:
        print("  element_counts:")
        for k, (ov, nv) in count_diff.items():
            arrow = "^" if nv > ov else "v"
            print(f"    {k}: {ov} -> {nv}  {arrow}")

    added = struct_diff["added"]
    removed = struct_diff["removed"]
    changed = struct_diff["changed"]

    if added:
        print(f"  ADDED ({len(added)} nodes):")
        for nid, info in added[:20]:
            print(f"    + {nid:12s} depth={info['depth']}  '{info['title'][:55]}'")
        if len(added) > 20:
            print(f"    ... and {len(added) - 20} more")

    if removed:
        print(f"  REMOVED ({len(removed)} nodes):")
        for nid, info in removed[:20]:
            print(f"    - {nid:12s} depth={info['depth']}  '{info['title'][:55]}'")
        if len(removed) > 20:
            print(f"    ... and {len(removed) - 20} more")

    if changed:
        print(f"  CHANGED ({len(changed)} nodes):")
        for nid, _, _, changes in changed[:20]:
            changes_str = []
            for field, (ov, nv) in changes.items():
                if field == "title":
                    changes_str.append(
                        f"title: '{str(ov)[:30]}' -> '{str(nv)[:30]}'"
                    )
                elif field == "text_len":
                    changes_str.append(f"text_len: {ov} -> {nv}")
                else:
                    changes_str.append(f"{field}: {ov} -> {nv}")
            print(f"    ~ {nid:12s}  {' | '.join(changes_str)}")
        if len(changed) > 20:
            print(f"    ... and {len(changed) - 20} more")

    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_registry() -> dict:
    """Load doc registry."""
    with open(DATA_RAW / "registry.json", encoding="utf-8") as f:
        return json.load(f)


def get_indexed_doc_ids(category: str | None = None) -> list[str]:
    """Return all doc_ids currently in the pasal index, optionally filtered by category."""
    doc_ids = []
    for jenis_dir in DATA_INDEX_PASAL.iterdir():
        if not jenis_dir.is_dir():
            continue
        if category and jenis_dir.name.upper() != category.upper():
            continue
        for f in sorted(jenis_dir.glob("*.json")):
            doc_ids.append(f.stem)
    return sorted(doc_ids)


def main():
    """Parse arguments and run the diff."""
    parser = argparse.ArgumentParser(description="Diff pasal index against fresh re-parse.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--doc-id", help="Single document ID to diff")
    group.add_argument("--category", help="Diff all docs in a category (e.g. PP, UU)")
    group.add_argument("--all", action="store_true", help="Diff all indexed documents")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print OK status for unchanged docs too")
    args = parser.parse_args()

    registry = load_registry()

    if args.doc_id:
        doc_ids = [args.doc_id]
    elif args.category:
        doc_ids = get_indexed_doc_ids(args.category)
        if not doc_ids:
            print(f"No indexed docs found for category '{args.category}'")
            sys.exit(0)
    else:
        doc_ids = get_indexed_doc_ids()

    print(f"Diffing {len(doc_ids)} document(s) against current parser (no LLM)...")

    any_diff = False
    for doc_id in doc_ids:
        had_diff = diff_doc(doc_id, registry, verbose=args.verbose)
        if had_diff:
            any_diff = True

    print()
    if any_diff:
        print("Result: DIFFS FOUND — see above for details.")
        sys.exit(1)
    else:
        print("Result: no structural changes detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
