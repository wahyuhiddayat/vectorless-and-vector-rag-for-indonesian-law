"""Rewrite node_ids from legacy "0001" format to readable "pasal_N" format.

Parser-produced docs have pasal-level node_ids like "0001", "0002". The new
LLM-parse script produces "pasal_1", "pasal_2", etc. Mixing both across the
corpus creates downstream inconsistency (GT roll-up, debugging, retrieval
citations). This script normalizes the legacy docs in-place.

Walks each doc's structure. For any node with a Pasal-shaped title
("Pasal 3", "Pasal 5A"), rewrite its node_id to "pasal_<number>". Does NOT
touch BAB/Bagian/Paragraf (parser's `0` etc. IDs are fine for containers).
Re-split indexes (ayat, full_split) need to be regenerated after this so
child node_ids inherit the new parent suffix.

Usage:
    python scripts/parser/rewrite_node_ids.py --dry-run     # preview counts
    python scripts/parser/rewrite_node_ids.py               # rewrite all pasal docs
    python scripts/parser/rewrite_node_ids.py --category OJK
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"

_PASAL_TITLE_RE = re.compile(r"^Pasal\s+(\d+[A-Z]?)$")
_BAB_TITLE_RE = re.compile(r"^BAB\s+", re.IGNORECASE)
_BAGIAN_TITLE_RE = re.compile(r"^Bagian\s+", re.IGNORECASE)
_PARAGRAF_TITLE_RE = re.compile(r"^Paragraf\s+", re.IGNORECASE)


def _container_id_from_title(title: str, arabic_index: int) -> str | None:
    """Return readable container id or None (keep existing)."""
    t = title.strip()
    if _BAB_TITLE_RE.match(t):
        # Extract Roman→arabic if present, else use sequence.
        m = re.match(r"BAB\s+([IVXLC]+)", t, re.IGNORECASE)
        if m:
            from vectorless.indexing.parser import roman_to_int
            n = roman_to_int(m.group(1).upper())
            return f"bab_{n}" if n else f"bab_{arabic_index}"
        return f"bab_{arabic_index}"
    if _BAGIAN_TITLE_RE.match(t):
        m = re.match(r"Bagian\s+(\w+)", t, re.IGNORECASE)
        if m:
            return f"bagian_{m.group(1).lower()}"
    if _PARAGRAF_TITLE_RE.match(t):
        m = re.match(r"Paragraf\s+(\d+)", t, re.IGNORECASE)
        if m:
            return f"paragraf_{m.group(1)}"
    return None


def rewrite_tree(
    nodes: list[dict], ancestor_id: str = "", bab_counter: list[int] | None = None
) -> int:
    """Rewrite node_ids in tree to readable format. Returns count rewritten."""
    bab_counter = bab_counter or [0]
    n_changed = 0
    for node in nodes:
        title = (node.get("title") or "").strip()
        old_id = node.get("node_id", "")
        new_id: str | None = None

        m = _PASAL_TITLE_RE.match(title)
        if m:
            num = m.group(1)
            # Amendment Pasal Roman handled separately — if title is "Pasal I/II/...",
            # keep roman in id.
            if re.match(r"^[IVX]+$", num, re.IGNORECASE):
                new_id = f"pasal_{num.upper()}"
            else:
                new_id = f"pasal_{num}"
        else:
            if _BAB_TITLE_RE.match(title):
                bab_counter[0] += 1
            container_id = _container_id_from_title(title, bab_counter[0])
            if container_id:
                new_id = (
                    f"{ancestor_id}_{container_id}" if ancestor_id else container_id
                )

        if new_id and new_id != old_id:
            node["node_id"] = new_id
            n_changed += 1

        if node.get("nodes"):
            n_changed += rewrite_tree(
                node["nodes"], ancestor_id=node["node_id"], bab_counter=bab_counter
            )
    return n_changed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", help="Only process this category (e.g. OJK)")
    ap.add_argument("--doc-id", help="Only process this doc_id")
    ap.add_argument("--dry-run", action="store_true", help="Preview, do not write")
    args = ap.parse_args()

    pattern = str(INDEX_PASAL / "*" / "*.json")
    files = sorted(glob.glob(pattern))
    if args.category:
        files = [f for f in files if Path(f).parent.name == args.category]
    if args.doc_id:
        files = [f for f in files if Path(f).stem == args.doc_id]
    files = [f for f in files if Path(f).name != "catalog.json"]

    print(f"Scanning {len(files)} doc files")
    total_changed = 0
    docs_changed = 0
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        structure = d.get("structure", [])
        changed = rewrite_tree(structure)
        if changed:
            docs_changed += 1
            total_changed += changed
            if not args.dry_run:
                with open(f, "w", encoding="utf-8") as fh:
                    json.dump(d, fh, ensure_ascii=False, indent=2)
            print(f"  {Path(f).name}: {changed} node_ids rewritten")
    print(f"\nTotal: {docs_changed} docs, {total_changed} node_ids rewritten")
    if args.dry_run:
        print("(dry-run — no files written)")
    else:
        print("Re-split ayat + full_split now to propagate new suffixes.")


if __name__ == "__main__":
    main()
