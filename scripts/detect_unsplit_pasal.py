"""Detect docs with embedded (unsplit) Pasal headings in index_pasal leaves.

Scans every leaf node's `text` field for a `\\nPasal N\\n<letter>` pattern, which
indicates the parser failed to detect and split a Pasal heading — the content of
the missed Pasal got merged into the preceding Pasal's body.

Amendment body nodes (title starts with `Angka N — `) are skipped because they
legitimately contain cross-references like `Pasal 117 berbunyi ...`.

Cross-references doc IDs against `data/ground_truth.json` so affected docs that
already have GT entries can be regenerated.

Usage:
    python scripts/detect_unsplit_pasal.py
"""

import json
import re
from pathlib import Path

INDEX_PASAL = Path("data/index_pasal")
GROUND_TRUTH = Path("data/ground_truth.json")

EMBEDDED_PASAL_RE = re.compile(r"\nPasal\s+\d+[A-Z]?\s*\n\s*[a-zA-Z]")


def walk_leaves(nodes):
    """Yield every leaf node in the structure."""
    for node in nodes:
        if "nodes" in node:
            yield from walk_leaves(node["nodes"])
        else:
            yield node


def scan_doc(doc_path: Path) -> list[tuple[str, str, list[str]]]:
    """Return list of (node_id, title_prefix, sample_matches) for suspicious leaves."""
    idx = json.load(open(doc_path, encoding="utf-8"))
    hits = []
    for leaf in walk_leaves(idx["structure"]):
        title = leaf.get("title", "")
        if title.startswith("Angka "):
            continue
        text = leaf.get("text", "")
        matches = EMBEDDED_PASAL_RE.findall(text)
        if matches:
            hits.append((leaf["node_id"], title[:40], matches[:3]))
    return hits


def main() -> None:
    """Run detection across all indexed docs and print report."""
    affected: dict[str, list] = {}
    for cat_dir in INDEX_PASAL.iterdir():
        if not cat_dir.is_dir():
            continue
        for doc_path in cat_dir.glob("*.json"):
            if doc_path.name == "catalog.json":
                continue
            hits = scan_doc(doc_path)
            if hits:
                affected[doc_path.stem] = hits

    gt_docs: set[str] = set()
    if GROUND_TRUTH.exists():
        gt = json.load(open(GROUND_TRUTH, encoding="utf-8"))
        gt_docs = {v["gold_doc_id"] for v in gt.values()}

    in_gt = sorted(d for d in affected if d in gt_docs)
    not_gt = sorted(d for d in affected if d not in gt_docs)

    print(f"=== {len(affected)} docs with likely unsplit Pasal ===\n")
    print(f"--- IN GT ({len(in_gt)}) — requires GT cleanup before rebuild ---")
    for doc in in_gt:
        print(f"{doc}: {len(affected[doc])} suspicious leaves")
        for nid, title, matches in affected[doc][:2]:
            print(f"  {nid} ({title}): {matches}")

    print(f"\n--- NOT IN GT ({len(not_gt)}) — rebuild before GT annotation ---")
    for doc in not_gt:
        print(f"{doc}: {len(affected[doc])} suspicious leaves")
        for nid, title, matches in affected[doc][:2]:
            print(f"  {nid} ({title}): {matches}")


if __name__ == "__main__":
    main()
