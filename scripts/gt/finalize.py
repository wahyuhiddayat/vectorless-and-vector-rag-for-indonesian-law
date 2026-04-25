"""
Ground Truth Finalizer.

Converts data/ground_truth.json (leaf-anchored annotations produced by
gt_collect.py) into data/validated_testset.pkl, which stores reference_mode
plus gold node ID sets for all three index granularities: pasal, ayat, and
rincian.

DESIGN: Anchor at finest granularity, roll UP.

Each GT question anchors at the most specific leaf in the rincian index
(e.g., a specific huruf or angka). Gold sets for coarser granularities are
derived by finding the parent node via prefix lookup:

  rincian: gold = {anchor}                          (exact, 1 node)
  ayat:       gold = {derive_parent(anchor, ayat)}     (parent, 1 node)
  pasal:      gold = {derive_parent(anchor, pasal)}    (parent, 1 node)

Every granularity has exactly 1 gold node per question, ensuring fair
evaluation: harder at fine granularity (larger corpus, same target).

EVALUATION USAGE (in your retrieval eval script):
  if granularity == "pasal":
      hit = retrieved_node_id in item["gold_pasal_node_ids"]
  elif granularity == "ayat":
      hit = retrieved_node_id in item["gold_ayat_node_ids"]
  elif granularity == "rincian":
      hit = retrieved_node_id in item["gold_rincian_node_ids"]

Usage:
    python scripts/gt/finalize.py
    python scripts/gt/finalize.py --check
    python scripts/gt/finalize.py --stats
"""

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GT_FILE = Path("data/ground_truth.json")
TESTSET_FILE = Path("data/validated_testset.pkl")
INDEX_PASAL = Path("data/index_pasal")
INDEX_AYAT = Path("data/index_ayat")
INDEX_FULL_SPLIT = Path("data/index_rincian")
VALID_REFERENCE_MODES = {"none", "legal_ref", "doc_only", "both"}
LEGAL_REFERENCE_RE = re.compile(r"\b(pasal|ayat|huruf|angka)\b", re.IGNORECASE)
DOC_REFERENCE_RE = re.compile(
    r"\b("
    r"peraturan pemerintah pengganti undang-?undang|perpu|"
    r"undang-?undang|uu|"
    r"peraturan pemerintah|pp|"
    r"peraturan presiden|perpres|"
    r"peraturan menteri(?:\s+[a-z][a-z-]*){0,4}|"
    r"pmk|permen[a-z-]+"
    r")\b",
    re.IGNORECASE,
)

# Cache to avoid re-reading the same index JSON multiple times
_doc_cache: dict[str, dict] = {}


def find_doc_path(doc_id: str, index_dir: Path) -> Path | None:
    """Find the index JSON file for a given doc_id within an index directory."""
    for path in index_dir.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path
    return None


def load_doc(doc_id: str, index_dir: Path) -> dict | None:
    """Load a document from a specific index directory (cached)."""
    cache_key = f"{index_dir}/{doc_id}"
    if cache_key in _doc_cache:
        return _doc_cache[cache_key]

    path = find_doc_path(doc_id, index_dir)
    if not path:
        return None

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    _doc_cache[cache_key] = doc
    return doc


def collect_leaf_ids(nodes: list[dict], results: set[str] | None = None) -> set[str]:
    """Recursively collect all leaf node IDs (nodes with text content)."""
    if results is None:
        results = set()
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            collect_leaf_ids(node["nodes"], results)
        elif node.get("text"):
            results.add(node["node_id"])
    return results


def get_leaf_ids(doc_id: str, index_dir: Path) -> set[str]:
    """Return all leaf node IDs for a document in a specific index."""
    doc = load_doc(doc_id, index_dir)
    if doc is None:
        return set()
    return collect_leaf_ids(doc["structure"])


def get_anchor_node_id(item: dict) -> str:
    """Return the leaf anchor node ID from a merged GT item."""
    return item.get("gold_anchor_node_id") or item["gold_node_id"]


def get_anchor_granularity(item: dict) -> str:
    """Return the anchor granularity from a merged GT item."""
    return item.get("gold_anchor_granularity", "rincian")


def infer_reference_mode(query: str) -> str:
    """Infer reference_mode from query text for backward compatibility."""
    query = query or ""
    has_legal = bool(LEGAL_REFERENCE_RE.search(query))
    has_doc = bool(DOC_REFERENCE_RE.search(query))

    if has_legal and has_doc:
        return "both"
    if has_legal:
        return "legal_ref"
    if has_doc:
        return "doc_only"
    return "none"


def derive_parent_node_id(anchor: str, doc_id: str, target_index: Path) -> str:
    """Roll an anchor up to its nearest leaf in `target_index` via prefix lookup.

    Walks progressively shorter prefixes; if no prefix matches, retries with the
    `_p` Pasal-container segments stripped (rincian uses them, ayat/pasal don't).
    """
    target_leaves = get_leaf_ids(doc_id, target_index)
    parts = anchor.split("_")
    for length in range(len(parts), 0, -1):
        candidate = "_".join(parts[:length])
        if candidate in target_leaves:
            return candidate
    stripped_parts = [p for p in parts if p != "p"]
    if stripped_parts != parts:
        for length in range(len(stripped_parts), 0, -1):
            candidate = "_".join(stripped_parts[:length])
            if candidate in target_leaves:
                return candidate
    raise ValueError(
        f"Cannot derive parent node from '{anchor}' in {target_index} for doc '{doc_id}'"
    )


def finalize(check_only: bool = False) -> dict:
    """
    Load ground_truth.json, expand each item with multi-granularity gold sets,
    and write to validated_testset.pkl (unless check_only=True).

    Returns the finalized testset dict.
    """
    if not GT_FILE.exists():
        print(f"ERROR: {GT_FILE} not found.")
        print("Run scripts/gt/collect.py first to produce ground_truth.json.")
        sys.exit(1)

    with open(GT_FILE, encoding="utf-8") as f:
        gt = json.load(f)

    if not gt:
        print("ERROR: ground_truth.json is empty.")
        sys.exit(1)

    print(f"\nLoaded {len(gt)} items from {GT_FILE}")
    print("Deriving gold sets: rincian (exact) -> ayat (parent) -> pasal (parent) ...")

    testset: dict[str, dict] = {}
    errors: list[str] = []

    for qid, item in gt.items():
        doc_id = item["gold_doc_id"]
        anchor_granularity = get_anchor_granularity(item)
        anchor_node_id = get_anchor_node_id(item)
        reference_mode = item.get("reference_mode") or infer_reference_mode(item.get("query", ""))

        if anchor_granularity != "rincian":
            errors.append(
                f"{qid}: gold_anchor_granularity must be 'rincian', got '{anchor_granularity}'"
            )
            continue

        if reference_mode not in VALID_REFERENCE_MODES:
            errors.append(
                f"{qid}: reference_mode must be one of {sorted(VALID_REFERENCE_MODES)}, "
                f"got '{reference_mode}'"
            )
            continue

        full_leaf_ids = get_leaf_ids(doc_id, INDEX_FULL_SPLIT)
        if not full_leaf_ids:
            errors.append(f"{qid}: doc '{doc_id}' not found in {INDEX_FULL_SPLIT}")
            continue

        if anchor_node_id not in full_leaf_ids:
            errors.append(
                f"{qid}: anchor node '{anchor_node_id}' is not a leaf in {INDEX_FULL_SPLIT}; "
                "the GT may use old ayat-anchored annotations — regenerate from rincian index"
            )
            continue

        # Roll UP: derive parent nodes at coarser granularities
        gold_full = {anchor_node_id}

        try:
            ayat_node_id = derive_parent_node_id(anchor_node_id, doc_id, INDEX_AYAT)
        except ValueError as e:
            errors.append(f"{qid}: {e}")
            continue
        gold_ayat = {ayat_node_id}

        try:
            pasal_node_id = derive_parent_node_id(anchor_node_id, doc_id, INDEX_PASAL)
        except ValueError as e:
            errors.append(f"{qid}: {e}")
            continue
        gold_pasal = {pasal_node_id}

        testset[qid] = {
            "query": item["query"],
            "query_style": item.get("query_style", ""),
            "difficulty": item.get("difficulty", ""),
            "reference_mode": reference_mode,
            "gold_doc_id": doc_id,
            "gold_anchor_granularity": "rincian",
            "gold_anchor_node_id": anchor_node_id,
            # Backward-compatible singular field used by load_testset.py
            "gold_node_id": anchor_node_id,
            "gold_pasal_node_ids": gold_pasal,
            "gold_ayat_node_ids": gold_ayat,
            "gold_rincian_node_ids": gold_full,
            "navigation_path": item.get("navigation_path", ""),
            "answer_hint": item.get("answer_hint", ""),
        }

    if errors:
        print(f"\n{len(errors)} hard errors:")
        for err in errors:
            print(f"  ✗ {err}")
        print("\nPerbaiki errors di atas sebelum melanjutkan.")
        sys.exit(1)

    n = len(testset)

    print("\nGold set sizes (per query):")
    print("  rincian : 1 node  (exact anchor — finest granularity)")
    print("  ayat       : 1 node  (derived parent in ayat index)")
    print("  pasal      : 1 node  (derived parent in pasal index)")

    if check_only:
        print("\n[check-only] Tidak menulis output.")
        return testset

    TESTSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TESTSET_FILE, "wb") as f:
        pickle.dump(testset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Disimpan ke {TESTSET_FILE}")
    print(f"  {len(testset)} queries siap dipakai untuk evaluasi 3 granularity")
    print("\nNext step:")
    print("  python scripts/gt/load_testset.py  # inspect result")

    return testset


def print_stats() -> None:
    """Print statistics about an existing validated_testset.pkl."""
    if not TESTSET_FILE.exists():
        print(f"ERROR: {TESTSET_FILE} not found.")
        print("Run scripts/gt/finalize.py first.")
        sys.exit(1)

    with open(TESTSET_FILE, "rb") as f:
        testset = pickle.load(f)

    total = len(testset)
    doc_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}
    anchor_counts: dict[str, int] = {}
    reference_mode_counts: dict[str, int] = {}

    for item in testset.values():
        doc_id = item["gold_doc_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        style = item.get("query_style") or "(missing)"
        style_counts[style] = style_counts.get(style, 0) + 1

        diff = item.get("difficulty") or "(missing)"
        diff_counts[diff] = diff_counts.get(diff, 0) + 1

        anchor = item.get("gold_anchor_granularity") or "(missing)"
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

        ref_mode = item.get("reference_mode") or "(missing)"
        reference_mode_counts[ref_mode] = reference_mode_counts.get(ref_mode, 0) + 1

        # All gold sets are now exactly size 1 (roll-up design)

    print(f"\nValidated testset stats: {TESTSET_FILE}")
    print(f"  Total queries     : {total}")
    print(f"  Documents covered : {len(doc_counts)}")

    print("\n  Anchor granularity distribution:")
    for anchor in sorted(anchor_counts.keys()):
        count = anchor_counts[anchor]
        print(f"    {anchor:15s}  {count:4d}  ({count/total*100:.1f}%)")

    print("\n  Query style distribution:")
    for style in sorted(style_counts.keys()):
        count = style_counts[style]
        print(f"    {style:15s}  {count:4d}  ({count/total*100:.1f}%)")

    print("\n  Reference mode distribution:")
    for ref_mode in ["none", "legal_ref", "doc_only", "both", "(missing)"]:
        count = reference_mode_counts.get(ref_mode, 0)
        if count:
            print(f"    {ref_mode:15s}  {count:4d}  ({count/total*100:.1f}%)")

    print("\n  Difficulty distribution:")
    for diff in ["easy", "medium", "tricky", "(missing)"]:
        count = diff_counts.get(diff, 0)
        if count:
            print(f"    {diff:15s}  {count:4d}  ({count/total*100:.1f}%)")

    print("\n  Gold set sizes (per query):")
    print("    rincian : 1.0  (exact anchor — finest granularity)")
    print("    ayat       : 1.0  (derived parent in ayat index)")
    print("    pasal      : 1.0  (derived parent in pasal index)")

    print("\n  Per document:")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"    {doc_id:35s}  {count:3d} queries")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(
        description=(
            "Convert ground_truth.json -> validated_testset.pkl "
            "with leaf-anchored multi-granularity gold sets (roll-up design)"
        )
    )
    ap.add_argument(
        "--check", action="store_true",
        help="Validate and show expansion stats without writing pkl",
    )
    ap.add_argument(
        "--stats", action="store_true",
        help="Print stats about existing validated_testset.pkl",
    )
    args = ap.parse_args()

    if args.stats:
        print_stats()
        return

    finalize(check_only=args.check)


if __name__ == "__main__":
    main()
