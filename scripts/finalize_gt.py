"""
Ground Truth Finalizer.

Converts data/ground_truth.json (ayat-anchored annotations produced by
gt_collect.py) into data/validated_testset.pkl, which stores gold node ID
sets for all three index granularities: pasal, ayat, and full_split.

EVALUATION USAGE (in your retrieval eval script):
  if granularity == "pasal":
      hit = retrieved_node_id in item["gold_pasal_node_ids"]
  elif granularity == "ayat":
      hit = retrieved_node_id in item["gold_ayat_node_ids"]
  elif granularity == "full_split":
      hit = retrieved_node_id in item["gold_full_split_node_ids"]

The semantic anchor is always an ayat-index leaf node:
  - pasal eval asks whether retrieval found the parent Pasal
  - ayat eval asks whether retrieval found the exact ayat anchor
  - full_split eval asks whether retrieval entered the full_split subtree
    under that ayat anchor

Usage:
    python scripts/finalize_gt.py
    python scripts/finalize_gt.py --check
    python scripts/finalize_gt.py --stats
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

GT_FILE = Path("data/ground_truth.json")
TESTSET_FILE = Path("data/validated_testset.pkl")
INDEX_PASAL = Path("data/index_pasal")
INDEX_AYAT = Path("data/index_ayat")
INDEX_FULL_SPLIT = Path("data/index_full_split")

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
    """Return the ayat anchor node ID from a merged GT item."""
    return item.get("gold_anchor_node_id") or item["gold_node_id"]


def get_anchor_granularity(item: dict) -> str:
    """Return the anchor granularity from a merged GT item."""
    return item.get("gold_anchor_granularity", "ayat")


def derive_pasal_node_id(anchor_node_id: str, doc_id: str) -> str:
    """
    Derive the parent pasal node ID for an ayat anchor.

    Common case:
      0002_a2 -> 0002
    If the ayat index leaf is unchanged from pasal level:
      0002 -> 0002
    """
    pasal_leaf_ids = get_leaf_ids(doc_id, INDEX_PASAL)
    candidates = [anchor_node_id]

    if "_a" in anchor_node_id:
        candidates.insert(0, anchor_node_id.split("_a", 1)[0])

    for candidate in candidates:
        if candidate in pasal_leaf_ids:
            return candidate

    raise ValueError(
        f"Could not derive pasal node_id from ayat anchor '{anchor_node_id}' in doc '{doc_id}'"
    )


def get_full_split_gold_ids(anchor_node_id: str, doc_id: str) -> set[str]:
    """
    Return the valid full_split gold node IDs for an ayat anchor.

    Any leaf that equals the ayat anchor or is a descendant of that anchor
    counts as correct.
    """
    full_leaf_ids = get_leaf_ids(doc_id, INDEX_FULL_SPLIT)
    if not full_leaf_ids:
        return {anchor_node_id}

    valid = {
        node_id for node_id in full_leaf_ids
        if node_id == anchor_node_id or node_id.startswith(anchor_node_id + "_")
    }
    return valid if valid else {anchor_node_id}


def finalize(check_only: bool = False) -> dict:
    """
    Load ground_truth.json, expand each item with multi-granularity gold sets,
    and write to validated_testset.pkl (unless check_only=True).

    Returns the finalized testset dict.
    """
    if not GT_FILE.exists():
        print(f"ERROR: {GT_FILE} not found.")
        print("Run scripts/gt_collect.py first to produce ground_truth.json.")
        sys.exit(1)

    with open(GT_FILE, encoding="utf-8") as f:
        gt = json.load(f)

    if not gt:
        print("ERROR: ground_truth.json is empty.")
        sys.exit(1)

    print(f"\nLoaded {len(gt)} items from {GT_FILE}")
    print("Expanding gold sets for pasal / ayat / full_split ...")

    testset: dict[str, dict] = {}
    errors: list[str] = []
    docs_missing_full: set[str] = set()

    for qid, item in gt.items():
        doc_id = item["gold_doc_id"]
        anchor_granularity = get_anchor_granularity(item)
        anchor_node_id = get_anchor_node_id(item)

        if anchor_granularity != "ayat":
            errors.append(
                f"{qid}: gold_anchor_granularity must be 'ayat', got '{anchor_granularity}'"
            )
            continue

        ayat_leaf_ids = get_leaf_ids(doc_id, INDEX_AYAT)
        if not ayat_leaf_ids:
            errors.append(f"{qid}: doc '{doc_id}' not found in {INDEX_AYAT}")
            continue

        if anchor_node_id not in ayat_leaf_ids:
            errors.append(
                f"{qid}: anchor node '{anchor_node_id}' is not a leaf in {INDEX_AYAT}; "
                "the GT likely still uses old pasal-anchored annotations"
            )
            continue

        try:
            pasal_node_id = derive_pasal_node_id(anchor_node_id, doc_id)
        except ValueError as e:
            errors.append(f"{qid}: {e}")
            continue

        if find_doc_path(doc_id, INDEX_FULL_SPLIT) is None:
            docs_missing_full.add(doc_id)

        gold_pasal = {pasal_node_id}
        gold_ayat = {anchor_node_id}
        gold_full = get_full_split_gold_ids(anchor_node_id, doc_id)

        testset[qid] = {
            "query": item["query"],
            "query_style": item.get("query_style", ""),
            "difficulty": item.get("difficulty", ""),
            "gold_doc_id": doc_id,
            "gold_anchor_granularity": "ayat",
            "gold_anchor_node_id": anchor_node_id,
            # Backward-compatible singular field used by load_testset.py
            "gold_node_id": anchor_node_id,
            "gold_pasal_node_ids": gold_pasal,
            "gold_ayat_node_ids": gold_ayat,
            "gold_full_split_node_ids": gold_full,
            "navigation_path": item.get("navigation_path", ""),
            "answer_hint": item.get("answer_hint", ""),
        }

    if errors:
        print(f"\n{len(errors)} hard errors:")
        for err in errors:
            print(f"  ✗ {err}")
        print("\nPerbaiki errors di atas sebelum melanjutkan.")
        sys.exit(1)

    if docs_missing_full:
        print(
            f"\n[WARN] {len(docs_missing_full)} doc(s) not in index_full_split "
            "(gold_full_split falls back to ayat anchor node_id):"
        )
        for doc_id in sorted(docs_missing_full):
            print(f"  - {doc_id}")

    n = len(testset)
    total_full = sum(len(v["gold_full_split_node_ids"]) for v in testset.values())
    multi_full = sum(1 for v in testset.values() if len(v["gold_full_split_node_ids"]) > 1)

    print("\nGold set sizes (avg per query):")
    print("  pasal      : 1 node  (parent pasal of ayat anchor)")
    print("  ayat       : 1 node  (exact ayat anchor)")
    print(
        f"  full_split : {total_full/n:.1f} nodes  "
        f"({multi_full}/{n} queries have >1 valid node)"
    )

    if check_only:
        print("\n[check-only] Tidak menulis output.")
        return testset

    TESTSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TESTSET_FILE, "wb") as f:
        pickle.dump(testset, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n✓ Disimpan ke {TESTSET_FILE}")
    print(f"  {len(testset)} queries siap dipakai untuk evaluasi 3 granularity")
    print("\nNext step:")
    print("  python scripts/load_testset.py  # inspect result")

    return testset


def print_stats() -> None:
    """Print statistics about an existing validated_testset.pkl."""
    if not TESTSET_FILE.exists():
        print(f"ERROR: {TESTSET_FILE} not found.")
        print("Run scripts/finalize_gt.py first.")
        sys.exit(1)

    with open(TESTSET_FILE, "rb") as f:
        testset = pickle.load(f)

    total = len(testset)
    doc_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}
    diff_counts: dict[str, int] = {}
    anchor_counts: dict[str, int] = {}
    total_full_ids = 0
    multi_full = 0

    for item in testset.values():
        doc_id = item["gold_doc_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        style = item.get("query_style") or "(missing)"
        style_counts[style] = style_counts.get(style, 0) + 1

        diff = item.get("difficulty") or "(missing)"
        diff_counts[diff] = diff_counts.get(diff, 0) + 1

        anchor = item.get("gold_anchor_granularity") or "(missing)"
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

        full_ids = item.get("gold_full_split_node_ids", set())
        total_full_ids += len(full_ids)
        if len(full_ids) > 1:
            multi_full += 1

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

    print("\n  Difficulty distribution:")
    for diff in ["easy", "medium", "tricky", "(missing)"]:
        count = diff_counts.get(diff, 0)
        if count:
            print(f"    {diff:15s}  {count:4d}  ({count/total*100:.1f}%)")

    print("\n  Gold set sizes (avg per query):")
    print("    pasal      : 1.0  (parent pasal of ayat anchor)")
    print("    ayat       : 1.0  (exact ayat anchor)")
    print(
        f"    full_split : {total_full_ids/total:.1f}  "
        f"({multi_full}/{total} queries have >1 valid node)"
    )

    print("\n  Per document:")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"    {doc_id:35s}  {count:3d} queries")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(
        description=(
            "Convert ground_truth.json -> validated_testset.pkl "
            "with ayat-anchored multi-granularity gold sets"
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
