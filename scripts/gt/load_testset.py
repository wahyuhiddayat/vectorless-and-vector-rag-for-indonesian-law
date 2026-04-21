"""
Load and inspect validated_testset.pkl ground truth dataset.

Utility script to load, validate, and preview the serialized ground truth
testset used for evaluation. Shows statistics, samples, and full data dump.

File format: pickle dict with structure:
  {
    "q001": {
      "query": str,
      "reference_mode": "none|legal_ref|doc_only|both",
      "gold_anchor_granularity": "ayat",
      "gold_anchor_node_id": str,
      "gold_node_id": str,
      "gold_doc_id": str,
      "gold_pasal_node_ids": set[str],
      "gold_ayat_node_ids": set[str],
      "gold_full_split_node_ids": set[str],
      "navigation_path": str,
      "answer_hint": str (optional)
    },
    ...
  }

Usage:
    python scripts/gt/load_testset.py                    # show stats + first 3 items
    python scripts/gt/load_testset.py --full             # dump entire dataset
    python scripts/gt/load_testset.py --filter perpu     # filter by doc_id pattern
    python scripts/gt/load_testset.py --query "keyword"  # search queries by keyword
    python scripts/gt/load_testset.py --stats            # detailed statistics
    python scripts/gt/load_testset.py --json             # output as JSON (to file/stdout)
    python scripts/gt/load_testset.py --doc perpu-1-2016 # filter by specific doc_id
"""

import argparse
import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

# Force UTF-8 on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TESTSET_FILE = Path("data/validated_testset.pkl")
ALT_TESTSET_FILE = Path("data/ground_truth.pkl")  # Alternative name


def load_testset(path: Path = None) -> dict:
    """Load testset from pickle file. Try multiple paths."""
    if path is None:
        # Try primary and fallback paths
        for candidate in [TESTSET_FILE, ALT_TESTSET_FILE]:
            if candidate.exists():
                path = candidate
                break
        if path is None:
            raise FileNotFoundError(
                f"Testset not found. Tried:\n  - {TESTSET_FILE}\n  - {ALT_TESTSET_FILE}"
            )

    with open(path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data)}")

    return data


def show_stats(data: dict):
    """Print comprehensive statistics about the testset."""
    doc_counts = defaultdict(int)
    anchor_nodes = defaultdict(set)
    reference_mode_counts = defaultdict(int)

    for item in data.values():
        doc_id = item.get("gold_doc_id", "?")
        node_id = item.get("gold_anchor_node_id") or item.get("gold_node_id", "?")
        doc_counts[doc_id] += 1
        anchor_nodes[doc_id].add(node_id)
        reference_mode_counts[item.get("reference_mode", "(missing)")] += 1

    print("\n" + "=" * 70)
    print("TESTSET STATISTICS")
    print("=" * 70)
    print(f"\nTotal questions       : {len(data)}")
    print(f"Documents covered     : {len(doc_counts)}")
    print(f"Unique anchor nodes   : {sum(len(nodes) for nodes in anchor_nodes.values())}")

    print(f"\nQuestions per document:")
    for doc_id in sorted(doc_counts.keys()):
        count = doc_counts[doc_id]
        unique_nodes = len(anchor_nodes[doc_id])
        print(f"  {doc_id:30s}  {count:3d} questions  ({unique_nodes} unique anchors)")

    print(f"\nReference mode distribution:")
    for ref_mode in ["none", "legal_ref", "doc_only", "both", "(missing)"]:
        count = reference_mode_counts.get(ref_mode, 0)
        if count:
            print(f"  {ref_mode:15s}  {count:3d}")

    print()


def show_preview(data: dict, num_items: int = 3):
    """Show first N items."""
    print("\n" + "=" * 70)
    print(f"PREVIEW: First {num_items} items")
    print("=" * 70 + "\n")

    for i, (qid, item) in enumerate(list(data.items())[:num_items], 1):
        print(f"[{qid}]")
        print(f"  Query        : {item.get('query', '?')[:100]}")
        if len(item.get('query', '')) > 100:
            print(f"                (truncated...)")
        print(f"  Doc ID       : {item.get('gold_doc_id', '?')}")
        print(f"  Anchor       : {item.get('gold_anchor_node_id', item.get('gold_node_id', '?'))}")
        print(f"  Granularity  : {item.get('gold_anchor_granularity', '?')}")
        print(f"  Ref mode     : {item.get('reference_mode', '?')}")
        print(f"  Path         : {item.get('navigation_path', '?')}")
        if item.get("answer_hint"):
            print(f"  Answer hint  : {item['answer_hint'][:80]}")
        print(f"  Gold sets    : pasal={len(item.get('gold_pasal_node_ids', set()))}, "
              f"ayat={len(item.get('gold_ayat_node_ids', set()))}, "
              f"full_split={len(item.get('gold_full_split_node_ids', set()))}")
        print()


def show_full(data: dict):
    """Pretty-print entire testset."""
    print("\n" + "=" * 70)
    print("FULL TESTSET")
    print("=" * 70 + "\n")

    for qid, item in data.items():
        print(f"[{qid}]")
        print(f"  Query        : {item.get('query', '?')}")
        print(f"  Doc ID       : {item.get('gold_doc_id', '?')}")
        print(f"  Anchor       : {item.get('gold_anchor_node_id', item.get('gold_node_id', '?'))}")
        print(f"  Granularity  : {item.get('gold_anchor_granularity', '?')}")
        print(f"  Ref mode     : {item.get('reference_mode', '?')}")
        print(f"  Path         : {item.get('navigation_path', '?')}")
        if item.get("answer_hint"):
            print(f"  Answer hint  : {item['answer_hint']}")
        print(f"  Gold pasal   : {sorted(item.get('gold_pasal_node_ids', set()))}")
        print(f"  Gold ayat    : {sorted(item.get('gold_ayat_node_ids', set()))}")
        print(f"  Gold full    : {sorted(item.get('gold_full_split_node_ids', set()))}")
        print()


def filter_by_doc(data: dict, doc_id: str) -> dict:
    """Return items matching a specific doc_id."""
    return {qid: item for qid, item in data.items()
            if item.get("gold_doc_id") == doc_id}


def filter_by_pattern(data: dict, pattern: str) -> dict:
    """Return items where doc_id contains pattern."""
    pattern = pattern.lower()
    return {qid: item for qid, item in data.items()
            if pattern in item.get("gold_doc_id", "").lower()}


def search_queries(data: dict, keyword: str) -> dict:
    """Return items where query contains keyword (case-insensitive)."""
    keyword = keyword.lower()
    return {qid: item for qid, item in data.items()
            if keyword in item.get("query", "").lower()}


def export_json(data: dict, output_file: str = None):
    """Export testset to JSON format."""
    def _convert(value):
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {k: _convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_convert(v) for v in value]
        return value

    json_str = json.dumps(_convert(data), ensure_ascii=False, indent=2)

    if output_file:
        Path(output_file).write_text(json_str, encoding="utf-8")
        print(f"Exported to: {output_file}")
    else:
        print(json_str)


def main():
    ap = argparse.ArgumentParser(
        description="Load and inspect validated_testset.pkl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/gt/load_testset.py              # show stats + preview
  python scripts/gt/load_testset.py --full       # dump entire dataset
  python scripts/gt/load_testset.py --stats      # detailed stats only
  python scripts/gt/load_testset.py --doc perpu-1-2016  # items from one doc
  python scripts/gt/load_testset.py --filter pp  # items from docs matching 'pp'
  python scripts/gt/load_testset.py --query "pidana"  # search in queries
  python scripts/gt/load_testset.py --json testset.json  # export to JSON
        """
    )

    ap.add_argument("--full", action="store_true",
                    help="Dump entire testset")
    ap.add_argument("--stats", action="store_true",
                    help="Show statistics only")
    ap.add_argument("--doc", type=str, default=None,
                    help="Filter by specific doc_id (exact match)")
    ap.add_argument("--filter", type=str, default=None,
                    help="Filter by doc_id pattern (substring match)")
    ap.add_argument("--query", type=str, default=None,
                    help="Search in query text (case-insensitive)")
    ap.add_argument("--json", type=str, default=None, nargs="?", const="-",
                    help="Export to JSON file (or stdout if no file specified)")
    ap.add_argument("--preview", type=int, default=3,
                    help="Number of items to show in preview (default: 3)")
    ap.add_argument("--file", type=str, default=None,
                    help="Load from specific pickle file")

    args = ap.parse_args()

    # Load testset
    try:
        testset_path = Path(args.file) if args.file else None
        data = load_testset(testset_path)
        print(f"Loaded: {len(data)} items")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading testset: {e}")
        sys.exit(1)

    # Apply filters
    if args.doc:
        data = filter_by_doc(data, args.doc)
        print(f"Filtered to doc '{args.doc}': {len(data)} items")
    elif args.filter:
        data = filter_by_pattern(data, args.filter)
        print(f"Filtered to pattern '{args.filter}': {len(data)} items")
    elif args.query:
        data = search_queries(data, args.query)
        print(f"Found {len(data)} items matching query '{args.query}'")

    # Show output
    if args.json is not None:
        export_json(data, None if args.json == "-" else args.json)
    elif args.full:
        show_full(data)
    elif args.stats:
        show_stats(data)
    else:
        # Default: stats + preview
        show_stats(data)
        show_preview(data, args.preview)


if __name__ == "__main__":
    main()
