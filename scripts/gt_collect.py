"""
Ground Truth Collector and Validator.

Validates raw ChatGPT-generated GT files and merges them into a single
ground_truth.json dataset. The benchmark is ayat-anchored: each accepted
item must point to exactly one leaf node from data/index_ayat.

Required fields (hard validation):
  query, reference_mode, gold_node_id, gold_doc_id, navigation_path,
  gold_anchor_granularity, gold_anchor_node_id

Optional fields (soft validation):
  query_style  - one of: formal, colloquial
  difficulty   - one of: easy, medium, hard
  answer_hint  - short text excerpt

Output:
  data/ground_truth.json - final merged dataset (keyed by q001, q002, ...)

Usage:
    python scripts/gt_collect.py
    python scripts/gt_collect.py --check-only
    python scripts/gt_collect.py --force-merge
    python scripts/gt_collect.py --stats
    python scripts/gt_collect.py --file <path>
"""

import argparse
import json
import re
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/ground_truth_raw")
GT_FILE = Path("data/ground_truth.json")
DATA_INDEX = Path("data/index_ayat")

VALID_QUERY_STYLES = {"formal", "colloquial"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_REFERENCE_MODES = {"none", "legal_ref", "doc_only", "both"}
REQUIRED_FIELDS = {
    "query",
    "reference_mode",
    "gold_node_id",
    "gold_doc_id",
    "navigation_path",
    "gold_anchor_granularity",
    "gold_anchor_node_id",
}

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
ANCHOR_CROSS_REFERENCE_RE = re.compile(
    r"\bayat\s*\(\d+\)|\bpasal\s+\d+[A-Z]*",
    re.IGNORECASE,
)

CONTEXT_WARNING_PATTERNS = [
    ("coreference phrase", re.compile(r"\b(aturan|ketentuan|pasal|ayat|hal|sanksi|pidana)\s+(ini|itu|tersebut)\b", re.IGNORECASE)),
    ("dangling reference", re.compile(r"\b(yang tadi|di atas|tersebut di atas|berikut ini)\b", re.IGNORECASE)),
    ("conversation carry-over", re.compile(r"\b(kalau begitu|kalau gitu|juga nggak|juga gak|juga tidak)\b", re.IGNORECASE)),
]

# Cache for loaded docs (avoid re-reading the same file multiple times)
_doc_cache: dict[str, dict] = {}


def find_doc_path(doc_id: str) -> Path | None:
    """Find the index JSON file for a given doc_id."""
    for path in DATA_INDEX.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path
    return None


def load_doc(doc_id: str) -> dict | None:
    """Load an ayat-index document from disk with caching."""
    if doc_id in _doc_cache:
        return _doc_cache[doc_id]

    path = find_doc_path(doc_id)
    if not path:
        return None

    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    _doc_cache[doc_id] = doc
    return doc


def get_leaf_meta_map(doc_id: str) -> dict[str, dict]:
    """Return metadata for all leaf nodes in an ayat-index document."""
    doc = load_doc(doc_id)
    if doc is None:
        return {}

    if not isinstance(doc, dict) or "structure" not in doc:
        # Index file exists but has wrong format (e.g. accidentally overwritten with GT content)
        return {}

    leaf_map: dict[str, dict] = {}

    def _walk(nodes: list[dict], parent_path: str = "") -> None:
        for node in nodes:
            node_path = node.get("navigation_path", "").strip()
            if not node_path and parent_path and node.get("title"):
                node_path = f"{parent_path} > {node['title']}"
            elif not node_path:
                node_path = parent_path

            if "nodes" in node and node["nodes"]:
                _walk(node["nodes"], parent_path=node_path)
            elif node.get("text"):
                leaf_map[node["node_id"]] = {
                    "title": node.get("title", ""),
                    "navigation_path": node_path,
                    "text": node.get("text", ""),
                }

    _walk(doc["structure"])
    return leaf_map


def parse_raw_json(path: Path) -> list[dict]:
    """Load raw JSON, tolerating markdown code fences."""
    with open(path, encoding="utf-8") as f:
        raw = f.read().strip()

    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])

    data = json.loads(raw)
    if not isinstance(data, list):
        raise ValueError("Top-level value must be a JSON array (list)")
    return data


def infer_reference_mode(query: str) -> str:
    """Infer reference_mode from the query text."""
    has_legal = bool(LEGAL_REFERENCE_RE.search(query or ""))
    has_doc = bool(DOC_REFERENCE_RE.search(query or ""))

    if has_legal and has_doc:
        return "both"
    if has_legal:
        return "legal_ref"
    if has_doc:
        return "doc_only"
    return "none"


def detect_context_warnings(query: str) -> list[str]:
    """Return soft warnings for suspicious context-dependent phrasing."""
    warnings = []
    for label, pattern in CONTEXT_WARNING_PATTERNS:
        if pattern.search(query):
            warnings.append(f"Possible context-dependent phrasing detected ({label})")
    return warnings


def strip_self_heading(anchor_text: str, title: str) -> str:
    """Remove the node's own heading from the beginning of anchor text before cross-ref checks."""
    text = (anchor_text or "").lstrip()
    node_title = (title or "").strip()
    if not text or not node_title:
        return text

    # Most leaf texts begin with their own title, e.g. "Pasal 1" or "Pasal 81 Ayat (2)".
    # That self-heading should not count as a cross-reference to another provision.
    if text.lower().startswith(node_title.lower()):
        return text[len(node_title):].lstrip()

    first_line, _, rest = text.partition("\n")
    if first_line.strip().lower() == node_title.lower():
        return rest.lstrip()

    return text


def validate_raw_file(path: Path) -> tuple[list[dict], list[str], list[str]]:
    """
    Validate a single raw GT file.

    Returns:
        (valid_items, hard_errors, warnings)
    """
    doc_id = path.stem
    hard_errors: list[str] = []
    warnings: list[str] = []

    try:
        data = parse_raw_json(path)
    except json.JSONDecodeError as e:
        return [], [f"JSON parse error: {e}"], []
    except ValueError as e:
        return [], [str(e)], []

    leaf_map = get_leaf_meta_map(doc_id)
    if not leaf_map:
        hard_errors.append(
            f"Document '{doc_id}' not found in ayat index - cannot validate anchor node_ids"
        )
        return [], hard_errors, warnings

    valid_items: list[dict] = []
    seen_anchor_ids: set[str] = set()

    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            hard_errors.append(f"  Item {i} (item_{i}): Item must be a JSON object")
            continue

        label = item.get("gold_anchor_node_id") or item.get("gold_node_id") or f"item_{i}"
        item_errors: list[str] = []
        item_warnings: list[str] = []

        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            item_errors.append(f"Missing required fields: {sorted(missing)}")

        if item.get("gold_doc_id") != doc_id:
            item_errors.append(
                f"gold_doc_id mismatch: got '{item.get('gold_doc_id')}', expected '{doc_id}'"
            )

        if item.get("gold_anchor_granularity") != "ayat":
            item_errors.append(
                "gold_anchor_granularity must be exactly 'ayat'"
            )

        anchor_node_id = item.get("gold_anchor_node_id")
        if anchor_node_id and anchor_node_id not in leaf_map:
            item_errors.append(
                f"gold_anchor_node_id '{anchor_node_id}' not found as leaf node in ayat index"
            )

        if anchor_node_id:
            if anchor_node_id in seen_anchor_ids:
                item_errors.append(
                    f"Duplicate gold_anchor_node_id within this batch: '{anchor_node_id}'"
                )
            else:
                seen_anchor_ids.add(anchor_node_id)

        if item.get("gold_node_id") != anchor_node_id:
            item_errors.append(
                "gold_node_id must match gold_anchor_node_id for ayat-anchored GT"
            )

        query = (item.get("query") or "").strip()
        if len(query) < 10:
            item_errors.append("Query too short (< 10 chars)")

        reference_mode = item.get("reference_mode")
        if reference_mode not in VALID_REFERENCE_MODES:
            item_errors.append(
                f"reference_mode must be one of {sorted(VALID_REFERENCE_MODES)}, "
                f"got '{reference_mode}'"
            )
        else:
            inferred_reference_mode = infer_reference_mode(query)
            if reference_mode != inferred_reference_mode:
                item_warnings.append(
                    f"reference_mode mismatch: declared '{reference_mode}', "
                    f"inferred '{inferred_reference_mode}'"
                )

        if "query_style" in item:
            if item["query_style"] not in VALID_QUERY_STYLES:
                item_warnings.append(
                    f"Unknown query_style '{item['query_style']}' "
                    f"(valid: {sorted(VALID_QUERY_STYLES)})"
                )
        else:
            item_warnings.append("Missing optional field: query_style")

        if "difficulty" in item:
            if item["difficulty"] not in VALID_DIFFICULTIES:
                item_warnings.append(
                    f"Unknown difficulty '{item['difficulty']}' "
                    f"(valid: {sorted(VALID_DIFFICULTIES)})"
                )
        else:
            item_warnings.append("Missing optional field: difficulty")

        if anchor_node_id in leaf_map:
            actual_path = leaf_map[anchor_node_id]["navigation_path"]
            if item.get("navigation_path") != actual_path:
                item_warnings.append(
                    "navigation_path does not match ayat index exactly "
                    f"(expected: {actual_path})"
                )

            anchor_meta = leaf_map[anchor_node_id]
            anchor_text = anchor_meta.get("text", "")
            anchor_body = strip_self_heading(anchor_text, anchor_meta.get("title", ""))
            if ANCHOR_CROSS_REFERENCE_RE.search(anchor_body):
                item_warnings.append(
                    "Anchor ayat text cites other provisions; manually verify the query "
                    "is answerable from this ayat alone"
                )

        item_warnings.extend(detect_context_warnings(query))

        if item_errors:
            hard_errors.append(f"  Item {i} ({label}): {'; '.join(item_errors)}")
        else:
            for msg in item_warnings:
                warnings.append(f"  [WARN] Item {i} ({label}): {msg}")
            valid_items.append(item)

    return valid_items, hard_errors, warnings


def load_existing_gt() -> dict:
    """Load existing ground_truth.json or return empty dict if not found."""
    if not GT_FILE.exists():
        return {}

    with open(GT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Preserve backward compatibility for older merged files.
    for item in data.values():
        if "gold_anchor_granularity" not in item:
            item["gold_anchor_granularity"] = "ayat"
        if "gold_anchor_node_id" not in item and "gold_node_id" in item:
            item["gold_anchor_node_id"] = item["gold_node_id"]
        if "reference_mode" not in item and "query" in item:
            item["reference_mode"] = infer_reference_mode(item["query"])

    return data


def assign_query_id(existing: dict) -> str:
    """Generate the next sequential query ID (q001, q002, ...)."""
    if not existing:
        return "q001"

    existing_nums = [
        int(key[1:])
        for key in existing
        if key.startswith("q") and key[1:].isdigit()
    ]
    next_num = max(existing_nums, default=0) + 1
    return f"q{next_num:03d}"


def print_stats(gt: dict) -> None:
    """Print coverage and distribution statistics for a ground truth dict."""
    total = len(gt)
    if total == 0:
        print("  (empty)")
        return

    doc_counts: dict[str, int] = {}
    style_counts: dict[str, int] = {}
    difficulty_counts: dict[str, int] = {}
    anchor_counts: dict[str, int] = {}
    reference_mode_counts: dict[str, int] = {}

    for item in gt.values():
        doc_id = item["gold_doc_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        style = item.get("query_style") or "(missing)"
        style_counts[style] = style_counts.get(style, 0) + 1

        diff = item.get("difficulty") or "(missing)"
        difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        anchor = item.get("gold_anchor_granularity") or "(missing)"
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

        ref_mode = item.get("reference_mode") or "(missing)"
        reference_mode_counts[ref_mode] = reference_mode_counts.get(ref_mode, 0) + 1

    print(f"\nGround truth stats: {GT_FILE}")
    print(f"  Total questions   : {total}")
    print(f"  Documents covered : {len(doc_counts)}")

    print("\n  Anchor granularity distribution:")
    for anchor in sorted(anchor_counts.keys()):
        count = anchor_counts[anchor]
        pct = count / total * 100
        print(f"    {anchor:15s}  {count:4d}  ({pct:.1f}%)")

    print("\n  Query style distribution:")
    for style in sorted(style_counts.keys()):
        count = style_counts[style]
        pct = count / total * 100
        print(f"    {style:15s}  {count:4d}  ({pct:.1f}%)")

    print("\n  Reference mode distribution:")
    for ref_mode in ["none", "legal_ref", "doc_only", "both", "(missing)"]:
        count = reference_mode_counts.get(ref_mode, 0)
        if count == 0:
            continue
        pct = count / total * 100
        print(f"    {ref_mode:15s}  {count:4d}  ({pct:.1f}%)")

    print("\n  Difficulty distribution:")
    for diff in ["easy", "medium", "hard", "(missing)"]:
        count = difficulty_counts.get(diff, 0)
        if count == 0:
            continue
        pct = count / total * 100
        print(f"    {diff:15s}  {count:4d}  ({pct:.1f}%)")

    print("\n  Per document:")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"    {doc_id:35s}  {count:3d} questions")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Validate and merge ayat-anchored GT raw files")
    ap.add_argument(
        "--check-only", action="store_true",
        help="Validate only, do not write output",
    )
    ap.add_argument(
        "--force-merge", action="store_true",
        help="Merge even if files only have [WARN] issues. Hard errors still block.",
    )
    ap.add_argument(
        "--stats", action="store_true",
        help="Print stats about current ground_truth.json",
    )
    ap.add_argument(
        "--file", type=str, default=None,
        help="Process a specific raw file only",
    )
    args = ap.parse_args()

    if args.stats:
        if not GT_FILE.exists():
            print("ground_truth.json not found.")
            return
        gt = load_existing_gt()
        print_stats(gt)
        return

    if args.file:
        raw_files = [Path(args.file)]
    else:
        if not RAW_DIR.exists():
            print(f"Raw directory not found: {RAW_DIR}")
            print("Simpan output ChatGPT ke data/ground_truth_raw/<KATEGORI>/<doc_id>.json")
            return
        raw_files = sorted(RAW_DIR.rglob("*.json"))

    if not raw_files:
        print(f"Tidak ada file di {RAW_DIR}")
        return

    existing_gt = load_existing_gt()
    existing_queries = {item["query"].lower() for item in existing_gt.values()}
    existing_doc_ids = {item["gold_doc_id"] for item in existing_gt.values()}
    existing_anchors: dict[tuple[str, str], str] = {
        (item["gold_doc_id"], item.get("gold_anchor_node_id", item.get("gold_node_id", ""))): qid
        for qid, item in existing_gt.items()
    }

    all_valid: list[dict] = []
    total_hard_errors = 0
    total_warnings = 0

    print(f"\nMemvalidasi {len(raw_files)} file...")
    print()

    for raw_path in raw_files:
        doc_id = raw_path.stem
        valid_items, hard_errors, warnings = validate_raw_file(raw_path)

        accepted: list[dict] = []
        dup_errors: list[str] = []
        for item in valid_items:
            q_lower = item["query"].lower()
            anchor_key = (item.get("gold_doc_id", ""), item.get("gold_anchor_node_id", ""))
            if q_lower in existing_queries:
                dup_errors.append(f"  Duplicate query: '{item['query'][:60]}'")
            elif anchor_key[1] and anchor_key in existing_anchors:
                dup_errors.append(
                    f"  Duplicate anchor ({anchor_key[0]}, {anchor_key[1]}) "
                    f"already in GT as {existing_anchors[anchor_key]}"
                )
            else:
                existing_queries.add(q_lower)
                existing_anchors[anchor_key] = item["query"][:40]
                accepted.append(item)

        all_errors_for_file = hard_errors + dup_errors
        status = "✓" if not all_errors_for_file else "✗"
        already = " [sudah ada di GT]" if doc_id in existing_doc_ids else ""
        warn_note = f", {len(warnings)} warnings" if warnings else ""

        print(
            f"  {status} {doc_id}{already}: "
            f"{len(accepted)} valid, {len(all_errors_for_file)} errors{warn_note}"
        )
        for msg in all_errors_for_file + warnings:
            print(f"    {msg}")

        all_valid.extend(accepted)
        total_hard_errors += len(all_errors_for_file)
        total_warnings += len(warnings)

    print(
        f"\nTotal: {len(all_valid)} pertanyaan valid, "
        f"{total_hard_errors} hard errors, {total_warnings} warnings"
    )

    if total_hard_errors > 0 and not args.force_merge:
        print("\nAda hard errors - perbaiki dulu sebelum merge.")
        print("Atau gunakan --force-merge untuk tetap merge meski ada warnings.")
        sys.exit(1)

    if total_hard_errors > 0 and args.force_merge:
        print("\n[WARN] Ada hard errors tapi --force-merge aktif - hanya item valid yang di-merge.")

    if args.check_only:
        print("\n[check-only] Tidak menulis output.")
        return

    if not all_valid:
        print("\nTidak ada item baru untuk ditambahkan.")
        return

    merged_gt = dict(existing_gt)
    for item in all_valid:
        qid = assign_query_id(merged_gt)
        merged_gt[qid] = {
            "query": item["query"],
            "query_style": item.get("query_style", ""),
            "difficulty": item.get("difficulty", ""),
            "reference_mode": item["reference_mode"],
            "gold_anchor_granularity": item["gold_anchor_granularity"],
            "gold_anchor_node_id": item["gold_anchor_node_id"],
            "gold_node_id": item["gold_node_id"],
            "gold_doc_id": item["gold_doc_id"],
            "navigation_path": item["navigation_path"],
            "answer_hint": item.get("answer_hint", ""),
        }

    GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_gt, f, ensure_ascii=False, indent=2)

    added = len(merged_gt) - len(existing_gt)
    print(f"\n✓ Disimpan ke {GT_FILE}")
    print(f"  Sebelumnya : {len(existing_gt)} pertanyaan")
    print(f"  Ditambahkan: {added} pertanyaan")
    print(f"  Total      : {len(merged_gt)} pertanyaan")


if __name__ == "__main__":
    main()
