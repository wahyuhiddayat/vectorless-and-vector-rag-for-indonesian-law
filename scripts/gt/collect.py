"""
Ground Truth Collector and Validator.

Runs structural validation on raw GT files (LLM-generated, then judged via
build_validate.py / external judge LLM) and merges accepted items into
data/ground_truth.json. The benchmark is leaf-anchored: each accepted item
must point to one or more leaf nodes from data/index_rincian.

Per design v3 (3-type stratified): allowed query types are factual,
paraphrased, multihop. factual + paraphrased = single anchor. multihop = 2
anchors in the same doc.

Required fields:
  query, reference_mode, gold_node_id, gold_doc_id, navigation_path,
  gold_anchor_granularity, gold_anchor_node_id

Semantic checks (cross-ref anchors, referential queries, multi-hop) are
delegated to the LLM judge step, see scripts/gt/build_validate.py.

Output:
  data/ground_truth.json - final merged dataset (keyed by q001, q002, ...)

Usage:
    python scripts/gt/collect.py
    python scripts/gt/collect.py --check-only
    python scripts/gt/collect.py --force-merge
    python scripts/gt/collect.py --stats
    python scripts/gt/collect.py --file <path>
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
DATA_INDEX = Path("data/index_rincian")
AUDIT_DIR = Path("data/gt_audit")

VALID_REFERENCE_MODES = {"none", "legal_ref", "doc_only", "both"}
VALID_QUERY_TYPES = {"factual", "paraphrased", "multihop"}
SINGLE_ANCHOR_TYPES = {"factual", "paraphrased"}
MULTI_ANCHOR_TYPES = {"multihop"}
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
# Cache for loaded docs (avoid re-reading the same file multiple times)
_doc_cache: dict[str, dict] = {}


def find_doc_path(doc_id: str) -> Path | None:
    """Find the index JSON file for a given doc_id."""
    for path in DATA_INDEX.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path
    return None


def load_doc(doc_id: str) -> dict | None:
    """Load a rincian-index document from disk with caching."""
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
    """Return metadata for all leaf nodes in a rincian-index document."""
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


def doc_id_from_path(path: Path) -> str:
    """Strip the optional `__<query_type>` suffix from a raw GT filename stem."""
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    return stem


def query_type_from_path(path: Path) -> str:
    """Extract the query type from a raw GT filename stem.

    Files named `<doc_id>.json` map to `factual`. Files named
    `<doc_id>__<type>.json` map to `<type>`.
    """
    stem = path.stem
    if "__" in stem:
        return stem.split("__", 1)[1]
    return "factual"


def validate_raw_file(path: Path) -> tuple[list[dict], list[str]]:
    """Run hard structural validation on a raw GT file.

    Returns (valid_items, hard_errors). Semantic checks (cross-ref anchors,
    referential queries, multi-hop sufficiency) are delegated to the LLM
    judge step.
    """
    doc_id = doc_id_from_path(path)
    hard_errors: list[str] = []

    try:
        data = parse_raw_json(path)
    except json.JSONDecodeError as e:
        return [], [f"JSON parse error: {e}"]
    except ValueError as e:
        return [], [str(e)]

    leaf_map = get_leaf_meta_map(doc_id)
    if not leaf_map:
        hard_errors.append(
            f"Document '{doc_id}' not found in rincian index - cannot validate anchor node_ids"
        )
        return [], hard_errors

    valid_items: list[dict] = []
    seen_anchor_ids: set[str] = set()

    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            hard_errors.append(f"  Item {i} (item_{i}): Item must be a JSON object")
            continue

        label = item.get("gold_anchor_node_id") or item.get("gold_node_id") or f"item_{i}"
        item_errors: list[str] = []

        missing = REQUIRED_FIELDS - set(item.keys())
        if missing:
            item_errors.append(f"Missing required fields: {sorted(missing)}")

        if item.get("gold_doc_id") != doc_id:
            item_errors.append(
                f"gold_doc_id mismatch: got '{item.get('gold_doc_id')}', expected '{doc_id}'"
            )

        if item.get("gold_anchor_granularity") != "rincian":
            item_errors.append("gold_anchor_granularity must be exactly 'rincian'")

        anchor_node_id = item.get("gold_anchor_node_id")
        if anchor_node_id and anchor_node_id not in leaf_map:
            item_errors.append(
                f"gold_anchor_node_id '{anchor_node_id}' not found as leaf node in rincian index"
            )

        if anchor_node_id:
            if anchor_node_id in seen_anchor_ids:
                item_errors.append(
                    f"Duplicate gold_anchor_node_id within this batch: '{anchor_node_id}'"
                )
            else:
                seen_anchor_ids.add(anchor_node_id)

        if item.get("gold_node_id") != anchor_node_id:
            item_errors.append("gold_node_id must match gold_anchor_node_id for leaf-anchored GT")

        if len((item.get("query") or "").strip()) < 10:
            item_errors.append("Query too short (< 10 chars)")

        if item.get("reference_mode") not in VALID_REFERENCE_MODES:
            item_errors.append(
                f"reference_mode must be one of {sorted(VALID_REFERENCE_MODES)}, "
                f"got '{item.get('reference_mode')}'"
            )

        # query_type is optional for legacy single-type files (default factual). When
        # present we validate against the allowed set and per-type anchor count.
        # gold_anchor_node_ids is required for multi-anchor types (multihop = 2 anchors).
        query_type = item.get("query_type", "factual")
        if query_type not in VALID_QUERY_TYPES:
            item_errors.append(
                f"query_type must be one of {sorted(VALID_QUERY_TYPES)}, got '{query_type}'"
            )
        else:
            anchor_ids = item.get("gold_anchor_node_ids")
            if query_type in MULTI_ANCHOR_TYPES and anchor_ids is None:
                item_errors.append(
                    f"query_type '{query_type}' requires field gold_anchor_node_ids "
                    f"(list of 2 anchor node ids)"
                )
            elif anchor_ids is not None:
                if not isinstance(anchor_ids, list) or not all(isinstance(a, str) for a in anchor_ids):
                    item_errors.append("gold_anchor_node_ids must be a list of strings when present")
                elif query_type in SINGLE_ANCHOR_TYPES and len(anchor_ids) != 1:
                    item_errors.append(
                        f"query_type '{query_type}' requires exactly 1 anchor, got {len(anchor_ids)}"
                    )
                elif query_type in MULTI_ANCHOR_TYPES and len(anchor_ids) != 2:
                    item_errors.append(
                        f"query_type '{query_type}' requires exactly 2 anchors, got {len(anchor_ids)}"
                    )
                else:
                    for aid in anchor_ids:
                        if aid not in leaf_map and aid != anchor_node_id:
                            # Multihop has both anchors in the same doc, so all
                            # anchor_ids must resolve in this doc's leaf_map.
                            item_errors.append(
                                f"gold_anchor_node_ids entry '{aid}' not found as leaf in this doc"
                            )

            # gold_doc_ids: for single-anchor mirror primary; for multihop mirror
            # primary doc twice (both anchors live in the same doc).
            doc_ids_field = item.get("gold_doc_ids")
            if query_type in MULTI_ANCHOR_TYPES and doc_ids_field is not None:
                if not isinstance(doc_ids_field, list) or len(doc_ids_field) != 2:
                    item_errors.append(
                        f"query_type '{query_type}' requires gold_doc_ids list of 2 doc ids"
                    )
                elif any(d != doc_id for d in doc_ids_field):
                    item_errors.append(
                        f"query_type '{query_type}' requires both gold_doc_ids entries to match '{doc_id}'"
                    )

        if item_errors:
            hard_errors.append(f"  Item {i} ({label}): {'; '.join(item_errors)}")
        else:
            valid_items.append(item)

    return valid_items, hard_errors


def load_audit_dropped_anchors(doc_id: str, query_type: str = "factual") -> set[str]:
    """Return primary anchor_node_ids the author flagged 'wrong' for (doc_id, query_type).

    Reads data/gt_audit/<doc_id>.json for factual or
    data/gt_audit/<doc_id>__<type>.json for other types. Returns an empty set
    when no log exists so audit logging stays optional.
    """
    name = f"{doc_id}__{query_type}"
    audit_path = AUDIT_DIR / f"{name}.json"
    if not audit_path.exists():
        return set()
    with open(audit_path, encoding="utf-8") as f:
        audit = json.load(f)
    return {
        entry["anchor_node_id"]
        for entry in audit.get("items", [])
        if entry.get("verdict") == "wrong" and entry.get("anchor_node_id")
    }


def load_existing_gt() -> dict:
    """Load existing ground_truth.json or return empty dict if not found."""
    if not GT_FILE.exists():
        return {}

    with open(GT_FILE, encoding="utf-8") as f:
        data = json.load(f)

    # Preserve backward compatibility for older merged files.
    for item in data.values():
        if "gold_anchor_granularity" not in item:
            item["gold_anchor_granularity"] = "rincian"
        if "gold_anchor_node_id" not in item and "gold_node_id" in item:
            item["gold_anchor_node_id"] = item["gold_node_id"]
        if "reference_mode" not in item and "query" in item:
            item["reference_mode"] = infer_reference_mode(item["query"])
        if "query_type" not in item:
            item["query_type"] = "factual"
        if "gold_anchor_node_ids" not in item:
            item["gold_anchor_node_ids"] = [item.get("gold_anchor_node_id", item.get("gold_node_id", ""))]
        if "gold_doc_ids" not in item:
            item["gold_doc_ids"] = [item.get("gold_doc_id", "")]

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
    anchor_counts: dict[str, int] = {}
    reference_mode_counts: dict[str, int] = {}
    query_type_counts: dict[str, int] = {}

    for item in gt.values():
        doc_id = item["gold_doc_id"]
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1

        style = item.get("query_style") or "(missing)"
        style_counts[style] = style_counts.get(style, 0) + 1

        anchor = item.get("gold_anchor_granularity") or "(missing)"
        anchor_counts[anchor] = anchor_counts.get(anchor, 0) + 1

        ref_mode = item.get("reference_mode") or "(missing)"
        reference_mode_counts[ref_mode] = reference_mode_counts.get(ref_mode, 0) + 1

        qtype = item.get("query_type") or "(missing)"
        query_type_counts[qtype] = query_type_counts.get(qtype, 0) + 1

    print(f"\nGround truth stats: {GT_FILE}")
    print(f"  Total questions   : {total}")
    print(f"  Documents covered : {len(doc_counts)}")

    print("\n  Query type distribution:")
    for qtype in ["factual", "paraphrased", "multihop", "(missing)"]:
        count = query_type_counts.get(qtype, 0)
        if count == 0:
            continue
        pct = count / total * 100
        print(f"    {qtype:15s}  {count:4d}  ({pct:.1f}%)")

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

    print("\n  Per document:")
    for doc_id, count in sorted(doc_counts.items()):
        print(f"    {doc_id:35s}  {count:3d} questions")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Validate and merge leaf-anchored GT raw files")
    ap.add_argument(
        "--check-only", action="store_true",
        help="Validate only, do not write output",
    )
    ap.add_argument(
        "--force-merge", action="store_true",
        help="Skip items with hard errors and merge the rest instead of aborting.",
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
            print("Simpan output annotator ke data/ground_truth_raw/<KATEGORI>/<doc_id>.json")
            return
        raw_files = sorted(
            p for p in RAW_DIR.rglob("*.json")
            if ".bak" not in p.parts
        )

    if not raw_files:
        print(f"Tidak ada file di {RAW_DIR}")
        return

    existing_gt = load_existing_gt()
    existing_queries = {item["query"].lower() for item in existing_gt.values()}
    existing_doc_ids = {item["gold_doc_id"] for item in existing_gt.values()}
    # Anchor dedup keys on (doc_id, anchor_node_id, query_type) so designs that
    # legitimately reuse an anchor across types (factual + paraphrased share the
    # same leaf by design) do not silently drop the second item.
    existing_anchors: dict[tuple[str, str, str], str] = {
        (
            item["gold_doc_id"],
            item.get("gold_anchor_node_id", item.get("gold_node_id", "")),
            item.get("query_type", "factual"),
        ): qid
        for qid, item in existing_gt.items()
    }

    all_valid: list[dict] = []
    total_hard_errors = 0

    print(f"\nMemvalidasi {len(raw_files)} file...")
    print()

    audit_drops_total = 0
    for raw_path in raw_files:
        doc_id = doc_id_from_path(raw_path)
        query_type_for_path = query_type_from_path(raw_path)
        valid_items, hard_errors = validate_raw_file(raw_path)

        dropped_anchors = load_audit_dropped_anchors(doc_id, query_type_for_path)
        if dropped_anchors:
            kept = [it for it in valid_items if it.get("gold_anchor_node_id") not in dropped_anchors]
            n_dropped = len(valid_items) - len(kept)
            if n_dropped:
                audit_drops_total += n_dropped
                print(f"  [audit] {doc_id}: dropping {n_dropped} item(s) flagged 'wrong' in gt_audit/")
            valid_items = kept

        accepted: list[dict] = []
        dup_errors: list[str] = []
        for item in valid_items:
            q_lower = item["query"].lower()
            anchor_key = (
                item.get("gold_doc_id", ""),
                item.get("gold_anchor_node_id", ""),
                item.get("query_type", "factual"),
            )
            if q_lower in existing_queries:
                dup_errors.append(f"  Duplicate query: '{item['query'][:60]}'")
            elif anchor_key[1] and anchor_key in existing_anchors:
                dup_errors.append(
                    f"  Duplicate anchor ({anchor_key[0]}, {anchor_key[1]}, type={anchor_key[2]}) "
                    f"already in GT as {existing_anchors[anchor_key]}"
                )
            else:
                existing_queries.add(q_lower)
                existing_anchors[anchor_key] = item["query"][:40]
                accepted.append(item)

        already_in_gt = doc_id in existing_doc_ids
        all_errors_for_file = hard_errors + dup_errors

        if already_in_gt and not hard_errors and not accepted:
            print(f"  -> {doc_id} [skip, sudah ada di GT]")
        else:
            status = "OK" if not all_errors_for_file else "FAIL"
            already = " [sudah ada di GT]" if already_in_gt else ""
            print(
                f"  {status} {doc_id}{already}: "
                f"{len(accepted)} valid, {len(all_errors_for_file)} errors"
            )
            for msg in all_errors_for_file:
                print(f"    {msg}")
            total_hard_errors += len(all_errors_for_file)

        all_valid.extend(accepted)

    print(f"\nTotal: {len(all_valid)} pertanyaan valid, {total_hard_errors} hard errors, {audit_drops_total} dropped via audit")

    if total_hard_errors > 0 and not args.force_merge:
        print("\nAda hard errors. Perbaiki sebelum merge atau gunakan --force-merge.")
        sys.exit(1)

    if total_hard_errors > 0 and args.force_merge:
        print("\n[force-merge aktif, item ber-error tidak akan di-merge]")

    if args.check_only:
        print("\n[check-only] Tidak menulis output.")
        return

    if not all_valid:
        print("\nTidak ada item baru untuk ditambahkan.")
        return

    merged_gt = dict(existing_gt)
    for item in all_valid:
        qid = assign_query_id(merged_gt)
        query_type = item.get("query_type", "factual")
        anchor_ids = item.get("gold_anchor_node_ids") or [item["gold_anchor_node_id"]]
        doc_ids = item.get("gold_doc_ids") or [item["gold_doc_id"]]
        merged_gt[qid] = {
            "query": item["query"],
            "query_type": query_type,
            "query_style": item.get("query_style", ""),
            "reference_mode": item["reference_mode"],
            "gold_anchor_granularity": item["gold_anchor_granularity"],
            "gold_anchor_node_id": item["gold_anchor_node_id"],
            "gold_anchor_node_ids": list(anchor_ids),
            "gold_node_id": item["gold_node_id"],
            "gold_doc_id": item["gold_doc_id"],
            "gold_doc_ids": list(doc_ids),
            "navigation_path": item["navigation_path"],
            "answer_hint": item.get("answer_hint", ""),
        }

    GT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(GT_FILE, "w", encoding="utf-8") as f:
        json.dump(merged_gt, f, ensure_ascii=False, indent=2)

    added = len(merged_gt) - len(existing_gt)
    print(f"\nDisimpan ke {GT_FILE}")
    print(f"  Sebelumnya : {len(existing_gt)} pertanyaan")
    print(f"  Ditambahkan: {added} pertanyaan")
    print(f"  Total      : {len(merged_gt)} pertanyaan")


if __name__ == "__main__":
    main()
