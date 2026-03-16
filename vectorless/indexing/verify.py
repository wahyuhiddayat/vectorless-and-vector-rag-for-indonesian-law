"""
verify.py — Verify structural integrity of indexed legal documents.

Runs automated checks on index JSONs to detect parsing issues:
  - Warning counts and categorization
  - Pasal count validation (element_counts vs actual tree)
  - Numbering completeness (gap detection)
  - Navigation path consistency
  - Leaf node quality (empty/short/long text, OCR leaks)
  - Page boundary sanity
  - Cross-granularity comparison

Usage:
    python -m vectorless.indexing.verify --granularity pasal
    python -m vectorless.indexing.verify --granularity pasal --doc-id uu-20-2025
    python -m vectorless.indexing.verify --all
    python -m vectorless.indexing.verify --granularity pasal --json
"""

import argparse
import json
import re
import sys
from pathlib import Path

GRANULARITY_INDEX_MAP = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "full_split": Path("data/index_full_split"),
}

OCR_LEAK_PATTERNS = [
    # Detect repeated PDF header "PRESIDEN REPUBLIK INDONESIA" leaked into text.
    # Exclude legitimate uses: closing signature and preambul always have a comma
    # ("PRESIDEN REPUBLIK INDONESIA, ttd..." or "PRESIDEN REPUBLIK INDONESIA, bahwa...")
    # and document titles ("PERATURAN PRESIDEN REPUBLIK INDONESIA").
    re.compile(r'(?<!PERATURAN )PRESIDEN\s+REPUBLIK\s+INDONESIA(?!\s*,)'),
    re.compile(r'(?<!PERATURAN )PRESIDEN\s*\nREPUBLIK\s+INDONESIA(?!\s*,)'),
    re.compile(r'(?<!TAMBAHAN )LEMBARAN NEGARA(?!\s+REPUBLIK)'),
    re.compile(r'TAMBAHAN LEMBARAN NEGARA'),
]


def collect_leaves(nodes: list[dict]) -> list[dict]:
    """Recursively collect all leaf nodes (no children, has text)."""
    leaves = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            leaves.extend(collect_leaves(node["nodes"]))
        else:
            leaves.append(node)
    return leaves


def check_nav_paths(nodes: list[dict], ancestors: list[str] | None = None) -> list[str]:
    """Check navigation_path consistency. Returns list of mismatches."""
    if ancestors is None:
        ancestors = []
    issues = []
    for node in nodes:
        expected = " > ".join(ancestors + [node["title"]])
        actual = node.get("navigation_path", "")
        if actual != expected:
            issues.append(f"Nav mismatch on {node.get('node_id', '?')}: expected '{expected[:60]}', got '{actual[:60]}'")
        if "nodes" in node and node["nodes"]:
            issues.extend(check_nav_paths(node["nodes"], ancestors + [node["title"]]))
    return issues


def check_page_boundaries(nodes: list[dict], parent_start=None, parent_end=None) -> list[str]:
    """Check page boundary sanity. Returns list of issues."""
    issues = []
    for node in nodes:
        start = node.get("start_index")
        end = node.get("end_index")
        nid = node.get("node_id", "?")

        if start is not None and end is not None and start > end:
            issues.append(f"Node {nid}: start_index ({start}) > end_index ({end})")

        if "nodes" in node and node["nodes"]:
            issues.extend(check_page_boundaries(node["nodes"], start, end))
    return issues


def categorize_warnings(warnings: list[str]) -> dict:
    """Categorize warnings into types."""
    cats = {"non_monotonic": 0, "gap": 0, "llm_failure": 0, "other": 0}
    for w in warnings:
        if "appears after" in w:
            cats["non_monotonic"] += 1
        elif "Gap in Pasal" in w:
            cats["gap"] += 1
        elif "Failed to parse LLM" in w:
            cats["llm_failure"] += 1
        else:
            cats["other"] += 1
    return cats


def verify_doc(doc: dict) -> dict:
    """Run all verification checks on a single document. Returns report dict."""
    doc_id = doc["doc_id"]
    structure = doc.get("structure", [])
    warnings = doc.get("warnings", [])
    element_counts = doc.get("element_counts", {})

    issues = []
    checks = {}

    # 1. Warnings
    warn_cats = categorize_warnings(warnings)
    checks["warnings"] = {"count": len(warnings), "categories": warn_cats}

    # 2. Pasal count validation
    leaves = collect_leaves(structure)
    pasal_leaves = [l for l in leaves if re.match(r"Pasal\s+", l.get("title", ""))]
    reported_pasal = element_counts.get("pasal", 0)
    checks["pasal_count"] = {
        "reported": reported_pasal,
        "leaf_nodes": len(leaves),
        "pasal_leaves": len(pasal_leaves),
    }

    # 3. Leaf quality
    empty_text = 0
    short_text = 0
    long_text = 0
    ocr_leaks = 0
    for leaf in leaves:
        text = leaf.get("text", "")
        if not text:
            empty_text += 1
            issues.append(f"Empty text: {leaf.get('node_id', '?')} ({leaf.get('title', '?')})")
        elif len(text) < 20:
            short_text += 1
        elif len(text) > 10000:
            long_text += 1
        for pattern in OCR_LEAK_PATTERNS:
            if pattern.search(text):
                ocr_leaks += 1
                break
    checks["leaf_quality"] = {
        "total_leaves": len(leaves),
        "empty_text": empty_text,
        "short_text": short_text,
        "long_text": long_text,
        "ocr_leaks": ocr_leaks,
    }
    if empty_text:
        issues.append(f"{empty_text} leaf node(s) with empty text")
    if ocr_leaks:
        issues.append(f"{ocr_leaks} leaf node(s) with OCR header leaks")

    # 4. Navigation path consistency
    nav_issues = check_nav_paths(structure)
    checks["nav_path_mismatches"] = len(nav_issues)
    if nav_issues:
        issues.extend(nav_issues[:3])  # Show first 3
        if len(nav_issues) > 3:
            issues.append(f"...and {len(nav_issues) - 3} more nav mismatches")

    # 5. Page boundary sanity
    boundary_issues = check_page_boundaries(structure)
    checks["boundary_issues"] = len(boundary_issues)
    if boundary_issues:
        issues.extend(boundary_issues[:3])
        if len(boundary_issues) > 3:
            issues.append(f"...and {len(boundary_issues) - 3} more boundary issues")

    # 6. Duplicate node_ids
    all_ids = []
    def collect_ids(nodes):
        for n in nodes:
            all_ids.append(n.get("node_id"))
            if "nodes" in n:
                collect_ids(n["nodes"])
    collect_ids(structure)
    dupes = len(all_ids) - len(set(all_ids))
    if dupes:
        checks["duplicate_node_ids"] = dupes
        issues.append(f"{dupes} duplicate node_id(s)")

    # Determine status
    if empty_text or dupes or len(boundary_issues) > 0:
        status = "FAIL"
    elif len(warnings) > 10 or ocr_leaks > 0:
        status = "WARN"
    elif len(warnings) > 0:
        status = "WARN"
    else:
        status = "OK"

    return {
        "doc_id": doc_id,
        "status": status,
        "checks": checks,
        "issues": issues,
    }


def verify_index(index_dir: Path, doc_id: str | None = None) -> list[dict]:
    """Verify all (or one) documents in an index directory."""
    results = []
    for path in sorted(index_dir.rglob("*.json")):
        if path.name == "catalog.json":
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc_id and doc["doc_id"] != doc_id:
            continue
        results.append(verify_doc(doc))
    return results


def cross_granularity_check(doc_id: str) -> list[str]:
    """Compare leaf counts across granularities for the same doc."""
    issues = []
    counts = {}
    for gran, idx_dir in GRANULARITY_INDEX_MAP.items():
        cat = doc_id.split("-")[0].upper()
        path = idx_dir / cat / f"{doc_id}.json"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        counts[gran] = len(collect_leaves(doc.get("structure", [])))

    if "pasal" in counts and "ayat" in counts:
        if counts["ayat"] < counts["pasal"]:
            issues.append(f"Ayat ({counts['ayat']}) < Pasal ({counts['pasal']}) leaves")
    if "ayat" in counts and "full_split" in counts:
        if counts["full_split"] < counts["ayat"]:
            issues.append(f"Full_split ({counts['full_split']}) < Ayat ({counts['ayat']}) leaves")

    return issues


def print_report(results: list[dict], index_dir: Path):
    """Print human-readable verification report."""
    print(f"\n{'='*60}")
    print(f"INDEX VERIFICATION REPORT")
    print(f"Index: {index_dir}/ ({len(results)} documents)")
    print(f"{'='*60}\n")

    ok = warn = fail = 0
    for r in results:
        status = r["status"]
        doc_id = r["doc_id"]
        leaves = r["checks"]["leaf_quality"]["total_leaves"]
        warn_count = r["checks"]["warnings"]["count"]

        if status == "OK":
            ok += 1
            print(f"[OK]   {doc_id:30s}  {leaves:4d} leaves, {warn_count} warnings")
        elif status == "WARN":
            warn += 1
            print(f"[WARN] {doc_id:30s}  {leaves:4d} leaves, {warn_count} warnings")
            cats = r["checks"]["warnings"]["categories"]
            lq = r["checks"]["leaf_quality"]
            parts = []
            if cats["non_monotonic"]:
                parts.append(f"{cats['non_monotonic']} non-monotonic")
            if cats["gap"]:
                parts.append(f"{cats['gap']} gaps")
            if cats["llm_failure"]:
                parts.append(f"{cats['llm_failure']} LLM failures")
            if lq["ocr_leaks"]:
                parts.append(f"{lq['ocr_leaks']} OCR leaks")
            if parts:
                print(f"         {', '.join(parts)}")
        else:
            fail += 1
            print(f"[FAIL] {doc_id:30s}  {leaves:4d} leaves, {warn_count} warnings")
            for issue in r["issues"][:5]:
                print(f"         - {issue}")

    print(f"\n{'-'*60}")
    print(f"Summary: {ok} OK, {warn} WARN, {fail} FAIL")


def main():
    ap = argparse.ArgumentParser(description="Verify structural integrity of indexed documents")
    ap.add_argument("--granularity", choices=["pasal", "ayat", "full_split"],
                    help="Granularity to verify")
    ap.add_argument("--all", action="store_true", help="Verify all granularities + cross-compare")
    ap.add_argument("--doc-id", type=str, help="Verify single document")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    if not args.granularity and not args.all:
        print("ERROR: Specify --granularity or --all")
        sys.exit(1)

    granularities = list(GRANULARITY_INDEX_MAP.keys()) if args.all else [args.granularity]

    all_results = {}
    for gran in granularities:
        index_dir = GRANULARITY_INDEX_MAP[gran]
        if not index_dir.exists():
            print(f"SKIP {gran}: {index_dir} does not exist")
            continue
        results = verify_index(index_dir, args.doc_id)
        all_results[gran] = results

        if not args.json:
            print_report(results, index_dir)

    # Cross-granularity check
    if args.all and not args.json:
        # Collect all doc_ids from pasal index
        pasal_dir = GRANULARITY_INDEX_MAP["pasal"]
        if pasal_dir.exists():
            doc_ids = set()
            for path in pasal_dir.rglob("*.json"):
                if path.name != "catalog.json":
                    with open(path, encoding="utf-8") as f:
                        doc_ids.add(json.load(f)["doc_id"])

            cross_issues = []
            for did in sorted(doc_ids):
                issues = cross_granularity_check(did)
                if issues:
                    cross_issues.append((did, issues))

            if cross_issues:
                print(f"\n{'='*60}")
                print("CROSS-GRANULARITY ISSUES")
                print(f"{'='*60}\n")
                for did, issues in cross_issues:
                    for issue in issues:
                        print(f"  {did}: {issue}")
            else:
                print(f"\nCross-granularity: All OK")

    if args.json:
        output = {}
        for gran, results in all_results.items():
            output[gran] = results
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
