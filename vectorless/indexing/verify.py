import argparse
import json
import re
import sys
from pathlib import Path

from .status import (
    apply_verify_results,
    load_registry,
    load_status_manifest,
    normalize_categories,
    sync_manifest_from_indexes,
    write_status_manifest,
)

GRANULARITY_INDEX_MAP = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "full_split": Path("data/index_full_split"),
}
PARSER_VERSION = "2026-04-02"
LLM_CLEANUP_VERSION = "2026-04-02"

# Detect recurring PDF headers that survive OCR cleanup and bleed into node text.
# Negative lookbehind/lookahead exclude legitimate uses:
#   "PERATURAN PRESIDEN REPUBLIK INDONESIA" (document title)
#   "PRESIDEN REPUBLIK INDONESIA, ttd..." (closing signature with comma)
OCR_LEAK_PATTERNS = [
    re.compile(r'(?<!PERATURAN )PRESIDEN\s+REPUBLIK\s+INDONESIA(?!\s*,)'),
    re.compile(r'(?<!PERATURAN )PRESIDEN\s*\nREPUBLIK\s+INDONESIA(?!\s*,)'),
    re.compile(r'(?<!TAMBAHAN )LEMBARAN NEGARA(?!\s+REPUBLIK)'),
    re.compile(r'TAMBAHAN LEMBARAN NEGARA'),
]


def collect_leaves(nodes: list[dict]) -> list[dict]:
    """Recursively collect all leaf nodes from a document structure.

    A leaf node has no children (no "nodes" key or empty "nodes" list).
    """
    leaves = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            leaves.extend(collect_leaves(node["nodes"]))
        else:
            leaves.append(node)
    return leaves


def check_nav_paths(nodes: list[dict], ancestors: list[str] | None = None) -> list[str]:
    """Validate that each node's navigation_path matches its actual tree position.

    Expected format: "Grandparent > Parent > Title". Returns a list of mismatch
    descriptions, or an empty list if all paths are correct.
    """
    if ancestors is None:
        ancestors = []
    issues = []
    status_issues = []
    # Check each node's navigation_path against its reconstructed ancestor chain.
    for node in nodes:
        expected = " > ".join(ancestors + [node["title"]])
        actual = node.get("navigation_path", "")
        if actual != expected:
            issues.append(f"Nav mismatch on {node.get('node_id', '?')}: expected '{expected[:60]}', got '{actual[:60]}'")
        if "nodes" in node and node["nodes"]:
            issues.extend(check_nav_paths(node["nodes"], ancestors + [node["title"]]))
    return issues


def check_page_boundaries(nodes: list[dict], parent_start=None, parent_end=None) -> list[str]:
    """Validate that start_index <= end_index for all nodes.

    Returns a list of violation descriptions, or an empty list if all
    boundaries are sane.
    """
    issues = []
    # Check each node for inverted page boundaries.
    for node in nodes:
        start = node.get("start_index")
        end = node.get("end_index")
        nid = node.get("node_id", "?")

        if start is not None and end is not None and start > end:
            issues.append(f"Node {nid}: start_index ({start}) > end_index ({end})")

        if "nodes" in node and node["nodes"]:
            issues.extend(check_page_boundaries(node["nodes"], start, end))
    return issues


def _find_preamble_sections(structure: list[dict]) -> dict:
    """Locate Menimbang, Mengingat, and Menetapkan nodes from the Pembukaan.

    Returns a dict with keys "menimbang", "mengingat", "menetapkan", each
    either a node dict or None if the section does not exist.
    """
    result = {"menimbang": None, "mengingat": None, "menetapkan": None}
    if not structure:
        return result
    pembukaan = structure[0]
    if "Pembukaan" not in pembukaan.get("title", ""):
        return result
    # Match Pembukaan children by their exact title.
    for child in pembukaan.get("nodes", []):
        title = child.get("title", "")
        if title == "Menimbang":
            result["menimbang"] = child
        elif title == "Mengingat":
            result["mengingat"] = child
        elif title == "Menetapkan":
            result["menetapkan"] = child
    return result


def _node_last_text(node: dict) -> str:
    """Return the text of the deepest last leaf under a node."""
    children = node.get("nodes")
    if children:
        return _node_last_text(children[-1])
    return node.get("text", "")


def _node_first_text(node: dict) -> str:
    """Return the text of the deepest first leaf under a node."""
    children = node.get("nodes")
    if children:
        return _node_first_text(children[0])
    return node.get("text", "")


def _node_all_text(node: dict) -> str:
    """Concatenate all leaf texts under a node, joined by newlines."""
    children = node.get("nodes")
    if children:
        return "\n".join(_node_all_text(c) for c in children)
    return node.get("text", "")


def _check_ocr_leaks_in_text(text: str) -> bool:
    """Return True if any OCR_LEAK_PATTERNS match inside the given text."""
    for pattern in OCR_LEAK_PATTERNS:
        if pattern.search(text):
            return True
    return False


def check_preamble(structure: list[dict]) -> list[str]:
    """Validate preamble integrity: Menimbang, Mengingat, and Menetapkan sections.

    Checks for boundary bleed, missing content, OCR artifacts, and structural
    issues. Returns a list of issue descriptions, or an empty list if all checks pass.
    """
    issues = []
    secs = _find_preamble_sections(structure)

    if not secs["menimbang"]:
        issues.append('Preamble: Menimbang section missing')
        return issues

    menimbang = secs["menimbang"]
    mengingat = secs["mengingat"]
    menetapkan = secs["menetapkan"]

    # -- Menimbang checks --

    # Menimbang should contain at least one "bahwa" clause.
    menimbang_all = _node_all_text(menimbang)
    if not re.search(r'\bbahwa\b', menimbang_all, re.IGNORECASE):
        issues.append('Preamble: Menimbang contains no "bahwa" clause (content may be garbled or misplaced)')

    # Boundary bleed: "Mengingat" keyword in the last Menimbang text chunk.
    last_text = _node_last_text(menimbang)
    if re.search(r'\bMengingat\b', last_text):
        issues.append('Preamble: Menimbang last text contains "Mengingat" keyword (boundary bleed)')
    if re.search(r'\bMengingat\s*:?\s*\d', last_text):
        issues.append('Preamble: Menimbang last text contains an absorbed Mengingat point (e.g. "Mengingat : 1.")')

    # OCR header leak in Menimbang text.
    if _check_ocr_leaks_in_text(menimbang_all):
        issues.append('Preamble: Menimbang contains OCR header leak (PRESIDEN/LEMBARAN NEGARA)')

    # -- Mengingat checks --

    if mengingat:
        mengingat_all = _node_all_text(mengingat)

        # Empty section detection.
        if not mengingat_all.strip():
            issues.append('Preamble: Mengingat section exists but has no text')
        else:
            # First child check: missing angka 1.
            first_child = (mengingat.get("nodes") or [None])[0]
            if first_child:
                # At full_split granularity, child titles reveal the numbering.
                title = first_child.get("title", "")
                if re.match(r'Mengingat Angka [2-9]', title):
                    issues.append(f'Preamble: Mengingat starts at "{title}" — angka 1 missing (absorbed into Menimbang?)')
            else:
                # At pasal/ayat granularity, check the raw text start.
                first_text = mengingat.get("text", "")
                if re.match(r'\s*2\.', first_text):
                    issues.append('Preamble: Mengingat text starts at "2." — first point missing (absorbed into Menimbang?)')

            # Mengingat should contain either numbered points ("1. Pasal...") or a direct legal
            # reference ("Pasal ...", "Undang-Undang ..."). Simple Perpres documents often skip numbering.
            has_numbered = re.search(r'(?:^|\n)\s*[1l][\.\s]', mengingat_all)
            has_legal_ref = re.search(r'(?:^|\n)\s*(?:Pasal|Undang|Peraturan)', mengingat_all)
            if not has_numbered and not has_legal_ref:
                issues.append('Preamble: Mengingat has no numbered points or legal references (structure may be garbled)')

            # Ghost "Mengingat" OCR keyword inside Mengingat text.
            if re.search(r'\bMengingat\b', mengingat_all):
                issues.append('Preamble: Mengingat text contains ghost "Mengingat" keyword (OCR bleed)')

            # "Dengan Persetujuan Bersama" boilerplate should have been trimmed.
            if re.search(r'Dengan\s+Persetujuan\s+Bersama', mengingat_all):
                issues.append('Preamble: Mengingat contains "Dengan Persetujuan Bersama" boilerplate (should be trimmed)')

            # Boundary bleed: MEMUTUSKAN or Menetapkan keyword at the end of Mengingat.
            if re.search(r'(?:MEMUTUS\S+|Menetapkan)\s*:?\s*$', mengingat_all.rstrip()):
                issues.append('Preamble: Mengingat ends with MEMUTUSKAN/Menetapkan keyword (boundary bleed)')

            # OCR header leak in Mengingat text.
            if _check_ocr_leaks_in_text(mengingat_all):
                issues.append('Preamble: Mengingat contains OCR header leak (PRESIDEN/LEMBARAN NEGARA)')

    # -- Menetapkan checks --

    if menetapkan:
        menetapkan_all = _node_all_text(menetapkan)

        # Empty section detection.
        if not menetapkan_all.strip():
            issues.append('Preamble: Menetapkan section exists but has no text')
        else:
            # Body text bleed: Pasal headings in Menetapkan indicate the boundary was wrong.
            if re.search(r'^Pasal\s+\d+', menetapkan_all, re.MULTILINE):
                issues.append('Preamble: Menetapkan contains Pasal headings (body text bled into preamble)')

            # Numbered legal references in Menetapkan suggest Mengingat content bled forward.
            if re.search(r'(?:^|\n)\s*\d+\.\s+(?:Pasal|Undang|Peraturan)', menetapkan_all):
                issues.append('Preamble: Menetapkan contains numbered legal references (Mengingat content may have bled)')

            # OCR header leak in Menetapkan text.
            if _check_ocr_leaks_in_text(menetapkan_all):
                issues.append('Preamble: Menetapkan contains OCR header leak (PRESIDEN/LEMBARAN NEGARA)')

    return issues


def categorize_warnings(warnings: list[str]) -> dict:
    """Categorize parser warnings by type based on keyword matching.

    Returns a dict with counts for: non_monotonic, gap, llm_failure, other.
    """
    cats = {"non_monotonic": 0, "gap": 0, "llm_failure": 0, "other": 0}
    # Classify each warning string by its content.
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
    """Run all 7 verification checks on a single document.

    Returns a report dict with: doc_id, status (OK/WARN/FAIL), checks (per-check
    details), and issues (flat list of human-readable problem descriptions).
    """
    doc_id = doc["doc_id"]
    structure = doc.get("structure", [])
    warnings = doc.get("warnings", [])
    element_counts = doc.get("element_counts", {})

    issues = []
    status_issues = []
    checks = {}

    # Check 1: Categorize parser warnings.
    warn_cats = categorize_warnings(warnings)
    checks["warnings"] = {"count": len(warnings), "categories": warn_cats}

    # Check 2: Pasal count — compare reported vs actual leaf nodes.
    leaves = collect_leaves(structure)
    pasal_leaves = [l for l in leaves if re.match(r"Pasal\s+", l.get("title", ""))]
    reported_pasal = element_counts.get("pasal", 0)
    checks["pasal_count"] = {
        "reported": reported_pasal,
        "leaf_nodes": len(leaves),
        "pasal_leaves": len(pasal_leaves),
    }

    # Check 3: Leaf quality — empty, short, long text, OCR leaks.
    empty_text = 0
    short_text = 0
    long_text = 0
    ocr_leaks = 0
    # Scan every leaf node for text quality issues.
    for leaf in leaves:
        text = leaf.get("text", "")
        if not text:
            empty_text += 1
            issue = f"Empty text: {leaf.get('node_id', '?')} ({leaf.get('title', '?')})"
            issues.append(issue)
            status_issues.append(issue)
        elif len(text) < 20:
            short_text += 1
        elif len(text) > 10000:
            long_text += 1
        # Check for OCR header bleed-through in leaf text.
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
        issue = f"{empty_text} leaf node(s) with empty text"
        issues.append(issue)
        status_issues.append(issue)
    if ocr_leaks:
        issue = f"{ocr_leaks} leaf node(s) with OCR header leaks"
        issues.append(issue)
        status_issues.append(issue)

    # Check 4: Navigation path consistency.
    nav_issues = check_nav_paths(structure)
    checks["nav_path_mismatches"] = len(nav_issues)
    if nav_issues:
        issues.extend(nav_issues[:3])
        if len(nav_issues) > 3:
            issues.append(f"...and {len(nav_issues) - 3} more nav mismatches")

    # Check 5: Page boundary sanity (start_index <= end_index).
    boundary_issues = check_page_boundaries(structure)
    checks["boundary_issues"] = len(boundary_issues)
    if boundary_issues:
        issues.extend(boundary_issues[:3])
        status_issues.extend(boundary_issues[:3])
        if len(boundary_issues) > 3:
            more_issue = f"...and {len(boundary_issues) - 3} more boundary issues"
            issues.append(more_issue)
            status_issues.append(more_issue)

    # Check 6: Duplicate node IDs.
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
        issue = f"{dupes} duplicate node_id(s)"
        issues.append(issue)
        status_issues.append(issue)

    # Check 7: Preamble integrity (Menimbang/Mengingat/Menetapkan boundaries and content).
    preamble_issues = check_preamble(structure)
    checks["preamble_issues"] = len(preamble_issues)
    if preamble_issues:
        issues.extend(preamble_issues)
        status_issues.extend(preamble_issues)

    # Determine overall status from issue severity.
    if empty_text or dupes or len(boundary_issues) > 0:
        status = "FAIL"
    elif len(warnings) > 0 or ocr_leaks > 0 or len(preamble_issues) > 0:
        status = "WARN"
    else:
        status = "OK"

    checks["issue_count"] = len(status_issues)
    checks["diagnostic_issue_count"] = len(issues)

    return {
        "doc_id": doc_id,
        "status": status,
        "checks": checks,
        "issues": issues,
    }


def verify_index(index_dir: Path, doc_id: str | None = None, category: str | None = None) -> list[dict]:
    """Verify all (or one specific) documents in an index directory.

    Loads each JSON file, runs verify_doc(), and returns a list of report dicts.
    Skips catalog.json and documents that don't match doc_id (when specified).
    """
    results = []
    categories = normalize_categories(category)
    # Walk all JSON files in the index directory.
    for path in sorted(index_dir.rglob("*.json")):
        if path.name.startswith("catalog"):
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if not isinstance(doc, dict) or "doc_id" not in doc:
            continue
        if doc_id and doc["doc_id"] != doc_id:
            continue
        if categories:
            doc_category = (doc.get("jenis_folder") or doc["doc_id"].split("-")[0]).upper()
            if doc_category not in categories:
                continue
        results.append(verify_doc(doc))
    return results


def cross_granularity_check(doc_id: str) -> list[str]:
    """Compare leaf counts across granularities for one document.

    Finer granularities must have >= the leaf count of coarser ones
    (pasal <= ayat <= full_split). Returns a list of violations.
    """
    issues = []
    counts = {}
    # Load leaf count from each available granularity index.
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
    """Print a human-readable verification report for one index directory.

    Shows [OK], [WARN], or [FAIL] per document with details for non-OK statuses.
    Ends with a summary line.
    """
    print(f"\n{'='*60}")
    print(f"INDEX VERIFICATION REPORT")
    print(f"Index: {index_dir}/ ({len(results)} documents)")
    print(f"{'='*60}\n")

    ok = warn = fail = 0
    # Print one line per document, with details for WARN/FAIL.
    for r in results:
        status = r["status"]
        doc_id = r["doc_id"]
        leaves = r["checks"]["leaf_quality"]["total_leaves"]
        parser_warn_count = r["checks"]["warnings"]["count"]
        issue_count = r["checks"].get("issue_count", len(r["issues"]))

        if status == "OK":
            ok += 1
            print(f"[OK]   {doc_id:30s}  {leaves:4d} leaves, {issue_count} issues, {parser_warn_count} parser warnings")
        elif status == "WARN":
            warn += 1
            print(f"[WARN] {doc_id:30s}  {leaves:4d} leaves, {issue_count} issues, {parser_warn_count} parser warnings")
            # Build a compact summary of warning categories.
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
            if r["checks"].get("preamble_issues", 0):
                parts.append(f"{r['checks']['preamble_issues']} preamble issue(s)")
            if parts:
                print(f"         {', '.join(parts)}")
            # Print each preamble issue on its own line for visibility.
            for issue in r["issues"]:
                if issue.startswith("Preamble:"):
                    print(f"           - {issue}")
        else:
            fail += 1
            print(f"[FAIL] {doc_id:30s}  {leaves:4d} leaves, {issue_count} issues, {parser_warn_count} parser warnings")
            for issue in r["issues"][:5]:
                print(f"         - {issue}")

    print(f"\n{'-'*60}")
    print(f"Summary: {ok} OK, {warn} WARN, {fail} FAIL")


def main():
    """CLI entry point for index verification."""
    ap = argparse.ArgumentParser(description="Verify structural integrity of indexed documents")
    ap.add_argument("--granularity", choices=["pasal", "ayat", "full_split"],
                    help="Granularity to verify")
    ap.add_argument("--all", action="store_true", help="Verify all granularities + cross-compare")
    ap.add_argument("--doc-id", type=str, help="Verify single document")
    ap.add_argument("--category", type=str, help="Verify only selected categories, e.g. UU,PP,PMK")
    ap.add_argument("--json", action="store_true", help="Output as JSON")
    args = ap.parse_args()

    if not args.granularity and not args.all:
        print("ERROR: Specify --granularity or --all")
        sys.exit(1)

    granularities = list(GRANULARITY_INDEX_MAP.keys()) if args.all else [args.granularity]
    registry = load_registry()
    manifest = load_status_manifest(PARSER_VERSION, LLM_CLEANUP_VERSION)
    sync_manifest_from_indexes(
        manifest,
        registry,
        PARSER_VERSION,
        LLM_CLEANUP_VERSION,
        doc_ids=[args.doc_id] if args.doc_id else None,
    )

    all_results = {}
    # Run verification on each requested granularity.
    for gran in granularities:
        index_dir = GRANULARITY_INDEX_MAP[gran]
        if not index_dir.exists():
            print(f"SKIP {gran}: {index_dir} does not exist")
            continue
        results = verify_index(index_dir, args.doc_id, args.category)
        all_results[gran] = results
        apply_verify_results(manifest, gran, results, registry=registry)

        if not args.json:
            print_report(results, index_dir)

    # Cross-granularity: compare leaf counts across pasal/ayat/full_split.
    if args.all and not args.json:
        pasal_dir = GRANULARITY_INDEX_MAP["pasal"]
        if pasal_dir.exists():
            doc_ids = set()
            # Collect all doc_ids from the pasal index as the reference set.
            for path in pasal_dir.rglob("*.json"):
                if path.name != "catalog.json":
                    with open(path, encoding="utf-8") as f:
                        doc = json.load(f)
                        if args.category:
                            doc_category = (doc.get("jenis_folder") or doc["doc_id"].split("-")[0]).upper()
                            if doc_category not in normalize_categories(args.category):
                                continue
                        doc_ids.add(doc["doc_id"])

            cross_issues = []
            # Check each document for leaf count inversions between granularities.
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

    write_status_manifest(manifest)


if __name__ == "__main__":
    main()
