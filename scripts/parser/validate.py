"""Validate parser output against raw PDF for all indexed documents.

Pure regex-based validation with four metrics per document:
    1. completeness  — actual_pasal_count / expected_pasal_count (from raw PDF)
    2. monotonic     — pasal numbers appear in ascending order
    3. gap_count     — number of missing pasals in the sequence
    4. bleed_count   — leaf nodes whose text contains an embedded Pasal heading

Amendment documents (is_perubahan=True) skip monotonic/gap checks since the
underlying legal structure intentionally uses scattered pasal numbers.

Output:
    data/parser_quality_report.json  — machine-readable per-doc report
    Console summary table

Usage:
    python scripts/parser/validate.py
    python scripts/parser/validate.py --category UU         # filter
    python scripts/parser/validate.py --verbose             # show issues per doc
"""

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vectorless.indexing.parser import extract_pages  # noqa: E402
from scripts._shared import find_pdf_path as _shared_find_pdf_path  # noqa: E402

# Lenient Pasal heading detector for raw PDF text.
# Matches standalone "Pasal N" lines including OCR variants:
#   "Pasal 5", "Pasal 10A", "Pasa1 7" (OCR l→1), "Pasa17" (OCR drops space)
# Does NOT match inline cross-references like "Pasal 5 diubah sehingga..."
# because those have trailing content on the same line.
PASAL_STANDALONE_RE = re.compile(
    r"(?m)^\s*[Pp]asa[l1]\s*(\d+[A-Z]?)\s*[']?\s*$"
)

# Leaf text containing an embedded Pasal heading = parser failed to split.
# Excludes amendment "Angka N — ..." nodes which legitimately quote pasal text.
EMBEDDED_PASAL_RE = re.compile(
    r"\n\s*[Pp]asa[l1]\s*\d+[A-Z]?\s*\n\s*[a-zA-Z]"
)

# Deeper-split indicators: structural markers embedded in leaf text that suggest
# the parser stopped splitting too early.
# Ayat markers "(1)", "(2)" at line start followed by uppercase letter.
UNSPLIT_AYAT_RE = re.compile(r"(?m)^\s*\((\d+)\)\s+[A-Z]")
# Huruf markers "a.", "b." at line start followed by uppercase letter.
UNSPLIT_HURUF_RE = re.compile(r"(?m)^\s*([a-z])\.\s+[A-Z]")
# Angka markers "1.", "2." at line start followed by uppercase letter + lowercase.
UNSPLIT_ANGKA_RE = re.compile(r"(?m)^\s*(\d+)\.\s+[A-Z][a-z]")

INDEX_DIR = Path("data/index_pasal")
REPORT_PATH = Path("data/parser_quality_report.json")


def count_pasal_markers_in_pdf(pdf_path: str) -> set[str]:
    """Extract unique pasal numbers from raw PDF text using lenient regex."""
    pages = extract_pages(pdf_path)
    nums: set[str] = set()
    for page in pages:
        for match in PASAL_STANDALONE_RE.finditer(page.get("raw_text", "")):
            nums.add(match.group(1).upper())
    return nums


def extract_pasal_info(index_data: dict) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str, str]], list[tuple[str, str]]]:
    """Return (ordered_pasal_numbers, leaf_bleeds, underspilts, empty_pasals).

    Walks the index tree and collects:
    - Pasal numbers at any depth
    - Leaf bleeds (embedded Pasal heading in body)
    - Underspilts: leaf nodes whose text contains 2+ structural markers that
      should have been split deeper (ayat/huruf/angka)
    - Empty pasals: Pasal-level leaf nodes whose text is missing or essentially
      just the title repeated (indicates parser failed to capture body content).
    """
    pasal_nums: list[str] = []
    leaf_bleeds: list[tuple[str, str]] = []
    underspilts: list[tuple[str, str, str]] = []
    empty_pasals: list[tuple[str, str]] = []

    def walk(nodes: list[dict], in_angka_ancestor: bool = False) -> None:
        for node in nodes:
            title = node.get("title", "")
            is_angka = title.startswith("Angka ") or in_angka_ancestor
            if title.startswith("Pasal "):
                raw = title.replace("Pasal", "").strip()
                m = re.match(r"(\d+[A-Z]?)", raw)
                if m:
                    pasal_nums.append(m.group(1))
            if "nodes" in node:
                walk(node["nodes"], in_angka_ancestor=is_angka)
            else:
                if is_angka:
                    continue  # amendment body legitimately quotes markers
                text = node.get("text", "")
                node_id = node.get("node_id", "?")
                # Empty content detector for Pasal-level leaves (no ayat children):
                # title starts with "Pasal N" AND text is very short OR equals title.
                is_pasal_leaf = bool(re.match(r"^Pasal\s+\d", title)) and not any(x in title for x in ("Ayat", "Huruf", "Angka"))
                stripped_text = (text or "").strip()
                if is_pasal_leaf and (len(stripped_text) < 30 or stripped_text == title.strip()):
                    empty_pasals.append((node_id, title[:40]))
                if EMBEDDED_PASAL_RE.search(text):
                    leaf_bleeds.append((node_id, title[:40]))
                has_ayat_in_title = "Ayat" in title
                has_huruf_in_title = "Huruf" in title
                has_angka_in_title = title.endswith("Angka ") or " Angka " in title
                ayat_nums = {m.group(1) for m in UNSPLIT_AYAT_RE.finditer(text)}
                huruf_nums = {m.group(1) for m in UNSPLIT_HURUF_RE.finditer(text)}
                angka_nums = {m.group(1) for m in UNSPLIT_ANGKA_RE.finditer(text)}
                if len(ayat_nums) >= 2 and not has_ayat_in_title:
                    underspilts.append((node_id, title[:40], f"unsplit_ayat:{sorted(ayat_nums)[:5]}"))
                elif len(huruf_nums) >= 3 and not has_huruf_in_title:
                    underspilts.append((node_id, title[:40], f"unsplit_huruf:{sorted(huruf_nums)[:5]}"))
                elif len(angka_nums) >= 3 and not has_angka_in_title and not has_huruf_in_title:
                    underspilts.append((node_id, title[:40], f"unsplit_angka:{sorted(angka_nums, key=int)[:5]}"))

    walk(index_data.get("structure", []))
    return pasal_nums, leaf_bleeds, underspilts, empty_pasals


def find_pdf_path(doc_id: str, jenis_folder: str | None = None) -> Path | None:
    """Locate the main PDF for a doc (thin wrapper around scripts._shared)."""
    return _shared_find_pdf_path(doc_id)


def parse_pasal_number(raw: str) -> int | None:
    """Return the integer part of a pasal number, or None for unparseable."""
    m = re.match(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def validate_doc(doc_id: str, index_path: Path) -> dict:
    """Compute quality metrics for one document."""
    index_data = json.load(open(index_path, encoding="utf-8"))
    is_perubahan = bool(index_data.get("is_perubahan", False))
    jenis_folder = index_data.get("jenis_folder")

    actual_pasals, leaf_bleeds, underspilts, empty_pasals = extract_pasal_info(index_data)
    actual_nums_int = [parse_pasal_number(p) for p in actual_pasals]
    actual_nums_int = [n for n in actual_nums_int if n is not None]

    pdf_path = find_pdf_path(doc_id, jenis_folder)
    expected_nums: set[str] = set()
    if pdf_path and pdf_path.exists():
        try:
            expected_nums = count_pasal_markers_in_pdf(str(pdf_path))
        except Exception as exc:
            return {
                "doc_id": doc_id,
                "status": "ERROR",
                "score": 0.0,
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    expected_count = len(expected_nums)
    actual_count = len(set(actual_pasals))
    completeness = (actual_count / expected_count) if expected_count else 1.0
    completeness = min(completeness, 1.0)

    # Monotonic ordering check (only meaningful for non-amendment docs).
    monotonic = True
    gap_list: list[int] = []
    if not is_perubahan and actual_nums_int:
        monotonic = actual_nums_int == sorted(actual_nums_int)
        full_range = set(range(min(actual_nums_int), max(actual_nums_int) + 1))
        gap_list = sorted(full_range - set(actual_nums_int))

    bleed_count = len(leaf_bleeds)
    underspilt_count = len(underspilts)
    empty_pasal_count = len(empty_pasals)

    # Score: weighted combination, max 100.
    score = 100.0 * completeness
    if not monotonic:
        score -= 15.0
    if gap_list:
        score -= min(15.0, len(gap_list) * 2.0)
    if bleed_count:
        score -= min(20.0, bleed_count * 5.0)
    # Empty-pasal penalty is harsh: body missing means retrieval cannot work.
    if empty_pasal_count:
        score -= min(25.0, empty_pasal_count * 8.0)
    # Underspilt penalty is softer: content is correct, just coarser granularity.
    if underspilt_count:
        score -= min(10.0, underspilt_count * 1.0)
    score = max(0.0, round(score, 1))

    if score >= 90:
        status = "OK"
    elif score >= 70:
        status = "PARTIAL"
    else:
        status = "FAIL"

    return {
        "doc_id": doc_id,
        "status": status,
        "score": score,
        "is_perubahan": is_perubahan,
        "expected_pasal_count": expected_count,
        "actual_pasal_count": actual_count,
        "completeness": round(completeness, 3),
        "monotonic": monotonic,
        "gap_count": len(gap_list),
        "gap_list": gap_list[:10],
        "bleed_count": bleed_count,
        "bleed_nodes": [n for n, _ in leaf_bleeds[:5]],
        "empty_pasal_count": empty_pasal_count,
        "empty_pasal_nodes": [f"{nid}:{t}" for nid, t in empty_pasals[:5]],
        "underspilt_count": underspilt_count,
        "underspilt_samples": [f"{nid}:{kind}" for nid, _, kind in underspilts[:5]],
    }


def main() -> None:
    """Validate all (or filtered) docs and write report."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--category", type=str, help="Filter by category folder name")
    ap.add_argument("--verbose", action="store_true", help="Print issues per doc")
    args = ap.parse_args()

    results: list[dict] = []
    for cat_dir in sorted(INDEX_DIR.iterdir()):
        if not cat_dir.is_dir():
            continue
        if args.category and cat_dir.name.upper() != args.category.upper():
            continue
        for doc_path in sorted(cat_dir.glob("*.json")):
            if doc_path.name == "catalog.json":
                continue
            results.append(validate_doc(doc_path.stem, doc_path))

    # Aggregate stats.
    by_status = {"OK": 0, "PARTIAL": 0, "FAIL": 0, "ERROR": 0}
    for r in results:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    total = len(results)

    # Print table header and rows.
    print(f"\n=== Parser Quality Report ({total} docs) ===\n")
    print(f"  OK      : {by_status['OK']:3d}  ({100*by_status['OK']/total:.1f}%)")
    print(f"  PARTIAL : {by_status['PARTIAL']:3d}  ({100*by_status['PARTIAL']/total:.1f}%)")
    print(f"  FAIL    : {by_status['FAIL']:3d}  ({100*by_status['FAIL']/total:.1f}%)")
    if by_status["ERROR"]:
        print(f"  ERROR   : {by_status['ERROR']:3d}")
    print()

    non_ok = [r for r in results if r["status"] != "OK"]
    if non_ok:
        print("--- Non-OK docs ---")
        for r in sorted(non_ok, key=lambda x: x["score"]):
            tag = "AMEND" if r.get("is_perubahan") else "REG"
            print(
                f"  [{r['status']:7}] {r['doc_id']:<28} "
                f"score={r['score']:5.1f} "
                f"[{tag}] "
                f"expected={r.get('expected_pasal_count', '?')} "
                f"actual={r.get('actual_pasal_count', '?')} "
                f"gaps={r.get('gap_count', 0)} "
                f"bleed={r.get('bleed_count', 0)} "
                f"empty={r.get('empty_pasal_count', 0)} "
                f"underspilt={r.get('underspilt_count', 0)}"
            )
            if args.verbose and r.get("gap_list"):
                print(f"              gap_list: {r['gap_list']}")
            if args.verbose and r.get("bleed_nodes"):
                print(f"              bleed_nodes: {r['bleed_nodes']}")
            if args.verbose and r.get("underspilt_samples"):
                for sample in r["underspilt_samples"]:
                    print(f"              underspilt: {sample}")

    # Write machine-readable report.
    report = {
        "summary": {
            "total_docs": total,
            "by_status": by_status,
            "ok_pct": round(100 * by_status["OK"] / total, 1) if total else 0,
        },
        "docs": sorted(results, key=lambda r: (r["status"], -r["score"])),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Report saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
