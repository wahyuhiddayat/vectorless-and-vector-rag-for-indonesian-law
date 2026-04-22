"""Granularity validation via deterministic sequence detection.

For each leaf in ayat and full_split indexes, use the same fuzzy-marker
sequence detector that the re-split pipeline uses. If the detector finds
a valid consecutive sequence (e.g. ayat (1)(2)(3) or huruf a.b.c.) in a
leaf's text, that leaf should have split but didn't — re-split bug or
novel pattern the splitter missed.

Reuses vectorless.indexing.parser._find_fuzzy_markers so the suspect
criteria exactly matches the actual splitter logic. Zero cost (regex).

Usage:
    python scripts/parser/check_granularity.py --doc-id uu-8-2025
    python scripts/parser/check_granularity.py --category UU
    python scripts/parser/check_granularity.py --category UU --verbose
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vectorless.indexing.parser import _find_fuzzy_markers  # noqa: E402

REGISTRY_PATH = REPO_ROOT / "data" / "raw" / "registry.json"
INDEX_DIR = {
    "pasal": REPO_ROOT / "data" / "index_pasal",
    "ayat": REPO_ROOT / "data" / "index_ayat",
    "full_split": REPO_ROOT / "data" / "index_full_split",
}
REPORT_PATH = REPO_ROOT / "data" / "granularity_report.json"

# Which sub-marker kinds each granularity should NOT leave unsplit:
#   ayat: no inline (N) markers should remain
#   full_split: no ayat/huruf/angka markers should remain anywhere
SUSPECT_KINDS = {
    "ayat": [("ayat", "1")],
    "full_split": [("ayat", "1"), ("huruf", "a"), ("angka", "1")],
}


def _iter_leaves(nodes: list[dict]):
    for n in nodes:
        if n.get("nodes"):
            yield from _iter_leaves(n["nodes"])
        else:
            yield n


def _find_leaf_suspect(leaf: dict, kinds: list[tuple[str, str]]) -> dict | None:
    """Return suspect record if any kind detects a consecutive sequence."""
    text = leaf.get("text") or ""
    if len(text) < 20:  # too short to plausibly contain a sub-sequence
        return None
    for kind, start in kinds:
        markers = _find_fuzzy_markers(text, kind, start)
        if markers and len(markers) >= 2:
            first_pos = markers[0][0]
            excerpt = text[max(0, first_pos - 30):first_pos + 150].replace("\n", " / ")
            return {
                "node_id": leaf.get("node_id"),
                "title": leaf.get("title"),
                "kind": kind,
                "labels": [m[1] for m in markers[:12]],
                "count": len(markers),
                "excerpt": excerpt[:220],
            }
    return None


def _find_doc_path(doc_id: str, granularity: str) -> Path | None:
    for p in INDEX_DIR[granularity].glob(f"*/{doc_id}.json"):
        return p
    return None


def check_doc(doc_id: str) -> dict:
    report: dict = {"doc_id": doc_id}
    total_suspects = 0

    for gran in ("pasal", "ayat", "full_split"):
        path = _find_doc_path(doc_id, gran)
        if not path:
            report[gran] = {"error": "index not found"}
            continue
        doc = json.load(open(path, encoding="utf-8"))
        leaves = list(_iter_leaves(doc.get("structure", [])))
        entry: dict = {
            "leaf_count": len(leaves),
            "total_chars": sum(len(leaf.get("text") or "") for leaf in leaves),
        }
        if gran in SUSPECT_KINDS:
            suspects = []
            for leaf in leaves:
                s = _find_leaf_suspect(leaf, SUSPECT_KINDS[gran])
                if s:
                    suspects.append(s)
            entry["suspects"] = suspects
            total_suspects += len(suspects)
        report[gran] = entry

    # Lossless sanity: pasal ≤ ayat ≤ full_split in leaf count; total chars
    # roughly comparable (re-split preserves text up to OCR-header stripping).
    counts = {g: report.get(g, {}).get("leaf_count") for g in ("pasal", "ayat", "full_split")}
    invariants_ok = (
        counts["pasal"] is not None
        and counts["ayat"] is not None
        and counts["full_split"] is not None
        and counts["pasal"] <= counts["ayat"] <= counts["full_split"]
    )
    report["invariants_ok"] = invariants_ok
    report["suspect_count"] = total_suspects
    report["verdict"] = (
        "OK" if total_suspects == 0 and invariants_ok
        else "SUSPECT" if total_suspects > 0 or not invariants_ok
        else "FAIL"
    )
    return report


def _resolve_targets(doc_ids: list[str], category: str | None) -> list[str]:
    if doc_ids:
        return doc_ids
    if not category:
        raise SystemExit("must pass --doc-id(s) or --category")
    reg = json.load(open(REGISTRY_PATH, encoding="utf-8"))
    target = category.upper()
    return sorted(
        did for did, entry in reg.items()
        if (entry.get("jenis_folder") or "").upper() == target
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Granularity validation via sequence scan")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[])
    ap.add_argument("--doc-ids", dest="doc_ids_csv", default="")
    ap.add_argument("--category")
    ap.add_argument("--verbose", action="store_true",
                    help="Print per-suspect details")
    args = ap.parse_args()

    doc_ids = list(args.doc_ids)
    if args.doc_ids_csv:
        doc_ids.extend([x.strip() for x in args.doc_ids_csv.split(",") if x.strip()])
    targets = _resolve_targets(doc_ids, args.category)

    print(f"checking {len(targets)} docs\n")
    reports = []
    for did in targets:
        r = check_doc(did)
        reports.append(r)
        counts = " / ".join(
            str(r.get(g, {}).get("leaf_count", "?"))
            for g in ("pasal", "ayat", "full_split")
        )
        print(f"  {r['verdict']:8s} {did:15s} leaves={counts:12s} suspects={r['suspect_count']}")
        if args.verbose and r["suspect_count"]:
            for gran in ("ayat", "full_split"):
                for s in (r.get(gran) or {}).get("suspects", []):
                    labels = ",".join(str(x) for x in s["labels"])
                    print(f"    [{gran}] {s['title']}: {s['kind']}=[{labels}]")
                    print(f"      ...{s['excerpt']}...")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump({"docs": reports}, f, ensure_ascii=False, indent=2)

    ok_count = sum(1 for r in reports if r["verdict"] == "OK")
    total_susp = sum(r["suspect_count"] for r in reports)
    print(f"\n{ok_count}/{len(reports)} OK  total suspects={total_susp}")
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
