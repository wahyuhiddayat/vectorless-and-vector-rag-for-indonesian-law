"""Per-doc interactive author verification logger for ground truth.

Author runs this once per doc after the Judge LLM step. The script walks
through each query in the cleaned raw GT, shows the anchor leaf text, and
records a verdict per item, correct, wrong, borderline, or skip. Output
goes to data/gt_audit/<doc_id>.json so the manual review pass leaves a
reproducible trail.

Usage:
    python scripts/gt/log_review.py <doc_id>
    python scripts/gt/log_review.py --report
    python scripts/gt/log_review.py --report --json
"""

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vectorless.ids import doc_category
from scripts.gt.collect import get_leaf_meta_map

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/ground_truth_raw")
AUDIT_DIR = Path("data/gt_audit")
META_SUFFIX = ".meta.json"

VALID_VERDICTS = {"c": "correct", "w": "wrong", "b": "borderline", "s": "skipped"}


def raw_path_for(doc_id: str) -> Path:
    """Return the path to the raw GT file for a doc_id."""
    return RAW_DIR / doc_category(doc_id) / f"{doc_id}.json"


def audit_path_for(doc_id: str) -> Path:
    """Return the path to the audit log for a doc_id."""
    return AUDIT_DIR / f"{doc_id}.json"


def meta_path_for(doc_id: str) -> Path:
    """Return the path to the provenance meta sidecar for a doc_id."""
    return RAW_DIR / doc_category(doc_id) / f"{doc_id}{META_SUFFIX}"


def load_items(path: Path) -> list[dict]:
    """Load and lightly validate the raw GT item array."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path} top-level must be a JSON array")
    return data


def load_meta(doc_id: str) -> dict:
    """Load the provenance sidecar if it exists, else return defaults."""
    path = meta_path_for(doc_id)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def prompt_verdict(query_label: str) -> tuple[str, str]:
    """Prompt the author for a verdict and optional notes for one item."""
    while True:
        raw = input(f"Verdict {query_label}, c=correct w=wrong b=borderline s=skip q=quit, ").strip().lower()
        if raw == "q":
            raise KeyboardInterrupt
        if raw in VALID_VERDICTS:
            verdict = VALID_VERDICTS[raw]
            notes = input("Notes (enter to skip), ").strip()
            return verdict, notes
        print("  invalid, try again")


def review_doc(doc_id: str, resume: bool) -> dict:
    """Run the interactive review loop for one doc and return the audit dict."""
    raw_path = raw_path_for(doc_id)
    if not raw_path.exists():
        raise SystemExit(f"raw GT not found, {raw_path}")

    items = load_items(raw_path)
    if not items:
        raise SystemExit(f"raw GT is empty, {raw_path}")

    leaf_map = get_leaf_meta_map(doc_id)
    if not leaf_map:
        raise SystemExit(f"doc {doc_id} not found in data/index_rincian, cannot resolve anchor text")

    audit_path = audit_path_for(doc_id)
    existing: dict = {}
    if audit_path.exists() and resume:
        with open(audit_path, encoding="utf-8") as f:
            existing = json.load(f)

    by_qid = {entry["qid"]: entry for entry in existing.get("items", [])}

    meta = load_meta(doc_id)
    print()
    print(f"Doc, {doc_id}")
    print(f"Items, {len(items)}")
    if meta:
        print(f"Annotator model, {meta.get('annotator_model', '<unknown>')}")
        print(f"Judge model    , {meta.get('judge_model', '<unknown>')}")
    print()

    results: list[dict] = []
    for idx, item in enumerate(items, start=1):
        anchor_id = item.get("gold_anchor_node_id") or item.get("gold_node_id") or "<missing>"
        leaf = leaf_map.get(anchor_id, {})
        text = (leaf.get("text") or "").strip()
        if len(text) > 600:
            text = text[:600].rstrip() + " ..."
        nav = leaf.get("navigation_path", "<unknown>")

        if idx in by_qid and resume:
            prev = by_qid[idx]
            print(f"[{idx}/{len(items)}] already reviewed as {prev['verdict']}, kept")
            results.append(prev)
            continue

        print(f"[{idx}/{len(items)}] {item.get('query_style', '?')}, {item.get('difficulty', '?')}, {item.get('reference_mode', '?')}")
        print(f"  Query  , {item.get('query', '')}")
        print(f"  Anchor , {anchor_id}")
        print(f"  Path   , {nav}")
        if item.get("answer_hint"):
            print(f"  Hint   , {item['answer_hint']}")
        print(f"  Gold   , {text or '<empty>'}")
        try:
            verdict, notes = prompt_verdict(f"[{idx}/{len(items)}]")
        except KeyboardInterrupt:
            print("\nQuit, partial progress saved")
            break
        results.append({
            "qid": idx,
            "anchor_node_id": anchor_id,
            "verdict": verdict,
            "notes": notes,
        })
        print()

    audit = {
        "doc_id": doc_id,
        "category": doc_category(doc_id),
        "reviewed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "annotator_model": meta.get("annotator_model"),
        "judge_model": meta.get("judge_model"),
        "prompt_version": meta.get("prompt_version"),
        "n_items": len(items),
        "n_reviewed": len(results),
        "items": results,
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote {audit_path}")
    return audit


def aggregate_report(json_out: bool) -> None:
    """Print or emit a JSON aggregate over all per-doc audit files."""
    if not AUDIT_DIR.exists():
        print("no audit logs yet")
        return

    docs = []
    totals = {"correct": 0, "wrong": 0, "borderline": 0, "skipped": 0}
    by_category: dict[str, dict[str, int]] = {}

    for path in sorted(AUDIT_DIR.glob("*.json")):
        if path.name.startswith("_"):
            continue
        with open(path, encoding="utf-8") as f:
            audit = json.load(f)
        counts = {"correct": 0, "wrong": 0, "borderline": 0, "skipped": 0}
        for entry in audit.get("items", []):
            counts[entry["verdict"]] = counts.get(entry["verdict"], 0) + 1
        for k, v in counts.items():
            totals[k] = totals.get(k, 0) + v
        cat = audit.get("category") or doc_category(audit["doc_id"])
        cat_bucket = by_category.setdefault(cat, {"correct": 0, "wrong": 0, "borderline": 0, "skipped": 0})
        for k, v in counts.items():
            cat_bucket[k] = cat_bucket.get(k, 0) + v

        docs.append({
            "doc_id": audit["doc_id"],
            "category": cat,
            "reviewed_at": audit.get("reviewed_at"),
            "n_items": audit.get("n_items", 0),
            "n_reviewed": audit.get("n_reviewed", 0),
            "counts": counts,
        })

    summary = {
        "total_docs": len(docs),
        "total_items_logged": sum(totals.values()),
        "totals": totals,
        "by_category": by_category,
        "docs": docs,
    }

    summary_path = AUDIT_DIR / "_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
        f.write("\n")

    if json_out:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    print(f"\nGT audit summary, {len(docs)} doc(s) reviewed")
    grand = sum(totals.values()) or 1
    for k in ("correct", "wrong", "borderline", "skipped"):
        v = totals[k]
        pct = 100.0 * v / grand
        print(f"  {k:11s} {v:5d}  ({pct:5.1f}%)")
    print()
    if by_category:
        print("Per category,")
        for cat in sorted(by_category):
            cb = by_category[cat]
            sub = sum(cb.values()) or 1
            acc = 100.0 * cb["correct"] / sub
            print(f"  {cat:20s} n={sub:4d}  correct={acc:5.1f}%  (w={cb['wrong']}, b={cb['borderline']}, s={cb['skipped']})")
    print()
    print("Per doc,")
    for d in docs:
        c = d["counts"]
        sub = sum(c.values()) or 1
        acc = 100.0 * c["correct"] / sub
        print(f"  {d['doc_id']:30s} n={sub:3d}  correct={acc:5.1f}%  reviewed={d['n_reviewed']}/{d['n_items']}")
    print()
    print(f"Wrote {summary_path}")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Interactive per-doc author verification logger.")
    ap.add_argument("doc_id", nargs="?", help="Document ID, e.g. perma-2-2022")
    ap.add_argument("--report", action="store_true", help="Aggregate audit logs into _summary.json")
    ap.add_argument("--json", action="store_true", help="Emit aggregate report as JSON")
    ap.add_argument("--no-resume", action="store_true",
                    help="Re-review every item even if a prior log exists")
    args = ap.parse_args()

    if args.report:
        aggregate_report(json_out=args.json)
        return

    if not args.doc_id:
        ap.error("doc_id is required unless --report is passed")

    review_doc(args.doc_id, resume=not args.no_resume)


if __name__ == "__main__":
    main()
