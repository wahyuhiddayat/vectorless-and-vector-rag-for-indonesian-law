"""Per-doc interactive author verification logger for ground truth.

Author runs this once per doc after the Judge LLM step. The script walks
through each query in the cleaned raw GT, shows the anchor leaf text, and
records a verdict per item, correct, wrong, borderline, or skip. Output
goes to data/gt_audit/<doc_id>__<type>.json so the manual review pass leaves
a reproducible trail.

Per design v3 (3-type stratified): supported query types are factual,
paraphrased, multihop. Provenance metadata (annotator/judge models, prompt
SHA-8) lives di data/gt_provenance.json, bukan per-file sidecar.

Usage:
    python scripts/gt/log_review.py uu-13-2025
    python scripts/gt/log_review.py uu-13-2025 --type paraphrased
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
PROVENANCE_FILE = Path("data/gt_provenance.json")
QUERY_TYPES = ("factual", "paraphrased", "multihop")

VALID_VERDICTS = {"c": "correct", "w": "wrong", "b": "borderline", "s": "skipped"}


def _basename(doc_id: str, query_type: str) -> str:
    return f"{doc_id}__{query_type}"


def raw_path_for(doc_id: str, query_type: str = "factual") -> Path:
    """Return the path to the raw GT file for a (doc_id, query_type)."""
    return RAW_DIR / doc_category(doc_id) / f"{_basename(doc_id, query_type)}.json"


def audit_path_for(doc_id: str, query_type: str = "factual") -> Path:
    """Return the path to the audit log for a (doc_id, query_type)."""
    return AUDIT_DIR / f"{_basename(doc_id, query_type)}.json"


def load_provenance() -> dict:
    """Load data/gt_provenance.json if present, else return empty defaults."""
    if not PROVENANCE_FILE.exists():
        return {}
    with open(PROVENANCE_FILE, encoding="utf-8") as f:
        return json.load(f)


def resolve_provenance(provenance: dict, doc_id: str, query_type: str) -> dict:
    """Pick effective provenance for a (doc_id, type), checking overrides first."""
    if not provenance:
        return {}
    overrides = provenance.get("overrides", []) or []
    for entry in overrides:
        if entry.get("doc_id") == doc_id and entry.get("type") == query_type:
            return {
                "annotator_model": entry.get("annotator_model"),
                "judge_model": entry.get("judge_model"),
                "prompt_version": (provenance.get("prompt_versions") or {}).get(query_type),
                "generated_at": entry.get("generated_at"),
            }
    models = provenance.get("models", {})
    return {
        "annotator_model": models.get("annotator"),
        "judge_model": models.get("judge"),
        "prompt_version": (provenance.get("prompt_versions") or {}).get(query_type),
        "generated_at": None,
    }


def load_items(path: Path) -> list[dict]:
    """Load and lightly validate the raw GT item array."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise SystemExit(f"{path} top-level must be a JSON array")
    return data


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


def _anchor_pairs(item: dict) -> list[tuple[str, str]]:
    """Return list of (doc_id, anchor_node_id) covering every anchor on an item."""
    anchor_ids = item.get("gold_anchor_node_ids")
    doc_ids = item.get("gold_doc_ids")
    if isinstance(anchor_ids, list) and anchor_ids:
        if isinstance(doc_ids, list) and len(doc_ids) == len(anchor_ids):
            return list(zip(doc_ids, anchor_ids))
        primary = item.get("gold_doc_id", "")
        return [(primary, aid) for aid in anchor_ids]
    nid = item.get("gold_anchor_node_id") or item.get("gold_node_id")
    did = item.get("gold_doc_id", "")
    return [(did, nid)] if nid else []


def review_doc(doc_id: str, query_type: str, resume: bool) -> dict:
    """Run the interactive review loop for one (doc_id, query_type) and return the audit dict."""
    raw_path = raw_path_for(doc_id, query_type)
    if not raw_path.exists():
        raise SystemExit(f"raw GT not found, {raw_path}")

    items = load_items(raw_path)
    if not items:
        raise SystemExit(f"raw GT is empty, {raw_path}")

    leaf_map_cache: dict[str, dict[str, dict]] = {}

    def _leaf_map(did: str) -> dict[str, dict]:
        if did not in leaf_map_cache:
            leaf_map_cache[did] = get_leaf_meta_map(did)
        return leaf_map_cache[did]

    if not _leaf_map(doc_id):
        raise SystemExit(f"doc {doc_id} not found in data/index_rincian, cannot resolve anchor text")

    audit_path = audit_path_for(doc_id, query_type)
    existing: dict = {}
    if audit_path.exists() and resume:
        with open(audit_path, encoding="utf-8") as f:
            existing = json.load(f)

    by_qid = {entry["qid"]: entry for entry in existing.get("items", [])}

    provenance = load_provenance()
    effective = resolve_provenance(provenance, doc_id, query_type)
    print()
    print(f"Doc, {doc_id}  type={query_type}")
    print(f"Items, {len(items)}")
    if effective:
        print(f"Annotator model, {effective.get('annotator_model') or '<unknown>'}")
        print(f"Judge model    , {effective.get('judge_model') or '<unknown>'}")
        print(f"Prompt version , {effective.get('prompt_version') or '<unknown>'}")
    print()

    results: list[dict] = []
    for idx, item in enumerate(items, start=1):
        if idx in by_qid and resume:
            prev = by_qid[idx]
            print(f"[{idx}/{len(items)}] already reviewed as {prev['verdict']}, kept")
            results.append(prev)
            continue

        anchors = _anchor_pairs(item)
        primary_anchor = anchors[0][1] if anchors else "<missing>"

        print(f"[{idx}/{len(items)}] type={item.get('query_type', '?')}, "
              f"{item.get('query_style', '?')}, {item.get('reference_mode', '?')}")
        print(f"  Query  , {item.get('query', '')}")
        for a_idx, (did, nid) in enumerate(anchors, start=1):
            leaf = _leaf_map(did).get(nid, {})
            text = (leaf.get("text") or "").strip()
            if len(text) > 600:
                text = text[:600].rstrip() + " ..."
            nav = leaf.get("navigation_path", "<unknown>")
            label = f"Anchor {a_idx}" if len(anchors) > 1 else "Anchor "
            print(f"  {label}, {did} :: {nid}")
            print(f"  Path   , {nav}")
            print(f"  Gold   , {text or '<empty>'}")
        if item.get("answer_hint"):
            print(f"  Hint   , {item['answer_hint']}")
        try:
            verdict, notes = prompt_verdict(f"[{idx}/{len(items)}]")
        except KeyboardInterrupt:
            print("\nQuit, partial progress saved")
            break
        results.append({
            "qid": idx,
            "anchor_node_id": primary_anchor,
            "anchor_node_ids": [nid for _, nid in anchors],
            "verdict": verdict,
            "notes": notes,
        })
        print()

    audit = {
        "doc_id": doc_id,
        "query_type": query_type,
        "category": doc_category(doc_id),
        "reviewed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "annotator_model": effective.get("annotator_model"),
        "judge_model": effective.get("judge_model"),
        "prompt_version": effective.get("prompt_version"),
        "n_items": len(items),
        "n_reviewed": len(results),
        "items": results,
    }

    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nReviewed.")
    print(f"  Audit -> {audit_path}  ({len(results)} of {len(items)})")
    print()
    print("Next.")
    print(f"  Continue with the next allocation item, or after all (doc, type) are done,")
    print(f"    python scripts/gt/collect.py")
    print(f"    python scripts/gt/finalize.py")
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
    ap.add_argument("doc_id", nargs="?", help="Document ID, e.g. uu-13-2025")
    ap.add_argument("--type", "-t", type=str, default="factual",
                    choices=list(QUERY_TYPES),
                    help="Query type to review (default factual)")
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

    review_doc(args.doc_id, query_type=args.type, resume=not args.no_resume)


if __name__ == "__main__":
    main()
