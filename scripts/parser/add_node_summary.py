"""Annotate an indexed document tree with per-node `summary` fields.

Bottom-up traversal: leaf nodes summarised from `text`, internal nodes
summarised from their children's summaries. The retrieval layer reads
`summary` opportunistically — annotating an index improves LLM
tree-navigation in low-signal trees (flat documents with generic
section titles).

Usage:
  python scripts/parser/add_node_summary.py --doc-id pmk-21-2026
  python scripts/parser/add_node_summary.py --category PMK
  python scripts/parser/add_node_summary.py --in-file path/to.json --out-file path/to.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vectorless.retrieval.common import (
    _doc_category, llm_call, get_token_stats, reset_token_counters,
)


LEAF_PROMPT = """\
Ringkas isi pasal hukum Indonesia berikut menjadi 1-2 kalimat ringkas
yang menjelaskan POKOK PENGATURAN-nya (subjek + tindakan/kewajiban/hak utama).
Hindari kalimat pembuka seperti "Pasal ini mengatur".

Judul: {title}
Konteks: {nav_path}

Isi:
{text}

Balas dalam JSON:
{{"summary": "<ringkasan 1-2 kalimat>"}}
"""

INTERNAL_PROMPT = """\
Ringkas cakupan bagian hukum berikut menjadi 1 kalimat ringkas yang
menjelaskan TOPIK utama yang dibahas oleh sub-bagian di bawahnya.

Judul: {title}
Konteks: {nav_path}

Sub-bagian (judul + ringkasan singkat):
{children}

Balas dalam JSON:
{{"summary": "<ringkasan 1 kalimat>"}}
"""


def _summarise_leaf(node: dict) -> str:
    text = (node.get("text") or "").strip()
    if not text:
        return ""
    prompt = LEAF_PROMPT.format(
        title=node.get("title", ""),
        nav_path=node.get("navigation_path", ""),
        text=text[:4000],
    )
    return (llm_call(prompt).get("summary") or "").strip()


def _summarise_internal(node: dict, child_summaries: list[tuple[str, str]]) -> str:
    children_text = "\n".join(f"- {t}: {s}" for t, s in child_summaries if s)
    if not children_text:
        return ""
    prompt = INTERNAL_PROMPT.format(
        title=node.get("title", ""),
        nav_path=node.get("navigation_path", ""),
        children=children_text,
    )
    return (llm_call(prompt).get("summary") or "").strip()


def annotate(nodes: list[dict], counter: dict | None = None, skip_existing: bool = True) -> dict:
    """Add `summary` to every node bottom-up. Returns counters for reporting."""
    if counter is None:
        counter = {"leaf": 0, "internal": 0, "skipped": 0}
    for node in nodes:
        children = node.get("nodes")
        if children:
            annotate(children, counter, skip_existing)
        if skip_existing and node.get("summary"):
            counter["skipped"] += 1
            continue
        t0 = time.time()
        if children:
            child_pairs = [(c.get("title", ""), c.get("summary", "")) for c in children]
            node["summary"] = _summarise_internal(node, child_pairs)
            counter["internal"] += 1
            tag = f"INT  {counter['internal']:>2}"
        else:
            node["summary"] = _summarise_leaf(node)
            counter["leaf"] += 1
            tag = f"LEAF {counter['leaf']:>2}"
        print(f"  [{tag}] {node.get('navigation_path', '')[:80]}  "
              f"({time.time()-t0:.1f}s)  --> {(node['summary'] or '')[:80]}")
    return counter


def annotate_file(in_path: Path, out_path: Path, force: bool = False) -> dict:
    """Read index JSON, annotate in place, write back. Returns LLM stats."""
    doc = json.loads(in_path.read_text(encoding="utf-8"))
    print(f"Annotating {doc['doc_id']} — {in_path.stat().st_size//1024}KB")

    reset_token_counters()
    t_start = time.time()
    counter = annotate(doc.get("structure", []), skip_existing=not force)
    elapsed = time.time() - t_start
    stats = get_token_stats()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done in {elapsed:.1f}s — leaf={counter['leaf']} internal={counter['internal']} "
          f"skipped={counter['skipped']} | {stats['llm_calls']} calls, "
          f"{stats['total_tokens']:,} tokens")
    return {"elapsed_s": elapsed, "counters": counter, **stats}


def _resolve_paths(args) -> list[tuple[Path, Path]]:
    """Resolve --in-file / --doc-id / --category to a list of (in, out) pairs."""
    if args.in_file:
        return [(Path(args.in_file), Path(args.out_file or args.in_file))]

    src_root = Path(args.source)
    out_root = Path(args.out_dir) if args.out_dir else src_root

    if args.doc_id:
        rel = Path(_doc_category(args.doc_id)) / f"{args.doc_id}.json"
        return [(src_root / rel, out_root / rel)]

    if args.category:
        cat_dir = src_root / args.category
        if not cat_dir.is_dir():
            raise FileNotFoundError(cat_dir)
        return [(p, out_root / p.relative_to(src_root))
                for p in sorted(cat_dir.glob("*.json"))]

    raise SystemExit("Must specify --in-file, --doc-id, or --category")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--source", default="data/index_pasal",
                    help="Source index root (default: data/index_pasal)")
    ap.add_argument("--out-dir", default=None,
                    help="Output index root (default: same as --source = annotate in place)")
    ap.add_argument("--category", help="Annotate every doc in <source>/<CATEGORY>/")
    ap.add_argument("--doc-id", help="Annotate a single doc")
    ap.add_argument("--in-file", help="Direct path to input JSON")
    ap.add_argument("--out-file", help="Direct path to output JSON (with --in-file)")
    ap.add_argument("--force", action="store_true",
                    help="Re-summarise nodes that already have a summary")
    args = ap.parse_args()

    pairs = _resolve_paths(args)
    print(f"Annotating {len(pairs)} file(s)")
    totals = {"elapsed_s": 0.0, "llm_calls": 0, "total_tokens": 0}
    for in_path, out_path in pairs:
        if not in_path.exists():
            print(f"  SKIP missing: {in_path}")
            continue
        stats = annotate_file(in_path, out_path, force=args.force)
        for k in totals:
            totals[k] += stats[k]
        print()

    if len(pairs) > 1:
        print(f"Total: {totals['elapsed_s']:.0f}s, {totals['llm_calls']} calls, "
              f"{totals['total_tokens']:,} tokens")


if __name__ == "__main__":
    main()
