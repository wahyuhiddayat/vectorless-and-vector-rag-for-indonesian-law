"""Annotate an indexed document tree with per-node `summary` fields."""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vectorless.ids import doc_category
from vectorless.llm import call as llm_call, get_stats, reset_counters

GRANULARITY_INDEX = {
    "pasal": Path("data/index_pasal"),
    "ayat": Path("data/index_ayat"),
    "rincian": Path("data/index_rincian"),
}

REGISTRY_PATH = ROOT / "data/raw/registry.json"


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


def _summarise_internal(node: dict, child_pairs: list[tuple[str, str]]) -> str:
    children_text = "\n".join(f"- {t}: {s}" for t, s in child_pairs if s)
    if not children_text:
        return ""
    prompt = INTERNAL_PROMPT.format(
        title=node.get("title", ""),
        nav_path=node.get("navigation_path", ""),
        children=children_text,
    )
    return (llm_call(prompt).get("summary") or "").strip()


def _collect_leaves(nodes: list[dict], force: bool, todo: list[dict], skipped: list[dict]) -> None:
    for node in nodes:
        if node.get("nodes"):
            _collect_leaves(node["nodes"], force, todo, skipped)
        elif force or not node.get("summary"):
            todo.append(node)
        else:
            skipped.append(node)


def _walk_internal(nodes: list[dict], counter: dict, force: bool, verbose: bool) -> None:
    for node in nodes:
        children = node.get("nodes")
        if not children:
            continue
        _walk_internal(children, counter, force, verbose)
        if not force and node.get("summary"):
            counter["skipped"] += 1
            continue
        t0 = time.time()
        pairs = [(c.get("title", ""), c.get("summary", "")) for c in children]
        node["summary"] = _summarise_internal(node, pairs)
        counter["internal"] += 1
        if verbose:
            print(f"  [INT  {counter['internal']:>2}] {node.get('navigation_path', '')[:80]}  "
                  f"({time.time()-t0:.1f}s)  --> {(node['summary'] or '')[:80]}")


def annotate_doc(doc_id: str, granularity: str = "pasal",
                 force: bool = False, verbose: bool = True) -> dict:
    """Annotate every node of one indexed doc in place. Returns LLM stats + counters."""
    if granularity not in GRANULARITY_INDEX:
        raise ValueError(f"granularity must be one of {list(GRANULARITY_INDEX)}")
    path = GRANULARITY_INDEX[granularity] / doc_category(doc_id) / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    doc = json.loads(path.read_text(encoding="utf-8"))
    if verbose:
        print(f"Annotating {doc_id} ({granularity}) — {path.stat().st_size//1024}KB")

    reset_counters()
    t_start = time.time()
    counter = {"leaf": 0, "internal": 0, "skipped": 0, "failed": 0}

    todo, leaf_skipped = [], []
    _collect_leaves(doc.get("structure", []), force, todo, leaf_skipped)
    counter["skipped"] = len(leaf_skipped)

    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_summarise_leaf, leaf): leaf for leaf in todo}
            for fut in as_completed(futures):
                leaf = futures[fut]
                try:
                    leaf["summary"] = fut.result()
                    counter["leaf"] += 1
                except Exception as e:
                    counter["failed"] += 1
                    leaf["summary"] = ""  # empty stays as "needs summary" on retry
                    if verbose:
                        print(f"  [LEAF FAIL] {leaf.get('navigation_path', '')[:80]} — {e!r}")
                    continue
                if verbose:
                    print(f"  [LEAF {counter['leaf']:>3}/{len(todo)}] "
                          f"{leaf.get('navigation_path', '')[:80]} "
                          f"--> {(leaf['summary'] or '')[:80]}")

    _walk_internal(doc.get("structure", []), counter, force, verbose)

    elapsed = time.time() - t_start
    stats = get_stats()
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        suffix = f" failed={counter['failed']}" if counter["failed"] else ""
        print(f"Done in {elapsed:.1f}s — leaf={counter['leaf']} "
              f"internal={counter['internal']} skipped={counter['skipped']}{suffix} | "
              f"{stats['llm_calls']} calls, {stats['total_tokens']:,} tokens")
    return {"elapsed_s": round(elapsed, 2), **counter, **stats}


def _registry_docs(category: str) -> list[str]:
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    target = category.upper()
    return sorted(did for did, entry in reg.items()
                  if (entry.get("jenis_folder") or "").upper() == target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[],
                    help="Doc to annotate (repeatable)")
    ap.add_argument("--category", help="Annotate every doc in this jenis_folder")
    ap.add_argument("--granularity", choices=list(GRANULARITY_INDEX), default="pasal")
    ap.add_argument("--force", action="store_true",
                    help="Re-summarise nodes that already have a summary")
    args = ap.parse_args()

    if args.doc_ids:
        targets = args.doc_ids
    elif args.category:
        targets = _registry_docs(args.category)
    else:
        raise SystemExit("must pass --doc-id or --category")

    print(f"Annotating {len(targets)} doc(s) at granularity={args.granularity}")
    totals = {"elapsed_s": 0.0, "llm_calls": 0, "total_tokens": 0, "failed": 0}
    for did in targets:
        try:
            stats = annotate_doc(did, granularity=args.granularity, force=args.force)
        except FileNotFoundError as e:
            print(f"  SKIP missing: {e}")
            continue
        for k in totals:
            totals[k] += stats[k]
        print()

    suffix = f" failed={totals['failed']}" if totals["failed"] else ""
    print(f"Total: {totals['elapsed_s']:.0f}s, {totals['llm_calls']} calls, "
          f"{totals['total_tokens']:,} tokens{suffix}")


if __name__ == "__main__":
    main()
