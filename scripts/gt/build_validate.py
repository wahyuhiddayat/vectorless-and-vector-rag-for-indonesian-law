"""Assemble a ready-to-paste validation prompt for the chosen Judge LLM.

Run this after pasting raw GT output into
data/ground_truth_raw/<CAT>/<doc_id>.json (factual) or
data/ground_truth_raw/<CAT>/<doc_id>__<type>.json (paraphrased, multihop). The script.

  1. Runs Layer 1 (struct check from collect.py), fails fast on hard errors.
  2. Runs Layer 2 deterministic gate for `paraphrased` (Sastrawi-stemmed
     Jaccard < 0.40). Other types skip Layer 2 and go straight to Layer 3.
  3. Inlines the items and the leaf-node context they reference (across both
     anchors for multihop) into the type-aware rules from validate_prompt.txt.
  4. Writes the assembled prompt to tmp/validate_<doc_id>__<type>.txt.

The Judge LLM output must be saved to a text file containing the
`---CLEANED---` separator, then applied through `apply_validation.py`. Never
overwrite the raw GT file directly.

Usage:
    python scripts/gt/build_validate.py --doc-id uu-13-2025 --type factual
    python scripts/gt/build_validate.py --doc-id uu-13-2025 --type paraphrased
    python scripts/gt/build_validate.py --doc-id uu-13-2025 --type multihop
    python scripts/gt/build_validate.py --doc-id uu-13-2025 --type paraphrased --skip-layer2
"""

import argparse
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vectorless.ids import doc_category
from scripts.gt.collect import validate_raw_file, get_leaf_meta_map

RAW_DIR = Path("data/ground_truth_raw")
TMP_DIR = Path("tmp")
RULES_FILE = Path(__file__).resolve().parent / "validate_prompt.txt"
DEFAULT_LEAF_TEXT_BUDGET = 600
MULTI_ANCHOR_TYPES = {"multihop"}
QUERY_TYPES = ("factual", "paraphrased", "multihop")


def _raw_filename(doc_id: str, query_type: str) -> str:
    return f"{doc_id}__{query_type}.json"


def _raw_path(doc_id: str, query_type: str = "factual") -> Path:
    return RAW_DIR / doc_category(doc_id) / _raw_filename(doc_id, query_type)


def _anchor_pairs(item: dict) -> list[tuple[str, str]]:
    """Return list of (doc_id, anchor_node_id) covering every anchor on an item.

    Falls back to the singular fields when the jamak fields are absent so
    legacy single-anchor items still work.
    """
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


def _build_doc_context(items: list[dict], text_budget: int) -> list[dict]:
    """Compact leaf list covering every anchor referenced by items.

    Walks every (doc_id, anchor_node_id) pair in the items so multihop
    includes both anchors. Loads each referenced doc lazily.
    """
    leaf_map_cache: dict[str, dict[str, dict]] = {}
    referenced: list[dict] = []
    seen: set[tuple[str, str]] = set()

    for item in items:
        for did, nid in _anchor_pairs(item):
            if not did or not nid:
                continue
            key = (did, nid)
            if key in seen:
                continue
            seen.add(key)
            if did not in leaf_map_cache:
                leaf_map_cache[did] = get_leaf_meta_map(did)
            leaf_map = leaf_map_cache[did]
            if nid not in leaf_map:
                continue
            meta = leaf_map[nid]
            text = (meta.get("text") or "").strip()
            if len(text) > text_budget:
                text = text[:text_budget].rstrip() + " ..."
            referenced.append({
                "doc_id": did,
                "node_id": nid,
                "navigation_path": meta.get("navigation_path", ""),
                "text": text,
            })
    return referenced


def _run_layer2(query_type: str, raw_path: Path) -> None:
    """Run the per-type deterministic gate, abort on any flag.

    paraphrased uses Sastrawi-stemmed Jaccard. factual and multihop skip
    Layer 2 and go directly to Layer 3 judge.
    """
    if query_type == "paraphrased":
        from scripts.gt.validators.paraphrase_overlap import validate_file as _para_validate
        n, n_flagged, flagged = _para_validate(raw_path)
        print(f"Layer 2 paraphrase, {n} item(s) checked, {n_flagged} flagged")
        if n_flagged:
            for f in flagged:
                print(f"  flag {f.get('item_index')}, {f.get('reason')}")
            sys.exit(1)
        return


def assemble_prompt(
    doc_id: str,
    text_budget: int,
    query_type: str = "factual",
    skip_layer2: bool = False,
) -> tuple[Path, list[dict]]:
    """Assemble the Judge LLM validation prompt for one (doc_id, query_type)."""
    raw_path = _raw_path(doc_id, query_type)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"raw GT not found, {raw_path}. "
            f"Generate it first via `python scripts/gt/prompt.py {doc_id} --type {query_type}`, "
            f"then paste the LLM output into the file."
        )

    valid_items, hard_errors = validate_raw_file(raw_path)
    if hard_errors:
        print("Layer 1 hard errors in raw GT, fix and retry,", file=sys.stderr)
        for e in hard_errors:
            print(e, file=sys.stderr)
        sys.exit(1)
    print(f"Layer 1 struct, {len(valid_items)} item(s) ok")

    if not skip_layer2:
        _run_layer2(query_type, raw_path)

    rules_text = RULES_FILE.read_text(encoding="utf-8").strip()
    context = _build_doc_context(valid_items, text_budget)

    body = (
        f"{rules_text}\n\n"
        f"=== ITEMS UNTUK DIVALIDASI (JSON) ===\n"
        f"{json.dumps(valid_items, ensure_ascii=False, indent=2)}\n\n"
        f"=== KONTEKS DOKUMEN, leaf nodes terkait ===\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TMP_DIR / f"validate_{doc_id}__{query_type}.txt"
    out_path.write_text(body, encoding="utf-8")
    return out_path, valid_items


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--type", "-t", type=str, default="factual",
                    choices=list(QUERY_TYPES),
                    help="Query type to validate (default factual)")
    ap.add_argument("--max-context-chars", type=int, default=DEFAULT_LEAF_TEXT_BUDGET,
                    help=f"Per-leaf text excerpt budget (default {DEFAULT_LEAF_TEXT_BUDGET})")
    ap.add_argument("--skip-layer2", action="store_true",
                    help="Skip the per-type deterministic gate (use only for diagnostic reruns)")
    args = ap.parse_args()

    out_path, items = assemble_prompt(
        args.doc_id,
        args.max_context_chars,
        query_type=args.type,
        skip_layer2=args.skip_layer2,
    )
    raw_path = _raw_path(args.doc_id, args.type)
    print(f"\nCreated.")
    print(f"  Judge prompt -> {out_path}  ({len(items)} item)")
    print()
    print("Next.")
    print(f"  1. Paste {out_path} to Judge LLM, paste full response over {raw_path}")
    print(f"  2. python scripts/gt/apply_validation.py --doc-id {args.doc_id} --type {args.type}")
    print(f"  3. python scripts/gt/log_review.py {args.doc_id} --type {args.type}")


if __name__ == "__main__":
    main()
