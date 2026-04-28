"""Assemble a ready-to-paste validation prompt for the chosen Judge LLM.

Run this after pasting raw GT output into data/ground_truth_raw/<CAT>/<doc_id>.json.
The script:
  1. Runs the structural check from collect.py (fails fast on hard errors).
  2. Inlines the items + the leaf-node context they reference into the
     static rules from validate_prompt.txt.
  3. Writes the assembled prompt to tmp/validate_<doc_id>.txt.

Usage:
    python scripts/gt/build_validate.py --doc-id perma-2-2022
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


def _raw_filename(doc_id: str, query_type: str) -> str:
    if query_type == "factual":
        return f"{doc_id}.json"
    return f"{doc_id}__{query_type}.json"


def _raw_path(doc_id: str, query_type: str = "factual") -> Path:
    return RAW_DIR / doc_category(doc_id) / _raw_filename(doc_id, query_type)


def _build_doc_context(items: list[dict], doc_id: str, text_budget: int) -> list[dict]:
    """Compact leaf list covering the anchors referenced by items."""
    leaf_map = get_leaf_meta_map(doc_id)
    referenced = []
    seen: set[str] = set()
    for item in items:
        nid = item.get("gold_anchor_node_id")
        if not nid or nid in seen or nid not in leaf_map:
            continue
        seen.add(nid)
        meta = leaf_map[nid]
        text = (meta.get("text") or "").strip()
        if len(text) > text_budget:
            text = text[:text_budget].rstrip() + " ..."
        referenced.append({
            "node_id": nid,
            "navigation_path": meta.get("navigation_path", ""),
            "text": text,
        })
    return referenced


def assemble_prompt(doc_id: str, text_budget: int, query_type: str = "factual") -> tuple[Path, list[dict]]:
    raw_path = _raw_path(doc_id, query_type)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"raw GT not found: {raw_path}\n"
            f"Generate it first via `python scripts/gt/prompt.py {doc_id}`, paste the LLM output here."
        )

    valid_items, hard_errors = validate_raw_file(raw_path)
    if hard_errors:
        print("Hard errors in raw GT — fix and retry:", file=sys.stderr)
        for e in hard_errors:
            print(e, file=sys.stderr)
        sys.exit(1)

    rules_text = RULES_FILE.read_text(encoding="utf-8").strip()
    context = _build_doc_context(valid_items, doc_id, text_budget)

    body = (
        f"{rules_text}\n\n"
        f"=== ITEMS UNTUK DIVALIDASI (JSON) ===\n"
        f"{json.dumps(valid_items, ensure_ascii=False, indent=2)}\n\n"
        f"=== KONTEKS DOKUMEN — leaf nodes terkait ===\n"
        f"{json.dumps(context, ensure_ascii=False, indent=2)}\n"
    )

    TMP_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "" if query_type == "factual" else f"__{query_type}"
    out_path = TMP_DIR / f"validate_{doc_id}{suffix}.txt"
    out_path.write_text(body, encoding="utf-8")
    return out_path, valid_items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--doc-id", required=True)
    ap.add_argument("--type", "-t", type=str, default="factual",
                    choices=["factual", "paraphrased", "multihop", "crossdoc", "adversarial"],
                    help="Query type to validate (default: factual)")
    ap.add_argument("--max-context-chars", type=int, default=DEFAULT_LEAF_TEXT_BUDGET,
                    help=f"Per-leaf text excerpt budget (default: {DEFAULT_LEAF_TEXT_BUDGET})")
    args = ap.parse_args()

    out_path, items = assemble_prompt(args.doc_id, args.max_context_chars, query_type=args.type)
    raw_path = _raw_path(args.doc_id, args.type)
    print(f"Validation prompt written: {out_path}")
    print(f"  items={len(items)} doc_id={args.doc_id}")
    print()
    print("Next:")
    print(f"  1. Open {out_path}, copy all, paste to your chosen Judge LLM")
    print(f"     (Copilot in IDE / Claude / GPT — NOT Gemini, to keep generator/judge family")
    print(f"     independent from the Gemini retrieval backbone).")
    print(f"  2. Have the Judge overwrite {raw_path} with the cleaned items array.")
    print(f"     (Copilot can edit the file in place; for browser Judges, paste the cleaned")
    print(f"     array section over the file's current contents.)")
    print(f"  3. python scripts/gt/collect.py --file {raw_path}")


if __name__ == "__main__":
    main()
