"""Stratified random selection of GT-source docs from each category.

Picks N docs per category to be the source of GT questions, leaving the
remaining docs in the index as distractors. Stratification is by number of
rincian leaf nodes, so selected docs cover a range of document sizes.

Usage:
    python scripts/gt/select_gt_docs.py --category PMK
    python scripts/gt/select_gt_docs.py --category PMK --per-category 5 --seed 42
    python scripts/gt/select_gt_docs.py --all
"""

import argparse
import json
import random
import sys
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_INDEX = Path("data/index_rincian")
SELECTION_FILE = Path("data/gt_doc_selection.json")
DEFAULT_PER_CATEGORY = 5
DEFAULT_SEED = 42


def count_leaves(structure: list[dict]) -> int:
    """Count leaf nodes (with text) in a parsed index structure."""
    total = 0
    for node in structure:
        if node.get("nodes"):
            total += count_leaves(node["nodes"])
        elif node.get("text"):
            total += 1
    return total


def list_docs_in_category(category: str) -> list[tuple[str, int]]:
    """Return [(doc_id, leaf_count)] sorted by leaf_count for one category."""
    cat_dir = DATA_INDEX / category
    if not cat_dir.exists():
        raise SystemExit(f"Category folder not found, {cat_dir}")

    rows: list[tuple[str, int]] = []
    for path in sorted(cat_dir.glob("*.json")):
        if path.name == "catalog.json":
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        rows.append((path.stem, count_leaves(doc.get("structure", []))))
    rows.sort(key=lambda r: (r[1], r[0]))
    return rows


def stratified_pick(rows: list[tuple[str, int]], n: int, rng: random.Random) -> list[str]:
    """Pick n doc_ids from rows via equal-size strata over leaf-count order.

    Splits the sorted rows into n strata of as-equal-as-possible size and
    picks one doc per stratum at random. If len(rows) < n, returns all.
    """
    total = len(rows)
    if total <= n:
        return [r[0] for r in rows]

    picks: list[str] = []
    for stratum_index in range(n):
        lo = (stratum_index * total) // n
        hi = ((stratum_index + 1) * total) // n
        choice = rng.choice(rows[lo:hi])
        picks.append(choice[0])
    return picks


def load_existing_selection() -> dict:
    """Load the existing selection file or return an empty dict."""
    if not SELECTION_FILE.exists():
        return {}
    with open(SELECTION_FILE, encoding="utf-8") as f:
        return json.load(f)


def write_selection(selection: dict) -> None:
    """Persist the selection map atomically."""
    SELECTION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SELECTION_FILE, "w", encoding="utf-8") as f:
        json.dump(selection, f, ensure_ascii=False, indent=2)
        f.write("\n")


def select_one_category(
    category: str,
    per_category: int,
    seed: int,
    force: bool,
) -> dict:
    """Run the stratified pick for one category and return the entry."""
    rows = list_docs_in_category(category)
    if not rows:
        raise SystemExit(f"No docs in {DATA_INDEX / category}")

    rng = random.Random(f"{seed}-{category}")
    selected = stratified_pick(rows, per_category, rng)
    distractors = [r[0] for r in rows if r[0] not in set(selected)]

    leaf_map = dict(rows)
    return {
        "seed": seed,
        "per_category": per_category,
        "stratification": "equal-size strata over leaf-count order",
        "selected": [
            {"doc_id": d, "leaf_count": leaf_map[d]} for d in sorted(selected)
        ],
        "distractors": [
            {"doc_id": d, "leaf_count": leaf_map[d]} for d in sorted(distractors)
        ],
    }


def list_all_categories() -> list[str]:
    """Return all category folder names that exist under data/index_rincian."""
    if not DATA_INDEX.exists():
        return []
    return sorted(
        p.name for p in DATA_INDEX.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Stratified random selection of GT-source docs.")
    ap.add_argument("--category", type=str, default=None, help="Category folder name, e.g. PMK")
    ap.add_argument("--all", action="store_true", help="Process every category under data/index_rincian")
    ap.add_argument("--per-category", type=int, default=DEFAULT_PER_CATEGORY,
                    help=f"Number of docs to pick per category (default {DEFAULT_PER_CATEGORY})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Base random seed (default {DEFAULT_SEED}). Per-category seed is f'{{seed}}-{{CAT}}'.")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite an existing entry for the same category")
    ap.add_argument("--show", action="store_true", help="Print current selection file and exit")
    args = ap.parse_args()

    if args.show:
        sel = load_existing_selection()
        print(json.dumps(sel, ensure_ascii=False, indent=2))
        return

    if not args.category and not args.all:
        ap.error("must pass --category CAT or --all")

    selection = load_existing_selection()
    targets = list_all_categories() if args.all else [args.category]

    for cat in targets:
        if cat in selection and not args.force:
            print(f"  skip {cat}, already selected (use --force to overwrite)")
            continue
        entry = select_one_category(cat, args.per_category, args.seed, args.force)
        selection[cat] = entry
        picked = ", ".join(d["doc_id"] for d in entry["selected"])
        print(f"  {cat}, selected {len(entry['selected'])} of {len(entry['selected']) + len(entry['distractors'])}, picks, {picked}")

    write_selection(selection)
    print(f"\nWrote {SELECTION_FILE}")


if __name__ == "__main__":
    main()
