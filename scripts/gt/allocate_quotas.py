"""Reproducible per-doc query-type quota allocation for typed GT generation.

Given a category, a global type distribution target, and the seed-locked GT-
source doc list from gt_doc_selection.json, produce a deterministic per-doc
allocation matrix. The matrix tells you exactly how many queries of each type
to generate per doc.

Algorithm. Largest-quota-first greedy with rotating offsets per type. For
each type, fill docs round-robin starting at a per-type seeded offset, capped
at ceil(quota / n_docs) per doc. The starting offset rotation balances loads
across docs so no doc gets all the extras of the biggest types.

Distribusi default mengikuti design v3 (3-type stratified, equal split):
factual 9, paraphrased 8, multihop 8 (sum=25 = ceiling alami 5 docs x 5 q/doc).

Usage:
    python scripts/gt/allocate_quotas.py --category UU
    python scripts/gt/allocate_quotas.py --category UU --total 25 --seed 42
    python scripts/gt/allocate_quotas.py --category UU --emit-commands
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SELECTION_FILE = Path("data/gt_doc_selection.json")
ALLOCATION_FILE = Path("data/gt_allocation.json")

DEFAULT_DISTRIBUTION = {
    "factual": 9,
    "paraphrased": 8,
    "multihop": 8,
}
TYPE_PRIORITY = ["factual", "paraphrased", "multihop"]


def load_selected_docs(category: str) -> list[str]:
    if not SELECTION_FILE.exists():
        raise SystemExit(f"{SELECTION_FILE} not found, run scripts/gt/select_gt_docs.py first")
    with open(SELECTION_FILE, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(category)
    if not entry:
        raise SystemExit(f"category '{category}' not in {SELECTION_FILE}")
    return [row["doc_id"] for row in entry.get("selected", [])]


def allocate(distribution: dict[str, int], n_docs: int, seed: int) -> list[dict[str, int]]:
    """Greedy fill with rotating per-type offsets. Returns list of per-doc dicts."""
    total = sum(distribution.values())
    if total % n_docs != 0:
        raise SystemExit(
            f"total queries {total} not divisible by n_docs {n_docs}, "
            f"adjust distribution or doc count"
        )
    per_doc_cap = total // n_docs
    rng = random.Random(seed)

    allocation = [{t: 0 for t in distribution} for _ in range(n_docs)]
    doc_counts = [0] * n_docs

    types_sorted = sorted(distribution.items(), key=lambda kv: -kv[1])

    for type_name, quota in types_sorted:
        if quota == 0:
            continue
        max_per_doc = math.ceil(quota / n_docs)
        offset = rng.randrange(n_docs)
        order = [(offset + i) % n_docs for i in range(n_docs)]

        remaining = quota
        for i in order:
            if remaining == 0:
                break
            room = per_doc_cap - doc_counts[i]
            if room <= 0:
                continue
            give = min(max_per_doc, remaining, room)
            allocation[i][type_name] += give
            doc_counts[i] += give
            remaining -= give

        # Fallback pass, ignore max_per_doc cap if quota still has leftovers.
        # Happens when small types collide with already-filled docs.
        if remaining > 0:
            for i in order:
                if remaining == 0:
                    break
                room = per_doc_cap - doc_counts[i]
                if room <= 0:
                    continue
                give = min(remaining, room)
                allocation[i][type_name] += give
                doc_counts[i] += give
                remaining -= give

        if remaining > 0:
            raise SystemExit(
                f"could not fully allocate type '{type_name}', {remaining} unplaced. "
                f"Distribution incompatible with per-doc cap {per_doc_cap}."
            )

    return allocation


def render_commands(doc_ids: list[str], allocation: list[dict[str, int]]) -> list[str]:
    """Produce shell commands matching the allocation."""
    commands: list[str] = []
    for doc_id, alloc in zip(doc_ids, allocation):
        for type_name in TYPE_PRIORITY:
            n = alloc.get(type_name, 0)
            if n == 0:
                continue
            commands.append(
                f"python scripts/gt/prompt.py {doc_id} --type {type_name} --questions {n}"
            )
    return commands


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible per-doc query-type allocation")
    ap.add_argument("--category", required=True, help="Category name as in gt_doc_selection.json (e.g., UU)")
    ap.add_argument("--total", type=int, default=None,
                    help="Total queries (defaults to sum of --distribution)")
    ap.add_argument("--distribution", type=str, default=None,
                    help='JSON dict of type quotas, e.g. \'{"factual":9,"paraphrased":8,"multihop":8}\'')
    ap.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default 42)")
    ap.add_argument("--emit-commands", action="store_true", help="Print prompt.py commands")
    ap.add_argument("--out", type=str, default=str(ALLOCATION_FILE),
                    help=f"Output allocation JSON (default {ALLOCATION_FILE})")
    args = ap.parse_args()

    if args.distribution:
        distribution = json.loads(args.distribution)
    else:
        distribution = dict(DEFAULT_DISTRIBUTION)

    # Validate distribution keys against allowed types.
    unknown = set(distribution) - set(TYPE_PRIORITY)
    if unknown:
        raise SystemExit(
            f"unknown query type(s) in --distribution: {sorted(unknown)}. "
            f"Allowed types: {TYPE_PRIORITY}"
        )

    if args.total is not None and args.total != sum(distribution.values()):
        raise SystemExit(
            f"--total {args.total} does not match sum of distribution {sum(distribution.values())}"
        )

    doc_ids = load_selected_docs(args.category)
    if not doc_ids:
        raise SystemExit(f"no selected docs for category {args.category}")

    allocation = allocate(distribution, len(doc_ids), seed=args.seed)

    payload = {
        "category": args.category,
        "seed": args.seed,
        "total_queries": sum(distribution.values()),
        "n_docs": len(doc_ids),
        "type_distribution_global": distribution,
        "per_doc_allocation": [
            {
                "doc_id": doc_id,
                **alloc,
                "total": sum(alloc.values()),
            }
            for doc_id, alloc in zip(doc_ids, allocation)
        ],
    }

    out_path = Path(args.out)
    existing: dict = {}
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing[args.category] = payload
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Allocation saved to {out_path}")

    print(f"\nAllocation matrix for category={args.category} seed={args.seed}:\n")
    header = ["doc_id"] + TYPE_PRIORITY + ["total"]
    widths = [max(len(h), 12) for h in header]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for doc_id, alloc in zip(doc_ids, allocation):
        row = [doc_id] + [str(alloc.get(t, 0)) for t in TYPE_PRIORITY] + [str(sum(alloc.values()))]
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))
    totals = ["TOTAL"]
    for t in TYPE_PRIORITY:
        totals.append(str(sum(a.get(t, 0) for a in allocation)))
    totals.append(str(sum(sum(a.values()) for a in allocation)))
    print("  " + "  ".join(c.ljust(w) for c, w in zip(totals, widths)))

    if args.emit_commands:
        print("\nManual route (per-prompt copy-paste):\n")
        for cmd in render_commands(doc_ids, allocation):
            print(f"  {cmd}")

    print()
    print("Next (auto route via Anthropic API).")
    print(f"  python scripts/gt/auto_annotate.py --category {args.category} --dry-run")
    print(f"  python scripts/gt/auto_annotate.py --category {args.category}")


if __name__ == "__main__":
    main()
