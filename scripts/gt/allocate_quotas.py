"""Reproducible per-doc query-type quota allocation for typed GT generation.

Given a category, a global type distribution target, and the seed-locked GT-
source doc list from gt_doc_selection.json, produce a deterministic per-doc
allocation matrix. The matrix tells you exactly how many queries of each type
to generate per doc, plus which docs to pair for crossdoc.

Algorithm. Largest-quota-first greedy with rotating offsets per type. For
each type, fill docs round-robin starting at a per-type seeded offset, capped
at ceil(quota / n_docs) per doc. The starting offset rotation balances loads
across docs so no doc gets all the extras of the biggest types.

Crossdoc pairing. For each doc that hosts a crossdoc query, pair it with the
next selected doc (round-robin, seeded shift). Pairs are deterministic from
the seed.

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
CATALOG_FILE = Path("data/index_pasal/catalog.json")

DEFAULT_DISTRIBUTION = {
    "factual": 8,
    "paraphrased": 6,
    "multihop": 6,
    "crossdoc": 3,
    "adversarial": 2,
}
TYPE_PRIORITY = ["factual", "paraphrased", "multihop", "crossdoc", "adversarial"]


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


def load_catalog_topics(doc_ids: list[str]) -> dict[str, dict[str, str]]:
    """Return {doc_id: {bidang, subjek, judul}} for the requested docs."""
    if not CATALOG_FILE.exists():
        return {}
    with open(CATALOG_FILE, encoding="utf-8") as f:
        catalog = json.load(f)
    by_id = {entry["doc_id"]: entry for entry in catalog if "doc_id" in entry}
    out: dict[str, dict[str, str]] = {}
    for did in doc_ids:
        entry = by_id.get(did, {})
        out[did] = {
            "bidang": entry.get("bidang", ""),
            "subjek": entry.get("subjek", ""),
            "judul": entry.get("judul", ""),
        }
    return out


def assign_crossdoc_pairs(
    doc_ids: list[str],
    allocation: list[dict[str, int]],
    seed: int,
    manual_pairs: dict[str, str] | None = None,
) -> tuple[list[str | None], list[str]]:
    """Pick a paired doc for each doc that hosts crossdoc queries.

    Pairing priority.
    1. Manual override via --manual-pairs.
    2. Same `subjek` partner (most semantically aligned, e.g. both APBN).
    3. Same `bidang` partner (broader category match).
    4. Arbitrary partner via seed-locked round-robin (last resort).

    Returns (pair_list, rationale_list) so the caller can surface why a pair
    was chosen. Pairs are deterministic from the seed when no manual override.
    """
    n = len(doc_ids)
    rationales: list[str] = [""] * n
    pairs: list[str | None] = [None] * n
    if n < 2:
        return pairs, rationales

    manual_pairs = manual_pairs or {}
    topics = load_catalog_topics(doc_ids)

    rng = random.Random(seed + 1)
    fallback_shift = 1 + rng.randrange(n - 1)

    crossdoc_indices = [i for i, a in enumerate(allocation) if a.get("crossdoc", 0) > 0]

    for i in crossdoc_indices:
        host = doc_ids[i]
        if host in manual_pairs:
            partner = manual_pairs[host]
            if partner not in doc_ids:
                raise SystemExit(
                    f"manual pair '{host}' -> '{partner}' but '{partner}' not in selected docs"
                )
            if partner == host:
                raise SystemExit(f"manual pair points doc '{host}' to itself")
            pairs[i] = partner
            rationales[i] = "manual"
            continue

        host_subjek = topics.get(host, {}).get("subjek", "")
        host_bidang = topics.get(host, {}).get("bidang", "")

        # Tier 2, same subjek
        candidates = [
            doc_ids[j] for j in range(n)
            if j != i and topics.get(doc_ids[j], {}).get("subjek", "") == host_subjek
            and host_subjek
        ]
        if candidates:
            pick = rng.choice(sorted(candidates))
            pairs[i] = pick
            rationales[i] = f"same subjek ({host_subjek})"
            continue

        # Tier 3, same bidang
        candidates = [
            doc_ids[j] for j in range(n)
            if j != i and topics.get(doc_ids[j], {}).get("bidang", "") == host_bidang
            and host_bidang
        ]
        if candidates:
            pick = rng.choice(sorted(candidates))
            pairs[i] = pick
            rationales[i] = f"same bidang ({host_bidang})"
            continue

        # Tier 4, arbitrary round-robin
        pairs[i] = doc_ids[(i + fallback_shift) % n]
        rationales[i] = "arbitrary (no topic match)"

    return pairs, rationales


def render_commands(category: str, doc_ids: list[str], allocation: list[dict[str, int]], pairs: list[str | None]) -> list[str]:
    """Produce shell commands matching the allocation."""
    commands: list[str] = []
    for doc_id, alloc, paired in zip(doc_ids, allocation, pairs):
        for type_name in TYPE_PRIORITY:
            n = alloc.get(type_name, 0)
            if n == 0:
                continue
            base = f"python scripts/gt/prompt.py {doc_id} --type {type_name} --questions {n}"
            if type_name == "crossdoc":
                if not paired:
                    raise SystemExit(f"doc {doc_id} has crossdoc>0 but no pair assigned")
                base += f" --paired-doc {paired}"
            commands.append(base)
    return commands


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible per-doc query-type allocation")
    ap.add_argument("--category", required=True, help="Category name as in gt_doc_selection.json (e.g., UU)")
    ap.add_argument("--total", type=int, default=None,
                    help="Total queries (defaults to sum of --distribution)")
    ap.add_argument("--distribution", type=str, default=None,
                    help='JSON dict of type quotas, e.g. \'{"factual":8,"paraphrased":6,"multihop":6,"crossdoc":3,"adversarial":2}\'')
    ap.add_argument("--seed", type=int, default=42, help="Seed for reproducibility (default 42)")
    ap.add_argument("--emit-commands", action="store_true", help="Print prompt.py commands")
    ap.add_argument("--out", type=str, default=str(ALLOCATION_FILE),
                    help=f"Output allocation JSON (default {ALLOCATION_FILE})")
    ap.add_argument("--manual-pairs", type=str, default=None,
                    help='Override crossdoc pairings, format: "host1=partner1,host2=partner2"')
    args = ap.parse_args()

    manual_pairs: dict[str, str] = {}
    if args.manual_pairs:
        for entry in args.manual_pairs.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if "=" not in entry:
                raise SystemExit(f"--manual-pairs entry '{entry}' must be host=partner")
            host, partner = entry.split("=", 1)
            manual_pairs[host.strip()] = partner.strip()

    if args.distribution:
        distribution = json.loads(args.distribution)
    else:
        distribution = dict(DEFAULT_DISTRIBUTION)
    if args.total is not None and args.total != sum(distribution.values()):
        raise SystemExit(
            f"--total {args.total} does not match sum of distribution {sum(distribution.values())}"
        )

    doc_ids = load_selected_docs(args.category)
    if not doc_ids:
        raise SystemExit(f"no selected docs for category {args.category}")

    allocation = allocate(distribution, len(doc_ids), seed=args.seed)
    pairs, rationales = assign_crossdoc_pairs(
        doc_ids, allocation, seed=args.seed, manual_pairs=manual_pairs,
    )

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
                "crossdoc_paired_with": paired,
                "crossdoc_pair_rationale": rationale,
                "total": sum(alloc.values()),
            }
            for doc_id, alloc, paired, rationale in zip(doc_ids, allocation, pairs, rationales)
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
    header = ["doc_id"] + TYPE_PRIORITY + ["total", "paired", "rationale"]
    widths = [max(len(h), 12) for h in header]
    print("  " + "  ".join(h.ljust(w) for h, w in zip(header, widths)))
    for doc_id, alloc, paired, rationale in zip(doc_ids, allocation, pairs, rationales):
        row = [doc_id] + [str(alloc.get(t, 0)) for t in TYPE_PRIORITY] + [
            str(sum(alloc.values())), paired or "-", rationale or "-",
        ]
        print("  " + "  ".join(c.ljust(w) for c, w in zip(row, widths)))
    totals = ["TOTAL"]
    for t in TYPE_PRIORITY:
        totals.append(str(sum(a.get(t, 0) for a in allocation)))
    totals.append(str(sum(sum(a.values()) for a in allocation)))
    totals.extend(["-", "-"])
    print("  " + "  ".join(c.ljust(w) for c, w in zip(totals, widths)))

    if args.emit_commands:
        print("\nCommands to run:\n")
        for cmd in render_commands(args.category, doc_ids, allocation, pairs):
            print(f"  {cmd}")


if __name__ == "__main__":
    main()
