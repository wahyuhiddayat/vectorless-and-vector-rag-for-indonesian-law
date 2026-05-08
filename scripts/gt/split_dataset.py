"""Stratified 70/15/15 train/val/test split of the GT testset.

Reads data/validated_testset.pkl and assigns each query to one of three
splits via per-cell allocation, where a cell is a (category, query_type)
pair. Per-cell seeds are derived from f"{seed}-{category}-{query_type}"
so the assignment is deterministic and independent across cells.

Outputs:
    data/splits/train_qids.json   list of qids
    data/splits/val_qids.json     list of qids
    data/splits/test_qids.json    list of qids
    data/splits/split_manifest.json   metadata, per-cell stats, sha256

Usage:
    python scripts/gt/split_dataset.py
    python scripts/gt/split_dataset.py --dry-run --stats
    python scripts/gt/split_dataset.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vectorless.ids import doc_category  # noqa: E402

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

TESTSET_FILE = Path("data/validated_testset.pkl")
SPLITS_DIR = Path("data/splits")
DEFAULT_SEED = 42
DEFAULT_RATIO = (0.70, 0.15, 0.15)


def load_testset() -> dict:
    """Load the validated testset pickle."""
    if not TESTSET_FILE.exists():
        raise SystemExit(f"Testset not found, {TESTSET_FILE}. Run scripts/gt/finalize.py first.")
    with open(TESTSET_FILE, "rb") as f:
        return pickle.load(f)


def cell_key(item: dict) -> tuple[str, str]:
    """Return (category, query_type) for a testset item."""
    doc_id = item.get("gold_doc_id") or ""
    cat = doc_category(doc_id) if doc_id else "(unknown)"
    qtype = item.get("query_type", "factual")
    return cat, qtype


def allocate_cell(
    qids: list[str],
    seed: int,
    cat: str,
    qtype: str,
    ratio: tuple[float, float, float],
    rr_state: dict,
) -> tuple[list[str], list[str], list[str]]:
    """Allocate one (category, query_type) cell into train, val, test.

    Determinism, sort qids alphabetically, then shuffle with a per-cell
    seed derived from f"{seed}-{cat}-{qtype}". Cells with N=1 go entirely
    to train. Cells with N=2 send the first to train and the second to
    whichever of val or test currently has fewer queries (round-robin via
    rr_state). Larger cells use rounded proportional allocation with a
    guard so test never drops below 1 when N>=4.
    """
    qids_sorted = sorted(qids)
    rng = random.Random(f"{seed}-{cat}-{qtype}")
    rng.shuffle(qids_sorted)

    n = len(qids_sorted)
    if n == 0:
        return [], [], []
    if n == 1:
        return qids_sorted[:], [], []
    if n == 2:
        first = qids_sorted[:1]
        second = qids_sorted[1:]
        if rr_state["val"] <= rr_state["test"]:
            rr_state["val"] += 1
            return first, second, []
        rr_state["test"] += 1
        return first, [], second

    train_p, val_p, _ = ratio
    n_train = round(train_p * n)
    n_val = round(val_p * n)
    n_test = n - n_train - n_val

    if n_test == 0 and n >= 4:
        n_train -= 1
        n_test = 1
    if n_val == 0 and n >= 4:
        if n_train > 0:
            n_train -= 1
            n_val = 1
        elif n_test > 1:
            n_test -= 1
            n_val = 1

    train = qids_sorted[:n_train]
    val = qids_sorted[n_train:n_train + n_val]
    test = qids_sorted[n_train + n_val:]
    return train, val, test


def split(testset: dict, seed: int, ratio: tuple[float, float, float]) -> dict:
    """Build the split assignment for the entire testset.

    Returns a dict with train, val, test qid lists plus per-cell stats.
    """
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, item in testset.items():
        cells[cell_key(item)].append(qid)

    train_all: list[str] = []
    val_all: list[str] = []
    test_all: list[str] = []
    rr_state = {"val": 0, "test": 0}
    cell_stats: list[dict] = []

    for (cat, qtype) in sorted(cells.keys()):
        qids = cells[(cat, qtype)]
        tr, va, te = allocate_cell(qids, seed, cat, qtype, ratio, rr_state)
        train_all.extend(tr)
        val_all.extend(va)
        test_all.extend(te)
        cell_stats.append({
            "category": cat,
            "query_type": qtype,
            "n_total": len(qids),
            "n_train": len(tr),
            "n_val": len(va),
            "n_test": len(te),
        })

    train_all.sort()
    val_all.sort()
    test_all.sort()

    return {
        "train": train_all,
        "val": val_all,
        "test": test_all,
        "cells": cell_stats,
    }


def sha256_of_list(items: list[str]) -> str:
    """Hash a sorted qid list to detect drift."""
    payload = "\n".join(sorted(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_outputs(result: dict, seed: int, ratio: tuple[float, float, float]) -> None:
    """Persist split files and manifest."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{name}_qids.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result[name], f, indent=2)
            f.write("\n")

    manifest = {
        "seed": seed,
        "ratio": {"train": ratio[0], "val": ratio[1], "test": ratio[2]},
        "stratification": "(category, query_type) joint, per-cell allocation",
        "totals": {
            "train": len(result["train"]),
            "val": len(result["val"]),
            "test": len(result["test"]),
            "all": len(result["train"]) + len(result["val"]) + len(result["test"]),
        },
        "fingerprints": {
            "train": sha256_of_list(result["train"]),
            "val": sha256_of_list(result["val"]),
            "test": sha256_of_list(result["test"]),
        },
        "cells": result["cells"],
    }
    with open(SPLITS_DIR / "split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")


def print_stats(testset: dict, result: dict) -> None:
    """Print summary stats by query_type, by category, and coverage notes."""
    by_split_type: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in ("train", "val", "test")
    }
    by_split_cat: dict[str, dict[str, int]] = {
        s: defaultdict(int) for s in ("train", "val", "test")
    }
    for split_name in ("train", "val", "test"):
        for qid in result[split_name]:
            item = testset[qid]
            by_split_type[split_name][item.get("query_type", "factual")] += 1
            by_split_cat[split_name][cell_key(item)[0]] += 1

    total = sum(len(result[s]) for s in ("train", "val", "test"))
    print()
    print("=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)
    print(f"\nTotal queries        : {total}")
    for split_name in ("train", "val", "test"):
        n = len(result[split_name])
        pct = 100.0 * n / total if total else 0.0
        print(f"  {split_name:5s}  {n:4d}  ({pct:5.1f}%)")

    print("\nBy query_type per split:")
    types = ["factual", "paraphrased", "multihop"]
    print(f"  {'split':6s}  " + "  ".join(f"{t:>12s}" for t in types))
    for split_name in ("train", "val", "test"):
        cells = "  ".join(f"{by_split_type[split_name].get(t, 0):>12d}" for t in types)
        print(f"  {split_name:6s}  {cells}")

    cats = sorted({c for s in by_split_cat.values() for c in s.keys()})
    print(f"\nBy category per split (n={len(cats)} categories):")
    print(f"  {'category':22s}  {'train':>6s}  {'val':>6s}  {'test':>6s}  {'total':>6s}")
    for cat in cats:
        tr = by_split_cat["train"].get(cat, 0)
        va = by_split_cat["val"].get(cat, 0)
        te = by_split_cat["test"].get(cat, 0)
        print(f"  {cat:22s}  {tr:>6d}  {va:>6d}  {te:>6d}  {tr + va + te:>6d}")

    cells_no_val = [c for c in result["cells"] if c["n_val"] == 0 and c["n_total"] > 0]
    cells_no_test = [c for c in result["cells"] if c["n_test"] == 0 and c["n_total"] > 0]
    print(f"\nCoverage notes:")
    print(f"  Cells with 0 val  : {len(cells_no_val)}  (mostly N=1 cells, all to train)")
    print(f"  Cells with 0 test : {len(cells_no_test)}")
    if cells_no_val:
        joined = ", ".join(f"{c['category']}/{c['query_type']}(N={c['n_total']})"
                           for c in cells_no_val[:8])
        suffix = "..." if len(cells_no_val) > 8 else ""
        print(f"  No-val cells      : {joined}{suffix}")

    print("\nFingerprints (sha256):")
    for name in ("train", "val", "test"):
        print(f"  {name:6s}  {sha256_of_list(result[name])}")
    print()


def verify(testset: dict, seed: int, ratio: tuple[float, float, float]) -> int:
    """Re-derive splits and compare against the on-disk manifest."""
    manifest_path = SPLITS_DIR / "split_manifest.json"
    if not manifest_path.exists():
        print(f"Manifest not found, {manifest_path}. Run without --verify first.")
        return 1

    result = split(testset, seed, ratio)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    expected = manifest.get("fingerprints", {})
    failed = False
    for name in ("train", "val", "test"):
        actual = sha256_of_list(result[name])
        match = "OK" if actual == expected.get(name) else "MISMATCH"
        print(f"  {name:6s}  {match}  expected={expected.get(name, '?')[:16]}...  actual={actual[:16]}...")
        if actual != expected.get(name):
            failed = True
    return 1 if failed else 0


def main() -> int:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Stratified 70/15/15 train/val/test split for GT.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"Base seed for per-cell shuffles (default {DEFAULT_SEED}).")
    ap.add_argument("--train", type=float, default=DEFAULT_RATIO[0],
                    help=f"Train ratio (default {DEFAULT_RATIO[0]}).")
    ap.add_argument("--val", type=float, default=DEFAULT_RATIO[1],
                    help=f"Val ratio (default {DEFAULT_RATIO[1]}).")
    ap.add_argument("--test", type=float, default=DEFAULT_RATIO[2],
                    help=f"Test ratio (default {DEFAULT_RATIO[2]}).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute split but do not write files.")
    ap.add_argument("--stats", action="store_true",
                    help="Print stats. Implied by --dry-run.")
    ap.add_argument("--verify", action="store_true",
                    help="Re-derive split and compare hashes against existing manifest.")
    args = ap.parse_args()

    ratio_sum = args.train + args.val + args.test
    if abs(ratio_sum - 1.0) > 1e-6:
        raise SystemExit(f"Ratios must sum to 1.0, got {ratio_sum}")
    ratio = (args.train, args.val, args.test)

    testset = load_testset()
    print(f"Loaded testset, {len(testset)} queries")

    if args.verify:
        return verify(testset, args.seed, ratio)

    result = split(testset, args.seed, ratio)

    if args.stats or args.dry_run:
        print_stats(testset, result)

    if args.dry_run:
        print("Dry run, no files written.")
        return 0

    write_outputs(result, args.seed, ratio)
    print(f"Wrote {SPLITS_DIR}/{{train,val,test}}_qids.json and split_manifest.json")
    print()
    print("Next.")
    print("  python scripts/gt/split_dataset.py --verify     # confirm reproducibility")
    print("  python scripts/sync_data.py --push              # backup to HF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
