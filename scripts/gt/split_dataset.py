"""Stratified 40/30/30 train/val/test split of the GT testset.

Reads data/validated_testset.pkl and assigns each query to one of three
splits via per-cell allocation, where a cell is a (category, query_type)
pair. Per-cell seeds are derived from f"{seed}-{category}-{query_type}"
so the assignment is deterministic and independent across cells.

Ratio rationale (retrieval-eval, not model training):
  - train (40%, ~285q): Stage 1 broad scan over all 18 (method, gran) cells.
    Largest because Stage 1 runs every method and needs statistical power to
    rank them reliably (Delta R@10 ~0.04 detection threshold).
  - val (30%, ~214q): Stage 2 hyperparameter tuning of the Stage 1 winner.
    Sized to discriminate 3-9 HP configs of a single method plus headroom
    for sub-group analysis.
  - test (30%, ~214q): Stage 3 final report. Sized for publication-quality
    bootstrap CI on R@10 (~ +/- 0.03) and RQ3 paired randomization test
    (n >= 100 threshold).

This deviates from the ML default 70/15/15 because there is no model
training here. LLMs are frozen, BM25 is deterministic; "train" is really
HP search, which needs less data than the broad method scan.

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
DEFAULT_RATIO = (0.40, 0.30, 0.30)


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


SPLIT_NAMES = ("train", "val", "test")


def hamilton_targets(total: int, ratio: tuple[float, float, float]) -> dict[str, int]:
    """Return global per-split target counts via Hamilton's largest-remainder.

    Floor each exact share, then distribute the leftover slots one by one to
    the splits with the largest fractional remainder. Tie-break by SPLIT_NAMES
    order (train > val > test) for determinism.
    """
    exact = {name: ratio[i] * total for i, name in enumerate(SPLIT_NAMES)}
    floors = {name: int(exact[name]) for name in SPLIT_NAMES}
    leftover = total - sum(floors.values())
    fractions = {name: exact[name] - floors[name] for name in SPLIT_NAMES}
    order = sorted(SPLIT_NAMES,
                   key=lambda s: (-fractions[s], SPLIT_NAMES.index(s)))
    counts = dict(floors)
    for i in range(leftover):
        counts[order[i]] += 1
    return counts


def allocate_cell(
    qids: list[str],
    seed: int,
    cat: str,
    qtype: str,
    ratio: tuple[float, float, float],
    rr_state: dict,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """Allocate one (category, query_type) cell via Hamilton's largest-remainder.

    Returns (allocs, fractions). allocs is {train, val, test: [qids]}. fractions
    is the per-split fractional preference (exact - floor) used later by the
    global rebalance pass to decide which cells donate slots.

    Determinism, sort qids alphabetically, then shuffle with a per-cell seed
    derived from f"{seed}-{cat}-{qtype}". Cell N=1 goes entirely to train. Cell
    N=2 sends the first to train and the second to whichever of val or test has
    fewer queries so far (round-robin via rr_state). Cells N>=3 use Hamilton,
    floor each exact share then assign the leftover slots to splits with the
    largest fractional remainder.
    """
    qids_sorted = sorted(qids)
    rng = random.Random(f"{seed}-{cat}-{qtype}")
    rng.shuffle(qids_sorted)

    n = len(qids_sorted)
    empty_fracs = {s: 0.0 for s in SPLIT_NAMES}
    if n == 0:
        return {"train": [], "val": [], "test": []}, empty_fracs
    if n == 1:
        return {"train": qids_sorted[:], "val": [], "test": []}, empty_fracs
    if n == 2:
        first = qids_sorted[:1]
        second = qids_sorted[1:]
        if rr_state["val"] <= rr_state["test"]:
            rr_state["val"] += 1
            return {"train": first, "val": second, "test": []}, empty_fracs
        rr_state["test"] += 1
        return {"train": first, "val": [], "test": second}, empty_fracs

    exact = {name: ratio[i] * n for i, name in enumerate(SPLIT_NAMES)}
    floors = {name: int(exact[name]) for name in SPLIT_NAMES}
    leftover = n - sum(floors.values())
    fractions = {name: exact[name] - floors[name] for name in SPLIT_NAMES}
    order = sorted(SPLIT_NAMES,
                   key=lambda s: (-fractions[s], SPLIT_NAMES.index(s)))
    counts = dict(floors)
    for i in range(leftover):
        counts[order[i]] += 1

    cursor = 0
    allocs: dict[str, list[str]] = {}
    for name in SPLIT_NAMES:
        allocs[name] = qids_sorted[cursor:cursor + counts[name]]
        cursor += counts[name]
    return allocs, fractions


def global_rebalance(per_cell: list[dict], targets: dict[str, int]) -> None:
    """Shift slots between splits within cells until totals match targets.

    per_cell is a list of {cell_key, allocs, fractions, n}. Mutates in place.
    Each iteration picks the most over-allocated split (donor side) and the
    most under-allocated split (recipient side), then moves one qid from a
    donor cell. Donor preference order, cells with the largest current
    over_split allocation first (so donating one slot does not skew the cell
    much), then cells where over_split's fractional was lowest (the slot was
    the cell's least-preferred), then by cell_key for deterministic tie-break.
    Sorting by current allocation descending spreads donations across many
    cells instead of draining a single cell repeatedly.
    """
    def totals() -> dict[str, int]:
        out = {s: 0 for s in SPLIT_NAMES}
        for entry in per_cell:
            for s in SPLIT_NAMES:
                out[s] += len(entry["allocs"][s])
        return out

    cur = totals()
    safety = 1000
    while cur != targets and safety > 0:
        safety -= 1
        diffs = {s: targets[s] - cur[s] for s in SPLIT_NAMES}
        under = max(diffs, key=lambda s: diffs[s])
        over = min(diffs, key=lambda s: diffs[s])
        if diffs[under] <= 0 or diffs[over] >= 0:
            break

        candidates = [e for e in per_cell if len(e["allocs"][over]) > 0]
        if not candidates:
            raise RuntimeError(
                f"Cannot rebalance, no cell has {over} allocation to donate."
            )
        candidates.sort(key=lambda e: (
            -len(e["allocs"][over]),
            e["fractions"][over],
            e["cell_key"],
        ))
        donor = candidates[0]

        qid = donor["allocs"][over].pop()
        donor["allocs"][under].append(qid)
        cur[over] -= 1
        cur[under] += 1

    if cur != targets:
        raise RuntimeError(f"Rebalance failed to converge, totals={cur} targets={targets}")


def split(testset: dict, seed: int, ratio: tuple[float, float, float]) -> dict:
    """Build the split assignment for the entire testset.

    Returns a dict with train, val, test qid lists plus per-cell stats.
    Two-phase, Hamilton per cell then global rebalance to hit exact targets.
    """
    cells: dict[tuple[str, str], list[str]] = defaultdict(list)
    for qid, item in testset.items():
        cells[cell_key(item)].append(qid)

    rr_state = {"val": 0, "test": 0}
    per_cell: list[dict] = []
    for (cat, qtype) in sorted(cells.keys()):
        qids = cells[(cat, qtype)]
        allocs, fractions = allocate_cell(qids, seed, cat, qtype, ratio, rr_state)
        per_cell.append({
            "cell_key": f"{cat}__{qtype}",
            "category": cat,
            "query_type": qtype,
            "n": len(qids),
            "allocs": allocs,
            "fractions": fractions,
        })

    targets = hamilton_targets(sum(len(e["allocs"]["train"])
                                    + len(e["allocs"]["val"])
                                    + len(e["allocs"]["test"])
                                    for e in per_cell), ratio)

    global_rebalance(per_cell, targets)

    train_all: list[str] = []
    val_all: list[str] = []
    test_all: list[str] = []
    cell_stats: list[dict] = []
    for entry in per_cell:
        train_all.extend(entry["allocs"]["train"])
        val_all.extend(entry["allocs"]["val"])
        test_all.extend(entry["allocs"]["test"])
        cell_stats.append({
            "category": entry["category"],
            "query_type": entry["query_type"],
            "n_total": entry["n"],
            "n_train": len(entry["allocs"]["train"]),
            "n_val": len(entry["allocs"]["val"]),
            "n_test": len(entry["allocs"]["test"]),
        })

    train_all.sort()
    val_all.sort()
    test_all.sort()

    return {
        "train": train_all,
        "val": val_all,
        "test": test_all,
        "cells": cell_stats,
        "targets": targets,
    }


def sha256_of_list(items: list[str]) -> str:
    """Hash a sorted qid list to detect drift."""
    payload = "\n".join(sorted(items)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def to_jsonable(value):
    """Convert sets to sorted lists so the record is JSON-serializable."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_jsonable(v) for v in value]
    return value


def write_jsonl_split(testset: dict, qids: list[str], path: Path) -> None:
    """Emit one JSONL per split, full GT records with sets serialized as lists.

    Powers the HF Dataset Viewer (tabular preview) for the published mirror.
    Eval pipeline does not read these files, it consumes validated_testset.pkl
    plus the qid lookup. Re-derived from the pickle every split run, so they
    cannot drift from the canonical source.
    """
    with open(path, "w", encoding="utf-8") as f:
        for qid in sorted(qids):
            item = testset[qid]
            record = {"query_id": qid}
            for key, value in item.items():
                record[key] = to_jsonable(value)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_outputs(testset: dict, result: dict, seed: int, ratio: tuple[float, float, float]) -> None:
    """Persist split files, manifest, and per-split JSONL records."""
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("train", "val", "test"):
        path = SPLITS_DIR / f"{name}_qids.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(result[name], f, indent=2)
            f.write("\n")
        write_jsonl_split(testset, result[name], SPLITS_DIR / f"{name}.jsonl")

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
    targets = result.get("targets", {})
    print()
    print("=" * 70)
    print("SPLIT STATISTICS")
    print("=" * 70)
    print(f"\nTotal queries        : {total}")
    for split_name in ("train", "val", "test"):
        n = len(result[split_name])
        pct = 100.0 * n / total if total else 0.0
        target = targets.get(split_name)
        if target is not None:
            delta = n - target
            delta_str = f"  target={target}  delta={delta:+d}"
        else:
            delta_str = ""
        print(f"  {split_name:5s}  {n:4d}  ({pct:5.1f}%){delta_str}")

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

    write_outputs(testset, result, args.seed, ratio)
    print(f"Wrote {SPLITS_DIR}/{{train,val,test}}_qids.json")
    print(f"Wrote {SPLITS_DIR}/{{train,val,test}}.jsonl  (HF Viewer)")
    print(f"Wrote {SPLITS_DIR}/split_manifest.json")
    print()
    print("Next.")
    print("  python scripts/gt/split_dataset.py --verify     # confirm reproducibility")
    print("  python scripts/sync_data.py --push              # backup to HF")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
