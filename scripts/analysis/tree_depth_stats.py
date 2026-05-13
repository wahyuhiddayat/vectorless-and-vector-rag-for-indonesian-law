"""Tree depth and branching statistics across granularities.

Scans index_{pasal,ayat,rincian} and reports depth distribution, branching
factors, and leaf counts. Used to justify beam width and max_rounds defaults
in bm25-tree retrieval.

Output:
  - stdout: per-granularity summary table
  - data/tree_depth_stats.json: full per-doc results for thesis appendix

Usage:
    python scripts/analysis/tree_depth_stats.py
    python scripts/analysis/tree_depth_stats.py --output data/tree_depth_stats.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path

GRANULARITIES = ("pasal", "ayat", "rincian")
BASE_DIR = Path("data")
INDEX_DIRS = {g: BASE_DIR / f"index_{g}" for g in GRANULARITIES}


def analyze_tree(node: dict, depth: int = 0) -> dict:
    """Recurse into one root node; return depth and branching stats."""
    children = node.get("nodes") or []
    kid_count = len(children)
    child_stats = [analyze_tree(c, depth + 1) for c in children]
    max_sub_depth = max((s["max_depth"] for s in child_stats), default=depth)
    total_internal = 1 if kid_count > 0 else 0  # this node is internal
    total_internal += sum(s["total_internal"] for s in child_stats)
    total_leaves = sum(s["total_leaves"] for s in child_stats) if kid_count > 0 else 1
    kids_at_this = {depth: 1} if kid_count > 0 else {}
    kid_count_by_depth: dict[int, int] = {}
    for s in child_stats:
        for d, c in s["kid_count_by_depth"].items():
            kid_count_by_depth[d] = kid_count_by_depth.get(d, 0) + c
    if kid_count > 0:
        kid_count_by_depth[depth] = kid_count_by_depth.get(depth, 0) + kid_count
    return {
        "max_depth": max_sub_depth,
        "total_internal": total_internal,
        "total_leaves": total_leaves,
        "total_nodes": total_internal + total_leaves,
        "kid_count_by_depth": kid_count_by_depth,
    }


def compute_granularity_stats(granularity: str) -> dict:
    """Scan all docs at one granularity and aggregate stats."""
    idx_dir = INDEX_DIRS[granularity]
    paths = list(idx_dir.rglob("*.json"))
    paths = [p for p in paths if p.name != "catalog.json"]
    per_doc = []
    all_depths = []
    all_branching: list[int] = []
    kid_count_dist: Counter = Counter()
    depth_dist: Counter = Counter()
    for path in paths:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            continue
        doc_id = doc.get("doc_id", path.stem)
        structure = doc.get("structure") or []
        if not structure:
            per_doc.append({"doc_id": doc_id, "depth": 0, "internal_nodes": 0, "leaf_nodes": 0, "total_nodes": 0, "note": "empty_structure"})
            continue
        doc_id = doc.get("doc_id", path.stem)
        merged = {"max_depth": 0, "total_internal": 0, "total_leaves": 0, "total_nodes": 0, "kid_count_by_depth": {}}
        for root in structure:
            s = analyze_tree(root)
            merged["max_depth"] = max(merged["max_depth"], s["max_depth"])
            merged["total_internal"] += s["total_internal"]
            merged["total_leaves"] += s["total_leaves"]
            merged["total_nodes"] += s["total_nodes"]
            for d, c in s["kid_count_by_depth"].items():
                merged["kid_count_by_depth"][d] = merged["kid_count_by_depth"].get(d, 0) + c
        all_depths.append(merged["max_depth"] + 1)  # 1-indexed depth
        depth_dist[merged["max_depth"] + 1] += 1
        for d, c in merged["kid_count_by_depth"].items():
            all_branching.append(c)
            kid_count_dist[c] += 1
        doc_info = {
            "doc_id": doc_id,
            "depth": merged["max_depth"] + 1,
            "internal_nodes": merged["total_internal"],
            "leaf_nodes": merged["total_leaves"],
            "total_nodes": merged["total_nodes"],
        }
        per_doc.append(doc_info)
    empty_docs = [d["doc_id"] for d in per_doc if d.get("note") == "empty_structure"]
    if not all_depths:
        return {"granularity": granularity, "n_docs": 0, "n_empty": len(empty_docs), "empty_docs": empty_docs}
    n = len(all_depths)
    avg_depth = sum(all_depths) / n
    sorted_depth = sorted(all_depths)
    p95_depth = sorted_depth[int(n * 0.95)]
    max_depth = max(all_depths)
    avg_branching = sum(all_branching) / len(all_branching) if all_branching else 0
    sorted_branch = sorted(all_branching)
    p95_branch = sorted_branch[int(len(all_branching) * 0.95)] if all_branching else 0
    max_branching = max(all_branching) if all_branching else 0
    return {
        "granularity": granularity,
        "n_files": len(paths),
        "n_docs": n,
        "n_empty": len(empty_docs),
        "empty_docs": empty_docs,
        "avg_depth": round(avg_depth, 2),
        "p95_depth": p95_depth,
        "max_depth": max_depth,
        "depth_distribution": dict(sorted(depth_dist.items())),
        "avg_branching": round(avg_branching, 2),
        "p95_branching": p95_branch,
        "max_branching": max_branching,
        "kid_count_distribution": dict(sorted(kid_count_dist.items())),
        "per_doc": per_doc,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Tree depth and branching statistics across granularities")
    ap.add_argument("--output", default="scripts/analysis/tree_depth_stats_output.json", help="Output path for full JSON (default: scripts/analysis/tree_depth_stats_output.json)")
    args = ap.parse_args()

    results = {}
    for g in GRANULARITIES:
        print(f"\n{'='*50}")
        print(f"  Granularity: {g}")
        print(f"{'='*50}")
        s = compute_granularity_stats(g)
        results[g] = s
        if s["n_docs"] == 0:
            print("  (no docs found)")
            continue
        print(f"  JSON files:          {s['n_files']}")
        print(f"  Valid docs:          {s['n_docs']}")
        if s['n_empty']:
            print(f"  Empty structure:     {s['n_empty']}  ({', '.join(s['empty_docs'])})")
        print(f"  Avg depth:           {s['avg_depth']}")
        print(f"  P95 depth:           {s['p95_depth']}")
        print(f"  Max depth:           {s['max_depth']}")
        print(f"  Depth distribution:  {s['depth_distribution']}")
        print(f"  Avg branching:       {s['avg_branching']}")
        print(f"  P95 branching:       {s['p95_branching']}")
        print(f"  Max branching:       {s['max_branching']}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nFull results written to {args.output}")


if __name__ == "__main__":
    main()
