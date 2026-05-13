"""Leaf text length distribution per granularity.

Output:
  - stdout: per-granularity summary table
  - scripts/analysis/leaf_text_lengths_output.json: full distribution data

Usage:
    python scripts/analysis/leaf_text_lengths.py
"""

import json
from pathlib import Path


def analyze_granularity(granularity: str) -> dict:
    """Walk all docs at one granularity and collect leaf text lengths."""
    idx_dir = Path("data") / f"index_{granularity}"
    lengths = []
    for p in sorted(idx_dir.rglob("*.json")):
        if p.name == "catalog.json":
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue

        def walk(nodes):
            for n in nodes or []:
                if n.get("nodes"):
                    yield from walk(n["nodes"])
                elif n.get("text"):
                    yield len(n["text"])

        lengths.extend(walk(doc.get("structure", [])))

    if not lengths:
        return {"granularity": granularity, "n_leaves": 0}

    n = len(lengths)
    s = sorted(lengths)
    avg = sum(s) / n
    med = s[n // 2]
    p95 = s[int(n * 0.95)]
    p99 = s[int(n * 0.99)]
    mx = s[-1]
    pct_above_5k = sum(1 for l in lengths if l > 5000) / n * 100
    pct_above_10k = sum(1 for l in lengths if l > 10000) / n * 100

    result = {
        "granularity": granularity,
        "n_leaves": n,
        "avg": round(avg),
        "median": med,
        "p95": p95,
        "p99": p99,
        "max": mx,
        "pct_above_5000": round(pct_above_5k, 1),
        "pct_above_10000": round(pct_above_10k, 1),
    }

    print(f"{granularity}:")
    print(f"  count={n}")
    print(f"  avg={result['avg']}  median={med}  p95={p95}  p99={p99}  max={mx}")
    print(f"  >5000 chars: {result['pct_above_5000']}%")
    print(f"  >10000 chars: {result['pct_above_10000']}%")

    return result


def main() -> None:
    results = {}
    for gran in ["pasal", "ayat", "rincian"]:
        r = analyze_granularity(gran)
        results[gran] = r
        print()

    out_path = Path("scripts/analysis/leaf_text_lengths_output.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Written to {out_path}")


if __name__ == "__main__":
    main()
