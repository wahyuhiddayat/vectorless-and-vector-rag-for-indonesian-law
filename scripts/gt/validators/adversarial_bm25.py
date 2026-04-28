"""Validate that adversarial GT queries are properly misleading for BM25.

For each adversarial item, run BM25 over the rincian leaf catalog of the
target's category. The target leaf must NOT appear in the BM25 top-3
(confirms surface mislead) and MUST appear in the BM25 top-50 (confirms
target is still retrievable so a reranker can rescue).

Threshold rationale.
- top-3: if target already at top-3, BM25 alone wins, no adversarial signal.
- top-50: if target outside top-50, even cross-encoder reranking on top-50
  candidates will miss it, query is too hard.

Usage:
    python -m scripts.gt.validators.adversarial_bm25 data/ground_truth_raw/UU/uu-1-2026__adversarial.json
    python -m scripts.gt.validators.adversarial_bm25 --max-rank-out 5 --min-rank-in 100 <path>
"""

import argparse
import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

DATA_INDEX = Path("data/index_rincian")
DEFAULT_MAX_RANK_OUT = 3
DEFAULT_MIN_RANK_IN = 50

INDONESIAN_STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
}


def tokenize(text: str) -> list[str]:
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    return [t for t in tokens if t not in INDONESIAN_STOPWORDS and len(t) > 1]


def collect_category_leaves(category: str) -> tuple[list[str], list[str]]:
    """Return (node_ids, tokenized_texts) across all docs in a category."""
    cat_dir = DATA_INDEX / category
    node_ids: list[str] = []
    texts: list[list[str]] = []

    def _walk(nodes: list[dict]) -> None:
        for node in nodes:
            if "nodes" in node and node["nodes"]:
                _walk(node["nodes"])
            elif node.get("text"):
                node_ids.append(node["node_id"])
                texts.append(tokenize(node["text"]))

    for path in sorted(cat_dir.glob("*.json")):
        if path.name == "catalog.json":
            continue
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        _walk(doc.get("structure", []))

    return node_ids, texts


def validate_file(
    path: Path,
    max_rank_out: int = DEFAULT_MAX_RANK_OUT,
    min_rank_in: int = DEFAULT_MIN_RANK_IN,
) -> tuple[int, int, list[dict]]:
    """Run BM25 gate. Returns (n_adversarial, n_flagged, flagged_items)."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} top-level must be a JSON array")

    flagged: list[dict] = []
    n_adv = 0
    bm25_cache: dict[str, tuple[BM25Okapi, list[str]]] = {}

    category = path.parent.name

    for i, item in enumerate(items, 1):
        if item.get("query_type") != "adversarial":
            continue
        n_adv += 1
        target_id = item.get("gold_anchor_node_id") or item.get("gold_node_id")
        query = item.get("query", "")
        if not target_id or not query:
            flagged.append({"item_index": i, "reason": "missing target or query", "rank": None})
            continue

        if category not in bm25_cache:
            node_ids, texts = collect_category_leaves(category)
            if not node_ids:
                flagged.append({"item_index": i, "reason": f"no leaves found for category {category}", "rank": None})
                continue
            bm25_cache[category] = (BM25Okapi(texts), node_ids)

        bm25, node_ids = bm25_cache[category]
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(zip(scores, node_ids), key=lambda x: -x[0])
        target_rank = next((r for r, (_, nid) in enumerate(ranked, 1) if nid == target_id), None)

        if target_rank is None:
            flagged.append({
                "item_index": i,
                "query": query,
                "target_node_id": target_id,
                "rank": None,
                "reason": "target not found in category index",
            })
            continue
        if target_rank <= max_rank_out:
            flagged.append({
                "item_index": i,
                "query": query,
                "target_node_id": target_id,
                "rank": target_rank,
                "reason": f"target at BM25 rank {target_rank} (must be > {max_rank_out} to qualify as adversarial)",
            })
            continue
        if target_rank > min_rank_in:
            flagged.append({
                "item_index": i,
                "query": query,
                "target_node_id": target_id,
                "rank": target_rank,
                "reason": f"target at BM25 rank {target_rank} (must be <= {min_rank_in} so reranker can rescue)",
            })

    return n_adv, len(flagged), flagged


def main() -> None:
    ap = argparse.ArgumentParser(description="Adversarial BM25 mislead validator")
    ap.add_argument("path", type=str, help="Path to raw GT JSON file (must live under data/ground_truth_raw/<CATEGORY>/)")
    ap.add_argument("--max-rank-out", type=int, default=DEFAULT_MAX_RANK_OUT,
                    help=f"Target must be ranked beyond this position (default {DEFAULT_MAX_RANK_OUT})")
    ap.add_argument("--min-rank-in", type=int, default=DEFAULT_MIN_RANK_IN,
                    help=f"Target must still be within this rank (default {DEFAULT_MIN_RANK_IN})")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    n_adv, n_flag, flagged = validate_file(
        path, max_rank_out=args.max_rank_out, min_rank_in=args.min_rank_in,
    )
    print(f"\nFile: {path}")
    print(f"Adversarial items inspected: {n_adv}")
    print(f"Flagged (rank <= {args.max_rank_out} or rank > {args.min_rank_in}): {n_flag}")
    if flagged:
        print("\nFlagged items:")
        for f in flagged:
            print(f"  item {f['item_index']}: rank={f.get('rank')}, reason={f['reason']}")
            if f.get("query"):
                print(f"    query: {f['query'][:100]}")
        sys.exit(1)
    if n_adv == 0:
        print("(no adversarial items in this file, nothing to validate)")
    else:
        print("All adversarial items pass BM25 mislead check.")


if __name__ == "__main__":
    main()
