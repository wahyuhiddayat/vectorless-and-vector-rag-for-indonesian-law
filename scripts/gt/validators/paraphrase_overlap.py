"""Validate that paraphrased GT queries have low lexical overlap with their anchor.

Computes Jaccard similarity over content unigrams (Indonesian stopwords stripped)
between the query and the anchor leaf text. Items above the threshold are flagged
as not sufficiently paraphrased and should be regenerated.

Usage:
    python -m scripts.gt.validators.paraphrase_overlap data/ground_truth_raw/UU/uu-1-2026__paraphrased.json
    python -m scripts.gt.validators.paraphrase_overlap --threshold 0.30 <path>
"""

import argparse
import json
import re
import sys
from pathlib import Path

DATA_INDEX = Path("data/index_rincian")
DEFAULT_THRESHOLD = 0.40

INDONESIAN_STOPWORDS = {
    "dan", "atau", "yang", "di", "ke", "dari", "untuk", "dengan",
    "pada", "dalam", "ini", "itu", "adalah", "oleh", "sebagai",
    "tidak", "akan", "telah", "dapat", "harus", "setiap", "suatu",
    "antara", "atas", "secara", "serta", "bahwa", "tentang",
    "berdasarkan", "sebagaimana", "dimaksud", "tersebut",
    "ayat", "huruf", "angka", "pasal", "bab", "bagian",
    "apa", "siapa", "kapan", "berapa", "bagaimana", "kenapa", "mengapa",
    "saja", "juga", "saat", "ada", "lain", "sama",
    "yaitu", "ialah", "menjadi", "lebih", "kurang", "atau",
    "semua", "para", "hal", "agar", "bila", "jika", "kalau",
}


def tokenize(text: str) -> set[str]:
    """Return content tokens after lowercase, alpha-only filter, and stopword removal."""
    lowered = (text or "").lower()
    tokens = re.findall(r"[a-zA-Z]+", lowered)
    return {t for t in tokens if t not in INDONESIAN_STOPWORDS and len(t) > 2}


def jaccard(a: set[str], b: set[str]) -> float:
    """Standard Jaccard similarity on token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_doc_path(doc_id: str) -> Path | None:
    for path in DATA_INDEX.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path
    return None


def collect_leaf_text_map(doc: dict) -> dict[str, str]:
    """Return mapping from node_id to leaf text for a parsed doc."""
    out: dict[str, str] = {}

    def _walk(nodes: list[dict]) -> None:
        for node in nodes:
            if "nodes" in node and node["nodes"]:
                _walk(node["nodes"])
            elif node.get("text"):
                out[node["node_id"]] = node["text"]

    _walk(doc.get("structure", []))
    return out


def validate_file(path: Path, threshold: float = DEFAULT_THRESHOLD) -> tuple[int, int, list[dict]]:
    """Validate a raw GT file. Returns (n_paraphrased, n_flagged, flagged_items)."""
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        raise ValueError(f"{path} top-level must be a JSON array")

    leaf_text_cache: dict[str, dict[str, str]] = {}
    flagged: list[dict] = []
    n_paraphrased = 0

    for i, item in enumerate(items, 1):
        if item.get("query_type") != "paraphrased":
            continue
        n_paraphrased += 1
        doc_id = item.get("gold_doc_id")
        anchor_id = item.get("gold_anchor_node_id") or item.get("gold_node_id")
        query = item.get("query", "")
        if doc_id not in leaf_text_cache:
            doc_path = find_doc_path(doc_id)
            if not doc_path:
                flagged.append({"item_index": i, "reason": f"doc '{doc_id}' not found", "score": None})
                continue
            with open(doc_path, encoding="utf-8") as f:
                leaf_text_cache[doc_id] = collect_leaf_text_map(json.load(f))
        anchor_text = leaf_text_cache[doc_id].get(anchor_id, "")
        if not anchor_text:
            flagged.append({"item_index": i, "reason": f"anchor '{anchor_id}' not in {doc_id}", "score": None})
            continue
        score = jaccard(tokenize(query), tokenize(anchor_text))
        if score > threshold:
            flagged.append({
                "item_index": i,
                "query": query,
                "anchor_node_id": anchor_id,
                "score": round(score, 3),
                "reason": f"Jaccard {score:.3f} > threshold {threshold}",
            })

    return n_paraphrased, len(flagged), flagged


def main() -> None:
    ap = argparse.ArgumentParser(description="Paraphrase lexical-overlap validator")
    ap.add_argument("path", type=str, help="Path to raw GT JSON file")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                    help=f"Jaccard reject threshold (default {DEFAULT_THRESHOLD})")
    args = ap.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    n_par, n_flag, flagged = validate_file(path, threshold=args.threshold)
    print(f"\nFile: {path}")
    print(f"Paraphrased items inspected: {n_par}")
    print(f"Flagged (overlap > {args.threshold}): {n_flag}")
    if flagged:
        print("\nFlagged items:")
        for f in flagged:
            print(f"  item {f['item_index']}: score={f.get('score')}, reason={f['reason']}")
            if f.get("query"):
                print(f"    query: {f['query'][:100]}")
        sys.exit(1)
    if n_par == 0:
        print("(no paraphrased items in this file, nothing to validate)")
    else:
        print("All paraphrased items pass overlap check.")


if __name__ == "__main__":
    main()
