"""Pre-run sanity checks.

Validates that the planned run is feasible before burning hours on it:
  - every gold_doc_id in the testset has an index file at each granularity
  - Gemini API is reachable (if any LLM-driven system is requested)
"""

from __future__ import annotations

import time
from collections import Counter
from pathlib import Path

# Mirror the category-derivation logic used by vectorless/retrieval/common.py
# without creating a hard import dependency on the retrieval package.
_MULTI_WORD_PREFIXES = {
    "peraturan-bssn": "PERATURAN_BSSN",
    "peraturan-ojk": "PERATURAN_OJK",
}


def doc_category(doc_id: str) -> str:
    low = (doc_id or "").lower()
    for prefix, folder in _MULTI_WORD_PREFIXES.items():
        if low.startswith(prefix + "-"):
            return folder
    return (doc_id or "").split("-")[0].upper()


GRANULARITY_INDEX_DIR = {
    "pasal": "data/index_pasal",
    "ayat": "data/index_ayat",
    "rincian": "data/index_rincian",
}


def check_index_coverage(
    repo_root: Path,
    testset: dict[str, dict],
    granularities: list[str],
) -> dict[str, list[str]]:
    """Return {granularity: [missing_doc_ids]} for docs referenced by the testset."""
    gold_doc_ids = sorted({
        item.get("gold_doc_id", "") for item in testset.values() if item.get("gold_doc_id")
    })
    missing_by_gran: dict[str, list[str]] = {}
    for gran in granularities:
        index_dir = repo_root / GRANULARITY_INDEX_DIR.get(gran, f"data/index_{gran}")
        missing: list[str] = []
        for doc_id in gold_doc_ids:
            path = index_dir / doc_category(doc_id) / f"{doc_id}.json"
            if not path.exists():
                missing.append(doc_id)
        if missing:
            missing_by_gran[gran] = missing
    return missing_by_gran


def check_gemini_reachable(timeout_s: float = 10.0) -> tuple[bool, str]:
    """Ping Gemini with a tiny request. Returns (ok, message)."""
    import os
    # retrieval/common.py loads .env at import time; the pre-flight check runs
    # outside that import path, so load it here too.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return False, "GEMINI_API_KEY not set"
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        return False, "google.genai package not installed"
    try:
        client = genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(timeout=int(timeout_s * 1000)),
        )
        t0 = time.time()
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="ping",
            config=genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(thinking_budget=0),
                max_output_tokens=8,
            ),
        )
        _ = response.text  # touch to force resolution
        elapsed_ms = int((time.time() - t0) * 1000)
        return True, f"{elapsed_ms}ms"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def query_distribution(testset: dict[str, dict], qids: list[str]) -> dict:
    """Summarise the selected queries across key strata."""
    selected = [testset[q] for q in qids if q in testset]
    docs = {item.get("gold_doc_id", "") for item in selected}
    categories = Counter(doc_category(item.get("gold_doc_id", "")) for item in selected)
    ref_mode = Counter(item.get("reference_mode", "") for item in selected)
    difficulty = Counter(item.get("difficulty", "") for item in selected)
    query_style = Counter(item.get("query_style", "") for item in selected)
    return {
        "num_queries": len(selected),
        "num_docs": len(docs),
        "reference_mode": dict(ref_mode),
        "category": dict(categories),
        "difficulty": dict(difficulty),
        "query_style": dict(query_style),
    }
