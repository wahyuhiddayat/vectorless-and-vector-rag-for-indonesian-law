"""Pre-run sanity checks.

Validates that the planned run is feasible before burning hours on it.
  - every gold_doc_id in the testset has an index file at each granularity
  - Gemini API is reachable (if any LLM-driven system is requested)
  - Qdrant is reachable (if running the vector path)
  - leaf-count invariant per doc, rincian >= ayat >= pasal
  - GT provenance, fingerprint of the testset file for reproducibility
"""

from __future__ import annotations

import hashlib
import json
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
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    if not os.environ.get("GEMINI_API_KEY"):
        return False, "GEMINI_API_KEY not set"
    try:
        from google.genai import types as genai_types
        from vectorless.llm import MODEL, client as gemini_client
    except ImportError as exc:
        return False, f"import failed: {exc}"
    try:
        cfg_kwargs: dict = {"max_output_tokens": 8}
        if MODEL.startswith("gemini-2.5"):
            cfg_kwargs["thinking_config"] = genai_types.ThinkingConfig(thinking_budget=0)
        t0 = time.time()
        response = gemini_client().models.generate_content(
            model=MODEL,
            contents="ping",
            config=genai_types.GenerateContentConfig(**cfg_kwargs),
        )
        _ = response.text
        elapsed_ms = int((time.time() - t0) * 1000)
        return True, f"{elapsed_ms}ms"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_qdrant_reachable(
    qdrant_path: str | None,
    qdrant_url: str | None,
    timeout_s: float = 10.0,
) -> tuple[bool, str]:
    """Verify the Qdrant store is reachable.

    Local mode, qdrant_path must point to an existing dir. Server mode,
    qdrant_url must respond to GET /collections within timeout_s. Returns
    (ok, message) following the convention of check_gemini_reachable().
    """
    if qdrant_path:
        path = Path(qdrant_path)
        if not path.exists():
            return False, f"qdrant path does not exist, {qdrant_path}"
        if not path.is_dir():
            return False, f"qdrant path is not a directory, {qdrant_path}"
        return True, f"local at {qdrant_path}"

    url = qdrant_url
    if not url:
        return False, "neither qdrant_path nor qdrant_url given"
    try:
        from urllib import request
        req = request.Request(url.rstrip("/") + "/collections")
        t0 = time.time()
        with request.urlopen(req, timeout=timeout_s) as resp:
            if resp.status != 200:
                return False, f"qdrant /collections returned HTTP {resp.status}"
            elapsed_ms = int((time.time() - t0) * 1000)
            return True, f"server at {url} ({elapsed_ms}ms)"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def check_corpus_consistency(
    repo_root: Path,
    testset: dict[str, dict],
    granularities: list[str],
) -> dict[str, list[str]]:
    """For every doc in the testset, verify rincian >= ayat >= pasal leaves.

    Catches partially re-split docs (the splitter ran on pasal but failed at
    ayat, or skipped rincian). Returns {doc_id, [reasons...]} for offenders.
    Empty dict when everything is consistent.
    """
    if "rincian" not in granularities or "ayat" not in granularities or "pasal" not in granularities:
        # Only a sub-set requested. Skip the cross-granularity invariant.
        return {}

    gold_doc_ids = sorted({
        item.get("gold_doc_id", "")
        for item in testset.values()
        if item.get("gold_doc_id")
    })
    offenders: dict[str, list[str]] = {}
    for doc_id in gold_doc_ids:
        counts = {}
        for gran in ("pasal", "ayat", "rincian"):
            path = repo_root / GRANULARITY_INDEX_DIR[gran] / doc_category(doc_id) / f"{doc_id}.json"
            counts[gran] = _count_leaves(path)
        reasons: list[str] = []
        if counts["rincian"] < counts["ayat"]:
            reasons.append(f"rincian({counts['rincian']}) < ayat({counts['ayat']})")
        if counts["ayat"] < counts["pasal"]:
            reasons.append(f"ayat({counts['ayat']}) < pasal({counts['pasal']})")
        if reasons:
            offenders[doc_id] = reasons
    return offenders


def _count_leaves(path: Path) -> int:
    """Count leaf nodes in an index JSON file. Zero if missing."""
    if not path.exists():
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (json.JSONDecodeError, OSError):
        return 0
    return _walk_leaves(doc.get("structure", []))


def _walk_leaves(nodes: list[dict]) -> int:
    total = 0
    for node in nodes:
        if node.get("nodes"):
            total += _walk_leaves(node["nodes"])
        elif node.get("text"):
            total += 1
    return total


def gt_fingerprint(testset_path: Path) -> dict:
    """Return a fingerprint of the testset file for reproducibility records.

    Captures byte size, mtime, and SHA-256 (truncated to 16 hex). The hash
    lets a future re-run detect that the GT changed even if the filename did
    not, so eval artifacts can be tied back to a specific GT snapshot.
    """
    if not testset_path.exists():
        return {"path": str(testset_path), "exists": False}

    sha = hashlib.sha256()
    size = 0
    with open(testset_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            sha.update(chunk)
            size += len(chunk)

    mtime = testset_path.stat().st_mtime
    return {
        "path": str(testset_path),
        "exists": True,
        "size_bytes": size,
        "sha256_16": sha.hexdigest()[:16],
        "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(mtime)),
    }


def gemini_model_name() -> str | None:
    """Return the configured Gemini model name, or None if unimportable."""
    try:
        from vectorless.llm import MODEL
        return str(MODEL)
    except Exception:
        return None


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
