"""Document-id parsing utilities shared by indexing and retrieval."""

_MULTI_WORD_PREFIXES = {
    "peraturan-bssn": "PERATURAN_BSSN",
    "peraturan-ojk": "PERATURAN_OJK",
}


def doc_category(doc_id: str) -> str:
    """Map doc_id to its category folder (e.g. 'uu-1-2025' -> 'UU')."""
    low = doc_id.lower()
    for prefix, folder in _MULTI_WORD_PREFIXES.items():
        if low.startswith(prefix + "-"):
            return folder
    return doc_id.split("-")[0].upper()
