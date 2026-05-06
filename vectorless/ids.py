"""Document-id parsing utilities shared by indexing and retrieval."""

_MULTI_WORD_PREFIXES = {
    "peraturan-bi": "PERATURAN_BI",
    "peraturan-bpom": "PERATURAN_BPOM",
    "peraturan-bssn": "PERATURAN_BSSN",
    "peraturan-menag": "PERMENAG",
    "peraturan-ojk": "PERATURAN_OJK",
    "peraturan-polri": "PERATURAN_POLRI",
    "permen-atr-kepala-bpn": "PERMEN_ATRBPN",
    "permen-bumn": "PERMENBUMN",
    "permen-esdm": "PERMEN_ESDM",
    "permen-komdigi": "PERMENKOMDIGI",
    "permen-perin": "PERMENPERIN",
    "permen-pupr": "PERMEN_PUPR",
    "perma": "PERATURAN_MA",
}


def doc_category(doc_id: str) -> str:
    """Map doc_id to its category folder (e.g. 'uu-1-2025' -> 'UU')."""
    low = doc_id.lower()
    for prefix, folder in _MULTI_WORD_PREFIXES.items():
        if low.startswith(prefix + "-"):
            return folder
    return doc_id.split("-")[0].upper()
