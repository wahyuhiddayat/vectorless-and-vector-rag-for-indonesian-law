"""Document-id parsing utilities shared by indexing and retrieval.

`doc_category(doc_id)` maps a doc_id like `peraturan-kpu-15-2024` to the
canonical jenis folder (`PERATURAN_KPU`). It uses the doc_id_prefix on
each `Category` entry in `vectorless.categories.CATEGORIES`, so adding a
new category in one place automatically wires the prefix lookup here.
"""
from .categories import CATEGORIES


def _build_prefix_table() -> list[tuple[str, str]]:
    """Return [(prefix, folder)] sorted by prefix length descending.

    Longer prefixes win on ambiguous matches (e.g. `permen-atr-kepala-bpn-`
    must take precedence over a hypothetical shorter `permen-atr-`).
    """
    table = [(c.doc_id_prefix(), c.folder) for c in CATEGORIES]
    return sorted(table, key=lambda pair: -len(pair[0]))


_PREFIX_TABLE = _build_prefix_table()


def doc_category(doc_id: str) -> str:
    """Map doc_id to its category folder (e.g. `uu-1-2025` -> `UU`).

    Raises ValueError when no registered Category prefix matches; this
    surfaces unregistered scraper output instead of silently falling back
    to the wrong folder.
    """
    low = doc_id.lower()
    for prefix, folder in _PREFIX_TABLE:
        if low.startswith(prefix + "-"):
            return folder
    raise ValueError(
        f"doc_id {doc_id!r} matches no registered Category prefix; "
        f"add an entry to vectorless.categories.CATEGORIES"
    )
