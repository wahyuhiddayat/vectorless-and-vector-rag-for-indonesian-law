"""Canonical BPK category registry.

Each Category bundles BPK jenis_id, short folder name, scope group, and
parser model. Scraper derives JENIS_MAP and KATEGORI_MAP from this tuple.
Parser resolves the per-doc model via parse_model_for_category().

Adding a category = one Category() entry. Removing = delete the entry.
No other config files to touch.

Parser model selection rationale.
  - 28 fixed BPK categories (per Notes/01-corpus/categories.md) use the
    DEFAULT_PARSER_MODEL (gpt-5), parser frozen per ADR-001.
  - Expansion categories added after 2026-05-07 use deepseek-v4-pro,
    validated via the bake-off in ADR-008 (verdict parity 5/6 OK with
    gpt-5, see Notes/05-experiments/2026-05-07-deepseek-bakeoff/results.md).
  - Cross-family rule (parser != judge) holds for both gpt-5 and
    deepseek-v4-pro against the Gemini 2.5 Pro judge.
"""
from dataclasses import dataclass

DEFAULT_PARSER_MODEL = "gpt-5"


@dataclass(frozen=True)
class Category:
    """One BPK category and the metadata downstream pipelines need.

    Attributes:
        jenis_id: BPK Search API jenis parameter value.
        folder: Short uppercase folder name used in data/<stage>/<folder>/.
        scope: Broad scope group, one of "Pusat", "Daerah",
            "Kementerian/Lembaga".
        parser_model: Model name passed to the LLM dispatcher when parsing
            documents in this category. Defaults to DEFAULT_PARSER_MODEL.
        prefix: Doc-id prefix used to map a scraped doc_id back to this
            category (`vectorless.ids.doc_category`). When empty, derived
            from the folder name. Override for categories where the
            scraped slug doesn't match the folder, e.g. `PERMENAG` slugs
            as `peraturan-menag` and `PERATURAN_MA` slugs as `perma`.
    """

    jenis_id: int
    folder: str
    scope: str
    parser_model: str = DEFAULT_PARSER_MODEL
    prefix: str = ""

    def doc_id_prefix(self) -> str:
        """Return the doc_id prefix for this category."""
        return self.prefix or self.folder.lower().replace("_", "-")


CATEGORIES: tuple[Category, ...] = (
    # Pusat (4)
    Category(8, "UU", "Pusat"),
    Category(9, "PERPU", "Pusat"),
    Category(10, "PP", "Pusat"),
    Category(11, "PERPRES", "Pusat"),
    # Daerah (4)
    Category(19, "PERDA", "Daerah"),
    Category(20, "PERGUB", "Daerah"),
    Category(23, "PERBUP", "Daerah"),
    Category(30, "PERWALI", "Daerah"),
    # Kementerian/Lembaga (20)
    Category(154, "PERMEN_PUPR", "Kementerian/Lembaga"),
    Category(40, "PERMENDAGRI", "Kementerian/Lembaga"),
    Category(42, "PMK", "Kementerian/Lembaga"),
    Category(69, "PERMENPERIN", "Kementerian/Lembaga"),
    Category(170, "PERMENAG", "Kementerian/Lembaga", prefix="peraturan-menag"),
    Category(241, "PERATURAN_POLRI", "Kementerian/Lembaga"),
    Category(54, "PERATURAN_BSSN", "Kementerian/Lembaga"),
    Category(202, "PERMENBUMN", "Kementerian/Lembaga", prefix="permen-bumn"),
    Category(67, "PERMENDAG", "Kementerian/Lembaga"),
    Category(186, "PERMENDIKBUD", "Kementerian/Lembaga"),
    Category(78, "PERATURAN_BI", "Kementerian/Lembaga"),
    Category(80, "PERATURAN_OJK", "Kementerian/Lembaga"),
    Category(95, "PERATURAN_MA", "Kementerian/Lembaga", prefix="perma"),
    Category(105, "PERMENAKER", "Kementerian/Lembaga"),
    Category(278, "PERMENKOMDIGI", "Kementerian/Lembaga"),
    Category(242, "PERMENDIKBUDRISTEK", "Kementerian/Lembaga"),
    Category(147, "PERMEN_ESDM", "Kementerian/Lembaga"),
    Category(111, "PERMEN_ATRBPN", "Kementerian/Lembaga", prefix="permen-atr-kepala-bpn"),
    Category(182, "PERMENKES", "Kementerian/Lembaga"),
    Category(230, "PERATURAN_BPOM", "Kementerian/Lembaga"),
    # Expansion categories (added 2026-05-07+).
    # Note. ADR-008 originally pinned deepseek-v4-pro for expansion based on
    # bake-off cost+quality parity. Reverted to gpt-5 in practice: thinking-on
    # was 14x slower per pasal, thinking-off untested on Indonesian legal text,
    # and the absolute cost saving (~$2 for 24 docs) didn't justify the latency
    # and methodology footnote. See ADR-009 (planned addendum).
    Category(59, "PERATURAN_KPU", "Kementerian/Lembaga"),
)


JENIS_MAP: dict[int, str] = {c.jenis_id: c.folder for c in CATEGORIES}
KATEGORI_MAP: dict[int, str] = {c.jenis_id: c.scope for c in CATEGORIES}
_PARSER_BY_FOLDER: dict[str, str] = {
    c.folder.upper(): c.parser_model for c in CATEGORIES
}


def parse_model_for_category(folder: str) -> str:
    """Return the parser model pinned for this folder.

    Falls back to DEFAULT_PARSER_MODEL when the folder is unknown, which
    keeps ad-hoc parser runs (without a registered category) on the safe
    default rather than failing.
    """
    return _PARSER_BY_FOLDER.get((folder or "").upper(), DEFAULT_PARSER_MODEL)
