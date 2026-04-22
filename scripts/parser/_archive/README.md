# Archived scripts

Scripts here are **no longer part of the active pipeline**. Kept for historical
reference and potential rollback only.

## Contents

- **`llm_fix.py`** — pre-LLM-parse pipeline that called Gemini to *patch*
  regex parser output with specific issue types (empty pasals, bleed,
  underspilt). Superseded by `scripts/parser/llm_parse.py` (full LLM-first
  structure generation). Shared helpers (`count_pasals_in_tree`,
  `format_pdf_pages`, `load_pdf_pages`, `load_pdf_text`, `parse_llm_json`,
  `_normalize_keys`) have been extracted to `scripts/parser/_common.py`.

- **`rewrite_node_ids.py`** — one-off migration from legacy `0001` /
  `P000` node_id format to readable `pasal_3_ayat_2_huruf_a` format.
  Already applied to the corpus; not needed again.

- **`detect_unsplit.py`** — diagnostic scanner for regex-parser output
  that looked for `\nPasal N\n<letter>` patterns (missed Pasal splits).
  Irrelevant for LLM-parse output which rarely has this failure mode.

## Do not import

Production code should not import from `_archive/`. If you need a helper
that lives here, copy it out to the active `scripts/parser/` folder or
`scripts/parser/_common.py`.
