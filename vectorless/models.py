"""Single source of truth for model pins (non-parser roles).

The parser model is per-category and lives in `vectorless/categories.py`
on each `Category(parser_model=...)` entry. Resolve via
`parse_model_for_category(folder)`.

Other roles each pin one model here. Switch a role here and every call
site picks it up. The `vectorless.llm` dispatcher routes by model prefix:
  gpt-*       -> OpenAI
  claude-*    -> Anthropic
  gemini-*    -> Vertex AI
  deepseek-*  -> DeepSeek (OpenAI-compatible)

Roles.
  SUMMARY_MODEL    Per-node summary. High volume, light per-call.
  OCR_CLEAN_MODEL  Targeted OCR repair on leaf text.
  JUDGE_MODEL      Post-index quality verdict. Cross-family from parser.
  RETRIEVAL_MODEL  LLM nav and hybrid rerank at query time.
"""

SUMMARY_MODEL = "gemini-2.5-flash-lite"
OCR_CLEAN_MODEL = "gemini-2.5-flash-lite"
JUDGE_MODEL = "gemini-2.5-pro"
RETRIEVAL_MODEL = "gemini-2.5-flash-lite"
