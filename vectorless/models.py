"""Single source of truth for model selection per role.

Each role pins one model. Switch a role here and every call site picks it up.
The vectorless.llm dispatcher routes by model prefix:
  gpt-*       -> OpenAI
  claude-*    -> Anthropic
  gemini-*    -> Vertex AI

Roles.
  PARSE_MODEL      Structure extraction from PDF text. Heavy long-context.
  SUMMARY_MODEL    Per-node summary. High volume, light per-call.
  OCR_CLEAN_MODEL  Targeted OCR repair on leaf text.
  JUDGE_MODEL      Post-index quality verdict. Cross-family from parser.
  RETRIEVAL_MODEL  LLM nav and hybrid rerank at query time.
"""

PARSE_MODEL = "gpt-5"
SUMMARY_MODEL = "gemini-2.5-flash-lite"
OCR_CLEAN_MODEL = "gemini-2.5-flash-lite"
JUDGE_MODEL = "gemini-2.5-pro"
RETRIEVAL_MODEL = "gemini-2.5-flash-lite"
