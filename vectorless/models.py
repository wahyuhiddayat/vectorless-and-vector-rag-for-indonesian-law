"""Single source of truth for OpenAI model selection per role.

Each role pins one model. Switch a role here and every call site picks it up.

Roles.
  PARSE_MODEL      Structure extraction from PDF text. Heavy long-context.
  SUMMARY_MODEL    Per-node summary. High volume, light per-call.
  OCR_CLEAN_MODEL  Targeted OCR repair on leaf text.
  JUDGE_MODEL      Post-index quality verdict. Stronger than parser.
  RETRIEVAL_MODEL  LLM nav and hybrid rerank at query time.
"""

PARSE_MODEL = "gpt-5"
SUMMARY_MODEL = "gpt-4.1-nano"
OCR_CLEAN_MODEL = "gpt-4.1-nano"
JUDGE_MODEL = "gpt-5"
RETRIEVAL_MODEL = "gpt-4.1-nano"
