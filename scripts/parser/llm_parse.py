"""CLI wrapper for the LLM-first structure parser.

Implementation lives at vectorless/indexing/llm_parse.py. Run via either:

    python scripts/parser/llm_parse.py --doc-id uu-3-2025
    python -m vectorless.indexing.llm_parse --doc-id uu-3-2025
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.llm_parse import main  # noqa: E402

if __name__ == "__main__":
    main()
