"""CLI wrapper for the parser-quality judge.

Implementation lives at vectorless/indexing/judge.py. Run via either:

    python scripts/parser/judge.py --doc-id uu-3-2025
    python -m vectorless.indexing.judge --doc-id uu-3-2025
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.judge import main  # noqa: E402

if __name__ == "__main__":
    main()
