"""CLI wrapper for the corpus-status auditor.

The implementation lives at vectorless/indexing/corpus_status.py. Run via
either entry point:

    python scripts/parser/corpus_status.py [--reconcile] [--dry-run] [--json]
    python -m vectorless.indexing.corpus_status [--reconcile] [--dry-run] [--json]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.corpus_status import main  # noqa: E402

if __name__ == "__main__":
    main()
