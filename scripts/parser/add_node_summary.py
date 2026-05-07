"""CLI wrapper for the per-node summary annotator.

Implementation lives at vectorless/indexing/summary.py. Run via either:

    python scripts/parser/add_node_summary.py --doc-id uu-3-2025
    python -m vectorless.indexing.summary --doc-id uu-3-2025
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.summary import main  # noqa: E402

if __name__ == "__main__":
    main()
