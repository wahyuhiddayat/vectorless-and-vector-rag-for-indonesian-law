"""CLI wrapper for the OCR-clean pass.

Implementation lives at vectorless/indexing/ocr.py. Run via either:

    python scripts/parser/clean_ocr.py --doc-id uu-3-2025
    python -m vectorless.indexing.ocr --doc-id uu-3-2025
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.ocr import main  # noqa: E402

if __name__ == "__main__":
    main()
