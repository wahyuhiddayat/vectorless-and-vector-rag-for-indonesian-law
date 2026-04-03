"""
Pretty-print a JSON file as UTF-8 without BOM.

Useful for raw GT files that come back as a flat one-line JSON array from ChatGPT.

Examples:
    python scripts/pretty_json.py data/ground_truth_raw/permenaker-1-2026.json
    python scripts/pretty_json.py data/ground_truth_raw/permenaker-1-2026.json --indent 4
"""

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Pretty-print JSON as UTF-8 without BOM")
    ap.add_argument("path", type=str, help="Path to JSON file")
    ap.add_argument("--indent", type=int, default=2, help="Indent size (default: 2)")
    args = ap.parse_args()

    path = Path(args.path)
    with open(path, encoding="utf-8-sig") as f:
        data = json.load(f)

    pretty = json.dumps(data, ensure_ascii=False, indent=args.indent) + "\n"
    path.write_text(pretty, encoding="utf-8")
    print(f"Pretty JSON written to: {path}")


if __name__ == "__main__":
    main()
