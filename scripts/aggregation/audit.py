"""Aggregate per-category audit log counts from the NotebookLM audit notes.

Parses every audit markdown file under Notes/Internal/Audit Corpus NotebookLM
(grouped by Pusat, Daerah, Kementerian Lembaga), counts the LAYAK and TIDAK
LAYAK rows in each Markdown table, and writes the result to
data/audit_aggregate.json plus a printed summary table.

Each audit file may contain multiple tables (one per BPK pagination batch).
All tables in a file are summed into a single per-category total.

Usage:
    python scripts/aggregation/audit.py
    python scripts/aggregation/audit.py --audit-dir "../Notes/Internal/Audit Corpus NotebookLM"
    python scripts/aggregation/audit.py --json-only
"""
import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DEFAULT_AUDIT_DIR = Path("../Notes/Internal/Audit Corpus NotebookLM")
DEFAULT_OUTPUT = Path("data/audit_aggregate.json")
SCOPE_FOLDERS = ("Pusat", "Daerah", "Kementerian Lembaga")


def normalize_layak(cell: str) -> str | None:
    """Map a Layak? cell value to one of LAYAK, TIDAK_LAYAK, or None.

    Accepts the variations seen across audit files, including bolded
    cells, Ya, Tidak, LAYAK, TIDAK LAYAK, and minor whitespace. Returns
    None for cells that cannot be classified, so the caller can flag
    them as unknown rows for manual inspection.
    """
    s = re.sub(r"[*_`]", "", cell or "").strip().lower()
    if not s:
        return None
    if s.startswith("tidak"):
        return "TIDAK_LAYAK"
    if s.startswith("layak") or s.startswith("ya"):
        return "LAYAK"
    return None


def parse_audit_file(path: Path) -> dict:
    """Count LAYAK and TIDAK LAYAK rows across every table in one audit file.

    A row is a Markdown table line whose Layak column resolves via
    normalize_layak. Multiple tables in the same file are summed. Rows
    with an unrecognised Layak cell are returned in the unknown list so
    they can be inspected manually.
    """
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    layak = 0
    tidak = 0
    unknown: list[str] = []
    in_table = False
    layak_idx: int | None = None

    for raw in lines:
        line = raw.rstrip()
        if not line.startswith("|"):
            in_table = False
            layak_idx = None
            continue

        cells = [c.strip() for c in line.strip().strip("|").split("|")]

        if not in_table:
            # First row of a new table must be the header.
            lowered = [c.lower() for c in cells]
            if any("layak" in c for c in lowered):
                layak_idx = next(i for i, c in enumerate(lowered) if "layak" in c)
                in_table = True
            continue

        # Skip the |---|---| separator row.
        if all(re.fullmatch(r":?-+:?", c) for c in cells if c):
            continue

        if layak_idx is None or layak_idx >= len(cells):
            continue

        verdict = normalize_layak(cells[layak_idx])
        if verdict == "LAYAK":
            layak += 1
        elif verdict == "TIDAK_LAYAK":
            tidak += 1
        else:
            unknown.append(line)

    scanned = layak + tidak + len(unknown)
    return {
        "scanned": scanned,
        "layak": layak,
        "tidak_layak": tidak,
        "unknown_rows": unknown,
    }


def collect_audit_dir(audit_dir: Path) -> "OrderedDict[str, dict]":
    """Walk all audit files under audit_dir and aggregate per category."""
    results: OrderedDict[str, dict] = OrderedDict()
    for scope in SCOPE_FOLDERS:
        scope_dir = audit_dir / scope
        if not scope_dir.exists():
            continue
        for md_path in sorted(scope_dir.glob("*.md")):
            if md_path.stem.startswith("_"):
                continue
            entry = parse_audit_file(md_path)
            entry["scope"] = scope
            entry["path"] = str(md_path.relative_to(audit_dir.parent.parent))
            results[md_path.stem] = entry
    return results


def print_table(results: "OrderedDict[str, dict]") -> None:
    """Render the aggregated counts as a fixed-width table on stdout."""
    print(f"{'category':<22} {'scope':<22} {'scanned':>8} {'LAYAK':>6} {'TIDAK':>6} {'unknown':>8}")
    print("-" * 76)
    total_scanned = total_layak = total_tidak = total_unknown = 0
    for cat, entry in results.items():
        unknown_n = len(entry["unknown_rows"])
        print(
            f"{cat:<22} {entry['scope']:<22} "
            f"{entry['scanned']:>8} {entry['layak']:>6} {entry['tidak_layak']:>6} {unknown_n:>8}"
        )
        total_scanned += entry["scanned"]
        total_layak += entry["layak"]
        total_tidak += entry["tidak_layak"]
        total_unknown += unknown_n
    print("-" * 76)
    print(
        f"{'TOTAL':<22} {'':<22} "
        f"{total_scanned:>8} {total_layak:>6} {total_tidak:>6} {total_unknown:>8}"
    )


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--audit-dir",
        type=Path,
        default=DEFAULT_AUDIT_DIR,
        help=f"Path to audit directory (default {DEFAULT_AUDIT_DIR}).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Path to write aggregated JSON (default {DEFAULT_OUTPUT}).",
    )
    ap.add_argument(
        "--json-only",
        action="store_true",
        help="Skip the printed table and only write the JSON output.",
    )
    args = ap.parse_args()

    if not args.audit_dir.exists():
        raise SystemExit(f"audit dir not found, {args.audit_dir}")

    results = collect_audit_dir(args.audit_dir)
    if not results:
        raise SystemExit(f"no audit files found under {args.audit_dir}")

    if not args.json_only:
        print_table(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
