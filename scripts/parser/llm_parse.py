"""CLI: parse documents into the canonical pasal-level index.

Resolves doc targets from --doc-id or --category, runs `parse_doc` for
each target, appends the audit record, and prints a one-line status per
doc. The actual parsing (LLM dispatch, chunking, structure normalization,
write-to-disk) lives in vectorless.indexing.llm_parse.

Usage:
    python scripts/parser/llm_parse.py --doc-id uu-3-2025
    python scripts/parser/llm_parse.py --category UU --dry-run
"""
import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.llm_parse import _append_audit, parse_doc  # noqa: E402
from vectorless.indexing.targets import resolve_targets  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[],
                    help="Doc to parse (repeatable)")
    ap.add_argument("--category",
                    help="Parse every doc in this jenis_folder (e.g. UU, OJK)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only, do not overwrite index")
    args = ap.parse_args()

    targets = resolve_targets(list(args.doc_ids), args.category)
    print(f"Targets: {len(targets)} docs")
    if not targets:
        return

    for i, doc_id in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {doc_id}")
        try:
            audit = parse_doc(doc_id, dry_run=args.dry_run)
        except Exception as exc:
            audit = {
                "doc_id": doc_id,
                "status": "error",
                "error": f"unhandled: {exc}",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        _append_audit(audit)
        msg = f"  status: {audit.get('status')}"
        if "pasal_count" in audit:
            msg += f"  pasals: {audit['pasal_count']} (pdf regex: {audit.get('pdf_pasal_regex_count', '?')})"
        if audit.get("validation_errors"):
            msg += f"  errors: {audit['validation_errors'][:2]}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
