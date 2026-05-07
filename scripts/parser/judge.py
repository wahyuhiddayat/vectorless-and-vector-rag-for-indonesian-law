"""CLI: run the Gemini judge on parsed indexes and refresh corpus status.

For each target doc, calls `judge_doc` and incrementally appends the
report to `data/judge_report.json`. Final pass refreshes
`data/corpus_status.json` so eligibility flags reflect the new verdicts.

Usage:
    python scripts/parser/judge.py --doc-id uu-3-2025
    python scripts/parser/judge.py --category UU
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.judge import REPORT_PATH, _save_reports, judge_doc  # noqa: E402
from vectorless.indexing.targets import resolve_targets  # noqa: E402
from vectorless.models import JUDGE_MODEL  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini-based judge for parsed index quality")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[])
    ap.add_argument("--category", help="Judge every doc in this jenis_folder")
    args = ap.parse_args()

    targets = resolve_targets(list(args.doc_ids), args.category)
    print(f"judging {len(targets)} docs with {JUDGE_MODEL}")

    reports: list[dict] = []
    for i, did in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {did} ...", flush=True)
        try:
            report = judge_doc(did)
        except Exception as exc:
            print(f"  error: {exc}", flush=True)
            report = {"doc_id": did, "verdict": "ERROR", "error": str(exc)}
        cov = report.get("coverage") or {}
        usage = report.get("usage") or {}
        print(
            f"  {report.get('verdict', '?')}  score={report.get('overall_score')}  "
            f"miss={len(cov.get('missing') or [])} "
            f"struct={len(report.get('structural_issues') or [])} "
            f"ocr={len(report.get('ocr_issues') or [])}  "
            f"tokens={usage.get('input_tokens') or 0:,}/{usage.get('output_tokens') or 0:,}  "
            f"${usage.get('cost_usd') or 0:.4f}",
            flush=True,
        )
        reports.append(report)
        _save_reports(reports)

    total_in = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in reports)
    total_out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in reports)
    total_cost = sum((r.get("usage") or {}).get("cost_usd") or 0 for r in reports)
    print(f"\ntotal tokens: {total_in:,} in / {total_out:,} out  cost ${total_cost:.4f}")
    print(f"report: {REPORT_PATH}")

    try:
        from vectorless.indexing.corpus_status import build_status, write_status
        write_status(build_status())
        print("corpus_status.json refreshed")
    except Exception as exc:
        print(f"warning: corpus_status refresh failed: {exc}")


if __name__ == "__main__":
    main()
