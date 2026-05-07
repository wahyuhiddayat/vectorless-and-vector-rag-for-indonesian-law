"""CLI: audit corpus state and optionally reconcile ineligible artifacts.

Default mode scans raw + registry + indexes + judge verdicts and writes
`data/corpus_status.json` plus a per-category summary table. With
`--reconcile`, ineligible doc artifacts (verdict in {MAJOR, FAIL, ERROR})
are deleted across raw, registry, all 3 index granularities, and judge
report so the artifact stores stay in lockstep.

Usage:
    python scripts/parser/corpus_status.py                # audit only
    python scripts/parser/corpus_status.py --reconcile    # drop INELIGIBLE
    python scripts/parser/corpus_status.py --reconcile --dry-run
    python scripts/parser/corpus_status.py --json         # print full JSON
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.corpus_status import (  # noqa: E402
    ROOT,
    STATUS_PATH,
    build_status,
    reconcile,
    write_status,
)


def _print_summary(status: dict, actions: dict | None = None) -> None:
    print(f"corpus snapshot at {status['generated_at']}")
    print()
    cols = ["total", "raw", "registry", "indexed", "judged", "eligible_gt",
            "ok", "minor", "major", "fail", "error"]
    header = f"{'category':20}" + "".join(f"{c:>11}" for c in cols)
    print(header)
    print("-" * len(header))
    for cat in sorted(status["by_category"]):
        c = status["by_category"][cat]
        row = f"{cat:20}" + "".join(f"{c.get(k,0):>11}" for k in cols)
        print(row)
    print()
    eligible = sum(1 for d in status["docs"] if d["eligible_for_gt"])
    not_eligible = sum(1 for d in status["docs"] if not d["eligible_for_gt"])
    not_indexed = sum(1 for d in status["docs"] if d["in_registry"] and not d["fully_indexed"])
    print(f"total docs known        : {len(status['docs'])}")
    print(f"GT-eligible             : {eligible}")
    print(f"INELIGIBLE (drop target): {not_eligible}")
    print(f"in registry but missing index: {not_indexed}")
    if actions is not None:
        print()
        print(f"reconcile dropped {len(actions['docs_dropped'])} docs:")
        for a in actions["docs_dropped"]:
            print(f"  - {a['doc_id']:30} {a['reason']}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconcile", action="store_true",
                    help="Drop ineligible doc artifacts (raw, registry, index, judge).")
    ap.add_argument("--dry-run", action="store_true",
                    help="With --reconcile, show what would be removed without touching files.")
    ap.add_argument("--json", action="store_true",
                    help="Print full corpus_status.json to stdout instead of summary.")
    args = ap.parse_args()

    status = build_status()
    actions = None
    if args.reconcile:
        actions = reconcile(status, dry_run=args.dry_run)
        if not args.dry_run:
            status = build_status()

    write_status(status)

    if args.json:
        print(json.dumps(status, indent=2, ensure_ascii=False))
    else:
        _print_summary(status, actions)
        print(f"\nwrote {STATUS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
