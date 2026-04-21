"""Restore index files from backup dirs.

Use cases:
    - Revert llm_fix-touched docs back to parser+cleanup state before running
      llm_parse (clean baseline, no confusion between old/new pipelines).
    - Revert llm_parse-touched docs back to llm_fix state (unlikely).

Usage:
    python scripts/parser/restore_backup.py --from llm_fix         # restore all
    python scripts/parser/restore_backup.py --from llm_fix --doc-id peraturan-ojk-23-2025
    python scripts/parser/restore_backup.py --from llm_parse --doc-id peraturan-ojk-17-2025
    python scripts/parser/restore_backup.py --from llm_fix --list  # list only
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"
BACKUP_DIRS = {
    "llm_fix": REPO_ROOT / "data" / "index_pasal_pre_llm_fix",
    "llm_parse": REPO_ROOT / "data" / "index_pasal_pre_llm_parse",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="source", choices=list(BACKUP_DIRS), required=True,
                    help="Backup dir to restore from")
    ap.add_argument("--doc-id", help="Restore only this doc_id")
    ap.add_argument("--list", action="store_true", help="List backups without restoring")
    args = ap.parse_args()

    backup_dir = BACKUP_DIRS[args.source]
    if not backup_dir.exists():
        print(f"No backups found at {backup_dir}")
        return

    backups = list(backup_dir.glob("*/*.json"))
    if args.doc_id:
        backups = [p for p in backups if p.stem == args.doc_id]

    if not backups:
        print("No matching backups")
        return

    if args.list:
        print(f"Backups in {backup_dir}:")
        for b in backups:
            print(f"  {b.parent.name}/{b.name}")
        return

    restored = 0
    for backup in backups:
        target = INDEX_PASAL / backup.parent.name / backup.name
        if not target.parent.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup, target)
        print(f"restored: {backup.parent.name}/{backup.name}")
        restored += 1
    print(f"\nRestored {restored} docs from {args.source}")
    print("Remember to re-run ayat + full_split re-split for restored docs.")


if __name__ == "__main__":
    main()
