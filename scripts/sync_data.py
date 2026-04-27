"""True mirror sync between local data/ and the HF backup repo.

`hf download` and `hf upload` are additive. Neither propagates deletions,
so files removed on one machine reappear on another after a roundtrip.
This script enforces real mirror semantics in both directions by listing
both sides, computing the diff, and applying creates plus deletes.

Modes.
  --pull   Make local match HF. Delete local files absent on HF, then
           download. Use after another machine pushed.
  --push   Make HF match local. Delete HF files absent locally, then
           upload. Use after editing local data/.

Usage:
    python scripts/sync_data.py --pull
    python scripts/sync_data.py --push
    python scripts/sync_data.py --pull --dry-run

Repo: wahyyuht/skripsi-data (dataset). Override with --repo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_REPO = "wahyyuht/skripsi-data"
DEFAULT_LOCAL = Path("data")


def list_local(local_dir: Path) -> set[str]:
    """Return relative POSIX paths of every file under local_dir."""
    if not local_dir.exists():
        return set()
    return {
        p.relative_to(local_dir).as_posix()
        for p in local_dir.rglob("*")
        if p.is_file()
    }


def list_remote(api: HfApi, repo: str) -> set[str]:
    """Return relative paths of every file in the HF dataset repo."""
    return set(api.list_repo_files(repo_id=repo, repo_type="dataset"))


def pull(api: HfApi, repo: str, local_dir: Path, dry_run: bool) -> None:
    """Make local mirror remote. Delete local extras, then snapshot pull."""
    local = list_local(local_dir)
    remote = list_remote(api, repo)
    extra_local = sorted(local - remote)
    missing_local = sorted(remote - local)

    print(f"local files:  {len(local)}")
    print(f"remote files: {len(remote)}")
    print(f"to delete locally: {len(extra_local)}")
    print(f"to download:       {len(missing_local)}")

    if dry_run:
        if extra_local:
            print("\n[dry-run] would delete local files:")
            for p in extra_local[:20]:
                print(f"  - {p}")
            if len(extra_local) > 20:
                print(f"  ... and {len(extra_local) - 20} more")
        return

    for rel in extra_local:
        (local_dir / rel).unlink(missing_ok=True)
    print(f"deleted {len(extra_local)} local files")

    # Prune empty directories left behind.
    for d in sorted(
        (p for p in local_dir.rglob("*") if p.is_dir()),
        key=lambda x: len(x.parts),
        reverse=True,
    ):
        try:
            d.rmdir()
        except OSError:
            pass

    if missing_local or extra_local:
        print("downloading from HF...")
        api.snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            local_dir=str(local_dir),
        )
        print("download done")


def push(api: HfApi, repo: str, local_dir: Path, dry_run: bool) -> None:
    """Make remote mirror local. Delete remote extras, then upload."""
    local = list_local(local_dir)
    remote = list_remote(api, repo)
    # Skip git internals on the remote side.
    remote_real = {r for r in remote if not r.startswith(".git")}
    extra_remote = sorted(remote_real - local)
    missing_remote = sorted(local - remote_real)

    print(f"local files:  {len(local)}")
    print(f"remote files: {len(remote_real)}")
    print(f"to delete on HF: {len(extra_remote)}")
    print(f"to upload:       {len(missing_remote)}")

    if dry_run:
        if extra_remote:
            print("\n[dry-run] would delete on HF:")
            for p in extra_remote[:20]:
                print(f"  - {p}")
            if len(extra_remote) > 20:
                print(f"  ... and {len(extra_remote) - 20} more")
        return

    if extra_remote:
        batch = 100
        for i in range(0, len(extra_remote), batch):
            chunk = extra_remote[i:i + batch]
            api.delete_files(
                repo_id=repo,
                repo_type="dataset",
                delete_patterns=chunk,
                commit_message=f"sync: remove {len(chunk)} stale files",
            )
            print(f"  deleted batch {i // batch + 1}: {len(chunk)} files")

    if missing_remote or not extra_remote:
        # huggingface_hub upload_folder behaves like a diff push, so it is
        # safe to run unconditionally to cover modified-but-existing files.
        print("uploading to HF...")
        api.upload_folder(
            repo_id=repo,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=".",
            commit_message="sync: push local data/",
        )
        print("upload done")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument("--pull", action="store_true", help="Mirror HF down to local data/")
    direction.add_argument("--push", action="store_true", help="Mirror local data/ up to HF")
    parser.add_argument("--repo", default=DEFAULT_REPO, help=f"HF dataset repo (default: {DEFAULT_REPO})")
    parser.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL, help=f"Local data dir (default: {DEFAULT_LOCAL})")
    parser.add_argument("--dry-run", action="store_true", help="Show what would change without applying")
    args = parser.parse_args()

    api = HfApi()
    if args.pull:
        pull(api, args.repo, args.local_dir, args.dry_run)
    else:
        push(api, args.repo, args.local_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
