"""
BPK JDIH Survey — Count documents per category without scraping.

Fetches only page 1 of each (jenis, tahun) combination to read the total count
from the "Menemukan X peraturan" text. No detail pages scraped, no PDFs downloaded.

Usage:
    python bpk_survey.py                            # all categories, total only
    python bpk_survey.py --tahun 2020-2026          # per-year breakdown
    python bpk_survey.py --group Pusat              # only Pusat categories
    python bpk_survey.py --jenis 8 9 10             # specific jenis IDs
    python bpk_survey.py --tahun 2020-2026 --output # save to survey_results.json
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime
from pathlib import Path

import requests

from common import (
    BASE_URL, HEADERS, JENIS_MAP, KATEGORI_MAP,
    DEFAULT_DELAY, fetch,
)

log = logging.getLogger("bpk_survey")


def get_doc_count(jenis_id: int, session: requests.Session, tahun: int | None = None) -> int:
    """Fetch page 1 and parse 'Menemukan X peraturan' to get total count."""
    url = f"{BASE_URL}/Search?jenis={jenis_id}&p=1"
    if tahun:
        url += f"&tahun={tahun}"
    resp = fetch(url, session)
    if resp is None:
        return -1
    # Parse: "Menemukan 62.038 peraturan dalam 0,014 detik"
    # Indonesian uses dots as thousands separator
    match = re.search(r"Menemukan\s+([\d.]+)\s+peraturan", resp.text)
    if not match:
        return 0
    count_str = match.group(1).replace(".", "")
    return int(count_str)


def parse_tahun_range(tahun_str: str) -> list[int]:
    """Parse '2020-2026' → [2020, 2021, ..., 2026] or '2024' → [2024]."""
    if "-" in tahun_str:
        start, end = tahun_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(tahun_str)]


def main():
    parser = argparse.ArgumentParser(description="Survey BPK JDIH document counts per category")
    parser.add_argument("--jenis", type=int, nargs="+", help="Specific jenis IDs to survey")
    parser.add_argument("--group", choices=["Pusat", "Daerah", "Kementerian/Lembaga"],
                        help="Filter by category group")
    parser.add_argument("--tahun", help='Year or range, e.g. "2024" or "2020-2026"')
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (default: 1.0s)")
    parser.add_argument("--output", action="store_true", help="Save results to data/survey_results.json")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # Determine which jenis to survey
    if args.jenis:
        jenis_ids = args.jenis
    elif args.group:
        jenis_ids = [jid for jid, grp in KATEGORI_MAP.items() if grp == args.group]
    else:
        jenis_ids = sorted(JENIS_MAP.keys())

    # Parse year range
    years = parse_tahun_range(args.tahun) if args.tahun else []

    session = requests.Session()
    session.headers.update(HEADERS)

    results = []
    total_requests = len(jenis_ids) * (1 + len(years))  # total + per-year
    done = 0

    print(f"\nSurveying {len(jenis_ids)} categories" +
          (f" × {len(years)} years ({years[0]}-{years[-1]})" if years else " (total only)"))
    print(f"Estimated: {total_requests} requests @ {args.delay}s = ~{total_requests * args.delay / 60:.0f} min\n")

    for jenis_id in sorted(jenis_ids):
        name = JENIS_MAP.get(jenis_id, f"jenis-{jenis_id}")
        group = KATEGORI_MAP.get(jenis_id, "Lainnya")

        # Get total (no year filter)
        total = get_doc_count(jenis_id, session)
        done += 1
        time.sleep(args.delay)

        by_year = {}
        if years:
            for year in years:
                count = get_doc_count(jenis_id, session, tahun=year)
                by_year[str(year)] = count
                done += 1
                time.sleep(args.delay)

            # Progress
            pct = done / total_requests * 100
            year_sum = sum(v for v in by_year.values() if v >= 0)
            print(f"[{pct:5.1f}%] {group:24s} | {jenis_id:>3d} | {name:28s} | Total: {total:>7,d} | "
                  + " | ".join(f"{y}: {by_year[str(y)]:>5,d}" for y in years)
                  + f" | Sum: {year_sum:>7,d}")
        else:
            pct = done / total_requests * 100
            print(f"[{pct:5.1f}%] {group:24s} | {jenis_id:>3d} | {name:28s} | Total: {total:>7,d}")

        results.append({
            "jenis_id": jenis_id,
            "name": name,
            "group": group,
            "total": total,
            "by_year": by_year,
        })

    # Summary
    print(f"\n{'=' * 80}")
    grand_total = sum(r["total"] for r in results if r["total"] >= 0)
    print(f"Grand total: {grand_total:,} documents across {len(results)} categories")

    if years:
        year_sum = sum(
            sum(v for v in r["by_year"].values() if v >= 0)
            for r in results
        )
        print(f"Year range {years[0]}-{years[-1]}: {year_sum:,} documents")

    # Group summary
    print(f"\nBy group:")
    for group_name in ["Pusat", "Daerah", "Kementerian/Lembaga"]:
        group_results = [r for r in results if r["group"] == group_name]
        group_total = sum(r["total"] for r in group_results if r["total"] >= 0)
        if years:
            group_year_sum = sum(
                sum(v for v in r["by_year"].values() if v >= 0)
                for r in group_results
            )
            print(f"  {group_name}: {len(group_results)} categories, "
                  f"{group_total:,} total, {group_year_sum:,} in {years[0]}-{years[-1]}")
        else:
            print(f"  {group_name}: {len(group_results)} categories, {group_total:,} total")

    # Save to JSON
    if args.output:
        output_path = Path("data/survey_results.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "surveyed_at": datetime.now().isoformat(),
            "year_range": years if years else None,
            "categories": results,
        }
        output_path.write_text(json.dumps(output_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
