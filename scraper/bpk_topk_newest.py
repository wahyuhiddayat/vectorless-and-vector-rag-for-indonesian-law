"""
Generate a Markdown report of the newest BPK JDIH documents per category.
"""

import argparse
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from common import BASE_URL, DEFAULT_DELAY, HEADERS, JENIS_MAP, KATEGORI_MAP, fetch

log = logging.getLogger("bpk_topk_newest")
DEFAULT_OUTPUT = Path("data/reports/bpk_topk_newest.md")

GROUP_PAGES = [
    (1, "Pusat"),
    (2, "Kementerian/Lembaga"),
    (3, "Daerah"),
]


def parse_jenis_page(html: str, group_name: str) -> list[dict]:
    """Parse a /Jenis/{group} page and return category rows in page order."""
    soup = BeautifulSoup(html, "html.parser")
    categories = []

    for row in soup.select("div.my-6.d-flex.justify-content-between.border-bottom"):
        name_link = row.select_one("a.fw-bold[href*='/Search?jenis=']")
        count_link = row.select_one("a.btn.btn-primary.btn-sm[href*='/Search?jenis=']")
        if not name_link:
            continue

        href = name_link.get("href", "")
        match = re.search(r"[?&]jenis=(\d+)", href)
        if not match:
            continue

        total_count = None
        if count_link:
            count_text = count_link.get_text(strip=True).replace(".", "").replace(",", "")
            if count_text.isdigit():
                total_count = int(count_text)

        categories.append(
            {
                "group": group_name,
                "jenis_id": int(match.group(1)),
                "name": name_link.get_text(" ", strip=True),
                "total_count": total_count,
            }
        )

    return categories


def parse_total_count_from_search_html(html: str) -> int | None:
    """Parse the total document count from search page summary text."""
    match = re.search(r"Menemukan\s+([\d.,]+)\s+peraturan", html)
    if not match:
        return None
    return int(match.group(1).replace(".", "").replace(",", ""))


def parse_total_pages(soup: BeautifulSoup) -> int:
    """Parse the last available search result page number from pagination."""
    pagination = soup.select_one("ul.pagination")
    if not pagination:
        return 1

    for link in pagination.select("a.page-link"):
        if link.get_text(strip=True) != "Last":
            continue
        href = link.get("href", "")
        match = re.search(r"[?&]p=(\d+)", href)
        if match:
            return int(match.group(1))

    max_page = 1
    for link in pagination.select("a.page-link"):
        text = link.get_text(strip=True)
        if text.isdigit():
            max_page = max(max_page, int(text))
    return max_page


def parse_search_results_page(html: str) -> list[dict]:
    """Parse one BPK search results page in on-page order."""
    soup = BeautifulSoup(html, "html.parser")
    items = []

    for card_body in soup.select("div.card-body.p-xl-10.p-8.d-flex"):
        nomor_div = card_body.select_one("div.col-lg-8.fw-semibold.fs-5.text-gray-600")
        title_link = card_body.select_one("div.col-lg-10.fs-2.fw-bold a[href^='/Details/']")
        if not title_link:
            continue

        href = title_link.get("href", "")
        parts = href.strip("/").split("/")
        if len(parts) < 3:
            continue

        items.append(
            {
                "nomor_tahun_text": nomor_div.get_text(" ", strip=True) if nomor_div else "",
                "judul": title_link.get_text(" ", strip=True),
                "detail_id": parts[1],
                "slug": parts[2],
                "detail_url": urljoin(BASE_URL, href),
            }
        )

    return items


def fetch_group_categories(group_id: int, group_name: str, session: requests.Session) -> list[dict] | None:
    """Fetch and parse categories for one top-level BPK group page."""
    url = f"{BASE_URL}/Jenis/{group_id}"
    response = fetch(url, session)
    if response is None:
        log.error("Failed to fetch group page for %s: %s", group_name, url)
        return None
    return parse_jenis_page(response.text, group_name)


def discover_categories(session: requests.Session) -> list[dict]:
    """Discover all categories from the three BPK /Jenis group pages."""
    categories = []

    for group_id, group_name in GROUP_PAGES:
        parsed = fetch_group_categories(group_id, group_name, session)
        if parsed is None:
            continue
        categories.extend(parsed)

    return categories


def filter_categories(categories: list[dict], group: str | None, jenis_ids: list[int] | None) -> list[dict]:
    """Filter discovered categories by group or explicit jenis IDs."""
    if jenis_ids:
        wanted = set(jenis_ids)
        filtered = [category for category in categories if category["jenis_id"] in wanted]
        ordered = []
        seen = set()
        for jenis_id in jenis_ids:
            for category in filtered:
                if category["jenis_id"] == jenis_id and jenis_id not in seen:
                    ordered.append(category)
                    seen.add(jenis_id)
                    break
        return ordered

    if group:
        return [category for category in categories if category["group"] == group]

    return categories


def make_fallback_categories(jenis_ids: list[int]) -> list[dict]:
    """Build best-effort category metadata when explicit IDs were not discovered."""
    categories = []
    for jenis_id in jenis_ids:
        categories.append(
            {
                "group": KATEGORI_MAP.get(jenis_id, "Lainnya"),
                "jenis_id": jenis_id,
                "name": JENIS_MAP.get(jenis_id, f"jenis-{jenis_id}"),
                "total_count": None,
            }
        )
    return categories


def fetch_search_page(jenis_id: int, page: int, session: requests.Session) -> tuple[list[dict], int | None, int]:
    """Fetch one search results page and return items, total count, and total pages."""
    url = f"{BASE_URL}/Search?jenis={jenis_id}&p={page}"
    response = fetch(url, session)
    if response is None:
        raise RuntimeError(f"request failed for {url}")

    soup = BeautifulSoup(response.text, "html.parser")
    items = parse_search_results_page(response.text)
    total_count = parse_total_count_from_search_html(response.text)
    total_pages = parse_total_pages(soup)
    return items, total_count, total_pages


def collect_top_k_newest(category: dict, k: int, session: requests.Session, delay: float) -> dict:
    """Collect the top-k newest documents for one category, paging as needed."""
    report = {
        "group": category["group"],
        "jenis_id": category["jenis_id"],
        "name": category["name"],
        "total_count": category.get("total_count"),
        "items": [],
        "error": None,
    }

    try:
        page = 1
        total_pages = 1
        while len(report["items"]) < k and page <= total_pages:
            items, fallback_total, total_pages = fetch_search_page(category["jenis_id"], page, session)
            if report["total_count"] is None and fallback_total is not None:
                report["total_count"] = fallback_total
            if not items:
                break
            remaining = k - len(report["items"])
            report["items"].extend(items[:remaining])
            page += 1
            if len(report["items"]) < k and page <= total_pages:
                time.sleep(delay)
    except RuntimeError as exc:
        report["error"] = str(exc)
        log.error("Failed to collect newest items for %s (%s): %s", category["name"], category["jenis_id"], exc)

    if report["total_count"] is None:
        report["total_count"] = 0

    return report


def format_category_heading(name: str, total_count: int) -> str:
    """Format a category heading with its total document count."""
    return f"### {name} - {total_count:,} dokumen"


def format_item_line(index: int, item: dict) -> str:
    """Format one Markdown list item for a newest-document entry."""
    link_text = item.get("nomor_tahun_text") or item.get("judul") or item.get("detail_url")
    title = item.get("judul", "").strip()
    if title and title != link_text:
        return f"{index}. [{link_text}]({item['detail_url']}) - {title}"
    return f"{index}. [{link_text}]({item['detail_url']})"


def render_markdown(reports: list[dict]) -> str:
    """Render grouped category reports into Markdown."""
    lines = ["# BPK Top-K Newest per Category", ""]

    grouped: dict[str, list[dict]] = {}
    for report in reports:
        grouped.setdefault(report["group"], []).append(report)

    for _, group_name in GROUP_PAGES:
        group_reports = grouped.get(group_name, [])
        if not group_reports:
            continue

        lines.append(f"## {group_name}")
        lines.append("")
        for report in group_reports:
            lines.append(format_category_heading(report["name"], report["total_count"]))
            if report["error"]:
                lines.append("")
                lines.append("_Failed to fetch newest items for this category._")
                lines.append("")
                continue

            if report["items"]:
                lines.append("")
                for index, item in enumerate(report["items"], start=1):
                    lines.append(format_item_line(index, item))
                lines.append("")
            else:
                lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_reports(categories: list[dict], k: int, session: requests.Session, delay: float) -> list[dict]:
    """Build category reports in the same order as the discovered categories."""
    reports = []
    for index, category in enumerate(categories):
        reports.append(collect_top_k_newest(category, k=k, session=session, delay=delay))
        if index < len(categories) - 1:
            time.sleep(delay)
    return reports


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the newest-documents report script."""
    parser = argparse.ArgumentParser(
        description="Generate a Markdown report of top-k newest BPK JDIH documents per category."
    )
    parser.add_argument("--k", type=int, default=3, help="Newest documents to list per category (default: 3)")
    parser.add_argument(
        "--group",
        choices=[group_name for _, group_name in GROUP_PAGES],
        help="Only include categories from one broad group",
    )
    parser.add_argument("--jenis", type=int, nargs="+", help="Explicit jenis IDs to include")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Markdown output file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()
    if args.k <= 0:
        parser.error("--k must be a positive integer")
    return args


def main() -> None:
    """Run the CLI entry point."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    session = requests.Session()
    session.headers.update(HEADERS)

    categories = discover_categories(session)
    filtered = filter_categories(categories, group=args.group, jenis_ids=args.jenis)

    if args.jenis:
        found_ids = {category["jenis_id"] for category in filtered}
        missing_ids = [jenis_id for jenis_id in args.jenis if jenis_id not in found_ids]
        if missing_ids:
            log.warning("Using fallback metadata for undiscovered jenis IDs: %s", ", ".join(map(str, missing_ids)))
            filtered.extend(make_fallback_categories(missing_ids))

    if not filtered:
        raise SystemExit("No categories matched the requested filters.")

    reports = build_reports(filtered, k=args.k, session=session, delay=args.delay)
    markdown = render_markdown(reports)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    log.info("Markdown report written to %s", output_path)


if __name__ == "__main__":
    main()
