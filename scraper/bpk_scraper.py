"""
BPK JDIH Scraper — Legal Document Acquisition
Scrapes metadata (JSON) and PDF files from peraturan.bpk.go.id
for Indonesian regulatory documents (UU, Perpu, PP, Perpres).
"""

import argparse
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import unquote, urljoin

import requests
from bs4 import BeautifulSoup

from common import (
    BASE_URL, HEADERS, JENIS_MAP, KATEGORI_MAP,
    DEFAULT_DELAY, MAX_RETRIES, fetch,
)

log = logging.getLogger("bpk_scraper")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def sanitize_filename(name: str) -> str:
    """Clean a string for safe use as a filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def make_doc_id(bentuk_singkat: str, nomor: str, tahun: str) -> str:
    """Build canonical doc_id like 'uu-1-2026'."""
    bs = bentuk_singkat.strip().lower() if bentuk_singkat else "unknown"
    n = nomor.strip() if nomor else "0"
    t = tahun.strip() if tahun else "0"
    return f"{bs}-{n}-{t}"


# ---------------------------------------------------------------------------
# Pagination
# ---------------------------------------------------------------------------


def get_total_pages(jenis_id: int, session: requests.Session, tahun: int | None = None) -> int:
    """Discover the last page number from the pagination on page 1."""
    url = f"{BASE_URL}/Search?jenis={jenis_id}&p=1"
    if tahun:
        url += f"&tahun={tahun}"
    resp = fetch(url, session)
    if resp is None:
        return 0
    soup = BeautifulSoup(resp.text, "html.parser")
    pagination = soup.select_one("ul.pagination")
    if not pagination:
        return 1
    # The "Last" link is the last <a class="page-link"> whose text is "Last"
    last_link = None
    for a in pagination.find_all("a", class_="page-link"):
        if a.get_text(strip=True) == "Last":
            last_link = a
            break
    if last_link and last_link.get("href"):
        match = re.search(r"[?&]p=(\d+)", last_link["href"])
        if match:
            return int(match.group(1))
    # Fallback: find the highest numeric page link
    max_page = 1
    for a in pagination.find_all("a", class_="page-link"):
        txt = a.get_text(strip=True)
        if txt.isdigit():
            max_page = max(max_page, int(txt))
    return max_page


# ---------------------------------------------------------------------------
# List page scraping
# ---------------------------------------------------------------------------


def scrape_list_page(jenis_id: int, page: int, session: requests.Session, tahun: int | None = None) -> list[dict]:
    """Scrape a single search results page. Returns list of item dicts."""
    url = f"{BASE_URL}/Search?jenis={jenis_id}&p={page}"
    if tahun:
        url += f"&tahun={tahun}"
    resp = fetch(url, session)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    items = []

    for card_body in soup.select("div.card-body.p-xl-10"):
        item = {}

        # Nomor + Tahun text (e.g. "Undang-undang (UU) Nomor 1 Tahun 2026")
        nomor_div = card_body.select_one("div.col-lg-8.fw-semibold.fs-5.text-gray-600")
        item["nomor_tahun_text"] = nomor_div.get_text(strip=True) if nomor_div else ""

        # Judul link → extract detail_id and slug
        judul_link = card_body.select_one("div.fs-2.fw-bold a[href^='/Details/']")
        if not judul_link:
            continue
        href = judul_link["href"]  # e.g. /Details/337869/uu-no-1-tahun-2026
        parts = href.strip("/").split("/")  # ['Details', '337869', 'uu-no-1-tahun-2026']
        if len(parts) < 3:
            continue
        item["detail_id"] = parts[1]
        item["slug"] = parts[2]
        item["judul"] = judul_link.get_text(strip=True)

        items.append(item)

    return items


# ---------------------------------------------------------------------------
# Detail page scraping
# ---------------------------------------------------------------------------


def scrape_detail_page(detail_id: str, slug: str, session: requests.Session) -> dict | None:
    """Scrape a detail page and return full metadata dict."""
    url = f"{BASE_URL}/Details/{detail_id}/{slug}"
    resp = fetch(url, session)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    metadata: dict = {"detail_id": detail_id, "slug": slug, "url": url}

    # --- MATERI POKOK ---
    # h4 has child <span>, so string= won't match; use lambda on get_text()
    materi_header = soup.find(
        lambda tag: tag.name == "h4" and "MATERI POKOK" in tag.get_text()
    )
    if materi_header:
        card_body = materi_header.find_parent("div", class_="card-body")
        if card_body:
            # The materi pokok text is in <p> elements after the separator
            paragraphs = card_body.find_all("p")
            materi_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            metadata["materi_pokok"] = " ".join(materi_texts)

    # --- METADATA PERATURAN ---
    meta_header = soup.find(
        lambda tag: tag.name == "h4" and "METADATA" in tag.get_text()
    )
    if meta_header:
        container = meta_header.find_parent("div", class_="card-body")
        if container:
            meta_container = container.select_one("div.container.fs-6")
            if meta_container:
                rows = meta_container.select("div.py-4")
                for row in rows:
                    key_div = row.select_one("div.col-lg-3.fw-bold")
                    val_div = row.select_one("div.col-lg-9")
                    if key_div and val_div:
                        key = key_div.get_text(strip=True)
                        val = val_div.get_text(strip=True)
                        # Normalize key to snake_case
                        norm_key = key.lower().replace(".", "").replace(" ", "_")
                        metadata[norm_key] = val

    # --- STATUS PERATURAN ---
    status_header = soup.find(
        lambda tag: tag.name == "h4" and tag.get_text(strip=True).startswith("STATUS")
    )
    if status_header:
        status_card = status_header.find_parent("div", class_="card-body")
        if status_card:
            relasi = _parse_status_peraturan(status_card)
            if relasi:
                metadata["relasi"] = relasi

    # --- PDF FILE INFO ---
    pdf_links = soup.select("a.download-file[href^='/Download/']")
    pdf_files = []
    for a in pdf_links:
        href = a.get("href", "")
        data_id = a.get("data-id", "")
        if not data_id or not href:
            continue
        # href like /Download/400929/UU%20Nomor%201%20Tahun%202026.pdf
        filename_part = href.split("/")[-1] if "/" in href else ""
        filename = unquote(filename_part)
        pdf_files.append({"file_id": data_id, "filename": filename, "href": href})
    # Deduplicate by file_id
    seen_ids = set()
    unique_pdfs = []
    for pf in pdf_files:
        if pf["file_id"] not in seen_ids:
            seen_ids.add(pf["file_id"])
            unique_pdfs.append(pf)
    if unique_pdfs:
        metadata["pdf_files"] = unique_pdfs

    # --- Build doc_id ---
    bentuk_singkat = metadata.get("bentuk_singkat", "")
    nomor = metadata.get("nomor", "")
    tahun = metadata.get("tahun", "")
    metadata["doc_id"] = make_doc_id(bentuk_singkat, nomor, tahun)

    return metadata


def _parse_status_peraturan(card_body) -> list[dict]:
    """Parse the STATUS PERATURAN section into a list of relasi dicts."""
    relasi = []
    current_type = None

    container = card_body.select_one("div.container.fs-6")
    if not container:
        return relasi

    for div in container.children:
        if not hasattr(div, "select_one"):
            continue

        # Check for relasi type header (e.g. "Mengubah :")
        type_div = div.select_one("div.fw-semibold.bg-light-primary")
        if type_div:
            current_type = type_div.get_text(strip=True).rstrip(" :")
            continue

        # Check for list of related regulations
        if current_type:
            for li in div.select("ol li"):
                ref = {}
                ref["tipe_relasi"] = current_type
                link = li.select_one("a[href^='/Details/']")
                if link:
                    ref["ref_display"] = link.get_text(strip=True)
                    ref_href = link["href"]
                    ref_parts = ref_href.strip("/").split("/")
                    if len(ref_parts) >= 3:
                        ref["ref_id"] = ref_parts[1]
                        ref["ref_slug"] = ref_parts[2]

                # Get the full text of the li minus the link text
                full_text = li.get_text(" ", strip=True)
                link_text = link.get_text(strip=True) if link else ""
                remainder = full_text.replace(link_text, "", 1).strip()
                if remainder.startswith("tentang "):
                    remainder = remainder[len("tentang "):]
                ref["keterangan"] = remainder

                relasi.append(ref)

    return relasi


# ---------------------------------------------------------------------------
# PDF download
# ---------------------------------------------------------------------------


def download_pdf(
    file_id: str,
    filename: str,
    href: str,
    output_dir: Path,
    session: requests.Session,
) -> bool:
    """Stream-download a PDF file. Skips if already exists."""
    safe_name = sanitize_filename(filename) if filename else f"{file_id}.pdf"
    if not safe_name.endswith(".pdf"):
        safe_name += ".pdf"
    dest = output_dir / safe_name

    if dest.exists() and dest.stat().st_size > 0:
        log.debug("PDF already exists, skipping: %s", dest)
        return True

    url = urljoin(BASE_URL, href)
    try:
        resp = session.get(url, headers=HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info("Downloaded PDF: %s", dest.name)
        return True
    except requests.RequestException as exc:
        log.error("Failed to download PDF %s: %s", url, exc)
        if dest.exists():
            dest.unlink()
        return False


# ---------------------------------------------------------------------------
# Registry generation
# ---------------------------------------------------------------------------


def generate_registry(output_dir: Path) -> dict:
    """Build a unified registry from all jenis subfolders' metadata.

    Merges with existing registry.json so incremental scraper runs
    (different --jenis each time) don't lose previous entries.
    """
    registry_path = output_dir / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        log.info("Loaded existing registry with %d entries", len(registry))
    else:
        registry = {}

    # Scan all {jenis}/metadata/ subdirectories
    for metadata_dir in sorted(output_dir.glob("*/metadata")):
        jenis_folder = metadata_dir.parent.name  # e.g. "UU", "PP"
        for json_file in sorted(metadata_dir.glob("*.json")):
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("Skipping invalid JSON %s: %s", json_file.name, exc)
                continue

            doc_id = data.get("doc_id", json_file.stem)
            entry = {
                "doc_id": doc_id,
                "detail_id": data.get("detail_id"),
                "jenis_folder": jenis_folder,
                "kategori": data.get("kategori", ""),
                "bentuk_singkat": data.get("bentuk_singkat", ""),
                "nomor": data.get("nomor", ""),
                "tahun": data.get("tahun", ""),
                "judul": data.get("judul", ""),
                "status": data.get("status", ""),
                "tanggal_penetapan": data.get("tanggal_penetapan", ""),
            }

            # Summarize relasi
            relasi_summary = []
            for rel in data.get("relasi", []):
                relasi_summary.append({
                    "tipe": rel.get("tipe_relasi", ""),
                    "ref": rel.get("ref_display", ""),
                    "ref_id": rel.get("ref_id", ""),
                })
            if relasi_summary:
                entry["relasi"] = relasi_summary

            entry["has_pdf"] = bool(data.get("pdf_files"))

            registry[doc_id] = entry

    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Registry written with %d entries: %s", len(registry), registry_path)
    return registry


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_page_range(page_str: str, total_pages: int) -> range:
    """Parse a page range string like '1-10', '5', or 'all'."""
    if page_str.lower() == "all":
        return range(1, total_pages + 1)
    if "-" in page_str:
        start, end = page_str.split("-", 1)
        return range(int(start), min(int(end), total_pages) + 1)
    return range(int(page_str), int(page_str) + 1)


def parse_tahun_arg(tahun_str: str | None) -> list[int | None]:
    """Parse tahun argument: '2024', '2020-2026', or None → [None]."""
    if tahun_str is None:
        return [None]
    if "-" in tahun_str:
        start, end = tahun_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(tahun_str)]


def main():
    parser = argparse.ArgumentParser(
        description="Scrape legal documents from BPK JDIH (peraturan.bpk.go.id)",
    )
    parser.add_argument(
        "--jenis",
        type=int,
        nargs="+",
        default=[8],
        help="Jenis ID(s) to scrape (8=UU, 9=Perpu, 10=PP, 11=Perpres). Default: 8",
    )
    parser.add_argument(
        "--pages",
        default="all",
        help='Page range, e.g. "1-10" or "5" or "all" (default: all)',
    )
    default_output = str(Path(__file__).resolve().parent.parent / "data" / "raw")
    parser.add_argument(
        "--output",
        default=default_output,
        help="Output directory (default: data/raw/)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY,
        help=f"Delay between requests in seconds (default: {DEFAULT_DELAY})",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Skip PDF download (metadata only)",
    )
    parser.add_argument(
        "--tahun",
        default=None,
        help='Year filter, e.g. "2024" or "2020-2026". BPK filters by single year, ranges iterate per year.',
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-scraped detail pages (resume interrupted scrape)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    output_dir = Path(args.output)

    session = requests.Session()
    session.headers.update(HEADERS)

    total_scraped = 0
    total_skipped = 0
    total_errors = 0

    tahun_list = parse_tahun_arg(args.tahun)

    for jenis_id in args.jenis:
        jenis_name = JENIS_MAP.get(jenis_id, f"jenis-{jenis_id}")
        kategori = KATEGORI_MAP.get(jenis_id, "Lainnya")

        # Per-jenis subdirectories: data/UU/metadata/, data/UU/pdfs/
        metadata_dir = output_dir / jenis_name / "metadata"
        pdfs_dir = output_dir / jenis_name / "pdfs"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        pdfs_dir.mkdir(parents=True, exist_ok=True)

        for tahun in tahun_list:
            tahun_label = f" tahun={tahun}" if tahun else ""
            log.info("=== Starting scrape for %s (jenis_id=%d, kategori=%s%s) ===",
                     jenis_name, jenis_id, kategori, tahun_label)

            # Discover total pages
            total_pages = get_total_pages(jenis_id, session, tahun=tahun)
            if total_pages == 0:
                log.info("No pages found for %s%s. Skipping.", jenis_name, tahun_label)
                continue
            log.info("Total pages for %s%s: %d", jenis_name, tahun_label, total_pages)
            time.sleep(args.delay)

            page_range = parse_page_range(args.pages, total_pages)
            log.info("Scraping pages %d to %d", page_range.start, page_range.stop - 1)

            for page_num in page_range:
                log.info("--- Page %d/%d ---", page_num, page_range.stop - 1)
                items = scrape_list_page(jenis_id, page_num, session, tahun=tahun)
                log.info("Found %d items on page %d", len(items), page_num)
                time.sleep(args.delay)

                for idx, item in enumerate(items, 1):
                    detail_id = item["detail_id"]
                    slug = item["slug"]
                    log.info(
                        "  [%d/%d] %s — %s",
                        idx, len(items), item.get("nomor_tahun_text", ""), item.get("judul", ""),
                    )

                    # Check for resume: if metadata JSON already exists, skip
                    # We check by detail_id since doc_id is only known after scraping
                    existing = list(metadata_dir.glob(f"*__{detail_id}.json"))
                    if args.resume and existing:
                        log.info("    Skipped (already scraped): %s", existing[0].name)
                        total_skipped += 1
                        continue

                    # Scrape detail page
                    meta = scrape_detail_page(detail_id, slug, session)
                    if meta is None:
                        log.error("    Failed to scrape detail page for %s", detail_id)
                        total_errors += 1
                        time.sleep(args.delay)
                        continue

                    doc_id = meta.get("doc_id", f"unknown-{detail_id}")
                    meta["kategori"] = kategori

                    # Add judul from list page if not in detail metadata
                    if "judul" not in meta or not meta["judul"]:
                        meta["judul"] = item.get("judul", "")

                    # Save metadata JSON
                    # Filename: {doc_id}__{detail_id}.json for uniqueness + lookup
                    json_path = metadata_dir / f"{doc_id}__{detail_id}.json"
                    json_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    log.info("    Saved metadata: %s", json_path.name)

                    # Download PDF
                    if not args.skip_pdf and meta.get("pdf_files"):
                        for pdf_info in meta["pdf_files"]:
                            download_pdf(
                                pdf_info["file_id"],
                                pdf_info["filename"],
                                pdf_info["href"],
                                pdfs_dir,
                                session,
                            )

                    total_scraped += 1
                    time.sleep(args.delay)

    # Generate unified registry from all jenis subfolders
    log.info("=== Generating registry ===")
    generate_registry(output_dir)

    log.info(
        "=== Done. Scraped: %d | Skipped: %d | Errors: %d ===",
        total_scraped, total_skipped, total_errors,
    )


if __name__ == "__main__":
    main()
