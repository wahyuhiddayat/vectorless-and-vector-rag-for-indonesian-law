"""Scrape BPK JDIH legal documents into per-jenis metadata JSON and PDF files.

Targets peraturan.bpk.go.id and supports UU, Perpu, PP, Perpres listings,
detail-page metadata extraction, deterministic PDF naming, and a unified
registry rebuild step.

Usage:
    python scraper/bpk_scraper.py --jenis 8 --pages 1-5
    python scraper/bpk_scraper.py --jenis 8 --resume
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


BASE_URL = "https://peraturan.bpk.go.id"

DEFAULT_DELAY = 1.5
MAX_RETRIES = 3

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "id-ID,id;q=0.9,en-US;q=0.8,en;q=0.7",
}

NON_RETRYABLE_STATUS = {400, 401, 403, 404, 405, 410, 451}

# Jenis ID to short folder name.
JENIS_MAP = {
    # Pusat
    8: "UU",
    9: "PERPU",
    10: "PP",
    11: "PERPRES",
    # Daerah
    19: "PERDA",
    20: "PERGUB",
    23: "PERBUP",
    30: "PERWALI",
    # Kementerian/Lembaga, fixed at 20 per Notes/01-corpus/categories.md
    154: "PERMEN_PUPR",
    40: "PERMENDAGRI",
    42: "PMK",
    69: "PERMENPERIN",
    170: "PERMENAG",
    241: "PERATURAN_POLRI",
    54: "PERATURAN_BSSN",
    202: "PERMENBUMN",
    67: "PERMENDAG",
    186: "PERMENDIKBUD",
    78: "PERATURAN_BI",
    80: "PERATURAN_OJK",
    95: "PERATURAN_MA",
    105: "PERMENAKER",
    278: "PERMENKOMDIGI",
    242: "PERMENDIKBUDRISET",
    147: "PERMEN_ESDM",
    111: "PERMEN_ATRBPN",
    182: "PERMENKES",
    230: "PERATURAN_BPOM",
}

# Jenis ID to broad category group.
KATEGORI_MAP = {
    # Pusat
    8: "Pusat",
    9: "Pusat",
    10: "Pusat",
    11: "Pusat",
    # Daerah
    19: "Daerah",
    20: "Daerah",
    23: "Daerah",
    30: "Daerah",
    # Kementerian/Lembaga
    154: "Kementerian/Lembaga",
    40: "Kementerian/Lembaga",
    42: "Kementerian/Lembaga",
    69: "Kementerian/Lembaga",
    170: "Kementerian/Lembaga",
    241: "Kementerian/Lembaga",
    54: "Kementerian/Lembaga",
    202: "Kementerian/Lembaga",
    67: "Kementerian/Lembaga",
    186: "Kementerian/Lembaga",
    78: "Kementerian/Lembaga",
    80: "Kementerian/Lembaga",
    95: "Kementerian/Lembaga",
    105: "Kementerian/Lembaga",
    278: "Kementerian/Lembaga",
    242: "Kementerian/Lembaga",
    147: "Kementerian/Lembaga",
    111: "Kementerian/Lembaga",
    182: "Kementerian/Lembaga",
    230: "Kementerian/Lembaga",
}

log = logging.getLogger("bpk_scraper")


def fetch(url: str, session: requests.Session, retries: int = MAX_RETRIES) -> requests.Response | None:
    """Fetch a page with bounded retries, skipping retries on non-recoverable status codes."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status in NON_RETRYABLE_STATUS:
                log.warning("Non-retryable status %d for %s", status, url)
                return None
            log.warning("Attempt %d/%d failed for %s, error %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s, error %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
    log.error("All %d attempts failed for %s", retries, url)
    return None


def sanitize_filename(name: str) -> str:
    """Normalize a string into a filesystem-safe filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


_DAERAH_RE = re.compile(
    r"\b(?:Provinsi|Kabupaten|Kota)\s+(.+?)\s+Nomor\b", re.IGNORECASE
)


def _daerah_slug(judul: str | None) -> str | None:
    """Extract daerah issuer slug from judul, or None for non-daerah docs.

    Pusat/Kementerian docs do not match the Provinsi/Kabupaten/Kota pattern
    and return None, so their doc_id format stays the legacy two-segment shape.
    """
    if not judul:
        return None
    m = _DAERAH_RE.search(judul)
    if not m:
        return None
    name = m.group(1).strip().lower()
    name = re.sub(r"[^\w\s-]", "", name)
    return re.sub(r"\s+", "-", name).strip("-")


def make_doc_id(bentuk_singkat: str, nomor: str, tahun: str,
                judul: str | None = None) -> str:
    """Build the canonical `doc_id`.

    Pusat/Kementerian docs use `{bentuk}-{nomor}-{tahun}` (e.g. `uu-1-2026`).
    Daerah docs (PERGUB, PERBUP, PERWAL, PERDA-Provinsi/Kab/Kota) include a
    daerah segment to disambiguate identical numbering across provinces, e.g.
    `pergub-dki-jakarta-11-2026` vs `pergub-jambi-1-2026`.
    """
    bs = bentuk_singkat.strip().lower() if bentuk_singkat else "unknown"
    bs = re.sub(r"[/\\:*?\"<>|\s]+", "-", bs).strip("-")
    n = nomor.strip() if nomor else "0"
    n = re.sub(r"[/\\:*?\"<>|\s]+", "-", n).strip("-").lower()
    t = tahun.strip() if tahun else "0"
    daerah = _daerah_slug(judul)
    if daerah:
        return f"{bs}-{daerah}-{n}-{t}"
    return f"{bs}-{n}-{t}"


def get_total_pages(jenis_id: int, session: requests.Session, tahun: int | None = None) -> int:
    """Return the last pagination page for a `jenis` listing."""
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
    last_link = None
    for a in pagination.find_all("a", class_="page-link"):
        if a.get_text(strip=True) == "Last":
            last_link = a
            break
    if last_link and last_link.get("href"):
        match = re.search(r"[?&]p=(\d+)", last_link["href"])
        if match:
            return int(match.group(1))
    max_page = 1
    for a in pagination.find_all("a", class_="page-link"):
        txt = a.get_text(strip=True)
        if txt.isdigit():
            max_page = max(max_page, int(txt))
    return max_page


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


def scrape_detail_page(detail_id: str, slug: str, session: requests.Session) -> dict | None:
    """Scrape a detail page into the metadata shape used on disk."""
    url = f"{BASE_URL}/Details/{detail_id}/{slug}"
    resp = fetch(url, session)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")
    metadata: dict = {"detail_id": detail_id, "slug": slug, "url": url}

    # `string=` does not match here because the heading contains child spans.
    materi_header = soup.find(
        lambda tag: tag.name == "h4" and "MATERI POKOK" in tag.get_text()
    )
    if materi_header:
        card_body = materi_header.find_parent("div", class_="card-body")
        if card_body:
            paragraphs = card_body.find_all("p")
            materi_texts = [p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)]
            metadata["materi_pokok"] = " ".join(materi_texts)

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
                        norm_key = key.lower().replace(".", "").replace(" ", "_")
                        metadata[norm_key] = val

    status_header = soup.find(
        lambda tag: tag.name == "h4" and tag.get_text(strip=True).startswith("STATUS")
    )
    if status_header:
        status_card = status_header.find_parent("div", class_="card-body")
        if status_card:
            relasi = _parse_status_peraturan(status_card)
            if relasi:
                metadata["relasi"] = relasi

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
    seen_ids = set()
    unique_pdfs = []
    for pf in pdf_files:
        if pf["file_id"] not in seen_ids:
            seen_ids.add(pf["file_id"])
            unique_pdfs.append(pf)
    if unique_pdfs:
        metadata["pdf_files"] = unique_pdfs

    bentuk_singkat = metadata.get("bentuk_singkat", "")
    nomor = metadata.get("nomor", "")
    tahun = metadata.get("tahun", "")
    judul = metadata.get("judul", "")
    metadata["doc_id"] = make_doc_id(bentuk_singkat, nomor, tahun, judul)

    return metadata


def _parse_status_peraturan(card_body) -> list[dict]:
    """Parse the STATUS PERATURAN card into relation entries."""
    relasi = []
    current_type = None

    container = card_body.select_one("div.container.fs-6")
    if not container:
        return relasi

    for div in container.children:
        if not hasattr(div, "select_one"):
            continue

        type_div = div.select_one("div.fw-semibold.bg-light-primary")
        if type_div:
            current_type = type_div.get_text(strip=True).rstrip(" :")
            continue

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

                full_text = li.get_text(" ", strip=True)
                link_text = link.get_text(strip=True) if link else ""
                remainder = full_text.replace(link_text, "", 1).strip()
                if remainder.startswith("tentang "):
                    remainder = remainder[len("tentang "):]
                ref["keterangan"] = remainder

                relasi.append(ref)

    return relasi


def download_pdf(
    dest_name: str,
    href: str,
    output_dir: Path,
    session: requests.Session,
) -> Path | None:
    """Stream-download a PDF and write it under the caller-supplied filename.

    Returns the destination Path on success or when the file already exists,
    None on error. The caller owns the naming policy so registry consumers do
    not need to re-derive it from the source filename.
    """
    dest = output_dir / dest_name

    if dest.exists() and dest.stat().st_size > 0:
        log.debug("PDF already exists, skipping: %s", dest)
        return dest

    url = urljoin(BASE_URL, href)
    try:
        resp = session.get(url, headers=HEADERS, stream=True, timeout=60)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        log.info("Downloaded PDF: %s", dest.name)
        return dest
    except requests.RequestException as exc:
        log.error("Failed to download PDF %s: %s", url, exc)
        if dest.exists():
            dest.unlink()
        return None


def generate_registry(output_dir: Path) -> dict:
    """Rebuild `registry.json` from the scraped metadata directories."""
    registry_path = output_dir / "registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        log.info("Loaded existing registry with %d entries", len(registry))
    else:
        registry = {}

    for metadata_dir in sorted(output_dir.glob("*/metadata")):
        jenis_folder = metadata_dir.parent.name
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
            # Deterministic PDF paths relative to data/raw/. Populated by the
            # download step. Consumers should prefer these over filename guessing.
            if data.get("pdf_path"):
                entry["pdf_path"] = data["pdf_path"]
            if data.get("lampiran_paths"):
                entry["lampiran_paths"] = data["lampiran_paths"]
            if data.get("extra_paths"):
                entry["extra_paths"] = data["extra_paths"]

            registry[doc_id] = entry

    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Registry written with %d entries: %s", len(registry), registry_path)
    return registry


def parse_page_range(page_str: str, total_pages: int) -> range:
    """Parse a page range string like '1-10', '5', or 'all'."""
    if page_str.lower() == "all":
        return range(1, total_pages + 1)
    if "-" in page_str:
        start, end = page_str.split("-", 1)
        return range(int(start), min(int(end), total_pages) + 1)
    page = min(int(page_str), total_pages)
    return range(page, page + 1)


def parse_tahun_arg(tahun_str: str | None) -> list[int | None]:
    """Parse tahun argument: '2024', '2020-2026', or None → [None]."""
    if tahun_str is None:
        return [None]
    if "-" in tahun_str:
        start, end = tahun_str.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(tahun_str)]


def main():
    """Run the BPK scraper CLI, scrape pages, save metadata and PDFs, then rebuild registry."""
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
        "--skip-doc-ids",
        default="",
        help="Comma-separated doc_ids to skip (e.g. uu-1-2026,uu-20-2025)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max docs to scrape per jenis (0 = no limit). Counts only "
             "successfully-scraped docs (skipped/errored do not count).",
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
    skip_doc_ids = {
        d.strip().lower() for d in args.skip_doc_ids.split(",") if d.strip()
    }
    dropped_log_path = Path("data/dropped_docs.json")
    if dropped_log_path.exists():
        try:
            dropped = json.load(dropped_log_path.open(encoding="utf-8"))
            blacklist = {d["doc_id"].lower() for d in dropped.get("docs", [])}
            new = blacklist - skip_doc_ids
            if new:
                log.info("Auto-skip %d doc_ids from dropped_docs.json", len(new))
            skip_doc_ids |= blacklist
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Could not load dropped_docs.json: %s", exc)
    if skip_doc_ids:
        log.info("Will skip doc_ids: %s", sorted(skip_doc_ids))

    for jenis_id in args.jenis:
        jenis_name = JENIS_MAP.get(jenis_id, f"jenis-{jenis_id}")
        if jenis_id not in KATEGORI_MAP:
            log.warning("Unknown jenis_id %d, falling back to kategori 'Lainnya'", jenis_id)
        kategori = KATEGORI_MAP.get(jenis_id, "Lainnya")
        jenis_scraped_count = 0  # counts successful new scrapes for --limit

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
                log.error("Failed to fetch listing for %s%s, skipping.", jenis_name, tahun_label)
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
                        "  [%d/%d] %s, %s",
                        idx, len(items), item.get("nomor_tahun_text", ""), item.get("judul", ""),
                    )

                    # Resume support. If metadata JSON already exists for this detail_id, skip.
                    # detail_id is the only id available before the detail page is scraped.
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
                    if doc_id.lower() in skip_doc_ids:
                        log.info("    Skipped (on skip list): %s", doc_id)
                        total_skipped += 1
                        time.sleep(args.delay)
                        continue
                    meta["kategori"] = kategori

                    # Add judul from list page if not in detail metadata
                    if "judul" not in meta or not meta["judul"]:
                        meta["judul"] = item.get("judul", "")

                    # Use {doc_id}__{detail_id}.json so the file is unique and resolvable by detail_id.
                    json_path = metadata_dir / f"{doc_id}__{detail_id}.json"
                    json_path.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    log.info("    Saved metadata: %s", json_path.name)

                    # Download PDF with deterministic doc_id-based filename.
                    # Record local paths in metadata so registry can resolve
                    # doc_id → PDF without filename guessing later.
                    if not args.skip_pdf and meta.get("pdf_files"):
                        main_pdf_path: Path | None = None
                        lampiran_paths: list[Path] = []
                        extra_paths: list[Path] = []
                        suffix_counter = {"lampiran": 0, "extra": 0}
                        for pdf_info in meta["pdf_files"]:
                            original_name = pdf_info.get("filename", "")
                            is_lampiran = bool(original_name) and (
                                "lampiran" in original_name.lower()
                                or "lamp_" in original_name.lower()
                            )
                            if is_lampiran:
                                suffix_counter["lampiran"] += 1
                                sfx = suffix_counter["lampiran"]
                                dest_name = f"{doc_id}_lampiran_{sfx}.pdf"
                            elif main_pdf_path is not None:
                                # Second non-lampiran PDF, treat as extra.
                                suffix_counter["extra"] += 1
                                sfx = suffix_counter["extra"]
                                dest_name = f"{doc_id}_extra_{sfx}.pdf"
                            else:
                                sfx = 0
                                dest_name = f"{doc_id}.pdf"
                            result = download_pdf(
                                dest_name,
                                pdf_info["href"],
                                pdfs_dir,
                                session,
                            )
                            if result is None:
                                continue
                            rel = result.relative_to(output_dir)
                            pdf_info["local_path"] = str(rel).replace("\\", "/")
                            if is_lampiran:
                                lampiran_paths.append(result)
                            elif sfx == 0:
                                main_pdf_path = result
                            else:
                                extra_paths.append(result)
                        # Persist resolved paths back to metadata (registry will pick up).
                        if main_pdf_path is not None:
                            meta["pdf_path"] = str(
                                main_pdf_path.relative_to(output_dir)
                            ).replace("\\", "/")
                        if lampiran_paths:
                            meta["lampiran_paths"] = [
                                str(p.relative_to(output_dir)).replace("\\", "/")
                                for p in lampiran_paths
                            ]
                        if extra_paths:
                            meta["extra_paths"] = [
                                str(p.relative_to(output_dir)).replace("\\", "/")
                                for p in extra_paths
                            ]
                        # Re-save metadata with resolved paths.
                        json_path.write_text(
                            json.dumps(meta, ensure_ascii=False, indent=2),
                            encoding="utf-8",
                        )

                    total_scraped += 1
                    jenis_scraped_count += 1
                    time.sleep(args.delay)

                    if args.limit and jenis_scraped_count >= args.limit:
                        log.info(
                            "    Reached --limit %d for %s. Stopping this jenis.",
                            args.limit, jenis_name,
                        )
                        break
                if args.limit and jenis_scraped_count >= args.limit:
                    break
            if args.limit and jenis_scraped_count >= args.limit:
                break

    # Generate unified registry from all jenis subfolders
    log.info("=== Generating registry ===")
    generate_registry(output_dir)

    log.info(
        "=== Done. Scraped: %d | Skipped: %d | Errors: %d ===",
        total_scraped, total_skipped, total_errors,
    )


if __name__ == "__main__":
    main()
