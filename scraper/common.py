"""Shared HTTP settings and type maps for the BPK JDIH scraper."""

import logging
import time

import requests

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

# Jenis ID to short folder name.
JENIS_MAP = {
    # Pusat
    8: "UU",
    10: "PP",
    11: "PERPRES",
    # Daerah
    19: "PERDA",
    20: "PERGUB",
    23: "PERBUP",
    30: "PERWALI",
    # Kementerian/Lembaga
    27: "PERATURAN_BPK",
    40: "PERMENDAGRI",
    42: "PMK",
    43: "PERMENKO_POLHUKAM",
    45: "PERMENLU",
    46: "PERMENKUMHAM",
    48: "PERMENHUB",
    49: "PERATURAN_KEJAKSAAN",
    50: "PERATURAN_KAPOLRI",
    52: "PERATURAN_BNN",
    53: "PERATURAN_BMKG",
    54: "PERATURAN_BSSN",
    56: "PERATURAN_KOMNAS_HAM",
    58: "PERATURAN_KPK",
    59: "PERATURAN_KPU",
    61: "PERATURAN_BNPT",
    67: "PERMENDAG",
    75: "PERATURAN_BKPM",
    78: "PERATURAN_BI",
    80: "PERATURAN_OJK",
    92: "PERATURAN_DPR",
    95: "PERATURAN_MA",
    98: "PERATURAN_MK",
    99: "PERATURAN_KY",
    105: "PERMENAKER",
    106: "PERMENKOMINFO",
    110: "PERMENRISTEKDIKTI",
    147: "PERMEN_ESDM",
    154: "PERMEN_PUPR",
    182: "PERMENKES",
    224: "PERATURAN_JAKSA_AGUNG",
    230: "PERATURAN_BPOM",
}

# Jenis ID to broad category group
KATEGORI_MAP = {
    # Pusat
    8: "Pusat",
    10: "Pusat",
    11: "Pusat",
    # Daerah
    19: "Daerah",
    20: "Daerah",
    23: "Daerah",
    30: "Daerah",
    # Kementerian/Lembaga
    27: "Kementerian/Lembaga",
    40: "Kementerian/Lembaga",
    42: "Kementerian/Lembaga",
    43: "Kementerian/Lembaga",
    45: "Kementerian/Lembaga",
    46: "Kementerian/Lembaga",
    48: "Kementerian/Lembaga",
    49: "Kementerian/Lembaga",
    50: "Kementerian/Lembaga",
    52: "Kementerian/Lembaga",
    53: "Kementerian/Lembaga",
    54: "Kementerian/Lembaga",
    56: "Kementerian/Lembaga",
    58: "Kementerian/Lembaga",
    59: "Kementerian/Lembaga",
    61: "Kementerian/Lembaga",
    67: "Kementerian/Lembaga",
    75: "Kementerian/Lembaga",
    78: "Kementerian/Lembaga",
    80: "Kementerian/Lembaga",
    92: "Kementerian/Lembaga",
    95: "Kementerian/Lembaga",
    98: "Kementerian/Lembaga",
    99: "Kementerian/Lembaga",
    105: "Kementerian/Lembaga",
    106: "Kementerian/Lembaga",
    110: "Kementerian/Lembaga",
    147: "Kementerian/Lembaga",
    154: "Kementerian/Lembaga",
    182: "Kementerian/Lembaga",
    224: "Kementerian/Lembaga",
    230: "Kementerian/Lembaga",
}

NON_RETRYABLE_STATUS = {400, 401, 403, 404, 405, 410, 451}

log = logging.getLogger("bpk")

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