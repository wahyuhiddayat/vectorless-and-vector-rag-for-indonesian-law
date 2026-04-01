"""
Shared constants and utilities for BPK JDIH scraper + survey scripts.
"""

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

# Jenis ID → short folder name (used for folder structure + display)
JENIS_MAP = {
    # Pusat
    8: "UU",
    9: "Perpu",
    10: "PP",
    11: "Perpres",
    12: "Keppres",
    13: "Inpres",
    36: "UU Darurat",
    273: "Permen Kebudayaan",
    # Daerah
    19: "Perda",
    20: "Pergub",
    23: "Perbup",
    30: "Perwali",
    34: "Kanun",
    35: "Perdasus Papua",
    38: "Perda Istimewa",
    # Kementerian/Lembaga
    27: "Peraturan BPK",
    28: "Keputusan BPK",
    40: "Permendagri",
    42: "PMK",
    43: "Permenko Polhukam",
    45: "Permenlu",
    46: "Permenkumham",
    47: "Permenhan",
    48: "Permenhub",
    49: "Peraturan Kejaksaan",
    50: "Peraturan Kapolri",
    52: "Peraturan BNN",
    53: "Peraturan BMKG",
    54: "Peraturan BSSN",
    56: "Peraturan Komnas HAM",
    58: "Peraturan KPK",
    59: "Peraturan KPU",
    61: "Peraturan BNPT",
    62: "Peraturan Bawaslu",
    66: "Keputusan Menkeu",
    67: "Permendag",
    69: "Permenperin",
    71: "Peraturan Bappenas",
    73: "Permenkop UKM",
    75: "Peraturan BKPM",
    76: "Peraturan BPS",
    77: "Keputusan BPS",
    78: "Peraturan BI",
    80: "Peraturan OJK",
    81: "Peraturan PPATK",
    83: "Peraturan LPS",
    86: "Keputusan BSN",
    87: "Peraturan LKPP",
    88: "Keputusan LKPP",
    89: "Peraturan KPPU",
    90: "Keputusan KPPU",
    92: "Peraturan DPR",
    93: "Peraturan DPD",
    95: "Peraturan MA",
    98: "Peraturan MK",
    99: "Peraturan KY",
    100: "Permenko PMK",
    101: "Permensesneg",
    103: "Permensos",
    104: "Permenparekraf",
    105: "Permenaker",
    106: "Permenkominfo",
    107: "Permenpan-RB",
    108: "Permen PPPA",
    109: "Permenpora",
    110: "Permenristekdikti",
    111: "Permen ATR-BPN",
    112: "Permendesa PDTT",
    113: "Peraturan Bapeten",
    114: "Peraturan BATAN",
    116: "Peraturan LIPI",
    118: "Peraturan Perpusnas",
    119: "Peraturan BNPB",
    121: "Peraturan BKKBN",
    122: "Peraturan BKN",
    123: "Peraturan BPKP",
    124: "Peraturan LAN",
    182: "Permenkes",
    219: "Permenko Kesra",
    223: "Peraturan BPS Baru",
    225: "Peraturan BSN",
    228: "Peraturan BATAN Baru",
    246: "Peraturan Perpusnas Baru",
    255: "Peraturan BRIN",
}

# Jenis ID → broad category group
KATEGORI_MAP = {
    # Pusat
    8: "Pusat", 9: "Pusat", 10: "Pusat", 11: "Pusat",
    12: "Pusat", 13: "Pusat", 36: "Pusat", 273: "Pusat",
    # Daerah
    19: "Daerah", 20: "Daerah", 23: "Daerah", 30: "Daerah",
    34: "Daerah", 35: "Daerah", 38: "Daerah",
    # Kementerian/Lembaga
    27: "Kementerian/Lembaga", 28: "Kementerian/Lembaga",
    40: "Kementerian/Lembaga", 42: "Kementerian/Lembaga",
    43: "Kementerian/Lembaga", 45: "Kementerian/Lembaga",
    46: "Kementerian/Lembaga", 47: "Kementerian/Lembaga",
    48: "Kementerian/Lembaga", 49: "Kementerian/Lembaga",
    50: "Kementerian/Lembaga", 52: "Kementerian/Lembaga",
    53: "Kementerian/Lembaga", 54: "Kementerian/Lembaga",
    56: "Kementerian/Lembaga", 58: "Kementerian/Lembaga",
    59: "Kementerian/Lembaga", 61: "Kementerian/Lembaga",
    62: "Kementerian/Lembaga", 66: "Kementerian/Lembaga",
    67: "Kementerian/Lembaga", 69: "Kementerian/Lembaga",
    71: "Kementerian/Lembaga", 73: "Kementerian/Lembaga",
    75: "Kementerian/Lembaga", 76: "Kementerian/Lembaga",
    77: "Kementerian/Lembaga", 78: "Kementerian/Lembaga",
    80: "Kementerian/Lembaga", 81: "Kementerian/Lembaga",
    83: "Kementerian/Lembaga", 86: "Kementerian/Lembaga",
    87: "Kementerian/Lembaga", 88: "Kementerian/Lembaga",
    89: "Kementerian/Lembaga", 90: "Kementerian/Lembaga",
    92: "Kementerian/Lembaga", 93: "Kementerian/Lembaga",
    95: "Kementerian/Lembaga", 98: "Kementerian/Lembaga",
    99: "Kementerian/Lembaga", 100: "Kementerian/Lembaga",
    101: "Kementerian/Lembaga", 103: "Kementerian/Lembaga",
    104: "Kementerian/Lembaga", 105: "Kementerian/Lembaga",
    106: "Kementerian/Lembaga", 107: "Kementerian/Lembaga",
    108: "Kementerian/Lembaga", 109: "Kementerian/Lembaga",
    110: "Kementerian/Lembaga", 111: "Kementerian/Lembaga",
    112: "Kementerian/Lembaga", 113: "Kementerian/Lembaga",
    114: "Kementerian/Lembaga", 116: "Kementerian/Lembaga",
    118: "Kementerian/Lembaga", 119: "Kementerian/Lembaga",
    121: "Kementerian/Lembaga", 122: "Kementerian/Lembaga",
    123: "Kementerian/Lembaga", 124: "Kementerian/Lembaga",
    182: "Kementerian/Lembaga",
    219: "Kementerian/Lembaga", 223: "Kementerian/Lembaga",
    225: "Kementerian/Lembaga", 228: "Kementerian/Lembaga",
    246: "Kementerian/Lembaga", 255: "Kementerian/Lembaga",
}

log = logging.getLogger("bpk")


def fetch(url: str, session: requests.Session, retries: int = MAX_RETRIES) -> requests.Response | None:
    """GET with retries. Returns Response or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            log.warning("Attempt %d/%d failed for %s: %s", attempt, retries, url, exc)
            if attempt < retries:
                time.sleep(2 * attempt)
    log.error("All %d attempts failed for %s", retries, url)
    return None
