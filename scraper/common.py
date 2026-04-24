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

# Jenis ID → short folder name (used for folder structure + display).
# Convention: short UPPERCASE acronyms, no spaces. Consistent with legal
# drafting convention in Indonesia (UU, PP, PERPRES, PMK, PERMENAKER).
JENIS_MAP = {
    # Pusat
    8: "UU",
    9: "PERPU",
    10: "PP",
    11: "PERPRES",
    12: "KEPPRES",
    13: "INPRES",
    36: "UUDRT",
    273: "PERMEN_KEBUDAYAAN",
    # Daerah
    19: "PERDA",
    20: "PERGUB",
    23: "PERBUP",
    30: "PERWALI",
    34: "KANUN",
    35: "PERDASUS_PAPUA",
    38: "PERDA_ISTIMEWA",
    # Kementerian/Lembaga
    27: "PERATURAN_BPK",
    28: "KEPUTUSAN_BPK",
    40: "PERMENDAGRI",
    42: "PMK",
    43: "PERMENKO_POLHUKAM",
    45: "PERMENLU",
    46: "PERMENKUMHAM",
    47: "PERMENHAN",
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
    62: "PERATURAN_BAWASLU",
    66: "KEPUTUSAN_MENKEU",
    67: "PERMENDAG",
    69: "PERMENPERIN",
    71: "PERATURAN_BAPPENAS",
    73: "PERMENKOP_UKM",
    75: "PERATURAN_BKPM",
    76: "PERATURAN_BPS",
    77: "KEPUTUSAN_BPS",
    78: "PERATURAN_BI",
    80: "PERATURAN_OJK",
    81: "PERATURAN_PPATK",
    83: "PERATURAN_LPS",
    86: "KEPUTUSAN_BSN",
    87: "PERATURAN_LKPP",
    88: "KEPUTUSAN_LKPP",
    89: "PERATURAN_KPPU",
    90: "KEPUTUSAN_KPPU",
    92: "PERATURAN_DPR",
    93: "PERATURAN_DPD",
    95: "PERATURAN_MA",
    98: "PERATURAN_MK",
    99: "PERATURAN_KY",
    100: "PERMENKO_PMK",
    101: "PERMENSESNEG",
    103: "PERMENSOS",
    104: "PERMENPAREKRAF",
    105: "PERMENAKER",
    106: "PERMENKOMINFO",
    107: "PERMENPAN_RB",
    108: "PERMEN_PPPA",
    109: "PERMENPORA",
    110: "PERMENRISTEKDIKTI",
    111: "PERMEN_ATR_BPN",
    112: "PERMENDESA_PDTT",
    113: "PERATURAN_BAPETEN",
    114: "PERATURAN_BATAN",
    116: "PERATURAN_LIPI",
    118: "PERATURAN_PERPUSNAS",
    119: "PERATURAN_BNPB",
    121: "PERATURAN_BKKBN",
    122: "PERATURAN_BKN",
    123: "PERATURAN_BPKP",
    124: "PERATURAN_LAN",
    182: "PERMENKES",
    219: "PERMENKO_KESRA",
    223: "PERATURAN_BPS_BARU",
    225: "PERATURAN_BSN",
    228: "PERATURAN_BATAN_BARU",
    246: "PERATURAN_PERPUSNAS_BARU",
    255: "PERATURAN_BRIN",
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
    """Fetch a page with bounded retries."""
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
