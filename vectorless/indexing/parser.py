import re
import json
import logging
import os
import sys
import time
import fitz
from pathlib import Path

log = logging.getLogger(__name__)

_LLM_BATCH_SIZE = 50_000 # ~50K chars ≈ ~12K tokens; smaller batches give more reliable Gemini JSON output.

# ============================================================
# TEXT EXTRACTION & CLEANING
# ============================================================

def _detect_two_columns(blocks: list[dict], page_width: float,
                        is_landscape: bool = False) -> list[dict]:
    """Reorder text blocks for correct reading order; handles two-column gazette layout."""
    if len(blocks) < 4 or not is_landscape:
        return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))

    # Landscape gazette pages use left/right columns.
    midpoint = page_width / 2

    left = []
    right = []
    for b in blocks:
        center_x = (b["x0"] + b["x1"]) / 2
        if center_x < midpoint:
            left.append(b)
        else:
            right.append(b)

    # Wide blocks span >60% of page width (full-width headers/titles)
    wide_blocks = [b for b in blocks if (b["x1"] - b["x0"]) > page_width * 0.6]

    # Two-column: both halves populated, <30% full-width blocks
    if len(left) >= 3 and len(right) >= 3 and len(wide_blocks) < len(blocks) * 0.3:
        left_sorted = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right_sorted = sorted(right, key=lambda b: (b["y0"], b["x0"]))
        return left_sorted + right_sorted

    # Fallback: sort by position
    return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))


def _extract_page_text(page) -> str:
    """Extract and column-sort text from a single PyMuPDF page."""
    page_dict = page.get_text("dict")
    page_width = page_dict.get("width", 595) # A4 default
    raw_blocks = page_dict.get("blocks", [])

    text_blocks = []
    for b in raw_blocks:
        if b.get("type") != 0: # 0 = text, 1 = image
            continue
        block_text = ""
        for line in b.get("lines", []):
            line_text = "".join(span["text"] for span in line.get("spans", []))
            block_text += line_text + "\n"
        if block_text.strip():
            text_blocks.append({
                "x0": b["bbox"][0], "y0": b["bbox"][1],
                "x1": b["bbox"][2], "y1": b["bbox"][3],
                "text": block_text,
            })

    if not text_blocks:
        return ""

    page_height = page_dict.get("height", 842)
    is_landscape = page_width > page_height

    ordered = _detect_two_columns(text_blocks, page_width, is_landscape=is_landscape)
    return "".join(b["text"] for b in ordered)


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract raw text from every page of a PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = _extract_page_text(page)
        pages.append({
            "page_num": i + 1, # 1-indexed
            "raw_text": text,
        })
    doc.close()
    return pages

def clean_page_text(text: str) -> str:
    """Remove common noise from Indonesian legal PDF text."""
    # OCR corruptions of "PRESIDEN REPUBLIK INDONESIA" (multi-line variant)
    text = re.sub(
        r'(?:SALINAN\s*)?(?:Menimbang\s*)?'
        r'(?:FRESIDEN|PRESIDEN|PNESIDEN|FTjTJTFiTIilNEEtrtrEIn!)\s*\n'
        r'\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|NEPUBUK)\s+'
        r'(?:TNDONESIA|INDONESIA|INDONESI,A)',
        '', text
    )
    # Standalone garbage strings from font-encoding errors
    text = re.sub(r'(?:LIrtrEIEtrN|iIitrEIEtrN|;?\*trEIEtrN|FTjTJTFiTIilNEEtrtrEIn!)', '', text)
    # Single-line PRESIDEN variants
    text = re.sub(r'^\s*(?:FRESIDEN|PRESIDEN|PNESIDEN|PRESTDEN)\s*$', '', text, flags=re.MULTILINE)
    # Single-line REPUBLIK INDONESIA variants
    text = re.sub(r'^\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|NEPUBUK)\s+(?:TNDONESIA|INDONESIA|INDONESI,A)\s*$', '', text, flags=re.MULTILINE)

    # Additional OCR-corrupted header variants (partial matches, truncated)
    text = re.sub(r'^\s*(?:R,EPUBLIK|REPIJBUK|REPI,IBLIK|REP[A-Z]*K)\s+(?:INDONES[A-Z,]*)\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:EUK|ELIK|BUK)\s+INDONESIA\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:PTTESIDEN|PRESTDEN)\s*$', '', text, flags=re.MULTILINE)

    # Remove page number markers like "-2-", "-3-", "- 10 -", "-28-", "-t2-", "-t4-"
    text = re.sub(r'\n\s*-\s*[t]?\d+\s*-?\s*\n', '\n', text)

    # Remove SK No footer like "SK No 273836A", "SK No248816A"
    text = re.sub(r'SK\s+No\s*\d+\s*A.*$', '', text, flags=re.MULTILINE)

    # Split glued headings: "diperolehPasal 2" → "diperoleh\nPasal 2"
    text = re.sub(r'([^\s])(?=Pasal\s+\d)', r'\1\n', text)
    # No-space variant: "Pasal22" → "Pasal 22"
    text = re.sub(r'^(Pasal)(\d)', r'\1 \2', text, flags=re.MULTILINE)

    # Split glued BAB headings similarly
    text = re.sub(r'([^\n])(?=BAB\s+[IVXLCDM])', r'\1\n', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix font-encoding OCR artifacts (O→0, l→1, I→1) in headings
    text = fix_ocr_artifacts(text)

    return text.strip()

def fix_ocr_artifacts(text: str) -> str:
    """Fix OCR artifacts in ayat numbering (O→0, l/I→1) and strip page-continuation noise."""
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.strip()

        # Fix misread ayat numbers inside closed parens: "(2l)" -> "(21)", "(l)" -> "(1)"
        line = re.sub(
            r'\(([0-9OlI]+)\)',
            lambda m: '(' + _normalize_ocr_digits(m.group(1)) + ')',
            line
        )

        # Fix malformed parens (missing/replaced closing paren): "(2t" -> "(2)"
        line = re.sub(r'\((\d+)[t]\b', lambda m: '(' + m.group(1) + ')', line)
        line = re.sub(r'\b[tl](\d+)[tl]\b', lambda m: '(' + m.group(1) + ')', line)
        line = re.sub(r'\(s\)', '(5)', line)

        # Remove page continuation markers: "Pasal 7...", "(3)DBH . , ."
        line = re.sub(r'^Pasa[lr]\s*\d+\s*\.{2,}\s*$', '', stripped and line or line)
        line = re.sub(r'\.\s*\.\s*\.\s*$', '', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def _normalize_ocr_digits(s: str) -> str:
    """Normalize OCR-misread digits in a string (O→0, l→1, I→1)."""
    # Remove spaces (handles "9 I" → "9I")
    s_nospace = s.replace(' ', '')
    normalized = ''
    for ch in s_nospace:
        if ch == 'O':
            normalized += '0'
        elif ch == 'l':
            normalized += '1'
        elif ch == 'I':
            normalized += '1'
        else:
            normalized += ch

    # Verify the result is actually numeric
    if normalized.isdigit():
        return normalized
    return s  # non-numeric, return original

def _parse_pasal_number(raw: str) -> tuple[str | None, str]:
    """Parse a raw Pasal number string into (number, suffix), handling OCR artifacts."""
    # Remove spaces (handles "9 I" case)
    s = raw.replace(' ', '')

    if not s:
        return None, ""

    # Check if last character is a letter (potential suffix)
    if s[-1].isalpha():
        last_char = s[-1].upper()
        body = s[:-1]

        # Try interpreting WITHOUT the last char as suffix
        num_without = _normalize_ocr_digits(body)

        # Try interpreting WITH the last char as part of the number
        num_with = _normalize_ocr_digits(s)

        if num_without.isdigit() and num_with.isdigit():
            # Both interpretations are valid numbers.
            # 'O' at end is almost always OCR'd '0' (suffix O doesn't exist in Indonesian law)
            # 'l' at end is almost always OCR'd '1' (suffix uses uppercase only)
            if last_char == 'O' or s[-1] == 'l':
                # Always treat as OCR'd digit
                return num_with, ""
            elif last_char == 'I' and len(num_without) >= 3:
                # "119I" → suffix I (Pasal 119I exists in UU perubahan)
                # but "19I" → OCR for 191, "9I" → OCR for 91
                return num_without, last_char
            else:
                # "9I" is most likely OCR for 91
                return num_with, ""
        elif num_without.isdigit():
            # Only works as number+suffix: "599A" → 599 + A
            return num_without, last_char
        elif num_with.isdigit():
            # Only works with last char as digit: "4O" → 40
            return num_with, ""
        else:
            return None, ""  # Neither interpretation works
    else:
        # No trailing letter, just normalize digits
        normalized = _normalize_ocr_digits(s)
        if normalized.isdigit():
            return normalized, ""
        return None, ""


# ============================================================
# 2. PENJELASAN (EXPLANATION SECTION) DETECTION
# ============================================================

def find_penjelasan_page(pages: list[dict]) -> int | None:
    """Return the page number where PENJELASAN starts, or None if not found."""
    for page in pages:
        text = page["raw_text"]  # use raw text since PENJELASAN is usually clean
        if re.search(r'PENJ\S*SAN\s*\n\s*ATAS', text):
            return page["page_num"]
    return None

def find_closing_page(pages: list[dict]) -> int | None:
    """Return the closing/pengesahan page number, or None if not found."""
    for page in pages:
        text = page["raw_text"]
        if re.search(r'Di(?:sahkan|tetapkan) di Jakarta|Agar setiap orang mengetahuinya', text):
            return page["page_num"]
    return None

def detect_perubahan(pages: list[dict]) -> bool:
    """Return True if the document is a Perubahan (amendment) UU/PP/Perpres."""
    if not pages:
        return False

    # Note: We don't use Roman Pasal detection because PyMuPDF often renders
    # "Pasal 1" as "Pasal I" due to font encoding, causing false positives.

    # Scan first 3 pages for the title (handles garbled page order)
    for page in pages[:3]:
        text = page["raw_text"]
        # Use \s* after TENTANG to match OCR without spaces
        title_m = re.search(r'TENTANG\s*(.+?)DENGAN\s+RAHMAT', text,
                            re.DOTALL | re.IGNORECASE)
        if not title_m:
            continue

        title_text = title_m.group(1)

        # Signal 1: explicit "PERUBAHAN" or "PENYESUAIAN"
        if re.search(r'PERUBAHAN|YESUAIAN', title_text, re.IGNORECASE):
            return True

        # Signal 2: "ATAS UNDANG-UNDANG" / "ATAS PERATURAN" pattern.
        # Catches OCR drops where "PERUBAHAN" was lost.
        if re.search(r'ATAS\s+UNDANG|ATAS\s+PERATURAN', title_text, re.IGNORECASE):
            return True

    return False

def detect_omnibus(pages: list[dict], elements: list[dict]) -> bool:
    """Return True if the document is an omnibus law (e.g. Cipta Kerja) where Pasal validation is skipped."""
    # Check title (between TENTANG and DENGAN RAHMAT) for omnibus keywords.
    # Must be in title, not in preamble references to other laws.
    for page in pages[:3]:
        title_m = re.search(r'TENTANG\s*(.+?)DENGAN\s+RAHMAT', page["raw_text"],
                            re.DOTALL | re.IGNORECASE)
        if title_m and re.search(r'CIPTA\s*KERJA', title_m.group(1), re.IGNORECASE):
            return True
    # Heuristic: >500 Pasals is likely omnibus
    pasal_count = sum(1 for e in elements if e["type"] == "pasal")
    return pasal_count > 500

# Contract: pages items must include page_num/raw_text/clean_text.
# Returns {"umum": str, "pasal": {pasal_number: explanation_text}}.
def parse_penjelasan(pages: list[dict], penjelasan_page: int,
                     total_pages: int) -> dict:
    """Parse the PENJELASAN section and return {"umum": str, "pasal": {number: text}}."""
    # Extract all PENJELASAN text from cleaned pages
    parts = []
    for page in pages:
        if page["page_num"] < penjelasan_page:
            continue
        if page["page_num"] > total_pages:
            break
        parts.append(page["clean_text"])
    full_text = "\n\n".join(parts)

    # Split into UMUM and PASAL DEMI PASAL sections.
    # "II." prefix is optional; some shorter UUs omit it or OCR drops it.
    # First space is \s* not \s+ because OCR sometimes merges "PASALDEMI".
    split_m = re.split(r'(?:II\.?\s*|[iI][lI1]\.?\s*)?PASAL\s*DEMI\s+PASAL', full_text, maxsplit=1,
                       flags=re.IGNORECASE)

    if len(split_m) == 2:
        umum_raw, pasal_section = split_m
    else:
        # No "PASAL DEMI PASAL" found; entire text is umum
        return {"umum": _clean_penjelasan_text(full_text), "pasal": {}}

    # Clean UMUM: strip the PENJELASAN header lines
    umum_text = re.sub(
        r'^.*?I\.\s*UMUM\s*', '', umum_raw, count=1,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()
    umum_text = _clean_penjelasan_text(umum_text)

    # Pre-process: fix OCR column artifacts in PASAL DEMI PASAL section.
    # PyMuPDF sometimes reads compact multi-column Pasals vertically:
    #   "Pasal\nPasal\nPasal\n51\nCukup jelas.\n52\nCukup jelas."
    # Fix these into "Pasal 51\nCukup jelas.\nPasal 52\nCukup jelas."
    pasal_section = _fix_penjelasan_columns(pasal_section)

    # Normalize OCR in Pasal numbers: "Pasal l0" → "Pasal 10", "Pasal 5O" → "Pasal 50"
    # In PENJELASAN context, also treat uppercase L as OCR'd 1 (no valid suffix L)
    def _normalize_penjelasan_pasal(m):
        num = m.group(2).replace('L', '1')
        return m.group(1) + _normalize_ocr_digits(num)

    pasal_section = re.sub(
        r'^(Pasa[l1]\s+)([0-9OlIL][0-9A-Za-z]*)\s*$',
        _normalize_penjelasan_pasal,
        pasal_section, flags=re.MULTILINE
    )

    # Split at each "Pasal X" heading (Arabic or Roman numerals)
    pasal_splits = re.split(
        r'^Pasa[l1]\s+(\d+[A-Z]?|[IVXLC]+)\s*$',
        pasal_section, flags=re.MULTILINE
    )
    # pasal_splits = [text_before_first_pasal, "1", explanation1, "2", explanation2, ...]

    pasal_dict = {}
    # Start from index 1 (skip text before first Pasal heading)
    i = 1
    while i + 1 < len(pasal_splits):
        pasal_num = pasal_splits[i].strip()
        explanation = pasal_splits[i + 1].strip()

        # Clean the explanation text
        explanation = _clean_penjelasan_text(explanation)

        # Empty explanation = implicitly "Cukup jelas."
        if not explanation:
            explanation = "Cukup jelas."

        pasal_dict[pasal_num] = explanation
        i += 2

    return {"umum": umum_text, "pasal": pasal_dict}

def _fix_penjelasan_columns(text: str) -> str:
    """Rebuild vertically-read OCR columns in PASAL DEMI PASAL.
    Converts stacked "Pasal\\nPasal\\n51\\n52" into "Pasal 51\\nPasal 52"."""
    def _consume_stacked_pasal(lines: list[str], start_idx: int) -> tuple[list[str] | None, int]:
        """Consume a stacked "Pasal" block and return rebuilt lines + next index."""
        pasal_count = 0
        j = start_idx
        while j < len(lines) and re.match(r'^Pasa[l1]\s*$', lines[j].strip()):
            pasal_count += 1
            j += 1

        collected: list[tuple[str, int]] = []
        k = j
        while k < len(lines) and len(collected) < pasal_count:
            num_line = lines[k].strip()
            num_m = re.match(r'^(\d+[A-Z]?)\s*$', num_line)
            if num_m:
                collected.append((num_m.group(1), k))
                k += 1
            elif collected:
                k += 1
            else:
                break

        if not collected:
            return None, start_idx + 1

        rebuilt: list[str] = []
        for idx, (num, start_k) in enumerate(collected):
            rebuilt.append(f"Pasal {num}")
            end_k = collected[idx + 1][1] if idx + 1 < len(collected) else k
            for exp_line_idx in range(start_k + 1, end_k):
                rebuilt.append(lines[exp_line_idx])
        return rebuilt, k

    def _consume_bare_number_sequence(lines: list[str], start_idx: int) -> tuple[list[str] | None, int]:
        """Consume bare-number Pasal continuation block and return rebuilt lines + next index."""
        j = start_idx
        bare_entries: list[tuple[str, int]] = []
        while j < len(lines):
            num_m = re.match(r'^(\d+)\s*$', lines[j].strip())
            if num_m:
                bare_entries.append((num_m.group(1), j))
                j += 1
            elif lines[j].strip().lower().startswith('cukup jelas') and bare_entries:
                j += 1
            elif bare_entries and not lines[j].strip():
                j += 1
            else:
                break

        if len(bare_entries) < 2:
            return None, start_idx + 1

        rebuilt: list[str] = []
        for idx, (num, start_j) in enumerate(bare_entries):
            rebuilt.append(f"Pasal {num}")
            end_j = bare_entries[idx + 1][1] if idx + 1 < len(bare_entries) else j
            for exp_idx in range(start_j + 1, end_j):
                rebuilt.append(lines[exp_idx])
        return rebuilt, j

    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        # Detect stacked "Pasal" lines (N consecutive lines that are just "Pasal")
        if re.match(r'^Pasa[l1]\s*$', stripped):
            rebuilt, next_i = _consume_stacked_pasal(lines, i)
            if rebuilt is not None:
                result.extend(rebuilt)
                i = next_i
                continue
            result.append(lines[i])
            i = next_i
            continue

        # Detect bare numbers on their own line (page-break continuation)
        # Only treat as Pasal if followed by "Cukup jelas." or another bare number
        elif re.match(r'^(\d+)\s*$', stripped):
            rebuilt, next_i = _consume_bare_number_sequence(lines, i)
            if rebuilt is not None:
                result.extend(rebuilt)
                i = next_i
                continue

        result.append(lines[i])
        i += 1

    return '\n'.join(result)

def _clean_penjelasan_text(text: str) -> str:
    """Clean noise from PENJELASAN text (headers, page markers, trailing metadata)."""
    # Remove PRESIDEN REPUBLIK INDONESIA headers (same OCR variants as body)
    _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN|ETIiEILtrN|FTjTJTFiTIilNEEtrtrEIn!|FTIESIDEN)'
    text = re.sub(r'\n?\s*' + _PRESIDEN_RE + r'\s*\n\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|REPUBTJK|NEPUBUK|REPI,IBLIK)\s+(?:TNDONESIA|INDONESIA|INDONESI,?A)\s*\n?', '\n', text)
    text = re.sub(r'^\s*' + _PRESIDEN_RE + r'\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|REPUBTJK|NEPUBUK|REPI,IBLIK)\s+(?:TNDONESIA|INDONESIA)\s*$', '', text, flags=re.MULTILINE)
    # Remove page number markers: "-5-", "- 10 -", "-t2-"
    text = re.sub(r'^\s*-\s*[t]?\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    # Remove SK No footer
    text = re.sub(r'SK\s+No\s*\d+\s*A.*$', '', text, flags=re.MULTILINE)
    # Remove trailing "TAMBAHAN LEMBARAN NEGARA..."
    text = re.sub(r'\n\s*TAMBAHAN\s+LEMBARAN\s+NEGARA.*$', '', text, flags=re.DOTALL)
    # Remove continuation markers: "Pasal 3...", "Huruf b. . ."
    text = re.sub(r'^\w[\w\s]*\.\s*\.\s*\.?\s*$', '', text, flags=re.MULTILINE)
    # Remove stacked standalone "Pasal" or "Ayat" lines (OCR column artifacts).
    # In valid text, "Pasal" is always followed by a number and "Ayat" by "(N)",
    # so bare standalone lines are always noise from column-reading.
    text = re.sub(r'(?:^(?:Pasal|Pasa1|Ayat)\s*\n){2,}', '', text, flags=re.MULTILINE)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def attach_penjelasan(nodes: list[dict], pasal_dict: dict[str, str]):
    """Attach per-Pasal penjelasan text to matching leaf nodes in the tree."""
    for node in iter_leaves(nodes):
        if "text" not in node:
            continue
        # Leaf node: match Pasal number from title (e.g. "Pasal 5" → "5", "Pasal 119I" → "119I")
        m = re.match(r'Pasal\s+(.+)', node.get("title", ""))
        if m:
            pasal_key = m.group(1).strip()
            # Try exact match first, then just the numeric part
            if pasal_key in pasal_dict:
                node["penjelasan"] = pasal_dict[pasal_key]
            else:
                # Try numeric-only match (strip suffix)
                num_m = re.match(r'(\d+)', pasal_key)
                if num_m and num_m.group(1) in pasal_dict:
                    node["penjelasan"] = pasal_dict[num_m.group(1)]
                else:
                    node["penjelasan"] = None
        # Non-Pasal leaf nodes (Pembukaan, etc.) don't get penjelasan

# ============================================================
# 3. STRUCTURAL ELEMENT DETECTION
# ============================================================

# Roman numerals with optional trailing letter (e.g. BAB VIIA in amendments).
ROMAN_NUMERAL = r'[IVXLCDM]+[A-Z]?'

PATTERNS = {
    "bab": re.compile(
        r'^BAB\s+(' + ROMAN_NUMERAL + r')\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE
    ),
    "bagian": re.compile(
        # Tolerates OCR typos in "Bagian" (Bagtan, Brgian, etc.); ordinal name anchors the match.
        r'^B[a-z]{1,5}an\s+'
        r'(Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|Kedelapan|'
        r'Kesembilan|Kesepuluh|Kesebelas|Kedua\s*belas|Ketiga\s*belas|'
        r'Keempat\s*belas|Kelima\s*belas|Keenam\s*belas|Ketujuh\s*belas|'
        r'Kedelapan\s*belas|Kesembilan\s*belas|Kedua\s*puluh)\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE | re.IGNORECASE
    ),
    "paragraf": re.compile(
        # Handle both arabic (1, 2, 3) and roman (I, II, III) numbering
        r'^Paragraf\s+(\d+|' + ROMAN_NUMERAL + r')\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE | re.IGNORECASE
    ),
    "pasal": re.compile(
        # Handles OCR variants ("Pasa1", "Pasal 4O", "Pasal 119I").
        # Negative lookahead prevents matching cross-references (ayat/huruf/angka on next line).
        r'^[Pp]asa[l1]\s+([0-9OlI][0-9A-Za-z \t]*?)\s*(?:\.\s*\.\s*[.\'])?$'
        r'(?:\n(?!ayat|huruf|angka|dan Pasal|sampai dengan|jo\.?\s)|\Z)',
        re.MULTILINE
    ),
}

# Preceding-line words that make "Pasal X" a cross-reference, not a heading.
_CROSS_REF_PRECEDING = frozenset({
    'dalam', 'pada', 'oleh', 'dari', 'dan', 'atau', 'dengan',
    'sebagaimana', 'berdasarkan', 'terhadap', 'menurut', 'tentang',
    'atas', 'bahwa', 'melalui', 'untuk', 'antara',
})

# Level mapping for hierarchy.
# For Perubahan (amendment) UUs:
#   Pasal I (roman, 0) > Angka items (1) > BAB (2) > Bagian (3) > Paragraf (4) > Pasal (5)
# For normal UUs: BAB (2) is the root; levels 0-1 are unused.
LEVEL_MAP = {
    "pasal_roman": 0,
    "angka": 1,
    "bab": 2,
    "bagian": 3,
    "paragraf": 4,
    "pasal": 5,
}

def roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral string to integer. Returns None if invalid."""
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    if not s or not all(c in values for c in s.upper()):
        return None
    s = s.upper()
    total = 0
    for i, c in enumerate(s):
        if i + 1 < len(s) and values[c] < values[s[i + 1]]:
            total -= values[c]  # subtractive notation (IV=4, IX=9)
        else:
            total += values[c]
    return total

def _clean_heading_title(title: str) -> str:
    """Clean OCR artifacts from heading titles (BAB, Bagian, Paragraf)."""
    # Remove header bleed: "Ganti RugiPRESIDEN" → "Ganti Rugi"
    title = re.sub(r'(?:PRESIDEN|FRESIDEN|PNESIDEN)(?:\s*REPUBLIK\s*INDONESIA)?', '', title)
    # Fix common OCR artifacts in heading text
    title = re.sub(r'Pe\{anjian', 'Perjanjian', title)
    title = re.sub(r'Pertanggungi\s*awaban', 'Pertanggungjawaban', title)
    return title.strip()

# Roman numeral Pasal headings in Perubahan UUs (Pasal I, II, III, ...).
PASAL_ROMAN = re.compile(
    r'^[Pp]asa[l1]\s+([IVXLC]+)\s*$',
    re.MULTILINE
)

# Numbered amendment instructions in Perubahan UUs, e.g. "76. Ketentuan Pasal 88 diubah..."
# Keyword anchor (Ketentuan/Pasal/Bab/Bagian) prevents false matches on regular list items.
ANGKA_PATTERN = re.compile(
    r'^(\d+)\.\s*'                              # number + period at start of line
    r'((?:Ketentuan\s+)?'                       # optional "Ketentuan "
    r'(?:Pasal|Bagian|Bab|Di\s+antara)\b'       # instruction keyword
    r'[^\n]*)',                                 # rest of first line
    re.MULTILINE
)

def _detect_page_elements(text: str, page_num: int, is_perubahan: bool) -> list[dict]:
    """Detect raw structural elements from one page and return unsorted items."""
    page_elements: list[dict] = []

    if is_perubahan:
        # Perubahan UUs: detect Roman numeral Pasal as root-level containers
        # (Pasal I, II, III, ...) at level 0 (above BAB).
        for m in PASAL_ROMAN.finditer(text):
            roman_str = m.group(1).strip()
            arabic_val = roman_to_int(roman_str)
            if arabic_val is None:
                continue
            page_elements.append({
                "type": "pasal_roman",
                "level": LEVEL_MAP["pasal_roman"],
                "number": roman_str,
                "title": f"Pasal {roman_str}",
                "page_num": page_num,
                "char_offset": m.start(),
            })

        # Detect Angka (numbered amendment instructions) inside Pasal I.
        # These are the primary structural items: "1. Ketentuan Pasal 1 diubah..."
        for m in ANGKA_PATTERN.finditer(text):
            angka_num = m.group(1)
            instruction = m.group(2).strip()
            title_text = instruction[:80] + ("..." if len(instruction) > 80 else "")
            page_elements.append({
                "type": "angka",
                "level": LEVEL_MAP["angka"],
                "number": angka_num,
                "title": f"Angka {angka_num} — {title_text}",
                "page_num": page_num,
                "char_offset": m.start(),
            })

    # Normal element detection (runs for both regular and perubahan UUs)
    for m in PATTERNS["bab"].finditer(text):
        heading_text = _clean_heading_title(m.group(2).strip())
        page_elements.append({
            "type": "bab",
            "level": LEVEL_MAP["bab"],
            "number": m.group(1).strip(),
            "title": f"BAB {m.group(1).strip()} - {heading_text}",
            "page_num": page_num,
            "char_offset": m.start(),
        })

    for m in PATTERNS["bagian"].finditer(text):
        heading_text = _clean_heading_title(m.group(2).strip())
        page_elements.append({
            "type": "bagian",
            "level": LEVEL_MAP["bagian"],
            "number": m.group(1).strip(),
            "title": f"Bagian {m.group(1).strip()} - {heading_text}",
            "page_num": page_num,
            "char_offset": m.start(),
        })

    for m in PATTERNS["paragraf"].finditer(text):
        raw_num = m.group(1).strip()
        roman_val = roman_to_int(raw_num)
        num = str(roman_val) if roman_val is not None else raw_num
        heading_text = _clean_heading_title(m.group(2).strip())
        page_elements.append({
            "type": "paragraf",
            "level": LEVEL_MAP["paragraf"],
            "number": num,
            "title": f"Paragraf {num} - {heading_text}",
            "page_num": page_num,
            "char_offset": m.start(),
        })

    for m in PATTERNS["pasal"].finditer(text):
        raw = m.group(1).strip()
        pasal_num, suffix = _parse_pasal_number(raw)
        if pasal_num is None:
            continue
        # Skip cross-references in body text (not true section headings).
        preceding = text[:m.start()].rstrip()
        if preceding:
            last_word = preceding.split()[-1].lower().rstrip('.,;:)')
            if last_word in _CROSS_REF_PRECEDING:
                continue
        page_elements.append({
            "type": "pasal",
            "level": LEVEL_MAP["pasal"],
            "number": f"{pasal_num}{suffix}",
            "title": f"Pasal {pasal_num}{suffix}",
            "page_num": page_num,
            "char_offset": m.start(),
        })

    return page_elements

def _dedupe_detected_elements(elements: list[dict]) -> list[dict]:
    """Deduplicate and filter sorted elements while preserving original rules."""
    deduped = []
    roman_positions = {(e["page_num"], e["char_offset"])
                       for e in elements if e["type"] == "pasal_roman"}
    first_roman = next(((e["page_num"], e["char_offset"])
                        for e in elements if e["type"] == "pasal_roman"), None)

    for elem in elements:
        if elem["type"] == "angka" and first_roman:
            if (elem["page_num"], elem["char_offset"]) < first_roman:
                continue
        if elem["type"] == "pasal" and (elem["page_num"], elem["char_offset"]) in roman_positions:
            continue
        if deduped and elem["type"] == "pasal" and deduped[-1]["type"] == "pasal":
            if (elem["number"] == deduped[-1]["number"]
                    and elem["page_num"] - deduped[-1]["page_num"] <= 1):
                continue
        deduped.append(elem)

    return deduped

# Contract: reads pages[*]["clean_text"], returns flat sorted element dicts.
# Element keys are stable: type, level, number, title, page_num, char_offset.
def detect_elements(pages: list[dict], body_end_page: int,
                    is_perubahan: bool = False) -> list[dict]:
    """Scan pages and return a flat list of structural elements (BAB, Bagian, Paragraf, Pasal) sorted by position."""
    elements = []

    for page in pages:
        # Skip pages after body ends (PENJELASAN, Lampiran, etc.)
        if page["page_num"] > body_end_page:
            continue
        text = page["clean_text"]
        page_num = page["page_num"]
        elements.extend(_detect_page_elements(text, page_num, is_perubahan))

    # Sort by page_num, then char_offset
    elements.sort(key=lambda e: (e["page_num"], e["char_offset"]))
    return _dedupe_detected_elements(elements)

# ============================================================
# 4. PASAL NUMBERING VALIDATION
# ============================================================

def validate_pasal_sequence(elements: list[dict],
                           is_perubahan: bool = False) -> list[str]:
    """Check Pasal sequence for gaps and reversals. Returns a list of warning strings."""
    if is_perubahan:
        return []  # Numbering is inherently non-monotonic in Perubahan UUs

    warnings = []
    last_pasal_num = 0

    for elem in elements:
        if elem["type"] != "pasal":
            continue
        try:
            num = int(re.match(r'(\d+)', elem["number"]).group(1))
        except (ValueError, AttributeError):
            continue

        if num < last_pasal_num:
            warnings.append(
                f"WARNING: Pasal {num} appears after Pasal {last_pasal_num} "
                f"(page {elem['page_num']}) — possible OCR error or PENJELASAN leak"
            )
        # Gap > 5 is unusual; smaller gaps (e.g. 2–3) are normal for sub-laws.
        elif num > last_pasal_num + 5:
            warnings.append(
                f"WARNING: Gap in Pasal numbering: {last_pasal_num} -> {num} "
                f"(page {elem['page_num']}) — possible missed Pasal"
            )
        last_pasal_num = num

    return warnings

# ============================================================
# 5. TREE BUILDING
# ============================================================

def assign_page_boundaries(elements: list[dict], total_pages: int):
    """Set start_index, end_index, and end_char_offset on each element in-place."""
    for i, elem in enumerate(elements):
        next_page = total_pages
        for j in range(i + 1, len(elements)):
            if elements[j]["level"] <= elem["level"]:
                next_page = elements[j]["page_num"]
                break
        elem["start_index"] = elem["page_num"]
        elem["end_index"] = next_page

        # Find the next element on the same end page to set end_char_offset
        elem["end_char_offset"] = None  # None means "to end of page"
        for j in range(i + 1, len(elements)):
            if elements[j]["page_num"] == elem["end_index"]:
                # Next element starts on the same end page; slice at its offset
                elem["end_char_offset"] = elements[j]["char_offset"]
                break
            elif elements[j]["page_num"] > elem["end_index"]:
                break

# Contract: accepts detect_elements() output and returns nested nodes.
# Node keys are stable: title, type, number, page bounds, node_id, nodes.
def build_tree(elements: list[dict], total_pages: int) -> list[dict]:
    """Convert a flat element list into a nested BAB > Bagian > Pasal tree."""
    if not elements:
        return []

    assign_page_boundaries(elements, total_pages)

    # Build nested tree using stack
    root_nodes = []
    stack = []  # stack of (level, node_dict)
    node_counter = 0

    for elem in elements:
        node = {
            "title": elem["title"],
            "type": elem["type"],
            "number": elem["number"],
            "start_index": elem["start_index"],
            "end_index": elem["end_index"],
            "start_char_offset": elem.get("char_offset", 0),
            "end_char_offset": None,  # will be set below
            "node_id": f"{node_counter:04d}",
            "nodes": [],
        }
        node_counter += 1

        # Pop stack until we find a parent (lower level number = higher in hierarchy)
        while stack and stack[-1][0] >= elem["level"]:
            stack.pop()

        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            root_nodes.append(node)

        stack.append((elem["level"], node))

    return root_nodes

def fix_node_boundaries(nodes: list[dict], parent_end: int, parent_end_char_offset: int | None = None):
    """Tighten node end_index and end_char_offset so siblings don't overlap.
    Last child inherits parent's end_char_offset to prevent text bleed."""
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            next_node = nodes[i + 1]
            next_start = next_node["start_index"]
            # Check before overwriting end_index
            old_end = node["end_index"]
            node["end_index"] = next_start
            if next_start == old_end:
                node["end_char_offset"] = next_node.get("start_char_offset", 0)
        else:
            # Pass parent's end_char_offset to last child to prevent text bleed.
            node["end_index"] = parent_end
            node["end_char_offset"] = parent_end_char_offset

        if node["nodes"]:
            fix_node_boundaries(node["nodes"], node["end_index"], node.get("end_char_offset"))
            child_max = max(c["end_index"] for c in node["nodes"])
            node["end_index"] = max(node["end_index"], child_max)

        node["end_index"] = max(node["end_index"], node["start_index"])

def consolidate_bab_in_perubahan(tree: list[dict]) -> int:
    """Move orphaned Pasals from the next Angka sibling under their BAB node.
    Handles amendment UUs where a new BAB and its Pasals land in separate Angkas.
    Mutates tree in place. Returns count of Pasals moved."""
    moved_count = 0

    for root_node in tree:
        if root_node.get("type") != "pasal_roman":
            continue
        children = root_node.get("nodes", [])
        if not children:
            continue

        indices_to_remove = []
        i = 0
        while i < len(children):
            angka = children[i]
            if angka.get("type") != "angka":
                i += 1
                continue

            # Find Angka whose ONLY child is a BAB leaf (no Pasal siblings).
            # This means the amendment inserted a BAB heading whose Pasals
            # are in the next Angka sibling. Excludes BAB renames where the
            # Angka also contains Pasals (e.g., Angka 89 has Pasal 108 + BAB IX).
            angka_children = angka.get("nodes", [])
            has_pasal = any(c.get("type") == "pasal" for c in angka_children)
            if has_pasal:
                i += 1
                continue
            bab_node = None
            for child in angka_children:
                if child.get("type") == "bab" and not child.get("nodes"):
                    bab_node = child
                    break
            if bab_node is None:
                i += 1
                continue

            # Look at next sibling Angka
            if i + 1 >= len(children):
                i += 1
                continue
            next_angka = children[i + 1]
            if next_angka.get("type") != "angka":
                i += 1
                continue

            # Collect Pasal children from next Angka
            pasal_children = [n for n in next_angka.get("nodes", [])
                              if n.get("type") == "pasal"]
            if not pasal_children:
                i += 1
                continue

            # Move Pasals under the BAB
            for pasal in pasal_children:
                pasal["_moved_from_angka"] = next_angka.get("number")
            bab_node["nodes"] = pasal_children
            moved_count += len(pasal_children)

            # Update BAB boundaries to encompass moved children
            bab_node["end_index"] = max(
                bab_node["end_index"],
                max(p["end_index"] for p in pasal_children),
            )
            angka["end_index"] = max(angka["end_index"], bab_node["end_index"])

            # Remove or trim the absorbed Angka
            remaining = [n for n in next_angka.get("nodes", [])
                         if n.get("type") != "pasal"]
            if remaining:
                next_angka["nodes"] = remaining
            else:
                indices_to_remove.append(i + 1)

            i += 2  # skip absorbed Angka

        # Remove absorbed Angka items (reverse to preserve indices)
        for idx in reversed(indices_to_remove):
            children.pop(idx)

    return moved_count

def iter_leaves(nodes: list[dict]):
    """Yield all leaf nodes from a tree (nodes without children)."""
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            yield from iter_leaves(node["nodes"])
        else:
            yield node

def clean_tree_for_output(nodes: list[dict], pages: list[dict] | None = None) -> list[dict]:
    """Strip internal fields and embed leaf node text for PageIndex output."""
    result = []
    for node in nodes:
        clean = {
            "title": node["title"],
            "node_id": node["node_id"],
            "start_index": node["start_index"],
            "end_index": node["end_index"],
        }
        if node.get("nodes"):
            clean["nodes"] = clean_tree_for_output(node["nodes"], pages)
            # Pasal_roman nodes in perubahan UUs may have their own text before
            # their first child. Recover it by extracting up to the first child.
            if pages is not None and node.get("type") == "pasal_roman":
                first_child = node["nodes"][0]
                if first_child["start_index"] == node["start_index"]:
                    # Child starts on same page; extract up to child's char_offset
                    own_end_page = node["start_index"]
                    own_end_char = first_child.get("start_char_offset") or None
                else:
                    # Child starts on a later page; extract up to the page before it
                    own_end_page = first_child["start_index"] - 1
                    own_end_char = None
                if own_end_page >= node["start_index"]:
                    intro = _extract_node_text(
                        pages, node["start_index"], own_end_page,
                        start_char_offset=node.get("start_char_offset", 0),
                        end_char_offset=own_end_char,
                    )
                    if intro.strip():
                        clean["text"] = intro
        elif pages is not None:
            # Leaf node: embed text with intra-page slicing
            clean["text"] = _extract_node_text(
                pages, node["start_index"], node["end_index"],
                start_char_offset=node.get("start_char_offset", 0),
                end_char_offset=node.get("end_char_offset"),
            )
        # Propagate penjelasan if attached (set by attach_penjelasan)
        if "penjelasan" in node:
            clean["penjelasan"] = node["penjelasan"]
        # Propagate source_angka for Pasals moved by BAB consolidation
        if "_moved_from_angka" in node:
            clean["source_angka"] = node["_moved_from_angka"]
        result.append(clean)
    return result

def _split_preamble(text: str, start_page: int, end_page: int) -> list[dict] | None:
    """Split preamble into Menimbang, Mengingat, and Menetapkan sub-nodes; returns None on failure."""
    def _clean_preamble_noise(raw_text: str) -> str:
        """Remove common OCR margin/header noise while preserving split keywords."""
        _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN)'
        noise_patterns = [
            re.compile(r'^\s*Mengingat\s*\n\s*Menetapkan\s*$', re.MULTILINE),
            re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s*\n\s*REP\S*\s+IND\S*\s*\n'),
            re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s*\n'),
            re.compile(r'^\s*SAL[IT]NAN\s*\n', re.MULTILINE),
            re.compile(r'^\s*-\d+-\s*\n', re.MULTILINE),
            re.compile(r'^\s*(?!Mengingat|Menetapkan)[A-Z]{3,8}\s*\n', re.MULTILINE),
        ]
        cleaned_text = raw_text
        for pat in noise_patterns:
            cleaned_text = pat.sub('\n', cleaned_text)
        return re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()

    def _strip_mengingat_prefix(candidate: str) -> str:
        """Strip 'Mengingat' prefix from start or within first 200 chars."""
        stripped = re.sub(r'^\s*Mengingat\s*:?\s*\n?\s*', '', candidate)
        if stripped != candidate:
            return stripped
        kw_m = re.search(r'\bMengingat\b', candidate[:200])
        if kw_m:
            after_kw = candidate[kw_m.end():]
            return re.sub(r'^\s*:?\s*\n?\s*', '', after_kw)
        return candidate

    def _split_by_content_transition(body: str) -> tuple[str | None, str | None]:
        """Try splitting Menimbang/Mengingat after final bahwa item punctuation."""
        last_bahwa_end = None
        for m in re.finditer(r'bahwa\s', body):
            last_bahwa_end = m.end()
        if not last_bahwa_end:
            return None, None

        after_last = body[last_bahwa_end:]
        for semi_m in re.finditer(r';\s*\n', after_last):
            split_pos = last_bahwa_end + semi_m.end()
            remaining = body[split_pos:]
            remaining_stripped = _strip_mengingat_prefix(remaining)
            if re.match(r'\s*[2-9]\d*[\.\s]+', remaining_stripped):
                continue
            if re.match(r'\s*(?:\d+[\.\s]+)?(?:Pasal|Undang|Peraturan)', remaining_stripped):
                return body[:split_pos].strip(), remaining_stripped.strip()

        for period_m in re.finditer(r'\.\s*\n', after_last):
            split_pos = last_bahwa_end + period_m.end()
            remaining = body[split_pos:]
            remaining_stripped = _strip_mengingat_prefix(remaining)
            if re.match(r'\s*1[\.\s]+(?:Pasal|Undang|Peraturan)', remaining_stripped):
                return body[:split_pos].strip(), remaining_stripped.strip()

        return None, None

    def _split_by_mengingat_keyword(body: str) -> tuple[str | None, str | None]:
        """Fallback split using explicit Mengingat heading."""
        mengingat_kw = re.search(r'\n\s*Mengingat\s*:?\s*\n', body)
        if not mengingat_kw:
            return None, None
        return body[:mengingat_kw.start()].strip(), body[mengingat_kw.end():].strip()

    def _split_by_first_pasal_ref(body: str) -> tuple[str | None, str | None]:
        """Last-resort split using first legal reference line."""
        mengingat_m = re.search(r'\n((?:\d+\.\s+)?Pasal\s+\d+)', body)
        if not mengingat_m:
            return None, None
        return body[:mengingat_m.start()].strip(), body[mengingat_m.start():].strip()

    cleaned = _clean_preamble_noise(text)

    # Step 2: Find Menimbang start
    menimbang_m = re.search(r'Menimbang\s*:?\s*(?:\n|(?=a[\.\s]))', cleaned)
    if menimbang_m:
        before_menimbang = cleaned[:menimbang_m.start()]
        after_menimbang = cleaned[menimbang_m.end():]
        # Some PDFs place the "Menimbang:" label block after its bahwa items due to
        # block sort order. If "a. bahwa" appears before the keyword, prepend it.
        bahwa_before_m = re.search(r'(?:^|\n)\s*a\.?\s+bahwa\s', before_menimbang)
        if bahwa_before_m:
            after_menimbang = (
                before_menimbang[bahwa_before_m.start():].lstrip('\n')
                + after_menimbang
            )
    else:
        # Fall back: find first "a. bahwa" or "a bahwa" pattern
        fallback_m = re.search(r'(?:^|\n)\s*a\.?\s+bahwa\s', cleaned)
        if not fallback_m:
            fallback_m = re.search(r'(?:^|\n)\s*a\.\s*\n\s*bahwa\s', cleaned)
        if not fallback_m:
            fallback_m = re.search(r'(?:^|\n)\s*bahwa\s', cleaned)
        if not fallback_m:
            return None
        before_menimbang = cleaned[:fallback_m.start()]
        after_menimbang = cleaned[fallback_m.start():].lstrip('\n')

    # Step 3: Find MEMUTUSKAN as end boundary and extract Menetapkan text
    memutuskan_m = re.search(r'MEMUTUS\S*\s*:', after_menimbang)
    menetapkan_text = None

    if memutuskan_m:
        preamble_body = after_menimbang[:memutuskan_m.start()].strip()
        # Text after "MEMUTUSKAN:" is the Menetapkan content
        raw_after = after_menimbang[memutuskan_m.end():].strip()
        # Strip "Menetapkan :" prefix if present
        menet_m = re.match(r'Menetapkan\s*:\s*', raw_after)
        if menet_m:
            menetapkan_text = raw_after[menet_m.end():].strip()
        elif raw_after:
            menetapkan_text = raw_after
    else:
        preamble_body = after_menimbang.strip()

    # Step 4: Split Menimbang from Mengingat by increasingly permissive strategies.
    menimbang_text, mengingat_text = _split_by_content_transition(preamble_body)
    if menimbang_text is None:
        menimbang_text, mengingat_text = _split_by_mengingat_keyword(preamble_body)
    if menimbang_text is None:
        menimbang_text, mengingat_text = _split_by_first_pasal_ref(preamble_body)

    # No split found; keep all as Menimbang
    if menimbang_text is None:
        menimbang_text = preamble_body
        mengingat_text = None

    # Step 5: Clean up and build children
    # Strip trailing "Mengingat" OCR keyword from Menimbang: margin bleed can
    # leave a lone "Mengingat" at the very end of the section.
    menimbang_text = re.sub(r'\s*\bMengingat\b\s*$', '', menimbang_text).strip()
    menimbang_text = re.sub(r'\n{3,}', '\n\n', menimbang_text).strip()
    if not menimbang_text:
        return None

    children = [
        {
            "title": "Menimbang",
            "node_id": "P001",
            "start_index": start_page,
            "end_index": end_page,
            "text": menimbang_text,
        },
    ]

    if mengingat_text:
        # Strip leading colon/whitespace OCR artifact (e.g. ": 1. Pasal..." →
        # "1. Pasal...") when "Mengingat :" label is on the same line as ref 1.
        mengingat_text = re.sub(r'^[\s:]+', '', mengingat_text)
        # Strip trailing "Dengan Persetujuan Bersama..." boilerplate between
        # Mengingat and MEMUTUSKAN. Fuzzy match since OCR garbles 'uj' → 'qj'.
        dpr_m = re.search(r'\n[^\n]*Dengan\s+Perse\w+\s+Bersama', mengingat_text)
        if dpr_m:
            mengingat_text = mengingat_text[:dpr_m.start()]
        # Strip lone "Mengingat" OCR margin-bleed lines from within Mengingat text.
        mengingat_text = re.sub(r'^\s*Mengingat\s*$', '', mengingat_text, flags=re.MULTILINE)
        mengingat_text = re.sub(r'\n{3,}', '\n\n', mengingat_text).strip()
        if mengingat_text:
            children.append({
                "title": "Mengingat",
                "node_id": "P002",
                "start_index": start_page,
                "end_index": end_page,
                "text": mengingat_text,
            })

    if menetapkan_text:
        menetapkan_text = re.sub(r'\n{3,}', '\n\n', menetapkan_text).strip()
        if menetapkan_text:
            children.append({
                "title": "Menetapkan",
                "node_id": "P003",
                "start_index": start_page,
                "end_index": end_page,
                "text": menetapkan_text,
            })

    return children

def _extract_node_text(
    pages: list[dict], start: int, end: int,
    start_char_offset: int = 0, end_char_offset: int | None = None,
) -> str:
    """Extract and join cleaned text from pages [start, end], slicing by char offsets."""
    parts = []
    for page in pages:
        if page["page_num"] < start:
            continue
        if page["page_num"] > end:
            break
        text = page["clean_text"]

        # Slice start page from char_offset onward
        if page["page_num"] == start and start_char_offset > 0:
            text = text[start_char_offset:]

        # Slice end page up to end_char_offset
        if page["page_num"] == end and end_char_offset is not None:
            # end_char_offset is relative to the full page text
            if page["page_num"] == start and start_char_offset > 0:
                # Adjust since we already sliced the start
                adjusted = end_char_offset - start_char_offset
                if adjusted > 0:
                    text = text[:adjusted]
            else:
                text = text[:end_char_offset]

        parts.append(text.strip())
    joined = "\n\n".join(p for p in parts if p)
    # Safety net: remove any remaining header bleed-through in extracted text
    # Match OCR variants: PRESIDEN, PRESIOEN, PRESTDEN, FRESIDEN, etc.
    _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN)'
    joined = re.sub(r'\n\s*' + _PRESIDEN_RE + r'\s*\n\s*REPUBLIK INDONESIA\s*\n', '\n', joined)
    joined = re.sub(r'\n\s*' + _PRESIDEN_RE + r'\s*\n', '\n', joined)
    # Remove trailing BAB/Bagian/Paragraf headers that bleed from page breaks
    # e.g., "BAB XV" at bottom of page before the actual BAB starts on next page
    joined = re.sub(r'\n\s*BAB\s+[IVXLCDM]+\s*$', '', joined)
    joined = re.sub(r'\n\s*B[a-z]{1,5}an\s+\w+\s*$', '', joined)
    joined = re.sub(r'\n\s*Paragraf\s+\w+\s*$', '', joined)
    return joined

# ============================================================
# 6. LLM TEXT CLEANUP
# ============================================================

LLM_CLEANUP_PROMPT = """\
You are an OCR correction tool for Indonesian legal documents.
Fix OCR artifacts in the text while preserving the original meaning and structure.

Rules:
- Fix garbled characters (e.g., "tqjuh" → "tujuh", "ruPiah" → "rupiah")
- Fix broken numbers (e.g., "OOO,OO" → "000,00", "47 |" → "471")
- Do NOT change legal terminology, Pasal references, or document structure
- Do NOT add or remove content
- You MUST return ALL keys from the input. Every node_id must appear in your output.
- Return ONLY a valid JSON object with the same keys, mapping each node_id to its cleaned text. No explanation, no markdown.

Input (JSON):
"""

def _process_batch(client, batch: dict[str, str], batch_idx: int, total: int, verbose: bool,
                   _label: str | None = None):
    """Run one Gemini batch with rate-limit retry. Returns (cleaned_dict, in_tokens, out_tokens, error_or_None)."""
    prompt = LLM_CLEANUP_PROMPT + json.dumps(batch, ensure_ascii=False)

    max_retries = 5
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            last_err = None
            break
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            is_rate_limit = "rate" in err_str or "quota" in err_str or "429" in err_str
            is_network = (
                "timeout" in err_str or "timed out" in err_str
                or "ssl" in err_str or "handshake" in err_str
                or "connection" in err_str or "reset" in err_str
                or "broken pipe" in err_str or "eof" in err_str
            )
            if (is_rate_limit or is_network) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                reason = "rate limited" if is_rate_limit else "network error"
                if verbose:
                    _lbl = _label or f"{batch_idx + 1}/{total}"
                    log.warning(f"batch {_lbl}: {reason} ({e.__class__.__name__}), retrying in {wait}s")
                time.sleep(wait)
            else:
                # Soft-fail: return original texts so caller can continue with remaining batches
                _lbl = _label or f"{batch_idx + 1}/{total}"
                msg = f"batch {_lbl}: {e.__class__.__name__}: {e}"
                if verbose:
                    log.warning(f"batch {_lbl}: failed after {attempt + 1} attempt(s), keeping raw OCR text")
                return batch, 0, 0, msg

    if last_err is not None:
        # Should not reach here, but guard anyway
        _lbl = _label or f"{batch_idx + 1}/{total}"
        return batch, 0, 0, f"batch {_lbl}: unexpected retry exit"

    # Track token usage
    usage = response.usage_metadata
    input_tok = usage.prompt_token_count or 0
    output_tok = usage.candidates_token_count or 0
    _lbl = _label or f"{batch_idx + 1}/{total}"
    if verbose:
        log.info(f"batch {_lbl}: {input_tok + output_tok:,} tokens ({input_tok:,} in, {output_tok:,} out)")

    # Parse JSON from response
    response_text = response.text.strip()
    # Strip markdown code fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        cleaned = json.loads(response_text)
    except json.JSONDecodeError as e:
        _lbl = _label or f"{batch_idx + 1}/{total}"
        msg = f"batch {_lbl}: {e}"
        if verbose:
            log.warning(f"failed to parse LLM response for {msg}")
        return batch, input_tok, output_tok, msg  # Fall back to original texts

    return cleaned, input_tok, output_tok, None

def llm_cleanup_texts(texts: list[tuple[str, str]], verbose: bool = True,
                      client=None) -> tuple[dict[str, str], list[str]]:
    """Clean OCR artifacts in texts with Gemini. Returns (cleaned_dict, failure_messages)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google import genai

    if client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")
        client = genai.Client(api_key=api_key, http_options={"timeout": 300})

    # Split into batches by character count
    batches: list[dict[str, str]] = []
    current_batch: dict[str, str] = {}
    current_size = 0

    for node_id, text in texts:
        text_len = len(text)
        if current_size + text_len > _LLM_BATCH_SIZE and current_batch:
            batches.append(current_batch)
            current_batch = {}
            current_size = 0
        current_batch[node_id] = text
        current_size += text_len

    if current_batch:
        batches.append(current_batch)

    if verbose:
        log.info(f"{len(texts)} nodes in {len(batches)} batch(es), processing in parallel")

    # Process batches in parallel
    results: dict[str, str] = {}
    failed_batches: list[tuple[int, dict[str, str]]] = []
    total_input_tokens = 0
    total_output_tokens = 0

    with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
        futures = {
            executor.submit(_process_batch, client, batch, i, len(batches), verbose): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_i = futures[future]
            cleaned, input_tok, output_tok, error = future.result()
            results.update(cleaned)
            total_input_tokens += input_tok
            total_output_tokens += output_tok
            if error:
                failed_batches.append((batch_i, batches[batch_i]))

    # Retry failed batches by splitting in half (smaller batch = less likely to truncate).
    failures: list[str] = []
    if failed_batches:
        if verbose:
            log.info(f"retrying {len(failed_batches)} failed batch(es) split in half")
        for batch_i, failed_batch in failed_batches:
            items = list(failed_batch.items())
            mid = max(1, len(items) // 2)
            for sub_idx, sub_items in enumerate([items[:mid], items[mid:]]):
                if not sub_items:
                    continue
                sub_batch = dict(sub_items)
                sub_label = f"{batch_i + 1}{'ab'[sub_idx]}"
                sub_cleaned, in_tok, out_tok, sub_err = _process_batch(
                    client, sub_batch, batch_i, len(batches), verbose, _label=sub_label
                )
                results.update(sub_cleaned)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                if sub_err:
                    failures.append(
                        f"LLM batch {batch_i + 1}{'ab'[sub_idx]} failed after retry: "
                        f"{len(sub_batch)} node(s) kept as raw OCR text"
                    )

    if verbose:
        log.info(f"total tokens: {total_input_tokens + total_output_tokens:,} "
                 f"({total_input_tokens:,} in, {total_output_tokens:,} out)")

    return results, failures

# Contract: mutates leaf text and penjelasan text in place.
# Returns warnings while preserving original text on failed cleanup.
def apply_llm_cleanup(output_nodes: list[dict],
                      penjelasan_data: dict | None = None,
                      verbose: bool = True,
                      client=None):
    """Run LLM OCR cleanup on all leaf node texts and penjelasan in-place."""
    # Collect all leaf texts (nodes with "text" field)
    # Also collect penjelasan texts (skip trivial "Cukup jelas.")
    texts_to_clean: list[tuple[str, str]] = []

    def _collect(nodes: list[dict]):
        for node in nodes:
            if "nodes" in node:
                _collect(node["nodes"])
            else:
                if "text" in node:
                    texts_to_clean.append((node["node_id"], node["text"]))
                if node.get("penjelasan") and node["penjelasan"] != "Cukup jelas.":
                    texts_to_clean.append((node["node_id"] + ":penjelasan", node["penjelasan"]))

    _collect(output_nodes)

    # Also clean penjelasan_umum if present
    if penjelasan_data and penjelasan_data.get("umum"):
        texts_to_clean.append(("__penjelasan_umum__", penjelasan_data["umum"]))

    if not texts_to_clean:
        if verbose:
            log.info("no texts to clean")
        return

    # Call LLM cleanup
    cleaned, llm_failures = llm_cleanup_texts(texts_to_clean, verbose=verbose, client=client)

    # Check for missing nodes
    expected_ids = {nid for nid, _ in texts_to_clean}
    missing = expected_ids - set(cleaned.keys())
    if missing and verbose:
        log.warning(f"{len(missing)} node(s) missing from LLM response, keeping original: {sorted(missing)}")

    # Replace texts in-place
    def _replace(nodes: list[dict]):
        for node in nodes:
            if "nodes" in node:
                _replace(node["nodes"])
            else:
                if "text" in node and node["node_id"] in cleaned:
                    node["text"] = cleaned[node["node_id"]]
                penj_key = node.get("node_id", "") + ":penjelasan"
                if penj_key in cleaned:
                    node["penjelasan"] = cleaned[penj_key]

    _replace(output_nodes)

    # Write back penjelasan_umum
    if penjelasan_data and "__penjelasan_umum__" in cleaned:
        penjelasan_data["umum"] = cleaned["__penjelasan_umum__"]

    if verbose:
        log.info(f"cleaned {len(cleaned)} nodes")

    return llm_failures

_OCR_HEADER_PATTERNS = [
    # Multi-line "PRESIDEN\nREPUBLIK INDONESIA" (clean version, post-LLM)
    re.compile(r'PRESIDEN\s*\n\s*REPUBLIK\s+INDONESIA'),
    # Single-line variant
    re.compile(r'^PRESIDEN\s+REPUBLIK\s+INDONESIA\s*$', re.MULTILINE),
    # Footer patterns
    re.compile(r'^LEMBARAN\s+NEGARA\s+REPUBLIK\s+INDONESIA.*$', re.MULTILINE),
    re.compile(r'^TAMBAHAN\s+LEMBARAN\s+NEGARA.*$', re.MULTILINE),
]

# Closing/pengesahan text that leaks into the last Pasal when both share a page.
# Must be tolerant of OCR noise (broken words, extra whitespace).
_CLOSING_TEXT_RE = re.compile(
    r'\n\s*(?:'
    r'Ditetapkan\s+di\s'
    r'|Diundangkan\s+di\s'
    r'|Agar\s+setiap\s+orang'
    r'|Agar\s+setiap\s*\n'  # OCR line-break variant
    r')'
    r'[\s\S]*$',
)


def strip_ocr_headers(nodes: list[dict]):
    """Strip residual PDF header/footer text from all leaf node texts in-place."""
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            strip_ocr_headers(node["nodes"])
        elif "text" in node:
            text = node["text"]
            for pat in _OCR_HEADER_PATTERNS:
                text = pat.sub('', text)
            # Strip closing/pengesahan text that leaked into content
            text = _CLOSING_TEXT_RE.sub('', text)
            # Clean up extra blank lines left behind
            text = re.sub(r'\n{3,}', '\n\n', text).strip()
            node["text"] = text


# ============================================================
# 7. MAIN PIPELINE
# ============================================================

# Contract: parser entry point for indexing pipeline.
# Returns metadata and stable 'structure' tree used by downstream indexing.
def parse_legal_pdf(pdf_path: str, verbose: bool = True,
                    use_llm_cleanup: bool = True,
                    is_perubahan: bool | None = None,
                    granularity: str = "pasal") -> dict:
    """Parse an Indonesian legal PDF into a PageIndex-compatible tree."""
    if granularity not in ("pasal", "ayat", "full_split"):
        raise ValueError(f"Unknown granularity: {granularity!r}. "
                         f"Use 'pasal', 'ayat', or 'full_split'.")
    pdf_path = str(pdf_path)
    pdf_name = Path(pdf_path).name
    total_steps = 6 if use_llm_cleanup else 5
    t_start = time.time()

    # Step 1: Extract text
    if verbose:
        log.info(f"[1/{total_steps}] extracting text from {pdf_name}")
    t0 = time.time()
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    if verbose:
        log.info(f"total pages: {total_pages} ({time.time() - t0:.1f}s)")

    # Step 2: Clean text
    if verbose:
        log.info(f"[2/{total_steps}] cleaning text")
    t0 = time.time()
    for page in pages:
        page["clean_text"] = clean_page_text(page["raw_text"])
    if verbose:
        log.info(f"done ({time.time() - t0:.1f}s)")

    # Step 3: Detect body boundaries
    if verbose:
        log.info(f"[3/{total_steps}] detecting document sections")
    closing_page = find_closing_page(pages)
    penjelasan_page = find_penjelasan_page(pages)

    # Body ends at whichever comes first: closing or penjelasan
    boundaries = [p for p in [closing_page, penjelasan_page] if p]
    if boundaries:
        body_end = min(boundaries) - 1
    else:
        body_end = total_pages

    if verbose:
        log.info(f"body: pages 1-{body_end}")
        if closing_page:
            log.info(f"closing/pengesahan: page {closing_page}")
        if penjelasan_page:
            log.info(f"penjelasan: pages {penjelasan_page}-{total_pages}")

    # Auto-detect Perubahan (amendment) UU if not specified
    if is_perubahan is None:
        is_perubahan = detect_perubahan(pages)
    if verbose and is_perubahan:
        log.info("detected as Perubahan (amendment)")

    # Step 4: Detect structural elements (only in body)
    if verbose:
        log.info(f"[4/{total_steps}] detecting structural elements")
    t0 = time.time()
    elements = detect_elements(pages, body_end, is_perubahan=is_perubahan)

    type_counts = {}
    for e in elements:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    if verbose:
        log.info(f"found: {type_counts} ({time.time() - t0:.1f}s)")

    # Validate Pasal sequence (skip for Perubahan and omnibus)
    is_omnibus = detect_omnibus(pages, elements)
    if verbose and is_omnibus:
        log.info("detected as Omnibus law")
    warnings = validate_pasal_sequence(elements, is_perubahan=is_perubahan or is_omnibus)
    for w in warnings:
        log.warning(w)

    # Step 5: Build tree
    if verbose:
        log.info(f"[5/{total_steps}] building tree structure")
    t0 = time.time()
    tree = build_tree(elements, body_end)
    fix_node_boundaries(tree, body_end)

    # Post-process: consolidate inserted BABs in perubahan documents
    if is_perubahan:
        bab_moved = consolidate_bab_in_perubahan(tree)
        if verbose and bab_moved:
            log.info(f"consolidated {bab_moved} Pasal(s) under inserted BAB(s)")

    # Add preamble node (text before first structural element)
    output_nodes = []

    if elements:
        first_elem = elements[0]
        first_elem_page = first_elem["page_num"]
        first_elem_offset = first_elem["char_offset"]
        # Only add preamble if there's content before the first element
        if first_elem_page > 1 or first_elem_offset > 0:
            preamble_text = _extract_node_text(pages, 1, first_elem_page,
                                               end_char_offset=first_elem_offset)
            preamble_children = _split_preamble(preamble_text, 1, first_elem_page)

            if preamble_children:
                output_nodes.append({
                    "title": "Pembukaan",
                    "node_id": "P000",
                    "start_index": 1,
                    "end_index": first_elem_page,
                    "nodes": preamble_children,
                })
            else:
                # Fallback: keep as single blob if splitting fails
                output_nodes.append({
                    "title": "Pembukaan (Menimbang, Mengingat, Memutuskan)",
                    "node_id": "P000",
                    "start_index": 1,
                    "end_index": first_elem_page,
                    "text": preamble_text,
                })

    output_nodes.extend(clean_tree_for_output(tree, pages))

    # Parse PENJELASAN per-Pasal and attach to leaf nodes
    penjelasan_data = None
    if penjelasan_page:
        penjelasan_data = parse_penjelasan(pages, penjelasan_page, total_pages)
        attach_penjelasan(output_nodes, penjelasan_data["pasal"])
        matched = sum(1 for n in iter_leaves(output_nodes) if n.get("penjelasan"))
        if verbose:
            log.info(f"penjelasan: {len(penjelasan_data['pasal'])} pasal parsed, "
                     f"{matched} matched to tree nodes")

    if verbose:
        log.info(f"done ({time.time() - t0:.1f}s)")

    # Step 6: LLM text cleanup (on by default, skip with --no-llm)
    llm_time = 0.0
    if use_llm_cleanup:
        if verbose:
            log.info(f"[6/{total_steps}] cleaning text with Gemini 2.5 Flash")
        t0 = time.time()
        llm_failures = apply_llm_cleanup(output_nodes, penjelasan_data, verbose=verbose)
        warnings.extend(llm_failures)
        llm_time = time.time() - t0
        if verbose:
            log.info(f"LLM cleanup: {llm_time:.1f}s")

    total_time = time.time() - t_start
    if verbose:
        if use_llm_cleanup:
            log.info(f"total time: {total_time:.1f}s  (LLM: {llm_time:.1f}s, parser: {total_time - llm_time:.1f}s)")
        else:
            log.info(f"total time: {total_time:.1f}s")

    # Sub-Pasal splitting based on requested granularity.
    if granularity == "ayat":
        output_nodes = ayat_split_leaves(output_nodes)
    elif granularity == "full_split":
        output_nodes = deep_split_leaves(output_nodes)

    # Final pass: strip residual OCR header/footer leaks from all leaf texts
    strip_ocr_headers(output_nodes)

    # Store unmatched penjelasan at doc level for retrieval agent fallback.
    # With perubahan support, amended Pasals are now leaf nodes and should
    # match normally; only fall back to doc-level if nothing matched.
    penjelasan_pasal = None
    if penjelasan_data and penjelasan_data["pasal"]:
        if matched == 0:
            penjelasan_pasal = penjelasan_data["pasal"]

    result = {
        "doc_name": pdf_name,
        "total_pages": total_pages,
        "body_pages": body_end,
        "penjelasan_page": penjelasan_page,
        "penjelasan_umum": penjelasan_data["umum"] if penjelasan_data else None,
        "penjelasan_pasal_demi_pasal": penjelasan_pasal,
        "is_perubahan": is_perubahan,
        "element_counts": type_counts,
        "warnings": warnings,
        "structure": output_nodes,
    }

    return result

def print_tree(nodes: list[dict], indent: int = 0):
    """Pretty-print tree structure."""
    for node in nodes:
        prefix = "  " * indent
        page_range = f"[p.{node['start_index']}-{node['end_index']}]"
        print(f"{prefix}{node.get('node_id', '----')} {node['title']} {page_range}")
        if "nodes" in node:
            print_tree(node["nodes"], indent + 1)

# ============================================================
# 8. SUB-PASAL LEAF SPLITTING
# ============================================================
# Post-processing step that splits Pasal leaf nodes into finer sub-nodes.
# Used by granularity="ayat" and granularity="full_split".
#
# Hierarchy: Pasal → Ayat (1)/(2)/... → Huruf a./b./... → Angka 1./2./...
#
# "ayat" mode:      splits to Ayat only (no Huruf/Angka recursion)
# "full_split" mode: recursively splits to the deepest level present

_AYAT_RE = re.compile(r'(?:^|\n)\((\d+)\)\s', re.MULTILINE)
_HURUF_RE = re.compile(r'(?:^|\n)([a-z])\.\s', re.MULTILINE)
_ANGKA_ITEM_RE = re.compile(r'(?:^|\n)(\d+)\.\s', re.MULTILINE)
_PENJ_AYAT_RE = re.compile(r'Ayat\s*\((\d+)\)\s*\n', re.MULTILINE)
_PENJ_HURUF_RE = re.compile(r'Huruf\s+([a-z])\s*\n', re.MULTILINE)


def _find_and_validate_markers(
    text: str, pattern: re.Pattern, expected_start: str
) -> list[tuple[int, str]] | None:
    """Find structural markers in text and validate they form a consecutive sequence.

    Consecutive means: starts at expected_start and increments by 1 (a→b→c or 1→2→3).
    Returns list of (char_pos, label) if valid sequence with ≥2 items, else None.
    """
    matches = list(pattern.finditer(text))
    if len(matches) < 2:
        return None

    # Deduplicate consecutive markers with the same label.
    # Page-break text overlap can create duplicates like "(5) Selain\n\n(5) Selain dikenai..."
    deduped: list[re.Match] = [matches[0]]
    for m in matches[1:]:
        if m.group(1) == deduped[-1].group(1):
            deduped[-1] = m  # replace with later (more complete) occurrence
        else:
            deduped.append(m)
    matches = deduped
    if len(matches) < 2:
        return None

    labels = [m.group(1) for m in matches]

    if expected_start.isdigit():
        try:
            nums = [int(l) for l in labels]
        except ValueError:
            return None
        if nums[0] != 1:
            return None
        if not all(nums[i + 1] == nums[i] + 1 for i in range(len(nums) - 1)):
            return None
    else:
        if labels[0] != expected_start:
            return None
        if not all(ord(labels[i + 1]) == ord(labels[i]) + 1 for i in range(len(labels) - 1)):
            return None

    return [(m.start(), m.group(1)) for m in matches]


def _split_text_by_markers(
    text: str, markers: list[tuple[int, str]]
) -> tuple[str, list[tuple[str, str]]]:
    """Split text into segments at each marker position.

    Returns (intro_text, [(label, segment_text), ...]).
    """
    positions = [pos for pos, _ in markers]
    labels = [label for _, label in markers]

    intro = text[:positions[0]].strip()

    segments = []
    for i, (pos, label) in enumerate(zip(positions, labels)):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segment = text[pos:end].strip()
        segments.append((label, segment))

    return intro, segments


def _distribute_penjelasan(
    penjelasan: str | None, sub_nodes: list[dict], kind: str
) -> None:
    """Distribute penjelasan text to sub-nodes in-place.

    Tries to parse penjelasan for per-item markers (Ayat (N) or Huruf X).
    Assigns matched sub-node its specific explanation; unmatched nodes get full text.
    """
    if not penjelasan:
        return

    if kind == "ayat":
        split_re = _PENJ_AYAT_RE
    elif kind == "huruf":
        split_re = _PENJ_HURUF_RE
    else:
        # angka: no known penjelasan distribution pattern
        for node in sub_nodes:
            node["penjelasan"] = penjelasan
        return

    parts = split_re.split(penjelasan)
    penj_map = {}
    for i in range(1, len(parts) - 1, 2):
        penj_map[parts[i]] = parts[i + 1].strip()

    for node in sub_nodes:
        label = node.get("_split_label")
        if label and label in penj_map:
            node["penjelasan"] = penj_map[label]
        else:
            node["penjelasan"] = penjelasan


def _strip_leading_junk(text: str) -> str:
    """Strip leading non-structural characters (colon, whitespace) from text."""
    return re.sub(r'^[\s:;,\-]+', '', text)


# Ayat-only splitting (granularity="ayat")

def _try_ayat_split(
    text: str,
    parent_id: str,
    parent_title: str,
    parent_start: int,
    parent_end: int,
    penjelasan: str | None,
) -> list[dict] | None:
    """Split text into Ayat sub-nodes only. No recursion into Huruf/Angka.

    Returns a list of Ayat child dicts if ayat markers found, else None.
    """
    text = _strip_leading_junk(text)

    ayat_markers = _find_and_validate_markers(text, _AYAT_RE, "1")
    if not ayat_markers:
        return None

    intro, segments = _split_text_by_markers(text, ayat_markers)
    sub_nodes = []
    for i, (label, segment) in enumerate(segments, 1):
        sub_id = f"{parent_id}_a{i}"
        sub_title = f"{parent_title} Ayat ({label})"
        node: dict = {
            "title": sub_title,
            "node_id": sub_id,
            "start_index": parent_start,
            "end_index": parent_end,
            "text": segment,
            "_split_label": label,
        }
        sub_nodes.append(node)
    _distribute_penjelasan(penjelasan, sub_nodes, "ayat")
    for n in sub_nodes:
        n.pop("_split_label", None)
    return sub_nodes


def _split_leaves_with(nodes: list[dict], split_func) -> list[dict]:
    """Apply a leaf split function recursively and preserve non-leaf structure."""
    result = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            node["nodes"] = _split_leaves_with(node["nodes"], split_func)
            result.append(node)
        elif "text" in node:
            sub_nodes = split_func(node)
            if sub_nodes:
                branch = {k: v for k, v in node.items() if k not in ("text", "penjelasan")}
                branch["nodes"] = sub_nodes
                result.append(branch)
            else:
                result.append(node)
        else:
            result.append(node)
    return result


def ayat_split_leaves(nodes: list[dict]) -> list[dict]:
    """Walk the tree and split every leaf node into Ayat sub-nodes only.

    Does NOT recurse deeper into Huruf/Angka. If a Pasal has no Ayat markers,
    it stays as a leaf unchanged.
    """
    def _split(node: dict):
        return _try_ayat_split(
            text=node["text"],
            parent_id=node["node_id"],
            parent_title=node["title"],
            parent_start=node["start_index"],
            parent_end=node["end_index"],
            penjelasan=node.get("penjelasan"),
        )

    return _split_leaves_with(nodes, _split)


# Recursive deep splitting (granularity="full_split")

def _try_deep_split(
    text: str,
    parent_id: str,
    parent_title: str,
    parent_start: int,
    parent_end: int,
    penjelasan: str | None,
) -> list[dict] | None:
    """Recursively split text into the smallest structural sub-nodes.

    Tries ayat first, then huruf, then angka items.
    Returns a list of child node dicts if a split was found, or None.
    """
    text = _strip_leading_junk(text)

    # Try Ayat: (1), (2), (3), ...
    ayat_markers = _find_and_validate_markers(text, _AYAT_RE, "1")
    if ayat_markers:
        intro, segments = _split_text_by_markers(text, ayat_markers)
        sub_nodes = []
        for i, (label, segment) in enumerate(segments, 1):
            sub_id = f"{parent_id}_a{i}"
            sub_title = f"{parent_title} Ayat ({label})"
            children = _try_deep_split(
                segment, sub_id, sub_title, parent_start, parent_end, penjelasan=None
            )
            node: dict = {
                "title": sub_title,
                "node_id": sub_id,
                "start_index": parent_start,
                "end_index": parent_end,
                "_split_label": label,
            }
            if children:
                node["nodes"] = children
            else:
                node["text"] = segment
            sub_nodes.append(node)
        _distribute_penjelasan(penjelasan, sub_nodes, "ayat")
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    # Try Huruf: a., b., c., ...
    huruf_markers = _find_and_validate_markers(text, _HURUF_RE, "a")
    if huruf_markers:
        intro, segments = _split_text_by_markers(text, huruf_markers)
        sub_nodes = []
        for i, (label, segment) in enumerate(segments, 1):
            sub_id = f"{parent_id}_h{i}"
            sub_title = f"{parent_title} Huruf {label}"
            children = _try_deep_split(
                segment, sub_id, sub_title, parent_start, parent_end, penjelasan=None
            )
            node = {
                "title": sub_title,
                "node_id": sub_id,
                "start_index": parent_start,
                "end_index": parent_end,
                "_split_label": label,
            }
            if children:
                node["nodes"] = children
            else:
                node["text"] = segment
            sub_nodes.append(node)
        _distribute_penjelasan(penjelasan, sub_nodes, "huruf")
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    # Try Angka items: 1., 2., 3., ...
    angka_markers = _find_and_validate_markers(text, _ANGKA_ITEM_RE, "1")
    if angka_markers:
        intro, segments = _split_text_by_markers(text, angka_markers)
        sub_nodes = []
        for i, (label, segment) in enumerate(segments, 1):
            sub_id = f"{parent_id}_n{i}"
            sub_title = f"{parent_title} Angka {label}"
            sub_nodes.append({
                "title": sub_title,
                "node_id": sub_id,
                "start_index": parent_start,
                "end_index": parent_end,
                "text": segment,
                "_split_label": label,
            })
        _distribute_penjelasan(penjelasan, sub_nodes, "angka")
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    return None


def deep_split_leaves(nodes: list[dict]) -> list[dict]:
    """Walk the tree and recursively split every leaf node to its deepest sub-structure.

    Tries Ayat → Huruf → Angka. If no sub-structure found, leaf stays unchanged.
    """
    def _split(node: dict):
        return _try_deep_split(
            text=node["text"],
            parent_id=node["node_id"],
            parent_title=node["title"],
            parent_start=node["start_index"],
            parent_end=node["end_index"],
            penjelasan=node.get("penjelasan"),
        )

    return _split_leaves_with(nodes, _split)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s",
                        datefmt="%H:%M:%S")

    # Parse flags
    args = sys.argv[1:]
    skip_llm = "--no-llm" in args
    if skip_llm:
        args.remove("--no-llm")
    use_llm = not skip_llm

    # Parse --granularity flag (default: pasal)
    gran = "pasal"
    if "--granularity" in args:
        idx = args.index("--granularity")
        if idx + 1 < len(args):
            gran = args[idx + 1]
            args = args[:idx] + args[idx + 2:]
        else:
            log.error("--granularity requires a value (pasal, ayat, full_split)")
            sys.exit(1)

    if gran not in ("pasal", "ayat", "full_split"):
        log.error(f"unknown granularity '{gran}'. Use: pasal, ayat, full_split")
        sys.exit(1)

    if use_llm and not os.environ.get("GEMINI_API_KEY"):
        log.error("GEMINI_API_KEY is not set. Set it for LLM cleanup, or use --no-llm to skip.")
        sys.exit(1)

    # Determine input
    # Join remaining args to handle paths with spaces
    pdf_arg = " ".join(args).strip() if args else ""
    if not pdf_arg:
        pdf_dir = Path("data/raw/UU/pdfs")
        # Process all main UU PDFs (not lampiran/salinan)
        pdf_files = sorted([
            f for f in pdf_dir.glob("UU Nomor *.pdf")
            if "Lampiran" not in f.name and "Salinan" not in f.name
        ])
        if not pdf_files:
            log.error("no PDF files found. Provide a path as argument.")
            sys.exit(1)
    else:
        pdf_files = [Path(pdf_arg)]

    for pdf_path in pdf_files:
        log.info(f"processing {pdf_path.name}")

        result = parse_legal_pdf(str(pdf_path), use_llm_cleanup=use_llm,
                                granularity=gran)

        print(f"\nTree structure ({gran})")
        print_tree(result["structure"])

        # Save output
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / (pdf_path.stem + "_structure.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.info(f"saved to {output_path}")