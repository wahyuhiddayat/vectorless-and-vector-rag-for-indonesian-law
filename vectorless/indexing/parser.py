"""
Rule-based parser for Indonesian legal documents (UU/PP/Perpres).
Converts PDF to hierarchical tree structure (PageIndex-compatible format).

Exploits the predictable hierarchy defined by UU No. 12/2011:
    BAB > Bagian > Paragraf > Pasal
to build a structural tree index without LLM-based indexing.

Supports 3 leaf granularity levels via the `granularity` parameter:
    - "pasal":      leaf = entire Pasal text (coarsest)
    - "ayat":       leaf = Ayat within each Pasal
    - "full_split": leaf = smallest unit: Ayat > Huruf > Angka (finest)
"""

import re
import json
import os
import sys
import time
import fitz  # PyMuPDF
from pathlib import Path

# ============================================================
# 1. TEXT EXTRACTION & CLEANING
# ============================================================

def _detect_two_columns(blocks: list[dict], page_width: float,
                        is_landscape: bool = False) -> list[dict]:
    """
    Detect two-column gazette layout and reorder blocks for correct reading order.

    Indonesian gazette PDFs (Lembaran Negara) use two-column landscape layout.
    Column detection is restricted to landscape pages (width > height) to avoid
    false positives on regular portrait PDFs where blocks may incidentally
    straddle the midpoint. For portrait pages, blocks are sorted by (y, x)
    which is still an improvement over PyMuPDF's unsorted default.

    Returns blocks in correct reading order.
    """
    if len(blocks) < 4 or not is_landscape:
        return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))

    # Two-column detection: only for landscape gazette pages.
    # Landscape pages (width > height) are gazette (Lembaran Negara) format
    # with left/right columns that need special handling.
    midpoint = page_width / 2

    # Classify each block into exactly one column using its center x-coordinate.
    left = []
    right = []
    for b in blocks:
        center_x = (b["x0"] + b["x1"]) / 2
        if center_x < midpoint:
            left.append(b)
        else:
            right.append(b)

    # Wide blocks span >60% of page width (e.g. full-width headers/titles)
    wide_blocks = [b for b in blocks if (b["x1"] - b["x0"]) > page_width * 0.6]

    # Two-column: both sides have content, and most blocks are not full-width
    if len(left) >= 3 and len(right) >= 3 and len(wide_blocks) < len(blocks) * 0.3:
        left_sorted = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right_sorted = sorted(right, key=lambda b: (b["y0"], b["x0"]))
        return left_sorted + right_sorted

    # Fallback: sort by position
    return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))


def _extract_page_text(page) -> str:
    """
    Extract text from a single PyMuPDF page with column-aware ordering.

    Uses get_text("dict") to get bounding box coordinates for each text block,
    detects two-column gazette layout, and reorders blocks so left column is
    read fully before right column.
    """
    page_dict = page.get_text("dict")
    page_width = page_dict.get("width", 595)  # A4 default fallback
    raw_blocks = page_dict.get("blocks", [])

    text_blocks = []
    for b in raw_blocks:
        if b.get("type") != 0:  # 0 = text block, 1 = image
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
    """Extract text from each page of a PDF using PyMuPDF.

    Uses column-aware extraction to handle two-column gazette PDFs
    (Lembaran Negara format) where PyMuPDF's default extraction reads
    across columns instead of reading each column top-to-bottom.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = _extract_page_text(page)
        pages.append({
            "page_num": i + 1,  # 1-indexed
            "raw_text": text,
        })
    doc.close()
    return pages

def clean_page_text(text: str) -> str:
    """Remove common noise from Indonesian legal PDF text."""
    # --- Header removal ---
    # OCR often garbles "PRESIDEN REPUBLIK INDONESIA" into many variants.
    # Each regex below targets a different corruption pattern observed across
    # 10+ UU PDFs from BPK JDIH. The strings look like garbage but are real
    # OCR outputs — do not "clean up" these regex patterns themselves.

    # Multi-line "PRESIDEN\nREPUBLIK INDONESIA" with OCR variants
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

    # Split glued Pasal/BAB headings where word is concatenated without space:
    #   "diperolehPasal 2" → "diperoleh\nPasal 2"
    #   "rupiah),Pasal 7" → "rupiah),\nPasal 7"
    #   "(3)Pasal 4" → "(3)\nPasal 4"
    # Uses lookbehind: only split when preceded by letter/digit/punct (not space/newline)
    # This preserves "dalam Pasal 3" on separate lines
    text = re.sub(r'([^\s])(?=Pasal\s+\d)', r'\1\n', text)
    # Handle glued Pasal with no space: "Pasal22" → "Pasal 22" (on its own line)
    text = re.sub(r'^(Pasal)(\d)', r'\1 \2', text, flags=re.MULTILINE)

    # Split glued BAB headings similarly
    text = re.sub(r'([^\n])(?=BAB\s+[IVXLCDM])', r'\1\n', text)

    # Normalize whitespace
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix font-encoding issues in headings:
    #   O (U+004F) → 0, l (U+006C) → 1, I (U+0049) → 1
    # Only apply to lines that look like structural headings (Pasal, BAB, etc.)
    text = fix_ocr_artifacts(text)

    return text.strip()

def fix_ocr_artifacts(text: str) -> str:
    """
    Fix OCR artifacts in ayat numbering and remove page-continuation noise.

    Handles font-encoding issues where heading fonts encode:
      0 (zero) as O (letter O)
      1 (one) as l (letter l) or I (letter I)

    Also fixes broken ayat patterns like "(2t" → "(2)" and removes
    trailing dots from Pasal continuation markers.

    NOTE: Pasal heading numbers are NOT fixed here — they use the smarter
    _parse_pasal_number() which distinguishes suffix letters from OCR digits.
    """
    lines = text.split('\n')
    fixed_lines = []

    for line in lines:
        stripped = line.strip()

        # Fix ayat pattern: "(2l)" → "(21)", "(l)" → "(1)", "(s)" → "(5)"
        # Also handles: "(2t" → "(2)", "t2t" → "(2)", "l2l" → "(2)"
        line = re.sub(
            r'\(([0-9OlI]+)\)',
            lambda m: '(' + _normalize_ocr_digits(m.group(1)) + ')',
            line
        )

        # Fix broken ayat patterns from OCR:
        # "(2t" (missing close paren, t is noise) → "(2)"
        # "t2t" (garbled parens) → "(2)"
        # "l2l" (garbled parens) → "(2)"
        # "(s)" where s should be 5 → "(5)"
        line = re.sub(r'\((\d+)[t]\b', lambda m: '(' + m.group(1) + ')', line)
        line = re.sub(r'\b[tl](\d+)[tl]\b', lambda m: '(' + m.group(1) + ')', line)
        line = re.sub(r'\(s\)', '(5)', line)

        # Remove page continuation markers: "Pasal 7...", "(3)DBH . , ."
        line = re.sub(r'^Pasa[lr]\s*\d+\s*\.{2,}\s*$', '', stripped and line or line)
        line = re.sub(r'\.\s*\.\s*\.\s*$', '', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)

def _normalize_ocr_digits(s: str) -> str:
    """
    Normalize a string that should be purely numeric but may contain
    OCR misreads: O→0, l→1, I→1, and spurious spaces.

    Only converts if the result would be a valid integer.
    """
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
    return s  # Return original if normalization doesn't produce a number

def _parse_pasal_number(raw: str) -> tuple[str | None, str]:
    """
    Parse a raw Pasal number string that may contain OCR artifacts.

    Returns (number, suffix) where suffix is a letter A-Z or empty string.

    Examples:
        "40"    → ("40", "")
        "4O"    → ("40", "")       # O is OCR'd 0
        "119I"  → ("119", "I")     # I is suffix, not OCR'd 1
        "9I"    → ("91", "")       # I is OCR'd 1 (since 9I with suffix would be unusual)
        "2l"    → ("21", "")       # l is OCR'd 1
        "9 I"   → ("91", "")       # space + I is OCR'd 1
        "599A"  → ("599", "A")
        "11G"   → ("11", "G")
        "17M"   → ("17", "M")
    """
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
                # Likely OCR: "9I" → Pasal 91
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
        # No trailing letter — just normalize digits
        normalized = _normalize_ocr_digits(s)
        if normalized.isdigit():
            return normalized, ""
        return None, ""

# ============================================================
# 2. PENJELASAN (EXPLANATION SECTION) DETECTION
# ============================================================

def find_penjelasan_page(pages: list[dict]) -> int | None:
    """
    Detect the page where PENJELASAN (Explanation) section starts.
    This section comes after the main body and should be separated.
    Returns page_num or None if not found.
    """
    for page in pages:
        text = page["raw_text"]  # use raw text since PENJELASAN is usually clean
        if re.search(r'PENJ\S*SAN\s*\n\s*ATAS', text):
            return page["page_num"]
    return None

def find_closing_page(pages: list[dict]) -> int | None:
    """
    Detect the closing page (pengesahan).
    UU uses 'Disahkan di Jakarta', PP/Perpres uses 'Ditetapkan di Jakarta'.
    This marks the end of the main body.
    """
    for page in pages:
        text = page["raw_text"]
        if re.search(r'Di(?:sahkan|tetapkan) di Jakarta|Agar setiap orang mengetahuinya', text):
            return page["page_num"]
    return None

def detect_perubahan(pages: list[dict]) -> bool:
    """
    Detect if a law is a Perubahan (amendment) UU/PP/Perpres.

    Checks the law's title (between TENTANG and DENGAN RAHMAT) for keywords
    "PERUBAHAN" or "PENYESUAIAN".

    Scans first 3 pages because some PDFs have garbled page ordering or
    the title spans multiple pages. Uses relaxed whitespace matching to
    handle OCR that concatenates words (e.g. "TENTANGPENYESUAIAN").

    Amendment UUs use Roman numeral articles (Pasal I, Pasal II) per UU 12/2011
    convention. Arabic-numbered Pasals inside are quoted text from the amended law.
    """
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
    """Detect if document is an omnibus law (e.g. Cipta Kerja).

    Omnibus laws amend multiple existing laws, each with independent Pasal
    numbering. Pasal validation is meaningless for these documents.
    """
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

def parse_penjelasan(pages: list[dict], penjelasan_page: int,
                     total_pages: int) -> dict:
    """
    Parse PENJELASAN section into umum (general) + per-Pasal explanations.

    The PENJELASAN section of Indonesian laws has two parts:
      I.  UMUM — general background/philosophy of the law
      II. PASAL DEMI PASAL — per-article explanations

    Returns:
        {
            "umum": "...",       # Text of section I. UMUM
            "pasal": {           # Per-Pasal explanations keyed by Pasal number string
                "1": "Cukup jelas.",
                "2": "Huruf a\nYang dimaksud dengan ...",
                "7": "Ayat (1)\nCukup jelas.\nAyat (2)\n...",
            }
        }
    """
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
    # "II." prefix is optional — some shorter UUs omit it or OCR drops it.
    # First space is \s* not \s+ because OCR sometimes merges "PASALDEMI".
    split_m = re.split(r'(?:II\.?\s*|[iI][lI1]\.?\s*)?PASAL\s*DEMI\s+PASAL', full_text, maxsplit=1,
                       flags=re.IGNORECASE)

    if len(split_m) == 2:
        umum_raw, pasal_section = split_m
    else:
        # No "PASAL DEMI PASAL" found — entire text is umum
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
    """
    Fix OCR column-reading artifacts in PENJELASAN's PASAL DEMI PASAL section.

    PyMuPDF reads multi-column compact layouts vertically, producing:
      "Pasal\\nPasal\\nPasal\\n51\\nCukup jelas.\\n52\\nCukup jelas.\\n53\\n..."

    This function detects sequences of N consecutive "Pasal" lines followed by
    bare numbers, and reconstructs them as "Pasal 51\\nCukup jelas.\\nPasal 52\\n..."

    Also handles bare numbers without "Pasal" prefix (continuation from stacked
    pattern across page breaks).
    """
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        # Detect stacked "Pasal" lines (N consecutive lines that are just "Pasal")
        if re.match(r'^Pasa[l1]\s*$', stripped):
            # Count how many consecutive "Pasal" lines
            pasal_count = 0
            j = i
            while j < len(lines) and re.match(r'^Pasa[l1]\s*$', lines[j].strip()):
                pasal_count += 1
                j += 1

            # Now collect the next entries: each should be a number possibly
            # followed by explanation lines until the next bare number
            collected = []
            k = j
            while k < len(lines) and len(collected) < pasal_count:
                num_line = lines[k].strip()
                num_m = re.match(r'^(\d+[A-Z]?)\s*$', num_line)
                if num_m:
                    collected.append((num_m.group(1), k))
                    k += 1
                else:
                    # This line is explanation text for the previous number
                    if collected:
                        k += 1
                    else:
                        break  # Can't pair, bail out

            if collected:
                # Rebuild: pair each number with "Pasal" and emit with
                # explanation text between this number and the next
                for idx, (num, start_k) in enumerate(collected):
                    result.append(f"Pasal {num}")
                    # Add explanation lines between this number and the next
                    if idx + 1 < len(collected):
                        end_k = collected[idx + 1][1]
                    else:
                        end_k = k
                    for exp_line_idx in range(start_k + 1, end_k):
                        result.append(lines[exp_line_idx])
                i = k
                continue
            else:
                # Failed to pair, keep original lines
                result.append(lines[i])
                i += 1
                continue

        # Detect bare numbers on their own line (page-break continuation)
        # Only treat as Pasal if followed by "Cukup jelas." or another bare number
        elif re.match(r'^(\d+)\s*$', stripped):
            # Look ahead to see if this is a sequence of bare Pasal numbers
            # Pattern: "59\nCukup jelas.\n60\nCukup jelas.\n..."
            j = i
            bare_entries = []
            while j < len(lines):
                num_m = re.match(r'^(\d+)\s*$', lines[j].strip())
                if num_m:
                    bare_entries.append((num_m.group(1), j))
                    j += 1
                elif lines[j].strip().lower().startswith('cukup jelas') and bare_entries:
                    j += 1  # Skip the "Cukup jelas." line
                elif bare_entries and not lines[j].strip():
                    j += 1  # Skip blank lines
                else:
                    break

            if len(bare_entries) >= 2:
                # This is a bare number sequence — prefix with "Pasal"
                for idx, (num, start_j) in enumerate(bare_entries):
                    result.append(f"Pasal {num}")
                    # Add explanation text between this and next bare number
                    if idx + 1 < len(bare_entries):
                        end_j = bare_entries[idx + 1][1]
                    else:
                        end_j = j
                    for exp_idx in range(start_j + 1, end_j):
                        result.append(lines[exp_idx])
                i = j
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

def _attach_penjelasan(nodes: list[dict], pasal_dict: dict[str, str]):
    """
    Attach per-Pasal penjelasan to leaf nodes in the tree.

    Walks the tree and for each leaf node that is a Pasal, looks up its
    number in pasal_dict and sets the 'penjelasan' field.
    """
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            _attach_penjelasan(node["nodes"], pasal_dict)
        elif "text" in node:
            # Leaf node — try to match Pasal number
            # Extract number from title: "Pasal 5" → "5", "Pasal 119I" → "119I"
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

# Roman numeral pattern for both BAB and Paragraf.
# Optional trailing letter (A, B, C) for amendment-inserted sections like BAB VIIA, BAB IXA.
ROMAN_NUMERAL = r'[IVXLCDM]+[A-Z]?'

PATTERNS = {
    "bab": re.compile(
        r'^BAB\s+(' + ROMAN_NUMERAL + r')\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE
    ),
    "bagian": re.compile(
        # Handle OCR typos: Bagian, Bagtan, Brgian, etc.
        # Uses B + 1-5 lowercase chars + "an" to catch OCR corruptions
        # while avoiding false matches (ordinal names like Kesatu anchor it)
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
        # Capture entire number+suffix part, parse in code
        # Handles: "Pasal 40", "Pasal 4O", "Pasal 119I", "Pasal 9 I", "Pasal 51 ..."
        # Also handles OCR variant "Pasa1" (l→1) via Pasa[l1]
        # Case-insensitive initial P handles OCR "pasal" → "Pasal"
        # Requires space between "Pasal" and number
        # Negative lookahead: next line must NOT start with reference continuation
        # (ayat, huruf, angka, dan Pasal, sampai, jo) — these indicate "Pasal X" is
        # a cross-reference, not a heading
        r'^[Pp]asa[l1]\s+([0-9OlI][0-9A-Za-z \t]*?)\s*(?:\.\s*\.\s*[.\'])?$'
        r'(?:\n(?!ayat|huruf|angka|dan Pasal|sampai dengan|jo\.?\s)|\Z)',
        re.MULTILINE
    ),
}

# Words whose presence on the line immediately before "Pasal X" indicate that
# the match is a cross-reference within body text, NOT a section heading.
# Example: "...sebagaimana dimaksud dalam\nPasal 76D\ndipidana..." — here "dalam"
# on the previous line means "Pasal 76D" is part of the sentence, not a heading.
_CROSS_REF_PRECEDING = frozenset({
    'dalam', 'pada', 'oleh', 'dari', 'dan', 'atau', 'dengan',
    'sebagaimana', 'berdasarkan', 'terhadap', 'menurut', 'tentang',
    'atas', 'bahwa', 'melalui', 'untuk', 'antara',
})

# Level mapping for hierarchy.
# For Perubahan (amendment) UUs:
#   Pasal I (roman, 0) > Angka items (1) > BAB (2) > Bagian (3) > Paragraf (4) > Pasal (5)
# For normal UUs: BAB (2) is the root — levels 0-1 are unused.
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

# Regex for Roman numeral Pasal headings used in Perubahan (amendment) UUs.
# These are the actual articles of the amendment law (Pasal I, II, III, ...).
PASAL_ROMAN = re.compile(
    r'^[Pp]asa[l1]\s+([IVXLC]+)\s*$',
    re.MULTILINE
)

# Regex for Angka (numbered amendment instructions) inside Pasal I of Perubahan UUs.
# Matches patterns like:
#   "76. Ketentuan Pasal 88 diubah sehingga..."
#   "82. Pasal 99 dihapus."
#   "86. Di antara Bab VII dan Bab VIII disisipkan..."
#   "81. Bagian Keenam Bab VII dihapus."
#   "1.\nKetentuan Pasal 1 ditambahkan..." (number on own line, text on next)
# The instruction keyword check (Ketentuan/Pasal/Bagian/Bab/Di antara) prevents
# false matches on numbered list items inside Pasal text ("1. Ibadah Haji adalah...").
ANGKA_PATTERN = re.compile(
    r'^(\d+)\.\s*'                              # number + period at start of line
    r'((?:Ketentuan\s+)?'                       # optional "Ketentuan "
    r'(?:Pasal|Bagian|Bab|Di\s+antara)\b'       # instruction keyword
    r'[^\n]*)',                                  # rest of first line
    re.MULTILINE
)

def detect_elements(pages: list[dict], body_end_page: int,
                    is_perubahan: bool = False) -> list[dict]:
    """
    Scan all pages and detect structural elements with their positions.
    Only scans pages within the main body (before PENJELASAN/closing).

    When is_perubahan=True (amendment UU), additionally detects:
    - Roman numeral Pasal (Pasal I, II, ...) as level-0 root containers
    - Angka items (numbered amendment instructions) as level-1 containers
    Normal elements (BAB, Bagian, Paragraf, Arabic Pasal) also detected and
    nest under the Angka items: Pasal I > Angka > BAB > Bagian > Pasal.

    Returns a flat list of elements sorted by (page_num, char_offset).
    """
    elements = []

    for page in pages:
        # Skip pages after body ends (PENJELASAN, Lampiran, etc.)
        if page["page_num"] > body_end_page:
            continue

        text = page["clean_text"]
        page_num = page["page_num"]

        if is_perubahan:
            # Perubahan UUs: detect Roman numeral Pasal as root-level containers
            # (Pasal I, II, III, ...) at level 0 (above BAB).
            for m in PASAL_ROMAN.finditer(text):
                roman_str = m.group(1).strip()
                arabic_val = roman_to_int(roman_str)
                if arabic_val is None:
                    continue
                elements.append({
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
                # Truncate long instructions for title
                title_text = instruction[:80] + ("..." if len(instruction) > 80 else "")
                elements.append({
                    "type": "angka",
                    "level": LEVEL_MAP["angka"],
                    "number": angka_num,
                    "title": f"Angka {angka_num} — {title_text}",
                    "page_num": page_num,
                    "char_offset": m.start(),
                })
            # Fall through to also detect normal elements (BAB, Bagian, Paragraf,
            # Arabic Pasal) — these nest under the Angka items via the tree builder.

        # --- Normal element detection (runs for both regular and perubahan UUs) ---

        # Detect BAB
        for m in PATTERNS["bab"].finditer(text):
            heading_text = _clean_heading_title(m.group(2).strip())
            elements.append({
                "type": "bab",
                "level": LEVEL_MAP["bab"],
                "number": m.group(1).strip(),
                "title": f"BAB {m.group(1).strip()} - {heading_text}",
                "page_num": page_num,
                "char_offset": m.start(),
            })

        # Detect Bagian
        for m in PATTERNS["bagian"].finditer(text):
            heading_text = _clean_heading_title(m.group(2).strip())
            elements.append({
                "type": "bagian",
                "level": LEVEL_MAP["bagian"],
                "number": m.group(1).strip(),
                "title": f"Bagian {m.group(1).strip()} - {heading_text}",
                "page_num": page_num,
                "char_offset": m.start(),
            })

        # Detect Paragraf
        for m in PATTERNS["paragraf"].finditer(text):
            raw_num = m.group(1).strip()
            # Convert roman to arabic if needed
            roman_val = roman_to_int(raw_num)
            if roman_val is not None:
                num = str(roman_val)
            else:
                num = raw_num
            heading_text = _clean_heading_title(m.group(2).strip())
            elements.append({
                "type": "paragraf",
                "level": LEVEL_MAP["paragraf"],
                "number": num,
                "title": f"Paragraf {num} - {heading_text}",
                "page_num": page_num,
                "char_offset": m.start(),
            })

        # Detect Pasal
        for m in PATTERNS["pasal"].finditer(text):
            raw = m.group(1).strip()
            pasal_num, suffix = _parse_pasal_number(raw)
            if pasal_num is None:
                continue  # Not a valid Pasal number
            # Skip cross-references: if the word right before "Pasal X" is a
            # preposition/conjunction (e.g. "dimaksud dalam\nPasal 76D"), this
            # is a reference inside body text, not a section heading.
            preceding = text[:m.start()].rstrip()
            if preceding:
                last_word = preceding.split()[-1].lower().rstrip('.,;:)')
                if last_word in _CROSS_REF_PRECEDING:
                    continue
            title = f"Pasal {pasal_num}{suffix}"
            elements.append({
                "type": "pasal",
                "level": LEVEL_MAP["pasal"],
                "number": f"{pasal_num}{suffix}",
                "title": title,
                "page_num": page_num,
                "char_offset": m.start(),
            })

    # Sort by page_num, then char_offset
    elements.sort(key=lambda e: (e["page_num"], e["char_offset"]))

    # Deduplicate and filter
    deduped = []
    # Build set of Roman Pasal positions for overlap removal
    roman_positions = {(e["page_num"], e["char_offset"])
                       for e in elements if e["type"] == "pasal_roman"}
    # Find first Pasal I position — Angka items before it are false positives
    # (e.g., "1. Pasal 20, Pasal 21..." in the Mengingat/preamble section)
    first_roman = next(((e["page_num"], e["char_offset"])
                        for e in elements if e["type"] == "pasal_roman"), None)
    for elem in elements:
        # Skip Angka items that appear before Pasal I (preamble false positives)
        if elem["type"] == "angka" and first_roman:
            if (elem["page_num"], elem["char_offset"]) < first_roman:
                continue
        # Skip Arabic Pasal that overlaps with a Roman Pasal at the same position.
        # Both regexes can match "Pasal I" — Roman sees it as Roman numeral,
        # Arabic sees "I" as OCR'd "1". Roman wins in perubahan UUs.
        if elem["type"] == "pasal" and (elem["page_num"], elem["char_offset"]) in roman_positions:
            continue
        # Same Pasal number on adjacent pages = likely page-break continuation
        if deduped and elem["type"] == "pasal" and deduped[-1]["type"] == "pasal":
            if (elem["number"] == deduped[-1]["number"]
                    and elem["page_num"] - deduped[-1]["page_num"] <= 1):
                continue
        deduped.append(elem)

    return deduped

# ============================================================
# 4. PASAL NUMBERING VALIDATION
# ============================================================

def validate_pasal_sequence(elements: list[dict],
                           is_perubahan: bool = False) -> list[str]:
    """
    Check that Pasal numbers are monotonically increasing.
    Returns list of warnings.

    Skips validation for Perubahan UUs — amendment laws have non-monotonic
    Pasal numbers by nature (Roman numeral articles I-IX, with arbitrary
    referenced article numbers in between).
    """
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
    """
    Assign start_index, end_index, and end_char_offset to each element.

    Each element's end_index is set to the page where the next same-or-higher
    level element begins. end_char_offset enables intra-page slicing when
    multiple elements share a page.

    Mutates elements in place.
    """
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
                # Next element starts on our end page — slice there
                elem["end_char_offset"] = elements[j]["char_offset"]
                break
            elif elements[j]["page_num"] > elem["end_index"]:
                break

def build_tree(elements: list[dict], total_pages: int) -> list[dict]:
    """
    Convert flat list of elements into a nested tree structure.

    First assigns page boundaries (start/end indices) to each element,
    then uses a stack-based approach where each element becomes a child
    of the most recent element with a higher level (lower number).

    Args:
        elements: flat list from detect_elements(), sorted by position.
        total_pages: total page count in the document body.

    Returns:
        list of root-level tree nodes, each with nested "nodes" children.
    """
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
    """
    Refine node boundaries after initial tree construction.

    Adjusts end_index so that:
    - Each node ends where its next sibling begins (no overlap)
    - The last sibling extends to its parent's end_index
    - No node ends before it starts
    - Parent expands if children extend beyond it

    Also sets end_char_offset for intra-page slicing between siblings.
    The last child inherits its parent's end_char_offset to prevent
    text from the next section bleeding into the last node.
    """
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
            # Propagate parent's end_char_offset to last child so it doesn't capture text from the next section
            node["end_index"] = parent_end
            node["end_char_offset"] = parent_end_char_offset

        if node["nodes"]:
            fix_node_boundaries(node["nodes"], node["end_index"], node.get("end_char_offset"))
            child_max = max(c["end_index"] for c in node["nodes"])
            node["end_index"] = max(node["end_index"], child_max)

        node["end_index"] = max(node["end_index"], node["start_index"])

def consolidate_bab_in_perubahan(tree: list[dict]) -> int:
    """
    Post-process a perubahan tree: move orphaned Pasals under their BAB.

    When an amendment inserts a new BAB, the BAB heading and its Pasals appear
    in separate Angka siblings. This finds BAB-leaf nodes and absorbs Pasals
    from the immediately following Angka sibling.

    Only handles explicit BAB insertions ("disisipkan ... bab").
    Mutates tree in place. Returns count of Pasals moved.
    """
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

def _iter_leaves(nodes: list[dict]):
    """Yield all leaf nodes from a tree (nodes without children)."""
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            yield from _iter_leaves(node["nodes"])
        else:
            yield node

def clean_tree_for_output(nodes: list[dict], pages: list[dict] | None = None) -> list[dict]:
    """
    Remove internal fields, keep only PageIndex-compatible fields.
    If pages is provided, embed text content in leaf nodes.
    """
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
            # Fix (Parser Fix 11): pasal_roman nodes in perubahan UUs can have
            # their own substantive text (e.g., ayats) before their first child
            # element (BAB heading, Arabic Pasal, Angka). The standard leaf-only
            # text extraction loses this content. Recover it by extracting the
            # text between the node's start and the first child's start position.
            if pages is not None and node.get("type") == "pasal_roman":
                first_child = node["nodes"][0]
                if first_child["start_index"] == node["start_index"]:
                    # Child starts on the same page — extract up to child's char_offset
                    own_end_page = node["start_index"]
                    own_end_char = first_child.get("start_char_offset") or None
                else:
                    # Child starts on a later page — extract pages up to the page before it
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
            # Leaf node — embed text with intra-page slicing
            clean["text"] = _extract_node_text(
                pages, node["start_index"], node["end_index"],
                start_char_offset=node.get("start_char_offset", 0),
                end_char_offset=node.get("end_char_offset"),
            )
        # Propagate penjelasan if attached (set by _attach_penjelasan)
        if "penjelasan" in node:
            clean["penjelasan"] = node["penjelasan"]
        # Propagate source_angka for Pasals moved by BAB consolidation
        if "_moved_from_angka" in node:
            clean["source_angka"] = node["_moved_from_angka"]
        result.append(clean)
    return result

def _split_preamble(text: str, start_page: int, end_page: int) -> list[dict] | None:
    """Split preamble text into Menimbang, Mengingat, and Menetapkan sub-nodes.

    Indonesian legal documents have a standard preamble structure:
      [Header] → Menimbang (a. bahwa...; b. bahwa...; ...)
               → Mengingat (Pasal/UU references)
               → MEMUTUSKAN: Menetapkan: ...

    Strategy: find section boundaries BEFORE stripping keywords, then use
    "Mengingat" keyword as primary boundary (fallback to content transition).

    Returns list of child nodes, or None if splitting fails (fallback to blob).
    """
    # Step 1: Clean OCR noise — but preserve Mengingat/Menetapkan keywords
    # so they can be used as section boundaries first.
    _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN)'
    noise_patterns = [
        # "Mengingat\nMenetapkan" combo = OCR margin bleed (both in margin column)
        re.compile(r'^\s*Mengingat\s*\n\s*Menetapkan\s*$', re.MULTILINE),
        # PRESIDEN REPUBLIK INDONESIA header bleed
        re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s*\n\s*REP\S*\s+IND\S*\s*\n'),
        re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s*\n'),
        # SALINAN/SATINAN watermark
        re.compile(r'^\s*SAL[IT]NAN\s*\n', re.MULTILINE),
        # Page numbers like "-2-", "-3-"
        re.compile(r'^\s*-\d+-\s*\n', re.MULTILINE),
        # Random OCR garbage (short all-caps nonsense, but NOT Mengingat/Menetapkan)
        re.compile(r'^\s*(?!Mengingat|Menetapkan)[A-Z]{3,8}\s*\n', re.MULTILINE),
    ]
    cleaned = text
    for pat in noise_patterns:
        cleaned = pat.sub('\n', cleaned)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    # Step 2: Find Menimbang start
    menimbang_m = re.search(r'Menimbang\s*:?\s*(?:\n|(?=a[\.\s]))', cleaned)
    if menimbang_m:
        before_menimbang = cleaned[:menimbang_m.start()]
        after_menimbang = cleaned[menimbang_m.end():]
        # Fix (Parser Fix 10): In some PDFs (e.g., perpu-2-2022, uu-1-2026), PyMuPDF
        # block sorting by (y0, x0) places the narrow "Menimbang :" label block AFTER
        # the bahwa item blocks in the sorted text, so "a. bahwa", "b. bahwa", etc.
        # appear BEFORE the "Menimbang" keyword.  _split_preamble() then slices from
        # menimbang_m.end(), silently discarding items a–d.
        # Recovery: if "a. bahwa" appears in the prefix (before_menimbang), prepend
        # the orphaned bahwa items to the front of after_menimbang.
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

    # Step 4: Split Menimbang from Mengingat
    # Priority: content-based detection first (most reliable), keyword fallback.
    # OCR margin keywords ("Mengingat") can appear at wrong positions, so we
    # prefer detecting the content transition after the last "bahwa" item.
    menimbang_text = None
    mengingat_text = None

    # Strategy A: content transition after last "bahwa" semicolon
    # Menimbang = "bahwa" items ending with ";", Mengingat = legal references
    last_bahwa_end = None
    for m in re.finditer(r'bahwa\s', preamble_body):
        last_bahwa_end = m.end()

    if last_bahwa_end:
        after_last = preamble_body[last_bahwa_end:]
        # Find the LAST semicolon in the final bahwa item (the item may
        # reference laws with intermediate semicolons, e.g. "(Lembaran...;")
        # so we need the semicolon that ends the whole bahwa clause.
        # Search for semicolon followed by newline, then check what comes next.
        for semi_m in re.finditer(r';\s*\n', after_last):
            split_pos = last_bahwa_end + semi_m.end()
            remaining = preamble_body[split_pos:]
            # Check if remaining starts with legal references (Mengingat content)
            # or is very short garbage (OCR noise before actual Mengingat)
            ref_m = re.match(
                r'\s*(?:\d+[\.\s]+)?(?:Pasal|Undang|Peraturan)',
                remaining,
            )
            if ref_m:
                menimbang_text = preamble_body[:split_pos].strip()
                mengingat_text = remaining.strip()
                break

    # Strategy B: "Mengingat" keyword (only if Strategy A didn't find a split)
    if menimbang_text is None:
        mengingat_kw = re.search(r'\n\s*Mengingat\s*:?\s*\n', preamble_body)
        if mengingat_kw:
            menimbang_text = preamble_body[:mengingat_kw.start()].strip()
            mengingat_text = preamble_body[mengingat_kw.end():].strip()

    # Strategy C (legacy fallback): first Pasal ref at start of line
    if menimbang_text is None:
        mengingat_m = re.search(
            r'\n((?:\d+\.\s+)?Pasal\s+\d+)',
            preamble_body,
        )
        if mengingat_m:
            menimbang_text = preamble_body[:mengingat_m.start()].strip()
            mengingat_text = preamble_body[mengingat_m.start():].strip()

    # No split found — keep all as Menimbang
    if menimbang_text is None:
        menimbang_text = preamble_body
        mengingat_text = None

    # Step 5: Clean up and build children
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
        mengingat_text = re.sub(r'\n{3,}', '\n\n', mengingat_text).strip()
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
    """
    Extract and join cleaned text from pages [start, end] (1-indexed, inclusive).
    Uses char offsets for intra-page slicing to avoid text bleeding between
    adjacent Pasal on the same page.
    """
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

# Max chars per batch (~100K chars ≈ ~25K tokens). Smaller batches = more reliable
# JSON output from Gemini. Speed comes from parallel processing, not larger batches.
_LLM_BATCH_SIZE = 100_000

def _process_batch(client, batch: dict[str, str], batch_idx: int, total: int, verbose: bool):
    """Process a single batch through Gemini with retry on rate limit."""
    prompt = LLM_CLEANUP_PROMPT + json.dumps(batch, ensure_ascii=False)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
            )
            break
        except Exception as e:
            err_str = str(e).lower()
            if ("rate" in err_str or "quota" in err_str or "429" in err_str) and attempt < max_retries - 1:
                wait = 30 * (attempt + 1)
                if verbose:
                    print(f"\n       Batch {batch_idx + 1}: rate limited, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    # Track token usage
    usage = response.usage_metadata
    input_tok = usage.prompt_token_count or 0
    output_tok = usage.candidates_token_count or 0
    if verbose:
        print(f"       Batch {batch_idx + 1}/{total} done: "
              f"{input_tok + output_tok:,} tokens ({input_tok:,} in, {output_tok:,} out)")

    # Parse JSON from response
    response_text = response.text.strip()
    # Strip markdown code fences if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        cleaned = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"       WARNING: Failed to parse LLM response for batch {batch_idx + 1}: {e}")
        cleaned = batch  # Fall back to original texts

    return cleaned, input_tok, output_tok

def llm_cleanup_texts(texts: list[tuple[str, str]], verbose: bool = True) -> dict[str, str]:
    """
    Clean OCR artifacts in texts using Gemini 2.5 Flash.
    Batches are processed in parallel for speed.

    Args:
        texts: list of (node_id, text) tuples
        verbose: print progress info

    Returns:
        dict mapping node_id to cleaned text
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    client = genai.Client(api_key=api_key)

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
        print(f"       {len(texts)} nodes in {len(batches)} batch(es), processing in parallel...")

    # Process batches in parallel
    results: dict[str, str] = {}
    total_input_tokens = 0
    total_output_tokens = 0

    with ThreadPoolExecutor(max_workers=min(len(batches), 4)) as executor:
        futures = {
            executor.submit(_process_batch, client, batch, i, len(batches), verbose): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            cleaned, input_tok, output_tok = future.result()
            results.update(cleaned)
            total_input_tokens += input_tok
            total_output_tokens += output_tok

    if verbose:
        print(f"       Total tokens: {total_input_tokens + total_output_tokens:,} "
              f"({total_input_tokens:,} in, {total_output_tokens:,} out)")

    return results

def _apply_llm_cleanup(output_nodes: list[dict],
                       penjelasan_data: dict | None = None,
                       verbose: bool = True):
    """
    Collect leaf node texts and penjelasan from the output tree,
    clean them with LLM, and replace texts in-place.
    """
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
            print("       No texts to clean.")
        return

    # Call LLM cleanup
    cleaned = llm_cleanup_texts(texts_to_clean, verbose=verbose)

    # Check for missing nodes
    expected_ids = {nid for nid, _ in texts_to_clean}
    missing = expected_ids - set(cleaned.keys())
    if missing and verbose:
        print(f"       WARNING: {len(missing)} node(s) missing from LLM response, keeping original: {sorted(missing)}")

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
        print(f"       Cleaned {len(cleaned)} nodes.")

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


def _strip_ocr_headers(nodes: list[dict]):
    """Strip residual PDF header/footer text from all leaf node texts in-place."""
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            _strip_ocr_headers(node["nodes"])
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

def parse_legal_pdf(pdf_path: str, verbose: bool = True,
                    use_llm_cleanup: bool = True,
                    is_perubahan: bool | None = None,
                    granularity: str = "pasal") -> dict:
    """
    Main function: parse an Indonesian legal PDF into PageIndex-compatible tree.

    Uses Gemini 2.5 Flash to fix remaining OCR artifacts in extracted text.
    Pass use_llm_cleanup=False to skip (dev/testing only).

    is_perubahan: Force Perubahan (amendment) mode. When None, auto-detects
    from the PDF title page. Perubahan UUs use Roman numeral articles
    (Pasal I, II, ...) as root-level containers, with the amended law's
    structure (BAB, Bagian, Pasal) nested inside.

    granularity: Leaf node granularity level.
        - "pasal":      leaf = entire Pasal (no sub-splitting)
        - "ayat":       leaf = Ayat within each Pasal
        - "full_split": leaf = smallest unit (Ayat > Huruf > Angka)
    """
    if granularity not in ("pasal", "ayat", "full_split"):
        raise ValueError(f"Unknown granularity: {granularity!r}. "
                         f"Use 'pasal', 'ayat', or 'full_split'.")
    pdf_path = str(pdf_path)
    pdf_name = Path(pdf_path).name
    total_steps = 6 if use_llm_cleanup else 5
    t_start = time.time()

    # Step 1: Extract text
    if verbose:
        print(f"[1/{total_steps}] Extracting text from {pdf_name}...")
    t0 = time.time()
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    if verbose:
        print(f"       Total pages: {total_pages} ({time.time() - t0:.1f}s)")

    # Step 2: Clean text
    if verbose:
        print(f"[2/{total_steps}] Cleaning text...")
    t0 = time.time()
    for page in pages:
        page["clean_text"] = clean_page_text(page["raw_text"])
    if verbose:
        print(f"       Done ({time.time() - t0:.1f}s)")

    # Step 3: Detect body boundaries
    if verbose:
        print(f"[3/{total_steps}] Detecting document sections...")
    closing_page = find_closing_page(pages)
    penjelasan_page = find_penjelasan_page(pages)

    # Body ends at whichever comes first: closing or penjelasan
    boundaries = [p for p in [closing_page, penjelasan_page] if p]
    if boundaries:
        body_end = min(boundaries) - 1
    else:
        body_end = total_pages

    if verbose:
        print(f"       Body: pages 1-{body_end}")
        if closing_page:
            print(f"       Closing/Pengesahan: page {closing_page}")
        if penjelasan_page:
            print(f"       Penjelasan: pages {penjelasan_page}-{total_pages}")

    # Auto-detect Perubahan (amendment) UU if not specified
    if is_perubahan is None:
        is_perubahan = detect_perubahan(pages)
    if verbose and is_perubahan:
        print(f"       Detected as Perubahan (amendment) UU")

    # Step 4: Detect structural elements (only in body)
    if verbose:
        print(f"[4/{total_steps}] Detecting structural elements...")
    t0 = time.time()
    elements = detect_elements(pages, body_end, is_perubahan=is_perubahan)

    type_counts = {}
    for e in elements:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    if verbose:
        print(f"       Found: {type_counts} ({time.time() - t0:.1f}s)")

    # Validate Pasal sequence (skip for Perubahan and omnibus)
    is_omnibus = detect_omnibus(pages, elements)
    if verbose and is_omnibus:
        print(f"       Detected as Omnibus law")
    warnings = validate_pasal_sequence(elements, is_perubahan=is_perubahan or is_omnibus)
    for w in warnings:
        print(f"       {w}")

    # Step 5: Build tree
    if verbose:
        print(f"[5/{total_steps}] Building tree structure...")
    t0 = time.time()
    tree = build_tree(elements, body_end)
    fix_node_boundaries(tree, body_end)

    # Post-process: consolidate inserted BABs in perubahan documents
    if is_perubahan:
        bab_moved = consolidate_bab_in_perubahan(tree)
        if verbose and bab_moved:
            print(f"       Consolidated {bab_moved} Pasal(s) under inserted BAB(s)")

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
        _attach_penjelasan(output_nodes, penjelasan_data["pasal"])
        matched = sum(1 for n in _iter_leaves(output_nodes) if n.get("penjelasan"))
        if verbose:
            print(f"       PENJELASAN: {len(penjelasan_data['pasal'])} pasal parsed, "
                  f"{matched} matched to tree nodes")

    if verbose:
        print(f"       Done ({time.time() - t0:.1f}s)")

    # Step 6: LLM text cleanup (on by default, skip with --no-llm)
    llm_time = 0.0
    if use_llm_cleanup:
        if verbose:
            print(f"[6/{total_steps}] Cleaning text with Gemini 2.5 Flash...")
        t0 = time.time()
        _apply_llm_cleanup(output_nodes, penjelasan_data, verbose=verbose)
        llm_time = time.time() - t0
        if verbose:
            print(f"       LLM cleanup took {llm_time:.1f}s")

    total_time = time.time() - t_start
    if verbose:
        print(f"\n       Total time: {total_time:.1f}s", end="")
        if use_llm_cleanup:
            print(f" (LLM: {llm_time:.1f}s, parser: {total_time - llm_time:.1f}s)")
        else:
            print()

    # Sub-Pasal splitting based on requested granularity.
    if granularity == "ayat":
        output_nodes = _ayat_split_leaves(output_nodes)
    elif granularity == "full_split":
        output_nodes = _deep_split_leaves(output_nodes)

    # Final pass: strip residual OCR header/footer leaks from all leaf texts
    _strip_ocr_headers(output_nodes)

    # Store unmatched penjelasan at doc level for retrieval agent fallback.
    # With perubahan support, amended Pasals are now leaf nodes and should
    # match normally — only fall back to doc-level if nothing matched.
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


# --- Ayat-only splitting (granularity="ayat") ---

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


def _ayat_split_leaves(nodes: list[dict]) -> list[dict]:
    """Walk the tree and split every leaf node into Ayat sub-nodes only.

    Does NOT recurse deeper into Huruf/Angka. If a Pasal has no Ayat markers,
    it stays as a leaf unchanged.
    """
    result = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            node["nodes"] = _ayat_split_leaves(node["nodes"])
            result.append(node)
        elif "text" in node:
            sub_nodes = _try_ayat_split(
                text=node["text"],
                parent_id=node["node_id"],
                parent_title=node["title"],
                parent_start=node["start_index"],
                parent_end=node["end_index"],
                penjelasan=node.get("penjelasan"),
            )
            if sub_nodes:
                branch = {k: v for k, v in node.items() if k not in ("text", "penjelasan")}
                branch["nodes"] = sub_nodes
                result.append(branch)
            else:
                result.append(node)
        else:
            result.append(node)
    return result


# --- Recursive deep splitting (granularity="full_split") ---

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

    # --- Try Ayat: (1), (2), (3), ... ---
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

    # --- Try Huruf: a., b., c., ... ---
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

    # --- Try Angka item: 1., 2., 3., ... ---
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


def _deep_split_leaves(nodes: list[dict]) -> list[dict]:
    """Walk the tree and recursively split every leaf node to its deepest sub-structure.

    Tries Ayat → Huruf → Angka. If no sub-structure found, leaf stays unchanged.
    """
    result = []
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            node["nodes"] = _deep_split_leaves(node["nodes"])
            result.append(node)
        elif "text" in node:
            sub_nodes = _try_deep_split(
                text=node["text"],
                parent_id=node["node_id"],
                parent_title=node["title"],
                parent_start=node["start_index"],
                parent_end=node["end_index"],
                penjelasan=node.get("penjelasan"),
            )
            if sub_nodes:
                branch = {k: v for k, v in node.items() if k not in ("text", "penjelasan")}
                branch["nodes"] = sub_nodes
                result.append(branch)
            else:
                result.append(node)
        else:
            result.append(node)
    return result


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
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
            print("ERROR: --granularity requires a value (pasal, ayat, full_split)")
            sys.exit(1)

    if gran not in ("pasal", "ayat", "full_split"):
        print(f"ERROR: Unknown granularity '{gran}'. Use: pasal, ayat, full_split")
        sys.exit(1)

    if use_llm and not os.environ.get("GEMINI_API_KEY"):
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        print("       Set it for LLM cleanup, or use --no-llm to skip.")
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
            print("No PDF files found. Provide a path as argument.")
            sys.exit(1)
    else:
        pdf_files = [Path(pdf_arg)]

    for pdf_path in pdf_files:
        print(f"\n{'='*70}")
        print(f" Processing: {pdf_path.name}")
        print(f"{'='*70}")

        result = parse_legal_pdf(str(pdf_path), use_llm_cleanup=use_llm,
                                granularity=gran)

        print(f"\nTREE STRUCTURE ({gran}):")
        print("-" * 60)
        print_tree(result["structure"])

        # Save output
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / (pdf_path.stem + "_structure.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\nSaved to: {output_path}")