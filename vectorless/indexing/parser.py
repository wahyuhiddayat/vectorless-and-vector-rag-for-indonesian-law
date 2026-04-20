import re
import json
import logging
import os
import sys
import time
import warnings
import fitz
from pathlib import Path

log = logging.getLogger(__name__)

# ============================================================
# 1. TEXT EXTRACTION & CLEANING
# ============================================================

def _detect_two_columns(blocks: list[dict], page_width: float, is_landscape: bool = False) -> list[dict]:
    """Reorder text blocks for correct reading order on landscape gazette pages.

    Portrait pages and pages with fewer than 4 blocks are returned sorted by
    top-to-bottom, left-to-right position. Landscape pages use a left-column-first
    ordering when both halves are sufficiently populated and fewer than 30% of
    blocks span the full page width (i.e. the page is genuinely two-column).
    """
    if len(blocks) < 4 or not is_landscape:
        return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))

    midpoint = page_width / 2

    left = []
    right = []
    # Classify each block as left or right column by its horizontal center.
    for b in blocks:
        center_x = (b["x0"] + b["x1"]) / 2
        if center_x < midpoint:
            left.append(b)
        else:
            right.append(b)

    # Full-width blocks (>60% of page width) are headers or titles, not column content.
    wide_blocks = [b for b in blocks if (b["x1"] - b["x0"]) > page_width * 0.6]

    # Treat as two-column only when both halves are populated and wide blocks are rare.
    if len(left) >= 3 and len(right) >= 3 and len(wide_blocks) < len(blocks) * 0.3:
        left_sorted = sorted(left, key=lambda b: (b["y0"], b["x0"]))
        right_sorted = sorted(right, key=lambda b: (b["y0"], b["x0"]))
        return left_sorted + right_sorted

    return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))


def _extract_page_text(page) -> str:
    """Extract text from a single PyMuPDF page, preserving column reading order.

    Tables are extracted as pipe-delimited rows instead of jumbled sequential
    text blocks. Requires PyMuPDF >= 1.23 for find_tables(); falls back to
    plain text extraction on older versions or pages with no detectable tables.
    """
    page_dict = page.get_text("dict")
    page_width = page_dict.get("width", 595)  # A4 default
    page_height = page_dict.get("height", 842)
    raw_blocks = page_dict.get("blocks", [])

    # Detect tables and build structured replacement text blocks.
    table_blocks: list[dict] = []
    table_rects: list[fitz.Rect] = []
    try:
        for tbl in page.find_tables():
            tbl_rect = fitz.Rect(tbl.bbox)
            table_rects.append(tbl_rect)
            rows = tbl.extract()  # list[list[str | None]]
            lines = []
            for row in rows:
                cells = [(cell or "").replace("\n", " ").strip() for cell in row]
                lines.append(" | ".join(cells))
            tbl_text = "\n".join(lines)
            if tbl_text.strip():
                table_blocks.append({
                    "x0": tbl.bbox[0], "y0": tbl.bbox[1],
                    "x1": tbl.bbox[2], "y1": tbl.bbox[3],
                    "text": tbl_text + "\n",
                })
    except Exception:
        pass  # graceful fallback: no table detection, proceed as plain text

    text_blocks = []
    # Collect text blocks with their bounding boxes, skipping image blocks
    # and any blocks whose region is already covered by a detected table.
    for b in raw_blocks:
        if b.get("type") != 0:  # 0 = text, 1 = image
            continue
        if table_rects:
            b_rect = fitz.Rect(b["bbox"])
            if any(tbl_rect.intersects(b_rect) for tbl_rect in table_rects):
                continue  # this block belongs to a table — skip raw text
        block_text = ""
        # Join all spans in each line, then append a newline after the line.
        for line in b.get("lines", []):
            line_text = "".join(span["text"] for span in line.get("spans", []))
            block_text += line_text + "\n"
        if block_text.strip():
            text_blocks.append({
                "x0": b["bbox"][0], "y0": b["bbox"][1],
                "x1": b["bbox"][2], "y1": b["bbox"][3],
                "text": block_text,
            })

    all_blocks = text_blocks + table_blocks
    if not all_blocks:
        return ""

    is_landscape = page_width > page_height

    ordered = _detect_two_columns(all_blocks, page_width, is_landscape=is_landscape)
    return "".join(b["text"] for b in ordered)


def extract_pages(pdf_path: str) -> list[dict]:
    """Extract raw text from every page of a PDF using PyMuPDF.

    Returns a list of dicts with keys: page_num (1-indexed), raw_text.
    """
    pages = []
    with fitz.open(pdf_path) as doc:
        # Extract text from each page in document order.
        for i, page in enumerate(doc):
            text = _extract_page_text(page)
            pages.append({
                "page_num": i + 1,
                "raw_text": text,
            })
    return pages


def clean_page_text(text: str) -> str:
    """Remove recurring OCR noise from Indonesian legal PDF page text.

    Handles: PRESIDEN REPUBLIK INDONESIA header variants (multi-line and single-line),
    font-encoding garbage strings, page number markers, SK No footers, glued Pasal/BAB
    headings, whitespace normalization, and digit OCR artifacts.
    """
    # Multi-line PRESIDEN REPUBLIK INDONESIA header (OCR variants + optional SALINAN prefix).
    text = re.sub(
        r'(?:SALINAN\s*)?(?:Menimbang\s*)?'
        r'(?:FRESIDEN|PRESIDEN|PNESIDEN|FTjTJTFiTIilNEEtrtrEIn!)\s*\n'
        r'\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|NEPUBUK)\s+'
        r'(?:TNDONESIA|INDONESIA|INDONESI,A)',
        '', text
    )
    # Standalone garbage strings produced by font-encoding errors.
    text = re.sub(r'(?:LIrtrEIEtrN|iIitrEIEtrN|;?\*trEIEtrN|FTjTJTFiTIilNEEtrtrEIn!)', '', text)
    # Single-line PRESIDEN variants.
    text = re.sub(r'^\s*(?:FRESIDEN|PRESIDEN|PNESIDEN|PRESTDEN)\s*$', '', text, flags=re.MULTILINE)
    # Single-line REPUBLIK INDONESIA variants.
    text = re.sub(r'^\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|NEPUBUK)\s+(?:TNDONESIA|INDONESIA|INDONESI,A)\s*$', '', text, flags=re.MULTILINE)
    # Additional truncated/partial OCR header variants.
    text = re.sub(r'^\s*(?:R,EPUBLIK|REPIJBUK|REPI,IBLIK|REP[A-Z]*K)\s+(?:INDONES[A-Z,]*)\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:EUK|ELIK|BUK)\s+INDONESIA\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:PTTESIDEN|PRESTDEN)\s*$', '', text, flags=re.MULTILINE)

    # Page number markers: "-2-", "- 10 -", "-t2-" (OCR'd "t" for digit 1).
    text = re.sub(r'\n\s*-\s*[t]?\d+\s*-?\s*\n', '\n', text)
    # SK No footer: "SK No 273836A".
    text = re.sub(r'SK\s+No\s*\d+\s*A.*$', '', text, flags=re.MULTILINE)

    # Glued Pasal heading: "diperolehPasal 2" → "diperoleh\nPasal 2".
    text = re.sub(r'([^\s])(?=Pasal\s+\d)', r'\1\n', text)
    # Missing space: "Pasal22" → "Pasal 22".
    text = re.sub(r'^(Pasal)(\d)', r'\1 \2', text, flags=re.MULTILINE)
    # Glued BAB heading: "sebelumnyaBAB IV" → "sebelumnya\nBAB IV".
    text = re.sub(r'([^\n])(?=BAB\s+[IVXLCDM])', r'\1\n', text)

    # Normalize whitespace.
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Fix digit OCR artifacts in ayat numbers (O→0, l→1, I→1).
    text = fix_ocr_artifacts(text)

    return text.strip()


def fix_ocr_artifacts(text: str) -> str:
    """Fix OCR digit misreads in ayat numbers, list prefixes, and page continuation markers."""
    lines = text.split('\n')
    fixed_lines = []

    # Apply fixes line by line to avoid accidental cross-line substitutions.
    for line in lines:
        # Ayat numbers inside parens: "(2l)" → "(21)", "(l)" → "(1)".
        line = re.sub(
            r'\(([0-9OlI]+)\)',
            lambda m: '(' + _normalize_ocr_digits(m.group(1)) + ')',
            line
        )
        # Malformed parens where closing paren was replaced or dropped: "(2t" → "(2)".
        line = re.sub(r'\((\d+)[t]\b', lambda m: '(' + m.group(1) + ')', line)
        line = re.sub(r'\b[tl](\d+)[tl]\b', lambda m: '(' + m.group(1) + ')', line)
        line = re.sub(r'\(s\)', '(5)', line)
        # Numbered list prefixes with OCR-garbled trailing digit: "1O." → "10.", "2l." → "21.".
        # Capital-O and lowercase-l are common OCR misreads of 0 and 1 respectively when they
        # follow another digit at the start of a list-item line.
        line = re.sub(
            r'^(\d+[OlI]+)\.',
            lambda m: _normalize_ocr_digits(m.group(1)) + '.',
            line
        )
        # Page continuation markers: "Pasal 7...", "(3)DBH . , ."
        line = re.sub(r'^Pasa[lr]\s*\d+\s*\.{2,}\s*$', '', line)
        line = re.sub(r'\.\s*\.\s*\.\s*$', '', line)

        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


def _normalize_ocr_digits(s: str) -> str:
    """Normalize OCR-misread characters in a potential digit string.

    Maps: O → 0, l → 1, I → 1. Strips spaces before normalizing (handles "9 I").
    Returns the normalized string only if the result is purely numeric;
    otherwise returns the original string unchanged to avoid false substitutions.
    """
    s_nospace = s.replace(' ', '')
    normalized = ''
    # Replace each OCR-corrupted letter with its correct digit.
    for ch in s_nospace:
        if ch == 'O':
            normalized += '0'
        elif ch == 'l':
            normalized += '1'
        elif ch == 'I':
            normalized += '1'
        else:
            normalized += ch

    if normalized.isdigit():
        return normalized
    return s


def _parse_pasal_number(raw: str) -> tuple[str | None, str]:
    """Parse a raw Pasal number string into (number, suffix), handling OCR artifacts.

    Distinguishes between a real letter suffix (e.g. "119I" in perubahan UUs)
    and an OCR-misread digit (e.g. "4O" meaning 40, "9I" meaning 91).
    Returns (None, "") when the string cannot be interpreted as a valid number.
    """
    s = raw.replace(' ', '')  # strip spaces ("9 I" → "9I")

    if not s:
        return None, ""

    if s[-1].isalpha():
        last_char = s[-1].upper()
        body = s[:-1]

        # Try both interpretations: last char as suffix vs. as part of the number.
        num_without = _normalize_ocr_digits(body)
        num_with = _normalize_ocr_digits(s)

        if num_without.isdigit() and num_with.isdigit():
            # Both are valid numbers; use context to decide.
            # 'O' at end is OCR'd '0' (no suffix O exists in Indonesian law).
            # Lowercase 'l' at end is OCR'd '1' (suffixes are always uppercase).
            if last_char == 'O' or s[-1] == 'l':
                return num_with, ""
            elif last_char == 'I' and len(num_without) >= 3:
                # "119I" is a real suffix (Pasal 119I in perubahan UUs).
                # Short forms like "19I" or "9I" are OCR for 191 / 91.
                return num_without, last_char
            else:
                return num_with, ""
        elif num_without.isdigit():
            # Unambiguous suffix: "599A" → (599, "A").
            return num_without, last_char
        elif num_with.isdigit():
            # Unambiguous OCR digit: "4O" → (40, "").
            return num_with, ""
        else:
            return None, ""
    else:
        normalized = _normalize_ocr_digits(s)
        if normalized.isdigit():
            return normalized, ""
        return None, ""

# ============================================================
# 2. PENJELASAN DETECTION & PARSING
# ============================================================

def find_penjelasan_page(pages: list[dict]) -> int | None:
    """Return the page number where PENJELASAN starts, or None if not found."""
    # Use raw text — PENJELASAN headers are usually clean OCR.
    for page in pages:
        text = page["raw_text"]
        if re.search(r'PENJ\S*SAN\s*\n\s*ATAS', text):
            return page["page_num"]
    return None


def find_closing_page(pages: list[dict]) -> int | None:
    """Return the page number of the pengesahan (closing) section, or None if not found."""
    for page in pages:
        text = page["raw_text"]
        if re.search(r'Di(?:sahkan|tetapkan) di Jakarta|Agar setiap orang mengetahuinya', text):
            return page["page_num"]
    return None


def detect_perubahan(pages: list[dict]) -> bool:
    """Return True if the document is a Perubahan (amendment) UU/PP/Perpres.

    Detection is title-based, not Pasal-number-based. PyMuPDF often renders
    "Pasal I" as "Pasal 1" due to font encoding, making Roman-numeral Pasal
    detection unreliable as a signal.
    """
    if not pages:
        return False

    # Scan the first 3 pages to handle garbled page order from OCR.
    for page in pages[:3]:
        text = page["raw_text"]
        # \s* after TENTANG handles OCR that merges "TENTANGPERUBAHAN" without a space.
        title_m = re.search(r'TENTANG\s*(.+?)DENGAN\s+RAHMAT', text,
                            re.DOTALL | re.IGNORECASE)
        if not title_m:
            continue

        title_text = title_m.group(1)

        # Explicit keyword in title.
        if re.search(r'PERUBAHAN|YESUAIAN', title_text, re.IGNORECASE):
            return True

        # "ATAS UNDANG-UNDANG" / "ATAS PERATURAN" catches cases where OCR dropped "PERUBAHAN".
        if re.search(r'ATAS\s+UNDANG|ATAS\s+PERATURAN', title_text, re.IGNORECASE):
            return True

    return False


def detect_omnibus(pages: list[dict], elements: list[dict]) -> bool:
    """Return True if the document is an omnibus law (e.g. UU Cipta Kerja).

    Omnibus laws skip Pasal sequence validation because their Pasal numbering
    is non-linear by design. Detection uses two signals: the document title
    and a Pasal count heuristic.
    """
    # Check title text for "CIPTA KERJA". Match must be in the title, not in
    # preamble references to other laws (hence the TENTANG...DENGAN RAHMAT bound).
    for page in pages[:3]:
        title_m = re.search(r'TENTANG\s*(.+?)DENGAN\s+RAHMAT', page["raw_text"],
                            re.DOTALL | re.IGNORECASE)
        if title_m and re.search(r'CIPTA\s*KERJA', title_m.group(1), re.IGNORECASE):
            return True
    # Heuristic: >500 Pasals strongly indicates an omnibus structure.
    pasal_count = sum(1 for e in elements if e["type"] == "pasal")
    return pasal_count > 500


def parse_penjelasan(pages: list[dict], penjelasan_page: int, total_pages: int) -> dict:
    """Parse the PENJELASAN section and return structured explanation text.

    Args:
        pages: Page dicts with page_num, raw_text, and clean_text fields.
        penjelasan_page: Page number where PENJELASAN begins.
        total_pages: Last page to include (exclusive of closing material).

    Returns:
        {"umum": str, "pasal": {pasal_number: explanation_text}}.
        If no PASAL DEMI PASAL heading is found, "pasal" is an empty dict.
    """
    parts = []
    # Collect cleaned text from PENJELASAN pages only.
    for page in pages:
        if page["page_num"] < penjelasan_page:
            continue
        if page["page_num"] > total_pages:
            break
        parts.append(page["clean_text"])
    full_text = "\n\n".join(parts)

    # Split into UMUM and PASAL DEMI PASAL sections.
    # "II." prefix is optional — some shorter UUs omit it or OCR drops it.
    # \s* between PASAL and DEMI handles OCR that merges them: "PASALDEMI".
    split_m = re.split(r'(?:II\.?\s*|[iI][lI1]\.?\s*)?PASAL\s*DEMI\s+PASAL', full_text, maxsplit=1, flags=re.IGNORECASE)

    if len(split_m) == 2:
        umum_raw, pasal_section = split_m
    else:
        return {"umum": _clean_penjelasan_text(full_text), "pasal": {}}

    # Strip the PENJELASAN header and "I. UMUM" label from the umum block.
    umum_text = re.sub(
        r'^.*?I\.\s*UMUM\s*', '', umum_raw, count=1,
        flags=re.DOTALL | re.IGNORECASE
    ).strip()
    umum_text = _clean_penjelasan_text(umum_text)

    # Fix OCR column-stacking artifacts before splitting at Pasal headings.
    pasal_section = _fix_penjelasan_columns(pasal_section)

    # Normalize OCR digit misreads in Pasal numbers: "Pasal l0" → "Pasal 10".
    # In PENJELASAN context, uppercase L is also treated as OCR'd 1 (no valid suffix L).
    def _normalize_penjelasan_pasal(m):
        num = m.group(2).replace('L', '1')
        return m.group(1) + _normalize_ocr_digits(num)

    pasal_section = re.sub(
        r'^(Pasa[l1]\s+)([0-9OlIL][0-9A-Za-z]*)\s*$',
        _normalize_penjelasan_pasal,
        pasal_section, flags=re.MULTILINE
    )

    # Split at each "Pasal X" heading (Arabic or Roman numerals).
    # Result: [text_before_first_pasal, "1", explanation1, "2", explanation2, ...]
    pasal_splits = re.split(
        r'^Pasa[l1]\s+(\d+[A-Z]?|[IVXLC]+)\s*$',
        pasal_section, flags=re.MULTILINE
    )

    pasal_dict = {}
    i = 1  # skip index 0 (text before the first Pasal heading)
    # Walk pairs of (pasal_number, explanation_text) from the split result.
    while i + 1 < len(pasal_splits):
        pasal_num = pasal_splits[i].strip()
        explanation = _clean_penjelasan_text(pasal_splits[i + 1].strip())
        # Pasals with no text are implicitly "Cukup jelas." in Indonesian law.
        if not explanation:
            explanation = "Cukup jelas."
        pasal_dict[pasal_num] = explanation
        i += 2

    return {"umum": umum_text, "pasal": pasal_dict}


def _consume_stacked_pasal(lines: list[str], start_idx: int) -> tuple[list[str] | None, int]:
    """Consume a run of stacked bare "Pasal" lines followed by their numbers and text.

    PyMuPDF sometimes reads compact Penjelasan columns vertically, producing:
      "Pasal\\nPasal\\n51\\ncukup jelas.\\n52\\ncukup jelas."
    This function detects such a run starting at start_idx, counts how many bare
    "Pasal" lines appear, collects the matching numbers and interleaved text, and
    returns rebuilt lines like ["Pasal 51", "cukup jelas.", "Pasal 52", "cukup jelas."].

    Returns (rebuilt_lines, next_index) on success, or (None, start_idx+1) if the
    pattern does not match.
    """
    pasal_count = 0
    j = start_idx
    # Count consecutive bare "Pasal" lines.
    while j < len(lines) and re.match(r'^Pasa[l1]\s*$', lines[j].strip()):
        pasal_count += 1
        j += 1

    collected: list[tuple[str, int]] = []
    k = j
    # Collect up to pasal_count number lines that follow the bare "Pasal" run.
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
    # Interleave each Pasal number with the explanation lines that follow it.
    for idx, (num, start_k) in enumerate(collected):
        rebuilt.append(f"Pasal {num}")
        end_k = collected[idx + 1][1] if idx + 1 < len(collected) else k
        for exp_line_idx in range(start_k + 1, end_k):
            rebuilt.append(lines[exp_line_idx])
    return rebuilt, k


def _consume_bare_number_sequence(lines: list[str], start_idx: int) -> tuple[list[str] | None, int]:
    """Consume a run of bare Pasal numbers separated from "Pasal" by a page break.

    Handles OCR artifacts where a page break splits a column into bare numbers
    like "51\\ncukup jelas.\\n52\\ncukup jelas." without "Pasal" prefixes.
    Returns (rebuilt_lines, next_index) on success, or (None, start_idx+1) if
    fewer than 2 bare numbers are found (single bare number may be a list item).
    """
    j = start_idx
    bare_entries: list[tuple[str, int]] = []
    # Collect consecutive bare-number lines, tolerating "Cukup jelas." and blank lines between them.
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
    # Rebuild each entry as "Pasal N" followed by its explanation lines.
    for idx, (num, start_j) in enumerate(bare_entries):
        rebuilt.append(f"Pasal {num}")
        end_j = bare_entries[idx + 1][1] if idx + 1 < len(bare_entries) else j
        for exp_idx in range(start_j + 1, end_j):
            rebuilt.append(lines[exp_idx])
    return rebuilt, j


def _fix_penjelasan_columns(text: str) -> str:
    """Rebuild OCR column-stacking artifacts in the PASAL DEMI PASAL section.

    PyMuPDF reads compact two-column Penjelasan pages vertically, producing runs like:
      "Pasal\\nPasal\\nPasal\\n51\\n52\\n53\\ncukup jelas.\\ncukup jelas.\\ncukup jelas."
    or bare-number variants after page breaks:
      "51\\ncukup jelas.\\n52\\ncukup jelas."

    Two detection strategies handle both forms:
    - Stacked "Pasal" lines: N consecutive bare "Pasal" lines, then N numbers + text.
    - Bare number sequences: 2+ consecutive bare numbers treated as Pasal headings.

    Returns the text with all such patterns replaced by correctly ordered lines.
    """
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()

        # Stacked bare "Pasal" lines signal a vertically-read column block.
        if re.match(r'^Pasa[l1]\s*$', stripped):
            rebuilt, next_i = _consume_stacked_pasal(lines, i)
            if rebuilt is not None:
                result.extend(rebuilt)
                i = next_i
                continue
            result.append(lines[i])
            i = next_i
            continue

        # Bare number on its own line — only treat as Pasal if part of a sequence of 2+.
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
    """Remove noise from PENJELASAN text: headers, page markers, and trailing metadata."""
    # PRESIDEN REPUBLIK INDONESIA headers — same OCR variants as in body text.
    _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN|ETIiEILtrN|FTjTJTFiTIilNEEtrtrEIn!|FTIESIDEN)'
    text = re.sub(r'\n?\s*' + _PRESIDEN_RE + r'\s*\n\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|REPUBTJK|NEPUBUK|REPI,IBLIK)\s+(?:TNDONESIA|INDONESIA|INDONESI,?A)\s*\n?', '\n', text)
    text = re.sub(r'^\s*' + _PRESIDEN_RE + r'\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*(?:REFUBUK|REPUEUK|REPUBUK|REPUBLIK|REPUBTJK|NEPUBUK|REPI,IBLIK)\s+(?:TNDONESIA|INDONESIA)\s*$', '', text, flags=re.MULTILINE)
    # Page number markers: "-5-", "- 10 -", "-t2-".
    text = re.sub(r'^\s*-\s*[t]?\d+\s*-?\s*$', '', text, flags=re.MULTILINE)
    # SK No footer.
    text = re.sub(r'SK\s+No\s*\d+\s*A.*$', '', text, flags=re.MULTILINE)
    # Trailing TAMBAHAN LEMBARAN NEGARA metadata.
    text = re.sub(r'\n\s*TAMBAHAN\s+LEMBARAN\s+NEGARA.*$', '', text, flags=re.DOTALL)
    # Page continuation markers: "Pasal 3...", "Huruf b. . ."
    text = re.sub(r'^\w[\w\s]*\.\s*\.\s*\.?\s*$', '', text, flags=re.MULTILINE)
    # Stacked bare "Pasal" or "Ayat" lines are always OCR column artifacts —
    # valid occurrences are always followed by a number or "(N)".
    text = re.sub(r'(?:^(?:Pasal|Pasa1|Ayat)\s*\n){2,}', '', text, flags=re.MULTILINE)
    # Trailing "Angka N" section headers from amendment docs: the penjelasan is split at
    # "Pasal X" boundaries, so the "Angka N" label preceding the next Pasal can bleed in.
    text = re.sub(r'(\s*\nAngka\s+\d+)+\s*$', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def attach_penjelasan(nodes: list[dict], pasal_dict: dict[str, str]):
    """Attach per-Pasal penjelasan text to matching Pasal leaf nodes in the tree."""
    # Walk every leaf node and match its Pasal number against pasal_dict.
    for node in iter_leaves(nodes):
        if "text" not in node:
            continue
        # Extract Pasal number from title: "Pasal 5" → "5", "Pasal 119I" → "119I".
        m = re.match(r'Pasal\s+(.+)', node.get("title", ""))
        if m:
            pasal_key = m.group(1).strip()
            if pasal_key in pasal_dict:
                node["penjelasan"] = pasal_dict[pasal_key]
            else:
                # Fall back to numeric-only match to handle suffix mismatches.
                num_m = re.match(r'(\d+)', pasal_key)
                if num_m and num_m.group(1) in pasal_dict:
                    node["penjelasan"] = pasal_dict[num_m.group(1)]
                else:
                    node["penjelasan"] = None
        # Non-Pasal leaf nodes (Pembukaan, etc.) are left without penjelasan.

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
        # Fuzzy prefix match on "Bagian" to tolerate OCR substitutions (e.g. Bagtan, Brgian).
        # The Indonesian ordinal word (Kesatu, Kedua, ...) anchors the match.
        r'^B[a-z]{1,5}an\s+'
        r'(Kesatu|Kedua|Ketiga|Keempat|Kelima|Keenam|Ketujuh|Kedelapan|'
        r'Kesembilan|Kesepuluh|Kesebelas|Kedua\s*belas|Ketiga\s*belas|'
        r'Keempat\s*belas|Kelima\s*belas|Keenam\s*belas|Ketujuh\s*belas|'
        r'Kedelapan\s*belas|Kesembilan\s*belas|Kedua\s*puluh)\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE | re.IGNORECASE
    ),
    "paragraf": re.compile(
        r'^Paragraf\s+(\d+|' + ROMAN_NUMERAL + r')\s*\n\s*(.+?)(?:\n|$)',
        re.MULTILINE | re.IGNORECASE
    ),
    "pasal": re.compile(
        # Matches OCR variants of "Pasal" (Pasa1, Pasal 4O, Pasal 119I).
        # Negative lookahead excludes lines followed by ayat/huruf/angka, which indicate
        # a cross-reference in body text rather than a section heading.
        # A lone trailing apostrophe (common OCR noise: "Pasal 9'") is accepted, but
        # "Pasal N..." (TOC/footer triple-dot artifacts) must NOT match.
        # Space after "Pasal" is optional: some scanned PDFs emit "Pasa12" (OCR
        # drops the space when the final `l` is rendered as a digit glyph).
        r"^[Pp]asa[l1]\s*([0-9OlI][0-9A-Za-z \t]*?)\s*[']?$"
        r"(?:\n(?!ayat|huruf|angka|dan Pasal|sampai dengan|jo\.?\s)|\Z)",
        re.MULTILINE
    ),
}

# Preceding-line words that make "Pasal X" a cross-reference, not a heading.
_CROSS_REF_PRECEDING = frozenset({
    'dalam', 'pada', 'oleh', 'dari', 'dan', 'atau', 'dengan',
    'sebagaimana', 'berdasarkan', 'terhadap', 'menurut', 'tentang',
    'atas', 'bahwa', 'melalui', 'untuk', 'antara',
})

# Numeric depth for each element type in the document hierarchy.
# In Perubahan (amendment) UUs, pasal_roman (0) is the root containing angka (1) and below.
# In normal UUs, bab (2) is the root and levels 0-1 are unused.
LEVEL_MAP = {
    "pasal_roman": 0,
    "angka": 1,
    "bab": 2,
    "bagian": 3,
    "paragraf": 4,
    "pasal": 5,
}

# Roman numeral Pasal headings in Perubahan UUs (Pasal I, II, III, ...).
PASAL_ROMAN = re.compile(
    r'^[Pp]asa[l1]\s*([IVXLCivxlc1l]+)\s*$',
    re.MULTILINE | re.IGNORECASE
)

# Numbered amendment instructions in Perubahan documents, e.g. "76. Ketentuan Pasal 88 diubah..."
#
# Indonesian amendment law uses several instruction forms that all begin with a number:
#   1. Ketentuan Pasal X diubah...            -- direct Pasal/Bagian/Bab reference
#   2. Ketentuan huruf a Pasal X diubah...    -- qualifier (huruf/ayat) before Pasal
#   3. Ketentuan ayat (1) Pasal X diubah...   -- same, with parenthesised ayat number
#   4. Judul Paragraf X Bagian Y diubah...    -- heading rename
#   5. Setelah Paragraf X ditambahkan...      -- insertion after a heading
#   6. Di antara Pasal X dan Pasal Y...       -- insertion between pasals
#
# The alternation below handles all six forms. Only runs for is_perubahan documents, so
# false-positive risk on regular numbered lists in normal laws is zero.
ANGKA_PATTERN = re.compile(
    r'^(\d+)\.\s*'                              # number + period at start of line
    r'((?:Ketentuan\s+)?'                       # optional "Ketentuan " prefix
    r'(?:'
    r'(?:Pasal|Bagian\s+Ke\w+|Bagian\s+[IVXLC]+|Bab|Di\s+antara|baris|Lampiran)\b'  # form 1/6: Bagian requires ordinal; baris/Lampiran for table/annex amendments
    r'|(?:(?:huruf|ayat|Ayat)\b(?:[^\n]*\n\s*){0,3}[^\n]*?(?:Pasal|Bagian\s+Ke\w+|Bagian\s+[IVXLC]+)\b)'  # form 2/3
    r'|(?:Penjelasan\b(?:[^\n]*\n\s*){0,3}[^\n]*?(?:Pasal|Bagian\s+Ke\w+|Bagian\s+[IVXLC]+|Bab)\b)'       # explanation amendment
    r'|(?:Judul\s+(?:Paragraf|Bagian|Bab)\b)'               # form 4: heading rename
    r'|(?:Setelah\s+(?:Pasal|Paragraf|Bagian|Bab)\b)'       # form 5: insertion after heading
    r')'
    r'[^\n]*(?:\n(?!\s*\d+\.)\s*[^\n]*){0,2})', # rest of instruction headline
    re.MULTILINE
)

# Matches the start of an amendment instruction line when the leading number is absent
# (used by wrap_orphan_pasals_in_angka1 to extract a title from the OCR gap text).
# Covers the same instruction forms as ANGKA_PATTERN but without the "N. " prefix.
_ORPHAN_INSTR_FIRST_RE = re.compile(
    r'^(?:Ketentuan\b'
    r'|Penjelasan\b'
    r'|Judul\s+(?:Paragraf|Bagian|Bab)\b'
    r'|Di\s+antara\b'
    r'|Setelah\s+(?:Pasal|Paragraf|Bagian|Bab)\b)',
    re.MULTILINE,
)

# Two-column PDF layout: in some amendment PDFs the amendment number appears alone on
# its own line (left column) while its instruction text appears later on the same page
# (right/left column interleaved by PyMuPDF). Matches a single digit 1-9 on its own
# line, searched only within the first ~300 chars of a page.
_STANDALONE_ANGKA_PAGE_TOP_RE = re.compile(
    r'^[ \t]*([1-9])[ \t]*$',
    re.MULTILINE,
)


def _normalize_roman_heading_token(token: str) -> str:
    """Normalize OCR-confused Roman heading tokens like 'l' or '1' into 'I'."""
    return (token or "").strip().replace("l", "I").replace("1", "I").upper()

def roman_to_int(s: str) -> int | None:
    """Convert a Roman numeral string to an integer.

    Returns None if the input is empty or contains non-Roman characters.
    """
    values = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
    s = _normalize_roman_heading_token(s)
    if not s or not all(c in values for c in s):
        return None
    total = 0
    # Accumulate value left to right, subtracting where a smaller numeral precedes a larger one.
    for i, c in enumerate(s):
        if i + 1 < len(s) and values[c] < values[s[i + 1]]:
            total -= values[c]
        else:
            total += values[c]
    return total


def _clean_heading_title(title: str) -> str:
    """Remove OCR artifacts from a heading title (BAB, Bagian, Paragraf).

    Fixes two categories of scanner errors: page-header text bleeding into the
    heading line, and character-level substitutions in specific Indonesian words.
    """
    # Strip "PRESIDEN REPUBLIK INDONESIA" header text that runs into the heading.
    title = re.sub(r'(?:PRESIDEN|FRESIDEN|PNESIDEN)(?:\s*REPUBLIK\s*INDONESIA)?', '', title)
    # Restore specific words where the scanner misread characters.
    title = re.sub(r'Pe\{anjian', 'Perjanjian', title)
    title = re.sub(r'Pertanggungi\s*awaban', 'Pertanggungjawaban', title)
    return title.strip()


def _prev_nonempty_line(text: str, pos: int) -> str:
    """Return the nearest previous non-empty line before pos."""
    lines = text[:pos].splitlines()
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _next_nonempty_line(text: str, pos: int) -> str:
    """Return the nearest next non-empty line after pos."""
    lines = text[pos:].splitlines()
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _is_caps_heading_fragment(line: str) -> bool:
    """Heuristic: line looks like an all-caps heading fragment, not body text."""
    stripped = line.strip()
    if not stripped or not any(ch.isalpha() for ch in stripped):
        return False
    if stripped != stripped.upper():
        return False
    if re.match(r'^(?:BAB|BAGIAN|PARAGRAF)\b', stripped):
        return True
    if re.match(r'^\(?\d+\)?[\.)]?\s', stripped):
        return False
    return True


def _looks_like_nonstructural_pasal(text: str, match: re.Match[str]) -> bool:
    """Return True when a Pasal regex hit is likely a title/reference leak."""
    prev_line = _prev_nonempty_line(text, match.start())
    next_line = _next_nonempty_line(text, match.end())

    # Real Pasal headings are followed by ayat markers, definition text, or other
    # sentence starts. Lowercase continuation strongly suggests the OCR split a
    # phrase such as "Pajak Penghasilan Pasal 21 ditanggung pemerintah".
    # Exception: if previous line is a sentence terminator, a page boundary
    # marker (page number "-9-"), or an all-caps page header like "PRESIDEN" or
    # "REPUBLIK INDONESIA", the Pasal is almost certainly a real heading even
    # if OCR lowercased the next word (e.g. "Laporan" → "laporan").
    if next_line and next_line[0].islower():
        prev_stripped = prev_line.rstrip()
        prev_ends_sentence = prev_stripped.endswith((".", ";", ":"))
        is_page_marker = bool(re.match(r'^-?\s*\d+\s*-?\s*$', prev_stripped))
        is_caps_header = _is_caps_heading_fragment(prev_line)
        if not (prev_ends_sentence or is_page_marker or is_caps_header):
            return True
    if next_line.startswith(("BAB ", "Bagian ", "Paragraf ")):
        return True
    if _is_caps_heading_fragment(prev_line) and _is_caps_heading_fragment(next_line):
        return True
    if prev_line.startswith(("BAB ", "Bagian ", "Paragraf ")) and _is_caps_heading_fragment(next_line):
        return True
    return False


def _detect_page_elements(text: str, page_num: int, is_perubahan: bool) -> list[dict]:
    """Extract structural elements from a single page's text.

    Returns an unsorted list of element dicts with keys:
    type, level, number, title, page_num, char_offset.
    """
    page_elements: list[dict] = []

    if is_perubahan:
        # Collect root-level article headings (Pasal I, II, III) that structure the amendment.
        for m in PASAL_ROMAN.finditer(text):
            roman_str = _normalize_roman_heading_token(m.group(1))
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

        # Collect numbered amendment instructions (e.g. "1. Ketentuan Pasal 3 diubah...").
        for m in ANGKA_PATTERN.finditer(text):
            angka_num = m.group(1)
            instruction = m.group(2).strip()
            title_text = instruction.replace("\n", " ").strip()
            page_elements.append({
                "type": "angka",
                "level": LEVEL_MAP["angka"],
                "number": angka_num,
                "title": f"Angka {angka_num} — {title_text}",
                "page_num": page_num,
                "char_offset": m.start(),
            })

        # Two-column layout: standalone amendment number separated from its instruction.
        # In some PDFs, PyMuPDF extracts the left-column amendment number as a separate
        # block from the right-column instruction text. Two sub-layouts exist:
        #
        # Layout B: digit appears BEFORE instruction in extracted text (digit is a small
        #   block at top of left column with lower y0 than the instruction block).
        #   Detection: find digit in first 300 chars, then search for instruction after it.
        #   Guard: skip pages with pasal_roman (Mengingat list numbers "2. Undang-Undang..."
        #   split across columns look like standalone digits at the top of the page).
        #
        # Layout A: digit appears AFTER instruction in extracted text (digit block has
        #   higher y0 — it sits below the instruction in the column). The digit appears
        #   immediately after the instruction's closing "berbunyi sebagai berikut:\n".
        #   Detection: find "sebagai berikut:\n[digit]\n" then search back for instruction.
        #   NO guard needed: the end-of-instruction marker is specific enough that Mengingat
        #   list numbers (e.g. "2. Undang-Undang") will never appear in this position.
        #   OCR note: amendment number 1 is sometimes extracted as "I" (Roman/OCR confusion).
        already_angka_nums = {e["number"] for e in page_elements if e["type"] == "angka"}
        has_roman_on_page = any(e["type"] == "pasal_roman" for e in page_elements)

        if not has_roman_on_page:
            # Layout B: digit in first 300 chars, instruction follows it.
            top_m = _STANDALONE_ANGKA_PAGE_TOP_RE.search(text[:300])
            if top_m:
                angka_num = top_m.group(1)
                if angka_num not in already_angka_nums:
                    instr_m = _ORPHAN_INSTR_FIRST_RE.search(text, top_m.end())
                    if instr_m:
                        raw_instr = text[instr_m.start():instr_m.start() + 300]
                        stop = re.search(r'\n[ \t]*\n|\n[ \t]*\d+\.', raw_instr)
                        if stop:
                            raw_instr = raw_instr[:stop.start()]
                        title_text = raw_instr.replace("\n", " ").strip()
                        page_elements.append({
                            "type": "angka",
                            "level": LEVEL_MAP["angka"],
                            "number": angka_num,
                            "title": f"Angka {angka_num} — {title_text}",
                            "page_num": page_num,
                            "char_offset": instr_m.start(),
                        })
                        already_angka_nums.add(angka_num)

        # Layout A: digit immediately follows "berbunyi sebagai berikut:\n[digit]\n".
        # Runs regardless of has_roman_on_page — the end-of-instruction marker is specific.
        # "I" is accepted in the digit position as OCR confusion for "1" (amendment no. 1).
        for berikut_m in re.finditer(
            r'(?:sebagai\s+berikut|berikut\s+ini)\s*:\s*\n([1-9I])\n', text, re.IGNORECASE
        ):
            raw_digit = berikut_m.group(1)
            angka_num = "1" if raw_digit == "I" else raw_digit
            if angka_num in already_angka_nums:
                continue
            text_before = text[:berikut_m.start()]
            instr_m = None
            for km in _ORPHAN_INSTR_FIRST_RE.finditer(text_before):
                instr_m = km  # keep last (nearest) match before the digit
            if instr_m is None:
                continue
            # Instruction text runs from keyword to end of "sebagai berikut:\n"
            # (berikut_m.end() minus the digit char and its trailing newline)
            instr_end = berikut_m.end() - len(raw_digit) - 1
            raw_instr = text[instr_m.start():instr_end]
            title_text = raw_instr.replace("\n", " ").strip()
            page_elements.append({
                "type": "angka",
                "level": LEVEL_MAP["angka"],
                "number": angka_num,
                "title": f"Angka {angka_num} — {title_text}",
                "page_num": page_num,
                "char_offset": instr_m.start(),
            })
            already_angka_nums.add(angka_num)

    # Collect BAB, Bagian, Paragraf, and Pasal headings present in all document types.
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
        if _looks_like_nonstructural_pasal(text, m):
            continue
        # Skip matches where the preceding word indicates a cross-reference, not a heading.
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
    """Remove spurious elements from a position-sorted element list.

    Applies four filters:
    - Angka items before the first pasal_roman are discarded (pre-body preamble noise).
    - A pasal entry at the same position as a pasal_roman is discarded (regex overlap).
    - Consecutive pasal entries with the same number on adjacent pages are collapsed to one.
    - Consecutive angka entries with the same number on adjacent pages are collapsed to one,
      keeping the entry with the longer title (the complete instruction from the second page).
    """
    deduped = []
    first_real_roman_idx = next((i for i, e in enumerate(elements) if e["type"] == "pasal_roman"), None)
    if first_real_roman_idx not in (None, 0):
        first_real_roman = elements[first_real_roman_idx]
        first_real_roman_num = roman_to_int(first_real_roman.get("number", ""))
        if first_real_roman_num and first_real_roman_num > 1:
            anchor_idx = next(
                (
                    i for i, elem in enumerate(elements[:first_real_roman_idx])
                    if elem["type"] in {"angka", "pasal", "bab", "bagian", "paragraf"}
                ),
                None,
            )
            if anchor_idx is not None:
                anchor = elements[anchor_idx]
                elements = elements[:anchor_idx] + [{
                    "type": "pasal_roman",
                    "level": LEVEL_MAP["pasal_roman"],
                    "number": "I",
                    "title": "Pasal I",
                    "page_num": anchor["page_num"],
                    "char_offset": anchor["char_offset"],
                }] + elements[anchor_idx:]
    roman_positions = {(e["page_num"], e["char_offset"]) for e in elements if e["type"] == "pasal_roman"}
    first_roman = next(((e["page_num"], e["char_offset"]) for e in elements if e["type"] == "pasal_roman"), None)

    # Apply each filter rule in order; append only elements that pass all checks.
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
        # Collapse duplicate pasal_roman elements with the same resolved number.
        # "Pasal 1" (Arabic, inside replacement content) can false-match PASAL_ROMAN
        # as "Pasal I" because '1' is in the OCR-confusion character class.
        if elem["type"] == "pasal_roman":
            existing_roman_nums = {
                roman_to_int(e.get("number", ""))
                for e in deduped if e["type"] == "pasal_roman"
            }
            this_num = roman_to_int(elem.get("number", ""))
            if this_num and this_num in existing_roman_nums:
                continue
        # Collapse page-break Angka duplicates: the first page often captures a truncated
        # instruction while the second page repeats it in full. Keep the longer title.
        if deduped and elem["type"] == "angka" and deduped[-1]["type"] == "angka":
            if (elem["number"] == deduped[-1]["number"]
                    and elem["page_num"] - deduped[-1]["page_num"] <= 1):
                if len(elem.get("title", "")) > len(deduped[-1].get("title", "")):
                    deduped[-1] = elem
                continue
        deduped.append(elem)

    # Filter angka elements with implausible numbering gaps.
    # Real amendment docs have consecutive Angka 1, 2, 3... under each Pasal Roman.
    # A jump from e.g. Angka 1 to Angka 80 signals a false positive from numbered
    # definition content inside replacement text.
    final_deduped = []
    last_angka_num = 0
    for elem in deduped:
        if elem["type"] == "angka":
            try:
                n = int(elem["number"])
            except (ValueError, TypeError):
                final_deduped.append(elem)
                continue
            if last_angka_num > 0 and n > last_angka_num + 20:
                log.warning(
                    "Discarding suspicious angka %d (page %d): gap from %d, likely false positive",
                    n, elem["page_num"], last_angka_num,
                )
                continue
            last_angka_num = n
        elif elem["type"] == "pasal_roman":
            last_angka_num = 0
        final_deduped.append(elem)

    return final_deduped


def detect_elements(pages: list[dict], body_end_page: int,
                    is_perubahan: bool = False) -> list[dict]:
    """Return all structural elements across the document body, sorted by position.

    Reads pages[*]["clean_text"] and ignores pages beyond body_end_page (e.g.
    PENJELASAN, Lampiran). Each returned dict has stable keys:
    type, level, number, title, page_num, char_offset.
    """
    elements = []
    body_pages: list[dict] = []

    # Collect elements from each body page; stop at post-body sections.
    for page in pages:
        if page["page_num"] > body_end_page:
            continue
        text = page["clean_text"]
        page_num = page["page_num"]
        elements.extend(_detect_page_elements(text, page_num, is_perubahan))
        body_pages.append(page)

    # For amendment docs, scan cross-page windows to catch angka instructions that split
    # at page boundaries. When "9. Ketentuan" ends page N and "Pasal 20 diubah..." begins
    # page N+1, neither half alone matches ANGKA_PATTERN on a single page. The merged
    # window catches both halves together. The dedup step below handles any duplicates
    # with same-page single-scan matches (keeps the longer title).
    if is_perubahan:
        _CROSS_TAIL = 800  # chars from end of page N to include in window
        _CROSS_HEAD = 400  # chars from start of page N+1 to include in window
        for i in range(len(body_pages) - 1):
            pa, pb = body_pages[i], body_pages[i + 1]
            tail_text = pa["clean_text"][-_CROSS_TAIL:]
            head_text = pb["clean_text"][:_CROSS_HEAD]
            window = tail_text + "\n" + head_text
            len_tail = len(tail_text)
            tail_start_in_pa = len(pa["clean_text"]) - len_tail
            for m in ANGKA_PATTERN.finditer(window):
                if m.start() >= len_tail:
                    # Match starts in page N+1's head — already covered by single-page scan
                    break
                angka_num = m.group(1)
                instruction = m.group(2).strip()
                title_text = instruction.replace("\n", " ").strip()
                elements.append({
                    "type": "angka",
                    "level": LEVEL_MAP["angka"],
                    "number": angka_num,
                    "title": f"Angka {angka_num} — {title_text}",
                    "page_num": pa["page_num"],
                    "char_offset": tail_start_in_pa + m.start(),
                })

    elements.sort(key=lambda e: (e["page_num"], e["char_offset"]))
    return _dedupe_detected_elements(elements)

# ============================================================
# 4. PASAL NUMBERING VALIDATION
# ============================================================

def validate_pasal_sequence(elements: list[dict], is_perubahan: bool = False) -> list[str]:
    """Check the Pasal numbering sequence for reversals and large gaps.

    Returns a list of warning strings. Skips validation for Perubahan UUs
    because amendment numbering is inherently non-monotonic.
    """
    if is_perubahan:
        return []

    warnings = []
    last_pasal_num = 0

    # Scan each Pasal in document order, checking for reversals and gaps greater than 5.
    for elem in elements:
        if elem["type"] != "pasal":
            continue
        try:
            num = int(re.match(r'(\d+)', elem["number"]).group(1))
        except (ValueError, AttributeError):
            continue

        # Ignore obvious front-matter / chapter-title leaks before the real body
        # sequence starts. These usually appear as a lone "Pasal N" citation near
        # the top of the document and are followed shortly by Pasal 1.
        if last_pasal_num == 0 and num > 20:
            next_pasals = [
                nxt for nxt in elements
                if nxt["type"] == "pasal"
                and (nxt["page_num"], nxt["char_offset"]) > (elem["page_num"], elem["char_offset"])
            ]
            if any(
                re.match(r'(\d+)', nxt["number"])
                and int(re.match(r'(\d+)', nxt["number"]).group(1)) == 1
                and nxt["page_num"] - elem["page_num"] <= 2
                for nxt in next_pasals[:5]
            ):
                continue

        if num < last_pasal_num:
            warnings.append(
                f"WARNING: Pasal {num} appears after Pasal {last_pasal_num} "
                f"(page {elem['page_num']}) — possible OCR error or PENJELASAN leak"
            )
        elif num > last_pasal_num + 5:
            # Gaps of 2-3 are normal in sub-laws; gaps larger than 5 likely indicate a missed page.
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
    """Set start_index, end_index, and end_char_offset on each element in-place.

    An element's end boundary is the page where the next element at the same or
    higher level begins. end_char_offset is set when that closing element starts
    on the same page, enabling intra-page text slicing.
    """
    for i, elem in enumerate(elements):
        next_page = total_pages
        # Walk forward to find the nearest element that closes this one.
        for j in range(i + 1, len(elements)):
            if elements[j]["level"] <= elem["level"]:
                next_page = elements[j]["page_num"]
                break
        elem["start_index"] = elem["page_num"]
        elem["end_index"] = next_page

        elem["end_char_offset"] = None
        # Walk forward to find an element that starts on the closing page and provides a char offset to slice at.
        for j in range(i + 1, len(elements)):
            if elements[j]["page_num"] == elem["end_index"]:
                elem["end_char_offset"] = elements[j]["char_offset"]
                break
            elif elements[j]["page_num"] > elem["end_index"]:
                break


def build_tree(elements: list[dict], total_pages: int) -> list[dict]:
    """Convert a flat element list into a nested hierarchy of document nodes.

    Accepts the output of detect_elements() and returns root nodes where each
    node may contain child nodes. Node dicts have stable keys:
    title, type, number, start_index, end_index, node_id, nodes.
    """
    if not elements:
        return []

    assign_page_boundaries(elements, total_pages)

    root_nodes = []
    stack = []  # each entry is (level, node_dict)
    node_counter = 0

    # Convert each flat element to a node and attach it under its nearest ancestor.
    for elem in elements:
        node = {
            "title": elem["title"],
            "type": elem["type"],
            "number": elem["number"],
            "start_index": elem["start_index"],
            "end_index": elem["end_index"],
            "start_char_offset": elem.get("char_offset", 0),
            "end_char_offset": None,  # refined later by fix_node_boundaries
            "node_id": f"{node_counter:04d}",
            "nodes": [],
        }
        node_counter += 1

        # Remove elements from the stack that are at the same or deeper level so the top of the stack is the correct parent.
        while stack and stack[-1][0] >= elem["level"]:
            stack.pop()

        if stack:
            stack[-1][1]["nodes"].append(node)
        else:
            root_nodes.append(node)

        stack.append((elem["level"], node))

    return root_nodes


def fix_node_boundaries(nodes: list[dict], parent_end: int, parent_end_char_offset: int | None = None):
    """Tighten sibling boundaries so their page ranges do not overlap.

    Each node's end is clamped to the next sibling's start. The last sibling
    inherits the parent's end boundary and char offset, which prevents text
    bleed across adjacent sections.
    """
    # Align each sibling's end to the next sibling's start; give the last sibling the parent's end.
    for i, node in enumerate(nodes):
        if i + 1 < len(nodes):
            next_node = nodes[i + 1]
            next_start = next_node["start_index"]
            old_end = node["end_index"]
            node["end_index"] = next_start
            if next_start == old_end:
                node["end_char_offset"] = next_node.get("start_char_offset", 0)
        else:
            node["end_index"] = parent_end
            node["end_char_offset"] = parent_end_char_offset

        if node["nodes"]:
            fix_node_boundaries(node["nodes"], node["end_index"], node.get("end_char_offset"))
            child_max = max(c["end_index"] for c in node["nodes"])
            node["end_index"] = max(node["end_index"], child_max)

        node["end_index"] = max(node["end_index"], node["start_index"])


def consolidate_bab_in_perubahan(tree: list[dict]) -> int:
    """Move orphaned Pasals into their BAB node when the amendment splits them across Angkas.

    In some Perubahan UUs, one Angka introduces a new BAB heading and the next Angka
    contains the Pasals that belong under it. This function detects that pattern and
    re-parents those Pasals under the BAB. Mutates the tree in place.

    Returns the total number of Pasals moved.
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
        # Walk each Angka child, looking for a BAB-only Angka followed by a Pasal-bearing Angka.
        while i < len(children):
            angka = children[i]
            if angka.get("type") != "angka":
                i += 1
                continue

            # Qualify the current Angka: it must contain a childless BAB and no Pasals.
            # Angkas that already have Pasals are BAB renames, not introductions.
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

            # Check the next sibling for Pasals to absorb.
            if i + 1 >= len(children):
                i += 1
                continue
            next_angka = children[i + 1]
            if next_angka.get("type") != "angka":
                i += 1
                continue

            pasal_children = [n for n in next_angka.get("nodes", []) if n.get("type") == "pasal"]
            if not pasal_children:
                i += 1
                continue

            # Re-parent each Pasal under the BAB and record which Angka it came from.
            for pasal in pasal_children:
                pasal["_moved_from_angka"] = next_angka.get("number")
            bab_node["nodes"] = pasal_children
            moved_count += len(pasal_children)

            # Expand BAB and Angka boundaries to cover the adopted children.
            bab_node["end_index"] = max(
                bab_node["end_index"],
                max(p["end_index"] for p in pasal_children),
            )
            angka["end_index"] = max(angka["end_index"], bab_node["end_index"])

            # Remove the absorbed Angka if it is now empty; otherwise strip the Pasals.
            remaining = [n for n in next_angka.get("nodes", []) if n.get("type") != "pasal"]
            if remaining:
                next_angka["nodes"] = remaining
            else:
                indices_to_remove.append(i + 1)

            i += 2

        # Delete absorbed Angkas in reverse order to keep earlier indices stable.
        for idx in reversed(indices_to_remove):
            children.pop(idx)

    return moved_count


def _collect_numeric_node_ids(nodes: list[dict], out: list[int]) -> None:
    """Accumulate all numeric node_ids found anywhere in the tree into out."""
    for node in nodes:
        try:
            out.append(int(node["node_id"]))
        except (ValueError, KeyError):
            pass
        if node.get("nodes"):
            _collect_numeric_node_ids(node["nodes"], out)


def wrap_orphan_pasals_in_angka1(
    tree: list[dict], pages: list[dict] | None = None
) -> int:
    """Wrap leading orphan Pasals under pasal_roman nodes in a synthetic Angka 1.

    In some Perubahan documents, OCR block reordering drops the "1." prefix from
    the first amendment instruction. The affected Pasal(s) then appear as direct
    children of the pasal_roman container instead of under an Angka 1 wrapper.
    The pattern identifying this case: one or more leading pasal children under a
    pasal_roman node that are followed by genuine angka siblings. A synthetic angka
    node is inserted to restore the correct pasal_roman > angka > pasal hierarchy.

    When pages is supplied, the text gap between the pasal_roman heading and the
    first orphan Pasal is scanned for an amendment instruction line starting with a
    keyword matched by _ORPHAN_INSTR_FIRST_RE. Consecutive non-blank lines are
    joined to recover split instructions and produce a title in the same format as
    naturally-detected Angka nodes (e.g. "Angka 1 — Ketentuan huruf a Pasal 6
    diubah sehingga berbunyi sebagai berikut:"). Falls back to "Angka 1" if no
    instruction line is found.

    Must be called after consolidate_bab_in_perubahan, on is_perubahan documents
    only. Mutates tree in place.

    Returns the number of Pasal nodes wrapped.
    """
    wrapped_count = 0

    for root_node in tree:
        if root_node.get("type") != "pasal_roman":
            continue
        children = root_node.get("nodes", [])
        if not children:
            continue

        # Collect all leading pasal children that appear before the first angka.
        n_leading = 0
        for child in children:
            if child.get("type") == "pasal":
                n_leading += 1
            else:
                break

        if n_leading == 0:
            continue

        # Only wrap when genuine angka siblings follow the orphan pasals, confirming
        # this is an amendment container and not a plain law with no Angka layer.
        has_angka_after = any(c.get("type") == "angka" for c in children[n_leading:])
        if not has_angka_after:
            continue

        leading_pasals = children[:n_leading]

        # Build a title from the dropped instruction text when page data is available.
        # The gap between the pasal_roman heading and the first orphan Pasal contains
        # the instruction that OCR block-reordering separated from its "1." prefix.
        angka_title = "Angka 1"
        if pages is not None:
            gap_text = _extract_node_text(
                pages,
                root_node["start_index"],
                leading_pasals[0]["start_index"],
                start_char_offset=root_node.get("start_char_offset", 0),
                end_char_offset=leading_pasals[0].get("start_char_offset"),
            ).strip()
            # Strip the pasal_roman heading line so only the instruction text remains.
            gap_text = re.sub(r'^[Pp]asa[l1]\s+[IVXLC]+\s*\n?', '', gap_text).strip()
            # Find the first line starting with an amendment instruction keyword,
            # skipping any document-header or preamble text that precedes it.
            m = _ORPHAN_INSTR_FIRST_RE.search(gap_text)
            if m:
                # Join consecutive non-blank lines to capture split instructions.
                # Stop at a blank line, after "berikut:" (natural instruction end),
                # or once 120 chars have been collected.
                parts: list[str] = []
                total = 0
                for line in gap_text[m.start():].split('\n'):
                    stripped = line.strip()
                    if not stripped:
                        break
                    parts.append(stripped)
                    total += len(stripped)
                    if total >= 120 or re.search(r'berikut\s*:', stripped):
                        break
                instr_text = ' '.join(parts)
                truncated = instr_text[:120] + ('...' if len(instr_text) > 120 else '')
                angka_title = f"Angka 1 — {truncated}"

        # Assign a unique node_id that does not collide with any existing numeric id.
        all_ids: list[int] = []
        _collect_numeric_node_ids(tree, all_ids)
        synthetic_id = f"{max(all_ids, default=-1) + 1:04d}"

        synthetic_angka: dict = {
            "title": angka_title,
            "type": "angka",
            "number": 1,
            "start_index": leading_pasals[0]["start_index"],
            "end_index": leading_pasals[-1]["end_index"],
            "start_char_offset": leading_pasals[0].get("start_char_offset", 0),
            "end_char_offset": leading_pasals[-1].get("end_char_offset"),
            "node_id": synthetic_id,
            "nodes": leading_pasals,
        }

        root_node["nodes"] = [synthetic_angka] + children[n_leading:]
        wrapped_count += n_leading
        log.debug(
            "synthetic Angka 1 created for %r: wrapped %d Pasal(s) (title=%r, node_id=%s)",
            root_node["title"], n_leading, angka_title, synthetic_id,
        )

    return wrapped_count


def iter_leaves(nodes: list[dict]):
    """Yield every leaf node in the tree (nodes that have no children)."""
    # Recurse into non-empty node lists; yield nodes with no children as leaves.
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            yield from iter_leaves(node["nodes"])
        else:
            yield node


def clean_tree_for_output(nodes: list[dict], pages: list[dict] | None = None) -> list[dict]:
    """Return a cleaned copy of the tree with internal fields removed.

    When pages is provided, text is embedded directly into each leaf node.
    pasal_roman container nodes also receive any intro text that precedes
    their first child.
    """
    result = []
    # Build each output node, embedding text for leaves and pasal_roman containers.
    for node in nodes:
        clean = {
            "title": node["title"],
            "node_id": node["node_id"],
            "start_index": node["start_index"],
            "end_index": node["end_index"],
        }
        if node.get("nodes"):
            clean["nodes"] = clean_tree_for_output(node["nodes"], pages)
            # A pasal_roman node may have intro text before its first child; extract it.
            if pages is not None and node.get("type") == "pasal_roman":
                first_child = node["nodes"][0]
                if first_child["start_index"] == node["start_index"]:
                    own_end_page = node["start_index"]
                    own_end_char = first_child.get("start_char_offset") or None
                else:
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
            clean["text"] = _extract_node_text(
                pages, node["start_index"], node["end_index"],
                start_char_offset=node.get("start_char_offset", 0),
                end_char_offset=node.get("end_char_offset"),
            )
        if "penjelasan" in node:
            clean["penjelasan"] = node["penjelasan"]
        if "_moved_from_angka" in node:
            clean["source_angka"] = node["_moved_from_angka"]
        result.append(clean)
    return result


def _clean_preamble_noise(raw_text: str) -> str:
    """Remove page-header and margin OCR noise from preamble text.

    Strips PRESIDEN REPUBLIK INDONESIA headers, SALINAN stamps, page number
    markers, and isolated all-caps words that bleed in from adjacent columns,
    while preserving the preamble keywords Mengingat and Menetapkan.
    """
    _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN|FR,ESIDEN|MENTERI)'
    noise_patterns = [
        re.compile(r'^\s*Mengingat\s*\n\s*Menetapkan\s*$', re.MULTILINE),
        re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s*\n\s*REP\S*\s+IND\S*\s*\n'),
        re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s+REP\S*\s+IND\S*\s*\n'),
        re.compile(r'\n\s*' + _PRESIDEN_RE + r'\s*\n'),
        re.compile(r'^\s*REP\S*\s+IND\S*\s*$', re.MULTILINE),
        re.compile(r'^\s*(?:MENTERI|FR,ESIDEN)\s+REP\S*\s+IND\S*\s*$', re.MULTILINE),
        re.compile(r'^\s*SAL[IT]NAN\s*\n', re.MULTILINE),
        re.compile(r'^\s*-\d+-\s*\n', re.MULTILINE),
        re.compile(r'^\s*(?!BAHWA|Mengingat|Menetapkan|MEMUTUSKAN)[A-Z]{3,8}\s*\n', re.MULTILINE),
    ]
    cleaned_text = raw_text
    # Apply each pattern in order; every match collapses to a single newline.
    for pat in noise_patterns:
        cleaned_text = pat.sub('\n', cleaned_text)
    return re.sub(r'\n{3,}', '\n\n', cleaned_text).strip()


def _clean_preamble_child_text(text: str, section: str) -> str:
    """Normalize one preamble child after section-level splitting."""
    cleaned = _clean_preamble_noise(text or "")
    cleaned = re.sub(
        r'(?im)^\s*(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN|FR,ESIDEN|MENTERI)\s*$',
        '',
        cleaned,
    )
    cleaned = re.sub(
        r'(?im)^\s*(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN|FR,ESIDEN|MENTERI)\s+REP\S*\s+IND\S*\s*$',
        '',
        cleaned,
    )
    cleaned = re.sub(
        r'(?im)^\s*REP\S*\s+IND\S*\s*$',
        '',
        cleaned,
    )
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()

    if section == "menimbang":
        cleaned = re.sub(r'(?is)\bMengingat\b\s*:?.*$', '', cleaned).strip()
        if re.search(r'(?i)\bbahwa\b', cleaned):
            cleaned = re.sub(r'(?is)(?:\n\s*[a-z]\.?\s*){2,}$', '', cleaned).strip()
    elif section == "mengingat":
        cleaned = re.sub(r'(?is)\b(?:MEMUTUSKAN|Menetapkan)\b\s*:?.*$', '', cleaned).strip()
        cleaned = re.sub(r'(?is)(?:\n\s*\d+\.\s*){2,}$', '', cleaned).strip()
    elif section == "menetapkan":
        cleaned = re.sub(r'(?is)^\s*Menetapkan\s*:?\s*', '', cleaned).strip()
        cleaned = re.sub(r'(?im)^\s*REP\S*\s+IND\S*\s*$', '', cleaned)
        cleaned = re.sub(
            r'(?im)^\s*(?:BAB\s*[IVXLC]+|BAB[IVXLC]+|KETENTUAN\s+UMUM|Agar\s+setiap\s+orang|Agar|Pasal\s+[0-9IVXLC]+|Pasa[Il1]\s+[0-9IVXLCIl1]+)\b[\s\S]*$',
            '',
            cleaned,
        ).strip()
        cleaned = re.sub(
            r'(?is)\b(?:Pasa[Il1l]\s+[0-9IVXLCIl1]+|Pasa[lI1]\s+[0-9IVXLCIl1]+)\b.*$',
            '',
            cleaned,
        ).strip()
        cleaned = re.sub(r'(?im)\n\s*Menetapkan\s*$', '', cleaned).strip()

    return re.sub(r'\n{3,}', '\n\n', cleaned).strip()


def _looks_like_marker_only_text(text: str) -> bool:
    """Return True when text collapses into bare list markers like 'a b c d'."""
    lines = [line.strip(" .:;\t") for line in (text or "").splitlines() if line.strip()]
    return bool(lines) and all(len(line) == 1 and line.isalpha() for line in lines)


def _extract_menimbang_candidate(text: str) -> str | None:
    """Recover misplaced Menimbang content that OCR placed before the label."""
    for pattern in (
        r'(?:^|\n)\s*a\.?\s*(?:\n\s*)?bahwa\b',
        r'(?:^|\n)\s*a\s*(?:\n\s*)?bahwa\b',
        r'(?:^|\n)\s*bahwa\b',
    ):
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return text[match.start():].lstrip('\n')
    return None


def _split_embedded_menetapkan(text: str) -> tuple[str, str | None]:
    """Split Mengingat text when an embedded Menetapkan heading slipped into it."""
    match = re.search(r'(?:^|\n)\s*Menetapkan\s*:?\s*(?:\n|(?=[A-Z]))', text)
    if not match:
        return text, None
    before = text[:match.start()].strip()
    after = text[match.end():].strip()
    return before, after or None


def _strip_mengingat_prefix(candidate: str) -> str:
    """Remove a leading 'Mengingat' keyword and surrounding punctuation from a string.

    Handles both the case where "Mengingat" appears at the very start and where
    it appears within the first 200 characters (e.g. after a newline).
    """
    stripped = re.sub(r'^\s*Mengingat\s*:?\s*\n?\s*', '', candidate)
    if stripped != candidate:
        return stripped
    kw_m = re.search(r'\bMengingat\b', candidate[:200])
    if kw_m:
        after_kw = candidate[kw_m.end():]
        return re.sub(r'^\s*:?\s*\n?\s*', '', after_kw)
    return candidate


def _recover_mengingat_tail(
    menimbang_text: str,
    mengingat_text: str | None,
) -> tuple[str, str | None]:
    """Move numbered legal references accidentally attached to Menimbang into Mengingat."""
    if not re.search(r'(?i)\bbahwa\b', menimbang_text):
        return menimbang_text, mengingat_text
    ref_m = re.search(
        r'(?:^|\n)((?:\d+\.\s+)?(?:Pasal\s+\d+|Undang-Undang|Peraturan(?:\s+Pemerintah)?))',
        menimbang_text,
    )
    if not ref_m:
        return menimbang_text, mengingat_text
    split_at = ref_m.start(1)
    moved_tail = menimbang_text[split_at:].strip()
    kept = menimbang_text[:split_at].strip()
    if not kept or not moved_tail:
        return menimbang_text, mengingat_text
    if mengingat_text:
        merged = moved_tail.rstrip() + "\n" + mengingat_text.lstrip()
    else:
        merged = moved_tail
    return kept, merged


def _split_by_content_transition(body: str) -> tuple[str | None, str | None]:
    """Split preamble body into Menimbang and Mengingat at the end of the last bahwa clause.

    Scans for the last "bahwa" keyword and then looks for the next semicolon- or
    period-terminated line where the remaining text begins with a legal reference
    (numbered Pasal, Undang-Undang, or Peraturan). Returns (menimbang, mengingat)
    or (None, None) if no valid split point is found.
    """
    last_bahwa_end = None
    # Find the position just after the last "bahwa" keyword.
    for m in re.finditer(r'bahwa\s', body):
        last_bahwa_end = m.end()
    if not last_bahwa_end:
        return None, None

    after_last = body[last_bahwa_end:]
    # Try each semicolon-terminated line break as a candidate split point.
    for semi_m in re.finditer(r';\s*\n', after_last):
        split_pos = last_bahwa_end + semi_m.end()
        remaining = body[split_pos:]
        remaining_stripped = _strip_mengingat_prefix(remaining)
        if re.match(r'\s*[2-9]\d*[\.\s]+', remaining_stripped):
            continue
        if re.match(r'\s*(?:\d+[\.\s]+)?(?:Pasal|Undang|Peraturan)', remaining_stripped):
            return body[:split_pos].strip(), remaining_stripped.strip()

    # Fall back to period-terminated line breaks.
    for period_m in re.finditer(r'\.\s*\n', after_last):
        split_pos = last_bahwa_end + period_m.end()
        remaining = body[split_pos:]
        remaining_stripped = _strip_mengingat_prefix(remaining)
        if re.match(r'\s*1[\.\s]+(?:Pasal|Undang|Peraturan)', remaining_stripped):
            return body[:split_pos].strip(), remaining_stripped.strip()

    return None, None


def _split_by_mengingat_keyword(body: str) -> tuple[str | None, str | None]:
    """Split preamble body at an explicit 'Mengingat' heading.

    Tries two detection strategies in order:

    1. Strict: "Mengingat" appears on its own line (well-formatted OCR).
    2. Relaxed: "Mengingat" follows immediately after a semicolon or period with no
       preceding newline -- a common OCR garbling where the line break between the last
       Menimbang clause and the Mengingat heading is lost.

    Returns (menimbang, mengingat) or (None, None) if the keyword is absent.
    """
    # Strategy 1: standalone Mengingat heading line.
    mengingat_kw = re.search(r'\n\s*Mengingat\s*:?\s*\n', body)
    if mengingat_kw:
        return body[:mengingat_kw.start()].strip(), body[mengingat_kw.end():].strip()

    # Strategy 2: Mengingat runs inline after the clause terminator (;/.) when the OCR
    # drops the intervening newline, e.g. "...Negara;Mengingat\n1. Pasal 4 ayat (1)..."
    # The terminator is kept with Menimbang; everything from Mengingat onward is Mengingat.
    mengingat_kw = re.search(r'([;.])\s*\n?\s*Mengingat\s*:?\s*\n?', body)
    if mengingat_kw:
        cut = mengingat_kw.start() + 1  # include the terminator in the Menimbang slice
        return body[:cut].strip(), body[mengingat_kw.end():].strip()

    return None, None


def _split_by_first_pasal_ref(body: str) -> tuple[str | None, str | None]:
    """Split preamble body at the first numbered legal reference.

    Last-resort fallback when no other split heuristic fires. Looks for a line
    starting with an optional number followed by "Pasal N".
    Returns (menimbang, mengingat) or (None, None) if not found.
    """
    mengingat_m = re.search(r'\n(\d+\.\s+Pasal\s+\d+)', body)
    if not mengingat_m:
        return None, None
    return body[:mengingat_m.start()].strip(), body[mengingat_m.start():].strip()


def _split_preamble(text: str, start_page: int, end_page: int) -> list[dict] | None:
    """Split preamble text into Menimbang, Mengingat, and Menetapkan child nodes.

    Returns a list of node dicts on success, or None if no Menimbang section
    can be located in the text.
    """
    cleaned = _clean_preamble_noise(text)

    # Locate the Menimbang section start.
    menimbang_m = re.search(r'Menimbang\s*:?\s*(?:\n|(?=a[\.\s]))', cleaned)
    if menimbang_m:
        before_menimbang = cleaned[:menimbang_m.start()]
        after_menimbang = cleaned[menimbang_m.end():]
        # OCR block reordering can place Menimbang clauses before the "Menimbang:" label.
        misplaced_menimbang = _extract_menimbang_candidate(before_menimbang)
        if misplaced_menimbang and (
            not re.search(r'(?i)\bbahwa\b', after_menimbang[:400])
            or _looks_like_marker_only_text(after_menimbang[:120])
        ):
            after_menimbang = (
                misplaced_menimbang.rstrip() + "\n" + after_menimbang.lstrip()
            )
    else:
        # No Menimbang keyword; locate the preamble body via the first bahwa clause.
        fallback_text = _extract_menimbang_candidate(cleaned)
        if not fallback_text:
            return None
        before_menimbang = cleaned[:cleaned.find(fallback_text)]
        after_menimbang = fallback_text.lstrip('\n')

    # Locate MEMUTUSKAN to bound the preamble body and extract Menetapkan text.
    # The colon-terminated form ("MEMUTUSKAN:") is tried first; some documents omit the
    # colon, in which case the bare heading on its own line is matched as a fallback.
    memutuskan_m = re.search(r'MEMUTUS\S*\s*:|\nMEMUTUS\S*\s*\n', after_menimbang)
    menetapkan_text = None

    if memutuskan_m:
        preamble_body = after_menimbang[:memutuskan_m.start()].strip()
        raw_after = after_menimbang[memutuskan_m.end():].strip()
        # Remove an optional "Menetapkan :" label that some documents include.
        menet_m = re.match(r'Menetapkan\s*:\s*', raw_after)
        if menet_m:
            menetapkan_text = raw_after[menet_m.end():].strip()
        elif raw_after:
            menetapkan_text = raw_after
    else:
        preamble_body = after_menimbang.strip()

    # Split the preamble body into Menimbang and Mengingat using fallback strategies.
    # Keyword-based detection is tried first: when "Mengingat" is present in the text it is
    # the most unambiguous signal regardless of OCR quality. Content-transition and first-ref
    # heuristics serve as fallbacks for documents where the keyword is absent or garbled beyond
    # recognition.
    menimbang_text, mengingat_text = _split_by_mengingat_keyword(preamble_body)
    if menimbang_text is None:
        menimbang_text, mengingat_text = _split_by_content_transition(preamble_body)
    if menimbang_text is None:
        menimbang_text, mengingat_text = _split_by_first_pasal_ref(preamble_body)

    if menimbang_text is None:
        menimbang_text = preamble_body
        mengingat_text = None

    menimbang_text, mengingat_text = _recover_mengingat_tail(menimbang_text, mengingat_text)

    if not menetapkan_text and mengingat_text:
        mengingat_text, embedded_menetapkan = _split_embedded_menetapkan(mengingat_text)
        if embedded_menetapkan:
            menetapkan_text = embedded_menetapkan

    # Finalize Menimbang text: remove a trailing "Mengingat" keyword (with any following colon
    # or OCR-garbled content) that bleeds in from margin noise at the end of the text.
    menimbang_text = re.sub(r'\s*\bMengingat\b\s*:?.*$', '', menimbang_text).strip()
    menimbang_text = _clean_preamble_child_text(menimbang_text, "menimbang")
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
        # Remove a leading colon artifact when the "Mengingat :" label shares a line with ref 1.
        mengingat_text = re.sub(r'^[\s:]+', '', mengingat_text)
        # Remove "Dengan Persetujuan Bersama" boilerplate that falls between Mengingat
        # and MEMUTUSKAN. The fuzzy word match handles common OCR garbling.
        # The (?:^|\n) prefix handles both the start-of-string case (when all Mengingat
        # legal references were absorbed into Menimbang and only this boilerplate remains)
        # and the mid-text case (when valid references precede it).
        dpr_m = re.search(r'(?:^|\n)[^\n]*Dengan\s+Perse\w+\s+Bersama', mengingat_text)
        if dpr_m:
            mengingat_text = mengingat_text[:dpr_m.start()]
        # Remove isolated "Mengingat" lines and leading keyword inserted by OCR/parsing.
        mengingat_text = re.sub(r'^\s*Mengingat\s*$', '', mengingat_text, flags=re.MULTILINE)
        mengingat_text = _strip_mengingat_prefix(mengingat_text)
        mengingat_text = _clean_preamble_child_text(mengingat_text, "mengingat")
        # Drop text that has no legal reference content — it is an OCR artifact (e.g. "Dengan", "2.").
        if not re.search(r'\b(?:Pasal|Undang|Peraturan)\b', mengingat_text):
            mengingat_text = ''
        if mengingat_text:
            children.append({
                "title": "Mengingat",
                "node_id": "P002",
                "start_index": start_page,
                "end_index": end_page,
                "text": mengingat_text,
            })

    if menetapkan_text:
        menetapkan_text = _clean_preamble_child_text(menetapkan_text, "menetapkan")
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
    """Extract and join page text for a node spanning pages [start, end].

    Applies char offset slicing at the start and end boundaries to return only
    the text belonging to this node.
    """
    parts = []
    # Collect text from each page in the range, slicing at the boundary pages.
    for page in pages:
        if page["page_num"] < start:
            continue
        if page["page_num"] > end:
            break
        text = page["clean_text"]

        if page["page_num"] == start and start_char_offset > 0:
            text = text[start_char_offset:]

        if page["page_num"] == end and end_char_offset is not None:
            # When start and end are the same page, the end offset is relative to
            # the original page text, so subtract the start slice already applied.
            if page["page_num"] == start and start_char_offset > 0:
                adjusted = end_char_offset - start_char_offset
                if adjusted > 0:
                    text = text[:adjusted]
            else:
                text = text[:end_char_offset]

        parts.append(text.strip())
    joined = "\n\n".join(p for p in parts if p)

    # Strip page header bleed-through that survived the initial OCR cleaning pass.
    _PRESIDEN_RE = r'(?:P(?:RE|NE|TT)?SI[DO]EN|FRESIDEN|PRESTDEN)'
    joined = re.sub(r'\n\s*' + _PRESIDEN_RE + r'\s*\n\s*REPUBLIK INDONESIA\s*\n', '\n', joined)
    joined = re.sub(r'\n\s*' + _PRESIDEN_RE + r'\s*\n', '\n', joined)
    # Strip structural headings at the very end of the text that bleed from the next page.
    joined = re.sub(r'\n\s*BAB\s+[IVXLCDM]+\s*$', '', joined)
    joined = re.sub(r'\n\s*B[a-z]{1,5}an\s+\w+\s*$', '', joined)
    joined = re.sub(r'\n\s*Paragraf\s+\w+\s*$', '', joined)
    return joined

# ============================================================
# 6. LLM TEXT CLEANUP
# ============================================================

# Conservative defaults prioritize reliability over throughput.
# Can be overridden via env vars when the connection/API is stable.
_DEFAULT_LLM_BATCH_SIZE = 40_000
_DEFAULT_LLM_MAX_WORKERS = 4
_DEFAULT_LLM_TIMEOUT = 300


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return max(minimum, int(raw))
    except ValueError:
        log.warning(f"invalid {name}={raw!r}, using default {default}")
        return default


def _llm_batch_size() -> int:
    return _env_int("VECTORLESS_LLM_BATCH_SIZE", _DEFAULT_LLM_BATCH_SIZE)


def _llm_max_workers() -> int:
    return _env_int("VECTORLESS_LLM_MAX_WORKERS", _DEFAULT_LLM_MAX_WORKERS)


def _llm_timeout() -> int:
    return _env_int("VECTORLESS_LLM_TIMEOUT", _DEFAULT_LLM_TIMEOUT)


def _disable_proxy_env_if_needed():
    """Temporarily clear proxy vars for Gemini requests on flaky Windows setups.

    IMPORTANT: this is a temporary operational workaround. The newer
    `google.genai` SDK proved unreliable in this environment, so LLM cleanup
    currently uses the older stable SDK path and disables proxy vars by
    default unless explicitly opted out.
    """
    if os.environ.get("VECTORLESS_GEMINI_DISABLE_PROXY", "1") != "1":
        return
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ[key] = ""


def _make_genai_model(api_key: str | None = None):
    """Create a Gemini model handle via the older stable SDK.

    IMPORTANT: temporary workaround until `google.genai` becomes reliable in
    this Windows environment. Keep the rest of the cleanup pipeline unchanged
    so we can swap the adapter back later.
    """
    _disable_proxy_env_if_needed()

    if api_key is None:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")

LLM_CLEANUP_PROMPT = """\
You are an OCR correction tool for Indonesian legal documents.
Fix OCR artifacts in the text while preserving the original meaning and structure.

Rules:
- Fix garbled characters (e.g., "tqjuh" → "tujuh", "ruPiah" → "rupiah")
- Fix broken numbers (e.g., "OOO,OO" → "000,00", "47 |" → "471")
- Fix broken ayat markers where the closing ")" was OCR'd as "1": e.g., "(21" → "(2)", "(31" → "(3)", "(41" → "(4)". These appear at the start of a line as ayat numbering.
- Remove duplicate paragraphs caused by page breaks: if the same sentence or ayat appears twice in a row (possibly with noise characters like "N", "I", "II" between them), keep only the complete version and remove the truncated duplicate.
- Remove isolated noise characters on their own lines (single letters like "N", "I", Roman numerals like "II", "III" that are clearly page artifacts, not legal content).
- Fix encoding corruption (e.g., "pang€rn" → "pangan", "rups€ah" → "rupiah").
- Do NOT change legal terminology, Pasal references, or document structure
- Do NOT add or remove substantive content
- You MUST return ALL keys from the input. Every node_id must appear in your output.
- Return ONLY a valid JSON object with the same keys, mapping each node_id to its cleaned text. No explanation, no markdown.

Input (JSON):
"""

# Page header and footer patterns that survive into extracted node text.
_OCR_HEADER_PATTERNS = [
    re.compile(r'PRESIDEN\s*\n\s*REPUBLIK\s+INDONESIA'),
    re.compile(r'^PRESIDEN\s+REPUBLIK\s+INDONESIA\s*$', re.MULTILINE),
    re.compile(r'^\s*(?:PRESIDEN|FRESIDEN|PNESIDEN|PRESTDEN|PTTESIDEN|FR,ESIDEN|MENTERI)\s*$', re.MULTILINE),
    re.compile(r'^\s*(?:REPUBLIK|R,EPUBLIK|REPIJBUK|REPI,IBLIK|REP[A-Z]*K)\s+INDONESIA\s*$', re.MULTILINE),
    re.compile(r'^\s*(?:FEPUEUK|IIEPUBUK|REP[A-Z,]*|[A-Z]{2,4}PUB[A-Z,]*)\s+IN[A-Z]{6,12}\s*$', re.MULTILINE),
    re.compile(r'^\s*(?:PRESIDEN|FRESIDEN|PNESIDEN|PRESTDEN|PTTESIDEN|FR,ESIDEN|MENTERI)\s+REP\S*\s+IND\S*\s*$', re.MULTILINE),
    re.compile(r'(?:PRESIDEN|FRESIDEN|PNESIDEN|PRESTDEN|PTTESIDEN|FR,ESIDEN|MENTERI)\s*\n\s*-?\d+\s*-\s*(?:\n\s*REP\S*\s+IND\S*)?', re.MULTILINE),
    re.compile(r'^\s*[A-Z]{1,4}\s+INDONESIA\s*$', re.MULTILINE),
    re.compile(r'^\s*_[0-9OlIt-]+_\s*$', re.MULTILINE),
    re.compile(r'^\s*-?\d+\s*-?\s*$', re.MULTILINE),
    re.compile(r'^SK\s+No\s+.*$', re.MULTILINE),
    re.compile(r'^LEMBARAN\s+NEGARA\s+REPUBLIK\s+INDONESIA.*$', re.MULTILINE),
    re.compile(r'^TAMBAHAN\s+LEMBARAN\s+NEGARA.*$', re.MULTILINE),
]

# Pengesahan (closing) text that bleeds into the last Pasal when they share a page.
# Written to tolerate OCR word breaks and extra whitespace.
_CLOSING_TEXT_RE = re.compile(
    r'\n\s*(?:'
    r'Ditetapkan\s+di\s'
    r'|Diundangkan\s+di\s'
    r'|Agar\s+setiap\s+orang'
    r'|Agar\s+setiap\s*\n'  # line-break OCR variant
    r')'
    r'[\s\S]*$',
)

_AMENDMENT_SPILLOVER_PATTERNS = [
    re.compile(r'\n\s*\d+\.\s+Ketentuan\s+(?:Pasal|ayat|huruf)\b[\s\S]*$', re.IGNORECASE),
    re.compile(r'\n\s*\d+\.\s+Di\s+antara\s+Pasal\b[\s\S]*$', re.IGNORECASE),
    re.compile(r'\n\s*\d+\.\s+Penjelasan\b[\s\S]*$', re.IGNORECASE),
    re.compile(r'\n\s*\d+\.\s+Setelah\b[\s\S]*$', re.IGNORECASE),
    re.compile(r'\n\s*Ketentuan\s+(?:Pasal|ayat|huruf)\b[\s\S]*$', re.IGNORECASE),
    re.compile(r'\n\s*Di\s+antara\s+Pasal\b[\s\S]*$', re.IGNORECASE),
    # Cross-line: number and "Ketentuan" split across PDF lines by OCR e.g. "5.\nKetentuan..."
    re.compile(r'\n\s*\d+\.\s*\n\s*Ketentuan\s+(?:Pasal|ayat|huruf)\b[\s\S]*$', re.IGNORECASE),
    # Roman-numeral closing articles that bleed into last ayat of amendment docs
    re.compile(r'\n\s*PASAL\s+[IVX]+\s*\n[\s\S]*$', re.IGNORECASE),
]


def _dedupe_repeated_heading(text: str, title: str | None) -> str:
    """Collapse OCR-induced duplicated headings at the start of a leaf."""
    if not title:
        return text
    title = title.strip()
    if not title:
        return text
    title_re = re.escape(title)
    return re.sub(
        rf'^\s*({title_re})\s*\n+\s*\1\b',
        r'\1',
        text,
        count=1,
        flags=re.IGNORECASE,
    )


def _strip_leading_title(text: str, title: str | None) -> str:
    """Remove the node's own title if it appears as the very first line of text.

    OCR output often starts with the heading text (e.g. "Pasal 1\\n...") even
    though the heading is already captured in the node's title field.  This
    function removes that redundant first-line heading when present.
    """
    if not title:
        return text
    words = title.strip().split()
    if not words:
        return text
    pattern = r'^\s*' + r'\s+'.join(re.escape(w) for w in words) + r'\s*\n'
    return re.sub(pattern, '', text, count=1, flags=re.IGNORECASE)


def _trim_amendment_spillover(text: str, title: str | None) -> str:
    """Trim duplicated amendment-intro text that bled into the previous Pasal leaf."""
    if not title or not title.startswith("Pasal "):
        return text
    trimmed = text
    for pattern in _AMENDMENT_SPILLOVER_PATTERNS:
        trimmed = pattern.sub('', trimmed)
    trimmed = re.sub(r'(?im)\n\s*Agar\s*$', '', trimmed)
    return trimmed


def _trim_trailing_structural_heading(text: str, title: str | None) -> str:
    """Remove next-section headings that bled into the current leaf."""
    trimmed = text
    if title and title.startswith("Pasal "):
        trimmed = re.sub(r'(?im)\n\s*BAB\s*[IVXLCDM]+\s*$', '', trimmed)
        trimmed = re.sub(r'(?im)\n\s*BAB[IVXLCDM]+\s*$', '', trimmed)
        trimmed = re.sub(r'(?im)\n\s*Bagian\s+[^\n]+\s*$', '', trimmed)
        trimmed = re.sub(r'(?im)\n\s*Paragraf\s+[^\n]+\s*$', '', trimmed)
    return trimmed


def _normalize_leaf_text(text: str, title: str | None = None) -> str:
    """Apply OCR cleanup, duplicate-heading collapse, and spillover trimming."""
    cleaned = text or ""
    if title == "Menimbang":
        cleaned = _clean_preamble_child_text(cleaned, "menimbang")
    elif title == "Mengingat":
        cleaned = _clean_preamble_child_text(cleaned, "mengingat")
    elif title == "Menetapkan":
        cleaned = _clean_preamble_child_text(cleaned, "menetapkan")
    for pat in _OCR_HEADER_PATTERNS:
        cleaned = pat.sub('', cleaned)
    cleaned = _CLOSING_TEXT_RE.sub('', cleaned)
    cleaned = _dedupe_repeated_heading(cleaned, title)
    cleaned = _strip_leading_title(cleaned, title)
    cleaned = _trim_amendment_spillover(cleaned, title)
    cleaned = _trim_trailing_structural_heading(cleaned, title)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned

def _process_batch(batch: dict[str, str], batch_idx: int, total: int, verbose: bool,
                   api_key: str | None = None, _label: str | None = None):
    """Send one batch to Gemini and return the cleaned texts with token counts.

    Returns a 4-tuple: (cleaned_dict, input_tokens, output_tokens, error_message).
    On unrecoverable error, returns the original batch unchanged with error_message set.
    Retries up to 5 times on rate-limit or transient network errors.
    """
    prompt = LLM_CLEANUP_PROMPT + json.dumps(batch, ensure_ascii=False)

    max_retries = 5
    last_err: Exception | None = None
    # Retry up to max_retries times, backing off linearly on rate limit or network errors.
    for attempt in range(max_retries):
        try:
            model = _make_genai_model(api_key)
            response = model.generate_content(
                prompt,
                request_options={"timeout": _llm_timeout()},
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
                # Return the original batch unchanged so the caller can proceed with other batches.
                _lbl = _label or f"{batch_idx + 1}/{total}"
                msg = f"batch {_lbl}: {e.__class__.__name__}: {e}"
                if verbose:
                    log.warning(f"batch {_lbl}: failed after {attempt + 1} attempt(s), keeping raw OCR text — {msg}")
                return batch, 0, 0, msg

    if last_err is not None:
        _lbl = _label or f"{batch_idx + 1}/{total}"
        return batch, 0, 0, f"batch {_lbl}: unexpected retry exit"

    usage = getattr(response, "usage_metadata", None)
    input_tok = getattr(usage, "prompt_token_count", 0) or 0
    output_tok = getattr(usage, "candidates_token_count", 0) or 0
    _lbl = _label or f"{batch_idx + 1}/{total}"
    if verbose:
        if input_tok or output_tok:
            log.info(f"batch {_lbl}: {input_tok + output_tok:,} tokens ({input_tok:,} in, {output_tok:,} out)")
        else:
            log.info(f"batch {_lbl}: response received")

    response_text = getattr(response, "text", "").strip()
    # Remove markdown code fences that the model sometimes wraps around JSON.
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
        return batch, input_tok, output_tok, msg

    return cleaned, input_tok, output_tok, None


def llm_cleanup_texts(texts: list[tuple[str, str]], verbose: bool = True, client=None) -> tuple[dict[str, str], list[str]]:
    """Send node texts to Gemini for OCR cleanup and return the results.

    Splits the input into character-bounded batches, processes them in parallel,
    and retries failed batches at half size. Returns a tuple of
    (cleaned_dict mapping node_id to cleaned text, list of failure messages).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")

    batch_size = _llm_batch_size()
    max_workers = _llm_max_workers()

    batches: list[dict[str, str]] = []
    current_batch: dict[str, str] = {}
    current_size = 0

    # Pack texts greedily into batches, flushing when the size limit is reached.
    for node_id, text in texts:
        text_len = len(text)
        if current_size + text_len > batch_size and current_batch:
            batches.append(current_batch)
            current_batch = {}
            current_size = 0
        current_batch[node_id] = text
        current_size += text_len

    if current_batch:
        batches.append(current_batch)

    if verbose:
        worker_label = "sequentially" if max_workers == 1 else f"with {min(len(batches), max_workers)} worker(s)"
        log.info(
            f"{len(texts)} nodes in {len(batches)} batch(es), "
            f"batch_size={batch_size:,}, timeout={_llm_timeout()}s, processing {worker_label}"
        )

    results: dict[str, str] = {}
    failed_batches: list[tuple[int, dict[str, str]]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_calls = 0

    # Submit all batches to a thread pool and collect results as they complete.
    with ThreadPoolExecutor(max_workers=min(len(batches), max_workers)) as executor:
        futures = {
            executor.submit(_process_batch, batch, i, len(batches), verbose, api_key): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            batch_i = futures[future]
            cleaned, input_tok, output_tok, error = future.result()
            results.update(cleaned)
            total_input_tokens += input_tok
            total_output_tokens += output_tok
            total_calls += 1
            if error:
                failed_batches.append((batch_i, batches[batch_i]))

    failures: list[str] = []
    if failed_batches:
        if verbose:
            log.info(f"retrying {len(failed_batches)} failed batch(es) split in half")
        # Retry each failed batch at half size to reduce the chance of truncation.
        for batch_i, failed_batch in failed_batches:
            items = list(failed_batch.items())
            mid = max(1, len(items) // 2)
            # Process each half independently; any remaining failure is treated as permanent.
            for sub_idx, sub_items in enumerate([items[:mid], items[mid:]]):
                if not sub_items:
                    continue
                sub_batch = dict(sub_items)
                sub_label = f"{batch_i + 1}{'ab'[sub_idx]}"
                sub_cleaned, in_tok, out_tok, sub_err = _process_batch(
                    sub_batch, batch_i, len(batches), verbose, api_key, _label=sub_label
                )
                results.update(sub_cleaned)
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                total_calls += 1
                if sub_err:
                    failures.append(
                        f"LLM batch {batch_i + 1}{'ab'[sub_idx]} failed after retry: "
                        f"{len(sub_batch)} node(s) kept as raw OCR text"
                    )

    if verbose:
        log.info(f"total tokens: {total_input_tokens + total_output_tokens:,} "f"({total_input_tokens:,} in, {total_output_tokens:,} out)")

    return results, failures, total_input_tokens, total_output_tokens, total_calls


def apply_llm_cleanup(output_nodes: list[dict], penjelasan_data: dict | None = None, verbose: bool = True, client=None):
    """Run LLM OCR cleanup on all leaf texts and penjelasan, mutating the tree in place.

    Preserves original text for any node whose cleaned version is missing from the
    LLM response. Returns (failure_messages, token_stats) where token_stats is a dict
    with keys: input_tokens, output_tokens, total_tokens, llm_calls.
    """
    texts_to_clean: list[tuple[str, str]] = []

    def _collect(nodes: list[dict]):
        """Recursively gather leaf node texts, Angka instruction titles, and penjelasan for cleaning."""
        for node in nodes:
            if "nodes" in node:
                # Angka instruction titles (e.g. "Angka 9 — Di antara Pasal 568...") may contain
                # OCR'd pasal refs like "5684" instead of "568A". Clean them so deep_split_leaves
                # later extracts the correct pasal reference from the cleaned title.
                title = node.get("title", "")
                if title and re.match(r'^Angka\s+\d+\s+—\s+', title):
                    texts_to_clean.append((node["node_id"] + ":title", title))
                _collect(node["nodes"])
            else:
                if "text" in node:
                    texts_to_clean.append((node["node_id"], node["text"]))
                if node.get("penjelasan") and node["penjelasan"] != "Cukup jelas.":
                    texts_to_clean.append((node["node_id"] + ":penjelasan", node["penjelasan"]))
                # Leaf Angka nodes also have instruction titles that may contain OCR errors.
                title = node.get("title", "")
                if title and re.match(r'^Angka\s+\d+\s+—\s+', title):
                    texts_to_clean.append((node["node_id"] + ":title", title))

    _collect(output_nodes)

    if penjelasan_data and penjelasan_data.get("umum"):
        texts_to_clean.append(("__penjelasan_umum__", penjelasan_data["umum"]))

    if not texts_to_clean:
        if verbose:
            log.info("no texts to clean")
        return [], {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "llm_calls": 0}

    cleaned, llm_failures, input_tok, output_tok, calls = llm_cleanup_texts(texts_to_clean, verbose=verbose, client=client)

    expected_ids = {nid for nid, _ in texts_to_clean}
    missing = expected_ids - set(cleaned.keys())
    if missing and verbose:
        log.warning(f"{len(missing)} node(s) missing from LLM response, keeping original: {sorted(missing)}")

    def _replace(nodes: list[dict]):
        """Recursively write cleaned text and titles back into nodes."""
        for node in nodes:
            if "nodes" in node:
                title_key = node["node_id"] + ":title"
                if title_key in cleaned:
                    new_title = cleaned[title_key]
                    node["title"] = new_title
                    # After the Angka title is corrected, also update any direct child Pasal
                    # node whose number is a corrupted version of the referenced pasal.
                    # Two sources of such children:
                    #   (a) Fix L2 intermediate node (node_id = parent + "_p")
                    #   (b) Parser-detected pasal heading with OCR'd number (e.g. "5684" for "568A")
                    # Match by numeric prefix: "568A" and "5684" both start with "568".
                    # Scan ALL Pasal refs in the cleaned title for one with a letter suffix.
                    # The lazy .*? in _ANGKA_TITLE_PASAL_RE finds the FIRST pasal ("Pasal 568"),
                    # but the inserted pasal with a letter suffix ("Pasal 568A") may appear later
                    # (e.g. "...Di antara Pasal 568 dan Pasal 569 disisipkan ... yakni Pasal 568A").
                    all_pasal_nums = re.findall(r'Pasal\s+(\d+\w*)', new_title)
                    new_pasal_ref = None
                    for ref in all_pasal_nums:
                        if len(ref) >= 2 and ref[-1].isalpha():
                            new_pasal_ref = f"Pasal {ref}"
                            break
                    if new_pasal_ref:
                        new_num = re.sub(r'^Pasal\s+', '', new_pasal_ref)  # "568A"
                        # Only propagate when the suffix is a letter (letter A → digit 4 OCR error).
                        # "5684": same prefix "568", trailing "4" (digit) vs "A" (letter) → MATCH
                        # "568":  different length → SKIP (avoids clobbering "Pasal 568")
                        # "568A": already correct → SKIP
                        prefix = new_num[:-1]  # "568"
                        for child in node.get("nodes", []):
                            child_title = child.get("title", "")
                            child_num = re.sub(r'^Pasal\s+', '', child_title)
                            # Case A: OCR replaced letter with digit → "5684" → same length
                            case_a = (len(child_num) == len(new_num)
                                      and child_num[:-1] == prefix
                                      and child_num[-1] != new_num[-1]
                                      and child_num[-1].isdigit())
                            # Case B: parse-time regex grabbed the bare prefix → "568" → shorter by 1
                            # (lazy .*? in _ANGKA_TITLE_PASAL_RE stops at first Pasal ref in title)
                            case_b = (child_num == prefix)
                            if case_a or case_b:
                                child["title"] = new_pasal_ref
                _replace(node["nodes"])
            else:
                if "text" in node and node["node_id"] in cleaned:
                    node["text"] = cleaned[node["node_id"]]
                penj_key = node.get("node_id", "") + ":penjelasan"
                if penj_key in cleaned:
                    node["penjelasan"] = cleaned[penj_key]
                title_key = node.get("node_id", "") + ":title"
                if title_key in cleaned:
                    node["title"] = cleaned[title_key]

    _replace(output_nodes)
    # Re-run local OCR/header normalization after the LLM pass so llm-only rebuilds
    # get the same cleanup guarantees as a full parse path.
    strip_ocr_headers(output_nodes)

    if penjelasan_data and "__penjelasan_umum__" in cleaned:
        penjelasan_data["umum"] = _normalize_leaf_text(cleaned["__penjelasan_umum__"])

    if verbose:
        log.info(f"cleaned {len(cleaned)} nodes")

    token_stats = {
        "input_tokens": input_tok,
        "output_tokens": output_tok,
        "total_tokens": input_tok + output_tok,
        "llm_calls": calls,
    }
    return llm_failures, token_stats


def strip_ocr_headers(nodes: list[dict]):
    """Remove residual page headers, footers, and pengesahan text from all leaf nodes in-place."""
    # Recurse into container nodes; apply patterns only to leaf text.
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            strip_ocr_headers(node["nodes"])
        if "text" in node:
            node["text"] = _normalize_leaf_text(node["text"], node.get("title"))
        if "penjelasan" in node:
            node["penjelasan"] = _normalize_leaf_text(node["penjelasan"], node.get("title"))

# ============================================================
# 7. MAIN PIPELINE
# ============================================================

def parse_legal_pdf(pdf_path: str, verbose: bool = True,
                    use_llm_cleanup: bool = True,
                    is_perubahan: bool | None = None,
                    granularity: str = "pasal") -> dict:
    """Parse an Indonesian legal PDF and return a PageIndex-compatible document dict.

    This is the primary entry point for the indexing pipeline. It runs text
    extraction, structural detection, tree building, penjelasan parsing, and
    optional LLM OCR cleanup in sequence.

    The returned dict contains document metadata and a 'structure' key holding
    the node tree consumed by downstream indexers.
    """
    if granularity not in ("pasal", "ayat", "full_split"):
        raise ValueError(f"Unknown granularity: {granularity!r}. " f"Use 'pasal', 'ayat', or 'full_split'.")
    pdf_path = str(pdf_path)
    pdf_name = Path(pdf_path).name
    total_steps = 6 if use_llm_cleanup else 5
    t_start = time.time()

    # Step 1: Extract raw text from all pages.
    if verbose:
        log.info(f"[1/{total_steps}] extracting text from {pdf_name}")
    t0 = time.time()
    pages = extract_pages(pdf_path)
    total_pages = len(pages)
    if verbose:
        log.info(f"total pages: {total_pages} ({time.time() - t0:.1f}s)")

    # Step 2: Apply OCR normalization to each page.
    if verbose:
        log.info(f"[2/{total_steps}] cleaning text")
    t0 = time.time()
    for page in pages:
        page["clean_text"] = clean_page_text(page["raw_text"])
    if verbose:
        log.info(f"done ({time.time() - t0:.1f}s)")

    # Step 3: Locate section boundaries (body, penjelasan, pengesahan).
    if verbose:
        log.info(f"[3/{total_steps}] detecting document sections")
    closing_page = find_closing_page(pages)
    penjelasan_page = find_penjelasan_page(pages)

    # Body ends at the page before whichever post-body section starts first.
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

    if is_perubahan is None:
        is_perubahan = detect_perubahan(pages)
    if verbose and is_perubahan:
        log.info("detected as Perubahan (amendment)")

    # Step 4: Detect BAB, Bagian, Paragraf, and Pasal elements within the body.
    if verbose:
        log.info(f"[4/{total_steps}] detecting structural elements")
    t0 = time.time()
    elements = detect_elements(pages, body_end, is_perubahan=is_perubahan)

    # Count each element type for the log summary.
    type_counts = {}
    for e in elements:
        type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
    if verbose:
        log.info(f"found: {type_counts} ({time.time() - t0:.1f}s)")

    is_omnibus = detect_omnibus(pages, elements)
    if verbose and is_omnibus:
        log.info("detected as Omnibus law")
    warnings = validate_pasal_sequence(elements, is_perubahan=is_perubahan or is_omnibus)
    # Emit each sequence warning to the log.
    for w in warnings:
        log.warning(w)

    # Step 5: Build the node tree and fix sibling boundaries.
    if verbose:
        log.info(f"[5/{total_steps}] building tree structure")
    t0 = time.time()
    tree = build_tree(elements, body_end)
    fix_node_boundaries(tree, body_end)

    # In Perubahan documents, wrap any orphan leading Pasals in a synthetic Angka 1
    # when OCR dropped the "1." prefix of the first amendment instruction.
    # NOTE: consolidate_bab_in_perubahan was removed — it incorrectly absorbed Pasals
    # from Angka N+1 into the BAB defined by Angka N, deleting the source Angka. Each
    # amendment Angka (BAB rename and Pasal change) should remain a separate node.
    if is_perubahan:
        angka_wrapped = wrap_orphan_pasals_in_angka1(tree, pages=pages)
        if verbose and angka_wrapped:
            log.info(f"wrapped {angka_wrapped} orphan Pasal(s) under synthetic Angka 1")

    # Build the preamble node from text before the first structural element.
    output_nodes = []

    if elements:
        first_elem = elements[0]
        first_elem_page = first_elem["page_num"]
        first_elem_offset = first_elem["char_offset"]
        # Skip if the document body starts at the very first character.
        if first_elem_page > 1 or first_elem_offset > 0:
            preamble_text = _extract_node_text(pages, 1, first_elem_page, end_char_offset=first_elem_offset)
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
                # If preamble splitting fails, emit the text as a single unsplit node.
                output_nodes.append({
                    "title": "Pembukaan (Menimbang, Mengingat, Memutuskan)",
                    "node_id": "P000",
                    "start_index": 1,
                    "end_index": first_elem_page,
                    "text": preamble_text,
                })

    output_nodes.extend(clean_tree_for_output(tree, pages))

    # Parse the PENJELASAN section and attach each entry to its Pasal leaf node.
    penjelasan_data = None
    if penjelasan_page:
        penjelasan_data = parse_penjelasan(pages, penjelasan_page, total_pages)
        attach_penjelasan(output_nodes, penjelasan_data["pasal"])
        matched = sum(1 for n in iter_leaves(output_nodes) if n.get("penjelasan"))
        if verbose:
            log.info(f"penjelasan: {len(penjelasan_data['pasal'])} pasal parsed, " f"{matched} matched to tree nodes")

    if verbose:
        log.info(f"done ({time.time() - t0:.1f}s)")

    # Step 6: LLM cleanup of remaining OCR artifacts.
    llm_time = 0.0
    if use_llm_cleanup:
        if verbose:
            log.info(f"[6/{total_steps}] cleaning text with Gemini 2.5 Flash")
        t0 = time.time()
        llm_failures, _token_stats = apply_llm_cleanup(output_nodes, penjelasan_data, verbose=verbose)
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

    # Normalize leaf text before splitting so that title-stripping (Fix C) and
    # spillover trimming are applied to the pasal-level text first.  When LLM
    # cleanup ran, strip_ocr_headers was already called inside apply_llm_cleanup;
    # calling it again here is a no-op.  When --no-llm is used this is the first
    # (and only) pre-split normalization pass.
    strip_ocr_headers(output_nodes)

    # Split Pasal leaves into finer sub-nodes based on the requested granularity.
    if granularity == "ayat":
        output_nodes = ayat_split_leaves(output_nodes)
    elif granularity == "full_split":
        output_nodes = deep_split_leaves(output_nodes)

    # Second pass: normalize any text on the newly-created child nodes.
    strip_ocr_headers(output_nodes)

    # Only store penjelasan at the document level if no Pasal nodes were matched.
    # In Perubahan documents, amended Pasals appear as leaf nodes and should match normally, making doc-level storage rarely needed.
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
    """Print each node's id, title, and page range, indented by depth."""
    for node in nodes:
        prefix = "  " * indent
        page_range = f"[p.{node['start_index']}-{node['end_index']}]"
        print(f"{prefix}{node.get('node_id', '----')} {node['title']} {page_range}")
        if "nodes" in node:
            print_tree(node["nodes"], indent + 1)

# ============================================================
# 8. SUB-PASAL LEAF SPLITTING
# ============================================================
# Splits Pasal leaf nodes into finer sub-nodes based on the requested granularity.
# The split hierarchy is: Pasal, then Ayat (1)/(2)/..., then Huruf a./b./..., then Angka 1./2./...
# The "ayat" granularity splits to Ayat only; "full_split" recurses to the deepest level present.

# Patterns for detecting sub-Pasal structure and penjelasan section headings.
_AYAT_RE = re.compile(r'(?:^|\n)\((\d+)\)\s', re.MULTILINE)
_HURUF_RE = re.compile(r'(?:^|\n)([a-z])\.\s', re.MULTILINE)
_ANGKA_ITEM_RE = re.compile(r'(?:^|\n)(\d+)\.\s', re.MULTILINE)
_PENJ_AYAT_RE = re.compile(r'Ayat\s*\((\d+)\)\s*\n', re.MULTILINE)
_PENJ_HURUF_RE = re.compile(r'Huruf\s+([a-z])\s*\n', re.MULTILINE)

# Words that, when appearing at the end of the line immediately before a marker,
# indicate the marker is part of an inline cross-line reference rather than a
# structural marker.  Example: "dimaksud pada ayat\n(1) harus" — the "(1)" here
# is a reference, not the start of Ayat 1.
_INLINE_REF_PREV_WORDS = frozenset(['ayat', 'pasal', 'huruf', 'angka', 'pada', 'di'])


def _is_inline_ref(text: str, match: re.Match) -> bool:
    """Return True if this marker match is a cross-line inline reference.

    Detects cases where OCR/PDF line-wrapping splits an inline reference such as
    "dimaksud pada ayat\\n(1) harus" — the "\\n(1)" looks like an Ayat marker but
    is actually a continuation of the previous line's reference.
    """
    nl_pos = text.rfind('\n', 0, match.start() + 1)
    if nl_pos == -1:
        return False  # match is at the very start of text
    prev_nl = text.rfind('\n', 0, nl_pos)
    prev_line = text[prev_nl + 1: nl_pos].strip().lower()
    last_word = prev_line.split()[-1] if prev_line.split() else ''
    return last_word in _INLINE_REF_PREV_WORDS


_FLAT_AYAT_NORM_RE = re.compile(r'\(\d+\)\s')


def _normalize_flat_structural_text(text: str) -> str:
    """Insert newlines before structural markers in flat (no-newline) text.

    LLM cleanup sometimes collapses multi-paragraph legal text to a single line,
    breaking the newline-anchored marker patterns used by _find_and_validate_markers.
    This recovers structure by:
    - Inserting \\n before ayat markers (N) not preceded by inline-reference words
    - Inserting \\n before huruf markers (a., b.) that follow list separators (: or ;)
    """
    if '\n' in text:
        return text  # Already structured, no normalization needed

    # Insert \n before (N) ayat markers, skipping inline cross-references.
    # Inline refs are preceded by words like "ayat", "pasal", "pada", etc.
    inline_words = frozenset(['ayat', 'pasal', 'huruf', 'angka', 'pada', 'di', 'dalam'])

    def _maybe_insert_ayat_nl(m: re.Match) -> str:
        if m.start() == 0:
            return m.group(0)
        window = text[max(0, m.start() - 40): m.start()]
        words = window.split()
        last_word = words[-1].lower().rstrip('.,;():') if words else ''
        return m.group(0) if last_word in inline_words else '\n' + m.group(0)

    text = _FLAT_AYAT_NORM_RE.sub(_maybe_insert_ayat_nl, text)

    # Insert \n before huruf markers (a., b., c.) following : or ;.
    # Handles the optional "dan" before the last item: "; dan c. " -> ";\nc. "
    text = re.sub(
        r'([;:])\s+(?:dan\s+)?([a-z])\.\s',
        lambda m: f'{m.group(1)}\n{m.group(2)}. ',
        text,
    )

    return text


def _find_and_validate_markers(text: str, pattern: re.Pattern, expected_start: str) -> list[tuple[int, str]] | None:
    """Find structural markers in text and verify they form a consecutive sequence.

    A valid sequence starts at expected_start and increments by one (letters a, b, c
    or numbers 1, 2, 3). Returns a list of (char_pos, label) tuples if at least two
    consecutive markers are found, or None otherwise.
    """
    matches = [m for m in pattern.finditer(text) if not _is_inline_ref(text, m)]
    if len(matches) < 2:
        return None

    # Collapse consecutive matches with the same label. Page-break text overlap
    # can produce duplicates, e.g. "(5) Selain" appearing twice on adjacent pages.
    deduped: list[re.Match] = [matches[0]]
    for m in matches[1:]:
        if m.group(1) == deduped[-1].group(1):
            deduped[-1] = m  # keep the later, more complete occurrence
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


def _split_text_by_markers(text: str, markers: list[tuple[int, str]]) -> tuple[str, list[tuple[str, str]]]:
    """Split text at each marker position into labelled segments.

    Returns a 2-tuple of (intro_text, segments) where intro_text is the content
    before the first marker and segments is a list of (label, segment_text) pairs.
    """
    positions = [pos for pos, _ in markers]
    labels = [label for _, label in markers]

    intro = text[:positions[0]].strip()

    segments = []
    # Slice between adjacent markers; the final segment extends to the end of the text.
    for i, (pos, label) in enumerate(zip(positions, labels)):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        segment = text[pos:end].strip()
        segments.append((label, segment))

    return intro, segments


def _distribute_penjelasan(penjelasan: str | None, sub_nodes: list[dict], kind: str) -> None:
    """Assign penjelasan text to each sub-node in-place.

    Parses the penjelasan for per-item section headings (Ayat (N) or Huruf X) and
    assigns each matched sub-node its specific excerpt. Sub-nodes with no matching
    entry receive the full penjelasan text. For Angka items, the full text is always
    assigned since there is no known per-item heading pattern.
    """
    if not penjelasan:
        return

    if kind == "ayat":
        split_re = _PENJ_AYAT_RE
    elif kind == "huruf":
        split_re = _PENJ_HURUF_RE
    else:
        # Angka has no per-item penjelasan heading pattern; assign full text to all.
        for node in sub_nodes:
            node["penjelasan"] = penjelasan
        return

    parts = split_re.split(penjelasan)
    # re.split with a capturing group produces alternating [text, label, text, label, ...]
    penj_map = {}
    for i in range(1, len(parts) - 1, 2):
        penj_map[parts[i]] = parts[i + 1].strip()

    # Assign each sub-node its specific excerpt, or the full text if no match found.
    for node in sub_nodes:
        label = node.get("_split_label")
        if label and label in penj_map:
            node["penjelasan"] = penj_map[label]
        else:
            node["penjelasan"] = penjelasan


def _strip_leading_junk(text: str) -> str:
    """Strip leading non-structural characters (colon, whitespace) from text."""
    return re.sub(r'^[\s:;,\-]+', '', text)


def _prepend_intro_to_first_child(sub_nodes: list[dict], intro: str) -> None:
    """Preserve text before the first split marker by prepending it to child 1."""
    if not intro or not sub_nodes:
        return
    intro = intro.strip()
    if not intro:
        return
    first = sub_nodes[0]
    existing = first.get("text", "").strip()
    first["text"] = f"{intro}\n{existing}".strip() if existing else intro


def _try_ayat_split(text: str, parent_id: str, parent_title: str, parent_start: int, parent_end: int, penjelasan: str | None,) -> list[dict] | None:
    """Split text into Ayat sub-nodes without recursing into Huruf or Angka.

    Returns a list of Ayat child dicts if Ayat markers are found, or None.
    """
    text = _strip_leading_junk(text)
    text = _normalize_flat_structural_text(text)

    ayat_markers = _find_and_validate_markers(text, _AYAT_RE, "1")
    if not ayat_markers:
        return None

    intro, segments = _split_text_by_markers(text, ayat_markers)
    sub_nodes = []
    # Build one Ayat sub-node for each matched segment.
    for i, (label, segment) in enumerate(segments, 1):
        # Strip the leading "(N) " ayat marker that _split_text_by_markers keeps.
        segment = re.sub(r'^\(\d+\)\s*', '', segment)
        if not segment.strip():
            continue  # Skip ayat with no content (PDF extraction gap between consecutive markers)
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
    _prepend_intro_to_first_child(sub_nodes, intro)
    _distribute_penjelasan(penjelasan, sub_nodes, "ayat")
    for n in sub_nodes:
        n.pop("_split_label", None)
    return sub_nodes


def _split_leaves_with(nodes: list[dict], split_func) -> list[dict]:
    """Apply a split function to every leaf node, preserving the container structure."""
    result = []
    # Recurse into container nodes; apply split_func only to leaves that have text.
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            # If node has both text and children (e.g. Pasal I amendment preamble),
            # promote the text to a synthetic intro child so it can be split too.
            if "text" in node and node["text"].strip():
                preamble = {
                    "title": f"{node['title']} Pembukaan",
                    "node_id": f"{node['node_id']}_intro",
                    "start_index": node["start_index"],
                    "end_index": node.get("end_index", node["start_index"]),
                    "text": node.pop("text"),
                }
                node["nodes"].insert(0, preamble)
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
    """Split every leaf node in the tree into Ayat sub-nodes.

    Only splits at the Ayat level; Huruf and Angka sub-structure is not examined.
    Leaves with no Ayat markers are kept unchanged.
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


def _try_huruf_split(text: str, parent_id: str, parent_title: str, parent_start: int, parent_end: int) -> list[dict] | None:
    """Try splitting text into Huruf sub-nodes (a., b., c., ...) without recursing further.

    Used by Angka items that may contain sub-lists like definitions with a./b./c./d.
    """
    text = _strip_leading_junk(text)
    text = _normalize_flat_structural_text(text)
    huruf_markers = _find_and_validate_markers(text, _HURUF_RE, "a")
    if not huruf_markers:
        return None
    intro, segments = _split_text_by_markers(text, huruf_markers)
    sub_nodes = []
    for i, (label, segment) in enumerate(segments, 1):
        segment = re.sub(r'^[a-z]\.\s*', '', segment)
        if not segment.strip():
            continue  # Skip huruf with no content (PDF extraction gap)
        sub_id = f"{parent_id}_h{i}"
        sub_title = f"{parent_title} Huruf {label}"
        sub_nodes.append({
            "title": sub_title,
            "node_id": sub_id,
            "start_index": parent_start,
            "end_index": parent_end,
            "text": segment,
            "_split_label": label,
        })
    _prepend_intro_to_first_child(sub_nodes, intro)
    for n in sub_nodes:
        n.pop("_split_label", None)
    return sub_nodes


def _try_deep_split(text: str, parent_id: str, parent_title: str, parent_start: int, parent_end: int, penjelasan: str | None,) -> list[dict] | None:
    """Recursively split text into the smallest structural sub-nodes present.

    Tries Ayat first, then Huruf, then Angka. Each level recurses into the next.
    Returns a list of child node dicts if a split was found, or None.
    """
    text = _strip_leading_junk(text)
    text = _normalize_flat_structural_text(text)

    # Try Ayat: (1), (2), (3), ...
    ayat_markers = _find_and_validate_markers(text, _AYAT_RE, "1")
    if ayat_markers:
        intro, segments = _split_text_by_markers(text, ayat_markers)
        sub_nodes = []
        # Build one Ayat sub-node per segment, recursing into each for deeper structure.
        for i, (label, segment) in enumerate(segments, 1):
            # Strip the leading "(N) " ayat marker that _split_text_by_markers keeps.
            segment = re.sub(r'^\(\d+\)\s*', '', segment)
            if not segment.strip():
                continue  # Skip ayat with no content (PDF extraction gap)
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
        _prepend_intro_to_first_child(sub_nodes, intro)
        _distribute_penjelasan(penjelasan, sub_nodes, "ayat")
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    # Try Angka items: 1., 2., 3., ...
    # Angka is tried before Huruf so that amendment Angka bodies (which contain numbered
    # definitions with nested huruf) split correctly at the Angka level first. Without this
    # ordering, Huruf markers deep inside (e.g. "a." in definition 3) would win over the
    # top-level Angka structure.
    angka_markers = _find_and_validate_markers(text, _ANGKA_ITEM_RE, "1")
    if angka_markers:
        intro, segments = _split_text_by_markers(text, angka_markers)
        sub_nodes = []
        # Build one Angka sub-node per segment, recursing into Huruf if present.
        for i, (label, segment) in enumerate(segments, 1):
            # Strip the leading "N. " angka marker that _split_text_by_markers keeps.
            segment = re.sub(r'^\d+\.\s*', '', segment)
            if not segment.strip():
                continue  # Skip angka with no content (PDF extraction gap)
            sub_id = f"{parent_id}_n{i}"
            sub_title = f"{parent_title} Angka {label}"
            # Recurse into Huruf sub-structure (e.g. definitions with a./b./c. lists).
            children = _try_huruf_split(
                segment, sub_id, sub_title, parent_start, parent_end,
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
        _prepend_intro_to_first_child(sub_nodes, intro)
        _distribute_penjelasan(penjelasan, sub_nodes, "angka")
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    # Try Huruf: a., b., c., ...
    huruf_markers = _find_and_validate_markers(text, _HURUF_RE, "a")
    if huruf_markers:
        intro, segments = _split_text_by_markers(text, huruf_markers)
        sub_nodes = []
        # Build one Huruf sub-node per segment, recursing into each for deeper structure.
        for i, (label, segment) in enumerate(segments, 1):
            # Strip the leading "x. " huruf marker that _split_text_by_markers keeps.
            segment = re.sub(r'^[a-z]\.\s*', '', segment)
            if not segment.strip():
                continue  # Skip huruf with no content (PDF extraction gap)
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
        _prepend_intro_to_first_child(sub_nodes, intro)
        _distribute_penjelasan(penjelasan, sub_nodes, "huruf")
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    return None


_ANGKA_TITLE_PASAL_RE = re.compile(r'^Angka\s+\d+\s+—\s+.*?(Pasal\s+\d+\w*)')

# Matches the amendment instruction at the start of an Angka's text, e.g.:
#   "1. \nKetentuan Pasal 1 angka 135 ... sehingga Pasal 1 berbunyi sebagai berikut:"
# Strips this so the replacement content (definitions list) can be split cleanly.
_AMENDMENT_INSTR_PREFIX_RE = re.compile(
    r'^\d+\.\s*\n?\s*(?:Ketentuan|Di\s+antara|Diantara)'
    r'.*?(?:sebagai\s+berikut|berikut)\s*:\s*\n',
    re.DOTALL | re.IGNORECASE,
)


def _clean_angka_for_deep_split(title: str, text: str) -> tuple[str, str]:
    """Clean amendment Angka title and text for better deep-split output.

    For amendment Angka nodes (title "Angka N — <instruction>..."):
    - Extracts the target Pasal reference for cleaner child titles
    - Strips the leading instruction text so definitions start cleanly
    """
    m = _ANGKA_TITLE_PASAL_RE.match(title)
    if m:
        title = m.group(1)
        text = _AMENDMENT_INSTR_PREFIX_RE.sub('', text, count=1)
    return title, text


def deep_split_leaves(nodes: list[dict]) -> list[dict]:
    """Recursively split every leaf node in the tree to its deepest sub-structure.

    Tries Ayat, then Angka, then Huruf at each level. Leaves with no detectable
    sub-structure are kept unchanged.

    For amendment Angka nodes whose title references a specific Pasal (e.g.
    "Angka 1 — Ketentuan Pasal 3 diubah..."), an intermediate Pasal node is
    inserted so the hierarchy reads: Angka N > Pasal X > [Ayat/Angka/Huruf].
    """
    def _split(node: dict):
        title = node["title"]
        text = node["text"]

        # If this Angka's title names a specific Pasal, wrap the split result in
        # an intermediate Pasal node instead of flattening children directly under
        # the Angka. This preserves the Pasal I > Angka N > Pasal X > ... hierarchy
        # for amendment documents.
        m = _ANGKA_TITLE_PASAL_RE.match(title)
        if m:
            pasal_ref = m.group(1)  # e.g. "Pasal 1"
            cleaned_text = _AMENDMENT_INSTR_PREFIX_RE.sub('', text, count=1)
            pasal_id = f"{node['node_id']}_p"
            children = _try_deep_split(
                text=cleaned_text,
                parent_id=pasal_id,
                parent_title=pasal_ref,
                parent_start=node["start_index"],
                parent_end=node["end_index"],
                penjelasan=node.get("penjelasan"),
            )
            if children:
                return [{
                    "title": pasal_ref,
                    "node_id": pasal_id,
                    "start_index": node["start_index"],
                    "end_index": node["end_index"],
                    "nodes": children,
                }]
            # No sub-structure found — fall through to plain deep split below.
            title, text = pasal_ref, cleaned_text

        return _try_deep_split(
            text=text,
            parent_id=node["node_id"],
            parent_title=title,
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

    # Extract boolean flags from sys.argv before processing positional args.
    args = sys.argv[1:]
    skip_llm = "--no-llm" in args
    if skip_llm:
        args.remove("--no-llm")
    use_llm = not skip_llm

    # Extract the --granularity flag and its value, defaulting to pasal.
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

    # Remaining args form the PDF path; join them to handle paths containing spaces.
    pdf_arg = " ".join(args).strip() if args else ""
    if not pdf_arg:
        pdf_dir = Path("data/raw/UU/pdfs")
        # When no path is given, process all main UU PDFs, excluding Lampiran and Salinan variants.
        pdf_files = sorted([
            f for f in pdf_dir.glob("UU Nomor *.pdf")
            if "Lampiran" not in f.name and "Salinan" not in f.name
        ])
        if not pdf_files:
            log.error("no PDF files found. Provide a path as argument.")
            sys.exit(1)
    else:
        pdf_files = [Path(pdf_arg)]

    # Parse each PDF and write its result to a JSON file.
    for pdf_path in pdf_files:
        log.info(f"processing {pdf_path.name}")

        result = parse_legal_pdf(str(pdf_path), use_llm_cleanup=use_llm,
                                granularity=gran)

        print(f"\nTree structure ({gran})")
        print_tree(result["structure"])

        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / (pdf_path.stem + "_structure.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        log.info(f"saved to {output_path}")
