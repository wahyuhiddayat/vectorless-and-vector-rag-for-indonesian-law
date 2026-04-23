"""PDF text utilities for Indonesian legal documents.

Four sections:
  1. TEXT EXTRACTION — PyMuPDF reader with multi-column reordering and
     OCR digit/header cleanup. Entry points: extract_pages, clean_page_text.
  2. PENJELASAN PARSING — detect + parse the explanatory section.
  3. LEAF-TEXT UTILITIES — OCR residual cleanup, heading dedup, spillover
     trim. Applied after re-split to keep leaf text tidy.
  4. SUB-PASAL LEAF SPLITTING — deterministic pasal → ayat → huruf → angka
     splitter (no LLM). Called from build.py to derive ayat and rincian
     granularities from the LLM-produced pasal tree.

Structure is produced by scripts/parser/llm_parse.py (LLM-first). This
module only handles text extraction and deterministic re-split.
"""
import json
import logging
import re

import fitz

log = logging.getLogger(__name__)

# ============================================================
# 1. TEXT EXTRACTION & CLEANING
# ============================================================

def _detect_two_columns(blocks: list[dict], page_width: float, is_landscape: bool = False) -> list[dict]:
    """Reorder text blocks for correct reading order on multi-column pages.

    Pages with fewer than 4 blocks are returned sorted top-to-bottom,
    left-to-right. Otherwise the function attempts column detection: if both
    halves are sufficiently populated and full-width blocks are rare (<30%),
    the page is treated as two-column and emitted as left column first then
    right column. Applies to both portrait and landscape pages — some
    Indonesian legal PDFs use narrow two-column layouts even in portrait
    orientation (e.g. government regulations with side-by-side articles).
    """
    if len(blocks) < 4:
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

    If a cached reordered-pages JSON exists at
    `data/reordered_pages/<doc_id>.json`, use it instead.

    Returns a list of dicts with keys: page_num (1-indexed), raw_text.
    """
    from pathlib import Path as _Path
    # Derive doc_id from pdf filename — find matching registry entry.
    pdf_name = _Path(pdf_path).name
    cache_dir = _Path("data/reordered_pages")
    if cache_dir.exists():
        try:
            reg_path = _Path("data/raw/registry.json")
            if reg_path.exists():
                reg = json.load(open(reg_path, encoding="utf-8"))
                for doc_id, entry in reg.items():
                    jenis = entry.get("jenis_folder", "")
                    if jenis and f"raw/{jenis}/pdfs/{pdf_name}" in pdf_path.replace("\\", "/"):
                        cache = cache_dir / f"{doc_id}.json"
                        if cache.exists():
                            log.info(f"extract_pages: using reordered cache {cache}")
                            return json.load(open(cache, encoding="utf-8"))
                        break
        except Exception as e:
            log.warning(f"extract_pages cache lookup failed: {e}")

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
        # Close paren OCR'd as trailing "1"/"l"/"I": "(21 " → "(2) " (common in
        # numbered-ayat lists where trailing char is misread).
        line = re.sub(r'\((\d)[1lI](?=\s)', lambda m: '(' + m.group(1) + ')', line)
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
# 3. LEAF-TEXT UTILITIES (OCR normalization, tree iteration)
# ------------------------------------------------------------
# Shared helpers used by section 4 (re-split) and by downstream
# consumers (build.py, retrieval). Safe to modify only if you
# understand the OCR artifacts they handle.
# ============================================================


def iter_leaves(nodes: list[dict]):
    """Yield every leaf node in the tree (nodes that have no children)."""
    # Recurse into non-empty node lists; yield nodes with no children as leaves.
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            yield from iter_leaves(node["nodes"])
        else:
            yield node


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
# 4. SUB-PASAL LEAF SPLITTING (pasal → ayat → huruf → angka)
# ------------------------------------------------------------
# Called by build.py --from-pasal to derive ayat + rincian
# granularities from pasal-level bodies (source of truth is LLM
# output, whether regex-produced or llm_parse-produced).
# OCR-tolerant via _find_fuzzy_markers. DO NOT simplify without
# preserving the fuzzy matcher's fallback chain (majority-match
# letter remap, one-slot gap-fill, OCR digit normalization).
# ============================================================
# Splits Pasal leaf nodes into finer sub-nodes based on the requested granularity.
# The split hierarchy is: Pasal, then Ayat (1)/(2)/..., then Huruf a./b./..., then Angka 1./2./...
# The "ayat" granularity splits to Ayat only; "rincian" recurses to the deepest level present.

# Patterns for detecting sub-Pasal structure and penjelasan section headings.
# Huruf: 1-2 letters (handles a..z and doubled aa..zz for lists longer than
# 26 items), period optional to tolerate OCR loss. Case-insensitive because
# OCR sometimes capitalizes isolated letters (e.g. "l." → "L "). Sequence
# validation downstream rejects misfires.
_HURUF_RE = re.compile(
    r'(?:^|\n)[ \t]*([A-Za-z]{1,2})\.?(?=\s+\S)',
    re.MULTILINE,
)
# Sub-huruf: "a) b) c)" style, used inside angka bodies as a deeper level.
_SUB_HURUF_RE = re.compile(r'(?:^|\n)\s*([a-z])\)\s', re.MULTILINE)

# Words that, when appearing at the end of the line immediately before a marker,
# indicate the marker is part of an inline cross-line reference rather than a
# structural marker.  Example: "dimaksud pada ayat\n(1) harus" — the "(1)" here
# is a reference, not the start of Ayat 1.
_INLINE_REF_PREV_WORDS = frozenset([
    # Prepositional anchors for a cross-reference that can OCR-break to next line.
    'pada', 'di', 'dalam',
    # Structural nouns — references read "Pasal N ayat (M)" where the next
    # token may break to the following line.
    'ayat', 'pasal', 'huruf', 'angka',
    # NOTE: conjunctions like "dan", "atau", "serta", "maupun" are NOT in this
    # set because Indonesian legal drafting convention uses them as LIST
    # conjunctions: "f. item; dan\ng. last item." — the "dan" ends a list
    # item, and the next line IS a structural marker, not an inline ref.
])


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


# Strict marker patterns — only match well-formed tokens. Fast path for
# clean LLM-produced text. Used before the OCR-tolerant fuzzy variants.
_STRICT_AYAT_RE = re.compile(r'(?:^|\n)[ \t]*\((\d+)\)(?=\s|$)', re.MULTILINE)
# Angka: digits at line start. Accept closing with period ("1."), closing
# paren ("1)"), or no closing char at all (OCR that drops the period).
# Sequence validation downstream rejects random digit lines that do not
# form a consecutive run starting at 1.
_STRICT_ANGKA_ITEM_RE = re.compile(
    r'(?:^|\n)[ \t]*(\d+)[.)]?(?=\s+\S)',
    re.MULTILINE,
)

# Fuzzy marker candidate regex — tolerate common OCR noise around digits.
# Ayat: digit-ish token optionally wrapped in () or [] with OCR noise.
#   Accepted: (1), (l), (I), (O), (21, 21), (2l), l2l, t2t, lN), (Nt, etc.
# The char-class 'OIilot' captures both digits and their common OCR letter
# substitutes (O/0, I/1, i/1, l/1, o/0, t as paren substitute).
_FUZZY_AYAT_RE = re.compile(
    r'(?:^|\n)[ \t]*[(\[lt]?[ \t]*(\d[0-9OIilot]{0,2}|[OIilot]\d[OIilot]?|[OIilo])[ \t]*[)\]lt]?(?=\s|$)',
    re.MULTILINE,
)
# Angka: digit-ish token followed by period.
_FUZZY_ANGKA_ITEM_RE = re.compile(
    r'(?:^|\n)[ \t]*(\d[0-9OIilot]{0,2}|[OIilo])\.(?=\s)',
    re.MULTILINE,
)


def _huruf_to_index(label: str) -> int | None:
    """Map a huruf label to its 1-based position: a=1..z=26, aa=27..zz=52."""
    lab = label.lower()
    if len(lab) == 1 and "a" <= lab <= "z":
        return ord(lab) - ord("a") + 1
    if len(lab) == 2 and lab[0] == lab[1] and "a" <= lab[0] <= "z":
        return 26 + ord(lab[0]) - ord("a") + 1
    return None


def _index_to_huruf(i: int) -> str:
    """Inverse of _huruf_to_index."""
    if 1 <= i <= 26:
        return chr(ord("a") + i - 1)
    if 27 <= i <= 52:
        c = chr(ord("a") + i - 27)
        return c + c
    return ""


def _find_fuzzy_markers(
    text: str, kind: str, expected_start: str
) -> list[tuple[int, str, int]] | None:
    """OCR-tolerant marker detection.

    Scans text for loose candidates matching the marker kind
    ("ayat", "angka", "huruf"), normalizes each via _normalize_ocr_digits
    (for digit kinds), filters inline cross-references, validates the
    sequence is consecutive starting at expected_start, and falls back
    to one-slot gap-fill if validation initially fails.

    Returns list of (match_start, normalized_label, match_end) tuples,
    or None if no valid sequence found.
    """
    if kind == "ayat":
        pattern = _FUZZY_AYAT_RE
        strict_pattern: re.Pattern | None = _STRICT_AYAT_RE
        is_numeric = True
    elif kind == "angka":
        pattern = _FUZZY_ANGKA_ITEM_RE
        strict_pattern = _STRICT_ANGKA_ITEM_RE
        is_numeric = True
    elif kind == "huruf":
        pattern = _HURUF_RE
        strict_pattern = None  # _HURUF_RE is already strict.
        is_numeric = False
    else:
        raise ValueError(f"unknown kind: {kind}")

    # Fast path: strict regex only. Clean LLM output has well-formed markers;
    # avoid the fuzzy pattern's false positives (bare digits matching as ayats
    # when the body actually contains angka items like "\n1 mengatasi...").
    if strict_pattern is not None:
        strict_hits: list[tuple[int, str, int]] = []
        for m in strict_pattern.finditer(text):
            if _is_inline_ref(text, m):
                continue
            strict_hits.append((m.start(), m.group(1), m.end()))
        if len(strict_hits) >= 2:
            dd: list[tuple[int, str, int]] = [strict_hits[0]]
            for c in strict_hits[1:]:
                if c[1] == dd[-1][1]:
                    dd[-1] = c  # page-break overlap dedup
                else:
                    dd.append(c)
            try:
                nums = [int(lab) for _, lab, _ in dd]
            except ValueError:
                nums = None
            # Accept any consecutive sequence; amendment PDFs sometimes start
            # ayats at a non-1 number (e.g. body literally shows "(4)...(5)...
            # (6)...(7)" because earlier ayats were lost in PDF extraction).
            if nums and all(
                nums[i + 1] == nums[i] + 1 for i in range(len(nums) - 1)
            ):
                return dd
        # Ayat only exists as "(N)" in Indonesian legal text. If strict found
        # no valid consecutive sequence, there ARE no ayats here — do NOT fall
        # through to fuzzy (which would match bare digits as ayat labels).
        if kind == "ayat":
            return None

    # Stage 1: collect raw candidates, normalize, filter inline refs.
    candidates: list[tuple[int, str, int]] = []
    for m in pattern.finditer(text):
        if _is_inline_ref(text, m):
            continue
        raw_label = m.group(1)
        if is_numeric:
            norm = _normalize_ocr_digits(raw_label)
            # Must normalize to pure digits with plausible magnitude.
            if not norm.isdigit():
                continue
            # Ayat/Angka rarely exceed 3 digits; reject noise like "000".
            if len(norm) > 3 or (norm.startswith("0") and norm != "0"):
                continue
        else:
            # Huruf: 1-2 letters (a..z, aa..zz). Normalize to lowercase
            # to tolerate OCR capitalization. Reject invalid shapes
            # (e.g. "ab" mixed letters, not a valid huruf label).
            norm = raw_label.lower()
            if _huruf_to_index(norm) is None:
                continue
        candidates.append((m.start(), norm, m.end()))

    if len(candidates) < 2:
        return None

    # Stage 2: deduplicate adjacent duplicates (page-break overlap).
    deduped: list[tuple[int, str, int]] = [candidates[0]]
    for c in candidates[1:]:
        if c[1] == deduped[-1][1]:
            deduped[-1] = c  # keep later, more complete occurrence
        else:
            deduped.append(c)
    if len(deduped) < 2:
        return None

    # Stage 3: validate sequence. Consecutive labels are required; starting
    # number is advisory only (amendment PDFs can legitimately start at a
    # non-1/non-a number when earlier entries are lost in extraction).
    def _validate(seq: list[tuple[int, str, int]]) -> bool:
        labels = [s[1] for s in seq]
        if is_numeric:
            try:
                nums = [int(l) for l in labels]
            except ValueError:
                return False
            return all(nums[i + 1] == nums[i] + 1 for i in range(len(nums) - 1))
        idxs = [_huruf_to_index(l) for l in labels]
        if any(i is None for i in idxs):
            return False
        return all(idxs[i + 1] == idxs[i] + 1 for i in range(len(idxs) - 1))

    if _validate(deduped):
        return deduped

    # Stage 4a: huruf sequence recovery.
    # If majority of positions match expected consecutive sequence
    # (≥70%), assume outliers are OCR artifacts and remap them to the
    # canonical expected letter. Bounded: requires solid majority + first
    # position anchored at expected_start.
    if not is_numeric:
        labels = [lab for _, lab, _ in deduped]
        start_idx = _huruf_to_index(expected_start)
        if not labels or start_idx is None or labels[0] != expected_start:
            return None
        expected_seq = [_index_to_huruf(start_idx + i) for i in range(len(labels))]
        matches = sum(1 for a, b in zip(labels, expected_seq) if a == b)
        if matches / len(labels) < 0.7:
            return None
        # Remap all outlier labels to canonical expected letters.
        remapped = [
            (pos, exp, end)
            for (pos, _, end), exp in zip(deduped, expected_seq)
        ]
        if _validate(remapped):
            return remapped
        return None

    try:
        nums = [int(lab) for _, lab, _ in deduped]
    except ValueError:
        return None
    # Gap-fill needs a plausible anchor but not necessarily start at 1;
    # amendment pasals can legitimately begin at a non-1 ayat number.
    # Find first gap: where nums[i+1] != nums[i] + 1.
    for i in range(len(nums) - 1):
        if nums[i + 1] != nums[i] + 1:
            # Missing number(s) between nums[i] and nums[i+1].
            missing = nums[i] + 1
            if nums[i + 1] != missing + 1:
                # Gap is >1; one-slot fill can't repair. Abort.
                return None
            # Scan raw pattern matches in text[between] for any candidate
            # that normalizes to `missing`. Permit slightly wider character
            # class (include letters that OCR-collapse to digit).
            window_start = deduped[i][2]
            window_end = deduped[i + 1][0]
            window = text[window_start:window_end]
            # Adjust match positions to global coordinates.
            for fm in pattern.finditer(window):
                # Filter inline-ref using global text position.
                global_start = window_start + fm.start()
                # Re-create match-like object for _is_inline_ref (simpler:
                # check prev-line heuristic manually).
                if _window_is_inline_ref(text, global_start):
                    continue
                raw = fm.group(1)
                norm = _normalize_ocr_digits(raw)
                if not norm.isdigit():
                    continue
                if int(norm) == missing:
                    new_marker = (
                        global_start,
                        str(missing),
                        window_start + fm.end(),
                    )
                    filled = deduped[: i + 1] + [new_marker] + deduped[i + 1 :]
                    if _validate(filled):
                        return filled
                    break  # Only try first fitting candidate per slot.
            return None  # No fitting candidate found.
    # Loop fell through = no gap found, but strict validation already failed
    # (possibly wrong start). Abort.
    return None


def _window_is_inline_ref(text: str, pos: int) -> bool:
    """Check if position is preceded by a structural word hinting at cross-ref."""
    nl = text.rfind('\n', 0, pos)
    if nl == -1:
        return False
    prev_nl = text.rfind('\n', 0, nl)
    prev_line = text[prev_nl + 1 : nl].strip().lower()
    if not prev_line:
        return False
    last_word = prev_line.split()[-1]
    return last_word in _INLINE_REF_PREV_WORDS


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


def _split_text_by_markers(
    text: str,
    markers: list[tuple[int, str]] | list[tuple[int, str, int]],
) -> tuple[str, list[tuple[str, str]]]:
    """Split text at each marker position into labelled segments.

    Accepts either 2-tuple (start, label) or 3-tuple (start, label, end).
    When the 3-tuple is used, the segment starts at `end` (skipping the
    marker text entirely) so callers don't need to strip OCR-varied
    marker prefixes. For 2-tuple legacy calls, segment starts at `start`
    and caller must strip the prefix.

    Returns (intro_text, segments) where intro_text is content before
    the first marker and segments is a list of (label, segment_text).
    """
    if not markers:
        return text.strip(), []

    has_end = len(markers[0]) == 3
    starts = [m[0] for m in markers]
    labels = [m[1] for m in markers]
    ends = [m[2] for m in markers] if has_end else None

    intro = text[:starts[0]].strip()

    segments = []
    for i, label in enumerate(labels):
        next_start = starts[i + 1] if i + 1 < len(starts) else len(text)
        # 3-tuple: slice after marker span. 2-tuple: slice from marker start
        # (caller must strip prefix).
        seg_start = ends[i] if has_end else starts[i]
        segment = text[seg_start:next_start].strip()
        segments.append((label, segment))

    return intro, segments


def _strip_leading_junk(text: str) -> str:
    """Strip leading non-structural characters (colon, whitespace) from text."""
    return re.sub(r'^[\s:;,\-]+', '', text)


def _stash_intro_for_parent(sub_nodes: list[dict], intro: str) -> None:
    """Stash parent intro text on the first sub-node for the caller to retrieve.

    The caller in `_split_leaves_with` moves `_intro_for_parent` off the first
    sub-node and places it as `text` on the parent container. This keeps
    intro text at the structurally-correct level (container) instead of
    contaminating the first child's body.
    """
    if not intro or not sub_nodes:
        return
    intro = intro.strip()
    if intro:
        sub_nodes[0]["_intro_for_parent"] = intro


def _try_ayat_split(text: str, parent_id: str, parent_title: str, parent_start: int, parent_end: int, penjelasan: str | None,) -> list[dict] | None:
    """Split text into Ayat sub-nodes without recursing into Huruf or Angka.

    Returns a list of Ayat child dicts if Ayat markers are found, or None.
    """
    text = _strip_leading_junk(text)
    text = _normalize_flat_structural_text(text)
    text = fix_ocr_artifacts(text)

    ayat_markers = _find_fuzzy_markers(text, "ayat", "1")
    if not ayat_markers:
        return None

    intro, segments = _split_text_by_markers(text, ayat_markers)
    sub_nodes = []
    # Build one Ayat sub-node for each matched segment.
    for i, (label, segment) in enumerate(segments, 1):
        if not segment.strip():
            continue  # Skip ayat with no content (PDF extraction gap)
        sub_id = f"{parent_id}_ayat_{label}"
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
    _stash_intro_for_parent(sub_nodes, intro)
    for n in sub_nodes:
        n.pop("_split_label", None)
    return sub_nodes


def _split_leaves_with(nodes: list[dict], split_func) -> list[dict]:
    """Apply a split function to every leaf node, preserving the container structure."""
    result = []
    # Recurse into container nodes; apply split_func only to leaves that have
    # text. Container text is intro text — kept on container (no synthetic
    # "Pembukaan" child — LLM-first output is already structured).
    for node in nodes:
        if "nodes" in node and node["nodes"]:
            node["nodes"] = _split_leaves_with(node["nodes"], split_func)
            result.append(node)
        elif "text" in node and node.get("text"):
            sub_nodes = split_func(node)
            if sub_nodes:
                # Keep penjelasan on the container. Per-child distribution
                # is handled by distribute_penjelasan_to_tree in llm_parse.
                # Recover intro text stashed by split function BEFORE adding
                # nodes, so JSON serializes as {title, ..., text, nodes}.
                branch = {k: v for k, v in node.items() if k != "text"}
                if sub_nodes and "_intro_for_parent" in sub_nodes[0]:
                    branch["text"] = sub_nodes[0].pop("_intro_for_parent")
                branch["nodes"] = sub_nodes
                result.append(branch)
            else:
                result.append(node)
        else:
            result.append(node)
    # Migrate deeper intros up through any container levels the split functions
    # produced (nested deep_split → huruf_split chains stash intro on the
    # deepest leaf; we walk up and surface each to its immediate parent).
    _migrate_stashed_intros(result)
    return result


def _migrate_stashed_intros(nodes: list[dict]) -> None:
    """Walk tree; for every container whose first child has _intro_for_parent,
    move that text onto the container and clear the stash. Also re-orders
    keys so `text` appears before `nodes` in JSON serialization."""
    for node in nodes:
        if node.get("nodes"):
            _migrate_stashed_intros(node["nodes"])
            first = node["nodes"][0] if node["nodes"] else None
            if first and "_intro_for_parent" in first:
                text_val = first.pop("_intro_for_parent")
                if not node.get("text"):
                    # Re-insert keys to put text before nodes.
                    nodes_val = node.pop("nodes")
                    node["text"] = text_val
                    node["nodes"] = nodes_val


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
    Falls back to paren-style "a) b) c)" when period style not found.
    """
    text = _strip_leading_junk(text)
    text = _normalize_flat_structural_text(text)
    text = fix_ocr_artifacts(text)

    # Prefer period-style (standard huruf); fall back to paren-style when absent.
    # Paren-style "a) b) c)" still uses legacy strict matcher — no OCR variation
    # known for parens-style markers in practice.
    huruf_markers = _find_fuzzy_markers(text, "huruf", "a")
    if not huruf_markers:
        legacy = _find_and_validate_markers(text, _SUB_HURUF_RE, "a")
        if legacy:
            huruf_markers = [(s, lab, s + 3) for s, lab in legacy]  # approx end
    if not huruf_markers:
        return None
    intro, segments = _split_text_by_markers(text, huruf_markers)
    sub_nodes = []
    for i, (label, segment) in enumerate(segments, 1):
        if not segment.strip():
            continue  # Skip huruf with no content (PDF extraction gap)
        sub_id = f"{parent_id}_huruf_{label}"
        sub_title = f"{parent_title} Huruf {label}"
        sub_nodes.append({
            "title": sub_title,
            "node_id": sub_id,
            "start_index": parent_start,
            "end_index": parent_end,
            "text": segment,
            "_split_label": label,
        })
    _stash_intro_for_parent(sub_nodes, intro)
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
    text = fix_ocr_artifacts(text)

    # Try Ayat: (1), (2), (3), ...
    ayat_markers = _find_fuzzy_markers(text, "ayat", "1")
    if ayat_markers:
        intro, segments = _split_text_by_markers(text, ayat_markers)
        sub_nodes = []
        # Build one Ayat sub-node per segment, recursing into each for deeper structure.
        for i, (label, segment) in enumerate(segments, 1):
            if not segment.strip():
                continue  # Skip ayat with no content (PDF extraction gap)
            sub_id = f"{parent_id}_ayat_{label}"
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
        _stash_intro_for_parent(sub_nodes, intro)
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    # Try Angka vs Huruf — pick whichever appears EARLIEST in text.
    # Text like "a. ... b. ... h. ... 1. ... 9." should split at a/b/c...
    # (huruf first), letting Huruf h's body recurse into Angka 1-9.
    # Text like "1. Definisi A... 2. Definisi B..." with occasional inner
    # a/b should split at 1/2/... (angka first), letting each Angka's body
    # recurse into Huruf a/b.
    angka_markers = _find_fuzzy_markers(text, "angka", "1")
    huruf_markers = _find_fuzzy_markers(text, "huruf", "a")
    if angka_markers and huruf_markers:
        first_angka_pos = angka_markers[0][0]
        first_huruf_pos = huruf_markers[0][0]
        if first_huruf_pos < first_angka_pos:
            angka_markers = None  # Skip Angka branch; Huruf is the parent level.
    if angka_markers:
        intro, segments = _split_text_by_markers(text, angka_markers)
        sub_nodes = []
        # Build one Angka sub-node per segment, recursing into Huruf if present.
        for i, (label, segment) in enumerate(segments, 1):
            if not segment.strip():
                continue  # Skip angka with no content (PDF extraction gap)
            sub_id = f"{parent_id}_angka_{label}"
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
        _stash_intro_for_parent(sub_nodes, intro)
        for n in sub_nodes:
            n.pop("_split_label", None)
        return sub_nodes

    # Try Huruf: a., b., c., ...
    huruf_markers = _find_fuzzy_markers(text, "huruf", "a")
    if huruf_markers:
        intro, segments = _split_text_by_markers(text, huruf_markers)
        sub_nodes = []
        # Build one Huruf sub-node per segment, recursing into each for deeper structure.
        for i, (label, segment) in enumerate(segments, 1):
            if not segment.strip():
                continue  # Skip huruf with no content (PDF extraction gap)
            sub_id = f"{parent_id}_huruf_{label}"
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
        _stash_intro_for_parent(sub_nodes, intro)
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
            # Extract the pasal number for a readable node_id suffix.
            ref_m = re.match(r"Pasal\s+(\d+[A-Z]?)", pasal_ref)
            ref_num = ref_m.group(1) if ref_m else "x"
            pasal_id = f"{node['node_id']}_pasal_{ref_num}"
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
