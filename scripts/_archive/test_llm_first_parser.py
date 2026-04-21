"""
test_llm_first_parser.py — Empirical comparison of LLM-first vs regex-first parsing.

For a single document, this script:
  1. Loads the regex-parsed structure from data/index_pasal/ (ground truth).
  2. Extracts raw PDF text using the same PyMuPDF pipeline.
  3. Sends the raw text to Gemini in page-chunks asking it to identify
     structural markers (Pasal, Angka, Bab, etc.) and correct OCR errors.
  4. Compares LLM-detected elements vs regex-detected elements.
  5. Reports: matches, mismatches, LLM-only, regex-only, title diffs, token cost.

This is a diagnostic/research tool — NOT a replacement parser.
Run it to empirically assess the tradeoff before committing to an architecture.

Usage:
    conda run -n skripsi python scripts/test_llm_first_parser.py --doc-id perpu-1-2022
    conda run -n skripsi python scripts/test_llm_first_parser.py --doc-id perpu-1-2022 --pages-per-chunk 8
"""
import argparse
import json
import os
import re
import sys
import time
import warnings
from pathlib import Path

# Add project root to path so we can import from vectorless.indexing
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vectorless.indexing.parser import extract_pages, clean_page_text
from vectorless.indexing.build import load_metadata, pick_main_pdf

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATA_INDEX_PASAL = Path("data/index_pasal")
DATA_RAW = Path("data/raw")
DEFAULT_PAGES_PER_CHUNK = 10

STRUCTURE_PROMPT = """\
You are parsing pages {start}–{end} of an Indonesian legal document.

Raw extracted text (may contain OCR artifacts):
---
{text}
---

Identify ALL structural markers found in this text. Return ONLY valid JSON — no explanation.

Format:
{{
  "elements": [
    {{"type": "bab",         "number": "I",   "title": "Ketentuan Umum"}},
    {{"type": "bagian",      "number": "Kesatu", "title": "..."}},
    {{"type": "pasal",       "number": "1"}},
    {{"type": "pasal_roman", "number": "I"}},
    {{"type": "angka",       "number": "9",   "title": "Di antara Pasal 568 dan Pasal 569 disisipkan 1 (satu) pasal, yakni Pasal 568A yang berbunyi sebagai berikut:"}},
    {{"type": "ayat",        "number": "1",   "parent": "3"}}
  ]
}}

Rules:
- Types: bab | bagian | pasal_roman | pasal | angka | ayat
- Include ONLY structural headings, not body text.
- Fix OCR errors in titles and numbers (e.g. "Pasal 5684" → "Pasal 568A", "Pasa1" → "Pasal").
- For "angka" elements, copy the full instruction text as "title".
- For "pasal_roman" (Roman numeral Pasal used in amendment docs, e.g. "Pasal I", "Pasal II"), use type "pasal_roman".
- If no structural markers found in these pages, return {{"elements": []}}.
"""


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _make_model():
    """Create Gemini model using same approach as parser.py."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable is not set")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def _call_llm(model, prompt: str) -> tuple[str, int, int]:
    """Call LLM and return (response_text, input_tokens, output_tokens)."""
    resp = model.generate_content(prompt)
    input_tok = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
    output_tok = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0
    return resp.text, input_tok, output_tok


def _parse_llm_response(text: str) -> list[dict]:
    """Extract JSON element list from LLM response, tolerating markdown fences."""
    # Strip markdown code fences if present
    text = re.sub(r'^```(?:json)?\s*', '', text.strip(), flags=re.MULTILINE)
    text = re.sub(r'\s*```$', '', text.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(text)
        return data.get("elements", [])
    except json.JSONDecodeError as e:
        print(f"  [WARN] LLM response not valid JSON: {e}")
        return []


# ---------------------------------------------------------------------------
# Ground truth extraction from regex-parsed index
# ---------------------------------------------------------------------------

def _find_pasal_json(doc_id: str) -> Path | None:
    """Find the pasal index JSON for a given doc_id across all categories."""
    for category_dir in DATA_INDEX_PASAL.iterdir():
        if not category_dir.is_dir():
            continue
        candidate = category_dir / f"{doc_id}.json"
        if candidate.exists():
            return candidate
    return None


def _find_pdf(doc_id: str) -> Path | None:
    """Find the PDF file for a given doc_id using the registry and metadata."""
    registry_path = DATA_RAW / "registry.json"
    if not registry_path.exists():
        return None
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    entry = registry.get(doc_id)
    if not entry:
        return None

    jenis_folder = entry.get("jenis_folder", "")
    detail_id = entry.get("detail_id", "")

    # Try metadata first (has exact filename list)
    metadata = load_metadata(doc_id, detail_id, jenis_folder)
    if metadata:
        filename = pick_main_pdf(metadata)
        if filename:
            pdf_path = DATA_RAW / jenis_folder / "pdfs" / filename
            if pdf_path.exists():
                return pdf_path

    # Fallback: construct filename from registry fields
    bentuk = entry.get("bentuk_singkat", "")
    nomor = entry.get("nomor", "")
    tahun = entry.get("tahun", "")
    fallback = DATA_RAW / jenis_folder / "pdfs" / f"{bentuk} Nomor {nomor} Tahun {tahun}.pdf"
    if fallback.exists():
        return fallback

    return None


def _extract_regex_elements(doc: dict) -> list[dict]:
    """Walk the parsed structure tree and extract (type, number, title) per node."""
    elements = []

    def _walk(nodes, depth=0):
        for node in nodes:
            title = node.get("title", "")
            nid = node.get("node_id", "")
            # Classify type from title pattern
            if re.match(r'^Pasal\s+[IVXLCivxlc]+$', title):
                elements.append({"type": "pasal_roman", "number": re.sub(r'^Pasal\s+', '', title), "title": title, "_nid": nid})
            elif re.match(r'^Pasal\s+', title):
                num = re.sub(r'^Pasal\s+', '', title)
                elements.append({"type": "pasal", "number": num, "title": title, "_nid": nid})
            elif re.match(r'^Angka\s+\d+', title):
                num = re.search(r'^Angka\s+(\d+)', title).group(1)
                elements.append({"type": "angka", "number": num, "title": title, "_nid": nid})
            elif re.match(r'^Bab\s+', title, re.IGNORECASE):
                num = re.sub(r'^(?i:Bab)\s+', '', title).split()[0] if title.split() else ""
                elements.append({"type": "bab", "number": num, "title": title, "_nid": nid})
            elif re.match(r'^Bagian\s+', title, re.IGNORECASE):
                elements.append({"type": "bagian", "number": "", "title": title, "_nid": nid})
            if "nodes" in node:
                _walk(node["nodes"], depth + 1)

    _walk(doc.get("structure", []))
    return elements


# ---------------------------------------------------------------------------
# Main comparison logic
# ---------------------------------------------------------------------------

def run_comparison(doc_id: str, pages_per_chunk: int = DEFAULT_PAGES_PER_CHUNK):
    """Run LLM-first parsing on doc_id and compare with existing regex parse."""

    # --- Load ground truth ---
    pasal_path = _find_pasal_json(doc_id)
    if not pasal_path:
        print(f"ERROR: No pasal index found for '{doc_id}' in {DATA_INDEX_PASAL}")
        sys.exit(1)
    doc = json.loads(pasal_path.read_text(encoding="utf-8"))
    regex_elements = _extract_regex_elements(doc)
    element_counts = doc.get("element_counts", {})

    print(f"\n{'='*60}")
    print(f"LLM-FIRST vs REGEX PARSER: {doc_id}")
    print(f"{'='*60}")
    print(f"Doc: {doc.get('judul', doc_id)[:80]}")
    print(f"Regex element_counts: {element_counts}")
    print(f"Regex structure nodes extracted: {len(regex_elements)}")

    # --- Find PDF ---
    pdf_path = _find_pdf(doc_id)
    if not pdf_path:
        print(f"ERROR: No PDF found for '{doc_id}' in {DATA_RAW}")
        sys.exit(1)
    print(f"PDF: {pdf_path.name}")

    # --- Extract raw pages ---
    print(f"\nExtracting raw text from PDF...")
    pages = extract_pages(str(pdf_path))
    print(f"Pages extracted: {len(pages)}")

    # --- LLM structure extraction ---
    print(f"\nRunning LLM structure extraction ({pages_per_chunk} pages/chunk)...")
    model = _make_model()
    llm_elements: list[dict] = []
    total_input_tokens = 0
    total_output_tokens = 0
    total_chunks = 0
    t_start = time.time()

    for chunk_start in range(0, len(pages), pages_per_chunk):
        chunk_pages = pages[chunk_start:chunk_start + pages_per_chunk]
        start_page = chunk_pages[0]["page_num"]
        end_page = chunk_pages[-1]["page_num"]

        # Concatenate and lightly clean page texts for this chunk
        chunk_text = "\n\n[PAGE BREAK]\n\n".join(
            clean_page_text(p["raw_text"]) for p in chunk_pages
        )

        prompt = STRUCTURE_PROMPT.format(
            start=start_page,
            end=end_page,
            text=chunk_text[:12000],  # cap per chunk to avoid context overflow
        )

        print(f"  Chunk pages {start_page}–{end_page}...", end=" ", flush=True)
        try:
            response_text, in_tok, out_tok = _call_llm(model, prompt)
            chunk_elements = _parse_llm_response(response_text)
            llm_elements.extend(chunk_elements)
            total_input_tokens += in_tok
            total_output_tokens += out_tok
            total_chunks += 1
            print(f"{len(chunk_elements)} elements, {in_tok}+{out_tok} tokens")
        except Exception as e:
            print(f"FAILED: {e}")

    elapsed = time.time() - t_start
    print(f"\nLLM extraction done: {total_chunks} chunks, {elapsed:.1f}s")
    print(f"LLM elements found: {len(llm_elements)}")

    # --- Compare ---
    print(f"\n{'─'*60}")
    print("COMPARISON")
    print(f"{'─'*60}")

    # Build lookup dicts by (type, number)
    def _key(e):
        return (e["type"], str(e.get("number", "")).strip())

    regex_by_key = {_key(e): e for e in regex_elements}
    llm_by_key = {_key(e): e for e in llm_elements}

    both_keys = set(regex_by_key) & set(llm_by_key)
    regex_only = set(regex_by_key) - set(llm_by_key)
    llm_only = set(llm_by_key) - set(regex_by_key)

    print(f"\nMatched (both detected):   {len(both_keys)}")
    print(f"Regex-only (LLM missed):   {len(regex_only)}")
    print(f"LLM-only (regex missed):   {len(llm_only)}")

    # Title differences among matched
    title_diffs = []
    for key in sorted(both_keys):
        rt = regex_by_key[key].get("title", "")
        lt = llm_by_key[key].get("title", "")
        if rt and lt and rt.strip() != lt.strip():
            title_diffs.append((key, rt, lt))

    if title_diffs:
        print(f"\nTitle differences ({len(title_diffs)}):")
        for (etype, enum), regex_title, llm_title in title_diffs:
            print(f"  {etype} {enum}:")
            print(f"    REGEX: {regex_title[:100]}")
            print(f"    LLM:   {llm_title[:100]}")

    if regex_only:
        print(f"\nRegex detected, LLM missed ({len(regex_only)}):")
        for key in sorted(regex_only):
            e = regex_by_key[key]
            print(f"  {key[0]} {key[1]}: {e.get('title', '')[:60]}")

    if llm_only:
        print(f"\nLLM detected, regex missed ({len(llm_only)}):")
        for key in sorted(llm_only):
            e = llm_by_key[key]
            print(f"  {key[0]} {key[1]}: {e.get('title', '')[:60]}")

    # --- Token cost comparison ---
    current_llm_tokens = (
        doc.get("llm_input_tokens", 0) + doc.get("llm_output_tokens", 0)
    )
    # Fall back to cost manifest if not in doc
    if current_llm_tokens == 0:
        cost_path = Path("data/indexing_logs/cost_pasal.json")
        if cost_path.exists():
            cost_data = json.loads(cost_path.read_text(encoding="utf-8"))
            entry = cost_data.get(doc_id, {})
            current_llm_tokens = entry.get("llm_total_tokens", 0)

    llm_first_tokens = total_input_tokens + total_output_tokens

    print(f"\n{'─'*60}")
    print("TOKEN COST COMPARISON")
    print(f"{'─'*60}")
    print(f"  LLM-first (structure extraction):  {llm_first_tokens:>8,} tokens  ({elapsed:.1f}s)")
    if current_llm_tokens > 0:
        delta_pct = (llm_first_tokens - current_llm_tokens) / current_llm_tokens * 100
        print(f"  Current cleanup (leaf text only):  {current_llm_tokens:>8,} tokens")
        print(f"  Delta: {delta_pct:+.1f}%")
    else:
        print(f"  (No current cleanup cost data found for comparison)")

    print(f"\n{'='*60}")
    print("VERDICT")
    print(f"{'='*60}")
    print(f"  LLM-first fixes {len(title_diffs)} title OCR issue(s)")
    print(f"  LLM misses {len(regex_only)} element(s) that regex caught")
    print(f"  LLM finds {len(llm_only)} extra element(s) (may be correct or hallucinated)")
    if current_llm_tokens > 0:
        print(f"  Token overhead vs current approach: {delta_pct:+.1f}%")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare LLM-first vs regex-first parsing for a single document."
    )
    parser.add_argument("--doc-id", required=True, help="Document ID, e.g. perpu-1-2022")
    parser.add_argument(
        "--pages-per-chunk",
        type=int,
        default=DEFAULT_PAGES_PER_CHUNK,
        help=f"Pages per LLM chunk (default: {DEFAULT_PAGES_PER_CHUNK})",
    )
    args = parser.parse_args()
    run_comparison(args.doc_id, args.pages_per_chunk)
