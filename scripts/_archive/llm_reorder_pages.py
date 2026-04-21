"""LLM-based page text reordering for PDFs with problematic multi-column layouts.

For problematic docs where pymupdf extracts blocks out of reading order, this
script uses Gemini to reorder the raw page text into a coherent reading flow.
The reordered text is then fed through the existing parser pipeline (parse +
LLM cleanup + re-split) so downstream logic stays unchanged.

Workflow:
    1. Extract per-page block text via pymupdf (current behavior).
    2. For each page, ask Gemini to reorder the blocks into proper reading
       order (top-to-bottom within each column, left column before right).
    3. Save the reordered page text as JSON cache so re-runs are fast.
    4. Patch the parser.extract_pages call to read from this cache when
       available for the targeted doc_ids.

Usage:
    python scripts/llm_reorder_pages.py --doc-id pp-9-2026
    python scripts/llm_reorder_pages.py --doc-ids pp-9-2026,uu-20-2025
    python scripts/llm_reorder_pages.py --list   # scan + auto-list problematic docs
"""

import argparse
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REORDER_PROMPT = """\
You are an expert at reading Indonesian legal document PDFs. Given the raw text
blocks extracted from a SINGLE page of a PDF (in arbitrary order), reorder them
to produce the correct reading flow.

Rules:
1. Preserve ALL original text content exactly — do not rewrite or summarize.
2. Drop noise-only blocks: page numbers like "- 3 -", stamps like "SK No 248736 A",
   page footers repeating "PRESIDEN\\nREPUBLIK INDONESIA".
3. For multi-column layouts (common in Indonesian government PDFs), emit the
   left column top-to-bottom first, then the right column.
4. Keep structural headings (BAB, Pasal, Bagian, Paragraf) on their own lines.
5. Output ONLY the reordered text, nothing else. No preamble, no JSON wrapping.

Raw blocks from the page (each block tagged with its bounding box):
{blocks}
"""


def extract_page_blocks(pdf_path: str) -> list[list[dict]]:
    """Extract per-page raw blocks with bbox, preserving all text verbatim."""
    import pymupdf
    pages = []
    with pymupdf.open(pdf_path) as doc:
        for page in doc:
            blocks = []
            for b in page.get_text("dict").get("blocks", []):
                if b.get("type") != 0:
                    continue
                text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        text += span["text"]
                    text += "\n"
                if text.strip():
                    bbox = b["bbox"]
                    blocks.append({
                        "x0": round(bbox[0], 1),
                        "y0": round(bbox[1], 1),
                        "x1": round(bbox[2], 1),
                        "y1": round(bbox[3], 1),
                        "text": text,
                    })
            pages.append(blocks)
    return pages


def llm_reorder_page(blocks: list[dict], client, page_num: int) -> str:
    """Call Gemini to produce reading-order-correct text from raw blocks."""
    if len(blocks) < 2:
        return "".join(b["text"] for b in blocks)

    blocks_fmt = "\n\n".join(
        f"[x0={b['x0']:.0f} y0={b['y0']:.0f} x1={b['x1']:.0f} y1={b['y1']:.0f}]\n{b['text']}"
        for b in blocks
    )
    prompt = REORDER_PROMPT.format(blocks=blocks_fmt)

    import google.generativeai as genai
    resp = client.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=8192,
        ),
    )
    text = resp.text or ""
    if not text.strip():
        print(f"  WARN page {page_num}: empty LLM response, falling back to top-down sort", file=sys.stderr)
        return "".join(
            b["text"] for b in sorted(blocks, key=lambda x: (x["y0"], x["x0"]))
        )
    return text


def reorder_doc(doc_id: str, pdf_path: str, cache_dir: Path) -> Path:
    """Produce a reordered per-page text cache JSON for one document.

    Saves progress after every page so partial progress is preserved on crashes
    or interruptions. Skips pages already in cache for resumability.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel("gemini-2.5-flash")

    pages_blocks = extract_page_blocks(pdf_path)
    total = len(pages_blocks)
    print(f"{doc_id}: {total} pages to reorder", flush=True)

    cache_path = cache_dir / f"{doc_id}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume from existing partial cache if present.
    reordered_pages: list[dict] = []
    if cache_path.exists():
        try:
            reordered_pages = json.load(open(cache_path, encoding="utf-8"))
            print(f"  resuming from {len(reordered_pages)} cached pages", flush=True)
        except Exception:
            reordered_pages = []

    done_page_nums = {p["page_num"] for p in reordered_pages}

    for i, blocks in enumerate(pages_blocks, 1):
        if i in done_page_nums:
            continue
        print(f"  page {i}/{total} ({len(blocks)} blocks) ...", flush=True)
        try:
            text = llm_reorder_page(blocks, client, i)
        except Exception as e:
            print(f"  ERROR page {i}: {e.__class__.__name__}: {e}", flush=True)
            # Fallback: top-down sort
            text = "".join(
                b["text"] for b in sorted(blocks, key=lambda x: (x["y0"], x["x0"]))
            )
            print(f"  (used fallback sort for page {i})", flush=True)
        reordered_pages.append({"page_num": i, "raw_text": text})
        # Persist after each page.
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(reordered_pages, f, ensure_ascii=False, indent=2)

    print(f"  done: saved {len(reordered_pages)} pages to {cache_path}", flush=True)
    return cache_path


def find_pdf_path(doc_id: str) -> str | None:
    """Locate the main PDF for a doc_id by scanning data/raw/<jenis>/pdfs/."""
    registry = json.load(open("data/raw/registry.json", encoding="utf-8"))
    entry = registry.get(doc_id)
    if not entry:
        return None
    jenis = entry.get("jenis_folder", "")
    nomor = entry.get("nomor", "")
    tahun = entry.get("tahun", "")
    pdf_dir = Path(f"data/raw/{jenis}/pdfs")
    if not pdf_dir.exists():
        return None
    candidates = []
    for pdf in pdf_dir.glob("*.pdf"):
        if "Lampiran" in pdf.name:
            continue
        name_lower = pdf.name.lower()
        if nomor and tahun and (
            f"nomor {nomor} " in name_lower
            or f"no. {nomor} " in name_lower
            or f"no {nomor} " in name_lower
            or (nomor in pdf.name and tahun in pdf.name)
        ):
            candidates.append(pdf)
    if not candidates:
        return None
    return str(min(candidates, key=lambda p: len(p.name)))


def main() -> None:
    """CLI entry: reorder pages for one or more docs."""
    ap = argparse.ArgumentParser(description="LLM reorder PDF pages for problematic docs")
    ap.add_argument("--doc-id", type=str, help="Single doc_id to reorder")
    ap.add_argument("--doc-ids", type=str, help="Comma-separated doc_ids")
    ap.add_argument("--cache-dir", type=str, default="data/reordered_pages",
                    help="Output cache directory")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if args.doc_id:
        doc_ids = [args.doc_id]
    elif args.doc_ids:
        doc_ids = [d.strip() for d in args.doc_ids.split(",") if d.strip()]
    else:
        print("Provide --doc-id or --doc-ids", file=sys.stderr)
        sys.exit(1)

    for doc_id in doc_ids:
        pdf_path = find_pdf_path(doc_id)
        if not pdf_path:
            print(f"{doc_id}: PDF not found", file=sys.stderr)
            continue
        reorder_doc(doc_id, pdf_path, cache_dir)


if __name__ == "__main__":
    main()
