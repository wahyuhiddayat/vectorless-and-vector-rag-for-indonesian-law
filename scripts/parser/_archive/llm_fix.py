"""Grounded LLM fix for non-OK parsed documents.

For each target doc, feed Gemini the current parser output + raw PDF text +
validator-flagged issues, and ask it to return a corrected structure tree.
LLM preserves correct parts and patches only flagged problems.

Workflow:
    1. Load data/parser_quality_report.json to determine targets
    2. For each doc: load parser JSON + raw PDF text + issues
    3. Build grounded prompt, call Gemini 2.5 Flash (temperature=0)
    4. Validate LLM output: JSON schema + sanity checks
    5. Backup original, save corrected, log audit trail

Safety:
    - data/index_pasal_pre_llm_fix/ holds backup of every doc before first edit
    - data/llm_fix_log.json records before/after scores per doc
    - --dry-run writes nothing, prints preview
    - Validation gate rejects output with fewer pasals than PDF (hallucination guard)

Usage:
    python scripts/parser/llm_fix.py --doc-id perpres-119-2025 --dry-run
    python scripts/parser/llm_fix.py --doc-id perpres-119-2025        # commit
    python scripts/parser/llm_fix.py --status FAIL                    # all FAIL docs
    python scripts/parser/llm_fix.py                                  # all non-OK docs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from vectorless.indexing.parser import extract_pages  # noqa: E402
from scripts._shared import find_pdf_path  # noqa: E402

INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"
BACKUP_DIR = REPO_ROOT / "data" / "index_pasal_pre_llm_fix"
QUALITY_REPORT = REPO_ROOT / "data" / "parser_quality_report.json"
AUDIT_LOG = REPO_ROOT / "data" / "llm_fix_log.json"
COST_MANIFEST = REPO_ROOT / "data" / "indexing_logs" / "cost_pasal.json"

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
MAX_INPUT_TOKENS_HINT = 200_000

# Docs with more pasals than this use BAB-chunked processing so each LLM
# response stays well below Gemini's ~65K output token ceiling.
CHUNK_PASAL_THRESHOLD = 30
# Fallback chunk size when a doc has no BAB structure (pure pasal list).
PASALS_PER_CHUNK = 15
# Max concurrent LLM calls per doc in chunked mode (Gemini 2.5 Flash paid tier
# allows 1000 RPM; 4 concurrent gives ~3x speedup without rate-limit risk).
CHUNK_PARALLEL_WORKERS = 4
# Buffer pages to include on each side of a chunk's page range (handles body
# content that spills into adjacent pages).
CHUNK_PAGE_BUFFER = 1
# For amendment docs: split Pasal Roman nodes with many Angka children into
# sub-chunks of this many Angka each, to keep LLM output within token budget.
ANGKA_PER_CHUNK = 8
_PASAL_ROMAN_TITLE_RE = re.compile(r"^Pasal\s+[IVX]+$")


PROMPT_TEMPLATE = """\
You are fixing the structural parse of an Indonesian legal document.

The regex-based parser produced an imperfect structure. Your job is to
return a CORRECTED structure, using the raw PDF text as ground truth.

=== DOCUMENT METADATA ===
doc_id       : {doc_id}
judul        : {judul}
is_perubahan : {is_perubahan}

=== CURRENT PARSER OUTPUT (tree view) ===
{parser_view}

=== KNOWN ISSUES (from automated validator) ===
{issues_list}

=== RAW PDF BLOCKS (ground truth, with spatial coordinates) ===
Each line is a text block extracted from the PDF, tagged with its position:
  [x=<column-left-edge> y=<top-edge>] <text content>

SPATIAL INTERPRETATION:
- Blocks with similar x0 belong to the same column. Indonesian gazette PDFs
  are often TWO-COLUMN (narrow x range ~50-280 = left; ~300-550 = right).
- Reading order: finish LEFT column top-to-bottom, THEN RIGHT column.
- Body content is NOT always adjacent to its heading in the block list:
  some PDFs cluster "Pasal 7\nPasal 8\nPasal 9" together as one block,
  and the actual body lives in another block further down. USE x,y coords
  to reconstruct the true reading flow.

{pdf_text}

=== INSTRUCTIONS ===
1. Preserve the structure tree as much as possible. Keep nodes that are already correct.
2. Fix ONLY the flagged issues:
   - empty_pasal     : attach the real body text from PDF to these Pasal nodes.
                       Use spatial coords to find which body block belongs to each
                       empty pasal — do NOT just take the next body block in list order.
   - bleed           : split content that leaked between adjacent Pasal nodes.
   - gaps            : add missing Pasal nodes that exist in the PDF body text.
   - underspilt_ayat : split a leaf into child Ayat nodes when (1), (2), (3) markers present.
   - underspilt_huruf: split a leaf into child Huruf nodes when a., b., c. markers present.
   - underspilt_angka: split a leaf into child Angka nodes when 1., 2., 3. numbered list present.
   - ordering        : reorder nodes to match sequential numbering.
3. NEVER invent content that is not in the PDF blocks.
4. NEVER change text of nodes that are already correct.
5. Preserve Preamble nodes (Pembukaan / Menimbang / Mengingat / Menetapkan) verbatim.
6. For amendment docs (is_perubahan=true), preserve the "Pasal I > Angka N > ..." structure.
7. Body text should be exact quotes from the PDF blocks (join multi-line spans cleanly).
8. Drop footer noise: "SK No ...", page numbers "- N -", repeated "PRESIDEN REPUBLIK INDONESIA".
9. Output ONLY a JSON object, no markdown fences, no prose.

=== OUTPUT SCHEMA ===

Every node has keys: "title" (required), and optionally "text" (intro/body) and/or "nodes" (children).

IMPORTANT — CONTENT ATTRIBUTION RULE:

When a Pasal (or Ayat, or Huruf) contains an intro sentence FOLLOWED by a list of
items (a., b., c.  OR  (1), (2), (3)  OR  1., 2., 3.), you MUST split:
- The parent node has "text" = INTRO ONLY (without any list items).
- Each list item goes in its own child node with "text" = ITEM BODY ONLY
  (without the "a." / "b." / "1." prefix and without the parent's intro).

EXAMPLE OF CORRECT SPLITTING:

PDF raw text:
    Pasal 6
    Fasilitas lainnya bagi X diberikan dalam bentuk:
    a. biaya perjalanan dinas; dan
    b. jaminan sosial.

Correct JSON:
    {{ "title": "Pasal 6",
       "text": "Fasilitas lainnya bagi X diberikan dalam bentuk:",
       "nodes": [
         {{ "title": "Pasal 6 Huruf a", "text": "biaya perjalanan dinas; dan" }},
         {{ "title": "Pasal 6 Huruf b", "text": "jaminan sosial." }}
       ]
    }}

WRONG (don't do this — the intro bled into the first child):
    {{ "title": "Pasal 6",
       "nodes": [
         {{ "title": "Pasal 6 Huruf a",
            "text": "Fasilitas lainnya bagi X diberikan dalam bentuk: a. biaya perjalanan dinas; dan" }},
         ...
       ]
    }}

WRONG (don't do this — list prefix kept in child text):
    {{ "title": "Pasal 6 Huruf a", "text": "a. biaya perjalanan dinas; dan" }}

The same rule applies RECURSIVELY at every depth (Ayat intro + Huruf children,
Huruf intro + Angka children, etc.).

Node shapes allowed:
1. Pure leaf:     {{ "title": "...", "text": "..." }}
2. Pure container:{{ "title": "...", "nodes": [...] }}
3. Intro+container:{{ "title": "...", "text": "intro only", "nodes": [...] }}

Title format: "Pasal 1" / "Pasal 2 Ayat (1)" / "Pasal 2 Ayat (1) Huruf a" /
"Pasal 3 Huruf a" / "Pasal 1 Angka 2".
BAB: "BAB I - TITLE". Bagian/Paragraf: "Bagian Kesatu - TITLE".

Return a JSON object with a single top-level key "structure" whose value is a list of nodes.

Return the corrected JSON now:
"""


def compact_tree_view(structure: list[dict], indent: int = 0) -> str:
    """Render the parser output tree as indented text for LLM context."""
    lines = []
    for node in structure:
        title = node.get("title", "?")
        prefix = "  " * indent
        if "nodes" in node and node["nodes"]:
            lines.append(f"{prefix}{title}:")
            lines.append(compact_tree_view(node["nodes"], indent + 1))
        else:
            text = (node.get("text") or "").strip().replace("\n", " ")
            snippet = text[:160] + ("..." if len(text) > 160 else "")
            lines.append(f'{prefix}{title} => "{snippet}"')
    return "\n".join(l for l in lines if l)


def build_issues_list(report_entry: dict) -> str:
    """Turn validator report entry into a concise bullet list for the prompt."""
    bullets = []
    if report_entry.get("empty_pasal_count", 0) > 0:
        nodes = report_entry.get("empty_pasal_nodes", [])
        bullets.append(f"- empty_pasal ({len(nodes)}x): {', '.join(nodes[:10])}")
    if report_entry.get("bleed_count", 0) > 0:
        nodes = report_entry.get("bleed_nodes", [])
        bullets.append(f"- bleed ({len(nodes)}x): {', '.join(nodes[:10])}")
    if report_entry.get("gap_count", 0) > 0:
        bullets.append(f"- gaps: missing Pasal {report_entry.get('gap_list', [])[:10]}")
    if not report_entry.get("monotonic", True):
        bullets.append("- ordering: Pasal numbers appear out of order")
    if report_entry.get("underspilt_count", 0) > 0:
        samples = report_entry.get("underspilt_samples", [])
        bullets.append(f"- underspilt ({report_entry['underspilt_count']}x): {', '.join(samples[:10])}")
    return "\n".join(bullets) if bullets else "- (no structural issues flagged — minor issues only)"


def load_pdf_text(doc_id: str) -> str:
    """Read the document's main PDF and concatenate all pages (flat text)."""
    pdf_path = find_pdf_path(doc_id)
    if not pdf_path:
        raise FileNotFoundError(f"PDF not found for {doc_id}")
    pages = extract_pages(str(pdf_path))
    return "\n\n".join(p.get("raw_text", "") for p in pages)


def load_pdf_pages(doc_id: str) -> list[dict]:
    """Read the PDF and return list of {page_num, blocks} per page.

    Each block: {x0, y0, text}. Blocks sorted (y, x) for reading order.
    Use format_pdf_pages() to render a page range as prompt text.
    """
    import pymupdf

    pdf_path = find_pdf_path(doc_id)
    if not pdf_path:
        raise FileNotFoundError(f"PDF not found for {doc_id}")

    pages: list[dict] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page_i, page in enumerate(doc, 1):
            blocks = []
            for b in page.get_text("dict").get("blocks", []):
                if b.get("type") != 0:
                    continue
                text = ""
                for line in b.get("lines", []):
                    for span in line.get("spans", []):
                        text += span["text"]
                    text += "\n"
                text = text.strip()
                if not text:
                    continue
                bbox = b["bbox"]
                blocks.append({
                    "x0": round(bbox[0]),
                    "y0": round(bbox[1]),
                    "text": text,
                })
            blocks.sort(key=lambda b: (b["y0"], b["x0"]))
            pages.append({"page_num": page_i, "blocks": blocks})
    return pages


def format_pdf_pages(
    pages: list[dict],
    start_page: int | None = None,
    end_page: int | None = None,
) -> str:
    """Render pages as '=== PAGE N ===' sections with [x=y=] tagged blocks.

    If start_page/end_page given, only include pages in that inclusive range
    (1-indexed). Used to scope PDF context per chunk for efficiency.
    """
    out: list[str] = []
    for p in pages:
        n = p["page_num"]
        if start_page is not None and n < start_page:
            continue
        if end_page is not None and n > end_page:
            continue
        out.append(f"=== PAGE {n} ===")
        for b in p["blocks"]:
            out.append(f"[x={b['x0']} y={b['y0']}] {b['text']}")
        out.append("")
    return "\n".join(out)


def load_pdf_blocks(doc_id: str) -> str:
    """Legacy: return all pages formatted as a single string (full doc scope)."""
    return format_pdf_pages(load_pdf_pages(doc_id))


def call_gemini(prompt: str, max_output_tokens: int = 65536) -> tuple[str, dict]:
    """Call Gemini 2.5 Flash; return (raw_text, usage_dict).

    Uses max_output_tokens near Gemini 2.5 Flash ceiling (65536) and disables
    thinking mode so full budget goes to actual text output (prevents JSON
    truncation on large legal docs).

    usage_dict keys: input_tokens, output_tokens, total_tokens, calls, elapsed_s.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    # Try to disable thinking via ThinkingConfig; fall back if SDK lacks support.
    thinking_kwarg: dict = {}
    try:
        tc = genai.types.ThinkingConfig(thinking_budget=0)  # type: ignore[attr-defined]
        thinking_kwarg = {"thinking_config": tc}
    except Exception:
        thinking_kwarg = {}

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0, "elapsed_s": 0.0}
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                    **thinking_kwarg,
                ),
            )
            usage["calls"] += 1
            meta = getattr(resp, "usage_metadata", None)
            if meta is not None:
                usage["input_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
                usage["output_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
                usage["total_tokens"] += getattr(meta, "total_token_count", 0) or 0
            text = resp.text or ""
            if text.strip():
                usage["elapsed_s"] = round(time.time() - t0, 3)
                return text, usage
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  retry {attempt}/{MAX_RETRIES}: {exc.__class__.__name__}: {exc}", flush=True)
    raise RuntimeError("LLM returned empty response after retries")


def parse_llm_json(raw: str) -> dict:
    """Strip markdown fences and parse as JSON, normalizing key aliases."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    obj = json.loads(cleaned)
    _normalize_keys(obj)
    return obj


def _normalize_keys(obj) -> None:
    """Rename 'name' -> 'title' recursively (LLM sometimes uses 'name')."""
    if isinstance(obj, dict):
        if "name" in obj and "title" not in obj:
            obj["title"] = obj.pop("name")
        for v in obj.values():
            _normalize_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            _normalize_keys(item)


def split_preamble_and_body(structure: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate top-level preamble nodes (Pembukaan) from body nodes."""
    preamble, body = [], []
    for node in structure:
        title = node.get("title", "")
        if title.startswith("Pembukaan") or title in (
            "Menimbang",
            "Mengingat",
            "Menetapkan",
        ):
            preamble.append(node)
        else:
            body.append(node)
    return preamble, body


def chunk_body_by_bab(body: list[dict]) -> list[list[dict]]:
    """Split body into chunks. One chunk per BAB, or groups of PASALS_PER_CHUNK
    pasals when the doc has no BAB structure."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    has_bab = any(n.get("title", "").startswith("BAB ") for n in body)
    for node in body:
        title = node.get("title", "")
        if has_bab:
            if title.startswith("BAB "):
                if current:
                    chunks.append(current)
                    current = []
                chunks.append([node])
            else:
                # Non-BAB top-level node (e.g. Pasal I amendment root): own chunk.
                if current:
                    chunks.append(current)
                    current = []
                chunks.append([node])
        else:
            current.append(node)
            if count_pasals_in_tree(current) >= PASALS_PER_CHUNK:
                chunks.append(current)
                current = []
    if current:
        chunks.append(current)
    return chunks


def chunk_body_for_amendment(
    body: list[dict],
) -> tuple[list[list[dict]], list[dict]]:
    """For amendment docs: split big Pasal Roman nodes by Angka group.

    Returns (chunks, meta). For each Pasal Roman with > ANGKA_PER_CHUNK Angka
    children, emits multiple chunks each wrapping a slice of the children.
    Non-splittable nodes become their own chunk.

    meta[i] is {"kind": "angka_group"|"whole", "parent_title": str|None}. The
    stitcher uses parent_title to merge consecutive angka_group chunks back
    under the same Pasal Roman wrapper.
    """
    chunks: list[list[dict]] = []
    meta: list[dict] = []
    for node in body:
        title = node.get("title", "")
        children = node.get("nodes", []) or []
        is_pasal_roman = bool(_PASAL_ROMAN_TITLE_RE.match(title))
        if is_pasal_roman and len(children) > ANGKA_PER_CHUNK:
            # Parse all Angka numbers present under this wrapper to detect gaps.
            present_nums: list[int] = []
            for c in children:
                m = re.search(r"Angka\s+(\d+)", c.get("title", ""))
                if m:
                    present_nums.append(int(m.group(1)))
            global_min = min(present_nums) if present_nums else 1
            global_max = max(present_nums) if present_nums else 0
            # Expected range starts at 1 (amendment Angka always starts at 1),
            # ends at max observed. Gaps anywhere in [1..global_max] are missing.
            for i in range(0, len(children), ANGKA_PER_CHUNK):
                subset = children[i : i + ANGKA_PER_CHUNK]
                wrapper = {
                    **{k: v for k, v in node.items() if k != "nodes"},
                    "nodes": subset,
                }
                # Per-chunk expected range: widen chunk 1 to include Angka 1..
                subset_nums = []
                for c in subset:
                    m = re.search(r"Angka\s+(\d+)", c.get("title", ""))
                    if m:
                        subset_nums.append(int(m.group(1)))
                if not subset_nums:
                    lo, hi = 0, 0
                else:
                    lo = 1 if i == 0 else min(subset_nums)
                    hi = max(subset_nums)
                missing = sorted(set(range(lo, hi + 1)) - set(subset_nums)) if hi else []
                chunks.append([wrapper])
                meta.append({
                    "kind": "angka_group",
                    "parent_title": title,
                    "angka_range": (lo, hi),
                    "angka_missing": missing,
                    "angka_global_max": global_max,
                })
        else:
            chunks.append([node])
            meta.append({"kind": "whole", "parent_title": None})
    return chunks, meta


def stitch_amendment_chunks(
    fixed_chunks: list[list[dict]], meta: list[dict]
) -> list[dict]:
    """Merge consecutive angka_group chunks sharing parent_title into one node."""
    out: list[dict] = []
    for chunk, m in zip(fixed_chunks, meta):
        if not chunk:
            continue
        if m["kind"] == "angka_group":
            wrapper = chunk[0]
            parent_title = m["parent_title"]
            if out and out[-1].get("title") == parent_title:
                out[-1].setdefault("nodes", []).extend(
                    wrapper.get("nodes", []) or []
                )
            else:
                out.append({**wrapper, "nodes": list(wrapper.get("nodes", []) or [])})
        else:
            out.extend(chunk)
    return out


def filter_issues_for_chunk(
    chunk_nodes: list[dict], all_issues: dict
) -> dict:
    """Return issues entry scoped to node_ids present in this chunk only."""
    node_ids: set[str] = set()

    def collect(nodes):
        for n in nodes:
            nid = n.get("node_id")
            if nid:
                node_ids.add(nid)
            if "nodes" in n:
                collect(n["nodes"])

    collect(chunk_nodes)
    if not node_ids:
        return {}
    out = dict(all_issues)
    # Filter node-specific lists to only entries touching this chunk.
    out["empty_pasal_nodes"] = [
        n for n in all_issues.get("empty_pasal_nodes", [])
        if _issue_node_id(n) in node_ids
    ]
    out["bleed_nodes"] = [
        n for n in all_issues.get("bleed_nodes", [])
        if _issue_node_id(n) in node_ids
    ]
    out["underspilt_samples"] = [
        s for s in all_issues.get("underspilt_samples", [])
        if _issue_node_id(s) in node_ids
    ]
    out["empty_pasal_count"] = len(out["empty_pasal_nodes"])
    out["bleed_count"] = len(out["bleed_nodes"])
    out["underspilt_count"] = len(out["underspilt_samples"])
    # Gaps / monotonic are per-doc, not per-chunk — leave as-is for context.
    return out


def _issue_node_id(entry: str) -> str:
    """Extract node_id prefix from issue sample string like '0012:Pasal 3' or '0012_a2_h1:unsplit_huruf'."""
    return entry.split(":", 1)[0] if isinstance(entry, str) else ""


_PASAL_TITLE_RE = re.compile(r"^Pasal\s+\d+[A-Z]?$")


def count_pasals_in_tree(structure: list[dict]) -> int:
    """Count nodes whose title is exactly a Pasal heading (e.g. 'Pasal 12', 'Pasal 5A').

    Uses a strict regex rather than substring matching to avoid false positives
    from titles like 'Pasal 1 1.' that can appear after title normalization of
    LLM output where children lacked proper Angka/Huruf markers.
    """
    count = 0
    for node in structure:
        title = (node.get("title") or "").strip()
        if _PASAL_TITLE_RE.match(title):
            count += 1
        if "nodes" in node:
            count += count_pasals_in_tree(node["nodes"])
    return count


def assign_node_ids(structure: list[dict]) -> None:
    """Assign node_id in-place. Preamble: P000+, body: 0000+."""
    body_counter = [0]
    preamble_counter = [0]
    for node in structure:
        title = node.get("title", "")
        is_preamble = (
            title in ("Pembukaan", "Menimbang", "Mengingat", "Menetapkan")
            or title.startswith("Pembukaan")
        )
        if is_preamble:
            node["node_id"] = f"P{preamble_counter[0]:03d}"
            preamble_counter[0] += 1
        else:
            node["node_id"] = f"{body_counter[0]:04d}"
            body_counter[0] += 1
        if "nodes" in node:
            _assign_child_ids(node)


def _assign_child_ids(parent: dict) -> None:
    """Recursively assign suffixed node_ids based on the FIRST marker in title.

    Amendment docs often have verbose titles like
    "Pasal I Angka 2 — Di antara Pasal 3 dan Pasal 4 disisipkan..."
    where the structural marker ("Angka 2") appears at the start, not the end.
    So we match the first Angka/Huruf/Ayat token in the title, preferring the
    most specific (innermost) one by type priority: Angka > Huruf > Ayat.

    Title format examples and resulting suffix:
        "Pasal 3 Ayat (1)"                             -> a1
        "Pasal 3 Huruf a"                              -> h1
        "Pasal I Angka 2 — Di antara Pasal 3..."       -> n2 (amendment)
        "Pasal 3 Ayat (2) Huruf a"                     -> h1 (deepest wins)
        "Pasal 3 Ayat (2) Huruf c Angka 4"             -> n4
    """
    parent_id = parent.get("node_id", "")
    for i, child in enumerate(parent.get("nodes", []), 1):
        title = child.get("title", "")
        # Find last occurrence of each marker type (to pick deepest for
        # clean short titles like "Pasal 3 Ayat (2) Huruf a").
        all_angka = list(re.finditer(r"Angka\s+(\d+)", title, re.IGNORECASE))
        all_huruf = list(re.finditer(r"Huruf\s+([a-z])", title, re.IGNORECASE))
        all_ayat = list(re.finditer(r"Ayat\s+\((\d+)\)", title, re.IGNORECASE))
        # Deepest means the marker type with highest specificity PRESENT in title.
        # But for verbose amendment titles like "Pasal I Angka 2 — ...",
        # we want Angka 2 (not Pasal).
        # Priority: Angka > Huruf > Ayat.
        suffix = None
        if all_angka:
            suffix = f"n{all_angka[0].group(1)}"
        elif all_huruf:
            letter = all_huruf[0].group(1).lower()
            suffix = f"h{ord(letter) - ord('a') + 1}"
        elif all_ayat:
            suffix = f"a{all_ayat[0].group(1)}"
        else:
            suffix = f"x{i}"
        # For normal titles with multiple markers, use the LAST one (deepest).
        # This matters when title is clean e.g. "Pasal 3 Ayat (2) Huruf a" —
        # we want h1 not a2. But for amendment "Pasal I Angka 2 — Pasal 3A..."
        # the first Angka is correct. Heuristic: if title ends with a marker,
        # trust the last marker; otherwise trust the first.
        m_end_angka = re.search(r"Angka\s+(\d+)\s*$", title, re.IGNORECASE)
        m_end_huruf = re.search(r"Huruf\s+([a-z])\s*$", title, re.IGNORECASE)
        m_end_ayat = re.search(r"Ayat\s+\((\d+)\)\s*$", title, re.IGNORECASE)
        if m_end_angka:
            suffix = f"n{m_end_angka.group(1)}"
        elif m_end_huruf:
            letter = m_end_huruf.group(1).lower()
            suffix = f"h{ord(letter) - ord('a') + 1}"
        elif m_end_ayat:
            suffix = f"a{m_end_ayat.group(1)}"
        child["node_id"] = f"{parent_id}_{suffix}"
        if "nodes" in child:
            _assign_child_ids(child)


def build_navigation_paths(structure: list[dict], ancestors: list[str] | None = None) -> None:
    """Populate navigation_path on every node in-place."""
    if ancestors is None:
        ancestors = []
    for node in structure:
        title = node.get("title", "?")
        node["navigation_path"] = " > ".join(ancestors + [title])
        if "nodes" in node:
            build_navigation_paths(node["nodes"], ancestors + [title])


def ensure_page_indices(
    structure: list[dict],
    original_structure: list[dict],
    doc_total_pages: int,
) -> None:
    """Populate start_index / end_index on every node in-place.

    LLM output lacks page indices. We derive them by matching node titles to
    the original parser output when possible; otherwise inherit from parent
    or fall back to the full doc page range.
    """
    # Build title -> (start, end) map from original parser structure.
    title_map: dict[str, tuple[int, int]] = {}

    def _collect(nodes: list[dict]):
        for n in nodes:
            t = n.get("title", "")
            s = n.get("start_index")
            e = n.get("end_index")
            if t and s is not None and e is not None and t not in title_map:
                title_map[t] = (int(s), int(e))
            if "nodes" in n:
                _collect(n["nodes"])

    _collect(original_structure)

    def _assign(nodes: list[dict], parent_range: tuple[int, int]) -> None:
        for n in nodes:
            t = n.get("title", "")
            if t in title_map:
                start, end = title_map[t]
            else:
                start, end = parent_range
            n["start_index"] = start
            n["end_index"] = end
            if "nodes" in n:
                _assign(n["nodes"], (start, end))

    _assign(structure, (1, doc_total_pages if doc_total_pages > 0 else 1))


def _expand_shortcut_title(title: str) -> str:
    """Expand shortcut titles and normalize marker casing.

    Examples:
        "1."            -> "Angka 1"
        "12."           -> "Angka 12"
        "a."            -> "Huruf a"
        "(1)"           -> "Ayat (1)"
        "Pasal 2 ayat (1)" -> "Pasal 2 Ayat (1)"      (casing fix)
        "Pasal 3 huruf a"  -> "Pasal 3 Huruf a"
        "Ayat (1)"      -> unchanged
    """
    t = title.strip()
    m_angka = re.match(r"^(\d+)\.?\s*$", t)
    if m_angka:
        return f"Angka {m_angka.group(1)}"
    m_huruf = re.match(r"^([a-z])\.?\s*$", t)
    if m_huruf:
        return f"Huruf {m_huruf.group(1)}"
    m_ayat = re.match(r"^\((\d+)\)\s*$", t)
    if m_ayat:
        return f"Ayat ({m_ayat.group(1)})"
    # Normalize casing for marker keywords embedded in longer titles.
    t = re.sub(r"\bayat\b", "Ayat", t)
    t = re.sub(r"\bhuruf\b", "Huruf", t)
    t = re.sub(r"\bangka\b", "Angka", t)
    return t


def normalize_titles(structure: list[dict], ancestor_titles: list[str] | None = None) -> None:
    """Expand shortcut child titles and prepend Pasal/Ayat prefix chain.

    LLM often emits short titles like "Ayat (1)" / "Huruf a" / "1." / "(2)"
    for child nodes when the parent is obvious. Existing parser/validator
    expects full titles like "Pasal 7 Ayat (1)" / "Pasal 7 Ayat (1) Huruf a".

    Steps per node:
    1. Expand shortcut forms (1. -> Angka 1, a. -> Huruf a, (1) -> Ayat (1)).
    2. If title is not a root-level structural heading and a Pasal ancestor
       exists, prepend the pasal-rooted ancestor chain.
    """
    if ancestor_titles is None:
        ancestor_titles = []
    for node in structure:
        raw_title = node.get("title", "")
        expanded = _expand_shortcut_title(raw_title)
        node["title"] = expanded

        is_root_level = (
            expanded.startswith(("Pasal ", "BAB ", "Bagian ", "Paragraf "))
            or expanded in ("Pembukaan", "Menimbang", "Mengingat", "Menetapkan", "Penutup")
        )
        pasal_chain: list[str] = []
        for anc in ancestor_titles:
            if anc.startswith("Pasal "):
                pasal_chain = [anc]
            elif pasal_chain:
                pasal_chain.append(anc)
        if not is_root_level and pasal_chain and expanded:
            node["title"] = " ".join(pasal_chain + [expanded])
        if "nodes" in node:
            normalize_titles(node["nodes"], ancestor_titles + [node["title"]])


def validate_fix(before: dict, after: dict, pdf_pasal_count: int) -> tuple[bool, list[str]]:
    """Sanity-check LLM output."""
    errors: list[str] = []
    if not isinstance(after, dict) or "structure" not in after:
        errors.append("missing 'structure' key")
        return False, errors
    new_structure = after["structure"]
    if not isinstance(new_structure, list):
        errors.append("'structure' must be a list")
        return False, errors

    before_count = count_pasals_in_tree(before.get("structure", []))
    after_count = count_pasals_in_tree(new_structure)

    # Upper bound: the PDF's own regex pasal count is the hard ceiling.
    # When parser under-counted (before << pdf_pasal_count), allow LLM to
    # recover up to pdf_pasal_count * 1.1 even if that exceeds before * 1.3.
    upper = max(before_count * 1.3, before_count + 5)
    if pdf_pasal_count > 0:
        upper = max(upper, pdf_pasal_count * 1.1)
    if after_count > upper:
        errors.append(
            f"too many pasal added: before={before_count} after={after_count} "
            f"upper_bound={upper:.0f} (pdf_count={pdf_pasal_count})"
        )
    if after_count < before_count * 0.7:
        errors.append(
            f"too many pasal dropped: before={before_count} after={after_count}"
        )
    if pdf_pasal_count > 0 and after_count > pdf_pasal_count * 2:
        errors.append(
            f"after_count={after_count} > 2x PDF regex count={pdf_pasal_count}"
        )
    return len(errors) == 0, errors


def _run_llm_fix_call(prompt: str) -> tuple[dict | None, dict, str | None]:
    """Shared LLM call + parse. Returns (llm_obj_or_None, usage, error_message)."""
    try:
        raw, usage = call_gemini(prompt)
    except Exception as exc:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0, "elapsed_s": 0.0}, f"llm call: {exc}"
    try:
        llm_obj = parse_llm_json(raw)
    except json.JSONDecodeError as exc:
        return None, usage, f"json parse: {exc} (raw_preview: {raw[:300]!r})"
    return llm_obj, usage, None


def _accumulate_usage(dst: dict, src: dict) -> None:
    """Add per-call usage metrics into dst."""
    for k in ("input_tokens", "output_tokens", "total_tokens", "calls"):
        dst[k] = dst.get(k, 0) + src.get(k, 0)
    dst["elapsed_s"] = round(dst.get("elapsed_s", 0.0) + src.get("elapsed_s", 0.0), 3)


CHUNK_PROMPT_TEMPLATE = """\
You are fixing ONE SECTION of a larger Indonesian legal document.

=== DOCUMENT METADATA ===
doc_id       : {doc_id}
judul        : {judul}
is_perubahan : {is_perubahan}
section      : {chunk_label} ({chunk_index}/{chunk_total})

=== SECTION TO FIX (parser output for THIS section only) ===
Parse the Pasal tree below. Your task is to RETURN A COMPLETE CORRECTED
VERSION of this exact section — same Pasal numbers, same scope, but with any
flagged issues fixed. Keep every Pasal and every child node. DO NOT summarize,
DO NOT drop content, DO NOT return only the BAB title.

{chunk_view}

=== KNOWN ISSUES IN THIS SECTION ===
{issues_list}
{missing_angka_hint}

=== FULL PDF TEXT (ground truth, with spatial coordinates) ===
Blocks tagged with [x=column-left y=top]. Two-column gazette layouts put
left column at x~50-280, right column at x~300-550. Use coords to
reconstruct body attribution when heading clusters (e.g. "Pasal 7\\nPasal 8")
and bodies are in different blocks.

{pdf_text}

=== INSTRUCTIONS ===

Output a JSON object with a single top-level key "structure". The "structure"
value must contain ALL nodes shown in the section above, preserved at the same
hierarchy depth. Every Pasal in the section must appear in your output with
full body text (either as "text" on a leaf, or as "nodes" on a container plus
optional intro "text" on the parent).

Example of CORRECT output shape for a BAB section containing 3 Pasals:
  {{
    "structure": [
      {{
        "title": "BAB I - KETENTUAN UMUM",
        "nodes": [
          {{ "title": "Pasal 1", "text": "..." }},
          {{ "title": "Pasal 2", "text": "..." }},
          {{ "title": "Pasal 3", "nodes": [...] }}
        ]
      }}
    ]
  }}

WRONG output (returning only the BAB header without its Pasal children) —
DO NOT do this:
  {{ "structure": [ {{ "title": "KETENTUAN UMUM" }} ] }}

Rules:
1. Preserve all Pasal numbers and their content from the section above.
2. Fix flagged issues (empty_pasal, underspilt, bleed, etc.) using the PDF
   blocks as ground truth.
3. Follow the content-attribution rule: intro text on parent, list items on
   children, no "a."/"1." prefix in child text.
4. Drop footer noise: "SK No ...", page numbers "- N -", repeated
   "PRESIDEN REPUBLIK INDONESIA".
5. Output ONLY valid JSON, no markdown fences, no prose.

=== AMENDMENT (PERUBAHAN) DOCS — SPECIAL HANDLING ===

When is_perubahan=true, the document amends another law. Structure is:
  Pasal I (roman, root)
  └── Angka 1, Angka 2, Angka 3, ...  (each is one amendment instruction)
       └── the NEW pasal/ayat/huruf text being inserted/modified

Rules for amendment:
- Keep "Pasal I" as the top-level root (title exactly "Pasal I").
- Each Angka instruction is a child of Pasal I. Title format:
    "Pasal I Angka N"   (short and clean — do NOT concatenate the instruction
                        text into the title).
- Inside each Angka, place the amended Pasal/Ayat/Huruf as nested children.
  For example, Angka 2 that inserts new Pasal 3A with 2 Ayat would look like:
    {{
      "title": "Pasal I Angka 2",
      "text": "Di antara Pasal 3 dan Pasal 4 disisipkan 1 (satu) pasal, yakni Pasal 3A sehingga berbunyi sebagai berikut:",
      "nodes": [
        {{
          "title": "Pasal 3A",
          "nodes": [
            {{ "title": "Pasal 3A Ayat (1)", "text": "..." }},
            {{ "title": "Pasal 3A Ayat (2)", "text": "..." }}
          ]
        }}
      ]
    }}
- Do NOT bleed next Angka's content into previous Angka. When you see
  "3. Bab II dihapus" appearing right after a Pasal body, that is Angka 3
  starting — put it in its OWN Angka sibling, not in the previous Ayat text.
"""


def _fix_chunk(
    doc_id: str,
    parser_doc: dict,
    chunk_nodes: list[dict],
    chunk_index: int,
    chunk_total: int,
    pdf_text: str,
    issues: dict,
    chunk_meta: dict | None = None,
) -> tuple[list[dict] | None, dict, str | None]:
    """Fix a single chunk of the body. Returns (fixed_nodes, usage, error)."""
    chunk_label = chunk_nodes[0].get("title", "")[:40] if chunk_nodes else ""
    # For amendment angka_group chunks, suffix the Angka range for clarity.
    if (
        chunk_nodes
        and _PASAL_ROMAN_TITLE_RE.match(chunk_nodes[0].get("title", ""))
        and chunk_nodes[0].get("nodes")
    ):
        inner = chunk_nodes[0]["nodes"]
        first = re.search(r"Angka\s+(\d+)", inner[0].get("title", ""))
        last = re.search(r"Angka\s+(\d+)", inner[-1].get("title", ""))
        if first and last:
            chunk_label += f" Angka {first.group(1)}-{last.group(1)}"
    chunk_issues = filter_issues_for_chunk(chunk_nodes, issues)
    # Build missing-Angka hint for amendment angka_group chunks.
    missing_angka_hint = ""
    if chunk_meta and chunk_meta.get("kind") == "angka_group":
        missing = chunk_meta.get("angka_missing") or []
        lo, hi = chunk_meta.get("angka_range", (0, 0))
        parent = chunk_meta.get("parent_title", "")
        if missing:
            hint_lines = [
                "",
                "=== MISSING ANGKA (parser skipped these — RECOVER FROM PDF) ===",
                f"Expected Angka range for {parent} in this section: {lo}-{hi}.",
                f"Parser is missing: Angka {', Angka '.join(str(n) for n in missing)}.",
                (
                    "Scan the PDF text for these numbered items and include them"
                    " as siblings in your output. Each missing Angka should be a"
                    f" direct child of \"{parent}\" with title \"{parent} Angka N\"."
                ),
                (
                    "If a missing number truly does not appear in the PDF text"
                    " (e.g. the document skips numbering), omit it rather than"
                    " hallucinating."
                ),
            ]
            missing_angka_hint = "\n".join(hint_lines)
    prompt = CHUNK_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        judul=parser_doc.get("judul", "(unknown)"),
        is_perubahan=parser_doc.get("is_perubahan", False),
        chunk_label=chunk_label,
        chunk_index=chunk_index,
        chunk_total=chunk_total,
        chunk_view=compact_tree_view(chunk_nodes),
        issues_list=build_issues_list(chunk_issues),
        missing_angka_hint=missing_angka_hint,
        pdf_text=pdf_text,
    )
    est_tokens = len(prompt) // 4
    print(
        f"  chunk {chunk_index}/{chunk_total} ({chunk_label}): "
        f"{est_tokens:,} input tokens...",
        flush=True,
    )
    llm_obj, usage, err = _run_llm_fix_call(prompt)
    # Save per-chunk output for debugging.
    debug_dir = REPO_ROOT / "tmp" / f"llm_fix_chunks_{doc_id}"
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_path = debug_dir / f"chunk_{chunk_index:02d}.json"
    try:
        with open(debug_path, "w", encoding="utf-8") as f:
            json.dump(llm_obj or {"_error": err}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    if err:
        return None, usage, err
    if not isinstance(llm_obj, dict) or "structure" not in llm_obj:
        return None, usage, "missing 'structure' key"
    struct = llm_obj["structure"]
    if not isinstance(struct, list):
        return None, usage, "'structure' must be a list"
    # Sanity check: if all returned nodes have no body AND no children,
    # the chunk output is degenerate (LLM returned only headers).
    body_nodes = 0
    for node in struct:
        if (node.get("text") and node["text"].strip()) or node.get("nodes"):
            body_nodes += 1
    if body_nodes == 0 and len(chunk_nodes) > 0:
        return None, usage, "degenerate output (all nodes empty)"
    return struct, usage, None


def fix_doc(doc_id: str, quality_report_docs: dict, dry_run: bool = False) -> dict:
    """Apply grounded LLM fix to one doc. Returns audit record.

    Routes big docs (>CHUNK_PASAL_THRESHOLD pasals) through BAB-chunked fix
    to stay under Gemini's 65K output-token ceiling.
    """
    audit = {
        "doc_id": doc_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "status": "pending",
    }

    index_path = None
    for cat_dir in INDEX_PASAL.iterdir():
        candidate = cat_dir / f"{doc_id}.json"
        if candidate.exists():
            index_path = candidate
            break
    if index_path is None:
        audit["status"] = "error"
        audit["error"] = "index file not found"
        return audit

    report_entry = quality_report_docs.get(doc_id, {})
    audit["before_score"] = report_entry.get("score")
    audit["before_status"] = report_entry.get("status")

    with open(index_path, encoding="utf-8") as f:
        parser_doc = json.load(f)

    try:
        pdf_text = load_pdf_blocks(doc_id)
    except Exception as exc:
        audit["status"] = "error"
        audit["error"] = f"pdf load: {exc}"
        return audit

    total_pasals = count_pasals_in_tree(parser_doc.get("structure", []))
    audit["before_pasal_count"] = total_pasals
    use_chunked = total_pasals > CHUNK_PASAL_THRESHOLD
    audit["mode"] = "chunked" if use_chunked else "whole"

    if not use_chunked:
        # Whole-doc single LLM call.
        prompt = PROMPT_TEMPLATE.format(
            doc_id=doc_id,
            judul=parser_doc.get("judul", "(unknown)"),
            is_perubahan=parser_doc.get("is_perubahan", False),
            parser_view=compact_tree_view(parser_doc.get("structure", [])),
            issues_list=build_issues_list(report_entry),
            pdf_text=pdf_text,
        )
        est_tokens = len(prompt) // 4
        if est_tokens > MAX_INPUT_TOKENS_HINT:
            print(f"  WARN large prompt: {est_tokens:,} tokens", flush=True)
        print(f"  calling Gemini ({est_tokens:,} input tokens)...", flush=True)
        llm_obj, usage, err = _run_llm_fix_call(prompt)
        audit["llm_fix_input_tokens"] = usage["input_tokens"]
        audit["llm_fix_output_tokens"] = usage["output_tokens"]
        audit["llm_fix_total_tokens"] = usage["total_tokens"]
        audit["llm_fix_time_s"] = usage["elapsed_s"]
        audit["llm_fix_calls"] = usage["calls"]
        if err:
            audit["status"] = "error"
            audit["error"] = err
            return audit
        new_structure = llm_obj["structure"]
    else:
        # Chunked flow: split body by BAB (or per-Angka for amendment docs).
        # Scope PDF per chunk + parallel execution + skip chunks without issues.
        preamble, body = split_preamble_and_body(parser_doc.get("structure", []))
        # Amendment mode triggers on Pasal Roman pattern, regardless of
        # is_perubahan flag (parser's detection can miss amendment docs).
        has_big_pasal_roman = any(
            _PASAL_ROMAN_TITLE_RE.match(n.get("title", ""))
            and len(n.get("nodes", []) or []) > ANGKA_PER_CHUNK
            for n in body
        )
        if has_big_pasal_roman:
            chunks, chunk_meta = chunk_body_for_amendment(body)
            chunk_mode = "amendment"
        else:
            chunks = chunk_body_by_bab(body)
            chunk_meta = None
            chunk_mode = "bab"
        audit["chunk_mode"] = chunk_mode
        pdf_pages = load_pdf_pages(doc_id)
        total_pdf_pages = len(pdf_pages)
        print(
            f"  chunked mode: {total_pasals} pasals across {len(chunks)} chunk(s), "
            f"parallel={CHUNK_PARALLEL_WORKERS}",
            flush=True,
        )
        agg_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "elapsed_s": 0.0,
        }
        fixed_chunks: list[list[dict]] = [None] * len(chunks)  # type: ignore
        chunk_errors: list[str] = []
        skipped_count = 0

        # Build per-chunk task list: (index, chunk_nodes, chunk_pdf_text,
        # chunk_issues, has_issues).
        from concurrent.futures import ThreadPoolExecutor, as_completed

        tasks: list[dict] = []
        for i, chunk_nodes in enumerate(chunks, 1):
            chunk_issues = filter_issues_for_chunk(chunk_nodes, report_entry)
            has_issues = (
                chunk_issues.get("empty_pasal_count", 0) > 0
                or chunk_issues.get("bleed_count", 0) > 0
                or chunk_issues.get("underspilt_count", 0) > 0
                or report_entry.get("gap_count", 0) > 0
                or report_entry.get("completeness", 1.0) < 1.0
                or not report_entry.get("monotonic", True)
            )
            if not has_issues:
                # Skip this chunk — parser output is clean. Keep original nodes.
                fixed_chunks[i - 1] = chunk_nodes
                skipped_count += 1
                print(f"  chunk {i}/{len(chunks)}: no issues, kept as-is", flush=True)
                continue
            # Scope PDF to this chunk's page range (+ buffer for body continuation).
            # For amendment angka_group chunks, derive range from inner Angka
            # children (not the wrapper, which spans the whole Pasal Roman).
            if (
                chunk_mode == "amendment"
                and chunk_meta is not None
                and chunk_meta[i - 1]["kind"] == "angka_group"
                and chunk_nodes
                and chunk_nodes[0].get("nodes")
            ):
                wrapper = chunk_nodes[0]
                inner = wrapper["nodes"]
                lo = chunk_meta[i - 1].get("angka_range", (0, 0))[0]
                # First chunk of wrapper (Angka 1 expected): parser's start_index
                # can be off (it may have missed the Pasal I header on an earlier
                # page). Scope from page 1 so the LLM sees the missing early
                # Angka + preamble→body transition.
                if lo == 1:
                    chunk_start = 1
                else:
                    chunk_start = inner[0].get("start_index")
                chunk_end = inner[-1].get("end_index")
            else:
                chunk_start = chunk_nodes[0].get("start_index") if chunk_nodes else None
                chunk_end = chunk_nodes[-1].get("end_index") if chunk_nodes else None
            if chunk_start is None or chunk_end is None:
                scoped_pdf = pdf_text  # fallback to full doc
            else:
                s = max(1, int(chunk_start) - CHUNK_PAGE_BUFFER)
                e = min(total_pdf_pages, int(chunk_end) + CHUNK_PAGE_BUFFER)
                scoped_pdf = format_pdf_pages(pdf_pages, s, e)
            tasks.append({
                "index": i,
                "chunk_nodes": chunk_nodes,
                "chunk_pdf": scoped_pdf,
                "chunk_issues": chunk_issues,
                "chunk_meta": (chunk_meta[i - 1] if chunk_meta is not None else None),
            })

        # Run remaining chunks in parallel.
        if tasks:
            with ThreadPoolExecutor(max_workers=CHUNK_PARALLEL_WORKERS) as pool:
                futures = {
                    pool.submit(
                        _fix_chunk,
                        doc_id,
                        parser_doc,
                        t["chunk_nodes"],
                        t["index"],
                        len(chunks),
                        t["chunk_pdf"],
                        t["chunk_issues"],
                        t["chunk_meta"],
                    ): t
                    for t in tasks
                }
                for future in as_completed(futures):
                    t = futures[future]
                    try:
                        fixed, usage, err = future.result()
                    except Exception as exc:
                        fixed, usage, err = None, {}, f"worker exception: {exc}"
                    _accumulate_usage(agg_usage, usage)
                    i = t["index"]
                    if err or fixed is None:
                        print(
                            f"  chunk {i} failed: {err} — keeping original",
                            flush=True,
                        )
                        fixed = t["chunk_nodes"]
                        chunk_errors.append(f"chunk{i}:{err}")
                    fixed_chunks[i - 1] = fixed

        audit["llm_fix_input_tokens"] = agg_usage["input_tokens"]
        audit["llm_fix_output_tokens"] = agg_usage["output_tokens"]
        audit["llm_fix_total_tokens"] = agg_usage["total_tokens"]
        audit["llm_fix_time_s"] = agg_usage["elapsed_s"]
        audit["llm_fix_calls"] = agg_usage["calls"]
        audit["chunk_count"] = len(chunks)
        audit["chunk_skipped"] = skipped_count
        audit["chunk_errors"] = chunk_errors
        # Stitch: preamble + fixed chunks (merge angka groups for amendment mode).
        if chunk_mode == "amendment":
            new_structure = list(preamble) + stitch_amendment_chunks(
                fixed_chunks, chunk_meta  # type: ignore[arg-type]
            )
        else:
            new_structure = list(preamble)
            for c in fixed_chunks:
                new_structure.extend(c or [])

    # Count pasals from raw PDF (flat text) for sanity check.
    try:
        flat_pdf = load_pdf_text(doc_id)
    except Exception:
        flat_pdf = pdf_text
    pdf_pasal_count = len(
        set(re.findall(r"(?m)^\s*[Pp]asa[l1]\s*(\d+[A-Z]?)\s*[']?\s*$", flat_pdf))
    )

    llm_wrapper = {"structure": new_structure}
    ok, errors = validate_fix(parser_doc, llm_wrapper, pdf_pasal_count)
    if not ok:
        audit["status"] = "rejected"
        audit["errors"] = errors
        reject_path = REPO_ROOT / "tmp" / f"llm_fix_rejected_{doc_id}.json"
        reject_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reject_path, "w", encoding="utf-8") as f:
            json.dump(llm_wrapper, f, ensure_ascii=False, indent=2)
        audit["rejected_preview_path"] = str(reject_path)
        return audit

    normalize_titles(new_structure)
    assign_node_ids(new_structure)
    build_navigation_paths(new_structure)
    # LLM output lacks start_index/end_index — re-populate from original
    # so downstream re-split (ayat/full_split) does not KeyError.
    doc_total_pages = parser_doc.get("total_pages") or 1
    for node in parser_doc.get("structure", []):
        end = node.get("end_index")
        if end is not None:
            doc_total_pages = max(doc_total_pages, int(end))
    ensure_page_indices(new_structure, parser_doc.get("structure", []), doc_total_pages)

    corrected = {**parser_doc}
    corrected["structure"] = new_structure
    corrected["llm_fix_applied_at"] = datetime.now(timezone.utc).isoformat()
    corrected["element_counts"] = {
        "pasal": count_pasals_in_tree(new_structure),
    }

    audit["after_pasal_count"] = corrected["element_counts"]["pasal"]

    if dry_run:
        audit["status"] = "dry_run_ok"
        preview_path = REPO_ROOT / "tmp" / f"llm_fix_preview_{doc_id}.json"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        with open(preview_path, "w", encoding="utf-8") as f:
            json.dump(corrected, f, ensure_ascii=False, indent=2)
        audit["preview_path"] = str(preview_path)
        return audit

    backup_cat = BACKUP_DIR / index_path.parent.name
    backup_cat.mkdir(parents=True, exist_ok=True)
    backup_path = backup_cat / index_path.name
    if not backup_path.exists():
        shutil.copy2(index_path, backup_path)
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(corrected, f, ensure_ascii=False, indent=2)
    audit["status"] = "fixed"
    audit["index_path"] = str(index_path)
    audit["backup_path"] = str(backup_path)
    audit["category"] = parser_doc.get("jenis_folder") or index_path.parent.name
    return audit


def load_targets(status_filter: str | None, specific: list[str] | None) -> list[str]:
    """Return ordered list of doc_ids to process."""
    if specific:
        return specific
    if not QUALITY_REPORT.exists():
        raise FileNotFoundError(
            f"{QUALITY_REPORT} missing - run scripts/parser/validate.py first"
        )
    report = json.load(open(QUALITY_REPORT, encoding="utf-8"))
    docs = report.get("docs", [])
    if status_filter:
        statuses = {s.strip().upper() for s in status_filter.split(",")}
        docs = [d for d in docs if d.get("status") in statuses]
    else:
        docs = [d for d in docs if d.get("status") != "OK"]
    return [d["doc_id"] for d in docs]


def append_audit(entry: dict) -> None:
    """Append entry to data/llm_fix_log.json."""
    records = []
    if AUDIT_LOG.exists():
        try:
            records = json.load(open(AUDIT_LOG, encoding="utf-8"))
        except Exception:
            records = []
    records.append(entry)
    with open(AUDIT_LOG, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def update_cost_manifest(doc_id: str, category: str, audit: dict) -> None:
    """Merge LLM fix cost into data/indexing_logs/cost_pasal.json.

    Appends llm_fix_* fields alongside existing parse_time_s / llm_* fields
    so all cost metrics live in one per-doc record.
    """
    if not audit.get("llm_fix_time_s"):
        return  # skip if no actual LLM call happened
    manifest = {}
    if COST_MANIFEST.exists():
        try:
            manifest = json.load(open(COST_MANIFEST, encoding="utf-8"))
        except Exception:
            manifest = {}
    entry = manifest.get(doc_id, {})
    if category and not entry.get("category"):
        entry["category"] = category
    entry["updated_at"] = datetime.now(timezone.utc).isoformat()
    entry["llm_fix_time_s"] = audit.get("llm_fix_time_s", 0.0)
    entry["llm_fix_input_tokens"] = audit.get("llm_fix_input_tokens", 0)
    entry["llm_fix_output_tokens"] = audit.get("llm_fix_output_tokens", 0)
    entry["llm_fix_total_tokens"] = audit.get("llm_fix_total_tokens", 0)
    entry["llm_fix_calls"] = audit.get("llm_fix_calls", 0)
    entry["llm_fix_applied_at"] = entry["updated_at"]
    manifest[doc_id] = entry
    COST_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(COST_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def main() -> None:
    """CLI entry for grounded LLM fix."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--doc-id", type=str, help="Fix a single doc_id")
    ap.add_argument("--doc-ids", type=str, help="Comma-separated doc_ids")
    ap.add_argument(
        "--status",
        type=str,
        help="Filter by validator status (FAIL, PARTIAL, or comma). Default: all non-OK.",
    )
    ap.add_argument("--dry-run", action="store_true", help="Preview only")
    args = ap.parse_args()

    specific = None
    if args.doc_id:
        specific = [args.doc_id]
    elif args.doc_ids:
        specific = [d.strip() for d in args.doc_ids.split(",") if d.strip()]

    targets = load_targets(args.status, specific)
    if not targets:
        print("No target docs.")
        return

    report = {}
    if QUALITY_REPORT.exists():
        report_docs = json.load(open(QUALITY_REPORT, encoding="utf-8")).get("docs", [])
        report = {d["doc_id"]: d for d in report_docs}

    print(
        f"Targets: {len(targets)} docs"
        f"{' (DRY-RUN)' if args.dry_run else ''}",
        flush=True,
    )

    for i, doc_id in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {doc_id}", flush=True)
        audit = fix_doc(doc_id, report, dry_run=args.dry_run)
        status = audit.get("status")
        err = audit.get("error", "") or ""
        print(f"  status: {status}  {err}", flush=True)
        if audit.get("before_pasal_count") is not None:
            print(
                f"  pasal count: {audit.get('before_pasal_count')} -> "
                f"{audit.get('after_pasal_count')}",
                flush=True,
            )
        append_audit(audit)
        if not args.dry_run and audit.get("status") == "fixed":
            update_cost_manifest(doc_id, audit.get("category", ""), audit)


if __name__ == "__main__":
    main()
