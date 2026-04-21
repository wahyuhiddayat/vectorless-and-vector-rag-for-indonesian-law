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

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3
MAX_INPUT_TOKENS_HINT = 200_000


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


def load_pdf_blocks(doc_id: str) -> str:
    """Read the PDF and return per-page blocks with spatial (x, y) coords.

    Format:
        === PAGE N ===
        [x=50 y=100] text block 1
        [x=300 y=100] text block 2
        ...

    Helps LLM disambiguate multi-column layouts where heading clusters appear
    in one column while body content is in another (common in Indonesian
    government gazette format).
    """
    import pymupdf

    pdf_path = find_pdf_path(doc_id)
    if not pdf_path:
        raise FileNotFoundError(f"PDF not found for {doc_id}")

    out: list[str] = []
    with pymupdf.open(str(pdf_path)) as doc:
        for page_i, page in enumerate(doc, 1):
            out.append(f"=== PAGE {page_i} ===")
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
            # Sort by (y, x) approximates reading order for most single-column
            # pages; multi-column ambiguity remains in x coordinates (which the
            # LLM can use to group into columns).
            blocks.sort(key=lambda b: (b["y0"], b["x0"]))
            for b in blocks:
                out.append(f"[x={b['x0']} y={b['y0']}] {b['text']}")
            out.append("")  # blank line between pages
    return "\n".join(out)


def call_gemini(prompt: str, max_output_tokens: int = 32768) -> str:
    """Call Gemini 2.5 Flash with temperature=0 and return raw text."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                ),
            )
            text = resp.text or ""
            if text.strip():
                return text
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


def count_pasals_in_tree(structure: list[dict]) -> int:
    """Count unique Pasal-titled nodes (at any depth)."""
    count = 0
    for node in structure:
        title = node.get("title", "")
        if title.startswith("Pasal ") and not any(x in title for x in ("Ayat", "Huruf", "Angka")):
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
    """Recursively assign suffixed node_ids based on the DEEPEST marker in title.

    Title format examples and resulting suffix:
        "Pasal 3 Ayat (1)"                    -> a1
        "Pasal 3 Huruf a"                     -> h1
        "Pasal 3 Angka 2"                     -> n2
        "Pasal 3 Ayat (2) Huruf a"            -> h1 (huruf is deepest)
        "Pasal 3 Ayat (2) Huruf c Angka 4"    -> n4 (angka is deepest)
    """
    parent_id = parent.get("node_id", "")
    for i, child in enumerate(parent.get("nodes", []), 1):
        title = child.get("title", "")
        # Try matching at end-of-string, most specific first.
        m_angka = re.search(r"Angka\s+(\d+)\s*$", title)
        m_huruf = re.search(r"Huruf\s+([a-z])\s*$", title)
        m_ayat = re.search(r"Ayat\s+\((\d+)\)\s*$", title)
        if m_angka:
            suffix = f"n{m_angka.group(1)}"
        elif m_huruf:
            # Map a=1, b=2, c=3 deterministically.
            suffix = f"h{ord(m_huruf.group(1)) - ord('a') + 1}"
        elif m_ayat:
            suffix = f"a{m_ayat.group(1)}"
        else:
            suffix = f"x{i}"
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


def normalize_titles(structure: list[dict], ancestor_titles: list[str] | None = None) -> None:
    """Rewrite short child titles to include Pasal/Ayat prefix chain.

    LLM often emits short titles like "Ayat (1)" or "Huruf a" for child nodes
    when the parent is obvious. Existing parser/validator expects full titles
    like "Pasal 7 Ayat (1)" or "Pasal 7 Ayat (1) Huruf a".

    Rule: if a child title does not start with "Pasal "/"BAB "/"Bagian "/
    "Paragraf "/preamble keyword and there is a pasal ancestor, prepend the
    pasal-rooted ancestor chain.
    """
    if ancestor_titles is None:
        ancestor_titles = []
    for node in structure:
        title = node.get("title", "")
        # Root-level structural titles don't need prefixing.
        is_root_level = (
            title.startswith(("Pasal ", "BAB ", "Bagian ", "Paragraf "))
            or title in ("Pembukaan", "Menimbang", "Mengingat", "Menetapkan", "Penutup")
            or title.startswith("Pasal I")  # amendment pasal_roman
        )
        # Pasal-rooted ancestors: chain from the first Pasal ancestor onward.
        pasal_chain = []
        for anc in ancestor_titles:
            if anc.startswith("Pasal "):
                pasal_chain = [anc]
            elif pasal_chain:
                pasal_chain.append(anc)
        if not is_root_level and pasal_chain and title:
            node["title"] = " ".join(pasal_chain + [title])
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

    if after_count > max(before_count * 1.3, before_count + 5):
        errors.append(
            f"too many pasal added: before={before_count} after={after_count}"
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


def fix_doc(doc_id: str, quality_report_docs: dict, dry_run: bool = False) -> dict:
    """Apply grounded LLM fix to one doc. Returns audit record."""
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
    try:
        raw = call_gemini(prompt)
    except Exception as exc:
        audit["status"] = "error"
        audit["error"] = f"llm call: {exc}"
        return audit

    try:
        llm_obj = parse_llm_json(raw)
    except json.JSONDecodeError as exc:
        audit["status"] = "error"
        audit["error"] = f"json parse: {exc}"
        audit["raw_preview"] = raw[:500]
        return audit

    # Count pasals from raw PDF (flat text) for sanity check.
    try:
        flat_pdf = load_pdf_text(doc_id)
    except Exception:
        flat_pdf = pdf_text
    pdf_pasal_count = len(
        set(re.findall(r"(?m)^\s*[Pp]asa[l1]\s*(\d+[A-Z]?)\s*[']?\s*$", flat_pdf))
    )

    ok, errors = validate_fix(parser_doc, llm_obj, pdf_pasal_count)
    if not ok:
        audit["status"] = "rejected"
        audit["errors"] = errors
        # Save rejected output for manual inspection.
        reject_path = REPO_ROOT / "tmp" / f"llm_fix_rejected_{doc_id}.json"
        reject_path.parent.mkdir(parents=True, exist_ok=True)
        with open(reject_path, "w", encoding="utf-8") as f:
            json.dump(llm_obj, f, ensure_ascii=False, indent=2)
        audit["rejected_preview_path"] = str(reject_path)
        return audit

    new_structure = llm_obj["structure"]
    normalize_titles(new_structure)  # rewrite short child titles into full form
    assign_node_ids(new_structure)
    build_navigation_paths(new_structure)

    corrected = {**parser_doc}
    corrected["structure"] = new_structure
    corrected["llm_fix_applied_at"] = datetime.now(timezone.utc).isoformat()
    corrected["element_counts"] = {
        "pasal": count_pasals_in_tree(new_structure),
    }

    audit["after_pasal_count"] = corrected["element_counts"]["pasal"]
    audit["before_pasal_count"] = count_pasals_in_tree(parser_doc.get("structure", []))

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


if __name__ == "__main__":
    main()
