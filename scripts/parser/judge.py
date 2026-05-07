"""LLM judge for parsed-index quality.

Compares raw PDF text against the parsed pasal-level JSON and produces a
per-doc report of coverage gaps, structural issues, and OCR corruption.
Runs on Gemini 2.5 Pro (configured in vectorless/models.py) for cross-family
verification of the OpenAI parser. Free under Vertex AI trial credit.

Output: data/judge_report.json. Requires Vertex AI ADC.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from scripts.parser._common import load_pdf_pages  # noqa: E402
from vectorless.models import JUDGE_MODEL  # noqa: E402

INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"
REPORT_PATH = REPO_ROOT / "data" / "judge_report.json"
REGISTRY_PATH = REPO_ROOT / "data" / "raw" / "registry.json"

MAX_OUTPUT_TOKENS = 16384

# Gemini 2.5 Pro pricing per 1M tokens (under 200K context). Used for cost logs.
PRICE_INPUT_PER_M = 1.25
PRICE_OUTPUT_PER_M = 10.0


JUDGE_PROMPT_TEMPLATE = """You audit the quality of an Indonesian legal document parse.

Compare the RAW PDF TEXT against the PARSED STRUCTURE (JSON). Produce a
concise, machine-readable defect report.

=== DOC METADATA ===
doc_id       : {doc_id}
judul        : {judul}
is_perubahan : {is_perubahan}

=== RAW PDF TEXT (body pages, plain text) ===
{pdf_text}

=== PARSED STRUCTURE (JSON, produced by the indexing LLM) ===
{structure_json}

=== YOUR TASK ===
Inspect the parsed structure against the PDF text. Identify defects in
these categories:

1. COVERAGE — pasals in PDF but missing from parse, or vice-versa.
   For amendment docs (is_perubahan=true), the parsed Pasal labels live
   inside "Pasal I"/"Pasal II" containers as nested arabic "Pasal N"
   nodes. Compare those against arabic Pasal references in the PDF
   amendment instructions (e.g. "Ketentuan Pasal 7 diubah ...").

2. STRUCTURAL — narrow scope. A separate deterministic Python check
   (scripts/parser/check_granularity.py) handles "unsplit" detection
   at the ayat/rincian levels using the exact splitter regex.
   **Do NOT report "unsplit" issues here** — those are caught upstream.

   The parser's design is INTENTIONAL: Pasal bodies stay FLAT as one
   "text" string, keeping "(1)", "(2)", "a.", "b.", "1.", "2." markers
   INLINE. You will see many Pasals with flat text and no children.
   That is CORRECT — do not flag it.

   Flag ONLY these issue types:
     - "hybrid": text contains inline markers AND has children (broken).
     - "misplaced": BAB/Bagian/Paragraf at wrong nesting depth (e.g. a
       Bagian emitted as top-level instead of a child of its BAB).
     - "duplicate": two or more container nodes share the same node_id.
     - "text_bleed": Pasal N text contains trailing ayat/content from
       Pasal N-1, or vice versa.
     - "sibling_order": children appear in wrong order vs PDF (e.g.
       Bagian Kesatu, Keempat, Kedua, Ketiga instead of Kesatu, Kedua,
       Ketiga, Keempat).
     - "title_incomplete": BAB/Bagian/Paragraf title truncated (missing
       words that the PDF clearly shows).

3. OCR — tokens in the parsed text that look garbled vs the PDF (e.g.
   "perenczrna.an" vs "perencanaan"). Only report clear corruption,
   not stylistic differences.

Output STRICT JSON (no markdown, no prose). Schema:

{{
  "doc_id": "{doc_id}",
  "coverage": {{
    "pdf_pasals": ["list of arabic Pasal labels found in PDF body"],
    "parsed_pasals": ["list of arabic Pasal labels found in structure"],
    "missing": ["labels in pdf_pasals not in parsed_pasals"],
    "extra":   ["labels in parsed_pasals not in pdf_pasals"]
  }},
  "structural_issues": [
    {{"node_id": "...", "issue": "hybrid|unsplit|empty|duplicate", "detail": "..."}}
  ],
  "ocr_issues": [
    {{"location": "node_id or path", "garbled": "...", "expected": "..."}}
  ],
  "overall_score": 0.0,
  "verdict": "OK|MINOR|MAJOR|FAIL",
  "notes": "1-2 sentence summary of quality"
}}

Scoring:
- 1.00 OK    — no issues found
- 0.80-0.99  MINOR — <=2 OCR issues OR 1 missing pasal
- 0.50-0.79  MAJOR — coverage gaps or multiple structural issues
- <0.50      FAIL  — severe coverage loss or broken structure

Cap ocr_issues at 10 most impactful samples. Reply with ONLY the JSON."""


def _load_registry() -> dict:
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        return json.load(f)


def _find_index_path(doc_id: str) -> Path | None:
    for p in INDEX_PASAL.glob(f"*/{doc_id}.json"):
        return p
    return None


def _pdf_body_text(doc_id: str, body_end: int | None) -> str:
    """Concatenate cleaned body-page text using the block-level PDF loader."""
    pages = load_pdf_pages(doc_id)
    if body_end is None:
        body_end = len(pages)
    out: list[str] = []
    for p in pages:
        if p["page_num"] > body_end:
            break
        for b in p.get("blocks", []):
            out.append(b["text"])
        out.append("")
    return "\n".join(out)


def _call_judge(prompt: str) -> tuple[dict, dict]:
    """Invoke the judge model via the central LLM wrapper.

    Returns (parsed_report, usage_meta). JSON parsing and transient retries
    are handled inside vectorless.llm.call.
    """
    from vectorless.llm import call as llm_call

    t0 = time.time()
    parsed, base_usage = llm_call(
        prompt,
        model=JUDGE_MODEL,
        max_completion_tokens=MAX_OUTPUT_TOKENS,
        return_usage=True,
    )
    elapsed = time.time() - t0
    input_tokens = base_usage["input_tokens"]
    output_tokens = base_usage["output_tokens"]
    cost = (
        input_tokens * PRICE_INPUT_PER_M / 1_000_000
        + output_tokens * PRICE_OUTPUT_PER_M / 1_000_000
    )
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": round(cost, 4),
        "elapsed_s": round(elapsed, 2),
    }
    return parsed, usage


def judge_doc(doc_id: str) -> dict:
    """Run the judge on one doc. Returns the parsed report dict."""
    idx_path = _find_index_path(doc_id)
    if not idx_path:
        return {"doc_id": doc_id, "verdict": "ERROR", "error": "no index found"}
    with open(idx_path, encoding="utf-8") as f:
        doc = json.load(f)

    pdf_text = _pdf_body_text(doc_id, doc.get("body_pages"))
    structure_json = json.dumps(doc.get("structure", []), ensure_ascii=False, indent=2)
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        judul=doc.get("judul", ""),
        is_perubahan=doc.get("is_perubahan", False),
        pdf_text=pdf_text,
        structure_json=structure_json,
    )

    t0 = datetime.now(timezone.utc)
    report, usage = _call_judge(prompt)
    report["judged_at"] = t0.isoformat()
    report["judge_model"] = JUDGE_MODEL
    report["usage"] = usage
    return report


def _save_reports(reports: list[dict]) -> None:
    existing: dict = {}
    if REPORT_PATH.exists():
        try:
            prev = json.load(open(REPORT_PATH, encoding="utf-8"))
            for r in prev.get("docs", []):
                existing[r["doc_id"]] = r
        except json.JSONDecodeError:
            pass
    for r in reports:
        existing[r["doc_id"]] = r
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
        "docs": sorted(existing.values(), key=lambda r: r["doc_id"]),
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def _resolve_targets(specific: list[str], category: str | None) -> list[str]:
    if specific:
        return specific
    if not category:
        raise SystemExit("must pass --doc-id(s) or --category")
    reg = _load_registry()
    target = category.upper()
    return sorted(
        did for did, entry in reg.items()
        if (entry.get("jenis_folder") or "").upper() == target
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemini-based judge for parsed index quality")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[])
    ap.add_argument("--category", help="Judge every doc in this jenis_folder")
    ap.add_argument("--source",
                    help="Override INDEX_PASAL source root (e.g. data/index_pasal_eval/v4-flash). "
                         "Use with --report to run judge on a bake-off parse without touching the canonical report.")
    ap.add_argument("--report",
                    help="Override REPORT_PATH output (e.g. data/judge_report_v4-flash.json).")
    args = ap.parse_args()

    global INDEX_PASAL, REPORT_PATH
    if args.source:
        INDEX_PASAL = (REPO_ROOT / args.source).resolve()
        print(f"Source override: {INDEX_PASAL}")
    if args.report:
        REPORT_PATH = (REPO_ROOT / args.report).resolve()
        print(f"Report override: {REPORT_PATH}")

    targets = _resolve_targets(list(args.doc_ids), args.category)

    print(f"judging {len(targets)} docs with {JUDGE_MODEL}")
    reports: list[dict] = []
    for i, did in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {did} ...", flush=True)
        try:
            report = judge_doc(did)
        except Exception as exc:
            print(f"  error: {exc}", flush=True)
            report = {"doc_id": did, "verdict": "ERROR", "error": str(exc)}
        verdict = report.get("verdict", "?")
        score = report.get("overall_score")
        cov = report.get("coverage") or {}
        miss = len(cov.get("missing") or [])
        struct = len(report.get("structural_issues") or [])
        ocr = len(report.get("ocr_issues") or [])
        usage = report.get("usage") or {}
        in_tok = usage.get("input_tokens") or 0
        out_tok = usage.get("output_tokens") or 0
        cost = usage.get("cost_usd") or 0
        print(
            f"  {verdict}  score={score}  miss={miss} struct={struct} ocr={ocr}  "
            f"tokens={in_tok:,}/{out_tok:,}  ${cost:.4f}",
            flush=True,
        )
        reports.append(report)
        _save_reports(reports)

    total_in = sum((r.get("usage") or {}).get("input_tokens") or 0 for r in reports)
    total_out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in reports)
    total_cost = sum((r.get("usage") or {}).get("cost_usd") or 0 for r in reports)
    print(
        f"\ntotal tokens: {total_in:,} in / {total_out:,} out  "
        f"cost ${total_cost:.4f}"
    )
    print(f"report: {REPORT_PATH}")

    if args.source:
        print("skipping corpus_status refresh (--source override active)")
    else:
        try:
            from scripts.parser.corpus_status import build_status, write_status
            write_status(build_status())
            print("corpus_status.json refreshed")
        except Exception as exc:
            print(f"warning: corpus_status refresh failed: {exc}")


if __name__ == "__main__":
    main()
