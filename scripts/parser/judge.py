"""Cross-model judge for LLM-parsed index quality.

For each doc: compare raw PDF text against the parsed pasal-level JSON.
Produces a per-doc report of coverage gaps (missing/extra pasals),
structural issues (hybrid/unsplit nodes), and OCR corruption samples.

Uses Claude Sonnet 4.6 via Claude Code subscription (no ANTHROPIC_API_KEY
needed). A different model family from the parser (Gemini 2.5 Flash)
reduces self-enhancement bias in the validation.

Usage:
    python scripts/parser/judge.py --doc-id uu-3-2025
    python scripts/parser/judge.py --doc-ids uu-3-2025,uu-14-2025
    python scripts/parser/judge.py --category UU

Output: data/judge_report.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.parser._common import load_pdf_pages  # noqa: E402

INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"
REPORT_PATH = REPO_ROOT / "data" / "judge_report.json"
REGISTRY_PATH = REPO_ROOT / "data" / "raw" / "registry.json"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # upgrade when newer Sonnet available
CALL_TIMEOUT_S = 600


JUDGE_PROMPT_TEMPLATE = """You audit the quality of an Indonesian legal document parse.

Compare the RAW PDF TEXT against the PARSED STRUCTURE (JSON). Produce a
concise, machine-readable defect report.

=== DOC METADATA ===
doc_id       : {doc_id}
judul        : {judul}
is_perubahan : {is_perubahan}

=== RAW PDF TEXT (body pages, plain text) ===
{pdf_text}

=== PARSED STRUCTURE (JSON, produced by Gemini 2.5 Flash) ===
{structure_json}

=== YOUR TASK ===
Inspect the parsed structure against the PDF text. Identify defects in
these categories:

1. COVERAGE — pasals in PDF but missing from parse, or vice-versa.
   For amendment docs (is_perubahan=true), the parsed Pasal labels live
   inside "Pasal I"/"Pasal II" containers as nested arabic "Pasal N"
   nodes. Compare those against arabic Pasal references in the PDF
   amendment instructions (e.g. "Ketentuan Pasal 7 diubah ...").

2. STRUCTURAL — nodes whose body text contains inline structural markers
   like "(1)", "(2)", "a.", "b.", "1.", "2." AND also have children,
   OR nodes whose flat text clearly hides un-extracted sub-structure
   (e.g. a Pasal leaf whose text has many "(N)" markers but no children
   — that is FINE because a deterministic re-split pass handles it).

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
    out = []
    for p in pages:
        if p["page_num"] > body_end:
            break
        for b in p.get("blocks", []):
            out.append(b["text"])
        out.append("")
    return "\n".join(out)


def _parse_judge_output(raw: str) -> dict:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    i = text.find("{")
    if i == -1:
        raise ValueError(f"no JSON in judge output: {text[:200]!r}")
    depth = 0
    end = -1
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end == -1:
        raise ValueError("unbalanced JSON in judge output")
    return json.loads(text[i:end])


def _call_claude(prompt: str) -> tuple[str, dict]:
    """Invoke `claude -p` headless. Returns (response_text, usage_meta).

    `--output-format json` makes Claude CLI emit a single JSON envelope
    with the response text plus token usage and cost. Prompt is piped
    via stdin to avoid command-line length limits.
    """
    try:
        result = subprocess.run(
            ["claude", "-p", "--model", JUDGE_MODEL, "--output-format", "json"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=CALL_TIMEOUT_S,
            encoding="utf-8",
        )
    except FileNotFoundError:
        raise RuntimeError("`claude` CLI not found on PATH")
    if result.returncode != 0:
        raise RuntimeError(
            f"claude CLI exit={result.returncode}: {result.stderr[:500]}"
        )
    try:
        envelope = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"claude CLI did not return JSON envelope: {result.stdout[:300]}"
        ) from exc
    text = envelope.get("result") or envelope.get("response") or ""
    u = envelope.get("usage", {}) or {}
    fresh_in = u.get("input_tokens") or 0
    cache_create = u.get("cache_creation_input_tokens") or 0
    cache_read = u.get("cache_read_input_tokens") or 0
    total_in = fresh_in + cache_create + cache_read
    usage = {
        "input_tokens_total": total_in,
        "input_tokens_fresh": fresh_in,
        "input_tokens_cache_create": cache_create,
        "input_tokens_cache_read": cache_read,
        "output_tokens": u.get("output_tokens") or 0,
        "cost_usd": envelope.get("total_cost_usd") or envelope.get("cost_usd"),
        "duration_ms": envelope.get("duration_ms"),
        "num_turns": envelope.get("num_turns"),
        "model_usage": envelope.get("modelUsage"),
    }
    return text, usage


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
    raw, usage = _call_claude(prompt)
    elapsed = (datetime.now(timezone.utc) - t0).total_seconds()
    report = _parse_judge_output(raw)
    report["judged_at"] = t0.isoformat()
    report["judge_model"] = JUDGE_MODEL
    report["elapsed_s"] = round(elapsed, 2)
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
    ap = argparse.ArgumentParser(description="Cross-model judge for parsed index quality")
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[])
    ap.add_argument("--doc-ids", dest="doc_ids_csv", default="")
    ap.add_argument("--category", help="Judge every doc in this jenis_folder")
    args = ap.parse_args()

    doc_ids = list(args.doc_ids)
    if args.doc_ids_csv:
        doc_ids.extend([x.strip() for x in args.doc_ids_csv.split(",") if x.strip()])

    targets = _resolve_targets(doc_ids, args.category)
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
        missing = (report.get("coverage") or {}).get("missing") or []
        structural = report.get("structural_issues") or []
        ocr = report.get("ocr_issues") or []
        usage = report.get("usage") or {}
        in_tok = usage.get("input_tokens_total") or 0
        out_tok = usage.get("output_tokens") or 0
        cost = usage.get("cost_usd") or 0
        print(
            f"  {verdict}  score={score}  missing={len(missing)} "
            f"structural={len(structural)} ocr={len(ocr)}  "
            f"tokens={in_tok:,} in / {out_tok:,} out  ${cost:.4f}",
            flush=True,
        )
        reports.append(report)
        _save_reports(reports)

    total_in = sum((r.get("usage") or {}).get("input_tokens_total") or 0 for r in reports)
    total_out = sum((r.get("usage") or {}).get("output_tokens") or 0 for r in reports)
    total_cost = sum((r.get("usage") or {}).get("cost_usd") or 0 for r in reports)
    print(
        f"\ntotal tokens: {total_in:,} in / {total_out:,} out  "
        f"total cost: ${total_cost:.4f}"
    )
    print(f"report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
