"""Repair OCR garbles in pasal-level leaf text via a focused LLM pass."""

import argparse
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from vectorless.ids import doc_category
from vectorless.llm import call as llm_call

INDEX_PASAL = Path("data/index_pasal")
REGISTRY_PATH = ROOT / "data/raw/registry.json"

# Length sanity bounds — reject LLM cleans that diverge wildly (likely hallucination).
LEN_MIN_RATIO = 0.85
LEN_MAX_RATIO = 1.15

# Numeric/structural markers that must survive the clean (rough — preserves count).
_MARKER_RE = re.compile(r"\((?:\d+|[a-z])\)|^\s*[a-z0-9]+\.\s+", re.MULTILINE)


PROMPT = """\
Kamu adalah corrector OCR untuk teks hukum Indonesia. Perbaiki HANYA
karakter yang jelas-jelas garbled karena OCR. JANGAN mengubah makna,
terminologi hukum, angka, atau struktur penomoran.

Pola OCR yang umum:
- Karakter berulang: "Hakirrr" -> "Hakim"
- Simbol di tengah kata: "berlal<:u" -> "berlaku", "men:alankan" -> "menjalankan"
- Digit menggantikan huruf: "ad3.lah" -> "adalah"
- Spasi salah: "keten tuan" -> "ketentuan"
- Pemisah hilang: "pengumumanyspanduk" -> "pengumuman/spanduk"
- Huruf serupa: "Nasionaf" -> "Nasional", "Pusar" -> "Pusat"

Aturan ketat:
- Jika ragu, BIARKAN kata apa adanya
- JANGAN parafrase, JANGAN tambah/hapus klausul
- Pertahankan SEMUA angka, kutipan, label "(1)", "a.", "1.", dll.
- Pertahankan struktur kalimat dan urutan paragraf

Teks asli:
{text}

Balas dalam JSON:
{{
  "cleaned": "<teks setelah perbaikan>",
  "fixes": [{{"from": "Hakirrr", "to": "Hakim"}}]
}}
"""


def _validate_clean(original: str, cleaned: str) -> str | None:
    """Return rejection reason if cleaned text fails sanity checks, else None."""
    ratio = len(cleaned) / max(1, len(original))
    if ratio < LEN_MIN_RATIO or ratio > LEN_MAX_RATIO:
        return f"length ratio {ratio:.2f} outside [{LEN_MIN_RATIO}, {LEN_MAX_RATIO}]"
    orig_markers = len(_MARKER_RE.findall(original))
    new_markers = len(_MARKER_RE.findall(cleaned))
    if new_markers < orig_markers:
        return f"marker count dropped: {orig_markers} -> {new_markers}"
    return None


def _clean_text(text: str, usage_acc: dict, lock: threading.Lock) -> tuple[str, list[dict], str | None]:
    """Return (cleaned_text, fixes_list, rejection_reason). text echoed back if rejected."""
    if not text.strip():
        return text, [], None
    result, usage = llm_call(PROMPT.format(text=text[:8000]), return_usage=True)
    with lock:
        usage_acc["input_tokens"] += usage["input_tokens"]
        usage_acc["output_tokens"] += usage["output_tokens"]
        usage_acc["calls"] += usage["calls"]
    cleaned = (result.get("cleaned") or "").strip()
    fixes = result.get("fixes") or []
    if not cleaned:
        return text, [], "empty cleaned response"
    reject = _validate_clean(text, cleaned)
    if reject:
        return text, fixes, reject
    return cleaned, fixes, None


def _walk_leaves(nodes: list[dict], todo: list[dict], force: bool) -> None:
    for node in nodes:
        if node.get("nodes"):
            _walk_leaves(node["nodes"], todo, force)
        elif node.get("text"):
            if force or not node.get("ocr_cleaned"):
                todo.append(node)


def clean_doc(doc_id: str, force: bool = False, verbose: bool = True) -> dict:
    """Clean OCR garbles in every pasal-level leaf. Returns LLM stats + counters."""
    path = INDEX_PASAL / doc_category(doc_id) / f"{doc_id}.json"
    if not path.exists():
        raise FileNotFoundError(path)

    doc = json.loads(path.read_text(encoding="utf-8"))
    if verbose:
        print(f"Cleaning OCR for {doc_id} — {path.stat().st_size//1024}KB")

    counter = {"cleaned": 0, "skipped": 0, "rejected": 0, "fixes_total": 0}
    usage_acc = {"input_tokens": 0, "output_tokens": 0, "calls": 0}
    lock = threading.Lock()

    todo: list[dict] = []
    _walk_leaves(doc.get("structure", []), todo, force)
    counter["skipped"] = sum(
        1 for _ in _iter_leaves(doc.get("structure", []))
    ) - len(todo)

    t_start = time.time()
    if todo:
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(_clean_text, leaf["text"], usage_acc, lock): leaf for leaf in todo}
            for fut in as_completed(futures):
                leaf = futures[fut]
                try:
                    cleaned, fixes, reject = fut.result()
                except Exception as e:
                    counter["rejected"] += 1
                    leaf["ocr_cleaned"] = "error"
                    if verbose:
                        print(f"  [ERR ] {leaf.get('navigation_path', '')[:80]} — {e!r}")
                    continue
                if reject:
                    counter["rejected"] += 1
                    leaf["ocr_cleaned"] = "skipped_validation"
                    if verbose:
                        print(f"  [REJ ] {leaf.get('navigation_path', '')[:80]} — {reject}")
                else:
                    leaf["text"] = cleaned
                    leaf["ocr_cleaned"] = True
                    if fixes:
                        leaf["ocr_fixes"] = fixes
                    counter["cleaned"] += 1
                    counter["fixes_total"] += len(fixes)
                    if verbose and fixes:
                        sample = ", ".join(f"{f.get('from','?')}->{f.get('to','?')}" for f in fixes[:3])
                        print(f"  [OK  {counter['cleaned']:>3}] {leaf.get('navigation_path', '')[:60]}  fixes={len(fixes)} ({sample})")

    elapsed = time.time() - t_start
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    if verbose:
        print(f"Done in {elapsed:.1f}s — cleaned={counter['cleaned']} "
              f"skipped={counter['skipped']} rejected={counter['rejected']} "
              f"fixes={counter['fixes_total']} | "
              f"{usage_acc['calls']} calls, "
              f"{usage_acc['input_tokens']+usage_acc['output_tokens']:,} tokens")
    return {
        "elapsed_s": round(elapsed, 2),
        **counter,
        "llm_calls": usage_acc["calls"],
        "input_tokens": usage_acc["input_tokens"],
        "output_tokens": usage_acc["output_tokens"],
        "total_tokens": usage_acc["input_tokens"] + usage_acc["output_tokens"],
    }


def _iter_leaves(nodes: list[dict]):
    for node in nodes:
        if node.get("nodes"):
            yield from _iter_leaves(node["nodes"])
        elif node.get("text"):
            yield node


def _registry_docs(category: str) -> list[str]:
    reg = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    target = category.upper()
    return sorted(did for did, entry in reg.items()
                  if (entry.get("jenis_folder") or "").upper() == target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.strip())
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[])
    ap.add_argument("--category", help="Process every doc in this jenis_folder")
    ap.add_argument("--force", action="store_true",
                    help="Re-clean leaves already marked ocr_cleaned")
    args = ap.parse_args()

    if args.doc_ids:
        targets = args.doc_ids
    elif args.category:
        targets = _registry_docs(args.category)
    else:
        raise SystemExit("must pass --doc-id or --category")

    print(f"Cleaning OCR for {len(targets)} doc(s)")
    totals = {"elapsed_s": 0.0, "llm_calls": 0, "total_tokens": 0,
              "fixes_total": 0, "rejected": 0}
    for did in targets:
        try:
            stats = clean_doc(did, force=args.force)
        except FileNotFoundError as e:
            print(f"  SKIP missing: {e}")
            continue
        for k in totals:
            totals[k] += stats[k]
        print()

    print(f"Total: {totals['elapsed_s']:.0f}s, {totals['llm_calls']} calls, "
          f"{totals['total_tokens']:,} tokens, fixes={totals['fixes_total']}, "
          f"rejected={totals['rejected']}")


if __name__ == "__main__":
    main()
