"""Build pasal-level index JSON with Gemini as the structure parser."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.parser._common import (  # noqa: E402
    count_pasals_in_tree,
    format_pdf_pages,
    load_pdf_pages,
    parse_llm_json,
    _normalize_keys,
)

import time  # noqa: E402

MODEL_NAME = "gemini-2.5-flash"
MAX_RETRIES = 3


def call_gemini(prompt: str, max_output_tokens: int = 65536) -> tuple[str, dict]:
    """Call the configured Gemini parser model."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    from google import genai as newgenai
    from google.genai import types as gtypes

    client = newgenai.Client(api_key=api_key)

    config_kwargs = dict(
        temperature=0.0,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )
    # Gemini 3.x rejects thinking_budget=0; 2.5 accepts it.
    if MODEL_NAME.startswith("gemini-2.5"):
        config_kwargs["thinking_config"] = gtypes.ThinkingConfig(thinking_budget=0)

    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0, "elapsed_s": 0.0}
    t0 = time.time()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config=gtypes.GenerateContentConfig(**config_kwargs),
            )
            usage["calls"] += 1
            meta = getattr(resp, "usage_metadata", None)
            if meta is not None:
                usage["input_tokens"] += getattr(meta, "prompt_token_count", 0) or 0
                usage["output_tokens"] += getattr(meta, "candidates_token_count", 0) or 0
                usage["total_tokens"] += getattr(meta, "total_token_count", 0) or 0
            text = getattr(resp, "text", None) or ""
            if text.strip():
                usage["elapsed_s"] = round(time.time() - t0, 3)
                return text, usage
        except Exception as exc:
            if attempt == MAX_RETRIES:
                raise
            print(f"  retry {attempt}/{MAX_RETRIES}: {type(exc).__name__}: {exc}", flush=True)
    raise RuntimeError("LLM returned empty response after retries")
from vectorless.indexing.metadata import build_metadata  # noqa: E402

INDEX_PASAL = REPO_ROOT / "data" / "index_pasal"
BACKUP_DIR = REPO_ROOT / "data" / "index_pasal_pre_llm_parse"
AUDIT_LOG = REPO_ROOT / "data" / "llm_parse_log.json"

MAX_WHOLE_DOC_PASALS = 35
MAX_WHOLE_DOC_INPUT_CHARS = 120_000
PAGES_PER_CHUNK = 10
CHUNK_OVERLAP_PAGES = 4
PARALLEL_WORKERS = 4


PROMPT_TEMPLATE = """\
You extract the hierarchical structure of an Indonesian legal document
from raw PDF text. Your output is the CANONICAL parse — it replaces a
fragile regex-based parser.

=== DOCUMENT METADATA ===
doc_id       : {doc_id}
judul        : {judul}
is_perubahan : {is_perubahan}
{page_range_hint}
=== RAW PDF TEXT (blocks tagged [page=N x=col-left y=top]) ===
Two-column gazette: left column x~50-280, right column x~300-550.
Use coords to reconstruct body attribution when headings and body are in
different blocks or columns.

{pdf_text}

=== HIERARCHY ===
Standard doc:  BAB > Bagian > Paragraf > Pasal > Ayat > Huruf > Angka
Amendment:     Pasal I (roman root, contains all amendment instructions)
                 └── Angka 1, 2, ...  (each = one amendment operation)
                      └── nested Pasal X when Angka inserts/modifies a pasal
               Pasal II (optional closing: 1-5 small Angka re: effective date)

=== AMENDMENT RULES (is_perubahan=true) ===
- Angka numbering under each Pasal Roman RESTARTS at 1.
- Do NOT split one Pasal I into Pasal I+II just because of a page break.
  Split ONLY when PDF literally has "Pasal II" as a heading — usually
  "Pada saat Peraturan Menteri ini mulai berlaku, ...".
- Recover ALL Angka 1..N from the PDF. Do not skip any numbers.

=== OUTPUT SCHEMA (JSON only, no markdown fences, no prose) ===

OUTPUT GRANULARITY: stop at Pasal level. Each Pasal is a LEAF with its
full body text as a single "text" string — including any "(1) ..." ayat
markers, "a. ..." huruf markers, "1. ..." angka markers that appear
verbatim in the PDF body. A separate re-split pass deterministically
breaks the body into sub-structure, so DO NOT nest Ayat/Huruf/Angka
inside Pasal nodes. Only BAB and Bagian can wrap Pasal nodes.

{{
  "structure": [
    {{
      "title": "BAB I - KETENTUAN UMUM",
      "node_id": "bab_1",
      "nodes": [
        {{
          "title": "Pasal 1",
          "node_id": "pasal_1",
          "text": "Dalam Peraturan ini yang dimaksud dengan:\\n1. Penyelenggara adalah ...\\n2. Pemodal adalah ..."
        }},
        {{
          "title": "Pasal 2",
          "node_id": "pasal_2",
          "text": "(1) Kegiatan Layanan Urun Dana wajib dilakukan oleh Penyelenggara.\\n(2) Penyelenggara sebagaimana dimaksud pada ayat (1) wajib berbadan hukum."
        }}
      ]
    }}
  ]
}}

For amendment docs (is_perubahan=true): emit "Pasal I" / "Pasal II" as
CONTAINERS with nested Angka children. Each Angka represents ONE
amendment instruction. When an Angka says "Ketentuan Pasal X diubah",
"Di antara Pasal X dan Pasal Y disisipkan Pasal XA", or similar, the
NEW Pasal MUST be emitted as a nested child of that Angka, NOT as a
flat text blob. Preserve the new Pasal's own structure (Ayat, Huruf)
as further nested children.

Example amendment structure (note Pasal I has BOTH text preamble AND nodes,
and prior-amendment references a./b./c. emitted as Huruf siblings alongside
Angka siblings):
{{
  "structure": [
    {{
      "title": "Pasal I",
      "node_id": "pasal_I",
      "text": "Undang-Undang Nomor 19 Tahun 2003 tentang Badan Usaha Milik Negara (Lembaran Negara Republik Indonesia Tahun 2003 Nomor 70, Tambahan Lembaran Negara Republik Indonesia Nomor 4297) yang telah beberapa kali diubah dengan Undang-Undang:",
      "nodes": [
        {{
          "title": "Pasal I Huruf a",
          "node_id": "pasal_I_huruf_a",
          "text": "Nomor 11 Tahun 2020 tentang Cipta Kerja (Lembaran Negara Republik Indonesia Tahun 2020 Nomor 245, Tambahan Lembaran Negara Republik Indonesia Nomor 6573);"
        }},
        {{
          "title": "Pasal I Huruf b",
          "node_id": "pasal_I_huruf_b",
          "text": "Nomor 6 Tahun 2023 tentang Penetapan Peraturan Pemerintah Pengganti Undang-Undang ...;"
        }},
        {{
          "title": "Pasal I Huruf c",
          "node_id": "pasal_I_huruf_c",
          "text": "Nomor 1 Tahun 2025 tentang Perubahan Ketiga atas Undang-Undang Nomor 19 Tahun 2003 ...;"
        }},
        {{
          "title": "Pasal I Angka 1",
          "node_id": "pasal_I_angka_1",
          "text": "Ketentuan Pasal 1 diubah sehingga berbunyi sebagai berikut:",
          "nodes": [
            {{
              "title": "Pasal 1",
              "node_id": "pasal_I_angka_1_pasal_1",
              "text": "Dalam Peraturan ini yang dimaksud dengan:\\n1. Penyelenggara adalah ...\\n2. Pemodal adalah ..."
            }}
          ]
        }},
        {{
          "title": "Pasal I Angka 2",
          "node_id": "pasal_I_angka_2",
          "text": "Di antara Pasal 3 dan Pasal 4 disisipkan 1 (satu) pasal, yakni Pasal 3A sehingga berbunyi sebagai berikut:",
          "nodes": [
            {{
              "title": "Pasal 3A",
              "node_id": "pasal_I_angka_2_pasal_3A",
              "text": "(1) Aset Keuangan Digital terdiri atas:\\na. Aset Kripto; dan\\nb. Aset Keuangan Digital lainnya."
            }}
          ]
        }},
        {{
          "title": "Pasal I Angka 3",
          "node_id": "pasal_I_angka_3",
          "text": "Pasal 5 dihapus."
        }}
      ]
    }},
    {{
      "title": "Pasal II",
      "node_id": "pasal_II",
      "text": "Peraturan ini mulai berlaku pada tanggal diundangkan."
    }}
  ]
}}

Rules for amendment nesting:
- Pasal I / Pasal II = CONTAINER (Roman numerals). Do NOT put amendment
  body as flat text on the Pasal Roman.
- Pasal I typically has a PREAMBLE paragraph before the first Angka, e.g.:
    "Undang-Undang Nomor X Tahun Y tentang ABC (Lembaran Negara ...),
     yang telah beberapa kali diubah dengan Undang-Undang:
     a. Nomor ... Tahun ... tentang ...;
     b. Nomor ... Tahun ... tentang ...;
     diubah sebagai berikut:"
  This preamble belongs to Pasal I ITSELF — emit as `"text"` on the Pasal I
  container node (alongside its "nodes" array of Angka children). DO NOT
  stuff it into Angka 1's intro — Angka 1 starts at the literal "1." marker.
- If the preamble contains a lettered list (a./b./c. listing prior amending
  laws), emit those as HURUF CHILDREN of Pasal I, sibling to the Angka
  children. Sequence: first Huruf a..c siblings (prior amendment refs),
  then Angka 1..N siblings (amendment instructions). Pasal I `text` then
  holds only the lead-in "UU Nomor X Tahun Y tentang ABC ..., yang telah
  beberapa kali diubah dengan Undang-Undang:" and the trailing transition
  "diubah sebagai berikut:" can be appended to the text or omitted
  (it's just a structural connector).
- Each Angka = child of Pasal Roman. Title format "Pasal I Angka N".
- Angka that inserts/modifies a Pasal = CONTAINER with the new Pasal as
  single child. The Angka's OWN text is just the instruction sentence
  ("Ketentuan Pasal X diubah...", "Di antara X dan Y disisipkan Pasal Z...").
- Angka that deletes ("Pasal 5 dihapus.") = LEAF with short text.

CRITICAL — nested amended Pasal numbering:
- The nested new Pasal inside an Angka is ALWAYS ARABIC: "Pasal 1",
  "Pasal 2", "Pasal 3A", "Pasal 568A" — NEVER Roman ("Pasal I", "Pasal II").
- Even if the PDF's visual rendering makes the arabic "1" look like Roman "I",
  USE ARABIC. Check the instruction text: "Ketentuan Pasal 1 diubah..." means
  nested title = "Pasal 1" (arabic), NOT "Pasal I".
- node_id for nested: pasal_I_angka_{{N}}_pasal_{{arabic}}
  e.g. pasal_I_angka_1_pasal_1, pasal_I_angka_2_pasal_3A.
- Do NOT reuse the outer Roman numeral as the nested pasal's number.

CRITICAL — nested Pasal body: keep ENTIRELY FLAT.
- Nested new Pasal body = ONE flat "text" string containing ALL markers
  inline: "(1) ...\\na. ...\\nb. ...\\n1. ...\\n2. ...", verbatim from PDF.
- DO NOT emit any children under nested Pasal. No Ayat, no Huruf, no Angka
  sub-nodes. The deterministic re-split pass handles Ayat/Huruf/Angka
  splitting based on inline markers.
- This applies REGARDLESS of depth: if nested Pasal 3F has Ayat (1), (2),
  Huruf a-h under (2), Angka 1-9 under h — ALL of that stays in ONE text
  string on Pasal 3F. Re-split produces the hierarchy deterministically.
- Example (what to emit, regardless of apparent nesting in PDF):
    {{ "title": "Pasal 3F", "node_id": "pasal_I_angka_8_pasal_3F",
       "text": "(1) Badan bertugas ...\\n(2) Dalam melaksanakan tugas ..., Badan berwenang:\\na. mengelola dividen ...\\nb. menyetujui ...\\nh. menetapkan pedoman/kebijakan strategis dalam bidang:\\n1. akuntansi ...\\n2. pengembangan ...\\n9. program ESG." }}
- This avoids LLM judgment calls on ambiguous deep-nesting layouts and
  keeps the output consistent. Only top-level containers (BAB, Bagian,
  Paragraf, Pasal I/II, Pasal I Angka N) are emitted as structural nodes.

- DO include every Angka number from the PDF; skip none.

NODE_ID CONVENTIONS (lowercase, underscore-separated):
- bab_{{arabic}}                          e.g. bab_1, bab_12
- bab_{{A}}_bagian_{{N}}                  e.g. bab_1_bagian_2
- {{parent}}_paragraf_{{N}}               e.g. bab_1_bagian_2_paragraf_3
- pasal_{{N}}                             e.g. pasal_3, pasal_5A
Amendment:
- pasal_I, pasal_II                       (uppercase roman for amendment)

TITLE CONVENTIONS:
- BAB: "BAB {{roman}} - {{name}}"      e.g. "BAB I - KETENTUAN UMUM"
- Bagian: "Bagian {{ordinal}} - {{name}}" e.g. "Bagian Kesatu - Kegiatan Usaha"
- Paragraf: "Paragraf {{N}} - {{name}}"
- Pasal: "Pasal {{N}}"                 e.g. "Pasal 3", "Pasal 5A"
- Amendment Pasal: "Pasal I" / "Pasal II"

=== CONTENT RULES ===

1. Body text preserves PDF WORD ORDER, STRUCTURE, NUMBERS, and MEANING
   exactly — but ACTIVELY REPAIR OCR corruption. Downstream retrieval
   (BM25) depends on clean Indonesian tokens. A garbled token that a
   fluent Indonesian legal reader would recognize MUST be fixed; leaving
   it verbatim means that passage becomes unsearchable.

   Standard: fix when a fluent reader would agree on the intended word.
   Leave verbatim only when the intended word is genuinely ambiguous
   (multiple valid Indonesian words could equally fit the context).
   "I am not 100% sure" is NOT the bar — near-total agreement is.

   REPAIR THESE (produce searchable Indonesian words without changing meaning):
   - Mid-word character corruption of any length (1-5 chars):
       "perenczrna.an"       → "perencanaan"
       "terfi.ang"           → "tentang"
       "kebiiakan"           → "kebijakan"
       "pemeriniah"          → "pemerintah"
       "Meneapkan"           → "Menetapkan"
       "Undang-tlndang"      → "Undang-Undang"
       "menyeleng.gartrkan"  → "menyelenggarakan"
       "Kera.jaan"           → "Kerajaan"  (proper nouns included)
       "INOONESIA"           → "INDONESIA"
   - Digit-for-letter in a word:
       "Pasa1" / "PasaT" / "PasaI"  → "Pasal"
       "5684" → "568A"  (only when adjacent Pasal sequence confirms)
   - Spurious punctuation inside words:
       "pe.rencanaan"        → "perencanaan"
       "menyelenggarak,an"   → "menyelenggarakan"
   - Missing spaces at column/page breaks:
       "KecamatanWonggeduku" → "Kecamatan Wonggeduku"
       "diaturdalam"         → "diatur dalam"
   - Collapse redundant whitespace; de-duplicate page-break headings.

   HARD FORBIDDEN (invalidates the parse):
   - Changing any digit or number, even if it looks wrong. "(3)" stays
     "(3)"; "Pasal 12" stays "Pasal 12"; "Tahun 2022" stays "Tahun 2022".
   - Changing legal terminology: "wajib" stays "wajib" (never "harus"),
     "memprioritaskan" stays "memprioritaskan", etc.
   - Paraphrasing, reordering words, translating, or summarizing.
   - Adding clauses not present in the PDF.
   - Guessing when genuinely ambiguous — leave verbatim.

2. Keep Pasal body as ONE flat "text" string. Preserve "(1) ...", "a. ..."
   and "1. ..." markers inline. A deterministic re-split pass handles
   Ayat/Huruf/Angka splitting. DO NOT pre-split them.

   This is the single most important rule. A Pasal is EITHER a leaf with
   flat text OR a container with children — NEVER both. If the body
   contains any "(1)", "(2)", "a.", "b.", "1.", "2." markers, they MUST
   all sit inline inside one "text" string. Do not extract any of them
   into child nodes, even when the PDF visually indents them.

   BAD (hybrid — ayat in text, huruf as children):
   {{
     "title": "Pasal 7",
     "text": "(1) Tugas pokok ...\\n(2) Tugas pokok ... dilakukan dengan:",
     "nodes": [
       {{"title": "Pasal 7 Huruf a", "text": "operasi militer untuk perang;"}},
       {{"title": "Pasal 7 Huruf b", "text": "operasi militer selain perang ..."}}
     ]
   }}

   GOOD (flat — everything inline, no children):
   {{
     "title": "Pasal 7",
     "text": "(1) Tugas pokok ...\\n(2) Tugas pokok ... dilakukan dengan:\\na. operasi militer untuk perang;\\nb. operasi militer selain perang ..., yaitu untuk:\\n1. mengatasi gerakan separatis ...\\n2. mengatasi pemberontakan ..."
   }}

3. Drop noise including OCR-garbled variants — page numbers ("- 5 -",
   "-2L-"), footers ("SK No 12345"), repeated headers ("PRESIDEN
   REPUBLIK INDONESIA", "MENTERI ..."), signing blocks. Garbled forms
   like "REFI.IBI.IK INOONESIA" or "FRESIDEN" are the SAME noise —
   strip them entirely, do not try to repair into body text.

4. Skip preamble entirely (Pembukaan, Menimbang, Mengingat, Menetapkan).
   Output begins at the first body section (first BAB or first Pasal).

5. Skip Penjelasan section entirely (parsed in separate pass).

6. Pasal sequence MUST be monotonic and contiguous. Every Pasal N in the
   body appears exactly once. Never skip, re-order, or duplicate. If a
   pasal is too corrupt to read in PDF, still include its header with
   text "[tidak terbaca dari PDF]" — do not omit its existence.

7. BAB membership: each Pasal belongs to the MOST RECENT BAB heading
   preceding it in PDF reading order. Use x/y coords to disambiguate
   multi-column layouts.

   CRITICAL — list continuation across page boundaries. A Pasal body
   often contains a list "a., b., c., ...h." that spans 2+ pages. The
   list continues UNTIL either the next Pasal heading, a BAB/Bagian/
   Paragraf heading, or the list explicitly ends. A page break alone
   does NOT end a Pasal body or its list — if page N+1 opens with
   "d. Ekuitas;\n e. ..." it is the CONTINUATION of Pasal 2's list,
   not a new node. Capture all items a through the last before the
   next heading.

   Example of the WRONG behavior to avoid:
     Page 4 ends: "c. kecukupan investasi;"
     Page 5 starts: "d. Ekuitas; e. Dana Jaminan; ...; h. ketentuan lain."
     followed by: "Bagian Kedua / Pasal 3 ..."
   → CORRECT output: Pasal 2 text = "...a. ...; b. ...; c. ...; d. ...; e. ...; f. ...; g. ...; h. ketentuan lain."
   → WRONG: Pasal 2 text = "...c. kecukupan investasi;" (missing d-h, truncated at page break).

8. No empty "nodes" arrays. Omit "nodes" if a node has no children.
   Omit "text" if a node has children but no intro; keep "text" as intro
   when both present.

9. Output ONLY valid JSON starting with {{ and ending with }}. No markdown
   code fences. No trailing explanation. One top-level key: "structure".
"""

_PASAL_TITLE_RE = re.compile(r"^Pasal\s+\d+[A-Z]?$")


def iter_nodes(structure: list[dict]):
    """Yield every node in the tree (depth-first)."""
    for n in structure:
        yield n
        if n.get("nodes"):
            yield from iter_nodes(n["nodes"])


def collect_pasal_numbers(structure: list[dict]) -> list[str]:
    """Return list of Pasal numbers (arabic only) in tree order."""
    out = []
    for n in iter_nodes(structure):
        t = (n.get("title") or "").strip()
        m = re.match(r"^Pasal\s+(\d+[A-Z]?)$", t)
        if m:
            out.append(m.group(1))
    return out


def _sanitize_node_id(s: str) -> str:
    """Enforce lowercase + underscores only; preserve Roman pasal (I, II, III)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", s).strip("_")


def _canonical_title_from_node_id(node_id: str, original_title: str) -> str:
    """Rebuild canonical pasal-family title from node_id.

    node_id is LLM-consistent and structural; the title may carry verbatim OCR
    artifacts. For nested amendment ids (pasal_I_angka_N_pasal_3A), the deepest
    `pasal` segment wins — so the title reflects the nested Pasal, not the
    Roman container. BAB/Bagian/Paragraf titles pass through unchanged.
    """
    parts = node_id.split("_")
    last_pasal_idx = -1
    for i, seg in enumerate(parts):
        if seg == "pasal" and i + 1 < len(parts):
            last_pasal_idx = i
    if last_pasal_idx == -1:
        return original_title
    tail = parts[last_pasal_idx:]
    out = []
    i = 0
    while i < len(tail):
        seg = tail[i]
        if seg == "pasal":
            if i + 1 < len(tail):
                num = tail[i + 1]
                if re.match(r"^[IVX]+$", num):
                    out.append(f"Pasal {num}")
                elif re.match(r"^\d+[A-Z]?$", num, re.IGNORECASE):
                    m = re.match(r"^(\d+)([A-Za-z]?)$", num)
                    if m:
                        n, suf = m.group(1), m.group(2).upper()
                        out.append(f"Pasal {n}{suf}")
                    else:
                        out.append(f"Pasal {num}")
                else:
                    out.append(f"Pasal {num}")
                i += 2
                continue
        elif seg == "ayat":
            if i + 1 < len(tail):
                out.append(f"Ayat ({tail[i + 1]})")
                i += 2
                continue
        elif seg == "huruf":
            if i + 1 < len(tail):
                out.append(f"Huruf {tail[i + 1]}")
                i += 2
                continue
        elif seg == "angka":
            if i + 1 < len(tail):
                out.append(f"Angka {tail[i + 1]}")
                i += 2
                continue
        i += 1
    canonical = " ".join(out)
    return canonical if canonical else original_title


def normalize_pasal_titles_in_tree(structure: list[dict]) -> int:
    """Rebuild pasal-family titles from node_id; return count modified."""
    count = 0
    for node in structure:
        title = node.get("title", "")
        nid = node.get("node_id", "")
        if nid and title.startswith("Pasal "):
            new = _canonical_title_from_node_id(nid, title)
            if new != title:
                node["title"] = new
                count += 1
        if node.get("nodes"):
            count += normalize_pasal_titles_in_tree(node["nodes"])
    return count


def assign_readable_node_ids(
    structure: list[dict], ancestor_id: str = ""
) -> None:
    """Keep conforming node_ids; re-derive from title + ancestor when missing/malformed."""
    for node in structure:
        title = (node.get("title") or "").strip()
        nid = (node.get("node_id") or "").strip()
        if not nid or not re.match(r"^[a-zA-Z0-9_]+$", nid):
            nid = _derive_node_id_from_title(title, ancestor_id)
        node["node_id"] = nid
        if node.get("nodes"):
            assign_readable_node_ids(node["nodes"], ancestor_id=nid)


def _derive_node_id_from_title(title: str, ancestor_id: str) -> str:
    """Fallback derivation when LLM output is missing/malformed node_id."""
    t = title.lower()
    m = re.match(r"^pasal\s+([ivx]+)(?:\s+angka\s+(\d+))?(?:\s+pasal\s+(\d+[a-z]?))?", t)
    if m:
        parts = [f"pasal_{m.group(1).upper()}"]
        if m.group(2):
            parts.append(f"angka_{m.group(2)}")
        if m.group(3):
            parts.append(f"pasal_{m.group(3).upper()}")
        return "_".join(parts)
    m = re.match(r"^bab\s+([ivxlc]+)", t)
    if m:
        from vectorless.indexing.parser import roman_to_int
        n = roman_to_int(m.group(1).upper())
        return f"bab_{n if n else m.group(1)}"
    m = re.match(r"^bagian\s+(\w+)", t)
    if m:
        return f"{ancestor_id}_bagian_{m.group(1).lower()}" if ancestor_id else f"bagian_{m.group(1).lower()}"
    m = re.match(r"^paragraf\s+(\d+)", t)
    if m:
        return f"{ancestor_id}_paragraf_{m.group(1)}" if ancestor_id else f"paragraf_{m.group(1)}"
    m = re.match(
        r"^pasal\s+(\d+[a-z]?)(?:\s+ayat\s+\((\d+)\))?(?:\s+huruf\s+([a-z]+))?(?:\s+angka\s+(\d+))?",
        t,
    )
    if m:
        parts = [f"pasal_{m.group(1).upper() if m.group(1) and not m.group(1).isdigit() else m.group(1)}"]
        if m.group(2):
            parts.append(f"ayat_{m.group(2)}")
        if m.group(3):
            parts.append(f"huruf_{m.group(3)}")
        if m.group(4):
            parts.append(f"angka_{m.group(4)}")
        return "_".join(parts)
    return _sanitize_node_id(title)[:60].lower() or "node"


def build_navigation_paths(
    structure: list[dict], ancestors: list[str] | None = None
) -> None:
    """Set navigation_path on every node using title ancestry."""
    ancestors = ancestors or []
    for node in structure:
        title = (node.get("title") or "").strip()
        path = ancestors + [title]
        node["navigation_path"] = " > ".join(path)
        if node.get("nodes"):
            build_navigation_paths(node["nodes"], path)


# Patterns for splitting a parent's penjelasan text into per-child slices.
# Tolerant to OCR quirks like "Hurufb" without space.
_PENJ_AYAT_HEADER_RE = re.compile(r"(?:^|\n)\s*Ayat\s*\(?(\d+)\)?\s*\n?", re.IGNORECASE)
_PENJ_HURUF_HEADER_RE = re.compile(r"(?:^|\n)\s*Huruf\s*([a-z])\b", re.IGNORECASE)
_PENJ_ANGKA_HEADER_RE = re.compile(r"(?:^|\n)\s*Angka\s+(\d+)\b", re.IGNORECASE)


def _split_penj_by_marker(
    parent_penj: str, regex: re.Pattern
) -> tuple[str, dict[str, str]]:
    """Split penjelasan at headers; return (lead_text, {label: slice})."""
    matches = list(regex.finditer(parent_penj))
    if not matches:
        return parent_penj, {}
    lead = parent_penj[: matches[0].start()].strip()
    slices: dict[str, str] = {}
    for i, m in enumerate(matches):
        label = m.group(1).lower()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(parent_penj)
        slices[label] = parent_penj[m.end() : end].strip()
    return lead, slices


def _child_key(title: str) -> tuple[str, str] | None:
    """Return (kind, label) for the DEEPEST sub-element in a child title."""
    t = title.strip()
    for kind, rex in (
        ("angka", r"Angka\s+(\d+)\s*$"),
        ("huruf", r"Huruf\s+([a-z])\s*$"),
        ("ayat", r"Ayat\s*\((\d+)\)\s*$"),
    ):
        m = re.search(rex, t, re.IGNORECASE)
        if m:
            return kind, m.group(1).lower()
    return None


def distribute_penjelasan_to_tree(structure: list[dict]) -> None:
    """Split each parent's penjelasan at child (Ayat/Huruf/Angka) headers.

    Fallback when no markers match: children inherit the full parent penjelasan.
    """
    for node in structure:
        children = node.get("nodes") or []
        if children and node.get("penjelasan"):
            parent_penj = node["penjelasan"]
            kind = None
            for c in children:
                k = _child_key(c.get("title", ""))
                if k:
                    kind = k[0]
                    break
            if kind == "ayat":
                lead, slices = _split_penj_by_marker(parent_penj, _PENJ_AYAT_HEADER_RE)
            elif kind == "huruf":
                lead, slices = _split_penj_by_marker(parent_penj, _PENJ_HURUF_HEADER_RE)
            elif kind == "angka":
                lead, slices = _split_penj_by_marker(parent_penj, _PENJ_ANGKA_HEADER_RE)
            else:
                lead, slices = parent_penj, {}
            # Lead text (often "Cukup jelas." intro) stays on the parent.
            node["penjelasan"] = lead if lead else parent_penj
            # Child gets its slice only when labels match — do not blind-copy parent.
            for c in children:
                key = _child_key(c.get("title", ""))
                if key and key[1] in slices and slices[key[1]]:
                    c["penjelasan"] = slices[key[1]]
        if children:
            distribute_penjelasan_to_tree(children)


def attach_penjelasan(
    structure: list[dict], penj_pasal_map: dict[str, str]
) -> int:
    """Attach per-Pasal penjelasan to Pasal nodes; return count matched."""
    matched = 0
    for node in iter_nodes(structure):
        title = (node.get("title") or "").strip()
        m = re.match(r"^Pasal\s+(\d+[A-Z]?)$", title)
        if not m:
            continue
        num = m.group(1).upper()
        if num in penj_pasal_map:
            node["penjelasan"] = penj_pasal_map[num]
            matched += 1
    return matched


def backfill_page_indices(
    structure: list[dict], pages: list[dict], doc_total_pages: int
) -> None:
    """Derive start_index / end_index for every node by matching title or text
    against PDF page raw_text. Missing defaults: 1..doc_total_pages."""
    page_text = {p["page_num"]: p.get("raw_text", "") for p in pages}

    def _find_title_page(title: str) -> int | None:
        if not title:
            return None
        needle = title.strip()
        needle_norm = re.sub(r"\s+", " ", needle)
        for n in sorted(page_text):
            t = re.sub(r"\s+", " ", page_text[n])
            if needle_norm and needle_norm[:40] in t:
                return n
        return None

    def _assign(node, default_start=1, default_end=doc_total_pages):
        title = node.get("title", "")
        start = _find_title_page(title) or default_start
        node["start_index"] = start
        children = node.get("nodes", []) or []
        if children:
            for i, child in enumerate(children):
                next_sib = children[i + 1] if i + 1 < len(children) else None
                next_start = _find_title_page(next_sib.get("title", "")) if next_sib else None
                child_default_end = (next_start - 1) if next_start else default_end
                _assign(child, default_start=start, default_end=child_default_end)
            node["end_index"] = children[-1].get("end_index", default_end)
        else:
            node["end_index"] = default_end

    for i, top in enumerate(structure):
        next_top = structure[i + 1] if i + 1 < len(structure) else None
        next_start = _find_title_page(next_top.get("title", "")) if next_top else None
        default_end = (next_start - 1) if next_start else doc_total_pages
        _assign(top, default_start=1, default_end=default_end)

_INLINE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*(?:\(\d+\)|[a-z]\.|\d+\.)\s",
    re.MULTILINE,
)


def _find_hybrid_nodes(structure: list[dict]) -> list[str]:
    """Return node_ids where text has inline ayat/huruf/angka markers AND
    the node also has children — indicates the LLM partially pre-split
    sub-structure, which breaks the deterministic re-split pass.

    Requires >=2 marker hits to avoid flagging amendment Angka N nodes
    whose text legitimately begins with "N. Ketentuan Pasal X diubah ...".
    """
    hybrids: list[str] = []
    for n in iter_nodes(structure):
        if not n.get("nodes"):
            continue
        text = n.get("text") or ""
        if len(_INLINE_MARKER_RE.findall(text)) >= 2:
            hybrids.append(n.get("node_id") or n.get("title") or "?")
    return hybrids


def validate_parse(
    output: dict, pdf_pasal_numbers: set[str], is_perubahan: bool = False
) -> tuple[bool, list[str]]:
    """Sanity-check LLM parse output; return (ok, errors).

    Amendment docs skip arabic pasal-count checks (output is Pasal Roman roots);
    only non-empty structure + at least one Pasal Roman is required.
    """
    errors: list[str] = []
    if not isinstance(output, dict) or "structure" not in output:
        return False, ["missing 'structure' key"]
    structure = output["structure"]
    if not isinstance(structure, list) or not structure:
        return False, ["'structure' must be a non-empty list"]

    hybrids = _find_hybrid_nodes(structure)
    if hybrids:
        errors.append(
            f"hybrid nodes (text has inline markers AND has children, "
            f"breaks re-split): {hybrids[:20]}"
        )

    if is_perubahan:
        romans = [
            n for n in iter_nodes(structure)
            if re.match(r"^Pasal\s+[IVX]+$", (n.get("title") or "").strip())
        ]
        if not romans:
            errors.append(
                "amendment doc has no Pasal Roman root (expected 'Pasal I' at least)"
            )
        for rn in romans:
            txt = (rn.get("text") or "").strip()
            has_children = bool(rn.get("nodes"))
            if not txt and not has_children:
                errors.append(
                    f"amendment {rn.get('title')} has no text nor children"
                )
        return len(errors) == 0, errors

    out_numbers = collect_pasal_numbers(structure)
    out_count = len(out_numbers)

    pdf_count = len(pdf_pasal_numbers)
    if pdf_count > 0:
        lower = max(1, int(pdf_count * 0.8))
        upper = int(pdf_count * 1.2) + 5
        if out_count < lower:
            errors.append(
                f"pasal count too low: {out_count} < {lower} (pdf regex={pdf_count})"
            )
        if out_count > upper:
            errors.append(
                f"pasal count too high: {out_count} > {upper} (pdf regex={pdf_count})"
            )

    dupes = {n for n in out_numbers if out_numbers.count(n) > 1}
    if dupes:
        errors.append(f"duplicate pasals: {sorted(dupes)}")

    out_set = set(out_numbers)
    missing = pdf_pasal_numbers - out_set
    if missing:
        errors.append(
            f"pasals missing from LLM output: {sorted(missing)[:20]}"
        )
    extra = out_set - pdf_pasal_numbers
    # Tolerate a few extras: LLM sees full PDF context and can recover pasals
    # the simple regex missed on tricky layouts. Flag only when >15% are extras.
    if extra and len(extra) > max(3, len(out_numbers) * 0.15):
        errors.append(
            f"too many pasals not in PDF regex (possible hallucination): "
            f"{sorted(extra)[:20]}"
        )

    empty_leaves: list[str] = []
    for n in iter_nodes(structure):
        if not n.get("nodes"):
            t = (n.get("text") or "").strip()
            if not t:
                empty_leaves.append(n.get("node_id", "?"))
    if len(empty_leaves) > max(5, out_count // 10):
        errors.append(
            f"too many empty leaves ({len(empty_leaves)}): {empty_leaves[:10]}"
        )

    return len(errors) == 0, errors


# --- main parse flow -----------------------------------------------------


def build_prompt(
    doc_id: str,
    judul: str,
    is_perubahan: bool,
    pdf_text: str,
    start_page: int | None = None,
    end_page: int | None = None,
    expected_pasals: list[str] | None = None,
) -> str:
    """Render the parse prompt for one doc (or page chunk)."""
    hint_lines: list[str] = []
    if start_page is not None and end_page is not None:
        hint_lines.append(f"page range   : {start_page}-{end_page}")
    if expected_pasals:
        hint_lines.append(
            "expected pasals in this range (from PDF regex scan): "
            + ", ".join(expected_pasals)
        )
        hint_lines.append(
            "REQUIRED: every Pasal listed above MUST appear in your output "
            "tree, even if its body is short or incomplete. Do not drop any."
        )
        hint_lines.append(
            "The page range is a focus zone, NOT a hard cutoff: if a Pasal "
            "body continues onto pages just past your range (visible in the "
            "included PDF text as overlap), capture the ENTIRE body verbatim. "
            "An adjacent chunk will also see these pages — the merger keeps "
            "the most complete copy, so emit fully rather than truncate."
        )
    page_range_hint = ("\n".join(hint_lines) + "\n") if hint_lines else ""
    return PROMPT_TEMPLATE.format(
        doc_id=doc_id,
        judul=judul,
        is_perubahan=str(is_perubahan).lower(),
        page_range_hint=page_range_hint,
        pdf_text=pdf_text,
    )


_PASAL_HEADING_RE = re.compile(
    r"(?m)^[\s\t]*[Pp]asa[l1]\s+(\d+[A-Z]?)\s*[']?\s*$"
)


def _is_plausible_pasal_num(num: str) -> bool:
    """Reject OCR noise: valid Pasal suffixes are A-H; O/I/L are usually misread 0/1."""
    m = re.match(r"^(\d+)([A-Z]*)$", num)
    if not m:
        return False
    suffix = m.group(2)
    if not suffix:
        return True
    return all(c in "ABCDEFGH" for c in suffix)


def _pasal_numbers_in_page_range(
    pages: list[dict], start: int, end: int
) -> set[str]:
    """Return set of Pasal numbers (e.g. {'1', '3A'}) found in the page range."""
    nums: set[str] = set()
    for p in pages:
        n = p["page_num"]
        if n < start or n > end:
            continue
        for block in p.get("blocks", []):
            for line in block.get("text", "").splitlines():
                m = re.match(r"\s*[Pp]asa[l1]\s+(\d+[A-Z]?)\s*[']?\s*$", line)
                if m:
                    num = m.group(1).upper()
                    if _is_plausible_pasal_num(num):
                        nums.add(num)
    return nums


def _pasals_in_page_range(
    pages: list[dict], start: int, end: int
) -> list[str]:
    """Return list of 'Pasal N' labels in numeric order for the given range."""
    nums = _pasal_numbers_in_page_range(pages, start, end)
    def _key(x: str) -> tuple[int, str]:
        m = re.match(r"(\d+)([A-Z]?)", x)
        return (int(m.group(1)), m.group(2) or "") if m else (10**9, x)
    return [f"Pasal {n}" for n in sorted(nums, key=_key)]


def _run_llm(prompt: str) -> tuple[dict | None, dict, str | None]:
    """Single Gemini call + JSON parse. Returns (obj, usage, error)."""
    try:
        raw, usage = call_gemini(prompt)
    except Exception as exc:
        return None, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0, "elapsed_s": 0.0}, f"llm call: {exc}"
    try:
        obj = parse_llm_json(raw)
    except Exception as exc:
        return None, usage, f"json parse: {exc} (preview: {raw[:200]!r})"
    _normalize_keys(obj)
    return obj, usage, None


_CONTAINER_TITLE_RE = re.compile(r"^(BAB|Bagian|Paragraf)\b", re.IGNORECASE)


def _is_container(node: dict) -> bool:
    t = (node.get("title") or "").strip()
    return bool(_CONTAINER_TITLE_RE.match(t)) and not _PASAL_TITLE_RE.match(t)


def _merge_chunk_structures(chunks: list[list[dict]]) -> list[dict]:
    """Merge chunk outputs into one tree, deduping Pasals under their container path.

    BAB > Bagian > Paragraf containers are matched across chunks by normalized
    title (node_id is not stable across chunks when a BAB header falls outside
    the chunk's page range).
    """
    if not chunks:
        return []

    container_stub: dict[tuple, dict] = {}
    container_order: list[tuple] = []

    pasal_best: dict[str, dict] = {}
    pasal_to_path: dict[str, tuple] = {}

    def _container_depth(title: str) -> int | None:
        t = title.strip()
        if t.upper().startswith("BAB "):
            return 0
        if t.lower().startswith("bagian "):
            return 1
        if t.lower().startswith("paragraf "):
            return 2
        return None

    def _norm_title(t: str) -> str:
        """Collapse dash/whitespace variants so near-identical titles share one key."""
        t = t.strip()
        t = re.sub(r"\s*[-–—]\s*", " ", t)
        t = re.sub(r"\s+", " ", t).lower()
        return t

    def _container_key(n: dict) -> str:
        return _norm_title(n.get("title") or "")

    current_path: list[str] = []  # display titles
    current_keys: list[str] = []  # normalized dedup keys

    def _walk(nodes: list[dict]):
        nonlocal current_path, current_keys
        for n in nodes:
            t = (n.get("title") or "").strip()
            if _PASAL_TITLE_RE.match(t):
                key_tuple = tuple(current_keys)
                existing = pasal_best.get(t)
                if existing is None:
                    pasal_best[t] = n
                    pasal_to_path[t] = key_tuple
                else:
                    if len(json.dumps(n, ensure_ascii=False)) > len(
                        json.dumps(existing, ensure_ascii=False)
                    ):
                        pasal_best[t] = n
                        if len(key_tuple) > len(pasal_to_path.get(t, ())):
                            pasal_to_path[t] = key_tuple
                continue
            depth = _container_depth(t)
            if depth is not None:
                current_path = current_path[:depth] + [t]
                current_keys = current_keys[:depth] + [_container_key(n)]
                key_tuple = tuple(current_keys)
                if key_tuple not in container_stub:
                    container_stub[key_tuple] = {k: v for k, v in n.items() if k != "nodes"}
                    container_order.append(key_tuple)
                _walk(n.get("nodes", []) or [])
            else:
                _walk(n.get("nodes", []) or [])

    for chunk in chunks:
        _walk(chunk)

    def _pasal_key(name: str) -> tuple[int, str]:
        m = re.match(r"Pasal\s+(\d+)([A-Z]?)", name)
        if not m:
            return (10**9, name)
        return (int(m.group(1)), m.group(2) or "")

    path_buckets: dict[tuple, list[str]] = {}
    unbucketed: list[str] = []
    for p in sorted(pasal_best.keys(), key=_pasal_key):
        path = pasal_to_path.get(p, ())
        if not path:
            unbucketed.append(p)
            continue
        path_buckets.setdefault(path, []).append(p)

    root_nodes: list[dict] = []
    node_index: dict[tuple, dict] = {}

    def _ensure_path(path: tuple) -> dict:
        """Return the deepest container on `path`, creating missing ancestors."""
        for depth in range(1, len(path) + 1):
            sub = path[:depth]
            if sub in node_index:
                continue
            template = container_stub.get(sub, {"title": sub[-1], "node_id": _sanitize_node_id(sub[-1]).lower()})
            node = {**template, "nodes": []}
            node_index[sub] = node
            parent = sub[:-1]
            if parent:
                node_index[parent].setdefault("nodes", []).append(node)
            else:
                root_nodes.append(node)
        return node_index[path]

    for key in container_order:
        _ensure_path(key)

    for path, pasal_list in path_buckets.items():
        container = _ensure_path(path)
        container.setdefault("nodes", []).extend(pasal_best[p] for p in pasal_list)

    if unbucketed:
        if root_nodes:
            last = root_nodes[-1]
            while last.get("nodes") and last["nodes"] and _is_container(last["nodes"][-1]):
                last = last["nodes"][-1]
            last.setdefault("nodes", []).extend(pasal_best[p] for p in unbucketed)
        else:
            root_nodes.extend(pasal_best[p] for p in unbucketed)

    def _prune(nodes: list[dict]) -> list[dict]:
        out = []
        for n in nodes:
            if n.get("nodes"):
                n["nodes"] = _prune(n["nodes"])
            if _is_container(n) and not n.get("nodes"):
                continue
            out.append(n)
        return out

    return _prune(root_nodes)


def _chunk_pages(
    pages: list[dict], pages_per_chunk: int, overlap: int
) -> list[tuple[int, int]]:
    """Page-based chunking — fallback when pasal-aware chunking finds no boundaries."""
    total = len(pages)
    if total <= pages_per_chunk:
        return [(1, total)]
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= total:
        end = min(total, start + pages_per_chunk - 1)
        ranges.append((start, end))
        if end == total:
            break
        start = end - overlap + 1
    return ranges


def _chunk_by_pasal(
    pages: list[dict],
    total_pages: int,
    pasals_per_chunk: int = 20,
    overlap_pages: int = 1,
) -> list[tuple[int, int]]:
    """Chunk into page ranges aligned on Pasal boundaries with overlap.

    Boundaries land BETWEEN Pasals, never mid-Pasal. Each range is extended by
    `overlap_pages` on each side so adjacent chunks share context — the LLM
    sees prior-Pasal context at the chunk head (avoids misattributing page-top
    spillover to the first heading), and the merger keeps the longest copy.

    Returns [] when no Pasals are detected — caller should fall back to
    `_chunk_pages`.
    """
    # Count heading OCCURRENCES, not unique pages: dense docs pack 3-5 pasals
    # per page, so page-counting would undercount.
    heading_re = re.compile(r"^\s*[Pp]asa[l1]\s+\d+[A-Z]?\b")
    pasal_occurrences: list[int] = []
    for p in pages:
        if p["page_num"] > total_pages:
            break
        for block in p.get("blocks", []):
            for line in block.get("text", "").splitlines():
                if heading_re.match(line):
                    pasal_occurrences.append(p["page_num"])
    if len(pasal_occurrences) < 2:
        return []

    ranges: list[tuple[int, int]] = []
    for i in range(0, len(pasal_occurrences), pasals_per_chunk):
        bucket = pasal_occurrences[i: i + pasals_per_chunk]
        aligned_start = 1 if i == 0 else bucket[0]
        if i + pasals_per_chunk < len(pasal_occurrences):
            next_first = pasal_occurrences[i + pasals_per_chunk]
            aligned_end = max(aligned_start, next_first - 1)
        else:
            aligned_end = total_pages
        start = max(1, aligned_start - overlap_pages)
        end = min(total_pages, aligned_end + overlap_pages)
        ranges.append((start, end))
    return ranges


def parse_doc(doc_id: str, dry_run: bool = False) -> dict:
    """Parse one document and return its audit record."""
    audit: dict = {
        "doc_id": doc_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "pending",
    }

    try:
        meta = build_metadata(doc_id)
    except Exception as exc:
        audit["status"] = "error"
        audit["error"] = f"metadata: {exc}"
        return audit

    judul = meta["judul"]
    is_perubahan = meta["is_perubahan"]
    total_pages = meta["total_pages"]
    body_end = meta["body_pages"]
    audit["is_perubahan"] = is_perubahan
    audit["total_pages"] = total_pages
    audit["body_pages"] = body_end

    try:
        pages = load_pdf_pages(doc_id)
    except Exception as exc:
        audit["status"] = "error"
        audit["error"] = f"pdf load: {exc}"
        return audit

    pdf_pasal_numbers = _pasal_numbers_in_page_range(pages, 1, body_end)
    audit["pdf_pasal_regex_count"] = len(pdf_pasal_numbers)

    # Chunk when char volume is large OR pasal count risks busting the ~65K
    # output-token budget (~35 pasals of nested text).
    pdf_text_full = format_pdf_pages(pages, 1, body_end)
    char_count = len(pdf_text_full)
    audit["pdf_chars"] = char_count
    # Amendment docs emit only 1-2 Pasal Roman roots regardless of inner Pasal N
    # count, so pasal-count chunking is meaningless — only char volume matters.
    if is_perubahan:
        use_chunked = char_count > MAX_WHOLE_DOC_INPUT_CHARS
    else:
        use_chunked = (
            char_count > MAX_WHOLE_DOC_INPUT_CHARS
            or len(pdf_pasal_numbers) > MAX_WHOLE_DOC_PASALS
        )
    audit["mode"] = "chunked" if use_chunked else "whole"

    agg_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0, "elapsed_s": 0.0}

    if not use_chunked:
        prompt = build_prompt(doc_id, judul, is_perubahan, pdf_text_full)
        est_tokens = len(prompt) // 4
        print(f"  calling Gemini ({est_tokens:,} input tokens, whole doc)...", flush=True)
        obj, usage, err = _run_llm(prompt)
        _accumulate_usage(agg_usage, usage)
        if err:
            audit["status"] = "error"
            audit["error"] = err
            audit["usage"] = agg_usage
            return audit
        structure = obj["structure"]
    else:
        body_pages_list = [p for p in pages if p["page_num"] <= body_end]
        ranges = _chunk_by_pasal(body_pages_list, body_end, pasals_per_chunk=20)
        strategy = "pasal-aware"
        if not ranges:
            ranges = _chunk_pages(body_pages_list, PAGES_PER_CHUNK, CHUNK_OVERLAP_PAGES)
            strategy = "page-based"
        print(
            f"  chunked mode ({strategy}): {total_pages} pages split into "
            f"{len(ranges)} chunks, parallel={PARALLEL_WORKERS}",
            flush=True,
        )
        chunk_structures: list[list[dict] | None] = [None] * len(ranges)
        errors: list[str] = []

        def _worker(i: int, start: int, end: int):
            scoped = format_pdf_pages(pages, start, end)
            expected = _pasals_in_page_range(pages, start, end)
            prompt = build_prompt(
                doc_id, judul, is_perubahan, scoped,
                start_page=start, end_page=end,
                expected_pasals=expected,
            )
            est = len(prompt) // 4
            print(
                f"  chunk {i+1}/{len(ranges)} (p{start}-{end}, {len(expected)} pasals expected): "
                f"{est:,} input tokens...",
                flush=True,
            )
            return i, _run_llm(prompt)

        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
            futures = [pool.submit(_worker, i, s, e) for i, (s, e) in enumerate(ranges)]
            for fut in as_completed(futures):
                try:
                    i, (obj, usage, err) = fut.result()
                except Exception as exc:
                    errors.append(f"worker exception: {exc}")
                    continue
                _accumulate_usage(agg_usage, usage)
                if err or not obj:
                    errors.append(f"chunk {i+1}: {err}")
                    continue
                chunk_structures[i] = obj["structure"]
        valid_chunks = [c for c in chunk_structures if c]
        if not valid_chunks:
            audit["status"] = "error"
            audit["error"] = "all chunks failed"
            audit["chunk_errors"] = errors
            audit["usage"] = agg_usage
            return audit
        audit["chunk_errors"] = errors
        structure = _merge_chunk_structures(valid_chunks)

    audit["usage"] = agg_usage

    # Normalize pasal titles BEFORE building navigation_path — paths must
    # reflect canonical form ("Pasal 15I", not OCR "Pasal l5I"). BM25
    # tokenization depends on clean title/text tokens.
    normalize_pasal_titles_in_tree(structure)
    assign_readable_node_ids(structure)
    build_navigation_paths(structure)
    backfill_page_indices(structure, pages, total_pages)

    # Penjelasan is intentionally NOT attached: GT targets body text, and the
    # regex-based attribution had layout-specific bugs mis-attributing across
    # pasals. Raw map stays at doc-level for optional display only.

    # Validation is diagnostic, NOT a gate — the regex validator false-positives
    # on valid LLM output (intentional Pasal gaps, container nodes).
    _, errors = validate_parse({"structure": structure}, pdf_pasal_numbers, is_perubahan=is_perubahan)
    audit["validation_errors"] = errors

    final_doc = dict(meta)
    final_doc["structure"] = structure
    final_doc["parser_method"] = "llm_parse"
    final_doc["llm_parse_model"] = MODEL_NAME
    final_doc["llm_parse_applied_at"] = datetime.now(timezone.utc).isoformat()

    audit["pasal_count"] = count_pasals_in_tree(structure)

    if dry_run:
        audit["status"] = "dry_run_ok"
        preview = REPO_ROOT / "tmp" / f"llm_parse_preview_{doc_id}.json"
        preview.parent.mkdir(parents=True, exist_ok=True)
        with open(preview, "w", encoding="utf-8") as f:
            json.dump(final_doc, f, ensure_ascii=False, indent=2)
        audit["preview_path"] = str(preview)
        return audit

    index_path = INDEX_PASAL / meta["jenis_folder"] / f"{doc_id}.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)

    if index_path.exists():
        backup_cat = BACKUP_DIR / meta["jenis_folder"]
        backup_cat.mkdir(parents=True, exist_ok=True)
        backup_path = backup_cat / index_path.name
        if not backup_path.exists():
            shutil.copy2(index_path, backup_path)
        audit["backup_path"] = str(backup_path)

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(final_doc, f, ensure_ascii=False, indent=2)

    audit["status"] = "parsed"
    audit["index_path"] = str(index_path)
    return audit


def _accumulate_usage(dst: dict, src: dict) -> None:
    for k in ("input_tokens", "output_tokens", "total_tokens", "calls", "elapsed_s"):
        dst[k] = (dst.get(k, 0) or 0) + (src.get(k, 0) or 0)


def _load_targets(specific: list[str] | None, category: str | None) -> list[str]:
    """Return the document IDs requested on the CLI."""
    if specific:
        return specific
    if not category:
        raise RuntimeError("must pass --doc-id(s) or --category")
    reg_path = REPO_ROOT / "data" / "raw" / "registry.json"
    if not reg_path.exists():
        raise RuntimeError(f"registry not found at {reg_path}")
    reg = json.load(open(reg_path, encoding="utf-8"))
    target = category.upper()
    return sorted(
        doc_id for doc_id, entry in reg.items()
        if (entry.get("jenis_folder") or "").upper() == target
    )


def _append_audit(entry: dict) -> None:
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    existing: list = []
    if AUDIT_LOG.exists():
        try:
            existing = json.load(open(AUDIT_LOG, encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    existing.append(entry)
    with open(AUDIT_LOG, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc-id", action="append", dest="doc_ids", default=[],
                    help="Doc to parse (repeatable)")
    ap.add_argument("--category",
                    help="Parse every doc in this jenis_folder (e.g. UU, OJK)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Preview only, do not overwrite index")
    args = ap.parse_args()

    targets = _load_targets(list(args.doc_ids) or None, args.category)
    print(f"Targets: {len(targets)} docs")
    if not targets:
        return

    for i, doc_id in enumerate(targets, 1):
        print(f"\n[{i}/{len(targets)}] {doc_id}")
        try:
            audit = parse_doc(doc_id, dry_run=args.dry_run)
        except Exception as exc:
            audit = {
                "doc_id": doc_id,
                "status": "error",
                "error": f"unhandled: {exc}",
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
        _append_audit(audit)
        status = audit.get("status")
        msg = f"  status: {status}"
        if "pasal_count" in audit:
            msg += f"  pasals: {audit['pasal_count']} (pdf regex: {audit.get('pdf_pasal_regex_count', '?')})"
        if audit.get("validation_errors"):
            msg += f"  errors: {audit['validation_errors'][:2]}"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
