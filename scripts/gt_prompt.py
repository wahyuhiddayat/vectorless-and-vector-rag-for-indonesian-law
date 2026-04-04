"""
Ground Truth Prompt Generator.

Generates one or more copy-paste prompts for an annotator LLM to create
self-contained, ayat-anchored ground truth question-answer pairs for
Indonesian legal retrieval evaluation.

Long documents are split into multiple prompt files automatically so the
annotator always sees full node text. Leaf nodes are never truncated and
never split across prompt parts.

Usage:
    python scripts/gt_prompt.py perpu-1-2016
    python scripts/gt_prompt.py perpu-1-2016 --questions 15
    python scripts/gt_prompt.py perpu-1-2016 --out tmp/custom_prompt.txt
    python scripts/gt_prompt.py perpu-1-2016 --stdout
    python scripts/gt_prompt.py --list
"""

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

# Force UTF-8 output on Windows (navigation paths contain Unicode chars like em-dash)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_INDEX = Path("data/index_ayat")
TMP_DIR = Path("tmp")
DEFAULT_PROMPT_CHAR_BUDGET = 45000
PROMPT_BUDGET_GROWTH = 1.25

# Preamble section names to exclude from GT (they are not substantive law text)
PREAMBLE_KEYWORDS = ["Menimbang", "Mengingat", "Menetapkan", "Pembukaan"]

# Full prompt template sent verbatim to ChatGPT.
PROMPT_TEMPLATE = """\
Kamu adalah asisten ahli hukum yang membantu membuat dataset evaluasi sistem \
temu kembali (retrieval) dokumen hukum Indonesia.

Dokumen: {judul}
doc_id: {doc_id}
{part_header}
=== LEAF NODE INDEX AYAT (dikelompokkan per Bab/Bagian) ===
{leaf_blocks_grouped}

=== INSTRUKSI UTAMA ===

Buat tepat {N} pertanyaan dalam bahasa Indonesia berdasarkan leaf node di atas.
Gunakan HANYA leaf node yang ditampilkan pada prompt ini. Jangan membuat item \
dengan `gold_anchor_node_id` di luar node yang muncul pada prompt ini.

ATURAN WAJIB:
1. Setiap pertanyaan harus SELF-CONTAINED: harus bisa dipahami sendirian tanpa \
percakapan atau query sebelumnya.
2. Setiap pertanyaan harus dijawab oleh TEPAT SATU leaf node pada index ayat \
(satu ayat anchor). Jika sebuah pasal tidak punya ayat eksplisit dan tetap menjadi \
leaf di index ayat, leaf tersebut boleh dipakai sebagai anchor.
3. Jangan buat pertanyaan yang butuh dua atau lebih ayat/pasal.
4. Nama dokumen BOLEH disebut, tetapi TIDAK WAJIB. Jangan paksa semua pertanyaan \
menyebut nama dokumen, Pasal, atau ayat.
5. Pertanyaan boleh menyebut:
   - tidak menyebut dokumen maupun referensi hukum,
   - hanya menyebut referensi hukum (Pasal/ayat/huruf/angka),
   - hanya menyebut dokumen,
   - atau menyebut keduanya.
6. Gunakan campuran reference_mode yang seimbang. Benchmark utama TIDAK boleh \
didominasi query yang eksplisit menyebut Pasal/ayat. Jika ragu, lebih baik \
parafrase menjadi `none` atau `doc_only` daripada terus memakai `legal_ref`.
7. Distribusikan pertanyaan ke BANYAK bagian yang berbeda - jangan terlalu banyak \
dari satu Bab yang sama.
8. Jawaban harus ada secara EKSPLISIT di teks node (bukan inferensi atau interpretasi).
9. Jangan buat pertanyaan tentang bagian Menimbang, Mengingat, atau Menetapkan.
10. DILARANG membuat query yang context-dependent atau coreferential, misalnya \
"aturan ini", "ketentuan tersebut", "hal itu", "yang tadi", atau "juga nggak?" \
jika acuan sebelumnya tidak disebut jelas dalam kalimat yang sama.
11. Jika memakai kata "ini", "itu", atau "tersebut", antecedent-nya harus disebut \
eksplisit di query yang sama, sehingga query tetap self-contained.
12. Pertanyaan INVALID jika ayat anchor hanya bisa menjawab dengan cara menunjuk \
ke ayat/pasal lain. Jika teks ayat hanya berkata "sebagaimana dimaksud pada ayat ..." \
dan detail jawabannya sebenarnya ada di sibling ayat, JANGAN pakai ayat itu sebagai GT.
13. Main benchmark ini HANYA untuk single-hop retrieval. Jika menjawab query \
memerlukan penggabungan informasi dari lebih dari satu ayat/pasal, query itu INVALID.
14. Main benchmark ini hanya mencakup body text dokumen. Jangan buat query yang \
bergantung pada metadata top-level dokumen atau bagian pembukaan.

=== REFERENCE MODE (reference_mode) ===

Setiap item HARUS punya `reference_mode` yang sesuai dengan bentuk query:

1. none - tidak menyebut nama dokumen dan tidak menyebut Pasal/ayat/huruf/angka.
   Contoh valid:
   "Kalau seseorang membujuk anak untuk melakukan persetubuhan dengan dirinya atau \
orang lain, apakah ketentuan pidana yang sama juga berlaku?"

2. legal_ref - menyebut Pasal/ayat/huruf/angka, tetapi tidak menyebut nama dokumen.
   Contoh valid:
   "Berapa pidana penjara paling singkat dan paling lama bagi setiap orang yang \
melanggar ketentuan dalam Pasal 76D?"

3. doc_only - menyebut nama/jenis dokumen, tetapi tidak menyebut Pasal/ayat/huruf/angka.
   Contoh valid:
   "Dalam Perpu Nomor 1 Tahun 2016, kapan peraturan ini mulai berlaku?"

4. both - menyebut nama dokumen dan juga referensi hukum seperti Pasal/ayat.
   Contoh valid:
   "Siapa saja pelaku yang ancaman pidananya ditambah sepertiga dalam Pasal 81 ayat \
(3) Perpu Nomor 1 Tahun 2016?"

Target distribusi reference_mode per batch:
- none: 3-5 item
- legal_ref: 3-5 item
- doc_only: 1-3 item
- both: 1-3 item

CATATAN:
- `both` harus menjadi minoritas.
- Query `none` dan `doc_only` sangat dianjurkan selama tetap self-contained dan unik.
- Jangan paksa semua query menjadi `legal_ref` atau `both`.
- `legal_ref` + `both` TIDAK boleh melebihi setengah total item.
- `both` maksimal 3 item.
- `legal_ref` maksimal 5 item.
- Invalid multi-ayat dependency:
  query yang di-anchor ke ayat yang hanya berkata "sebagaimana dimaksud pada ayat \
  (4) dan ayat (5)" tetapi pertanyaannya justru meminta rincian kondisi dari ayat \
  (4)/(5).

=== PRE-FLIGHT CHECK SEBELUM OUTPUT ===

Sebelum mengembalikan JSON final, lakukan pengecekan internal:
1. Hitung jumlah item per `reference_mode`.
2. Jika `legal_ref` + `both` > setengah total item, ubah beberapa query menjadi \
   `none` atau `doc_only`.
3. Jika `both` > 3, kurangi.
4. Jika `legal_ref` > 5, kurangi.
5. Jika `none` < 3, tambahkan query `none`.
6. Pastikan tidak ada query yang hanya bagus karena terlalu eksplisit menyebut \
   Pasal/ayat padahal bisa ditulis lebih natural.

=== JENIS PERTANYAAN (query_style) ===

Distribusi target: masing-masing sekitar 25% dari total pertanyaan.

1. formal - bahasa hukum resmi, seperti yang digunakan dalam dokumen atau persidangan.
   Contoh: "Apa yang dimaksud dengan 'kebiri kimia' sebagaimana diatur dalam Pasal 81 ayat (7)?"

2. natural - bahasa sehari-hari orang awam yang tidak berlatar hukum.
   Contoh: "Kalau ada yang melakukan kekerasan seksual terhadap anak, hukumannya apa?"

3. paraphrase - kata-kata berbeda tapi maknanya sama. Hindari menggunakan kata kunci \
yang persis sama dengan teks node.
   Contoh: "Berapa lama sanksi tambahan dapat diterapkan setelah narapidana selesai \
menjalani hukuman pokoknya?"

4. vague - wording lebih umum atau lebih awam, seperti pertanyaan dari orang yang \
belum tahu istilah hukumnya, TETAPI tetap self-contained dan tetap harus uniquely \
answerable by ONE leaf node.
   `vague` TIDAK berarti conversational, underspecified, atau referential.
   Jika query vague masih bisa cocok ke lebih dari satu leaf node dalam dokumen yang \
   sama, query itu INVALID dan harus ditulis ulang sampai unik.
   Valid vague: "Apa aturan soal pembaruan susunan P2K3 kalau ketua atau sekretaris berubah?"
   Valid vague: "Apa aturan soal hukuman tambahan buat pelaku kejahatan terhadap anak?"
   Invalid vague: "Bagaimana aturannya?"
   Invalid vague: "Kalau ada perubahan, apa yang harus dilakukan?"
   Invalid vague: "Apa ketentuannya soal itu?"
   Disallowed context-dependent: "Kalau begitu, aturan ini juga berlaku nggak?"

=== TINGKAT KESULITAN (difficulty) ===

Distribusi target: 40% easy, 40% medium, 20% tricky.

1. easy - jawabannya ada dalam satu kalimat atau frasa eksplisit. Orang bisa menjawab \
langsung tanpa membaca keseluruhan node.
   Contoh: "Berapa denda maksimal yang dikenakan?" -> jawaban ada dalam satu angka/kalimat.

2. medium - butuh membaca satu ayat atau satu leaf node utuh dan memahami konteksnya. \
Jawaban tidak ada dalam satu kalimat saja, mungkin ada kondisi atau pengecualian.
   Contoh: "Siapa saja yang dapat dikenai pidana tambahan?"

3. tricky - butuh membedakan kondisi spesifik, atau mudah tertukar dengan ayat lain \
yang mirip. Pertanyaan yang sepertinya sederhana tapi jawabannya membutuhkan ketelitian.
   CATATAN: Pertanyaan tricky tetap harus dijawab oleh SATU ayat anchor.
   Contoh: "Dalam kondisi apa sanksi A berlaku, bukan sanksi B?" (jika keduanya ada di \
ayat yang berbeda dalam pasal yang sama)

=== FORMAT OUTPUT ===

Kembalikan HANYA JSON array berikut, tanpa markdown, tanpa penjelasan tambahan:

[ 
  {{
    "query": "pertanyaan dalam bahasa Indonesia",
    "query_style": "formal|natural|paraphrase|vague",
    "difficulty": "easy|medium|tricky",
    "reference_mode": "none|legal_ref|doc_only|both",
    "gold_anchor_granularity": "ayat",
    "gold_anchor_node_id": "node_id leaf index ayat yang menjawab",
    "gold_node_id": "sama dengan gold_anchor_node_id",
    "gold_doc_id": "{doc_id}",
    "navigation_path": "navigation_path dari leaf node tersebut",
    "answer_hint": "kutipan singkat dari teks yang menjadi kunci jawaban, maksimal 100 karakter"
  }}
]

Buat tepat {N} item. Gunakan distribusi query_style ~25% tiap jenis dan difficulty \
~40% easy, ~40% medium, ~20% tricky. Gunakan distribusi reference_mode yang \
seimbang seperti di atas. Benchmark ini untuk retrieval stateless, jadi query \
harus membantu menguji document discovery, bukan cuma pencocokan kata `Pasal/ayat`. \
Pastikan semua query self-contained, single-hop, tidak terlalu didominasi explicit legal \
references, tidak mencakup metadata/pembukaan, dan semua gold node mengacu ke leaf \
node pada index ayat.
Kembalikan HANYA JSON array. Tidak ada teks lain di luar JSON.
"""


def default_output_path(doc_id: str) -> Path:
    """Return the default single-prompt path under the repo-local tmp folder."""
    return TMP_DIR / f"gt_{doc_id}.txt"


def default_output_prefix(doc_id: str) -> Path:
    """Return the default multipart prefix under the repo-local tmp folder."""
    return TMP_DIR / f"gt_{doc_id}"


def manifest_path_from_prefix(prefix: Path) -> Path:
    """Return the manifest path for a multipart prompt run."""
    return prefix.parent / f"{prefix.name}_manifest.json"


def part_path_from_prefix(prefix: Path, part_index: int) -> Path:
    """Return the prompt file path for one part."""
    return prefix.parent / f"{prefix.name}_part{part_index:02d}.txt"


def make_output_target(doc_id: str, out_arg: str | None, multipart: bool) -> Path:
    """Resolve output file/prefix for single or multipart prompt generation."""
    if not out_arg:
        return default_output_prefix(doc_id) if multipart else default_output_path(doc_id)

    out_path = Path(out_arg)
    if multipart:
        return out_path.parent / out_path.stem
    return out_path


def find_doc(doc_id: str) -> Path | None:
    """Search for doc_id across all category subfolders."""
    for path in DATA_INDEX.rglob("*.json"):
        if path.stem == doc_id and path.name != "catalog.json":
            return path
    return None


def collect_leaf_nodes(
    nodes: list[dict],
    results: list[dict] | None = None,
    parent_path: str = "",
) -> list[dict]:
    """Recursively collect all leaf nodes (nodes with text content) from tree."""
    if results is None:
        results = []
    for node in nodes:
        node_path = node.get("navigation_path", "").strip()
        if not node_path and parent_path and node.get("title"):
            node_path = f"{parent_path} > {node['title']}"
        elif not node_path:
            node_path = parent_path

        if "nodes" in node and node["nodes"]:
            collect_leaf_nodes(node["nodes"], results, parent_path=node_path)
        elif node.get("text"):
            results.append({
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "navigation_path": node_path,
                "text": node["text"].strip(),
            })
    return results


def filter_preamble(leaves: list[dict]) -> list[dict]:
    """Remove preamble leaves (Menimbang, Mengingat, Menetapkan, Pembukaan)."""
    filtered = [
        leaf for leaf in leaves
        if not any(kw in leaf.get("navigation_path", "") for kw in PREAMBLE_KEYWORDS)
    ]
    return filtered if filtered else leaves


def compute_adaptive_n(leaf_count: int) -> int:
    """
    Compute the adaptive question count based on number of ayat-index leaf nodes.

    Formula: max(10, min(leaf_count, 20))
    - Minimum 10 ensures statistical significance even for short documents.
    - Cap at 20 keeps annotation effort manageable.
    """
    return max(10, min(leaf_count, 20))


def section_name(leaf: dict) -> str:
    """Return the top-level section for a leaf node."""
    nav = leaf.get("navigation_path", "")
    if " > " in nav:
        return nav.split(" > ")[0].strip()
    return "Lainnya"


def render_leaf_block(leaf: dict) -> str:
    """Render one full, untruncated leaf block for the prompt."""
    return (
        f"[node_id: {leaf['node_id']}]\n"
        f"Judul: {leaf['title']}\n"
        f"Path: {leaf['navigation_path']}\n"
        f"Teks:\n{leaf['text']}"
    )


def render_grouped_blocks(leaf_nodes: list[dict]) -> str:
    """
    Render full ayat-index leaves grouped by top-level section.

    Each section starts with a header line "--- SECTION TITLE ---" followed by
    individual leaf blocks.
    """
    groups: dict[str, list[str]] = OrderedDict()
    for leaf in leaf_nodes:
        sec = section_name(leaf)
        groups.setdefault(sec, [])
        groups[sec].append(render_leaf_block(leaf))

    section_parts = []
    for section_title, blocks in groups.items():
        section_parts.append(f"\n--- {section_title} ---\n\n" + "\n\n".join(blocks))
    return "\n".join(section_parts)


def render_part_header(part_index: int, total_parts: int, quota: int, leaf_count: int) -> str:
    """Render part metadata shown to the annotator."""
    if total_parts <= 1:
        return ""
    return (
        "=== PART INFO ===\n"
        f"Part: {part_index} of {total_parts}\n"
        f"Question quota for this part: {quota}\n"
        f"Leaf nodes in this part: {leaf_count}\n\n"
    )


def build_prompt(doc: dict, leaf_nodes: list[dict], n_questions: int, part_index: int = 1, total_parts: int = 1) -> str:
    """Build one prompt for a specific part."""
    prompt = PROMPT_TEMPLATE.format(
        judul=doc["judul"],
        doc_id=doc["doc_id"],
        part_header=render_part_header(part_index, total_parts, n_questions, len(leaf_nodes)),
        leaf_blocks_grouped=render_grouped_blocks(leaf_nodes),
        N=n_questions,
    )
    return prompt


def pack_prompt_parts(doc: dict, leaf_nodes: list[dict], total_questions: int, base_budget: int = DEFAULT_PROMPT_CHAR_BUDGET) -> tuple[list[list[dict]], int]:
    """
    Split the document into prompt parts using whole-node packing.

    The budget may grow automatically if needed so that each part can still
    receive at least one question.
    """
    if not leaf_nodes:
        return [[]], base_budget

    budget = base_budget
    parts: list[list[dict]] = []

    while True:
        parts = []
        current: list[dict] = []
        for leaf in leaf_nodes:
            trial = current + [leaf]
            # Use a placeholder quota of 1 here; final quotas are assigned later.
            trial_prompt = build_prompt(doc, trial, n_questions=1)
            if current and len(trial_prompt) > budget:
                parts.append(current)
                current = [leaf]
            else:
                current = trial
        if current:
            parts.append(current)

        if len(parts) <= max(total_questions, 1):
            return parts, budget
        budget = math.ceil(budget * PROMPT_BUDGET_GROWTH)


def allocate_question_quotas(parts: list[list[dict]], total_questions: int) -> list[int]:
    """Allocate total question count proportionally across prompt parts."""
    if not parts:
        return []
    if len(parts) == 1:
        return [total_questions]

    leaf_counts = [len(part) for part in parts]
    total_leaves = sum(leaf_counts)
    quotas = [1 for _ in parts]
    remaining = total_questions - len(parts)
    if remaining < 0:
        raise ValueError("Cannot allocate at least one question per part")

    if remaining == 0:
        return quotas

    raw_extra = [remaining * (count / total_leaves) for count in leaf_counts]
    base_extra = [math.floor(x) for x in raw_extra]
    quotas = [q + extra for q, extra in zip(quotas, base_extra)]
    assigned = sum(quotas)
    leftovers = total_questions - assigned

    remainders = sorted(
        ((raw_extra[i] - base_extra[i], i) for i in range(len(parts))),
        reverse=True,
    )
    for _, idx in remainders[:leftovers]:
        quotas[idx] += 1

    return quotas


def build_prompt_parts(doc: dict, n_questions: int, char_budget: int = DEFAULT_PROMPT_CHAR_BUDGET) -> tuple[list[dict], int]:
    """Build single or multipart prompt payloads for one document."""
    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)
    parts, final_budget = pack_prompt_parts(doc, leaf_nodes, n_questions, base_budget=char_budget)
    quotas = allocate_question_quotas(parts, n_questions)
    total_parts = len(parts)

    prompt_parts = []
    for idx, (part_leaves, quota) in enumerate(zip(parts, quotas), start=1):
        prompt_parts.append({
            "part_index": idx,
            "total_parts": total_parts,
            "question_quota": quota,
            "leaf_count": len(part_leaves),
            "node_ids": [leaf["node_id"] for leaf in part_leaves],
            "prompt": build_prompt(doc, part_leaves, n_questions=quota, part_index=idx, total_parts=total_parts),
        })
    return prompt_parts, final_budget


def write_single_prompt(output_path: Path, prompt: str) -> None:
    """Write one prompt file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")


def write_multipart_prompts(prefix: Path, doc: dict, prompt_parts: list[dict], total_questions: int, final_budget: int) -> tuple[list[Path], Path]:
    """Write multipart prompt files plus a manifest."""
    prefix.parent.mkdir(parents=True, exist_ok=True)

    part_paths = []
    for part in prompt_parts:
        path = part_path_from_prefix(prefix, part["part_index"])
        path.write_text(part["prompt"], encoding="utf-8")
        part_paths.append(path)

    manifest = {
        "doc_id": doc["doc_id"],
        "judul": doc["judul"],
        "total_parts": len(prompt_parts),
        "total_questions": total_questions,
        "prompt_char_budget": final_budget,
        "parts": [
            {
                "part_index": part["part_index"],
                "question_quota": part["question_quota"],
                "leaf_count": part["leaf_count"],
                "node_ids": part["node_ids"],
                "prompt_file": str(path),
                "expected_raw_part_file": f"data/ground_truth_parts/{doc['doc_id']}/part{part['part_index']:02d}.json",
            }
            for part, path in zip(prompt_parts, part_paths)
        ],
    }
    manifest_path = manifest_path_from_prefix(prefix)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return part_paths, manifest_path


def list_available_docs() -> None:
    """Print all available doc_ids from the index."""
    docs = []
    for path in DATA_INDEX.rglob("*.json"):
        if path.name != "catalog.json":
            docs.append((path.parent.name, path.stem))
    docs.sort()
    print(f"\nDokumen tersedia di {DATA_INDEX}:\n")
    current_cat = None
    for cat, doc_id in docs:
        if cat != current_cat:
            print(f"\n  [{cat}]")
            current_cat = cat
        print(f"    {doc_id}")
    print(f"\nTotal: {len(docs)} dokumen\n")


def main() -> None:
    """CLI entrypoint."""
    ap = argparse.ArgumentParser(description="Generate ayat-anchored GT prompt for ChatGPT")
    ap.add_argument("doc_id", nargs="?", help="Document ID (e.g. perpu-1-2016)")
    ap.add_argument(
        "--questions", "-q", type=int, default=None,
        help="Override question count (default: adaptive, max(10, min(leaf_count, 20)))",
    )
    ap.add_argument(
        "--out", "-o", type=str, default=None,
        help="Save prompt to file; for multipart output this acts as a filename prefix",
    )
    ap.add_argument(
        "--stdout", action="store_true",
        help="Print prompt to stdout (single-prompt docs only)",
    )
    ap.add_argument(
        "--list", "-l", action="store_true",
        help="List all available doc_ids",
    )
    ap.add_argument(
        "--char-budget", type=int, default=DEFAULT_PROMPT_CHAR_BUDGET,
        help=f"Approximate prompt-size budget before automatic multipart splitting (default: {DEFAULT_PROMPT_CHAR_BUDGET})",
    )
    args = ap.parse_args()

    if args.list or not args.doc_id:
        list_available_docs()
        if not args.doc_id:
            ap.print_help()
        return

    doc_path = find_doc(args.doc_id)
    if not doc_path:
        print(f"ERROR: doc_id '{args.doc_id}' tidak ditemukan di {DATA_INDEX}")
        print("Gunakan --list untuk melihat semua doc_id yang tersedia.")
        sys.exit(1)

    with open(doc_path, encoding="utf-8") as f:
        doc = json.load(f)

    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)
    adaptive_n = compute_adaptive_n(len(leaf_nodes))
    n_used = args.questions if args.questions is not None else adaptive_n

    prompt_parts, final_budget = build_prompt_parts(doc, n_questions=n_used, char_budget=args.char_budget)
    multipart = len(prompt_parts) > 1

    print(f"\nDokumen    : {doc['judul'][:80]}")
    print(f"doc_id     : {doc['doc_id']}")
    print(f"Leaf nodes : {len(leaf_nodes)} ayat-index leaf nodes")
    print(f"Target N   : {n_used} pertanyaan (adaptive: {adaptive_n})", end="")
    if args.questions is not None:
        print(f"  [override: --questions {args.questions}]", end="")
    print()
    print(f"Prompt mode: {'multipart full-text' if multipart else 'single full-text'}")
    print(f"Budget     : {final_budget} chars")

    if multipart and args.stdout:
        print("\nERROR: Prompt multipart terlalu besar untuk --stdout.")
        print("Jalankan tanpa --stdout agar file part otomatis disimpan ke tmp/.")
        sys.exit(1)

    if not multipart:
        output_path = make_output_target(doc["doc_id"], args.out, multipart=False)
        output_label = "stdout (copy-paste)" if args.stdout else str(output_path)
        print(f"Output     : {output_label}\n")

        if args.stdout:
            print("=" * 70)
            print("COPY PROMPT DI BAWAH INI KE CHATGPT (BUKAN Gemini):")
            print("=" * 70)
            print(prompt_parts[0]["prompt"])
            print("=" * 70)
            print("\nLangkah selanjutnya:")
            print("  1. Copy prompt di atas -> paste ke ChatGPT")
            print("  2. Copy JSON output dari ChatGPT")
            print(f"  3. Simpan ke: data/ground_truth_raw/{args.doc_id}.json")
            print("  4. Jalankan: python scripts/gt_collect.py")
            return

        write_single_prompt(output_path, prompt_parts[0]["prompt"])
        print(f"Prompt disimpan ke: {output_path}")
        print("\nLangkah selanjutnya:")
        print(f"  1. Buka {output_path} dan copy isinya")
        print("  2. Copy JSON output dari ChatGPT")
        print(f"  3. Simpan ke: data/ground_truth_raw/{args.doc_id}.json")
        print("  4. Jalankan: python scripts/gt_collect.py")
        return

    prefix = make_output_target(doc["doc_id"], args.out, multipart=True)
    part_paths, manifest_path = write_multipart_prompts(prefix, doc, prompt_parts, total_questions=n_used, final_budget=final_budget)

    print(f"Output     : {len(part_paths)} part files + manifest")
    for part, path in zip(prompt_parts, part_paths):
        print(f"  - part {part['part_index']:02d}: {path}  [{part['question_quota']} pertanyaan, {part['leaf_count']} leaf]")
    print(f"  - manifest: {manifest_path}")

    parts_dir = Path("data/ground_truth_parts") / doc["doc_id"]
    print("\nLangkah selanjutnya:")
    print(f"  1. Untuk setiap part, buka file prompt di tmp/ lalu paste ke ChatGPT")
    print(f"  2. Simpan output JSON per part ke: {parts_dir}\\part01.json, part02.json, dst.")
    print(f"  3. Jalankan: python scripts/merge_gt_parts.py {doc['doc_id']}")
    print(f"  4. Jalankan: python scripts/gt_collect.py --check-only --file \"data\\ground_truth_raw\\{doc['doc_id']}.json\"")
    print("  5. Jika valid, merge dengan gt_collect.py seperti biasa")


if __name__ == "__main__":
    main()
