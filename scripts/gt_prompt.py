"""
Ground Truth Prompt Generator.

Generates a copy-paste prompt for ChatGPT to create self-contained,
ayat-anchored ground truth question-answer pairs for Indonesian legal
QA evaluation.

Workflow:
  1. Run: python scripts/gt_prompt.py <doc_id>
  2. Copy the printed prompt
  3. Paste into ChatGPT (NOT Gemini - avoid retrieval bias)
  4. Copy the JSON output -> save to data/ground_truth_raw/<doc_id>.json
  5. Run: python scripts/gt_collect.py  (validates + merges all raw files)

Usage:
    python scripts/gt_prompt.py perpu-1-2016
    python scripts/gt_prompt.py perpu-1-2016 --questions 15
    python scripts/gt_prompt.py perpu-1-2016 --out prompt.txt
    python scripts/gt_prompt.py --list
"""

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

# Force UTF-8 output on Windows (navigation paths contain Unicode chars like em-dash)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_INDEX = Path("data/index_ayat")

# Preamble section names to exclude from GT (they are not substantive law text)
PREAMBLE_KEYWORDS = ["Menimbang", "Mengingat", "Menetapkan", "Pembukaan"]

# Full prompt template sent verbatim to ChatGPT.
# Placeholders: {judul}, {doc_id}, {leaf_blocks_grouped}, {N}
PROMPT_TEMPLATE = """\
Kamu adalah asisten ahli hukum yang membantu membuat dataset evaluasi sistem \
temu kembali (retrieval) dokumen hukum Indonesia.

Dokumen: {judul}
doc_id: {doc_id}

=== LEAF NODE INDEX AYAT (dikelompokkan per Bab/Bagian) ===
{leaf_blocks_grouped}

=== INSTRUKSI UTAMA ===

Buat tepat {N} pertanyaan dalam bahasa Indonesia berdasarkan leaf node di atas.

ATURAN WAJIB:
1. Setiap pertanyaan harus SELF-CONTAINED: harus bisa dipahami sendirian tanpa \
percakapan atau query sebelumnya.
2. Setiap pertanyaan harus dijawab oleh TEPAT SATU leaf node pada index ayat \
(satu ayat anchor). Jika sebuah pasal tidak punya ayat eksplisit dan tetap menjadi \
leaf di index ayat, leaf tersebut boleh dipakai sebagai anchor.
3. Jangan buat pertanyaan yang butuh dua atau lebih ayat/pasal.
4. Distribusikan pertanyaan ke BANYAK bagian yang berbeda - jangan terlalu banyak \
dari satu Bab yang sama.
5. Jawaban harus ada secara EKSPLISIT di teks node (bukan inferensi atau interpretasi).
6. Jangan buat pertanyaan tentang bagian Menimbang, Mengingat, atau Menetapkan.
7. DILARANG membuat query yang context-dependent atau coreferential, misalnya \
"aturan ini", "ketentuan tersebut", "hal itu", "yang tadi", atau "juga nggak?" \
jika acuan sebelumnya tidak disebut jelas dalam kalimat yang sama.
8. Jika memakai kata "ini", "itu", atau "tersebut", antecedent-nya harus disebut \
eksplisit di query yang sama, sehingga query tetap self-contained.

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

4. vague - agak umum atau ambigu, seperti pertanyaan dari orang yang belum tahu \
pasti istilahnya, TETAPI tetap self-contained dan tidak boleh bergantung pada \
konteks percakapan sebelumnya.
   Allowed vague: "Apa aturan soal hukuman tambahan buat pelaku kejahatan terhadap anak?"
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
    "gold_anchor_granularity": "ayat",
    "gold_anchor_node_id": "node_id leaf index ayat yang menjawab",
    "gold_node_id": "sama dengan gold_anchor_node_id",
    "gold_doc_id": "{doc_id}",
    "navigation_path": "navigation_path dari leaf node tersebut",
    "answer_hint": "kutipan singkat dari teks yang menjadi kunci jawaban, maksimal 100 karakter"
  }}
]

Buat tepat {N} item. Gunakan distribusi query_style ~25% tiap jenis dan difficulty \
~40% easy, ~40% medium, ~20% tricky. Pastikan semua query self-contained dan semua \
gold node mengacu ke leaf node pada index ayat.
Kembalikan HANYA JSON array. Tidak ada teks lain di luar JSON.
"""


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
                "text": node["text"],
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
    - Cap at 20 keeps prompts within a reasonable ChatGPT context window.
    """
    return max(10, min(leaf_count, 20))


def group_leaves_by_section(leaf_nodes: list[dict]) -> dict[str, list[dict]]:
    """
    Group ayat-index leaf nodes by their top-level section (Bab/Bagian/Paragraf).

    The top-level section is the first path component before " > " in navigation_path.
    Nodes without " > " in their path fall into a catch-all "Lainnya" bucket.

    Returns an OrderedDict preserving the order sections first appear.
    This grouping is shown in the prompt to help ChatGPT distribute queries
    across different parts of the document.
    """
    groups: dict[str, list[dict]] = OrderedDict()
    for leaf in leaf_nodes:
        nav = leaf.get("navigation_path", "")
        if " > " in nav:
            section = nav.split(" > ")[0].strip()
        else:
            section = "Lainnya"
        if section not in groups:
            groups[section] = []
        groups[section].append(leaf)
    return groups


def truncate_text(text: str, max_chars: int = 800) -> str:
    """Truncate text for prompt, preserving the beginning."""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...terpotong]"


def render_grouped_blocks(groups: dict[str, list[dict]]) -> str:
    """
    Render ayat-index leaves grouped by section into a formatted string for the prompt.

    Each section starts with a header line "--- SECTION TITLE ---" followed by
    individual leaf blocks showing node_id, title, navigation path, and text.
    """
    section_parts = []
    for section_title, leaves in groups.items():
        leaf_blocks = []
        for leaf in leaves:
            block = (
                f"[node_id: {leaf['node_id']}]\n"
                f"Judul: {leaf['title']}\n"
                f"Path: {leaf['navigation_path']}\n"
                f"Teks:\n{truncate_text(leaf['text'])}"
            )
            leaf_blocks.append(block)
        section_block = f"\n--- {section_title} ---\n\n" + "\n\n".join(leaf_blocks)
        section_parts.append(section_block)
    return "\n".join(section_parts)


def build_prompt(doc: dict, n_questions: int | None = None) -> tuple[str, int]:
    """
    Build the copy-paste prompt for ChatGPT.

    Args:
        doc: Loaded document JSON from index_ayat/.
        n_questions: Override question count. If None, computed adaptively via
            compute_adaptive_n(leaf_count).

    Returns:
        Tuple of (prompt_string, n_used) where n_used is the actual count
        injected into the prompt.
    """
    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)

    n = n_questions if n_questions is not None else compute_adaptive_n(len(leaf_nodes))
    groups = group_leaves_by_section(leaf_nodes)
    leaf_blocks_grouped = render_grouped_blocks(groups)

    prompt = PROMPT_TEMPLATE.format(
        judul=doc["judul"],
        doc_id=doc["doc_id"],
        leaf_blocks_grouped=leaf_blocks_grouped,
        N=n,
    )
    return prompt, n


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
        help="Save prompt to file instead of printing to stdout",
    )
    ap.add_argument(
        "--list", "-l", action="store_true",
        help="List all available doc_ids",
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

    print(f"\nDokumen    : {doc['judul'][:80]}")
    print(f"doc_id     : {doc['doc_id']}")
    print(f"Leaf nodes : {len(leaf_nodes)} ayat-index leaf nodes")
    print(f"Target N   : {n_used} pertanyaan (adaptive: {adaptive_n})", end="")
    if args.questions is not None:
        print(f"  [override: --questions {args.questions}]", end="")
    print()
    print(f"Output     : {args.out or 'stdout (copy-paste)'}\n")

    prompt, _ = build_prompt(doc, n_questions=args.questions)

    if args.out:
        Path(args.out).write_text(prompt, encoding="utf-8")
        print(f"Prompt disimpan ke: {args.out}")
        print("\nLangkah selanjutnya:")
        print(f"  1. Buka {args.out} dan copy isinya")
    else:
        print("=" * 70)
        print("COPY PROMPT DI BAWAH INI KE CHATGPT (BUKAN Gemini):")
        print("=" * 70)
        print(prompt)
        print("=" * 70)
        print("\nLangkah selanjutnya:")
        print("  1. Copy prompt di atas -> paste ke ChatGPT")

    raw_dir = Path("data/ground_truth_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    print("  2. Copy JSON output dari ChatGPT")
    print(f"  3. Simpan ke: data/ground_truth_raw/{args.doc_id}.json")
    print("  4. Jalankan: python scripts/gt_collect.py")


if __name__ == "__main__":
    main()
