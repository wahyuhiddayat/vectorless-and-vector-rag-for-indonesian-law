"""
Ground Truth Prompt Generator.

Generates one or more copy-paste prompts for an annotator LLM to create
self-contained, leaf-anchored ground truth question-answer pairs for
Indonesian legal retrieval evaluation.

Anchors at the finest granularity (rincian index) so that evaluation
at coarser levels (ayat, pasal) can be derived by rolling UP to parents.

Long documents are split into multiple prompt files automatically so the
annotator always sees full node text. Leaf nodes are never truncated and
never split across prompt parts.

Usage:
    python scripts/gt/prompt.py perpu-1-2016
    python scripts/gt/prompt.py perpu-1-2016 --questions 15
    python scripts/gt/prompt.py perpu-1-2016 --out tmp/custom_prompt.txt
    python scripts/gt/prompt.py perpu-1-2016 --stdout
    python scripts/gt/prompt.py --list
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from pathlib import Path

# Force UTF-8 output on Windows (navigation paths contain Unicode chars like em-dash)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DATA_INDEX = Path("data/index_rincian")
TMP_DIR = Path("tmp")
SELECTION_FILE = Path("data/gt_doc_selection.json")
PROMPTS_DIR = Path("scripts/gt/prompts")
META_SUFFIX = ".meta.json"
DEFAULT_PROMPT_CHAR_BUDGET = 45000
PROMPT_BUDGET_GROWTH = 1.25

VALID_QUERY_TYPES = {"factual", "paraphrased", "multihop", "crossdoc", "adversarial"}
SINGLE_DOC_TYPES = {"factual", "paraphrased", "multihop", "adversarial"}
PAIRED_DOC_TYPES = {"crossdoc"}

# Preamble section names to exclude from GT (they are not substantive law text)
PREAMBLE_KEYWORDS = ["Menimbang", "Mengingat", "Menetapkan", "Pembukaan"]

# Minimum body leaf nodes required for GT generation (short docs produce unavoidable duplicates)
MIN_LEAF_FOR_GT = 5

# Legacy prompt template, kept for reference only. Active templates live in
# scripts/gt/prompts/<query_type>.txt and are loaded via load_template().
_LEGACY_PROMPT_TEMPLATE = """\
Kamu adalah asisten ahli hukum yang membantu membuat dataset evaluasi sistem \
temu kembali (retrieval) dokumen hukum Indonesia.

Dokumen: {judul}
doc_id: {doc_id}
{part_header}
=== LEAF NODE INDEX (dikelompokkan per Bab/Bagian) ===
{leaf_blocks_grouped}

=== INSTRUKSI UTAMA ===

Buat tepat {N} pertanyaan dalam bahasa Indonesia berdasarkan leaf node di atas.
Gunakan HANYA leaf node yang ditampilkan pada prompt ini. Jangan membuat item \
dengan `gold_anchor_node_id` di luar node yang muncul pada prompt ini.

ATURAN WAJIB:
1. Setiap pertanyaan harus SELF-CONTAINED: harus bisa dipahami sendirian tanpa \
percakapan atau query sebelumnya.
2. Setiap pertanyaan harus dijawab oleh TEPAT SATU leaf node (satu anchor). \
Pilih leaf node PALING SPESIFIK yang tersedia: jika sebuah ayat dipecah menjadi \
huruf a, b, c, dst., maka anchor harus salah satu huruf tersebut (bukan ayat \
parent-nya). Jika ayat tidak dipecah lebih lanjut, ayat itu sendiri boleh jadi \
anchor. Jika pasal tidak punya ayat eksplisit, pasal itu sendiri boleh jadi anchor.
3. Jangan buat pertanyaan yang butuh dua atau lebih leaf node untuk dijawab.
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
7. Periksa apakah ada gold_anchor_node_id yang sama pada lebih dari satu item. \
   Jika ada duplikat, HAPUS item yang lebih lemah atau ganti anchornya dengan node \
   lain. Setiap gold_anchor_node_id dalam batch ini harus UNIK.
8. Untuk setiap item berlabel "hard", konfirmasi ada leaf sibling di parent yang sama \
   yang bisa tertukar. Jika tidak ada, turunkan ke "medium".
9. Untuk setiap item, periksa apakah gold_anchor_node_id sudah pada granularity \
   PALING SPESIFIK. Jika leaf node yang dipilih punya sibling huruf/angka, anchor \
   harus di level huruf/angka, bukan di level ayat/pasal parent-nya.

=== JENIS PERTANYAAN (query_style) ===

Ada tepat 2 jenis pertanyaan yang mencerminkan persona pengguna RAG hukum.
Distribusi target: ~50% formal, ~50% colloquial.

1. formal - bahasa hukum resmi, seperti yang digunakan oleh praktisi hukum, konsultan, \
atau dalam dokumen resmi. Diksi presisi, struktur kalimat lengkap. Bisa menyebut \
Pasal/ayat/huruf secara eksplisit, bisa juga tidak.
   Contoh formal + legal_ref: "Apa yang dimaksud dengan 'kebiri kimia' sebagaimana \
diatur dalam Pasal 81 ayat (7)?"
   Contoh formal + none: "Apa tujuan pembentukan P2K3?"
   Contoh formal + doc_only: "Dalam Peraturan Menteri Ketenagakerjaan Nomor 13 Tahun \
2025, siapa yang melakukan pembinaan terhadap P2K3?"

2. colloquial - bahasa sehari-hari orang awam yang tidak berlatar hukum. Boleh \
informal, boleh pakai singkatan, boleh colloquial Indonesian.
   DILARANG menggunakan kata atau frasa yang sama persis dengan teks yang akan kamu \
jadikan answer_hint. Parafrasakan dengan sinonim atau ungkapan berbeda. Bayangkan \
seseorang yang belum membaca dokumen ini mencari informasinya, mereka tidak tahu \
istilah teknisnya. Jika istilah teknis spesifik benar-benar tidak ada sinonimnya, \
boleh disebut, tapi JANGAN menyalin struktur kalimat dari teks node.
   TETAPI tetap self-contained dan uniquely answerable by ONE leaf node.
   `colloquial` TIDAK berarti conversational, underspecified, atau referential.
   Jika query colloquial masih bisa cocok ke lebih dari satu leaf node dalam dokumen \
   yang sama, query itu INVALID dan harus ditulis ulang sampai unik.
   Contoh valid: "Kalau ada yang melakukan kekerasan seksual terhadap anak, hukumannya apa?"
   Contoh valid: "Perusahaan wajib bikin P2K3 itu kalau kondisi karyawannya gimana?"
   Contoh valid: "Berapa lama verifikasi dokumen buat perpanjangan P2K3?"
   Invalid: "Bagaimana aturannya?"
   Invalid: "Kalau ada perubahan, apa yang harus dilakukan?"
   Invalid: "Apa ketentuannya soal itu?"
   Disallowed context-dependent: "Kalau begitu, aturan ini juga berlaku nggak?"

=== TINGKAT KESULITAN (difficulty) ===

Distribusi target: 40% easy, 40% medium, 20% hard.

1. easy - jawabannya ada dalam satu kalimat atau frasa eksplisit. Orang bisa menjawab \
langsung tanpa membaca keseluruhan node.
   Contoh: "Berapa denda maksimal yang dikenakan?" -> jawaban ada dalam satu angka/kalimat.

2. medium - butuh membaca satu ayat atau satu leaf node utuh dan memahami konteksnya. \
Jawaban tidak ada dalam satu kalimat saja, mungkin ada kondisi atau pengecualian.
   Contoh: "Siapa saja yang dapat dikenai pidana tambahan?"

3. hard - query yang mudah tertukar dengan LEAF NODE SIBLING yang lain di parent yang \
sama. WAJIB: Sebelum memberi label "hard", identifikasi leaf node lain di parent yang \
sama yang bisa menjawab query secara plausibel tapi SALAH. Jika tidak ada sibling \
yang bisa tertukar, gunakan "medium" saja.
   CATATAN: Pertanyaan hard tetap harus dijawab oleh SATU leaf node anchor.
   Contoh VALID: Node 0005_a2_h1 vs 0005_a2_h2 sama-sama tentang persyaratan peserta \
dengan kondisi berbeda, query spesifik ke salah satunya.
   Contoh INVALID: "Hard" hanya karena jawabannya panjang atau banyak item, itu medium.

=== FORMAT OUTPUT ===

Kembalikan HANYA JSON array berikut, tanpa markdown, tanpa penjelasan tambahan:

[ 
  {{
    "query": "pertanyaan dalam bahasa Indonesia",
    "query_style": "formal|colloquial",
    "difficulty": "easy|medium|hard",
    "reference_mode": "none|legal_ref|doc_only|both",
    "gold_anchor_granularity": "rincian",
    "gold_anchor_node_id": "node_id leaf PALING SPESIFIK yang menjawab",
    "gold_node_id": "sama dengan gold_anchor_node_id",
    "gold_doc_id": "{doc_id}",
    "navigation_path": "navigation_path dari leaf node tersebut",
    "answer_hint": "kutipan singkat dari teks yang menjadi kunci jawaban, maksimal 100 karakter"
  }}
]

Buat tepat {N} item. Gunakan distribusi query_style ~50% formal dan ~50% colloquial, \
dan difficulty ~40% easy, ~40% medium, ~20% hard. Gunakan distribusi reference_mode yang \
seimbang seperti di atas. Benchmark ini untuk retrieval stateless, jadi query \
harus membantu menguji document discovery, bukan cuma pencocokan kata `Pasal/ayat`. \
Pastikan semua query self-contained, single-hop, tidak terlalu didominasi explicit legal \
references, tidak mencakup metadata/pembukaan, dan semua gold node mengacu ke leaf \
node PALING SPESIFIK yang tersedia (huruf/angka jika ada, ayat jika tidak dipecah \
lebih lanjut, pasal jika tidak punya ayat).
Kembalikan HANYA JSON array. Tidak ada teks lain di luar JSON.
"""


def load_template(query_type: str) -> str:
    """Load a prompt template file from scripts/gt/prompts/<type>.txt."""
    path = PROMPTS_DIR / f"{query_type}.txt"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt template for query_type='{query_type}' not found at {path}. "
            f"Expected one of: {sorted(VALID_QUERY_TYPES)}"
        )
    return path.read_text(encoding="utf-8")


def prompt_template_version(query_type: str = "factual") -> str:
    """Return the SHA-8 hash of the current prompt template for provenance."""
    template = load_template(query_type)
    return hashlib.sha256(template.encode("utf-8")).hexdigest()[:8]


def raw_filename(doc_id: str, query_type: str) -> str:
    """Return the raw GT filename including query type tag."""
    return f"{doc_id}__{query_type}.json"


def meta_filename(doc_id: str, query_type: str) -> str:
    """Return the meta sidecar filename matching raw_filename() conventions."""
    return f"{doc_id}__{query_type}{META_SUFFIX}"


def write_meta_sidecar(category: str, doc_id: str, n_questions: int, total_parts: int, query_type: str = "factual") -> Path:
    """Write a provenance sidecar next to the raw GT placeholder."""
    raw_dir = Path("data/ground_truth_raw") / category
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta_path = raw_dir / meta_filename(doc_id, query_type)
    if meta_path.exists():
        return meta_path

    payload = {
        "doc_id": doc_id,
        "category": category,
        "query_type": query_type,
        "prompt_version": prompt_template_version(query_type),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "n_questions_target": n_questions,
        "n_parts": total_parts,
        "annotator_model": "<FILL>",
        "judge_model": "<FILL>",
        "notes": "",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return meta_path


def load_selection_for(category: str) -> set[str] | None:
    """Return the selected doc_id set for a category, or None if absent."""
    if not SELECTION_FILE.exists():
        return None
    with open(SELECTION_FILE, encoding="utf-8") as f:
        data = json.load(f)
    entry = data.get(category)
    if not entry:
        return None
    return {row["doc_id"] for row in entry.get("selected", [])}


def default_output_path(doc_id: str, query_type: str = "factual") -> Path:
    """Return the default single-prompt path under the repo-local tmp folder."""
    return TMP_DIR / f"gt_{doc_id}__{query_type}.txt"


def default_output_prefix(doc_id: str, query_type: str = "factual") -> Path:
    """Return the default multipart prefix under the repo-local tmp folder."""
    return TMP_DIR / f"gt_{doc_id}__{query_type}"


def manifest_path_from_prefix(prefix: Path) -> Path:
    """Return the manifest path for a multipart prompt run."""
    return prefix.parent / f"{prefix.name}_manifest.json"


def part_path_from_prefix(prefix: Path, part_index: int) -> Path:
    """Return the prompt file path for one part."""
    return prefix.parent / f"{prefix.name}_part{part_index:02d}.txt"


def make_output_target(doc_id: str, out_arg: str | None, multipart: bool, query_type: str = "factual") -> Path:
    """Resolve output file/prefix for single or multipart prompt generation."""
    if not out_arg:
        if multipart:
            return default_output_prefix(doc_id, query_type)
        return default_output_path(doc_id, query_type)

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
            leaf = {
                "node_id": node["node_id"],
                "title": node.get("title", ""),
                "navigation_path": node_path,
                "text": node["text"].strip(),
            }
            if node.get("penjelasan"):
                leaf["penjelasan"] = node["penjelasan"].strip()
            results.append(leaf)
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
    Compute the adaptive question count based on number of rincian-index leaf nodes.

    Returns 0 if leaf_count < MIN_LEAF_FOR_GT, doc too small for meaningful GT.
    Otherwise returns min(leaf_count, 5): cap at 5 keeps annotation diverse;
    n <= leaf_count guarantees no forced anchor reuse.
    """
    if leaf_count < MIN_LEAF_FOR_GT:
        return 0
    return min(leaf_count, 5)


def section_name(leaf: dict) -> str:
    """Return the top-level section for a leaf node."""
    nav = leaf.get("navigation_path", "")
    if " > " in nav:
        return nav.split(" > ")[0].strip()
    return "Lainnya"


def render_leaf_block(leaf: dict) -> str:
    """Render one full, untruncated leaf block for the prompt."""
    block = (
        f"[node_id: {leaf['node_id']}]\n"
        f"Judul: {leaf['title']}\n"
        f"Path: {leaf['navigation_path']}\n"
        f"Teks:\n{leaf['text']}"
    )
    penjelasan = leaf.get("penjelasan", "")
    if penjelasan and penjelasan.strip().lower().rstrip(".") != "cukup jelas":
        block += f"\nPenjelasan:\n{penjelasan}"
    return block


def render_grouped_blocks(leaf_nodes: list[dict]) -> str:
    """
    Render full ayat-index leaves with section headers inserted inline,
    preserving document order (depth-first traversal order from the index).

    A new "--- SECTION ---" header is emitted whenever the section name changes.
    This avoids reordering nodes (e.g. single-leaf pasals would otherwise all
    be batched into one "Lainnya" block that appears before multi-ayat pasals).
    """
    parts: list[str] = []
    current_section: str | None = None
    for leaf in leaf_nodes:
        sec = section_name(leaf)
        if sec != current_section:
            parts.append(f"\n--- {sec} ---\n")
            current_section = sec
        parts.append(render_leaf_block(leaf))
    return "\n".join(parts)


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


def build_prompt(
    doc: dict,
    leaf_nodes: list[dict],
    n_questions: int,
    part_index: int = 1,
    total_parts: int = 1,
    query_type: str = "factual",
    secondary_doc: dict | None = None,
    secondary_leaf_nodes: list[dict] | None = None,
) -> str:
    """Build one prompt for a specific part using the per-type template.

    For crossdoc (paired-doc) the secondary_doc and secondary_leaf_nodes are
    rendered in additional template placeholders. For all other types those
    are ignored.
    """
    template = load_template(query_type)
    fmt_kwargs = {
        "judul": doc["judul"],
        "doc_id": doc["doc_id"],
        "part_header": render_part_header(part_index, total_parts, n_questions, len(leaf_nodes)),
        "leaf_blocks_grouped": render_grouped_blocks(leaf_nodes),
        "N": n_questions,
    }
    if query_type == "crossdoc":
        if secondary_doc is None or secondary_leaf_nodes is None:
            raise ValueError("crossdoc query_type requires secondary_doc and secondary_leaf_nodes")
        fmt_kwargs["judul_secondary"] = secondary_doc["judul"]
        fmt_kwargs["doc_id_secondary"] = secondary_doc["doc_id"]
        fmt_kwargs["leaf_blocks_grouped_secondary"] = render_grouped_blocks(secondary_leaf_nodes)
    return template.format(**fmt_kwargs)


def pack_prompt_parts(
    doc: dict,
    leaf_nodes: list[dict],
    total_questions: int,
    base_budget: int = DEFAULT_PROMPT_CHAR_BUDGET,
    query_type: str = "factual",
    secondary_doc: dict | None = None,
    secondary_leaf_nodes: list[dict] | None = None,
) -> tuple[list[list[dict]], int]:
    """Split the primary doc into prompt parts using whole-node packing.

    The secondary doc (for crossdoc) is included in EVERY part because the
    annotator needs both docs visible to construct comparative queries.
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
            trial_prompt = build_prompt(
                doc, trial, n_questions=1, query_type=query_type,
                secondary_doc=secondary_doc, secondary_leaf_nodes=secondary_leaf_nodes,
            )
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


def build_prompt_parts(
    doc: dict,
    n_questions: int,
    char_budget: int = DEFAULT_PROMPT_CHAR_BUDGET,
    query_type: str = "factual",
    secondary_doc: dict | None = None,
) -> tuple[list[dict], int]:
    """Build single or multipart prompt payloads for one document."""
    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)

    secondary_leaf_nodes: list[dict] | None = None
    if secondary_doc is not None:
        secondary_leaves = collect_leaf_nodes(secondary_doc["structure"])
        secondary_leaf_nodes = filter_preamble(secondary_leaves)

    parts, final_budget = pack_prompt_parts(
        doc, leaf_nodes, n_questions, base_budget=char_budget, query_type=query_type,
        secondary_doc=secondary_doc, secondary_leaf_nodes=secondary_leaf_nodes,
    )
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
            "prompt": build_prompt(
                doc, part_leaves, n_questions=quota,
                part_index=idx, total_parts=total_parts,
                query_type=query_type,
                secondary_doc=secondary_doc,
                secondary_leaf_nodes=secondary_leaf_nodes,
            ),
        })
    return prompt_parts, final_budget


def write_single_prompt(output_path: Path, prompt: str) -> None:
    """Write one prompt file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt, encoding="utf-8")


def write_multipart_prompts(prefix: Path, doc: dict, prompt_parts: list[dict], total_questions: int, final_budget: int, category: str = "") -> tuple[list[Path], Path]:
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
                "expected_raw_part_file": f"data/ground_truth_parts/{category}/{doc['doc_id']}/part{part['part_index']:02d}.json" if category else f"data/ground_truth_parts/{doc['doc_id']}/part{part['part_index']:02d}.json",
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
    ap = argparse.ArgumentParser(description="Generate leaf-anchored GT prompt for ChatGPT")
    ap.add_argument("doc_id", nargs="?", help="Document ID (e.g. perpu-1-2016)")
    ap.add_argument(
        "--questions", "-q", type=int, default=None,
        help="Override question count (default: adaptive min(leaf_count, 5), skips docs with < 5 body leaves)",
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
    ap.add_argument(
        "--allow-unselected", action="store_true",
        help="Allow generating a prompt for a doc that is not in data/gt_doc_selection.json",
    )
    ap.add_argument(
        "--type", "-t", type=str, default="factual",
        choices=sorted(VALID_QUERY_TYPES),
        help="Query type to generate (factual, paraphrased, multihop, crossdoc, adversarial). Default: factual.",
    )
    ap.add_argument(
        "--paired-doc", type=str, default=None,
        help="Secondary doc_id for crossdoc generation. Required when --type crossdoc.",
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

    category = doc_path.parent.name
    selected = load_selection_for(category)
    if selected is not None and args.doc_id not in selected and not args.allow_unselected:
        print(f"ERROR: doc '{args.doc_id}' is NOT in {SELECTION_FILE} for category {category}")
        print("It will act as a distractor in the corpus, not a GT source.")
        print("If this is intentional, pass --allow-unselected.")
        sys.exit(1)

    with open(doc_path, encoding="utf-8") as f:
        doc = json.load(f)

    query_type = args.type
    secondary_doc: dict | None = None
    if query_type == "crossdoc":
        if not args.paired_doc:
            print("ERROR: --type crossdoc requires --paired-doc <secondary_doc_id>")
            sys.exit(1)
        sec_path = find_doc(args.paired_doc)
        if not sec_path:
            print(f"ERROR: paired-doc '{args.paired_doc}' not found in {DATA_INDEX}")
            sys.exit(1)
        with open(sec_path, encoding="utf-8") as f:
            secondary_doc = json.load(f)
    elif args.paired_doc:
        print(f"WARNING: --paired-doc ignored for --type {query_type}")

    leaves = collect_leaf_nodes(doc["structure"])
    leaf_nodes = filter_preamble(leaves)
    adaptive_n = compute_adaptive_n(len(leaf_nodes))
    n_used = args.questions if args.questions is not None else adaptive_n

    if n_used == 0:
        print(f"\nDokumen    : {doc['judul'][:80]}")
        print(f"doc_id     : {doc['doc_id']}")
        print(f"Leaf nodes : {len(leaf_nodes)} body leaf nodes (setelah filter preamble)")
        if len(leaf_nodes) == 0:
            print(f"\n[SKIP] Tidak ada body leaf nodes, kemungkinan hanya preamble.")
        else:
            print(f"\n[SKIP] Hanya {len(leaf_nodes)} leaf nodes (< {MIN_LEAF_FOR_GT}); "
                  f"dokumen terlalu kecil untuk dijadikan GT.")
        sys.exit(0)

    prompt_parts, final_budget = build_prompt_parts(
        doc, n_questions=n_used, char_budget=args.char_budget,
        query_type=query_type, secondary_doc=secondary_doc,
    )
    multipart = len(prompt_parts) > 1

    paired = f", paired={secondary_doc['doc_id']}" if secondary_doc is not None else ""
    override = f" override" if args.questions is not None else ""
    mode = "multipart" if multipart else "single"
    print(f"\n{doc['doc_id']} type={query_type}{paired}, "
          f"{len(leaf_nodes)} leaves, N={n_used}{override} (adaptive {adaptive_n}), "
          f"{mode}, budget={final_budget}")

    if multipart and args.stdout:
        print("\nERROR: Prompt multipart terlalu besar untuk --stdout.")
        print("Jalankan tanpa --stdout agar file part otomatis disimpan ke tmp/.")
        sys.exit(1)

    if not multipart:
        output_path = make_output_target(doc["doc_id"], args.out, multipart=False, query_type=query_type)
        raw_target_name = raw_filename(args.doc_id, query_type)
        raw_dir = Path("data/ground_truth_raw") / category
        raw_dir.mkdir(parents=True, exist_ok=True)
        placeholder = raw_dir / raw_target_name
        meta_path = write_meta_sidecar(category, args.doc_id, n_used, total_parts=1, query_type=query_type)

        if args.stdout:
            print("=" * 70)
            print("COPY PROMPT DI BAWAH INI KE GENERATOR LLM (Claude/GPT, bukan Gemini):")
            print("=" * 70)
            print(prompt_parts[0]["prompt"])
            print("=" * 70)
            print(f"\nRaw target -> {placeholder}")
            print("\nNext.")
            print(f"  1. Paste output JSON to {placeholder}")
            print(f"  2. python scripts/gt/build_validate.py --doc-id {args.doc_id} --type {query_type}")
            print(f"  3. Paste tmp/validate_{args.doc_id}__{query_type}.txt to Judge LLM, paste full response over {placeholder}")
            print(f"  4. python scripts/gt/apply_validation.py --doc-id {args.doc_id} --type {query_type}")
            print(f"  5. python scripts/gt/log_review.py {args.doc_id} --type {query_type}")
            return

        write_single_prompt(output_path, prompt_parts[0]["prompt"])
        if not placeholder.exists():
            placeholder.write_text("[]", encoding="utf-8")

        print("\nCreated.")
        print(f"  Annotator prompt -> {output_path}")
        print(f"  Raw placeholder  -> {placeholder}")
        print(f"  Provenance       -> {meta_path}")
        print("\nNext.")
        print(f"  1. Paste {output_path} to Generator LLM, paste JSON output to {placeholder}")
        print(f"  2. python scripts/gt/build_validate.py --doc-id {args.doc_id} --type {query_type}")
        print(f"  3. Paste tmp/validate_{args.doc_id}__{query_type}.txt to Judge LLM, paste full response over {placeholder}")
        print(f"  4. python scripts/gt/apply_validation.py --doc-id {args.doc_id} --type {query_type}")
        print(f"  5. python scripts/gt/log_review.py {args.doc_id} --type {query_type}")
        return

    prefix = make_output_target(doc["doc_id"], args.out, multipart=True, query_type=query_type)
    part_paths, manifest_path = write_multipart_prompts(prefix, doc, prompt_parts, total_questions=n_used, final_budget=final_budget, category=category)

    parts_basename = f"{doc['doc_id']}__{query_type}"
    parts_dir = Path("data/ground_truth_parts") / category / parts_basename
    parts_dir.mkdir(parents=True, exist_ok=True)
    for part in prompt_parts:
        ph = parts_dir / f"part{part['part_index']:02d}.json"
        if not ph.exists():
            ph.write_text("[]", encoding="utf-8")
    raw_placeholder = Path("data/ground_truth_raw") / category / raw_filename(doc["doc_id"], query_type)
    (Path("data/ground_truth_raw") / category).mkdir(parents=True, exist_ok=True)
    meta_path = write_meta_sidecar(category, doc["doc_id"], n_used, total_parts=len(prompt_parts), query_type=query_type)
    print("\nCreated.")
    print(f"  Multipart prompts -> {len(part_paths)} files in {prefix.parent}")
    for part, path in zip(prompt_parts, part_paths):
        print(f"    - part {part['part_index']:02d}: {path}  ({part['question_quota']} pertanyaan, {part['leaf_count']} leaf)")
    print(f"  Manifest          -> {manifest_path}")
    print(f"  Part placeholders -> {parts_dir}/part01.json ..")
    print(f"  Raw target        -> {raw_placeholder}")
    print(f"  Provenance        -> {meta_path}")
    print("\nNext.")
    print(f"  1. Paste each part to Generator LLM, save outputs to {parts_dir}/part01.json, part02.json, ...")
    print(f"  2. python scripts/gt/merge_parts.py {doc['doc_id']} --type {query_type}")
    print(f"  3. python scripts/gt/build_validate.py --doc-id {doc['doc_id']} --type {query_type}")
    print(f"  4. Paste tmp/validate_{doc['doc_id']}__{query_type}.txt to Judge LLM, paste full response over {raw_placeholder}")
    print(f"  5. python scripts/gt/apply_validation.py --doc-id {doc['doc_id']} --type {query_type}")
    print(f"  6. python scripts/gt/log_review.py {doc['doc_id']} --type {query_type}")


if __name__ == "__main__":
    main()
