# Flow Vectorless RAG

## Project Structure

```
vectorless-rag-for-indonesian-law/
  indexing/                  # PDF parsing + index building
    parser.py                # Parse PDF → structured text (BAB, Bagian, Pasal, Ayat)
    build_pasal.py           # Tree JSON, leaf = Pasal
    build_ayat.py            # Tree JSON, leaf = Ayat (lebih granular)
    build_full_split.py      # Tree JSON, leaf = Huruf/Angka (paling granular)
  retrieval/                 # Query → answer pipeline
    common.py                # Shared: LLM client, data loading, tree helpers, answer gen, logging
    bm25_flat.py             # BM25 flat single-stage — *baseline untuk thesis*
    bm25_2stage.py           # BM25 2-stage (doc→node) — *arsitektur lama, tidak dipakai*
    llm.py                   # LLM-only 2-stage (doc search + tree navigation)
    hybrid.py                # Hybrid BM25+LLM 2-stage (union doc search + BM25 candidates + LLM rerank)
    bm25_no_sw.py            # Ablation: BM25 2-stage tanpa stopword removal
    hybrid_no_sw.py          # Ablation: Hybrid tanpa stopword removal
  compare_retrieval.py       # Side-by-side comparison across granularities
  scraper/                   # BPK JDIH scraper (separate, self-contained)
  data/
    raw/                     # Scraper output (metadata + PDFs)
    index_pasal/             # Pasal-level index (20 docs) + catalog.json
    index_ayat/              # Ayat-level index
    index_full_split/        # Full-split index
    retrieval_logs/          # Experiment logs
```

## Phase 1: Indexing (offline, run once)
PDF → `indexing/parser.py` → `indexing/build_*.py` → `data/index_pasal/` (tree JSONs + catalog.json)

**CLI:**
```bash
python -m indexing.build_pasal                        # index all (with LLM cleanup)
python -m indexing.build_pasal --doc-id uu-20-2025    # single document
python -m indexing.build_ayat --force                 # re-index ayat level
```

## Phase 2: Retrieval + Generation (per query)

### 3 Strategi Retrieval Utama (untuk thesis)

#### 1. BM25 Flat (`retrieval/bm25_flat.py`) — Baseline
- **Arsitektur:** Single-stage flat search
- **Corpus:** Semua ~2438 leaf nodes dari semua dokumen sekaligus
- **Metadata enrichment:** Teks Pasal ditempeli judul dokumen + navigation path
- **Tidak pakai catalog doc search** — langsung cari ke semua Pasal
- **Kenapa flat:** BM25 itu keyword matching, memaksakan 2-stage (filter doc dulu) menyebabkan cascading failure saat keyword gak ada di metadata (lexical gap). Flat search = standar IR untuk BM25.

#### 2. LLM-only (`retrieval/llm.py`) — Vectorless Innovation
- **Arsitektur:** 2-stage hierarchical navigation
- **Stage 1 - Doc search:** LLM baca catalog.json, pilih dokumen yang relevan secara semantik
- **Stage 2 - Tree search:** LLM navigasi tree JSON level-by-level (BAB → Bagian → Pasal)
  - Mode `stepwise`: navigasi per level, max 8 rounds
  - Mode `full`: LLM lihat seluruh skeleton tree sekaligus
- **Kenapa 2-stage:** LLM bisa bridge semantic gap ("penyadapan" → "hukum acara pidana"), jadi 2-stage aman.

#### 3. Hybrid BM25+LLM (`retrieval/hybrid.py`) — Best of Both
- **Arsitektur:** 2-stage with combined methods
- **Stage 1 - Doc search:** Union of BM25 catalog hits + LLM semantic picks
- **Stage 2 - Node search:** BM25 retrieves candidate Pasal, LLM reranks with KWIC snippets
- **Kenapa hybrid:** BM25 catches exact keyword matches, LLM catches semantic matches. Union eliminates blind spots.

### Alur per query (umum):
1. **Search** — cari Pasal yang relevan (method beda per strategy)
2. **Generate answer** — kirim retrieved chunks ke LLM (Gemini 2.5 Flash), generate jawaban grounded
3. **Save log** — simpan result ke `data/retrieval_logs/`

**CLI:**
```bash
python -m retrieval.bm25_flat "Apa syarat penyadapan?"
python -m retrieval.llm "Apa syarat penyadapan?" --strategy full
python -m retrieval.hybrid "Apa definisi penyadapan?" --bm25_top_k 10
```

### Catatan: BM25 2-stage (`retrieval/bm25_2stage.py`) — Arsitektur Lama
File ini masih ada tapi **tidak dipakai untuk eksperimen thesis**. Arsitektur 2-stage untuk BM25 punya masalah lexical gap: jika keyword query tidak ada di metadata catalog, doc search gagal dan node search tidak pernah jalan. Contoh: query "Apa syarat penyadapan?" → 0 results karena "penyadapan" tidak ada di judul/subjek dokumen manapun.

BM25 flat (`retrieval/bm25_flat.py`) menggantikan peran ini sebagai baseline BM25 yang fair.

## Preprocessing
Preprocessing cuma dipake di BM25 path (baik flat maupun 2-stage). LLM path dan answer generation tidak ada preprocessing.

| Component | Preprocessing? | Details |
|---|---|---|
| BM25 search (`retrieval/bm25_flat.py`, `retrieval/hybrid.py`) | Yes — `tokenize()` | Tokenize enriched text |
| LLM doc search (`retrieval/llm.py`) | No | Raw text langsung masuk prompt |
| LLM tree search (`retrieval/llm.py`) | No | Raw tree structure masuk prompt |
| Answer generation (`retrieval/common.py`) | No | Raw retrieved text masuk prompt |

Tokenizer:
```python
def tokenize(text: str) -> list[str]:
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
```
3 langkah:
1. **Lowercase** — `text.lower()`
2. **Regex split** — `re.findall(r'[a-z0-9]+', ...)` — split on non-alphanumeric
3. **Stopword removal** — filter ~35 Indonesian stopwords (yang, dan, di, dari, untuk, dengan, pada, dll)

No stemming, no lemmatization.

### Ablation Variants
- **`retrieval/bm25_no_sw.py`** — BM25 2-stage tanpa stopword removal
- **`retrieval/hybrid_no_sw.py`** — Hybrid tanpa stopword removal
- Berdasarkan Faisal et al. (2024): stopword removal bisa menurunkan performa BM25 di domain hukum karena kata seperti "dan"/"atau" punya makna logis (kumulatif vs alternatif)
- Bisa dilaporkan sebagai ablation study di Bab 4

## Bentuk catalog.json
Metadata catalog.json **identik** di semua index (`index_pasal/`, `index_ayat/`, `index_full_split/`):
```json
[
  {
    "doc_id": "perpu-1-2016",
    "judul": "Peraturan Pemerintah Pengganti Undang-Undang...",
    "nomor": "1",
    "tahun": "2016",
    "bentuk_singkat": "Perpu",
    "status": "Berlaku",
    "tanggal_penetapan": "25 Mei 2016",
    "bidang": "",
    "subjek": "HAK ASASI MANUSIA",
    "materi_pokok": "",
    "relasi": [{"jenis": "Mengubah", "judul": "UU No. 23 Tahun 2002", "keterangan": "Pasal 81 dan 82"}],
    "total_pages": 10,
    "element_counts": { "pasal_roman": 2, "angka": 4, "pasal": 4 },
    "jenis_folder": "Perpu"
  }
]
```
Dipake buat: LLM doc search, BM25 flat metadata enrichment.

## Citation System
Answer generation pakai **label-based citations** untuk mencegah LLM mengarang referensi:

1. Setiap retrieved chunk diberi label `[R1]`, `[R2]`, dst.
2. LLM diminta menyisipkan label di jawaban: "Pidana penjara paling singkat 5 tahun [R1]."
3. LLM mengembalikan `"cited": ["R1", "R2"]`
4. System maps label kembali ke `node_id` → `citations` array di log

Contoh output log:
```json
{
  "answer": "Pidana penjara paling singkat 5 tahun [R1]. Hukuman tambahan berupa pengumuman identitas [R2].",
  "citations": [
    {"label": "R1", "node_id": "pasal-81", "title": "Pasal 81", "navigation_path": "BAB ... > Pasal 81"},
    {"label": "R2", "node_id": "pasal-81-ayat-6", "title": "Pasal 81 Ayat (6)", "navigation_path": "..."}
  ]
}
```
Keuntungan: LLM tidak bisa cite "Pasal 99" yang tidak ada di retrieved chunks.

## Bentuk Tree JSON (Index)
Root metadata sama, `structure` beda per granularity:
```json
{
  "doc_id": "perpu-1-2016",
  "judul": "...",
  "nomor": "1", "tahun": "2016",
  "bentuk_singkat": "Perpu",
  "status": "Berlaku",
  "tanggal_penetapan": "25 Mei 2016",
  "relasi": [],
  "element_counts": {...},
  "warnings": [],
  "structure": [...]   // <-- INI YANG BEDA
}
```

| Granularity | Leaf node = | Contoh |
|---|---|---|
| `index_pasal/` | Pasal utuh (semua ayat digabung) | `"text": "(1) Setiap orang... (2) Ketentuan..."` |
| `index_ayat/` | Ayat individual | `"Pasal 81 Ayat (1)" → "text": "(1) Setiap orang..."` |
| `index_full_split/` | Huruf/Angka (paling dalam) | `"Menimbang Huruf a" → "text": "a. bahwa negara..."` |

Semua bagian dokumen masuk index (termasuk Pembukaan/Menimbang/Mengingat). Yang beda cuma granularity pemecahannya:
- **Pasal-level:** Menimbang = 1 blob
- **Ayat/Deep leaf:** Menimbang dipecah per huruf (a, b, c, d)
