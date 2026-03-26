# Vectorless and Vector RAG for Indonesian Law

Thesis project: comparing vectorless RAG (hierarchical tree navigation) with vector RAG (dense retrieval) for Indonesian legal question answering.

## Project Structure

```
vectorless-and-vector-rag-for-indonesian-law/
  scraper/              # BPK JDIH document acquisition (see scraper/README.md)
  vectorless/           # Vectorless RAG system
    indexing/           # PDF parsing + index building
    retrieval/          # Query -> answer pipelines (BM25, LLM, hybrid)
  vector/               # Vector RAG system (comparison baseline)
  data/                 # Shared data layer (gitignored)
    raw/                # Scraper output: metadata JSONs + PDFs
    index_pasal/        # Pasal-level index + catalog.json
    index_ayat/         # Ayat-level index
    index_full_split/   # Full-split index (finest granularity)
    retrieval_logs/     # Experiment result logs
```

All commands run from project root using `python -m`.

---

## 1. Scraper

See [scraper/README.md](scraper/README.md) for full documentation.

```bash
# Scrape UU pages 1-5 with PDFs
python scraper/bpk_scraper.py --jenis 8 --pages 1-5

# Resume interrupted scraping
python scraper/bpk_scraper.py --jenis 8 --resume
```

---

## 2. Indexing (Vectorless)

Parses scraped PDFs into hierarchical tree indices. Single command with `--granularity` flag:

```bash
python -m vectorless.indexing.build --granularity <pasal|ayat|full_split>
```

| Granularity | Leaf node = | Output |
|-------------|-------------|--------|
| `pasal` | Pasal (coarsest) | `data/index_pasal/` |
| `ayat` | Ayat (mid) | `data/index_ayat/` |
| `full_split` | Huruf/Angka (finest) | `data/index_full_split/` |

### Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--granularity` | *(required)* | Leaf node granularity: `pasal`, `ayat`, or `full_split` |
| `--doc-id ID` | all docs | Operate on a single document, e.g. `--doc-id uu-20-2025` |
| `--parse-only` | off | Pass 1 only: PDF parsing, no LLM. Use when iterating parser fixes |
| `--llm-only` | off | Pass 2 only: LLM cleanup on already-parsed docs. Resumes after network failure |
| `--rebuild WHAT` | skip existing | What to rebuild: `all`, `uncleaned`, or comma-separated doc_ids |
| `--from-pasal` | off | Re-split from existing pasal index (no PDF parsing, no LLM). Only for `ayat`/`full_split` |
| `--full-pipeline` | off | Run complete pipeline: pasal parse+LLM → ayat resplit → full_split resplit → verify |
| `--no-llm` | — | *(legacy)* Alias for `--parse-only` |
| `--force` | — | *(legacy)* Alias for `--rebuild all` |

### Examples

```bash
# Index everything at Pasal level (recommended for final thesis data)
python -m vectorless.indexing.build --granularity pasal

# Full pipeline in one command: parse, clean, resplit all three granularities, verify
python -m vectorless.indexing.build --granularity pasal --full-pipeline

# Two-pass workflow: parse first, then run LLM cleanup separately
python -m vectorless.indexing.build --granularity pasal --parse-only
python -m vectorless.indexing.build --granularity pasal --llm-only

# Quick test: parse one doc without LLM cleanup
python -m vectorless.indexing.build --granularity pasal --doc-id uu-20-2025 --parse-only

# Re-index specific docs after a parser fix
python -m vectorless.indexing.build --granularity pasal --rebuild uu-20-2025,uu-1-2026

# Re-index all docs
python -m vectorless.indexing.build --granularity pasal --rebuild all

# Fast: re-split ayat/full_split from existing pasal index (~0.2s, no LLM)
python -m vectorless.indexing.build --granularity ayat --from-pasal --rebuild all
python -m vectorless.indexing.build --granularity full_split --from-pasal --rebuild all
```

### Parser pipeline (internal)

`vectorless/indexing/parser.py` processes each PDF in 8 stages:

1. **Text extraction & cleaning** — PyMuPDF extraction, two-column gazette layout reorder, OCR artifact fixes
2. **Penjelasan detection & parsing** — locate PENJELASAN section, fix column-stacking OCR artifacts, attach to tree
3. **Structural element detection** — regex-based heading detection (BAB / Bagian / Paragraf / Pasal)
4. **Pasal numbering validation** — sequence checks, gap/jump detection
5. **Tree building** — stack-based tree assembly, preamble splitting (Menimbang / Mengingat / Menetapkan), boundary fixes
6. **LLM text cleanup** — Gemini batch OCR correction (~50K chars/batch)
7. **Main pipeline** — top-level orchestration (`parse_legal_pdf`)
8. **Sub-Pasal leaf splitting** — Ayat and deep (huruf/angka) granularity expansion

### Verification

Verify structural integrity of indexed documents:

```bash
python -m vectorless.indexing.verify --granularity pasal
python -m vectorless.indexing.verify --all                    # all 3 granularities + cross-compare
python -m vectorless.indexing.verify --granularity pasal --doc-id uu-20-2025
python -m vectorless.indexing.verify --granularity pasal --json
```

---

## 3. Retrieval (Vectorless)

Six retrieval strategies, each in its own file. All share the same data from `data/index_*/`.

### Switching granularity

All retrieval modules read from `data/index_pasal` by default. Switch granularity with the DATA_INDEX env var:

```bash
# Default (pasal)
python -m vectorless.retrieval.bm25_flat "query"

# Ayat-level index
DATA_INDEX=data/index_ayat python -m vectorless.retrieval.bm25_flat "query"

# Full-split index
DATA_INDEX=data/index_full_split python -m vectorless.retrieval.hybrid "query"
```

For Python API usage, set the env var **before** importing:
```python
import os
os.environ["DATA_INDEX"] = "data/index_ayat"
from vectorless.retrieval.bm25_flat import retrieve  # picks up env var at import time
```

### 3.1 BM25 Flat (single-stage keyword search)

Searches ALL leaf nodes from ALL documents at once using BM25 scoring.

```bash
python -m vectorless.retrieval.bm25_flat "Apa syarat penyadapan?"
python -m vectorless.retrieval.bm25_flat "Apa syarat penyadapan?" --top_k 10
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--top_k N` | `5` | How many Pasal results to return |

### 3.2 LLM-only (semantic tree navigation)

LLM navigates the document tree to find relevant Pasal. No keyword matching.

```bash
python -m vectorless.retrieval.llm "Apa syarat penyadapan?"
python -m vectorless.retrieval.llm "Apa syarat penyadapan?" --strategy full
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--strategy` | `stepwise` | `stepwise` = navigate level-by-level (BAB -> Bagian -> Pasal). `full` = LLM sees all Pasal at once and picks |

### 3.3 Hybrid (BM25 + LLM)

BM25 finds keyword-relevant candidates, LLM reranks using text snippets.

```bash
python -m vectorless.retrieval.hybrid "Apa syarat penyadapan?"
python -m vectorless.retrieval.hybrid "Apa syarat penyadapan?" --bm25_top_k 15
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--bm25_top_k N` | `10` | How many BM25 candidates to pass to LLM for reranking. The LLM then picks 1-3 from these |

### 3.4 Ablations

Same as above but without stopword removal (testing if stopwords help or hurt BM25 for Indonesian legal text):

```bash
python -m vectorless.retrieval.bm25_no_sw "Apa syarat penyadapan?"
python -m vectorless.retrieval.hybrid_no_sw "Apa syarat penyadapan?"
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--top_k_docs N` | `3` | Max documents from doc-level search |
| `--top_k_nodes N` | `5` | Max Pasal nodes from node-level search |

### 3.5 BM25 2-Stage (legacy, not used in experiments)

Two-stage BM25: first find relevant documents, then search within. Superseded by BM25 flat.

```bash
python -m vectorless.retrieval.bm25_2stage "Apa syarat penyadapan?"
```

Same flags as ablations (`--top_k_docs`, `--top_k_nodes`).

---

## 4. Retrieval (Vector)

Vector RAG baseline using Qdrant. Requires Qdrant running on `localhost:6333`.

### 4.1 Index documents into Qdrant

```bash
python -m vector.index_vector
```

### 4.2 Dense retrieval

Pure embedding similarity search.

```bash
python -m vector.retrieve_vector "Apa syarat penyadapan?"
python -m vector.retrieve_vector "Apa syarat penyadapan?" --top_k 10
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--top_k N` | `5` | How many results to return |

### 4.3 Hybrid (BM25 + dense)

Combines BM25 sparse scoring with dense embedding scoring.

```bash
python -m vector.retrieve_vector_hybrid "Apa syarat penyadapan?"
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--top_k N` | `5` | Top-K results per method (BM25 and dense), then merged |

---

## 5. Using retrieval from Python (for experiments)

All retrieval modules export a `retrieve()` function. For thesis experiments, call this directly instead of using CLI:

```python
from vectorless.retrieval.bm25_flat import retrieve
result = retrieve("Apa syarat penyadapan?", top_k=5)

print(result["answer"])        # LLM-generated answer
print(result["sources"])       # Retrieved Pasal with scores
print(result["metrics"])       # Token usage, elapsed time
```

Compare strategies programmatically:

```python
from vectorless.retrieval import bm25_flat, hybrid, llm

strategies = {
    "bm25_flat": lambda q: bm25_flat.retrieve(q, top_k=5),
    "hybrid": lambda q: hybrid.retrieve(q, bm25_top_k=10),
    "llm_full": lambda q: llm.retrieve(q, strategy="full"),
}

for name, fn in strategies.items():
    result = fn("Apa syarat penyadapan?")
    print(f"{name}: {len(result.get('sources', []))} sources, "
          f"{result['metrics']['total_tokens']} tokens")
```

---

## Environment

Requires `.env` at project root:

```
GEMINI_API_KEY=your_api_key_here
```

Optional env vars:

| Variable | Default | What it does |
|----------|---------|--------------|
| `DATA_INDEX` | `data/index_pasal` | Which granularity index retrieval modules read from. Set to `data/index_ayat` or `data/index_full_split` to switch. |