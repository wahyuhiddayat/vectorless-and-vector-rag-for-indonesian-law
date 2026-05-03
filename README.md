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
    index_status.json   # Central manifest for indexing progress/status
    index_pasal/        # Pasal-level index + catalog.json
    index_ayat/         # Ayat-level index
    index_rincian/   # Rincian index (finest granularity)
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

# Generate a Markdown report of the newest documents per category
python scraper/bpk_topk_newest.py --k 3
```

---

## 2. Indexing (Vectorless)

Parses scraped PDFs into hierarchical tree indices. Single command with `--granularity` flag:

```bash
python -m vectorless.indexing.build --granularity <pasal|ayat|rincian>
```

For the day-to-day operational guide, see [vectorless/indexing/README.md](vectorless/indexing/README.md).

| Granularity | Leaf node = | Output |
|-------------|-------------|--------|
| `pasal` | Pasal (coarsest) | `data/index_pasal/` |
| `ayat` | Ayat (mid) | `data/index_ayat/` |
| `rincian` | Huruf/Angka (finest) | `data/index_rincian/` |

### Flags

| Flag | Default | What it does |
|------|---------|--------------|
| `--granularity` | *(required)* | Leaf node granularity: `pasal`, `ayat`, or `rincian` |
| `--doc-id ID` | all docs | Operate on a single document, e.g. `--doc-id uu-20-2025` |
| `--category CAT` | all categories | Filter docs by category/folder, e.g. `UU`, `PP`, `PMK`, `PERMENAKER` |
| `--parse-only` | off | Pass 1 only: PDF parsing, no LLM. Use when iterating parser fixes |
| `--llm-only` | off | Pass 2 only: LLM cleanup on already-parsed docs. Resumes after network failure |
| `--rebuild WHAT` | skip existing | What to rebuild: `all`, `uncleaned`, `stale`, or comma-separated doc_ids |
| `--from-pasal` | off | Re-split from existing pasal index (no PDF parsing, no LLM). Only for `ayat`/`rincian` |
| `--full-pipeline` | off | Run complete pipeline: pasal parse+LLM â†’ ayat resplit â†’ rincian resplit â†’ verify |
| `--no-llm` | â€” | *(legacy)* Alias for `--parse-only` |
| `--force` | â€” | *(legacy)* Alias for `--rebuild all` |

### Examples

```bash
# Index everything at Pasal level (recommended for final thesis data)
python -m vectorless.indexing.build --granularity pasal

# Full pipeline in one command: parse, clean, resplit all three granularities, verify
python -m vectorless.indexing.build --granularity pasal --full-pipeline

# Two-pass workflow: parse first, then run LLM cleanup separately
python -m vectorless.indexing.build --granularity pasal --parse-only
python -m vectorless.indexing.build --granularity pasal --llm-only --rebuild uncleaned

# Quick test: parse one doc without LLM cleanup
python -m vectorless.indexing.build --granularity pasal --doc-id uu-20-2025 --parse-only

# Run per category
python -m vectorless.indexing.build --granularity pasal --category PMK --llm-only --rebuild uncleaned
python -m vectorless.indexing.build --granularity ayat --category PMK --from-pasal --rebuild stale
python -m vectorless.indexing.build --granularity rincian --category PMK --from-pasal --rebuild stale
python -m vectorless.indexing.status --category PMK --refresh-verify

# Re-index specific docs after a parser fix
python -m vectorless.indexing.build --granularity pasal --rebuild uu-20-2025,uu-1-2026

# Re-index all docs
python -m vectorless.indexing.build --granularity pasal --rebuild all

# Re-index only stale docs after a parser-version bump
python -m vectorless.indexing.build --granularity pasal --rebuild stale

# Fast: re-split ayat/rincian from existing pasal index (~0.2s, no LLM)
python -m vectorless.indexing.build --granularity ayat --from-pasal --rebuild stale
python -m vectorless.indexing.build --granularity rincian --from-pasal --rebuild stale

# Inspect central indexing status
python -m vectorless.indexing.status
python -m vectorless.indexing.status --refresh-verify
```

### Incremental workflow

The indexing pipeline now maintains `data/index_status.json` as the operational
source of truth for parse progress, LLM cleanup, stale derived outputs, and
verify summaries.

```bash
# 1) Parse everything offline first
python -m vectorless.indexing.build --granularity pasal --parse-only

# 2) Spend Gemini budget only on docs that still need cleanup
python -m vectorless.indexing.build --granularity pasal --llm-only --rebuild uncleaned

# 3) After parser fixes, rebuild only stale pasal docs
python -m vectorless.indexing.build --granularity pasal --rebuild stale

# 4) Then refresh derived granularities from pasal only
python -m vectorless.indexing.build --granularity ayat --from-pasal --rebuild stale
python -m vectorless.indexing.build --granularity rincian --from-pasal --rebuild stale
```

LLM cleanup now defaults to conservative settings for reliability:

- sequential batches by default (`VECTORLESS_LLM_MAX_WORKERS=1`)
- smaller batch size (`VECTORLESS_LLM_BATCH_SIZE=20000`)
- per-batch fresh Gemini model handles

IMPORTANT TEMPORARY NOTE:

- LLM cleanup currently uses `google-generativeai` as a workaround because the
  newer `google.genai` SDK was timing out repeatedly in the current Windows
  environment even on tiny requests.
- This is an operational workaround for thesis progress, not the preferred
  long-term SDK choice.
- Install it if needed with:

```powershell
python -m pip install google-generativeai
```

If the connection is stable and you want more speed, you can tune it explicitly:

```powershell
$env:VECTORLESS_LLM_MAX_WORKERS="2"
$env:VECTORLESS_LLM_BATCH_SIZE="30000"
python -m vectorless.indexing.build --granularity pasal --category PMK --llm-only --rebuild uncleaned
```

### Parser pipeline (internal)

`vectorless/indexing/parser.py` processes each PDF in 8 stages:

1. **Text extraction & cleaning** â€” PyMuPDF extraction, two-column gazette layout reorder, OCR artifact fixes
2. **Penjelasan detection & parsing** â€” locate PENJELASAN section, fix column-stacking OCR artifacts, attach to tree
3. **Structural element detection** â€” regex-based heading detection (BAB / Bagian / Paragraf / Pasal)
4. **Pasal numbering validation** â€” sequence checks, gap/jump detection
5. **Tree building** â€” stack-based tree assembly, preamble splitting (Menimbang / Mengingat / Menetapkan), boundary fixes
6. **LLM text cleanup** â€” Gemini batch OCR correction (~50K chars/batch)
7. **Main pipeline** â€” top-level orchestration (`parse_legal_pdf`)
8. **Sub-Pasal leaf splitting** â€” Ayat and deep (huruf/angka) granularity expansion

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

Three retrieval strategies are kept for the final experiments: BM25 flat, pure
LLM navigation, and hybrid BM25+LLM. All share the same data from
`data/index_*/`.

### Switching granularity

All retrieval modules read from `data/index_pasal` by default. Switch granularity with the DATA_INDEX env var:

```bash
# Default (pasal)
python -m vectorless.retrieval.bm25.flat "query"

# Ayat-level index
DATA_INDEX=data/index_ayat python -m vectorless.retrieval.bm25.flat "query"

# Rincian index
DATA_INDEX=data/index_rincian python -m vectorless.retrieval.hybrid.search "query"
```

For Python API usage, set the env var **before** importing:
```python
import os
os.environ["DATA_INDEX"] = "data/index_ayat"
from vectorless.retrieval.bm25.flat import retrieve  # picks up env var at import time
```

### 3.1 BM25 Flat (single-stage keyword search)

Searches ALL leaf nodes from ALL documents at once using BM25 scoring.

```bash
python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?"
python -m vectorless.retrieval.bm25.flat "Apa syarat penyadapan?" --top_k 10
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--top_k N` | `5` | How many Pasal results to return |

### 3.2 LLM-only (semantic tree navigation)

LLM navigates the document tree to find relevant Pasal. No keyword matching.

```bash
python -m vectorless.retrieval.llm.search "Apa syarat penyadapan?"
python -m vectorless.retrieval.llm.search "Apa syarat penyadapan?" --strategy full
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--strategy` | `stepwise` | `stepwise` = navigate level-by-level (BAB -> Bagian -> Pasal). `full` = LLM sees all Pasal at once and picks |

### 3.3 Hybrid (BM25 + LLM)

BM25 finds keyword-relevant candidates, LLM reranks using text snippets.

```bash
python -m vectorless.retrieval.hybrid.search "Apa syarat penyadapan?"
python -m vectorless.retrieval.hybrid.search "Apa syarat penyadapan?" --bm25_top_k 15
```

| Flag | Default | What it does |
|------|---------|--------------|
| `query` | *(required)* | Legal question in Indonesian |
| `--bm25_top_k N` | `10` | How many BM25 candidates to pass to LLM for reranking. The LLM then picks 1-3 from these |

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

---

## 5. Using retrieval from Python (for experiments)

All retrieval modules export a `retrieve()` function. For thesis experiments, call this directly instead of using CLI:

```python
from vectorless.retrieval.bm25.flat import retrieve
result = retrieve("Apa syarat penyadapan?", top_k=5)

print(result["answer"])        # LLM-generated answer
print(result["sources"])       # Retrieved Pasal with scores
print(result["metrics"])       # Token usage, elapsed time
```

Compare strategies programmatically:

```python
from vectorless.retrieval.bm25 import flat as bm25_flat
from vectorless.retrieval.hybrid import search as hybrid
from vectorless.retrieval.llm import search as llm

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

## 6. Ground Truth Workflow

Ground truth annotation uses **rincian-index leaves as the semantic anchor**
(the finest available granularity: huruf, angka, or ayat/pasal if no finer split
exists). Gold sets for coarser granularities (ayat, pasal) are derived by
**rolling UP** to parent nodes via prefix lookup.

Each benchmark query must be **self-contained**: vague wording is allowed, but
context-dependent/coreferential queries are not part of the main benchmark.
The main benchmark is also intentionally limited to **single-hop retrieval over
body text**: substantive pasal/ayat content and closing provisions in the body
are in scope, while top-level metadata and preamble sections
(`Pembukaan` / `Menimbang` / `Mengingat` / `Menetapkan`) are out of scope.
If answering a query requires combining information from more than one
leaf node, the query is invalid for the main GT and should be treated as a
future secondary benchmark instead.
The main benchmark should also use a **balanced mix** of query reference styles:
no legal reference, legal reference only, document only, and both.
For the detailed operational guide, see [scripts/README_GT.md](scripts/README_GT.md).
By default, `gt_prompt.py` now writes either:

- one full-text prompt file to `tmp/gt_<doc_id>.txt`, or
- multiple full-text prompt files plus a manifest in `tmp/` for long documents.

Long GT prompts are no longer built by truncating node text.

```bash
# List available documents from data/index_rincian
python scripts/gt_prompt.py --list

# Generate prompt for ChatGPT/Claude
python scripts/gt_prompt.py perpu-1-2016
python scripts/gt_prompt.py perpu-1-2016 --out %TEMP%\\gt_perpu-1-2016.txt
python scripts/gt_prompt.py perpu-1-2016 --stdout

# Merge multipart raw GT outputs when a doc was split into several prompt parts
python scripts/merge_gt_parts.py permenaker-13-2025 --pretty

# Validate raw annotation files
python scripts/gt_collect.py --check-only

# Merge validated items
python scripts/gt_collect.py

# Build multi-granularity evaluation testset (roll-up: rincian -> ayat -> pasal)
python scripts/finalize_gt.py

# Inspect the final pickle
python scripts/load_testset.py --stats

# Evaluate current vectorless systems on the validated GT
python scripts/evaluate_vectorless.py
python scripts/evaluate_vectorless.py --doc-id permenaker-1-2026 --query-limit 5 --verbose
python scripts/evaluate_vectorless.py --systems bm25,hybrid --granularities ayat
```

Semantics of `validated_testset.pkl` (roll-up design):

- `reference_mode` - one of `none`, `legal_ref`, `doc_only`, `both`
- `gold_rincian_node_ids` - exact anchor (1 node, finest granularity)
- `gold_ayat_node_ids` - derived parent in ayat index (1 node)
- `gold_pasal_node_ids` - derived parent in pasal index (1 node)

Every granularity has exactly 1 gold node per question, ensuring fair evaluation:
harder at finer granularity (larger corpus, same single target). The anchor is
always the most specific leaf available (huruf/angka if the pasal is split that
deep, otherwise ayat or pasal). Each query stays self-contained, single-hop, and
answerable by one leaf node.

---

## Environment

The pipeline uses three LLM vendors by role (OpenAI for parsing and GT
judge, Anthropic for GT annotator and parser judge fallback, Vertex AI
Gemini for high-volume summary/OCR/retrieval). Setup all three:

```bash
# OpenAI + Anthropic API keys go in .env
cp .env.example .env
# Edit .env and fill OPENAI_API_KEY + ANTHROPIC_API_KEY

# Vertex AI uses Application Default Credentials, one-time login
gcloud auth application-default login
gcloud config set project YOUR_GCP_PROJECT_ID
```

Required `.env`:

```
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GOOGLE_CLOUD_PROJECT=your_gcp_project_id
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=True
```

### LLM distribution by role

| Role | Vendor | Model | Why |
|---|---|---|---|
| Parser | OpenAI | `gpt-5` | Top-tier reasoning for PDF structure |
| Summary | Vertex AI | `gemini-2.5-flash-lite` | High-volume, low-stakes; trial-covered |
| OCR clean | Vertex AI | `gemini-2.5-flash-lite` | Cheap text fix; trial-covered |
| Retrieval LLM | Vertex AI | `gemini-2.5-flash-lite` | Largest token volume in eval; trial-covered |
| Parser judge | Vertex AI | `gemini-2.5-pro` | Cross-family from OpenAI parser |
| GT annotator | Anthropic | `claude-sonnet-4-6` | Cross-family from Gemini retrieval LLM |
| GT judge | OpenAI | `gpt-5` | Cross-family from Anthropic annotator |

Bias-free comparison boundaries: Parser ≠ Parser-judge, Annotator ≠ Retrieval LLM, Annotator ≠ GT-judge.

Optional env vars:

| Variable | Default | What it does |
|----------|---------|--------------|
| `DATA_INDEX` | `data/index_pasal` | Which granularity index retrieval modules read from. Set to `data/index_ayat` or `data/index_rincian` to switch. |

Evaluation artifacts are written to `data/eval_runs/<timestamp>_<label>/` and include:

- `config.json`
- `per_query.jsonl`
- `summary_overall.json`
- `summary_by_system_granularity.csv`
- `summary_by_slice.csv`
