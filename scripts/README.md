# Ground Truth Workflow

Operational guide for building ground truth (GT) for retrieval evaluation.

Run all commands from the project root:

```powershell
cd "d:\Fasilkom UI\Kuliah\Semester 8\TA - Skripsi\02 Codebase\vectorless-and-vector-rag-for-indonesian-law"
```

## GT policy

- GT is **leaf-anchored** at the finest available granularity (`rincian` index)
- the anchor must be the **most specific leaf node** available: huruf/angka if present, otherwise ayat, otherwise pasal
- gold sets for coarser granularities (ayat, pasal) are **derived by rolling UP** via prefix lookup in `finalize_gt.py`
- every granularity has **exactly 1 gold node** per question
- main benchmark covers **body text only**
- substantive pasal/ayat content is in scope
- closing provisions in the body such as `mulai berlaku` are in scope if they appear as body nodes
- top-level document metadata is out of scope for the main benchmark
- preamble sections (`Pembukaan`, `Menimbang`, `Mengingat`, `Menetapkan`) are out of scope
- main benchmark queries are **single-hop**
- main benchmark queries must be **self-contained**
- every query must be answerable by **exactly one leaf node**
- `colloquial` queries must still be uniquely answerable by one anchor — vague/underspecified queries are not allowed
- context-dependent or conversational carry-over queries are not allowed
- multihop queries are out of scope for the main benchmark
- document mention is optional
- legal reference mention is optional
- documents with fewer than `MIN_LEAF_FOR_GT` (currently 5) body leaves are auto-skipped by `prompt.py`
- per-doc question count is adaptive: `min(leaf_count, 5)` — no anchor reuse, no duplicates

Methodology notes:

- `gold_node_id` intentionally mirrors `gold_anchor_node_id` in raw GT as a compatibility alias
- cross-granularity gold sets are derived later by `finalize_gt.py` using roll-up prefix lookup
- `answer_hint` is a short evidence snippet for reviewer sanity-checking, not a full gold answer and not the main basis for automated scoring
- GT generation runs through an external Generator LLM and is judged by a separate Judge LLM (both must be a different model family from the Gemini retrieval backbone to avoid self-evaluation bias)
- the prompt fed to the Generator contains raw `text` and `navigation_path` per leaf only — Gemini-generated `summary` fields are deliberately excluded from the GT prompt to keep the bias boundary clean
- consistency is enforced through prompt constraints + LLM-as-Judge validation; structural integrity is enforced by `collect.py`
- main benchmark queries must be single-hop retrieval queries. If answering the query requires combining information from more than one leaf node, the query is invalid for this GT.

For the main GT set, prefer documents with:

- `verify_status.rincian = OK`

If you want a stricter subset later, use documents that are `OK` on all three:

- `pasal = OK`
- `ayat = OK`
- `rincian = OK`

---

## Step 0. Pick eligible documents

Eligible = passed indexing pipeline cleanly. Inspect `data/judge_report.json`
for each candidate doc; prefer verdict `OK` or `MINOR`. `MAJOR` with only
cosmetic OCR issues is acceptable. `FAIL` or `MAJOR` with `coverage.missing`
items must be excluded or replaced.

Indexing artifacts to consult:

- `data/judge_report.json` — overall verdict and per-doc issue list
- `data/granularity_report.json` — splitter sanity (must show 0 suspects)
- `data/indexing_logs/cost_pasal.json` — `ocr_fixes_total`, `ocr_rejected`,
  parse + summary cost per doc

For category-level GT, pick the top 5 docs by judge verdict + topic
diversity (avoid two docs covering the same legal sub-domain).

---

## Step 1. Generate prompt

```powershell
python scripts/gt/prompt.py <doc_id>
```

Output:

- Single file: `tmp/gt_<doc_id>.txt`
- Large document: `tmp/gt_<doc_id>_part01.txt`, `tmp/gt_<doc_id>_part02.txt`, ... + manifest

Notes:

- Reads from `data/index_rincian` — the LLM sees the finest-grained leaf nodes (huruf, angka, or ayat/pasal when not split further).
- If the document has no body leaf nodes (pure preamble), `gt_prompt.py` will print `[SKIP]` and exit.
- Short documents automatically get fewer questions (`n = leaf_count`) — one question per leaf, no duplicates.
- For large documents, do not use `--stdout`; let the files be written to `tmp/`.

Optional flags:

```powershell
python scripts/gt/prompt.py <doc_id> --stdout
python scripts/gt/prompt.py <doc_id> --out "$env:TEMP\gt_<doc_id>.txt"
python scripts/gt/prompt.py --list
```

---

## Step 2. Generate raw GT via Generator LLM

Paste the generated prompt file into your chosen Generator LLM
(Claude / GPT / etc.) — anything **except** the Gemini family, since
Gemini is the retrieval backbone and using it for generation would
introduce self-evaluation bias.

**Single-part document:**

Save the JSON array output to:

```text
data/ground_truth_raw/<KATEGORI>/<doc_id>.json
```

`prompt.py` creates this file as an empty placeholder (`[]`) automatically — overwrite it with the LLM output.

**Multipart document:**

Save each part's JSON array to:

```text
data/ground_truth_parts/<KATEGORI>/<doc_id>/part01.json
data/ground_truth_parts/<KATEGORI>/<doc_id>/part02.json
...
```

Then merge:

```powershell
python scripts/gt/merge_parts.py <doc_id>
```

This writes the merged file to `data/ground_truth_raw/<KATEGORI>/<doc_id>.json`.

---

## Step 3. Build the validation prompt

```powershell
python scripts/gt/build_validate.py --doc-id <doc_id>
```

This runs structural validation (required fields, anchor exists in
`index_rincian`, no duplicate anchors) and **fails fast** if any hard
error is present — fix the raw file and rerun.

On success it writes `tmp/validate_<doc_id>.txt`: a self-contained
validation prompt with the 6 semantic rules + items + the leaf nodes
referenced by those items, all inlined. No external attachments needed.

---

## Step 4. Semantic validation via Judge LLM

Open the `tmp/validate_<doc_id>.txt` file, copy everything, paste to
your chosen Judge LLM (Copilot in IDE / Claude / GPT — again, **not**
Gemini).

The Judge returns:

- A validation JSON with `rejected[]` and `flagged[]`
- A `---CLEANED---` separator followed by the cleaned items array
  (rejected items removed; flagged items auto-fixed per rule actions)

**Apply the cleaned output to the raw file:**

- If using Copilot in IDE: ask it to overwrite
  `data/ground_truth_raw/<KATEGORI>/<doc_id>.json` with the cleaned
  items array directly. Copilot can edit files in place.
- If using a browser Judge (Claude web, ChatGPT): copy the array
  section after `---CLEANED---` and paste it over the raw file's
  contents manually.

---

## Step 5. Merge into ground truth

```powershell
python scripts/gt/collect.py --file data/ground_truth_raw/<doc_id>.json
```

Merge all raw files at once:

```powershell
python scripts/gt/collect.py
```

Output: `data/ground_truth.json`

---

## Step 6. Finalize to evaluation artifact

```powershell
python scripts/gt/finalize.py
```

Output: `data/validated_testset.pkl`

Semantics (roll-up design — every level has exactly 1 gold node):

- `gold_rincian_node_ids`: exact anchor (finest granularity)
- `gold_ayat_node_ids`: derived parent in ayat index
- `gold_pasal_node_ids`: derived parent in pasal index

---

## Step 7. Inspect and sanity-check

```powershell
python scripts/gt/collect.py --stats
python scripts/gt/finalize.py --stats
python scripts/gt/load_testset.py
```

What to check:

- total question count
- documents covered
- balanced `reference_mode` distribution (none / legal_ref / doc_only / both)
- all gold set sizes = 1.0 (roll-up design)
- no metadata-only questions, no preamble, no multi-hop
- `colloquial` items are self-contained and uniquely anchored

---

## Raw GT schema

Each item in `data/ground_truth_raw/<doc_id>.json` must look like:

```json
{
  "query": "...",
  "query_style": "formal|colloquial",
  "difficulty": "easy|medium|hard",
  "reference_mode": "none|legal_ref|doc_only|both",
  "gold_anchor_granularity": "rincian",
  "gold_anchor_node_id": "...",
  "gold_node_id": "...",
  "gold_doc_id": "...",
  "navigation_path": "...",
  "answer_hint": "..."
}
```

Notes:

- `gold_node_id` must match `gold_anchor_node_id`
- `gold_anchor_granularity` must be `rincian`
- `gold_anchor_node_id` must be the most specific leaf available (huruf/angka if present)
- `answer_hint` should stay short and evidence-like; it is not a full canonical answer span
- raw GT must be a JSON array

---

## Reset and rebuild GT from scratch

If you need to start over:

```powershell
# Reset merged GT only (keep raw files)
echo {} > data/ground_truth.json

# Start over from scratch (delete all raw files too)
Remove-Item data\ground_truth_raw\*.json
Remove-Item data\ground_truth_parts -Recurse -Force
echo {} > data/ground_truth.json
Remove-Item data\validated_testset.pkl
```

---

## Recommended habits

- work from docs with `rincian = OK`
- validate structurally before semantic review
- always run semantic validation (Step 4) before merging — never merge directly from ChatGPT output
- merge often, finalize only when the merged GT looks clean
