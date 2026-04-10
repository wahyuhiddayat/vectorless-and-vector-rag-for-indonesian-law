# Ground Truth Workflow

Operational guide for building ground truth (GT) for retrieval evaluation.

Run all commands from the project root:

```powershell
cd "d:\Fasilkom UI\Kuliah\Semester 8\TA - Skripsi\02 Codebase\vectorless-and-vector-rag-for-indonesian-law"
```

## GT policy

- GT is **ayat-anchored**
- main benchmark covers **body text only**
- substantive pasal/ayat content is in scope
- closing provisions in the body such as `mulai berlaku` are in scope if they appear as body nodes
- top-level document metadata is out of scope for the main benchmark
- preamble sections (`Pembukaan`, `Menimbang`, `Mengingat`, `Menetapkan`) are out of scope
- main benchmark queries are **single-hop**
- main benchmark queries must be **self-contained**
- every query must be answerable by **exactly one ayat anchor**
- `colloquial` queries must still be uniquely answerable by one anchor — vague/underspecified queries are not allowed
- context-dependent or conversational carry-over queries are not allowed
- multihop queries are out of scope for the main benchmark
- document mention is optional
- legal reference mention is optional
- documents with zero body leaf nodes (pure preamble) are auto-skipped by `gt_prompt.py`
- short documents generate fewer questions (`n = leaf_count`) — no anchor reuse, no duplicates

Methodology notes:

- `gold_node_id` intentionally mirrors `gold_anchor_node_id` in raw GT as a compatibility alias
- cross-granularity gold sets are derived later by `finalize_gt.py`
- `answer_hint` is a short evidence snippet for reviewer sanity-checking, not a full gold answer and not the main basis for automated scoring
- GT generation is a curated one-shot annotator workflow: rerunning the same prompt may produce different outputs, so consistency is enforced through prompt constraints, validation, and manual review
- Main benchmark queries must be single-hop retrieval queries. If answering the query requires combining information from more than one ayat/pasal, the query is invalid for this GT.

For the main GT set, prefer documents with:

- `verify_status.ayat = OK`

If you want a stricter subset later, use documents that are `OK` on all three:

- `pasal = OK`
- `ayat = OK`
- `full_split = OK`

---

## Step 0. Find eligible documents

Refresh index status:

```powershell
python -m vectorless.indexing.status --refresh-verify --json > status.json
```

Save eligible doc IDs to `gt_eligible.txt` (criterion: all three granularities verified OK):

```powershell
(Get-Content status.json -Raw | ConvertFrom-Json).docs.PSObject.Properties.Value | Where-Object { $_.verify_status.pasal -eq "OK" -and $_.verify_status.ayat -eq "OK" -and $_.verify_status.full_split -eq "OK" } | Select-Object -ExpandProperty doc_id | Set-Content gt_eligible.txt
```

`gt_eligible.txt` is your working list. Only generate GT for documents in this file.

Documents that are not triple-OK can be re-indexed later (after fixing the parser) without affecting already-OK documents — registry merge only updates the re-indexed doc.

---

## Step 1. Generate prompt

```powershell
python scripts/gt_prompt.py <doc_id>
```

Output:

- Single file: `tmp/gt_<doc_id>.txt`
- Large document: `tmp/gt_<doc_id>_part01.txt`, `tmp/gt_<doc_id>_part02.txt`, ... + manifest

Notes:

- If the document has no body leaf nodes (pure preamble), `gt_prompt.py` will print `[SKIP]` and exit.
- Short documents automatically get fewer questions (`n = leaf_count`) — one question per leaf, no duplicates.
- For large documents, do not use `--stdout`; let the files be written to `tmp/`.

Optional flags:

```powershell
python scripts/gt_prompt.py <doc_id> --stdout
python scripts/gt_prompt.py <doc_id> --out "$env:TEMP\gt_<doc_id>.txt"
python scripts/gt_prompt.py --list
```

---

## Step 2. Send prompt to ChatGPT and save raw JSON

Open the generated prompt file and paste it into ChatGPT.

**Single-part document:**

Save the JSON array output to:

```text
data/ground_truth_raw/<doc_id>.json
```

**Multipart document:**

Save each part's JSON array to:

```text
data/ground_truth_parts/<doc_id>/part01.json
data/ground_truth_parts/<doc_id>/part02.json
...
```

Then merge:

```powershell
python scripts/merge_gt_parts.py <doc_id>
```

This writes the merged file to `data/ground_truth_raw/<doc_id>.json`.

---

## Step 3. Structural validation

```powershell
python scripts/gt_collect.py --check-only --file data/ground_truth_raw/<doc_id>.json
```

Fix any hard errors in the raw JSON file before proceeding. Copy the `[WARN]` lines from the output — you will need them in the next step.

Expected clean output: `N valid, 0 errors, 0 warnings`

---

## Step 4. Semantic validation with Copilot

Use `scripts/gt_validate_prompt.txt` as a template (the file is gitignored — customize locally).

In Copilot Chat:

1. **Attach** `data/index_ayat/[KATEGORI]/<doc_id>.json` so Copilot can verify node text and sibling structure.
2. Fill in `[KATEGORI]` and `[DOC_ID]` in the template header.
3. Paste the raw JSON array from `data/ground_truth_raw/<doc_id>.json` into the `[PASTE ...]` placeholder.
4. Paste the `[WARN]` lines from Step 3 below the JSON (as additional hints).
5. Send to Copilot.

Copilot will return:

- A validation JSON with `rejected[]` and `flagged[]`
- A `---CLEANED---` section: the JSON array with rejected items removed and flagged items auto-fixed

**Replace the raw file with the cleaned array:**

Copy everything after `---CLEANED---` and overwrite `data/ground_truth_raw/<doc_id>.json` with it.

---

## Step 5. Merge into ground truth

```powershell
python scripts/gt_collect.py --file data/ground_truth_raw/<doc_id>.json
```

Merge all raw files at once:

```powershell
python scripts/gt_collect.py
```

Output: `data/ground_truth.json`

---

## Step 6. Finalize to evaluation artifact

```powershell
python scripts/finalize_gt.py
```

Output: `data/validated_testset.pkl`

Semantics:

- `gold_pasal_node_ids`: parent pasal of the ayat anchor
- `gold_ayat_node_ids`: exact ayat anchor
- `gold_full_split_node_ids`: descendant leaves under that ayat anchor

---

## Step 7. Inspect and sanity-check

```powershell
python scripts/gt_collect.py --stats
python scripts/finalize_gt.py --stats
python scripts/load_testset.py
```

What to check:

- total question count
- documents covered
- balanced `reference_mode`
- balanced `query_style` (~50% formal, ~50% colloquial)
- average gold set sizes make sense
- no metadata-only questions
- no preamble questions
- no multihop queries
- `colloquial` items are still self-contained and uniquely anchored

---

## Raw GT schema

Each item in `data/ground_truth_raw/<doc_id>.json` must look like:

```json
{
  "query": "...",
  "query_style": "formal|colloquial",
  "difficulty": "easy|medium|hard",
  "reference_mode": "none|legal_ref|doc_only|both",
  "gold_anchor_granularity": "ayat",
  "gold_anchor_node_id": "...",
  "gold_node_id": "...",
  "gold_doc_id": "...",
  "navigation_path": "...",
  "answer_hint": "..."
}
```

Notes:

- `gold_node_id` must match `gold_anchor_node_id`
- `gold_anchor_granularity` must be `ayat`
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

- work from docs with `ayat = OK`
- validate structurally before semantic review
- always run semantic validation (Step 4) before merging — never merge directly from ChatGPT output
- merge often, finalize only when the merged GT looks clean
