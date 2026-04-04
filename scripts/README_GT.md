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

Methodology notes:

- `gold_node_id` intentionally mirrors `gold_anchor_node_id` in raw GT as a compatibility alias
- cross-granularity gold sets are derived later by `finalize_gt.py`
- `answer_hint` is a short evidence snippet for reviewer sanity-checking, not a full gold answer and not the main basis for automated scoring
- GT generation is a curated one-shot annotator workflow: rerunning the same prompt may produce different outputs, so consistency is enforced through prompt constraints, validation, and manual review
- Main benchmark queries must be single-hop retrieval queries. If answering the query requires combining information from more than one ayat/pasal, the query is invalid for this GT.
- Metadata/preamble retrieval and multihop retrieval can be added later as separate benchmark tracks, but they are intentionally excluded from the current main evaluation setup

For the main GT set, prefer documents with:

- `verify_status.ayat = OK`

If you want a stricter subset later, use documents that are `OK` on all three:

- `pasal = OK`
- `ayat = OK`
- `full_split = OK`

## Step 0. Extract candidate docs with ayat = OK

Refresh status first:

```powershell
python -m vectorless.indexing.status --refresh-verify --json > status.json
```

List documents safe for GT main:

```powershell
(Get-Content status.json -Raw | ConvertFrom-Json).docs.PSObject.Properties.Value | Where-Object { $_.verify_status.ayat -eq "OK" } | Select-Object -ExpandProperty doc_id
```

Count them:

```powershell
((Get-Content status.json -Raw | ConvertFrom-Json).docs.PSObject.Properties.Value | Where-Object { $_.verify_status.ayat -eq "OK" }).Count
```

Save them to a helper file:

```powershell
(Get-Content status.json -Raw | ConvertFrom-Json).docs.PSObject.Properties.Value | Where-Object { $_.verify_status.ayat -eq "OK" } | Select-Object -ExpandProperty doc_id | Set-Content gt_ayat_ok.txt
```

Stricter triple-OK subset:

```powershell
(Get-Content status.json -Raw | ConvertFrom-Json).docs.PSObject.Properties.Value | Where-Object { $_.verify_status.pasal -eq "OK" -and $_.verify_status.ayat -eq "OK" -and $_.verify_status.full_split -eq "OK" } | Select-Object -ExpandProperty doc_id
```

Review later subset:

```powershell
(Get-Content status.json -Raw | ConvertFrom-Json).docs.PSObject.Properties.Value | Where-Object { $_.verify_status.ayat -eq "WARN" } | Select-Object -ExpandProperty doc_id
```

## Step 1. Generate prompt for one document

Default behavior:

- if the document is small enough, generate one file: `tmp/gt_<doc_id>.txt`
- if the document is too large, generate multiple full-text prompt files:
  - `tmp/gt_<doc_id>_part01.txt`
  - `tmp/gt_<doc_id>_part02.txt`
  - ...
  - plus a manifest file in `tmp/`

```powershell
python scripts/gt_prompt.py permenaker-1-2026
```

Optional:

```powershell
python scripts/gt_prompt.py permenaker-1-2026 --stdout
python scripts/gt_prompt.py permenaker-1-2026 --out "$env:TEMP\gt_permenaker-1-2026.txt"
python scripts/gt_prompt.py --list
```

Notes:

- Long documents are now split automatically so the annotator always sees full node text.
- Leaf nodes are never truncated and never split across prompt parts.
- If a document becomes multipart, do not use `--stdout`; let the files be written to `tmp/`.
- The main benchmark still excludes metadata-only, preamble-only, and multihop queries even when a long document is split across multiple prompt parts.

## Step 2. Send prompt to LLM and save raw JSON

Recommended workflow:

If output is a single prompt:

1. Open `tmp/gt_<doc_id>.txt`
2. Paste into ChatGPT or another annotator LLM
3. Save the JSON array to:

```text
data/ground_truth_raw/<doc_id>.json
```

If output is multipart:

1. Open each prompt file in `tmp/`
2. Paste each part into the annotator LLM separately
3. Save each JSON array to:

```text
data/ground_truth_parts/<doc_id>/part01.json
data/ground_truth_parts/<doc_id>/part02.json
...
```

4. Merge them:

```powershell
python scripts/merge_gt_parts.py <doc_id>
```

Example:

```powershell
python scripts/merge_gt_parts.py permenaker-13-2025 --pretty
```

## Step 3. Validate raw GT file

```powershell
python scripts/gt_collect.py --check-only --file "data\ground_truth_raw\permenaker-1-2026.json"
```

Expected good output:

- `10 valid, 0 errors, 0 warnings`

If there are warnings:

- `reference_mode mismatch` usually means the label should be edited
- anchor cross-reference warnings should be reviewed, but self-heading false positives are already filtered

## Step 4. Merge into ground truth

Merge one file:

```powershell
python scripts/gt_collect.py --file "data\ground_truth_raw\permenaker-1-2026.json"
```

Merge all raw GT files:

```powershell
python scripts/gt_collect.py
```

Output:

```text
data/ground_truth.json
```

## Step 5. Finalize to evaluation artifact

```powershell
python scripts/finalize_gt.py
```

Output:

```text
data/validated_testset.pkl
```

Semantics:

- `gold_pasal_node_ids`: parent pasal of the ayat anchor
- `gold_ayat_node_ids`: exact ayat anchor
- `gold_full_split_node_ids`: descendant leaves under that ayat anchor

## Step 6. Inspect and sanity-check

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

## Current raw GT schema

Each raw item should look like:

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
- `gold_node_id` mirrors the ayat anchor in raw GT; cross-granularity gold sets are derived later during finalization
- `answer_hint` should stay short and evidence-like; it is not meant to be a full canonical answer span
- raw GT should be saved as a JSON array

## Reset and rebuild GT from scratch

If `ground_truth.json` still contains old experimental items:

```powershell
Remove-Item data\ground_truth.json
python scripts/gt_collect.py
python scripts/finalize_gt.py
```

If you also want to start over from raw GT:

```powershell
Remove-Item data\ground_truth_raw\*.json
Remove-Item data\ground_truth.json
Remove-Item data\validated_testset.pkl
```

## JSON formatting and encoding tips

`gt_collect.py` expects UTF-8 JSON. In PowerShell, avoid using `>` for pretty-printing JSON because it may introduce the wrong encoding.

Safe pretty-print command:

```powershell
$content = python -m json.tool data\ground_truth_raw\permenaker-1-2026.json
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText((Resolve-Path "data\ground_truth_raw\permenaker-1-2026.json"), ($content -join "`n"), $utf8NoBom)
```

Helper script version:

```powershell
python scripts/pretty_json.py data\ground_truth_raw\permenaker-1-2026.json --indent 4
```

Multipart merge helper:

```powershell
python scripts/merge_gt_parts.py permenaker-13-2025 --pretty
```

## Recommended habit

- work from docs with `ayat = OK`
- validate every raw file before merge
- merge often, but finalize only after the merged GT looks clean
- treat old pasal-anchored experiments as disposable unless regenerated
