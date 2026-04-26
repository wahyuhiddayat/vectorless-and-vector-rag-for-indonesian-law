# Ground Truth Workflow

Operational guide for building the GT testset used in retrieval evaluation.

Run all commands from the project root.

```powershell
cd "d:\Fasilkom UI\Kuliah\Semester 8\TA - Skripsi\02 Codebase\vectorless-and-vector-rag-for-indonesian-law"
```

For the design rationale (anchor granularity, roll-up, fairness arguments), see [`Notes/design/Ground Truth Design.md`](../../../Notes/design/Ground%20Truth%20Design.md).

## Pipeline at a glance

```
0. SELECT      select_gt_docs.py        pick 5 of 10 docs per category (stratified random)
1. PROMPT      prompt.py                generate annotator prompt + provenance sidecar
2. ANNOTATE    Generator LLM (manual)   paste prompt, get JSON back
3. MERGE       merge_parts.py           combine multipart outputs (only if parts > 1)
4. VALIDATE    build_validate.py        emit Judge LLM prompt
5. JUDGE       Judge LLM (manual)       semantic clean of items
6. APPLY       apply_validation.py      gate Judge output, overwrite raw file
7. REVIEW      log_review.py            author per-doc spot-check
8. COLLECT     collect.py               struct-validate + merge into ground_truth.json
9. FINALIZE    finalize.py              roll up to 3 granularities, write pkl
```

Steps 2 and 5 are external LLM calls. Steps 0, 1, 3, 4, 6, 7, 8, 9 are scripts.

## GT policy summary

- GT is leaf-anchored at `rincian` (finest granularity).
- The anchor must be the most specific leaf available, huruf or angka if present, otherwise ayat, otherwise pasal.
- Gold sets for ayat and pasal are derived by rolling UP via prefix lookup. Every granularity has exactly 1 gold node per question.
- Body text only. Preamble (`Menimbang`, `Mengingat`, `Menetapkan`, `Pembukaan`) and top-level metadata are out of scope.
- Single-hop only. Multi-hop is out of scope for the main benchmark.
- Queries must be self-contained. No referential pronouns ("aturan ini", "ketentuan tersebut") without antecedent.
- Annotator and Judge LLM must be a different model family from the Gemini retrieval backbone.

## Step 0. Select GT-source docs

Stratified random pick by leaf count, persistent seed.

```powershell
python scripts/gt/select_gt_docs.py --category PMK --seed 42
python scripts/gt/select_gt_docs.py --all --seed 42
python scripts/gt/select_gt_docs.py --show
```

Output, `data/gt_doc_selection.json`. The 5 picked docs are GT sources, the 5 unpicked stay in the corpus as distractors.

## Step 1. Generate prompt

```powershell
python scripts/gt/prompt.py <doc_id>
```

Outputs.

- Single doc, `tmp/gt_<doc_id>.txt`
- Long doc, `tmp/gt_<doc_id>_part01.txt`, `..._part02.txt`, `..._manifest.json`
- Empty placeholder JSON at `data/ground_truth_raw/<CAT>/<doc_id>.json`
- Provenance sidecar at `data/ground_truth_raw/<CAT>/<doc_id>.meta.json`

Notes.

- Reads from `data/index_rincian`.
- Doc must appear in `gt_doc_selection.json` for its category. Pass `--allow-unselected` to bypass.
- Adaptive question count, `n = min(leaf_count, 5)`. Docs with fewer than 5 body leaves are skipped automatically.
- `prompt_version` recorded in the sidecar is the SHA-8 of `PROMPT_TEMPLATE`. If you edit the template, the hash changes and old GT becomes a different version.

## Step 2. Run the Generator LLM

Paste the generated prompt file into Claude or ChatGPT (anything except the Gemini family, since Gemini is the retrieval backbone).

Save the JSON array output to:

- Single, `data/ground_truth_raw/<CAT>/<doc_id>.json` (overwrite the placeholder).
- Multipart, `data/ground_truth_parts/<CAT>/<doc_id>/part01.json`, `part02.json`, ...

Then update the sidecar field `annotator_model` with the model name and version.

## Step 3. Merge multipart parts (only if multipart)

```powershell
python scripts/gt/merge_parts.py <doc_id>
```

Output, `data/ground_truth_raw/<CAT>/<doc_id>.json` (merged JSON array).

## Step 4. Build validation prompt

```powershell
python scripts/gt/build_validate.py --doc-id <doc_id>
```

Runs the same struct check as `collect.py` and emits a self-contained validation prompt at `tmp/validate_<doc_id>.txt`. Hard-gates if structural errors exist.

The prompt inlines the 6 semantic rules from `validate_prompt.txt`, the items, and the leaf nodes referenced by those items.

## Step 5. Run the Judge LLM

Paste `tmp/validate_<doc_id>.txt` into Copilot, Claude, or ChatGPT (must differ from the Generator model and must not be Gemini).

The Judge returns a validation JSON, then a `---CLEANED---` separator, then a JSON array of cleaned items.

## Step 6. Apply Judge output through the gate

```powershell
python scripts/gt/apply_validation.py --doc-id <doc_id> --judge-file <judge_output.txt>
```

The gate.

1. Extracts the array after `---CLEANED---`.
2. Runs structural validation against `data/index_rincian`.
3. On pass, overwrites `data/ground_truth_raw/<CAT>/<doc_id>.json` and saves the previous file under `.bak/`.
4. On fail, prints errors and exits 1 without touching the raw file.

Pass `--dry-run` to validate only. Pass `--stdin` if you prefer pipe input.

After applying, update sidecar field `judge_model`.

## Step 7. Author spot-check

Run once per doc, immediately after the Judge step while the doc is still fresh in your head.

```powershell
python scripts/gt/log_review.py <doc_id>
```

For each item, the script prints query, anchor, navigation path, answer hint, and full leaf text, then prompts for a verdict.

- `c` correct
- `w` wrong (will be dropped at collect step)
- `b` borderline (kept, recorded for the methodology section)
- `s` skipped (recorded explicitly, not silent)
- `q` quit (saves partial progress, resumable)

Output, `data/gt_audit/<doc_id>.json`.

Aggregate report across all docs.

```powershell
python scripts/gt/log_review.py --report
python scripts/gt/log_review.py --report --json
```

Output, `data/gt_audit/_summary.json`.

## Step 8. Collect into the merged GT

```powershell
python scripts/gt/collect.py --file data/ground_truth_raw/<CAT>/<doc_id>.json
python scripts/gt/collect.py                    # process every raw file
python scripts/gt/collect.py --check-only       # validate, do not write
python scripts/gt/collect.py --stats            # show distribution stats
```

Behavior.

- Hard structural validation per item (required fields, anchor exists in `data/index_rincian`, no duplicate anchors).
- Items flagged `wrong` in `data/gt_audit/<doc_id>.json` are dropped.
- Cross-batch deduplication of queries (case-insensitive) and anchors.
- Output, `data/ground_truth.json` keyed by `q001`, `q002`, ...

## Step 9. Finalize the testset

```powershell
python scripts/gt/finalize.py
```

Output, `data/validated_testset.pkl`. Each item gets gold sets at all 3 granularities derived from the rincian anchor via prefix lookup.

Inspect.

```powershell
python scripts/gt/load_testset.py
python scripts/gt/load_testset.py --stats
python scripts/gt/load_testset.py --doc <doc_id>
python scripts/gt/load_testset.py --query "<keyword>"
```

`--stats` prints reference_mode and per-category cross-tab so you can spot starved sub-tasks early.

## Files in this directory

| File | Role |
|---|---|
| `select_gt_docs.py` | Stratified random doc selection (Step 0) |
| `prompt.py` | Generator prompt + provenance sidecar (Step 1) |
| `merge_parts.py` | Combine multipart outputs (Step 3) |
| `build_validate.py` | Assemble Judge prompt (Step 4) |
| `validate_prompt.txt` | Static semantic rules used by Judge |
| `apply_validation.py` | Judge output gate (Step 6) |
| `log_review.py` | Author spot-check logger (Step 7) |
| `collect.py` | Struct-validate + merge (Step 8) |
| `finalize.py` | Roll-up to 3 granularities (Step 9) |
| `load_testset.py` | Inspect the final pkl |
| `build_catalog.py` | Filter index catalogs to GT-only doc set |

## Reset

```powershell
# Reset merged GT only, keep raw and audit
echo {} > data/ground_truth.json

# Full reset, keep selection file
Remove-Item data\ground_truth_raw\*\*.json
Remove-Item data\ground_truth_raw\*\*.meta.json
Remove-Item data\ground_truth_parts -Recurse -Force
Remove-Item data\gt_audit -Recurse -Force
echo {} > data/ground_truth.json
Remove-Item data\validated_testset.pkl

# Drop selection too (regenerate from scratch)
Remove-Item data\gt_doc_selection.json
```

## Raw GT schema

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

`gold_node_id` mirrors `gold_anchor_node_id` for compatibility with older code paths. `answer_hint` is an evidence snippet for reviewer sanity checking, not a full canonical answer.

## Habits

- Generate one category at a time. Validate, judge, review, collect before moving to the next.
- Always run `log_review.py` while the doc context is still in your head.
- Never paste Judge output directly to the raw file. Always go through `apply_validation.py` so the struct gate catches malformed JSON.
- Keep `gt_doc_selection.json` under version control if you start over with a different seed.
- After every change to `data/`, push the HF dataset backup (see project CLAUDE.md).
