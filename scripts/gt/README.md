# Ground Truth Workflow (v2, typed-query benchmark)

Operational guide for building the GT testset used in retrieval evaluation.

Run all commands from the project root.

```bash
cd "d:/Fasilkom UI/Kuliah/Semester 8/TA - Skripsi/02 Codebase/vectorless-and-vector-rag-for-indonesian-law"
```

For the design rationale (5 query types, 4-layer validation, multi-anchor rollup), see [`Notes/02-ground-truth/design_v2.md`](../../../Notes/02-ground-truth/design_v2.md).

## Pipeline at a glance

```
0. SELECT       select_gt_docs.py        pick 5 of 10 docs per category (stratified random)
0.5 ALLOCATE    allocate_quotas.py       distribute per-type quotas across selected docs
1. PROMPT       prompt.py --type <t>     emit annotator prompt + provenance sidecar
2. ANNOTATE     Generator LLM (manual)   paste prompt, save JSON output
3. MERGE        merge_parts.py           combine multipart outputs (only if parts > 1)
4. VALIDATE     build_validate.py        Layer 1 struct + Layer 2 deterministic + Judge prompt
5. JUDGE        Judge LLM (manual)       semantic clean of items
6. APPLY        apply_validation.py      Layer 1 re-check on cleaned array, overwrite raw
7. REVIEW       log_review.py            Layer 4 author per-doc spot-check
8. COLLECT      collect.py               struct-validate + merge into ground_truth.json
9. FINALIZE     finalize.py              roll up to 3 granularities, write pkl
```

Steps 2 and 5 are external LLM calls. All other steps are scripts.

## Five query types

| Type | Anchor count | Stress-tests |
|---|:-:|---|
| `factual` | 1 | Literal lookup baseline |
| `paraphrased` | 1 | Dense embedding vs BM25 separation |
| `multihop` | 2 (same doc, different pasal) | Hierarchical tree navigation |
| `crossdoc` | 2 (different docs in same category) | Catalog-level retrieval |
| `adversarial` | 1 target + distractor | Cross-encoder reranking |

Per-type rules and gates are detailed in `Notes/02-ground-truth/design_v2.md`.

## GT policy summary

- GT is leaf-anchored at `rincian` (finest granularity).
- Single-anchor queries roll up to gold size 1 at every granularity.
- Multi-anchor queries roll up to gold size 1..N (collapses if anchors share parent).
- Body text only. Preamble (`Menimbang`, `Mengingat`, `Menetapkan`, `Pembukaan`) and top-level metadata are out of scope.
- Annotator and Judge LLM must be a different model family from the Gemini retrieval backbone, and ideally different from each other (cross-family judge per design v2).

## Step 0. Select GT-source docs

Stratified random pick by leaf count, persistent seed.

```bash
python scripts/gt/select_gt_docs.py --category UU --seed 42
python scripts/gt/select_gt_docs.py --all --seed 42
python scripts/gt/select_gt_docs.py --show
```

Output, `data/gt_doc_selection.json`. The 5 picked docs are GT sources, the 5 unpicked stay in the corpus as distractors.

## Step 0.5. Allocate per-type quotas

```bash
python scripts/gt/allocate_quotas.py --category UU --seed 42 --emit-commands
```

Reads selected docs from Step 0. Greedy fill with rotating per-type offsets. Crossdoc pairing 4-tier priority (manual override, same `subjek`, same `bidang`, round-robin). Output, `data/gt_allocation.json`. With `--emit-commands` the script prints ready-to-paste invocations for Step 1.

## Step 1. Generate annotator prompt

```bash
python scripts/gt/prompt.py uu-1-2026 --type factual --questions 8
python scripts/gt/prompt.py uu-1-2026 --type paraphrased --questions 6
python scripts/gt/prompt.py uu-1-2026 --type multihop --questions 6
python scripts/gt/prompt.py uu-1-2026 --type crossdoc --paired-doc uu-2-2026 --questions 3
python scripts/gt/prompt.py uu-1-2026 --type adversarial --questions 2
```

Outputs.

- Single doc, `tmp/gt_<doc_id>(__<type>).txt`
- Long doc, `tmp/gt_<doc_id>(__<type>)_part01.txt`, `..._part02.txt`, `..._manifest.json`
- Empty placeholder JSON at `data/ground_truth_raw/<CAT>/<doc_id>(__<type>).json`
- Provenance sidecar at `data/ground_truth_raw/<CAT>/<doc_id>(__<type>).meta.json`

Notes.

- Reads from `data/index_rincian`.
- Doc must appear in `gt_doc_selection.json` for its category. Pass `--allow-unselected` to bypass.
- Templates live under `scripts/gt/prompts/<type>.txt`.
- `prompt_version` in the sidecar is the SHA-8 of the resolved template. If you edit the template, the hash changes and old GT becomes a different version.
- `--type crossdoc` requires `--paired-doc <secondary_doc_id>`.

## Step 2. Run the Generator LLM

Paste the generated prompt file into Claude or GPT (anything except the Gemini family, since Gemini is the retrieval backbone).

Save the JSON array output to:

- Single, `data/ground_truth_raw/<CAT>/<doc_id>(__<type>).json` (overwrite the placeholder).
- Multipart, `data/ground_truth_parts/<CAT>/<doc_id>(__<type>)/part01.json`, `part02.json`, ...

Then update the sidecar field `annotator_model` with the model name and version.

## Step 3. Merge multipart parts (only if multipart)

```bash
python scripts/gt/merge_parts.py <doc_id>
```

Output, `data/ground_truth_raw/<CAT>/<doc_id>(__<type>).json`.

## Step 4. Build validation prompt (Layer 1 + Layer 2 + Judge prompt)

```bash
python scripts/gt/build_validate.py --doc-id uu-1-2026 --type factual
python scripts/gt/build_validate.py --doc-id uu-1-2026 --type paraphrased
python scripts/gt/build_validate.py --doc-id uu-1-2026 --type adversarial
```

The script.

1. Runs Layer 1 struct check from `collect.py`. Hard-gates if any structural error exists.
2. Runs the per-type Layer 2 deterministic gate, paraphrase Jaccard for `paraphrased`, BM25 cascade rank for `adversarial`. Other types skip Layer 2. Hard-gates on any flag.
3. Inlines items + leaf-node context for **every** anchor (both anchors for multihop, both docs for crossdoc) into the type-aware rules from `validate_prompt.txt`.
4. Writes the assembled prompt to `tmp/validate_<doc_id>(__<type>).txt`.

Pass `--skip-layer2` only for diagnostic reruns when you intentionally want to bypass the deterministic gate.

## Step 5. Run the Judge LLM

Paste `tmp/validate_<doc_id>(__<type>).txt` into Claude or GPT (must differ from the Generator and must not be Gemini, cross-family judge per design v2).

The Judge returns a validation summary, then the `---CLEANED---` separator, then a JSON array of cleaned items, then `---END---`.

Save the full Judge response to `tmp/judge_<doc_id>(__<type>).txt`.

## Step 6. Apply Judge output through the gate

```bash
python scripts/gt/apply_validation.py \
    --doc-id uu-1-2026 \
    --type multihop \
    --judge-file tmp/judge_uu-1-2026__multihop.txt
```

The gate.

1. Extracts the array after `---CLEANED---`.
2. Runs Layer 1 structural validation against `data/index_rincian`.
3. On pass, overwrites `data/ground_truth_raw/<CAT>/<doc_id>(__<type>).json` and saves the previous file under `.bak/<doc_id>(__<type>).<timestamp>.json`.
4. On fail, prints errors and exits 1 without touching the raw file.

Pass `--dry-run` to validate only. Pass `--stdin` if you prefer pipe input.

After applying, update sidecar field `judge_model`.

Never overwrite the raw GT file by hand or via the Judge directly. Always go through `apply_validation.py` so the struct gate catches malformed JSON.

## Step 7. Author spot-check (Layer 4)

Run once per (doc, type), immediately after the Judge step while the doc is still fresh in your head.

```bash
python scripts/gt/log_review.py uu-1-2026 --type factual
python scripts/gt/log_review.py uu-1-2026 --type multihop
```

For each item the script prints query, every anchor (with its leaf text and navigation path), distractor (if adversarial), answer hint, then prompts for a verdict.

- `c` correct
- `w` wrong (will be dropped at collect step)
- `b` borderline (kept, recorded for the methodology section)
- `s` skipped (recorded explicitly, not silent)
- `q` quit (saves partial progress, resumable)

Output, `data/gt_audit/<doc_id>(__<type>).json`.

Aggregate report across all docs and types.

```bash
python scripts/gt/log_review.py --report
python scripts/gt/log_review.py --report --json
```

Output, `data/gt_audit/_summary.json`.

## Step 8. Collect into the merged GT

```bash
python scripts/gt/collect.py --file data/ground_truth_raw/<CAT>/<doc_id>(__<type>).json
python scripts/gt/collect.py                    # process every raw file
python scripts/gt/collect.py --check-only       # validate, do not write
python scripts/gt/collect.py --stats            # show distribution stats
```

Behavior.

- Hard structural validation per item (required fields, anchor exists in `data/index_rincian`, per-type anchor count, `gold_anchor_node_ids` required for multi-anchor types, no within-batch duplicate anchors).
- Items flagged `wrong` in the matching `data/gt_audit/<doc_id>(__<type>).json` are dropped.
- Cross-batch deduplication of queries (case-insensitive) and anchor key `(doc_id, anchor_node_id, query_type)`. Anchor reuse across types is allowed by design (factual and paraphrased intentionally share anchors).
- Output, `data/ground_truth.json` keyed by `q001`, `q002`, ...

## Step 9. Finalize the testset

```bash
python scripts/gt/finalize.py
```

Output, `data/validated_testset.pkl`. Each item gets gold sets at all 3 granularities (pasal, ayat, rincian), unioned across every anchor.

Inspect.

```bash
python scripts/gt/load_testset.py
python scripts/gt/load_testset.py --stats
python scripts/gt/load_testset.py --doc <doc_id>
python scripts/gt/load_testset.py --query "<keyword>"
```

`--stats` prints reference_mode and per-category cross-tab so you can spot starved sub-tasks early.

## Batch orchestrator (Step 4 + Step 6 over the whole allocation)

When the Judge runs in your IDE (Copilot, Codex, Claude Code), it can process
many prompt files in one workspace pass. `run_allocation.py` walks
`gt_allocation.json` and runs the build phase then the apply phase in bulk.

```bash
# State matrix for every (doc, type) in the plan
python scripts/gt/run_allocation.py
python scripts/gt/run_allocation.py --category UU

# Build all Judge prompts at once (Layer 1 + Layer 2 + emit prompt)
python scripts/gt/run_allocation.py --build --category UU

# (manual) Tell the IDE Judge to process every tmp/validate_*.txt and write
#         the cleaned output to tmp/judge_<same-name>.txt next to it.

# Apply every tmp/judge_*.txt through the struct gate
python scripts/gt/run_allocation.py --apply --category UU
```

Filter by `--type <factual|paraphrased|multihop|crossdoc|adversarial>` to scope
either phase. Both phases continue past per-item failures and report counts at
the end. State is derived from filesystem, so the orchestrator is safely
re-runnable.

## Files in this directory

| File | Role |
|---|---|
| `select_gt_docs.py` | Stratified random doc selection (Step 0) |
| `allocate_quotas.py` | Per-type quota allocation and crossdoc pairing (Step 0.5) |
| `prompt.py` | Generator prompt + provenance sidecar (Step 1) |
| `prompts/<type>.txt` | Per-type annotator templates |
| `merge_parts.py` | Combine multipart outputs (Step 3) |
| `build_validate.py` | Layer 1 + Layer 2 + Judge prompt assembly (Step 4) |
| `validators/paraphrase_overlap.py` | Layer 2 Jaccard gate for `paraphrased` |
| `validators/adversarial_bm25.py` | Layer 2 BM25 cascade gate for `adversarial` |
| `validate_prompt.txt` | Type-aware semantic rules used by Judge (Step 5) |
| `apply_validation.py` | Judge output gate (Step 6) |
| `log_review.py` | Author spot-check logger (Step 7) |
| `collect.py` | Struct-validate + merge (Step 8) |
| `finalize.py` | Roll-up to 3 granularities (Step 9) |
| `run_allocation.py` | Batch orchestrator over Step 4 + Step 6 |
| `load_testset.py` | Inspect the final pkl |
| `build_catalog.py` | Filter index catalogs to GT-only doc set |

## Reset

```bash
# Reset merged GT only, keep raw and audit
echo {} > data/ground_truth.json

# Full reset, keep selection and allocation
rm -rf data/ground_truth_raw/*/*.json data/ground_truth_raw/*/*.meta.json
rm -rf data/ground_truth_parts data/gt_audit
echo {} > data/ground_truth.json
rm -f data/validated_testset.pkl

# Drop selection and allocation too (regenerate from scratch)
rm -f data/gt_doc_selection.json data/gt_allocation.json
```

## Raw GT schema

```json
{
  "query": "...",
  "query_type": "factual|paraphrased|multihop|crossdoc|adversarial",
  "query_style": "formal|colloquial",
  "difficulty": "easy|medium|hard",
  "reference_mode": "none|legal_ref|doc_only|both",
  "gold_anchor_granularity": "rincian",
  "gold_anchor_node_id": "...",
  "gold_anchor_node_ids": ["..."],
  "gold_node_id": "...",
  "gold_doc_id": "...",
  "gold_doc_ids": ["..."],
  "navigation_path": "...",
  "answer_hint": "...",
  "distractor_node_id": "..."
}
```

`gold_anchor_node_ids` is **required** for `multihop` and `crossdoc`, optional otherwise (defaults to `[gold_anchor_node_id]`). `gold_doc_ids` mirrors `gold_anchor_node_ids`. `distractor_node_id` is required for `adversarial`. `answer_hint` is an evidence snippet for reviewer sanity checking, not a full canonical answer.

## Habits

- Generate one (category, type) at a time. Validate, judge, review, collect before moving on.
- Always run `log_review.py` while the doc context is still in your head.
- Never paste Judge output directly to the raw file. Always go through `apply_validation.py`.
- Keep `gt_doc_selection.json` and `gt_allocation.json` under version control if you start over with a different seed.
- After every change to `data/`, push the HF dataset backup (see project CLAUDE.md).
