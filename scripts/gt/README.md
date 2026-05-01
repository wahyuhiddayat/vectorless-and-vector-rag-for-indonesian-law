# Ground Truth Workflow (v3, 3-type stratified benchmark)

Operational guide for building the GT testset used in retrieval evaluation.

Run all commands from the project root.

```bash
cd "d:/Fasilkom UI/Kuliah/Semester 8/TA - Skripsi/02 Codebase/vectorless-and-vector-rag-for-indonesian-law"
```

For the design rationale (3 query types, 4-layer validation, multi-anchor rollup), see [`Notes/02-ground-truth/design_v3.md`](../../../Notes/02-ground-truth/design_v3.md). The v2 → v3 scope-down decision is documented in [`Notes/06-decisions/2026-05-01-3type-gt-benchmark.md`](../../../Notes/06-decisions/2026-05-01-3type-gt-benchmark.md).

## Pipeline at a glance

```
0. SELECT       select_gt_docs.py        pick 5 of 10 docs per category (stratified random)
0.5 ALLOCATE    allocate_quotas.py       distribute per-type quotas across selected docs
1. PROMPT       prompt.py --type <t>     emit annotator prompt; refresh prompt SHA-8 di data/gt_provenance.json
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

## Three query types

| Type | Anchor count | Stress-tests | Target ratio |
|---|:-:|---|:-:|
| `factual` | 1 | Literal lookup baseline | ~33% (9 of 25) |
| `paraphrased` | 1 | Dense embedding vs BM25 separation | ~33% (8 of 25) |
| `multihop` | 2 (same doc, different pasal) | Hierarchical tree navigation | ~34% (8 of 25) |

Per-type rules and gates are detailed in `Notes/02-ground-truth/design_v3.md`. `crossdoc` and `adversarial` (from v2) dropped per ADR-002 — adversarial requires deferred reranker, crossdoc adds limited retrieval-skill information beyond multihop.

## GT policy summary

- GT is leaf-anchored at `rincian` (finest granularity).
- Single-anchor queries roll up to gold size 1 at every granularity.
- Multi-anchor queries roll up to gold size 1..N (collapses if anchors share parent).
- Body text only. Preamble (`Menimbang`, `Mengingat`, `Menetapkan`, `Pembukaan`) and top-level metadata are out of scope.
- Annotator and Judge LLM must be a different model family from the Gemini retrieval backbone, and different from each other (cross-family judge per design v3). Default: annotator Claude Sonnet 4.6, judge GPT-5.

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

Reads selected docs from Step 0. Greedy fill with rotating per-type offsets, per-doc cap 5 queries. Type ratio target equal split ~33/33/34 (factual 9 / paraphrased 8 / multihop 8 of 25 per category), turunkan kuota per-type kalau doc affordance tidak mendukung dan compensate dengan type lain. Output, `data/gt_allocation.json` mencatat realized distribution. With `--emit-commands` the script prints ready-to-paste invocations for Step 1.

## Step 1. Generate annotator prompt

```bash
python scripts/gt/prompt.py uu-1-2026 --type factual --questions 2
python scripts/gt/prompt.py uu-1-2026 --type paraphrased --questions 2
python scripts/gt/prompt.py uu-1-2026 --type multihop --questions 1
```

Outputs.

- Single doc, `tmp/gt_<doc_id>__<type>.txt`
- Long doc, `tmp/gt_<doc_id>__<type>_part01.txt`, `..._part02.txt`, `..._manifest.json`
- Empty placeholder JSON at `data/ground_truth_raw/<CAT>/<doc_id>__<type>.json`
- Refresh `prompt_versions.<type>` SHA-8 di `data/gt_provenance.json` (auto kalau template berubah)

Notes.

- Reads from `data/index_rincian`.
- Doc must appear in `gt_doc_selection.json` for its category. Pass `--allow-unselected` to bypass.
- Templates live under `scripts/gt/prompts/<type>.txt`.
- `prompt_versions.<type>` di `data/gt_provenance.json` = SHA-8 resolved template. Kalau edit template, hash berubah → GT lama linguistic-version berbeda dari yang baru di-generate.

## Step 2. Run the Generator LLM

Paste the generated prompt file into Claude or GPT (anything except the Gemini family, since Gemini is the retrieval backbone).

Save the JSON array output to:

- Single, `data/ground_truth_raw/<CAT>/<doc_id>__<type>.json` (overwrite the placeholder).
- Multipart, `data/ground_truth_parts/<CAT>/<doc_id>__<type>/part01.json`, `part02.json`, ...

Default annotator model (Claude Sonnet 4.6) sudah pinned di `data/gt_provenance.json` — tidak perlu update per-file. Kalau swap model untuk batch tertentu, tambah entry ke `gt_provenance.json` `overrides` array.

## Step 3. Merge multipart parts (only if multipart)

```bash
python scripts/gt/merge_parts.py <doc_id> --type <type>
```

Output, `data/ground_truth_raw/<CAT>/<doc_id>__<type>.json`.

## Step 4. Build validation prompt (Layer 1 + Layer 2 + Judge prompt)

```bash
python scripts/gt/build_validate.py --doc-id uu-1-2026 --type factual
python scripts/gt/build_validate.py --doc-id uu-1-2026 --type paraphrased
python scripts/gt/build_validate.py --doc-id uu-1-2026 --type multihop
```

The script.

1. Runs Layer 1 struct check from `collect.py`. Hard-gates if any structural error exists.
2. Runs the per-type Layer 2 deterministic gate, paraphrase Jaccard for `paraphrased`. Other types skip Layer 2. Hard-gates on any flag.
3. Inlines items + leaf-node context for **every** anchor (both anchors for multihop) into the type-aware rules from `validate_prompt.txt`.
4. Writes the assembled prompt to `tmp/validate_<doc_id>__<type>.txt`.

Pass `--skip-layer2` only for diagnostic reruns when you intentionally want to bypass the deterministic gate.

## Step 5. Run the Judge LLM

Paste `tmp/validate_<doc_id>__<type>.txt` into GPT-5 (judge default, cross-family from Claude annotator and from Gemini retrieval backbone).

The Judge returns a validation summary, the `---CLEANED---` separator, the JSON array of cleaned items, then `---END---`.

Paste the **full Judge response** (framing, JSON, summary prose, all of it) directly over `data/ground_truth_raw/<CAT>/<doc_id>__<type>.json`. Do not strip the framing, the gate handles that.

## Step 6. Apply Judge output through the gate

```bash
python scripts/gt/apply_validation.py --doc-id uu-1-2026 --type multihop
```

The gate reads the dirty raw file, extracts the array after `---CLEANED---`, runs Layer 1 structural validation against `data/index_rincian`, backs up the dirty raw under `.bak/<doc_id>__<type>.<timestamp>.json`, then rewrites the raw file as pure JSON.

Alternative input modes.

```bash
# Read from a separate file the Judge wrote to
python scripts/gt/apply_validation.py --doc-id <id> --type <t> --judge-file tmp/judge_<id>.txt

# Pipe via stdin
python scripts/gt/apply_validation.py --doc-id <id> --type <t> --stdin < paste.txt

# Validate only, do not write
python scripts/gt/apply_validation.py --doc-id <id> --type <t> --dry-run
```

If validation fails the raw file is left untouched, fix the items in the raw file (or rerun the Judge) and try again.

Default judge model (GPT-5) sudah pinned di `data/gt_provenance.json` — tidak perlu update per-file. Kalau swap model untuk batch tertentu, tambah entry ke `gt_provenance.json` `overrides` array.

## Step 7. Author spot-check (Layer 4)

Run once per (doc, type), immediately after the Judge step while the doc is still fresh in your head.

```bash
python scripts/gt/log_review.py uu-1-2026 --type factual
python scripts/gt/log_review.py uu-1-2026 --type multihop
```

For each item the script prints query, every anchor (with its leaf text and navigation path), answer hint, then prompts for a verdict.

- `c` correct
- `w` wrong (will be dropped at collect step)
- `b` borderline (kept, recorded for the methodology section)
- `s` skipped (recorded explicitly, not silent)
- `q` quit (saves partial progress, resumable)

Output, `data/gt_audit/<doc_id>__<type>.json`.

Aggregate report across all docs and types.

```bash
python scripts/gt/log_review.py --report
python scripts/gt/log_review.py --report --json
```

Output, `data/gt_audit/_summary.json`.

## Step 8. Collect into the merged GT

```bash
python scripts/gt/collect.py --file data/ground_truth_raw/<CAT>/<doc_id>__<type>.json
python scripts/gt/collect.py                    # process every raw file
python scripts/gt/collect.py --check-only       # validate, do not write
python scripts/gt/collect.py --stats            # show distribution stats
```

Behavior.

- Hard structural validation per item (required fields, anchor exists in `data/index_rincian`, per-type anchor count, `gold_anchor_node_ids` required for multi-anchor types, no within-batch duplicate anchors).
- Items flagged `wrong` in the matching `data/gt_audit/<doc_id>__<type>.json` are dropped.
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

## Optional: auto-annotate via OpenAI API (Step 1 + 2)

When the manual paste-paste cycle for the annotator step gets tedious,
`auto_annotate.py` calls the Anthropic API (Claude Sonnet 4.6) for every (doc, type)
in the allocation. Default annotator is Sonnet 4.6, judge is GPT-5 (cross-family
per design v3 — see `Notes/03-pipeline/llm-distribution.md`).

Setup once,

```bash
echo "OPENAI_API_KEY=sk-..." >> .env
pip install 'openai>=1.50'
```

Usage,

```bash
# Preview cost only, do not call API
python scripts/gt/auto_annotate.py --category UU --dry-run

# Real run
python scripts/gt/auto_annotate.py --category UU

# Single item
python scripts/gt/auto_annotate.py --doc-id uu-13-2025 --type paraphrased

# Pin to a snapshot
python scripts/gt/auto_annotate.py --category UU --model gpt-5.5-2026-04-23

# Raise the safety cap
python scripts/gt/auto_annotate.py --category UU --max-cost 5
```

Defaults, model `gpt-5.5`, cost cap `$1.00`. Items already annotated are
skipped unless `--force` is passed. Multipart docs (too long for single
prompt) are skipped with a clear log, run them via `prompt.py` manually.

After auto-annotate,

```bash
python scripts/gt/run_allocation.py --build --category UU
# Paste each tmp/validate_*.txt to GPT-5 (cross-family from Claude annotator)
# Paste full Judge response over the matching raw GT file
python scripts/gt/run_allocation.py --apply --category UU
# log_review per (doc, type), then collect.py + finalize.py
```

Cost estimate for pilot UU 25 items, around `$0.40-1.00` with caching.

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

# (manual) Tell the IDE Judge to process every tmp/validate_*.txt and paste
#         the full response (with ---CLEANED--- framing) over the matching
#         raw GT file in data/ground_truth_raw/<CAT>/<doc>__<type>.json.

# Apply every Judge response through the struct gate
python scripts/gt/run_allocation.py --apply --category UU
```

Filter by `--type <factual|paraphrased|multihop>` to scope
either phase. Both phases continue past per-item failures and report counts at
the end.

State derivation. The orchestrator infers each (doc, type) state from the raw
file content plus tmp/validate mtime:

| State | Meaning |
|---|---|
| `not-annotated` | raw missing or empty placeholder |
| `annotated` | raw is bare JSON, build_validate not run yet |
| `built` | raw is bare JSON AND tmp/validate_*.txt exists |
| `judged` | raw contains `---CLEANED---` (Judge response pasted, awaiting apply) |
| `applied` | raw is bare JSON AND raw mtime > validate mtime (apply ran after build) |

Re-runnable safely: `--build` skips items that don't have a raw yet,
`--apply` skips items not in `judged` state.

## Files in this directory

| File | Role |
|---|---|
| `select_gt_docs.py` | Stratified random doc selection (Step 0) |
| `allocate_quotas.py` | Per-type quota allocation, ratio target equal split ~33/33/34, per-doc cap 5 (Step 0.5) |
| `prompt.py` | Generator prompt; refresh `data/gt_provenance.json` prompt SHA-8 (Step 1) |
| `prompts/<type>.txt` | Per-type annotator templates (factual, paraphrased, multihop) |
| `merge_parts.py` | Combine multipart outputs (Step 3) |
| `build_validate.py` | Layer 1 + Layer 2 + Judge prompt assembly (Step 4) |
| `validators/paraphrase_overlap.py` | Layer 2 Jaccard gate for `paraphrased` |
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
rm -rf data/ground_truth_raw/*/*.json
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
  "query_type": "factual|paraphrased|multihop",
  "query_style": "formal|colloquial",
  "reference_mode": "none|legal_ref|doc_only|both",
  "gold_anchor_granularity": "rincian",
  "gold_anchor_node_id": "...",
  "gold_anchor_node_ids": ["..."],
  "gold_node_id": "...",
  "gold_doc_id": "...",
  "gold_doc_ids": ["..."],
  "navigation_path": "...",
  "answer_hint": "..."
}
```

`gold_anchor_node_ids` is **required** for `multihop` (size 2), optional otherwise (defaults to `[gold_anchor_node_id]`). `gold_doc_ids` mirrors `gold_anchor_node_ids`. `answer_hint` is an evidence snippet for reviewer sanity checking, not a full canonical answer. (`difficulty` field dari v1/v2 dropped di v3.)

## Habits

- Generate one (category, type) at a time. Validate, judge, review, collect before moving on.
- Always run `log_review.py` while the doc context is still in your head.
- Never paste Judge output directly to the raw file. Always go through `apply_validation.py`.
- Keep `gt_doc_selection.json` and `gt_allocation.json` under version control if you start over with a different seed.
- After every change to `data/`, push the HF dataset backup (see project CLAUDE.md).
