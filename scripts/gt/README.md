# Ground Truth Workflow

Operational guide for building and maintaining the retrieval ground truth dataset.

Run commands from the project root:

```bash
cd "d:/Fasilkom UI/Kuliah/Semester 8/TA - Skripsi/02 Codebase/vectorless-and-vector-rag-for-indonesian-law"
```

## Pipeline

```text
0. SELECT      select_gt_docs.py       choose 5 GT-source docs per category
1. ALLOCATE    allocate_quotas.py      assign query quotas per doc and type
2. PROMPT      prompt.py               build annotator prompts
3. ANNOTATE    external LLM/API        write raw GT JSON
4. VALIDATE    build_validate.py       build judge prompt after local gates
5. JUDGE       external LLM/API        clean semantic issues
6. APPLY       apply_validation.py     extract cleaned judge output
7. REVIEW      log_review.py           author spot-check
8. COLLECT     collect.py              merge accepted raw GT
9. FINALIZE    finalize.py             build validated testset pickle
10. SPLIT      split_dataset.py        write train/val/test qid splits
```

Steps 3 and 5 may be manual LLM calls or API calls. All other steps are local scripts.

## Query Types

| Type | Anchors | Purpose |
|---|---:|---|
| `factual` | 1 | Literal lookup |
| `paraphrased` | 1 | Lexical variation and semantic robustness |
| `multihop` | 2 | Two-anchor retrieval inside one document |

All anchors must be leaf nodes in `data/index_rincian`.

## Active Artifacts

| Artifact | Role |
|---|---|
| `data/gt_doc_selection.json` | GT-source and distractor document selection |
| `data/gt_allocation.json` | Planned query allocation per selected document |
| `data/ground_truth_raw/<CAT>/<doc_id>__<type>.json` | Raw per-document GT items |
| `data/gt_audit/<doc_id>__<type>.json` | Author review decisions |
| `data/ground_truth.json` | Merged accepted GT items |
| `data/validated_testset.pkl` | Final evaluator input |
| `data/splits/{train,val,test}_qids.json` | Split qid lists |
| `data/splits/split_manifest.json` | Split config, stats, and fingerprints |

## Select GT-Source Docs

```bash
python scripts/gt/select_gt_docs.py --category UU --seed 42
python scripts/gt/select_gt_docs.py --all --seed 42
python scripts/gt/select_gt_docs.py --show
```

Output: `data/gt_doc_selection.json`.

Each category uses 5 GT-source docs. The unselected docs stay in the corpus as same-category distractors.

## Allocate Query Quotas

```bash
python scripts/gt/allocate_quotas.py --category UU --seed 42 --emit-commands
python scripts/gt/allocate_quotas.py --all --seed 42
```

Output: `data/gt_allocation.json`.

The allocation caps each selected document at 5 queries and targets an approximately even split across `factual`, `paraphrased`, and `multihop`.

## Generate Prompts

```bash
python scripts/gt/prompt.py uu-13-2025 --type factual --questions 2
python scripts/gt/prompt.py uu-13-2025 --type paraphrased --questions 2
python scripts/gt/prompt.py uu-13-2025 --type multihop --questions 1
```

Outputs:

- prompt files under `tmp/`
- placeholder raw GT files under `data/ground_truth_raw/<CAT>/`
- prompt template versions in `data/gt_provenance.json`

Prompt templates live in `scripts/gt/prompts/`.

## Annotate

Manual path:

1. Paste the prompt file into the annotator model.
2. Save the JSON array to `data/ground_truth_raw/<CAT>/<doc_id>__<type>.json`.
3. If the output is split across multiple files, save parts under `data/ground_truth_parts/<CAT>/<doc_id>__<type>/`.

Merge multipart output:

```bash
python scripts/gt/merge_parts.py uu-13-2025 --type factual
```

API path:

```bash
python scripts/gt/auto_annotate.py --category UU --dry-run
python scripts/gt/auto_annotate.py --category UU
python scripts/gt/auto_annotate.py --doc-id uu-13-2025 --type paraphrased
```

Defaults:

- provider: Anthropic
- model: `claude-sonnet-4-6`
- temperature: `0.0`
- seed: `42`
- cost cap: `$1.00`

Set `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` when using API paths.

## Build Validation Prompt

```bash
python scripts/gt/build_validate.py --doc-id uu-13-2025 --type factual
python scripts/gt/build_validate.py --doc-id uu-13-2025 --type paraphrased
python scripts/gt/build_validate.py --doc-id uu-13-2025 --type multihop
```

This script:

- runs structural validation from `collect.py`
- runs the paraphrase overlap gate for `paraphrased`
- adds relevant leaf text context
- writes `tmp/validate_<doc_id>__<type>.txt`

## Judge and Apply

Manual judge path:

1. Paste `tmp/validate_<doc_id>__<type>.txt` into the judge model.
2. Save the full judge response over the matching raw GT file.
3. Run `apply_validation.py`.

```bash
python scripts/gt/apply_validation.py --doc-id uu-13-2025 --type multihop
```

Alternative input modes:

```bash
python scripts/gt/apply_validation.py --doc-id uu-13-2025 --type multihop --judge-file tmp/judge.txt
python scripts/gt/apply_validation.py --doc-id uu-13-2025 --type multihop --stdin < paste.txt
python scripts/gt/apply_validation.py --doc-id uu-13-2025 --type multihop --dry-run
```

API judge path:

```bash
python scripts/gt/auto_judge.py --category UU --dry-run
python scripts/gt/auto_judge.py --category UU
```

Defaults:

- provider: OpenAI
- model: `gpt-5`
- temperature: `0.0`
- seed: `42`
- cost cap: `$1.00`

## Author Review

Run after judge/apply while the document context is still fresh.

```bash
python scripts/gt/log_review.py uu-13-2025 --type factual
python scripts/gt/log_review.py uu-13-2025 --type multihop
```

Review commands:

| Key | Meaning |
|---|---|
| `c` | correct |
| `w` | wrong, excluded during collect |
| `b` | borderline, kept but recorded |
| `s` | skipped |
| `q` | quit and save partial progress |

Aggregate report:

```bash
python scripts/gt/log_review.py --report
python scripts/gt/log_review.py --report --json
```

## Collect

```bash
python scripts/gt/collect.py --file data/ground_truth_raw/<CAT>/<doc_id>__<type>.json
python scripts/gt/collect.py
python scripts/gt/collect.py --check-only
python scripts/gt/collect.py --stats
```

`collect.py` validates raw files and writes `data/ground_truth.json`.

Validation includes:

- required fields
- valid `query_type`
- anchor existence in `data/index_rincian`
- required anchor count per query type
- duplicate query and duplicate anchor checks
- author-review exclusions from `data/gt_audit/`

## Finalize

```bash
python scripts/gt/finalize.py
python scripts/gt/finalize.py --check
python scripts/gt/finalize.py --stats
```

Output: `data/validated_testset.pkl`.

`finalize.py` rolls `rincian` anchors up to gold sets for `ayat` and `pasal`, then writes the evaluator input.

Inspect the final dataset:

```bash
python scripts/gt/load_testset.py
python scripts/gt/load_testset.py --stats
python scripts/gt/load_testset.py --doc <doc_id>
python scripts/gt/load_testset.py --query "<keyword>"
```

## Split Dataset

```bash
python scripts/gt/split_dataset.py
python scripts/gt/split_dataset.py --dry-run --stats
python scripts/gt/split_dataset.py --verify
```

Outputs:

- `data/splits/dev_qids.json`
- `data/splits/val_qids.json`
- `data/splits/test_qids.json`
- `data/splits/dev.jsonl`
- `data/splits/val.jsonl`
- `data/splits/test.jsonl`
- `data/splits/split_manifest.json`

The split is deterministic with seed `42` and stratifies by `(category, query_type)`.

## Batch Orchestrator

`run_allocation.py` walks `data/gt_allocation.json` and batches validation build/apply phases.

```bash
python scripts/gt/run_allocation.py
python scripts/gt/run_allocation.py --category UU
python scripts/gt/run_allocation.py --build --category UU
python scripts/gt/run_allocation.py --apply --category UU
python scripts/gt/run_allocation.py --category UU --type paraphrased
```

State names:

| State | Meaning |
|---|---|
| `not-annotated` | raw file missing or placeholder |
| `annotated` | raw JSON exists, validation prompt not built |
| `built` | validation prompt exists |
| `judged` | raw file contains judge response framing |
| `applied` | cleaned JSON has been applied after validation prompt build |

## Files

| File | Role |
|---|---|
| `select_gt_docs.py` | choose GT-source docs |
| `allocate_quotas.py` | assign per-type query quotas |
| `prompt.py` | build annotator prompts |
| `prompts/factual.txt` | factual query template |
| `prompts/paraphrased.txt` | paraphrased query template |
| `prompts/multihop.txt` | multihop query template |
| `merge_parts.py` | merge multipart annotator outputs |
| `build_validate.py` | build judge prompts |
| `validators/paraphrase_overlap.py` | paraphrase overlap gate |
| `validate_prompt.txt` | judge rules |
| `apply_validation.py` | extract and validate judge-cleaned output |
| `auto_annotate.py` | API annotator runner |
| `auto_judge.py` | API judge runner |
| `log_review.py` | author review logger |
| `collect.py` | merge accepted raw GT |
| `finalize.py` | write `validated_testset.pkl` |
| `split_dataset.py` | write train/val/test splits |
| `load_testset.py` | inspect final testset |
| `run_allocation.py` | batch validation build/apply |
| `build_catalog.py` | filter index catalogs to GT docs |

## Raw GT Schema

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

Rules:

- `gold_anchor_node_ids` is required for `multihop` and must contain 2 IDs.
- `factual` and `paraphrased` must have 1 anchor.
- `gold_doc_ids` mirrors `gold_anchor_node_ids`.
- `answer_hint` is reviewer evidence, not a canonical answer.

## Maintenance

Use these only when intentionally rebuilding GT artifacts.

```bash
# Reset merged GT only, keep raw files and review logs.
echo {} > data/ground_truth.json

# Remove finalized evaluator artifacts.
rm -f data/validated_testset.pkl
rm -rf data/splits

# Remove selection/allocation plans.
rm -f data/gt_doc_selection.json data/gt_allocation.json
```
