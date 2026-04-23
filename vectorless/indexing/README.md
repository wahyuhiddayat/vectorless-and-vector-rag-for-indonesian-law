# Indexing Playbook

Operational guide for the vectorless indexing pipeline.

Run all commands from the project root:

```powershell
cd "d:\Fasilkom UI\Kuliah\Semester 8\TA - Skripsi\02 Codebase\vectorless-and-vector-rag-for-indonesian-law"
```

## What each granularity means

- `pasal`: main parsed index and LLM-cleanup base
- `ayat`: derived from `pasal`
- `rincian`: finest derived index from `pasal`

Only `pasal` talks to the parser and Gemini cleanup. `ayat` and `rincian` should normally be regenerated from `pasal`.

## Status first

Check current indexing progress before doing anything else:

```powershell
python -m vectorless.indexing.status --refresh-verify
python -m vectorless.indexing.status --category PMK --refresh-verify
python -m vectorless.indexing.status --doc-id pmk-10-2026 --refresh-verify
```

Important fields:

- `LLM cleaned current`: pasal files already cleaned with the current cleanup version
- `Uncleaned / cleanup-stale`: docs that still need `--llm-only --rebuild uncleaned`
- `Stale parse`: docs that need pasal rebuild after a parser-version bump
- `Stale derived`: docs whose `ayat` or `rincian` are not synced to the latest pasal
- `GT candidates`: docs structurally safe enough for ground-truth work
- `Clean retrieval candidates`: docs fully synced and cleaned for retrieval experiments

## Daily workflows

### 1. New documents after scraping

Use this when the PDFs are new and have not been indexed yet.

```powershell
python -m vectorless.indexing.build --granularity pasal --category PMK --parse-only
python -m vectorless.indexing.build --granularity pasal --category PMK --llm-only --rebuild uncleaned
python -m vectorless.indexing.build --granularity ayat --category PMK --from-pasal --rebuild all
python -m vectorless.indexing.build --granularity rincian --category PMK --from-pasal --rebuild all
python -m vectorless.indexing.status --category PMK --refresh-verify
```

### 2. Continue interrupted Gemini cleanup

Use this when pasal files already exist but cleanup failed halfway.

```powershell
python -m vectorless.indexing.build --granularity pasal --category PMK --llm-only --rebuild uncleaned
python -m vectorless.indexing.build --granularity ayat --category PMK --from-pasal --rebuild all
python -m vectorless.indexing.build --granularity rincian --category PMK --from-pasal --rebuild all
python -m vectorless.indexing.status --category PMK --refresh-verify
```

### 3. After a parser fix

Use this when `PARSER_VERSION` is bumped or a parser bug was fixed.

```powershell
python -m vectorless.indexing.build --granularity pasal --category PMK --rebuild stale
python -m vectorless.indexing.build --granularity pasal --category PMK --llm-only --rebuild uncleaned
python -m vectorless.indexing.build --granularity ayat --category PMK --from-pasal --rebuild stale
python -m vectorless.indexing.build --granularity rincian --category PMK --from-pasal --rebuild stale
python -m vectorless.indexing.status --category PMK --refresh-verify
```

### 4. Rebuild one problematic document only

Use this for `WARN` docs or a single parser experiment.

```powershell
python -m vectorless.indexing.build --granularity pasal --doc-id pmk-6-2026 --rebuild all
python -m vectorless.indexing.build --granularity pasal --doc-id pmk-6-2026 --llm-only --rebuild uncleaned
python -m vectorless.indexing.build --granularity ayat --doc-id pmk-6-2026 --from-pasal --rebuild all
python -m vectorless.indexing.build --granularity rincian --doc-id pmk-6-2026 --from-pasal --rebuild all
python -m vectorless.indexing.status --doc-id pmk-6-2026 --refresh-verify
```

## Category workflow template

Replace `PMK` with `PERMENAKER`, `PERMENDAG`, `PERMENKES`, `UU`, `PERPU`, `PP`, or `PERPRES`.

```powershell
python -m vectorless.indexing.status --category PMK --refresh-verify
python -m vectorless.indexing.build --granularity pasal --category PMK --llm-only --rebuild uncleaned
python -m vectorless.indexing.build --granularity ayat --category PMK --from-pasal --rebuild all
python -m vectorless.indexing.build --granularity rincian --category PMK --from-pasal --rebuild all
python -m vectorless.indexing.status --category PMK --refresh-verify
```

## Environment tuning

Current default cleanup settings are conservative on purpose.

If Gemini is stable and you want to go faster:

```powershell
$env:VECTORLESS_LLM_MAX_WORKERS="2"
$env:VECTORLESS_LLM_BATCH_SIZE="30000"
```

If Gemini starts timing out again, open a fresh shell or set them back:

```powershell
$env:VECTORLESS_LLM_MAX_WORKERS="1"
$env:VECTORLESS_LLM_BATCH_SIZE="20000"
```

Optional timeout override:

```powershell
$env:VECTORLESS_LLM_TIMEOUT="300"
```

## Temporary Gemini note

IMPORTANT: LLM cleanup currently uses `google-generativeai` as a temporary Windows workaround because `google.genai` was repeatedly timing out in the current environment.

Install it if needed:

```powershell
python -m pip install google-generativeai
```

This workaround should be revisited later if `google.genai` becomes stable again.

## Interpreting verify results

- `OK`: structurally clean enough
- `WARN`: usable, but has parser anomalies worth noting
- `FAIL`: broken enough to avoid for main experiments until fixed
- `MISSING`: index file does not exist yet

`WARN` does not automatically mean you must rebuild immediately. It usually means:

- keep using it if retrieval/GT still look reasonable
- revisit it after a parser fix
- rebuild that one doc only when you are ready

## Recommended practice for this thesis

- Do not block GT work on LLM cleanup if structure is already good enough
- Use `status --refresh-verify` before and after major indexing work
- Prefer category-by-category cleanup to control Gemini budget
- Prefer doc-by-doc rebuilds for `WARN` cases instead of `rebuild all`
