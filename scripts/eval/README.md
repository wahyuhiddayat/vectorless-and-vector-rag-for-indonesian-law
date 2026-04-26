# Evaluation Framework

Orchestrates RQ1, RQ2, and RQ3 experiments for vectorless vs vector RAG on the Indonesian legal corpus.

Run all commands from the project root.

```powershell
cd "d:\Fasilkom UI\Kuliah\Semester 8\TA - Skripsi\02 Codebase\vectorless-and-vector-rag-for-indonesian-law"
```

For experimental design, supervisor decisions, and the RQ3 winner-selection rule see [`Notes/design/Retrieval Experiments.md`](../../../Notes/design/Retrieval%20Experiments.md). For ground-truth construction see [`scripts/gt/README.md`](../gt/README.md).

## Files

| File | Role |
|---|---|
| `vectorless.py` | RQ1 CLI (3 systems x 3 granularities, 9 combos) |
| `vector.py` | RQ2 CLI (2 systems x 3 granularities x 3 embedding models, 18 combos) |
| `vectorless_worker.py` | Subprocess worker for one vectorless retrieval call |
| `vector_worker.py` | Subprocess worker for one vector retrieval call |
| `core/runner.py` | EvalRunner, preflight, per-combo loop, retry, finalize |
| `core/metrics.py` | Pure IR metrics, retrieval + answer scoring + rank stats |
| `core/aggregation.py` | Per-combo, per-slice, per-reference_mode summaries + bootstrap CI |
| `core/preflight.py` | Index coverage, Gemini, Qdrant, corpus consistency, GT fingerprint |
| `core/records.py` | Per-query record schema and worker payload normalisation |
| `core/io.py` | JSONL/CSV writers, testset loader, resume helpers |
| `core/logger.py` | Tee logger, stdout + progress.log |

## Pipeline at a glance

```
preflight
  - index coverage   (every gold_doc_id has a JSON at each granularity)
  - corpus consistency  (rincian >= ayat >= pasal leaf counts per doc)
  - Gemini reachable    (LLM-driven systems only)
  - Qdrant reachable    (vector path only)
  - resume config diff  (refuse to mix configs in one run dir)

execute (per combo)
  - subprocess worker per query, timeout + categorised retry
  - JSONL flush per query, resumable mid-run

finalize
  - aggregate, per-combo, per-slice, per-reference_mode
  - bootstrap percentile CI on Recall@k and MRR@10
  - write CSVs, summary_overall.json, errors.jsonl, progress.log
```

## RQ1, vectorless

```powershell
# Pilot, 10 random queries, fixed seed
python scripts/eval/vectorless.py --label pilot_10q --query-limit 10 --random-seed 42

# Full run, 9 combos x N queries
python scripts/eval/vectorless.py --label main_rq1_140q --strict

# Resume after a crash
python scripts/eval/vectorless.py --label main_rq1_140q --resume

# Subset (one system)
python scripts/eval/vectorless.py --label test_bm25 --systems bm25

# Single doc debug
python scripts/eval/vectorless.py --label debug --doc-id pmk-21-2026 --query-limit 3

# Metric self-test (no LLM calls)
python scripts/eval/vectorless.py --self-test-metrics
```

## RQ2, vector

```powershell
# Full run, 18 combos
python scripts/eval/vector.py --label main_rq2_140q --qdrant-path ./qdrant_local --strict

# One embedding model, debug
python scripts/eval/vector.py --label test_bge --embedding-models bge-m3 --query-limit 5 --qdrant-path ./qdrant_local

# Resume
python scripts/eval/vector.py --label main_rq2_140q --resume --qdrant-path ./qdrant_local
```

## RQ3, head-to-head

After RQ1 and RQ2 finish, pick winners per the criterion in [Retrieval Experiments.md](../../../Notes/design/Retrieval%20Experiments.md), section "RQ3 Winner-Selection Criterion".

Report from `summary_overall.json` of each run.

```python
# Pseudocode for the laporan section
v_winner = best_combo(rq1_summary, by="recall@10")
e_winner = best_combo(rq2_summary, by="recall@10")
report(v_winner, e_winner, ci=rq{1,2}_summary["bootstrap_ci"])
```

There is no separate RQ3 runner. RQ3 is a comparison of the existing artifacts.

## Outputs

```
data/eval_runs/<label>/
  config.json                              # full config + GT fingerprint + LLM model
  records/
    <system>__<granularity>.jsonl          # vectorless (one file per combo)
    <system>__<gran>__<model>.jsonl        # vector (one file per combo)
  summary_overall.json                     # overall + per-combo + bootstrap CI + per-ref-mode
  summary_by_system_granularity.csv        # one row per combo, 30+ metric cols
  summary_by_slice.csv                     # per (combo, slice_field, slice_value)
  summary_by_reference_mode.csv            # per (combo, reference_mode)
  errors.jsonl                             # only if any errors
  progress.log                             # tee'd terminal output
```

## Metrics

Per query, retrieval.

- `hit@k`, `recall@k` (equal for single-gold, both reported)
- `mrr@k` for every cutoff in {1, 3, 5, 10}
- `ndcg@k`, `map@k`
- `first_relevant_rank`, `full_reciprocal_rank` (no cutoff, captures gold beyond k)
- `exact_top1_hit`

Per combo, descriptive stats.

- `mean_rank_on_hit`, `median_rank_on_hit`, `max_rank_on_hit`
- `hits_anywhere` (count of queries with gold somewhere in retrieved list)

Per query, answer.

- `answer_nonempty`, `num_citations`
- `citation_precision`, `citation_recall`, `citation_hit`, `fully_grounded_citations`
- `answer_hint_token_recall`, `answer_hint_token_f1` (sanity-check only, not headline)

Per query, cost.

- `llm_calls`, `input_tokens`, `output_tokens`, `total_tokens`, `elapsed_s`
- `step_metrics` (per-stage breakdown if the retrieval module exposes it)

Per combo, statistical.

- `bootstrap_ci`, percentile bootstrap CI for `recall@k` (every cutoff) and `mrr@10`. 1000 resamples, seed 42, 95% interval.

## Methodology notes

Documented in [`core/metrics.py`](core/metrics.py) docstring N1, N2, N3.

- N1, single-gold GT collapses several IR metrics. Recall@k = Hit@k, MAP@k = MRR@k. NDCG@k monotone-equivalent. Reported in full so reviewers from any sub-field can cross-reference.
- N2, retrieval may return fewer than k items. Recall@k for a 3-item list with the gold at rank 2 is still a hit, no padding to k.
- N3, full_reciprocal_rank ignores the cutoff. Useful when gold appears at rank 12 with k=10, where mrr@10=0 hides the signal.

## Subprocess pattern

Each query runs in a fresh subprocess.

- vectorless, `DATA_INDEX` env var picks the granularity (pasal, ayat, rincian).
- vector, `VECTOR_EMBEDDING_MODEL`, `VECTOR_COLLECTION`, `VECTOR_GRANULARITY`, `QDRANT_PATH`.

Env vars are set before the retrieval module is imported, so module-level reads of those constants pick up the right values.

## Resume and idempotency

`--resume` skips already-recorded `(query_id, system, granularity[, embedding_model])` combos. Truncated records (crash mid-flush) and records missing required fields are dropped on read so those queries get re-run automatically. The runner refuses to resume into a directory whose stored config disagrees with the current invocation.

`--overwrite` deletes the run directory and starts fresh.

`--strict` aborts on any pre-flight failure. Without it, warnings are surfaced and execution continues.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Worker timed out" | LLM stepwise stuck or Gemini slow | `--worker-timeout-s 1500` |
| "missing-index" errors only | Re-split incomplete or doc outside the corpus | Run `vectorless.indexing.build` for that doc and granularity |
| "Qdrant check failed" | `--qdrant-path` does not exist | Ensure indexing wrote the local store, or pass a server URL |
| "Resume config mismatch" | Different `--systems` or `--granularities` than the stored run | Use `--overwrite`, or pick a new `--label` |
| Gemini API check failed | `GEMINI_API_KEY` unset | Set `.env`, retry without `--strict` to validate other parts |
| Bootstrap CI too wide | Few queries in this combo | Increase query count or accept the wider CI in the writeup |

## Relation to GT

The eval reads `data/validated_testset.pkl` produced by the GT pipeline ([scripts/gt/](../gt/README.md)). The runner records a SHA-256 fingerprint of the testset in `config.json` so eval artifacts can be tied back to the exact GT snapshot they were produced against. If the testset is regenerated between runs, the fingerprint diff is surfaced as a warning.

## Adding a new retrieval system

1. Add the module under `vectorless/retrieval/<name>/` (or `vector/...`).
2. Register it in the worker dispatcher (`vectorless_worker.py` or `vector_worker.py`).
3. Add the system name to the `SYSTEMS` list in `vectorless.py` or `vector.py`, and to `LLM_SYSTEMS` if it calls Gemini.
4. Run `--label test_<name> --query-limit 5` to smoke test before a full run.

No changes needed in the runner, metrics, or aggregation, the pattern is generic over `(system, granularity)` (and `embedding_model` for vector).
