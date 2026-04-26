# Scripts

Top-level index for everything under `scripts/`. Each sub-package owns its own README.

| Package | Purpose | Doc |
|---|---|---|
| `scripts/gt/` | Ground truth generation, validation, audit, finalization | [scripts/gt/README.md](gt/README.md) |
| `scripts/eval/` | RQ1 to RQ3 retrieval experiments runner | [scripts/eval/README.md](eval/README.md) |
| `scripts/parser/` | Indexing helpers, OCR clean, judge, granularity check | (see CLAUDE.md, `Notes/ops/Indexing Flow.md`) |
| `scripts/_shared/` | Common utilities shared across the above |  |

For pipeline-level guides see `Notes/`, start at [Notes/README.md](../../Notes/README.md). The most-referenced files are `ops/Indexing Flow.md`, `design/Ground Truth Design.md`, and `design/Overview.md`.

## When to read which doc

- Building or repairing the corpus index, `Notes/ops/Indexing Flow.md`.
- Generating or validating ground truth, [scripts/gt/README.md](gt/README.md) plus `Notes/design/Ground Truth Design.md`.
- Running retrieval experiments, project root `CLAUDE.md` and `Notes/design/Retrieval Experiments.md`.
- Reading or extending the corpus scraper, `scraper/` source files (no separate doc).
