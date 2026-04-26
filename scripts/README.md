# Scripts

Top-level index for everything under `scripts/`. Each sub-package owns its own README.

| Package | Purpose | Doc |
|---|---|---|
| `scripts/gt/` | Ground truth generation, validation, audit, finalization | [scripts/gt/README.md](gt/README.md) |
| `scripts/eval/` | RQ1 to RQ3 retrieval experiments runner | [scripts/eval/README.md](eval/README.md) |
| `scripts/parser/` | Indexing helpers, OCR clean, judge, granularity check | (see CLAUDE.md, `Notes/Indexing Flow.md`) |
| `scripts/_shared/` | Common utilities shared across the above |  |

For pipeline-level guides see `Notes/`, in particular `Indexing Flow.md`, `Ground Truth Design.md`, and `Overview.md`.

## When to read which doc

- Building or repairing the corpus index, `Notes/Indexing Flow.md`.
- Generating or validating ground truth, [scripts/gt/README.md](gt/README.md) plus `Notes/Ground Truth Design.md`.
- Running retrieval experiments, project root `CLAUDE.md` and `Notes/Retrieval Experiments.md`.
- Reading or extending the corpus scraper, `scraper/` source files (no separate doc).
