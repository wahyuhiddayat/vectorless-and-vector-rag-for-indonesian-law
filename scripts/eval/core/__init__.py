"""Shared core modules for the evaluation harness.

Pure building blocks for any retrieval-evaluation pipeline:
  - metrics:      retrieval + answer scoring
  - records:      per-query record building
  - aggregation:  summary computation across records
  - io:           disk read/write, resume helpers
  - preflight:    pre-run sanity checks
  - logger:       progress + summary formatting
  - runner:       orchestrator class tying everything together
"""
