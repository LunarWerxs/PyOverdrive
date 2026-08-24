# Opportunity record schema

One YAML file per opportunity: `lab/corpus/OPP-XXXXXX.yaml`. These files are
the source of truth; `lab/corpus/index.sqlite` is a derived index (rebuild with
`python lab/cli/rebuild_index.py`). The human-readable analysis lives in
`docs/research/opportunities/OPP-XXXXXX.md`.

## Fields

```yaml
id: OPP-000001                # stable, never reused
title: one-line description
sources:                      # every source, with retrieval metadata
  - url: https://github.com/numpy/numpy/issues/31969
    repository: numpy/numpy
    type: issue | pr | commit | gist | mailing-list | paper | benchmark
    number: 31969
    state_at_ingestion: open | closed | merged
    retrieved_at: ISO-8601 UTC
claim:                        # the ORIGINAL claim, verbatim in spirit
  speedup: 46.0               # as reported, not as verified
  baseline: what was measured as slow
  candidate: what was reported faster
  hardware: as reported, or unknown
affected_operations: [numpy.unique]
input_regime:                 # where the claim applies
  dtypes: [int64]
  sizes: [1000000]
  notes: cardinality, sortedness, layout conditions
status: unverified | reproduced | not_reproduced | prototyped |
        correctness_validated | performance_validated | accepted |
        rejected | shipped | monitored
license_review: pending | clean | blocked
reproducer: benchmarks/historical/opp_000001_unique.py   # or null
current_numpy_result:         # filled by Dyno runs, null until then
  verified_speedup: null      # measured on fingerprinted hardware
  fingerprints: []            # machines that contributed evidence
main_branch_result: null      # requires NumPy main build (ADR-0001 item 4)
correctness_risk: unknown | low | medium | high
implementation_risk: unknown | low | medium | high
maintainer_objections: []     # verbatim concerns; never assume they were wrong
linked_prs: []
linked_commits: []
related_opportunities: []     # graph edges (spec section 5.5)
notes: []
```

## Workflow states (spec section 13)

discovered -> extracted -> deduplicated -> reproduced_or_rejected ->
algorithm_audited -> prototyped -> correctness_validated ->
performance_validated -> integration_reviewed -> accepted_or_rejected ->
shipped_or_upstreamed -> continuously_monitored

A record's `status` must only advance when the corresponding artifact exists
(e.g. `reproduced` requires a committed Dyno result file).
