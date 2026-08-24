# PyOverdrive: agent operating instructions

Read `docs/BUILD_SPEC.md` first. It is the authoritative brief. This file is the
short operational digest.

## What this project is

Two systems in one monorepo:

- **ProsPyctor Performance Lab** (`lab/`): mines NumPy and adjacent ecosystems
  for performance opportunities, reproduces claims with **Dyno** (`lab/dyno/`),
  and keeps evidence-backed opportunity records.
- **PyOverdrive Runtime** (`src/pyoverdrive/`): the shipped accelerator:
  **Gearbox** dispatcher, fast paths, **PyRallel** parallel core, SIMD backends,
  always with stock-NumPy fallback.

## Non-negotiables (spec §4, abridged)

1. Measure first; reproducible baseline before any optimization.
2. Correctness before speed. Differential-test every candidate against stock NumPy.
3. Every accelerated path has a stock fallback and a kill switch.
4. No unverified claims. Issue comments and papers are leads until Dyno reproduces them.
5. Preserve provenance (source URL, author, license, adaptation history) for every borrowed idea.
6. Never copy license-incompatible code. Reimplement from descriptions where legal.
7. Hardware decides. Agents propose; benchmark results decide.
8. Closed/rejected issues are still leads; never assume maintainers were wrong,
   extract and test their objections explicitly.

## Mechanics

- Environment: `.venv/` (Windows: `.venv/Scripts/python`). Pinned stock NumPy is
  the baseline; record versions in every result.
- Every benchmark result carries a machine fingerprint (`tools/fingerprint.py`)
  and the foreign CPU load Dyno sampled before and after the run
  (`conditions` in the JSON). This box is shared by many agent sessions: a
  result flagged `contended` understates every multi-thread candidate, so
  treat its thresholds as conservative and rerun when the box is quiet
  before claiming a crossover moved. Never benchmark while another heavy
  run of yours is in flight.
- Recalibrate PyRallel on new hardware with
  `benchmarks/micro/bench_pyrallel_calibration.py`, then read the table
  `lab/cli/calibrate_pyrallel.py` prints into
  `src/pyoverdrive/fastpaths/parallel_ufunc.py` by hand.
- Opportunity records live in `lab/corpus/` (YAML, schema in
  `lab/corpus/SCHEMA.md`) with reports in `docs/research/opportunities/`.
- Reproducers live in `benchmarks/historical/opp_XXXXXX_<slug>.py` and use the
  Dyno harness (`lab/dyno`). Results JSON goes to
  `benchmarks/results/OPP-XXXXXX/<fingerprint>.json` and IS committed.
- Commit small, path-scoped changes (`git add <paths>`, never `git add -A`).
- Naming: PyOverdrive, ProsPyctor, Dyno, Gearbox, PyRallel in docs and
  diagnostics; descriptive module paths (`dispatcher/`, `parallel/`) in code.
- Trademark: never put "NumPy" in the package/repo/product name. Say
  "PyOverdrive for NumPy" / "PyOverdrive accelerates NumPy".

## Agents must not

- Invent or extrapolate benchmark numbers.
- Cherry-pick favorable input sizes; sweep sizes/dtypes/layouts.
- Hide correctness differences behind loose tolerances.
- Treat one CPU as universal evidence.
- Copy code without a provenance + license record.
