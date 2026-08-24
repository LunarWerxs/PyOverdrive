# ADR-0001: Repository layout and Phase 0 tooling

Date: 2026-08-23 · Status: accepted

## Decisions

1. **src-layout for the shipped package.** The spec's tree (§3) shows `runtime/`
   at the repo root. We ship an importable `pyoverdrive` package, so the runtime
   lives at `src/pyoverdrive/` (dispatcher/, fastpaths/, parallel/, simd/,
   hardware/, fallback/). Everything else in the spec tree keeps its place.
   Rationale: standard packaging hygiene; prevents accidentally importing the
   repo checkout instead of the installed wheel; keeps `lab/` unshippable.

2. **Dyno harness is custom-first, pyperf-compatible later.** Phase 0 needs
   reproducers that (a) run on Windows today, (b) check correctness outside the
   timed region, (c) sweep input regimes, and (d) emit fingerprinted JSON.
   `pyperf`'s process-spawning model complicates that for agent-written sweeps,
   so `lab/dyno` implements warmup + repeated samples + median/MAD/min
   reporting directly. pyperf stays installed; ASV + pyperf integration is a
   Phase 0 follow-up gated by CI, not a blocker for first evidence.

3. **SQLite for the corpus index; YAML records are the source of truth.**
   Records live as YAML files in `lab/corpus/` (reviewable, diffable);
   `lab/corpus/index.sqlite` is a derived index rebuilt by
   `lab/cli/rebuild_index.py`. Raw GitHub API dumps land in `lab/corpus/raw/`
   (gitignored: refetchable, and issue bodies are third-party content we
   should not redistribute wholesale).

4. **NumPy `main` builds are deferred.** Building NumPy main on Windows needs
   an MSVC + meson toolchain we have not set up. First reproductions run
   against pinned stable (2.4.5). Every opportunity record carries a
   `main_branch_result: null` until a Linux CI leg or local toolchain exists.
   This is a known evidence gap, recorded per record.

## Consequences

- `pip install -e .` installs only the runtime; lab tooling is repo-only.
- Benchmark evidence (`benchmarks/results/**`) is committed; raw API dumps are not.
- A future CI matrix (Linux/macOS) must add the NumPy-main leg before any
  "still slow on main" claim is published.
