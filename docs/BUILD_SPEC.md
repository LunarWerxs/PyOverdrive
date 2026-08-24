# PyOverdrive: Autonomous AI Build Specification

**Document purpose:** This file is the primary implementation brief for an AI engineering team building a drop-in NumPy performance accelerator and the research system that continuously discovers optimizations for it.

**Project name:** PyOverdrive

**Tagline:** **NumPy at full throttle.**

**Python package and repository name:** `pyoverdrive`

**Positioning:** PyOverdrive is an independent adaptive accelerator for NumPy. It is not affiliated with or endorsed by NumPy or NumFOCUS. Use “PyOverdrive for NumPy” or “PyOverdrive accelerates NumPy” in descriptive text; do not combine `NumPy` into the package, repository, domain, or product name without written permission from the trademark owner.

**Status:** Named and ready for initial implementation.

---

## 1. Executive mandate

Build PyOverdrive as one umbrella project containing two complementary systems:

1. **ProsPyctor Performance Lab:** An automated research system that mines NumPy and adjacent open-source ecosystems for performance opportunities, reproduces them, tests alternative algorithms, and produces evidence-backed implementation candidates. Its benchmark and verification subsystem is named **Dyno**.
2. **PyOverdrive Runtime:** An installable runtime that accelerates real `numpy.ndarray` operations through the **Gearbox** adaptive dispatcher, specialized fast-path algorithms, SIMD kernels, and the **PyRallel** adaptive parallel execution core while falling back safely to stock NumPy whenever an accelerated path is not demonstrably safe and faster.

ProsPyctor discovers optimization opportunities, Dyno validates them, and the PyOverdrive Runtime applies them. Gearbox selects the appropriate implementation at runtime. PyRallel is one execution backend available to Gearbox, not a separate competing product.

The project must pursue substantial wins aggressively, including credible 10–50× improvements on particular operations and input regimes. It must never market or report a speedup that has not been reproduced under controlled conditions. The objective is not to make every NumPy call 10× faster; that is physically impossible for many memory-bound and already optimized operations. The objective is to discover and deliver as many meaningful, broadly useful improvements as possible without breaking NumPy semantics.

This project is intended to be fully AI-written. AI agents may plan, research, implement, test, review, benchmark, document, and maintain the code. Empirical results—not agent confidence—are the final authority.

---

## 2. Product definition

### 2.1 User-facing goal

The eventual user experience should be as close as technically possible to:

```bash
pip install pyoverdrive
```

```python
import numpy as np
import pyoverdrive

pyoverdrive.enable()
```

Existing arrays and ordinary NumPy code should continue working. A future distribution may provide an alternate NumPy wheel or upstream changes that require no explicit `enable()` call. Do not commit to the final activation mechanism before prototyping the available NumPy extension and dispatch hooks.

### 2.2 What the runtime does

For each supported operation, the runtime considers relevant facts such as:

- Operation and arguments
- Dtype and casting rules
- Input size and dimensionality
- Shape, strides, contiguity, and alignment
- Axis and reduction configuration
- Expected cardinality, sortedness, or other safely known properties
- Available CPU instruction sets
- Core count, current thread utilization, and memory topology
- Active BLAS/OpenMP/thread-pool state
- Estimated computation-to-memory ratio
- Measured crossover thresholds

It then selects one of:

- Stock NumPy
- A specialized algorithmic fast path
- A scalar optimized kernel
- A single-threaded SIMD kernel
- A parallel SIMD kernel
- An optimized external library with a compatible license and semantics

Unsupported, ambiguous, unsafe, or unprofitable cases must fall back to stock NumPy.

### 2.3 Non-goals for the first releases

- Replacing `numpy.ndarray` with a different array type
- Requiring decorators, graph compilation, or a new expression language
- Becoming a general GPU tensor framework
- Reimplementing all of NumPy
- Promising universal 10× acceleration
- Sacrificing correctness silently for speed
- Maintaining a permanent full NumPy fork unless every lighter integration path fails
- Optimizing large matrix multiplication already delegated effectively to tuned BLAS libraries

Optional relaxed-precision or fast-math modes may be considered later, but they must be explicit, clearly documented, and disabled by default.

---

## 3. Core architectural rule: one project, separate modules

Use a monorepo with strong boundaries between research tooling and shipped runtime code.

```text
pyoverdrive/
├── AGENTS.md
├── README.md
├── LICENSE
├── pyproject.toml
├── meson.build
├── lab/
│   ├── collectors/             # GitHub, release notes, papers, benchmarks
│   ├── extractors/             # Reproducers, claims, patches, algorithms
│   ├── corpus/                 # Normalized opportunity records
│   ├── runners/                # Isolated benchmark execution
│   ├── ranking/                # Evidence-based opportunity scoring
│   ├── reports/                # Generated research reports
│   └── cli/
├── runtime/
│   ├── dispatcher/
│   ├── fastpaths/
│   ├── parallel/
│   ├── simd/
│   ├── hardware/
│   ├── thread_control/
│   └── fallback/
├── native/                     # C/C++ extension code
├── compatibility/
│   ├── differential/
│   ├── fuzzing/
│   └── fixtures/
├── benchmarks/
│   ├── micro/
│   ├── workload/
│   ├── historical/
│   └── results/
├── docs/
│   ├── architecture/
│   ├── decisions/
│   ├── research/
│   │   └── opportunities/
│   └── performance/
├── tools/
└── tests/
```

ProsPyctor may become its own package later, but it should initially live beside the PyOverdrive Runtime so that it can share Dyno benchmark schemas, compatibility tests, hardware fingerprints, and opportunity records.

### 3.1 Canonical component names

| Component | Canonical name | Role |
|---|---|---|
| Overall project and installable runtime | **PyOverdrive** | Accelerates real NumPy workloads |
| Public tagline | **NumPy at full throttle.** | Speed-first positioning |
| Issue and PR mining system | **ProsPyctor** | Finds historical and current optimization opportunities |
| Benchmark and verification system | **Dyno** | Measures speed, memory, scaling, and regressions |
| Adaptive runtime dispatcher | **Gearbox** | Selects stock NumPy, fast paths, SIMD, or parallel execution |
| Adaptive parallel execution core | **PyRallel** | Executes appropriate operations across CPU cores |

Use these names consistently in code, documentation, reports, CI jobs, and user-facing diagnostics. Descriptive names such as `dispatcher` and `parallel` should remain in module paths where they make the code clearer; branding must not reduce maintainability.

---

## 4. Non-negotiable engineering principles

1. **Measure first.** Every optimization begins with a reproducible baseline.
2. **Correctness before speed.** An incorrect 50× result is a failure.
3. **Stock fallback.** Every accelerated path needs a reliable fallback.
4. **No unverified claims.** Treat issue comments, papers, and benchmarks as leads until independently reproduced.
5. **Avoid regressions.** A fast path that slows common cases must be guarded by accurate dispatch.
6. **Optimize actual workloads.** Microbenchmarks identify mechanisms; application benchmarks determine value.
7. **Preserve provenance.** Record the source, author, URL, license, commit, and adaptation history of every borrowed idea or implementation.
8. **Do not copy incompatible code.** Algorithms may be reimplemented from descriptions where legally appropriate; code must only be reused under compatible terms.
9. **Small upstreamable changes are preferred.** Keep optimizations modular and independently reviewable.
10. **Hardware decides.** Agents propose; benchmark results decide.
11. **Closed does not mean solved.** A closed issue may have been rejected for scope, API uncertainty, lack of a contributor, stale discussion, or an incomplete patch.
12. **Never assume maintainers were wrong.** Extract and test their objections explicitly.

---

## 5. Mandatory performance gold-mine program

Issue and pull-request mining is a primary R&D program. It must not be treated as optional research performed after implementation begins.

### 5.1 Mine the entire history, not only open issues

Ingest and analyze:

- Every open and closed NumPy issue
- Every open, closed, and merged NumPy pull request
- Full comment timelines, reviews, linked commits, branches, forks, gists, benchmark attachments, and cross-references
- NumPy NEPs, roadmap items, release notes, mailing-list discussions, benchmarks, and developer documentation
- Performance regressions that were fixed, because the old and new implementations may reveal reusable dispatch strategies
- Rejected proposals and abandoned PRs
- Issues closed as duplicates, stale, wontfix, out-of-scope, or requiring design decisions
- Discussions mentioning faster behavior in older NumPy versions
- Reports where an equivalent NumPy operation takes radically different paths depending on shape or dtype

Do not assume that an old issue is irrelevant. Re-run its reproducer against the latest stable NumPy and NumPy `main`. Hardware, compilers, SIMD support, and internal implementations change over time.

### 5.2 Mine NumPy-adjacent repositories

Search the issue, PR, benchmark, and source histories of at least:

- SciPy
- pandas
- CPython and Faster CPython
- Numba and llvmlite
- NumExpr
- CuPy
- JAX and XLA
- PyTorch
- Dask and Xarray
- Pythran
- Bohrium and cuPyNumeric
- Bodo and Codon
- Apache Arrow
- Polars and DuckDB
- Julia array and linear-algebra packages
- xtensor and Eigen
- OpenBLAS, BLIS, and oneMKL-related projects
- SLEEF, Google Highway, xsimd, and compiler vector-math projects
- sortednp, fastfunc, and other small packages linked from NumPy performance discussions
- Array API standard discussions

The goal is not merely to copy code. Look for:

- Algorithms NumPy does not use
- Dispatch thresholds NumPy lacks
- Specialized handling for sorted, contiguous, aligned, low-cardinality, or repeated data
- SIMD kernels missing for certain operations or dtypes
- Cache-aware traversal and tiling strategies
- Parallel implementations and their crossover points
- Abandoned patches with unresolved but solvable compatibility objections
- Workarounds used by downstream libraries because NumPy is slow
- Cases where downstream projects bypass NumPy or reshape/cast data before calling it
- Benchmarks where two mathematically equivalent NumPy formulations differ dramatically
- Regressions introduced by newer algorithms

### 5.3 Search vocabulary

Search titles, bodies, comments, diffs, commits, and benchmark names using broad and combinable terms:

```text
performance, perf, slow, slower, regression, speed, speedup, faster,
benchmark, bottleneck, optimize, optimization, SIMD, AVX, AVX2, AVX-512,
SSE, NEON, SVE, vectorize, vectorization, parallel, threading, multicore,
OpenMP, cache, locality, allocation, temporary, copy, stride, contiguous,
broadcast, dispatch, specialize, specialization, fusion, fused, BLAS,
LAPACK, reduction, indexing, scatter, gather, sort, unique, set,
transpose, relayout, dtype, float16, float32, casting, GIL, free-threaded,
10x, 20x, 50x, order of magnitude, memory bandwidth, NUMA
```

Search for comparative phrases such as:

```text
faster than NumPy
NumPy is slower than
equivalent to
workaround
reshape before
avoids temporary
no pull request
abandoned
stale
prototype
proof of concept
```

### 5.4 Preserve complete evidence

For each lead, create a normalized opportunity record:

```yaml
id: OPP-000001
title: Adaptive algorithm selection for numeric unique
sources:
  - url: https://github.com/numpy/numpy/issues/31969
    repository: numpy/numpy
    type: issue
    number: 31969
    state_at_ingestion: open
    retrieved_at: YYYY-MM-DDTHH:MM:SSZ
claim:
  speedup: 46.0
  baseline: hash-based unique
  candidate: SIMD sorting path
  hardware: unknown-or-recorded
affected_operations: [numpy.unique]
input_regime:
  dtypes: [int32, int64]
  sizes: [1000000]
status: unverified
license_review: pending
reproducer_status: not_started
current_numpy_result: null
main_branch_result: null
correctness_risk: unknown
implementation_risk: unknown
maintainer_objections: []
linked_prs: []
linked_commits: []
notes: []
```

Every opportunity record must retain:

- Original performance claim
- Exact code or best reconstructed reproducer
- Original and current environment details
- Current issue/PR status
- All linked implementation attempts
- Maintainer concerns and rejection reasons
- License and provenance information
- Reproduction results on stable NumPy and `main`
- Correctness findings
- Performance distributions, not only best runs
- Recommendation: reject, monitor, prototype, implement, or upstream

### 5.5 Build an opportunity graph

Do not analyze issues independently when they describe the same underlying mechanism. Link opportunities by:

- Operation
- Dtype
- Algorithm family
- SIMD/vector-math library
- Memory-access pattern
- Dispatch overhead
- Threading concern
- Duplicate issue/PR
- Shared maintainer objection
- Shared downstream workaround

The graph should reveal clusters such as “scatter/reduction bottlenecks,” “small-array dispatch overhead,” “missing float16 SIMD,” and “shape-dependent BLAS routing.” A cluster with ten partial solutions may be more valuable than any individual issue.

### 5.6 Gold-mine ranking

Rank opportunities using an explicit score, for example:

```text
opportunity_score =
    verified_speedup_weight
  × estimated_usage_frequency
  × affected_ecosystem_reach
  × confidence
  × compatibility_probability
  ÷ implementation_cost
  ÷ maintenance_cost
```

Use bounded normalized factors rather than raw multiplication in the actual implementation. Publish each component so an agent cannot hide subjective judgment inside one opaque score.

Prioritize:

- Large verified speedup
- Common operation
- Small and reviewable patch
- Exact semantic equivalence
- Cross-platform benefit
- No maintained existing accelerator already solving it
- Clear route to independent packaging or upstream acceptance

---

## 6. Initial high-value leads

These are starting points, not accepted truths. Recheck their current status before work begins and reproduce every claim.

| Lead | Reported opportunity | Initial action |
|---|---:|---|
| NumPy issue [#31969](https://github.com/numpy/numpy/issues/31969) | `np.unique` reportedly 46× slower on a large numeric case due to hash-path selection | Reproduce immediately; test adaptive thresholds by dtype, size, and cardinality |
| NumPy issue [#12778](https://github.com/numpy/numpy/issues/12778) | `np.inner` reportedly ~10× slower than an equivalent `tensordot` formulation | Reproduce across shapes; investigate bitwise-result and BLAS-routing differences |
| NumPy issues [#5922](https://github.com/numpy/numpy/issues/5922) and [#11156](https://github.com/numpy/numpy/issues/11156) | `ufunc.at` reportedly 10–25× slower for some grouped operations | Reconstruct current benchmarks; inspect indexing, ordering, aliasing, and atomicity semantics |
| NumPy issue [#21655](https://github.com/numpy/numpy/issues/21655) | Cache-blocked and parallel relayout produced large multi-fold gains | Port reproducer; implement native tiled transpose/relayout prototype |
| NumPy issue [#27456](https://github.com/numpy/numpy/issues/27456) | CPython call-site specialization reportedly exceeds 2× for small/medium arrays | Rebuild prototype on current CPython/NumPy; isolate reusable work-plan caching |
| NumPy issue [#23068](https://github.com/numpy/numpy/issues/23068) | SLEEF/SIMD prototype reportedly made `log10` ~4× faster | Audit current SIMD math paths and accuracy constraints; compare SLEEF and alternatives |
| NumPy issue [#27042](https://github.com/numpy/numpy/issues/27042) | Sorted-set cases reportedly 10–1000× faster with specialized algorithms | Reproduce; audit `sortednp`; determine safe opt-in property/API and license constraints |
| NumPy issue [#8208](https://github.com/numpy/numpy/issues/8208) | Four-thread `sin` demonstration reportedly ~3× faster | Rebuild using persistent native pool; measure crossover and bandwidth limits |
| NumPy issue [#27786](https://github.com/numpy/numpy/issues/27786) | Poor free-threaded scaling for repeated small-array workloads | Profile lock/cache contention and evaluate specialization/allocator fixes |
| NumPy roadmap | Better parallel execution remains an acknowledged opportunity | Map roadmap language to runtime architecture and upstream strategy |

Also search for newer issues superseding these leads. Do not waste weeks solving an old report that current `main` already fixed.

---

## 7. ProsPyctor Performance Lab implementation

### 7.1 Suggested technology

- Python 3.12+ for collection, orchestration, parsing, analysis, and reporting
- GitHub GraphQL or REST APIs with authenticated rate-limit handling
- Local git clones for complete diff, commit, blame, and branch analysis
- SQLite initially; DuckDB or PostgreSQL only when demonstrated scale requires it
- `pyperf` for Python-facing microbenchmarks
- NumPy's ASV benchmark suite for historical and upstream-compatible measurements
- Linux `perf` where available for cycles, instructions, cache misses, branches, and bandwidth clues
- Containerized or otherwise isolated reproducible environments
- Machine-readable JSON/Parquet results plus generated Markdown reports

Avoid constructing a large distributed platform before one-machine ingestion and reproduction works.

### 7.2 Ingestion requirements

- Incremental sync with resumable cursors
- Respect API limits and repository terms
- Store raw source metadata separately from normalized interpretations
- Content hashing to avoid duplicate processing
- Link resolution for duplicate issues, PRs, commits, gists, and external repositories
- Preserve deleted or unavailable-link metadata when content disappears
- Record ingestion version so extraction improvements can be rerun
- Never execute untrusted issue code on a host machine without isolation

### 7.3 AI extraction requirements

AI agents should extract:

- Claimed bottleneck
- Proposed cause
- Proposed algorithm or patch
- Benchmark and input-generation code
- Claimed hardware/software environment
- Correctness caveats
- Maintainer feedback
- Whether the work landed elsewhere
- Whether a third-party project already productized it
- Likely current relevance

Use deterministic parsers for metadata and AI for interpretation. Store quoted evidence locations so every extracted claim is auditable.

### 7.4 Reproducer generation

For every promising lead, generate a standalone benchmark that:

- Pins random seeds
- Records package versions, compiler, CPU, instruction sets, BLAS, OS, and thread settings
- Separates setup from timed work
- Warms up relevant caches and code paths
- Runs enough samples for stable distributions
- Checks output correctness outside the timed region
- Measures memory consumption when relevant
- Compares stock stable NumPy, NumPy `main`, candidate code, and known competitors
- Tests the reported case plus neighboring sizes/shapes to expose threshold cliffs

Generated reproducers are untrusted until an independent review agent confirms that they measure what they claim.

---

## 8. Benchmarking standard

### 8.1 Required controls

Record or control:

- CPU model, microcode, core topology, and available instruction sets
- Operating system and kernel
- Compiler and optimization flags
- Python and NumPy commit/version
- Linked BLAS/LAPACK implementation
- BLAS, OpenMP, and accelerator thread counts
- CPU affinity where available
- Warm-up policy
- Frequency scaling/turbo state when relevant
- Array alignment, order, strides, shape, dtype, byte order, and cardinality
- Allocation inclusion or exclusion
- Peak resident memory and temporary allocation volume when important

Report median, dispersion, sample count, and confidence—not only the best observation.

### 8.2 Performance matrices

For each operation, sweep:

- Tiny through very large input sizes
- Contiguous C order, contiguous Fortran order, and representative strided views
- Common dtypes and supported unusual dtypes
- Aligned and unaligned data where relevant
- Different axes and dimensionalities
- Different cardinalities or sortedness when they affect the algorithm
- Single thread and representative multicore counts
- Intel, AMD, and ARM machines when the candidate is intended to be portable

### 8.3 Benchmark interpretation

Classify wins as:

- **Algorithmic:** Better complexity or data-structure choice
- **Dispatch:** Better selection among existing algorithms
- **SIMD:** More work per instruction
- **Parallel:** More cores used effectively
- **Memory:** Fewer temporaries, copies, allocations, or cache misses
- **Call overhead:** Less repeated planning, casting, broadcasting, or Python/C dispatch
- **Composite:** Multiple mechanisms

This classification guides whether the improvement belongs in a fast path, dispatcher, SIMD backend, or parallel core.

---

## 9. Correctness and compatibility standard

Differentially compare every candidate against the applicable stock NumPy behavior. Test at minimum:

- Zero-length and singleton dimensions
- Scalars and zero-dimensional arrays
- C- and Fortran-contiguous arrays
- Positive, negative, zero, and unusual strides
- Transposed, sliced, broadcast, and misaligned views
- Read-only inputs
- Overlapping inputs and outputs
- `out=` and `where=` behavior where supported
- Axis tuples, negative axes, `keepdims`, initial values, and dtype overrides
- Casting rules and byte order
- NaN, infinity, signed zero, subnormal, and extreme values
- Integer overflow and warning behavior
- Floating-point status flags and exceptions where observable
- Complex values
- Datetime/timedelta where applicable
- Structured and object dtypes, normally through fallback
- ndarray subclasses and override protocols
- Free-threaded Python behavior
- Concurrent calls and nested external thread pools
- Determinism requirements for reductions

Use property-based testing and fuzzing to generate shapes, strides, dtypes, values, aliases, and operation arguments. A candidate cannot ship solely because curated fixtures pass.

Define comparison modes explicitly:

- Bit-identical required
- Numerically equivalent under NumPy's documented tolerance
- Ordering-equivalent where output ordering is not guaranteed
- Explicit opt-in relaxed mode

Default behavior must not silently weaken NumPy's guarantees.

---

## 10. PyOverdrive Runtime architecture

### 10.1 Gearbox adaptive dispatcher

Gearbox is the center of the runtime. It should be table-driven and testable independently from kernels.

Inputs may include operation, dtype signature, dimensions, size, strides, flags, CPU features, available threads, and calibrated thresholds. Output is a selected implementation plus a reason code.

Requirements:

- Extremely low dispatch overhead
- Deterministic decision for the same environment and inputs
- Conservative fallback
- Per-hardware default threshold tables produced by benchmarks
- Optional startup or installation-time calibration only if its value is proven
- Debug mode explaining why a path was selected
- Environment variable or API to disable individual optimizations
- Telemetry only if explicitly enabled; never require remote telemetry

Do not use a heavyweight learned model in a hot path. If machine learning assists threshold discovery offline, compile its conclusions into small decision tables or formulas.

### 10.2 Algorithmic fast paths

Each fast path must be isolated behind a stable internal interface and include:

- Applicability predicate
- Correctness contract
- Crossover benchmark
- Stock fallback
- Provenance record
- Unit and fuzz tests
- Performance regression benchmark
- Kill switch

Prefer narrow, independently shippable fast paths over a monolithic rewrite.

### 10.3 PyRallel adaptive parallel execution core

Begin PyRallel with a persistent native thread pool. Creating Python threads or a new executor for every NumPy call will erase many gains.

Initial requirements:

- Static chunking for large uniform contiguous work
- Dynamic scheduling only for demonstrably irregular work
- Low-overhead wake/sleep behavior
- Size- and operation-specific parallel crossover thresholds
- Coordination with BLAS/OpenMP and other thread pools
- Protection against nested oversubscription
- Clean shutdown and fork safety
- Exception and floating-point error propagation
- Safe handling of output aliasing and overlapping memory
- Deterministic or documented reduction behavior
- User-configurable maximum threads
- Single-thread fallback

Start with compute-heavy, independently chunkable operations such as selected transcendental ufuncs. Do not expect memory-bound addition or copying to scale linearly with cores.

Use `threadpoolctl` or compatible inspection/control where useful, but do not make correctness depend on every third-party library cooperating.

### 10.4 SIMD and vector math

- Reuse NumPy's existing CPU-feature detection where possible
- Compare native intrinsics, compiler auto-vectorization, SLEEF, Highway, and other compatible backends
- Maintain scalar fallbacks
- Test accuracy across the entire relevant numeric domain, not only random ordinary values
- Monitor binary-size and compile-time costs
- Dispatch by supported instruction set at runtime where practical
- Never execute unsupported instructions based on build-machine assumptions

### 10.5 Integration route investigation

Prototype and compare:

1. Public NumPy extension/registration hooks
2. A CPython extension operating on real `numpy.ndarray` objects
3. Limited opt-in Python activation
4. Small upstream NumPy PRs
5. An alternate binary wheel as a last resort

Document limitations before choosing. Avoid fragile monkey-patching as the permanent architecture. The best long-term outcome is likely a combination of an independent research/runtime project and small improvements upstreamed into NumPy.

---

## 11. Development phases and gates

### Phase 0: Repository and measurement foundation

Deliver:

- Monorepo scaffolding
- Reproducible development environment
- CI on Linux, Windows, and macOS
- Baseline compatibility harness
- `pyperf` and ASV integration
- Machine fingerprinting
- Opportunity-record schema
- Architecture decision records

Gate: A trivial no-op candidate can be benchmarked, correctness-tested, packaged, and compared with stock NumPy automatically.

### Phase 1: ProsPyctor gold-mine ingestion

Deliver:

- Full NumPy issue/PR incremental collector
- Search and AI extraction pipeline
- Duplicate/cross-reference graph
- At least 100 automatically extracted performance leads
- Human/agent-auditable Markdown reports
- Reproducers for the initial high-value leads

Gate: At least ten historical performance claims are successfully rerun on current NumPy, including rejected and stale results.

### Phase 2: First verified fast paths

Target three to five small opportunities with strong evidence. `np.unique`, multidimensional `inner`, selected `ufunc.at` cases, and relayout are initial candidates, not mandatory choices.

Gate:

- At least three verified improvements
- At least one 10× result on a meaningful documented input regime, or a documented conclusion that initial 10× claims do not survive reproduction
- Complete differential and fuzz coverage for supported cases
- No greater than 2% statistically credible regression on unaffected benchmark cases

### Phase 3: Gearbox adaptive dispatcher

Deliver:

- Low-overhead decision engine
- Calibrated thresholds
- Explain/debug mode
- Fallback and kill switches
- Integration of the first fast paths

Gate: Dispatch overhead does not erase gains, and threshold boundaries have no severe performance cliffs.

### Phase 4: PyRallel execution prototype

Implement the persistent pool for a small operation set, initially considering `sin`, `exp`, `log`, selected reductions, and mask/reduction combinations.

Gate:

- Demonstrated scaling on compute-heavy large arrays
- No oversubscription catastrophe with common BLAS/OpenMP configurations
- Correct behavior under concurrency, exceptions, and process shutdown
- Automatic single-thread selection below measured thresholds

### Phase 5: Integrated MVP

Deliver an experimental `pyoverdrive` wheel containing Gearbox, fast paths, PyRallel, configuration, diagnostics, documentation, and Dyno benchmark reports.

MVP success targets:

- Several meaningful 2×+ wins
- Multiple 5×+ wins
- At least one credible 10×+ case if the evidence supports it
- Less than 2% regression across the protected baseline suite
- Correct fallback for unsupported cases
- Installation and rollback documented

### Phase 6: Expansion and upstreaming

- Continuously mine new and historical leads
- Add operations based on measured ecosystem value
- Submit small upstream NumPy PRs where appropriate
- Add real workloads from scientific computing, data processing, simulation, and AI preprocessing
- Publish transparent performance reports including losses and limitations

---

## 12. AI-team operating instructions

The lead AI agent should divide work into bounded research, implementation, benchmarking, and adversarial-review tasks. Independent agents should verify important conclusions rather than merely reviewing one another's prose.

For each opportunity:

1. A research agent reconstructs the complete history and competitor landscape.
2. A reproduction agent creates and runs the benchmark.
3. An implementation agent develops the candidate optimization.
4. A correctness agent attempts to break it with edge cases and fuzzing.
5. A performance-review agent checks methodology and searches for regressions.
6. A licensing/provenance agent confirms what may be reused.
7. The lead agent accepts, rejects, or requests another iteration based on artifacts and measured evidence.

Agents must:

- Commit small, reviewable changes
- Record assumptions and unresolved risks
- Add tests before declaring work complete
- Preserve unrelated repository changes
- Re-run relevant benchmarks after every material optimization
- Compare against current stable NumPy and NumPy `main`
- Search for an existing implementation before building anything substantial
- Check whether a claimed “new” idea already exists under another name
- Update the opportunity record and architecture documentation
- Prefer complete working scripts and reproducible commands over fragments

Agents must not:

- Invent benchmark results
- Select only favorable input sizes
- compare against incorrectly configured competitors
- Hide correctness differences behind loose tolerances
- Treat one CPU as universal evidence
- rewrite large subsystems before proving a narrow mechanism
- copy code without provenance and license review
- close an opportunity as impossible merely because the first implementation fails
- continue an approach after its predefined kill criteria are met

---

## 13. Opportunity workflow

Every candidate follows this state machine:

```text
discovered
→ extracted
→ deduplicated
→ reproduced_or_rejected
→ algorithm_audited
→ prototyped
→ correctness_validated
→ performance_validated
→ integration_reviewed
→ accepted_or_rejected
→ shipped_or_upstreamed
→ continuously_monitored
```

Required output files per serious opportunity:

```text
docs/research/opportunities/OPP-XXXXXX.md
benchmarks/historical/opp_xxxxxx.py
benchmarks/results/OPP-XXXXXX/<machine-fingerprint>.json
compatibility/fixtures/opp_xxxxxx/
```

The Markdown report should answer:

- What is slow?
- Why is it slow?
- Who reported it and when?
- Does it still reproduce?
- What existing projects already solve it?
- What alternative algorithms exist?
- What semantics differ?
- Where is the crossover point?
- Is it common enough to matter?
- Can it be implemented safely and maintained?
- Should it be shipped independently, upstreamed, monitored, or rejected?

---

## 14. Initial benchmark workloads

Microbenchmarks must be supplemented with representative workload kernels:

- Data cleaning and filtering
- Feature preprocessing for machine learning
- Reinforcement-learning environment simulation with many small arrays
- Image/signal transformations on CPU
- Scientific parameter sweeps
- Statistical aggregation and grouped updates
- Mask creation followed by reductions
- Sorted set intersection and membership queries
- Array relayout before repeated axis-oriented computation
- Repeated small linear algebra and geometry operations

Do not claim that accelerating these kernels automatically accelerates GPU model training. Measure end-to-end effects where possible.

---

## 15. Distribution, licensing, and upstream strategy

- Prefer a permissive license compatible with NumPy's BSD-3-Clause ecosystem, subject to legal review.
- Maintain a machine-readable third-party provenance inventory.
- Keep research reproductions of incompatible code isolated and never ship that code.
- Avoid using NumPy's name in a way that implies official endorsement.
- Follow the NumPy fair-play guidance and NumFOCUS trademark policy: keep `NumPy` out of PyOverdrive's package, repository, domain, and formal product name unless written permission is obtained. Truthfully describe compatibility using phrasing such as “PyOverdrive for NumPy.”
- Make disabling and uninstalling the accelerator straightforward.
- Publish reproducible benchmark commands and raw results.
- Submit focused improvements upstream where they naturally belong.
- Do not allow upstream review latency to prevent independent experimental validation.
- Avoid a permanent hard fork if extension hooks or upstreamable modules can achieve the goal.

ProsPyctor itself may become a valuable open-source component even if some runtime experiments fail. Its issue-mining and Dyno benchmark-reproduction system should be general enough to apply to SciPy and other numerical libraries later, but do not generalize prematurely.

---

## 16. Kill criteria and anti-delusion rules

Stop or redesign a candidate when:

- The claimed speedup does not reproduce on current stable NumPy or `main`
- The win exists only because the baseline is misconfigured
- Correctness requires an unacceptable semantic change
- Dispatch overhead eliminates the practical gain
- The accelerated case is vanishingly rare and maintenance cost is high
- A maintained existing project already provides the same drop-in solution
- Cross-platform behavior cannot be made safe
- Performance regresses materially in common neighboring cases
- Licensing prevents a clean implementation

Do not stop the entire project because individual leads fail. The Lab exists specifically to reject weak leads cheaply and redirect effort toward stronger ones.

Conversely, do not expand scope based on one spectacular benchmark. Require a performance surface across sizes, shapes, dtypes, and hardware.

---

## 17. Definition of done for the first public release

The first public release is complete only when:

- Installation works from documented wheels on supported platforms
- Supported operations and exact applicability regimes are documented
- All unsupported cases fall back correctly
- Differential, property-based, fuzz, concurrency, and packaging tests pass
- Benchmarks are reproducible from published commands
- Raw benchmark data and machine fingerprints are included
- At least three optimizations provide meaningful verified wins
- No protected baseline shows an unexplained material regression
- Thread-pool interaction and oversubscription behavior are documented
- Every shipped implementation has a provenance and license record
- Users can inspect, disable, and troubleshoot selected paths
- Marketing language accurately distinguishes targeted speedups from whole-program speedups

---

## 18. Immediate first tasks

The first lead agent should execute these tasks in order:

1. Create the `pyoverdrive` monorepo structure and development instructions.
2. Pin current stable NumPy and create an automated build of NumPy `main`.
3. Integrate NumPy's ASV suite and create the machine-fingerprint format.
4. Implement the opportunity schema and SQLite database.
5. Ingest all NumPy issues and PR metadata incrementally.
6. Add full-text and structured searches for the performance vocabulary above.
7. Produce initial records for issues #31969, #12778, #5922, #11156, #21655, #27456, #23068, #27042, #8208, and #27786.
8. Independently reproduce #31969 and #12778 against stable NumPy and `main`.
9. Audit whether existing PRs, packages, or newer issues already solve them.
10. Select the first implementation only after the benchmark and competitor audit are complete.
11. Build the smallest safe fast path and differential-test it.
12. Publish the first evidence report, including failures and negative results.

Do not begin the general parallel core before the measurement infrastructure can demonstrate when parallel execution helps and when it hurts. Once that foundation exists, prototype the persistent pool on a small set of compute-heavy ufuncs.

---

## 19. Primary references

- [NumPy repository](https://github.com/numpy/numpy)
- [NumPy roadmap](https://numpy.org/neps/roadmap.html)
- [NumPy NEP 36: fair-play guidance for external projects](https://numpy.org/neps/nep-0036-fair-play.html)
- [NumFOCUS trademark guidelines](https://numfocus.org/trademark-guidelines)
- [NumPy global configuration and threading notes](https://numpy.org/doc/2.4/reference/global_state.html)
- [NumPy thread-safety documentation](https://numpy.org/doc/2.4/reference/thread_safety.html)
- [NumPy ASV benchmarks](https://github.com/numpy/numpy/tree/main/benchmarks)
- [NEP 38: SIMD optimizations](https://numpy.org/neps/nep-0038-SIMD-optimizations.html)
- [NEP 11: deferred ufunc evaluation](https://numpy.org/neps/nep-0011-deferred-ufunc-evaluation.html)
- [NumExpr performance rationale](https://numexpr.readthedocs.io/en/stable/intro.html)
- [Intel mkl_umath](https://github.com/IntelPython/mkl_umath)
- [cuPyNumeric documentation](https://docs.nvidia.com/cupynumeric/latest/)
- [Bohrium repository](https://github.com/bh107/bohrium)
- [PyTorch compilation of NumPy code](https://pytorch.org/blog/compiling-numpy-code/)

All links and issue states are time-sensitive. Refresh them during ingestion and record the retrieval time.

---

## 20. Final directive to the AI team

Be aggressively curious and empirically ruthless.

Assume the NumPy ecosystem contains years of overlooked performance work: abandoned prototypes, forgotten benchmarks, downstream workarounds, rejected patches, shape-specific algorithm gaps, missing SIMD kernels, poor dispatch thresholds, unnecessary temporary arrays, and operations that never received enough maintainer time. Search all of it.

At the same time, assume every exciting claim may be stale, narrow, incorrectly measured, semantically incompatible, or already productized elsewhere. Prove it again.

PyOverdrive's advantage is not one imagined magical algorithm. Its advantage is the ability to apply enormous AI research and engineering effort across thousands of historical clues, test them systematically with Dyno, combine compatible discoveries, and convert verified wins into a safe adaptive runtime for the actual NumPy ecosystem.

Build **PyOverdrive — NumPy at full throttle.**
