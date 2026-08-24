# ADR-0002: PyRallel prototype: shared thread pool for unary elementwise ufuncs

Date: 2026-08-23 · Status: accepted

Context: Phase 4 (spec section 11) asks for a persistent pool for a small
operation set with automatic single-thread selection below measured
thresholds. OPP-000008 is the founding evidence: np.sin over 16 threads
reached 7.2x at 1e7 elements on the first calibrated machine because NumPy's
compute-bound ufunc loops release the GIL.

## Decisions

1. **One process-wide `ThreadPoolExecutor`, created lazily, never per call.**
   Executor startup is on the order of the whole win at 1e5 elements; the
   OPP-000008 reproducer had to pre-create its executors for the same reason.
   The pool is capped at `min(cpu_count, 16)` (16 is the widest configuration
   with committed evidence) and overridable with `PYOVERDRIVE_THREADS`.
   `PYOVERDRIVE_THREADS=1` is the whole-core kill switch on top of the
   per-op `PYOVERDRIVE_DISABLE=pyrallel_<op>` switches Gearbox provides.

2. **Workers only ever run stock NumPy kernels, and tasks never wait on
   other tasks.** Chunk tasks call the real ufunc obtained through
   `Gearbox.stock_fn` (never the patched public name; the OPP-000000
   recursion incident), and they never submit further work, so pool deadlock
   is impossible by construction and user code calling patched ufuncs from
   its own threads merely queues on the shared pool.

3. **Unary elementwise ufuncs only, bit-identical contract.** Elementwise
   kernels have no cross-element data flow, so chunking cannot change a
   result bit; the calibration battery checks bit-identity on every measured
   case and the differential suite asserts it. Reductions are deliberately
   excluded from this prototype: chunking a reduction changes the pairwise
   summation tree, which is a NUMERIC contract, a different promise needing
   its own evidence and tolerance policy. Binary ufuncs (np.add measured
   2.4-2.75x at 1e7 on this machine) are also deferred: the win is memory
   bandwidth, not compute, so it is far more machine dependent and needs a
   broadcasting-aware predicate.

4. **Dispatch surface is `op(x)` and `op(x, out=o)` with exact-match
   `out`.** The allocating form is what users write; the `out=` form is the
   source issue's own idiom and avoids first-touch page faults in the
   workers, measured 25-40% slower than a pre-touched buffer at 1e6 float64.
   `out` must be a plain, writeable, C-contiguous ndarray of identical shape
   and dtype (`out is x` in-place is fine: each chunk owns its index range).
   Anything needing a cast, broadcast, `where=`, or a strided write stays on
   stock, untouched.

5. **Thresholds are a table of measured crossovers, per (op, dtype), with a
   shared size-to-thread-count schedule.** Both live in
   `src/pyoverdrive/fastpaths/parallel_ufunc.py` as literals with their
   evidence path, regenerated from
   `benchmarks/micro/bench_pyrallel_calibration.py`. A pair with no measured
   win is absent from the table and runs on stock; that is the automatic
   single-thread selection the Phase 4 gate asks for. The battery measured
   float32 needing roughly 3x more elements than float64 to break even (its
   kernels are ~2x faster per element, so fixed dispatch cost weighs more)
   and sqrt, the in-battery negative control, scaling poorly (memory bound).

6. **Patched ufuncs mirror their ufunc attributes.** Patching replaces a
   `np.ufunc` with a Python function, so Gearbox now copies `nin`, `nout`,
   `types`, `identity`, `at`, `reduce`, `accumulate`, `outer`, and friends
   onto the wrapper. Duck-typed callers keep working;
   `isinstance(np.sin, np.ufunc)` still goes False while patched, a
   documented limit of the experimental patching route (spec 10.5) that the
   extension-hook route is meant to remove.

7. **Dyno records foreign CPU load with every result.** The first
   calibration battery ran while other agent sessions held the machine at
   65-96% CPU: single-thread baselines matched the quiet OPP-000008 run
   exactly (5.07 vs 5.08 ms) while every multi-thread candidate came out ~2x
   worse. `lab/dyno/load.py` now samples busy fraction before and after a
   suite, stores it under `conditions`, and flags the run `contended` above
   20%, so a reader can tell a real crossover from a starved one. Thresholds
   calibrated from a contended run are conservative (wins understated), never
   optimistic, which is the safe direction for a dispatch table.

## Consequences

- PyRallel's oversubscription posture is "degrade by OS scheduling": chunk
  kernels call no BLAS/OpenMP, so PyRallel never nests inside another
  runtime, and the pool is bounded; a concurrent BLAS workload slows the
  chunks down rather than triggering runaway thread creation. Coarse
  measurement 2026-08-23 (28% foreign load): three 2500x2500 OpenBLAS
  matmuls in one thread concurrent with three threaded `np.sin` calls on
  1e7 float64 in another took 0.85-1.17x the wall time of running the two
  sequentially. No catastrophe; a proper Dyno case sweeping BLAS thread
  counts is still worth adding.
- Windows thread wake-up latency puts the crossover at ~1e5 elements for
  float64 transcendentals here; Linux is expected to cross lower. Hardware
  decides: the table is per fingerprint, and a second machine's battery is
  the next evidence to collect.
- Exceptions inside a chunk surface as a Gearbox fallback plus a one-time
  RuntimeWarning; the half-written buffer is private to the failed call,
  except in the `out=` form where the caller's buffer may be partially
  written before stock recomputes it in full. Stock then overwrites every
  element, so the final state is still exactly stock's.
