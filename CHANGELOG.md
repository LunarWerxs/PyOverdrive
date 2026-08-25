# Changelog

## 0.1.0 (unreleased)

First public release candidate. MIT licensed.

### Fast paths

Forty-five always-on fast-path families (67 always-on paths of 69
registered; the remaining two are one calibration-gated path and a
disabled dispatch-overhead test artifact) behind `pyoverdrive.enable()`.
Every threshold comes from committed, machine-fingerprinted benchmark
evidence under `benchmarks/results/`; the headline end-to-end wins on
the reference machines include:

- `np.apply_along_axis` with one of NumPy's own reducers, served as the
  equivalent `axis=` call instead of a Python loop over 1-D slices
  (55-292x, bit-identical; the margin scales with slice count)
- `np.vectorize` wrapping one of NumPy's own ufuncs, called directly
  instead of through the object loop (14-101x, bit-identical; served by
  installing a subclass of `np.vectorize`, so `isinstance` and `type()`
  keep working)
- the singular-value family on 2x2/3x3 stacks from one closed-form
  core: `np.linalg.pinv` via the adjugate (8.6-24x), `np.linalg.norm`
  with `ord=2` (8.0-15.1x) and `np.linalg.svd(compute_uv=False)`
  (7.8-12.1x), each under its own measured conditioning band plus a
  degeneracy guard, with out-of-band matrices split out to stock
- `np.isin` on StringDType (2317x at the upstream-reported shape) and
  on object arrays (240-262x at scale)
- `np.searchsorted` with out-of-dtype-range Python int keys (~1000x:
  an O(1) provable answer vs a per-element bigint walk)
- `np.quantile`/`np.percentile` with dense quantile arrays (34-40x;
  up to 918x candidate-level), `np.nanquantile`/`np.nanpercentile`
  many-slice (22-229x / 3.3-178x)
- integer `np.matmul`/`np.dot` via exact float64 BLAS under a provable
  2^53 bound (2.6-28.5x, bit-exact), and the nan-family wrappers
  (`nanmean` up to 12.4x, `nanargmax` up to 5.3x, `nanmedian` 2-D up
  to 2.6x, `nan_to_num` ~2x) via a single isnan scan
- batched `np.linalg.det`/`slogdet` closed forms on 2x2/3x3/4x4 stacks
  and `solve` on 2x2/3x3, joining the shipped `inv` adjugate family.
  Measured END TO END with the result consumed: det 1.23-4.9x, slogdet
  1.19-2.8x, Cramer solve 1.53-3.1x, inv 1.3-11.5x. The previously
  advertised "up to 91.9x" was a candidate-level figure that excluded the
  guard, and the floors derived from it put the 2x2 path at 0.70x - a
  dispatched REGRESSION - at its own floor. The guard is fused into the run
  and every window was re-derived from end-to-end measurements.
- batched small-matrix linear algebra: `np.linalg.eigvalsh` on 2x2
  stacks (26-31x) and 3x3 stacks via the trigonometric closed form
  (2.7-4.4x; near-degenerate cells are split out and served by stock,
  so a few coalesced pairs no longer cost the stack its speedup),
  `np.linalg.inv` on 2x2/3x3 stacks (1.3-11.5x end-to-end), `np.linalg.cholesky`
  on 2x2/3x3 stacks via a fused guard-in-run Cholesky-Crout pass in
  cache-sized chunks (1.6-2.3x from batch 1000 up, no cap), and
  `np.linalg.qr` on 2x2/3x3 stacks via unrolled Householder
  reflectors in LAPACK's own sign convention (2.2-11x with Q,
  1.7-8.2x R-only; rank-deficient matrices split out to stock, where
  two valid factorizations can legitimately disagree)
- `np.einsum` chains of three or more operands routed through numpy's
  own `optimize=True` above a measured naive-loop-volume gate (7.4x at
  volume 332k rising to 178-234x; stock's default runs one fused loop
  over every index), extending the shipped two-operand regime;
  ellipsis spellings ('...ij,...jk->...ik' class) admitted for both
  regimes under their own measured floors (up to 163x end-to-end)
- `np.interp` on uniformly spaced sample grids via direct index
  arithmetic instead of per-query bisection (2.8-6x from 3000 queries
  up), and `np.take` with `out=` via fancy-index gather + assign
  (1.3-3.3x from 1000 gathered elements up, bit-exact, preserving
  stock's out-unchanged-on-bad-index guarantee)
- `np.sort`/`np.unique` on single-character strings (33x / 26.6x),
  `np.unique(axis=0)` on integer rows (1.8-5x), sort-based integer
  `np.unique` (37-55x)
- `np.convolve`/`np.correlate` via FFT (3.6-14x end-to-end),
  `np.einsum` two-operand reordering (3.4-27x), `np.inner` on stacked
  operands (1.2-7.0x inside its measured regime), `np.intersect1d`
  (1.5x at its floor rising to 21-90x at scale)
- `np.matmul` complex-times-real in upcast-dominated shapes (1.55-7.5x),
  `np.dot` real-matrix-times-complex-vector (12-44x)
- channel-style reductions (`np.mean`/`np.sum` over leading axes with a
  tiny trailing axis, 2.2-14.4x), small-array `np.median` (1.5-3.4x),
  small 1-D `np.roll` (2.4-5.9x), uniform-bin `np.histogram2d`
  (1.4-2.4x), threaded elementwise ufuncs (1.34-1.63x at their floors,
  1.8-2.2x at 1e7, measured end-to-end as the worst of sorted/shuffled
  input and bare/consumed result)
- 1-D constant-mode `np.pad` as one allocation plus one assignment
  (1.5-4.6x, bit-identical). Quoted CONSUMED, i.e. with the padded array
  summed before the clock stops: the no-constant route allocates with
  calloc, so a bare timing reports up to 342x for a shape that is
  actually 0.86x once the result is read. The size cap is set on the
  length of the RESULT for the same reason.

### Safety machinery

- Gearbox dispatcher: conservative predicates, automatic stock
  fallback, per-path kill switches (`PYOVERDRIVE_DISABLE`,
  `disable_path`), `explain()` for dispatch decisions, and
  `selfcheck()` proving every path against stock on the host machine.
- Class-backed paths: where the slow work lives on an instance rather
  than in a function (`np.vectorize`), the dispatcher installs a
  SUBCLASS of the stock class rather than wrapping the name with a
  function, so `isinstance`, `type()`, attributes and `__name__` keep
  working and `disable()` restores the original class object. Their
  kill switch is live: the installed subclass consults it per call.
- Per-machine calibration (`python -m pyoverdrive --calibrate`) for
  paths whose wins are architecture-dependent: verdicts persist per
  machine fingerprint; foreign or stale calibration files are ignored;
  with no file, gated paths stay off.
- That now covers the THREADED families too, which ship enabled and whose
  thresholds are the least transferable numbers here (core count, cache,
  memory channels). `--calibrate` re-times every threaded row at its own
  floor - the weakest cell its predicate admits - and switches off the ones
  that do not pay on the host. It only ever removes rows: finding where a
  row starts paying needs the full sweep, not a probe. Two guards make it
  trustworthy where an ordinary probe would not be. Each cell runs in a
  FRESH process that re-draws until it lands on a fast core, because a
  threaded candidate measured against a single-threaded baseline inherits
  the P-core/E-core coin flip; and a row is removed only when two
  independent readings AGREE, since there is no portable way to ask whether
  the machine is busy and disagreement is exactly what a busy one produces.
  Disagreement keeps the shipped row.
- Bit-identical results on 43 of the 67 always-on paths. Two more (the
  FFT convolve/correlate pair) are bit-identical for integer dtypes
  and numeric for floats, and the remaining 22 run in documented
  numeric mode, each with a measured tolerance. Every path's
  comparison mode is recorded in its provenance, checked by the
  differential suite, and the counts above are re-derived from the
  live registry by tools/verify_claims.py rather than maintained by
  hand - they had drifted before that check existed.

### Verification

- 2207 tests: hand-written differential suites per path plus a
  hypothesis property net over every patched operation.
- The threaded-ufunc thresholds were re-derived from scratch after the old
  ones were found to rest on a measurement artifact, and 15 of their 16 rows
  moved. On a hybrid CPU (P-cores + E-cores) a single-threaded process is
  placed on one class of core and stays there, so the BASELINE of every
  threading speedup is a per-process coin flip - the same np.sin float64
  n=1e5 baseline measured 344 us in 15 of 25 fresh processes and 497 us in
  the other 10, 1.44x apart. The threaded candidate spans cores and averages
  over the split, so the flip moves only the denominator, always in the
  candidate's favour, and it is invisible to re-runs, larger sample counts,
  medians and idle-machine checks alike because it is perfectly reproducible
  *within* a process. Thresholds now come from `tools/calibrate_dispatch.py`:
  end to end through the patched name, one cell per process, sides
  interleaved, only on processes that drew a fast core, and each threshold
  must clear 1.3x on the worst of {sorted, shuffled} x {bare, consumed}
  input at that size and every larger measured one, on TWO independent
  sweeps with the worse reading kept per cell - several margins are within
  0.06x of the bar, and a threshold is a green, so it has to reproduce just
  as a red does. The second sweep moved no floor. `np.sqrt` left the
  threaded family entirely - it is memory-bandwidth bound and reaches 1.3x
  at no measured size, having shipped at 1.05x at its own advertised floor.
  Write-up: `docs/research/hybrid-cpu-baseline-coin-flip.md`.
- The threaded BINARY family (`np.add` and friends) came from the same
  battery and was corrected the same way, with one extra rule: its wins are
  bandwidth, they cross the 1.3x bar only between 1e7 and 3e7 elements, and
  the run-to-run spread there is as wide as the margin being measured - one
  sweep read `subtract` float64 at 1.23x, 1.14x, 1.33x on consecutive sizes.
  So its floors come from TWO independent sweeps with the worse reading kept
  per cell; a row ships only if it cleared the bar twice. At the old 1e6
  floors the family actually delivered 1.04-1.20x, and `subtract` float32 at
  its 3e6 floor ran at 0.97x - a dispatched loss. What survives is `np.add`,
  `maximum` and `minimum` on float64/int64 and `subtract`/`multiply` on
  int64, at 1e7-2e7 elements. Every float32 row is gone, and `np.divide`
  left the family entirely. This is the strongest candidate in the project
  for per-machine calibration rather than a shipped table.
- An audit of all 44 fast-path modules against the measurement defects
  already found in three of them turned up two more, both confirmed by
  measurement rather than by reading:
  - `np.linalg.inv` computed the determinant in its PREDICATE for the
    conditioning check and again in the run, plus a separate finiteness
    scan of the whole stack. At batch 4096 that guard cost 128.5us against
    25.8us for the entire 2x2 inverse it was protecting. Fused into the run
    (the pattern det/slogdet/solve already use), and the finiteness test now
    falls out of the conditioning scale for free: 2x2 went 3.2x -> 11.5x at
    batch 10k, 3x3 1.38x -> 1.5x.
  - The det/slogdet/solve guard computed that same conditioning scale TWICE
    per call, once in the shared helper and again in the check, and computed
    it with a multi-axis reduction that numpy does badly on a tiny trailing
    shape. Threaded through and folded over the entry views instead, with a
    measured per-dimension crossover because folding loses on small batches
    where per-call overhead is the whole cost. det 2x2 at its own floor went
    0.97x -> 1.37x, slogdet 3x3 1.01x -> 1.19x.
- `np.inner` on stacked operands had NO size gate - any ndim>2 pair was
  accepted - and its only quoted speedup was the upstream issue's own "~10x",
  never re-measured through the predicate and run. Measured end to end it
  ranges from 0.38x to 6.98x, i.e. it was making ordinary small calls 2.6x
  SLOWER. The selfcheck could not see it because its canonical input was the
  first shape in the sweep that happened to win. Worse, the wins and losses
  INTERLEAVE - (4,256,512) is 0.43x while (20,16,512) is 1.23x - so no
  function of volume or output size separates them. The gate is therefore
  restrictive rather than clever: it admits only the corner where every
  measured cell won (1.27x-6.98x) and leaves the rest on stock, forfeiting
  real wins mixed in with the losses. Confirmed to hold outside the
  measuring grid too, including the operand orientation the grid never used.
- `np.histogram2d`'s uniform-bin path gated on BIN count alone, which is the
  wrong axis by itself: its cost scales with the bins it allocates and
  clears, stock's with the samples it walks. Few samples into many bins was
  therefore its losing corner and it was shipping - 0.75-0.81x at 200
  samples and 0.82-0.98x at 500, across 30x30, 60x60 and 100x100 bins
  alike. It has a sample floor now (2000, the first count with real
  headroom: 1.57-2.02x).
- The pessimization sweep grew a `--sizes` mode: every cell also judged at
  3x, 10x, 30x and 100x its canonical size, since each canonical input sits
  near the BOTTOM of what its path accepts - and at a third, a tenth, a
  thirtieth and a hundredth of it, since downward is where a path with no
  floor at all keeps accepting. 693 cells. Both directions found real
  losses: upward the three det/slogdet cells, downward np.inner and
  np.histogram2d. A red is re-measured in a second process
  and only reported if it reproduces.
- ...and a `--shapes` mode, closing the aspect-ratio class the size sweep
  can never reach: each 2-D-or-deeper cell is re-judged at roughly constant
  volume with its trailing axis grown 4x and 16x while the leading axis
  shrinks by the same factor, and the reverse - np.inner's 0.38x corner had
  the same element count as its canonical input, so no size multiple would
  ever have produced it. Operand coupling is handled by a shared-trailing
  rule plus a chain variant for (m,k)x(k,n) products. On the idle Intel
  box, 33 aspect cells dispatch and none is below 1.0x; the refused cells
  were spot-checked to be genuine predicate refusals (valid shapes, the
  gate declining), including the long-contraction corner np.inner used to
  lose in (docs/research/batch15-notes.md).
- Two more audit findings were REFUTED by measuring them, which is the point
  of measuring: `np.vectorize` needs no floor (it wins 3.1x even at size 1,
  because stock is slow at every size), and `intersect1d`'s 1.19x at its
  floor was a contended reading - the idle box says 1.50x. `svd`'s
  "PROVISIONAL" calibration label was discharged the same way at 2.38x.
- Three det/slogdet cells were dispatching into a LOSS at the TOP of their
  window - det 3x3 1.01x at 1e5, slogdet 3x3 0.84x at 1e5, slogdet 4x4
  0.85x at 3e4 - and nothing caught them, because the pessimization sweep
  judges one canonical input per path and for these it sits near the floor.
  A path can be honest where it is checked and lose where it is not. det 3x3
  and slogdet 3x3 gained upper caps (they had none) and the two 4x4 caps
  came down from 3e4 to 1e4. Every cell inside every window now measures at
  least 1.22x.
- `tools/verify_no_pessimization.py` inherited the same defect and could
  return a false green on threaded paths; it now rejects measurements taken
  on a slow core too. Its other limit is now stated rather than implied: it
  probes ONE canonical input per path, so a loss confined to another dtype
  does not appear there - which is exactly how `subtract` float32 hid.
- Full gate green on Windows x86-64 (AMD Zen 4 + Intel Alder Lake) and on
  Linux x86-64 in CI across CPython 3.12/3.13/3.14 against numpy 2.0.2,
  2.4.5 and latest, plus a clean-venv wheel-install check.
- Two genuine upstream numpy findings made along the way and FILED:
  StringDType NUL-handling defects (numpy/numpy#32414) and a uint64
  searchsorted promotion case added to numpy/numpy#29727. A third bug was
  hit independently while getting the Linux gate green - np.linalg.pinv
  never returns for a matrix 3x3 or larger carrying an infinity on the
  diagonal - and turned out to be numpy/numpy#7461, open upstream since
  2016; it is narrowed here to compute_uv=True on Linux only
  (docs/research/upstream-pinv-inf-hang.md).
