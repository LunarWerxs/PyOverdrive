# Changelog

## 0.1.0 (unreleased)

First public release candidate. MIT licensed.

### Fast paths

Forty-five always-on fast-path families (69 always-on paths of 71
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
- batched 2x2/3x3 `np.linalg.det/slogdet/solve` closed forms (det 2x2
  up to 91.9x), joining the shipped `inv` adjugate family
- batched small-matrix linear algebra: `np.linalg.eigvalsh` on 2x2
  stacks (26-31x) and 3x3 stacks via the trigonometric closed form
  (2.7-4.4x; near-degenerate cells are split out and served by stock,
  so a few coalesced pairs no longer cost the stack its speedup),
  `np.linalg.inv` on 2x2/3x3 stacks (3-16.5x), `np.linalg.cholesky`
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
  operands (18-24x), `np.intersect1d` (21-90x)
- `np.matmul` complex-times-real in upcast-dominated shapes (1.55-7.5x),
  `np.dot` real-matrix-times-complex-vector (12-44x)
- channel-style reductions (`np.mean`/`np.sum` over leading axes with a
  tiny trailing axis, 2.2-14.4x), small-array `np.median` (1.5-3.4x),
  small 1-D `np.roll` (2.4-5.9x), uniform-bin `np.histogram2d`
  (1.4-2.4x), threaded elementwise ufuncs (5.8-9.7x at size)
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
- Bit-identical results on 45 of the 69 always-on paths. Two more (the
  FFT convolve/correlate pair) are bit-identical for integer dtypes
  and numeric for floats, and the remaining 22 run in documented
  numeric mode, each with a measured tolerance. Every path's
  comparison mode is recorded in its provenance, checked by the
  differential suite, and the counts above are re-derived from the
  live registry by tools/verify_claims.py rather than maintained by
  hand - they had drifted before that check existed.

### Verification

- 2261 tests: hand-written differential suites per path plus a
  hypothesis property net over every patched operation.
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
