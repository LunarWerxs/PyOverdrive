# PyOverdrive

**NumPy at full throttle.** Two lines, and NumPy code you have already written
gets faster. Nothing else changes.

```bash
pip install pyoverdrive
```

```python
import numpy as np
import pyoverdrive

pyoverdrive.enable()
# your existing code, unchanged. Supported calls take a measured fast path;
# everything else is stock NumPy, untouched.
```

## Why it is fast

Not by rewriting NumPy. NumPy's kernels are excellent - but a handful of its
functions reach them by a slow road: a Python loop where a vectorized call
exists, a per-matrix LAPACK dispatch on a stack of 2x2s, a planner switched off
by default to protect tiny inputs. PyOverdrive recognizes those calls and takes
the short road, and stays out of the way everywhere else.

| Your call | What PyOverdrive does instead | Measured |
|---|---|---:|
| `np.isin(a, b)` on `StringDType` | hash-set membership | **1593x** |
| `np.searchsorted(a, huge_python_int)` | an O(1) provable answer | **1041x** |
| `np.apply_along_axis(np.mean, -1, a)` | the `axis=` reduction, once | **217x** |
| `np.vectorize(np.sqrt)(x)` | calls the wrapped ufunc directly | **186x** |
| `np.einsum('ij,jk,kl->il', a, b, c)` | NumPy's own planner, above a measured size gate | **174x** |
| `np.linalg.qr(stack_of_10k_3x3)` | closed-form Householder, vectorized | **3.8x** |

Measured end-to-end through the public API - `pyoverdrive.enable()` on, your
call unchanged - on an idle Intel i7-12700K, NumPy 2.5.2, at 0% background
load. Every number in this repository ships with the machine that produced it
and the JSON it came from, under
[`benchmarks/results/`](benchmarks/results). Nothing is admitted on one
machine's word: a fast path ships only if it wins on both benchmark machines,
and the slower one's number is the one quoted. The THREADED ufunc thresholds
are the exception and say so: they come from the one box here that is
reliably idle, and `python -m pyoverdrive --calibrate` re-checks them on
yours (see "Baselining a machine").

## Why it is safe

The point of an accelerator you can leave switched on is that you never have to
wonder. So:

- **Results match stock NumPy.** Bit-identical on 43 of the 67 always-on paths;
  the other 23 run in a documented numeric mode with a measured tolerance.
- **Every fast path is conservative.** It runs only when a predicate proves the
  input is in its calibrated regime. Anything else - odd dtypes, edge shapes,
  non-finite values, subclasses - falls back to stock, automatically.
- **Verify it on your own machine**, not on our claims:
  ```bash
  python -m pyoverdrive --selfcheck   # every path vs stock NumPy, here
  python -m pyoverdrive --calibrate   # re-measure the hardware-dependent ones
  ```
  `--calibrate` re-times the threaded paths at their own thresholds on your
  CPU and switches off any that do not pay there. It never speeds a
  threshold up on a guess, and on a busy machine it keeps the shipped
  setting rather than judging from unstable numbers.
- **Kill any path, or all of them**, without touching your code:
  `pyoverdrive.disable_path("qr_small_batch")`, `PYOVERDRIVE_DISABLE=...`,
  or `pyoverdrive.disable()` to restore stock NumPy exactly.
- **Ask what it did:** `pyoverdrive.explain("numpy.linalg.qr", a)` reports the
  decision and the reason without running anything.

Over 2,000 tests - hand-written differential suites per path plus property
fuzzing against stock - run green on Windows (AMD Zen 4 and Intel hybrid) and
Linux x86-64.

PyOverdrive is an independent project. It is not affiliated with or endorsed by
NumPy or NumFOCUS.

## Under the hood

| Component | Name | Role |
|---|---|---|
| Overall project + installable runtime | **PyOverdrive** | Accelerates real `numpy.ndarray` workloads |
| Issue/PR mining system | **ProsPyctor** | Finds historical and current optimization opportunities |
| Benchmark + verification system | **Dyno** | Measures speed, memory, scaling, and regressions |
| Adaptive runtime dispatcher | **Gearbox** | Selects stock NumPy, fast paths, SIMD, or parallel execution |
| Adaptive parallel execution core | **PyRallel** | Executes appropriate operations across CPU cores |

## Ground rules

1. **Measure first.** Every optimization begins with a reproducible baseline.
2. **Correctness before speed.** An incorrect 50× result is a failure.
3. **Stock fallback.** Every accelerated path needs a reliable fallback.
4. **No unverified claims.** Issue comments, papers, and benchmarks are leads
   until independently reproduced with Dyno on fingerprinted hardware.

The full implementation brief is [docs/BUILD_SPEC.md](docs/BUILD_SPEC.md).

## Repository layout

```
lab/           ProsPyctor Performance Lab (research tooling, not shipped)
  collectors/    GitHub ingestion
  corpus/        Normalized opportunity records (+ raw API dumps, uncommitted)
  dyno/          Dyno benchmark harness
src/pyoverdrive/  The shipped runtime (Gearbox, fast paths, PyRallel, SIMD)
benchmarks/    Reproducers (historical/), workloads, and committed results
compatibility/ Differential + fuzz testing against stock NumPy
docs/          Architecture, decisions (ADRs), research reports
```

## Install, verify, roll back

```
python -m pip install git+https://github.com/LunarWerxs/PyOverdrive.git   # or a built wheel
python -m pyoverdrive --selfcheck     # every fast path vs stock NumPy on THIS machine
python -m pyoverdrive --demo          # headline ops timed live, stock vs PyOverdrive (~20s)
python -m pyoverdrive --calibrate     # re-measure the hardware-dependent paths HERE (~2 min)
```

How this compares to numexpr, Numba, Cython/Pythran, bottleneck, JAX and
friends: [docs/COMPARISON.md](docs/COMPARISON.md).

```python
import numpy as np
import pyoverdrive

pyoverdrive.enable()                      # patch every supported operation
pyoverdrive.enable(["numpy.sin"])         # or just some
pyoverdrive.report()                      # what is registered, enabled, patched
pyoverdrive.explain("numpy.sin", x)       # which path a call would take, and why
pyoverdrive.configure(threads=8, disable=["pyrallel_sin"])
pyoverdrive.disable()                     # restore stock NumPy exactly
```

Rollback, in order of bluntness: `pyoverdrive.disable()` restores every
patched name to the original object (`isinstance(np.sin, np.ufunc)` is True
again); `PYOVERDRIVE_DISABLE=name1,name2` or `disable_path(name)` kills
individual fast paths; `PYOVERDRIVE_THREADS=1` switches every parallel path
off while keeping the serial ones; `pip uninstall pyoverdrive` removes the
package, and nothing else on the system was touched. Every fast path also
falls back to stock on its own if it raises (one RuntimeWarning, then stock).

Build a wheel with `python -m pip wheel . --no-deps -w dist`; the wheel
contains only the runtime (no lab, no benchmarks).

## Development

```
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]
.venv/Scripts/python -m pytest tests compatibility
```

Run a historical reproducer (evidence goes to `benchmarks/results/`):

```
.venv/Scripts/python benchmarks/historical/opp_000001_unique.py
```

## Baselining a machine

One command runs every calibration battery, the public-API proof, the test
suite, and prints the calibration table the evidence proposes next to the
shipped one:

```
.venv/Scripts/python benchmarks/run_baseline.py                 # ~10 min
.venv/Scripts/python benchmarks/run_baseline.py --historical    # + every reproducer
.venv/Scripts/python benchmarks/run_baseline.py --require-quiet # refuse on a busy box
```

Dyno samples foreign CPU load before and after every suite and stamps it
into the result JSON. Evidence exists for two machines: fingerprint
`8f8198d9abab` (AMD Zen 4 AVX-512, 16C/32T, numpy 2.4.5, a working machine
and so rarely quiet) and `9bbe7063c555` (Intel i7-12700K hybrid 8P+4E, numpy
2.5.2, Python 3.13, kept idle: 0-1% load on every suite, and the box every
threading number comes from). The full correctness gate (differential, property fuzz, and the
all-paths self-check; 2065 tests and 67 paths at that run, 2026-08-25)
additionally runs green on a third environment - Linux x86-64 under
Docker (python:3.13-slim, numpy 2.5.2) - and on a clean-venv wheel
install, so the compatibility claims hold across two OSes, two
architectures, and both supported numpy minor lines. The single-threaded
algorithmic families win on both benchmark machines and transfer at full
strength or better (intersect1d sorted 83x, correlate int64 15.4x,
unique(axis=0) 37.9x on Intel).

The THREADED ufunc thresholds are a different story and are stated
narrowly on purpose. They were re-derived on 2026-08-24 on the Intel box -
the only reliably idle machine here, and threading numbers taken under load
are worthless - after the previous ones turned out to rest on a per-process
P-core/E-core coin flip in the single-threaded baseline
(`docs/research/hybrid-cpu-baseline-coin-flip.md`); 15 of 16 rows moved and
`np.sqrt` left the threaded family altogether.

One machine is not enough for numbers this hardware-dependent, and the
honest answer to that is not to promise a second box some day: it is to let
the threshold be checked where it matters, on yours. `python -m pyoverdrive
--calibrate` now re-measures every threaded row at its own floor and drops
the ones that do not pay on the machine it is run on. Each cell is measured
in a fresh process that re-draws until it lands on a fast core, and a row is
only removed when two independent readings agree - a busy machine produces
disagreement, and disagreement keeps the shipped row rather than guessing.
It never lowers a floor: finding where a row starts paying needs the full
sweep in `tools/calibrate_dispatch.py`, not a probe.

A run flagged CONTENDED understates every multi-thread number; do not move a
threshold on its say-so.

## Status

Phases 0-4 prototyped on the first machine (Zen 4 AVX-512, 16C/32T, numpy
2.4.5, fingerprint `8f8198d9abab`). Forty-five fast-path families are live behind
`enable()` (plus one calibration-gated), every threshold calibrated from
committed Dyno evidence. Results are bit-identical to stock on 43 of the 67 always-on paths
registered paths; the other 23 run in documented numeric mode
(`inner_tensordot`, float fftconvolve,
`nanquantile_masked`, `nanpercentile_masked`, `einsum_optimize`,
`reduce_tiny_trailing`, `eigvalsh_2x2_closed`, `eigvalsh_3x3_trig`,
`matmul_split_complex`, `inv_small_batch`, `linalg_small_batch`
det/slogdet/solve, `cholesky_small_batch`, `qr_small_batch`,
`interp_uniform_grid`), each with a tight measured tolerance:

| Fast path | Regime | Measured end-to-end (public API) |
|---|---|---:|
| `unique_sort` | `np.unique`/`unique_values`, {int32,int64,uint32,uint64} n >= 64; {int8,uint8,uint16} n >= 1000, int16 n >= 10k via radix (`kind='stable'`) | 37-55x (1M int64); small ints 1.6-27.8x; up to 101x candidate-level |
| `inner_tensordot` | `np.inner`, float32/float64, at least one operand ndim > 2, inside a MEASURED regime: >= 8 rows on the left, >= 64 on the right, >= 1024 output cells, contraction <= 128 | 1.21-6.98x inside that regime. It previously had no size gate at all and ran at 0.38x on small operands; the wins and losses interleave, so the gate admits only the corner where every measured cell won |
| `intersect_sorted` | `np.intersect1d`, same int dtype, combined size >= 400 (32/64-bit) or >= 12k (8/16-bit) | 1.5x at the floor rising to 21.6x random inputs and 90.7x already-sorted (1e6 x 1e5); small ints 1.9-10.1x; up to 433x candidate-level |
| `pyrallel_<op>` | `np.sin cos tan exp log log10 tanh`, float64/float32, C-contiguous, op/dtype-calibrated size floor (3e5-3e6 elements) | 1.34-1.63x at the floor, 1.8-2.2x at 1e7, measured end-to-end as the WORST of sorted/shuffled input and bare/consumed result. `np.sqrt` is not in this family: bandwidth bound, never reaches 1.3x |
| `relayout_blocked` | `np.ascontiguousarray` of a transposed/F-ordered 2-D float64/float32/int64 array, >= 512x512 (int64 1024x1024) | 2.8-3.3x at 2048x2048 end-to-end; up to 6.6x candidate-level (float32 8192x1024) |
| `unique_axis0_column` | `np.unique(a, axis=0)`, single int column (8- to 64-bit), >= 1000 rows | 42x at 10k rows int64; 40-298x small ints |
| `pyrallel_<op>` (binary) | `np.add maximum minimum` float64/int64, `np.subtract` int64, `np.multiply` int64; same-shape same-dtype C-contiguous; floors of 1e7-2e7 elements | 1.31-1.38x at the floor, worst of two independent sweeps (bandwidth bound; `a + b` is not reachable, only explicit `np.add`). `np.divide` and every float32 row left the family: they clear 1.3x at no measured size |
| `fftconvolve` / `fftcorrelate` | `np.convolve`/`np.correlate`, all three modes (full/same/valid), 1-D same-dtype float64/int64/int32, min length 1000, per-mode naive-work floors; floats all-finite and non-overflowing, ints under the 2^52 exactness bound | 3.6x float64 / 13.5x int64 full-mode end-to-end (10k x 1k); same-mode (the smoothing idiom) 2.9-4.9x, valid 1.6-3.3x, int modes ~13-14x; 1518x candidate-level at 20k x 20k. Ints bit-identical, floats ~1e-12 |
| `nanquantile_masked` | `np.nanquantile(a, q, axis=<int>)`, 2-D+ float64, scalar q, >= 300 elements, guarded against the few-long-slices anti-regime | 22.6-229x across the many-slice region (51.9-63.4x at the reporter's 27x100); results bit-exact vs stock in every probe |
| `einsum_optimize` | two-operand subscripts-form `np.einsum`, float64/float32, min operand >= 10k (matmul-shaped) or >= 1e6 (scalar output); label and ellipsis (`...ij,...jk`) spellings; routes through numpy's own `optimize=True` | 3.4-27.4x float64 from the floor up (45x candidate-level; ellipsis 3.2-4.3x measured); tiny contractions (the reason optimize is off by default) stay on stock |
| `searchsorted_sortqueries` | `np.searchsorted`, 1-D same-dtype float64/int64, haystack >= 1e4, queries >= 1e4 (f64) / 1e5 (i64), <= 10x haystack, and a sampled disorder gate (only random-like query orders dispatch) | 2.2-3.6x float64, up to 14.5x int64, bit-identical; sorted/nearly-sorted/descending queries measured losing and refused |
| `isclose_fused` | `np.isclose`, finite scalar pairs or small all-finite same-shape float64 (<= 1000) / float32 (<= 10k) arrays, default-style finite tolerances | 2.8x scalar pairs, 1.5-1.8x small arrays, bit-identical; NaN-bearing refusal costs 0.82x (documented guard row) |
| `isin_string_hash` | `np.isin`, 1-D default StringDType pairs, combined >= 300; pure-NUL strings refused (stock bug guarded) | 2319x at the reporter's shape, 13-299x across regimes, bit-identical |
| `dot_mixed_view` | `np.dot(real f64 2-D, complex128 1-D)`, all-finite, A >= 20k elements | 12-44x via one real GEMV; reverse direction measured no gap and stays stock |
| `quantile_dense_sort` | `np.quantile` with a q ARRAY (4-16384 quantiles), float64, 1-D or 2-D last-axis, reduced length 512-65536 | 1.8x at 4 quantiles to 918x dense; bit-identical incl. NaN slices (stock's own lerp arithmetic replicated) |
| `percentile_dense` | `np.percentile`, same regime with q in [0, 100] | 1.83x at nq=4 to 805.8x at nq=16384; bit-identical (numpy's own q/100 scaling) |
| `nanpercentile_masked` | `np.nanpercentile(a, q, axis=<int>)`, the nanquantile_masked regime with q in [0, 100] | 3.3-178x across the many-slice region (48x at 27x100); anti-regime refused (measured 0.84x there) |
| `sort_char_view` / `unique_char_view` | `np.sort` / `np.unique` (all flag combos) on 1-D single-char U1/S1, native byte order, per-route floors 300-10k | sort U1 33x at 10k (25x at 1M), S1 2.1-2.5x; unique+counts 26.6x, index/inverse up to 8.7x, plain 1.4-1.9x; bit-identical (the int view is a monotone bijection) |
| `mean_tiny_trailing` / `sum_tiny_trailing` | `np.mean` / `np.sum` over all leading axes, C-order float64/float32, trailing axis 2-5, >= 10k rows | 2.2-14.4x across the measured op/dtype/k grid; the issue's own (1000,1000,3) mean 3.9x; numeric (summation order changes; rtol 1e-9 f64) |
| `eigvalsh_2x2_closed` | `np.linalg.eigvalsh` on (..., 2, 2) float64/float32 batches >= 100, UPLO 'L', all-finite | 2.5x at batch 100 up to 31x at 10k (38.8x f32), 5.9x at 1M; numeric at LAPACK's own abs-error standard; full `eigh` stays stock (eigenvector sign has no comparison contract) |
| `eigvalsh_3x3_trig` | `np.linalg.eigvalsh` on (..., 3, 3) float64/float32 batches >= 300, UPLO 'L', all-finite | trigonometric closed form, 2.5-2.6x at batch 300 up to 4.4x at 1000, 2.9x at 100k; near-degenerate cells (a coalescing eigenvalue pair) are split out and served by stock rather than ship sqrt-eps accuracy (3.7x still at 1% degenerate; whole stack goes to stock past 25%) |
| `cholesky_small_batch` | `np.linalg.cholesky` on (..., 2, 2) / (..., 3, 3) float64 batches >= 1000, no cap; positive-definite by pivot guard fused into the factorization pass | Cholesky-Crout in cache-sized chunks, 1.9-2.3x (2x2) and 1.6-1.9x (3x3) from batch 1000 to 1M; non-PD input keeps stock's exact LinAlgError via mid-run fallback |
| `qr_small_batch` | `np.linalg.qr` on (..., 2, 2) / (..., 3, 3) float64 batches >= 300, modes reduced/complete/'r' | unrolled Householder reflectors in LAPACK's own sign convention: 4.1-11x (2x2) and 2.2-4.2x (3x3) with Q, 1.7-8.2x R-only; rank-deficient matrices are split out and served by stock (two valid factorizations can disagree there) |
| `einsum_optimize_chain` | `np.einsum` with three or more operands, clean subscripts, naive loop volume >= 65 536 (262 144 scalar output; 76 832 for ellipsis spellings, which cross later) | 1.7x at the floor rising to 150-231x at volume 3-17M end-to-end, 2161x raw at a 4-operand chain, 163x for an ellipsis chain at 67M (stock's default runs ONE fused loop over every index of the chain) |
| `matmul_split_complex` | `np.matmul(C complex 2-D, R real 2-D)`, C rows <= 256, n >= 1000, q >= 500, matched dtype pairs, all-finite | 1.55-7.5x across the measured m x n x q grid (upcast-copy-dominated shapes); square/tall C measured LOSING and stays stock |
| `roll_concat_1d` | `np.roll(a, int_shift)` on 1-D int64/float64/int32/float32/bool, size 1-10k | 5.9x at n=8, 4.7x at 1000, 2.4x at 10k (fixed ~4us Python machinery saved); shift=0 copy route up to 11.3x; dies above 10k and stays stock there |
| `pad_1d_constant` | `np.pad(a, pad_width)` in constant mode on plain 1-D numeric arrays, result length <= 16384 | 4.6x at output 14 falling to 1.5x at 16k with no constant, 2.6x to 1.6x with one. Quoted CONSUMED: the no-constant route allocates with calloc, so a bare timing reports up to 342x for a shape that is 0.86x once the array is read - which is why the cap is on the OUTPUT length |
| `argmax_blocked_transpose` (calibration-gated, OFF by default) | `np.argmax(a, axis=0)`, C-order 2-D float64/float32/int64, rows >= 3000 and size >= 9e6 | 2.2-4.05x on Intel Alder Lake; a measured 0.65-0.84x REGRESSION on AMD Zen 4, so it only turns on where `python -m pyoverdrive --calibrate` proves the win on that machine |
| `inv_small_batch` | `np.linalg.inv` on (..., 2, 2)/(..., 3, 3) float64/float32 stacks, batch floors 300-10k, det-vs-scale guard fused into the run (measured condition ceiling: passes 1e6, fails 1e8) | measured end-to-end, consumed: 2x2 4.6-11.5x, 3x3 1.3-2.0x. The guard used to run in the predicate and cost 128.5us against 25.8us for the 2x2 inverse it protected; fusing it took that cell from 5.0x to 11.5x. Numeric mode |
| `isin_object_hash` | `np.isin` on 1-D object arrays, combined >= 300; NaN-like and unhashable inputs answered via stock inside the run | 262x at 30k x 3k (guarded, end-measured), 457x raw; 2.96x at the floor; bit-identical by construction |
| `median_partition` | `np.median` on 1-D float64, 10 <= n <= 5001 | 3.4x at n=11 to 1.5x at 5000 (overhead-class); bit-identical incl. NaN check, scalar type, warnings |
| `hist2d_uniform` | `np.histogram2d`, int bins (>= 900 total) + explicit range, float64 samples, optional weights | 1.65x at the reporter's 5e6 case, 2.41x at 1000x1000 bins; bit-identical incl. values exactly on bin edges |
| `unique_rows_lexsort` | `np.unique(a, axis=0)`, int64/int32 2-D, 2-8 columns, >= 1000 rows, counts supported | 4.4-5x (k=2) to 1.84x (k=8); bit-identical incl. numeric-lexicographic row order |
| `searchsorted_extreme_key` | `np.searchsorted(int_array, python_int)` where the key is OUTSIDE the dtype's range | O(1) vs stock's per-element bigint walk (163 ms at n=1e5); provably identical answer |
| `nan{mean,sum,std,var}_scan` | `np.nanmean/nansum/nanstd/nanvar`, float64, no other kwargs, per-op floors 100-10k; one isnan probe, then the plain reduction | nanmean 2.0-12.4x, nansum 1.3-2.9x, nanstd/nanvar 2.2-2.6x; NaN-present input falls back inside the run at a measured 0.96x; bit-identical |
| `nanarg{max,min}_scan` | `np.nanargmax/nanargmin`, float64, >= 300 elements | 1.75-3.04x 1-D, 5.34x at (1000,1000) axis=1; all-NaN raises stock's ValueError with no spurious warning; bit-identical |
| `nanmedian_scan` | `np.nanmedian`, 2-D C-order float64, axis 1/-1, >= 200k elements, slices <= 2000 long | 1.42-2.59x in the many-slice region; few-long-slices anti-regime (measured 0.94x) refused; bit-identical |
| `matmul_int_blas` / `dot_int_blas` | `np.matmul`/`np.dot`, 2-D int64/int32 pairs, min dim >= 50, under the exactness bound k*max\|A\|*max\|B\| < 2^53 (2^31 for int32) | 2.6x at n=50 to 28.5x at n=800, PROVABLY bit-exact (every f64 partial sum an exact integer); over-bound inputs refused |
| `det/slogdet/solve_small_batch` | `np.linalg.det/slogdet/solve` on (..., 2, 2)/(..., 3, 3)/(..., 4, 4) float64 stacks, per-op batch WINDOWS (floors 200-1000, upper caps where the closed form stops paying), inv_small_batch's det-vs-scale conditioning guard fused into the run | measured end-to-end with the result consumed: det 1.23-4.9x, slogdet 1.19-2.8x, Cramer solve 1.53-3.1x, each at its floor and through its window. The previously advertised "up to 91.9x" was candidate-level - it excluded the guard, and the floors derived from it had the 2x2 path dispatching at 0.70x |
| `nan_to_num_where` | `np.nan_to_num(x)` default args or scalar `nan=`/`posinf=`/`neginf=` overrides, float64, >= 10k elements | 1.9-2.6x (1.1-1.4x with an inf mix); bit-identical, always a fresh copy |
| `interp_uniform_grid` | `np.interp(x, xp, fp)`, all plain 1-D float64, xp uniformly spaced (linspace/arange grids), >= 3000 queries, finite x and fp | direct index arithmetic replaces per-query bisection: 2.8-6x across the measured regimes; left=/right=/period= stay on stock |
| `take_index_assign` | `np.take(a, idx, out=...)`, 1-D float64/int64 a, intp indices >= 1000 | fancy-index gather + assign, 1.3-3.3x, bit-identical, same out object returned; out stays untouched on a bad index exactly like stock |
| `pinv_small_batch` | `np.linalg.pinv` on (..., 2, 2) / (..., 3, 3) float64 batches >= 100, no kwargs, within a measured conditioning band | adjugate inverse (for a well-conditioned square matrix the pseudo-inverse IS the inverse): 8.6-24x. Ill-conditioned or near-degenerate matrices are split out to stock and scattered back |
| `norm2_small_batch` | `np.linalg.norm(a, ord=2, axis=(-2,-1))` on the same batches | the largest singular value from the gram's closed form: 8.0-15.1x, eps-accurate at any condition number; near-degenerate matrices split out to stock |
| `svdvals_small_batch` | `np.linalg.svd(a, compute_uv=False)` on the same batches | all singular values from the gram's closed form: 7.8-12.1x, absolute error against \|\|A\|\| (the standard LAPACK itself guarantees). `compute_uv=True`, `cond` and `matrix_rank` are deliberately not served |
| `apply_along_axis_reduce` | `np.apply_along_axis(f, axis, a)` where `f` IS one of NumPy's reducers (identity-matched), plain ndarray, >= 16 slices, no zero-length dims | the `axis=` reduction instead of a Python loop over slices: 15-178x measured (mean 178x at 20k slices), bit-identical. Order-sensitive reducers (mean/sum/std/var/prod) serve the last axis only - off it, NumPy accumulates in a different order and the last ulp disagrees |
| `vectorize_ufunc_direct` | `np.vectorize(f)` where `f` is one of 34 served unary ufuncs, called on a plain float64 array with size > 0 | calls the wrapped ufunc directly instead of the object loop: 13-112x measured, bit-identical. Installs a subclass of `np.vectorize`, so `isinstance`/`type()` keep working; scalars, 0-d, empty (stock raises), float32 and otypes/excluded/signature/cache all stay on stock |

## Per-machine calibration

Some wins are architecture-dependent: the blocked-transpose argmax above
beats Alder Lake's stock argmax 2.2-4x yet loses to Zen 4's (whose
strided argmax is ~2.3x faster at equal sizes). Paths like that register
disabled and are gated by a live probe:

```
python -m pyoverdrive --calibrate
```

runs each gated path's regime-edge cells against stock on YOUR machine
(a few seconds), stores the verdict per machine fingerprint in
`~/.pyoverdrive/calibration.json`, and enables only what measured a win
there. A calibration file from different hardware or a different
numpy/Python stack is ignored, not trusted; with no file, gated paths
simply stay off. Both committed outcomes exist as evidence: the Intel
box enables the argmax path, the Zen 4 box declines it (0.65x/0.75x at
the probe cells), and both are correct.

`pyrallel_<op>` is the Phase 4 PyRallel prototype (`docs/decisions/ADR-0002`):
one persistent thread pool, a byte-keyed thread schedule, the caller's
`np.errstate` mirrored into every chunk, `PYOVERDRIVE_THREADS` as the cap and
`=1` as the whole-core kill switch. Its thresholds were re-derived from
scratch on 2026-08-24 and 15 of the 16 rows moved, because the old ones
rested on a measurement artifact rather than on the code: on a hybrid CPU a
single-threaded process is placed on a P-core or an E-core and stays there,
so the BASELINE of a threading speedup is a per-process coin flip - the same
`np.sin` float64 n=1e5 baseline measured 344 us in 15 of 25 fresh processes
and 497 us in the other 10. The threaded candidate spans cores and averages
over the split, so the flip inflates every ratio by up to 1.44x and never
deflates one. `np.sqrt` left the family altogether over it. Details and the
measurements in `docs/research/hybrid-cpu-baseline-coin-flip.md`.

Honest numbers from the same runs: float64/float32/int16/int8 unique show no
win (int8 would LOSE 7x, so the predicate excludes them); tiny patched calls
pay ~300 ns dispatch tax (a 10-element `np.add` or `np.sin` in a hot loop
drops to ~0.5x; under 1% from 1e5 elements up; documented in
`benchmarks/results/MVP-BASELINE/`); threading any of these ufuncs
below its floor LOSES, by up to 50x at 1e4, which is exactly why the
size floors exist. `np.isin` stays on stock: the searchsorted approach lost
to NumPy's table method at every size. Rejected as stale after measurement:
the classic `ufunc.at` slowness, fixed upstream in 2023 (OPP-000003).

Full evidence, including losses: `docs/research/opportunities/` and
`benchmarks/results/`. Recalibrate the threaded ufuncs on new hardware with
`tools/calibrate_dispatch.py`, which measures end-to-end through the patched
name, one cell per process, and refuses to trust a process that drew a slow
core on a hybrid CPU.

## About

PyOverdrive is a [Lunarwerx](https://github.com/LunarWerxs) project, from
the team behind [Connections.ICU](https://connections.icu). MIT licensed
(see `LICENSE`).
