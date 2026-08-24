# Batch-6 lead shortlist (mined 2026-08-24)

Sweep: 16 angles via the now-banked collector
(`lab/collectors/github_sweep.py`), deliberately opening ground earlier
batches never touched: CLOSED numpy issues (fixed-partially / wontfix /
by-design gold), re-sorted open queries (comments-desc, reactions-desc),
and four adjacent trackers (scipy, pandas, bottleneck, numexpr). 583
fresh candidates after dedup against the 101 previously-seen issues.
Title triage by 8 parallel scorer agents kept ~110; the flagged topics
plus my own classics list (Bottleneck's published benchmark table read
as a menu of numpy headroom) were probed directly on the dev box before
any ingestion decision.

Ingested (OPP-000041..046), every one probe-verified BEFORE recording:

| lead | OPP | evidence |
|---|---|---|
| nanmean/nansum/nanstd/nanvar wrapper overhead | OPP-000041 | numpy#5691 (class) + bottleneck table; idle-box 1.31-12.4x |
| nanargmax/nanargmin masked-copy overhead | OPP-000042 | bottleneck table; idle-box 1.75-5.34x |
| nanmedian 2-D per-slice Python loop | OPP-000043 | numpy#4683 (mechanism class); idle-box 1.42-2.59x |
| integer matmul has no BLAS path | OPP-000044 | numpy#14556 + #16158; idle-box 2.62-28.5x BIT-EXACT under the 2^53 bound |
| batched 2x2/3x3 det/slogdet/solve LAPACK overhead | OPP-000045 | numpy#20052 + OPP-000035 sibling; idle-box up to 91.9x (det 2x2) |
| nan_to_num where-chain allocations | OPP-000046 | numpy#23140; idle-box 1.20-2.21x |

Also shipped from this campaign without a new record:
`nanpercentile_masked` (OPP-000013 sibling surface, the percentile_dense
recipe applied to nanquantile_masked; idle-box 3.26-178x, anti-regime
correctly refused at 0.84x).

Probed and DECLINED with measured proof (dev box, order-of-magnitude):

- numpy#26510 StringDType unique: via-U rebuild AND python-set-sort both
  0.57x - numpy 2.4 already handles it. Stale.
- numpy#21804 setdiff1d: sorted-searchsorted route 0.37x. Stock fine.
- numpy#13284 log/sqrt on int64: astype-f64 route 0.67-0.79x. Stale.
- numpy#619 / #7569 / #8957 stacked matmul not BLAS: stock beats both a
  Python loop (0.78x) and einsum (0.12x) - numpy 2.x batches into BLAS.
  Fixed upstream.
- numpy#5507 np.average(weights=None): 0.90-1.03x vs mean. No gap left.
- nanmax/nanmin (bottleneck table rows): 0.61-1.08x - numpy fixed the
  2015-era #5691 gap for these two ops; only the four aggregations and
  the arg-reductions retain it.
- solve via inv-route: 0.57x (the LAPACK-inv path is not the win;
  Cramer closed form IS, 6.25x, which is what shipped).
- np.random.choice family (#2764, #7543, #25371, #4188): any faster
  algorithm draws a DIFFERENT random stream; bit-identity is impossible
  by construction. Permanently out of scope.
- numpy#2269 first-nonzero, #3446 any/all short-circuit: no numpy
  function call to intercept (idiom-level, not op-level).
- bottleneck move_* rolling family: no numpy equivalent op to patch.

Strong bench for batch 7 (triage-kept, unprobed): numpy#23140-adjacent
nan_to_num kwargs regimes; numpy#16573 median averaging overhead;
numpy#11714 einsum 1.15 regression (check against shipped
einsum_optimize regimes); numpy#20790 __array_function__ overhead
(likely core-C, unpatchable); scipy#24474 batched-linalg demand signal
(supports extending OPP-000045 to more ops: eig, cholesky 2x2/3x3
closed forms); numpy#3994 complex abs via sqrt(re^2+im^2) (overflow
semantics vs hypot need care); numpy#18153 tril/triu_indices;
numpy#28921 ndindex.

Sweep artifacts: scratchpad sweep6/ (candidates.jsonl + 8 chunk files),
not committed; the collector itself is.
