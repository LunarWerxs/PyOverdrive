# Batch-8 leads: sweep, triage, probe verdicts (2026-08-24)

Sweep: the banked collector (lab/collectors/github_sweep.py) re-run over
its 16 angles - 576 fresh candidates after dedup. Most were batch-6
triage rejects resurfacing, which exposed two instrument gaps, both
fixed and committed the same day: triage verdicts now BANK into
lab/corpus/triage-rejects.jsonl (read by the collector's dedup, so a
rejected title stays rejected), and the shortlist dedup now matches the
"numpy#123" shorthand these documents actually use, not just full URLs.
Dedup baseline grew 110 -> 556.

Triage: 8 parallel scorer agents over title+metadata, 137 keeps / 439
banked rejects (one dropped candidate hand-triaged). Already-declined
leads that predate the shorthand fix (complex abs, setdiff1d, any/all,
stacked matmul, log-int, StringDType unique) were skipped by hand this
once; the fixed dedup drops them from future sweeps.

Probes: one deterministic dev-box script over the fresh ground, then a
guard-inclusive idle-box pass (fp 9bbe7063c555) for the survivors.

Ingested (records OPP-000050..051), idle-box BATCH8-CAL numbers
(guard-inclusive candidates):

| lead | OPP | idle-box evidence |
|---|---|---|
| np.interp on uniform grids: per-query bisection vs direct index arithmetic (numpy#2513 class) | OPP-000050 | 3.37x at nq=3k/grid 1k, 5.95x at 10k, 2.81-3.32x at 30k-100k, 3.19x at 1M/10k, 1.46x at 1M/100 arange; 1.06x at nq=1k (floor evidence); rel err 1e-14..1e-12 |
| np.take with out= slower than fancy-index gather + assign (numpy#28636; seberg's diagnosis: stock buffers so out survives a bad index - the gather route keeps that guarantee for free) | OPP-000051 | 3.34x at 1k gathered, 2.15x at 10k, 1.82x at 100k (1.80x int64), 1.58x at 1M, 1.30x at 10M, bit-exact |

Probed and DECLINED with measured proof (dev box unless noted):

- np.add.at via bincount (numpy#23176): 1.08-1.26x - numpy 2.x already
  closed the famous 10x ufunc.at gap. Stale.
- np.clip scalar bounds via minimum(maximum) (numpy#14281/#1023):
  0.47-1.00x - the 1.17-era clip regression is long fixed. Stale.
- np.full via empty+fill (numpy#13001/#16180/#17206): 2.67x at n=100
  but the margin is ~0.5us absolute; the ~300 ns dispatch tax
  (ADR-0003) eats it. Declined by tax, not by direction.
- np.min tiny arrays (numpy#12350): tolist+builtin wins 15x raw at n=5,
  but NaN-correctness needs a scan, the result must be rewrapped
  np.float64, and the tax lands on every tiny call - the honest net is
  ~1.5x on a ~1.5us call, below the bar. Declined by tax + guards.
- np.mod scalar divisor recomposition: 0.49x. Stock is fine.
- np.allclose as isclose().all() (numpy#23483): 0.98-1.13x, wrapper
  overhead negligible. Stale.
- np.array_equal via tobytes memcmp (numpy#2926): 2.37x at n=1k but
  0.06x at 1M (tobytes copies); no copy-free comparison surface exists
  at Python level, and float bitwise-equality is semantically wrong
  (NaN, signed zero). Out of reach.
- np.histogram uniform bins, the 1-D sibling of shipped hist2d_uniform
  (numpy#6099/#8699): WITHDRAWN AFTER FULL IMPLEMENTATION - the naive
  bincount route probes 2.3-4.4x in a mid-size window, but it is only
  bit-identical by luck on random data; the edge-correction passes the
  contract requires (the same machinery hist2d_uniform ships) cost the
  entire margin. Idle-box battery with the honest candidate: 0.72-1.27x
  across n=300..300k, 0.74x edge-salted, 0.85x weighted - no winning
  cell at the 1.3x bar. The 2-D path wins because histogramdd has no C
  shortcut to compete with; the 1-D op does (numpy#6100 landed
  astrofrog's ~10x fix in 2015). Battery cells retained in BATCH8-CAL
  as the decline evidence; no OPP record.

Two more bench items probed and DECLINED (dev box, post-ship):

- np.searchsorted mixed-int-dtype promotion (numpy#13579): casting
  queries to the array dtype measures 1.29-1.33x for int64/int32
  mixes BEFORE paying the value-preservation scan a real path would
  need, and the uint64-vs-int64 mix (the upstream promotion trap this
  project already documented) measures 0.23x - the cast route is the
  slow direction there. Below the bar with guards priced.
- structured-array field-access overhead (numpy#9934/#6467 class):
  field extraction itself is a 0.1us strided-view creation, the
  per-call overhead on real math is 1.21x, and a[field] is
  __getitem__ - no numpy function call to intercept anyway.

Still on the bench, genuinely open but each needing an infrastructure
decision before probing: np.fromfile/np.save/np.load I/O family (a
disk-bound harness - dyno measures CPU time, so the evidence standard
itself needs an owner call), np.linalg.qr gufunc demand (numpy#7179 -
C-level feasibility research, values-only qr has no obvious closed
form), and the remaining einsum planner shapes.
