# ADR-0005: Predicate-hoisted guards, sampled gates, and the property net

Date: 2026-08-23 · Status: accepted

Context: five families shipped in one day (nanquantile_masked,
einsum_optimize, searchsorted_sortqueries, the small-int radix extension,
isclose_fused) on top of the morning's fftconvolve. Three patterns
recurred and one piece of standing infrastructure was added; recording
them here so the next family starts from them instead of rediscovering.

## Decisions

1. **Hoist every semantic guard into the predicate so the run is exact by
   construction.** isclose_fused is the cleanest case: stock's cost IS
   its guards (errstate, isfinite reductions, conversions), so the
   predicate performs the guards once (finite tolerances, finite values,
   matched shapes) and refuses anything else - the dispatched computation
   is then the bare arithmetic and bit-identical. fftconvolve's abs-max
   scan (NaN + inf + overflow in one pass) and nanquantile's argument
   normalization follow the same shape. The refusal path pays the guard
   twice (ours, then stock's own); measure that tax and pin it as an
   MVP-BASELINE guard row (isclose NaN refusal: 0.82x, accepted).

2. **When the win depends on a property of the DATA, gate on a sampled
   estimate, never a full scan and never a guess.** searchsorted's win
   needs genuinely disordered queries; sortedness cannot be checked
   exactly for less than the win costs, and "large therefore disordered"
   is false (nearly-sorted lost 0.48x). A 4096-pair strided sample
   estimates the descent fraction within ~0.016 for microseconds. The
   gate metric must be measured against the battery's own arrays
   (descent fractions 0.0/0.01/0.09/0.165/0.316/0.5/1.0 mapped to
   0.48x-3.28x), and min(descents, ascents) rather than descents alone,
   because DESCENDING order has maximal descents and still loses.

3. **Routing to numpy's own alternative machinery beats reimplementing
   it.** einsum_optimize ships no contraction code: it is a size gate in
   front of stock optimize=True, which upstream would not default for
   tiny-call reasons that a dispatch layer can respect per call. Where
   numpy has a second, better route behind a keyword or a kind=
   (small-int radix via kind='stable'), the fast path is the gate, not
   the algorithm. This also shrinks the correctness surface to return-
   type fidelity (einsum's 0-d-vs-scalar normalization was the one real
   difference).

4. **The hypothesis property net is standing infrastructure, not a
   one-off.** compatibility/property/ fuzzes every risky family's full
   argument space against stock - values, shapes straddling every floor,
   kwargs, exception parity - and found two real holes on day one: the
   fftconvolve overflow smear (finite ~8e213 inputs, patched with the
   abs-max bound) and an unsound tolerance model in its own convolve
   check. Every new family MUST add a property test alongside its
   differential file. Related, found by the searchsorted differential
   battery and kept there as a strict xfail: numpy's batched searchsorted
   chains a locality hint across consecutive queries, so an unsorted
   haystack returns query-ORDER-dependent garbage on stock itself - a
   documented-precondition violation, but one the docstring must not
   claim to reproduce byte-for-byte.

5. **A reproduced lead may be parked, with the reasons in the corpus.**
   OPP-000021 (cov post-hoc) stays reproduced-but-unshipped: its float32
   headline is a precision downgrade a transparent drop-in cannot claim
   (stock upcasts internally), and the float64-matched regime is under
   min-win in the claim's own shape. Parking notes live in the corpus
   record so the decision is auditable and reversible, not forgotten.

## Consequences

- The dispatch tax now has three measured shapes: the ~300-390 ns
  wrapper floor (ADR-0003), guard-scan refusal taxes (isclose 0.82x on
  NaN input), and sampled-gate costs (searchsorted ~30 us, invisible at
  1.00x on its guard row). All pinned in MVP-BASELINE.
- Twelve families, 27 registered paths, selfcheck 27/27; every
  batch-2 reproduced lead is shipped, refuted, or parked with reasons.
