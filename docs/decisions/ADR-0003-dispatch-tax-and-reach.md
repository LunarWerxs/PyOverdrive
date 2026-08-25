# ADR-0003: Dispatch tax policy, reach of module patching, and the Phase 5 surface

Date: 2026-08-23 · Status: accepted

Context: shipping the binary PyRallel family exposed the cost of the
experimental patching route on tiny calls (a 10-element `np.add` went from
460 ns to 1.4 us), and the relayout path exposed a second shape of it (stock
`np.ascontiguousarray` on an already-C array is a 50-70 ns no-op). Both had
to be decided, not papered over.

## Decisions

1. **The tax is paid in the wrapper, so the wrapper is the hot path.**
   Gearbox keeps one precompiled list of ENABLED paths per operation,
   shared by reference with that operation's wrapper closure and rebuilt in
   place on `register` / `set_path_enabled`, and the dispatch loop runs
   inside the closure itself (no `dispatch()` call level, no dict lookup,
   no `enabled` check per call). Every PyRallel-family predicate checks the
   cheapest, most selective fact first (array size against the smallest
   threshold in its table) so a small array leaves in one comparison.
   Measured: tiny-call tax ~950 ns -> ~300 ns; under 1% of the call from
   1e5 elements up. `tests/test_gearbox.py` pins that a kill switch is
   honored by the live wrapper without re-patching.

2. **~300 ns on a tiny call is accepted and documented, not hidden.** It is
   the floor of a Python-level wrapper around a sub-microsecond C call and
   it violates the letter of the "no more than 2% regression on unaffected
   cases" gate for explicit tiny-array hot loops (`np.add(a, b)` on 10
   elements runs at ~0.55x). The MVP-BASELINE suite keeps such cases as
   protected-baseline rows so the number is always on the record, the
   README states it, and every family has a kill switch. Removing the tax
   altogether needs the extension-hook route (spec 10.5), not more Python.

3. **Reach is the patched NAME, nothing else, and each family says so.**
   `a + b` resolves to `ndarray.__add__` in C and never touches
   `numpy.add`; `x.copy(order="C")`, `np.array(x)`, `np.asarray(x,
   order="C")` never touch `numpy.ascontiguousarray`; NumPy's own internal
   calls (e.g. `intersect1d` calling `unique` by its module-internal name)
   bypass the patched public names. Each fast-path module states its reach
   in its docstring and the proof suite carries an "operator form not
   patched" row so the limit is measured, not assumed.

4. **Phase 5 surface ships inside the wheel, with no lab dependency.**
   `status()` / `report()` / `configure(threads=, disable=, enable=,
   debug=)` / `selfcheck()` and `python -m pyoverdrive --selfcheck`. The
   self-check runs every registered path on a dispatching input against
   stock under the path's declared comparison mode (bit-identical, numeric,
   or set-equal); it is the post-install proof and the first thing to run
   on new hardware. It caught a wrong declaration on day one
   (`unique_values_sort` claimed bit-identity against an unordered stock
   result; the contract is set-equality). Proven from a fresh-venv install
   of the built wheel: install, self-check 20/20, `disable()` restores real
   ufuncs, uninstall leaves nothing.

5. **Bandwidth-bound wins get their own table and their own caveat.** The
   binary family's win is a property of how many cores it takes to saturate
   this machine's memory channels, so it lives in `parallel_binary.py` with
   its own battery and a "recalibrate on every new box" note, separate from
   the compute-bound transcendental table.

   SUPERSEDED NUMBERS, 2026-08-24: this ADR recorded that family at
   "1.3-2.2x at 1e6-1e7 elements". Measured end to end it was 1.04-1.20x at
   those floors, because the battery behind it compares a threaded candidate
   against a single-threaded baseline that lands on a P-core or an E-core per
   process and stays there. Re-derived from two independent sweeps its floors
   are 1e7-2e7, every float32 row and all of `np.divide` failed to clear
   1.3x at any size, and the survivors clear it by 0.01-0.08x. The caveat
   this point makes was right; the numbers under it were not. See
   `docs/research/hybrid-cpu-baseline-coin-flip.md`.

## Consequences

- A user with an explicit tiny-array `np.<op>` hot loop should disable that
  family (`PYOVERDRIVE_DISABLE=pyrallel_add`) or not enable that operation;
  `enable([...])` takes an explicit list for exactly this reason.
- `benchmarks/run_baseline.py` is the one command for a new machine; its
  calibration-diff output is a proposal to paste by hand, never applied
  automatically, because every table row is a measured number with a
  fingerprint behind it.
