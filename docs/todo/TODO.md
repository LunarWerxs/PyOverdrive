# PyOverdrive: open work

Add new work as a section here; DELETE a section when it is finished.

## 50 tests are red on main, and none of them are about the code under test

*Filed 2026-09-18 by the Odin fleet burn-down. Measured on this checkout, with every unrelated
working-tree change stashed, so these are main's own failures and not somebody's in-flight edit.*

`python -m pytest -q` on main: **50 failed, 2155 passed, 2 skipped, 1 xfailed.**

**49 of the 50 are in `compatibility/differential/`** - the suites that compare a fast path
against stock NumPy. Examples: `test_relayout_fastpath_differential.py::test_3d_input_falls_back`,
`test_searchsorted_differential.py::test_dispatch_int64_wide_range`,
`test_searchsorted_differential.py::test_dispatch_int64_duplicates_heavy_both_sides`,
`test_unique_fastpath_differential.py::test_unsupported_dtypes_fall_back`. They fail as a block,
which usually means one shared assumption moved rather than 49 separate regressions - a dispatch
predicate, a fallback rule, or the NumPy version they were written against.

**The 50th is `tests/test_packaging.py::test_version_matches_pyproject`**, and it is a one-line
fact: `pyoverdrive.__version__` is `0.0.1.dev0` while `pyproject.toml` declares `1.0.0`. The
package is published, so the version a user sees and the version the tree claims disagree.

**Why this matters more than the count suggests.** A suite that has been red for a while stops
being read: the next real regression lands in a run that already says "50 failed" and nobody
notices. That is also why this was filed rather than fixed in passing - 49 differential failures
are a real investigation into what the fast paths now do, not a tidy-up, and doing it badly would
be worse than leaving it visible.

**Start here:** run ONE of the differential failures and read what it actually asserts -
`python -m pytest -q compatibility/differential/test_unique_fastpath_differential.py::test_unsupported_dtypes_fall_back -x -vv`.
If the shared assumption theory holds, fixing it closes most of the 49 at once. The packaging
one is independent and can be closed on its own by deciding which version is the true one.

**Done when** `python -m pytest -q` is green, or each remaining failure is named here with the
reason it is expected.

