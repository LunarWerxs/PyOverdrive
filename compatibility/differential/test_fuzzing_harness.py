"""Tests for the fuzzing harness itself (fuzzing.py), not for a fast path.

Contract: a corpus case replays the exact input that failed - same values
AND the same memory layout - and the same seed always yields the same
samples; the shrinker keeps a failure failing while making it small.
Regressions caught: saving the view instead of values + layout (np.savez
silently makes it contiguous, so a strided-only bug would replay green),
sampling from global random state (a reported seed would not reproduce),
and a shrinker that drops the property that made the case fail.
"""

from __future__ import annotations

import numpy as np

from fuzzing import (
    Case, FuzzedArray, FuzzedParameter, FuzzSpec, Fuzzer, Layout, default_values, load_case,
    materialize, save_case, shrink,
)


def _strided_case() -> Case:
    values = np.arange(30, dtype=np.float64).reshape(6, 5) - 7.25
    return Case({"x": (values, Layout(perm=(1, 0), step=(2, -3)))}, {"axis": None})


def test_corpus_round_trip_keeps_values_and_layout(tmp_path):
    case = _strided_case()
    spec = FuzzSpec("some_path", "numpy.positive", Fuzzer((), ()), lambda a, p: ((a["x"],), {}))
    file = save_case(spec, case, "values differ", corpus=tmp_path)
    assert file.parent.name == "some_path"
    assert save_case(spec, case, "values differ", corpus=tmp_path) == file  # content-named

    meta, loaded = load_case(file)
    assert meta["path"] == "some_path" and meta["params"] == {"axis": None}
    original = case.materialize()["x"]
    replayed = loaded.materialize()["x"]
    np.testing.assert_array_equal(replayed, original)
    assert replayed.strides == original.strides
    assert not replayed.flags.c_contiguous and not replayed.flags.f_contiguous


def test_materialize_realizes_the_layout():
    values = np.arange(12, dtype=np.int32).reshape(3, 4)
    view = materialize(values, Layout(perm=(0, 1), step=(1, -1)))
    np.testing.assert_array_equal(view, values)
    assert view.strides[1] < 0


def test_same_seed_same_samples_and_constraints_hold():
    fuzzer = Fuzzer(
        (FuzzedParameter("n", 0, 500, "loguniform"), FuzzedParameter("m", 1, 9)),
        (FuzzedArray("x", ("n", "m"), dtypes=(np.float64, np.int16),
                     min_elements=4, max_elements=900, max_allocation_bytes=40_000),),
        constraints=(lambda p: p["m"] != 5,),
    )
    first = list(fuzzer.take(30, seed=11))
    again = list(fuzzer.take(30, seed=11))
    for a, b in zip(first, again):
        (va, la), (vb, lb) = a.arrays["x"], b.arrays["x"]
        assert la == lb and va.dtype == vb.dtype
        np.testing.assert_array_equal(va, vb)
    for case in first:
        values, layout = case.arrays["x"]
        assert case.params["m"] != 5
        assert 4 <= values.size <= 900
        view = case.materialize()["x"]
        assert view.base is None or view.base.nbytes <= 40_000
    assert any(not c.arrays["x"][1].is_contiguous for c in first)


def test_rank_zero_float_draws_take_special_values():
    # Regression: a rank-0 float draw used to be a numpy scalar, so the
    # special-value assignment raised TypeError and `python fuzz.py` crashed
    # on the first rank-0 sample of noop_positive.
    rng = np.random.default_rng(0)
    for dtype in (np.float32, np.float64, np.complex128):
        for _ in range(20):
            vals = default_values(rng, np.dtype(dtype), (), special_probability=1.0)
            assert isinstance(vals, np.ndarray) and vals.shape == () and vals.dtype == dtype


def test_shrink_keeps_the_failure_and_minimizes():
    rng = np.random.default_rng(3)
    values = rng.uniform(-50, 500, size=(8, 6))

    def fails(case: Case) -> bool:
        x = case.materialize()["x"]
        return x.size > 0 and not x.flags.c_contiguous and bool((x > 100).any())

    case = Case({"x": (values, Layout(perm=(1, 0), step=(3, 2)))})
    assert fails(case)
    small = shrink(case, fails)
    assert fails(small)
    x = small.materialize()["x"]
    assert x.size == 2
    assert np.count_nonzero(x) == 1
