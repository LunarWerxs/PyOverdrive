"""Per-fast-path fuzz specs: the seeded input space each path is checked on.

WHY: one declarative spec per path replaces "the shapes its author thought
of" with a space that straddles the path's size floors and caps and mixes
in the layouts, ranks, dtypes and special values its predicate must either
handle bit-for-bit or refuse. Both test_fuzzed_differential.py (a small
seeded sample on every plain run) and fuzz.py (long runs that write
corpus cases) draw from SPECS.

Adding a path: build its Fuzzer from the size/dtype/axis space around its
predicate, name the comparison discipline its module docstring declares,
and register it in SPECS under the fast path's name.
"""

from __future__ import annotations

import numpy as np

from fuzzing import FuzzedArray, FuzzedParameter, Fuzzer, FuzzSpec, byte_exact, exact

RANK = FuzzedParameter("rank", 1, 3)
AXIS = FuzzedParameter("axis", distribution={None: 0.4, 0: 0.25, -1: 0.25, 2: 0.1})


def _dims(first: str, first_max: int) -> tuple[FuzzedParameter, ...]:
    """Leading axis drawn ``first`` ("uniform" to straddle a high floor,
    "loguniform" for a low one), trailing axes small, so every rank fits
    the element budget without starving the resampler."""
    return (
        FuzzedParameter("n0", 0, first_max, first),
        FuzzedParameter("n1", 0, 64, "loguniform"),
        FuzzedParameter("n2", 0, 16, "loguniform"),
    )


def _x_axis(arrays: dict, params: dict) -> tuple[tuple, dict]:
    if params["axis"] is None:
        return (arrays["x"],), {}
    return (arrays["x"],), {"axis": params["axis"]}


def _float_reduction(path: str, op: str, dist: str, n_max: int = 40_000) -> FuzzSpec:
    # float32 draws are there to prove the refusal side stays stock
    x = FuzzedArray(
        "x", ("n0", "n1", "n2"), dtypes=(np.float64, np.float64, np.float64, np.float32),
        rank="rank", max_elements=60_000,
    )
    return FuzzSpec(path, op, Fuzzer((RANK, *_dims(dist, n_max), AXIS), (x,)), _x_axis, exact)


def _positive() -> FuzzSpec:
    x = FuzzedArray(
        "x", ("n0", "n1", "n2"),
        dtypes=(np.float64, np.float32, np.int64, np.int32, np.uint8, np.complex128, np.bool_),
        rank="rank",
    )
    rank = FuzzedParameter("rank", 0, 3)
    return FuzzSpec(
        "noop_positive", "numpy.positive",
        Fuzzer((rank, *_dims("loguniform", 64)), (x,)),
        lambda arrays, params: ((arrays["x"],), {}),
        exact,
        enable_paths=("noop_positive",),
    )


def _nan_to_num() -> FuzzSpec:
    x = FuzzedArray(
        "x", ("n0", "n1", "n2"), dtypes=(np.float64, np.float64, np.float64, np.float32),
        rank="rank", max_elements=60_000, special_probability=0.7,
    )
    fills = (
        FuzzedParameter("nan", distribution={None: 0.5, -0.0: 0.2, -1.5: 0.2, 1e300: 0.1}),
        FuzzedParameter("posinf", distribution={None: 0.6, 9.0: 0.2, -0.0: 0.2}),
        FuzzedParameter("neginf", distribution={None: 0.6, -9.0: 0.2, 0.0: 0.2}),
    )

    def call(arrays, params):
        kwargs = {k: params[k] for k in ("nan", "posinf", "neginf") if params[k] is not None}
        return (arrays["x"],), kwargs

    # byte_exact: a caller-supplied -0.0 fill must keep its sign bit
    return FuzzSpec(
        "nan_to_num_where", "numpy.nan_to_num",
        Fuzzer((RANK, *_dims("uniform", 40_000), *fills), (x,)), call, byte_exact,
    )


def _median() -> FuzzSpec:
    x = FuzzedArray("x", ("n0",), dtypes=(np.float64, np.float64, np.float64, np.float32))
    return FuzzSpec(
        "median_partition", "numpy.median",
        Fuzzer((FuzzedParameter("n0", 0, 6_000, "loguniform"),), (x,)),
        lambda arrays, params: ((arrays["x"],), {}),
        exact,
    )


def _roll() -> FuzzSpec:
    x = FuzzedArray(
        "x", ("n0",),
        dtypes=(np.int64, np.float64, np.int32, np.float32, np.bool_, np.uint8),
    )
    params = (
        FuzzedParameter("n0", 0, 12_000, "loguniform"),
        FuzzedParameter("shift", -25_000, 25_000, "uniform"),
    )
    return FuzzSpec(
        "roll_concat_1d", "numpy.roll", Fuzzer(params, (x,)),
        lambda arrays, params: ((arrays["x"], params["shift"]), {}),
        exact,
    )


SPECS: dict[str, FuzzSpec] = {
    spec.path: spec
    for spec in (
        _positive(),
        _float_reduction("nanmean_scan", "numpy.nanmean", "loguniform"),
        _float_reduction("nansum_scan", "numpy.nansum", "uniform"),
        _float_reduction("nanstd_scan", "numpy.nanstd", "uniform"),
        _float_reduction("nanvar_scan", "numpy.nanvar", "uniform"),
        _float_reduction("nanargmax_scan", "numpy.nanargmax", "loguniform"),
        _float_reduction("nanargmin_scan", "numpy.nanargmin", "loguniform"),
        _nan_to_num(),
        _median(),
        _roll(),
    )
}
