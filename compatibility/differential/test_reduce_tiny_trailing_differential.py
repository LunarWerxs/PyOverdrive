"""Differential tests: mean_tiny_trailing / sum_tiny_trailing fast paths vs
stock numpy.mean / numpy.sum.

Contract (src/pyoverdrive/fastpaths/reduce_tiny_trailing.py): applies to
mean(a, axis)/sum(a, axis) where a is a plain C-contiguous float64/float32
ndarray with ndim >= 2, the axis argument reduces exactly all leading axes
(every axis except the last), a.shape[-1] is in [K_MIN, K_MAX], rows
(a.size // a.shape[-1]) is >= ROWS_MIN, and no other arguments are given.
dtype/out/keepdims/where/initial kwargs, other dtypes, other axes, integer
arrays, non-contiguous/F-order arrays, and non-ndarray inputs all stay on
stock. The reroute changes summation order (one stock reduction per kept
column instead of one reduction over the whole array), so results are
numerically equal, not bit-identical: comparison mode is numeric.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.reduce_tiny_trailing import K_MAX, K_MIN, ROWS_MIN

OPS = [("numpy.mean", "mean_tiny_trailing"), ("numpy.sum", "sum_tiny_trailing")]
OP_IDS = ["mean", "sum"]

ROWS = ROWS_MIN * 2
K = 3


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable(["numpy.mean", "numpy.sum"])
    yield
    pyoverdrive.disable()


def _np_fn(op):
    return getattr(np, op.split(".")[-1])


def _stock(op, *args, **kwargs):
    return GEARBOX.stock_fn(op)(*args, **kwargs)


def _arr(shape, dtype=np.float64, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, size=shape).astype(dtype)


def _tol(dtype):
    if dtype == np.float64:
        return 1e-9, 0.0
    if dtype == np.float32:
        return 1e-3, 0.0
    raise ValueError(f"unexpected dtype: {dtype}")


def _assert_dispatched(op, path, args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == path, (op, args, kwargs, decision, reason)
    got = _np_fn(op)(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    return got, stock


def _assert_dispatched_close(op, path, args, kwargs=None):
    got, stock = _assert_dispatched(op, path, args, kwargs)
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    rtol, atol = _tol(stock.dtype)
    assert np.allclose(got, stock, rtol=rtol, atol=atol)
    return got, stock


def _assert_refused_equal(op, args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, args, kwargs, decision, reason)
    got = _np_fn(op)(*args, **kwargs)
    stock = _stock(op, *args, **kwargs)
    assert type(got) is type(stock)
    assert np.array_equal(got, stock, equal_nan=True)
    return got, stock


def _assert_refused_raises(op, args, kwargs=None):
    kwargs = kwargs or {}
    decision, reason = GEARBOX.decide(op, args, kwargs)
    assert decision == "stock", (op, args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        _np_fn(op)(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(op, *args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + numeric equality: shapes and axis spellings
# ---------------------------------------------------------------------------

_SHAPE_AXIS_CASES = [
    ((ROWS, K), 0, 1),
    ((ROWS, K), -2, 2),
    ((ROWS, K), (0,), 3),
    ((ROWS, K), (-2,), 4),
    ((200, 100, 3), (0, 1), 5),
    ((200, 100, 3), (1, 0), 6),
    ((200, 100, 3), (-3, -2), 7),
    ((30, 30, 20, 2), (0, 1, 2), 8),
]
_SHAPE_AXIS_IDS = [
    "2d-axis0", "2d-axis-neg2", "2d-axis-tuple0", "2d-axis-tuple-neg2",
    "3d-axis01", "3d-axis10", "3d-axis-neg3neg2", "4d-axis012",
]


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
@pytest.mark.parametrize("shape, axis, seed", _SHAPE_AXIS_CASES, ids=_SHAPE_AXIS_IDS)
def test_dispatch_shapes_and_axes(op, path, shape, axis, seed):
    a = _arr(shape, np.float64, seed)
    _assert_dispatched_close(op, path, (a, axis), {})


def test_dispatch_positional_axis_mean():
    # np.mean(a, 0): axis passed positionally, not as a kwarg.
    a = _arr((ROWS, K), np.float64, seed=50)
    _assert_dispatched_close("numpy.mean", "mean_tiny_trailing", (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_dispatch_float32(op, path):
    a = _arr((ROWS, K), np.float32, seed=51)
    _assert_dispatched_close(op, path, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
@pytest.mark.parametrize("k", [K_MIN, K_MAX], ids=["k-min", "k-max"])
def test_dispatch_k_boundaries(op, path, k):
    a = _arr((ROWS, k), np.float64, seed=52 + k)
    _assert_dispatched_close(op, path, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_dispatch_rows_exactly_at_floor(op, path):
    a = _arr((ROWS_MIN, K), np.float64, seed=60)
    assert a.size // K == ROWS_MIN
    _assert_dispatched_close(op, path, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_rows_just_below_floor(op, path):
    a = _arr((ROWS_MIN - 1, K), np.float64, seed=61)
    assert a.size // K == ROWS_MIN - 1
    _assert_refused_equal(op, (a, 0), {})


# ---------------------------------------------------------------------------
# 2. special values: these still DISPATCH (the predicate does not scan
#    values), so parity has to hold through NaN / inf as well.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_special_values_nan_salted_column(op, path):
    a = _arr((ROWS, K), np.float64, seed=300)
    a[::997, 1] = np.nan
    got, stock = _assert_dispatched(op, path, (a, 0), {})
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(np.isnan(got), np.isnan(stock))
    mask = ~np.isnan(stock)
    rtol, atol = _tol(stock.dtype)
    assert np.allclose(got[mask], stock[mask], rtol=rtol, atol=atol)


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_special_values_posinf_column(op, path):
    a = _arr((ROWS, K), np.float64, seed=301)
    a[::997, 1] = np.inf
    got, stock = _assert_dispatched(op, path, (a, 0), {})
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.array_equal(np.isinf(got), np.isinf(stock))
    mask = np.isfinite(stock)
    rtol, atol = _tol(stock.dtype)
    assert np.allclose(got[mask], stock[mask], rtol=rtol, atol=atol)


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_special_values_mixed_inf_column(op, path):
    a = _arr((ROWS, K), np.float64, seed=302)
    a[997, 1] = np.inf
    a[1997, 1] = -np.inf
    got, stock = _assert_dispatched(op, path, (a, 0), {})
    assert got.dtype == stock.dtype
    assert got.shape == stock.shape
    assert np.isnan(got[1])
    assert np.isnan(stock[1])
    mask = np.array([True, False, True])
    rtol, atol = _tol(stock.dtype)
    assert np.allclose(got[mask], stock[mask], rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# 3. refusal routes: stock-vs-stock parity
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
@pytest.mark.parametrize("k", [1, K_MAX + 1], ids=["k-below-min", "k-above-max"])
def test_refusal_k_out_of_range(op, path, k):
    a = _arr((ROWS, k), np.float64, seed=105 + k)
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_axis_none(op, path):
    a = _arr((ROWS, K), np.float64, seed=110)
    _assert_refused_equal(op, (a, None), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_axis1_on_2d(op, path):
    a = _arr((ROWS, K), np.float64, seed=111)
    _assert_refused_equal(op, (a, 1), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_axis_0_2_on_3d(op, path):
    a = _arr((200, 100, 3), np.float64, seed=112)
    _assert_refused_equal(op, (a, (0, 2)), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_axis_all_axes_2d(op, path):
    a = _arr((ROWS, K), np.float64, seed=113)
    _assert_refused_equal(op, (a, (0, 1)), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_int64_dtype(op, path):
    a = (_arr((ROWS, K), np.float64, seed=114) * 100).astype(np.int64)
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_float16_dtype(op, path):
    a = _arr((ROWS, K), np.float16, seed=115)
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_f_order(op, path):
    a = np.asfortranarray(_arr((ROWS, K), np.float64, seed=116))
    assert a.flags.f_contiguous and not a.flags.c_contiguous
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_noncontiguous_view(op, path):
    full = _arr((ROWS * 2, K), np.float64, seed=117)
    a = full[::2]
    assert not a.flags.c_contiguous
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_keepdims(op, path):
    a = _arr((ROWS, K), np.float64, seed=118)
    _assert_refused_equal(op, (a,), {"axis": 0, "keepdims": True})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_dtype_kwarg(op, path):
    a = _arr((ROWS, K), np.float32, seed=119)
    _assert_refused_equal(op, (a,), {"axis": 0, "dtype": np.float64})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_out_kwarg(op, path):
    a = _arr((ROWS, K), np.float64, seed=120)
    out_got = np.empty(K, dtype=np.float64)
    out_stock = np.empty(K, dtype=np.float64)
    decision, reason = GEARBOX.decide(op, (a, 0), {"out": out_got})
    assert decision == "stock", (op, decision, reason)
    got = _np_fn(op)(a, 0, out=out_got)
    stock = _stock(op, a, 0, out=out_stock)
    assert got is out_got
    assert np.array_equal(got, stock, equal_nan=True)


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_where_kwarg(op, path):
    a = _arr((ROWS, K), np.float64, seed=121)
    where = np.ones(a.shape, dtype=bool)
    where[::777, 0] = False
    _assert_refused_equal(op, (a, 0), {"where": where})


def test_refusal_initial_kwarg_sum():
    a = _arr((ROWS, K), np.float64, seed=122)
    _assert_refused_equal("numpy.sum", (a, 0), {"initial": 5.0})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_1d_array(op, path):
    a = _arr((ROWS * K,), np.float64, seed=123)
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_python_nested_list(op, path):
    a = _arr((ROWS, K), np.float64, seed=124).tolist()
    _assert_refused_equal(op, (a, 0), {})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_duplicate_axis_raises_typeerror(op, path):
    # np.mean(a, 0, axis=0): duplicate axis binding raises before the
    # predicate's own "1 <= len(args) <= 2 and no axis kwarg" logic even
    # matters; both routes must raise the same exception type.
    a = _arr((ROWS, K), np.float64, seed=125)
    _assert_refused_raises(op, (a, 0), {"axis": 0})


@pytest.mark.parametrize("op, path", OPS, ids=OP_IDS)
def test_refusal_axis_out_of_range_raises(op, path):
    a = _arr((ROWS, K), np.float64, seed=126)
    _assert_refused_raises(op, (a, 5), {})


# ---------------------------------------------------------------------------
# 4. kill switch
# ---------------------------------------------------------------------------

def test_kill_switch_mean_tiny_trailing():
    a = _arr((ROWS, K), np.float64, seed=200)
    decision, reason = GEARBOX.decide("numpy.mean", (a, 0), {})
    assert decision == "mean_tiny_trailing", (decision, reason)
    pyoverdrive.disable_path("mean_tiny_trailing")
    try:
        decision, reason = GEARBOX.decide("numpy.mean", (a, 0), {})
        assert decision == "stock", (decision, reason)
        got = np.mean(a, 0)
        stock = _stock("numpy.mean", a, 0)
        assert np.array_equal(got, stock, equal_nan=True)
    finally:
        pyoverdrive.enable_path("mean_tiny_trailing")


def test_kill_switch_sum_tiny_trailing():
    a = _arr((ROWS, K), np.float64, seed=201)
    decision, reason = GEARBOX.decide("numpy.sum", (a, 0), {})
    assert decision == "sum_tiny_trailing", (decision, reason)
    pyoverdrive.disable_path("sum_tiny_trailing")
    try:
        decision, reason = GEARBOX.decide("numpy.sum", (a, 0), {})
        assert decision == "stock", (decision, reason)
        got = np.sum(a, 0)
        stock = _stock("numpy.sum", a, 0)
        assert np.array_equal(got, stock, equal_nan=True)
    finally:
        pyoverdrive.enable_path("sum_tiny_trailing")
