"""Differential tests: hist2d_uniform fast path vs stock numpy.histogram2d.

Contract (src/pyoverdrive/fastpaths/hist2d_uniform.py): applies only to
histogram2d(x, y, bins=..., range=...) where x and y are plain 1-D
float64 ndarrays of equal length, bins is an int or a pair of ints each
>= 2 with product >= BINS_MIN_TOTAL, range is a pair of finite (lo, hi)
pairs with lo < hi, weights is absent or a plain 1-D float64 ndarray of
the same length, and density/normed are absent. Everything else stays
on stock. Comparison mode: bit-identical - counts (H), xedges, yedges,
dtypes included.
"""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import GEARBOX
from pyoverdrive.fastpaths.hist2d_uniform import SAMPLES_MIN, BINS_MIN_TOTAL

OP = "numpy.histogram2d"
PATH = "hist2d_uniform"


@pytest.fixture(scope="module", autouse=True)
def _enabled():
    pyoverdrive.enable([OP])
    yield
    pyoverdrive.disable()


def _stock(*args, **kwargs):
    return GEARBOX.stock_fn(OP)(*args, **kwargs)


def _xy(n, lo=-3.0, hi=3.0, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=n).astype(np.float64)
    y = rng.uniform(lo, hi, size=n).astype(np.float64)
    return x, y


def _edge_salted(seed=100):
    edges = np.linspace(-3, 3, 41)
    rng = np.random.default_rng(seed)
    n_interior = 400
    on_edges_x = edges[rng.integers(0, len(edges), size=n_interior)]
    on_edges_y = edges[rng.integers(0, len(edges), size=n_interior)]
    endpoints = np.array([-3.0, 3.0] * 50)
    just_outside = np.array(
        [np.nextafter(-3.0, -np.inf), np.nextafter(3.0, np.inf)] * 25
    )
    random_x = rng.uniform(-3, 3, size=SAMPLES_MIN)
    random_y = rng.uniform(-3, 3, size=SAMPLES_MIN)
    x = np.concatenate([on_edges_x, endpoints, just_outside, random_x])
    y = np.concatenate([on_edges_y, endpoints, just_outside, random_y])
    return x.astype(np.float64), y.astype(np.float64)


def _assert_arrays_exactly_equal(got, stock):
    gH, gxe, gye = got
    sH, sxe, sye = stock
    assert type(gH) is type(sH)
    assert type(gxe) is type(sxe)
    assert type(gye) is type(sye)
    assert gH.dtype == sH.dtype, (gH.dtype, sH.dtype)
    assert gxe.dtype == sxe.dtype
    assert gye.dtype == sye.dtype
    assert gH.shape == sH.shape
    assert np.array_equal(gH, sH)
    assert np.array_equal(gxe, sxe)
    assert np.array_equal(gye, sye)


def _assert_dispatched_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == PATH, (decision, reason)
    got = np.histogram2d(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_arrays_exactly_equal(got, stock)
    return got, stock


def _assert_refused_equal(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    got = np.histogram2d(*args, **kwargs)
    stock = _stock(*args, **kwargs)
    _assert_arrays_exactly_equal(got, stock)
    return got


def _assert_refused_raises(args, kwargs):
    decision, reason = GEARBOX.decide(OP, args, kwargs)
    assert decision == "stock", (args, kwargs, decision, reason)
    with pytest.raises(Exception) as got_exc:
        np.histogram2d(*args, **kwargs)
    with pytest.raises(Exception) as stock_exc:
        _stock(*args, **kwargs)
    assert type(got_exc.value) is type(stock_exc.value)


# ---------------------------------------------------------------------------
# 1. dispatch + bit-identity
# ---------------------------------------------------------------------------


def test_dispatch_bins_40x40_with_range():
    x, y = _xy(2000, seed=1)
    _assert_dispatched_equal((x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]})


def test_dispatch_bins_scalar_int_40():
    x, y = _xy(2000, seed=2)
    _assert_dispatched_equal((x, y), {"bins": 40, "range": [[-3, 3], [-3, 3]]})


def test_dispatch_bins_asymmetric_40x60():
    x, y = _xy(2000, seed=3)
    _assert_dispatched_equal((x, y), {"bins": (40, 60), "range": [[-3, 3], [-3, 3]]})


def test_dispatch_positional_bins_and_range():
    x, y = _xy(2000, seed=4)
    _assert_dispatched_equal((x, y, [40, 40], [[-3, 3], [-3, 3]]), {})


def test_dispatch_weights_f64():
    x, y = _xy(2000, seed=5)
    rng = np.random.default_rng(6)
    w = rng.uniform(0.1, 5.0, size=x.size).astype(np.float64)
    _assert_dispatched_equal(
        (x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]], "weights": w}
    )


def test_dispatch_edge_salted_data():
    x, y = _edge_salted(seed=100)
    got, stock = _assert_dispatched_equal(
        (x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    )
    # Extra explicit count-parity check on the decisive edge-salted cell.
    assert int(got[0].sum()) == int(stock[0].sum())


def test_dispatch_samples_exactly_at_outer_edges():
    # tiled above SAMPLES_MIN: this test is about the EDGE semantics, and it
    # must still take the dispatched route to assert them
    x = np.tile(np.array([-3.0, 3.0, -3.0, 3.0, 0.0]), SAMPLES_MIN).astype(np.float64)
    y = np.tile(np.array([-3.0, -3.0, 3.0, 3.0, 0.0]), SAMPLES_MIN).astype(np.float64)
    _assert_dispatched_equal((x, y), {"bins": [30, 30], "range": [[-3, 3], [-3, 3]]})


def test_dispatch_samples_just_outside_edges():
    just_below = np.nextafter(-3.0, -np.inf)
    just_above = np.nextafter(3.0, np.inf)
    x = np.tile([just_below, just_above, just_below, just_above],
                SAMPLES_MIN).astype(np.float64)
    y = np.tile([just_below, just_below, just_above, just_above],
                SAMPLES_MIN).astype(np.float64)
    got, stock = _assert_dispatched_equal(
        (x, y), {"bins": [30, 30], "range": [[-3, 3], [-3, 3]]}
    )
    assert got[0].sum() == 0.0
    assert stock[0].sum() == 0.0


def test_dispatch_negative_range():
    x, y = _xy(SAMPLES_MIN * 2, lo=-7.0, hi=9.0, seed=7)
    _assert_dispatched_equal(
        (x, y), {"bins": [35, 35], "range": [[-7, -1], [2, 9]]}
    )


def test_empty_x_and_y_stays_on_stock_and_matches():
    """An empty input cannot reach SAMPLES_MIN, so it is stock's now.

    It used to dispatch. The sample floor exists because this path's cost
    scales with BINS while stock's scales with SAMPLES, so few samples into
    many bins was measured losing (0.75x at 200 samples) - and zero samples
    is the extreme of that. What the test has to prove is unchanged: the
    answer is exactly stock's.
    """
    x = np.array([], dtype=np.float64)
    y = np.array([], dtype=np.float64)
    kwargs = {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    decision, reason = GEARBOX.decide(OP, (x, y), kwargs)
    assert decision == "stock", (decision, reason)
    _assert_arrays_exactly_equal(np.histogram2d(x, y, **kwargs),
                                 _stock(x, y, **kwargs))


def test_dispatch_all_out_of_range_samples():
    x = np.full(SAMPLES_MIN * 2, 1000.0, dtype=np.float64)
    y = np.full(SAMPLES_MIN * 2, -1000.0, dtype=np.float64)
    got, stock = _assert_dispatched_equal(
        (x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    )
    assert got[0].sum() == 0.0


def test_dispatch_boundary_bins_product_equals_min_total():
    assert BINS_MIN_TOTAL == 900
    x, y = _xy(2000, seed=8)
    decision, reason = GEARBOX.decide(
        OP, (x, y), {"bins": [30, 30], "range": [[-3, 3], [-3, 3]]}
    )
    assert decision == PATH, (decision, reason)
    _assert_dispatched_equal((x, y), {"bins": [30, 30], "range": [[-3, 3], [-3, 3]]})


def test_refusal_bins_product_just_below_min_total():
    assert 29 * 30 < BINS_MIN_TOTAL
    x, y = _xy(2000, seed=9)
    decision, reason = GEARBOX.decide(
        OP, (x, y), {"bins": [29, 30], "range": [[-3, 3], [-3, 3]]}
    )
    assert decision == "stock", (decision, reason)
    _assert_refused_equal((x, y), {"bins": [29, 30], "range": [[-3, 3], [-3, 3]]})


# ---------------------------------------------------------------------------
# 2. refusal routes
# ---------------------------------------------------------------------------


def test_refusal_bins_as_edge_array():
    x, y = _xy(500, seed=10)
    edges = np.linspace(-3, 3, 11)
    _assert_refused_equal((x, y), {"bins": edges})


def test_refusal_bins_40x40_without_range():
    x, y = _xy(500, seed=11)
    _assert_refused_equal((x, y), {"bins": [40, 40]})


def test_refusal_float32_x():
    x, y = _xy(500, seed=12)
    x32 = x.astype(np.float32)
    _assert_refused_equal(
        (x32, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    )


def test_refusal_int64_x():
    rng = np.random.default_rng(13)
    x = rng.integers(-3, 3, size=500).astype(np.int64)
    y = rng.uniform(-3, 3, size=500).astype(np.float64)
    _assert_refused_equal((x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]})


def test_refusal_mismatched_xy_lengths_raises_same_exception():
    x, _ = _xy(500, seed=14)
    _, y = _xy(300, seed=15)
    _assert_refused_raises(
        (x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    )


def test_refusal_density_true():
    x, y = _xy(500, seed=16)
    _assert_refused_equal(
        (x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]], "density": True}
    )


def test_refusal_weights_wrong_length_raises_same_exception():
    x, y = _xy(500, seed=17)
    rng = np.random.default_rng(18)
    w = rng.uniform(0.1, 5.0, size=300).astype(np.float64)
    _assert_refused_raises(
        (x, y),
        {"bins": [40, 40], "range": [[-3, 3], [-3, 3]], "weights": w},
    )


def test_refusal_weights_int64():
    x, y = _xy(500, seed=19)
    rng = np.random.default_rng(20)
    w = rng.integers(1, 10, size=500).astype(np.int64)
    _assert_refused_equal(
        (x, y), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]], "weights": w}
    )


def test_refusal_python_lists():
    x, y = _xy(200, seed=21)
    xl, yl = x.tolist(), y.tolist()
    decision, reason = GEARBOX.decide(
        OP, (xl, yl), {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    )
    assert decision == "stock", (decision, reason)
    got = np.histogram2d(xl, yl, bins=[40, 40], range=[[-3, 3], [-3, 3]])
    stock = _stock(xl, yl, bins=[40, 40], range=[[-3, 3], [-3, 3]])
    _assert_arrays_exactly_equal(got, stock)


# ---------------------------------------------------------------------------
# 3. kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_restores_stock_routing():
    x, y = _xy(2000, seed=22)
    kwargs = {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    decision, reason = GEARBOX.decide(OP, (x, y), kwargs)
    assert decision == PATH, (decision, reason)
    pyoverdrive.disable_path(PATH)
    try:
        decision, reason = GEARBOX.decide(OP, (x, y), kwargs)
        assert decision == "stock", (decision, reason)
        got = np.histogram2d(x, y, **kwargs)
        stock = _stock(x, y, **kwargs)
        _assert_arrays_exactly_equal(got, stock)
    finally:
        pyoverdrive.enable_path(PATH)


@pytest.mark.parametrize("n", [0, 1, 200, 500, SAMPLES_MIN - 1])
def test_below_the_sample_floor_stays_on_stock(n):
    """Few samples into many bins is this path's losing corner and it was
    shipping: measured end to end on the idle box, 200 samples ran at
    0.75-0.81x and 500 at 0.82-0.98x, across every bin count tried.

    The gate was on BIN count alone, which is the wrong axis by itself -
    this path's cost scales with the bins it allocates and clears, stock's
    with the samples it walks. Nothing had ever looked below the canonical
    input, because the canonical input is one somebody picked while the path
    was working.
    """
    x, y = _xy(n, seed=11)
    kwargs = {"bins": [40, 40], "range": [[-3, 3], [-3, 3]]}
    decision, reason = GEARBOX.decide(OP, (x, y), kwargs)
    assert decision == "stock", (n, decision, reason)
    _assert_arrays_exactly_equal(np.histogram2d(x, y, **kwargs),
                                 _stock(x, y, **kwargs))
