"""Fast path registry.

Each module in this package defines ``register(gearbox)`` and is listed in
``_MODULES``. A fast path may only be listed here once it has:

- an applicability predicate and stock fallback (automatic via Gearbox),
- a Dyno crossover benchmark under benchmarks/,
- differential tests under compatibility/,
- a provenance record (opportunity id + sources),
- a kill switch (automatic via Gearbox name registration).

See docs/BUILD_SPEC.md section 10.2.
"""

from __future__ import annotations

from . import (
    argmax_blocked,
    char_view,
    cholesky_small_batch,
    eigvalsh_3x3,
    hist2d_uniform,
    interp_uniform_grid,
    inv_small_batch,
    isin_object_hash,
    linalg_small_batch,
    matmul_int_blas,
    median_partition,
    nan_to_num_where,
    nanargminmax_scan,
    nanmedian_scan,
    nanreduce_scan,
    apply_along_axis_reduce,
    percentile_dense,
    qr_small_batch,
    svd_small_batch,
    vectorize_ufunc,
    searchsorted_extreme_key,
    take_index_assign,
    unique_rows_lexsort,
    dot_mixed_view,
    eigvalsh_2x2,
    matmul_split_complex,
    nanpercentile_masked,
    roll_concat,
    einsum_optimize,
    isclose_fused,
    isin_string_hash,
    fftconvolve,
    nanquantile_masked,
    inner_tensordot,
    intersect_sorted,
    noop,
    parallel_binary,
    parallel_ufunc,
    quantile_dense_sort,
    reduce_tiny_trailing,
    relayout_blocked,
    searchsorted_sortqueries,
    unique_axis0_column,
    unique_sort,
)

_MODULES = [
    noop,
    unique_sort,
    inner_tensordot,
    parallel_ufunc,
    parallel_binary,
    intersect_sorted,
    relayout_blocked,
    unique_axis0_column,
    fftconvolve,
    nanquantile_masked,
    einsum_optimize,
    searchsorted_sortqueries,
    isclose_fused,
    isin_string_hash,
    dot_mixed_view,
    quantile_dense_sort,
    char_view,
    reduce_tiny_trailing,
    eigvalsh_2x2,
    matmul_split_complex,
    roll_concat,
    argmax_blocked,  # calibration-gated: registers disabled
    inv_small_batch,
    isin_object_hash,
    median_partition,
    hist2d_uniform,
    unique_rows_lexsort,
    searchsorted_extreme_key,
    percentile_dense,
    nanpercentile_masked,
    nanreduce_scan,
    nanargminmax_scan,
    nanmedian_scan,
    matmul_int_blas,
    linalg_small_batch,
    nan_to_num_where,
    cholesky_small_batch,
    eigvalsh_3x3,
    interp_uniform_grid,
    take_index_assign,
    qr_small_batch,
    apply_along_axis_reduce,
    vectorize_ufunc,
    svd_small_batch,
]


def register_all(gearbox) -> None:
    for mod in _MODULES:
        mod.register(gearbox)
