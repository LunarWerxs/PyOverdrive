"""PyRallel: persistent thread-pool execution core. See pyrallel.py."""

from .pyrallel import (  # noqa: F401
    available,
    errstate_parallel_safe,
    max_threads,
    parallel_elementwise,
    parallel_unary,
    pool_size,
    set_max_threads,
    shutdown,
)
