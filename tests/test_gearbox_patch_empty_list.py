"""patch([]) targets zero operations; only None means "all supported"."""

import numpy as np
import pytest

import pyoverdrive
from pyoverdrive.dispatcher.gearbox import FastPath, Gearbox


@pytest.fixture(autouse=True)
def _always_unpatched():
    yield
    pyoverdrive.disable()


def test_patch_with_empty_list_leaves_numpy_untouched():
    box = Gearbox()
    box.register(FastPath(
        name="noop_positive",
        op="numpy.positive",
        applicable=lambda args, kwargs: True,
        run=lambda x: x,
    ))
    original = np.positive
    try:
        box.patch([])
        assert np.positive is original
    finally:
        box.unpatch()
