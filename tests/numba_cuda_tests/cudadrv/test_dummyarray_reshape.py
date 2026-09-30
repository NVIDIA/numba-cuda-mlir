# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import unittest

import numpy as np

from numba_cuda_mlir.numba_cuda.cudadrv import dummyarray


def make_dummy(arr):
    return dummyarray.Array.from_desc(0, arr.shape, arr.strides, arr.itemsize)


class TestDummyArrayReshape(unittest.TestCase):
    """Regression tests for Array.reshape() honoring the requested order.

    ``attempt_nocopy_reshape`` (and the fast paths above it) must only
    produce a zero-copy view when the requested `order` actually matches
    how the array is laid out in memory. Taking the fast path whenever the
    array is contiguous in *either* order - regardless of the requested
    order - yields a view with the wrong strides: the read succeeds, but
    silently returns the wrong data instead of raising or copying.

    Each reshape here is checked from both directions: when `order`
    matches the array's own layout, a no-copy view must still be produced
    (guarding against an overly conservative fix), and when it doesn't,
    NumPy itself would need to copy, so `Array.reshape` must raise rather
    than fabricate an incorrect view.
    """

    def _check_matching_order_reshape(self, base_shape, newshape, order):
        base = np.arange(int(np.prod(base_shape)), dtype=np.int64).reshape(base_shape)
        real = np.ascontiguousarray(base) if order == "C" else np.asfortranarray(base)
        dummy = make_dummy(real)

        newarr, extents = dummy.reshape(*newshape, order=order)

        view = np.lib.stride_tricks.as_strided(real, shape=newshape, strides=newarr.strides)
        expected = real.reshape(newshape, order=order)
        np.testing.assert_array_equal(
            view,
            expected,
            err_msg=(
                f"reshape({newshape}, order={order!r}) on an already "
                f"{order}-contiguous {base_shape} array should stay a "
                f"correct no-copy view"
            ),
        )

    def _check_mismatched_order_reshape(self, base_shape, newshape, order):
        base = np.arange(int(np.prod(base_shape)), dtype=np.int64).reshape(base_shape)
        other_order = "F" if order == "C" else "C"
        real = np.ascontiguousarray(base) if other_order == "C" else np.asfortranarray(base)
        dummy = make_dummy(real)

        with self.assertRaises(
            NotImplementedError,
            msg=(
                f"reshape({newshape}, order={order!r}) on a "
                f"{other_order}-contiguous {base_shape} array requires a copy "
                f"and must not silently return a wrong no-copy view"
            ),
        ):
            dummy.reshape(*newshape, order=order)

    def test_reshape_flatten_order_c(self):
        self._check_matching_order_reshape((2, 3), (6,), "C")
        self._check_mismatched_order_reshape((2, 3), (6,), "C")

    def test_reshape_flatten_order_f(self):
        self._check_matching_order_reshape((2, 3), (6,), "F")
        self._check_mismatched_order_reshape((2, 3), (6,), "F")

    def test_reshape_2d_to_2d_order_c(self):
        self._check_matching_order_reshape((2, 3), (3, 2), "C")
        self._check_mismatched_order_reshape((2, 3), (3, 2), "C")

    def test_reshape_2d_to_2d_order_f(self):
        self._check_matching_order_reshape((2, 3), (3, 2), "F")
        self._check_mismatched_order_reshape((2, 3), (3, 2), "F")

    def test_reshape_3d(self):
        for order in ("C", "F"):
            self._check_matching_order_reshape((2, 3, 4), (4, 3, 2), order)
            self._check_mismatched_order_reshape((2, 3, 4), (4, 3, 2), order)


if __name__ == "__main__":
    unittest.main()
