# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from numba_cuda_mlir.numba_cuda.api import _dtype_from_cuda_array_interface


class _Owner:
    def __init__(self, dtype):
        self.dtype = dtype


def test_cuda_array_interface_uses_supported_owner_dtype_for_void_typestr():
    owner_dtype = np.dtype([("x", np.float16), ("y", np.float16)])
    dtype = _dtype_from_cuda_array_interface({"typestr": "|V4"}, _Owner(owner_dtype))
    assert dtype == owner_dtype


def test_cuda_array_interface_unsupported_owner_dtype_falls_back_to_typestr():
    owner_dtype = np.dtype([("bad", object)])
    dtype = _dtype_from_cuda_array_interface({"typestr": "|V8"}, _Owner(owner_dtype))
    assert dtype == np.dtype("|V8")


@pytest.mark.parametrize(
    "dtype, offset, stride, aligned",
    [
        (np.float64, 0, 8, True),
        (np.float64, 1, 8, False),
        (np.float64, 0, 12, False),
        (np.complex128, 8, 16, True),
        (np.complex128, 0, 24, False),
        (np.complex128, 0, -16, True),
        (np.dtype([("x", np.float64), ("y", np.int32)]), 0, 12, False),
        (np.dtype([("x", np.float64), ("y", np.int32)], align=True), 0, 16, True),
        (np.dtype("S0"), 0, 0, False),
        (np.dtype("U0"), 0, 0, False),
        (np.dtype([]), 0, 0, False),
    ],
)
def test_host_and_cuda_interface_classify_alignment(dtype, offset, stride, aligned):
    from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

    dtype = np.dtype(dtype)
    raw = np.empty(128, dtype=np.uint8)
    offset += 2 * abs(stride) if stride < 0 else 0
    host = np.ndarray((3,), dtype=dtype, buffer=raw, offset=offset, strides=(stride,))
    owner = _Owner(dtype)
    owner.__cuda_array_interface__ = {
        "version": 3,
        "typestr": dtype.str,
        "shape": host.shape,
        "strides": host.strides,
        "data": (host.ctypes.data, False),
    }
    for array in (host, owner):
        ty = typeof(array)
        assert ty.aligned == aligned


@pytest.mark.parametrize(
    "strides, offset, layout, aligned",
    [
        (None, 0, "C", True),
        ((6, 2), 0, "C", True),
        ((2, 4), 0, "F", True),
        ((12, 4), 0, "A", True),
        ((7, 2), 0, "A", False),
        (None, 1, "C", False),
    ],
)
def test_bfloat_cuda_interface_preserves_dtype_layout_and_alignment(
    strides, offset, layout, aligned
):
    import ml_dtypes

    from numba_cuda_mlir.numba_cuda import types
    from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

    raw = np.empty(128, dtype=np.uint8)
    owner = _Owner(np.dtype(ml_dtypes.bfloat16))
    owner.__cuda_array_interface__ = {
        "version": 3,
        "typestr": "|V2",
        "shape": (2, 3),
        "strides": strides,
        "data": (raw.ctypes.data + offset, False),
    }
    ty = typeof(owner)
    assert ty.dtype == types.bfloat16
    assert ty.ndim == 2
    assert ty.layout == layout
    assert ty.aligned == aligned


@pytest.mark.parametrize("owner_dtype", ["float64", np.float64])
def test_cuda_interface_normalizes_dtype_like_owner(owner_dtype):
    assert _dtype_from_cuda_array_interface({"typestr": "|V8"}, _Owner(owner_dtype)) == np.dtype(
        "float64"
    )
