# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typing conversion tests.
"""

import ctypes
import numpy as np
import pytest

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.lowering_utilities.type_conversions import to_numba_type


@pytest.mark.parametrize(
    "np_type, numba_type",
    [
        (np.float32, types.float32),
        (np.float64, types.float64),
        (np.float16, types.float16),
        (np.int32, types.int32),
        (np.int64, types.int64),
        (np.complex64, types.complex64),
        (np.complex128, types.complex128),
    ],
)
def test_numpy_scalar_type_conversion(np_type, numba_type):
    """Verify that numpy scalar types convert to correct numba types."""
    assert to_numba_type(np_type) == numba_type


@pytest.mark.parametrize(
    "np_type, numba_type",
    [
        (np.float32, types.float32),
        (np.float64, types.float64),
        (np.int32, types.int32),
    ],
)
def test_numpy_dtype_conversion(np_type, numba_type):
    """Verify that numpy dtypes convert to correct numba types."""
    assert to_numba_type(np.dtype(np_type)) == numba_type


@pytest.mark.parametrize(
    "ctype, numba_type",
    [
        (ctypes.c_float, types.float32),
        (ctypes.c_double, types.float64),
        (ctypes.c_int32, types.int32),
    ],
)
def test_ctypes_conversion(ctype, numba_type):
    """Verify that ctypes types convert to correct numba types."""
    assert to_numba_type(ctype) == numba_type


def test_custom_type_conversion():
    """Verify that a custom type registered through the extension api converts correctly."""

    class CustomFrontendType:
        pass

    class CustomNumbaType(types.Type):
        def __init__(self):
            super().__init__(name="CustomNumbaType")

    custom_numba_type_instance = CustomNumbaType()

    @to_numba_type.register(CustomFrontendType)
    def _(val: CustomFrontendType) -> types.Type:
        return custom_numba_type_instance

    frontend_val = CustomFrontendType()
    assert to_numba_type(frontend_val) == custom_numba_type_instance
