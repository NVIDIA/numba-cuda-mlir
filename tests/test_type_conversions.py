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

def test_numpy_scalar_type_conversion():
    """Verify that numpy scalar types convert to correct numba types."""
    assert to_numba_type(np.float32) == types.float32
    assert to_numba_type(np.float64) == types.float64
    assert to_numba_type(np.float16) == types.float16
    assert to_numba_type(np.int32) == types.int32
    assert to_numba_type(np.int64) == types.int64
    assert to_numba_type(np.complex64) == types.complex64
    assert to_numba_type(np.complex128) == types.complex128

def test_numpy_dtype_conversion():
    """Verify that numpy dtypes convert to correct numba types."""
    assert to_numba_type(np.dtype(np.float32)) == types.float32
    assert to_numba_type(np.dtype(np.float64)) == types.float64
    assert to_numba_type(np.dtype(np.int32)) == types.int32

def test_ctypes_conversion():
    """Verify that ctypes types convert to correct numba types."""
    assert to_numba_type(ctypes.c_float) == types.float32
    assert to_numba_type(ctypes.c_double) == types.float64
    assert to_numba_type(ctypes.c_int32) == types.int32
