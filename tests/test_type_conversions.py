# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typing conversion tests.
"""

import ctypes
import numpy as np
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.lowering_utilities import get_conversion_signedness
from numba_cuda_mlir.lowering_utilities.type_conversions import to_numba_type
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir.type_defs.vector_types import VectorType


@pytest.mark.parametrize(
    "np_type, numba_type",
    [
        (np.float16, types.float16),
        (np.float32, types.float32),
        (np.float64, types.float64),
        (np.int8, types.int8),
        (np.int16, types.int16),
        (np.int32, types.int32),
        (np.int64, types.int64),
        (np.uint8, types.uint8),
        (np.uint16, types.uint16),
        (np.uint32, types.uint32),
        (np.uint64, types.uint64),
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
        (np.uint32, types.uint32),
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
        (ctypes.c_int8, types.int8),
        (ctypes.c_int16, types.int16),
        (ctypes.c_int32, types.int32),
        (ctypes.c_int64, types.int64),
        (ctypes.c_uint8, types.uint8),
        (ctypes.c_uint16, types.uint16),
        (ctypes.c_uint32, types.uint32),
        (ctypes.c_uint64, types.uint64),
        (ctypes.c_longlong, types.int64),
        (ctypes.c_ulonglong, types.uint64),
        (np.complex64, types.complex64),
        (np.complex128, types.complex128),
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


NUMERIC_CAST_CASES = [
    pytest.param(np.bool_(True), np.int64, np.int64(1), id="bool-to-int64"),
    pytest.param(np.uint8(200), np.float64, 200.0, id="uint8-to-float64"),
    pytest.param(np.uint16(60000), np.float64, 60000.0, id="uint16-to-float64"),
    pytest.param(
        np.uint32(3_000_000_000),
        np.float64,
        3_000_000_000.0,
        id="uint32-to-float64",
    ),
    pytest.param(
        np.uint32(3_000_000_000),
        np.float32,
        np.float32(3_000_000_000),
        id="uint32-to-float32",
    ),
    pytest.param(
        np.float64(3_000_000_000),
        np.uint32,
        np.uint32(3_000_000_000),
        id="float64-to-uint32",
    ),
    pytest.param(np.int8(-56), np.float64, -56.0, id="int8-to-float64"),
    pytest.param(np.uint8(200), np.int64, np.int64(200), id="uint8-to-int64"),
    pytest.param(
        np.int8(-56),
        np.uint32,
        np.uint32(4_294_967_240),
        id="int8-to-uint32",
    ),
    pytest.param(
        np.uint32(3_000_000_000),
        np.int64,
        np.int64(3_000_000_000),
        id="uint32-to-int64",
    ),
    pytest.param(np.int8(-56), np.int64, np.int64(-56), id="int8-to-int64"),
    pytest.param(np.int32(-5), np.int64, np.int64(-5), id="int32-to-int64"),
]


@pytest.mark.parametrize("source, destination_dtype, expected", NUMERIC_CAST_CASES)
def test_numeric_assignment_preserves_signedness(source, destination_dtype, expected):
    @cuda.jit
    def cast(source, destination):
        destination[0] = source[0]

    source = cuda.to_device(np.array([source]))
    destination = cuda.device_array(1, dtype=destination_dtype)
    cast[1, 1](source, destination)
    np.testing.assert_equal(destination.copy_to_host()[0], expected)


@pytest.mark.parametrize("source, destination_dtype, expected", NUMERIC_CAST_CASES)
def test_explicit_numeric_cast_preserves_signedness(source, destination_dtype, expected):
    @cuda.jit
    def cast(source, destination):
        destination[0] = destination_dtype(source[0])

    source = cuda.to_device(np.array([source]))
    destination = cuda.device_array(1, dtype=destination_dtype)
    cast[1, 1](source, destination)
    np.testing.assert_equal(destination.copy_to_host()[0], expected)


@pytest.mark.parametrize(
    "source_type,target_type,expected",
    [
        (types.int8, types.int64, True),
        (types.uint8, types.int64, False),
        (types.int8, types.uint32, True),
        (types.float64, types.int32, True),
        (types.float64, types.uint32, False),
        (types.boolean, types.int64, False),
        (types.float64, types.boolean, False),
        (VectorType(types.int8, 2), VectorType(types.float64, 2), True),
        (VectorType(types.uint8, 2), VectorType(types.float64, 2), False),
        (VectorType(types.float64, 2), VectorType(types.int32, 2), True),
        (VectorType(types.float64, 2), VectorType(types.uint32, 2), False),
        (types.float32, types.float64, None),
    ],
    ids=[
        "signed-source",
        "unsigned-source",
        "source-precedes-target",
        "signed-target",
        "unsigned-target",
        "boolean-source",
        "boolean-target",
        "signed-vector-source",
        "unsigned-vector-source",
        "signed-vector-target",
        "unsigned-vector-target",
        "no-integer-type",
    ],
)
def test_get_conversion_signedness(source_type, target_type, expected):
    assert get_conversion_signedness(source_type, target_type) is expected


@pytest.mark.parametrize(
    "source_type,target_type,expected_signed",
    [
        (VectorType(types.int8, 2), VectorType(types.float64, 2), True),
        (VectorType(types.uint8, 2), VectorType(types.float64, 2), False),
        (VectorType(types.float64, 2), VectorType(types.int32, 2), True),
        (VectorType(types.float64, 2), VectorType(types.uint32, 2), False),
    ],
    ids=["int-to-float", "uint-to-float", "float-to-int", "float-to-uint"],
)
def test_lower_cast_uses_vector_integer_signedness(
    monkeypatch, source_type, target_type, expected_signed
):
    class CastRegistry:
        def find(self, _):
            return None

    captured = {}

    def fake_convert(value, target, *, signed):
        captured["target"] = target
        captured["signed"] = signed
        return value

    lower = object.__new__(MLIRLower)
    lower.context = type("Context", (), {"_casts": CastRegistry()})()
    monkeypatch.setattr(MLIRLower, "get_mlir_type", lambda _, ty: ty)
    monkeypatch.setattr("numba_cuda_mlir.mlir_lowering.convert", fake_convert)

    value = object()
    assert lower.lower_cast(source_type, target_type, value) is value
    assert captured == {"target": target_type, "signed": expected_signed}
