# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for NumPy ufunc operations with output array parameters.

Tests element-wise operations with signatures:
- (scalar, array) -> array: Scalar input stored in output array
- (array, array) -> array: Element-wise unary operation with output array
- (scalar, scalar, array) -> array: Binary op with scalar inputs
- (array, array, array) -> array: Element-wise binary operation with output array
"""

import numpy as np
import pytest
from numba_cuda_mlir import cuda

DeviceNDArray = cuda.DeviceNDArray


# =============================================================================
# Unary ufuncs: (Array, Array) signature
# =============================================================================

UNARY_FLOAT_UFUNCS = [
    # Trig functions
    pytest.param(np.sin, np.array([0.0, np.pi / 6, np.pi / 2]), id="sin"),
    pytest.param(np.cos, np.array([0.0, np.pi / 3, np.pi]), id="cos"),
    pytest.param(np.tan, np.array([0.0, np.pi / 4, np.pi / 6]), id="tan"),
    # Inverse trig functions
    pytest.param(np.arcsin, np.array([0.0, 0.5, 1.0]), id="arcsin"),
    pytest.param(np.arccos, np.array([0.0, 0.5, 1.0]), id="arccos"),
    pytest.param(np.arctan, np.array([0.0, 0.5, 1.0]), id="arctan"),
    # Hyperbolic functions
    pytest.param(np.sinh, np.array([0.0, 0.5, 1.0]), id="sinh"),
    pytest.param(np.cosh, np.array([0.0, 0.5, 1.0]), id="cosh"),
    pytest.param(np.tanh, np.array([0.0, 0.5, 1.0]), id="tanh"),
    # Inverse hyperbolic functions
    pytest.param(np.arcsinh, np.array([0.0, 0.5, 1.0]), id="arcsinh"),
    pytest.param(np.arccosh, np.array([1.0, 1.5, 2.0]), id="arccosh"),
    pytest.param(np.arctanh, np.array([0.0, 0.3, 0.6]), id="arctanh"),
    # Angle conversion
    pytest.param(np.deg2rad, np.array([0.0, 90.0, 180.0]), id="deg2rad"),
    pytest.param(np.rad2deg, np.array([0.0, np.pi / 2, np.pi]), id="rad2deg"),
    pytest.param(np.degrees, np.array([0.0, np.pi / 2, np.pi]), id="degrees"),
    pytest.param(np.radians, np.array([0.0, 90.0, 180.0]), id="radians"),
    # Log functions
    pytest.param(np.log, np.array([1.0, np.e, np.e**2]), id="log"),
    pytest.param(np.log2, np.array([1.0, 2.0, 8.0]), id="log2"),
    pytest.param(np.log10, np.array([1.0, 10.0, 100.0]), id="log10"),
]


@pytest.mark.parametrize("ufunc,input_data", UNARY_FLOAT_UFUNCS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_unary_float_ufunc_array_to_array(ufunc, input_data, dtype):
    """Test unary ufuncs with (array, out_array) signature."""

    @cuda.jit
    def kernel(inp: DeviceNDArray, out: DeviceNDArray):
        ufunc(inp, out)

    inp = cuda.to_device(input_data.astype(dtype))
    out = cuda.to_device(np.zeros_like(input_data, dtype=dtype))
    kernel[1, 1, 0, 0](inp, out)

    result = out.copy_to_host()
    expected = ufunc(input_data.astype(dtype))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


# Test with single negative value (matches numba-cuda test pattern)
TRIG_UFUNCS_NEGATIVE_INPUT = [
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh,
    np.arctanh,
]


@pytest.mark.parametrize("ufunc", TRIG_UFUNCS_NEGATIVE_INPUT)
def test_unary_ufunc_negative_float32(ufunc):
    """Test unary ufuncs with negative float32 input (matches numba-cuda test)."""

    @cuda.jit
    def kernel(inp: DeviceNDArray, out: DeviceNDArray):
        ufunc(inp, out)

    inp = cuda.to_device(np.array([-0.5], dtype=np.float32))
    out = cuda.to_device(np.array([0.0], dtype=np.float32))
    kernel[1, 1, 0, 0](inp, out)

    result = out.copy_to_host()
    expected = ufunc(np.array([-0.5], dtype=np.float32))
    np.testing.assert_array_almost_equal(result, expected, decimal=5)


# =============================================================================
# Binary ufuncs: (Array, Array, Array) signature
# =============================================================================

BINARY_FLOAT_UFUNCS = [
    pytest.param(
        np.arctan2,
        np.array([1.0, 0.0, -1.0]),
        np.array([1.0, 1.0, 1.0]),
        id="arctan2",
    ),
    pytest.param(
        np.hypot,
        np.array([3.0, 5.0, 8.0]),
        np.array([4.0, 12.0, 15.0]),
        id="hypot",
    ),
    pytest.param(
        np.maximum,
        np.array([1.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        id="maximum",
    ),
    pytest.param(
        np.minimum,
        np.array([1.0, 5.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        id="minimum",
    ),
    pytest.param(
        np.fmax,
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 1.0, 4.0]),
        id="fmax",
    ),
    pytest.param(
        np.fmin,
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 1.0, 4.0]),
        id="fmin",
    ),
]


@pytest.mark.parametrize("ufunc,arr1,arr2", BINARY_FLOAT_UFUNCS)
def test_binary_float_ufunc_array_to_array(ufunc, arr1, arr2):
    """Test binary float ufuncs with (array, array, out_array) signature."""

    @cuda.jit
    def kernel(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray):
        ufunc(a, b, out)

    a = cuda.to_device(arr1.astype(np.float64))
    b = cuda.to_device(arr2.astype(np.float64))
    out = cuda.to_device(np.zeros_like(arr1, dtype=np.float64))
    kernel[1, 1, 0, 0](a, b, out)

    result = out.copy_to_host()
    expected = ufunc(arr1, arr2)
    np.testing.assert_array_almost_equal(result, expected, decimal=10)


# =============================================================================
# Comparison ufuncs: return bool, stored as float or int
# =============================================================================

COMPARISON_UFUNCS = [
    pytest.param(np.greater, id="greater"),
    pytest.param(np.less, id="less"),
    pytest.param(np.equal, id="equal"),
    pytest.param(np.not_equal, id="not_equal"),
    pytest.param(np.greater_equal, id="greater_equal"),
    pytest.param(np.less_equal, id="less_equal"),
]

COMPARISON_TEST_DATA = [
    (np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])),
]


@pytest.mark.parametrize("ufunc", COMPARISON_UFUNCS)
def test_comparison_ufunc_array_to_array_float_output(ufunc):
    """Test comparison ufuncs with float output array."""
    arr1, arr2 = np.array([1.0, 2.0, 3.0]), np.array([2.0, 2.0, 2.0])

    @cuda.jit
    def kernel(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray):
        ufunc(a, b, out)

    a = cuda.to_device(arr1.astype(np.float64))
    b = cuda.to_device(arr2.astype(np.float64))
    out = cuda.to_device(np.zeros_like(arr1, dtype=np.float64))
    kernel[1, 1, 0, 0](a, b, out)

    result = out.copy_to_host()
    expected = ufunc(arr1, arr2).astype(np.float64)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("ufunc", COMPARISON_UFUNCS)
def test_comparison_ufunc_array_to_array_int_output(ufunc):
    """Test comparison ufuncs with int output array."""
    arr1, arr2 = np.array([1, 2, 3], dtype=np.int32), np.array(
        [2, 2, 2], dtype=np.int32
    )

    @cuda.jit
    def kernel(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray):
        ufunc(a, b, out)

    a = cuda.to_device(arr1)
    b = cuda.to_device(arr2)
    out = cuda.to_device(np.zeros_like(arr1, dtype=np.int32))
    kernel[1, 1, 0, 0](a, b, out)

    result = out.copy_to_host()
    expected = ufunc(arr1, arr2).astype(np.int32)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("ufunc", COMPARISON_UFUNCS)
def test_comparison_ufunc_scalar_to_array(ufunc):
    """Test comparison ufuncs with scalar inputs and output array."""

    @cuda.jit
    def kernel(a: cuda.float64, b: cuda.float64, out: DeviceNDArray):
        ufunc(a, b, out)

    out = cuda.to_device(np.zeros(1, dtype=np.float64))
    kernel[1, 1, 0, 0](np.float64(1.0), np.float64(2.0), out)

    result = out.copy_to_host()
    expected = np.array([float(ufunc(1.0, 2.0))])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Logical ufuncs
# =============================================================================

BINARY_LOGICAL_UFUNCS = [
    pytest.param(np.logical_and, id="logical_and"),
    pytest.param(np.logical_or, id="logical_or"),
    pytest.param(np.logical_xor, id="logical_xor"),
]


@pytest.mark.parametrize("ufunc", BINARY_LOGICAL_UFUNCS)
def test_logical_ufunc_array_to_array(ufunc):
    """Test binary logical ufuncs with array inputs."""
    arr1 = np.array([1.0, 0.0, 1.0, 0.0])
    arr2 = np.array([1.0, 1.0, 0.0, 0.0])

    @cuda.jit
    def kernel(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray):
        ufunc(a, b, out)

    a = cuda.to_device(arr1.astype(np.float64))
    b = cuda.to_device(arr2.astype(np.float64))
    out = cuda.to_device(np.zeros_like(arr1, dtype=np.float64))
    kernel[1, 1, 0, 0](a, b, out)

    result = out.copy_to_host()
    expected = ufunc(arr1, arr2).astype(np.float64)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("ufunc", BINARY_LOGICAL_UFUNCS)
def test_logical_ufunc_scalar_to_array(ufunc):
    """Test binary logical ufuncs with scalar inputs."""

    @cuda.jit
    def kernel(a: cuda.float64, b: cuda.float64, out: DeviceNDArray):
        ufunc(a, b, out)

    out = cuda.to_device(np.zeros(1, dtype=np.float64))
    kernel[1, 1, 0, 0](np.float64(1.0), np.float64(0.0), out)

    result = out.copy_to_host()
    expected = np.array([float(ufunc(1.0, 0.0))])
    np.testing.assert_array_equal(result, expected)


def test_logical_not_array_to_array():
    """Test logical_not with array input."""
    arr = np.array([1.0, 0.0, 1.0, 0.0])

    @cuda.jit
    def kernel(inp: DeviceNDArray, out: DeviceNDArray):
        np.logical_not(inp, out)

    inp = cuda.to_device(arr.astype(np.float64))
    out = cuda.to_device(np.zeros_like(arr, dtype=np.float64))
    kernel[1, 1, 0, 0](inp, out)

    result = out.copy_to_host()
    expected = np.logical_not(arr).astype(np.float64)
    np.testing.assert_array_equal(result, expected)


def test_logical_not_scalar_to_array():
    """Test logical_not with scalar input."""

    @cuda.jit
    def kernel(val: cuda.float64, out: DeviceNDArray):
        np.logical_not(val, out)

    out = cuda.to_device(np.zeros(1, dtype=np.float64))
    kernel[1, 1, 0, 0](np.float64(1.0), out)

    result = out.copy_to_host()
    expected = np.array([float(np.logical_not(1.0))])
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Bitwise ufuncs (integer only)
# =============================================================================

BINARY_BITWISE_UFUNCS = [
    pytest.param(np.bitwise_and, id="bitwise_and"),
    pytest.param(np.bitwise_or, id="bitwise_or"),
    pytest.param(np.bitwise_xor, id="bitwise_xor"),
]


@pytest.mark.parametrize("ufunc", BINARY_BITWISE_UFUNCS)
def test_bitwise_ufunc_array_to_array(ufunc):
    """Test binary bitwise ufuncs with int32 array inputs."""
    arr1 = np.array([0b1100, 0b1010, 0b1111], dtype=np.int32)
    arr2 = np.array([0b1010, 0b1100, 0b0000], dtype=np.int32)

    @cuda.jit
    def kernel(a: DeviceNDArray, b: DeviceNDArray, out: DeviceNDArray):
        ufunc(a, b, out)

    a = cuda.to_device(arr1)
    b = cuda.to_device(arr2)
    out = cuda.to_device(np.zeros_like(arr1))
    kernel[1, 1, 0, 0](a, b, out)

    result = out.copy_to_host()
    expected = ufunc(arr1, arr2)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize("ufunc", BINARY_BITWISE_UFUNCS)
def test_bitwise_ufunc_scalar_to_array(ufunc):
    """Test binary bitwise ufuncs with int32 scalar inputs."""

    @cuda.jit
    def kernel(a: cuda.int32, b: cuda.int32, out: DeviceNDArray):
        ufunc(a, b, out)

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1, 0, 0](np.int32(0b1100), np.int32(0b1010), out)

    result = out.copy_to_host()
    expected = np.array([ufunc(0b1100, 0b1010)], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


def test_invert_array_to_array():
    """Test bitwise invert with int32 array input."""
    arr = np.array([0b0000, 0b1111, 0b1010], dtype=np.int32)

    @cuda.jit
    def kernel(inp: DeviceNDArray, out: DeviceNDArray):
        np.invert(inp, out)

    inp = cuda.to_device(arr)
    out = cuda.to_device(np.zeros_like(arr))
    kernel[1, 1, 0, 0](inp, out)

    result = out.copy_to_host()
    expected = np.invert(arr)
    np.testing.assert_array_equal(result, expected)


def test_invert_scalar_to_array():
    """Test bitwise invert with int32 scalar input."""

    @cuda.jit
    def kernel(val: cuda.int32, out: DeviceNDArray):
        np.invert(val, out)

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    kernel[1, 1, 0, 0](np.int32(0b1010), out)

    result = out.copy_to_host()
    expected = np.array([np.invert(np.int32(0b1010))], dtype=np.int32)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# Binary ufuncs with scalar inputs: (scalar, scalar, Array) signature
# =============================================================================

BINARY_SCALAR_UFUNCS = [
    pytest.param(np.arctan2, 1.0, 1.0, id="arctan2"),
    pytest.param(np.hypot, 3.0, 4.0, id="hypot"),
    pytest.param(np.maximum, 3.0, 5.0, id="maximum"),
    pytest.param(np.minimum, 3.0, 5.0, id="minimum"),
    pytest.param(np.fmax, 3.0, 5.0, id="fmax"),
    pytest.param(np.fmin, 3.0, 5.0, id="fmin"),
]


@pytest.mark.parametrize("ufunc,val1,val2", BINARY_SCALAR_UFUNCS)
def test_binary_float_ufunc_scalar_to_array(ufunc, val1, val2):
    """Test binary float ufuncs with (scalar, scalar, out_array) signature."""

    @cuda.jit
    def kernel(a: cuda.float64, b: cuda.float64, out: DeviceNDArray):
        ufunc(a, b, out)

    out = cuda.to_device(np.zeros(1, dtype=np.float64))
    kernel[1, 1, 0, 0](np.float64(val1), np.float64(val2), out)

    result = out.copy_to_host()
    expected = np.array([ufunc(val1, val2)])
    np.testing.assert_array_almost_equal(result, expected, decimal=10)
