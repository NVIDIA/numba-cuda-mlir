# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Typing and MLIR generation tests for narrow float types (fp8, fp4, fp6).
"""

from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler
from numba_cuda_mlir.testing import filecheck
import pytest


_FP8_TYPES = [
    types.f8E3M4,
    types.f8E4M3B11FNUZ,
    types.f8E4M3FN,
    types.f8E4M3FNUZ,
    types.f8E4M3,
    types.f8E5M2FNUZ,
    types.f8E5M2,
    types.f8E8M0FNU,
]

_FP4_FP6_TYPES = [
    types.f4E2M1FN,
    types.f6E2M3FN,
    types.f6E3M2FN,
]

_ALL_NARROW_TYPES = _FP8_TYPES + _FP4_FP6_TYPES


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_array_parameter(dtype):
    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        print(x)

    mlir = compiler.compile_mlir(k, types.void(dtype[:]))
    assert dtype.name in mlir


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_array_parameter_optimized(dtype):
    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        print(x)

    mlir = compiler.compile_mlir(k, types.void(dtype[:]), optimized=True)
    assert "test_compile_array_parameter_optimized" in mlir


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_add(dtype):
    """Verify that add(dtype, dtype) -> dtype compiles to correct MLIR."""

    @cuda.jit(opt_level=3)
    def k(out: cuda.DeviceNDArray, a: cuda.DeviceNDArray, b: cuda.DeviceNDArray):
        out[0] = a[0] + b[0]

    mlir = compiler.compile_mlir(k, types.void(dtype[:], dtype[:], dtype[:]))
    filecheck(
        f"""
        CHECK: arith.extf
        CHECK-SAME: {dtype.name} to f32
        CHECK: arith.addf
        CHECK-SAME: f32
        CHECK: arith.truncf
        CHECK-SAME: f32 to {dtype.name}
        """,
        mlir,
    )


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_sub(dtype):
    @cuda.jit(opt_level=3)
    def k(out: cuda.DeviceNDArray, a: cuda.DeviceNDArray, b: cuda.DeviceNDArray):
        out[0] = a[0] - b[0]

    mlir = compiler.compile_mlir(k, types.void(dtype[:], dtype[:], dtype[:]))
    filecheck(
        """
        CHECK: arith.subf
        CHECK-SAME: f32
        """,
        mlir,
    )


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_mul(dtype):
    @cuda.jit(opt_level=3)
    def k(out: cuda.DeviceNDArray, a: cuda.DeviceNDArray, b: cuda.DeviceNDArray):
        out[0] = a[0] * b[0]

    mlir = compiler.compile_mlir(k, types.void(dtype[:], dtype[:], dtype[:]))
    filecheck(
        """
        CHECK: arith.mulf
        CHECK-SAME: f32
        """,
        mlir,
    )


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_div(dtype):
    @cuda.jit(opt_level=3)
    def k(out: cuda.DeviceNDArray, a: cuda.DeviceNDArray, b: cuda.DeviceNDArray):
        out[0] = a[0] / b[0]

    mlir = compiler.compile_mlir(k, types.void(dtype[:], dtype[:], dtype[:]))
    filecheck(
        """
        CHECK: arith.divf
        CHECK-SAME: f32
        """,
        mlir,
    )


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_comparison(dtype):
    @cuda.jit(opt_level=3)
    def k(out: cuda.DeviceNDArray, a: cuda.DeviceNDArray, b: cuda.DeviceNDArray):
        if a[0] < b[0]:
            out[0] = a[0]
        else:
            out[0] = b[0]

    mlir = compiler.compile_mlir(k, types.void(dtype[:], dtype[:], dtype[:]))
    filecheck(
        """
        CHECK: arith.cmpf
        CHECK-SAME: f32
        """,
        mlir,
    )


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_negation(dtype):
    @cuda.jit(opt_level=3)
    def k(out: cuda.DeviceNDArray, a: cuda.DeviceNDArray):
        out[0] = -a[0]

    mlir = compiler.compile_mlir(k, types.void(dtype[:], dtype[:]))
    filecheck(
        """
        CHECK: arith.negf
        CHECK-SAME: f32
        """,
        mlir,
    )


@pytest.mark.parametrize("dtype", _ALL_NARROW_TYPES, ids=lambda t: t.name)
def test_compile_2d_array(dtype):
    @cuda.jit(opt_level=3)
    def k(x: cuda.DeviceNDArray):
        print(x)

    mlir = compiler.compile_mlir(k, types.void(dtype[:, :]))
    assert dtype.name in mlir


def test_type_conversion_roundtrip():
    """Verify to_mlir_type and to_numba_type are inverses for all narrow types."""
    from numba_cuda_mlir._mlir import ir

    with ir.Context():
        from numba_cuda_mlir.lowering_utilities.type_conversions import (
            to_mlir_type,
            to_numba_type,
        )

        for ty in _ALL_NARROW_TYPES:
            mlir_ty = to_mlir_type(ty)
            numba_ty = to_numba_type(mlir_ty)
            assert str(numba_ty) == str(ty), f"Round-trip failed for {ty}"
