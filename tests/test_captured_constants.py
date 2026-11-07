# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import ModuleType

import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.lowering_utilities import signless_int

NP_TRUE = np.bool_(True)
NP_FALSE = np.bool_(False)
NP_F32 = np.float32(3.5)
NP_I8 = np.int8(7)
PY_COMPLEX = 2.0 - 1.5j
NP_C64 = np.complex64(2.0 - 1.5j)
COMPLEX_TUPLE = (2.0 - 1.5j, 0.5j)

# Integer constants of 2**63 or more, which do not fit int64_t (issue #372).
GOLDEN_GAMMA = 0x9E3779B97F4A7C15
NP_GOLDEN_GAMMA = np.uint64(GOLDEN_GAMMA)
ALL_ONES = 0xFFFFFFFFFFFFFFFF
INT64_MAX = 2**63 - 1
UINT64_MIN_UNSIGNED = 2**63

# Attributes of a module are lowered through getattr rather than as globals.
CONSTANTS = ModuleType("constants")
CONSTANTS.ONE = np.uint64(1)
CONSTANTS.F32 = NP_F32
CONSTANTS.I8 = NP_I8
CONSTANTS.TRUE = NP_TRUE
CONSTANTS.C64 = NP_C64
CONSTANTS.PY_TWO = 2
CONSTANTS.NP_TWO = np.int32(2)
CONSTANTS.PY_HALF = 0.5
CONSTANTS.NP_HALF = np.float32(0.5)
CONSTANTS.NP_ZERO = np.float32(0.0)


def test_np_bool_global_in_branch():
    @cuda.jit
    def k(out):
        if NP_TRUE:
            out[0] = 1
        if NP_FALSE:
            out[1] = 1

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 0])


def test_np_bool_global_assignment():
    @cuda.jit
    def k(out):
        flag = NP_TRUE
        out[0] = 1 if flag else 0
        other = NP_FALSE
        out[1] = 1 if other else 0

    out = cuda.to_device(np.zeros(2, dtype=np.int32))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 0])


def test_np_bool_closure_freevar():
    def make_kernel(flag):
        @cuda.jit
        def k(out):
            captured = flag
            out[0] = 1 if captured else 0

        return k

    out = cuda.to_device(np.zeros(1, dtype=np.int32))
    make_kernel(np.bool_(True))[1, 1](out)
    assert out.copy_to_host()[0] == 1
    make_kernel(np.bool_(False))[1, 1](out)
    assert out.copy_to_host()[0] == 0


def test_np_scalar_globals_arithmetic():
    @cuda.jit
    def k(out):
        out[0] = NP_F32 * 2.0
        out[1] = NP_I8 + 1

    out = cuda.to_device(np.zeros(2, dtype=np.float64))
    k[1, 1](out)
    np.testing.assert_allclose(out.copy_to_host(), [7.0, 8.0])


def test_np_scalar_module_attributes():
    @cuda.jit
    def k(out, cout):
        out[0] = np.uint64(5) + CONSTANTS.ONE
        out[1] = CONSTANTS.F32 * 2.0
        out[2] = CONSTANTS.I8 + 1
        out[3] = 1 if CONSTANTS.TRUE else 0
        cout[0] = CONSTANTS.C64 + 1

    out = cuda.to_device(np.zeros(4, dtype=np.float64))
    cout = cuda.to_device(np.zeros(1, dtype=np.complex128))
    k[1, 1](out, cout)
    np.testing.assert_allclose(out.copy_to_host(), [6.0, 7.0, 8.0, 1.0])
    np.testing.assert_array_equal(cout.copy_to_host(), [3.0 - 1.5j])


def test_numeric_module_attributes_as_conditions():
    @cuda.jit
    def k(out):
        out[0] = 1 if CONSTANTS.PY_TWO else 0
        out[1] = 1 if CONSTANTS.NP_TWO else 0
        out[2] = 1 if CONSTANTS.PY_HALF else 0
        out[3] = 1 if CONSTANTS.NP_HALF else 0
        out[4] = 1 if CONSTANTS.NP_ZERO else 0

    out = cuda.to_device(np.zeros(5, dtype=np.int32))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [1, 1, 1, 1, 0])


@pytest.mark.parametrize("dtype", [np.complex64, np.complex128], ids=["c64", "c128"])
def test_complex_global_cast(dtype):
    @cuda.jit
    def k(out):
        out[0] = dtype(PY_COMPLEX)

    out = cuda.to_device(np.zeros(1, dtype=dtype))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [PY_COMPLEX])


def test_complex_global_arithmetic():
    @cuda.jit
    def k(out):
        out[0] = PY_COMPLEX * 2
        out[1] = NP_C64 + 1

    out = cuda.to_device(np.zeros(2, dtype=np.complex128))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), [4.0 - 3.0j, 3.0 - 1.5j])


def test_complex_global_tuple():
    @cuda.jit
    def k(out):
        out[0] = np.complex128(COMPLEX_TUPLE[0])
        out[1] = COMPLEX_TUPLE[1]

    out = cuda.to_device(np.zeros(2, dtype=np.complex128))
    k[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), list(COMPLEX_TUPLE))


def test_complex_closure_tuple():
    def make_kernel(captured):
        @cuda.jit
        def k(out):
            out[0] = np.complex128(captured[0])
            out[1] = captured[1]

        return k

    out = cuda.to_device(np.zeros(2, dtype=np.complex128))
    make_kernel(COMPLEX_TUPLE)[1, 1](out)
    np.testing.assert_array_equal(out.copy_to_host(), list(COMPLEX_TUPLE))


def test_uint64_int_global_at_or_above_2_63():
    @cuda.jit
    def k(out):
        out[0] = np.uint64(3) * GOLDEN_GAMMA

    out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    k[1, 1](out)
    assert int(out.copy_to_host()[0]) == (3 * GOLDEN_GAMMA) % 2**64


def test_uint64_np_global_at_or_above_2_63():
    @cuda.jit
    def k(out):
        out[0] = NP_GOLDEN_GAMMA

    out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    k[1, 1](out)
    assert int(out.copy_to_host()[0]) == GOLDEN_GAMMA


def test_uint64_global_boundaries():
    @cuda.jit
    def k(out):
        out[0] = INT64_MAX
        out[1] = UINT64_MIN_UNSIGNED
        out[2] = ALL_ONES

    out = cuda.to_device(np.zeros(3, dtype=np.uint64))
    k[1, 1](out)
    assert [int(v) for v in out.copy_to_host()] == [INT64_MAX, UINT64_MIN_UNSIGNED, ALL_ONES]


def test_uint64_closure_freevar_at_or_above_2_63():
    def make_kernel(constant):
        @cuda.jit
        def k(out):
            out[0] = np.uint64(3) * constant

        return k

    out = cuda.to_device(np.zeros(1, dtype=np.uint64))
    make_kernel(GOLDEN_GAMMA)[1, 1](out)
    assert int(out.copy_to_host()[0]) == (3 * GOLDEN_GAMMA) % 2**64


def test_signless_int_only_rewrites_values_that_do_not_fit_int64():
    with ir.Context():
        i64 = ir.IntegerType.get_signless(64)
        i32 = ir.IntegerType.get_signless(32)
        assert signless_int(ALL_ONES, i64) == -1
        assert signless_int(GOLDEN_GAMMA, i64) == GOLDEN_GAMMA - 2**64
        assert signless_int(UINT64_MIN_UNSIGNED, i64) == -(2**63)
        # Values that IntegerAttr already accepts are left alone.
        assert signless_int(INT64_MAX, i64) == INT64_MAX
        assert signless_int(-1, i64) == -1
        assert signless_int(0xFFFFFFFF, i32) == 0xFFFFFFFF
        assert signless_int(True, ir.IntegerType.get_signless(1)) is True
