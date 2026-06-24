# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import itertools
import operator

import numpy as np
import numba_cuda_mlir
import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype
from numba_cuda_mlir.numba_cuda.types import f2, i2, u2, b1
from numba_cuda_mlir.numba_cuda.typing import signature
from numba_cuda_mlir.testing import NumbaCUDATestCase


def simple_fp16_div_scalar(ary, a, b):
    ary[0] = a / b


def simple_fp16add(ary, a, b):
    ary[0] = a + b


def simple_fp16_iadd(ary, a):
    ary[0] += a


def simple_fp16_isub(ary, a):
    ary[0] -= a


def simple_fp16_imul(ary, a):
    ary[0] *= a


def simple_fp16_idiv(ary, a):
    ary[0] /= a


def simple_fp16sub(ary, a, b):
    ary[0] = a - b


def simple_fp16mul(ary, a, b):
    ary[0] = a * b


def simple_fp16neg(ary, a):
    ary[0] = -a


def simple_fp16abs(ary, a):
    ary[0] = abs(a)


def simple_fp16_gt(ary, a, b):
    ary[0] = a > b


def simple_fp16_ge(ary, a, b):
    ary[0] = a >= b


def simple_fp16_lt(ary, a, b):
    ary[0] = a < b


def simple_fp16_le(ary, a, b):
    ary[0] = a <= b


def simple_fp16_eq(ary, a, b):
    ary[0] = a == b


def simple_fp16_ne(ary, a, b):
    ary[0] = a != b


@cuda.jit("b1(f2, f2)", device=True)
def hlt_func_1(x, y):
    return x < y


@cuda.jit("b1(f2, f2)", device=True)
def hlt_func_2(x, y):
    return x < y


def multiple_hcmp_1(r, a, b, c):
    # float16 predicates used in two separate functions
    r[0] = hlt_func_1(a, b) and hlt_func_2(b, c)


def multiple_hcmp_2(r, a, b, c):
    # The same float16 predicate used in the caller and callee
    r[0] = hlt_func_1(a, b) and b < c


def multiple_hcmp_3(r, a, b, c):
    # Different float16 predicates used in the caller and callee
    r[0] = hlt_func_1(a, b) and c >= b


def multiple_hcmp_4(r, a, b, c):
    # The same float16 predicates used twice in a function
    r[0] = a < b and b < c


def multiple_hcmp_5(r, a, b, c):
    # Different float16 predicates used in a function
    r[0] = a < b and c >= b


class TestOperatorModule:
    @pytest.fixture(autouse=True)
    def setUp(self):
        np.random.seed(0)

    """
    Test if operator module is supported by the CUDA target.
    """

    def operator_template(self, op):
        @cuda.jit
        def foo(a, b):
            i = 0
            a[i] = op(a[i], b[i])

        a = np.ones(1)
        b = np.ones(1)
        res = a.copy()
        foo[1, 1](res, b)

        np.testing.assert_equal(res, op(a, b))

    def test_add(self):
        self.operator_template(operator.add)

    def test_sub(self):
        self.operator_template(operator.sub)

    def test_mul(self):
        self.operator_template(operator.mul)

    def test_truediv(self):
        self.operator_template(operator.truediv)

    def test_floordiv(self):
        self.operator_template(operator.floordiv)

    def test_fp16_binary(self):
        functions = (
            simple_fp16add,
            simple_fp16sub,
            simple_fp16mul,
            simple_fp16_div_scalar,
        )
        ops = (operator.add, operator.sub, operator.mul, operator.truediv)

        for fn, op in zip(functions, ops):
            kernel = cuda.jit("void(f2[:], f2, f2)")(fn)

            got = np.zeros(1, dtype=np.float16)
            arg1 = np.random.random(1).astype(np.float16)
            arg2 = np.random.random(1).astype(np.float16)

            kernel[1, 1](got, arg1[0], arg2[0])
            expected = op(arg1, arg2)
            np.testing.assert_allclose(got, expected)

    def test_fp16_binary_ptx(self):
        functions = (simple_fp16add, simple_fp16sub, simple_fp16mul)
        instrs = ("add.f16", "sub.f16", "mul.f16")
        args = (f2[:], f2, f2)
        for fn, instr in zip(functions, instrs):
            compiled = cuda.jit("void(f2[:], f2, f2)", lto=True)(fn)
            ptx = compiled.inspect_lto_ptx(args)
            assert instr in ptx

    def test_mixed_fp16_binary_arithmetic(self):
        functions = (
            simple_fp16add,
            simple_fp16sub,
            simple_fp16mul,
            simple_fp16_div_scalar,
        )
        ops = (operator.add, operator.sub, operator.mul, operator.truediv)
        types = (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)
        for (fn, op), ty in itertools.product(zip(functions, ops), types):
            kernel = cuda.jit(fn, lto=True)

            arg1 = np.random.random(1).astype(np.float16)
            arg2 = (np.random.random(1) * 100).astype(ty)
            res_ty = np.result_type(np.float16, ty)

            got = np.zeros(1, dtype=res_ty)
            kernel[1, 1](got, arg1[0], arg2[0])
            expected = op(arg1, arg2)
            np.testing.assert_allclose(got, expected)

    def test_fp16_inplace_binary_ptx(self):
        functions = (simple_fp16_iadd, simple_fp16_isub, simple_fp16_imul)
        instrs = ("add.f16", "sub.f16", "mul.f16")
        args = (f2[:], f2)

        for fn, instr in zip(functions, instrs):
            compiled = cuda.jit("void(f2[:], f2)", lto=True)(fn)
            ptx = compiled.inspect_lto_ptx(args)
            assert instr in ptx

    def test_fp16_inplace_binary(self):
        functions = (
            simple_fp16_iadd,
            simple_fp16_isub,
            simple_fp16_imul,
            simple_fp16_idiv,
        )
        ops = (operator.iadd, operator.isub, operator.imul, operator.itruediv)

        for fn, op in zip(functions, ops):
            kernel = cuda.jit("void(f2[:], f2)")(fn)

            got = np.random.random(1).astype(np.float16)
            expected = got.copy()
            arg = np.random.random(1).astype(np.float16)[0]
            kernel[1, 1](got, arg)
            op(expected, arg)
            np.testing.assert_allclose(got, expected)

    def test_fp16_unary(self):
        functions = (simple_fp16neg, simple_fp16abs)
        ops = (operator.neg, operator.abs)

        for fn, op in zip(functions, ops):
            kernel = cuda.jit("void(f2[:], f2)")(fn)

            got = np.zeros(1, dtype=np.float16)
            arg1 = np.random.random(1).astype(np.float16)

            kernel[1, 1](got, arg1[0])
            expected = op(arg1)
            np.testing.assert_allclose(got, expected)

    @pytest.mark.xfail(True, reason="NVVM verify error")
    def test_fp16_neg_ptx(self):
        args = (f2[:], f2)
        compiled = cuda.jit("void(f2[:], f2)", lto=True)(simple_fp16neg)
        ptx = compiled.inspect_lto_ptx(args)
        assert "neg.f16" in ptx

    @pytest.mark.xfail(True, reason="Bitcode parsing error")
    def test_fp16_abs_ptx(self):
        args = (f2[:], f2)
        compiled = cuda.jit("void(f2[:], f2)", lto=True)(simple_fp16abs)
        ptx = compiled.inspect_lto_ptx(args)
        assert "abs.f16" in ptx

    @pytest.mark.parametrize(
        "func,op",
        [
            pytest.param(simple_fp16_gt, operator.gt, id="gt"),
            pytest.param(simple_fp16_ge, operator.ge, id="ge"),
            pytest.param(simple_fp16_lt, operator.lt, id="lt"),
            pytest.param(simple_fp16_le, operator.le, id="le"),
            pytest.param(simple_fp16_eq, operator.eq, id="eq"),
            pytest.param(simple_fp16_ne, operator.ne, id="ne"),
        ],
    )
    def test_int16_comparison(self, func, op):
        kernel = cuda.jit(f"void(b1[:], i2, i2)")(func)

        got = np.zeros(1, dtype=np.bool_)
        arg1 = np.array([-50], dtype=np.int16)
        arg2 = np.array([0], dtype=np.int16)

        kernel[1, 1](got, arg1[0], arg2[0])
        expected = op(arg1, arg2)
        assert got[0] == expected

    @pytest.mark.parametrize(
        "func,op",
        [
            pytest.param(simple_fp16_gt, operator.gt, id="gt"),
            pytest.param(simple_fp16_ge, operator.ge, id="ge"),
            pytest.param(simple_fp16_lt, operator.lt, id="lt"),
            pytest.param(simple_fp16_le, operator.le, id="le"),
            pytest.param(simple_fp16_eq, operator.eq, id="eq"),
            pytest.param(simple_fp16_ne, operator.ne, id="ne"),
        ],
    )
    def test_uint16_comparison(self, func, op):
        kernel = cuda.jit("void(b1[:], u2, u2)")(func)

        got = np.zeros(1, dtype=np.bool_)
        arg1 = np.array([40000], dtype=np.uint16)
        arg2 = np.array([5], dtype=np.uint16)

        kernel[1, 1](got, arg1[0], arg2[0])
        expected = op(arg1, arg2)
        assert got[0] == expected

    def test_fp16_comparison(self):
        functions = (
            simple_fp16_gt,
            simple_fp16_ge,
            simple_fp16_lt,
            simple_fp16_le,
            simple_fp16_eq,
            simple_fp16_ne,
        )
        ops = (
            operator.gt,
            operator.ge,
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
        )

        for fn, op in zip(functions, ops):
            kernel = cuda.jit("void(b1[:], f2, f2)")(fn)

            got = np.zeros(1, dtype=np.bool_)
            arg1 = np.random.random(1).astype(np.float16)
            arg2 = np.random.random(1).astype(np.float16)

            kernel[1, 1](got, arg1[0], arg2[0])
            expected = op(arg1, arg2)
            assert got[0] == expected

    def test_mixed_fp16_comparison(self):
        functions = (
            simple_fp16_gt,
            simple_fp16_ge,
            simple_fp16_lt,
            simple_fp16_le,
            simple_fp16_eq,
            simple_fp16_ne,
        )
        ops = (
            operator.gt,
            operator.ge,
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
        )
        types = (np.int8, np.int16, np.int32, np.int64, np.float32, np.float64)

        for (fn, op), ty in itertools.product(zip(functions, ops), types):
            kernel = cuda.jit(fn)

            got = np.zeros(1, dtype=np.bool_)
            arg1 = np.random.random(1).astype(np.float16)
            arg2 = (np.random.random(1) * 100).astype(ty)

            kernel[1, 1](got, arg1[0], arg2[0])
            expected = op(arg1, arg2)
            assert got[0] == expected

    def test_multiple_float16_comparisons(self):
        functions = (
            multiple_hcmp_1,
            multiple_hcmp_2,
            multiple_hcmp_3,
            multiple_hcmp_4,
            multiple_hcmp_5,
        )
        for fn in functions:
            compiled = cuda.jit("void(b1[:], f2, f2, f2)")(fn)
            ary = np.zeros(1, dtype=np.bool_)
            arg1 = np.float16(2.0)
            arg2 = np.float16(3.0)
            arg3 = np.float16(4.0)
            compiled[1, 1](ary, arg1, arg2, arg3)
            assert ary[0] == True

    def test_multiple_float16_comparisons_false(self):
        functions = (
            multiple_hcmp_1,
            multiple_hcmp_2,
            multiple_hcmp_3,
            multiple_hcmp_4,
            multiple_hcmp_5,
        )
        for fn in functions:
            compiled = cuda.jit("void(b1[:], f2, f2, f2)")(fn)
            ary = np.zeros(1, dtype=np.bool_)
            arg1 = np.float16(2.0)
            arg2 = np.float16(3.0)
            arg3 = np.float16(1.0)
            compiled[1, 1](ary, arg1, arg2, arg3)
            assert ary[0] == False

    @pytest.mark.parametrize(
        "func,opstring",
        [
            pytest.param(simple_fp16_gt, "setp.gt.s16", id="gt"),
            pytest.param(simple_fp16_ge, "setp.ge.s16", id="ge"),
            pytest.param(simple_fp16_lt, "setp.lt.s16", id="lt"),
            pytest.param(simple_fp16_le, "setp.le.s16", id="le"),
            pytest.param(simple_fp16_eq, "setp.eq.b16", id="eq"),
            pytest.param(simple_fp16_ne, "setp.ne.b16", id="ne"),
        ],
    )
    def test_int16_comparison_ptx(self, func, opstring):
        args = (b1[:], i2, i2)
        compiled = cuda.jit("void(b1[:], i2, i2)", lto=True)(func)
        ptx = compiled.inspect_lto_ptx(args)
        assert opstring in ptx, f"{opstring} not in PTX"

    @pytest.mark.parametrize(
        "func,opstring",
        [
            pytest.param(simple_fp16_gt, "setp.gt.u16", id="gt"),
            pytest.param(simple_fp16_ge, "setp.ge.u16", id="ge"),
            pytest.param(simple_fp16_lt, "setp.lt.u16", id="lt"),
            pytest.param(simple_fp16_le, "setp.le.u16", id="le"),
            pytest.param(simple_fp16_eq, "setp.eq.b16", id="eq"),
            pytest.param(simple_fp16_ne, "setp.ne.b16", id="ne"),
        ],
    )
    def test_uint16_comparison_ptx(self, func, opstring):
        args = (b1[:], u2, u2)
        compiled = cuda.jit("void(b1[:], u2, u2)", lto=True)(func)
        ptx = compiled.inspect_lto_ptx(args)
        assert opstring in ptx, f"{opstring} not in PTX"

    @pytest.mark.xfail(True, reason="NVVM verify error")
    def test_fp16_comparison_ptx(self):
        functions = (
            simple_fp16_gt,
            simple_fp16_ge,
            simple_fp16_lt,
            simple_fp16_le,
            simple_fp16_eq,
            simple_fp16_ne,
        )
        ops = (
            operator.gt,
            operator.ge,
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
        )
        opstring = (
            "setp.gt.f16",
            "setp.ge.f16",
            "setp.lt.f16",
            "setp.le.f16",
            "setp.eq.f16",
            "setp.neu.f16",
        )
        args = (b1[:], f2, f2)

        for fn, op, s in zip(functions, ops, opstring):
            compiled = cuda.jit("void(b1[:], f2, f2)", lto=True)(fn)
            ptx = compiled.inspect_lto_ptx(args)
            assert s in ptx

    @pytest.mark.xfail(True, reason="NVVM verify error")
    def test_fp16_int8_comparison_ptx(self):
        # Test that int8 can be safely converted to fp16
        # in a comparison
        functions = (
            simple_fp16_gt,
            simple_fp16_ge,
            simple_fp16_lt,
            simple_fp16_le,
            simple_fp16_eq,
            simple_fp16_ne,
        )
        ops = (
            operator.gt,
            operator.ge,
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
        )

        opstring = {
            operator.gt: "setp.gt.f16",
            operator.ge: "setp.ge.f16",
            operator.lt: "setp.lt.f16",
            operator.le: "setp.le.f16",
            operator.eq: "setp.eq.f16",
            operator.ne: "setp.neu.f16",
        }
        for fn, op in zip(functions, ops):
            args = (b1[:], f2, from_dtype(np.int8))
            compiled = cuda.jit(signature(types.void, *args), lto=True)(fn)
            ptx = compiled.inspect_lto_ptx(args)
            assert opstring[op] in ptx

    @pytest.mark.xfail(True, reason="NVVM verify error")
    def test_mixed_fp16_comparison_promotion_ptx(self):
        functions = (
            simple_fp16_gt,
            simple_fp16_ge,
            simple_fp16_lt,
            simple_fp16_le,
            simple_fp16_eq,
            simple_fp16_ne,
        )
        ops = (
            operator.gt,
            operator.ge,
            operator.lt,
            operator.le,
            operator.eq,
            operator.ne,
        )

        types_promote = (np.int16, np.int32, np.int64, np.float32, np.float64)
        opstring = {
            operator.gt: "setp.gt.",
            operator.ge: "setp.ge.",
            operator.lt: "setp.lt.",
            operator.le: "setp.le.",
            operator.eq: "setp.eq.",
            operator.ne: "setp.neu.",
        }
        opsuffix = {
            np.dtype("int32"): "f64",
            np.dtype("int64"): "f64",
            np.dtype("float32"): "f32",
            np.dtype("float64"): "f64",
        }

        for (fn, op), ty in itertools.product(zip(functions, ops), types_promote):
            arg2_ty = np.result_type(np.float16, ty)
            args = (b1[:], f2, from_dtype(arg2_ty))
            compiled = cuda.jit(signature(types.void, *args), lto=True)(fn)
            ptx = compiled.inspect_lto_ptx(args)

            ops = opstring[op] + opsuffix[arg2_ty]
            assert ops in ptx
