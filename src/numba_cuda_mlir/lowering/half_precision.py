# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering for half-precision (fp16/bf16) intrinsic functions."""

import operator
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
lower = registry.lower
from numba_cuda_mlir.numba_cuda.types.ext_types import bfloat16 as bf16
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import arith, math as math_dialect


def _lower_unary_intrinsic(func, mlir_op, numba_type):
    """Register lowering for a unary half-precision intrinsic."""

    @lower(func, numba_type)
    def impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = mlir_op(value)
        builder.store_var(target, result)


def _lower_binary_intrinsic(func, mlir_op, numba_type):
    """Register lowering for a binary half-precision intrinsic."""

    @lower(func, numba_type, numba_type)
    def impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = mlir_op(lhs, rhs)
        builder.store_var(target, result)


def _lower_ternary_intrinsic(func, mlir_op, numba_type):
    """Register lowering for a ternary half-precision intrinsic."""

    @lower(func, numba_type, numba_type, numba_type)
    def impl(builder, target, args, kwargs):
        a = builder.load_var(args[0])
        b = builder.load_var(args[1])
        c = builder.load_var(args[2])
        result = mlir_op(a, b, c)
        builder.store_var(target, result)


def register_bf16_lowering():
    """Register lowering for bfloat16 intrinsics from numba_cuda_mlir.numba_cuda._internal.cuda_bf16."""
    from numba_cuda_mlir.numba_cuda._internal.cuda_bf16 import (
        nv_bfloat16,
        htrunc,
        hceil,
        hfloor,
        hrint,
        hsqrt,
        hrsqrt,
        hrcp,
        hlog,
        hlog2,
        hlog10,
        hcos,
        hsin,
        hexp,
        hexp2,
        hexp10,
        htanh,
        htanh_approx,
        # Additional intrinsics
        __habs,
        __hneg,
        __hadd,
        __hsub,
        __hmul,
        __hdiv,
        __hadd_rn,
        __hsub_rn,
        __hmul_rn,
        __hadd_sat,
        __hsub_sat,
        __hmul_sat,
        __hfma,
        __hfma_sat,
        __hfma_relu,
        # Comparison intrinsics
        __heq,
        __hne,
        __hgt,
        __hlt,
        __hge,
        __hle,
        __hequ,
        __hneu,
        __hgtu,
        __hltu,
        __hgeu,
        __hleu,
        __hmax,
        __hmin,
        __hmax_nan,
        __hmin_nan,
        __hisinf,
        __hisnan,
        # Conversion intrinsics
        __float2bfloat16,
        __float2bfloat16_rn,
        __float2bfloat16_rz,
        __float2bfloat16_rd,
        __float2bfloat16_ru,
        __double2bfloat16,
        __bfloat162float,
        __short2bfloat16_rn,
        __short2bfloat16_rz,
        __short2bfloat16_rd,
        __short2bfloat16_ru,
        __bfloat162short_rn,
        __bfloat162short_rz,
        __bfloat162short_rd,
        __bfloat162short_ru,
        __ushort2bfloat16_rn,
        __ushort2bfloat16_rz,
        __ushort2bfloat16_rd,
        __ushort2bfloat16_ru,
        __bfloat162ushort_rn,
        __bfloat162ushort_rz,
        __bfloat162ushort_rd,
        __bfloat162ushort_ru,
        __int2bfloat16_rn,
        __int2bfloat16_rz,
        __int2bfloat16_rd,
        __int2bfloat16_ru,
        __bfloat162int_rn,
        __bfloat162int_rz,
        __bfloat162int_rd,
        __bfloat162int_ru,
        __uint2bfloat16_rn,
        __uint2bfloat16_rz,
        __uint2bfloat16_rd,
        __uint2bfloat16_ru,
        __bfloat162uint_rn,
        __bfloat162uint_rz,
        __bfloat162uint_rd,
        __bfloat162uint_ru,
        __ll2bfloat16_rn,
        __ll2bfloat16_rz,
        __ll2bfloat16_rd,
        __ll2bfloat16_ru,
        __bfloat162ll_rn,
        __bfloat162ll_rz,
        __bfloat162ll_rd,
        __bfloat162ll_ru,
        __ull2bfloat16_rn,
        __ull2bfloat16_rz,
        __ull2bfloat16_rd,
        __ull2bfloat16_ru,
        __bfloat162ull_rn,
        __bfloat162ull_rz,
        __bfloat162ull_rd,
        __bfloat162ull_ru,
        __bfloat162char_rz,
        __bfloat162uchar_rz,
        # Bitcast intrinsics
        __bfloat16_as_short,
        __bfloat16_as_ushort,
        __short_as_bfloat16,
        __ushort_as_bfloat16,
    )

    # Constructor: nv_bfloat16(float64)
    @lower(nv_bfloat16, types.float64)
    def bf16_ctor_f64(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(float32)
    @lower(nv_bfloat16, types.float32)
    def bf16_ctor_f32(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(float16)
    @lower(nv_bfloat16, types.float16)
    def bf16_ctor_f16(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(int16)
    @lower(nv_bfloat16, types.int16)
    def bf16_ctor_i16(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(int32)
    @lower(nv_bfloat16, types.int32)
    def bf16_ctor_i32(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(int64)
    @lower(nv_bfloat16, types.int64)
    def bf16_ctor_i64(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(uint16)
    @lower(nv_bfloat16, types.uint16)
    def bf16_ctor_u16(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(uint32)
    @lower(nv_bfloat16, types.uint32)
    def bf16_ctor_u32(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(uint64)
    @lower(nv_bfloat16, types.uint64)
    def bf16_ctor_u64(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(IntegerLiteral) - handles literal ints like 5
    @lower(nv_bfloat16, types.IntegerLiteral)
    def bf16_ctor_intlit(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Constructor: nv_bfloat16(Literal) - handles other literals
    @lower(nv_bfloat16, types.Literal)
    def bf16_ctor_lit(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = convert(value, T.bf16())
        builder.store_var(target, result)

    # Binary operators: bf16 + bf16, bf16 - bf16, etc.
    @lower(operator.add, bf16, bf16)
    def bf16_add(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.addf(lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.sub, bf16, bf16)
    def bf16_sub(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.subf(lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.mul, bf16, bf16)
    def bf16_mul(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.mulf(lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.truediv, bf16, bf16)
    def bf16_div(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.divf(lhs, rhs)
        builder.store_var(target, result)

    # Unary operators
    @lower(operator.neg, bf16)
    def bf16_neg(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.negf(value)
        builder.store_var(target, result)

    @lower(operator.pos, bf16)
    def bf16_pos(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        builder.store_var(target, value)

    # Comparison operators
    @lower(operator.eq, bf16, bf16)
    def bf16_eq(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OEQ, lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.ne, bf16, bf16)
    def bf16_ne(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.ONE, lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.lt, bf16, bf16)
    def bf16_lt(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OLT, lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.le, bf16, bf16)
    def bf16_le(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OLE, lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.gt, bf16, bf16)
    def bf16_gt(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OGT, lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.ge, bf16, bf16)
    def bf16_ge(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OGE, lhs, rhs)
        builder.store_var(target, result)

    # In-place operators
    @lower(operator.iadd, bf16, bf16)
    def bf16_iadd(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.addf(lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.isub, bf16, bf16)
    def bf16_isub(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.subf(lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.imul, bf16, bf16)
    def bf16_imul(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.mulf(lhs, rhs)
        builder.store_var(target, result)

    @lower(operator.itruediv, bf16, bf16)
    def bf16_itruediv(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.divf(lhs, rhs)
        builder.store_var(target, result)

    # Unary math operations
    _lower_unary_intrinsic(htrunc, math_dialect.trunc, bf16)
    _lower_unary_intrinsic(hceil, math_dialect.ceil, bf16)
    _lower_unary_intrinsic(hfloor, math_dialect.floor, bf16)
    _lower_unary_intrinsic(hrint, math_dialect.roundeven, bf16)
    _lower_unary_intrinsic(hsqrt, math_dialect.sqrt, bf16)
    _lower_unary_intrinsic(hrsqrt, math_dialect.rsqrt, bf16)
    _lower_unary_intrinsic(hlog, math_dialect.log, bf16)
    _lower_unary_intrinsic(hlog2, math_dialect.log2, bf16)
    _lower_unary_intrinsic(hlog10, math_dialect.log10, bf16)
    _lower_unary_intrinsic(hcos, math_dialect.cos, bf16)
    _lower_unary_intrinsic(hsin, math_dialect.sin, bf16)
    _lower_unary_intrinsic(hexp, math_dialect.exp, bf16)
    _lower_unary_intrinsic(hexp2, math_dialect.exp2, bf16)
    _lower_unary_intrinsic(htanh, math_dialect.tanh, bf16)
    _lower_unary_intrinsic(htanh_approx, math_dialect.tanh, bf16)

    # hrcp (reciprocal): 1/x
    @lower(hrcp, bf16)
    def hrcp_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        one = arith.constant(result=T.bf16(), value=1.0)
        result = arith.divf(one, value)
        builder.store_var(target, result)

    # hexp10: 10^x = exp(x * ln(10))
    @lower(hexp10, bf16)
    def hexp10_impl(builder, target, args, kwargs):
        import math

        value = builder.load_var(args[0])
        ln10 = arith.constant(result=T.bf16(), value=math.log(10))
        scaled = arith.mulf(value, ln10)
        result = math_dialect.exp(scaled)
        builder.store_var(target, result)

    # __habs: absolute value
    @lower(__habs, bf16)
    def habs_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = math_dialect.absf(value)
        builder.store_var(target, result)

    # __hneg: negation
    @lower(__hneg, bf16)
    def hneg_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.negf(value)
        builder.store_var(target, result)

    # Binary arithmetic intrinsics
    @lower(__hadd, bf16, bf16)
    def hadd_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.addf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hsub, bf16, bf16)
    def hsub_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.subf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hmul, bf16, bf16)
    def hmul_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.mulf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hdiv, bf16, bf16)
    def hdiv_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.divf(lhs, rhs)
        builder.store_var(target, result)

    # Round-to-nearest variants (same as regular for now)
    @lower(__hadd_rn, bf16, bf16)
    def hadd_rn_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.addf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hsub_rn, bf16, bf16)
    def hsub_rn_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.subf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hmul_rn, bf16, bf16)
    def hmul_rn_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.mulf(lhs, rhs)
        builder.store_var(target, result)

    # Saturating operations: clamp result to [0.0, 1.0]
    def saturate_bf16(value):
        zero = arith.constant(result=T.bf16(), value=0.0)
        one = arith.constant(result=T.bf16(), value=1.0)
        clamped_low = arith.maximumf(value, zero)
        return arith.minimumf(clamped_low, one)

    @lower(__hadd_sat, bf16, bf16)
    def hadd_sat_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        sum_val = arith.addf(lhs, rhs)
        result = saturate_bf16(sum_val)
        builder.store_var(target, result)

    @lower(__hsub_sat, bf16, bf16)
    def hsub_sat_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        diff = arith.subf(lhs, rhs)
        result = saturate_bf16(diff)
        builder.store_var(target, result)

    @lower(__hmul_sat, bf16, bf16)
    def hmul_sat_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        prod = arith.mulf(lhs, rhs)
        result = saturate_bf16(prod)
        builder.store_var(target, result)

    # FMA operations
    @lower(__hfma, bf16, bf16, bf16)
    def hfma_impl(builder, target, args, kwargs):
        a = builder.load_var(args[0])
        b = builder.load_var(args[1])
        c = builder.load_var(args[2])
        result = math_dialect.fma(a, b, c)
        builder.store_var(target, result)

    @lower(__hfma_sat, bf16, bf16, bf16)
    def hfma_sat_impl(builder, target, args, kwargs):
        a = builder.load_var(args[0])
        b = builder.load_var(args[1])
        c = builder.load_var(args[2])
        fma_result = math_dialect.fma(a, b, c)
        result = saturate_bf16(fma_result)
        builder.store_var(target, result)

    @lower(__hfma_relu, bf16, bf16, bf16)
    def hfma_relu_impl(builder, target, args, kwargs):
        a = builder.load_var(args[0])
        b = builder.load_var(args[1])
        c = builder.load_var(args[2])
        fma_result = math_dialect.fma(a, b, c)
        zero = arith.constant(result=T.bf16(), value=0.0)
        result = arith.maximumf(fma_result, zero)
        builder.store_var(target, result)

    # Comparison intrinsics (ordered)
    @lower(__heq, bf16, bf16)
    def heq_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OEQ, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hne, bf16, bf16)
    def hne_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.ONE, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hgt, bf16, bf16)
    def hgt_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OGT, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hlt, bf16, bf16)
    def hlt_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OLT, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hge, bf16, bf16)
    def hge_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OGE, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hle, bf16, bf16)
    def hle_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.OLE, lhs, rhs)
        builder.store_var(target, result)

    # Comparison intrinsics (unordered - returns true if either operand is NaN)
    @lower(__hequ, bf16, bf16)
    def hequ_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.UEQ, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hneu, bf16, bf16)
    def hneu_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.UNE, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hgtu, bf16, bf16)
    def hgtu_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.UGT, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hltu, bf16, bf16)
    def hltu_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.ULT, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hgeu, bf16, bf16)
    def hgeu_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.UGE, lhs, rhs)
        builder.store_var(target, result)

    @lower(__hleu, bf16, bf16)
    def hleu_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.cmpf(arith.CmpFPredicate.ULE, lhs, rhs)
        builder.store_var(target, result)

    # Min/max operations
    # CUDA __hmax returns the non-NaN operand (IEEE 754-2008 maxNum semantics)
    @lower(__hmax, bf16, bf16)
    def hmax_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.maxnumf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hmin, bf16, bf16)
    def hmin_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.minnumf(lhs, rhs)
        builder.store_var(target, result)

    # CUDA __hmax_nan propagates NaN (IEEE 754-2019 maximum semantics)
    @lower(__hmax_nan, bf16, bf16)
    def hmax_nan_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.maximumf(lhs, rhs)
        builder.store_var(target, result)

    @lower(__hmin_nan, bf16, bf16)
    def hmin_nan_impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        result = arith.minimumf(lhs, rhs)
        builder.store_var(target, result)

    # Special value checks
    @lower(__hisinf, bf16)
    def hisinf_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        abs_val = math_dialect.absf(value)
        inf = arith.constant(result=T.bf16(), value=float("inf"))
        result = arith.cmpf(arith.CmpFPredicate.OEQ, abs_val, inf)
        builder.store_var(target, result)

    @lower(__hisnan, bf16)
    def hisnan_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.cmpf(arith.CmpFPredicate.UNO, value, value)
        builder.store_var(target, result)

    # Conversion intrinsics: float -> bf16
    @lower(__float2bfloat16, types.float32)
    def float2bfloat16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.truncf(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__float2bfloat16_rn, types.float32)
    def float2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.truncf(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__float2bfloat16_rz, types.float32)
    def float2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.truncf(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__float2bfloat16_rd, types.float32)
    def float2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.truncf(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__float2bfloat16_ru, types.float32)
    def float2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.truncf(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__double2bfloat16, types.float64)
    def double2bfloat16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.truncf(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162float, bf16)
    def bfloat162float_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.extf(out=T.f32(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: short (int16) <-> bf16
    @lower(__short2bfloat16_rn, types.int16)
    def short2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__short2bfloat16_rz, types.int16)
    def short2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__short2bfloat16_rd, types.int16)
    def short2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__short2bfloat16_ru, types.int16)
    def short2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162short_rn, bf16)
    def bfloat162short_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.roundeven(value)
        result = arith.fptosi(out=T.i16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162short_rz, bf16)
    def bfloat162short_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptosi(out=T.i16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162short_rd, bf16)
    def bfloat162short_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.floor(value)
        result = arith.fptosi(out=T.i16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162short_ru, bf16)
    def bfloat162short_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.ceil(value)
        result = arith.fptosi(out=T.i16(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: ushort (uint16) <-> bf16
    @lower(__ushort2bfloat16_rn, types.uint16)
    def ushort2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ushort2bfloat16_rz, types.uint16)
    def ushort2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ushort2bfloat16_rd, types.uint16)
    def ushort2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ushort2bfloat16_ru, types.uint16)
    def ushort2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ushort_rn, bf16)
    def bfloat162ushort_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.roundeven(value)
        result = arith.fptoui(out=T.i16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ushort_rz, bf16)
    def bfloat162ushort_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptoui(out=T.i16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ushort_rd, bf16)
    def bfloat162ushort_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.floor(value)
        result = arith.fptoui(out=T.i16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ushort_ru, bf16)
    def bfloat162ushort_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.ceil(value)
        result = arith.fptoui(out=T.i16(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: int (int32) <-> bf16
    @lower(__int2bfloat16_rn, types.int32)
    def int2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__int2bfloat16_rz, types.int32)
    def int2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__int2bfloat16_rd, types.int32)
    def int2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__int2bfloat16_ru, types.int32)
    def int2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162int_rn, bf16)
    def bfloat162int_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.roundeven(value)
        result = arith.fptosi(out=T.i32(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162int_rz, bf16)
    def bfloat162int_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptosi(out=T.i32(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162int_rd, bf16)
    def bfloat162int_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.floor(value)
        result = arith.fptosi(out=T.i32(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162int_ru, bf16)
    def bfloat162int_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.ceil(value)
        result = arith.fptosi(out=T.i32(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: uint (uint32) <-> bf16
    @lower(__uint2bfloat16_rn, types.uint32)
    def uint2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__uint2bfloat16_rz, types.uint32)
    def uint2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__uint2bfloat16_rd, types.uint32)
    def uint2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__uint2bfloat16_ru, types.uint32)
    def uint2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162uint_rn, bf16)
    def bfloat162uint_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.roundeven(value)
        result = arith.fptoui(out=T.i32(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162uint_rz, bf16)
    def bfloat162uint_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptoui(out=T.i32(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162uint_rd, bf16)
    def bfloat162uint_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.floor(value)
        result = arith.fptoui(out=T.i32(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162uint_ru, bf16)
    def bfloat162uint_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.ceil(value)
        result = arith.fptoui(out=T.i32(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: ll (int64) <-> bf16
    @lower(__ll2bfloat16_rn, types.int64)
    def ll2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ll2bfloat16_rz, types.int64)
    def ll2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ll2bfloat16_rd, types.int64)
    def ll2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ll2bfloat16_ru, types.int64)
    def ll2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.sitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ll_rn, bf16)
    def bfloat162ll_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.roundeven(value)
        result = arith.fptosi(out=T.i64(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ll_rz, bf16)
    def bfloat162ll_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptosi(out=T.i64(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ll_rd, bf16)
    def bfloat162ll_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.floor(value)
        result = arith.fptosi(out=T.i64(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ll_ru, bf16)
    def bfloat162ll_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.ceil(value)
        result = arith.fptosi(out=T.i64(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: ull (uint64) <-> bf16
    @lower(__ull2bfloat16_rn, types.uint64)
    def ull2bfloat16_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ull2bfloat16_rz, types.uint64)
    def ull2bfloat16_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ull2bfloat16_rd, types.uint64)
    def ull2bfloat16_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__ull2bfloat16_ru, types.uint64)
    def ull2bfloat16_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.uitofp(out=T.bf16(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ull_rn, bf16)
    def bfloat162ull_rn_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.roundeven(value)
        result = arith.fptoui(out=T.i64(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ull_rz, bf16)
    def bfloat162ull_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptoui(out=T.i64(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ull_rd, bf16)
    def bfloat162ull_rd_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.floor(value)
        result = arith.fptoui(out=T.i64(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162ull_ru, bf16)
    def bfloat162ull_ru_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        value = math_dialect.ceil(value)
        result = arith.fptoui(out=T.i64(), in_=value)
        builder.store_var(target, result)

    # Conversion intrinsics: char/uchar
    @lower(__bfloat162char_rz, bf16)
    def bfloat162char_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptosi(out=T.i8(), in_=value)
        builder.store_var(target, result)

    @lower(__bfloat162uchar_rz, bf16)
    def bfloat162uchar_rz_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.fptoui(out=T.i8(), in_=value)
        builder.store_var(target, result)

    # Bitcast intrinsics: bf16 -> int16
    @lower(__bfloat16_as_short, bf16)
    def bfloat16_as_short_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.bitcast(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_ushort, bf16)
    def bfloat16_as_ushort_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.bitcast(T.i16(), value)
        builder.store_var(target, result)

    # Bitcast intrinsics: int16 -> bf16
    @lower(__short_as_bfloat16, types.int16)
    def short_as_bfloat16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.bitcast(T.bf16(), value)
        builder.store_var(target, result)

    @lower(__ushort_as_bfloat16, types.uint16)
    def ushort_as_bfloat16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.bitcast(T.bf16(), value)
        builder.store_var(target, result)

    # Bitcast intrinsics with int64/int32 args (for when scalars are typed as int64)
    # These truncate to i16 first, then treat that as the bf16 bit pattern
    @lower(__bfloat16_as_short, types.int64)
    def bfloat16_as_short_i64_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.trunci(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_short, types.int32)
    def bfloat16_as_short_i32_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.trunci(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_short, types.int16)
    def bfloat16_as_short_i16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        builder.store_var(target, value)

    @lower(__bfloat16_as_ushort, types.int64)
    def bfloat16_as_ushort_i64_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.trunci(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_ushort, types.int32)
    def bfloat16_as_ushort_i32_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.trunci(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_ushort, types.int16)
    def bfloat16_as_ushort_i16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        builder.store_var(target, value)

    @lower(__bfloat16_as_ushort, types.uint64)
    def bfloat16_as_ushort_u64_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.trunci(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_ushort, types.uint32)
    def bfloat16_as_ushort_u32_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        result = arith.trunci(T.i16(), value)
        builder.store_var(target, result)

    @lower(__bfloat16_as_ushort, types.uint16)
    def bfloat16_as_ushort_u16_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        builder.store_var(target, value)

    @lower(__short_as_bfloat16, types.int64)
    def short_as_bfloat16_i64_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)

    @lower(__short_as_bfloat16, types.int32)
    def short_as_bfloat16_i32_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)

    @lower(__ushort_as_bfloat16, types.int64)
    def ushort_as_bfloat16_i64_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)

    @lower(__ushort_as_bfloat16, types.int32)
    def ushort_as_bfloat16_i32_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)

    @lower(__ushort_as_bfloat16, types.uint64)
    def ushort_as_bfloat16_u64_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)

    @lower(__ushort_as_bfloat16, types.uint32)
    def ushort_as_bfloat16_u32_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)

    @lower(__short_as_bfloat16, types.IntegerLiteral)
    def short_as_bfloat16_lit_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        truncated = arith.trunci(T.i16(), value)
        result = arith.bitcast(T.bf16(), truncated)
        builder.store_var(target, result)


def register_fp16_lowering():
    """Register lowering for fp16 intrinsics from numba_cuda_mlir.cuda.fp16."""
    from numba_cuda_mlir.cuda import fp16

    # Unary operations missing from cuda.py
    _lower_unary_intrinsic(fp16.htrunc, math_dialect.trunc, types.float16)
    _lower_unary_intrinsic(fp16.hceil, math_dialect.ceil, types.float16)
    _lower_unary_intrinsic(fp16.hfloor, math_dialect.floor, types.float16)
    _lower_unary_intrinsic(fp16.hrint, math_dialect.roundeven, types.float16)
    _lower_unary_intrinsic(fp16.hsqrt, math_dialect.sqrt, types.float16)
    _lower_unary_intrinsic(fp16.hrsqrt, math_dialect.rsqrt, types.float16)
    _lower_unary_intrinsic(fp16.htanh, math_dialect.tanh, types.float16)
    _lower_unary_intrinsic(fp16.htanh_approx, math_dialect.tanh, types.float16)

    # hrcp (reciprocal): 1/x
    @lower(fp16.hrcp, types.float16)
    def fp16_hrcp_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        one = arith.constant(result=T.f16(), value=1.0)
        result = arith.divf(one, value)
        builder.store_var(target, result)

    # hexp10: 10^x = exp(x * ln(10))
    @lower(fp16.hexp10, types.float16)
    def fp16_hexp10_impl(builder, target, args, kwargs):
        import math

        value = builder.load_var(args[0])
        ln10 = arith.constant(result=T.f16(), value=math.log(10))
        scaled = arith.mulf(value, ln10)
        result = math_dialect.exp(scaled)
        builder.store_var(target, result)


register_bf16_lowering()
register_fp16_lowering()
