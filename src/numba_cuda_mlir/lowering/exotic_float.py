# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering for exotic float types (fp8, fp4, fp6, tf32).

Arithmetic on sub-32-bit exotic floats must promote to f32, operate, then
truncate back because LLVM IR has no native fp8/fp4/fp6 types.
"""

import operator
from numba_cuda_mlir import types
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.types import (
    bfloat16_raw_type,
    cvt_e8m0_to_bf16raw,
    fp8_e4m3,
    fp8_e5m2,
    fp8_e8m0,
)
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, llvm
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.numba_cuda.types.ext_types import bfloat16 as bf16

registry = LoweringRegistry()
lower = registry.lower

_EXOTIC_FLOAT_TYPES = [
    types.f4E2M1FN,
    types.f6E2M3FN,
    types.f6E3M2FN,
    types.f8E3M4,
    types.f8E4M3B11FNUZ,
    types.f8E4M3FN,
    types.f8E4M3FNUZ,
    types.f8E4M3,
    types.f8E5M2FNUZ,
    types.f8E5M2,
    types.f8E8M0FNU,
    types.tf32,
]

_BINARY_ARITH = {
    operator.add: arith.addf,
    operator.sub: arith.subf,
    operator.mul: arith.mulf,
    operator.truediv: arith.divf,
    operator.iadd: arith.addf,
    operator.isub: arith.subf,
    operator.imul: arith.mulf,
    operator.itruediv: arith.divf,
}

_COMPARISONS = {
    operator.eq: arith.CmpFPredicate.OEQ,
    operator.ne: arith.CmpFPredicate.ONE,
    operator.lt: arith.CmpFPredicate.OLT,
    operator.le: arith.CmpFPredicate.OLE,
    operator.gt: arith.CmpFPredicate.OGT,
    operator.ge: arith.CmpFPredicate.OGE,
}

_FP8_VARIANTS = [
    (fp8_e5m2, T.f8E5M2),
    (fp8_e4m3, T.f8E4M3FN),
    (fp8_e8m0, T.f8E8M0FNU),
]

_FP8_CTOR_INPUT_TYPES = [
    types.float16,
    bf16,
    types.float32,
    types.float64,
    types.int8,
    types.int16,
    types.int32,
    types.int64,
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
]


def _get_mlir_type(numba_ty):
    return to_mlir_type(numba_ty)


def _make_binary(op, mlir_op, ty):
    @lower(op, ty, ty, api="values")
    def impl(builder, signature, args):
        lhs = args[0]
        rhs = args[1]
        target_mlir_ty = _get_mlir_type(ty)
        lhs_f32 = arith.extf(out=T.f32(), in_=lhs)
        rhs_f32 = arith.extf(out=T.f32(), in_=rhs)
        result_f32 = mlir_op(lhs_f32, rhs_f32)
        result = arith.truncf(out=target_mlir_ty, in_=result_f32)
        return result


def _make_cmp(op, predicate, ty):
    @lower(op, ty, ty, api="values")
    def impl(builder, signature, args):
        lhs = args[0]
        rhs = args[1]
        lhs_f32 = arith.extf(out=T.f32(), in_=lhs)
        rhs_f32 = arith.extf(out=T.f32(), in_=rhs)
        return arith.cmpf(predicate, lhs_f32, rhs_f32)


for _ty in _EXOTIC_FLOAT_TYPES:
    for _op, _mlir_op in _BINARY_ARITH.items():
        _make_binary(_op, _mlir_op, _ty)

    for _op, _pred in _COMPARISONS.items():
        _make_cmp(_op, _pred, _ty)

    @lower(operator.neg, _ty, api="values")
    def neg_impl(builder, signature, args):
        value = args[0]
        target_mlir_ty = _get_mlir_type(signature.return_type)
        val_f32 = arith.extf(out=T.f32(), in_=value)
        neg_f32 = arith.negf(val_f32)
        result = arith.truncf(out=target_mlir_ty, in_=neg_f32)
        return result

    @lower(operator.pos, _ty, api="values")
    def pos_impl(builder, signature, args):
        value = args[0]
        return value


for _py_type, _mlir_type_fn in _FP8_VARIANTS:
    for _input_ty in _FP8_CTOR_INPUT_TYPES:

        @lower(_py_type, _input_ty, api="values")
        def fp8_ctor(builder, signature, args):
            value = args[0]
            result = convert(value, _get_mlir_type(signature.return_type))
            return result


@lower(cvt_e8m0_to_bf16raw, types.uint8, api="values")
def cvt_e8m0_to_bf16raw_impl(builder, signature, args):
    val = args[0]
    i16_val = arith.extui(T.i16(), val)
    bf16_bits = arith.shli(i16_val, arith.constant(T.i16(), 7))
    target_ty = to_mlir_type(bfloat16_raw_type)
    desc = llvm.UndefOp(target_ty).result
    desc = llvm.insertvalue(container=desc, value=bf16_bits, position=ir.DenseI64ArrayAttr.get([0]))
    return desc


@registry.lower_getattr(bfloat16_raw_type, "x")
def bfloat16_raw_getattr_x(context, builder, target, value):
    struct_val = builder.load_var(value)
    x_val = llvm.extractvalue(T.i16(), struct_val, position=ir.DenseI64ArrayAttr.get([0]))
    builder.store_var(target, x_val)
