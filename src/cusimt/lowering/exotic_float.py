# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lowering for exotic float types (fp8, fp4, fp6, tf32).

Arithmetic on sub-32-bit exotic floats must promote to f32, operate, then
truncate back because LLVM IR has no native fp8/fp4/fp6 types.
"""

import operator
from cusimt import types
from cusimt.lowering_utilities import convert
from cusimt.lowering_utilities.type_conversions import to_mlir_type
from cusimt.mlir_lowering_registry import MLIRLoweringRegistry
from cusimt.types import (
    bfloat16_raw_type,
    cvt_e8m0_to_bf16raw,
    fp8_e4m3,
    fp8_e5m2,
    fp8_e8m0,
)
from cusimt._mlir import ir
from cusimt._mlir.dialects import arith, llvm
from cusimt._mlir.extras import types as T
from cusimt.numba_cuda.types.ext_types import bfloat16 as bf16

registry = MLIRLoweringRegistry()
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
    @lower(op, ty, ty)
    def impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        target_mlir_ty = _get_mlir_type(ty)
        lhs_f32 = arith.extf(out=T.f32(), in_=lhs)
        rhs_f32 = arith.extf(out=T.f32(), in_=rhs)
        result_f32 = mlir_op(lhs_f32, rhs_f32)
        result = arith.truncf(out=target_mlir_ty, in_=result_f32)
        builder.store_var(target, result)


def _make_cmp(op, predicate, ty):
    @lower(op, ty, ty)
    def impl(builder, target, args, kwargs):
        lhs = builder.load_var(args[0])
        rhs = builder.load_var(args[1])
        lhs_f32 = arith.extf(out=T.f32(), in_=lhs)
        rhs_f32 = arith.extf(out=T.f32(), in_=rhs)
        builder.store_var(target, arith.cmpf(predicate, lhs_f32, rhs_f32))


for _ty in _EXOTIC_FLOAT_TYPES:
    for _op, _mlir_op in _BINARY_ARITH.items():
        _make_binary(_op, _mlir_op, _ty)

    for _op, _pred in _COMPARISONS.items():
        _make_cmp(_op, _pred, _ty)

    @lower(operator.neg, _ty)
    def neg_impl(builder, target, args, kwargs, _bound_ty=_ty):
        value = builder.load_var(args[0])
        target_mlir_ty = _get_mlir_type(_bound_ty)
        val_f32 = arith.extf(out=T.f32(), in_=value)
        neg_f32 = arith.negf(val_f32)
        result = arith.truncf(out=target_mlir_ty, in_=neg_f32)
        builder.store_var(target, result)

    @lower(operator.pos, _ty)
    def pos_impl(builder, target, args, kwargs):
        value = builder.load_var(args[0])
        builder.store_var(target, value)


for _py_type, _mlir_type_fn in _FP8_VARIANTS:
    for _input_ty in _FP8_CTOR_INPUT_TYPES:

        @lower(_py_type, _input_ty)
        def fp8_ctor(builder, target, args, kwargs, _fn=_mlir_type_fn):
            value = builder.load_var(args[0])
            result = convert(value, _fn())
            builder.store_var(target, result)


@lower(cvt_e8m0_to_bf16raw, types.uint8)
def cvt_e8m0_to_bf16raw_impl(builder, target, args, kwargs):
    val = builder.load_var(args[0])
    i16_val = arith.extui(T.i16(), val)
    bf16_bits = arith.shli(i16_val, arith.constant(T.i16(), 7))
    target_ty = to_mlir_type(bfloat16_raw_type)
    desc = llvm.UndefOp(target_ty).result
    desc = llvm.insertvalue(container=desc, value=bf16_bits, position=ir.DenseI64ArrayAttr.get([0]))
    builder.store_var(target, desc)


@registry.lower_getattr(bfloat16_raw_type, "x")
def bfloat16_raw_getattr_x(context, builder, target, value):
    struct_val = builder.load_var(value)
    x_val = llvm.extractvalue(T.i16(), struct_val, position=ir.DenseI64ArrayAttr.get([0]))
    builder.store_var(target, x_val)
