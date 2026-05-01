# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator

from numba_cuda_mlir._mlir.dialects import arith, math as mlir_math
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.np import npdatetime_helpers

from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower

# datetime64 and timedelta64 are i64 under the hood, so all ops are integer ops.
# When operand units differ we scale to the result unit before operating.


def _scale_to_unit(val, src_unit, dest_unit):
    """Multiply an i64 value by the conversion factor from src_unit to dest_unit."""
    if src_unit == dest_unit or src_unit == "" or dest_unit == "":
        return val
    factor = npdatetime_helpers.get_timedelta_conversion_factor(src_unit, dest_unit)
    if factor is None or factor == 1:
        return val
    return arith.muli(val, arith.constant(T.i64(), factor))


def _load_scaled(builder, var, dest_unit):
    """Load a datetime/timedelta variable and scale it to dest_unit."""
    val = builder.load_var(var)
    src_unit = builder.get_numba_type(var).unit
    return _scale_to_unit(val, src_unit, dest_unit)


@lower(operator.sub, types.NPDatetime, types.NPDatetime)
@lower(operator.sub, types.NPDatetime, types.NPTimedelta)
@lower(operator.sub, types.NPTimedelta, types.NPTimedelta)
def datetime_sub(builder, target, args, kwargs):
    dest_unit = builder.get_numba_type(target).unit
    lhs = _load_scaled(builder, args[0], dest_unit)
    rhs = _load_scaled(builder, args[1], dest_unit)
    builder.store_var(target, arith.subi(lhs, rhs))


@lower(operator.add, types.NPDatetime, types.NPTimedelta)
@lower(operator.add, types.NPTimedelta, types.NPDatetime)
@lower(operator.add, types.NPTimedelta, types.NPTimedelta)
def datetime_add(builder, target, args, kwargs):
    dest_unit = builder.get_numba_type(target).unit
    lhs = _load_scaled(builder, args[0], dest_unit)
    rhs = _load_scaled(builder, args[1], dest_unit)
    builder.store_var(target, arith.addi(lhs, rhs))


def _datetime_cmp(predicate):
    def impl(builder, target, args, kwargs):
        lty = builder.get_numba_type(args[0])
        rty = builder.get_numba_type(args[1])
        common = npdatetime_helpers.get_best_unit(lty.unit, rty.unit)
        lhs = _load_scaled(builder, args[0], common)
        rhs = _load_scaled(builder, args[1], common)
        builder.store_var(target, arith.cmpi(predicate, lhs, rhs))

    return impl


@lower(operator.eq, types.NPDatetime, types.NPDatetime)
@lower(operator.eq, types.NPTimedelta, types.NPTimedelta)
def datetime_eq(builder, target, args, kwargs):
    return _datetime_cmp(arith.CmpIPredicate.eq)(builder, target, args, kwargs)


@lower(operator.ne, types.NPDatetime, types.NPDatetime)
@lower(operator.ne, types.NPTimedelta, types.NPTimedelta)
def datetime_ne(builder, target, args, kwargs):
    return _datetime_cmp(arith.CmpIPredicate.ne)(builder, target, args, kwargs)


@lower(operator.lt, types.NPDatetime, types.NPDatetime)
@lower(operator.lt, types.NPTimedelta, types.NPTimedelta)
def datetime_lt(builder, target, args, kwargs):
    return _datetime_cmp(arith.CmpIPredicate.slt)(builder, target, args, kwargs)


@lower(operator.le, types.NPDatetime, types.NPDatetime)
@lower(operator.le, types.NPTimedelta, types.NPTimedelta)
def datetime_le(builder, target, args, kwargs):
    return _datetime_cmp(arith.CmpIPredicate.sle)(builder, target, args, kwargs)


@lower(operator.gt, types.NPDatetime, types.NPDatetime)
@lower(operator.gt, types.NPTimedelta, types.NPTimedelta)
def datetime_gt(builder, target, args, kwargs):
    return _datetime_cmp(arith.CmpIPredicate.sgt)(builder, target, args, kwargs)


@lower(operator.ge, types.NPDatetime, types.NPDatetime)
@lower(operator.ge, types.NPTimedelta, types.NPTimedelta)
def datetime_ge(builder, target, args, kwargs):
    return _datetime_cmp(arith.CmpIPredicate.sge)(builder, target, args, kwargs)


# --- timedelta * {int, float} and {int, float} * timedelta ---


def _td_mul(builder, target, td_var, other_var):
    td = builder.load_var(td_var)
    other = builder.load_var(other_var)
    other_ty = builder.get_numba_type(other_var)
    if isinstance(other_ty, types.Float):
        prod = arith.mulf(convert(td, T.f64()), other)
        builder.store_var(target, convert(prod, T.i64()))
    else:
        builder.store_var(target, arith.muli(td, convert(other, T.i64())))


@lower(operator.mul, types.NPTimedelta, types.Integer)
@lower(operator.mul, types.NPTimedelta, types.Float)
def timedelta_mul(builder, target, args, kwargs):
    _td_mul(builder, target, args[0], args[1])


@lower(operator.mul, types.Integer, types.NPTimedelta)
@lower(operator.mul, types.Float, types.NPTimedelta)
def timedelta_rmul(builder, target, args, kwargs):
    _td_mul(builder, target, args[1], args[0])


# --- timedelta / {int, float} and timedelta // {int, float} ---


def _td_div_scalar(builder, target, td_var, other_var):
    td = builder.load_var(td_var)
    other = builder.load_var(other_var)
    other_ty = builder.get_numba_type(other_var)
    if isinstance(other_ty, types.Float):
        quot = arith.divf(convert(td, T.f64()), other)
        builder.store_var(target, convert(quot, T.i64()))
    else:
        builder.store_var(target, arith.divsi(td, convert(other, T.i64())))


@lower(operator.truediv, types.NPTimedelta, types.Integer)
@lower(operator.truediv, types.NPTimedelta, types.Float)
@lower(operator.floordiv, types.NPTimedelta, types.Integer)
@lower(operator.floordiv, types.NPTimedelta, types.Float)
def timedelta_div_scalar(builder, target, args, kwargs):
    _td_div_scalar(builder, target, args[0], args[1])


# --- timedelta / timedelta -> float64 ---


@lower(operator.truediv, types.NPTimedelta, types.NPTimedelta)
def timedelta_div_timedelta(builder, target, args, kwargs):
    lty = builder.get_numba_type(args[0])
    rty = builder.get_numba_type(args[1])
    common = npdatetime_helpers.get_best_unit(lty.unit, rty.unit)
    lhs = convert(_load_scaled(builder, args[0], common), T.f64())
    rhs = convert(_load_scaled(builder, args[1], common), T.f64())
    builder.store_var(target, arith.divf(lhs, rhs))


@lower(operator.floordiv, types.NPTimedelta, types.NPTimedelta)
def timedelta_floordiv_timedelta(builder, target, args, kwargs):
    lty = builder.get_numba_type(args[0])
    rty = builder.get_numba_type(args[1])
    common = npdatetime_helpers.get_best_unit(lty.unit, rty.unit)
    lhs = convert(_load_scaled(builder, args[0], common), T.f64())
    rhs = convert(_load_scaled(builder, args[1], common), T.f64())
    quot = arith.divf(lhs, rhs)
    builder.store_var(target, convert(arith.floorf(quot), T.i64()))


# --- unary: -timedelta, +timedelta, abs(timedelta) ---


@lower(operator.neg, types.NPTimedelta)
def timedelta_neg(builder, target, args, kwargs):
    val = builder.load_var(args[0])
    zero = arith.constant(T.i64(), 0)
    builder.store_var(target, arith.subi(zero, val))


@lower(operator.pos, types.NPTimedelta)
def timedelta_pos(builder, target, args, kwargs):
    builder.store_var(target, builder.load_var(args[0]))


@lower(abs, types.NPTimedelta)
def timedelta_abs(builder, target, args, kwargs):
    builder.store_var(target, mlir_math.absi(builder.load_var(args[0])))
