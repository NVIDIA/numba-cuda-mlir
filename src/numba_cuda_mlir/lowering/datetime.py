# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import operator

from numba_cuda_mlir._mlir.dialects import arith, math as mlir_math
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.np import npdatetime_helpers

from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
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


@lower(operator.sub, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.sub, types.NPDatetime, types.NPTimedelta, api="values")
@lower(operator.sub, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_sub(builder, signature, args):
    dest_unit = signature.return_type.unit
    lhs = _scale_to_unit(args[0], signature.args[0].unit, dest_unit)
    rhs = _scale_to_unit(args[1], signature.args[1].unit, dest_unit)
    return arith.subi(lhs, rhs)


@lower(operator.add, types.NPDatetime, types.NPTimedelta, api="values")
@lower(operator.add, types.NPTimedelta, types.NPDatetime, api="values")
@lower(operator.add, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_add(builder, signature, args):
    dest_unit = signature.return_type.unit
    lhs = _scale_to_unit(args[0], signature.args[0].unit, dest_unit)
    rhs = _scale_to_unit(args[1], signature.args[1].unit, dest_unit)
    return arith.addi(lhs, rhs)


def _datetime_cmp(predicate):
    def impl(builder, signature, args):
        lty = signature.args[0]
        rty = signature.args[1]
        common = npdatetime_helpers.get_best_unit(lty.unit, rty.unit)
        lhs = _scale_to_unit(args[0], signature.args[0].unit, common)
        rhs = _scale_to_unit(args[1], signature.args[1].unit, common)
        return arith.cmpi(predicate, lhs, rhs)

    return impl


@lower(operator.eq, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.eq, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_eq(builder, signature, args):
    return _datetime_cmp(arith.CmpIPredicate.eq)(builder, signature, args)


@lower(operator.ne, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.ne, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_ne(builder, signature, args):
    return _datetime_cmp(arith.CmpIPredicate.ne)(builder, signature, args)


@lower(operator.lt, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.lt, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_lt(builder, signature, args):
    return _datetime_cmp(arith.CmpIPredicate.slt)(builder, signature, args)


@lower(operator.le, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.le, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_le(builder, signature, args):
    return _datetime_cmp(arith.CmpIPredicate.sle)(builder, signature, args)


@lower(operator.gt, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.gt, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_gt(builder, signature, args):
    return _datetime_cmp(arith.CmpIPredicate.sgt)(builder, signature, args)


@lower(operator.ge, types.NPDatetime, types.NPDatetime, api="values")
@lower(operator.ge, types.NPTimedelta, types.NPTimedelta, api="values")
def datetime_ge(builder, signature, args):
    return _datetime_cmp(arith.CmpIPredicate.sge)(builder, signature, args)


# --- timedelta * {int, float} and {int, float} * timedelta ---


def _td_mul(td, other, other_type):
    if isinstance(other_type, types.Float):
        result = arith.mulf(convert(td, T.f64()), other)
        return convert(result, T.i64())
    return arith.muli(td, convert(other, T.i64()))


@lower(operator.mul, types.NPTimedelta, types.Integer, api="values")
@lower(operator.mul, types.NPTimedelta, types.Float, api="values")
def timedelta_mul(builder, signature, args):
    return _td_mul(args[0], args[1], signature.args[1])


@lower(operator.mul, types.Integer, types.NPTimedelta, api="values")
@lower(operator.mul, types.Float, types.NPTimedelta, api="values")
def timedelta_rmul(builder, signature, args):
    return _td_mul(args[1], args[0], signature.args[0])


# --- timedelta / {int, float} and timedelta // {int, float} ---


def _td_div_scalar(td, other, other_type):
    if isinstance(other_type, types.Float):
        result = arith.divf(convert(td, T.f64()), other)
        return convert(result, T.i64())
    return arith.divsi(td, convert(other, T.i64()))


@lower(operator.truediv, types.NPTimedelta, types.Integer, api="values")
@lower(operator.truediv, types.NPTimedelta, types.Float, api="values")
@lower(operator.floordiv, types.NPTimedelta, types.Integer, api="values")
@lower(operator.floordiv, types.NPTimedelta, types.Float, api="values")
def timedelta_div_scalar(builder, signature, args):
    return _td_div_scalar(args[0], args[1], signature.args[1])


# --- timedelta / timedelta -> float64 ---


@lower(operator.truediv, types.NPTimedelta, types.NPTimedelta, api="values")
def timedelta_div_timedelta(builder, signature, args):
    lty = signature.args[0]
    rty = signature.args[1]
    common = npdatetime_helpers.get_best_unit(lty.unit, rty.unit)
    lhs = convert(_scale_to_unit(args[0], signature.args[0].unit, common), T.f64())
    rhs = convert(_scale_to_unit(args[1], signature.args[1].unit, common), T.f64())
    return arith.divf(lhs, rhs)


@lower(operator.floordiv, types.NPTimedelta, types.NPTimedelta, api="values")
def timedelta_floordiv_timedelta(builder, signature, args):
    lty = signature.args[0]
    rty = signature.args[1]
    common = npdatetime_helpers.get_best_unit(lty.unit, rty.unit)
    lhs = convert(_scale_to_unit(args[0], signature.args[0].unit, common), T.f64())
    rhs = convert(_scale_to_unit(args[1], signature.args[1].unit, common), T.f64())
    quot = arith.divf(lhs, rhs)
    return convert(arith.floorf(quot), T.i64())


# --- unary: -timedelta, +timedelta, abs(timedelta) ---


@lower(operator.neg, types.NPTimedelta, api="values")
def timedelta_neg(builder, signature, args):
    val = args[0]
    zero = arith.constant(T.i64(), 0)
    return arith.subi(zero, val)


@lower(operator.pos, types.NPTimedelta, api="values")
def timedelta_pos(builder, signature, args):
    return args[0]


@lower(abs, types.NPTimedelta, api="values")
def timedelta_abs(builder, signature, args):
    return mlir_math.absi(args[0])
