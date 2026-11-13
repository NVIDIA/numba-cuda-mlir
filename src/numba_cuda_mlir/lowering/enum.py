# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR lowering for enum operations."""

import operator

import numpy as np
from numba_cuda_mlir._mlir.dialects import arith

from numba_cuda_mlir import types
from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.lowering_utilities import (
    convert,
    get_conversion_signedness,
    numpy_implicit_type_promotion,
)
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type

registry = LoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
lower_cast = registry.lower_cast


def _is_float_enum(enum_type):
    return isinstance(enum_type.dtype, types.Float)


def _enum_cmp(builder, signature, args, int_pred, float_pred, different_type_result):
    """Compare two EnumMembers. Different types return constant result."""
    lhs, rhs = args
    lhs_type = signature.args[0]
    rhs_type = signature.args[1]
    if lhs_type == rhs_type:
        lhs_val = lhs
        rhs_val = rhs
        if _is_float_enum(lhs_type):
            if float_pred is None:
                raise ValueError(
                    f"float comparison predicate required for float-backed enum type {lhs_type}"
                )
            result = arith.cmpf(float_pred, lhs_val, rhs_val)
        else:
            result = arith.cmpi(int_pred, lhs_val, rhs_val)
    else:
        result = arith.constant(arith.IntegerType.get_signless(1), different_type_result)
    return result


@lower(operator.eq, types.EnumMember, types.EnumMember, api="values")
@lower(operator.is_, types.EnumMember, types.EnumMember, api="values")
def enum_eq(builder, signature, args):
    return _enum_cmp(builder, signature, args, arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ, 0)


@lower(operator.ne, types.EnumMember, types.EnumMember, api="values")
@lower(operator.is_not, types.EnumMember, types.EnumMember, api="values")
def enum_ne(builder, signature, args):
    return _enum_cmp(builder, signature, args, arith.CmpIPredicate.ne, arith.CmpFPredicate.UNE, 1)


@lower_getattr(types.EnumMember, "value")
@lower_getattr(types.IntEnumMember, "value")
def enum_value(context, builder, target, value):
    val = builder.load_var(value)
    builder.store_var(target, val)


@lower_cast(types.IntEnumMember, types.Integer)
def int_enum_to_int(builder, target, value):
    val = builder.load_var(value)
    target_type = builder.get_numba_type(target.name)
    result = convert(val, to_mlir_type(target_type))
    builder.store_var(target, result)


def _int_enum_value_cmp(builder, signature, args, predicate):
    """IntEnum comparison by value (not type)."""
    lhs, rhs = args
    lhs_val = convert(lhs, rhs.type)
    rhs_val = rhs
    result = arith.cmpi(predicate, lhs_val, rhs_val)
    return result


@lower(operator.eq, types.IntEnumMember, types.IntEnumMember, api="values")
def int_enum_eq(builder, signature, args):
    return _int_enum_value_cmp(builder, signature, args, arith.CmpIPredicate.eq)


@lower(operator.ne, types.IntEnumMember, types.IntEnumMember, api="values")
def int_enum_ne(builder, signature, args):
    return _int_enum_value_cmp(builder, signature, args, arith.CmpIPredicate.ne)


@lower(operator.is_, types.IntEnumMember, types.IntEnumMember, api="values")
def int_enum_is(builder, signature, args):
    return _enum_cmp(builder, signature, args, arith.CmpIPredicate.eq, None, 0)


@lower(operator.is_not, types.IntEnumMember, types.IntEnumMember, api="values")
def int_enum_is_not(builder, signature, args):
    return _enum_cmp(builder, signature, args, arith.CmpIPredicate.ne, None, 1)


@lower("static_getitem", types.EnumClass, types.StringLiteral, api="values")
@lower(operator.getitem, types.EnumClass, types.StringLiteral, api="values")
def enum_class_getitem(builder, signature, args):
    """Return an enum member by string key: Color["red"]."""
    enum_class_var, key_var = args
    enum_class_type = signature.args[0]
    key = signature.args[1].literal_value
    member = enum_class_type.instance_class[key]
    mlir_type = to_mlir_type(enum_class_type.dtype)
    return arith.constant(mlir_type, member.value)


def _int_enum_mixed_values(builder, signature, args):
    """Load and convert int/IntEnum operands to compatible types."""
    lhs, rhs = args
    lhs_val = lhs
    rhs_val = rhs
    lhs_type = signature.args[0]
    rhs_type = signature.args[1]
    if isinstance(lhs_type, types.IntEnumMember):
        lhs_val = convert(lhs_val, rhs_val.type)
    elif isinstance(rhs_type, types.IntEnumMember):
        rhs_val = convert(rhs_val, lhs_val.type)
    return lhs_val, rhs_val


def _int_enum_mixed_cmp(builder, signature, args, predicate):
    """Compare integer values without narrowing or confusing sign bits."""
    lhs, rhs = args
    lhs_type = signature.args[0]
    rhs_type = signature.args[1]
    lhs_val = lhs
    rhs_val = rhs
    common_type = numpy_implicit_type_promotion(lhs_val.type, rhs_val.type)
    lhs_signed = get_conversion_signedness(lhs_type, rhs_type)
    rhs_signed = get_conversion_signedness(rhs_type, lhs_type)
    lhs_val = convert(lhs_val, common_type, signed=lhs_signed)
    rhs_val = convert(rhs_val, common_type, signed=rhs_signed)
    result = arith.cmpi(predicate, lhs_val, rhs_val)
    if lhs_signed != rhs_signed:
        signed_val = lhs_val if lhs_signed else rhs_val
        # Equal-width signed/unsigned values can share bits but not values.
        # This also handles int64/uint64 without requiring wider GPU integers.
        zero = arith.constant(common_type, 0)
        if predicate == arith.CmpIPredicate.eq:
            nonnegative = arith.cmpi(arith.CmpIPredicate.sge, signed_val, zero)
            result = arith.andi(result, nonnegative)
        else:
            negative = arith.cmpi(arith.CmpIPredicate.slt, signed_val, zero)
            result = arith.ori(result, negative)
    return result


@lower(operator.eq, types.Integer, types.IntEnumMember, api="values")
@lower(operator.eq, types.IntEnumMember, types.Integer, api="values")
def int_enum_eq_mixed(builder, signature, args):
    return _int_enum_mixed_cmp(builder, signature, args, arith.CmpIPredicate.eq)


@lower(operator.ne, types.Integer, types.IntEnumMember, api="values")
@lower(operator.ne, types.IntEnumMember, types.Integer, api="values")
def int_enum_ne_mixed(builder, signature, args):
    return _int_enum_mixed_cmp(builder, signature, args, arith.CmpIPredicate.ne)


@lower(operator.gt, types.Integer, types.IntEnumMember, api="values")
@lower(operator.gt, types.IntEnumMember, types.Integer, api="values")
def int_enum_gt(builder, signature, args):
    lhs, rhs = _int_enum_mixed_values(builder, signature, args)
    return arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs)


@lower(operator.lt, types.Integer, types.IntEnumMember, api="values")
@lower(operator.lt, types.IntEnumMember, types.Integer, api="values")
def int_enum_lt(builder, signature, args):
    lhs, rhs = _int_enum_mixed_values(builder, signature, args)
    return arith.cmpi(arith.CmpIPredicate.slt, lhs, rhs)


@lower(operator.ge, types.Integer, types.IntEnumMember, api="values")
@lower(operator.ge, types.IntEnumMember, types.Integer, api="values")
def int_enum_ge(builder, signature, args):
    lhs, rhs = _int_enum_mixed_values(builder, signature, args)
    return arith.cmpi(arith.CmpIPredicate.sge, lhs, rhs)


@lower(operator.le, types.Integer, types.IntEnumMember, api="values")
@lower(operator.le, types.IntEnumMember, types.Integer, api="values")
def int_enum_le(builder, signature, args):
    lhs, rhs = _int_enum_mixed_values(builder, signature, args)
    return arith.cmpi(arith.CmpIPredicate.sle, lhs, rhs)


@lower(operator.add, types.Integer, types.IntEnumMember, api="values")
@lower(operator.add, types.IntEnumMember, types.Integer, api="values")
def int_enum_add(builder, signature, args):
    lhs, rhs = _int_enum_mixed_values(builder, signature, args)
    return arith.addi(lhs, rhs)


@lower(operator.sub, types.Integer, types.IntEnumMember, api="values")
@lower(operator.sub, types.IntEnumMember, types.Integer, api="values")
def int_enum_sub(builder, signature, args):
    lhs, rhs = _int_enum_mixed_values(builder, signature, args)
    return arith.subi(lhs, rhs)


def _enum_to_number(builder, signature, args):
    """Convert IntEnumMember or EnumMember to numeric type: int(x), float(x), etc."""
    val = args[0]
    target_type = signature.return_type
    return convert(val, to_mlir_type(target_type))


_numeric_types = [
    int,
    float,
    types.int8,
    types.int16,
    types.int32,
    types.int64,
    types.uint8,
    types.uint16,
    types.uint32,
    types.uint64,
    types.float32,
    types.float64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
]
for typ in _numeric_types:
    lower(typ, types.IntEnumMember, api="values")(_enum_to_number)
    lower(typ, types.EnumMember, api="values")(_enum_to_number)
