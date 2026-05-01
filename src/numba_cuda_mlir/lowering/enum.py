# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLIR lowering for enum operations."""

import operator

import numpy as np
from numba_cuda_mlir._mlir.dialects import arith

from numba_cuda_mlir import types
from numba_cuda_mlir.mlir_lowering_registry import MLIRLoweringRegistry
from numba_cuda_mlir.lowering_utilities import convert
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type

registry = MLIRLoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
lower_cast = registry.lower_cast


def _is_float_enum(enum_type):
    return isinstance(enum_type.dtype, types.Float)


def _enum_cmp(builder, target, args, int_pred, float_pred, different_type_result):
    """Compare two EnumMembers. Different types return constant result."""
    lhs, rhs = args
    lhs_type = builder.get_numba_type(lhs.name)
    rhs_type = builder.get_numba_type(rhs.name)
    if lhs_type == rhs_type:
        lhs_val = builder.load_var(lhs)
        rhs_val = builder.load_var(rhs)
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
    builder.store_var(target, result)


@lower(operator.eq, types.EnumMember, types.EnumMember)
@lower(operator.is_, types.EnumMember, types.EnumMember)
def enum_eq(builder, target, args, kws):
    _enum_cmp(builder, target, args, arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ, 0)


@lower(operator.ne, types.EnumMember, types.EnumMember)
@lower(operator.is_not, types.EnumMember, types.EnumMember)
def enum_ne(builder, target, args, kws):
    _enum_cmp(builder, target, args, arith.CmpIPredicate.ne, arith.CmpFPredicate.UNE, 1)


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


def _int_enum_value_cmp(builder, target, args, predicate):
    """IntEnum comparison by value (not type)."""
    lhs, rhs = args
    lhs_val = convert(builder.load_var(lhs), builder.load_var(rhs).type)
    rhs_val = builder.load_var(rhs)
    result = arith.cmpi(predicate, lhs_val, rhs_val)
    builder.store_var(target, result)


@lower(operator.eq, types.IntEnumMember, types.IntEnumMember)
def int_enum_eq(builder, target, args, kws):
    _int_enum_value_cmp(builder, target, args, arith.CmpIPredicate.eq)


@lower(operator.ne, types.IntEnumMember, types.IntEnumMember)
def int_enum_ne(builder, target, args, kws):
    _int_enum_value_cmp(builder, target, args, arith.CmpIPredicate.ne)


@lower(operator.is_, types.IntEnumMember, types.IntEnumMember)
def int_enum_is(builder, target, args, kws):
    _enum_cmp(builder, target, args, arith.CmpIPredicate.eq, None, 0)


@lower(operator.is_not, types.IntEnumMember, types.IntEnumMember)
def int_enum_is_not(builder, target, args, kws):
    _enum_cmp(builder, target, args, arith.CmpIPredicate.ne, None, 1)


@lower("static_getitem", types.EnumClass, types.StringLiteral)
@lower(operator.getitem, types.EnumClass, types.StringLiteral)
def enum_class_getitem(builder, target, args, kwargs):
    """Return an enum member by string key: Color["red"]."""
    enum_class_var, key_var = args
    enum_class_type = builder.get_numba_type(enum_class_var.name)
    key = builder.get_numba_type(key_var.name).literal_value
    member = enum_class_type.instance_class[key]
    mlir_type = to_mlir_type(enum_class_type.dtype)
    builder.store_var(target, arith.constant(mlir_type, member.value))


def _int_enum_mixed_values(builder, args):
    """Load and convert int/IntEnum operands to compatible types."""
    lhs, rhs = args
    lhs_val = builder.load_var(lhs)
    rhs_val = builder.load_var(rhs)
    lhs_type = builder.get_numba_type(lhs.name)
    rhs_type = builder.get_numba_type(rhs.name)
    if isinstance(lhs_type, types.IntEnumMember):
        lhs_val = convert(lhs_val, rhs_val.type)
    elif isinstance(rhs_type, types.IntEnumMember):
        rhs_val = convert(rhs_val, lhs_val.type)
    return lhs_val, rhs_val


@lower(operator.gt, types.Integer, types.IntEnumMember)
@lower(operator.gt, types.IntEnumMember, types.Integer)
def int_enum_gt(builder, target, args, kws):
    lhs, rhs = _int_enum_mixed_values(builder, args)
    builder.store_var(target, arith.cmpi(arith.CmpIPredicate.sgt, lhs, rhs))


@lower(operator.lt, types.Integer, types.IntEnumMember)
@lower(operator.lt, types.IntEnumMember, types.Integer)
def int_enum_lt(builder, target, args, kws):
    lhs, rhs = _int_enum_mixed_values(builder, args)
    builder.store_var(target, arith.cmpi(arith.CmpIPredicate.slt, lhs, rhs))


@lower(operator.ge, types.Integer, types.IntEnumMember)
@lower(operator.ge, types.IntEnumMember, types.Integer)
def int_enum_ge(builder, target, args, kws):
    lhs, rhs = _int_enum_mixed_values(builder, args)
    builder.store_var(target, arith.cmpi(arith.CmpIPredicate.sge, lhs, rhs))


@lower(operator.le, types.Integer, types.IntEnumMember)
@lower(operator.le, types.IntEnumMember, types.Integer)
def int_enum_le(builder, target, args, kws):
    lhs, rhs = _int_enum_mixed_values(builder, args)
    builder.store_var(target, arith.cmpi(arith.CmpIPredicate.sle, lhs, rhs))


@lower(operator.add, types.Integer, types.IntEnumMember)
@lower(operator.add, types.IntEnumMember, types.Integer)
def int_enum_add(builder, target, args, kws):
    lhs, rhs = _int_enum_mixed_values(builder, args)
    builder.store_var(target, arith.addi(lhs, rhs))


@lower(operator.sub, types.Integer, types.IntEnumMember)
@lower(operator.sub, types.IntEnumMember, types.Integer)
def int_enum_sub(builder, target, args, kws):
    lhs, rhs = _int_enum_mixed_values(builder, args)
    builder.store_var(target, arith.subi(lhs, rhs))


def _enum_to_number(builder, target, args, kws):
    """Convert IntEnumMember or EnumMember to numeric type: int(x), float(x), etc."""
    val = builder.load_var(args[0])
    target_type = builder.get_numba_type(target.name)
    builder.store_var(target, convert(val, to_mlir_type(target_type)))


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
    lower(typ, types.IntEnumMember)(_enum_to_number)
    lower(typ, types.EnumMember)(_enum_to_number)
