# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from cusimt.errors import InternalCompilerError
import math as builtin_math
import operator
from cusimt.lowering_utilities import (
    DeferredMethodCall,
    type_conversions,
    convert,
    constant,
    numpy_implicit_type_promotion,
    memref_to_tensor,
    broadcast_shapes_for_binary_op,
    index_of,
    tensor_to_memref,
    simple_scalar_conversion_op,
    try_extract_constant,
)
from cusimt.lowering_utilities.linalg_lowering import lower_np_binop
from cusimt.logging import trace
from cusimt.mlir_lowering import MLIRLower
from cusimt import types
from numba.core.errors import TypingError
from numba.extending import intrinsic
from cusimt._mlir.extras import types as T
from cusimt._mlir.dialects import (
    linalg,
    tensor,
    arith,
    shape,
    math,
    nvvm,
    complex as complex_dialect,
)
from cusimt.mlir.dialect_exts.math import ipowi
from cusimt.mlir_lowering_registry import MLIRLoweringRegistry

registry = MLIRLoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
from cusimt.lowering_utilities import (
    RangeObject,
    int_of,
)
from cusimt._mlir import ir
from .ufunc_registry import UFuncRegistry

ufunc_registry = UFuncRegistry("builtins")


def _get_range_object(builder, args: list[ir.Value]) -> tuple[ir.Value, ir.Value, ir.Value]:
    match args:
        case [stop]:
            stop_type = builder.get_mlir_type(stop)
            return (
                int_of(0, ty=stop_type),
                int_of(builder.load_var(stop), ty=stop_type, signed=True),
                int_of(1, ty=stop_type),
            )
        case [start, stop]:
            start_type = builder.get_mlir_type(start)
            stop_type = builder.get_mlir_type(stop)
            return (
                int_of(builder.load_var(start), ty=start_type, signed=True),
                int_of(builder.load_var(stop), ty=stop_type, signed=True),
                int_of(1, ty=stop_type),
            )
        case [start, stop, step]:
            return (
                int_of(
                    builder.load_var(start),
                    ty=builder.get_mlir_type(start),
                    signed=True,
                ),
                int_of(builder.load_var(stop), ty=builder.get_mlir_type(stop), signed=True),
                int_of(builder.load_var(step), ty=builder.get_mlir_type(step), signed=True),
            )
        case _:
            raise ValueError(f"Invalid arguments for range: {args}")


@lower(operator.mul, types.UniTuple, types.Literal)
@lower(operator.mul, types.Tuple, types.Literal)
@lower(operator.mul, types.Literal, types.UniTuple)
def lower_tuple_multiply(builder, target, args, kwargs):
    trace()
    assert not kwargs and len(args) == 2, "tuple multiply expects 2 arguments"
    tup, how_many = builder.load_vars(args)
    assert isinstance(tup, tuple), f"Expected tuple, got {type(tup)}"
    match tup, how_many:
        case tuple(), (int() | ir.Value()):
            pass
        case ((int() | ir.Value()), tuple()):
            tup, how_many = how_many, tup
        case _:
            raise TypeError(f"Invalid arguments for tuple multiply: {tup=} {how_many=}")

    if how_many := try_extract_constant(how_many):
        builder.store_var(target, tup * how_many)
        return

    raise InternalCompilerError(f"Expected literal integer, got {type(how_many)}")


@lower(tuple, types.UniTuple)
@lower(tuple, types.Tuple)
def lower_tuple_builtin(builder, target, args, kwargs):
    trace()
    assert not kwargs and len(args) == 1, "tuple() expects 1 argument"
    value = builder.load_var(args[0])
    if not isinstance(value, tuple):
        value = (value,)
    builder.store_var(target, value)


@lower(operator.add, types.UniTuple, types.UniTuple)
@lower(operator.add, types.Tuple, types.Tuple)
@lower(operator.add, types.UniTuple, types.Tuple)
@lower(operator.add, types.Tuple, types.UniTuple)
def lower_tuple_concat(builder, target, args, kwargs):
    trace()
    assert not kwargs and len(args) == 2, "tuple concat expects 2 arguments"
    lhs, rhs = builder.load_vars(args)
    if not isinstance(lhs, tuple):
        lhs = (lhs,)
    if not isinstance(rhs, tuple):
        rhs = (rhs,)
    builder.store_var(target, lhs + rhs)


@lower(range, types.Number)
@lower(range, types.Number, types.Number)
@lower(range, types.Number, types.Number, types.Number)
def lower_range(builder: MLIRLower, target, args, kwargs):
    start, stop, step = _get_range_object(builder, args)
    ro = RangeObject(builder, start, stop, step)
    builder.store_var(target, ro)


@lower(operator.contains, types.Tuple, types.Number)
def tuple_contains_cg(builder, target, args, kwargs):
    tup = builder.load_var(args[0])
    item = builder.load_var(args[1])
    from cusimt.lowering_utilities import bool_of

    # Extract constant values from the tuple elements
    constant_values = []
    for x in tup:
        const_val = try_extract_constant(x)
        if const_val is None:
            raise NotImplementedError(
                f"Tuple contains is not implemented for non-constant values: {x}"
            )
        constant_values.append(const_val)

    # Check if item is a constant and can be checked at compile time
    item_const = try_extract_constant(item)
    if item_const is not None:
        builder.store_var(target, bool_of(item_const in constant_values))
    else:
        # Runtime check - item is not a constant
        from cusimt.lowering_utilities import false, equal

        result = false()
        for const_val in constant_values:
            result = arith.ori(result, equal(item, int_of(const_val, item.type)))
        builder.store_var(target, result)


@lower(operator.contains, types.UniTuple, types.Number)
def unituple_contains_cg(builder, target, args, kwargs):
    trace("args=%s", args)
    from cusimt.lowering_utilities import (
        false,
        equal,
        bool_of,
        concretize_tuple_to_tensor,
    )

    tup: tuple = builder.load_var(args[0])
    assert isinstance(tup, tuple), f"Expected Python tuple, got {type(tup)}"
    tup = concretize_tuple_to_tensor(tup)

    item: ir.Value = builder.load_var(args[1])

    mr_type = tup.type
    rank = mr_type.rank
    if not mr_type.has_rank:
        raise NotImplementedError("NYI: unranked memrefs")
    if mr_type.rank != 1:
        raise TypeError(f"Expected a 1-dimensional memref, got a {rank}-dimensional memref")

    def body(op: linalg.ReduceOp, element: ir.Value, accumulator: ir.Value):
        found = equal(element, item)
        found = arith.ori(found, accumulator)
        linalg.yield_([found])

    result_type = ir.RankedTensorType.get((), T.bool())
    init = tensor.splat(result_type, false(), [])
    dims_attr = ir.DenseI64ArrayAttr.get([0])
    op = linalg.ReduceOp(
        result=[result_type],
        inputs=[tup],
        inits=[init],
        dimensions=dims_attr,
    )
    block = op.combiner.blocks.append(tup.type.element_type, result_type.element_type)
    with ir.InsertionPoint(block):
        body(op, *block.arguments)
    result = tensor.extract(op.results[0], [])
    builder.store_var(target, bool_of(result))


@intrinsic
def unituple_contains(typingctx, tup: types.UniTuple, item: types.Number):
    return types.bool(tup), unituple_contains_cg


for exc_type in (
    ArithmeticError,
    AssertionError,
    AttributeError,
    BufferError,
    EOFError,
    ImportError,
    LookupError,
    MemoryError,
    NameError,
    OSError,
    ReferenceError,
    RuntimeError,
    SystemError,
    TypeError,
    ValueError,
):

    @lower(exc_type, types.StringLiteral)
    def lower_exc_type(builder, target, args, kwargs, exc_type=exc_type):
        message = builder.load_var(args[0])
        builder.store_var(target, exc_type(message))


@lower(operator.eq, types.StringLiteral, types.StringLiteral)
def operator_eq_string_literal_lower(builder, target, args, kwargs):
    """Lower str == str for string literals (constant-fold at compile time)."""
    left_type = builder.get_numba_type(args[0].name)
    right_type = builder.get_numba_type(args[1].name)
    result = arith.constant(
        result=T.bool(),
        value=(left_type.literal_value == right_type.literal_value),
    )
    builder.store_var(target, result)


@lower(operator.not_, types.Number)
@lower(operator.not_, types.Boolean)
def lower_not(builder, target, args, kwargs):
    """Boolean negation: not x == (x == 0)."""
    import cusimt.lowering_utilities

    operand = builder.load_var(args[0])
    c0 = cusimt.lowering_utilities.constant(0, operand.type)
    res = cusimt.lowering_utilities.equal(operand, c0)
    builder.store_var(target, res)


def lower_broadcasted_binary(builder, target, args, kwargs):
    from cusimt.lowering_utilities import (
        index_of,
        broadcast_shapes_for_binary_op,
        tensor_to_memref,
    )

    args = [builder.load_var(arg) for arg in args]
    trace("args=%s", args)
    dtype = builder.get_mlir_type(target)
    dtype = T.tensor(*dtype.shape, dtype.element_type)
    etype = dtype.element_type

    op, lhs, rhs = args
    lhs, rhs = broadcast_shapes_for_binary_op(lhs, rhs, builder)

    sh = shape.shape_of(lhs)
    dims = [shape.get_extent(sh, index_of(i)) for i in range(lhs.type.rank)]
    empty = tensor.empty(sizes=dims, element_type=etype)

    lhs_conv, rhs_conv = (
        simple_scalar_conversion_op(lhs.type.element_type, etype),
        simple_scalar_conversion_op(rhs.type.element_type, etype),
    )
    lhs_etype, rhs_etype = lhs.type.element_type, rhs.type.element_type

    @linalg.map(
        result=[dtype],
        inputs=[lhs, rhs],
        init=empty,
    )
    def binop(lhs: lhs_etype, rhs: rhs_etype, init: etype):
        lhs, rhs = lhs_conv(lhs), rhs_conv(rhs)
        match op:
            case operator.sub:
                return (
                    arith.subi(lhs, rhs)
                    if isinstance(etype, ir.IntegerType)
                    else arith.subf(lhs, rhs)
                )
            case operator.add:
                return (
                    arith.addi(lhs, rhs)
                    if isinstance(etype, ir.IntegerType)
                    else arith.addf(lhs, rhs)
                )
            case operator.mul:
                return (
                    arith.muli(lhs, rhs)
                    if isinstance(etype, ir.IntegerType)
                    else arith.mulf(lhs, rhs)
                )
            case operator.truediv | operator.itruediv:
                return arith.divf(lhs, rhs)
            case operator.floordiv | operator.ifloordiv:
                return arith.divsi(lhs, rhs)
            case operator.pow | builtin_math.pow:
                match lhs.type, rhs.type:
                    case ir.FloatType(), ir.FloatType():
                        return math.powf(lhs, rhs)
                    case ir.FloatType(), ir.IntegerType():
                        return math.fpowi(lhs, rhs)
                    case ir.IntegerType(), ir.FloatType():
                        raise InternalCompilerError("NYI: integer power of float, unreachable")
                    case ir.IntegerType(), ir.IntegerType():
                        return ipowi(lhs, rhs)
            case _:
                raise NotImplementedError(f"Not implemented for operator {op}")

    result = tensor_to_memref(binop)
    builder.store_var(target, result)


@intrinsic
def broadcasted_uniform_binary_intrinsic(typingctx, op, a, b):
    """
    Broadcastable binary operation where the result type's element type
    and shape are coerced from the two operands.
    operator.add _would_ be a valid use of this, but math.pow would _not_ be,
    because it is not uniform.
    """
    match a, b:
        case (types.Array() as arr, types.Number() as n) | (
            types.Number() as n,
            types.Array() as arr,
        ):
            ety = numpy_implicit_type_promotion(arr.dtype, n)
            retty = arr.copy(dtype=ety, ndim=arr.ndim)
            return retty(a, b), lower_broadcasted_binary
        case types.Array() as arr1, types.Array() as arr2:
            ety = numpy_implicit_type_promotion(arr1.dtype, arr2.dtype)
            retty = arr1.copy(dtype=ety, ndim=max(arr1.ndim, arr2.ndim))
            return retty(a, b), lower_broadcasted_binary
        case _:
            raise NotImplementedError(f"Not implemented for types {type(a)} and {type(b)}")


@ufunc_registry.register(operator.floordiv)
def lower_broadcasted_floor_division(builder, target, args, kwargs):
    trace()

    # Use tensor types for linalg operations
    dtype = builder.get_mlir_type(target)
    etype = dtype.element_type
    dtype = T.tensor(*dtype.shape, etype)
    if isinstance(etype, ir.FloatType):
        etype = type_conversions.integer_of_width(etype.width)
        etype = type_conversions.to_mlir_type(etype)
        dtype = T.tensor(*dtype.shape, etype)

    # Convert inputs to tensors and broadcast shapes
    lhs, rhs = [builder.load_var(arg) for arg in args]
    lhs, rhs = memref_to_tensor(lhs), memref_to_tensor(rhs)
    lhs, rhs = broadcast_shapes_for_binary_op(lhs, rhs, builder)
    sh = shape.shape_of(lhs)
    dims = [shape.get_extent(sh, index_of(i)) for i in range(lhs.type.rank)]
    empty = tensor.empty(sizes=dims, element_type=dtype.element_type)

    lhs_conv, rhs_conv = (
        simple_scalar_conversion_op(lhs.type.element_type, etype),
        simple_scalar_conversion_op(rhs.type.element_type, etype),
    )
    lhs_etype, rhs_etype = lhs.type.element_type, rhs.type.element_type

    @linalg.map(
        result=[dtype],
        inputs=[lhs, rhs],
        init=empty,
    )
    def casted_div(lhs: lhs_etype, rhs: rhs_etype, init: etype):
        lhs, rhs = lhs_conv(lhs), rhs_conv(rhs)
        return arith.divsi(lhs, rhs)

    result = tensor_to_memref(casted_div)
    builder.store_var(target, result)


@ufunc_registry.register(operator.truediv)
def lower_broadcasted_div(builder, target, args, kwargs):
    trace()

    # Use tensor types for linalg operations
    dtype = builder.get_mlir_type(target)
    dtype = T.tensor(*dtype.shape, dtype.element_type)
    etype = dtype.element_type
    assert isinstance(etype, ir.FloatType)

    # Convert inputs to tensors and broadcast shapes
    lhs, rhs = [builder.load_var(arg) for arg in args]
    lhs, rhs = memref_to_tensor(lhs), memref_to_tensor(rhs)
    lhs, rhs = broadcast_shapes_for_binary_op(lhs, rhs, builder)
    sh = shape.shape_of(lhs)
    dims = [shape.get_extent(sh, index_of(i)) for i in range(lhs.type.rank)]
    empty = tensor.empty(sizes=dims, element_type=dtype.element_type)

    lhs_conv, rhs_conv = (
        simple_scalar_conversion_op(lhs.type.element_type, etype),
        simple_scalar_conversion_op(rhs.type.element_type, etype),
    )
    lhs_etype, rhs_etype = lhs.type.element_type, rhs.type.element_type

    @linalg.map(
        result=[dtype],
        inputs=[lhs, rhs],
        init=empty,
    )
    def casted_div(lhs: lhs_etype, rhs: rhs_etype, init: etype):
        lhs, rhs = lhs_conv(lhs), rhs_conv(rhs)
        return arith.divf(lhs, rhs)

    result = tensor_to_memref(casted_div)
    builder.store_var(target, result)


@intrinsic
def broadcasted_div_intrinsic(typingctx, a, b):
    arr = a if isinstance(a, types.Array) else b
    res_dt = arr.dtype
    bw = max(res_dt.bitwidth, 32)
    res_dt = type_conversions.float_of_width(bw)
    resty = arr.copy(dtype=res_dt)
    return resty(a, b), lower_broadcasted_div


@intrinsic
def broadcasted_floor_division_intrinsic(typingctx, a, b):
    arr = a if isinstance(a, types.Array) else b
    res_dt = arr.dtype
    bw = res_dt.bitwidth
    res_dt = type_conversions.integer_of_width(bw)
    resty = arr.copy(dtype=res_dt)
    return resty(a, b), lower_broadcasted_floor_division


@intrinsic
def broadcasted_pow_intrinsic(typingctx, op, a, b):
    match a, b:
        case (types.Array() as arr, types.Number() as n) | (
            types.Number() as n,
            types.Array() as arr,
        ):
            ety = numpy_implicit_type_promotion(arr.dtype, n)
            retty = arr.copy(dtype=ety, ndim=arr.ndim)
            return retty(a, b), lower_broadcasted_binary
        case types.Array() as arr1, types.Array() as arr2:
            ety = numpy_implicit_type_promotion(arr1.dtype, arr2.dtype)
            retty = arr1.copy(dtype=ety, ndim=max(arr1.ndim, arr2.ndim))
            return retty(a, b), lower_broadcasted_binary
        case _:
            raise NotImplementedError(f"Not implemented for types {type(a)} and {type(b)}")


@lower(int, types.Number)
@lower(int, types.Boolean)
@lower(bool, types.Number)
@lower(bool, types.Boolean)
@lower(float, types.Number)
@lower(float, types.Boolean)
def type_convert(builder, target, args, kwargs):
    target_numba_ty = builder.get_numba_type(target)
    to_type = builder.get_mlir_type(target)
    to_signed = isinstance(target_numba_ty, types.Integer) and target_numba_ty.signed

    if isinstance(target_numba_ty, (types.IntegerLiteral, types.Literal)):
        result = constant(target_numba_ty.literal_value, to_type)
        builder.store_var(target, result)
        return

    value = builder.load_var(args[0])

    if (const_val := try_extract_constant(value)) is not None:
        result = constant(const_val, to_type)
        builder.store_var(target, result)
        return

    if isinstance(value.type, ir.BF16Type) and isinstance(to_type, ir.IntegerType):
        value = (
            arith.fptosi(out=to_type, in_=value)
            if to_type.width > 1
            else arith.fptoui(out=to_type, in_=value)
        )
    else:
        value = convert(value, to_type, signed=to_signed)
    builder.store_var(target, value)


@lower(float, types.StringLiteral)
def float_from_string_literal(builder, target, args, kwargs):
    """Lower float("nan"), float("inf"), float("-inf"), etc."""
    from cusimt._mlir.extras import types as T

    # Get the string literal value from the type
    arg_type = builder.get_numba_type(args[0])
    string_val = arg_type.literal_value.lower()

    if string_val == "nan":
        val = float("nan")
    elif string_val == "inf" or string_val == "infinity":
        val = float("inf")
    elif string_val == "-inf" or string_val == "-infinity":
        val = float("-inf")
    else:
        raise NotImplementedError(f"float() with string literal '{string_val}' not supported")

    result = arith.constant(result=T.f64(), value=val)
    builder.store_var(target, result)


for type_name in dir(types):
    typ = getattr(types, type_name)
    if isinstance(typ, types.Type):
        lower(typ, types.Number)(type_convert)

# Register numpy type constructors (e.g. np.float32(x), np.int64(x))
import numpy as np

_numpy_type_constructors = [
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float16,
    np.float32,
    np.float64,
    np.bool_,
    np.complex64,
    np.complex128,
]
for np_type in _numpy_type_constructors:
    lower(np_type, types.Number)(type_convert)

# When callee is a type (e.g. uint8(x)), lookup uses types.NumberClass
lower(types.NumberClass, types.Number)(type_convert)
lower(types.NumberClass, types.Boolean)(type_convert)


@lower(max, types.Number, types.Number)
def lower_max(builder, target, args, kwargs):
    """Lower built-in max(a, b) for numeric types."""
    from cusimt.lowering_utilities import coerce_numpy_scalars_for_binary_op

    a = builder.load_var(args[0])
    b = builder.load_var(args[1])

    # Promote types using numpy implicit type promotion
    a, b = coerce_numpy_scalars_for_binary_op(a, b)

    # Use appropriate max operation based on type
    if isinstance(a.type, ir.FloatType):
        result = arith.maximumf(a, b)
    elif isinstance(a.type, ir.IntegerType):
        # For simplicity, use signed max (could enhance to detect signed/unsigned)
        result = arith.maxsi(a, b)
    else:
        raise NotImplementedError(f"max not implemented for type {a.type}")

    builder.store_var(target, result)


@lower(min, types.Number, types.Number)
def lower_min(builder, target, args, kwargs):
    """Lower built-in min(a, b) for numeric types."""
    from cusimt.lowering_utilities import coerce_numpy_scalars_for_binary_op

    a = builder.load_var(args[0])
    b = builder.load_var(args[1])

    # Promote types using numpy implicit type promotion
    a, b = coerce_numpy_scalars_for_binary_op(a, b)

    # Use appropriate min operation based on type
    if isinstance(a.type, ir.FloatType):
        result = arith.minimumf(a, b)
    elif isinstance(a.type, ir.IntegerType):
        # For simplicity, use signed min (could enhance to detect signed/unsigned)
        result = arith.minsi(a, b)
    else:
        raise NotImplementedError(f"min not implemented for type {a.type}")

    builder.store_var(target, result)


@lower(round, types.Number, types.Number)
def lower_round(builder, target, args, kwargs):
    from cusimt.runtime import round

    trace()
    value = builder.load_var(args[0])
    ndigits = builder.load_var(args[1])
    fname = f"round_ndigits_type_{str(value.type)}_type_{str(ndigits.type)}"
    if func := getattr(round, fname, None):
        builder.lower_call_external_mlir_library_function(target, func, args, {})
        return
    raise NotImplementedError(f"round intrinsic for types {value.type} and {ndigits.type}")


@lower(round, types.Number)
def lower_round_single(builder, target, args, kwargs):
    from cusimt._mlir.dialects import math

    value = builder.load_var(args[0])
    res = math.round(value)
    builder.store_var(target, res)


@lower(breakpoint)
def lower_breakpoint(builder, target, args, kwargs):
    nvvm.breakpoint()
    builder.store_var(target, None)


@lower(abs, types.Number)
def lower_abs_number(builder, target, args, kwargs):
    """Lower built-in abs() for numeric types (int, float, complex)."""
    assert len(args) == 1, "abs expects 1 argument"
    value = builder.load_var(args[0])
    arg_type = builder.get_numba_type(args[0].name)

    if isinstance(arg_type, types.Complex):
        result = complex_dialect.abs(value)
    elif isinstance(arg_type, types.Float):
        result = math.absf(value)
    elif isinstance(arg_type, types.Integer):
        if arg_type.signed:
            result = math.absi(value)
        else:
            result = value
    else:
        result = math.absi(value)

    builder.store_var(target, result)


@lower_getattr(types.Number, "bit_count")
def lower_number_bit_count(_, builder, target, num):
    def lower(builder, target, args, kwargs):
        from cusimt._mlir.dialects import math

        value = builder.load_var(num)
        res = math.ctpop(value)
        builder.store_var(target, res)

    builder.store_var(target, DeferredMethodCall(num, lower))


@lower(operator.add, types.Array, types.Array)
@lower(operator.add, types.Array, types.Number)
@lower(operator.add, types.Number, types.Array)
def operator_add_array_lower(builder, target, args, kwargs):
    """Lower operator.add for arrays by using linalg.add"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.add)


@lower(operator.iadd, types.Array, types.Array)
@lower(operator.iadd, types.Array, types.Number)
@lower(operator.iadd, types.Number, types.Array)
def operator_iadd_array_lower(builder, target, args, kwargs):
    """Lower operator.iadd for arrays by using linalg.add"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.add)


@lower(operator.sub, types.Array, types.Array)
@lower(operator.sub, types.Array, types.Number)
@lower(operator.sub, types.Number, types.Array)
def operator_sub_array_lower(builder, target, args, kwargs):
    """Lower operator.sub for arrays by using linalg.sub"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.sub)


@lower(operator.isub, types.Array, types.Array)
@lower(operator.isub, types.Array, types.Number)
@lower(operator.isub, types.Number, types.Array)
def operator_isub_array_lower(builder, target, args, kwargs):
    """Lower operator.isub for arrays by using linalg.sub"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.sub)


@lower(operator.mul, types.Array, types.Array)
@lower(operator.mul, types.Array, types.Number)
@lower(operator.mul, types.Number, types.Array)
def operator_mul_array_lower(builder, target, args, kwargs):
    """Lower operator.mul for arrays by using linalg.mul"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.mul)


@lower(operator.imul, types.Array, types.Array)
@lower(operator.imul, types.Array, types.Number)
@lower(operator.imul, types.Number, types.Array)
def operator_imul_array_lower(builder, target, args, kwargs):
    """Lower operator.imul for arrays by using linalg.mul"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.mul)


@lower(operator.neg, types.Number)
def operator_neg_number_lower(builder, target, args, kwargs):
    """Lower operator.neg for numbers by using arith.negf"""
    value = builder.load_var(args[0])
    c0 = constant(0, value.type)
    result = c0 - value
    builder.store_var(target, result)


@lower(operator.eq, types.Number, types.NoneType)
@lower(operator.eq, types.NoneType, types.Number)
def operator_eq_none_lower(builder, target, args, kwargs):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    builder.store_var(target, result)


@lower(operator.ne, types.Number, types.NoneType)
@lower(operator.ne, types.NoneType, types.Number)
def operator_ne_none_lower(builder, target, args, kwargs):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    builder.store_var(target, result)


@lower(operator.eq, types.NoneType, types.NoneType)
def operator_eq_none_none_lower(builder, target, args, kwargs):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    builder.store_var(target, result)


@lower(operator.ne, types.NoneType, types.NoneType)
def operator_ne_none_none_lower(builder, target, args, kwargs):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    builder.store_var(target, result)


@lower(operator.is_, types.Number, types.NoneType)
@lower(operator.is_, types.NoneType, types.Number)
def operator_is_none_lower(builder, target, args, kwargs):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    builder.store_var(target, result)


@lower(operator.is_, types.NoneType, types.NoneType)
def operator_is_none_none_lower(builder, target, args, kwargs):
    """Lower 'None is None' - always True."""
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    builder.store_var(target, result)


@lower(operator.is_, types.Boolean, types.Literal)
@lower(operator.is_, types.Literal, types.Boolean)
def operator_is_bool_literal_lower(builder, target, args, kwargs):
    """Lower 'x is True' or 'x is False' - compare values."""
    left = builder.load_var(args[0])
    right = builder.load_var(args[1])
    left_val = try_extract_constant(left)
    right_val = try_extract_constant(right)
    if left_val is not None and right_val is not None:
        result = arith.constant(result=ir.IntegerType.get_signless(1), value=left_val is right_val)
    elif left_val is not None:
        result = arith.cmpi(
            arith.CmpIPredicate.eq,
            arith.constant(result=ir.IntegerType.get_signless(1), value=left_val),
            right,
        )
    elif right_val is not None:
        result = arith.cmpi(
            arith.CmpIPredicate.eq,
            left,
            arith.constant(result=ir.IntegerType.get_signless(1), value=right_val),
        )
    else:
        result = arith.cmpi(arith.CmpIPredicate.eq, left, right)
    builder.store_var(target, result)


@lower(operator.is_not, types.Number, types.NoneType)
@lower(operator.is_not, types.NoneType, types.Number)
def operator_is_not_none_lower(builder, target, args, kwargs):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    builder.store_var(target, result)


@lower(operator.is_not, types.NoneType, types.NoneType)
def operator_is_not_none_none_lower(builder, target, args, kwargs):
    """Lower 'None is not None' - always False."""
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    builder.store_var(target, result)


@lower(operator.is_not, types.Boolean, types.Literal)
@lower(operator.is_not, types.Literal, types.Boolean)
def operator_is_not_bool_literal_lower(builder, target, args, kwargs):
    """Lower 'x is not True' or 'x is not False' - compare values."""
    left = builder.load_var(args[0])
    right = builder.load_var(args[1])
    left_val = try_extract_constant(left)
    right_val = try_extract_constant(right)
    if left_val is not None and right_val is not None:
        result = arith.constant(
            result=ir.IntegerType.get_signless(1), value=left_val is not right_val
        )
    elif left_val is not None:
        result = arith.cmpi(
            arith.CmpIPredicate.ne,
            arith.constant(result=ir.IntegerType.get_signless(1), value=left_val),
            right,
        )
    elif right_val is not None:
        result = arith.cmpi(
            arith.CmpIPredicate.ne,
            left,
            arith.constant(result=ir.IntegerType.get_signless(1), value=right_val),
        )
    else:
        result = arith.cmpi(arith.CmpIPredicate.ne, left, right)
    builder.store_var(target, result)


@lower(operator.pos, types.Number)
def operator_pos_number_lower(builder, target, args, kwargs):
    """Lower operator.neg for numbers by using arith.negf"""
    value = builder.load_var(args[0])
    builder.store_var(target, value)
