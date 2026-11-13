# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
from numba_cuda_mlir.errors import InternalCompilerError
import math as builtin_math
import operator
from numba_cuda_mlir.lowering_utilities import (
    DeferredMethodCall,
    type_conversions,
    convert,
    constant,
    numpy_implicit_type_promotion,
    memref_to_tensor,
    memref_to_value_tensor,
    value_tensor_to_storage_memref,
    broadcast_shapes_for_binary_op,
    index_of,
    tensor_to_memref,
    simple_scalar_conversion_op,
    expensive_coerce_tensor_type,
    get_conversion_signedness,
    try_extract_constant,
)
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.extending import intrinsic
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import (
    arith,
    math,
    nvvm,
)
from numba_cuda_mlir.mlir.dialect_exts.math import ipowi
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
from numba_cuda_mlir.lowering_utilities import (
    RangeObject,
    int_of,
)
from numba_cuda_mlir._mlir import ir
from .ufunc_registry import UFuncRegistry

ufunc_registry = UFuncRegistry("builtins")


def _get_range_object(builder, args: list[ir.Value]) -> tuple[ir.Value, ir.Value, ir.Value]:
    match args:
        case [stop]:
            stop_type = stop.type
            return (
                int_of(0, ty=stop_type),
                int_of(stop, ty=stop_type, signed=True),
                int_of(1, ty=stop_type),
            )
        case [start, stop]:
            start_type = start.type
            stop_type = stop.type
            return (
                int_of(start, ty=start_type, signed=True),
                int_of(stop, ty=stop_type, signed=True),
                int_of(1, ty=stop_type),
            )
        case [start, stop, step]:
            return (
                int_of(
                    start,
                    ty=start.type,
                    signed=True,
                ),
                int_of(stop, ty=stop.type, signed=True),
                int_of(step, ty=step.type, signed=True),
            )
        case _:
            raise ValueError(f"Invalid arguments for range: {args}")


@lower(operator.mul, types.UniTuple, types.Literal, api="values")
@lower(operator.mul, types.Tuple, types.Literal, api="values")
@lower(operator.mul, types.Literal, types.UniTuple, api="values")
def lower_tuple_multiply(builder, signature, args):
    trace()
    assert len(args) == 2, "tuple multiply expects 2 arguments"
    tup, how_many = args
    assert isinstance(tup, tuple), f"Expected tuple, got {type(tup)}"
    match tup, how_many:
        case tuple(), (int() | ir.Value()):
            pass
        case ((int() | ir.Value()), tuple()):
            tup, how_many = how_many, tup
        case _:
            raise TypeError(f"Invalid arguments for tuple multiply: {tup=} {how_many=}")

    if how_many := try_extract_constant(how_many):
        return tup * how_many

    raise InternalCompilerError(f"Expected literal integer, got {type(how_many)}")


@lower(tuple, types.UniTuple, api="values")
@lower(tuple, types.Tuple, api="values")
def lower_tuple_builtin(builder, signature, args):
    trace()
    assert len(args) == 1, "tuple() expects 1 argument"
    value = args[0]
    if not isinstance(value, tuple):
        value = (value,)
    return value


@lower(operator.add, types.UniTuple, types.UniTuple, api="values")
@lower(operator.add, types.Tuple, types.Tuple, api="values")
@lower(operator.add, types.UniTuple, types.Tuple, api="values")
@lower(operator.add, types.Tuple, types.UniTuple, api="values")
def lower_tuple_concat(builder, signature, args):
    trace()
    assert len(args) == 2, "tuple concat expects 2 arguments"
    lhs, rhs = args
    if not isinstance(lhs, tuple):
        lhs = (lhs,)
    if not isinstance(rhs, tuple):
        rhs = (rhs,)
    return lhs + rhs


def _tuple_equal_value(builder, signature, args):
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.cuda.experimental import consteval

    lhs_type, rhs_type = signature.args
    if len(lhs_type) != len(rhs_type):
        return arith.constant(T.bool(), False)

    # Unroll by field so heterogeneous and nested tuples use their own
    # element comparison lowerings, including the usual numeric promotions.
    indices = tuple(range(len(lhs_type)))

    @cuda.jit(device=True)
    def impl(lhs, rhs):
        result = True
        for i in consteval(indices):
            result = result & (lhs[i] == rhs[i])
        return result

    return builder.build_user_defined_function_call_values(impl, signature, args)


@lower(operator.eq, types.BaseTuple, types.BaseTuple, api="values")
def lower_tuple_eq(builder, signature, args):
    return _tuple_equal_value(builder, signature, args)


@lower(operator.ne, types.BaseTuple, types.BaseTuple, api="values")
def lower_tuple_ne(builder, signature, args):
    result = _tuple_equal_value(builder, signature, args)
    return arith.xori(result, arith.constant(T.bool(), True))


@lower(range, types.Number, api="values")
@lower(range, types.Number, types.Number, api="values")
@lower(range, types.Number, types.Number, types.Number, api="values")
def lower_range(builder: MLIRLower, signature, args):
    start, stop, step = _get_range_object(builder, args)
    ro = RangeObject(builder, start, stop, step)
    return ro


@lower(operator.contains, types.Tuple, types.Number, api="values")
def tuple_contains_cg(builder, signature, args):
    tup = args[0]
    item = args[1]
    from numba_cuda_mlir.lowering_utilities import bool_of

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
        return bool_of(item_const in constant_values)
    else:
        # Runtime check - item is not a constant
        from numba_cuda_mlir.lowering_utilities import false, equal

        result = false()
        for const_val in constant_values:
            result = arith.ori(result, equal(item, int_of(const_val, item.type)))
        return result


@lower(operator.contains, types.UniTuple, types.Number, api="values")
def unituple_contains_cg(builder, signature, args):
    trace("args=%s", args)
    from numba_cuda_mlir._mlir.dialects import linalg, tensor
    from numba_cuda_mlir.lowering_utilities import (
        false,
        equal,
        bool_of,
        concretize_tuple_to_tensor,
    )

    tup: tuple = args[0]
    assert isinstance(tup, tuple), f"Expected Python tuple, got {type(tup)}"
    tup = concretize_tuple_to_tensor(tup)

    item: ir.Value = args[1]

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
    return bool_of(result)


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

    @lower(exc_type, types.StringLiteral, api="values")
    def lower_exc_type(builder, signature, args):
        message = args[0]
        return signature.return_type.exc_class(message)


@lower(operator.eq, types.StringLiteral, types.StringLiteral, api="values")
def operator_eq_string_literal_lower(builder, signature, args):
    """Lower str == str for string literals (constant-fold at compile time)."""
    left_type = signature.args[0]
    right_type = signature.args[1]
    result = arith.constant(
        result=T.bool(),
        value=(left_type.literal_value == right_type.literal_value),
    )
    return result


@lower(operator.not_, types.Number, api="values")
@lower(operator.not_, types.Boolean, api="values")
def lower_not(builder, signature, args):
    """Boolean negation: not x == (x == 0)."""
    import numba_cuda_mlir.lowering_utilities

    operand = args[0]
    c0 = numba_cuda_mlir.lowering_utilities.constant(0, operand.type)
    res = numba_cuda_mlir.lowering_utilities.equal(operand, c0)
    return res


def lower_broadcasted_binary(builder, signature, args):
    from numba_cuda_mlir._mlir.dialects import linalg, shape, tensor
    from numba_cuda_mlir.lowering_utilities import index_of, tensor_to_memref

    op, lhs, rhs = args
    lhs_type = signature.args[1]
    rhs_type = signature.args[2]
    target_type = signature.return_type
    result_element_type = builder.get_value_type(target_type.dtype)

    if isinstance(lhs_type, types.Array):
        lhs = memref_to_value_tensor(lhs_type, lhs)
    if isinstance(rhs_type, types.Array):
        rhs = memref_to_value_tensor(rhs_type, rhs)

    lhs, rhs = broadcast_shapes_for_binary_op(lhs, rhs, builder)
    lhs = expensive_coerce_tensor_type(lhs, result_element_type)
    rhs = expensive_coerce_tensor_type(rhs, result_element_type)

    sh = shape.shape_of(lhs)
    dims = [shape.get_extent(sh, index_of(i)) for i in range(lhs.type.rank)]
    dtype = T.tensor(*lhs.type.shape, result_element_type)
    empty = tensor.empty(sizes=dims, element_type=result_element_type)

    lhs_conv, rhs_conv = (
        simple_scalar_conversion_op(lhs.type.element_type, result_element_type),
        simple_scalar_conversion_op(rhs.type.element_type, result_element_type),
    )
    lhs_etype, rhs_etype = lhs.type.element_type, rhs.type.element_type

    @linalg.map(
        result=[dtype],
        inputs=[lhs, rhs],
        init=empty,
    )
    def binop(lhs: lhs_etype, rhs: rhs_etype, init: result_element_type):
        lhs, rhs = lhs_conv(lhs), rhs_conv(rhs)
        match op:
            case operator.sub:
                return (
                    arith.subi(lhs, rhs)
                    if isinstance(result_element_type, ir.IntegerType)
                    else arith.subf(lhs, rhs)
                )
            case operator.add:
                return (
                    arith.addi(lhs, rhs)
                    if isinstance(result_element_type, ir.IntegerType)
                    else arith.addf(lhs, rhs)
                )
            case operator.mul:
                return (
                    arith.muli(lhs, rhs)
                    if isinstance(result_element_type, ir.IntegerType)
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
                        rhs = convert(rhs, lhs.type)
                        return ipowi(lhs, rhs)
            case _:
                raise NotImplementedError(f"Not implemented for operator {op}")

    result = value_tensor_to_storage_memref(target_type, binop)
    return result


@ufunc_registry.register(operator.floordiv)
@lower(operator.floordiv, types.Array, types.Array, api="values")
def lower_broadcasted_floor_division(builder, signature, args):
    from numba_cuda_mlir._mlir.dialects import linalg, shape, tensor

    trace()

    target_type = signature.return_type
    result_value_type = builder.get_value_type(target_type.dtype)
    compute_type = result_value_type
    if isinstance(result_value_type, ir.FloatType):
        compute_type = type_conversions.integer_of_width(result_value_type.width)
        compute_type = type_conversions.to_mlir_type(compute_type)

    lhs, rhs = args
    lhs_type = signature.args[0]
    rhs_type = signature.args[1]
    if isinstance(lhs_type, types.Array):
        lhs = memref_to_value_tensor(lhs_type, lhs)
    if isinstance(rhs_type, types.Array):
        rhs = memref_to_value_tensor(rhs_type, rhs)

    lhs, rhs = broadcast_shapes_for_binary_op(lhs, rhs, builder)
    lhs = expensive_coerce_tensor_type(lhs, compute_type)
    rhs = expensive_coerce_tensor_type(rhs, compute_type)
    sh = shape.shape_of(lhs)
    dims = [shape.get_extent(sh, index_of(i)) for i in range(lhs.type.rank)]
    result_type = T.tensor(*lhs.type.shape, result_value_type)
    empty = tensor.empty(sizes=dims, element_type=result_value_type)

    lhs_conv, rhs_conv = (
        simple_scalar_conversion_op(lhs.type.element_type, compute_type),
        simple_scalar_conversion_op(rhs.type.element_type, compute_type),
    )
    lhs_etype, rhs_etype = lhs.type.element_type, rhs.type.element_type

    @linalg.map(
        result=[result_type],
        inputs=[lhs, rhs],
        init=empty,
    )
    def casted_div(lhs: lhs_etype, rhs: rhs_etype, init: result_value_type):
        lhs, rhs = lhs_conv(lhs), rhs_conv(rhs)
        result = arith.divsi(lhs, rhs)
        return convert(result, result_value_type)

    result = value_tensor_to_storage_memref(target_type, casted_div)
    return result


@ufunc_registry.register(operator.truediv)
@lower(operator.truediv, types.Array, types.Array, api="values")
def lower_broadcasted_div(builder, signature, args):
    from numba_cuda_mlir._mlir.dialects import linalg, shape, tensor

    trace()

    target_type = signature.return_type
    result_value_type = builder.get_value_type(target_type.dtype)
    assert isinstance(result_value_type, ir.FloatType)

    lhs, rhs = args
    lhs_type = signature.args[0]
    rhs_type = signature.args[1]
    if isinstance(lhs_type, types.Array):
        lhs = memref_to_value_tensor(lhs_type, lhs)
    if isinstance(rhs_type, types.Array):
        rhs = memref_to_value_tensor(rhs_type, rhs)

    lhs, rhs = broadcast_shapes_for_binary_op(lhs, rhs, builder)
    lhs = expensive_coerce_tensor_type(lhs, result_value_type)
    rhs = expensive_coerce_tensor_type(rhs, result_value_type)
    sh = shape.shape_of(lhs)
    dims = [shape.get_extent(sh, index_of(i)) for i in range(lhs.type.rank)]
    result_type = T.tensor(*lhs.type.shape, result_value_type)
    empty = tensor.empty(sizes=dims, element_type=result_value_type)

    lhs_conv, rhs_conv = (
        simple_scalar_conversion_op(lhs.type.element_type, result_value_type),
        simple_scalar_conversion_op(rhs.type.element_type, result_value_type),
    )
    lhs_etype, rhs_etype = lhs.type.element_type, rhs.type.element_type

    @linalg.map(
        result=[result_type],
        inputs=[lhs, rhs],
        init=empty,
    )
    def casted_div(lhs: lhs_etype, rhs: rhs_etype, init: result_value_type):
        lhs, rhs = lhs_conv(lhs), rhs_conv(rhs)
        return arith.divf(lhs, rhs)

    result = value_tensor_to_storage_memref(target_type, casted_div)
    return result


@lower(int, types.Number, api="values")
@lower(int, types.Boolean, api="values")
@lower(bool, types.Number, api="values")
@lower(bool, types.Boolean, api="values")
@lower(float, types.Number, api="values")
@lower(float, types.Boolean, api="values")
def type_convert(builder, signature, args):
    target_numba_ty = signature.return_type
    to_type = builder.get_mlir_type(signature.return_type)
    source_numba_ty = signature.args[0]
    signed = get_conversion_signedness(source_numba_ty, target_numba_ty)

    if isinstance(target_numba_ty, (types.IntegerLiteral, types.Literal)):
        result = constant(target_numba_ty.literal_value, to_type)
        return result

    value = args[0]

    if (const_val := try_extract_constant(value)) is not None:
        result = constant(const_val, to_type)
        return result

    # An i1 target is excluded so it falls through to `convert`, which compares against
    # zero rather than truncating; fptosi/fptoui to i1 keeps the low bit, so bf16(0.0)
    # came out True and bf16(2.0) came out False.
    if (
        isinstance(value.type, ir.BF16Type)
        and isinstance(to_type, ir.IntegerType)
        and to_type.width > 1
    ):
        value = (
            arith.fptosi(out=to_type, in_=value) if signed else arith.fptoui(out=to_type, in_=value)
        )
    else:
        value = convert(
            value,
            to_type,
            signed=signed,
        )
    return value


@lower(float, types.StringLiteral, api="values")
def float_from_string_literal(builder, signature, args):
    """Lower float("nan"), float("inf"), float("-inf"), etc."""
    from numba_cuda_mlir._mlir.extras import types as T

    # Get the string literal value from the type
    arg_type = signature.args[0]
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
    return result


for type_name in dir(types):
    typ = getattr(types, type_name)
    if isinstance(typ, types.Type):
        lower(typ, types.Number, api="values")(type_convert)

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
    lower(np_type, types.Number, api="values")(type_convert)

# When callee is a type (e.g. uint8(x)), lookup uses types.NumberClass
lower(types.NumberClass, types.Number, api="values")(type_convert)
lower(types.NumberClass, types.Boolean, api="values")(type_convert)


def _minmax_operands(builder, signature, args):
    """Convert operands to the inferred common type using source signedness."""
    target_type = signature.return_type
    mlir_type = builder.get_mlir_type(target_type)
    return tuple(
        convert(value, mlir_type, signed=get_conversion_signedness(source_type, target_type))
        for value, source_type in zip(args, signature.args)
    )


@lower(max, types.VarArg(types.Any), api="values")
@lower(max, types.Number, types.Number, api="values")
def lower_max(builder, signature, args):
    """Lower built-in max(a, b, ...) using the inferred common numeric type."""
    values = _minmax_operands(builder, signature, args)
    result = values[0]
    for value in values[1:]:
        if isinstance(result.type, ir.FloatType):
            if len(args) == 2:
                result = arith.maxnumf(result, value)
            else:
                greater = arith.cmpf(arith.CmpFPredicate.OGT, value, result)
                result = arith.select(greater, value, result)
        elif isinstance(result.type, ir.IntegerType | ir.IndexType):
            result = (
                arith.maxsi(result, value)
                if signature.return_type.signed
                else arith.maxui(result, value)
            )
        else:
            raise NotImplementedError(f"max not implemented for type {result.type}")

    return result


@lower(min, types.VarArg(types.Any), api="values")
@lower(min, types.Number, types.Number, api="values")
def lower_min(builder, signature, args):
    """Lower built-in min(a, b, ...) using the inferred common numeric type."""
    values = _minmax_operands(builder, signature, args)
    result = values[0]
    for value in values[1:]:
        if isinstance(result.type, ir.FloatType):
            if len(args) == 2:
                result = arith.minnumf(result, value)
            else:
                less = arith.cmpf(arith.CmpFPredicate.OLT, value, result)
                result = arith.select(less, value, result)
        elif isinstance(result.type, ir.IntegerType | ir.IndexType):
            result = (
                arith.minsi(result, value)
                if signature.return_type.signed
                else arith.minui(result, value)
            )
        else:
            raise NotImplementedError(f"min not implemented for type {result.type}")

    return result


@lower(round, types.Number, types.Number, api="values")
def lower_round(builder, signature, args):
    from numba_cuda_mlir.runtime import round

    trace()
    value = args[0]
    ndigits = args[1]
    value_type = value.type
    if isinstance(value_type, ir.IntegerType):
        # Numba types round(int, ndigits) as (float, intp) -> float, so pick the
        # float variant the signature declares. The call lowering converts each
        # operand to the callee's declared input type.
        value_type = builder.get_mlir_type(signature.return_type)
    fname = f"round_ndigits_type_{str(value_type)}_type_{str(ndigits.type)}"
    if func := getattr(round, fname, None):
        return builder.lower_call_external_mlir_library_function_values(func, signature, args)
    raise NotImplementedError(f"round intrinsic for types {value.type} and {ndigits.type}")


@lower(round, types.Integer, api="values")
@lower(round, types.Boolean, api="values")
def lower_round_integer(builder, signature, args):
    # Boolean is not a subclass of Integer in Numba's type hierarchy, so it
    # needs its own registration to reach this already-rounded path.
    source_type = signature.args[0]
    target_type = signature.return_type
    value = builder.lower_cast(source_type, target_type, args[0])
    return value


@lower(round, types.Number, api="values")
def lower_round_single(builder, signature, args):
    from numba_cuda_mlir._mlir.dialects import math

    value = args[0]
    # math.round takes halfway cases away from zero; Python, NumPy and
    # numba-cuda take them to even, which is math.roundeven.  The two-argument
    # lowering already goes through runtime.round_intrinsics for this reason.
    res = math.roundeven(value)
    return res


@lower(breakpoint, api="values")
def lower_breakpoint(builder, signature, args):
    nvvm.breakpoint()
    return None


@lower(abs, types.Number, api="values")
def lower_abs_number(builder, signature, args):
    """Lower built-in abs() for numeric types (int, float, complex)."""
    assert len(args) == 1, "abs expects 1 argument"
    value = args[0]
    arg_type = signature.args[0]

    if isinstance(arg_type, types.Complex):
        from numba_cuda_mlir._mlir.dialects import complex as complex_dialect

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

    return result


@lower_getattr(types.Number, "bit_count")
def lower_number_bit_count(_, builder, target, num):
    def lower(builder, signature, args):
        from numba_cuda_mlir._mlir.dialects import math

        value = args[0]
        res = math.ctpop(value)
        return res

    builder.store_var(
        target, DeferredMethodCall(builder.load_var(num), builder.get_numba_type(num), lower)
    )


@lower(operator.add, types.Array, types.Array, api="values")
@lower(operator.add, types.Array, types.Number, api="values")
@lower(operator.add, types.Number, types.Array, api="values")
def operator_add_array_lower(builder, signature, args):
    """Lower operator.add for arrays by using linalg.add"""
    from numba_cuda_mlir._mlir.dialects import linalg
    from numba_cuda_mlir.lowering_utilities.linalg_lowering import lower_np_binop

    return lower_np_binop(builder, signature, args, linalg.add)


@lower(operator.iadd, types.Array, types.Array, api="values")
@lower(operator.iadd, types.Array, types.Number, api="values")
@lower(operator.iadd, types.Number, types.Array, api="values")
def operator_iadd_array_lower(builder, signature, args):
    """Lower operator.iadd for arrays by using linalg.add"""
    from numba_cuda_mlir._mlir.dialects import linalg
    from numba_cuda_mlir.lowering_utilities.linalg_lowering import lower_np_binop

    return lower_np_binop(builder, signature, args, linalg.add)


@lower(operator.sub, types.Array, types.Array, api="values")
@lower(operator.sub, types.Array, types.Number, api="values")
@lower(operator.sub, types.Number, types.Array, api="values")
def operator_sub_array_lower(builder, signature, args):
    """Lower operator.sub for arrays by using linalg.sub"""
    from numba_cuda_mlir._mlir.dialects import linalg
    from numba_cuda_mlir.lowering_utilities.linalg_lowering import lower_np_binop

    return lower_np_binop(builder, signature, args, linalg.sub)


@lower(operator.isub, types.Array, types.Array, api="values")
@lower(operator.isub, types.Array, types.Number, api="values")
@lower(operator.isub, types.Number, types.Array, api="values")
def operator_isub_array_lower(builder, signature, args):
    """Lower operator.isub for arrays by using linalg.sub"""
    from numba_cuda_mlir._mlir.dialects import linalg
    from numba_cuda_mlir.lowering_utilities.linalg_lowering import lower_np_binop

    return lower_np_binop(builder, signature, args, linalg.sub)


@lower(operator.mul, types.Array, types.Array, api="values")
@lower(operator.mul, types.Array, types.Number, api="values")
@lower(operator.mul, types.Number, types.Array, api="values")
def operator_mul_array_lower(builder, signature, args):
    """Lower operator.mul for arrays by using linalg.mul"""
    from numba_cuda_mlir._mlir.dialects import linalg
    from numba_cuda_mlir.lowering_utilities.linalg_lowering import lower_np_binop

    return lower_np_binop(builder, signature, args, linalg.mul)


@lower(operator.imul, types.Array, types.Array, api="values")
@lower(operator.imul, types.Array, types.Number, api="values")
@lower(operator.imul, types.Number, types.Array, api="values")
def operator_imul_array_lower(builder, signature, args):
    """Lower operator.imul for arrays by using linalg.mul"""
    from numba_cuda_mlir._mlir.dialects import linalg
    from numba_cuda_mlir.lowering_utilities.linalg_lowering import lower_np_binop

    return lower_np_binop(builder, signature, args, linalg.mul)


@lower(operator.neg, types.Number, api="values")
def operator_neg_number_lower(builder, signature, args):
    """Lower operator.neg for numbers"""
    value = args[0]
    match value.type:
        case ir.FloatType():
            # Flip the sign bit: 0.0 - x would give +0.0 for x = 0.0, and fold
            # constants such as -nan to +nan
            result = arith.negf(value)
        case ir.ComplexType():
            from numba_cuda_mlir._mlir.dialects import complex as complex_dialect

            result = complex_dialect.neg(value)
        case _:
            result = constant(0, value.type) - value
    source_type = signature.args[0]
    if isinstance(source_type, types.Integer) and not source_type.signed:
        # Wrap at the input width before zero-extending, like stock numba-cuda.
        target_type = signature.return_type
        result = convert(result, builder.get_mlir_type(target_type), signed=False)
    return result


@lower(operator.eq, types.Number, types.NoneType, api="values")
@lower(operator.eq, types.NoneType, types.Number, api="values")
def operator_eq_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    return result


@lower(operator.ne, types.Number, types.NoneType, api="values")
@lower(operator.ne, types.NoneType, types.Number, api="values")
def operator_ne_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    return result


@lower(operator.eq, types.NoneType, types.NoneType, api="values")
def operator_eq_none_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    return result


@lower(operator.ne, types.NoneType, types.NoneType, api="values")
def operator_ne_none_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    return result


@lower(operator.is_, types.Number, types.NoneType, api="values")
@lower(operator.is_, types.NoneType, types.Number, api="values")
def operator_is_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    return result


@lower(operator.is_, types.BaseTuple, types.NoneType, api="values")
@lower(operator.is_, types.NoneType, types.BaseTuple, api="values")
def operator_is_tuple_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    return result


@lower(operator.is_, types.NoneType, types.NoneType, api="values")
def operator_is_none_none_lower(builder, signature, args):
    """Lower 'None is None' - always True."""
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    return result


@lower(operator.is_, types.Optional, types.NoneType, api="values")
@lower(operator.is_, types.NoneType, types.Optional, api="values")
def operator_is_optional_none_lower(builder, signature, args):
    """Lower 'optional_val is None' - check valid bit is false."""
    from numba_cuda_mlir.mlir.dialect_exts import llvm

    lhs_type = signature.args[0]
    opt_arg = args[0] if isinstance(lhs_type, types.Optional) else args[1]
    opt_val = opt_arg
    i1 = ir.IntegerType.get_signless(1)
    valid = llvm.extractvalue(i1, opt_val, [1])
    result = arith.xori(valid, arith.constant(i1, 1))
    return result


@lower(operator.is_, types.Boolean, types.Boolean, api="values")
@lower(operator.is_, types.Boolean, types.Literal, api="values")
@lower(operator.is_, types.Literal, types.Boolean, api="values")
def operator_is_bool_literal_lower(builder, signature, args):
    """Lower 'x is True' or 'x is False' - compare values."""
    left = args[0]
    right = args[1]
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
    return result


@lower(operator.is_not, types.Number, types.NoneType, api="values")
@lower(operator.is_not, types.NoneType, types.Number, api="values")
def operator_is_not_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    return result


@lower(operator.is_not, types.BaseTuple, types.NoneType, api="values")
@lower(operator.is_not, types.NoneType, types.BaseTuple, api="values")
def operator_is_not_tuple_none_lower(builder, signature, args):
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=True)
    return result


@lower(operator.is_not, types.NoneType, types.NoneType, api="values")
def operator_is_not_none_none_lower(builder, signature, args):
    """Lower 'None is not None' - always False."""
    result = arith.constant(result=ir.IntegerType.get_signless(1), value=False)
    return result


@lower(operator.is_not, types.Optional, types.NoneType, api="values")
@lower(operator.is_not, types.NoneType, types.Optional, api="values")
def operator_is_not_optional_none_lower(builder, signature, args):
    """Lower 'optional_val is not None' - check valid bit is true."""
    from numba_cuda_mlir.mlir.dialect_exts import llvm

    lhs_type = signature.args[0]
    opt_arg = args[0] if isinstance(lhs_type, types.Optional) else args[1]
    opt_val = opt_arg
    i1 = ir.IntegerType.get_signless(1)
    valid = llvm.extractvalue(i1, opt_val, [1])
    return valid


@lower(operator.is_not, types.Boolean, types.Boolean, api="values")
@lower(operator.is_not, types.Boolean, types.Literal, api="values")
@lower(operator.is_not, types.Literal, types.Boolean, api="values")
def operator_is_not_bool_literal_lower(builder, signature, args):
    """Lower 'x is not True' or 'x is not False' - compare values."""
    left = args[0]
    right = args[1]
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
    return result


@lower(operator.pos, types.Number, api="values")
def operator_pos_number_lower(builder, signature, args):
    """Lower operator.neg for numbers by using arith.negf"""
    value = args[0]
    return value
