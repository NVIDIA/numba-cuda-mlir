# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import functools
from typing import Callable
from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir.lowering_utilities import (
    numpy_implicit_type_promotion,
    convert,
    i64_of,
    get_conversion_signedness,
    get_or_insert_function,
)
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
lower = registry.lower
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import ir as numba_ir
import numba_cuda_mlir.types
import math
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir.dialects import (
    complex as complex_dialect,
    arith,
    func,
    math as math_dialect,
    memref,
    llvm,
)
from numba_cuda_mlir.mlir.dialect_exts.math import ipowi
import numba_cuda_mlir._mlir.ir as ir
import operator

MaybeOp = None | Callable[[ir.Value, ir.Value], ir.OpResult]


@dataclass
class OpForType:
    """
    Descriptor for a builtin operator describing the conversion semantics
    and the builder for the operation given operand types.
    """

    float: MaybeOp
    signed_integer: MaybeOp
    unsigned_integer: MaybeOp
    complex: MaybeOp
    cast_to_return_type: bool = False


def _get_float_of_same_size_as(ty: ir.Type) -> ir.Type:
    """
    This could be more general. We will need more complicated casting
    logic depending on the operation, and we should generalize this
    and move into mlir utils.
    """
    match ty:
        case ir.IndexType():
            return T.f64()
        case ir.IntegerType():
            match ty.width:
                case 64:
                    return T.f64()
                case 1 | 8 | 16 | 32:
                    return T.f32()
                case _:
                    raise ValueError(f"Unsupported integer type width: {ty.width}")
        case ir.FloatType():
            return ty
        case _:
            raise ValueError(f"Unsupported type: {ty}")


def _cast_to_float_of_same_size(value: ir.Value, source_type: types.Type | None = None) -> ir.Value:
    float_type = _get_float_of_same_size_as(value.type)
    signed = None
    if source_type is not None:
        target_type = types.float32 if float_type == T.f32() else types.float64
        signed = get_conversion_signedness(source_type, target_type)
    return lowering_utilities.convert(value, float_type, signed=signed)


def _ensure_float(value: ir.Value, source_type: types.Type | None = None) -> ir.Value:
    """Ensure value is floating-point, converting integers to floats if needed."""
    if isinstance(value.type, ir.IntegerType) or isinstance(value.type, ir.IndexType):
        return _cast_to_float_of_same_size(value, source_type)
    return value


def _is_integer_type(ty: ir.Type) -> bool:
    """Check if the type is an integer or index type."""
    return isinstance(ty, (ir.IntegerType, ir.IndexType))


def _call_libdevice_unary(mlir_lower, value: ir.Value, f64_name: str, f32_name: str):
    """Call a unary libdevice function based on the value type."""
    float_type = value.type
    if float_type == T.f64():
        name = f64_name
    elif float_type == T.f32():
        name = f32_name
    else:
        value = convert(value, T.f64())
        float_type = T.f64()
        name = f64_name
    fn_type = ir.FunctionType.get([float_type], [float_type])
    callee = get_or_insert_function(name, fn_type, mlir_lower.mlir_gpu_module)
    return func.call(result=[float_type], callee=callee.name.value, operands_=[value])


def _call_libdevice_binary(mlir_lower, x: ir.Value, y: ir.Value, f64_name: str, f32_name: str):
    """Call a binary libdevice function based on the value types."""
    float_type = x.type
    if float_type == T.f64():
        name = f64_name
    elif float_type == T.f32():
        name = f32_name
    else:
        x = convert(x, T.f64())
        y = convert(y, T.f64())
        float_type = T.f64()
        name = f64_name
    fn_type = ir.FunctionType.get([float_type, float_type], [float_type])
    callee = get_or_insert_function(name, fn_type, mlir_lower.mlir_gpu_module)
    return func.call(result=[float_type], callee=callee.name.value, operands_=[x, y])


def _casted_div(lhs, rhs) -> ir.OpResult:
    lhs = _cast_to_float_of_same_size(lhs)
    rhs = _cast_to_float_of_same_size(rhs)
    return arith.divf(lhs, rhs)


def _make_fcmp(
    predicate: arith.CmpFPredicate,
) -> Callable[[ir.Value, ir.Value], ir.OpResult]:
    return functools.partial(arith.cmpf, predicate)


def _make_icmp(
    predicate: arith.CmpIPredicate,
) -> Callable[[ir.Value, ir.Value], ir.OpResult]:
    return functools.partial(arith.cmpi, predicate)


@functools.lru_cache(maxsize=1)
def _operator_mapping() -> dict:
    return {
        operator.add: OpForType(
            arith.addf,
            arith.addi,
            arith.addi,
            complex_dialect.add,
            cast_to_return_type=True,
        ),
        operator.iadd: OpForType(
            arith.addi, arith.addi, arith.addi, None, cast_to_return_type=True
        ),
        operator.ior: OpForType(None, arith.ori, arith.ori, None, cast_to_return_type=True),
        operator.iand: OpForType(None, arith.andi, arith.andi, None, cast_to_return_type=True),
        operator.ixor: OpForType(None, arith.xori, arith.xori, None, cast_to_return_type=True),
        operator.sub: OpForType(
            arith.subf,
            arith.subi,
            arith.subi,
            complex_dialect.sub,
            cast_to_return_type=True,
        ),
        operator.mul: OpForType(
            arith.mulf,
            arith.muli,
            arith.muli,
            complex_dialect.mul,
            cast_to_return_type=True,
        ),
        operator.truediv: OpForType(
            _casted_div,
            _casted_div,
            _casted_div,
            complex_dialect.div,
            cast_to_return_type=True,
        ),
        operator.itruediv: OpForType(
            _casted_div, _casted_div, _casted_div, None, cast_to_return_type=True
        ),
        operator.mod: OpForType(arith.remf, arith.remsi, arith.remui, None),
        operator.lt: OpForType(
            _make_fcmp(arith.CmpFPredicate.OLT),
            _make_icmp(arith.CmpIPredicate.slt),
            _make_icmp(arith.CmpIPredicate.ult),
            None,
        ),
        operator.le: OpForType(
            _make_fcmp(arith.CmpFPredicate.OLE),
            _make_icmp(arith.CmpIPredicate.sle),
            _make_icmp(arith.CmpIPredicate.ule),
            None,
        ),
        operator.gt: OpForType(
            _make_fcmp(arith.CmpFPredicate.OGT),
            _make_icmp(arith.CmpIPredicate.sgt),
            _make_icmp(arith.CmpIPredicate.ugt),
            None,
        ),
        operator.ge: OpForType(
            _make_fcmp(arith.CmpFPredicate.OGE),
            _make_icmp(arith.CmpIPredicate.sge),
            _make_icmp(arith.CmpIPredicate.uge),
            None,
        ),
        operator.eq: OpForType(
            _make_fcmp(arith.CmpFPredicate.OEQ),
            _make_icmp(arith.CmpIPredicate.eq),
            _make_icmp(arith.CmpIPredicate.eq),
            None,
        ),
        operator.ne: OpForType(
            _make_fcmp(arith.CmpFPredicate.UNE),
            _make_icmp(arith.CmpIPredicate.ne),
            _make_icmp(arith.CmpIPredicate.ne),
            None,
        ),
        operator.rshift: OpForType(
            None,
            arith.shrsi,
            arith.shrui,
            None,
        ),
        operator.irshift: OpForType(
            None,
            arith.shrsi,
            arith.shrui,
            None,
        ),
        operator.lshift: OpForType(
            None,
            arith.shli,
            arith.shli,
            None,
        ),
        operator.ilshift: OpForType(
            None,
            arith.shli,
            arith.shli,
            None,
        ),
    }


def _get_operator_mapping(fn) -> OpForType | None:
    info = _operator_mapping().get(fn)
    trace("info: %s", info)
    return info


def _get_operation_for_op_and_type(op, type, is_unsigned) -> tuple[OpForType, MaybeOp] | None:
    mapping = _get_operator_mapping(op)
    if not mapping:
        return None
    match type:
        case ir.FloatType():
            return mapping, mapping.float
        case ir.IntegerType():
            if type.is_signed:
                return mapping, mapping.signed_integer
            elif type.is_unsigned:
                return mapping, mapping.unsigned_integer
            else:
                # Default to signedness
                return mapping, mapping.unsigned_integer if is_unsigned else mapping.signed_integer
        case ir.ComplexType():
            return mapping, mapping.complex
        case _:
            raise ValueError(f"Unsupported type: {type}")


@lower(operator.pow, types.Number, types.Number, api="values")
@lower(operator.ipow, types.Number, types.Number, api="values")
@lower(pow, types.Number, types.Number, api="values")
def pow_cg(builder, signature, args):
    assert len(args) == 2, "pow_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs_mlir_type, rhs_mlir_type = lhs.type, rhs.type
    match (lhs_mlir_type, rhs_mlir_type):
        case (ir.FloatType(), ir.FloatType()):
            lhs = builder.mlir_convert(lhs, target_mlir_type)
            rhs = builder.mlir_convert(rhs, target_mlir_type)
            op = math_dialect.powf
        case (ir.FloatType(), ir.IntegerType()):
            # NVVM/libdevice only defines __nv_powi/__nv_powif with an i32 exponent.
            # Feeding `math.fpowi` an i64 exponent will lower to a call with the wrong ABI
            # (e.g. __nv_powi(f64, i64)) and can crash the kernel at runtime.
            if rhs_mlir_type.width != 32:
                rhs = convert(rhs, T.i32())
            op = math_dialect.fpowi
        case (ir.IntegerType(), ir.IntegerType()):
            rhs = builder.mlir_convert(rhs, lhs_mlir_type)
            op = ipowi
        case (ir.ComplexType(), ir.ComplexType()):
            op = complex_dialect.pow
        case (ir.ComplexType(), ir.IntegerType()):
            op = complex_dialect.powi
        case (ir.ComplexType(), ir.FloatType()):
            rhs = builder.mlir_convert(rhs, target_mlir_type)
            op = complex_dialect.pow
        case _:
            raise ValueError(f"Unsupported types: {lhs_mlir_type} and {rhs_mlir_type}")

    res = op(lhs, rhs)
    res = builder.mlir_convert(res, target_mlir_type)
    return res


def _convert_operand(value, source_type, target_type, target_mlir_type):
    signed = get_conversion_signedness(source_type, target_type)
    return lowering_utilities.convert(value, target_mlir_type, signed=signed)


def _bin_op_cg(op, builder, signature, args):
    assert len(args) == 2, "add_cg expects 2 arguments"
    lhs, rhs = args
    target_type, lhs_type, rhs_type = (signature.return_type, signature.args[0], signature.args[1])
    target_mlir_type = builder.get_mlir_type(target_type)

    # Ensure numbers whose MLIR types are signless end up with the correct
    # operation. Without this logic, two input values with two unsigned integer
    # types get a signed comparator operator and vice versa. Booleans lower to
    # i1 and must compare as unsigned, otherwise True (all-ones) orders below
    # False.
    def _is_unsigned_operand(operand_type):
        if isinstance(operand_type, types.Boolean):
            return True
        return isinstance(operand_type, types.Integer) and not operand_type.signed

    is_unsigned = _is_unsigned_operand(lhs_type) and _is_unsigned_operand(rhs_type)

    promoted_numba_type = None
    unified_type = None
    mixed_integer_boolean_comparison = op in (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    ) and (
        isinstance(lhs_type, types.Boolean)
        and isinstance(rhs_type, types.Integer)
        or isinstance(lhs_type, types.Integer)
        and isinstance(rhs_type, types.Boolean)
    )
    if mixed_integer_boolean_comparison or (
        isinstance(lhs_type, types.Integer) and isinstance(rhs_type, types.Integer)
    ):
        # Ask which promotion was actually resolved
        resolved = builder.context.typing_context.resolve_function_type(
            op, (lhs_type, rhs_type), {}
        )
        if resolved.args[0] == resolved.args[1]:
            # Comparisons: operands are reported already promoted to a common
            # type, and the return type is bool.
            promoted_numba_type = resolved.args[0]
        else:
            # Arithmetic: operands are reported unpromoted, and the promotion
            # is carried by the return type.
            promoted_numba_type = resolved.return_type
        assert promoted_numba_type is not None
        unified_type = builder.get_mlir_type(promoted_numba_type)

    # Handle lowered Python/numpy scalars alongside MLIR values
    # This can happen with module-level constants
    if not isinstance(lhs, ir.Value):
        # Convert numpy scalars to Python scalars
        if hasattr(lhs, "item"):
            lhs = lhs.item()
        lhs = lowering_utilities.constant(
            lhs, unified_type if promoted_numba_type is not None else target_mlir_type
        )
    if not isinstance(rhs, ir.Value):
        # Convert numpy scalars to Python scalars
        if hasattr(rhs, "item"):
            rhs = rhs.item()
        rhs = lowering_utilities.constant(
            rhs, unified_type if promoted_numba_type is not None else target_mlir_type
        )

    if promoted_numba_type is None:
        unified_type = lowering_utilities.numpy_implicit_type_promotion(lhs.type, rhs.type)

    trace("op: %s, target_mlir_type: %s", op, target_mlir_type)
    if found_op := _get_operation_for_op_and_type(op, unified_type, is_unsigned):
        info, op = found_op
        assert op is not None, "Expected operation"
        if info.cast_to_return_type:
            lhs = _convert_operand(lhs, lhs_type, target_type, target_mlir_type)
            rhs = _convert_operand(rhs, rhs_type, target_type, target_mlir_type)
        elif promoted_numba_type is not None:
            lhs = _convert_operand(lhs, lhs_type, promoted_numba_type, unified_type)
            rhs = _convert_operand(rhs, rhs_type, promoted_numba_type, unified_type)
        else:
            lhs, rhs = lowering_utilities.coerce_numpy_scalars_for_binary_op(lhs, rhs)
        if (
            isinstance(target_type, types.Integer)
            and target_type.signed
            and op in (arith.addi, arith.subi, arith.muli)
        ):
            # Match Numba: ignore signed overflow so index checks can fold.
            res = op(lhs, rhs, overflow_flags="nsw")
        else:
            res = op(lhs, rhs)
    else:
        raise ValueError(f"No operation found for {op=} and {target_mlir_type=}")

    res = builder.mlir_convert(res, target_mlir_type)
    return res


@lower(operator.sub, types.Number, types.Number, api="values")
@lower(operator.isub, types.Number, types.Number, api="values")
@lower(operator.sub, types.Boolean, types.Number, api="values")
@lower(operator.sub, types.Number, types.Boolean, api="values")
@lower(operator.sub, types.Boolean, types.Boolean, api="values")
@lower(operator.isub, types.Boolean, types.Number, api="values")
@lower(operator.isub, types.Number, types.Boolean, api="values")
@lower(operator.isub, types.Boolean, types.Boolean, api="values")
def sub_cg(builder, signature, args):
    return _bin_op_cg(operator.sub, builder, signature, args)


@lower(operator.mul, types.Number, types.Number, api="values")
@lower(operator.imul, types.Number, types.Number, api="values")
@lower(operator.mul, types.Number, types.Boolean, api="values")
@lower(operator.mul, types.Boolean, types.Number, api="values")
@lower(operator.mul, types.Boolean, types.Boolean, api="values")
def mul_cg(builder, signature, args):
    return _bin_op_cg(operator.mul, builder, signature, args)


@lower(operator.truediv, types.Number, types.Number, api="values")
@lower(operator.itruediv, types.Number, types.Number, api="values")
def truediv_cg(builder, signature, args):
    return _bin_op_cg(operator.truediv, builder, signature, args)


@lower(operator.ne, types.Number, types.Number, api="values")
@lower(operator.ne, types.Boolean, types.Boolean, api="values")
@lower(operator.ne, types.Boolean, types.Integer, api="values")
@lower(operator.ne, types.Integer, types.Boolean, api="values")
def ne_cg(builder, signature, args):
    return _bin_op_cg(operator.ne, builder, signature, args)


@lower(operator.eq, types.Number, types.Number, api="values")
@lower(operator.eq, types.Boolean, types.Boolean, api="values")
@lower(operator.eq, types.Boolean, types.Integer, api="values")
@lower(operator.eq, types.Integer, types.Boolean, api="values")
def eq_cg(builder, signature, args):
    return _bin_op_cg(operator.eq, builder, signature, args)


@lower(operator.lt, types.Number, types.Number, api="values")
@lower(operator.lt, types.Boolean, types.Boolean, api="values")
@lower(operator.lt, types.Boolean, types.Integer, api="values")
@lower(operator.lt, types.Integer, types.Boolean, api="values")
def lt_cg(builder, signature, args):
    return _bin_op_cg(operator.lt, builder, signature, args)


@lower(operator.le, types.Number, types.Number, api="values")
@lower(operator.le, types.Boolean, types.Boolean, api="values")
@lower(operator.le, types.Boolean, types.Integer, api="values")
@lower(operator.le, types.Integer, types.Boolean, api="values")
def le_cg(builder, signature, args):
    return _bin_op_cg(operator.le, builder, signature, args)


@lower(operator.gt, types.Number, types.Number, api="values")
@lower(operator.gt, types.Boolean, types.Boolean, api="values")
@lower(operator.gt, types.Boolean, types.Integer, api="values")
@lower(operator.gt, types.Integer, types.Boolean, api="values")
def gt_cg(builder, signature, args):
    return _bin_op_cg(operator.gt, builder, signature, args)


@lower(operator.ge, types.Number, types.Number, api="values")
@lower(operator.ge, types.Boolean, types.Boolean, api="values")
@lower(operator.ge, types.Boolean, types.Integer, api="values")
@lower(operator.ge, types.Integer, types.Boolean, api="values")
def ge_cg(builder, signature, args):
    return _bin_op_cg(operator.ge, builder, signature, args)


@lower(operator.add, types.Number, types.Number, api="values")
@lower(operator.iadd, types.Number, types.Number, api="values")
@lower(operator.add, types.Boolean, types.Number, api="values")
@lower(operator.add, types.Number, types.Boolean, api="values")
@lower(operator.add, types.Boolean, types.Boolean, api="values")
@lower(operator.iadd, types.Boolean, types.Number, api="values")
@lower(operator.iadd, types.Number, types.Boolean, api="values")
@lower(operator.iadd, types.Boolean, types.Boolean, api="values")
def iadd_cg(builder, signature, args):
    """
    This operator is weird. Sometimes, Numba resolves addition of
    two integers to operator.add instead of operator.iadd.
    To accommodate this, we handle both cases here, and distinguish
    based on the target type.
    """
    target_type = signature.return_type
    if isinstance(target_type, (types.Integer, types.Boolean)):
        return _bin_op_cg(operator.iadd, builder, signature, args)
    elif isinstance(target_type, (types.Float, types.Complex)):
        return _bin_op_cg(operator.add, builder, signature, args)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


@lower(operator.ior, types.Integer, types.Integer, api="values")
@lower(operator.ior, types.Boolean, types.Boolean, api="values")
@lower(operator.ior, types.Boolean, types.Integer, api="values")
@lower(operator.ior, types.Integer, types.Boolean, api="values")
def ior_cg(builder, signature, args):
    return _bin_op_cg(operator.ior, builder, signature, args)


@lower(operator.iand, types.Integer, types.Integer, api="values")
@lower(operator.iand, types.Boolean, types.Boolean, api="values")
@lower(operator.iand, types.Boolean, types.Integer, api="values")
@lower(operator.iand, types.Integer, types.Boolean, api="values")
def iand_cg(builder, signature, args):
    return _bin_op_cg(operator.iand, builder, signature, args)


@lower(operator.ixor, types.Integer, types.Integer, api="values")
@lower(operator.ixor, types.Boolean, types.Boolean, api="values")
@lower(operator.ixor, types.Boolean, types.Integer, api="values")
@lower(operator.ixor, types.Integer, types.Boolean, api="values")
def ixor_cg(builder, signature, args):
    return _bin_op_cg(operator.ixor, builder, signature, args)


@lower(numba_cuda_mlir.types.ptr, types.Array, api="values")
def pointer_array_cg(builder, signature, args):
    trace()
    assert len(args) == 1, "calling types.ptr() is only supported on arrays, pointers, and integers"
    array = args[0]
    return lowering_utilities.memref_data_pointer(array)


@lower(operator.sub, numba_cuda_mlir.types.ptr, types.Integer, api="values")
@lower(operator.sub, types.Integer, numba_cuda_mlir.types.ptr, api="values")
def pointer_sub_cg(builder, signature, args):
    trace()
    assert len(args) == 2, "pointer_sub_cg expects 2 arguments"
    lhs, rhs = args
    match lhs.type, rhs.type:
        case ir.IntegerType(), llvm.PointerType():
            I, P = lhs, rhs
        case llvm.PointerType(), ir.IntegerType():
            P, I = lhs, rhs
        case _:
            raise ValueError(f"Unsupported types: {lhs.type} and {rhs.type}")
    I = convert(I, T.i64())
    P = convert(P, llvm.PointerType.get())
    return lowering_utilities.llvm_ptr_add_bytes(P, i64_of(0) - I)


@lower(operator.add, numba_cuda_mlir.types.ptr, types.Integer, api="values")
@lower(operator.add, types.Integer, numba_cuda_mlir.types.ptr, api="values")
def pointer_add_cg(builder, signature, args):
    trace()
    assert len(args) == 2, "pointer_add_cg expects 2 arguments"
    lhs, rhs = args
    match lhs.type, rhs.type:
        case ir.IntegerType(), llvm.PointerType():
            I, P = lhs, rhs
        case llvm.PointerType(), ir.IntegerType():
            P, I = lhs, rhs
        case _:
            raise ValueError(f"Unsupported types: {lhs.type} and {rhs.type}")
    I = convert(I, T.i64())
    P = convert(P, llvm.PointerType.get())
    return lowering_utilities.llvm_ptr_add_bytes(P, I)


@lower(math.ceil, types.Number, api="values")
def math_ceil_cg(mlir_lower, signature, args):
    value = args[0]
    if _is_integer_type(value.type):
        result = value
    else:
        result = math_dialect.ceil(value)
    return mlir_lower.mlir_convert(result, mlir_lower.get_mlir_type(signature.return_type))


@lower(math.floor, types.Number, api="values")
def math_floor_cg(mlir_lower, signature, args):
    value = args[0]
    if _is_integer_type(value.type):
        result = value
    else:
        result = math_dialect.floor(value)
    return mlir_lower.mlir_convert(result, mlir_lower.get_mlir_type(signature.return_type))


@lower(math.trunc, types.Number, api="values")
def math_trunc_cg(mlir_lower, signature, args):
    value = args[0]
    if _is_integer_type(value.type):
        result = value
    else:
        result = math_dialect.trunc(value)
    return mlir_lower.mlir_convert(result, mlir_lower.get_mlir_type(signature.return_type))


@lower(math.sin, types.Number, api="values")
def math_sin_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.sin(value)
    return result


@lower(math.cos, types.Number, api="values")
def math_cos_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.cos(value)
    return result


@lower(math.tan, types.Number, api="values")
def math_tan_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.tan(value)
    return result


@lower(math.sqrt, types.Number, api="values")
def math_sqrt_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.sqrt(value)
    return result


@lower(math.exp, types.Number, api="values")
def math_exp_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.exp(value)
    return result


@lower(math.log, types.Number, api="values")
def math_log_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.log(value)
    return result


@lower(math.log2, types.Number, api="values")
def math_log2_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.log2(value)
    return result


@lower(math.log10, types.Number, api="values")
def math_log10_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.log10(value)
    return result


@lower(math.fabs, types.Number, api="values")
def math_fabs_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.absf(value)
    return result


@lower(math.isfinite, types.Number, api="values")
def math_isfinite_cg(mlir_lower, signature, args):
    assert len(args) == 1, "math_isfinite expects 1 argument"

    value = args[0]
    if _is_integer_type(value.type):
        # Integers are always finite
        result = arith.constant(T.bool(), True)
    else:
        result = math_dialect.isfinite(value)
    return result


@lower(math.isnan, types.Number, api="values")
def math_isnan_cg(mlir_lower, signature, args):
    assert len(args) == 1, "math_isnan expects 1 argument"

    value = args[0]
    if _is_integer_type(value.type):
        # Integers are never NaN
        result = arith.constant(T.bool(), False)
    else:
        result = math_dialect.isnan(value)
    return result


@lower(math.exp2, types.Number, api="values")
def math_exp2_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.exp2(value)
    return result


@lower(math.tanh, types.Number, api="values")
def math_tanh_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.tanh(value)
    return result


@lower(math.sinh, types.Number, api="values")
def math_sinh_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.sinh(value)
    return result


@lower(math.cosh, types.Number, api="values")
def math_cosh_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.cosh(value)
    return result


@lower(math.asin, types.Number, api="values")
def math_asin_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.asin(value)
    return result


@lower(math.acos, types.Number, api="values")
def math_acos_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.acos(value)
    return result


@lower(math.atan, types.Number, api="values")
def math_atan_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.atan(value)
    return result


@lower(math.asinh, types.Number, api="values")
def math_asinh_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.asinh(value)
    return result


@lower(math.acosh, types.Number, api="values")
def math_acosh_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.acosh(value)
    return result


@lower(math.atanh, types.Number, api="values")
def math_atanh_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.atanh(value)
    return result


@lower(math.isinf, types.Number, api="values")
def math_isinf_cg(mlir_lower, signature, args):
    assert len(args) == 1, "math_isinf expects 1 argument"
    value = args[0]
    if _is_integer_type(value.type):
        # Integers are never infinity
        result = arith.constant(T.bool(), False)
    else:
        result = math_dialect.isinf(value)
    return result


@lower(math.copysign, types.Number, types.Number, api="values")
def math_copysign_cg(mlir_lower, signature, args):
    assert len(args) == 2, "math_copysign expects 2 arguments"
    x = _ensure_float(args[0], signature.args[0])
    y = _ensure_float(args[1], signature.args[1])
    # Ensure both have the same type
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    result = math_dialect.copysign(x, y)
    return result


@lower(math.atan2, types.Number, types.Number, api="values")
def math_atan2_cg(mlir_lower, signature, args):
    assert len(args) == 2, "math_atan2 expects 2 arguments"
    y = _ensure_float(args[0], signature.args[0])
    x = _ensure_float(args[1], signature.args[1])
    # Ensure both have the same type
    unified_type = lowering_utilities.numpy_implicit_type_promotion(y.type, x.type)
    y = convert(y, unified_type)
    x = convert(x, unified_type)
    result = math_dialect.atan2(y, x)
    return result


@lower(math.hypot, types.Number, types.Number, api="values")
def math_hypot_cg(mlir_lower, signature, args):
    """hypot(x, y) = sqrt(x^2 + y^2), without overflow or underflow"""
    assert len(args) == 2, "math_hypot expects 2 arguments"
    x = _ensure_float(args[0], signature.args[0])
    y = _ensure_float(args[1], signature.args[1])
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    # libdevice's hypot is exact; sqrt(x * x + y * y) gives inf for
    # math.hypot(1e200, 1e200)
    result = _call_libdevice_binary(mlir_lower, x, y, "__nv_hypot", "__nv_hypotf")
    return convert(result, unified_type)


@lower(math.log1p, types.Number, api="values")
def math_log1p_cg(mlir_lower, signature, args):
    assert len(args) == 1, "math_log1p expects 1 argument"
    x = _ensure_float(args[0], signature.args[0])
    result = math_dialect.log1p(x)
    return result


def _float_divmod(mlir_lower, x: ir.Value, y: ir.Value) -> tuple[ir.Value, ir.Value]:
    """Python's float floor division and modulo (CPython's _float_div_mod).

    They start from libdevice's exact fmod: x - floor(x / y) * y is off
    whenever x / y rounds, e.g. 1.0 % 0.1 would give -5.6e-17 and 1.0 // 0.1
    10.0, instead of 0.09999999999999995 and 9.0.
    """
    ty = x.type
    zero = lowering_utilities.constant(0.0, ty)
    one = lowering_utilities.constant(1.0, ty)
    half = lowering_utilities.constant(0.5, ty)
    mod = convert(_call_libdevice_binary(mlir_lower, x, y, "__nv_fmod", "__nv_fmodf"), ty)
    div = arith.divf(arith.subf(x, mod), y)
    # A nonzero remainder takes the sign of the divisor
    mod_nonzero = arith.cmpf(arith.CmpFPredicate.UNE, mod, zero)
    y_negative = arith.cmpf(arith.CmpFPredicate.OLT, y, zero)
    mod_negative = arith.cmpf(arith.CmpFPredicate.OLT, mod, zero)
    adjust = arith.andi(mod_nonzero, arith.xori(y_negative, mod_negative))
    mod = arith.select(adjust, arith.addf(mod, y), mod)
    div = arith.select(adjust, arith.subf(div, one), div)
    mod = arith.select(mod_nonzero, mod, math_dialect.copysign(zero, y))
    # Snap the quotient to the nearest integral value
    floordiv = math_dialect.floor(div)
    round_up = arith.cmpf(arith.CmpFPredicate.OGT, arith.subf(div, floordiv), half)
    floordiv = arith.select(round_up, arith.addf(floordiv, one), floordiv)
    quotient = arith.divf(x, y)
    div_nonzero = arith.cmpf(arith.CmpFPredicate.UNE, div, zero)
    floordiv = arith.select(div_nonzero, floordiv, math_dialect.copysign(zero, quotient))
    # A zero divisor still gives inf or nan
    y_zero = arith.cmpf(arith.CmpFPredicate.OEQ, y, zero)
    floordiv = arith.select(y_zero, quotient, floordiv)
    return floordiv, mod


@lower(operator.mod, types.Number, types.Number, api="values")
@lower(operator.imod, types.Number, types.Number, api="values")
def mod_cg(builder, signature, args):
    assert len(args) == 2, "mod_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)

    match target_mlir_type:
        case ir.IntegerType():
            if target_type.signed:
                # Python floored modulo: the result takes the sign of the
                # divisor. arith.remsi truncates toward zero, so add the
                # divisor when the remainder is nonzero and its sign
                # disagrees with the divisor's ((rem ^ rhs) < 0).
                rem = arith.remsi(lhs, rhs)
                zero = lowering_utilities.constant(0, rem.type)
                sign_mismatch = arith.cmpi(arith.CmpIPredicate.slt, arith.xori(rem, rhs), zero)
                rem_nonzero = arith.cmpi(arith.CmpIPredicate.ne, rem, zero)
                needs_fix = arith.andi(sign_mismatch, rem_nonzero)
                result = arith.addi(rem, arith.select(needs_fix, rhs, zero))
            else:
                result = arith.remui(lhs, rhs)
        case ir.FloatType():
            _, result = _float_divmod(builder, lhs, rhs)
        case _:
            raise NotImplementedError(f"mod not implemented for {target_mlir_type=}")
    return result


@lower(operator.lshift, types.Integer, types.Integer, api="values")
@lower(operator.ilshift, types.Integer, types.Integer, api="values")
def lshift_cg(builder, signature, args):
    assert len(args) == 2, "lshift_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)
    result = arith.shli(lhs, rhs)
    return result


@lower(operator.rshift, types.Integer, types.Integer, api="values")
@lower(operator.irshift, types.Integer, types.Integer, api="values")
def rshift_cg(builder, signature, args):
    assert len(args) == 2, "rshift_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)
    # Use unsigned right shift for unsigned types, signed for signed types
    if target_type.signed:
        result = arith.shrsi(lhs, rhs)
    else:
        result = arith.shrui(lhs, rhs)
    return result


@lower(operator.irshift, types.Integer, types.Integer, api="values")
def irshift_cg(builder, signature, args):
    """In-place right shift (>>=). In SSA form, this is the same as regular rshift."""
    return rshift_cg(builder, signature, args)


@lower(operator.ilshift, types.Integer, types.Integer, api="values")
def ilshift_cg(builder, signature, args):
    """In-place left shift (<<=). In SSA form, this is the same as regular lshift."""
    return lshift_cg(builder, signature, args)


@lower(operator.and_, types.Integer, types.Integer, api="values")
@lower(operator.and_, types.Boolean, types.Boolean, api="values")
@lower(operator.and_, types.Boolean, types.Integer, api="values")
@lower(operator.and_, types.Integer, types.Boolean, api="values")
def and_cg(builder, signature, args):
    assert len(args) == 2, "and_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)
    result = arith.andi(lhs, rhs)
    return result


@lower(operator.or_, types.Integer, types.Integer, api="values")
@lower(operator.or_, types.Boolean, types.Boolean, api="values")
@lower(operator.or_, types.Boolean, types.Integer, api="values")
@lower(operator.or_, types.Integer, types.Boolean, api="values")
def or_cg(builder, signature, args):
    assert len(args) == 2, "or_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)
    result = arith.ori(lhs, rhs)
    return result


@lower(operator.xor, types.Integer, types.Integer, api="values")
@lower(operator.xor, types.Boolean, types.Boolean, api="values")
@lower(operator.xor, types.Boolean, types.Integer, api="values")
@lower(operator.xor, types.Integer, types.Boolean, api="values")
def xor_cg(builder, signature, args):
    assert len(args) == 2, "xor_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)
    result = arith.xori(lhs, rhs)
    return result


@lower(operator.invert, types.Integer, api="values")
@lower(operator.invert, types.Boolean, api="values")
def invert_cg(builder, signature, args):
    """Bitwise NOT: xor with all-ones (~True is False for booleans)."""
    assert len(args) == 1, "invert_cg expects 1 argument"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    source_type = signature.args[0]
    unsigned = isinstance(source_type, types.Integer) and not source_type.signed
    if unsigned:
        # Invert at the input width; widening first changes unsigned results.
        operand = args[0]
    else:
        operand = _convert_operand(args[0], signature.args[0], target_type, target_mlir_type)
    fill = 1 if isinstance(target_type, types.Boolean) else -1
    all_ones = lowering_utilities.constant(fill, operand.type)
    result = arith.xori(operand, all_ones)
    if unsigned:
        result = _convert_operand(result, source_type, target_type, target_mlir_type)
    return result


@lower(operator.floordiv, types.Number, types.Number, api="values")
@lower(operator.ifloordiv, types.Number, types.Number, api="values")
def floordiv_cg(builder, signature, args):
    assert len(args) == 2, "floordiv_cg expects 2 arguments"
    target_type = signature.return_type
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = _convert_operand(lhs, signature.args[0], target_type, target_mlir_type)
    rhs = _convert_operand(rhs, signature.args[1], target_type, target_mlir_type)

    match target_mlir_type:
        case ir.FloatType():
            result, _ = _float_divmod(builder, lhs, rhs)
        case ir.IntegerType():
            # Result type selects signed or unsigned floor division. Operand
            # conversion signedness was handled independently above.
            if target_type.signed:
                result = arith.floordivsi(lhs, rhs)
            else:
                # For unsigned, regular division is the same as floor division
                result = arith.divui(lhs, rhs)
        case _:
            raise ValueError(f"Unsupported type for floor division: {target_mlir_type}")

    result = builder.mlir_convert(result, target_mlir_type)
    return result


def _get_attr_from_value(value: float | int, element_type: ir.Type):
    match element_type:
        case ir.F32Type():
            return ir.FloatAttr.get_f32(value)
        case ir.F64Type():
            return ir.FloatAttr.get_f64(value)
        case ir.IndexType():
            return ir.IntegerAttr.get(element_type, int(value))
        case _:
            raise ValueError(f"Unsupported element type: {element_type}")


@lower(complex, types.Number, types.Number, api="values")
def complex_cg(mlir_lower, signature, args):
    target_type = signature.return_type
    mlir_target_type = mlir_lower.get_mlir_type(target_type)
    assert len(args) == 2, "complex_cg expects 2 arguments"
    real, imag = args
    element_type = mlir_target_type.element_type

    # Check if args are constants (float/int) or variables
    if isinstance(real, (float, int)) and isinstance(imag, (float, int)):
        # Create constant complex number
        real_attr = _get_attr_from_value(real, element_type)
        imag_attr = _get_attr_from_value(imag, element_type)
        const_attr = ir.ArrayAttr.get([real_attr, imag_attr])
        complex_const_op = complex_dialect.constant(
            complex=mlir_target_type,
            value=const_attr,
        )
        return complex_const_op
    else:
        # Create complex from runtime values
        real_val = (
            real if isinstance(real, ir.Value) else lowering_utilities.constant(real, element_type)
        )
        imag_val = (
            imag if isinstance(imag, ir.Value) else lowering_utilities.constant(imag, element_type)
        )

        # Convert to the target element type if needed
        real_val = lowering_utilities.convert(real_val, element_type)
        imag_val = lowering_utilities.convert(imag_val, element_type)

        # Create complex value from real and imaginary parts
        complex_val = complex_dialect.create_(
            complex=mlir_target_type,
            real=real_val,
            imaginary=imag_val,
        )
        return complex_val


@registry.lower_getattr(types.Complex, "real")
def complex_real_getattr(
    context,
    builder,
    target: numba_ir.Var,
    value: numba_ir.Var,
):
    """Lower .real attribute access for complex numbers."""
    trace("Lowering .real for complex type")
    complex_value = builder.load_var(value)
    real_part = complex_dialect.re(complex_value)
    builder.store_var(target, real_part)


@registry.lower_getattr(types.Complex, "imag")
def complex_imag_getattr(
    context,
    builder,
    target: numba_ir.Var,
    value: numba_ir.Var,
):
    """Lower .imag attribute access for complex numbers."""
    trace("Lowering .imag for complex type")
    complex_value = builder.load_var(value)
    imag_part = complex_dialect.im(complex_value)
    builder.store_var(target, imag_part)


@lower(math.frexp, types.Float, api="values")
def math_frexp_cg(mlir_lower, signature, args):
    """
    Lower math.frexp(x) using LLVM intrinsic.
    Returns (mantissa, exponent) where mantissa is float and exponent is int32.
    """
    assert len(args) == 1, "math.frexp expects 1 argument"

    value = args[0]
    float_type = value.type

    # llvm.intr.frexp returns struct { float, i32 }
    result_type = llvm.StructType.get_literal([float_type, T.i32()])
    result = llvm.intr_frexp(result_type, value)

    # Extract mantissa and exponent
    mantissa = llvm.extractvalue(float_type, result, [0])
    exp = llvm.extractvalue(T.i32(), result, [1])

    return (mantissa, exp)


@lower(math.ldexp, types.Float, types.Integer, api="values")
def math_ldexp_cg(mlir_lower, signature, args):
    """
    Lower math.ldexp(x, exp) using LLVM intrinsic.
    Returns x * 2^exp.
    """
    assert len(args) == 2, "math.ldexp expects 2 arguments"

    value = args[0]
    exp = args[1]

    # ldexp expects i32 exponent
    exp = convert(exp, T.i32())

    result = llvm.intr_ldexp(value.type, value, exp)
    return result


@lower(math.expm1, types.Number, api="values")
def math_expm1_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.expm1(value)
    return result


@lower(math.degrees, types.Number, api="values")
def math_degrees_cg(mlir_lower, signature, args):
    """Convert radians to degrees: x * (180 / pi)"""
    value = _ensure_float(args[0], signature.args[0])
    # 180 / pi = 57.29577951308232
    factor = lowering_utilities.constant(57.29577951308232, value.type)
    result = arith.mulf(value, factor)
    return result


@lower(math.radians, types.Number, api="values")
def math_radians_cg(mlir_lower, signature, args):
    """Convert degrees to radians: x * (pi / 180)"""
    value = _ensure_float(args[0], signature.args[0])
    # pi / 180 = 0.017453292519943295
    factor = lowering_utilities.constant(0.017453292519943295, value.type)
    result = arith.mulf(value, factor)
    return result


@lower(math.erf, types.Number, api="values")
def math_erf_cg(mlir_lower, signature, args):
    value = _ensure_float(args[0], signature.args[0])
    result = math_dialect.erf(value)
    return result


@lower(math.erfc, types.Number, api="values")
def math_erfc_cg(mlir_lower, signature, args):
    """erfc(x) = complementary error function, using libdevice for precision"""
    value = _ensure_float(args[0], signature.args[0])
    result = _call_libdevice_unary(mlir_lower, value, "__nv_erfc", "__nv_erfcf")
    return result


@lower(math.gamma, types.Number, api="values")
def math_gamma_cg(mlir_lower, signature, args):
    """Lower math.gamma using libdevice tgamma"""
    value = _ensure_float(args[0], signature.args[0])
    result = _call_libdevice_unary(mlir_lower, value, "__nv_tgamma", "__nv_tgammaf")
    return result


@lower(math.lgamma, types.Number, api="values")
def math_lgamma_cg(mlir_lower, signature, args):
    """Lower math.lgamma using libdevice lgamma"""
    value = _ensure_float(args[0], signature.args[0])
    result = _call_libdevice_unary(mlir_lower, value, "__nv_lgamma", "__nv_lgammaf")
    return result


@lower(math.fmod, types.Number, types.Number, api="values")
def math_fmod_cg(mlir_lower, signature, args):
    """fmod(x, y) - floating point remainder with sign of x (truncated division)"""
    assert len(args) == 2, "math_fmod expects 2 arguments"
    x = _ensure_float(args[0], signature.args[0])
    y = _ensure_float(args[1], signature.args[1])
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    # libdevice computes the remainder exactly; x - trunc(x / y) * y is off
    # whenever x / y rounds, and gives nan for an infinite y
    result = _call_libdevice_binary(mlir_lower, x, y, "__nv_fmod", "__nv_fmodf")
    return convert(result, unified_type)


@lower(math.remainder, types.Number, types.Number, api="values")
def math_remainder_cg(mlir_lower, signature, args):
    """IEEE 754 remainder: x - round(x/y) * y"""
    assert len(args) == 2, "math_remainder expects 2 arguments"
    x = _ensure_float(args[0], signature.args[0])
    y = _ensure_float(args[1], signature.args[1])
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    # libdevice computes the IEEE remainder exactly; x - roundeven(x / y) * y
    # is off whenever x / y rounds, and gives nan for an infinite y
    result = _call_libdevice_binary(mlir_lower, x, y, "__nv_remainder", "__nv_remainderf")
    return convert(result, unified_type)


@lower(math.pow, types.Number, types.Number, api="values")
def math_pow_cg(mlir_lower, signature, args):
    """math.pow(x, y) - x raised to power y"""
    assert len(args) == 2, "math_pow expects 2 arguments"
    x = _ensure_float(args[0], signature.args[0])
    y = _ensure_float(args[1], signature.args[1])
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    result = math_dialect.powf(x, y)
    return result


@lower(math.nextafter, types.Number, types.Number, api="values")
def math_nextafter_cg(mlir_lower, signature, args):
    """nextafter(x, y) - next representable floating-point value after x towards y"""
    assert len(args) == 2, "math_nextafter expects 2 arguments"
    x = _ensure_float(args[0], signature.args[0])
    y = _ensure_float(args[1], signature.args[1])
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    result = _call_libdevice_binary(mlir_lower, x, y, "__nv_nextafter", "__nv_nextafterf")
    return result


@lower(math.modf, types.Float, api="values")
def math_modf_cg(mlir_lower, signature, args):
    """
    modf(x) returns (fractional_part, integer_part).
    fractional_part has the same sign as x.
    Special cases:
    - modf(inf) = (0.0, inf)
    - modf(-inf) = (-0.0, -inf)
    - modf(nan) = (nan, nan)
    """
    assert len(args) == 1, "math_modf expects 1 argument"

    value = args[0]
    float_type = value.type

    # Integer part: trunc(x)
    int_part = math_dialect.trunc(value)
    # Fractional part: x - trunc(x)
    frac_part_normal = arith.subf(value, int_part)

    # Handle infinity: for ±inf, fractional part should be ±0.0, not nan
    # Check if value is infinite
    is_inf = math_dialect.isinf(value)
    # Create ±0.0 with same sign as value
    zero = lowering_utilities.constant(0.0, float_type)
    signed_zero = math_dialect.copysign(zero, value)
    # Select: if infinite use signed_zero, else use normal fractional part
    frac_part = arith.select(is_inf, signed_zero, frac_part_normal)

    return (frac_part, int_part)
