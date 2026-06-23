# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import functools
from typing import Callable
from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir.lowering_utilities import (
    coerce_numpy_scalars_for_binary_op,
    numpy_implicit_type_promotion,
    convert,
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
                case 32:
                    return T.f32()
                case 64:
                    return T.f64()
                case _:
                    raise ValueError(f"Unsupported integer type width: {ty.width}")
        case ir.FloatType():
            return ty
        case _:
            raise ValueError(f"Unsupported type: {ty}")


def _cast_to_float_of_same_size(value: ir.Value) -> ir.Value:
    float_type = _get_float_of_same_size_as(value.type)
    return lowering_utilities.convert(value, float_type)


def _ensure_float(value: ir.Value) -> ir.Value:
    """Ensure value is floating-point, converting integers to floats if needed."""
    if isinstance(value.type, ir.IntegerType) or isinstance(value.type, ir.IndexType):
        return _cast_to_float_of_same_size(value)
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
            _make_fcmp(arith.CmpFPredicate.ONE),
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


def _get_operation_for_op_and_type(op, type) -> tuple[OpForType, MaybeOp] | None:
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
                return mapping, mapping.signed_integer
        case ir.ComplexType():
            return mapping, mapping.complex
        case _:
            raise ValueError(f"Unsupported type: {type}")


@lower(operator.pow, types.Number, types.Number)
@lower(operator.ipow, types.Number, types.Number)
@lower(pow, types.Number, types.Number)
def pow_cg(builder, target, args, kwargs):
    assert not kwargs, "pow_cg does not accept any keyword arguments"
    assert len(args) == 2, "pow_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs, rhs = builder.load_var(lhs), builder.load_var(rhs)
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
    builder.store_var(target, res)


def _bin_op_cg(op, builder, target, args, kwargs):
    assert not kwargs, "add_cg does not accept any keyword arguments"
    assert len(args) == 2, "add_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs, rhs = builder.load_var(lhs), builder.load_var(rhs)

    # Handle cases where load_var returns Python/numpy scalars instead of MLIR values
    # This can happen with module-level constants
    if not isinstance(lhs, ir.Value):
        # Convert numpy scalars to Python scalars
        if hasattr(lhs, "item"):
            lhs = lhs.item()
        lhs = lowering_utilities.constant(lhs, target_mlir_type)
    if not isinstance(rhs, ir.Value):
        # Convert numpy scalars to Python scalars
        if hasattr(rhs, "item"):
            rhs = rhs.item()
        rhs = lowering_utilities.constant(rhs, target_mlir_type)

    unified_type = lowering_utilities.numpy_implicit_type_promotion(lhs.type, rhs.type)

    trace("op: %s, target_mlir_type: %s", op, target_mlir_type)
    if found_op := _get_operation_for_op_and_type(op, unified_type):
        info, op = found_op
        assert op is not None, "Expected operation"
        if info.cast_to_return_type:
            lhs = lowering_utilities.convert(lhs, target_mlir_type)
            rhs = lowering_utilities.convert(rhs, target_mlir_type)
        else:
            lhs, rhs = coerce_numpy_scalars_for_binary_op(lhs, rhs)
        res = op(lhs, rhs)
    else:
        raise ValueError(f"No operation found for {op=} and {target_mlir_type=}")

    res = builder.mlir_convert(res, target_mlir_type)
    builder.store_var(target, res)


@lower(operator.sub, types.Number, types.Number)
@lower(operator.isub, types.Number, types.Number)
@lower(operator.sub, types.Boolean, types.Number)
@lower(operator.sub, types.Number, types.Boolean)
@lower(operator.sub, types.Boolean, types.Boolean)
@lower(operator.isub, types.Boolean, types.Number)
@lower(operator.isub, types.Number, types.Boolean)
@lower(operator.isub, types.Boolean, types.Boolean)
def sub_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.sub, builder, target, args, kwargs)


@lower(operator.mul, types.Number, types.Number)
@lower(operator.imul, types.Number, types.Number)
@lower(operator.mul, types.Number, types.Boolean)
@lower(operator.mul, types.Boolean, types.Number)
@lower(operator.mul, types.Boolean, types.Boolean)
def mul_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.mul, builder, target, args, kwargs)


@lower(operator.truediv, types.Number, types.Number)
@lower(operator.itruediv, types.Number, types.Number)
def truediv_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.truediv, builder, target, args, kwargs)


@lower(operator.ne, types.Number, types.Number)
def ne_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.ne, builder, target, args, kwargs)


@lower(operator.eq, types.Number, types.Number)
def eq_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.eq, builder, target, args, kwargs)


@lower(operator.lt, types.Number, types.Number)
def lt_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.lt, builder, target, args, kwargs)


@lower(operator.le, types.Number, types.Number)
def le_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.le, builder, target, args, kwargs)


@lower(operator.gt, types.Number, types.Number)
def gt_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.gt, builder, target, args, kwargs)


@lower(operator.ge, types.Number, types.Number)
def ge_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.ge, builder, target, args, kwargs)


@lower(operator.add, types.Number, types.Number)
@lower(operator.iadd, types.Number, types.Number)
@lower(operator.add, types.Boolean, types.Number)
@lower(operator.add, types.Number, types.Boolean)
@lower(operator.add, types.Boolean, types.Boolean)
@lower(operator.iadd, types.Boolean, types.Number)
@lower(operator.iadd, types.Number, types.Boolean)
@lower(operator.iadd, types.Boolean, types.Boolean)
def iadd_cg(builder, target, args, kwargs):
    """
    This operator is weird. Sometimes, Numba resolves addition of
    two integers to operator.add instead of operator.iadd.
    To accommodate this, we handle both cases here, and distinguish
    based on the target type.
    """
    target_type = builder.get_numba_type(target.name)
    if isinstance(target_type, (types.Integer, types.Boolean)):
        return _bin_op_cg(operator.iadd, builder, target, args, kwargs)
    elif isinstance(target_type, (types.Float, types.Complex)):
        return _bin_op_cg(operator.add, builder, target, args, kwargs)
    else:
        raise ValueError(f"Unsupported target type: {target_type}")


@lower(operator.ior, types.Integer, types.Integer)
def ior_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.ior, builder, target, args, kwargs)


@lower(operator.iand, types.Integer, types.Integer)
def iand_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.iand, builder, target, args, kwargs)


@lower(operator.ixor, types.Integer, types.Integer)
def ixor_cg(builder, target, args, kwargs):
    return _bin_op_cg(operator.ixor, builder, target, args, kwargs)


@lower(numba_cuda_mlir.types.ptr, types.Array)
def pointer_array_cg(builder, target, args, kwargs):
    trace()
    assert len(args) == 1 and not kwargs, (
        "calling types.ptr() is only supported on arrays, pointers, and integers"
    )
    array = builder.load_var(args[0])
    result = lowering_utilities.memref_data_pointer_as_index(array)
    result = arith.index_cast(T.i64(), result)
    result = llvm.inttoptr(res=llvm.PointerType.get(), arg=result)
    builder.store_var(target, result)


@lower(operator.sub, numba_cuda_mlir.types.ptr, types.Integer)
@lower(operator.sub, types.Integer, numba_cuda_mlir.types.ptr)
def pointer_sub_cg(builder, target, args, kwargs):
    trace()
    assert not kwargs, "pointer_sub_cg does not accept any keyword arguments"
    assert len(args) == 2, "pointer_sub_cg expects 2 arguments"
    lhs, rhs = builder.load_vars(args)
    match lhs.type, rhs.type:
        case ir.IntegerType(), llvm.PointerType():
            I, P = lhs, rhs
        case llvm.PointerType(), ir.IntegerType():
            P, I = lhs, rhs
        case _:
            raise ValueError(f"Unsupported types: {lhs.type} and {rhs.type}")
    I = convert(I, T.i64())
    P = convert(P, llvm.PointerType.get())
    P = llvm.ptrtoint(res=T.i64(), arg=P)
    P -= I
    P = llvm.inttoptr(res=llvm.PointerType.get(), arg=P)
    builder.store_var(target, P)


@lower(operator.add, numba_cuda_mlir.types.ptr, types.Integer)
@lower(operator.add, types.Integer, numba_cuda_mlir.types.ptr)
def pointer_add_cg(builder, target, args, kwargs):
    trace()
    assert not kwargs, "pointer_add_cg does not accept any keyword arguments"
    assert len(args) == 2, "pointer_add_cg expects 2 arguments"
    lhs, rhs = builder.load_vars(args)
    match lhs.type, rhs.type:
        case ir.IntegerType(), llvm.PointerType():
            I, P = lhs, rhs
        case llvm.PointerType(), ir.IntegerType():
            P, I = lhs, rhs
        case _:
            raise ValueError(f"Unsupported types: {lhs.type} and {rhs.type}")
    I = convert(I, T.i64())
    P = convert(P, llvm.PointerType.get())
    P = llvm.ptrtoint(res=T.i64(), arg=P)
    P += I
    P = llvm.inttoptr(res=llvm.PointerType.get(), arg=P)
    builder.store_var(target, P)


@lower(math.ceil, types.Number)
def math_ceil_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_ceil_intrinsic does not accept any keyword arguments"
    value = mlir_lower.load_var(args[0])
    if _is_integer_type(value.type):
        # ceil of an integer is the integer itself, but convert to float for return type
        result = _cast_to_float_of_same_size(value)
    else:
        result = math_dialect.ceil(value)
    mlir_lower.store_var(target, result)


@lower(math.floor, types.Number)
def math_floor_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_floor does not accept any keyword arguments"
    value = mlir_lower.load_var(args[0])
    if _is_integer_type(value.type):
        # floor of an integer is the integer itself, but convert to float for return type
        result = _cast_to_float_of_same_size(value)
    else:
        result = math_dialect.floor(value)
    mlir_lower.store_var(target, result)


@lower(math.trunc, types.Number)
def math_trunc_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_trunc does not accept any keyword arguments"
    value = mlir_lower.load_var(args[0])
    if _is_integer_type(value.type):
        # trunc of an integer is the integer itself, but convert to float for return type
        result = _cast_to_float_of_same_size(value)
    else:
        result = math_dialect.trunc(value)
    mlir_lower.store_var(target, result)


@lower(math.sin, types.Number)
def math_sin_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_sin does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.sin(value)
    mlir_lower.store_var(target, result)


@lower(math.cos, types.Number)
def math_cos_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_cos does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.cos(value)
    mlir_lower.store_var(target, result)


@lower(math.tan, types.Number)
def math_tan_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_tan does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.tan(value)
    mlir_lower.store_var(target, result)


@lower(math.sqrt, types.Number)
def math_sqrt_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_sqrt does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.sqrt(value)
    mlir_lower.store_var(target, result)


@lower(math.exp, types.Number)
def math_exp_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_exp does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.exp(value)
    mlir_lower.store_var(target, result)


@lower(math.log, types.Number)
def math_log_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_log does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.log(value)
    mlir_lower.store_var(target, result)


@lower(math.log2, types.Number)
def math_log2_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_log2 does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.log2(value)
    mlir_lower.store_var(target, result)


@lower(math.log10, types.Number)
def math_log10_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_log10 does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.log10(value)
    mlir_lower.store_var(target, result)


@lower(math.fabs, types.Number)
def math_fabs_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_fabs does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.absf(value)
    mlir_lower.store_var(target, result)


@lower(math.isfinite, types.Number)
def math_isfinite_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_isfinite does not accept any keyword arguments"
    assert len(args) == 1, "math_isfinite expects 1 argument"

    value = mlir_lower.load_var(args[0])
    if _is_integer_type(value.type):
        # Integers are always finite
        result = arith.constant(T.bool(), True)
    else:
        result = math_dialect.isfinite(value)
    mlir_lower.store_var(target, result)


@lower(math.isnan, types.Number)
def math_isnan_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_isnan does not accept any keyword arguments"
    assert len(args) == 1, "math_isnan expects 1 argument"

    value = mlir_lower.load_var(args[0])
    if _is_integer_type(value.type):
        # Integers are never NaN
        result = arith.constant(T.bool(), False)
    else:
        result = math_dialect.isnan(value)
    mlir_lower.store_var(target, result)


@lower(math.exp2, types.Number)
def math_exp2_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_exp2 does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.exp2(value)
    mlir_lower.store_var(target, result)


@lower(math.tanh, types.Number)
def math_tanh_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_tanh does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.tanh(value)
    mlir_lower.store_var(target, result)


@lower(math.sinh, types.Number)
def math_sinh_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_sinh does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.sinh(value)
    mlir_lower.store_var(target, result)


@lower(math.cosh, types.Number)
def math_cosh_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_cosh does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.cosh(value)
    mlir_lower.store_var(target, result)


@lower(math.asin, types.Number)
def math_asin_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_asin does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.asin(value)
    mlir_lower.store_var(target, result)


@lower(math.acos, types.Number)
def math_acos_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_acos does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.acos(value)
    mlir_lower.store_var(target, result)


@lower(math.atan, types.Number)
def math_atan_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_atan does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.atan(value)
    mlir_lower.store_var(target, result)


@lower(math.asinh, types.Number)
def math_asinh_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_asinh does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.asinh(value)
    mlir_lower.store_var(target, result)


@lower(math.acosh, types.Number)
def math_acosh_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_acosh does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.acosh(value)
    mlir_lower.store_var(target, result)


@lower(math.atanh, types.Number)
def math_atanh_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_atanh does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.atanh(value)
    mlir_lower.store_var(target, result)


@lower(math.isinf, types.Number)
def math_isinf_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_isinf does not accept any keyword arguments"
    assert len(args) == 1, "math_isinf expects 1 argument"
    value = mlir_lower.load_var(args[0])
    if _is_integer_type(value.type):
        # Integers are never infinity
        result = arith.constant(T.bool(), False)
    else:
        result = math_dialect.isinf(value)
    mlir_lower.store_var(target, result)


@lower(math.copysign, types.Number, types.Number)
def math_copysign_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_copysign does not accept any keyword arguments"
    assert len(args) == 2, "math_copysign expects 2 arguments"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    y = _ensure_float(mlir_lower.load_var(args[1]))
    # Ensure both have the same type
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    result = math_dialect.copysign(x, y)
    mlir_lower.store_var(target, result)


@lower(math.atan2, types.Number, types.Number)
def math_atan2_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_atan2 does not accept any keyword arguments"
    assert len(args) == 2, "math_atan2 expects 2 arguments"
    y = _ensure_float(mlir_lower.load_var(args[0]))
    x = _ensure_float(mlir_lower.load_var(args[1]))
    # Ensure both have the same type
    unified_type = lowering_utilities.numpy_implicit_type_promotion(y.type, x.type)
    y = convert(y, unified_type)
    x = convert(x, unified_type)
    result = math_dialect.atan2(y, x)
    mlir_lower.store_var(target, result)


@lower(math.hypot, types.Number, types.Number)
def math_hypot_cg(mlir_lower, target, args, kwargs):
    """hypot(x, y) = sqrt(x^2 + y^2), computed in float64 for precision"""
    assert not kwargs, "math_hypot does not accept any keyword arguments"
    assert len(args) == 2, "math_hypot expects 2 arguments"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    y = _ensure_float(mlir_lower.load_var(args[1]))
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    # Compute in float64 for better precision (matches numpy behavior)
    original_type = unified_type
    if unified_type == T.f32():
        x = convert(x, T.f64())
        y = convert(y, T.f64())
    # hypot(x, y) = sqrt(x^2 + y^2)
    x_sq = arith.mulf(x, x)
    y_sq = arith.mulf(y, y)
    sum_sq = arith.addf(x_sq, y_sq)
    result = math_dialect.sqrt(sum_sq)
    # Convert back to original type if needed
    if original_type == T.f32():
        result = convert(result, T.f32())
    mlir_lower.store_var(target, result)


@lower(math.log1p, types.Number)
def math_log1p_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_log1p does not accept any keyword arguments"
    assert len(args) == 1, "math_log1p expects 1 argument"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.log1p(x)
    mlir_lower.store_var(target, result)


@lower(operator.mod, types.Number, types.Number)
@lower(operator.imod, types.Number, types.Number)
def mod_cg(builder, target, args, kwargs):
    assert not kwargs, "mod_cg does not accept any keyword arguments"
    assert len(args) == 2, "mod_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = lowering_utilities.convert(builder.load_var(lhs), target_mlir_type)
    rhs = lowering_utilities.convert(builder.load_var(rhs), target_mlir_type)

    match target_mlir_type:
        case ir.IntegerType():
            # For integers: use signed integer remainder
            result = arith.remsi(lhs, rhs)
        case ir.FloatType():
            # For floats: implement a % b = a - floor(a/b) * b
            div = arith.divf(lhs, rhs)
            floored = math_dialect.floor(div)
            mult = arith.mulf(floored, rhs)
            result = arith.subf(lhs, mult)
        case _:
            raise NotImplementedError(f"mod not implemented for {target_mlir_type=}")
    builder.store_var(target, result)


@lower(operator.lshift, types.Integer, types.Integer)
@lower(operator.ilshift, types.Integer, types.Integer)
def lshift_cg(builder, target, args, kwargs):
    assert not kwargs, "lshift_cg does not accept any keyword arguments"
    assert len(args) == 2, "lshift_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = lowering_utilities.convert(builder.load_var(lhs), target_mlir_type)
    rhs = lowering_utilities.convert(builder.load_var(rhs), target_mlir_type)
    result = arith.shli(lhs, rhs)
    builder.store_var(target, result)


@lower(operator.rshift, types.Integer, types.Integer)
@lower(operator.irshift, types.Integer, types.Integer)
def rshift_cg(builder, target, args, kwargs):
    assert not kwargs, "rshift_cg does not accept any keyword arguments"
    assert len(args) == 2, "rshift_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = lowering_utilities.convert(builder.load_var(lhs), target_mlir_type)
    rhs = lowering_utilities.convert(builder.load_var(rhs), target_mlir_type)
    # Use unsigned right shift for unsigned types, signed for signed types
    if target_type.signed:
        result = arith.shrsi(lhs, rhs)
    else:
        result = arith.shrui(lhs, rhs)
    builder.store_var(target, result)


@lower(operator.irshift, types.Integer, types.Integer)
def irshift_cg(builder, target, args, kwargs):
    """In-place right shift (>>=). In SSA form, this is the same as regular rshift."""
    return rshift_cg(builder, target, args, kwargs)


@lower(operator.ilshift, types.Integer, types.Integer)
def ilshift_cg(builder, target, args, kwargs):
    """In-place left shift (<<=). In SSA form, this is the same as regular lshift."""
    return lshift_cg(builder, target, args, kwargs)


@lower(operator.and_, types.Integer, types.Integer)
def and_cg(builder, target, args, kwargs):
    assert not kwargs, "and_cg does not accept any keyword arguments"
    assert len(args) == 2, "and_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = lowering_utilities.convert(builder.load_var(lhs), target_mlir_type)
    rhs = lowering_utilities.convert(builder.load_var(rhs), target_mlir_type)
    result = arith.andi(lhs, rhs)
    builder.store_var(target, result)


@lower(operator.or_, types.Integer, types.Integer)
def or_cg(builder, target, args, kwargs):
    assert not kwargs, "or_cg does not accept any keyword arguments"
    assert len(args) == 2, "or_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = lowering_utilities.convert(builder.load_var(lhs), target_mlir_type)
    rhs = lowering_utilities.convert(builder.load_var(rhs), target_mlir_type)
    result = arith.ori(lhs, rhs)
    builder.store_var(target, result)


@lower(operator.xor, types.Integer, types.Integer)
def xor_cg(builder, target, args, kwargs):
    assert not kwargs, "xor_cg does not accept any keyword arguments"
    assert len(args) == 2, "xor_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs = lowering_utilities.convert(builder.load_var(lhs), target_mlir_type)
    rhs = lowering_utilities.convert(builder.load_var(rhs), target_mlir_type)
    result = arith.xori(lhs, rhs)
    builder.store_var(target, result)


@lower(operator.floordiv, types.Number, types.Number)
@lower(operator.ifloordiv, types.Number, types.Number)
def floordiv_cg(builder, target, args, kwargs):
    assert not kwargs, "floordiv_cg does not accept any keyword arguments"
    assert len(args) == 2, "floordiv_cg expects 2 arguments"
    target_type = builder.get_numba_type(target.name)
    target_mlir_type = builder.get_mlir_type(target_type)
    lhs, rhs = args
    lhs, rhs = builder.load_var(lhs), builder.load_var(rhs)

    match target_mlir_type:
        case ir.FloatType():
            # For floats: floor(a / b)
            lhs = lowering_utilities.convert(lhs, target_mlir_type)
            rhs = lowering_utilities.convert(rhs, target_mlir_type)
            div_result = arith.divf(lhs, rhs)
            result = math_dialect.floor(div_result)
        case ir.IntegerType():
            # For signed integers, use floordivsi
            lhs = lowering_utilities.convert(lhs, target_mlir_type)
            rhs = lowering_utilities.convert(rhs, target_mlir_type)
            if target_mlir_type.is_signed:
                result = arith.floordivsi(lhs, rhs)
            else:
                # For unsigned, regular division is the same as floor division
                result = arith.divui(lhs, rhs)
        case _:
            raise ValueError(f"Unsupported type for floor division: {target_mlir_type}")

    result = builder.mlir_convert(result, target_mlir_type)
    builder.store_var(target, result)


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


@lower(complex, types.Number, types.Number)
def complex_cg(mlir_lower, target, args, kwargs):
    target_type = mlir_lower.get_numba_type(target.name)
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
        mlir_lower.store_var(target, complex_const_op)
    else:
        # Create complex from runtime values
        real_val = (
            mlir_lower.load_var(real)
            if hasattr(real, "name")
            else lowering_utilities.constant(real, element_type)
        )
        imag_val = (
            mlir_lower.load_var(imag)
            if hasattr(imag, "name")
            else lowering_utilities.constant(imag, element_type)
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
        mlir_lower.store_var(target, complex_val)


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


@lower(math.frexp, types.Float)
def math_frexp_cg(mlir_lower, target, args, kwargs):
    """
    Lower math.frexp(x) using LLVM intrinsic.
    Returns (mantissa, exponent) where mantissa is float and exponent is int32.
    """
    assert not kwargs, "math.frexp does not accept any keyword arguments"
    assert len(args) == 1, "math.frexp expects 1 argument"

    value = mlir_lower.load_var(args[0])
    float_type = value.type

    # llvm.intr.frexp returns struct { float, i32 }
    result_type = llvm.StructType.get_literal([float_type, T.i32()])
    result = llvm.intr_frexp(result_type, value)

    # Extract mantissa and exponent
    mantissa = llvm.extractvalue(float_type, result, [0])
    exp = llvm.extractvalue(T.i32(), result, [1])

    mlir_lower.store_var(target, (mantissa, exp))


@lower(math.ldexp, types.Float, types.Integer)
def math_ldexp_cg(mlir_lower, target, args, kwargs):
    """
    Lower math.ldexp(x, exp) using LLVM intrinsic.
    Returns x * 2^exp.
    """
    assert not kwargs, "math.ldexp does not accept any keyword arguments"
    assert len(args) == 2, "math.ldexp expects 2 arguments"

    value = mlir_lower.load_var(args[0])
    exp = mlir_lower.load_var(args[1])

    # ldexp expects i32 exponent
    exp = convert(exp, T.i32())

    result = llvm.intr_ldexp(value.type, value, exp)
    mlir_lower.store_var(target, result)


@lower(math.expm1, types.Number)
def math_expm1_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_expm1 does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.expm1(value)
    mlir_lower.store_var(target, result)


@lower(math.degrees, types.Number)
def math_degrees_cg(mlir_lower, target, args, kwargs):
    """Convert radians to degrees: x * (180 / pi)"""
    assert not kwargs, "math_degrees does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    # 180 / pi = 57.29577951308232
    factor = lowering_utilities.constant(57.29577951308232, value.type)
    result = arith.mulf(value, factor)
    mlir_lower.store_var(target, result)


@lower(math.radians, types.Number)
def math_radians_cg(mlir_lower, target, args, kwargs):
    """Convert degrees to radians: x * (pi / 180)"""
    assert not kwargs, "math_radians does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    # pi / 180 = 0.017453292519943295
    factor = lowering_utilities.constant(0.017453292519943295, value.type)
    result = arith.mulf(value, factor)
    mlir_lower.store_var(target, result)


@lower(math.erf, types.Number)
def math_erf_cg(mlir_lower, target, args, kwargs):
    assert not kwargs, "math_erf does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = math_dialect.erf(value)
    mlir_lower.store_var(target, result)


@lower(math.erfc, types.Number)
def math_erfc_cg(mlir_lower, target, args, kwargs):
    """erfc(x) = complementary error function, using libdevice for precision"""
    assert not kwargs, "math_erfc does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = _call_libdevice_unary(mlir_lower, value, "__nv_erfc", "__nv_erfcf")
    mlir_lower.store_var(target, result)


@lower(math.gamma, types.Number)
def math_gamma_cg(mlir_lower, target, args, kwargs):
    """Lower math.gamma using libdevice tgamma"""
    assert not kwargs, "math_gamma does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = _call_libdevice_unary(mlir_lower, value, "__nv_tgamma", "__nv_tgammaf")
    mlir_lower.store_var(target, result)


@lower(math.lgamma, types.Number)
def math_lgamma_cg(mlir_lower, target, args, kwargs):
    """Lower math.lgamma using libdevice lgamma"""
    assert not kwargs, "math_lgamma does not accept any keyword arguments"
    value = _ensure_float(mlir_lower.load_var(args[0]))
    result = _call_libdevice_unary(mlir_lower, value, "__nv_lgamma", "__nv_lgammaf")
    mlir_lower.store_var(target, result)


@lower(math.fmod, types.Number, types.Number)
def math_fmod_cg(mlir_lower, target, args, kwargs):
    """fmod(x, y) - floating point remainder with sign of x (truncated division)"""
    assert not kwargs, "math_fmod does not accept any keyword arguments"
    assert len(args) == 2, "math_fmod expects 2 arguments"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    y = _ensure_float(mlir_lower.load_var(args[1]))
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    # fmod(x, y) = x - trunc(x/y) * y
    # This gives the remainder with the sign of x (truncated toward zero)
    div = arith.divf(x, y)
    truncated = math_dialect.trunc(div)
    mult = arith.mulf(truncated, y)
    result = arith.subf(x, mult)
    mlir_lower.store_var(target, result)


@lower(math.remainder, types.Number, types.Number)
def math_remainder_cg(mlir_lower, target, args, kwargs):
    """IEEE 754 remainder: x - round(x/y) * y"""
    assert not kwargs, "math_remainder does not accept any keyword arguments"
    assert len(args) == 2, "math_remainder expects 2 arguments"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    y = _ensure_float(mlir_lower.load_var(args[1]))
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    # IEEE remainder: x - round(x/y) * y (round to nearest even)
    div = arith.divf(x, y)
    rounded = math_dialect.roundeven(div)
    mult = arith.mulf(rounded, y)
    result = arith.subf(x, mult)
    mlir_lower.store_var(target, result)


@lower(math.pow, types.Number, types.Number)
def math_pow_cg(mlir_lower, target, args, kwargs):
    """math.pow(x, y) - x raised to power y"""
    assert not kwargs, "math_pow does not accept any keyword arguments"
    assert len(args) == 2, "math_pow expects 2 arguments"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    y = _ensure_float(mlir_lower.load_var(args[1]))
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    result = math_dialect.powf(x, y)
    mlir_lower.store_var(target, result)


@lower(math.nextafter, types.Number, types.Number)
def math_nextafter_cg(mlir_lower, target, args, kwargs):
    """nextafter(x, y) - next representable floating-point value after x towards y"""
    assert not kwargs, "math_nextafter does not accept any keyword arguments"
    assert len(args) == 2, "math_nextafter expects 2 arguments"
    x = _ensure_float(mlir_lower.load_var(args[0]))
    y = _ensure_float(mlir_lower.load_var(args[1]))
    unified_type = lowering_utilities.numpy_implicit_type_promotion(x.type, y.type)
    x = convert(x, unified_type)
    y = convert(y, unified_type)
    result = _call_libdevice_binary(mlir_lower, x, y, "__nv_nextafter", "__nv_nextafterf")
    mlir_lower.store_var(target, result)


@lower(math.modf, types.Float)
def math_modf_cg(mlir_lower, target, args, kwargs):
    """
    modf(x) returns (fractional_part, integer_part).
    fractional_part has the same sign as x.
    Special cases:
    - modf(inf) = (0.0, inf)
    - modf(-inf) = (-0.0, -inf)
    - modf(nan) = (nan, nan)
    """
    assert not kwargs, "math_modf does not accept any keyword arguments"
    assert len(args) == 1, "math_modf expects 1 argument"

    value = mlir_lower.load_var(args[0])
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

    mlir_lower.store_var(target, (frac_part, int_part))
