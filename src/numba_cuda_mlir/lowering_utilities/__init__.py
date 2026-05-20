# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.errors import ensure_verifies
from numba_cuda_mlir.descriptor import MLIRDispatcher
from numba_cuda_mlir.descriptor import MLIRDispatcherType
from numba_cuda_mlir.errors import InternalCompilerError
from functools import singledispatch
from dataclasses import dataclass
from abc import abstractmethod
from collections.abc import Sequence
import functools
import numpy as np
from numba_cuda_mlir.numba_cuda import types, typing
from numba_cuda_mlir.annotations import AnyCallable, PS
from numba_cuda_mlir.lowering_utilities import type_conversions
from numba_cuda_mlir.lowering_utilities.type_conversions import (
    np_dtype_to_mlir_type as mlir_type_from_numpy_dtype,
    to_mlir_type,
)
from numba_cuda_mlir.type_defs.float_types import BFloat16Type
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import (
    arith,
    memref,
    func,
    gpu,
    llvm,
    cf,
    builtin,
    complex as complex_dialect,
    vector as vector_dialect,
)
from numba_cuda_mlir.mlir.dialect_exts import scf
from numba_cuda_mlir._mlir.extras.meta import region_op
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir._mlir.extras import types as T
import operator

# std::numeric_limits<int32_t>::min() — marks a dynamic index in llvm.getelementptr
GEP_DYNAMIC_INDEX = -2147483648


def memref_to_llvm_ptr(array: ir.Value, indices: list[ir.Value], element_type: ir.Type) -> ir.Value:
    """Convert memref + indices to LLVM pointer.

    Extracts base pointer from potentially strided memref and computes
    element pointer using getelementptr with linearized indices.

    Args:
        array: Memref value (potentially strided)
        indices: List of index values
        element_type: Element type for getelementptr

    Returns:
        LLVM pointer (!llvm.ptr) to the indexed element
    """
    # Extract base pointer from memref and convert to LLVM pointer
    base_ptr_idx = memref.extract_aligned_pointer_as_index(array)
    base_ptr = convert(base_ptr_idx, llvm.PointerType.get())

    # Compute linear offset and use getelementptr
    if len(indices) == 1:
        # 1D case: simple offset
        idx = convert(indices[0], T.i64())
        element_ptr = llvm.getelementptr(
            llvm.PointerType.get(),
            base_ptr,
            [idx],
            [GEP_DYNAMIC_INDEX],
            element_type,
            None,
        )
    else:
        # N-D case: linearize indices using row-major strides
        ndim = len(indices)

        # Compute strides for row-major layout (C order)
        strides = [None] * ndim
        strides[-1] = arith.constant(T.i64(), 1)
        for d in range(ndim - 2, -1, -1):
            dim_size = memref.dim(array, index_of(d + 1))
            dim_size = convert(dim_size, T.i64())
            strides[d] = dim_size * strides[d + 1]

        # Compute linear index: sum(index[d] * stride[d])
        linear_idx = arith.constant(T.i64(), 0)
        for d in range(ndim):
            idx_val = convert(indices[d], T.i64())
            linear_idx = linear_idx + idx_val * strides[d]

        element_ptr = llvm.getelementptr(
            llvm.PointerType.get(),
            base_ptr,
            [linear_idx],
            [GEP_DYNAMIC_INDEX],
            element_type,
            None,
        )

    return element_ptr


def memref_to_tensor(memref):
    """
    Utility function to convert from a memref to tensor
    """
    from numba_cuda_mlir._mlir.dialects import bufferization

    match memref.type:
        case ir.MemRefType():
            tensor_type = ir.RankedTensorType.get(
                shape=memref.type.shape, element_type=memref.type.element_type
            )
            tensor = bufferization.to_tensor(result=tensor_type, buffer=memref, restrict=True)
            return tensor
        case ir.RankedTensorType():
            return memref
        case _:
            raise NotImplementedError(f"Not implemented for type {memref.type}")


def tensor_to_memref(tensor):
    """
    Utility function to convert from a tensor to memref
    """
    from numba_cuda_mlir._mlir.dialects import bufferization

    match tensor.type:
        case ir.MemRefType():
            return tensor
        case ir.RankedTensorType():
            # Create memref with strided layout
            rank = len(tensor.type.shape)
            dyn_stride = ir.MemRefType.get_dynamic_stride_or_offset()
            layout = ir.StridedLayoutAttr.get(offset=dyn_stride, strides=[dyn_stride] * rank)
            memref_type = ir.MemRefType.get(
                shape=tensor.type.shape,
                element_type=tensor.type.element_type,
                layout=layout,
            )
            memref = bufferization.to_buffer(buffer=memref_type, tensor=tensor)
            return memref
        case _:
            raise NotImplementedError(f"Not implemented for type {tensor.type}")


def get_type_width(ty: ir.Type) -> int:
    match ty:
        case ir.IntegerType():
            return ty.width
        case ir.IndexType():
            return 64
        case ir.FloatType():
            return ty.width
        case ir.ComplexType():
            return 2 * ir.ComplexType(ty).element_type.width
        case ir.VectorType():
            vt = ir.VectorType(ty)
            count = 1
            for d in vt.shape:
                count *= d
            return count * get_type_width(vt.element_type)
        case _:
            raise NotImplementedError(f"Not implemented for type {ty}")


def is_complex_type(mlir_type):
    return isinstance(mlir_type, ir.ComplexType)


def get_llvm_struct_for_complex(complex_type):
    elem_type = complex_type.element_type
    return llvm.StructType.get_literal([elem_type, elem_type])


def complex_to_llvm_struct(value):
    complex_type = value.type
    if not is_complex_type(complex_type):
        raise TypeError(f"Expected MLIR complex type, got {complex_type}")
    struct_type = get_llvm_struct_for_complex(complex_type)
    real = complex_dialect.re(value)
    imag = complex_dialect.im(value)
    undef = llvm.UndefOp(struct_type)
    with_real = llvm.insertvalue(
        container=undef,
        value=real,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    return llvm.insertvalue(
        container=with_real,
        value=imag,
        position=ir.DenseI64ArrayAttr.get([1]),
    )


def llvm_struct_to_complex(value, complex_type):
    elem_type = complex_type.element_type
    real = llvm.extractvalue(
        res=elem_type,
        container=value,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    imag = llvm.extractvalue(
        res=elem_type,
        container=value,
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return complex_dialect.create_(complex=complex_type, real=real, imaginary=imag)


def _lookup_datamodel_type(numba_type: types.Type, context: str) -> ir.Type:
    from numba_cuda_mlir.models import mlir_data_manager

    model = mlir_data_manager.lookup(numba_type)
    match context:
        case "value":
            return model.get_value_type()
        case "storage" | "data":
            return model.get_data_type()
        case "argument":
            return model.get_argument_type()
        case "return":
            return model.get_return_type()
        case _:
            raise InternalCompilerError(f"Unknown datamodel context: {context}")


def get_value_type(numba_type: types.Type) -> ir.Type:
    return _lookup_datamodel_type(numba_type, "value")


def get_storage_type(numba_type: types.Type) -> ir.Type:
    return _lookup_datamodel_type(numba_type, "storage")


def get_argument_type(numba_type: types.Type) -> ir.Type:
    return _lookup_datamodel_type(numba_type, "argument")


def get_return_type(numba_type: types.Type) -> ir.Type:
    return _lookup_datamodel_type(numba_type, "return")


def _numba_type_bitwidth(numba_type: types.Type, value_type: ir.Type | None = None) -> int:
    bitwidth = getattr(numba_type, "bitwidth", None)
    if bitwidth is not None:
        return int(bitwidth)
    if value_type is not None:
        return get_type_width(value_type)
    return get_type_width(get_value_type(numba_type))


def storage_bitwidth(numba_type: types.Type) -> int:
    storage_type = get_storage_type(numba_type)
    return get_type_width(storage_type)


def storage_itemsize_bytes(numba_type: types.Type) -> int:
    from numba_cuda_mlir.types import NestedArray, Record

    if isinstance(numba_type, Record):
        return numba_type.size
    if isinstance(numba_type, NestedArray):
        return storage_itemsize_bytes(numba_type.dtype)

    dtype = getattr(numba_type, "dtype", numba_type)
    if isinstance(dtype, Record):
        return dtype.size
    if isinstance(dtype, NestedArray):
        return storage_itemsize_bytes(dtype.dtype)
    if isinstance(dtype, np.dtype):
        return dtype.itemsize
    if isinstance(dtype, types.UnicodeCharSeq):
        return getattr(dtype, "count", 1) * np.dtype("U1").itemsize
    if isinstance(dtype, types.CharSeq):
        return getattr(dtype, "count", 1)
    bitwidth = storage_bitwidth(dtype)
    return (bitwidth + 7) // 8


def _is_bool_numba_type(numba_type: types.Type) -> bool:
    return isinstance(numba_type, (types.Boolean, types.BooleanLiteral))


def _is_float_storage_numba_type(numba_type: types.Type) -> bool:
    from numba_cuda_mlir.type_defs import float_types
    from numba_cuda_mlir.numba_cuda.types.ext_types import Bfloat16

    return isinstance(numba_type, (types.Float, float_types.SpecialFloatType, Bfloat16))


def _integer_storage_type_for_value(numba_type: types.Type, value_type: ir.Type) -> ir.IntegerType:
    return ir.IntegerType.get_signless(_numba_type_bitwidth(numba_type, value_type))


def value_to_storage(numba_type: types.Type, value: ir.Value) -> ir.Value:
    """Convert a source-level value to its memory/ABI storage representation."""
    storage_type = get_storage_type(numba_type)
    value_type = get_value_type(numba_type)
    if getattr(value, "type", None) == storage_type and value_type == storage_type:
        return value
    if getattr(value, "type", None) != value_type:
        value = convert(value, value_type)
    if value_type == storage_type:
        return value

    if _is_bool_numba_type(numba_type):
        value = convert(value, T.bool())
        return arith.extui(out=storage_type, in_=value)

    if _is_float_storage_numba_type(numba_type) and isinstance(storage_type, ir.IntegerType):
        bits_type = _integer_storage_type_for_value(numba_type, value_type)
        bits = arith.bitcast(out=bits_type, in_=value)
        if bits_type.width == storage_type.width:
            return bits if bits_type == storage_type else arith.bitcast(out=storage_type, in_=bits)
        if bits_type.width < storage_type.width:
            return arith.extui(out=storage_type, in_=bits)
        return arith.trunci(out=storage_type, in_=bits)

    return convert(value, storage_type)


def storage_to_value(numba_type: types.Type, value: ir.Value) -> ir.Value:
    """Convert a memory/ABI storage payload to its source-level value representation."""
    storage_type = get_storage_type(numba_type)
    value_type = get_value_type(numba_type)
    if getattr(value, "type", None) == value_type and value_type == storage_type:
        return value
    if getattr(value, "type", None) != storage_type:
        value = convert(value, storage_type)
    if value_type == storage_type:
        return value

    if _is_bool_numba_type(numba_type):
        zero = arith.constant(storage_type, 0)
        return arith.cmpi(arith.CmpIPredicate.ne, value, zero)

    if _is_float_storage_numba_type(numba_type) and isinstance(storage_type, ir.IntegerType):
        bits_type = _integer_storage_type_for_value(numba_type, value_type)
        bits = value
        if storage_type.width > bits_type.width:
            bits = arith.trunci(out=bits_type, in_=value)
        elif storage_type.width < bits_type.width:
            bits = arith.extui(out=bits_type, in_=value)
        elif value.type != bits_type:
            bits = arith.bitcast(out=bits_type, in_=value)
        return arith.bitcast(out=value_type, in_=bits)

    return convert(value, value_type)


def array_element_value_load(array_type: types.Array, array: ir.Value, indices: Sequence[ir.Value]):
    stored = memref.load(array, list(indices))
    return storage_to_value(array_type.dtype, stored)


def array_element_value_store(
    array_type: types.Array,
    array: ir.Value,
    indices: Sequence[ir.Value],
    value: ir.Value,
):
    stored = value_to_storage(array_type.dtype, value)
    memref.store(value=stored, memref=array, indices=list(indices))


def memref_to_value_tensor(array_type: types.Array, array: ir.Value) -> ir.Value:
    from numba_cuda_mlir._mlir.dialects import linalg, tensor

    tensor_value = memref_to_tensor(array)
    value_element_type = get_value_type(array_type.dtype)
    if tensor_value.type.element_type == value_element_type:
        return tensor_value

    result_type = T.tensor(*tensor_value.type.shape, value_element_type)
    init = tensor.empty(
        sizes=dims_of_tensor_shape(tensor_value),
        element_type=value_element_type,
    )
    result = linalg.MapOp(
        result=[result_type],
        inputs=[tensor_value],
        init=init,
    )
    block = result.mapper.blocks.append(tensor_value.type.element_type, value_element_type)
    with ir.InsertionPoint(block):
        linalg.yield_([storage_to_value(array_type.dtype, block.arguments[0])])
    return result.result[0]


def value_tensor_to_storage_memref(array_type: types.Array, tensor_value: ir.Value) -> ir.Value:
    from numba_cuda_mlir._mlir.dialects import linalg, tensor

    storage_element_type = get_storage_type(array_type.dtype)
    if tensor_value.type.element_type == storage_element_type:
        return tensor_to_memref(tensor_value)

    result_type = T.tensor(*tensor_value.type.shape, storage_element_type)
    init = tensor.empty(
        sizes=dims_of_tensor_shape(tensor_value),
        element_type=storage_element_type,
    )
    result = linalg.MapOp(
        result=[result_type],
        inputs=[tensor_value],
        init=init,
    )
    block = result.mapper.blocks.append(tensor_value.type.element_type, storage_element_type)
    with ir.InsertionPoint(block):
        linalg.yield_([value_to_storage(array_type.dtype, block.arguments[0])])
    return tensor_to_memref(result.result[0])


@singledispatch
def numpy_implicit_type_promotion(
    a: ir.Type | types.Type, b: ir.Type | types.Type
) -> ir.Type | types.Type:
    raise NotImplementedError(f"{a=} {b=}")


@numpy_implicit_type_promotion.register(types.Type)
def _(a: types.Type, b: types.Type) -> types.Type:
    float_types = {
        16: types.float16,
        32: types.float32,
        64: types.float64,
    }
    integer_types = {
        8: types.int8,
        16: types.int16,
        32: types.int32,
        64: types.int64,
    }
    from numba_cuda_mlir.type_defs.float_types import SpecialFloatType

    match a, b:
        case ty1, ty2 if ty1 == ty2:
            return a
        case (BFloat16Type(), _ as other) | (_ as other, BFloat16Type()):
            raise TypeError(f"Trying to convert to/from BFloat16 and {other}")
        case (SpecialFloatType() as s, _ as other) | (
            _ as other,
            SpecialFloatType() as s,
        ):
            raise TypeError(f"Implicit promotion between {s} and {other} is not supported")
        case types.Float() as f, types.Integer() as i:
            return float_types[max(f.bitwidth, i.bitwidth)]
        case types.Integer() as i, types.Float() as f:
            return float_types[max(f.bitwidth, i.bitwidth)]
        case types.Integer() as i1, types.Integer() as i2:
            return integer_types[max(i1.bitwidth, i2.bitwidth)]
        case types.Float() as f1, types.Float() as f2:
            return float_types[max(f1.bitwidth, f2.bitwidth)]
        case _:
            raise NotImplementedError(f"{a=} {b=}")


@numpy_implicit_type_promotion.register(ir.Type)
def _(a: ir.Type, b: ir.Type) -> ir.Type:
    match a, b:
        case ty1, ty2 if ty1 == ty2:
            return a
        case (
            (ir.IntegerType(), ir.FloatType())
            | (ir.FloatType(), ir.IntegerType())
            | (ir.IndexType(), ir.FloatType())
            | (ir.FloatType(), ir.IndexType())
        ):
            larger_width = max(get_type_width(a), get_type_width(b))
            ty = T.f32() if larger_width == 32 else T.f64()
            return ty
        case (
            (ir.IntegerType(), ir.IntegerType())
            | (ir.IndexType(), ir.IntegerType())
            | (ir.IntegerType(), ir.IndexType())
        ):
            larger_width = max(get_type_width(a), get_type_width(b))
            ty = ir.IntegerType.get_signless(width=larger_width)
            return ty
        case ir.FloatType(), ir.FloatType():
            larger_width = max(get_type_width(a), get_type_width(b))
            ty = T.f32() if larger_width == 32 else T.f64()
            return ty
        case (ir.ComplexType(), _) | (_, ir.ComplexType()):

            def _scalar_width(t: ir.Type) -> int:
                return t.element_type.width if isinstance(t, ir.ComplexType) else get_type_width(t)

            larger_width = max(_scalar_width(a), _scalar_width(b))
            elem = T.f32() if larger_width <= 32 else T.f64()
            return T.complex(elem)
        case _:
            raise NotImplementedError(f"Not implemented for type {a} and {b}")


def coerce_numpy_scalars_for_binary_op(a: ir.Value, b: ir.Value) -> tuple[ir.Value, ir.Value]:
    coerced = numpy_implicit_type_promotion(a.type, b.type)
    return convert(a, coerced), convert(b, coerced)


def mul(a: ir.Value, b: ir.Value) -> ir.Value:
    """Multiply two values with automatic type coercion."""
    a, b = coerce_numpy_scalars_for_binary_op(a, b)
    match a.type:
        case ir.IntegerType() | ir.IndexType():
            return arith.muli(a, b)
        case ir.FloatType():
            return arith.mulf(a, b)
        case _:
            raise NotImplementedError(f"Not implemented for type {a.type}")


def div(a: ir.Value, b: ir.Value) -> ir.Value:
    """Divide two values with automatic type coercion."""
    a, b = coerce_numpy_scalars_for_binary_op(a, b)
    match a.type:
        case ir.IntegerType() | ir.IndexType():
            return arith.divsi(a, b)  # signed division
        case ir.FloatType():
            return arith.divf(a, b)
        case _:
            raise NotImplementedError(f"Not implemented for type {a.type}")


def less_than(a: ir.Value, b: ir.Value) -> ir.Value:
    a, b = coerce_numpy_scalars_for_binary_op(a, b)
    match a.type:
        case ir.IntegerType() | ir.IndexType():
            return arith.cmpi(arith.CmpIPredicate.slt, a, b)
        case ir.FloatType():
            return arith.cmpf(arith.CmpFPredicate.OLT, a, b)
        case _:
            raise NotImplementedError(f"Not implemented for type {a.type}")


def _get_mlir_bin_op_for_operator(op):
    import functools

    match op:
        case operator.or_:
            return (
                functools.partial(arith.ori),
                None,
            )
        case operator.and_:
            return (
                functools.partial(arith.andi),
                None,
            )
        case operator.xor:
            return (
                functools.partial(arith.xori),
                None,
            )
        case operator.lt:
            return (
                functools.partial(arith.cmpi, arith.CmpIPredicate.slt),
                functools.partial(arith.cmpf, arith.CmpFPredicate.ULT),
            )
        case operator.le:
            return (
                functools.partial(arith.cmpi, arith.CmpIPredicate.sle),
                functools.partial(arith.cmpf, arith.CmpFPredicate.ULE),
            )
        case operator.gt:
            return (
                functools.partial(arith.cmpi, arith.CmpIPredicate.sgt),
                functools.partial(arith.cmpf, arith.CmpFPredicate.UGT),
            )
        case operator.ge:
            return (
                functools.partial(arith.cmpi, arith.CmpIPredicate.sge),
                functools.partial(arith.cmpf, arith.CmpFPredicate.UGE),
            )
        case operator.eq:
            return (
                functools.partial(arith.cmpi, arith.CmpIPredicate.eq),
                functools.partial(arith.cmpf, arith.CmpFPredicate.UEQ),
            )
        case operator.ne:
            return (
                functools.partial(arith.cmpi, arith.CmpIPredicate.ne),
                functools.partial(arith.cmpf, arith.CmpFPredicate.UNE),
            )
        case operator.add | operator.iadd:
            return (
                functools.partial(arith.addi),
                functools.partial(arith.addf),
            )
        case operator.sub | operator.isub:
            return (
                functools.partial(arith.subi),
                functools.partial(arith.subf),
            )
        case operator.mul | operator.imul:
            return (
                functools.partial(arith.muli),
                functools.partial(arith.mulf),
            )
        case operator.truediv | operator.floordiv | operator.itruediv | operator.ifloordiv:
            return (
                functools.partial(arith.divsi),
                functools.partial(arith.divf),
            )
        case operator.mod | operator.imod:
            return (
                functools.partial(arith.remsi),
                functools.partial(arith.remf),
            )
        case _:
            raise NotImplementedError(f"Not implemented for operator {op}")


def _create_utility_bin_op(op):
    iop, fop = _get_mlir_bin_op_for_operator(op)

    def bin_op(a: ir.Value, b: ir.Value, *rest) -> ir.Value:
        if rest:
            return bin_op(bin_op(a, b), *rest)
        a, b = coerce_numpy_scalars_for_binary_op(a, b)
        match a.type:
            case ir.IntegerType() | ir.IndexType() if iop is not None:
                return iop(a, b)
            case ir.FloatType() if fop is not None:
                return fop(a, b)
            case _:
                raise NotImplementedError(f"Not implemented for type {a.type}")

    return bin_op


sub = _create_utility_bin_op(operator.sub)
add = _create_utility_bin_op(operator.add)
mul = _create_utility_bin_op(operator.mul)
div = _create_utility_bin_op(operator.truediv)
less_than = _create_utility_bin_op(operator.lt)
less_than_or_equal = _create_utility_bin_op(operator.le)
greater_than = _create_utility_bin_op(operator.gt)
greater_than_or_equal = _create_utility_bin_op(operator.ge)
equal = _create_utility_bin_op(operator.eq)
not_equal = _create_utility_bin_op(operator.ne)
or_ = _create_utility_bin_op(operator.or_)
and_ = _create_utility_bin_op(operator.and_)
xor = _create_utility_bin_op(operator.xor)


def true() -> ir.Value:
    return arith.constant(T.bool(), 1)


def false() -> ir.Value:
    return arith.constant(T.bool(), 0)


_generic_rmw = region_op(
    memref.GenericAtomicRMWOp,
    terminator=lambda results: memref.AtomicYieldOp(results[0]),
)


def set_error_code_if_zero(error_ptr: ir.Value, error_code: int):
    """Set error code if currently 0 (first error wins).

    Uses LLVM cmpxchg to atomically compare-and-swap the error code.
    Only the first error is recorded (subsequent errors are ignored).
    """
    zero = llvm.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 0)).result
    error_val = llvm.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), error_code)).result
    # Atomic CAS: if current value is 0, set to error_code
    llvm.cmpxchg(
        error_ptr,
        zero,
        error_val,
        llvm.AtomicOrdering.monotonic,
        llvm.AtomicOrdering.monotonic,
    )


def set_error_and_return(
    condition: ir.Value, error_memref: ir.Value, error_code: int, return_block: ir.Block
):
    """If condition is false, set error code and branch to return block."""
    with scf.if_ctx_manager(arith.cmpi(arith.CmpIPredicate.eq, condition, false())):
        set_error_code_if_zero(error_memref, error_code)
        cf.br([], return_block)
        scf.yield_([])


def bool_of(value: ir.Value | bool) -> ir.Value:
    match value:
        case ir.Value():
            return convert(value, T.bool())
        case _:
            return true() if value else false()


def concretize_tuple_to_tensor(tup: tuple[ir.Value]) -> ir.Value:
    from numba_cuda_mlir._mlir.dialects import tensor

    ty = tup[0].type
    if not all(ty == t.type for t in tup):
        raise NotImplementedError("All elements of the tuple must have the same type")
    tensor_type = T.tensor(len(tup), ty)
    t = tensor.splat(tensor_type, tup[0], [])
    for i, element in enumerate(tup[1:]):
        t = tensor.insert(element, t, [index_of(i)])
    return t


def convert_tuple_like(values: list[ir.Value], target_type: ir.Type) -> ir.Value:
    from numba_cuda_mlir._mlir.dialects import tensor

    match target_type:
        case tuple():
            return tuple(convert(value, ty) for value, ty in zip(values, target_type))
        case ir.MemRefType():
            tty = T.tensor(*target_type.shape, target_type.element_type)
            tens = tensor.from_elements(tty, [*values])
            return tensor_to_memref(tens)
        case ir.RankedTensorType():
            return tensor.from_elements(target_type, *values)
        case _:
            raise NotImplementedError(f"Not implemented for type {target_type}")


def _convert_integer_to_integer(
    value: ir.Value, target_type: ir.IntegerType, *, signed: bool = False
) -> ir.Value:
    """
    If possible, we perform the conversion on the types as they are given to us.
    However, if the signedness of the types do not match, we first bitcast to signless forms
    and then extend/truncate as necessary.
    """
    value_type: ir.IntegerType = value.type

    if value_type == target_type:
        return value

    def _signedness(ty: ir.IntegerType) -> str:
        if ty.is_signed:
            return "signed"
        if ty.is_unsigned:
            return "unsigned"
        return "signless"

    source_signedness = _signedness(value_type)
    use_signed_extend = signed or source_signedness == "signed"

    work_value_type = value_type
    work_target_type = target_type
    if not work_target_type.is_signless:
        work_target_type = ir.IntegerType.get_signless(work_target_type.width)

    if work_value_type.width > work_target_type.width:
        # Special case: when converting to i1 (boolean), use comparison against zero
        # instead of truncation, which only keeps the LSB and fails for values like 2
        if work_target_type.width == 1:
            trace("converting to i1 (boolean) via comparison against zero")
            zero = arith.constant(work_value_type, value=0)
            value = arith.cmpi(arith.CmpIPredicate.ne, value, zero)
        else:
            trace("value_type.width > target_type.width, truncating")
            value = arith.trunci(out=work_target_type, in_=value)
    elif work_value_type.width < work_target_type.width:
        trace("value_type.width < target_type.width, extending")
        # Default to unsigned extension for signless integers (common in GPU code)
        # Use signed extension only when explicitly marked as signed
        extend_op = arith.extsi if use_signed_extend else arith.extui
        value = extend_op(out=work_target_type, in_=value)
    elif work_value_type != work_target_type:
        trace("bitcasting to match intermediate target signedness")
        value = arith.bitcast(out=work_target_type, in_=value)

    if work_target_type != target_type:
        trace("restoring requested target signedness via bitcast")
        value = arith.bitcast(out=target_type, in_=value)

    return value


def convert(value, target_type, *, signed: bool = False):
    if getattr(value, "type", None) == target_type:
        return value
    return ensure_verifies(unverified_convert(value, target_type, signed=signed))


@singledispatch
def unverified_convert(value, target_type, *, signed: bool = False):
    raise NotImplementedError(f"Not implemented for type {type(value)}")


@unverified_convert.register
def convert_none(value: ir.NoneType, target_type: ir.NoneType, **_):
    if value != target_type:
        raise InternalCompilerError("Cannot convert NoneType to anything other than NoneType")
    return value


@unverified_convert.register
def opaque_data_model_convert(value: type | MLIRDispatcher, target_type: ir.NoneType, **_):
    """
    For types with an opaque data model, we defer the real lowering until later - we
    hopefully resolve this at compile time anyways.
    """
    return value


@unverified_convert.register
def number_class_convert(value: types.Type, target_type: types.functions.NumberClass, **_):
    return value


def _memory_spaces_match(lhs: ir.MemRefType, rhs: ir.MemRefType) -> bool:
    """
    `lhs.memory_space == rhs.memory_space` works when both memrefs _have_ a memory
    space, but when one of them does not, `memref.memory_space` returns None and
    attributes do not have an __eq__ overload for None, so an error is raised.
    That's why the weird XOR.
    """
    match lhs.memory_space, rhs.memory_space:
        case ir.Attribute() as a, ir.Attribute() as b if a == b:
            return True
        case None, ir.Attribute() | ir.Attribute(), None:
            return False
        case None, None:
            return True


@unverified_convert.register
def unverified_basic_mlir_convert(
    value: ir.Value | int | float | bool | complex,
    target_type: ir.Type,
    *,
    signed: bool = False,
) -> ir.Value:
    from numba_cuda_mlir._mlir.dialects import (
        complex as complex_dialect,
        nvgpu,
        tensor,
        vector,
    )

    if isinstance(value, (int, float, bool, complex)):
        if isinstance(target_type, ir.IndexType):
            return index_of(int(value))
        return constant(value, target_type)
    value_type = value.type
    trace("value_type: %s, target_type: %s", value_type, target_type)
    match value_type, target_type:
        case ir.Type() as x, ir.Type() as y if x == y:
            trace("value_type == target_type, returning value")
            return value
        case (ir.IndexType(), ir.IntegerType()) | (ir.IntegerType(), ir.IndexType()):
            return arith.index_cast(out=target_type, in_=value)
        case (ir.IndexType(), ir.FloatType()) | (ir.FloatType(), ir.IndexType()):
            return convert(int_of(value, T.i64()), target_type)
        case ir.IntegerType() as a, ir.IntegerType() as b:
            return _convert_integer_to_integer(value, target_type, signed=signed)
        case ir.FloatType(), ir.FloatType() if (
            value_type.width == target_type.width and value_type != target_type
        ):
            # Same width but different float types (e.g. f16 <-> bf16)
            # Convert via f32 as intermediate
            trace("same-width float conversion via f32: %s -> %s", value_type, target_type)
            intermediate = arith.extf(out=T.f32(), in_=value)
            return arith.truncf(out=target_type, in_=intermediate)
        case ir.FloatType(), ir.FloatType() if value_type.width != target_type.width:
            if value_type.width > target_type.width:
                trace("value_type.width > target_type.width, truncating")
                return arith.truncf(out=target_type, in_=value)
            else:
                trace("value_type.width < target_type.width, extending")
                return arith.extf(out=target_type, in_=value)
        case ir.VectorType() as vt1, ir.VectorType() as vt2 if vt1.shape == vt2.shape:
            # We can use arith ops to convert vectors element-wise
            elem1 = vt1.element_type
            elem2 = vt2.element_type
            if isinstance(elem1, ir.FloatType) and isinstance(elem2, ir.FloatType):
                if elem1.width > elem2.width:
                    return arith.truncf(out=target_type, in_=value)
                else:
                    return arith.extf(out=target_type, in_=value)
            elif isinstance(elem1, ir.IntegerType) and isinstance(elem2, ir.IntegerType):
                if elem1.width > elem2.width:
                    return arith.trunci(out=target_type, in_=value)
                else:
                    return arith.extsi(out=target_type, in_=value)
            elif isinstance(elem1, ir.IntegerType) and isinstance(elem2, ir.FloatType):
                return (
                    arith.sitofp(out=target_type, in_=value)
                    if elem1.width > 1
                    else arith.uitofp(out=target_type, in_=value)
                )
            elif isinstance(elem1, ir.FloatType) and isinstance(elem2, ir.IntegerType):
                return (
                    arith.fptosi(out=target_type, in_=value)
                    if elem2.width > 1
                    else arith.fptoui(out=target_type, in_=value)
                )
            else:
                raise NotImplementedError(f"Vector conversion not implemented: {vt1} to {vt2}")
        case ir.IntegerType(), ir.FloatType():
            return (
                arith.sitofp(out=target_type, in_=value)
                if value_type.width > 1
                else arith.uitofp(out=target_type, in_=value)
            )
        case ir.BF16Type(), ir.IntegerType() if target_type.width == 16:
            # bf16 to int16/uint16: use bitcast to preserve bit pattern
            trace("bf16 -> i16 bitcast conversion")
            return arith.bitcast(out=target_type, in_=value)
        case ir.FloatType(), ir.IntegerType():
            return (
                arith.fptosi(out=target_type, in_=value)
                if target_type.width > 1
                else arith.fptoui(out=target_type, in_=value)
            )
        case ir.IntegerType(), ir.ComplexType():
            # Convert integer to complex: int -> float -> complex(float, 0)
            float_type = target_type.element_type
            float_val = (
                arith.sitofp(out=float_type, in_=value)
                if value_type.width > 1
                else arith.uitofp(out=float_type, in_=value)
            )
            zero = arith.constant(result=float_type, value=0.0)
            return complex_dialect.create_(complex=target_type, real=float_val, imaginary=zero)
        case ir.FloatType(), ir.ComplexType():
            # Convert float to complex: float -> complex(float, 0)
            float_type = target_type.element_type
            real_val = convert(value, float_type)
            zero = arith.constant(result=float_type, value=0.0)
            return complex_dialect.create_(complex=target_type, real=real_val, imaginary=zero)
        case ir.ComplexType(), ir.ComplexType():
            assert value_type.element_type != target_type.element_type, (
                "how did we get here? the types should compare-equal."
            )
            target_element_type = target_type.element_type
            real = complex_dialect.re(value)
            real = convert(real, target_element_type)
            imag = complex_dialect.im(value)
            imag = convert(imag, target_element_type)
            return complex_dialect.create_(complex=target_type, real=real, imaginary=imag)
        case ir.MemRefType() as mr, ptr_type if str(ptr_type) == "!llvm.ptr":
            idx = memref.extract_aligned_pointer_as_index(value)
            return convert(idx, target_type)
        case ptr_type, ir.IntegerType() if str(ptr_type) == "!llvm.ptr":
            ptrtoi = llvm.ptrtoint(res=T.i64(), arg=value)
            return convert(ptrtoi, target_type)
        case ptr_type, ir.IndexType() if str(ptr_type) == "!llvm.ptr":
            value = llvm.ptrtoint(res=T.i64(), arg=value)
            return convert(value, target_type)
        case ir.IntegerType(), ptr_type if str(ptr_type) == "!llvm.ptr":
            itoptr = convert(value, T.i64())
            return llvm.inttoptr(res=target_type, arg=itoptr)
        case ir.IndexType(), ptr_type if str(ptr_type) == "!llvm.ptr":
            value = convert(value, T.i64())
            return convert(value, target_type)
        case (
            ir.MemRefType() as value_type,
            ir.MemRefType() as target_type,
        ) if not _memory_spaces_match(value_type, target_type):
            memref_type_with_memory_space = ir.MemRefType.get(
                value_type.shape,
                value_type.element_type,
                value_type.layout,
                target_type.memory_space,
            )
            mr = memref.memory_space_cast(dest=memref_type_with_memory_space, source=value)
            return convert(mr, target_type)
        case ir.MemRefType() as a, ir.MemRefType() as b if not a.has_rank or not b.has_rank:
            raise NotImplementedError("Conversions between unranked memrefs")
        case ir.MemRefType() as a, ir.MemRefType() as b if a.rank == b.rank:
            return memref.cast(dest=target_type, source=value)
        case ir.MemRefType(), ir.MemRefType():
            value = memref_to_tensor(value)
            shape = [tensor.dim(value, index_of(i)) for i in range(target_type.rank)]
            shape = tensor.from_elements(T.tensor(target_type.rank, T.index()), shape)
            t_type = T.tensor(*target_type.shape, target_type.element_type)
            value = tensor.reshape(t_type, value, shape)
            value = tensor_to_memref(value)
            value = convert(
                value, target_type
            )  # should be compatible with a memref cast if not equal
            return value
        case nvgpu.TensorMapDescriptorType(), ir.Type() if str(target_type) == "!llvm.ptr":
            return builtin.unrealized_conversion_cast([llvm.PointerType.get()], [value])
        case ir.Type() as a, nvgpu.TensorMapDescriptorType() if str(a) == "!llvm.ptr":
            return builtin.unrealized_conversion_cast([target_type], [value])
        case ir.ComplexType(), ir.FloatType():
            return complex_dialect.re(value)
        case ir.FloatType(), ir.ComplexType():
            return complex_dialect.create_(
                complex=target_type,
                real=value,
                imaginary=float_of(0, target_type.element_type),
            )
        case ir.ComplexType() as ct, ir.VectorType() as vt if vt.shape == [2]:
            real = complex_dialect.re(value)
            imag = complex_dialect.im(value)
            if ct.element_type != vt.element_type:
                real = convert(real, vt.element_type)
                imag = convert(imag, vt.element_type)
            v = llvm.mlir_undef(vt)
            v = vector.insert(real, v, [], [0])
            v = vector.insert(imag, v, [], [1])
            return v
        case ir.VectorType() as vt, ir.ComplexType() as ct if vt.shape == [2]:
            real = vector.extract(value, [], [0])
            imag = vector.extract(value, [], [1])
            if vt.element_type != ct.element_type:
                real = convert(real, ct.element_type)
                imag = convert(imag, ct.element_type)
            return complex_dialect.create_(complex=ct, real=real, imaginary=imag)
        case ir.Type() as x, ir.Type() as y if x != y:
            raise NotImplementedError(
                f"Type cast not implemented: {x} to {y}. Should this be a bitcast? "
                "Please file an issue with your use case. Thank you!"
            )
        case _:
            raise NotImplementedError(f"NotImplemented converting {value_type} to {target_type}")


def index_of(value: ir.Value | int) -> ir.Value:
    match value:
        case int():
            return arith.constant(T.index(), value=value)
        case ir.Value():
            return convert(value, T.index())
        case _:
            raise NotImplementedError(f"Not implemented for type {type(value)}")


def int_of(value: ir.Value | int | float | bool, ty: ir.Type, *, signed: bool = False) -> ir.Value:
    match value:
        case int() | float() | bool():
            return arith.constant(ty, value=int(value))
        case ir.Value():
            return convert(value, ty, signed=signed)
        case _:
            raise NotImplementedError(f"Not implemented for type {type(value)}")


def float_of(value: ir.Value | float | int, ty: ir.Type) -> ir.Value:
    match value:
        case int() | float():
            return arith.constant(ty, value=float(value))
        case ir.Value():
            return convert(value, ty)
        case _:
            raise NotImplementedError(f"Not implemented for type {type(value)}")


def i32_of(value: ir.Value | int) -> ir.Value:
    return int_of(value, T.i32())


def i64_of(value: ir.Value | int) -> ir.Value:
    return int_of(value, T.i64())


def f32_of(value: ir.Value | float) -> ir.Value:
    return float_of(value, T.f32())


def f64_of(value: ir.Value | float) -> ir.Value:
    return float_of(value, T.f64())


def user_signature_to_external_abi_signature(
    signature: typing.Signature,
) -> typing.Signature:
    """
    The user's function must return an integer indicating if a Python exception occurred,
    and the return value is passed to the callee by pointer.

    See: https://nvidia.github.io/numba-cuda/user/cuda_ffi.html#Device-Function-ABI
    """
    return types.int32(types.CPointer(signature.return_type), *signature.args)


def coerce_to_shape_tuple(
    value: ir.Value | tuple[ir.Value | int, ...],
) -> tuple[ir.Value, ...]:
    def to_indices(values):
        return tuple(map(lambda x: index_of(x), values))

    match value:
        case tuple() if all(isinstance(x, ir.Value) for x in value):
            return to_indices(value)
        case tuple() if all(isinstance(x, int) for x in value):
            return to_indices(value)
        case ir.Value() if isinstance(value.type, ir.MemRefType):
            mr_type = value.type
            assert mr_type.has_rank and mr_type.rank == 1, "Value must be a 1-dimensional memref"
            assert mr_type.has_static_shape, "Shape of memref must be static"
            tuple_result = tuple(
                memref.load(value, [index_of(i)]) for i in range(mr_type.get_dim_size(0))
            )
            return to_indices(tuple_result)
        case _:
            raise NotImplementedError(f"Not implemented for type {type(value)}")


class DeferredLowering:
    """
    Class representing a method call that defers lowering to a later stage.
    When a method's attribute is retrieved from an object, we must
    return a function ready for lowering, but the lowering function
    may require access to the object itself. In this case, we capture
    the object and any additional required context and return a function
    that can be called later to perform the lowering.
    """

    @abstractmethod
    def __call__(self, builder, target, args, kwargs): ...


class DeferredMethodCall(DeferredLowering):
    def __init__(self, _self, lowering_function: AnyCallable[PS]):
        self._self = _self
        self.lowering_function = lowering_function

    def __call__(self, builder, target, args, kwargs):
        return self.lowering_function(builder, target, [self._self] + args, kwargs)


@dataclass
class RangeObject:
    """
    Holds the state of a range object.
    """

    @staticmethod
    def _unify_integer_types(*values: ir.Value) -> ir.Type:
        max_width = 0
        for val in values:
            match val.type:
                case ir.IntegerType():
                    max_width = max(max_width, val.type.width)
                case ir.IndexType():
                    max_width = max(max_width, 64)
                case _:
                    raise InternalCompilerError(f"Unsupported integer type: {val.type}")
        return ir.IntegerType.get_signless(max_width)

    def __init__(self, lower, start: ir.Value, stop: ir.Value, step: ir.Value):
        element_type = self._unify_integer_types(start, stop, step)

        with lower.alloca_insertion_point():
            self._memref = memref.alloca(T.memref(5, element_type), [], [])

        start_val = convert(start, element_type, signed=True)
        stop_val = convert(stop, element_type, signed=True)
        step_val = convert(step, element_type, signed=True)
        zero = arith.constant(element_type, 0)
        one = arith.constant(element_type, 1)

        # Store start/stop/step
        memref.store(start_val, self._memref, [index_of(0)])
        memref.store(stop_val, self._memref, [index_of(1)])
        memref.store(step_val, self._memref, [index_of(2)])

        # Step must not be zero - add runtime check if error checking is enabled
        error_memref = lower._get_or_create_error_global()
        if error_memref is not None:
            from numba_cuda_mlir.mlir_lowering import KERNEL_ERROR_CODES

            step_is_zero = arith.cmpi(predicate=arith.CmpIPredicate.eq, lhs=step_val, rhs=zero)
            with scf.if_ctx_manager(step_is_zero):
                set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ZeroDivisionError])
                scf.yield_([])

        # Compute trip-count in i64 to avoid issues with mixed signed/unsigned
        # semantics on narrow integer types, then truncate back.
        i64 = T.i64()
        start_i64 = convert(start_val, i64, signed=True)
        stop_i64 = convert(stop_val, i64, signed=True)
        step_i64 = convert(step_val, i64, signed=True)
        zero_i64 = arith.constant(i64, 0)
        one_i64 = arith.constant(i64, 1)

        diff = arith.subi(lhs=stop_i64, rhs=start_i64)
        neg_diff = arith.subi(lhs=zero_i64, rhs=diff)
        neg_step = arith.subi(lhs=zero_i64, rhs=step_i64)
        step_positive = arith.cmpi(arith.CmpIPredicate.sgt, lhs=step_i64, rhs=zero_i64)

        # Positive step: ceildiv(diff, step) if diff > 0 else 0
        pos_numerator = arith.addi(lhs=diff, rhs=arith.subi(lhs=step_i64, rhs=one_i64))
        pos_count = arith.divui(lhs=pos_numerator, rhs=step_i64)
        diff_positive = arith.cmpi(arith.CmpIPredicate.sgt, lhs=diff, rhs=zero_i64)
        pos_result = arith.select(diff_positive, pos_count, zero_i64)

        # Negative step: ceildiv(neg_diff, neg_step) if neg_diff > 0 else 0
        neg_numerator = arith.addi(lhs=neg_diff, rhs=arith.subi(lhs=neg_step, rhs=one_i64))
        neg_count = arith.divui(lhs=neg_numerator, rhs=neg_step)
        neg_diff_positive = arith.cmpi(arith.CmpIPredicate.sgt, lhs=neg_diff, rhs=zero_i64)
        neg_result = arith.select(neg_diff_positive, neg_count, zero_i64)

        count = arith.select(step_positive, pos_result, neg_result)
        count = convert(count, element_type, signed=True)
        memref.store(count, self._memref, [index_of(3)])
        memref.store(start_val, self._memref, [index_of(4)])

    def get_mlir_type(self) -> ir.Type:
        return self._memref.type

    @property
    def start(self) -> ir.Value:
        return memref.load(self._memref, [index_of(0)])

    @property
    def stop(self) -> ir.Value:
        return memref.load(self._memref, [index_of(1)])

    @property
    def step(self) -> ir.Value:
        return memref.load(self._memref, [index_of(2)])

    @property
    def count(self) -> ir.Value:
        return memref.load(self._memref, [index_of(3)])

    @property
    def iter(self) -> ir.Value:
        return memref.load(self._memref, [index_of(4)])

    def __next__(self) -> ir.Value:
        raise NotImplementedError()

    def next(self) -> ir.Value:
        """
        Compute (value, is_valid) and update internal state.
        """
        int_type = self._memref.type.element_type
        zero = int_of(0, int_type)
        one = int_of(1, int_type)

        current_count = self.count
        is_valid = arith.cmpi(predicate=arith.CmpIPredicate.sgt, lhs=current_count, rhs=zero)
        current_value = self.iter

        # Compute updated state (predicated without control flow)
        decremented = arith.subi(lhs=current_count, rhs=one)
        next_value = arith.addi(lhs=current_value, rhs=self.step)
        updated_count = arith.select(
            condition=is_valid, true_value=decremented, false_value=current_count
        )
        updated_iter = arith.select(
            condition=is_valid, true_value=next_value, false_value=current_value
        )

        # Store updates
        memref.store(updated_count, self._memref, [index_of(3)])
        memref.store(updated_iter, self._memref, [index_of(4)])

        result_type = T.memref(2, int_type)
        result = memref.alloca(memref=result_type, dynamic_sizes=[], symbol_operands=[])
        memref.store(current_value, result, [index_of(0)])
        memref.store(convert(is_valid, int_type), result, [index_of(1)])
        return result


class IterResult:
    """Holds (value, is_valid) result from iterator next() operation."""

    def __init__(self, value: ir.Value, is_valid: ir.Value):
        self.value = value
        self.is_valid = is_valid


def _zero_value_for_type(mlir_type):
    if isinstance(mlir_type, (ir.IntegerType, ir.IndexType)):
        return arith.constant(result=mlir_type, value=0)
    if isinstance(mlir_type, ir.FloatType):
        return arith.constant(result=mlir_type, value=0.0)
    if isinstance(mlir_type, ir.ComplexType):
        zero = _zero_value_for_type(mlir_type.element_type)
        return complex_dialect.create_(complex=mlir_type, real=zero, imaginary=zero)
    if isinstance(mlir_type, ir.VectorType):
        zero = _zero_value_for_type(mlir_type.element_type)
        return vector_dialect.broadcast(mlir_type, zero)
    if str(mlir_type).startswith("!llvm."):
        return llvm.mlir_zero(res=mlir_type)
    raise InternalCompilerError(f"Cannot materialize iterator dummy value for {mlir_type}.")


@dataclass
class UniTupleIterObject:
    """
    Holds the state of a UniTuple iterator object.

    Supports two storage modes:
    - memref (for builtin scalar element types like i32, f64, index)
    - llvm alloca (for LLVM dialect types like !llvm.ptr, !llvm.struct)

    The storage mode is chosen automatically based on the element type.
    """

    def __init__(
        self,
        lower,
        tuple_storage: ir.Value,
        count: int,
        element_type: ir.Type,
        *,
        uses_llvm_storage: bool = False,
    ):
        self._tuple_storage = tuple_storage
        self._count = count
        self._element_type = element_type
        self._uses_llvm = uses_llvm_storage
        with lower.alloca_insertion_point():
            self._index_memref = memref.alloca(T.memref(1, T.i64()), [], [])
        memref.store(int_of(0, ty=T.i64()), self._index_memref, [index_of(0)])

    @property
    def index(self) -> ir.Value:
        return memref.load(self._index_memref, [index_of(0)])

    def next(self) -> IterResult:
        """
        Compute (value, is_valid) and update internal state.
        Returns an IterResult containing the value and validity flag.
        """
        one = int_of(1, ty=T.i64())
        count = int_of(self._count, ty=T.i64())

        current_index = self.index
        is_valid = arith.cmpi(predicate=arith.CmpIPredicate.slt, lhs=current_index, rhs=count)

        load_if_valid = scf.IfOp(is_valid, results_=[self._element_type], has_else=True)
        with ir.InsertionPoint(load_if_valid.then_block):
            if self._uses_llvm:
                elem_ptr = llvm.getelementptr(
                    llvm.PointerType.get(),
                    self._tuple_storage,
                    [current_index],
                    [GEP_DYNAMIC_INDEX],
                    self._element_type,
                    None,
                )
                current_value = llvm.load(res=self._element_type, addr=elem_ptr)
            else:
                index_as_index = arith.index_cast(out=ir.IndexType.get(), in_=current_index)
                current_value = memref.load(self._tuple_storage, [index_as_index])
            scf.yield_([current_value])
        with ir.InsertionPoint(load_if_valid.else_block):
            scf.yield_([_zero_value_for_type(self._element_type)])
        current_value = load_if_valid.results[0]

        next_index = arith.addi(lhs=current_index, rhs=one)
        updated_index = arith.select(
            condition=is_valid, true_value=next_index, false_value=current_index
        )
        memref.store(updated_index, self._index_memref, [index_of(0)])

        return IterResult(current_value, is_valid)


@dataclass
class ArrayIterObject:
    """
    Holds the state of an array iterator object.
    """

    def __init__(self, lower, array: ir.Value, element_type: ir.Type, length: ir.Value):
        self._array = array
        self._element_type = element_type
        with lower.alloca_insertion_point():
            self._index_memref = memref.alloca(T.memref(1, T.i64()), [], [])
            self._length_memref = memref.alloca(T.memref(1, T.i64()), [], [])
        memref.store(int_of(0, ty=T.i64()), self._index_memref, [index_of(0)])
        memref.store(int_of(length, ty=T.i64()), self._length_memref, [index_of(0)])

    @property
    def index(self) -> ir.Value:
        return memref.load(self._index_memref, [index_of(0)])

    @property
    def length(self) -> ir.Value:
        return memref.load(self._length_memref, [index_of(0)])

    def next(self) -> IterResult:
        """
        Compute (value, is_valid) and update internal state.
        Returns an IterResult containing the value and validity flag.
        """
        one = int_of(1, ty=T.i64())

        current_index = self.index
        length = self.length

        is_valid = arith.cmpi(predicate=arith.CmpIPredicate.slt, lhs=current_index, rhs=length)

        load_if_valid = scf.IfOp(is_valid, results_=[self._element_type], has_else=True)
        with ir.InsertionPoint(load_if_valid.then_block):
            index_as_index = arith.index_cast(out=ir.IndexType.get(), in_=current_index)
            current_value = memref.load(self._array, [index_as_index])
            scf.yield_([current_value])
        with ir.InsertionPoint(load_if_valid.else_block):
            scf.yield_([_zero_value_for_type(self._element_type)])
        current_value = load_if_valid.results[0]

        next_index = arith.addi(lhs=current_index, rhs=one)
        updated_index = arith.select(
            condition=is_valid, true_value=next_index, false_value=current_index
        )
        memref.store(updated_index, self._index_memref, [index_of(0)])

        return IterResult(current_value, is_valid)


def _make_zero_d_subview(array_mr: ir.Value, multi_idx: list[ir.Value]) -> ir.Value:
    """Build a rank-0 memref view at `multi_idx` of `array_mr`.

    All elements of multi_idx must be IndexType values; their count must equal
    the rank of array_mr.  The sizes and strides are passed as the integer 1
    (not as SSA values) so MLIR sees them as static and can rank-reduce the
    result to rank 0.
    """
    source_type: ir.MemRefType = array_mr.type
    dyn_stride = ir.ShapedType.get_dynamic_stride_or_offset()
    result_type = ir.MemRefType.get(
        shape=[],
        element_type=source_type.element_type,
        layout=ir.StridedLayoutAttr.get(dyn_stride, []),
        memory_space=source_type.memory_space,
    )
    rank = source_type.rank
    return memref.subview(
        array_mr,
        offsets=multi_idx,
        sizes=[1] * rank,
        strides=[1] * rank,
        result_type=result_type,
    )


@dataclass
class NdIterIterObject:
    """
    Iterator state for ``np.nditer`` over one or more arrays.

    Each call to ``next()`` yields either a rank-0 memref view of the current
    element (single-array case) or a tuple of rank-0 views (multi-array case).
    Iteration proceeds in C order over the broadcast shape.
    """

    def __init__(
        self,
        lower,
        arrays: list[ir.Value],
        ndim: int,
        broadcast_shape: list[ir.Value] | None = None,
    ):
        # arrays may have differing rank when broadcasting; broadcast_shape
        # is the iteration shape (length == ndim, all IndexType values).
        # When all arrays are the same shape, callers may pass None and the
        # shape is taken from the first array.
        self._arrays = list(arrays)
        self._ndim = ndim
        if broadcast_shape is None:
            broadcast_shape = [memref.dim(arrays[0], index_of(d)) for d in range(ndim)]
        self._shape = list(broadcast_shape)
        with lower.alloca_insertion_point():
            self._index_memref = memref.alloca(T.memref(1, T.i64()), [], [])
        memref.store(int_of(0, ty=T.i64()), self._index_memref, [index_of(0)])
        # Total element count = product of broadcast shape.
        if ndim == 0:
            size_i64 = int_of(1, ty=T.i64())
        else:
            size_idx = self._shape[0]
            for d in range(1, ndim):
                size_idx = arith.muli(size_idx, self._shape[d])
            size_i64 = arith.index_cast(out=T.i64(), in_=size_idx)
        with lower.alloca_insertion_point():
            self._size_memref = memref.alloca(T.memref(1, T.i64()), [], [])
        memref.store(size_i64, self._size_memref, [index_of(0)])

    @property
    def index(self) -> ir.Value:
        return memref.load(self._index_memref, [index_of(0)])

    @property
    def size(self) -> ir.Value:
        return memref.load(self._size_memref, [index_of(0)])

    def _unravel_c_order(self, linear_idx_i64: ir.Value) -> list[ir.Value]:
        """Convert linear index (i64) to a list of IndexType per-dim indices, C order."""
        if self._ndim == 0:
            return []
        if self._ndim == 1:
            return [index_of(linear_idx_i64)]
        per_dim_i64 = [None] * self._ndim
        remaining = linear_idx_i64
        for d in reversed(range(self._ndim)):
            dim_size_i64 = arith.index_cast(out=T.i64(), in_=self._shape[d])
            per_dim_i64[d] = arith.remsi(lhs=remaining, rhs=dim_size_i64)
            remaining = arith.divsi(lhs=remaining, rhs=dim_size_i64)
        return [index_of(v) for v in per_dim_i64]

    def next(self) -> IterResult:
        one_i64 = int_of(1, ty=T.i64())
        current_index = self.index
        size = self.size
        is_valid = arith.cmpi(predicate=arith.CmpIPredicate.slt, lhs=current_index, rhs=size)

        multi_idx = self._unravel_c_order(current_index)
        views = [_make_zero_d_subview(arr, multi_idx) for arr in self._arrays]

        next_index = arith.addi(lhs=current_index, rhs=one_i64)
        updated_index = arith.select(
            condition=is_valid, true_value=next_index, false_value=current_index
        )
        memref.store(updated_index, self._index_memref, [index_of(0)])

        if len(views) == 1:
            return IterResult(views[0], is_valid)
        return IterResult(tuple(views), is_valid)


def _types_match(ty1, ty2, exact=False):
    if exact:
        return ty1 == ty2
    match ty1, ty2:
        case a, b if a == b:
            return True
        case ir.MemRefType() as mr1, ir.MemRefType() as mr2:
            matches = mr1.element_type == mr2.element_type
            matches &= mr1.rank == mr2.rank
            matches &= _memory_spaces_match(mr1, mr2)
            return matches
        case (ir.IndexType(), ir.IntegerType() as i) | (
            ir.IntegerType() as i,
            ir.IndexType(),
        ):
            return i.width == 64
        case _:
            return False


def get_func_type(op: func.FuncOp | gpu.GPUFuncOp) -> ir.FunctionType:
    """Get the FunctionType from a func.FuncOp or gpu.GPUFuncOp."""
    if isinstance(op, func.FuncOp):
        return op.type
    else:
        return op.function_type.value


def _func_types_match(ty1: ir.FunctionType, ty2: ir.FunctionType, exact=False):
    if exact:
        return ty1 == ty2
    if len(ty1.inputs) != len(ty2.inputs):
        return False
    if len(ty1.results) != len(ty2.results):
        return False
    for i in range(len(ty1.inputs)):
        if not _types_match(ty1.inputs[i], ty2.inputs[i], exact):
            return False
    for i in range(len(ty1.results)):
        if not _types_match(ty1.results[i], ty2.results[i], exact):
            return False
    return True


def lookup_callee_in_module(
    name: str,
    mlir_type: ir.FunctionType,
    module: ir.Module | gpu.GPUModuleOp,
    exact=False,
) -> func.FuncOp | gpu.GPUFuncOp | None:
    trace("looking up callee %s", name)
    name = ir.StringAttr.get(name)
    body = module.regions[0].blocks[0]
    func_ops = list(filter(lambda x: isinstance(x, (func.FuncOp, gpu.GPUFuncOp)), body))
    matches = list(
        filter(
            lambda x: x.name == name and _func_types_match(get_func_type(x), mlir_type, exact),
            func_ops,
        )
    )
    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        raise ValueError(
            f"Multiple functions with name {name} and type {mlir_type} found in module"
        )
    return None


def declare_external_function(
    name: str, mlir_type: ir.FunctionType, ip: ir.InsertionPoint
) -> func.FuncOp:
    with ip:
        func_op = func.FuncOp(name=name, type=mlir_type, visibility="private")
    return func_op


def get_or_insert_function(
    name: str, mlir_type: ir.FunctionType, module: ir.Module | gpu.GPUModuleOp
) -> func.FuncOp | gpu.GPUFuncOp:
    if callee := lookup_callee_in_module(name, mlir_type, module):
        return callee
    body = module.regions[0].blocks[0]
    return declare_external_function(name, mlir_type, ir.InsertionPoint(body))


def constant(
    value: ir.Value | int | float | bool | complex | np.number, ty: ir.Type | None
) -> ir.Value:
    """
    Creating an arith.constant for a float type and passing an integer value
    will cause a segfault in MLIR(!!!) so this utility converts the value
    to the appropriate type. This should really be handled in MLIR.
    TODO: send patch
    """
    from numba_cuda_mlir._mlir.dialects import complex as complex_dialect

    match value, ty:
        # First, if the type is None, we infer the MLIR type.
        # This is useful for converting literals to the appropriate type.
        case ir.Value(), t if t is None:
            return value
        case int(), ir.IntegerType():
            return arith.constant(ty, value=value)
        case int(), ir.IndexType():
            return arith.constant(ty, value=value)
        case float(), ir.FloatType():
            return arith.constant(ty, value=value)
        case bool(), None:
            return arith.constant(T.bool(), value=value)
        case complex(), None:
            element_type = T.f64()
            complex_type = T.complex(element_type)
            real_attr = ir.FloatAttr.get(element_type, value.real)
            imag_attr = ir.FloatAttr.get(element_type, value.imag)
            const_attr = ir.ArrayAttr.get([real_attr, imag_attr])
            return complex_dialect.constant(complex=complex_type, value=const_attr)
        case ir.Value(), _:
            return convert(value, ty)
        case _, ir.IntegerType():
            return arith.constant(ty, value=int(value))
        case _, ir.IndexType():
            return arith.constant(ty, value=int(value))
        case _, ir.FloatType():
            return arith.constant(ty, value=float(value))
        case _, ir.ComplexType():
            element_type = ty.element_type
            cplx = complex(value)
            real_attr = ir.FloatAttr.get(element_type, cplx.real)
            imag_attr = ir.FloatAttr.get(element_type, cplx.imag)
            const_attr = ir.ArrayAttr.get([real_attr, imag_attr])
            return complex_dialect.constant(complex=ty, value=const_attr)
        case _:
            # If we get here with a type that doesn't support constants,
            # provide a helpful error message
            import traceback

            stack = "".join(traceback.format_stack())
            raise NotImplementedError(
                f"constant() not implemented for value={value} (type: {type(value).__name__}), "
                f"ty={ty} (type class: {type(ty).__name__ if ty else 'None'})\n"
                f"Stack trace:\n{stack}"
            )


def dims_of_tensor_shape(tens: ir.Value) -> list[ir.Value]:
    from numba_cuda_mlir._mlir.dialects import shape

    ty = tens.type
    sh = shape.shape_of(tens)
    return [shape.get_extent(sh, index_of(i)) for i in range(ty.rank)]


def broadcast_shapes_for_binary_op(
    a: ir.Value, b: ir.Value, builder=None
) -> tuple[ir.Value, ir.Value]:
    from numba_cuda_mlir._mlir.dialects import shape, tensor

    if isinstance(a.type, ir.MemRefType):
        a = memref_to_tensor(a)
    if isinstance(b.type, ir.MemRefType):
        b = memref_to_tensor(b)
    match a.type, b.type:
        case ir.RankedTensorType(), ir.RankedTensorType():
            if a.type.rank == b.type.rank:
                return a, b
            sa, sb = shape.shape_of(a), shape.shape_of(b)
            is_broadcastable = shape.is_broadcastable([sa, sb])
            if builder is not None:
                error_memref = builder._get_or_create_error_global()
                if error_memref is not None:
                    from numba_cuda_mlir.mlir_lowering import KERNEL_ERROR_CODES

                    not_broadcastable = arith.xori(
                        is_broadcastable, arith.constant(result=T.i(1), value=1)
                    )
                    with scf.if_ctx_manager(not_broadcastable):
                        set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
                        scf.yield_([])
            sh = shape.broadcast(shapes=[sb, sa], result=sa.type)
            a, b = tensor.reshape(a, sh), tensor.reshape(b, sh)
            return a, b
        case ir.RankedTensorType() as t, ir.IntegerType() | ir.FloatType():
            tensor_value, scalar_value = a, b
            scalar_value = constant(scalar_value, t.element_type)
            splatted = tensor.splat(t, scalar_value, dims_of_tensor_shape(tensor_value))
            return tensor_value, splatted
        case ir.IntegerType() | ir.FloatType(), ir.RankedTensorType() as t:
            scalar_value, tensor_value = a, b
            scalar_value = constant(scalar_value, t.element_type)
            splatted = tensor.splat(t, scalar_value, dims_of_tensor_shape(tensor_value))
            return splatted, tensor_value
        case ir.IntegerType() | ir.FloatType(), ir.IntegerType() | ir.FloatType():
            return a, b
        case _:
            raise NotImplementedError(f"types {a.type} and {b.type}")


def _simple_scalar_conversion_op(src: ir.Type, dst: ir.Type):
    match src, dst:
        case x, y if x == y:
            return lambda dst, x: x
        case ir.FloatType(), ir.FloatType():
            return arith.truncf if src.width > dst.width else arith.extf
        case ir.IntegerType(), ir.IntegerType():
            return arith.trunci if src.width > dst.width else arith.extsi
        case ir.FloatType(), ir.IntegerType():
            return arith.fptosi
        case ir.IntegerType(), ir.FloatType():
            return arith.sitofp if src.width > 1 else arith.uitofp
        case _:
            raise NotImplementedError(f"Not implemented for types {src} and {dst}")


def simple_scalar_conversion_op(src: ir.Type, dst: ir.Type):
    op = _simple_scalar_conversion_op(src, dst)
    return functools.partial(op, dst)


def expensive_coerce_tensor_type(a: ir.Value, target_element_type: ir.Type) -> ir.Value:
    from numba_cuda_mlir._mlir.dialects import linalg, tensor

    if a.type.element_type == target_element_type:
        return a
    output_type = T.tensor(*a.type.shape, target_element_type)
    src, dst = a.type.element_type, target_element_type
    op = simple_scalar_conversion_op(src, dst)
    init = tensor.empty(sizes=dims_of_tensor_shape(a), element_type=target_element_type)
    result = linalg.MapOp(
        result=[output_type],
        inputs=[a],
        init=init,
    )
    block = result.mapper.blocks.append(a.type.element_type, target_element_type)
    with ir.InsertionPoint(block):
        linalg.yield_([op(block.arguments[0])])
    return result.result[0]


def try_extract_constant(
    value: ir.Value | ir.OpResult | int | float | bool | None,
) -> int | float | bool | None:
    match value:
        case int() | float() | bool() | None:
            return value
    value = ir.Value(value)
    owner = value.owner
    if isinstance(owner, ir.Block):
        return None

    # Get the OpView - owner might be ir.Operation or already an OpView
    if isinstance(owner, ir.Operation):
        opview = owner.opview
    else:
        # owner is already an OpView (e.g., arith.ConstantOp)
        opview = owner

    match opview:
        case arith.ConstantOp(value=v):
            return v.value
        case (
            arith.IndexCastOp()
            | arith.FPToSIOp()
            | arith.FPToUIOp()
            | arith.SIToFPOp()
            | arith.UIToFPOp()
        ):
            return try_extract_constant(opview.operands[0])
        case _:
            return None


def is_nonelike(ty):
    """returns if 'ty' is none"""
    return ty is None or isinstance(ty, types.NoneType) or isinstance(ty, types.Omitted)
