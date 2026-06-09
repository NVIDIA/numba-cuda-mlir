# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda import typing
import inspect
from numba_cuda_mlir import types
from functools import singledispatch

import importlib.util
import numba_cuda_mlir._mlir.ir as ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir._mlir.extras import types as T
import numpy as np
import ctypes


@singledispatch
def to_numba_type(obj):
    raise TypeError(f"No conversion found for type {obj}")


@to_numba_type.register(ir.Value)
def _(val: ir.Value) -> types.Type:
    """Convert an MLIR Value to a Numba type by extracting its type attribute."""
    return to_numba_type(val.type)


@to_numba_type.register(inspect.Signature)
def numba_sig_from_pyfunc(pysig: inspect.Signature) -> typing.Signature:
    args = list(pysig.parameters.values())
    if not all(hasattr(arg, "annotation") for arg in args):
        raise ValueError("All arguments must have type annotations")
    argtypes = [arg.annotation for arg in args]

    if not all(isinstance(argtype, types.Type) for argtype in argtypes):
        raise TypeError(
            "All arguments must have type annotations of type numba_cuda_mlir.types.Type"
        )

    retty = pysig.return_annotation
    if retty is inspect._empty:
        retty = types.none
    if retty and not isinstance(retty, types.NoneType) and retty is not None:
        raise TypeError("Return type must be None or numba_cuda_mlir.types.none")

    return typing.signature(types.none, *argtypes)


@to_numba_type.register(ir.TypeAttr)
def _(ty: ir.TypeAttr) -> types.Signature:
    return to_numba_type(ty.value)


@to_numba_type.register(ir.Type)
@to_numba_type.register(ir.FunctionType)
def _(ty: ir.Type | ir.FunctionType) -> types.Type | typing.Signature:
    match ty:
        case ty if str(ty) == "!llvm.ptr":
            return types.CPointer(types.void)
        case ir.MemRefType():
            dtype = to_numba_type(ty.element_type)
            return types.Array(dtype, ty.rank, "A")
        case ir.VectorType():
            from numba_cuda_mlir.type_defs.vector_types import VectorType

            elem_type = to_numba_type(ty.element_type)
            return VectorType(elem_type, tuple(ty.shape))
        case ir.IntegerType():
            match ty.width:
                case 1:
                    return types.bool
                case 8:
                    return types.int8
                case 16:
                    return types.int16
                case 32:
                    return types.int32
                case 64:
                    return types.int64
                case _:
                    raise NotImplementedError(f"Not implemented for type {ty}")
        case ir.IndexType():
            return types.int64
        case ir.BF16Type():
            from numba_cuda_mlir.numba_cuda.types.ext_types import bfloat16

            return bfloat16
        case ir.Float4E2M1FNType():
            return types.f4E2M1FN
        case ir.Float6E2M3FNType():
            return types.f6E2M3FN
        case ir.Float6E3M2FNType():
            return types.f6E3M2FN
        case ir.Float8E3M4Type():
            return types.f8E3M4
        case ir.Float8E4M3B11FNUZType():
            return types.f8E4M3B11FNUZ
        case ir.Float8E4M3FNType():
            return types.f8E4M3FN
        case ir.Float8E4M3FNUZType():
            return types.f8E4M3FNUZ
        case ir.Float8E4M3Type():
            return types.f8E4M3
        case ir.Float8E5M2FNUZType():
            return types.f8E5M2FNUZ
        case ir.Float8E5M2Type():
            return types.f8E5M2
        case ir.Float8E8M0FNUType():
            return types.f8E8M0FNU
        case ir.FloatTF32Type():
            return types.tf32
        case ir.FloatType():
            match ty.width:
                case 16:
                    return types.float16
                case 32:
                    return types.float32
                case 64:
                    return types.float64
                case _:
                    raise NotImplementedError(f"Not implemented for type {ty}")
        case ir.FunctionType():
            result_type = types.void
            if len(ty.results) == 1:
                result_type = to_numba_type(ty.results[0])
            elif len(ty.results) > 1:
                # Multiple return values -> create a Tuple type
                result_types = [to_numba_type(r) for r in ty.results]
                # Check if all types are the same for UniTuple
                if all(t == result_types[0] for t in result_types):
                    result_type = types.UniTuple(result_types[0], len(result_types))
                else:
                    result_type = types.Tuple(result_types)
            return typing.signature(result_type, *[to_numba_type(arg) for arg in ty.inputs])
        case ir.Type():
            # we need to downcast LLVM types for some reason...
            # TODO(ajm): this has been fixed, need to update this everywhere
            tystr = str(ty)
            if "llvm.ptr" in tystr:
                return types.ptr
            elif "llvm.struct" in tystr:
                st = llvm.StructType(ty)
                assert not st.opaque, f"Struct type {st.name} is opaque, how did this happen?"
                name = st.name
                named_fe_type = types.AggregateType.get_named_type(name)
                assert named_fe_type is not None, f"No named type found for {name}"
                return named_fe_type
        case _:
            raise NotImplementedError(f"Not implemented for type {ty}")


@to_numba_type.register(type)
def _(obj: type) -> types.Type:
    if ctypes._SimpleCData in obj.mro():
        return ctypes_type_to_numba_type(obj)
    if issubclass(obj, np.generic):
        return np_dtype_to_numba_dtype(np.dtype(obj))
    raise NotImplementedError(f"Not implemented for type {obj}")


def ctypes_type_to_numba_type(obj: ctypes._SimpleCData) -> types.Type:
    match obj:
        case ctypes.c_void_p:
            return types.ptr
        case ctypes.c_bool:
            return types.bool
        case ctypes.c_int8:
            return types.int8
        case ctypes.c_uint8:
            return types.uint8
        case ctypes.c_int16:
            return types.int16
        case ctypes.c_uint16:
            return types.uint16
        case ctypes.c_int32:
            return types.int32
        case ctypes.c_uint32:
            return types.uint32
        case ctypes.c_int64:
            return types.int64
        case ctypes.c_uint64:
            return types.uint64
        case ctypes.c_float:
            return types.float32
        case ctypes.c_double:
            return types.float64
        case _:
            raise NotImplementedError(f"Not implemented for type {obj}")


@singledispatch
def to_mlir_type(obj):
    raise TypeError(f"No conversion found for type {type(obj)}")


@to_mlir_type.register(ir.Value)
def _(val: ir.Value) -> ir.Type:
    """Extract the MLIR type from an MLIR Value (including ScalarValue)."""
    return val.type


@to_mlir_type.register(type)
def _(obj: type) -> ir.Type:
    mro = obj.mro()
    if ctypes._SimpleCData in mro:
        return ctypes_type_to_mlir_type(obj)
    elif issubclass(obj, np.generic):
        return np_dtype_to_mlir_type(obj)
    raise NotImplementedError(f"Not implemented for type {obj}")


def ctypes_type_to_mlir_type(obj: ctypes._SimpleCData) -> ir.Type:
    """
    TODO(ajm): verify signedness matches user's expectations.
    NOTE: The arith dialect is exclusively signless, so it is on us to maintain the
    signedness of the types in the numba type system.
    """
    from numba_cuda_mlir._mlir.dialects import llvm

    match obj:
        case ctypes.c_void_p:
            return llvm.PointerType.get()
        case ctypes.c_bool:
            return T.bool()
        case ctypes.c_int8:
            return T.i8()
        case ctypes.c_uint8:
            return T.ui8()
        case ctypes.c_int16:
            return T.i16()
        case ctypes.c_uint16:
            return T.i16()
        case ctypes.c_int32:
            return T.i32()
        case ctypes.c_uint32:
            return T.i32()
        case ctypes.c_int64:
            return T.i64()
        case ctypes.c_uint64:
            return T.i64()
        case ctypes.c_float:
            return T.f32()
        case ctypes.c_double:
            return T.f64()
        case ctypes.POINTER():
            raise NotImplementedError(f"Not implemented for type {type(obj)}")
        case _:
            raise NotImplementedError(f"Not implemented for type {type(obj)}")


@to_mlir_type.register(typing.Signature)
def _(ty: typing.Signature) -> ir.FunctionType:
    if ty.return_type is types.void:
        return_types = []
    elif isinstance(ty.return_type, types.UniTuple):
        # Multiple returns of same type
        elem_type = to_mlir_type(ty.return_type.dtype)
        return_types = [elem_type] * ty.return_type.count
    elif isinstance(ty.return_type, types.Tuple):
        # Multiple returns of different types
        return_types = [to_mlir_type(t) for t in ty.return_type.types]
    else:
        return_types = [to_mlir_type(ty.return_type)]
    arg_types = tuple(to_mlir_type(arg_type) for arg_type in ty.args)
    return ir.FunctionType.get(results=return_types, inputs=arg_types)


@to_mlir_type.register(np.dtype)
def np_dtype_to_mlir_type(dtype: np.dtype) -> ir.Type:
    match dtype:
        case np.int8:
            return T.i8()
        case np.int16:
            return T.i16()
        case np.int32:
            return T.i32()
        case np.int64:
            return T.i64()
        case np.uint8:
            return T.i8()
        case np.uint16:
            return T.i16()
        case np.uint32:
            return T.i32()
        case np.uint64:
            return T.i64()
        case np.bool:
            return T.bool()
        case np.float16:
            return T.f16()
        case np.float32:
            return T.f32()
        case np.float64:
            return T.f64()
        case np.complex64:
            return T.complex(T.f32())
        case np.complex128:
            return T.complex(T.f64())
        case _ if dtype.kind in ("M", "m"):
            return T.i64()
        case _:
            raise ValueError(f"Cannot convert dtype {dtype} to MLIR type.")


@to_mlir_type.register(types.Type)
def _(ty: types.Type) -> ir.Type:
    from numba_cuda_mlir.type_defs import float_types
    from numba_cuda_mlir.numba_cuda.types.ext_types import (
        Bfloat16,
        GridGroup as GridGroupClass,
    )

    match ty:
        case GridGroupClass():
            return T.i64()
        case types.BooleanLiteral():
            return T.bool()
        case types.bool:
            return T.bool()
        case float_types.BFloat16Type():
            return T.bf16()
        case Bfloat16():
            return T.bf16()
        case _ if isinstance(ty, float_types.SpecialFloatType) or isinstance(
            ty,
            (
                type(types._type_fp8_e5m2),
                type(types._type_fp8_e4m3),
                type(types._type_fp8_e8m0),
                types.bfloat16_raw_class,
            ),
        ):
            from numba_cuda_mlir.models import mlir_data_manager

            return mlir_data_manager.lookup(ty).get_value_type()
        case types.float16:
            return T.f16()
        case types.float32:
            return T.f32()
        case types.float64:
            return T.f64()
        case types.int8 | types.uint8:
            return T.i8()
        case types.int16 | types.uint16:
            return T.i16()
        case types.int32 | types.uint32:
            return T.i32()
        case types.int64 | types.uint64:
            return T.i64()
        case types.NPDatetime() | types.NPTimedelta():
            return T.i64()
        case types.CPointer():
            return llvm.PointerType.get()
        case types.Complex() as ct:
            inner_type = to_mlir_type(ct.underlying_float)
            return T.complex(inner_type)
        case types.IntegerLiteral():
            return T.i64()
        case types.Array():
            from numba_cuda_mlir.types import Record

            dyn = ir.MemRefType.get_dynamic_size()
            dyn_stride = ir.MemRefType.get_dynamic_stride_or_offset()
            layout = ir.StridedLayoutAttr.get(
                offset=dyn_stride,
                strides=[dyn_stride] * ty.ndim,
            )
            # For Record arrays, use byte-based memref (i8) since llvm.ptr
            # is not a valid memref element type
            if isinstance(ty.dtype, Record):
                return ir.MemRefType.get([dyn] * ty.ndim, T.i8(), layout=layout)
            dtype = to_mlir_type(ty.dtype)
            return ir.MemRefType.get([dyn] * ty.ndim, dtype, layout=layout)
        case types.AggregateType():
            st = llvm.StructType.get_identified(ty.name)
            if st.opaque:
                # Check if this is a bitfield struct
                if ty.is_bitfield_struct:
                    # Bitfield struct: get the storage type and convert to MLIR
                    storage_type = ty.get_bitfield_storage_type()
                    mlir_types = [to_mlir_type(storage_type)]
                else:
                    # Regular struct: create a field for each struct field
                    numba_types = [i[1] for i in ty.fields if i[0] is not None]
                    mlir_types = [to_mlir_type(ty) for ty in numba_types]
                st = llvm.StructType.new_identified(ty.name, mlir_types)
            return st
        case types.UnionType():
            # Union is represented as an integer of the size of the largest variant
            # Use the size_bits property which handles all variant types
            max_bits = ty.size_bits

            # Return integer type of appropriate size
            if max_bits <= 8:
                return T.i8()
            elif max_bits <= 16:
                return T.i16()
            elif max_bits <= 32:
                return T.i32()
            elif max_bits <= 64:
                return T.i64()
            else:
                raise NotImplementedError(
                    f"Union '{ty.name}' requires {max_bits} bits but only up to 64-bit unions are currently supported"
                )
        case types.EnumMember() | types.IntEnumMember():
            return to_mlir_type(ty.dtype)
        case _:
            from numba_cuda_mlir.type_defs.vector_types import VectorType

            if isinstance(ty, VectorType):
                elem_type = to_mlir_type(ty.dtype)
                return ir.VectorType.get(list(ty.shape), elem_type)

            try:
                from numba_cuda_mlir.models import mlir_data_manager

                model = mlir_data_manager.lookup(ty)
                return model.get_value_type()
            except KeyError:
                pass

            raise TypeError(f"Unsupported MLIR dtype: {ty}")


@to_mlir_type.register
def _typeref_to_mlir(ty) -> ir.Type:
    """Handle TypeRef by extracting and converting its instance_type."""
    from numba_cuda_mlir.numba_cuda.types.abstract import TypeRef

    if isinstance(ty, TypeRef):
        return to_mlir_type(ty.instance_type)
    raise TypeError(f"Expected TypeRef, got {type(ty)}")


# Explicitly register TypeRef to avoid singledispatch issues with subclassing
try:
    from numba_cuda_mlir.numba_cuda.types.abstract import TypeRef

    to_mlir_type.register(TypeRef, _typeref_to_mlir)
except ImportError:
    pass


@to_numba_type.register(np.dtype)
def np_dtype_to_numba_dtype(dtype: np.dtype) -> types.Type:
    match dtype:
        case np.bool:
            return types.bool
        case np.float16:
            return types.float16
        case np.float32:
            return types.float32
        case np.float64:
            return types.float64
        case np.int8:
            return types.int8
        case np.int16:
            return types.int16
        case np.int32:
            return types.int32
        case np.int64:
            return types.int64
        case np.uint8:
            return types.uint8
        case np.uint16:
            return types.uint16
        case np.uint32:
            return types.uint32
        case np.uint64:
            return types.uint64
        case np.complex64:
            return types.complex64
        case np.complex128:
            return types.complex128
        case _ if dtype.kind == "M":
            return types.NPDatetime(dtype.str[4:-1] if "[" in dtype.str else "")
        case _ if dtype.kind == "m":
            return types.NPTimedelta(dtype.str[4:-1] if "[" in dtype.str else "")
        case _:
            raise TypeError(f"Unsupported numpy dtype: {dtype}")


def integer_of_width(width: int) -> types.Type:
    match width:
        case 1:
            return types.bool
        case 8:
            return types.int8
        case 16:
            return types.int16
        case 32:
            return types.int32
        case 64:
            return types.int64
        case _:
            raise ValueError(f"Unsupported integer width: {width}")


def float_of_width(width: int) -> types.Type:
    match width:
        case 16:
            # TODO: this is ambiguous... as long as we don't allow bad conversions between
            # exotic float types, this won't be a problem.
            return types.float16
        case 32:
            return types.float32
        case 64:
            return types.float64
        case _:
            raise ValueError(f"Unsupported float width: {width}")


if importlib.util.find_spec("torch") is not None:
    import torch

    @to_mlir_type.register(torch.dtype)
    def torch_dtype_to_mlir_dtype(dtype: torch.dtype) -> ir.Type:
        return to_mlir_type(to_numba_type(dtype))

    @to_numba_type.register(torch.dtype)
    def torch_dtype_to_numba_dtype(dtype: torch.dtype) -> types.Type:
        from numba_cuda_mlir.type_defs import float_types

        match dtype:
            case torch.bool:
                return types.bool
            case torch.bfloat16:
                return float_types.BFloat16Type()
            case torch.float16:
                return types.float16
            case torch.float32:
                return types.float32
            case torch.float64:
                return types.float64
            case torch.int8:
                return types.int8
            case torch.int16:
                return types.int16
            case torch.int32:
                return types.int32
            case torch.int64:
                return types.int64


def inline_ptx_type_constraint_to_numba_type(constraint: str) -> types.Type:
    if len(constraint) != 1:
        raise ValueError(f"Invalid inline ptx type constraint: {constraint}")
    numba_types = {
        "h": types.int16,
        "r": types.int32,
        "l": types.int64,
        "f": types.float32,
        "d": types.float64,
        # "q": types.int128, # TODO
        "C": types.ptr,
    }
    if constraint not in numba_types:
        raise ValueError(f"Invalid inline ptx type constraint: {constraint}")
    return numba_types[constraint]


def inline_ptx_type_constraint_to_mlir_type(constraint: str) -> ir.Type:
    if len(constraint) != 1:
        raise ValueError(f"Invalid inline ptx type constraint: {constraint}")
    mlir_types = {
        "h": T.i16(),
        "r": T.i32(),
        "l": T.i64(),
        "f": T.f32(),
        "d": T.f64(),
        # "q": T.i128(),
        "C": llvm.PointerType.get(),
    }
    if constraint not in mlir_types:
        raise ValueError(f"Invalid inline ptx type constraint: {constraint}")
    return mlir_types[constraint]
