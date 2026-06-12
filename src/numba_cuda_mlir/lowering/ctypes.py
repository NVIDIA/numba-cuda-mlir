# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.lowering_registry import LoweringRegistry
import operator
import ctypes
from numba_cuda_mlir.numba_cuda.types import ffi as _numba_ffi_type
from numba_cuda_mlir import types
from numba_cuda_mlir.lowering_utilities.type_conversions import (
    to_mlir_type,
    to_numba_type,
)
from numba_cuda_mlir.lowering_utilities import (
    convert,
    i64_of,
    DeferredLowering,
    index_of,
    storage_itemsize_bytes,
)
from numba_cuda_mlir._mlir.dialects import llvm, arith, memref
from numba_cuda_mlir._mlir.dialects import complex as complex_dialect
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir._mlir.extras import types as T

registry = LoweringRegistry()
lower_getattr = registry.lower_getattr

llvm_kDynamic = (
    -2147483648
)  # std::numeric_limits<int32_t>::min(), ugh we need this in the bindings...


def _complex_storage_type(mlir_type):
    if not isinstance(mlir_type, ir.ComplexType):
        return None
    elem_type = mlir_type.element_type
    return llvm.StructType.get_literal([elem_type, elem_type])


def _complex_to_storage(value):
    storage_type = _complex_storage_type(value.type)
    assert storage_type is not None
    real = complex_dialect.re(value)
    imag = complex_dialect.im(value)
    stored = llvm.UndefOp(storage_type)
    stored = llvm.insertvalue(stored, real, [0])
    return llvm.insertvalue(stored, imag, [1])


def _complex_from_storage(value, mlir_type):
    real = llvm.extractvalue(mlir_type.element_type, value, [0])
    imag = llvm.extractvalue(mlir_type.element_type, value, [1])
    return complex_dialect.create_(mlir_type, real, imag)


@registry.lower(ctypes.pointer, types.Array)
def lower_pointer(builder, target, args, kwargs):
    trace()
    args = builder.load_vars(args)
    value = args[0]
    value = convert(value, llvm.PointerType.get())
    builder.store_var(target, value)


@registry.lower_cast(types.Integer, types.Integer)
def lower_integer_cast(builder, target, args, kwargs):
    trace()
    args = builder.load_vars(args)
    value, to_type = args
    to_type = to_mlir_type(to_type)
    value = convert(value, to_type)
    builder.store_var(target, value)


@registry.lower_cast(types.CPointer, types.Integer)
def lower_pointer_to_int_cast(builder, target, args, kwargs):
    """Cast pointer to integer (ptrtoint)."""
    trace()
    ptr, to_type = builder.load_vars(args)
    result = llvm.ptrtoint(res=T.i64(), arg=ptr)
    to_type = to_mlir_type(to_type)
    result = convert(result, to_type)
    builder.store_var(target, result)


@registry.lower_cast(types.Integer, types.CPointer)
def lower_int_to_pointer_cast(builder, target, args, kwargs):
    """Cast integer to pointer (inttoptr)."""
    trace()
    int_val, _ = builder.load_vars(args)
    int_val = convert(int_val, T.i64())
    result = llvm.inttoptr(res=llvm.PointerType.get(), arg=int_val)
    builder.store_var(target, result)


@registry.lower(ctypes.cast, types.Any, types.Any)
def lower_cast(builder, target, args, kwargs):
    trace()
    args = builder.load_vars(args)
    value, to_type = args
    to_type = to_mlir_type(to_type)
    value = convert(value, to_type)
    builder.store_var(target, value)


@registry.lower(ctypes.POINTER, types.Any)
def lower_pointer(builder, target, args, kwargs):
    trace()
    args = builder.load_vars(args)
    value = args[0]
    nb_type = to_numba_type(value)
    nb_type = types.CPointer(nb_type)
    builder.store_var(target, nb_type)


@registry.lower(operator.setitem, types.CPointer, types.Number, types.Any)
def lower_pointer_setitem(builder, target, args, kwargs):
    trace()
    ptr, idx, value = builder.load_vars(args)
    # Get the numba type of the pointer - LLVM pointers are opaque
    nb_type = builder.get_numba_type(args[0])
    ele_ty = nb_type.dtype
    if ele_ty == types.none:
        raise TypeError(
            f"Cannot set item on pointer of type {nb_type}, must cast to a typed pointer first"
        )
    idx = convert(idx, T.i64())
    ele_mlir_ty = builder.get_mlir_type(ele_ty)
    storage_ty = builder.get_storage_type(ele_ty)
    if complex_storage_ty := _complex_storage_type(ele_mlir_ty):
        storage_ty = complex_storage_ty
        value = convert(value, ele_mlir_ty)
        value = _complex_to_storage(value)
    else:
        value = builder.as_storage(ele_ty, value)
    elementptr = llvm.getelementptr(
        llvm.PointerType.get(), ptr, [idx], [llvm_kDynamic], storage_ty, None
    )
    llvm.store(value, elementptr)


@registry.lower(operator.getitem, types.CPointer, types.Number)
def lower_pointer_getitem(builder, target, args, kwargs):
    trace()
    ptr, idx = builder.load_vars(args)
    nb_type = builder.get_numba_type(args[0])
    ele_ty = nb_type.dtype
    if ele_ty == types.none:
        raise TypeError(
            f"Cannot get item on pointer of type {nb_type}, must cast to a typed pointer first"
        )
    idx = convert(idx, T.i64())
    ele_mlir_ty = builder.get_mlir_type(ele_ty)
    storage_ty = builder.get_storage_type(ele_ty)
    complex_storage_ty = _complex_storage_type(ele_mlir_ty)
    if complex_storage_ty is not None:
        storage_ty = complex_storage_ty
    elementptr = llvm.getelementptr(
        llvm.PointerType.get(), ptr, [idx], [llvm_kDynamic], storage_ty, None
    )
    # Load the value from the computed address
    value = llvm.load(storage_ty, elementptr)
    if complex_storage_ty is not None:
        value = _complex_from_storage(value, ele_mlir_ty)
    else:
        value = builder.from_storage(nb_type.dtype, value)
    builder.incref(nb_type.dtype, value)
    builder.store_var(target, value)


@registry.lower(operator.add, types.CPointer, types.Number)
@registry.lower(operator.iadd, types.CPointer, types.Number)
def lower_pointer_add(builder, target, args, kwargs):
    trace()
    ptr, num = builder.load_vars(args)
    nb_type = builder.get_numba_type(args[0])
    ele_ty = nb_type.dtype
    if ele_ty == types.none:
        raise TypeError(
            f"Cannot add to pointer of type {nb_type}, must cast to a typed pointer first"
        )
    w = storage_itemsize_bytes(ele_ty)
    num = convert(num, T.i64())
    ptri = llvm.ptrtoint(res=T.i64(), arg=ptr)
    ptri += num * w
    ptr = llvm.inttoptr(ptr.type, ptri)
    builder.store_var(target, ptr)


@registry.lower(operator.sub, types.CPointer, types.Number)
@registry.lower(operator.isub, types.CPointer, types.Number)
def lower_pointer_sub(builder, target, args, kwargs):
    trace()
    ptr, num = builder.load_vars(args)
    nb_type = builder.get_numba_type(args[0])
    ele_ty = nb_type.dtype
    if ele_ty == types.none:
        raise TypeError(
            f"Cannot subtract from pointer of type {nb_type}, must cast to a typed pointer first"
        )
    w = storage_itemsize_bytes(ele_ty)
    num = convert(num, T.i64())
    ptri = llvm.ptrtoint(res=T.i64(), arg=ptr)
    ptri -= num * w
    ptr = llvm.inttoptr(ptr.type, ptri)
    builder.store_var(target, ptr)


@registry.lower(types.ptr, types.CPointer)
def lower_aggregate_type_ptr(builder, target, args, kwargs):
    ptr = builder.load_var(args[0])
    assert "llvm.ptr" in str(ptr.type), f"Expected LLVM pointer, got {ptr.type}"
    builder.store_var(target, ptr)


@registry.lower(types.ptr, types.AggregateType)
def lower_aggregate_type_ptr(builder, target, args, kwargs):
    var = builder.load_var(args[0])
    with builder.alloca_insertion_point():
        fe_type = to_numba_type(var.type)
        attrs = types.get_numba_cuda_mlir_attributes(fe_type)
        alignment = attrs.get("align", None)
        ptr = llvm.alloca(llvm.PointerType.get(), i64_of(1), var.type, alignment=alignment)
    builder.store_var(target, ptr)


class DeferredFFIFromBuffer(DeferredLowering):
    def __call__(self, builder, target, args, kwargs):
        from numba_cuda_mlir._mlir import ir
        from numba_cuda_mlir.lowering_utilities import get_type_width

        array_value = builder.load_var(args[0])
        mr_type = ir.MemRefType(array_value.type)

        if mr_type.memory_space is not None:
            generic_mr_type = ir.MemRefType.get(
                shape=mr_type.shape,
                element_type=mr_type.element_type,
                layout=mr_type.layout,
            )
            array_value = memref.memory_space_cast(dest=generic_mr_type, source=array_value)
            mr_type = ir.MemRefType(array_value.type)

        ptr_as_index = memref.extract_aligned_pointer_as_index(array_value)

        md = memref.extract_strided_metadata(array_value)
        offset = index_of(md[1])

        elem_type = mr_type.element_type
        elem_bytes = get_type_width(elem_type) // 8

        byte_offset = arith.muli(
            offset,
            arith.constant(T.index(), elem_bytes),
        )
        data_ptr_index = arith.addi(ptr_as_index, byte_offset)

        ptr_as_i64 = arith.index_cast(T.i64(), data_ptr_index)
        ptr = llvm.inttoptr(res=llvm.PointerType.get(), arg=ptr_as_i64)
        builder.store_var(target, ptr)


@lower_getattr(_numba_ffi_type, "from_buffer")
def lower_ffi_from_buffer_getattr(_, mlir_lower, target, ffi_var):
    mlir_lower.store_var(target, DeferredFFIFromBuffer())
