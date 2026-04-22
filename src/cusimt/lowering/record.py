# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lowering for NumPy Record (structured dtype) types.

Records are represented as pointers to byte arrays. Field access is done via
byte offset calculations using LLVM GEP and bitcast operations.
"""

from numba.core import types
from cusimt.types import Record, NestedArray
from numba.core import ir as numba_ir
import operator

from cusimt._mlir.dialects import (
    llvm,
    arith,
    memref as memref_dialect,
    complex as complex_dialect,
    scf,
)
from cusimt._mlir import ir
from cusimt._mlir.extras import types as T
from cusimt._mlir.dialects import arith as arith_dialect


def _is_complex_type(mlir_type):
    """Check if a type is MLIR complex type."""
    return isinstance(mlir_type, ir.ComplexType)


def _get_llvm_struct_for_complex(complex_type):
    """Get the LLVM struct type for a complex type."""
    elem_type = complex_type.element_type
    return llvm.StructType.get_literal([elem_type, elem_type])


def _complex_to_llvm_struct(value):
    """Convert MLIR complex value to LLVM struct for storage."""
    complex_type = value.type
    elem_type = complex_type.element_type
    struct_type = _get_llvm_struct_for_complex(complex_type)

    # Extract real and imaginary parts
    real = complex_dialect.re(value)
    imag = complex_dialect.im(value)

    # Build LLVM struct
    undef = llvm.UndefOp(struct_type)
    with_real = llvm.insertvalue(
        container=undef,
        value=real,
        position=ir.DenseI64ArrayAttr.get([0]),
    )
    with_imag = llvm.insertvalue(
        container=with_real,
        value=imag,
        position=ir.DenseI64ArrayAttr.get([1]),
    )
    return with_imag


def _llvm_struct_to_complex(value, complex_type):
    """Convert LLVM struct back to MLIR complex value."""
    elem_type = complex_type.element_type

    # Extract real and imaginary from struct
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

    # Create complex value
    return complex_dialect.create_(complex=complex_type, real=real, imaginary=imag)


# For llvm.getelementptr - dynamic offset marker
LLVM_DYNAMIC = -2147483648

from cusimt.mlir_lowering_registry import MLIRLoweringRegistry
from cusimt.descriptor import MLIRTargetContext

registry = MLIRLoweringRegistry()
lower = registry.lower
from cusimt.logging import trace
from cusimt.lowering_utilities import index_of, convert


def get_record_field_ptr(builder, record_ptr, record_type, field_name):
    """
    Get a pointer to a field within a record.

    Args:
        builder: MLIRLower builder
        record_ptr: LLVM pointer to the record's byte storage
        record_type: The Numba Record type
        field_name: Name of the field to access

    Returns:
        LLVM pointer to the field
    """
    offset = record_type.offset(field_name)
    field_numba_type = record_type.typeof(field_name)

    trace(
        f"get_record_field_ptr: field={field_name}, offset={offset}, type={field_numba_type}"
    )

    # GEP with byte offset
    # llvm.getelementptr ptr[offset] : (!llvm.ptr) -> !llvm.ptr
    offset_val = arith_dialect.constant(T.i64(), offset)
    field_ptr = llvm.getelementptr(
        llvm.PointerType.get(),
        record_ptr,
        [offset_val],
        [LLVM_DYNAMIC],
        T.i8(),
        None,
    )

    return field_ptr, field_numba_type


@registry.lower_getattr_generic(Record)
def lower_record_getattr(
    context: MLIRTargetContext,
    builder: "MLIRLower",
    target: numba_ir.Var,
    value: numba_ir.Var,
    attr: str,
):
    """
    Lower getattr for Record field access.

    record.field_name -> load value from record at field offset
    """
    record_type = builder.get_numba_type(value.name)
    trace("Lowering Record getattr: %s.%s", record_type, attr)

    # Check field exists
    if attr not in record_type.fields:
        available = list(record_type.fields.keys())
        raise AttributeError(
            f"Record has no field '{attr}'. Available fields: {available}"
        )

    # Load the record pointer
    record_ptr = builder.load_var(value)

    # Get pointer to the field
    field_ptr, field_numba_type = get_record_field_ptr(
        builder, record_ptr, record_type, attr
    )

    if isinstance(field_numba_type, NestedArray):
        # Nested array field - create a memref pointing to the data
        result = _load_nested_array_field(
            builder, field_ptr, field_numba_type, record_type
        )
    else:
        # Scalar field - load directly
        field_mlir_type = builder.get_mlir_type(field_numba_type)
        if _is_complex_type(field_mlir_type):
            # Complex types: load as LLVM struct, then convert to complex
            struct_type = _get_llvm_struct_for_complex(field_mlir_type)
            struct_val = llvm.load(struct_type, field_ptr)
            result = _llvm_struct_to_complex(struct_val, field_mlir_type)
        else:
            result = llvm.load(field_mlir_type, field_ptr)

    builder.store_var(target, result)
    trace("Stored Record field '%s' into %s", attr, target.name)


def _load_nested_array_field(builder, field_ptr, nested_type, record_type):
    """
    Load a nested array field from a record.

    Returns the LLVM pointer to the nested array data directly.
    Shape and stride information is available from nested_type at compile time.
    """
    trace(
        "Loading nested array field: shape=%s, dtype=%s",
        nested_type.shape,
        nested_type.dtype,
    )

    # Just return the pointer to the field data directly
    # The caller will use nested_type for shape/stride info when indexing
    trace("Returning pointer to nested array data")
    return field_ptr


@registry.lower_setattr_generic(Record)
def lower_record_setattr(
    context: MLIRTargetContext,
    builder: "MLIRLower",
    sig,
    args,
    attr: str,
):
    """
    Lower setattr for Record field assignment.

    record.field_name = value -> store value to record at field offset
    """
    record_type, value_type = sig.args
    record_var, value_var = args

    trace("Lowering Record setattr: %s.%s = %s", record_type, attr, value_type)

    # Check field exists
    if attr not in record_type.fields:
        available = list(record_type.fields.keys())
        raise AttributeError(
            f"Record has no field '{attr}'. Available fields: {available}"
        )

    # Load the record pointer and value
    record_ptr = builder.load_var(record_var)
    value = builder.load_var(value_var)

    # Get pointer to the field
    field_ptr, field_numba_type = get_record_field_ptr(
        builder, record_ptr, record_type, attr
    )

    if isinstance(field_numba_type, NestedArray):
        # Nested array field - need to copy data
        _store_nested_array_field(
            builder, field_ptr, value, field_numba_type, value_type
        )
    else:
        # Scalar field - store directly
        field_mlir_type = builder.get_mlir_type(field_numba_type)

        # Convert value to field type if needed
        value = convert(value, field_mlir_type)

        if _is_complex_type(field_mlir_type):
            # Complex types: convert to LLVM struct before storing
            value = _complex_to_llvm_struct(value)

        llvm.store(value, field_ptr)

    trace("Stored value into Record field '%s'", attr)


def _store_nested_array_field(builder, field_ptr, value, nested_type, value_type):
    """
    Store a value into a nested array field.

    This copies data from the source array into the nested array storage.
    Source may be a memref or LLVM pointer, destination is an LLVM pointer.
    Uses scf.for to iterate over elements, supporting any rank.
    """
    trace("Storing into nested array field: %s", nested_type)

    elem_size = nested_type.dtype.bitwidth // 8
    elem_mlir_type = builder.get_mlir_type(nested_type.dtype)

    # Determine if source is memref or LLVM pointer
    is_source_memref = isinstance(value.type, ir.MemRefType)
    is_source_llvm_ptr = isinstance(value.type, llvm.PointerType)

    trace(
        "Source type: %s, is_memref=%s, is_llvm_ptr=%s",
        value.type,
        is_source_memref,
        is_source_llvm_ptr,
    )

    shape = list(nested_type.shape)
    strides = list(nested_type.strides)
    ndim = len(shape)

    # Compute total number of elements
    total_elements = 1
    for s in shape:
        total_elements *= s

    # Use scf.for to iterate over linear index, then decompose to multi-dim indices
    for linear_idx in scf.for_(0, total_elements, 1):
        # Convert linear index to i64 for arithmetic
        linear_i64 = arith.index_cast(T.i64(), linear_idx)

        # Decompose linear index to multi-dimensional indices (row-major order)
        # For shape [A, B, C]: idx0 = linear // (B*C), idx1 = (linear // C) % B, idx2 = linear % C
        indices = []
        remaining = linear_i64
        for dim in range(ndim):
            # Compute product of trailing dimensions
            trailing_product = 1
            for d in range(dim + 1, ndim):
                trailing_product *= shape[d]
            trailing_val = arith_dialect.constant(T.i64(), trailing_product)

            if trailing_product > 1:
                idx = arith.divui(remaining, trailing_val)
                remaining = arith.remui(remaining, trailing_val)
            else:
                idx = remaining
            indices.append(idx)

        # Load from source
        if is_source_memref:
            # Convert indices to index type for memref.load
            memref_indices = [arith.index_cast(T.index(), idx) for idx in indices]
            src_elem = memref_dialect.load(value, memref_indices)
        elif is_source_llvm_ptr:
            # Linear byte offset for source (row-major, contiguous)
            src_byte_offset = arith.muli(
                linear_i64, arith_dialect.constant(T.i64(), elem_size)
            )
            src_elem_ptr = llvm.getelementptr(
                llvm.PointerType.get(),
                value,
                [src_byte_offset],
                [LLVM_DYNAMIC],
                T.i8(),
                None,
            )
            src_elem = llvm.load(elem_mlir_type, src_elem_ptr)
        else:
            raise NotImplementedError(
                f"Cannot store from source type {value.type} into nested array"
            )

        # Destination byte offset using strides: sum(idx[i] * stride[i])
        dst_byte_offset = arith_dialect.constant(T.i64(), 0)
        for dim in range(ndim):
            stride_val = arith_dialect.constant(T.i64(), strides[dim])
            term = arith.muli(indices[dim], stride_val)
            dst_byte_offset = arith.addi(dst_byte_offset, term)

        dst_elem_ptr = llvm.getelementptr(
            llvm.PointerType.get(),
            field_ptr,
            [dst_byte_offset],
            [LLVM_DYNAMIC],
            T.i8(),
            None,
        )

        # Handle complex types
        if _is_complex_type(elem_mlir_type):
            src_elem = _complex_to_llvm_struct(src_elem)

        llvm.store(src_elem, dst_elem_ptr)
        scf.yield_([])


# Static getitem/setitem for bracket notation: record['field']


@lower("static_getitem", Record, types.StringLiteral)
@lower(operator.getitem, Record, types.StringLiteral)
def lower_record_static_getitem_str(builder, target, args, kwargs):
    """
    Lower record['field_name'] to getattr.
    """
    record_var = args[0]
    field_name = args[1]  # This is the literal string value

    record_type = builder.get_numba_type(record_var.name)

    # Get the field name from the string literal
    field_name_type = builder.get_numba_type(field_name)
    attr = field_name_type.literal_value

    trace("Lowering Record static_getitem: %s['%s']", record_type, attr)

    # Delegate to getattr lowering
    # We need to manually invoke the getattr logic
    record_ptr = builder.load_var(record_var)

    if attr not in record_type.fields:
        available = list(record_type.fields.keys())
        raise AttributeError(
            f"Record has no field '{attr}'. Available fields: {available}"
        )

    field_ptr, field_numba_type = get_record_field_ptr(
        builder, record_ptr, record_type, attr
    )

    if isinstance(field_numba_type, NestedArray):
        result = _load_nested_array_field(
            builder, field_ptr, field_numba_type, record_type
        )
    else:
        field_mlir_type = builder.get_mlir_type(field_numba_type)
        if _is_complex_type(field_mlir_type):
            # Complex types: load as LLVM struct, then convert to complex
            struct_type = _get_llvm_struct_for_complex(field_mlir_type)
            struct_val = llvm.load(struct_type, field_ptr)
            result = _llvm_struct_to_complex(struct_val, field_mlir_type)
        else:
            result = llvm.load(field_mlir_type, field_ptr)

    builder.store_var(target, result)


@lower("static_getitem", Record, types.IntegerLiteral)
@lower(operator.getitem, Record, types.IntegerLiteral)
def lower_record_static_getitem_int(builder, target, args, kwargs):
    """
    Lower record[0] to getattr using field index.
    """
    record_var = args[0]
    index_var = args[1]

    record_type = builder.get_numba_type(record_var.name)

    # Get the field index from the integer literal
    index_type = builder.get_numba_type(index_var)
    field_index = index_type.literal_value

    # Get field name by index
    field_names = list(record_type.fields.keys())
    if field_index < 0 or field_index >= len(field_names):
        raise IndexError(
            f"Record field index {field_index} out of range (0-{len(field_names)-1})"
        )
    attr = field_names[field_index]

    trace(
        "Lowering Record static_getitem: %s[%s] -> field '%s'",
        record_type,
        field_index,
        attr,
    )

    record_ptr = builder.load_var(record_var)

    field_ptr, field_numba_type = get_record_field_ptr(
        builder, record_ptr, record_type, attr
    )

    if isinstance(field_numba_type, NestedArray):
        result = _load_nested_array_field(
            builder, field_ptr, field_numba_type, record_type
        )
    else:
        field_mlir_type = builder.get_mlir_type(field_numba_type)
        if _is_complex_type(field_mlir_type):
            struct_type = _get_llvm_struct_for_complex(field_mlir_type)
            struct_val = llvm.load(struct_type, field_ptr)
            result = _llvm_struct_to_complex(struct_val, field_mlir_type)
        else:
            result = llvm.load(field_mlir_type, field_ptr)

    builder.store_var(target, result)


@lower("static_setitem", Record, types.StringLiteral, types.Any)
@lower(operator.setitem, Record, types.StringLiteral, types.Any)
def lower_record_static_setitem_str(builder, target, args, kwargs):
    """
    Lower record['field_name'] = value to setattr.
    """
    record_var = args[0]
    field_name_var = args[1]
    value_var = args[2]

    record_type = builder.get_numba_type(record_var.name)
    value_type = builder.get_numba_type(value_var.name)

    # Get the field name from the string literal
    field_name_type = builder.get_numba_type(field_name_var)
    attr = field_name_type.literal_value

    trace(
        "Lowering Record static_setitem: %s['%s'] = %s", record_type, attr, value_type
    )

    record_ptr = builder.load_var(record_var)
    value = builder.load_var(value_var)

    if attr not in record_type.fields:
        available = list(record_type.fields.keys())
        raise AttributeError(
            f"Record has no field '{attr}'. Available fields: {available}"
        )

    field_ptr, field_numba_type = get_record_field_ptr(
        builder, record_ptr, record_type, attr
    )

    if isinstance(field_numba_type, NestedArray):
        _store_nested_array_field(
            builder, field_ptr, value, field_numba_type, value_type
        )
    else:
        field_mlir_type = builder.get_mlir_type(field_numba_type)
        value = convert(value, field_mlir_type)
        if _is_complex_type(field_mlir_type):
            # Complex types: convert to LLVM struct before storing
            value = _complex_to_llvm_struct(value)
        llvm.store(value, field_ptr)


# NestedArray getitem/setitem - work with raw LLVM pointers


def _compute_nested_array_elem_ptr(builder, base_ptr, nested_type, indices):
    """
    Compute pointer to element in nested array using LLVM pointer arithmetic.

    Args:
        base_ptr: LLVM pointer to start of nested array data
        nested_type: The NestedArray type (contains shape, strides, dtype)
        indices: List of index values (LLVM i64)

    Returns:
        LLVM pointer to the element
    """
    elem_size = nested_type.dtype.bitwidth // 8
    strides = list(nested_type.strides)

    # Compute linear byte offset: sum(index[i] * stride[i])
    byte_offset = None
    for i, (idx, stride) in enumerate(zip(indices, strides)):
        # stride is in bytes
        stride_val = arith_dialect.constant(T.i64(), stride)
        term = arith.muli(idx, stride_val)
        if byte_offset is None:
            byte_offset = term
        else:
            byte_offset = arith.addi(byte_offset, term)

    if byte_offset is None:
        # Scalar access (0-D nested array?)
        byte_offset = arith_dialect.constant(T.i64(), 0)

    # GEP to compute element pointer
    elem_ptr = llvm.getelementptr(
        llvm.PointerType.get(),
        base_ptr,
        [byte_offset],
        [LLVM_DYNAMIC],
        T.i8(),
        None,
    )

    return elem_ptr


def lower_nested_array_getitem_int(builder, target, args, kwargs):
    """
    Lower nested_array[i] for 1D nested arrays.
    """
    array_var = args[0]
    index_var = args[1]

    nested_type = builder.get_numba_type(array_var.name)
    trace("Lowering NestedArray getitem: %s[int]", nested_type)

    # Load the base pointer and index
    base_ptr = builder.load_var(array_var)
    index = builder.load_var(index_var)

    # Convert index to i64
    index = convert(index, T.i64())

    # Compute element pointer
    elem_ptr = _compute_nested_array_elem_ptr(builder, base_ptr, nested_type, [index])

    # Load and return the element
    elem_mlir_type = builder.get_mlir_type(nested_type.dtype)
    if _is_complex_type(elem_mlir_type):
        struct_type = _get_llvm_struct_for_complex(elem_mlir_type)
        struct_val = llvm.load(struct_type, elem_ptr)
        result = _llvm_struct_to_complex(struct_val, elem_mlir_type)
    else:
        result = llvm.load(elem_mlir_type, elem_ptr)

    builder.store_var(target, result)
    trace("Loaded NestedArray element into %s", target.name)


def lower_nested_array_setitem_int(builder, target, args, kwargs):
    """
    Lower nested_array[i] = value for 1D nested arrays.
    """
    array_var = args[0]
    index_var = args[1]
    value_var = args[2]

    nested_type = builder.get_numba_type(array_var.name)
    trace("Lowering NestedArray setitem: %s[int] = value", nested_type)

    # Load the base pointer, index, and value
    base_ptr = builder.load_var(array_var)
    index = builder.load_var(index_var)
    value = builder.load_var(value_var)

    # Convert index to i64
    index = convert(index, T.i64())

    # Compute element pointer
    elem_ptr = _compute_nested_array_elem_ptr(builder, base_ptr, nested_type, [index])

    # Convert value to element type if needed
    elem_mlir_type = builder.get_mlir_type(nested_type.dtype)
    value = convert(value, elem_mlir_type)

    # Store the value
    if _is_complex_type(elem_mlir_type):
        value = _complex_to_llvm_struct(value)
    llvm.store(value, elem_ptr)

    trace("Stored value into NestedArray element")


def _unpack_tuple_indices(indices_val, num_indices):
    """Unpack tuple indices from a Python tuple or LLVM struct."""
    indices = []
    if isinstance(indices_val, tuple):
        for idx in indices_val:
            indices.append(convert(idx, T.i64()))
    elif hasattr(indices_val, "type"):
        for i in range(num_indices):
            idx = llvm.extractvalue(
                T.i64(),
                indices_val,
                position=ir.DenseI64ArrayAttr.get([i]),
            )
            indices.append(idx)
    else:
        raise NotImplementedError(f"Cannot unpack indices of type {type(indices_val)}")
    return indices


def lower_nested_array_getitem_tuple(builder, target, args, kwargs):
    """
    Lower nested_array[i, j, ...] for multi-dimensional nested arrays.
    """
    array_var = args[0]
    indices_var = args[1]

    nested_type = builder.get_numba_type(array_var.name)
    indices_type = builder.get_numba_type(indices_var.name)

    trace("Lowering NestedArray getitem: %s[tuple]", nested_type)

    base_ptr = builder.load_var(array_var)
    indices_val = builder.load_var(indices_var)
    indices = _unpack_tuple_indices(indices_val, len(indices_type))

    elem_ptr = _compute_nested_array_elem_ptr(builder, base_ptr, nested_type, indices)

    # Load and return the element
    elem_mlir_type = builder.get_mlir_type(nested_type.dtype)
    if _is_complex_type(elem_mlir_type):
        struct_type = _get_llvm_struct_for_complex(elem_mlir_type)
        struct_val = llvm.load(struct_type, elem_ptr)
        result = _llvm_struct_to_complex(struct_val, elem_mlir_type)
    else:
        result = llvm.load(elem_mlir_type, elem_ptr)

    builder.store_var(target, result)
    trace("Loaded NestedArray element into %s", target.name)


def lower_nested_array_setitem_tuple(builder, target, args, kwargs):
    """
    Lower nested_array[i, j, ...] = value for multi-dimensional nested arrays.
    """
    array_var = args[0]
    indices_var = args[1]
    value_var = args[2]

    nested_type = builder.get_numba_type(array_var.name)
    indices_type = builder.get_numba_type(indices_var.name)

    trace("Lowering NestedArray setitem: %s[tuple] = value", nested_type)

    base_ptr = builder.load_var(array_var)
    value = builder.load_var(value_var)
    indices_val = builder.load_var(indices_var)
    indices = _unpack_tuple_indices(indices_val, len(indices_type))

    elem_ptr = _compute_nested_array_elem_ptr(builder, base_ptr, nested_type, indices)

    # Convert value to element type and store
    elem_mlir_type = builder.get_mlir_type(nested_type.dtype)
    value = convert(value, elem_mlir_type)

    if _is_complex_type(elem_mlir_type):
        value = _complex_to_llvm_struct(value)

    llvm.store(value, elem_ptr)
    trace("Stored value into NestedArray element")
