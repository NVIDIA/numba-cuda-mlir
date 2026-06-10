# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.lowering_registry import LoweringRegistry

registry = LoweringRegistry()
lower = registry.lower
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir.lowering_utilities import index_of, memref_to_llvm_ptr, convert, get_type_width
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.cuda import vector as vector_module
from numba_cuda_mlir import types
from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir._mlir.dialects import vector, arith, llvm
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir import ir
from typing import Any

# For llvm.getelementptr - dynamic offset marker
llvm_kDynamic = -2147483648


def _cpointer_to_element_ptr(ptr: ir.Value, index: ir.Value, element_type: ir.Type) -> ir.Value:
    """Convert CPointer + index to element pointer using getelementptr."""
    idx = convert(index, T.i64())
    return llvm.getelementptr(
        llvm.PointerType.get(),
        ptr,
        [idx],
        [llvm_kDynamic],
        element_type,
        None,
    )


def _get_alignment(lower: MLIRLower, args: list[Any], kwargs) -> int:
    """Extract alignment value from positional args or kwargs."""
    if len(args) > 3:
        # Passed as positional argument (4th arg)
        return lower.get_numba_type(args[3].name).literal_value
    else:
        # Search in keyword arguments
        kwargs_dict = dict(kwargs)
        if "alignment" not in kwargs_dict:
            raise ValueError("alignment parameter not found in args or kwargs")
        return lower.get_numba_type(kwargs_dict["alignment"].name).literal_value


def _get_padding_value(element_type: ir.Type) -> ir.Value:
    """Get a zero padding value for the given element type."""
    if isinstance(element_type, ir.FloatType):
        return arith.constant(element_type, 0.0)
    elif isinstance(element_type, ir.IntegerType):
        return arith.constant(element_type, 0)
    else:
        raise NotImplementedError(f"Unsupported element type for vector: {element_type}")


def _get_permutation_map(memref_rank: int, vec_rank: int) -> ir.AffineMap:
    """Get a permutation map for vector transfer ops.

    For a memref of rank M and vector of rank V, the permutation map
    maps from M input dimensions to V output dimensions.
    We map the last V dimensions of the memref to the vector.

    e.g., for memref<8x4xf32> and vector<4xf32>:
    permutation_map = (d0, d1) -> (d1)
    """
    offset = memref_rank - vec_rank
    dims = [ir.AffineDimExpr.get(offset + i) for i in range(vec_rank)]
    return ir.AffineMap.get(memref_rank, 0, dims)


def _vector_with_element_type(vec_type: ir.VectorType, element_type: ir.Type) -> ir.VectorType:
    return ir.VectorType.get(list(ir.VectorType(vec_type).shape), element_type)


def _integer_bits_vector_type(value_vec_type: ir.VectorType) -> ir.VectorType:
    value_vec_type = ir.VectorType(value_vec_type)
    return _vector_with_element_type(
        value_vec_type,
        ir.IntegerType.get_signless(get_type_width(value_vec_type.element_type)),
    )


def _array_storage_vector_type(array: ir.Value, value_vec_type: ir.VectorType) -> ir.VectorType:
    return _vector_with_element_type(value_vec_type, array.type.element_type)


def _scalar_storage_vector_type(lower: MLIRLower, vec_numba_type: VectorType) -> ir.VectorType:
    value_vec_type = lower.get_mlir_type(vec_numba_type)
    return _vector_with_element_type(value_vec_type, lower.get_storage_type(vec_numba_type.dtype))


def _storage_vector_to_value(
    vec_numba_type: VectorType, value: ir.Value, value_vec_type: ir.VectorType
):
    if value.type == value_vec_type:
        return value

    storage_vec_type = ir.VectorType(value.type)
    value_vec_type = ir.VectorType(value_vec_type)
    storage_elem_type = storage_vec_type.element_type
    value_elem_type = value_vec_type.element_type

    if isinstance(vec_numba_type.dtype, (types.Boolean, types.BooleanLiteral)):
        from numba_cuda_mlir.mlir.dialect_exts import arith as arith_ext

        zero = arith_ext.constant(0, storage_vec_type)
        return arith.cmpi(arith.CmpIPredicate.ne, value, zero)

    if isinstance(value_elem_type, ir.FloatType) and isinstance(storage_elem_type, ir.IntegerType):
        bits_type = _integer_bits_vector_type(value_vec_type)
        bits = value
        bits_elem_type = ir.VectorType(bits_type).element_type
        if storage_elem_type.width > bits_elem_type.width:
            bits = arith.trunci(out=bits_type, in_=value)
        elif storage_elem_type.width < bits_elem_type.width:
            bits = arith.extui(out=bits_type, in_=value)
        elif value.type != bits_type:
            bits = arith.bitcast(out=bits_type, in_=value)
        return arith.bitcast(out=value_vec_type, in_=bits)

    return convert(value, value_vec_type)


def _value_vector_to_storage(
    vec_numba_type: VectorType, value: ir.Value, storage_vec_type: ir.VectorType
):
    if value.type == storage_vec_type:
        return value

    value_vec_type = ir.VectorType(value.type)
    storage_vec_type = ir.VectorType(storage_vec_type)
    value_elem_type = value_vec_type.element_type
    storage_elem_type = storage_vec_type.element_type

    if isinstance(vec_numba_type.dtype, (types.Boolean, types.BooleanLiteral)):
        return arith.extui(out=storage_vec_type, in_=value)

    if isinstance(value_elem_type, ir.FloatType) and isinstance(storage_elem_type, ir.IntegerType):
        bits_type = _integer_bits_vector_type(value_vec_type)
        bits = arith.bitcast(out=bits_type, in_=value)
        bits_elem_type = ir.VectorType(bits_type).element_type
        if bits_elem_type.width == storage_elem_type.width:
            return (
                bits
                if bits_type == storage_vec_type
                else arith.bitcast(out=storage_vec_type, in_=bits)
            )
        if bits_elem_type.width < storage_elem_type.width:
            return arith.extui(out=storage_vec_type, in_=bits)
        return arith.trunci(out=storage_vec_type, in_=bits)

    return convert(value, storage_vec_type)


# =============================================================================
# Unaligned Vector Load (3-arg) - uses vector.transfer_read
# =============================================================================


@lower(vector_module.load, types.Array, types.Integer, types.IntegerLiteral)
@lower(vector_module.load, types.Array, types.Integer, types.Tuple)
@lower(vector_module.load, types.Array, types.Integer, types.UniTuple)
def vector_load_1d_index(lower: MLIRLower, target, args: list[Any], kwargs):
    """Unaligned vector load."""
    array = lower.load_var(args[0])
    index = lower.load_var(args[1])

    target_type = lower.get_numba_type(target.name)
    value_vec_type = lower.get_mlir_type(target_type)
    vec_type = _array_storage_vector_type(array, value_vec_type)
    vec_rank = len(value_vec_type.shape)
    memref_rank = array.type.rank

    index = index_of(index)
    padding = _get_padding_value(vec_type.element_type)
    perm_map = _get_permutation_map(memref_rank, vec_rank)
    in_bounds = ir.ArrayAttr.get([ir.BoolAttr.get(True)] * vec_rank)
    result = vector.transfer_read(vec_type, array, [index], perm_map, padding, in_bounds)
    lower.store_var(target, _storage_vector_to_value(target_type, result, value_vec_type))


@lower(vector_module.load, types.Array, types.UniTuple, types.IntegerLiteral)
@lower(vector_module.load, types.Array, types.UniTuple, types.Tuple)
@lower(vector_module.load, types.Array, types.UniTuple, types.UniTuple)
@lower(vector_module.load, types.Array, types.Tuple, types.IntegerLiteral)
@lower(vector_module.load, types.Array, types.Tuple, types.Tuple)
@lower(vector_module.load, types.Array, types.Tuple, types.UniTuple)
def vector_load_nd_index(lower: MLIRLower, target, args: list[Any], kwargs):
    """Unaligned vector load."""
    array = lower.load_var(args[0])
    indices = lower.load_var(args[1])

    target_type = lower.get_numba_type(target.name)
    value_vec_type = lower.get_mlir_type(target_type)
    vec_type = _array_storage_vector_type(array, value_vec_type)
    vec_rank = len(value_vec_type.shape)
    memref_rank = array.type.rank

    indices = [index_of(i) for i in indices]
    padding = _get_padding_value(vec_type.element_type)
    perm_map = _get_permutation_map(memref_rank, vec_rank)
    in_bounds = ir.ArrayAttr.get([ir.BoolAttr.get(True)] * vec_rank)
    result = vector.transfer_read(vec_type, array, indices, perm_map, padding, in_bounds)
    lower.store_var(target, _storage_vector_to_value(target_type, result, value_vec_type))


# =============================================================================
# Aligned Vector Load (4-arg) - uses llvm.load with alignment
# =============================================================================


@lower(
    vector_module.load,
    types.Array,
    types.Integer,
    types.IntegerLiteral,
    types.IntegerLiteral,
)
@lower(vector_module.load, types.Array, types.Integer, types.Tuple, types.IntegerLiteral)
@lower(vector_module.load, types.Array, types.Integer, types.UniTuple, types.IntegerLiteral)
def vector_load_1d_index_aligned(lower: MLIRLower, target, args: list[Any], kwargs):
    """Aligned vector load using llvm.load (1D index)."""
    array = lower.load_var(args[0])
    index = lower.load_var(args[1])

    target_type = lower.get_numba_type(target.name)
    value_vec_type = lower.get_mlir_type(target_type)
    vec_type = _array_storage_vector_type(array, value_vec_type)
    index = index_of(index)

    # Convert memref to LLVM pointer
    ptr = memref_to_llvm_ptr(array, [index], vec_type.element_type)

    # Use llvm.load with alignment
    alignment = _get_alignment(lower, args, kwargs)
    result = llvm.load(vec_type, ptr, alignment=alignment)
    lower.store_var(target, _storage_vector_to_value(target_type, result, value_vec_type))


@lower(
    vector_module.load,
    types.Array,
    types.UniTuple,
    types.IntegerLiteral,
    types.IntegerLiteral,
)
@lower(vector_module.load, types.Array, types.UniTuple, types.Tuple, types.IntegerLiteral)
@lower(
    vector_module.load,
    types.Array,
    types.UniTuple,
    types.UniTuple,
    types.IntegerLiteral,
)
@lower(
    vector_module.load,
    types.Array,
    types.Tuple,
    types.IntegerLiteral,
    types.IntegerLiteral,
)
@lower(vector_module.load, types.Array, types.Tuple, types.Tuple, types.IntegerLiteral)
@lower(vector_module.load, types.Array, types.Tuple, types.UniTuple, types.IntegerLiteral)
def vector_load_nd_index_aligned(lower: MLIRLower, target, args: list[Any], kwargs):
    """Aligned vector load using llvm.load (N-D index)."""
    array = lower.load_var(args[0])
    indices = lower.load_var(args[1])

    target_type = lower.get_numba_type(target.name)
    value_vec_type = lower.get_mlir_type(target_type)
    vec_type = _array_storage_vector_type(array, value_vec_type)
    indices = [index_of(i) for i in indices]

    # Convert memref to LLVM pointer
    ptr = memref_to_llvm_ptr(array, indices, vec_type.element_type)

    # Use llvm.load with alignment
    alignment = _get_alignment(lower, args, kwargs)
    result = llvm.load(vec_type, ptr, alignment=alignment)
    lower.store_var(target, _storage_vector_to_value(target_type, result, value_vec_type))


# =============================================================================
# Unaligned Vector Store (3-arg) - uses vector.transfer_write
# =============================================================================


@lower(vector_module.store, types.Array, types.Integer, VectorType)
def vector_store_1d_index(lower: MLIRLower, target, args: list[Any], kwargs):
    """Unaligned vector store."""
    array = lower.load_var(args[0])
    index = lower.load_var(args[1])
    vec = lower.load_var(args[2])

    vec_numba_type = lower.get_numba_type(args[2].name)
    vec_type = _array_storage_vector_type(array, lower.get_mlir_type(vec_numba_type))
    vec = _value_vector_to_storage(vec_numba_type, vec, vec_type)
    vec_rank = len(vec_type.shape)
    memref_rank = array.type.rank

    index = index_of(index)
    perm_map = _get_permutation_map(memref_rank, vec_rank)
    in_bounds = ir.ArrayAttr.get([ir.BoolAttr.get(True)] * vec_rank)
    vector.transfer_write(None, vec, array, [index], perm_map, in_bounds)
    lower.store_var(target, None)


@lower(vector_module.store, types.Array, types.UniTuple, VectorType)
@lower(vector_module.store, types.Array, types.Tuple, VectorType)
def vector_store_nd_index(lower: MLIRLower, target, args: list[Any], kwargs):
    """Unaligned vector store."""
    array = lower.load_var(args[0])
    indices = lower.load_var(args[1])
    vec = lower.load_var(args[2])

    vec_numba_type = lower.get_numba_type(args[2].name)
    vec_type = _array_storage_vector_type(array, lower.get_mlir_type(vec_numba_type))
    vec = _value_vector_to_storage(vec_numba_type, vec, vec_type)
    vec_rank = len(vec_type.shape)
    memref_rank = array.type.rank

    indices = [index_of(i) for i in indices]
    perm_map = _get_permutation_map(memref_rank, vec_rank)
    in_bounds = ir.ArrayAttr.get([ir.BoolAttr.get(True)] * vec_rank)
    vector.transfer_write(None, vec, array, indices, perm_map, in_bounds)
    lower.store_var(target, None)


# =============================================================================
# Aligned Vector Store (4-arg) - uses llvm.store with alignment
# =============================================================================


@lower(vector_module.store, types.Array, types.Integer, VectorType, types.IntegerLiteral)
def vector_store_1d_index_aligned(lower: MLIRLower, target, args: list[Any], kwargs):
    """Aligned vector store using llvm.store (1D index)."""
    array = lower.load_var(args[0])
    index = lower.load_var(args[1])
    vec = lower.load_var(args[2])
    index = index_of(index)

    vec_numba_type = lower.get_numba_type(args[2].name)
    vec_type = _array_storage_vector_type(array, lower.get_mlir_type(vec_numba_type))
    vec = _value_vector_to_storage(vec_numba_type, vec, vec_type)

    # Convert memref to LLVM pointer
    ptr = memref_to_llvm_ptr(array, [index], vec_type.element_type)

    # Use llvm.store with alignment
    alignment = _get_alignment(lower, args, kwargs)
    llvm.store(vec, ptr, alignment=alignment)
    lower.store_var(target, None)


@lower(vector_module.store, types.Array, types.UniTuple, VectorType, types.IntegerLiteral)
@lower(vector_module.store, types.Array, types.Tuple, VectorType, types.IntegerLiteral)
def vector_store_nd_index_aligned(lower: MLIRLower, target, args: list[Any], kwargs):
    """Aligned vector store using llvm.store (N-D index)."""
    array = lower.load_var(args[0])
    indices = lower.load_var(args[1])
    vec = lower.load_var(args[2])
    indices = [index_of(i) for i in indices]

    vec_numba_type = lower.get_numba_type(args[2].name)
    vec_type = _array_storage_vector_type(array, lower.get_mlir_type(vec_numba_type))
    vec = _value_vector_to_storage(vec_numba_type, vec, vec_type)

    # Convert memref to LLVM pointer
    ptr = memref_to_llvm_ptr(array, indices, vec_type.element_type)

    # Use llvm.store with alignment
    alignment = _get_alignment(lower, args, kwargs)
    llvm.store(vec, ptr, alignment=alignment)
    lower.store_var(target, None)


# =============================================================================
# Aligned CPointer operations - uses llvm.load/store with alignment
# =============================================================================


@lower(
    vector_module.load,
    types.CPointer,
    types.Integer,
    types.IntegerLiteral,
    types.IntegerLiteral,
)
@lower(
    vector_module.load,
    types.CPointer,
    types.Integer,
    types.Tuple,
    types.IntegerLiteral,
)
@lower(
    vector_module.load,
    types.CPointer,
    types.Integer,
    types.UniTuple,
    types.IntegerLiteral,
)
def vector_load_cpointer_1d(lower: MLIRLower, target, args: list[Any], kwargs):
    """Vector load from CPointer using llvm.load with alignment (1D index)."""
    ptr = lower.load_var(args[0])
    index = lower.load_var(args[1])

    target_type = lower.get_numba_type(target.name)
    value_vec_type = lower.get_mlir_type(target_type)
    vec_type = _scalar_storage_vector_type(lower, target_type)

    element_ptr = _cpointer_to_element_ptr(ptr, index, vec_type.element_type)

    alignment = _get_alignment(lower, args, kwargs)
    result = llvm.load(vec_type, element_ptr, alignment=alignment)
    lower.store_var(target, _storage_vector_to_value(target_type, result, value_vec_type))


@lower(vector_module.store, types.CPointer, types.Integer, VectorType, types.IntegerLiteral)
def vector_store_cpointer_1d(lower: MLIRLower, target, args: list[Any], kwargs):
    """Vector store to CPointer using llvm.store with alignment (1D index)."""
    ptr = lower.load_var(args[0])
    index = lower.load_var(args[1])
    vec = lower.load_var(args[2])

    vec_numba_type = lower.get_numba_type(args[2].name)
    vec_type = _scalar_storage_vector_type(lower, vec_numba_type)
    vec = _value_vector_to_storage(vec_numba_type, vec, vec_type)
    element_ptr = _cpointer_to_element_ptr(ptr, index, vec_type.element_type)

    alignment = _get_alignment(lower, args, kwargs)
    llvm.store(vec, element_ptr, alignment=alignment)
    lower.store_var(target, None)
