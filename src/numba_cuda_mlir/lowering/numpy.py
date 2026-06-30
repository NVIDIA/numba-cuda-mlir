# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass
import functools
import operator
from numba_cuda_mlir import lowering_utilities
from numba_cuda_mlir.descriptor import MLIRTargetContext
from numba_cuda_mlir.extending import overload, overload_method, typing_registry
from numba_cuda_mlir.errors import InternalCompilerError, ensure_verifies
from numba_cuda_mlir.mlir_lowering import MLIRLower
from numba_cuda_mlir.mlir.dialect_exts import memref, scf, arith, tensor
from numba_cuda_mlir._mlir.dialects import (
    arith as arith_dialect,
    builtin,
    math as math_dialect,
    llvm,
    tensor,
    memref as memref_dialect,
    complex as complex_dialect,
    linalg,
)
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.type_defs.vector_types import VectorType
from numba_cuda_mlir.numba_cuda.np.npyimpl import _make_dtype_object

registry = LoweringRegistry()
lower = registry.lower
lower_getattr = registry.lower_getattr
lower_getattr_generic = registry.lower_getattr_generic
lower_constant = registry.lower_constant
from numba_cuda_mlir.numba_cuda import types
import numba_cuda_mlir.numba_cuda.core.ir as numba_ir
from numba_cuda_mlir.numba_cuda.core import errors
from typing import Any, cast
from numba_cuda_mlir.logging import trace
import numpy as np
from numba_cuda_mlir.numba_cuda.np.arrayobj import numpy_empty_like_nd, _zero_fill_array_method
from .ufunc_registry import UFuncRegistry

from numba_cuda_mlir.lowering_utilities import (
    tensor_to_memref,
    convert,
    constant,
    float_of,
    memref_to_tensor,
    DeferredMethodCall,
    index_of,
    set_error_code_if_zero,
    try_extract_constant,
    NdIterIterObject,
    is_nonelike,
    storage_itemsize_bytes,
    false as false_,
)
from numba_cuda_mlir.mlir_lowering import KERNEL_ERROR_CODES
from numba_cuda_mlir.lowering_utilities.linalg_lowering import (
    lower_np_binop,
    lower_matmul,
    lower_linalg_dot,
    lower_transpose,
)

ufunc_registry = UFuncRegistry("numpy")

# Import to_fixed_tuple for lowering registration
from numba_cuda_mlir.numba_cuda.np.unsafe.ndarray import to_fixed_tuple


@lower(to_fixed_tuple, types.Array, types.IntegerLiteral)
def lower_to_fixed_tuple(builder, target, args, kwargs):
    """
    Convert a 1D array to a fixed-length tuple.
    to_fixed_tuple(array, length) -> UniTuple(array.dtype, length)
    """
    array_var, length_var = args
    array = builder.load_var(array_var)
    array_type = builder.get_numba_type(array_var.name)

    length_type = builder.get_numba_type(length_var)
    tuple_size = int(length_type.literal_value)
    elements = []
    for i in range(tuple_size):
        idx = index_of(i)
        elem = lowering_utilities.array_element_value_load(array_type, array, [idx])
        elements.append(elem)

    builder.store_var(target, tuple(elements))


@lower(_zero_fill_array_method, types.Array)
def lower_zero_fill_array_method(builder: MLIRLower, target, args, kwargs):
    array_var = args[0]
    array = builder.load_var(array_var)
    array_type = builder.get_numba_type(array_var.name)
    ptr_as_index = memref_dialect.extract_aligned_pointer_as_index(array)
    dst_ptr = llvm.inttoptr(llvm.PointerType.get(), convert(ptr_as_index, T.i64()))

    nbytes = constant(storage_itemsize_bytes(array_type.dtype), T.i64())
    for dim in range(array.type.rank):
        extent = convert(memref_dialect.dim(array, index_of(dim)), T.i64())
        nbytes = arith.muli(nbytes, extent)

    llvm.MemsetOp(dst_ptr, constant(0, T.i8()), nbytes, false_())
    if target is not None:
        builder.store_var(target, ir.NoneType.get())


@lower_getattr(types.Array, "size")
def lower_get_size(
    _: MLIRTargetContext,
    builder: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    array = builder.load_var(array)
    rank = array.type.rank
    dims = [memref.dim(array, index_of(i)) for i in range(rank)]
    size = functools.reduce(operator.mul, dims, index_of(1))
    builder.store_var(target, size)


def _lower_array_complex_real_imag(builder, target, array_var, attr):
    """Lower array.real / array.imag for complex arrays.

    For a complex array in memory: [real0, imag0, real1, imag1, ...]
    .real returns a float view at offset 0 with stride 2
    .imag returns a float view at offset 1 with stride 2
    Uses memref.reinterpret_cast to preserve the source memory space
    """
    complex_array = builder.load_var(array_var)
    array_type = complex_array.type
    rank = array_type.rank
    float_type = array_type.element_type.element_type

    dyn = ir.ShapedType.get_dynamic_size()
    dyn_s = ir.ShapedType.get_dynamic_stride_or_offset()

    target_mr_type = ir.MemRefType.get(
        shape=[dyn] * rank,
        element_type=float_type,
        layout=ir.StridedLayoutAttr.get(dyn_s, [dyn_s] * rank),
        memory_space=array_type.memory_space,
    )

    retyped = builtin.unrealized_conversion_cast([target_mr_type], [complex_array])

    md = memref_dialect.extract_strided_metadata(complex_array)
    src_offset = md[1]
    sizes = list(md[2 : 2 + rank])
    strides = list(md[2 + rank : 2 + 2 * rank])

    # Each complex element spans 2 floats in memory.
    two = index_of(2)
    new_offset = arith.muli(src_offset, two)
    if attr == "imag":
        new_offset = arith.addi(new_offset, index_of(1))
    new_strides = [arith.muli(s, two) for s in strides]

    result = memref_dialect.reinterpret_cast(
        target_mr_type,
        retyped,
        offsets=[new_offset],
        sizes=sizes,
        strides=new_strides,
        static_offsets=[dyn_s],
        static_sizes=[dyn] * rank,
        static_strides=[dyn_s] * rank,
    )
    builder.store_var(target, result)


@lower_getattr(types.Array, "real")
def lower_array_real(
    _: MLIRTargetContext,
    builder: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    array_type = builder.get_numba_type(array.name)
    if array_type.dtype in types.complex_domain:
        _lower_array_complex_real_imag(builder, target, array, "real")
    else:
        # For non-complex arrays, .real is identity
        builder.store_var(target, builder.load_var(array))


@lower_getattr(types.Array, "imag")
def lower_array_imag(
    _: MLIRTargetContext,
    builder: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    array_type = builder.get_numba_type(array.name)
    if array_type.dtype in types.complex_domain:
        _lower_array_complex_real_imag(builder, target, array, "imag")
    else:
        raise NotImplementedError(
            f"array.imag not implemented for non-complex dtype {array_type.dtype}"
        )


@lower_getattr(np, "nan")
def lower_numpy_nan_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    target_type = mlir_lower.get_mlir_type(target)
    nan = float_of(np.nan, target_type)
    mlir_lower.store_var(target, nan)


@lower_getattr(np, "float32")
@lower_getattr(np, "int32")
@lower_getattr(np, "int64")
@lower_getattr(np, "float64")
def lower_numpy_dtype_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    target_type = mlir_lower.get_numba_type(array.name)
    dtype = target_type.dtype
    mlir_type = mlir_lower.get_mlir_type(dtype)
    constant = 0.0 if isinstance(dtype, types.Float) else 0
    dtype_dummy_const = arith.constant(
        result=mlir_type,
        value=constant,
    )
    mlir_lower.store_var(target, dtype_dummy_const)


@lower(np.sum, types.Array)
def np_sum_cg(builder, target, args, kwargs):
    """
    Sum over all dimensions of the array using linalg.reduce.
    """
    assert len(args) == 1 and len(kwargs) == 0, "np.sum takes exactly one argument"
    arg = args[0]

    # Extract element types from input array and target
    input_array_type = builder.get_numba_type(arg.name)
    input_element_type = input_array_type.dtype
    input_dtype = builder.get_value_type(input_element_type)

    target_numba_type = builder.get_numba_type(target.name)
    target_dtype = builder.get_value_type(target_numba_type)

    array = builder.load_var(arg)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, array)
    loc = ir.Location.unknown()

    # Create a rank-0 tensor initialized with the neutral element (0) for the reduction.
    c0 = _zero_literal_for_numba_type(input_element_type)
    c0 = arith.constant(result=input_dtype, value=c0)
    result_type = ir.RankedTensorType.get((), input_dtype)
    init = tensor.splat(result_type, c0, [])

    # Reduce across all dimensions of the input tensor.
    rank = array.type.rank
    dims_attr = ir.DenseI64ArrayAttr.get(list(range(rank)))

    reduce_op = linalg.ReduceOp(
        result=[result_type],
        inputs=[array],
        inits=[init],
        dimensions=dims_attr,
    )
    region = reduce_op.combiner
    block = region.blocks.append(input_dtype, input_dtype)
    with ir.InsertionPoint(block):
        out_arg, in_arg = block.arguments
        linalg.yield_([lowering_utilities.add(out_arg, in_arg)])

    # Extract the scalar from the rank-0 tensor result.
    reduce_result_tensor = reduce_op.results[0]
    result_scalar = tensor.extract(reduce_result_tensor, [])

    # Convert to target type if needed
    result_scalar = lowering_utilities.convert(result_scalar, target_dtype)

    builder.store_var(target, result_scalar)


def _get_output_shape_from_reduction(
    arr: types.Array, axis: None | int | tuple[int, ...] = None
) -> types.Type:
    """
    Determine the output type of the sum operation.
    Because the axis and initial keyword arguments are not supported by upstream
    (and therefore by us), we just return the datatype of the array.
    """
    match axis:
        case None:
            return arr.dtype  # scalar
        case int():
            return types.Array(arr.dtype, arr.ndim - 1, arr.layout)
        case tuple() as axes:
            return types.Array(arr.dtype, arr.ndim - len(axes), arr.layout)
        case _:
            raise types.TypingError(f"Invalid axis: {arr=} {axis=}")


def _zero_literal_for_numba_type(numba_type):
    return 0.0 if isinstance(numba_type, (types.Float, types.Complex)) else 0


def _one_literal_for_numba_type(numba_type):
    return 1.0 if isinstance(numba_type, (types.Float, types.Complex)) else 1


def _bool_storage_literal(builder, target_type, value):
    dtype = target_type.dtype
    if isinstance(dtype, (types.Boolean, types.BooleanLiteral)):
        return arith.constant(result=builder.get_storage_type(dtype), value=value)
    return None


def _store_first_output_value(builder, output_arg, output_memref, value):
    output_array_type = builder.get_numba_type(output_arg.name)
    value = lowering_utilities.convert(value, builder.get_value_type(output_array_type.dtype))
    value = lowering_utilities.value_to_storage(output_array_type.dtype, value)
    memref.store(value, output_memref, [index_of(0)])


def _bool_to_value_type(result, value_type):
    if isinstance(value_type, ir.IntegerType):
        if value_type.width == 1:
            return result
        return arith.extui(value_type, result)
    if isinstance(value_type, ir.ComplexType):
        int32_type = ir.IntegerType.get_signless(32)
        result = arith.extui(int32_type, result)
        float_type = value_type.element_type
        real = arith.uitofp(float_type, result)
        zero = arith.constant(result=float_type, value=0.0)
        return complex_dialect.create_(value_type, real, zero)
    int32_type = ir.IntegerType.get_signless(32)
    result = arith.extui(int32_type, result)
    return arith.uitofp(value_type, result)


def _value_is_nonzero(value):
    value_type = value.type
    if isinstance(value_type, ir.ComplexType):
        real = complex_dialect.re(value)
        imag = complex_dialect.im(value)
        zero = arith.constant(result=value_type.element_type, value=0.0)
        real_nonzero = arith.cmpf(arith.CmpFPredicate.UNE, real, zero)
        imag_nonzero = arith.cmpf(arith.CmpFPredicate.UNE, imag, zero)
        return arith.ori(real_nonzero, imag_nonzero)
    if isinstance(value_type, (ir.IntegerType, ir.IndexType)):
        zero = (
            index_of(0)
            if isinstance(value_type, ir.IndexType)
            else arith.constant(result=value_type, value=0)
        )
        return arith.cmpi(arith.CmpIPredicate.ne, value, zero)
    zero = arith.constant(result=value_type, value=0.0)
    return arith.cmpf(arith.CmpFPredicate.UNE, value, zero)


@lower(np.any, types.Array)
def np_any_cg(builder, target, args, kwargs):
    """
    Sum over all dimensions of the array using linalg.reduce.
    """
    assert len(args) == 1 and len(kwargs) == 0, "np.sum takes exactly one argument"
    from numba_cuda_mlir.lowering_utilities import constant, false

    arg = args[0]
    element_type = builder.get_numba_type(target.name)
    dtype = builder.get_value_type(element_type)
    trace("dtype=%s", dtype)
    input_array_type = builder.get_numba_type(arg.name)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, builder.load_var(arg))
    input_dtype = builder.get_value_type(input_array_type.dtype)
    loc = ir.Location.unknown()

    # Create a rank-0 tensor initialized with the neutral element (0) for the reduction.
    result_type = ir.RankedTensorType.get((), dtype)
    init = tensor.splat(result_type, false(), [])

    # Reduce across all dimensions of the input tensor.
    rank = array.type.rank
    dims_attr = ir.DenseI64ArrayAttr.get(list(range(rank)))

    reduce_op = linalg.ReduceOp(
        result=[result_type],
        inputs=[array],
        inits=[init],
        dimensions=dims_attr,
    )
    region = reduce_op.combiner
    block = region.blocks.append(input_dtype, T.bool())
    with ir.InsertionPoint(block):
        from numba_cuda_mlir.lowering_utilities import or_

        in_arg, out_arg = block.arguments
        result = or_(out_arg, _value_is_nonzero(in_arg))
        linalg.yield_([result])

    # Extract the scalar from the rank-0 tensor result.
    reduce_result_tensor = reduce_op.results[0]
    result_scalar = tensor.extract(reduce_result_tensor, [])

    builder.store_var(target, result_scalar)


@lower(np.var, types.Array)
def np_var_cg(builder, target, args, kwargs):
    assert len(args) == 1 and len(kwargs) == 0, "np.var takes exactly one argument"
    arg = args[0]
    element_type = builder.get_numba_type(target.name)
    dtype = builder.get_value_type(element_type)
    trace("dtype=%s", dtype)
    input_array_type = builder.get_numba_type(arg.name)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, builder.load_var(arg))
    input_dtype = builder.get_value_type(input_array_type.dtype)
    single_result_type = ir.RankedTensorType.get((), dtype)
    dims = [tensor.dim(array, index_of(i)) for i in range(array.type.rank)]

    def get_empty():
        return tensor.empty(*dims, dtype)

    def get_single_result():
        return tensor.EmptyOp([], dtype).result

    dims_attr = ir.DenseI64ArrayAttr.get(list(range(array.type.rank)))

    @ensure_verifies
    @linalg.reduce(
        result=[single_result_type],
        inputs=[array],
        inits=[get_single_result()],
        dimensions=dims_attr,
    )
    def reduced(element: input_dtype, acc: dtype):
        return acc + convert(element, dtype)

    summed = tensor.extract(reduced, [])
    num_elements = convert(functools.reduce(operator.mul, dims), dtype)
    mean = summed / num_elements

    @ensure_verifies
    @linalg.reduce(
        result=[single_result_type],
        inputs=[array],
        inits=[get_single_result()],
        dimensions=dims_attr,
    )
    def summed_squares_t(element: input_dtype, acc: dtype):
        diff = convert(element, dtype) - mean
        square = diff * diff
        return acc + square

    summed_squares = tensor.extract(summed_squares_t, [])
    variance = summed_squares / num_elements
    builder.store_var(target, variance)


@lower(np.all, types.Array)
def np_all_cg(builder, target, args, kwargs):
    """
    Sum over all dimensions of the array using linalg.reduce.
    """
    assert len(args) == 1 and len(kwargs) == 0, "np.all takes exactly one argument"
    from numba_cuda_mlir.lowering_utilities import true

    arg = args[0]
    element_type = builder.get_numba_type(target.name)
    dtype = builder.get_value_type(element_type)
    trace("dtype=%s", dtype)
    input_array_type = builder.get_numba_type(arg.name)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, builder.load_var(arg))
    input_dtype = builder.get_value_type(input_array_type.dtype)
    loc = ir.Location.unknown()

    result_type = ir.RankedTensorType.get((), dtype)
    init = tensor.splat(result_type, true(), [])

    rank = array.type.rank
    dims_attr = ir.DenseI64ArrayAttr.get(list(range(rank)))

    reduce_op = linalg.ReduceOp(
        result=[result_type],
        inputs=[array],
        inits=[init],
        dimensions=dims_attr,
    )
    region = reduce_op.combiner
    block = region.blocks.append(input_dtype, T.bool())
    with ir.InsertionPoint(block):
        from numba_cuda_mlir.lowering_utilities import and_

        in_arg, out_arg = block.arguments
        result = and_(out_arg, _value_is_nonzero(in_arg))
        linalg.yield_([result])

    reduce_result_tensor = reduce_op.results[0]
    result_scalar = tensor.extract(reduce_result_tensor, [])

    builder.store_var(target, result_scalar)


@lower(len, types.Array)
def len_cg(builder, target, args, kwargs):
    assert len(args) == 1 and len(kwargs) == 0, "len takes exactly one argument"
    arg = args[0]
    element_type = builder.get_numba_type(target.name)
    dtype = builder.get_mlir_type(element_type)
    array = ir.Value(builder.load_var(arg))
    ty = ir.MemRefType(array.type)
    rank = ty.rank
    product = arith.constant(result=T.index(), value=1)
    for i in range(rank):
        dim_size = memref.dim(source=array, index=arith.constant(result=T.index(), value=i))
        product = lowering_utilities.mul(product, dim_size)
    product = lowering_utilities.convert(product, dtype)
    builder.store_var(target, product)


@lower(len, types.UniTuple)
@lower(len, types.Tuple)
def tuple_len_cg(builder, target, args, kwargs):
    tup = builder.load_var(args[0])
    builder.store_var(target, arith.constant(result=T.index(), value=len(tup)))


@lower(np.mean, types.Array)
def np_mean_cg(builder, target, args, kwargs):
    assert len(args) == 1 and len(kwargs) == 0, "np.mean takes exactly one argument"
    arg = args[0]
    element_type = builder.get_numba_type(target.name)
    dtype = builder.get_value_type(element_type)
    trace("dtype=%s", dtype)
    input_array_type = builder.get_numba_type(arg.name)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, builder.load_var(arg))
    input_elem_type = builder.get_value_type(input_array_type.dtype)

    @ensure_verifies
    @linalg.reduce(
        result=[ir.RankedTensorType.get((), dtype)],
        inputs=[array],
        inits=[tensor.EmptyOp([], dtype)],
        dimensions=ir.DenseI64ArrayAttr.get(list(range(array.type.rank))),
    )
    def summed(element: input_elem_type, acc: dtype):
        return acc + convert(element, dtype)

    summed = tensor.extract(summed, [])
    num_elements = convert(
        functools.reduce(
            operator.mul,
            [tensor.dim(array, index_of(i)) for i in range(array.type.rank)],
        ),
        dtype,
    )
    mean = summed / num_elements
    builder.store_var(target, mean)


def np_dynamic_reshape_cg(builder, target, args, kwargs):
    to_type = builder.get_mlir_type(builder.get_numba_type(target))
    array = builder.load_var(args[0])
    shape = builder.load_var(args[1])
    shty = T.memref(to_type.rank, shape.type.element_type)
    shape = memref.cast(shty, shape)
    reshaped = memref.reshape(
        result=to_type,
        source=array,
        shape=shape,
    )
    builder.store_var(target, reshaped)


@lower(np.reshape, types.Array)
def np_reshape_cg(builder, target, args, kwargs):
    assert len(args) == 2 and len(kwargs) == 0, "np.reshape takes exactly two arguments"
    shape = builder.load_var(args[1])

    # Handle tuple shapes (e.g., out.shape which is a Python tuple of MLIR values)
    if isinstance(shape, (list, tuple)):
        # Convert tuple of shape values to a memref for memref.reshape
        shape_numba_ty = builder.get_numba_type(args[1].name)
        ndim = len(shape)

        # Create a memref to hold the shape values
        shape_memref_type = ir.MemRefType.get([ndim], T.index())
        with builder.alloca_insertion_point():
            alloca_op = memref_dialect.AllocaOp(
                memref=shape_memref_type,
                dynamicSizes=[],
                symbolOperands=[],
            )
        shape_memref = alloca_op.memref

        # Store each shape dimension into the memref
        for i, dim_val in enumerate(shape):
            dim_idx = lowering_utilities.convert(dim_val, T.index())
            memref.store(dim_idx, shape_memref, [index_of(i)])

        # Now use the memref for reshape
        to_type = builder.get_mlir_type(builder.get_numba_type(target))
        array = builder.load_var(args[0])
        reshaped = memref.reshape(
            result=to_type,
            source=array,
            shape=shape_memref,
        )
        builder.store_var(target, reshaped)
        return

    # Handle memref shape (e.g., from np.array or similar)
    shape_ty = shape.type
    match shape_ty:
        case ir.MemRefType():
            return np_dynamic_reshape_cg(builder, target, args, kwargs)
        case _:
            raise NotImplementedError(f"reshape with: {shape_ty=}")


@lower_getattr(types.Array, "reshape")
def lower_array_reshape_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, np_reshape_cg))


@lower_getattr(types.Array, "sum")
def lower_array_sum_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, np_sum_cg))


def _array_item_cg(builder, target, args, kwargs):
    """Lower ``arr.item()``. Implemented for size-1 arrays only."""
    assert not kwargs, "array.item() takes no keyword arguments"

    if len(args) != 1:
        raise NotImplementedError(
            f"array.item() with positional indices is not implemented; got {len(args) - 1} indices"
        )
    array_var = args[0]
    array = builder.load_var(array_var)
    rank = array.type.rank
    indices = [index_of(0)] * rank
    value = memref.load(array, indices)
    builder.store_var(target, value)


@lower_getattr(types.Array, "item")
def lower_array_item_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, _array_item_cg))


def _array_ravel_cg(builder, target, args, kwargs):
    """Lower ``arr.ravel()`` as a 1-D view over ``arr``'s existing storage -
    implements the non-copying case of ravel only.

    A non-copying flat view is only well-defined when the source is
    C-contiguous, so we reject any other layout."""

    # ``memref.reshape`` would be the obvious tool, but it requires the source
    # to have an identity affine map. The ``types.Array`` data model maps
    # every array to a memref with a dynamic strided layout, so we use
    # ``memref.reinterpret_cast`` to produce a 1-D view of length
    # ``prod(shape)`` with unit stride.

    assert not kwargs, "array.ravel() takes no keyword arguments"
    if len(args) != 1:
        raise NotImplementedError(
            f"array.ravel() with positional arguments is not implemented; got {len(args) - 1}"
        )
    array_var = args[0]
    array_ty = builder.get_numba_type(array_var)
    if array_ty.layout != "C":
        raise errors.TypingError(
            f"array.ravel() requires a C-contiguous array, got layout {array_ty.layout!r}"
        )

    src = builder.load_var(array_var)
    src_type: ir.MemRefType = src.type
    ndim = src_type.rank

    md = memref_dialect.extract_strided_metadata(src)
    src_offset = md[1]
    src_sizes = list(md[2 : 2 + ndim])

    if ndim == 0:
        total_size = index_of(1)
    else:
        total_size = src_sizes[0]
        for s in src_sizes[1:]:
            total_size = arith.muli(total_size, s)

    result_type = builder.get_mlir_type(builder.get_numba_type(target))
    dyn = ir.ShapedType.get_dynamic_size()
    dyn_s = ir.ShapedType.get_dynamic_stride_or_offset()
    reshaped = memref_dialect.reinterpret_cast(
        result_type,
        src,
        offsets=[src_offset],
        sizes=[total_size],
        strides=[index_of(1)],
        static_offsets=[dyn_s],
        static_sizes=[dyn],
        static_strides=[dyn_s],
    )
    builder.store_var(target, reshaped)


@lower_getattr(types.Array, "ravel")
def lower_array_ravel_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, _array_ravel_cg))


@overload(np.take, typing_registry=typing_registry)
@overload_method(types.Array, "take", typing_registry=typing_registry)
def numpy_take(a, indices, axis=None):
    # Moved from numba_cuda - not all branches will be fully supported in
    # numba-cuda-mlir
    if is_nonelike(axis):
        if isinstance(a, types.Array) and isinstance(indices, types.Integer):

            def take_impl(a, indices, axis=None):
                if indices > (a.size - 1) or indices < -a.size:
                    raise IndexError("Index out of bounds")
                return a.ravel()[indices]

            return take_impl

        if isinstance(a, types.Array) and isinstance(indices, types.Array):
            F_order = indices.layout == "F"

            def take_impl(a, indices, axis=None):
                ret = np.empty(indices.size, dtype=a.dtype)
                if F_order:
                    walker = indices.copy()  # get C order
                else:
                    walker = indices
                it = np.nditer(walker)
                i = 0
                flat = a.ravel()
                for x in it:
                    if x > (a.size - 1) or x < -a.size:
                        raise IndexError("Index out of bounds")
                    ret[i] = flat[x]
                    i = i + 1
                return ret.reshape(indices.shape)

            return take_impl

        if isinstance(a, types.Array) and isinstance(indices, (types.List, types.BaseTuple)):

            def take_impl(a, indices, axis=None):
                convert = np.array(indices)
                return np.take(a, convert)

            return take_impl
    else:
        if isinstance(a, types.Array) and isinstance(indices, types.Integer):
            t = (0,) * (a.ndim - 1)

            # np.squeeze is too hard to implement in Numba as the tuple "t"
            # needs to be allocated beforehand we don't know it's size until
            # code gets executed.
            @register_jitable
            def _squeeze(r, axis):
                tup = tuple(t)
                j = 0
                assert axis < len(r.shape) and r.shape[axis] == 1, r.shape
                for idx in range(len(r.shape)):
                    s = r.shape[idx]
                    if idx != axis:
                        tup = tuple_setitem(tup, j, s)
                        j += 1
                return r.reshape(tup)

            def take_impl(a, indices, axis=None):
                r = np.take(a, (indices,), axis=axis)
                if a.ndim == 1:
                    return r[0]
                if axis < 0:
                    axis += a.ndim
                return _squeeze(r, axis)

            return take_impl

        if isinstance(a, types.Array) and isinstance(
            indices, (types.Array, types.List, types.BaseTuple)
        ):
            ndim = a.ndim

            _getitem = generate_getitem_setitem_with_axis(ndim, "getitem")
            _setitem = generate_getitem_setitem_with_axis(ndim, "setitem")

            def take_impl(a, indices, axis=None):
                if axis < 0:
                    axis += a.ndim

                if axis < 0 or axis >= a.ndim:
                    msg = f"axis {axis} is out of bounds for array of dimension {a.ndim}"
                    raise ValueError(msg)

                shape = tuple_setitem(a.shape, axis, len(indices))
                out = np.empty(shape, dtype=a.dtype)
                for i in range(len(indices)):
                    y = _getitem(a, indices[i], axis)
                    _setitem(out, i, axis, y)
                return out

            return take_impl


class Slice:
    def __init__(
        self,
        start: ir.Value,
        stop: ir.Value | None = None,
        step: ir.Value | None = None,
    ):
        # Handle None for start (e.g., x[:5] has start=None meaning 0)
        if isinstance(start, ir.NoneType):
            start = None
        self.start: ir.Value = (
            lowering_utilities.convert(start, T.index())
            if start is not None
            else arith.index_cast(arith.constant(result=T.i64(), value=0), to=T.index())
        )
        # Handle None for stop (e.g., x[2:] has stop=None meaning end of array)
        if isinstance(stop, ir.NoneType):
            stop = None
        self.stop: ir.Value | None = lowering_utilities.convert(stop, T.index()) if stop else None
        # Handle None for step (default to 1)
        if isinstance(step, ir.NoneType):
            step = None
        self.step: ir.Value = (
            lowering_utilities.convert(step, T.index())
            if step
            else arith.constant(result=T.index(), value=1)
        )

    def __str__(self):
        return f"Slice(start={self.start}, stop={self.stop}, step={self.step})"

    def __repr__(self):
        return str(self)

    def __iter__(self):
        yield self.start
        yield self.stop
        yield self.step


@lower(slice, types.VarArg(types.Any))
def lower_array_slice(builder, target, args, kwargs):
    args = map(builder.load_var, args)
    builder.store_var(target, Slice(*args))


from numba_cuda_mlir.types import Record


def _lower_record_array_getitem(builder, target, args, kwargs):
    """
    Handle array[index] where array.dtype is Record.

    For record arrays, we compute a byte offset and return an llvm.ptr
    to the record's storage within the array.
    """
    array_var = args[0]
    index_var = args[1]

    array_numba_type = builder.get_numba_type(array_var.name)
    record_type = array_numba_type.dtype
    record_size = record_type.size

    trace("Record array getitem: record_size=%s", record_size)

    # Load the array (memref<?xi8>) and index
    array = builder.load_var(array_var)
    # Handle both variable and constant indices
    if isinstance(index_var, int):
        # Static/constant index - create constant directly
        index = arith_dialect.constant(T.i64(), index_var)
    else:
        index = builder.load_var(index_var)

    # For Record arrays, the memref is memref<?xi8> with byte strides.
    # We need to compute: base_ptr + index * record_size
    # Note: This assumes contiguous arrays (no views with non-zero offsets).
    # Supporting views would require extract_strided_metadata, but that breaks
    # pointer extraction on some memref types.

    # Get the aligned pointer directly from the array
    ptr_as_index = memref_dialect.extract_aligned_pointer_as_index(array)

    # Use record_size as stride (assumes contiguous layout)
    stride = arith_dialect.constant(T.i64(), record_size)

    # Convert index to i64 for arithmetic
    index_i64 = convert(index, T.i64())

    # Compute byte offset = index * stride
    byte_offset = arith.muli(index_i64, stride)

    # Add offset to base pointer
    ptr_as_i64 = convert(ptr_as_index, T.i64())
    result_ptr_i64 = arith.addi(ptr_as_i64, byte_offset)

    # Convert back to pointer
    result_ptr = llvm.inttoptr(llvm.PointerType.get(), result_ptr_i64)

    builder.store_var(target, result_ptr)
    trace("Record array getitem: stored ptr to %s", target.name)


def _get_memref_strides(memref_type: ir.MemRefType) -> list[int]:
    """Extract static strides from a memref's layout, computing C-contiguous
    strides when the layout is the default identity map."""
    dyn = ir.ShapedType.get_dynamic_stride_or_offset()
    dyn_size = ir.ShapedType.get_dynamic_size()
    layout = memref_type.layout
    if isinstance(layout, ir.StridedLayoutAttr):
        return list(layout.strides)
    source_shape = list(memref_type.shape)
    result = []
    for i in range(memref_type.rank):
        suffix = source_shape[i + 1 :]
        if all(d != dyn_size for d in suffix):
            stride = 1
            for d in suffix:
                stride *= d
            result.append(stride)
        else:
            result.append(dyn)
    return result


def _rank_reducing_subview(
    array: ir.Value,
    sv_offsets: list[ir.Value],
    sv_sizes: list[ir.Value],
    sv_strides: list[ir.Value],
    dims_to_drop: list[bool],
) -> ir.Value:
    """Create a memref.subview that drops dimensions marked in *dims_to_drop*,
    then collapse_shape to produce the rank-reduced result.  This is used for
    both ``arr[i]`` and ``arr[(i, j)]`` style indexing where scalar indices
    reduce the array rank."""
    array_type: ir.MemRefType = array.type
    source_rank = array_type.rank
    dyn = ir.ShapedType.get_dynamic_stride_or_offset()
    dyn_size = ir.ShapedType.get_dynamic_size()

    subview_shape = [1 if dims_to_drop[i] else dyn_size for i in range(source_rank)]
    source_strides = _get_memref_strides(array_type)

    subview_type = ir.MemRefType.get(
        subview_shape,
        array_type.element_type,
        layout=ir.StridedLayoutAttr.get(dyn, source_strides),
        memory_space=array_type.memory_space,
    )
    subview = memref.subview(array, sv_offsets, sv_sizes, sv_strides, result_type=subview_type)

    # Build reassociation: group each dropped dim with the next kept dim.
    reassociation: list[list[int]] = []
    pending: list[int] = []
    for i in range(source_rank):
        pending.append(i)
        if not dims_to_drop[i]:
            reassociation.append(pending)
            pending = []
    if pending:
        if reassociation:
            reassociation[-1].extend(pending)
        else:
            reassociation.append(pending)

    # Compute result strides for the collapsed memref. A dimension group
    # requires a dynamic stride if any source dimension in the group has a
    # dynamic stride OR a dynamic size (since the product of sizes affects
    # the effective stride after collapsing).
    result_strides = []
    for group in reassociation:
        needs_dyn = any(source_strides[d] == dyn or subview_shape[d] == dyn_size for d in group)
        if needs_dyn:
            result_strides.append(dyn)
        else:
            result_strides.append(source_strides[group[-1]])
    n_kept = len(result_strides)
    result_type = ir.MemRefType.get(
        [dyn_size] * n_kept,
        array_type.element_type,
        layout=ir.StridedLayoutAttr.get(dyn, result_strides),
        memory_space=array_type.memory_space,
    )
    return memref.collapse_shape(result_type, subview, reassociation)


@lower(operator.getitem, types.Array, types.Number)
@lower(operator.getitem, types.Array, types.Integer)
@lower(operator.getitem, types.Buffer, types.Integer)
def lower_array_getitem(builder, target, args, kwargs):
    trace()

    # Check if this is a record array
    array_numba_type = builder.get_numba_type(args[0].name)
    from numba_cuda_mlir.types import NestedArray

    if isinstance(array_numba_type, NestedArray):
        from numba_cuda_mlir.lowering.record import lower_nested_array_getitem_int

        return lower_nested_array_getitem_int(builder, target, args, kwargs)

    if isinstance(array_numba_type.dtype, Record):
        return _lower_record_array_getitem(builder, target, args, kwargs)

    array = builder.load_var(args[0])
    # Handle both variable and constant indices
    if isinstance(args[1], int):
        index = index_of(args[1])
    else:
        index = builder.load_var(args[1])
        index = index_of(index)
    array_type = array.type

    if not array_type.has_rank:
        raise NotImplementedError("NYI: unranked memrefs")

    if array_type.rank == 1:
        value = lowering_utilities.array_element_value_load(array_numba_type, array, [index])
    else:
        rank = array_type.rank
        sv_offsets = [index] + [index_of(0)] * (rank - 1)
        sv_sizes = [index_of(1)] + [memref.dim(array, index_of(i)) for i in range(1, rank)]
        sv_strides = [index_of(1)] * rank
        dims_to_drop = [True] + [False] * (rank - 1)
        value = _rank_reducing_subview(array, sv_offsets, sv_sizes, sv_strides, dims_to_drop)

    builder.store_var(target, value)


@dataclass
class MemRefSlice:
    offset: ir.Value
    size: ir.Value
    stride: ir.Value
    static_offset: int
    static_size: int
    static_stride: int

    @staticmethod
    def get(array: ir.Value, slice: Slice):
        from numba_cuda_mlir.lowering_utilities import try_extract_constant as extract

        # first, last, step, static first, static last, static step
        f, l, s, sf, sl, ss = [None for _ in range(6)]

        # size, static size
        sz, ssz = None, None

        if v := extract(slice.start):
            sf = v
        else:
            f = slice.start
        if v := extract(slice.stop):
            sl = v
        elif slice.stop is None:
            l = memref.dim(array, index_of(0))
        else:
            l = slice.stop
        if v := extract(slice.step):
            ss = v
        else:
            s = slice.step

        if sf and sl:
            ssz = sl - sf
        else:
            sz = l - f

        return MemRefSlice(
            offset=f,
            size=sz,
            stride=s,
            static_offset=sf,
            static_size=ssz,
            static_stride=ss,
        )


class MemRefSlices:
    def __init__(self, *slices: list[MemRefSlice]):
        self.slices = slices

    def __add__(self, other: "MemRefSlices") -> "MemRefSlices":
        return MemRefSlices(*self.slices, *other.slices)

    def __len__(self):
        return len(self.slices)

    def subview(self, mr: ir.Value) -> dict[str, list[ir.Value | int]]:
        # TODO(ajm): replace with tensor.generate
        kws = {
            "offsets": [
                (slice.static_offset if slice.static_offset is not None else slice.offset)
                for slice in self.slices
            ],
            "sizes": [
                (slice.static_size if slice.static_size is not None else slice.size)
                for slice in self.slices
            ],
            "strides": [
                (slice.static_stride if slice.static_stride is not None else slice.stride)
                for slice in self.slices
            ],
        }
        dyn = ir.ShapedType.get_dynamic_stride_or_offset()
        source_strides, _ = mr.type.get_strides_and_offset()
        result_strides = []
        for src_stride, s in zip(source_strides, self.slices):
            if s.static_stride is not None and src_stride != dyn:
                result_strides.append(src_stride * s.static_stride)
            else:
                result_strides.append(dyn)
        layout = ir.StridedLayoutAttr.get(offset=dyn, strides=result_strides)
        mrt = ir.MemRefType.get(
            element_type=mr.type.element_type,
            shape=[
                (slice.static_size if slice.static_size is not None else dyn)
                for slice in self.slices
            ],
            layout=layout,
            memory_space=mr.type.memory_space,
        )
        return memref.subview(mr, **kws, result_type=mrt)


def slice_memref(mr: ir.Value, mrs: MemRefSlices) -> ir.Value:
    value = mrs.subview(mr)
    return value


@lower(operator.getitem, types.Array, types.SliceType)
def lower_array_slice_getitem(builder, target, args, kwargs):
    trace()
    mr = builder.load_var(args[0])
    mr_type = mr.type
    dtype = mr_type.element_type
    rank = mr_type.rank
    slice = builder.load_var(args[1])
    start, stop, step = slice.start, slice.stop, slice.step
    if start is None:
        start = arith.index_cast(arith.constant(result=T.i64(), value=0), to=T.index())
    if stop is None:
        stop = memref.dim(mr, index_of(0))
    if step is None:
        step = index_of(1)
    starts, stops, steps = [start], [stop], [step]
    for i in range(1, rank):
        starts.append(index_of(0))
        stops.append(memref.dim(mr, index_of(i)))
        steps.append(index_of(1))

    dyn = ir.ShapedType.get_dynamic_stride_or_offset()
    source_strides, _ = mr_type.get_strides_and_offset()
    result_strides = []
    for src_stride, step in zip(source_strides, steps):
        step_val = try_extract_constant(step)
        if step_val is not None and src_stride != dyn:
            result_strides.append(src_stride * step_val)
        else:
            result_strides.append(dyn)
    layout = ir.StridedLayoutAttr.get(offset=dyn, strides=result_strides)
    mrt = ir.MemRefType.get(
        element_type=dtype,
        shape=[dyn for _ in range(rank)],
        layout=layout,
        memory_space=mr_type.memory_space,
    )
    sizes = [(stop - start) // step for start, stop, step in zip(starts, stops, steps)]
    mr = memref.subview(mr, offsets=starts, sizes=sizes, strides=steps, result_type=mrt)
    builder.store_var(target, mr)


def _idx_c(value: int) -> ir.Value:
    return arith.constant(result=T.index(), value=value)


def _view_transform_axis(
    sizes: list[ir.Value],
    strides: list[ir.Value],
    axis: int,
    old_size: int,
    new_size: int,
) -> tuple[list[ir.Value], list[ir.Value]]:
    new_sizes = list(sizes)
    new_strides = list(strides)
    if new_size < old_size:
        ratio = old_size // new_size
        new_sizes[axis] = arith.muli(sizes[axis], _idx_c(ratio))
    else:
        bytelen = arith.muli(sizes[axis], _idx_c(old_size))
        new_sizes[axis] = arith.divsi(bytelen, _idx_c(new_size))
    for j in range(len(strides)):
        if j == axis:
            new_strides[j] = _idx_c(1)
        else:
            byte_stride = arith.muli(strides[j], _idx_c(old_size))
            new_strides[j] = arith.divsi(byte_stride, _idx_c(new_size))
    return new_sizes, new_strides


def lower_array_view_cg(builder, target, args, kwargs):
    _self, dtype = [builder.load_var(arg) for arg in args]
    mr_ty: ir.MemRefType = _self.type
    assert mr_ty.has_rank, "NYI: unranked memrefs"

    new_dtype = lowering_utilities.to_mlir_type(dtype)
    old_dtype = mr_ty.element_type

    if old_dtype == new_dtype:
        builder.store_var(target, _self)
        return

    new_dtype_bytes = lowering_utilities.get_type_width(new_dtype) // 8
    old_dtype_bytes = lowering_utilities.get_type_width(old_dtype) // 8

    rank = mr_ty.rank
    if rank == 0:
        raise ValueError(f"Cannot .view() a 0-d array (target {target.name!r})")

    target_ty = builder.get_numba_type(target.name)
    target_mr = builder.get_mlir_type(target_ty)
    layout = target_ty.layout

    # To support shared memory, memory space must be propagated with the memref
    target_mr = ir.MemRefType.get(
        shape=target_mr.shape,
        element_type=target_mr.element_type,
        layout=target_mr.layout,
        memory_space=mr_ty.memory_space,
    )
    retyped = builtin.unrealized_conversion_cast([target_mr], [_self])

    if old_dtype_bytes == new_dtype_bytes:
        builder.store_var(target, retyped)
        return

    md = memref_dialect.extract_strided_metadata(_self)
    offset_e = md[1]
    sizes = list(md[2 : 2 + rank])
    strides = list(md[2 + rank : 2 + 2 * rank])

    new_offset = arith.divsi(
        arith.muli(offset_e, _idx_c(old_dtype_bytes)),
        _idx_c(new_dtype_bytes),
    )

    def _reinterpret(axis: int) -> ir.Value:
        new_sizes, new_strides = _view_transform_axis(
            sizes,
            strides,
            axis,
            old_dtype_bytes,
            new_dtype_bytes,
        )

        dyn = ir.ShapedType.get_dynamic_size()
        dyn_s = ir.ShapedType.get_dynamic_stride_or_offset()
        return memref_dialect.reinterpret_cast(
            target_mr,
            retyped,
            offsets=[new_offset],
            sizes=new_sizes,
            strides=new_strides,
            static_offsets=[dyn_s],
            static_sizes=[dyn] * rank,
            static_strides=[dyn_s] * rank,
        )

    # ====================
    # "C", "F" and "A" with rank 1 layouts
    # ====================

    if layout in ("C", "F") or rank == 1:
        axis = (rank - 1) if (layout == "C" or rank == 1) else 0
        builder.store_var(target, _reinterpret(axis))
        return

    # ====================
    # "A" layout with rank >= 2
    # ====================

    error_memref = builder._get_or_create_error_global()
    one_idx = _idx_c(1)

    def _dispatch(axis_idx: int) -> ir.Value:
        if axis_idx < 0:
            if error_memref is not None:
                set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
            return retyped
        is_contig = arith.cmpi(arith.CmpIPredicate.eq, strides[axis_idx], one_idx)
        if_op = scf.IfOp(is_contig, results_=[target_mr], has_else=True)
        with ir.InsertionPoint(if_op.then_block):
            scf.yield_([_reinterpret(axis_idx)])
        with ir.InsertionPoint(if_op.else_block):
            scf.yield_([_dispatch(axis_idx - 1)])
        return if_op.results[0]

    builder.store_var(target, _dispatch(rank - 1))


@lower_getattr(types.Array, "view")
def lower_array_view_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, lower_array_view_cg))


@lower(operator.getitem, types.UniTuple, types.Number)
@lower(operator.getitem, types.Tuple, types.Number)
def lower_uni_tuple_getitem(builder, target, args, kwargs):
    """
    Tuples are always Python tuples in the varmap. Static integer indices
    resolve at compile time; dynamic indices emit an scf.index_switch.
    """
    trace("args=%s", args)
    from numba_cuda_mlir.lowering_utilities import convert

    target_type = builder.get_numba_type(target.name)
    tup = builder.load_var(args[0])
    index = builder.load_var(args[1]) if isinstance(args[1], numba_ir.Var) else args[1]
    assert isinstance(tup, tuple), f"Expected Python tuple, got {type(tup)}"
    match tup, index:
        case tuple(), ir.Value():
            tup = builder.lower_literal_if_needed(tup)
            index = index_of(index)
            result_type = builder.get_mlir_type(target_type)
            error_memref = builder._get_or_create_error_global()

            def default(op):
                if error_memref is not None:
                    set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[IndexError])
                result = convert(tup[0], result_type)
                scf.yield_([result])

            def case_builder(op, case_index, case_value):
                result = convert(tup[case_value], result_type)
                scf.yield_([result])

            cases = range(len(tup))
            cases = ir.DenseI64ArrayAttr.get(cases)
            result = scf.index_switch(
                results=[result_type],
                arg=index,
                cases=cases,
                default_body_builder=default,
                case_body_builder=case_builder,
            )
            builder.store_var(target, result)
            if isinstance(result, ir.Value):
                builder.incref(target_type, result)
        case tuple(), int() as index:
            val = tup[index]
            builder.store_var(target, val)
            if isinstance(val, ir.Value):
                builder.incref(target_type, val)
        case _:
            raise InternalCompilerError(f"Tuple index must be an integer, got {type(args[1])}")


@lower("static_getitem", types.UniTuple, types.SliceLiteral)
@lower("static_getitem", types.Tuple, types.SliceLiteral)
def lower_static_tuple_slice(builder, target, args, kwargs):
    tup = builder.load_var(args[0])
    index = args[1]
    assert isinstance(tup, tuple), f"Expected Python tuple, got {type(tup)}"
    assert isinstance(index, slice), f"Expected slice, got {type(index)}"
    builder.store_var(target, tup[index])


def _lower_record_array_setitem(builder, target, args, kwargs):
    """
    Handle array[index] = record where array.dtype is Record.

    Copies the entire record bytes from source to destination.
    """
    array_var = args[0]
    index_var = args[1]
    value_var = args[2]

    array_numba_type = builder.get_numba_type(array_var.name)
    record_type = array_numba_type.dtype
    record_size = record_type.size

    trace("Record array setitem: record_size=%s", record_size)

    # Load the array (memref), index, and source record pointer
    array = builder.load_var(array_var)
    index = builder.load_var(index_var)
    src_ptr = builder.load_var(value_var)

    # Get destination pointer - assumes contiguous arrays (no views with non-zero offsets)
    ptr_as_index = memref_dialect.extract_aligned_pointer_as_index(array)

    # Convert index and pointer to i64 - use convert which handles any source type
    index_i64 = lowering_utilities.convert(index, T.i64())
    # Use record_size as stride (assumes contiguous layout)
    stride = arith_dialect.constant(T.i64(), record_size)
    byte_offset = arith.muli(index_i64, stride)
    ptr_as_i64 = lowering_utilities.convert(ptr_as_index, T.i64())
    dest_ptr_i64 = arith.addi(ptr_as_i64, byte_offset)
    dest_ptr = llvm.inttoptr(llvm.PointerType.get(), dest_ptr_i64)

    # Copy record_size bytes from src to dest using llvm.memcpy
    size_val = arith_dialect.constant(T.i64(), record_size)
    is_volatile = arith_dialect.constant(T.bool(), 0)
    llvm.intr_memcpy(dest_ptr, src_ptr, size_val, is_volatile)

    trace("Record array setitem: copied %s bytes", record_size)


@lower(operator.setitem, types.Array, types.Integer, types.StringLiteral)
def lower_charseq_array_setitem_string(builder: MLIRLower, target, args, kwargs):
    """arr[i] = "XYZ" for arrays with CharSeq or UnicodeCharSeq dtype."""
    from numba_cuda_mlir.lowering_utilities import GEP_DYNAMIC_INDEX, false as false_

    array_numba_type = builder.get_numba_type(args[0].name)
    element_type = array_numba_type.dtype
    string_type = builder.get_numba_type(args[2].name)
    string_val = (
        string_type.literal_value
        if isinstance(string_type, types.StringLiteral)
        else builder.load_var(args[2])
    )
    assert isinstance(string_val, str)

    array = builder.load_var(args[0])
    index = builder.load_var(args[1])
    index_i64 = convert(index, T.i64())

    if isinstance(element_type, types.CharSeq):
        element_size = element_type.count
        encoded = string_val.encode("utf-8")
    elif isinstance(element_type, types.UnicodeCharSeq):
        unicode_byte_width = np.dtype("U1").itemsize
        element_size = element_type.count * unicode_byte_width
        code_points = [ord(c) for c in string_val]
        encoded = b""
        for cp in code_points:
            encoded += cp.to_bytes(unicode_byte_width, byteorder="little")
    else:
        raise NotImplementedError(
            f"String literal assignment not supported for array dtype {element_type}"
        )

    ptr_as_index = memref.extract_aligned_pointer_as_index(array)
    stride = constant(element_size, T.i64())
    byte_offset = arith.muli(index_i64, stride)
    ptr_as_i64 = convert(ptr_as_index, T.i64())
    result_ptr_i64 = arith.addi(ptr_as_i64, byte_offset)
    dst_ptr = llvm.inttoptr(llvm.PointerType.get(), result_ptr_i64)

    zero = constant(0, T.i8())
    size_val = constant(element_size, T.i64())
    llvm.MemsetOp(dst_ptr, zero, size_val, false_())

    for i, byte_val in enumerate(encoded):
        if i >= element_size:
            break
        byte_const = constant(byte_val, T.i8())
        offset = constant(i, T.i64())
        byte_ptr = llvm.getelementptr(
            llvm.PointerType.get(),
            dst_ptr,
            [offset],
            [GEP_DYNAMIC_INDEX],
            T.i8(),
            None,
        )
        llvm.store(byte_const, byte_ptr)


@lower(operator.setitem, types.Array, types.Integer, types.Boolean)
@lower(operator.setitem, types.Array, types.Integer, types.Number)
@lower(operator.setitem, types.Array, types.Integer, types.NPDatetime)
@lower(operator.setitem, types.Array, types.Integer, types.NPTimedelta)
@lower(operator.setitem, types.Array, types.Integer, Record)
@lower(operator.setitem, types.Array, types.Integer, VectorType)
def lower_array_setitem(builder: MLIRLower, target, args, kwargs):
    trace()

    # Check if this is a record array
    array_numba_type = builder.get_numba_type(args[0].name)
    from numba_cuda_mlir.types import NestedArray

    if isinstance(array_numba_type, NestedArray):
        from numba_cuda_mlir.lowering.record import lower_nested_array_setitem_int

        return lower_nested_array_setitem_int(builder, target, args, kwargs)

    if isinstance(array_numba_type.dtype, Record):
        return _lower_record_array_setitem(builder, target, args, kwargs)

    array = builder.load_var(args[0])
    index = builder.load_var(args[1])
    index = lowering_utilities.index_of(index)
    value = builder.load_var(args[2])
    mrt = array.type
    if mrt.rank == 1:
        lowering_utilities.array_element_value_store(array_numba_type, array, [index], value)
    else:
        rankm1 = mrt.rank - 1

        @scf.forall_(
            [0] * rankm1,
            [memref.dim(source=array, index=index_of(i + 1)) for i in range(rankm1)],
            [1] * rankm1,
        )
        def assign_slice(*indices):
            lowering_utilities.array_element_value_store(
                array_numba_type, array, [index] + list(indices), value
            )


def _setitem_index_to_memref_index(index: ir.Value | int) -> ir.Value:
    match index:
        case ir.Value() | int():
            return index_of(index)
        case _:
            raise InternalCompilerError(f"Index must be an integer or a value, got {type(index)}")


def _setitem_indices_to_memref_indices(
    indices: tuple[ir.Value | int, ...] | ir.Value,
) -> tuple[ir.Value, ...]:
    match indices:
        case tuple():
            return tuple(_setitem_index_to_memref_index(i) for i in indices)
        case ir.Value() as value if (
            isinstance(value.type, ir.MemRefType)
            and value.type.has_rank
            and value.type.rank == 1
            and value.type.has_static_shape
        ):
            mr_type = value.type
            num_elements = mr_type.get_dim_size(0)
            return tuple(index_of(memref.load(value, [index_of(i)])) for i in range(num_elements))
        case _:
            raise InternalCompilerError(
                f"Indices must be a tuple of integers or a value, got {type(indices)}"
            )


@lower(operator.setitem, types.Array, types.Tuple, types.Any)
@lower(operator.setitem, types.Array, types.UniTuple, types.Any)
def lower_array_setitem_tuple(builder, target, args, kwargs):
    # Check if this is a nested array (embedded in a record)
    from numba_cuda_mlir.types import NestedArray

    array_numba_type = builder.get_numba_type(args[0].name)
    if isinstance(array_numba_type, NestedArray):
        from numba_cuda_mlir.lowering.record import lower_nested_array_setitem_tuple

        return lower_nested_array_setitem_tuple(builder, target, args, kwargs)

    array = builder.load_var(args[0])
    tup = args[1]
    tup = builder.load_var(tup) if isinstance(tup, numba_ir.Var) else tup
    indices = _setitem_indices_to_memref_indices(tup)
    value = builder.load_var(args[2])
    lowering_utilities.array_element_value_store(array_numba_type, array, indices, value)


@lower(operator.setitem, types.Array, types.SliceType, types.Any)
def lower_array_slice_setitem(builder, target, args, kwargs):
    """Lower arr[slice] = value - fill sliced region with scalar value."""
    trace()
    array = builder.load_var(args[0])
    slice_val = builder.load_var(args[1])
    value = builder.load_var(args[2])

    array_numba_type = builder.get_numba_type(args[0].name)
    mr_type = array.type
    rank = mr_type.rank

    # Parse slice bounds for dimension 0
    start, stop, step = slice_val.start, slice_val.stop, slice_val.step
    if start is None:
        start = index_of(0)
    else:
        start = index_of(start)
    if stop is None:
        stop = memref.dim(array, index_of(0))
    else:
        stop = index_of(stop)
    if step is None:
        step = index_of(1)
    else:
        step = index_of(step)

    # Build bounds for all dimensions - first dim uses slice, rest use full range
    starts = [start] + [index_of(0)] * (rank - 1)
    stops = [stop] + [memref.dim(array, index_of(i + 1)) for i in range(rank - 1)]
    steps = [step] + [index_of(1)] * (rank - 1)

    @scf.forall_(starts, stops, steps)
    def fill_all(*indices):
        lowering_utilities.array_element_value_store(array_numba_type, array, list(indices), value)


@lower(operator.getitem, types.Array, types.UniTuple)
@lower(operator.getitem, types.Array, types.Tuple)
def lower_array_tuple_getitem(builder: MLIRLower, target, args, kwargs):
    if len(args) != 2:
        raise InternalCompilerError(f"Tuple getitem takes exactly two arguments, got {len(args)}")

    # Check if this is a nested array (embedded in a record)
    from numba_cuda_mlir.types import NestedArray

    array_numba_type = builder.get_numba_type(args[0].name)
    if isinstance(array_numba_type, NestedArray):
        from numba_cuda_mlir.lowering.record import lower_nested_array_getitem_tuple

        return lower_nested_array_getitem_tuple(builder, target, args, kwargs)

    array = builder.load_var(args[0])
    tuple_indices = builder.load_var(args[1])

    array_type = array.type
    if not isinstance(array_type, (ir.MemRefType, ir.RankedTensorType)) or not array_type.has_rank:
        raise InternalCompilerError(
            f"Array must be a statically-ranked memref or tensor, got {array_type}"
        )

    if not isinstance(tuple_indices, tuple):
        raise InternalCompilerError(f"Tuple indices must be a tuple, got {type(tuple_indices)}")

    target_type = builder.get_numba_type(target.name)
    source_rank = array_type.rank
    n_indexed = len(tuple_indices)
    n_trailing = source_rank - n_indexed

    error_memref = builder._get_or_create_error_global()
    zero = arith.constant(result=T.index(), value=0)
    dims = [memref.dim(array, index_of(i)) for i in range(source_rank)]

    offsets, sizes, strides, is_scalar = [], [], [], []
    for i, index in enumerate(tuple_indices):
        match index:
            case Slice(start=start, stop=stop, step=step):
                if not isinstance(target_type, types.Array):
                    raise TypeError(
                        f"Target type {target_type} is not an array, but a slice was used to index it"
                    )
                offsets.append(start)
                end = stop or dims[i]
                sizes.append(end - start)
                strides.append(step or 1)
                is_scalar.append(False)
                if error_memref is not None:
                    is_not_positive = arith.cmpi(arith.CmpIPredicate.sle, strides[-1], zero)
                    with scf.if_ctx_manager(is_not_positive):
                        set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
                        scf.yield_([])
            case int() as i:
                offsets.append(arith.constant(result=T.index(), value=i))
                sizes.append(1)
                strides.append(1)
                is_scalar.append(True)
            case ir.Value() as value:
                offsets.append(lowering_utilities.convert(value, T.index()))
                sizes.append(1)
                strides.append(1)
                is_scalar.append(True)
            case _:
                raise InternalCompilerError(
                    f"Tuple indices must be a slice or int, got {type(index)}"
                )

    # Extend with full-extent entries for unindexed trailing dimensions
    full_offsets = list(offsets) + [index_of(0)] * n_trailing
    full_sizes = list(sizes) + [dims[n_indexed + i] for i in range(n_trailing)]
    full_strides = list(strides) + [index_of(1)] * n_trailing
    full_is_scalar = list(is_scalar) + [False] * n_trailing

    match target_type:
        case types.Array():
            n_kept = sum(1 for sc in full_is_scalar if not sc)
            if n_kept != target_type.ndim:
                raise InternalCompilerError(
                    f"Result rank {n_kept} does not match target type ndim {target_type.ndim}"
                )
            value = _rank_reducing_subview(
                array, full_offsets, full_sizes, full_strides, full_is_scalar
            )
            builder.store_var(target, value)
        case types.Number() | types.Boolean():
            value = lowering_utilities.array_element_value_load(
                array_numba_type, array, full_offsets
            )
            builder.store_var(target, value)
        case _:
            raise InternalCompilerError(
                f"Target type {target_type} is not an array or number, but a tuple was used to index it"
            )


# ============================================================================
# Reduction Operations: min, max, prod
# ============================================================================


def create_reduction_op(builder, target, args, kwargs, init_value_fn, combiner_fn, op_name):
    """
    Public helper for implementing reduction operations using linalg.ReduceOp.

    This function provides a reusable pattern for implementing NumPy reduction
    operations that collapse all dimensions of an array to a scalar result.
    Use this to add new reduction operations following the established pattern.

    Args:
        builder: MLIR builder instance
        target: Target variable for the result
        args: Function arguments (should be single array)
        kwargs: Keyword arguments (should be empty)
        init_value_fn: Function(element_type) -> init_value for the reduction
        combiner_fn: Function(element_type, out_arg, in_arg) -> combined_value
        op_name: Name of the operation for error messages

    Examples:
        See np_min_cg, np_max_cg, np_prod_cg for complete usage examples.
    """
    assert len(args) == 1 and len(kwargs) == 0, f"{op_name} takes exactly one argument"
    arg = args[0]

    # Extract element types from input array and target
    input_array_type = builder.get_numba_type(arg.name)
    input_element_type = input_array_type.dtype
    input_dtype = builder.get_value_type(input_element_type)

    target_numba_type = builder.get_numba_type(target.name)
    target_dtype = builder.get_value_type(target_numba_type)

    array = builder.load_var(arg)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, array)
    loc = ir.Location.unknown()

    # Get initialization value from the provided function using input element type
    init_val = init_value_fn(input_element_type)

    init_const = arith.constant(result=input_dtype, value=init_val)
    result_type = ir.RankedTensorType.get((), input_dtype)
    init = tensor.splat(result_type, init_const, [])

    # Reduce across all dimensions
    rank = array.type.rank
    dims_attr = ir.DenseI64ArrayAttr.get(list(range(rank)))

    reduce_op = linalg.ReduceOp(
        result=[result_type],
        inputs=[array],
        inits=[init],
        dimensions=dims_attr,
    )
    region = reduce_op.combiner
    block = region.blocks.append()
    in_arg = block.add_argument(input_dtype, loc=loc)
    out_arg = block.add_argument(input_dtype, loc=loc)
    with ir.InsertionPoint(block):
        # Apply combiner operation using input element type
        result_val = combiner_fn(input_element_type, out_arg, in_arg)
        linalg.yield_([result_val])

    # Extract the scalar from the rank-0 tensor result
    reduce_result_tensor = next(iter(cast(Any, reduce_op.operation.results)))
    result_scalar = tensor.extract(reduce_result_tensor, [])

    # Convert to target type if needed
    result_scalar = lowering_utilities.convert(result_scalar, target_dtype)

    builder.store_var(target, result_scalar)


@ufunc_registry.register(np.min)
@lower(np.min, types.Array)
def np_min_cg(builder, target, args, kwargs):
    """
    Find minimum over all dimensions of the array using linalg.reduce.
    """

    def min_init_value(element_type):
        """Initialize with maximum value for the type (using two's complement for unsigned)"""
        if isinstance(element_type, types.Float):
            return float("inf")
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                # For signed: max value is 2^(bits-1) - 1
                return (2 ** (element_type.bitwidth - 1)) - 1
            else:
                # For unsigned: max value is -1 in two's complement representation
                # This works for all bitwidths (uint8=-1, uint16=-1, uint32=-1, uint64=-1)
                return -1
        else:
            raise ValueError(f"Unsupported type for np.min: {element_type}")

    def min_combiner(element_type, out_arg, in_arg):
        """Select minimum value"""
        if isinstance(element_type, types.Float):
            return arith.minimumf(out_arg, in_arg)
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return arith.minsi(out_arg, in_arg)
            else:
                return arith.minui(out_arg, in_arg)
        else:
            raise ValueError(f"Unsupported type for np.min: {element_type}")

    create_reduction_op(builder, target, args, kwargs, min_init_value, min_combiner, "np.min")


@lower_getattr(types.Array, "min")
def lower_array_min_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, np_min_cg))


@lower(np.max, types.Array)
def np_max_cg(builder, target, args, kwargs):
    """
    Find maximum over all dimensions of the array using linalg.reduce.
    """

    def max_init_value(element_type):
        """Initialize with minimum value for the type"""
        if isinstance(element_type, types.Float):
            return float("-inf")
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return -(2 ** (element_type.bitwidth - 1))
            else:
                return 0
        else:
            raise ValueError(f"Unsupported type for np.max: {element_type}")

    def max_combiner(element_type, out_arg, in_arg):
        """Select maximum value"""
        if isinstance(element_type, types.Float):
            return arith.maximumf(out_arg, in_arg)
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return arith.maxsi(out_arg, in_arg)
            else:
                return arith.maxui(out_arg, in_arg)
        else:
            raise ValueError(f"Unsupported type for np.max: {element_type}")

    create_reduction_op(builder, target, args, kwargs, max_init_value, max_combiner, "np.max")


@lower_getattr(types.Array, "max")
def lower_array_max_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, np_max_cg))


@lower(np.prod, types.Array)
def np_prod_cg(builder, target, args, kwargs):
    """
    Product over all dimensions of the array using linalg.reduce.
    """

    def prod_init_value(element_type):
        """Initialize with 1 (multiplicative identity)"""
        return 1

    def prod_combiner(element_type, out_arg, in_arg):
        """Multiply values - no type coercion needed as operands are same type"""
        return lowering_utilities.mul(out_arg, in_arg)

    create_reduction_op(builder, target, args, kwargs, prod_init_value, prod_combiner, "np.prod")


@lower_getattr(types.Array, "prod")
def lower_array_prod_getattr(
    _: MLIRTargetContext,
    mlir_lower: MLIRLower,
    target: numba_ir.Var,
    array: numba_ir.Var,
):
    mlir_lower.store_var(target, DeferredMethodCall(array, np_prod_cg))


# ============================================================================
# NaN-aware Reduction Operations: nanmin, nanmax, nansum, nanprod, nanmean
# ============================================================================


@lower(np.nanmin, types.Array)
def np_nanmin_cg(builder, target, args, kwargs):
    """
    Find minimum over all dimensions, ignoring NaN values.
    Uses fminimum which propagates non-NaN values over NaN.
    """

    def nanmin_init_value(element_type):
        if isinstance(element_type, types.Float):
            return float("inf")
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return 2 ** (element_type.bitwidth - 1) - 1
            else:
                return 2**element_type.bitwidth - 1
        else:
            raise ValueError(f"Unsupported type for np.nanmin: {element_type}")

    def nanmin_combiner(element_type, out_arg, in_arg):
        if isinstance(element_type, types.Float):
            # arith.minnumf ignores NaN (propagates non-NaN)
            return arith.minnumf(out_arg, in_arg)
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return arith.minsi(out_arg, in_arg)
            else:
                return arith.minui(out_arg, in_arg)
        else:
            raise ValueError(f"Unsupported type for np.nanmin: {element_type}")

    create_reduction_op(
        builder, target, args, kwargs, nanmin_init_value, nanmin_combiner, "np.nanmin"
    )


@lower(np.nanmax, types.Array)
def np_nanmax_cg(builder, target, args, kwargs):
    """
    Find maximum over all dimensions, ignoring NaN values.
    Uses fmaximum which propagates non-NaN values over NaN.
    """

    def nanmax_init_value(element_type):
        if isinstance(element_type, types.Float):
            return float("-inf")
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return -(2 ** (element_type.bitwidth - 1))
            else:
                return 0
        else:
            raise ValueError(f"Unsupported type for np.nanmax: {element_type}")

    def nanmax_combiner(element_type, out_arg, in_arg):
        if isinstance(element_type, types.Float):
            # arith.maxnumf ignores NaN (propagates non-NaN)
            return arith.maxnumf(out_arg, in_arg)
        elif isinstance(element_type, types.Integer):
            if element_type.signed:
                return arith.maxsi(out_arg, in_arg)
            else:
                return arith.maxui(out_arg, in_arg)
        else:
            raise ValueError(f"Unsupported type for np.nanmax: {element_type}")

    create_reduction_op(
        builder, target, args, kwargs, nanmax_init_value, nanmax_combiner, "np.nanmax"
    )


@lower(np.nansum, types.Array)
def np_nansum_cg(builder, target, args, kwargs):
    """
    Sum over all dimensions, treating NaN as zero.
    """
    from numba_cuda_mlir.lowering_utilities import add

    def nansum_init_value(element_type):
        return 0

    def nansum_combiner(element_type, out_arg, in_arg):
        if isinstance(element_type, types.Float):
            # Check if in_arg is NaN, if so use 0 instead
            is_nan = arith.cmpf(arith.CmpFPredicate.UNO, in_arg, in_arg)
            zero = arith.constant(result=in_arg.type, value=0.0)
            safe_in = arith.select(is_nan, zero, in_arg)
            return add(out_arg, safe_in)
        else:
            return add(out_arg, in_arg)

    create_reduction_op(
        builder, target, args, kwargs, nansum_init_value, nansum_combiner, "np.nansum"
    )


@lower(np.nanprod, types.Array)
def np_nanprod_cg(builder, target, args, kwargs):
    """
    Product over all dimensions, treating NaN as one.
    """
    from numba_cuda_mlir.lowering_utilities import mul

    def nanprod_init_value(element_type):
        return 1

    def nanprod_combiner(element_type, out_arg, in_arg):
        if isinstance(element_type, types.Float):
            # Check if in_arg is NaN, if so use 1 instead
            is_nan = arith.cmpf(arith.CmpFPredicate.UNO, in_arg, in_arg)
            one = arith.constant(result=in_arg.type, value=1.0)
            safe_in = arith.select(is_nan, one, in_arg)
            return mul(out_arg, safe_in)
        else:
            return mul(out_arg, in_arg)

    create_reduction_op(
        builder,
        target,
        args,
        kwargs,
        nanprod_init_value,
        nanprod_combiner,
        "np.nanprod",
    )


@lower(np.nanmean, types.Array)
def np_nanmean_cg(builder, target, args, kwargs):
    """
    Mean over all dimensions, ignoring NaN values.
    Computes sum of non-NaN values divided by count of non-NaN values.
    """
    from numba_cuda_mlir.lowering_utilities import add
    from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type

    arr = builder.load_var(args[0])
    arr_type = builder.get_numba_type(args[0])
    element_type = arr_type.dtype
    mlir_element_type = builder.get_value_type(element_type)

    arr_tensor = lowering_utilities.memref_to_value_tensor(arr_type, arr)
    ndim = arr_type.ndim
    dims_attr = ir.DenseI64ArrayAttr.get(list(range(ndim)))
    loc = ir.Location.unknown()

    # Compute nansum using manual ReduceOp construction
    sum_init_const = arith.constant(
        result=mlir_element_type,
        value=0.0 if isinstance(element_type, types.Float) else 0,
    )
    sum_result_type = ir.RankedTensorType.get((), mlir_element_type)
    sum_init = tensor.splat(sum_result_type, sum_init_const, [])

    sum_reduce_op = linalg.ReduceOp(
        result=[sum_result_type],
        inputs=[arr_tensor],
        inits=[sum_init],
        dimensions=dims_attr,
    )
    sum_region = sum_reduce_op.combiner
    sum_block = sum_region.blocks.append()
    sum_in_arg = sum_block.add_argument(mlir_element_type, loc=loc)
    sum_out_arg = sum_block.add_argument(mlir_element_type, loc=loc)
    with ir.InsertionPoint(sum_block):
        if isinstance(element_type, types.Float):
            is_nan = arith.cmpf(arith.CmpFPredicate.UNO, sum_in_arg, sum_in_arg)
            zero = arith.constant(result=mlir_element_type, value=0.0)
            safe_val = arith.select(is_nan, zero, sum_in_arg)
            new_sum = add(sum_out_arg, safe_val)
        else:
            new_sum = add(sum_out_arg, sum_in_arg)
        linalg.yield_([new_sum])

    sum_result_tensor = next(iter(cast(Any, sum_reduce_op.operation.results)))
    total_sum = tensor.extract(sum_result_tensor, [])

    # Compute count of non-NaN values
    count_init_const = arith.constant(result=T.f64(), value=0.0)
    count_result_type = ir.RankedTensorType.get((), T.f64())
    count_init = tensor.splat(count_result_type, count_init_const, [])

    count_reduce_op = linalg.ReduceOp(
        result=[count_result_type],
        inputs=[arr_tensor],
        inits=[count_init],
        dimensions=dims_attr,
    )
    count_region = count_reduce_op.combiner
    count_block = count_region.blocks.append()
    count_in_arg = count_block.add_argument(mlir_element_type, loc=loc)
    count_out_arg = count_block.add_argument(T.f64(), loc=loc)
    with ir.InsertionPoint(count_block):
        one = arith.constant(result=T.f64(), value=1.0)
        zero_count = arith.constant(result=T.f64(), value=0.0)
        if isinstance(element_type, types.Float):
            is_nan = arith.cmpf(arith.CmpFPredicate.UNO, count_in_arg, count_in_arg)
            incr = arith.select(is_nan, zero_count, one)
        else:
            incr = one
        new_count = arith.addf(count_out_arg, incr)
        linalg.yield_([new_count])

    count_result_tensor = next(iter(cast(Any, count_reduce_op.operation.results)))
    total_count = tensor.extract(count_result_tensor, [])

    # Compute mean = sum / count
    if isinstance(element_type, types.Float):
        sum_f64 = convert(total_sum, T.f64())
        mean_f64 = arith.divf(sum_f64, total_count)
        mean = convert(mean_f64, mlir_element_type)
    else:
        sum_f64 = arith.sitofp(T.f64(), total_sum)
        mean_f64 = arith.divf(sum_f64, total_count)
        mean = arith.fptosi(mlir_element_type, mean_f64)

    builder.store_var(target, mean)


# ============================================================================
# Element-wise Operations: abs, sqrt, exp, log
# ============================================================================
# NOTE: These element-wise operations use linalg.GenericOp with proper dynamic
# shape handling. When NumPy ufuncs like np.abs() are called on arrays, they
# are routed through the arrayexpr code path (mlir_lowering.py:lower_arrayexpr_assign)
# which looks them up in numba_cuda_mlir.ufunc_db and delegates to these lowering functions.


def create_elementwise_op(builder, target, args, kwargs, math_fn, op_name):
    """
    Public helper for implementing element-wise operations using linalg.GenericOp.

    This function provides a reusable pattern for implementing NumPy element-wise
    operations that apply a mathematical function to each element of an array.
    Use this to add new element-wise operations following the established pattern.

    Args:
        builder: MLIR builder instance
        target: Target variable for the result
        args: Function arguments (should be single array)
        kwargs: Keyword arguments (should be empty)
        math_fn: Function(input_element_type, target_element_type, input_mlir_type, target_mlir_type, in_elem) -> result
        op_name: Name of the operation for error messages

    Examples:
        See np_abs_cg, np_sqrt_cg, np_exp_cg, np_log_cg for complete usage examples.
    """
    assert len(args) == 1 and len(kwargs) == 0, f"{op_name} takes exactly one argument"
    arg = args[0]

    # Extract types from input array and target
    input_array_type = builder.get_numba_type(arg.name)
    input_element_type = input_array_type.dtype
    input_mlir_type = builder.get_value_type(input_element_type)

    target_array_type = builder.get_numba_type(target.name)
    target_element_type = target_array_type.dtype
    target_mlir_type = builder.get_value_type(target_element_type)

    array_memref = builder.load_var(arg)
    array = lowering_utilities.memref_to_value_tensor(input_array_type, array_memref)
    input_tensor_type = array.type

    # Extract dynamic dimensions for tensor.empty (use target type for output)
    rank = input_tensor_type.rank
    dynamic_sizes = []
    for i in range(rank):
        if input_tensor_type.shape[i] == ir.ShapedType.get_dynamic_size():
            # Extract dynamic dimension from the original memref
            dim_size = memref.dim(
                source=array_memref, index=arith.constant(result=T.index(), value=i)
            )
            dynamic_sizes.append(dim_size)

    # Create empty output tensor with target element type
    init = tensor.EmptyOp(dynamic_sizes, element_type=target_mlir_type).result

    # Use linalg.GenericOp for element-wise operation
    affine_map = ir.AffineMap.get_identity(rank)
    indexing_maps_attr = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(affine_map),  # input
            ir.AffineMapAttr.get(affine_map),  # output
        ]
    )
    iterator_types_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )

    generic_op = linalg.GenericOp(
        [init.type],  # result_tensors
        [array],  # inputs
        [init],  # outputs
        indexing_maps_attr,
        iterator_types_attr,
    )
    region = generic_op.regions[0]
    block = region.blocks.append(input_mlir_type, target_mlir_type)
    in_elem = block.arguments[0]
    with ir.InsertionPoint(block):
        # Apply the mathematical operation (may include type conversion)
        result = math_fn(
            input_element_type,
            target_element_type,
            input_mlir_type,
            target_mlir_type,
            in_elem,
        )
        linalg.yield_([result])

    # Get the result and convert back to memref
    map_result = next(iter(cast(Any, generic_op.operation.results)))
    result_memref = lowering_utilities.value_tensor_to_storage_memref(target_array_type, map_result)
    builder.store_var(target, result_memref)


@ufunc_registry.register(np.abs)
@ufunc_registry.register(np.absolute)
@lower(np.abs, types.Array)
def np_abs_cg(builder, target, args, kwargs):
    """
    Element-wise absolute value using linalg.GenericOp.
    """

    def abs_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        """Select absf for floats, absi for integers, with type conversion if needed"""
        # Apply abs operation on input type
        if isinstance(input_element_type, types.Float):
            result = math_dialect.absf(in_elem)
        else:
            result = math_dialect.absi(in_elem)

        # Convert to target type if needed
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, abs_fn, "np.abs")


@ufunc_registry.register(np.sqrt)
@lower(np.sqrt, types.Array)
def np_sqrt_cg(builder, target, args, kwargs):
    """
    Element-wise square root using linalg.GenericOp.
    """

    def sqrt_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        """Compute square root with type conversion if needed"""
        result = math_dialect.sqrt(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, sqrt_fn, "np.sqrt")


@ufunc_registry.register(np.exp)
@lower(np.exp, types.Array)
def np_exp_cg(builder, target, args, kwargs):
    """
    Element-wise exponential using linalg.GenericOp.
    """

    def exp_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        """Compute exponential with type conversion if needed"""
        result = math_dialect.exp(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, exp_fn, "np.exp")


@ufunc_registry.register(np.log)
@lower(np.log, types.Array)
def np_log_cg(builder, target, args, kwargs):
    """
    Element-wise natural logarithm using linalg.GenericOp.
    """

    def log_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        """Compute natural logarithm with type conversion if needed"""
        result = math_dialect.log(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, log_fn, "np.log")


# Angle conversion constants
_RAD_TO_DEG = 180.0 / np.pi
_DEG_TO_RAD = np.pi / 180.0


@lower(np.rad2deg, types.Float)
@lower(np.degrees, types.Float)
def np_rad2deg_scalar_cg(builder, target, args, kwargs):
    """Scalar radians to degrees: x * (180 / pi)"""
    assert len(args) == 1 and len(kwargs) == 0
    in_val = builder.load_var(args[0])
    target_mlir_type = builder.get_mlir_type(target)
    in_val = convert(in_val, target_mlir_type)
    factor = float_of(_RAD_TO_DEG, target_mlir_type)
    result = arith.mulf(in_val, factor)
    builder.store_var(target, result)


@lower(np.rad2deg, types.Float, types.Array)
@lower(np.degrees, types.Float, types.Array)
def np_rad2deg_scalar_to_array_cg(builder, target, args, kwargs):
    """rad2deg(scalar, out_array) - broadcast scalar to output array"""
    assert len(args) == 2 and len(kwargs) == 0
    in_val = builder.load_var(args[0])
    out_arr = builder.load_var(args[1])
    output_array_type = builder.get_numba_type(args[1].name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    in_val = convert(in_val, elem_type)
    factor = float_of(_RAD_TO_DEG, elem_type)
    result_val = arith.mulf(in_val, factor)
    # Fill all elements of output array with the result
    _store_first_output_value(builder, args[1], out_arr, result_val)
    builder.store_var(target, out_arr)


@lower(np.deg2rad, types.Float)
@lower(np.radians, types.Float)
def np_deg2rad_scalar_cg(builder, target, args, kwargs):
    """Scalar degrees to radians: x * (pi / 180)"""
    assert len(args) == 1 and len(kwargs) == 0
    in_val = builder.load_var(args[0])
    target_mlir_type = builder.get_mlir_type(target)
    in_val = convert(in_val, target_mlir_type)
    factor = float_of(_DEG_TO_RAD, target_mlir_type)
    result = arith.mulf(in_val, factor)
    builder.store_var(target, result)


@lower(np.deg2rad, types.Float, types.Array)
@lower(np.radians, types.Float, types.Array)
def np_deg2rad_scalar_to_array_cg(builder, target, args, kwargs):
    """deg2rad(scalar, out_array) - broadcast scalar to output array"""
    assert len(args) == 2 and len(kwargs) == 0
    in_val = builder.load_var(args[0])
    out_arr = builder.load_var(args[1])
    output_array_type = builder.get_numba_type(args[1].name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    in_val = convert(in_val, elem_type)
    factor = float_of(_DEG_TO_RAD, elem_type)
    result_val = arith.mulf(in_val, factor)
    _store_first_output_value(builder, args[1], out_arr, result_val)
    builder.store_var(target, out_arr)


@ufunc_registry.register(np.rad2deg)
@ufunc_registry.register(np.degrees)
@lower(np.rad2deg, types.Array)
@lower(np.degrees, types.Array)
def np_rad2deg_cg(builder, target, args, kwargs):
    """Element-wise radians to degrees: x * (180 / pi)"""

    def rad2deg_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        factor = float_of(_RAD_TO_DEG, target_mlir_type)
        result = arith.mulf(in_elem, factor)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, rad2deg_fn, "np.rad2deg")


@ufunc_registry.register(np.deg2rad)
@ufunc_registry.register(np.radians)
@lower(np.deg2rad, types.Array)
@lower(np.radians, types.Array)
def np_deg2rad_cg(builder, target, args, kwargs):
    """Element-wise degrees to radians: x * (pi / 180)"""

    def deg2rad_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        factor = float_of(_DEG_TO_RAD, target_mlir_type)
        result = arith.mulf(in_elem, factor)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, deg2rad_fn, "np.deg2rad")


def _deg2rad_elem(in_elem):
    """Degrees to radians element-wise"""
    factor = float_of(_DEG_TO_RAD, in_elem.type)
    return arith.mulf(in_elem, factor)


def _rad2deg_elem(in_elem):
    """Radians to degrees element-wise"""
    factor = float_of(_RAD_TO_DEG, in_elem.type)
    return arith.mulf(in_elem, factor)


@lower(np.deg2rad, types.Array, types.Array)
@lower(np.radians, types.Array, types.Array)
def np_deg2rad_array_to_array_cg(builder, target, args, kwargs):
    """deg2rad(array, out_array) - element-wise degrees to radians"""
    create_elementwise_op_with_output(builder, target, args, _deg2rad_elem)


@lower(np.rad2deg, types.Array, types.Array)
@lower(np.degrees, types.Array, types.Array)
def np_rad2deg_array_to_array_cg(builder, target, args, kwargs):
    """rad2deg(array, out_array) - element-wise radians to degrees"""
    create_elementwise_op_with_output(builder, target, args, _rad2deg_elem)


@lower(operator.truediv, types.Array, types.Array)
@lower(operator.truediv, types.Array, types.Number)
@lower(operator.truediv, types.Number, types.Array)
def operator_truediv_array_lower(builder, target, args, kwargs):
    """Lower operator.truediv for arrays by using linalg.div"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.div)


@lower(operator.itruediv, types.Array, types.Array)
@lower(operator.itruediv, types.Array, types.Number)
@lower(operator.itruediv, types.Number, types.Array)
def operator_itruediv_array_lower(builder, target, args, kwargs):
    """Lower operator.itruediv for arrays by using linalg.div"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.div)


@lower(operator.neg, types.Array)
def operator_neg_array_lower(builder, target, args, kwargs):
    """Lower operator.neg for arrays by using linalg.sub(0.0, array)"""
    assert len(args) == 1, "operator.neg expects 1 argument"

    # Get the array argument
    array_arg = args[0]
    array_type = builder.get_numba_type(array_arg.name)
    target_type = builder.get_numba_type(target.name)

    # Create a scalar 0.0 value
    from numba_cuda_mlir.mlir.dialect_exts import arith

    element_type = builder.get_value_type(array_type.dtype)
    zero_scalar = arith.constant(
        result=element_type, value=_zero_literal_for_numba_type(array_type.dtype)
    )

    # Store the zero scalar in a temporary variable for lower_np_binop
    import numba_cuda_mlir.numba_cuda.core.ir as numba_ir

    zero_var = numba_ir.Var(
        scope=array_arg.scope, name=f"$const_zero_{array_arg.name}", loc=array_arg.loc
    )
    builder.store_var(zero_var, zero_scalar)
    builder.typemap[zero_var.name] = array_type.dtype

    # Call lower_np_binop with 0.0 - array
    lower_np_binop(builder, target, target_type, [zero_var, array_arg], linalg.sub)


@lower(abs, types.Array)
def abs_array_lower(builder, target, args, kwargs):
    """Lower abs() for arrays"""
    np_abs_cg(builder, target, args, kwargs)


@lower(operator.matmul, types.Array, types.Array)
def operator_matmul_array_lower(builder, target, args, kwargs):
    """Lower operator.matmul (@) for arrays"""
    target_type = builder.get_numba_type(target.name)
    lower_matmul(builder, target, target_type, args)


def _allocate_array(builder, target, args):
    """Helper to allocate array memory without storing to variable.

    Returns the allocated memref value.
    """
    assert len(args) in [1, 2], "allocation expects 1 or 2 arguments"

    shape_arg = args[0]

    shape_type = builder.get_numba_type(shape_arg.name)
    target_type = builder.get_numba_type(target.name)

    # Determine ndim from shape
    if isinstance(shape_type, types.Integer):
        ndim = 1
        shape_vals = [builder.load_var(shape_arg)]
    elif isinstance(shape_type, (types.UniTuple, types.Tuple)):
        ndim = shape_type.count if isinstance(shape_type, types.UniTuple) else len(shape_type.types)
        # Extract tuple elements
        shape_loaded = builder.load_var(shape_arg)

        assert isinstance(shape_loaded, (list, tuple)), (
            f"Expected Python tuple for shape, got {type(shape_loaded)}"
        )
        shape_vals = list(shape_loaded)
    else:
        raise NotImplementedError(f"Unsupported shape type: {shape_type}")

    # Convert shape values to index type
    shape_vals = [builder.mlir_convert(s, T.index()) for s in shape_vals]

    # Get the storage element type from target
    element_type = builder.get_storage_type(target_type.dtype)

    # Create a simple contiguous memref type for allocation (no strided layout)
    # This produces a row-major/C-contiguous array
    dyn = ir.MemRefType.get_dynamic_size()
    alloca_memref_type = ir.MemRefType.get([dyn] * ndim, element_type)

    # For dynamic allocations, create alloca at current point (not entry block)
    # because the size values may not be defined at entry block yet
    # Static allocations could use alloca_insertion_point() but for simplicity
    # we always create at current point since MLIR handles this fine
    alloca_op = memref_dialect.AllocaOp(
        memref=alloca_memref_type,
        dynamicSizes=shape_vals,
        symbolOperands=[],
    )

    return alloca_op.memref


@lower(np.empty, types.UniTuple, types.DTypeSpec)
@lower(np.empty, types.Tuple, types.DTypeSpec)
@lower(np.empty, types.Integer, types.DTypeSpec)
@lower(np.empty, types.UniTuple)
@lower(np.empty, types.Tuple)
@lower(np.empty, types.Integer)
def np_empty_lower(builder, target, args, kwargs):
    """Lower np.empty to memref.alloc"""
    result_memref = _allocate_array(builder, target, args)
    builder.store_var(target, result_memref)


@lower(np.zeros, types.UniTuple, types.DTypeSpec)
@lower(np.zeros, types.Tuple, types.DTypeSpec)
@lower(np.zeros, types.Integer, types.DTypeSpec)
def np_zeros_lower(builder, target, args, kwargs):
    """Lower np.zeros to memref.alloc + linalg.fill"""
    # Allocate the array
    array_memref = _allocate_array(builder, target, args)

    # Fill with zeros
    target_type = builder.get_numba_type(target.name)
    zero = _bool_storage_literal(builder, target_type, 0)
    if zero is None:
        element_type = builder.get_value_type(target_type.dtype)
        zero = arith.constant(
            result=element_type, value=_zero_literal_for_numba_type(target_type.dtype)
        )
        zero = builder.as_storage(target_type.dtype, zero)

    array_tensor = memref_to_tensor(array_memref)
    filled_tensor = linalg.fill(zero, outs=[array_tensor])
    result_memref = tensor_to_memref(filled_tensor)

    builder.store_var(target, result_memref)


@lower(np.ones, types.UniTuple, types.DTypeSpec)
@lower(np.ones, types.Tuple, types.DTypeSpec)
@lower(np.ones, types.Integer, types.DTypeSpec)
def np_ones_lower(builder, target, args, kwargs):
    """Lower np.ones to memref.alloc + linalg.fill"""
    # Allocate the array
    array_memref = _allocate_array(builder, target, args)

    # Fill with ones
    target_type = builder.get_numba_type(target.name)
    one = _bool_storage_literal(builder, target_type, 1)
    if one is None:
        element_type = builder.get_value_type(target_type.dtype)
        one = arith.constant(
            result=element_type, value=_one_literal_for_numba_type(target_type.dtype)
        )
        one = builder.as_storage(target_type.dtype, one)

    array_tensor = memref_to_tensor(array_memref)
    filled_tensor = linalg.fill(one, outs=[array_tensor])
    result_memref = tensor_to_memref(filled_tensor)

    builder.store_var(target, result_memref)


@lower(np.full, types.UniTuple, types.Number, types.DTypeSpec)
@lower(np.full, types.Tuple, types.Number, types.DTypeSpec)
@lower(np.full, types.Integer, types.Number, types.DTypeSpec)
def np_full_lower(builder, target, args, kwargs):
    """Lower np.full to memref.alloc + linalg.fill"""
    assert len(args) in [2, 3], "np.full expects 2 or 3 arguments"

    # Allocate the array (use args[0] for shape and args[2] for dtype if present)
    shape_dtype_args = [args[0]]
    if len(args) > 2:
        shape_dtype_args.append(args[2])
    array_memref = _allocate_array(builder, target, shape_dtype_args)

    # Get the fill value
    target_type = builder.get_numba_type(target.name)
    value_arg = args[1]
    fill_value = builder.load_var(value_arg)

    # Convert fill value to target element type if needed
    element_type = builder.get_value_type(target_type.dtype)
    fill_value = builder.mlir_convert(fill_value, element_type)
    fill_value = builder.as_storage(target_type.dtype, fill_value)

    # Fill the array
    array_tensor = memref_to_tensor(array_memref)
    filled_tensor = linalg.fill(fill_value, outs=[array_tensor])
    result_memref = tensor_to_memref(filled_tensor)

    builder.store_var(target, result_memref)


@lower(np.add, types.Number, types.Number)
def np_add_scalar_lower(builder, target, args, kwargs):
    """Lower np.add for scalars"""
    assert len(args) == 2, "np.add expects 2 arguments"
    lhs = builder.load_var(args[0])
    rhs = builder.load_var(args[1])
    result = (
        arith.addf(lhs, rhs)
        if isinstance(builder.get_numba_type(target.name), types.Float)
        else arith.addi(lhs, rhs)
    )
    builder.store_var(target, result)


@lower(np.add, types.Array, types.Array)
@lower(np.add, types.Array, types.Number)
@lower(np.add, types.Number, types.Array)
def np_add_array_lower(builder, target, args, kwargs):
    """Lower np.add for arrays"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.add)


@lower(np.subtract, types.Number, types.Number)
def np_subtract_scalar_lower(builder, target, args, kwargs):
    """Lower np.subtract for scalars"""
    assert len(args) == 2, "np.subtract expects 2 arguments"
    lhs = builder.load_var(args[0])
    rhs = builder.load_var(args[1])
    result = (
        arith.subf(lhs, rhs)
        if isinstance(builder.get_numba_type(target.name), types.Float)
        else arith.subi(lhs, rhs)
    )
    builder.store_var(target, result)


@lower(np.subtract, types.Array, types.Array)
@lower(np.subtract, types.Array, types.Number)
@lower(np.subtract, types.Number, types.Array)
def np_subtract_array_lower(builder, target, args, kwargs):
    """Lower np.subtract for arrays"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.sub)


@lower(np.multiply, types.Number, types.Number)
def np_multiply_scalar_lower(builder, target, args, kwargs):
    """Lower np.multiply for scalars"""
    assert len(args) == 2, "np.multiply expects 2 arguments"
    lhs = builder.load_var(args[0])
    rhs = builder.load_var(args[1])
    result = (
        arith.mulf(lhs, rhs)
        if isinstance(builder.get_numba_type(target.name), types.Float)
        else arith.muli(lhs, rhs)
    )
    builder.store_var(target, result)


@lower(np.multiply, types.Array, types.Array)
@lower(np.multiply, types.Array, types.Number)
@lower(np.multiply, types.Number, types.Array)
def np_multiply_array_lower(builder, target, args, kwargs):
    """Lower np.multiply for arrays"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.mul)


@lower(np.divide, types.Number, types.Number)
def np_divide_scalar_lower(builder, target, args, kwargs):
    """Lower np.divide for scalars."""

    lhs = builder.load_var(args[0])
    rhs = builder.load_var(args[1])
    target_type = builder.get_numba_type(target.name)

    if isinstance(target_type, types.Float):
        # Convert operands to the target type so mixed-type calls like
        # ``np.divide(float64, int64)`` provide float operands as required by
        # ``arith.divf``.
        target_mlir_type = builder.get_mlir_type(target_type)
        lhs = convert(lhs, target_mlir_type)
        rhs = convert(rhs, target_mlir_type)
        result = arith.divf(lhs, rhs)
    else:
        result = arith.divsi(lhs, rhs)

    builder.store_var(target, result)


@lower(np.divide, types.Array, types.Array)
@lower(np.divide, types.Array, types.Number)
@lower(np.divide, types.Number, types.Array)
def np_divide_array_lower(builder, target, args, kwargs):
    """Lower np.divide for arrays"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.div)


@lower(np.negative, types.Array)
def np_negative_array_lower(builder, target, args, kwargs):
    """Lower np.negative for arrays"""
    # Same as operator.neg implementation
    operator_neg_array_lower(builder, target, args, kwargs)


@lower(np.absolute, types.Number)
def np_absolute_scalar_lower(builder, target, args, kwargs):
    """Lower np.absolute for scalars"""
    assert len(args) == 1, "np.absolute expects 1 argument"
    value = builder.load_var(args[0])
    result = (
        math_dialect.absf(value)
        if isinstance(builder.get_numba_type(args[0].name), types.Float)
        else math_dialect.absi(value)
    )
    builder.store_var(target, result)


@lower(np.absolute, types.Array)
def np_absolute_array_lower(builder, target, args, kwargs):
    """Lower np.absolute for arrays"""
    # Use the existing np.abs lowering
    np_abs_cg(builder, target, args, kwargs)


@lower(np.ceil, types.Number)
def np_ceil_scalar_lower(builder, target, args, kwargs):
    """Lower np.ceil for scalars"""
    assert len(args) == 1, "np.ceil expects 1 argument"
    value = builder.load_var(args[0])
    result = math_dialect.ceil(value)
    builder.store_var(target, result)


@lower(np.ceil, types.Array)
def np_ceil_array_lower(builder, target, args, kwargs):
    """Lower np.ceil for arrays"""

    def ceil_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        result = math_dialect.ceil(in_elem)
        return convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, ceil_fn, "np.ceil")


@lower(np.isnan, types.Number)
def np_isnan_scalar_lower(builder, target, args, kwargs):
    """Lower np.isnan for scalars.

    Integers are never NaN; floats use ``math.isnan``. ``arith.cmpf`` with the
    UNO predicate would also work (``NaN`` is the only value where ``v != v``),
    but using the math dialect mirrors the existing ``math.isnan`` lowering.
    """
    assert len(args) == 1, "np.isnan expects 1 argument"
    assert not kwargs, "np.isnan does not accept keyword arguments"
    value = builder.load_var(args[0])
    arg_ty = builder.get_numba_type(args[0].name)
    if isinstance(arg_ty, (types.Integer, types.Boolean)):
        result = arith.constant(T.bool(), False)
    else:
        result = math_dialect.isnan(value)
    builder.store_var(target, result)


@lower(np.isnan, types.Array)
def np_isnan_array_lower(builder, target, args, kwargs):
    """Lower np.isnan element-wise over an array."""

    def isnan_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, (types.Integer, types.Boolean)):
            return arith.constant(T.bool(), False)
        return math_dialect.isnan(in_elem)

    create_elementwise_op(builder, target, args, kwargs, isnan_fn, "np.isnan")


@lower(np.floor, types.Number)
def np_floor_scalar_lower(builder, target, args, kwargs):
    """Lower np.floor for scalars"""
    assert len(args) == 1, "np.floor expects 1 argument"
    value = builder.load_var(args[0])
    result = math_dialect.floor(value)
    builder.store_var(target, result)


@lower(np.floor, types.Array)
def np_floor_array_lower(builder, target, args, kwargs):
    """Lower np.floor for arrays"""

    def floor_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        result = math_dialect.floor(in_elem)
        return convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, floor_fn, "np.floor")


@lower(np.log, types.Number)
def np_log_scalar_lower(builder, target, args, kwargs):
    """Lower np.log for scalars"""
    assert len(args) == 1, "np.log expects 1 argument"
    value = builder.load_var(args[0])
    result = math_dialect.log(value)
    builder.store_var(target, result)


@lower(np.exp, types.Number)
def np_exp_scalar_lower(builder, target, args, kwargs):
    """Lower np.exp for scalars"""
    assert len(args) == 1, "np.exp expects 1 argument"
    value = builder.load_var(args[0])
    result = math_dialect.exp(value)
    builder.store_var(target, result)


@lower(np.matmul, types.Array, types.Array)
def np_matmul_lower(builder, target, args, kwargs):
    """Lower np.matmul for 2D arrays"""
    target_type = builder.get_numba_type(target.name)
    lower_matmul(builder, target, target_type, args)


@lower(np.dot, types.Number, types.Number)
def np_dot_scalar_lower(builder, target, args, kwargs):
    """Lower np.dot for scalars (multiplication)"""
    assert len(args) == 2, "np.dot expects 2 arguments"
    lhs = builder.load_var(args[0])
    rhs = builder.load_var(args[1])
    result = (
        arith.mulf(lhs, rhs)
        if isinstance(builder.get_numba_type(target.name), types.Float)
        else arith.muli(lhs, rhs)
    )
    builder.store_var(target, result)


@lower(np.dot, types.Number, types.Array)
@lower(np.dot, types.Array, types.Number)
def np_dot_scalar_array_lower(builder, target, args, kwargs):
    """Lower np.dot for scalar * array"""
    target_type = builder.get_numba_type(target.name)
    lower_np_binop(builder, target, target_type, args, linalg.mul)


@lower(np.dot, types.Array, types.Array)
def np_dot_array_lower(builder, target, args, kwargs):
    """Lower np.dot for arrays"""
    lhs_type = builder.get_numba_type(args[0].name)
    rhs_type = builder.get_numba_type(args[1].name)
    target_type = builder.get_numba_type(target.name)

    # If both are 2D, use matmul
    if lhs_type.ndim == 2 and rhs_type.ndim == 2:
        lower_matmul(builder, target, target_type, args)
    else:
        # Use the general dot lowering
        lower_linalg_dot(builder, target, target_type, args)


@lower(np.transpose, types.Array)
def np_transpose_lower(builder, target, args, kwargs):
    """Lower np.transpose for arrays"""
    assert len(args) == 1, "np.transpose expects 1 argument"
    lower_transpose(builder, target, args[0])


def create_elementwise_op_with_output(builder, target, args, math_fn):
    """
    Helper for element-wise operations that write to an output array.

    Handles signatures like: np.arccos(input_array, output_array)
    where both input and output are arrays of the same shape.
    """
    assert len(args) == 2
    input_arg, output_arg = args

    input_array_type = builder.get_numba_type(input_arg.name)
    output_array_type = builder.get_numba_type(output_arg.name)
    input_mlir_elem_type = builder.get_value_type(input_array_type.dtype)
    output_mlir_elem_type = builder.get_value_type(output_array_type.dtype)

    input_memref = builder.load_var(input_arg)
    output_memref = builder.load_var(output_arg)

    input_tensor = lowering_utilities.memref_to_value_tensor(input_array_type, input_memref)
    output_tensor = memref_to_tensor(output_memref)
    output_storage_elem_type = output_tensor.type.element_type

    rank = input_tensor.type.rank
    affine_map = ir.AffineMap.get_identity(rank)

    indexing_maps_attr = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(affine_map),  # input
            ir.AffineMapAttr.get(affine_map),  # output
        ]
    )
    iterator_types_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )

    generic_op = linalg.GenericOp(
        [output_tensor.type],  # result_tensors
        [input_tensor],  # inputs
        [output_tensor],  # outputs
        indexing_maps_attr,
        iterator_types_attr,
    )
    region = generic_op.regions[0]
    block = region.blocks.append(input_mlir_elem_type, output_storage_elem_type)
    in_elem = block.arguments[0]
    with ir.InsertionPoint(block):
        in_elem = convert(in_elem, output_mlir_elem_type)
        result = math_fn(in_elem)
        result = convert(result, output_mlir_elem_type)
        result = lowering_utilities.value_to_storage(output_array_type.dtype, result)
        linalg.yield_([result])

    map_result = next(iter(cast(Any, generic_op.operation.results)))
    result_memref = tensor_to_memref(map_result)
    # Copy result back to the original output memref (in-place semantics)
    memref.copy(result_memref, output_memref)
    builder.store_var(target, output_memref)


def create_scalar_to_output_array(builder, target, args, math_fn, op_name):
    """
    Helper for element-wise operations with scalar input and output array.

    Handles signatures like: np.arccos(scalar, output_array)
    The scalar result is stored in output_array[0].
    """
    assert len(args) == 2, f"{op_name} expects 2 arguments (scalar, output)"
    scalar_arg, output_arg = args

    in_val = builder.load_var(scalar_arg)
    out_arr = builder.load_var(output_arg)
    output_array_type = builder.get_numba_type(output_arg.name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    in_val = convert(in_val, elem_type)
    result_val = math_fn(in_val)
    result_val = convert(result_val, elem_type)
    _store_first_output_value(builder, output_arg, out_arr, result_val)
    builder.store_var(target, out_arr)


@lower(np.arccos, types.Float)
def np_arccos_scalar_cg(builder, target, args, kwargs):
    """Scalar arccos using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.acos(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arccos, types.Float, types.Array)
def np_arccos_scalar_to_array_cg(builder, target, args, kwargs):
    """arccos(scalar, out_array) - compute arccos and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.acos, "np.arccos")


@ufunc_registry.register(np.arccos)
@lower(np.arccos, types.Array)
def np_arccos_array_cg(builder, target, args, kwargs):
    """Element-wise arccos using linalg.GenericOp."""

    def arccos_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_acos(in_elem)
        else:
            result = math_dialect.acos(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, arccos_fn, "np.arccos")


@lower(np.arccos, types.Array, types.Array)
def np_arccos_array_to_array_cg(builder, target, args, kwargs):
    """arccos(array, out_array) - element-wise arccos to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_acos)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.acos)


@lower(np.arcsin, types.Float)
def np_arcsin_scalar_cg(builder, target, args, kwargs):
    """Scalar arcsin using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.asin(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arcsin, types.Float, types.Array)
def np_arcsin_scalar_to_array_cg(builder, target, args, kwargs):
    """arcsin(scalar, out_array) - compute arcsin and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.asin, "np.arcsin")


@ufunc_registry.register(np.arcsin)
@lower(np.arcsin, types.Array)
def np_arcsin_array_cg(builder, target, args, kwargs):
    """Element-wise arcsin using linalg.GenericOp."""

    def arcsin_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_asin(in_elem)
        else:
            result = math_dialect.asin(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, arcsin_fn, "np.arcsin")


@lower(np.arcsin, types.Array, types.Array)
def np_arcsin_array_to_array_cg(builder, target, args, kwargs):
    """arcsin(array, out_array) - element-wise arcsin to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_asin)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.asin)


@lower(np.arctan, types.Float)
def np_arctan_scalar_cg(builder, target, args, kwargs):
    """Scalar arctan using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.atan(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arctan, types.Float, types.Array)
def np_arctan_scalar_to_array_cg(builder, target, args, kwargs):
    """arctan(scalar, out_array) - compute arctan and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.atan, "np.arctan")


@ufunc_registry.register(np.arctan)
@lower(np.arctan, types.Array)
def np_arctan_array_cg(builder, target, args, kwargs):
    """Element-wise arctan using linalg.GenericOp."""

    def arctan_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_atan(in_elem)
        else:
            result = math_dialect.atan(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, arctan_fn, "np.arctan")


@lower(np.arctan, types.Array, types.Array)
def np_arctan_array_to_array_cg(builder, target, args, kwargs):
    """arctan(array, out_array) - element-wise arctan to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_atan)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.atan)


@lower(np.sin, types.Float)
def np_sin_scalar_cg(builder, target, args, kwargs):
    """Scalar sin using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.sin(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.sin, types.Float, types.Array)
def np_sin_scalar_to_array_cg(builder, target, args, kwargs):
    """sin(scalar, out_array) - compute sin and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.sin, "np.sin")


@ufunc_registry.register(np.sin)
@lower(np.sin, types.Array)
def np_sin_array_cg(builder, target, args, kwargs):
    """Element-wise sin using linalg.GenericOp."""

    def sin_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = complex_dialect.sin(in_elem)
        else:
            result = math_dialect.sin(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, sin_fn, "np.sin")


@lower(np.sin, types.Array, types.Array)
def np_sin_array_to_array_cg(builder, target, args, kwargs):
    """sin(array, out_array) - element-wise sin to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.sin)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.sin)


@lower(np.cos, types.Float)
def np_cos_scalar_cg(builder, target, args, kwargs):
    """Scalar cos using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.cos(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.cos, types.Float, types.Array)
def np_cos_scalar_to_array_cg(builder, target, args, kwargs):
    """cos(scalar, out_array) - compute cos and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.cos, "np.cos")


@ufunc_registry.register(np.cos)
@lower(np.cos, types.Array)
def np_cos_array_cg(builder, target, args, kwargs):
    """Element-wise cos using linalg.GenericOp."""

    def cos_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = complex_dialect.cos(in_elem)
        else:
            result = math_dialect.cos(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, cos_fn, "np.cos")


@lower(np.cos, types.Array, types.Array)
def np_cos_array_to_array_cg(builder, target, args, kwargs):
    """cos(array, out_array) - element-wise cos to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.cos)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.cos)


@lower(np.tan, types.Float)
def np_tan_scalar_cg(builder, target, args, kwargs):
    """Scalar tan using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.tan(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.tan, types.Float, types.Array)
def np_tan_scalar_to_array_cg(builder, target, args, kwargs):
    """tan(scalar, out_array) - compute tan and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.tan, "np.tan")


@ufunc_registry.register(np.tan)
@lower(np.tan, types.Array)
def np_tan_array_cg(builder, target, args, kwargs):
    """Element-wise tan using linalg.GenericOp."""

    def tan_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = complex_dialect.tan(in_elem)
        else:
            result = math_dialect.tan(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, tan_fn, "np.tan")


@lower(np.tan, types.Array, types.Array)
def np_tan_array_to_array_cg(builder, target, args, kwargs):
    """tan(array, out_array) - element-wise tan to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.tan)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.tan)


@lower(np.sinh, types.Float)
def np_sinh_scalar_cg(builder, target, args, kwargs):
    """Scalar sinh using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.sinh(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.sinh, types.Float, types.Array)
def np_sinh_scalar_to_array_cg(builder, target, args, kwargs):
    """sinh(scalar, out_array) - compute sinh and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.sinh, "np.sinh")


@ufunc_registry.register(np.sinh)
@lower(np.sinh, types.Array)
def np_sinh_array_cg(builder, target, args, kwargs):
    """Element-wise sinh using linalg.GenericOp."""

    def sinh_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_sinh(in_elem)
        else:
            result = math_dialect.sinh(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, sinh_fn, "np.sinh")


@lower(np.sinh, types.Array, types.Array)
def np_sinh_array_to_array_cg(builder, target, args, kwargs):
    """sinh(array, out_array) - element-wise sinh to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_sinh)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.sinh)


@lower(np.cosh, types.Float)
def np_cosh_scalar_cg(builder, target, args, kwargs):
    """Scalar cosh using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.cosh(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.cosh, types.Float, types.Array)
def np_cosh_scalar_to_array_cg(builder, target, args, kwargs):
    """cosh(scalar, out_array) - compute cosh and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.cosh, "np.cosh")


@ufunc_registry.register(np.cosh)
@lower(np.cosh, types.Array)
def np_cosh_array_cg(builder, target, args, kwargs):
    """Element-wise cosh using linalg.GenericOp."""

    def cosh_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_cosh(in_elem)
        else:
            result = math_dialect.cosh(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, cosh_fn, "np.cosh")


@lower(np.cosh, types.Array, types.Array)
def np_cosh_array_to_array_cg(builder, target, args, kwargs):
    """cosh(array, out_array) - element-wise cosh to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_cosh)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.cosh)


@lower(np.tanh, types.Float)
def np_tanh_scalar_cg(builder, target, args, kwargs):
    """Scalar tanh using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.tanh(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.tanh, types.Float, types.Array)
def np_tanh_scalar_to_array_cg(builder, target, args, kwargs):
    """tanh(scalar, out_array) - compute tanh and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.tanh, "np.tanh")


@ufunc_registry.register(np.tanh)
@lower(np.tanh, types.Array)
def np_tanh_array_cg(builder, target, args, kwargs):
    """Element-wise tanh using linalg.GenericOp."""

    def tanh_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = complex_dialect.tanh(in_elem)
        else:
            result = math_dialect.tanh(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, tanh_fn, "np.tanh")


@lower(np.tanh, types.Array, types.Array)
def np_tanh_array_to_array_cg(builder, target, args, kwargs):
    """tanh(array, out_array) - element-wise tanh to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.tanh)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.tanh)


@lower(np.arcsinh, types.Float)
def np_arcsinh_scalar_cg(builder, target, args, kwargs):
    """Scalar arcsinh using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.asinh(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arcsinh, types.Float, types.Array)
def np_arcsinh_scalar_to_array_cg(builder, target, args, kwargs):
    """arcsinh(scalar, out_array) - compute arcsinh and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.asinh, "np.arcsinh")


@ufunc_registry.register(np.arcsinh)
@lower(np.arcsinh, types.Array)
def np_arcsinh_array_cg(builder, target, args, kwargs):
    """Element-wise arcsinh using linalg.GenericOp."""

    def arcsinh_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_asinh(in_elem)
        else:
            result = math_dialect.asinh(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, arcsinh_fn, "np.arcsinh")


@lower(np.arcsinh, types.Array, types.Array)
def np_arcsinh_array_to_array_cg(builder, target, args, kwargs):
    """arcsinh(array, out_array) - element-wise arcsinh to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_asinh)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.asinh)


@lower(np.arccosh, types.Float)
def np_arccosh_scalar_cg(builder, target, args, kwargs):
    """Scalar arccosh using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.acosh(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arccosh, types.Float, types.Array)
def np_arccosh_scalar_to_array_cg(builder, target, args, kwargs):
    """arccosh(scalar, out_array) - compute arccosh and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.acosh, "np.arccosh")


@ufunc_registry.register(np.arccosh)
@lower(np.arccosh, types.Array)
def np_arccosh_array_cg(builder, target, args, kwargs):
    """Element-wise arccosh using linalg.GenericOp."""

    def arccosh_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_acosh(in_elem)
        else:
            result = math_dialect.acosh(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, arccosh_fn, "np.arccosh")


@lower(np.arccosh, types.Array, types.Array)
def np_arccosh_array_to_array_cg(builder, target, args, kwargs):
    """arccosh(array, out_array) - element-wise arccosh to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_acosh)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.acosh)


@lower(np.arctanh, types.Float)
def np_arctanh_scalar_cg(builder, target, args, kwargs):
    """Scalar arctanh using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.atanh(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arctanh, types.Float, types.Array)
def np_arctanh_scalar_to_array_cg(builder, target, args, kwargs):
    """arctanh(scalar, out_array) - compute arctanh and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.atanh, "np.arctanh")


@ufunc_registry.register(np.arctanh)
@lower(np.arctanh, types.Array)
def np_arctanh_array_cg(builder, target, args, kwargs):
    """Element-wise arctanh using linalg.GenericOp."""

    def arctanh_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_atanh(in_elem)
        else:
            result = math_dialect.atanh(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, arctanh_fn, "np.arctanh")


@lower(np.arctanh, types.Array, types.Array)
def np_arctanh_array_to_array_cg(builder, target, args, kwargs):
    """arctanh(array, out_array) - element-wise arctanh to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_atanh)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.atanh)


@lower(np.sqrt, types.Float, types.Array)
def np_sqrt_scalar_to_array_cg(builder, target, args, kwargs):
    """sqrt(scalar, out_array) - compute sqrt and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.sqrt, "np.sqrt")


@lower(np.sqrt, types.Array, types.Array)
def np_sqrt_array_to_array_cg(builder, target, args, kwargs):
    """sqrt(array, out_array) - element-wise sqrt to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.sqrt)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.sqrt)


@lower(np.exp, types.Float, types.Array)
def np_exp_scalar_to_array_cg(builder, target, args, kwargs):
    """exp(scalar, out_array) - compute exp and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.exp, "np.exp")


@lower(np.exp, types.Array, types.Array)
def np_exp_array_to_array_cg(builder, target, args, kwargs):
    """exp(array, out_array) - element-wise exp to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.exp)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.exp)


@lower(np.log, types.Float, types.Array)
def np_log_scalar_to_array_cg(builder, target, args, kwargs):
    """log(scalar, out_array) - compute log and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.log, "np.log")


@lower(np.log, types.Array, types.Array)
def np_log_array_to_array_cg(builder, target, args, kwargs):
    """log(array, out_array) - element-wise log to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, complex_dialect.log)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.log)


# Binary ufunc helpers


def create_binary_elementwise_op(builder, target, args, kwargs, math_fn, op_name):
    """
    Helper for implementing binary element-wise operations using linalg.GenericOp.

    This function creates an output array automatically, similar to create_elementwise_op
    but for binary operations.

    Args:
        builder: MLIR builder instance
        target: Target variable for the result
        args: Function arguments (should be two arrays)
        kwargs: Keyword arguments (should be empty)
        math_fn: Function(in1_elem, in2_elem) -> result
        op_name: Name of the operation for error messages
    """
    assert len(args) == 2 and len(kwargs) == 0, f"{op_name} takes exactly two arguments"
    in1_arg, in2_arg = args

    # Extract types from input arrays and target
    in1_array_type = builder.get_numba_type(in1_arg.name)
    in2_array_type = builder.get_numba_type(in2_arg.name)
    in1_element_type = in1_array_type.dtype
    in2_element_type = in2_array_type.dtype
    in1_mlir_type = builder.get_value_type(in1_element_type)
    in2_mlir_type = builder.get_value_type(in2_element_type)

    target_array_type = builder.get_numba_type(target.name)
    target_element_type = target_array_type.dtype
    target_mlir_type = builder.get_value_type(target_element_type)

    in1_memref = builder.load_var(in1_arg)
    in2_memref = builder.load_var(in2_arg)
    in1_tensor = lowering_utilities.memref_to_value_tensor(in1_array_type, in1_memref)
    in2_tensor = lowering_utilities.memref_to_value_tensor(in2_array_type, in2_memref)
    input_tensor_type = in1_tensor.type

    # Extract dynamic dimensions for tensor.empty (use target type for output)
    rank = input_tensor_type.rank
    dynamic_sizes = []
    for i in range(rank):
        if input_tensor_type.shape[i] == ir.ShapedType.get_dynamic_size():
            dim_size = memref.dim(
                source=in1_memref, index=arith.constant(result=T.index(), value=i)
            )
            dynamic_sizes.append(dim_size)

    # Create empty output tensor with target element type
    init = tensor.EmptyOp(dynamic_sizes, element_type=target_mlir_type).result

    # Use linalg.GenericOp for element-wise operation
    affine_map = ir.AffineMap.get_identity(rank)
    indexing_maps_attr = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(affine_map),  # input1
            ir.AffineMapAttr.get(affine_map),  # input2
            ir.AffineMapAttr.get(affine_map),  # output
        ]
    )
    iterator_types_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )

    generic_op = linalg.GenericOp(
        [init.type],  # result_tensors
        [in1_tensor, in2_tensor],  # inputs
        [init],  # outputs
        indexing_maps_attr,
        iterator_types_attr,
    )
    region = generic_op.regions[0]
    block = region.blocks.append(in1_mlir_type, in2_mlir_type, target_mlir_type)
    in1_elem = block.arguments[0]
    in2_elem = block.arguments[1]
    with ir.InsertionPoint(block):
        # Convert inputs to target type
        in1_elem_conv = lowering_utilities.convert(in1_elem, target_mlir_type)
        in2_elem_conv = lowering_utilities.convert(in2_elem, target_mlir_type)
        result = math_fn(in1_elem_conv, in2_elem_conv)
        result = lowering_utilities.convert(result, target_mlir_type)
        linalg.yield_([result])

    # Get the result and convert back to memref
    map_result = next(iter(cast(Any, generic_op.operation.results)))
    result_memref = lowering_utilities.value_tensor_to_storage_memref(target_array_type, map_result)
    builder.store_var(target, result_memref)


def create_binary_scalar_to_output_array(builder, target, args, math_fn, op_name):
    """
    Helper for binary operations with two scalar inputs and output array.
    Handles signatures like: np.arctan2(y, x, output_array)
    The scalar result is stored in output_array[0].
    """
    assert len(args) == 3, f"{op_name} expects 3 arguments (in1, in2, output)"
    in1_arg, in2_arg, output_arg = args

    in1_val = builder.load_var(in1_arg)
    in2_val = builder.load_var(in2_arg)
    out_arr = builder.load_var(output_arg)
    output_array_type = builder.get_numba_type(output_arg.name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    in1_val = convert(in1_val, elem_type)
    in2_val = convert(in2_val, elem_type)
    result_val = math_fn(in1_val, in2_val)
    result_val = convert(result_val, elem_type)
    _store_first_output_value(builder, output_arg, out_arr, result_val)
    builder.store_var(target, out_arr)


def create_binary_elementwise_op_with_output(
    builder, target, args, math_fn, op_name, convert_inputs=True
):
    """
    Helper for binary element-wise operations with output array.
    Handles signatures like: np.arctan2(y_array, x_array, output_array)

    Args:
        convert_inputs: If True, convert inputs to output type before operation.
                       Set to False for comparisons where inputs stay in original type.
    """
    assert len(args) == 3, f"{op_name} expects 3 arguments (in1, in2, output)"
    in1_arg, in2_arg, output_arg = args

    in1_array_type = builder.get_numba_type(in1_arg.name)
    in2_array_type = builder.get_numba_type(in2_arg.name)
    output_array_type = builder.get_numba_type(output_arg.name)
    in1_mlir_elem_type = builder.get_value_type(in1_array_type.dtype)
    in2_mlir_elem_type = builder.get_value_type(in2_array_type.dtype)
    output_mlir_elem_type = builder.get_value_type(output_array_type.dtype)

    in1_memref = builder.load_var(in1_arg)
    in2_memref = builder.load_var(in2_arg)
    output_memref = builder.load_var(output_arg)

    in1_tensor = lowering_utilities.memref_to_value_tensor(in1_array_type, in1_memref)
    in2_tensor = lowering_utilities.memref_to_value_tensor(in2_array_type, in2_memref)
    output_tensor = memref_to_tensor(output_memref)
    output_storage_elem_type = output_tensor.type.element_type

    rank = in1_tensor.type.rank
    affine_map = ir.AffineMap.get_identity(rank)

    indexing_maps_attr = ir.ArrayAttr.get(
        [
            ir.AffineMapAttr.get(affine_map),  # input1
            ir.AffineMapAttr.get(affine_map),  # input2
            ir.AffineMapAttr.get(affine_map),  # output
        ]
    )
    iterator_types_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )

    generic_op = linalg.GenericOp(
        [output_tensor.type],  # result_tensors
        [in1_tensor, in2_tensor],  # inputs
        [output_tensor],  # outputs
        indexing_maps_attr,
        iterator_types_attr,
    )
    region = generic_op.regions[0]
    block = region.blocks.append(in1_mlir_elem_type, in2_mlir_elem_type, output_storage_elem_type)
    in1_elem = block.arguments[0]
    in2_elem = block.arguments[1]
    with ir.InsertionPoint(block):
        if convert_inputs:
            in1_elem = convert(in1_elem, output_mlir_elem_type)
            in2_elem = convert(in2_elem, output_mlir_elem_type)
        result = math_fn(in1_elem, in2_elem)
        result = convert(result, output_mlir_elem_type)
        result = lowering_utilities.value_to_storage(output_array_type.dtype, result)
        linalg.yield_([result])

    map_result = next(iter(cast(Any, generic_op.operation.results)))
    result_memref = tensor_to_memref(map_result)
    # Copy result back to the original output memref (in-place semantics)
    memref.copy(result_memref, output_memref)
    builder.store_var(target, output_memref)


# arctan2 lowerings


@lower(np.arctan2, types.Float, types.Float)
def np_arctan2_scalar_cg(builder, target, args, kwargs):
    """Scalar arctan2 using MLIR math dialect"""
    assert len(args) == 2 and len(kwargs) == 0
    y = builder.load_var(args[0])
    x = builder.load_var(args[1])
    result = math_dialect.atan2(y, x)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.arctan2, types.Float, types.Float, types.Array)
def np_arctan2_scalar_to_array_cg(builder, target, args, kwargs):
    """arctan2(y, x, out_array) - compute arctan2 and store in output array"""
    create_binary_scalar_to_output_array(builder, target, args, math_dialect.atan2, "np.arctan2")


@ufunc_registry.register(np.arctan2)
@lower(np.arctan2, types.Array, types.Array)
def np_arctan2_array_cg(builder, target, args, kwargs):
    """Element-wise arctan2 using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, math_dialect.atan2, "np.arctan2")


@lower(np.arctan2, types.Array, types.Array, types.Array)
def np_arctan2_array_to_array_cg(builder, target, args, kwargs):
    """arctan2(y_array, x_array, out_array) - element-wise arctan2"""
    create_binary_elementwise_op_with_output(
        builder, target, args, math_dialect.atan2, "np.arctan2"
    )


# hypot lowerings (implemented as sqrt(x*x + y*y))


def _hypot_fn(x, y):
    """Compute hypot(x, y) = sqrt(x*x + y*y)"""
    x_sq = arith.mulf(x, x)
    y_sq = arith.mulf(y, y)
    sum_sq = arith.addf(x_sq, y_sq)
    return math_dialect.sqrt(sum_sq)


@lower(np.hypot, types.Float, types.Float)
def np_hypot_scalar_cg(builder, target, args, kwargs):
    """Scalar hypot using MLIR math dialect"""
    assert len(args) == 2 and len(kwargs) == 0
    x = builder.load_var(args[0])
    y = builder.load_var(args[1])
    result = _hypot_fn(x, y)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.hypot, types.Float, types.Float, types.Array)
def np_hypot_scalar_to_array_cg(builder, target, args, kwargs):
    """hypot(x, y, out_array) - compute hypot and store in output array"""
    create_binary_scalar_to_output_array(builder, target, args, _hypot_fn, "np.hypot")


@ufunc_registry.register(np.hypot)
@lower(np.hypot, types.Array, types.Array)
def np_hypot_array_cg(builder, target, args, kwargs):
    """Element-wise hypot using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _hypot_fn, "np.hypot")


@lower(np.hypot, types.Array, types.Array, types.Array)
def np_hypot_array_to_array_cg(builder, target, args, kwargs):
    """hypot(x_array, y_array, out_array) - element-wise hypot"""
    create_binary_elementwise_op_with_output(builder, target, args, _hypot_fn, "np.hypot")


# Comparison ufuncs


def _is_complex_type(mlir_type):
    """Check if an MLIR type is a complex type."""
    return isinstance(mlir_type, ir.ComplexType)


def _create_comparison_fn(int_pred, float_pred, complex_fn=None):
    """Create a comparison function for int, float, and complex types."""

    def cmp_fn(a, b):
        if isinstance(a.type, ir.IntegerType):
            return arith.cmpi(int_pred, a, b)
        elif _is_complex_type(a.type):
            if complex_fn is not None:
                return complex_fn(a, b)
            else:
                raise NotImplementedError(f"Ordering comparison not supported for complex types")
        else:
            return arith.cmpf(float_pred, a, b)

    return cmp_fn


def _complex_equal(a, b):
    """Complex equality: both real and imag parts equal."""
    return complex_dialect.eq(a, b)


def _complex_not_equal(a, b):
    """Complex inequality: either real or imag part differs."""
    return complex_dialect.neq(a, b)


def _complex_greater(a, b):
    """Complex greater: compare real first, then imag (NumPy convention)."""
    a_real = complex_dialect.re(a)
    b_real = complex_dialect.re(b)
    a_imag = complex_dialect.im(a)
    b_imag = complex_dialect.im(b)
    real_gt = arith.cmpf(arith.CmpFPredicate.OGT, a_real, b_real)
    real_eq = arith.cmpf(arith.CmpFPredicate.OEQ, a_real, b_real)
    imag_gt = arith.cmpf(arith.CmpFPredicate.OGT, a_imag, b_imag)
    return arith.ori(real_gt, arith.andi(real_eq, imag_gt))


def _complex_greater_equal(a, b):
    """Complex greater_equal: compare real first, then imag (NumPy convention)."""
    a_real = complex_dialect.re(a)
    b_real = complex_dialect.re(b)
    a_imag = complex_dialect.im(a)
    b_imag = complex_dialect.im(b)
    real_gt = arith.cmpf(arith.CmpFPredicate.OGT, a_real, b_real)
    real_eq = arith.cmpf(arith.CmpFPredicate.OEQ, a_real, b_real)
    imag_ge = arith.cmpf(arith.CmpFPredicate.OGE, a_imag, b_imag)
    return arith.ori(real_gt, arith.andi(real_eq, imag_ge))


def _complex_less(a, b):
    """Complex less: compare real first, then imag (NumPy convention)."""
    a_real = complex_dialect.re(a)
    b_real = complex_dialect.re(b)
    a_imag = complex_dialect.im(a)
    b_imag = complex_dialect.im(b)
    real_lt = arith.cmpf(arith.CmpFPredicate.OLT, a_real, b_real)
    real_eq = arith.cmpf(arith.CmpFPredicate.OEQ, a_real, b_real)
    imag_lt = arith.cmpf(arith.CmpFPredicate.OLT, a_imag, b_imag)
    return arith.ori(real_lt, arith.andi(real_eq, imag_lt))


def _complex_less_equal(a, b):
    """Complex less_equal: compare real first, then imag (NumPy convention)."""
    a_real = complex_dialect.re(a)
    b_real = complex_dialect.re(b)
    a_imag = complex_dialect.im(a)
    b_imag = complex_dialect.im(b)
    real_lt = arith.cmpf(arith.CmpFPredicate.OLT, a_real, b_real)
    real_eq = arith.cmpf(arith.CmpFPredicate.OEQ, a_real, b_real)
    imag_le = arith.cmpf(arith.CmpFPredicate.OLE, a_imag, b_imag)
    return arith.ori(real_lt, arith.andi(real_eq, imag_le))


_greater_fn = _create_comparison_fn(
    arith.CmpIPredicate.sgt, arith.CmpFPredicate.OGT, _complex_greater
)
_greater_equal_fn = _create_comparison_fn(
    arith.CmpIPredicate.sge, arith.CmpFPredicate.OGE, _complex_greater_equal
)
_less_fn = _create_comparison_fn(arith.CmpIPredicate.slt, arith.CmpFPredicate.OLT, _complex_less)
_less_equal_fn = _create_comparison_fn(
    arith.CmpIPredicate.sle, arith.CmpFPredicate.OLE, _complex_less_equal
)
_equal_fn = _create_comparison_fn(arith.CmpIPredicate.eq, arith.CmpFPredicate.OEQ, _complex_equal)
_not_equal_fn = _create_comparison_fn(
    arith.CmpIPredicate.ne, arith.CmpFPredicate.ONE, _complex_not_equal
)


def _create_comparison_scalar_lowering(cmp_fn, op_name):
    """Create a scalar comparison lowering that stores result in output array."""

    def lowering(builder, target, args, kwargs):
        assert len(args) == 3 and len(kwargs) == 0
        a = builder.load_var(args[0])
        b = builder.load_var(args[1])
        out_arr = builder.load_var(args[2])
        result = cmp_fn(a, b)
        # Convert i1 result to output value type, then to storage.
        output_array_type = builder.get_numba_type(args[2].name)
        elem_type = builder.get_value_type(output_array_type.dtype)
        result = _bool_to_value_type(result, elem_type)
        _store_first_output_value(builder, args[2], out_arr, result)
        builder.store_var(target, out_arr)

    return lowering


@lower(np.greater, types.Number, types.Number, types.Array)
def np_greater_scalar_to_array_cg(builder, target, args, kwargs):
    """greater(a, b, out) - scalar a > b to output array"""
    _create_comparison_scalar_lowering(_greater_fn, "np.greater")(builder, target, args, kwargs)


@lower(np.greater, types.Array, types.Array, types.Array)
def np_greater_array_to_array_cg(builder, target, args, kwargs):
    """greater(a, b, out) - element-wise a > b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _greater_fn, "np.greater", convert_inputs=False
    )


@lower(np.greater_equal, types.Number, types.Number, types.Array)
def np_greater_equal_scalar_to_array_cg(builder, target, args, kwargs):
    _create_comparison_scalar_lowering(_greater_equal_fn, "np.greater_equal")(
        builder, target, args, kwargs
    )


@lower(np.greater_equal, types.Array, types.Array, types.Array)
def np_greater_equal_array_to_array_cg(builder, target, args, kwargs):
    """greater_equal(a, b, out) - element-wise a >= b"""
    create_binary_elementwise_op_with_output(
        builder,
        target,
        args,
        _greater_equal_fn,
        "np.greater_equal",
        convert_inputs=False,
    )


@lower(np.less, types.Number, types.Number, types.Array)
def np_less_scalar_to_array_cg(builder, target, args, kwargs):
    _create_comparison_scalar_lowering(_less_fn, "np.less")(builder, target, args, kwargs)


@lower(np.less, types.Array, types.Array, types.Array)
def np_less_array_to_array_cg(builder, target, args, kwargs):
    """less(a, b, out) - element-wise a < b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _less_fn, "np.less", convert_inputs=False
    )


@lower(np.less_equal, types.Number, types.Number, types.Array)
def np_less_equal_scalar_to_array_cg(builder, target, args, kwargs):
    _create_comparison_scalar_lowering(_less_equal_fn, "np.less_equal")(
        builder, target, args, kwargs
    )


@lower(np.less_equal, types.Array, types.Array, types.Array)
def np_less_equal_array_to_array_cg(builder, target, args, kwargs):
    """less_equal(a, b, out) - element-wise a <= b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _less_equal_fn, "np.less_equal", convert_inputs=False
    )


@lower(np.equal, types.Number, types.Number, types.Array)
def np_equal_scalar_to_array_cg(builder, target, args, kwargs):
    _create_comparison_scalar_lowering(_equal_fn, "np.equal")(builder, target, args, kwargs)


@lower(np.equal, types.Array, types.Array, types.Array)
def np_equal_array_to_array_cg(builder, target, args, kwargs):
    """equal(a, b, out) - element-wise a == b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _equal_fn, "np.equal", convert_inputs=False
    )


@lower(np.not_equal, types.Number, types.Number, types.Array)
def np_not_equal_scalar_to_array_cg(builder, target, args, kwargs):
    _create_comparison_scalar_lowering(_not_equal_fn, "np.not_equal")(builder, target, args, kwargs)


@lower(np.not_equal, types.Array, types.Array, types.Array)
def np_not_equal_array_to_array_cg(builder, target, args, kwargs):
    """not_equal(a, b, out) - element-wise a != b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _not_equal_fn, "np.not_equal", convert_inputs=False
    )


# Logical ufuncs


def _to_bool(val):
    """Convert value to i1 boolean (non-zero = true)."""
    if isinstance(val.type, ir.IntegerType):
        zero = arith.constant(result=val.type, value=0)
        return arith.cmpi(arith.CmpIPredicate.ne, val, zero)
    elif _is_complex_type(val.type):
        # For complex, check if value != 0+0j
        elem_type = val.type.element_type
        zero_r = arith.constant(result=elem_type, value=0.0)
        zero_c = complex_dialect.create_(val.type, zero_r, zero_r)
        return complex_dialect.neq(val, zero_c)
    else:
        zero = arith.constant(result=val.type, value=0.0)
        return arith.cmpf(arith.CmpFPredicate.ONE, val, zero)


def _logical_and_fn(a, b):
    """Logical AND: convert to bool, AND, return i1."""
    a_bool = _to_bool(a)
    b_bool = _to_bool(b)
    return arith.andi(a_bool, b_bool)


def _logical_or_fn(a, b):
    """Logical OR: convert to bool, OR, return i1."""
    a_bool = _to_bool(a)
    b_bool = _to_bool(b)
    return arith.ori(a_bool, b_bool)


def _logical_xor_fn(a, b):
    """Logical XOR: convert to bool, XOR, return i1."""
    a_bool = _to_bool(a)
    b_bool = _to_bool(b)
    return arith.xori(a_bool, b_bool)


@ufunc_registry.register(np.logical_and)
@lower(np.logical_and, types.Array, types.Array)
def np_logical_and_array_cg(builder, target, args, kwargs):
    """Element-wise logical_and using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _logical_and_fn, "np.logical_and")


@lower(np.logical_and, types.Array, types.Array, types.Array)
def np_logical_and_array_to_array_cg(builder, target, args, kwargs):
    """logical_and(a, b, out) - element-wise a and b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _logical_and_fn, "np.logical_and"
    )


@ufunc_registry.register(np.logical_or)
@lower(np.logical_or, types.Array, types.Array)
def np_logical_or_array_cg(builder, target, args, kwargs):
    """Element-wise logical_or using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _logical_or_fn, "np.logical_or")


@lower(np.logical_or, types.Array, types.Array, types.Array)
def np_logical_or_array_to_array_cg(builder, target, args, kwargs):
    """logical_or(a, b, out) - element-wise a or b"""
    create_binary_elementwise_op_with_output(builder, target, args, _logical_or_fn, "np.logical_or")


@ufunc_registry.register(np.logical_xor)
@lower(np.logical_xor, types.Array, types.Array)
def np_logical_xor_array_cg(builder, target, args, kwargs):
    """Element-wise logical_xor using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _logical_xor_fn, "np.logical_xor")


@lower(np.logical_xor, types.Array, types.Array, types.Array)
def np_logical_xor_array_to_array_cg(builder, target, args, kwargs):
    """logical_xor(a, b, out) - element-wise a xor b"""
    create_binary_elementwise_op_with_output(
        builder, target, args, _logical_xor_fn, "np.logical_xor"
    )


def _logical_not_fn(a):
    """Logical NOT: convert to bool, invert, return i1."""
    a_bool = _to_bool(a)
    true_val = arith.constant(result=ir.IntegerType.get_signless(1), value=1)
    return arith.xori(a_bool, true_val)


@ufunc_registry.register(np.logical_not)
@lower(np.logical_not, types.Array)
def np_logical_not_array_cg(builder, target, args, kwargs):
    """Element-wise logical_not using linalg.GenericOp."""

    def logical_not_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        result = _logical_not_fn(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, logical_not_fn, "np.logical_not")


@lower(np.logical_not, types.Array, types.Array)
def np_logical_not_array_to_array_cg(builder, target, args, kwargs):
    """logical_not(a, out) - element-wise not a"""
    create_elementwise_op_with_output(builder, target, args, _logical_not_fn)


# Min/max ufuncs


def _maximum_fn(a, b):
    if isinstance(a.type, ir.IntegerType):
        cmp = arith.cmpi(arith.CmpIPredicate.sgt, a, b)
    elif _is_complex_type(a.type):
        cmp = _complex_greater(a, b)
    else:
        cmp = arith.cmpf(arith.CmpFPredicate.OGT, a, b)
    return arith.select(cmp, a, b)


def _minimum_fn(a, b):
    if isinstance(a.type, ir.IntegerType):
        cmp = arith.cmpi(arith.CmpIPredicate.slt, a, b)
    elif _is_complex_type(a.type):
        cmp = _complex_less(a, b)
    else:
        cmp = arith.cmpf(arith.CmpFPredicate.OLT, a, b)
    return arith.select(cmp, a, b)


def _fmax_fn(a, b):
    """fmax ignores NaN: if one is NaN, return the other"""
    if isinstance(a.type, ir.IntegerType):
        # For integers, same as maximum
        cmp = arith.cmpi(arith.CmpIPredicate.sgt, a, b)
        return arith.select(cmp, a, b)
    elif _is_complex_type(a.type):
        # For complex, compare like maximum
        cmp = _complex_greater(a, b)
        return arith.select(cmp, a, b)
    else:
        return arith.maximumf(a, b)


def _fmin_fn(a, b):
    """fmin ignores NaN: if one is NaN, return the other"""
    if isinstance(a.type, ir.IntegerType):
        # For integers, same as minimum
        cmp = arith.cmpi(arith.CmpIPredicate.slt, a, b)
        return arith.select(cmp, a, b)
    elif _is_complex_type(a.type):
        # For complex, compare like minimum
        cmp = _complex_less(a, b)
        return arith.select(cmp, a, b)
    else:
        return arith.minimumf(a, b)


@ufunc_registry.register(np.maximum)
@lower(np.maximum, types.Array, types.Array)
def np_maximum_array_cg(builder, target, args, kwargs):
    """Element-wise maximum using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _maximum_fn, "np.maximum")


@lower(np.maximum, types.Array, types.Array, types.Array)
def np_maximum_array_to_array_cg(builder, target, args, kwargs):
    """maximum(a, b, out) - element-wise max(a, b)"""
    create_binary_elementwise_op_with_output(builder, target, args, _maximum_fn, "np.maximum")


@ufunc_registry.register(np.minimum)
@lower(np.minimum, types.Array, types.Array)
def np_minimum_array_cg(builder, target, args, kwargs):
    """Element-wise minimum using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _minimum_fn, "np.minimum")


@lower(np.minimum, types.Array, types.Array, types.Array)
def np_minimum_array_to_array_cg(builder, target, args, kwargs):
    """minimum(a, b, out) - element-wise min(a, b)"""
    create_binary_elementwise_op_with_output(builder, target, args, _minimum_fn, "np.minimum")


@ufunc_registry.register(np.fmax)
@lower(np.fmax, types.Array, types.Array)
def np_fmax_array_cg(builder, target, args, kwargs):
    """Element-wise fmax using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _fmax_fn, "np.fmax")


@lower(np.fmax, types.Array, types.Array, types.Array)
def np_fmax_array_to_array_cg(builder, target, args, kwargs):
    """fmax(a, b, out) - element-wise fmax (NaN-ignoring max)"""
    create_binary_elementwise_op_with_output(builder, target, args, _fmax_fn, "np.fmax")


@ufunc_registry.register(np.fmin)
@lower(np.fmin, types.Array, types.Array)
def np_fmin_array_cg(builder, target, args, kwargs):
    """Element-wise fmin using linalg.GenericOp."""
    create_binary_elementwise_op(builder, target, args, kwargs, _fmin_fn, "np.fmin")


@lower(np.fmin, types.Array, types.Array, types.Array)
def np_fmin_array_to_array_cg(builder, target, args, kwargs):
    """fmin(a, b, out) - element-wise fmin (NaN-ignoring min)"""
    create_binary_elementwise_op_with_output(builder, target, args, _fmin_fn, "np.fmin")


# Bitwise ufuncs


@lower(np.bitwise_and, types.Array, types.Array, types.Array)
def np_bitwise_and_array_to_array_cg(builder, target, args, kwargs):
    """bitwise_and(a, b, out) - element-wise a & b"""
    create_binary_elementwise_op_with_output(builder, target, args, arith.andi, "np.bitwise_and")


@lower(np.bitwise_or, types.Array, types.Array, types.Array)
def np_bitwise_or_array_to_array_cg(builder, target, args, kwargs):
    """bitwise_or(a, b, out) - element-wise a | b"""
    create_binary_elementwise_op_with_output(builder, target, args, arith.ori, "np.bitwise_or")


@lower(np.bitwise_xor, types.Array, types.Array, types.Array)
def np_bitwise_xor_array_to_array_cg(builder, target, args, kwargs):
    """bitwise_xor(a, b, out) - element-wise a ^ b"""
    create_binary_elementwise_op_with_output(builder, target, args, arith.xori, "np.bitwise_xor")


def _bitwise_not_fn(a):
    """Bitwise NOT: ~a = a XOR all_ones"""
    all_ones = arith.constant(result=a.type, value=-1)
    return arith.xori(a, all_ones)


@lower(np.invert, types.Array, types.Array)
@lower(np.bitwise_not, types.Array, types.Array)
def np_bitwise_not_array_to_array_cg(builder, target, args, kwargs):
    """bitwise_not(a, out) - element-wise ~a"""
    create_elementwise_op_with_output(builder, target, args, _bitwise_not_fn)


# Log ufuncs with output array


@lower(np.log2, types.Float)
def np_log2_scalar_cg(builder, target, args, kwargs):
    """Scalar log2 using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.log2(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.log2, types.Float, types.Array)
def np_log2_scalar_to_array_cg(builder, target, args, kwargs):
    """log2(scalar, out_array) - compute log2 and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.log2, "np.log2")


@ufunc_registry.register(np.log2)
@lower(np.log2, types.Array)
def np_log2_array_cg(builder, target, args, kwargs):
    """Element-wise log2 using linalg.GenericOp."""

    def log2_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_log2(in_elem)
        else:
            result = math_dialect.log2(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, log2_fn, "np.log2")


@lower(np.log2, types.Array, types.Array)
def np_log2_array_to_array_cg(builder, target, args, kwargs):
    """log2(array, out_array) - element-wise log2 to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_log2)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.log2)


@lower(np.log10, types.Float)
def np_log10_scalar_cg(builder, target, args, kwargs):
    """Scalar log10 using MLIR math dialect"""
    assert len(args) == 1 and len(kwargs) == 0
    value = builder.load_var(args[0])
    result = math_dialect.log10(value)
    result = convert(result, builder.get_mlir_type(target))
    builder.store_var(target, result)


@lower(np.log10, types.Float, types.Array)
def np_log10_scalar_to_array_cg(builder, target, args, kwargs):
    """log10(scalar, out_array) - compute log10 and store in output array"""
    create_scalar_to_output_array(builder, target, args, math_dialect.log10, "np.log10")


@ufunc_registry.register(np.log10)
@lower(np.log10, types.Array)
def np_log10_array_cg(builder, target, args, kwargs):
    """Element-wise log10 using linalg.GenericOp."""

    def log10_fn(
        input_element_type,
        target_element_type,
        input_mlir_type,
        target_mlir_type,
        in_elem,
    ):
        if isinstance(input_element_type, types.Complex):
            result = _complex_log10(in_elem)
        else:
            result = math_dialect.log10(in_elem)
        return lowering_utilities.convert(result, target_mlir_type)

    create_elementwise_op(builder, target, args, kwargs, log10_fn, "np.log10")


@lower(np.log10, types.Array, types.Array)
def np_log10_array_to_array_cg(builder, target, args, kwargs):
    """log10(array, out_array) - element-wise log10 to output array"""
    input_array_type = builder.get_numba_type(args[0].name)
    if isinstance(input_array_type.dtype, types.Complex):
        create_complex_elementwise_op_with_output(builder, target, args, _complex_log10)
    else:
        create_elementwise_op_with_output(builder, target, args, math_dialect.log10)


# =============================================================================
# Scalar lowerings for logical, min/max, and bitwise ufuncs
# =============================================================================


def _create_logical_binary_scalar_lowering(logic_fn, op_name):
    """Create a scalar logical lowering that stores result in output array."""

    def lowering(builder, target, args, kwargs):
        assert len(args) == 3 and len(kwargs) == 0
        a = builder.load_var(args[0])
        b = builder.load_var(args[1])
        out_arr = builder.load_var(args[2])
        result = logic_fn(a, b)  # Returns i1
        output_array_type = builder.get_numba_type(args[2].name)
        elem_type = builder.get_value_type(output_array_type.dtype)
        result = _bool_to_value_type(result, elem_type)
        _store_first_output_value(builder, args[2], out_arr, result)
        builder.store_var(target, out_arr)

    return lowering


def _create_logical_unary_scalar_lowering(logic_fn, op_name):
    """Create a scalar logical NOT lowering that stores result in output array."""

    def lowering(builder, target, args, kwargs):
        assert len(args) == 2 and len(kwargs) == 0
        a = builder.load_var(args[0])
        out_arr = builder.load_var(args[1])
        result = logic_fn(a)  # Returns i1
        output_array_type = builder.get_numba_type(args[1].name)
        elem_type = builder.get_value_type(output_array_type.dtype)
        result = _bool_to_value_type(result, elem_type)
        _store_first_output_value(builder, args[1], out_arr, result)
        builder.store_var(target, out_arr)

    return lowering


@lower(np.logical_and, types.Number, types.Number, types.Array)
def np_logical_and_scalar_to_array_cg(builder, target, args, kwargs):
    """logical_and(a, b, out) - scalar a and b to output array"""
    _create_logical_binary_scalar_lowering(_logical_and_fn, "np.logical_and")(
        builder, target, args, kwargs
    )


@lower(np.logical_or, types.Number, types.Number, types.Array)
def np_logical_or_scalar_to_array_cg(builder, target, args, kwargs):
    """logical_or(a, b, out) - scalar a or b to output array"""
    _create_logical_binary_scalar_lowering(_logical_or_fn, "np.logical_or")(
        builder, target, args, kwargs
    )


@lower(np.logical_xor, types.Number, types.Number, types.Array)
def np_logical_xor_scalar_to_array_cg(builder, target, args, kwargs):
    """logical_xor(a, b, out) - scalar a xor b to output array"""
    _create_logical_binary_scalar_lowering(_logical_xor_fn, "np.logical_xor")(
        builder, target, args, kwargs
    )


@lower(np.logical_not, types.Number, types.Array)
def np_logical_not_scalar_to_array_cg(builder, target, args, kwargs):
    """logical_not(a, out) - scalar not a to output array"""
    _create_logical_unary_scalar_lowering(_logical_not_fn, "np.logical_not")(
        builder, target, args, kwargs
    )


def _create_binary_scalar_to_output_array(builder, target, args, math_fn, op_name):
    """Helper for binary scalar operations that write to an output array."""
    assert len(args) == 3
    a = builder.load_var(args[0])
    b = builder.load_var(args[1])
    out_arr = builder.load_var(args[2])
    output_array_type = builder.get_numba_type(args[2].name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    a = convert(a, elem_type)
    b = convert(b, elem_type)
    result = math_fn(a, b)
    result = convert(result, elem_type)
    _store_first_output_value(builder, args[2], out_arr, result)
    builder.store_var(target, out_arr)


@lower(np.maximum, types.Number, types.Number, types.Array)
def np_maximum_scalar_to_array_cg(builder, target, args, kwargs):
    """maximum(a, b, out) - scalar max(a, b) to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, _maximum_fn, "np.maximum")


@lower(np.minimum, types.Number, types.Number, types.Array)
def np_minimum_scalar_to_array_cg(builder, target, args, kwargs):
    """minimum(a, b, out) - scalar min(a, b) to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, _minimum_fn, "np.minimum")


@lower(np.fmax, types.Number, types.Number, types.Array)
def np_fmax_scalar_to_array_cg(builder, target, args, kwargs):
    """fmax(a, b, out) - scalar fmax(a, b) to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, _fmax_fn, "np.fmax")


@lower(np.fmin, types.Number, types.Number, types.Array)
def np_fmin_scalar_to_array_cg(builder, target, args, kwargs):
    """fmin(a, b, out) - scalar fmin(a, b) to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, _fmin_fn, "np.fmin")


@lower(np.bitwise_and, types.Integer, types.Integer, types.Array)
def np_bitwise_and_scalar_to_array_cg(builder, target, args, kwargs):
    """bitwise_and(a, b, out) - scalar a & b to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, arith.andi, "np.bitwise_and")


@lower(np.bitwise_or, types.Integer, types.Integer, types.Array)
def np_bitwise_or_scalar_to_array_cg(builder, target, args, kwargs):
    """bitwise_or(a, b, out) - scalar a | b to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, arith.ori, "np.bitwise_or")


@lower(np.bitwise_xor, types.Integer, types.Integer, types.Array)
def np_bitwise_xor_scalar_to_array_cg(builder, target, args, kwargs):
    """bitwise_xor(a, b, out) - scalar a ^ b to output array"""
    _create_binary_scalar_to_output_array(builder, target, args, arith.xori, "np.bitwise_xor")


@lower(np.invert, types.Integer, types.Array)
@lower(np.bitwise_not, types.Integer, types.Array)
def np_bitwise_not_scalar_to_array_cg(builder, target, args, kwargs):
    """bitwise_not(a, out) - scalar ~a to output array"""
    assert len(args) == 2 and len(kwargs) == 0
    a = builder.load_var(args[0])
    out_arr = builder.load_var(args[1])
    output_array_type = builder.get_numba_type(args[1].name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    a = convert(a, elem_type)
    result = _bitwise_not_fn(a)
    result = convert(result, elem_type)
    _store_first_output_value(builder, args[1], out_arr, result)
    builder.store_var(target, out_arr)


# =============================================================================
# Complex number ufunc lowerings
# =============================================================================


def _complex_acos(z):
    """acos(z) = -i * log(z + i * sqrt(1 - z^2))"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    one = arith.constant(result=element_type, value=1.0)
    zero = arith.constant(result=element_type, value=0.0)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)
    i_complex = complex_dialect.create_(target_mlir_type, zero, one)
    neg_i = arith.constant(result=element_type, value=-1.0)
    neg_i_complex = complex_dialect.create_(target_mlir_type, zero, neg_i)

    z_squared = complex_dialect.mul(z, z)
    one_minus_z2 = complex_dialect.sub(one_complex, z_squared)
    sqrt_term = complex_dialect.sqrt(one_minus_z2)
    i_sqrt = complex_dialect.mul(i_complex, sqrt_term)
    sum_term = complex_dialect.add(z, i_sqrt)
    log_term = complex_dialect.log(sum_term)
    return complex_dialect.mul(neg_i_complex, log_term)


def _complex_asin(z):
    """asin(z) = -i * log(i*z + sqrt(1 - z^2))"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    one = arith.constant(result=element_type, value=1.0)
    zero = arith.constant(result=element_type, value=0.0)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)
    i_complex = complex_dialect.create_(target_mlir_type, zero, one)
    neg_i = arith.constant(result=element_type, value=-1.0)
    neg_i_complex = complex_dialect.create_(target_mlir_type, zero, neg_i)

    iz = complex_dialect.mul(i_complex, z)
    z_squared = complex_dialect.mul(z, z)
    one_minus_z2 = complex_dialect.sub(one_complex, z_squared)
    sqrt_term = complex_dialect.sqrt(one_minus_z2)
    sum_term = complex_dialect.add(iz, sqrt_term)
    log_term = complex_dialect.log(sum_term)
    return complex_dialect.mul(neg_i_complex, log_term)


def _complex_atan(z):
    """atan(z) = i/2 * log((1 - i*z)/(1 + i*z))"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    one = arith.constant(result=element_type, value=1.0)
    zero = arith.constant(result=element_type, value=0.0)
    half = arith.constant(result=element_type, value=0.5)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)
    i_complex = complex_dialect.create_(target_mlir_type, zero, one)
    i_half_complex = complex_dialect.create_(target_mlir_type, zero, half)

    iz = complex_dialect.mul(i_complex, z)
    one_minus_iz = complex_dialect.sub(one_complex, iz)
    one_plus_iz = complex_dialect.add(one_complex, iz)
    ratio = complex_dialect.div(one_minus_iz, one_plus_iz)
    log_term = complex_dialect.log(ratio)
    return complex_dialect.mul(i_half_complex, log_term)


def _complex_sinh(z):
    """sinh(z) = (exp(z) - exp(-z)) / 2"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    two = arith.constant(result=element_type, value=2.0)
    zero = arith.constant(result=element_type, value=0.0)
    two_complex = complex_dialect.create_(target_mlir_type, two, zero)

    exp_z = complex_dialect.exp(z)
    neg_z = complex_dialect.neg(z)
    exp_neg_z = complex_dialect.exp(neg_z)
    diff = complex_dialect.sub(exp_z, exp_neg_z)
    return complex_dialect.div(diff, two_complex)


def _complex_cosh(z):
    """cosh(z) = (exp(z) + exp(-z)) / 2"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    two = arith.constant(result=element_type, value=2.0)
    zero = arith.constant(result=element_type, value=0.0)
    two_complex = complex_dialect.create_(target_mlir_type, two, zero)

    exp_z = complex_dialect.exp(z)
    neg_z = complex_dialect.neg(z)
    exp_neg_z = complex_dialect.exp(neg_z)
    sum_exp = complex_dialect.add(exp_z, exp_neg_z)
    return complex_dialect.div(sum_exp, two_complex)


def _complex_asinh(z):
    """asinh(z) = log(z + sqrt(z^2 + 1))"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    one = arith.constant(result=element_type, value=1.0)
    zero = arith.constant(result=element_type, value=0.0)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)

    z_squared = complex_dialect.mul(z, z)
    z_squared_plus_one = complex_dialect.add(z_squared, one_complex)
    sqrt_term = complex_dialect.sqrt(z_squared_plus_one)
    z_plus_sqrt = complex_dialect.add(z, sqrt_term)
    return complex_dialect.log(z_plus_sqrt)


def _complex_acosh(z):
    """acosh(z) = log(z + sqrt(z+1)*sqrt(z-1))"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    one = arith.constant(result=element_type, value=1.0)
    zero = arith.constant(result=element_type, value=0.0)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)

    z_plus_1 = complex_dialect.add(z, one_complex)
    z_minus_1 = complex_dialect.sub(z, one_complex)
    sqrt_z_plus_1 = complex_dialect.sqrt(z_plus_1)
    sqrt_z_minus_1 = complex_dialect.sqrt(z_minus_1)
    sqrt_product = complex_dialect.mul(sqrt_z_plus_1, sqrt_z_minus_1)
    z_plus_sqrt = complex_dialect.add(z, sqrt_product)
    return complex_dialect.log(z_plus_sqrt)


def _complex_atanh(z):
    """atanh(z) = 0.5 * log((1+z)/(1-z))"""
    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    one = arith.constant(result=element_type, value=1.0)
    half = arith.constant(result=element_type, value=0.5)
    zero = arith.constant(result=element_type, value=0.0)
    one_complex = complex_dialect.create_(target_mlir_type, one, zero)
    half_complex = complex_dialect.create_(target_mlir_type, half, zero)

    one_plus_z = complex_dialect.add(one_complex, z)
    one_minus_z = complex_dialect.sub(one_complex, z)
    ratio = complex_dialect.div(one_plus_z, one_minus_z)
    log_term = complex_dialect.log(ratio)
    return complex_dialect.mul(half_complex, log_term)


def _complex_log2(z):
    """log2(z) = log(z) / log(2), with edge case handling for z=0"""
    import math

    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    zero = arith.constant(result=element_type, value=0.0)
    zero_c = complex_dialect.create_(target_mlir_type, zero, zero)
    is_zero = complex_dialect.eq(z, zero_c)

    ln2 = arith.constant(result=element_type, value=math.log(2))
    ln2_complex = complex_dialect.create_(target_mlir_type, ln2, zero)
    log_z = complex_dialect.log(z)
    result = complex_dialect.div(log_z, ln2_complex)

    neg_inf = arith.constant(result=element_type, value=float("-inf"))
    zero_result = complex_dialect.create_(target_mlir_type, neg_inf, zero)

    return arith.select(is_zero, zero_result, result)


def _complex_log10(z):
    """log10(z) = log(z) / log(10), with edge case handling for z=0"""
    import math

    target_mlir_type = z.type
    element_type = target_mlir_type.element_type
    zero = arith.constant(result=element_type, value=0.0)
    zero_c = complex_dialect.create_(target_mlir_type, zero, zero)
    is_zero = complex_dialect.eq(z, zero_c)

    ln10 = arith.constant(result=element_type, value=math.log(10))
    ln10_complex = complex_dialect.create_(target_mlir_type, ln10, zero)
    log_z = complex_dialect.log(z)
    result = complex_dialect.div(log_z, ln10_complex)

    neg_inf = arith.constant(result=element_type, value=float("-inf"))
    zero_result = complex_dialect.create_(target_mlir_type, neg_inf, zero)

    return arith.select(is_zero, zero_result, result)


def create_complex_scalar_to_output_array(builder, target, args, math_fn, op_name):
    """Helper for complex scalar operations that write to an output array."""
    assert len(args) == 2
    scalar_arg = builder.load_var(args[0])
    out_arr = builder.load_var(args[1])
    output_array_type = builder.get_numba_type(args[1].name)
    elem_type = builder.get_value_type(output_array_type.dtype)
    scalar_val = convert(scalar_arg, elem_type)
    result = math_fn(scalar_val)
    result = convert(result, elem_type)
    _store_first_output_value(builder, args[1], out_arr, result)
    builder.store_var(target, out_arr)


def create_complex_elementwise_op_with_output(builder, target, args, math_fn):
    """Helper for complex element-wise operations that write to an output array."""
    assert len(args) == 2
    input_arg, output_arg = args

    input_array_type = builder.get_numba_type(input_arg.name)
    output_array_type = builder.get_numba_type(output_arg.name)
    input_mlir_elem_type = builder.get_value_type(input_array_type.dtype)
    output_mlir_elem_type = builder.get_value_type(output_array_type.dtype)

    input_memref = builder.load_var(input_arg)
    output_memref = builder.load_var(output_arg)

    input_tensor = lowering_utilities.memref_to_value_tensor(input_array_type, input_memref)
    output_tensor = memref_to_tensor(output_memref)
    output_storage_elem_type = output_tensor.type.element_type

    rank = input_tensor.type.rank
    affine_map = ir.AffineMap.get_identity(rank)

    indexing_maps_attr = ir.ArrayAttr.get(
        [ir.AffineMapAttr.get(affine_map), ir.AffineMapAttr.get(affine_map)]
    )
    iterator_types_attr = ir.ArrayAttr.get(
        [ir.Attribute.parse("#linalg.iterator_type<parallel>") for _ in range(rank)]
    )

    generic_op = linalg.GenericOp(
        [output_tensor.type],
        [input_tensor],
        [output_tensor],
        indexing_maps_attr,
        iterator_types_attr,
    )
    region = generic_op.regions[0]
    block = region.blocks.append(input_mlir_elem_type, output_storage_elem_type)
    in_elem = block.arguments[0]
    with ir.InsertionPoint(block):
        in_elem = convert(in_elem, output_mlir_elem_type)
        result = math_fn(in_elem)
        result = convert(result, output_mlir_elem_type)
        result = lowering_utilities.value_to_storage(output_array_type.dtype, result)
        linalg.yield_([result])

    map_result = next(iter(cast(Any, generic_op.operation.results)))
    result_memref = tensor_to_memref(map_result)
    memref.copy(result_memref, output_memref)
    builder.store_var(target, output_memref)


# Complex trig ufuncs with direct dialect support
@lower(np.sin, types.Complex, types.Array)
def np_sin_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """sin(complex, out_array) - compute complex sin and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.sin, "np.sin")


@lower(np.cos, types.Complex, types.Array)
def np_cos_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """cos(complex, out_array) - compute complex cos and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.cos, "np.cos")


@lower(np.tan, types.Complex, types.Array)
def np_tan_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """tan(complex, out_array) - compute complex tan and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.tan, "np.tan")


@lower(np.sqrt, types.Complex, types.Array)
def np_sqrt_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """sqrt(complex, out_array) - compute complex sqrt and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.sqrt, "np.sqrt")


@lower(np.exp, types.Complex, types.Array)
def np_exp_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """exp(complex, out_array) - compute complex exp and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.exp, "np.exp")


@lower(np.log, types.Complex, types.Array)
def np_log_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """log(complex, out_array) - compute complex log and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.log, "np.log")


@lower(np.tanh, types.Complex, types.Array)
def np_tanh_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """tanh(complex, out_array) - compute complex tanh and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, complex_dialect.tanh, "np.tanh")


# Complex inverse trig ufuncs (using formula implementations)
@lower(np.arccos, types.Complex, types.Array)
def np_arccos_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """arccos(complex, out_array) - compute complex arccos and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_acos, "np.arccos")


@lower(np.arcsin, types.Complex, types.Array)
def np_arcsin_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """arcsin(complex, out_array) - compute complex arcsin and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_asin, "np.arcsin")


@lower(np.arctan, types.Complex, types.Array)
def np_arctan_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """arctan(complex, out_array) - compute complex arctan and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_atan, "np.arctan")


# Complex hyperbolic ufuncs
@lower(np.sinh, types.Complex, types.Array)
def np_sinh_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """sinh(complex, out_array) - compute complex sinh and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_sinh, "np.sinh")


@lower(np.cosh, types.Complex, types.Array)
def np_cosh_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """cosh(complex, out_array) - compute complex cosh and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_cosh, "np.cosh")


@lower(np.arcsinh, types.Complex, types.Array)
def np_arcsinh_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """arcsinh(complex, out_array) - compute complex arcsinh and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_asinh, "np.arcsinh")


@lower(np.arccosh, types.Complex, types.Array)
def np_arccosh_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """arccosh(complex, out_array) - compute complex arccosh and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_acosh, "np.arccosh")


@lower(np.arctanh, types.Complex, types.Array)
def np_arctanh_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """arctanh(complex, out_array) - compute complex arctanh and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_atanh, "np.arctanh")


@lower(np.log2, types.Complex, types.Array)
def np_log2_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """log2(complex, out_array) - compute complex log2 and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_log2, "np.log2")


@lower(np.log10, types.Complex, types.Array)
def np_log10_complex_scalar_to_array_cg(builder, target, args, kwargs):
    """log10(complex, out_array) - compute complex log10 and store in output array"""
    create_complex_scalar_to_output_array(builder, target, args, _complex_log10, "np.log10")


@lower(numpy_empty_like_nd, types.Array, types.DTypeSpec, types.TypeRef)
@lower(numpy_empty_like_nd, types.Array, types.NoneType, types.TypeRef)
def numpy_empty_like_nd_lower(builder, target, args, kwargs):
    """Lower numpy_empty_like_nd intrinsic to memref.alloc with same shape as prototype."""
    prototype_arg = args[0]
    target_type = builder.get_numba_type(target.name)

    prototype = builder.load_var(prototype_arg)
    prototype_type = prototype.type

    if isinstance(prototype_type, ir.MemRefType):
        ndim = prototype_type.rank
    elif isinstance(prototype_type, ir.RankedTensorType):
        ndim = prototype_type.rank
    else:
        raise NotImplementedError(
            f"numpy_empty_like_nd: unsupported prototype type {prototype_type}"
        )

    # Extract shape from prototype
    shape_vals = []
    for i in range(ndim):
        dim = memref.dim(prototype, index_of(i))
        shape_vals.append(dim)

    # Get the storage element type from target
    element_type = builder.get_storage_type(target_type.dtype)

    # Create a simple contiguous memref type for allocation (no strided layout)
    # This produces a row-major/C-contiguous array
    dyn = ir.MemRefType.get_dynamic_size()
    alloca_memref_type = ir.MemRefType.get([dyn] * ndim, element_type)

    # Allocate memref with same shape using alloca in the alloca insertion point
    with builder.alloca_insertion_point():
        alloca_op = memref_dialect.AllocaOp(
            memref=alloca_memref_type,
            dynamicSizes=shape_vals,
            symbolOperands=[],
        )

    builder.store_var(target, alloca_op.memref)


@lower(_make_dtype_object, types.StringLiteral)
def make_dtype_object_cg(builder, target, args, kws):
    target_type = builder.get_numba_type(target.name)
    builder.store_var(target, builder._materialize_type_token(target_type))


@overload(np.dtype, typing_registry=typing_registry)
def numpy_dtype(dtype, align=False, copy=False):
    """Provide an implementation so that numpy.dtype function can be lowered."""
    if isinstance(dtype, (types.Literal, types.functions.NumberClass)):

        def imp(dtype, align=False, copy=False):
            return _make_dtype_object(dtype)

        return imp
    else:
        raise errors.NumbaTypeError("unknown dtype descriptor: {}".format(dtype))


@lower(np.nditer, types.Any)
def make_array_nditer(builder, target, args, kws):
    if len(args) != 1:
        raise NotImplementedError(
            f"np.nditer expects exactly one positional argument, got {len(args)}"
        )
    operand_var = args[0]
    operand_ty = builder.get_numba_type(operand_var.name)

    if isinstance(operand_ty, types.BaseTuple):
        member_tys = list(operand_ty)
        if not all(isinstance(t, types.Array) for t in member_tys):
            raise NotImplementedError(
                "np.nditer over tuples currently requires all members to be Arrays; "
                f"got {member_tys}"
            )
        if len({t.ndim for t in member_tys}) > 1:
            raise NotImplementedError(
                "np.nditer with broadcasting across different-rank inputs is not "
                "yet supported in the MLIR backend"
            )
        tup = builder.load_var(operand_var)
        if isinstance(tup, (tuple, list)):
            array_values = list(tup)
        else:
            raise NotImplementedError(
                f"Expected tuple storage for np.nditer tuple input, got {type(tup)}"
            )
        ndim = member_tys[0].ndim
    elif isinstance(operand_ty, types.Array):
        array_values = [builder.load_var(operand_var)]
        ndim = operand_ty.ndim
    else:
        raise NotImplementedError(f"np.nditer not implemented for {operand_ty!r}")

    iter_obj = NdIterIterObject(builder, array_values, ndim)
    builder.store_var(target, iter_obj)


@lower("number.item", types.Boolean)
@lower("number.item", types.Number)
def number_item_impl(builder, target, args, kws):
    """
    The no-op .item() method on booleans and numbers.
    """
    builder.store_var(target, builder.load_var(args[0]))
