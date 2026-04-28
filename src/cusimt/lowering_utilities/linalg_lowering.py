# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt._mlir import ir
from cusimt._mlir.dialects import arith, memref, linalg, tensor, shape, gpu
from cusimt.mlir.dialect_exts import scf
from cusimt._mlir.extras import types as T
from cusimt.lowering_utilities import (
    memref_to_tensor,
    tensor_to_memref,
    set_error_code_if_zero,
    convert,
    mul,
    add,
)
from numba import types
from cusimt.mlir_lowering import KERNEL_ERROR_CODES


def lower_matmul(mlir_lower, target, target_type, args):
    """
    linalg_matmul(array: types.Array, array: types.Array) -> array: types.Array
    """
    lhs = args[0]
    rhs = args[1]

    lhs_type = mlir_lower.get_numba_type(lhs.name)
    rhs_type = mlir_lower.get_numba_type(rhs.name)

    assert lhs_type.ndim == 2, f"linalg.matmul requires 2D arrays, got {lhs_type.ndim}D for lhs"
    assert rhs_type.ndim == 2, f"linalg.matmul requires 2D arrays, got {rhs_type.ndim}D for rhs"

    lhs_dims = []
    rhs_dims = []
    for i in range(2):
        index = arith.constant(result=ir.IndexType.get(), value=i)
        lhs_dim = memref.dim(source=mlir_lower.load_var(lhs), index=index)
        rhs_dim = memref.dim(source=mlir_lower.load_var(rhs), index=index)
        lhs_dims.append(lhs_dim)
        rhs_dims.append(rhs_dim)

    # Verify matrix dimensions are compatible for multiplication
    # For A(m,n) * B(n,p), we need the inner dimensions to match
    dims_match = arith.cmpi(
        predicate=arith.CmpIPredicate.eq,
        lhs=lhs_dims[1],  # A's columns
        rhs=rhs_dims[0],  # B's rows
    )
    error_memref = mlir_lower._get_or_create_error_global()
    if error_memref is not None:
        dims_mismatch = arith.xori(dims_match, arith.constant(result=T.i(1), value=1))
        with scf.if_ctx_manager(dims_mismatch):
            set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
            scf.yield_([])

    result_dims = [lhs_dims[0], rhs_dims[1]]

    memref_type = mlir_lower.get_mlir_type(target_type)
    result_memref = memref.alloc(
        memref=memref_type,
        dynamic_sizes=result_dims,
        symbol_operands=[],
    )

    lhs_tensor = memref_to_tensor(mlir_lower.load_var(lhs))
    rhs_tensor = memref_to_tensor(mlir_lower.load_var(rhs))
    result_tensor = memref_to_tensor(result_memref)

    # Initialize result tensor with zeros
    element_type = mlir_lower.get_mlir_type(target_type.dtype)
    zero_value = 0.0 if isinstance(target_type.dtype, types.Float) else 0
    zero = arith.constant(result=element_type, value=zero_value)
    result_tensor = linalg.fill(zero, outs=[result_tensor])

    matmul_result = linalg.matmul(lhs_tensor, rhs_tensor, outs=[result_tensor])

    mlir_lower.store_var(target, tensor_to_memref(matmul_result))


def get_array_dims(mlir_lower, arg, arg_type):
    if isinstance(arg_type, types.Array):
        dims = []
        for i in range(arg_type.ndim):
            index = arith.constant(result=ir.IndexType.get(), value=i)
            dim = memref.dim(source=mlir_lower.load_var(arg), index=index)
            dims.append(dim)
        return dims, arg_type.ndim
    else:
        return [], 0


def broadcast_tensor(mlir_lower, input_tensor, input_dims, target_ndim, target_shape, target_type):
    """Tensor broadcasting using nested SCF loops"""
    input_ndim = len(input_dims)

    # Create empty output tensor
    empty = tensor.empty(
        sizes=target_shape, element_type=mlir_lower.get_mlir_type(target_type.dtype)
    )

    def build_nested_tensor_broadcast_loops(tensor_arg, indices_so_far, dim_idx):
        if dim_idx == target_ndim:
            # Base case - compute input indices and extract/insert value
            input_indices = []

            # Compute input indices based on broadcasting rules
            for in_dim_idx in range(input_ndim):
                # Map input dimension to output dimension (right-aligned)
                out_dim_idx = target_ndim - input_ndim + in_dim_idx

                # Check if input dimension is 1 (broadcasting dimension)
                input_dim_size = input_dims[in_dim_idx]
                one = arith.constant(result=ir.IndexType.get(), value=1)
                zero = arith.constant(result=ir.IndexType.get(), value=0)

                # If input dimension is 1, use index 0; otherwise use output index
                is_broadcast_dim = arith.cmpi(
                    predicate=arith.CmpIPredicate.eq, lhs=input_dim_size, rhs=one
                )
                input_index = arith.select(
                    condition=is_broadcast_dim,
                    true_value=zero,
                    false_value=indices_so_far[out_dim_idx],
                )
                input_indices.append(input_index)

            # Extract value from input tensor
            extracted_value = tensor.extract(tensor=input_tensor, indices=input_indices)
            converted_value = convert(extracted_value, mlir_lower.get_mlir_type(target_type.dtype))

            # Insert into output tensor
            return tensor.insert(scalar=converted_value, dest=tensor_arg, indices=indices_so_far)

        # Create loop for current dimension
        lb = arith.constant(result=ir.IndexType.get(), value=0)
        ub = target_shape[dim_idx]
        step = arith.constant(result=ir.IndexType.get(), value=1)

        loop = scf.ForOp(lower_bound=lb, upper_bound=ub, step=step, iter_args=[tensor_arg])

        with ir.InsertionPoint(loop.body):
            iv = loop.induction_variable
            current_tensor = loop.inner_iter_args[0]
            new_indices = indices_so_far + [iv]

            # Recurse to next dimension
            updated_tensor = build_nested_tensor_broadcast_loops(
                current_tensor, new_indices, dim_idx + 1
            )
            scf.yield_([updated_tensor])

        return loop.results[0]

    return build_nested_tensor_broadcast_loops(empty, [], 0)


def broadcast_if_needed(
    mlir_lower,
    tensor_arg,
    tensor_dims,
    tensor_ndim,
    tensor_shape,
    result_ndim,
    broadcast_shape,
    broadcast_dims,
    target_type,
):
    if tensor_ndim == result_ndim and target_type.dtype == tensor_arg.type.element_type:
        broadcast_op = scf.IfOp(
            shape.shape_eq([tensor_shape, broadcast_shape]),
            results_=[tensor_arg.type],
            hasElse=True,
        )
        with ir.InsertionPoint(broadcast_op.then_block):
            scf.yield_([tensor_arg])
        with ir.InsertionPoint(broadcast_op.else_block):
            scf.yield_(
                [
                    broadcast_tensor(
                        mlir_lower,
                        tensor_arg,
                        tensor_dims,
                        result_ndim,
                        broadcast_dims,
                        target_type,
                    )
                ]
            )
        return broadcast_op.results[0]
    else:
        return broadcast_tensor(
            mlir_lower,
            tensor_arg,
            tensor_dims,
            result_ndim,
            broadcast_dims,
            target_type,
        )


def lower_np_binop(mlir_lower, target, target_type, args, linalg_op):
    """
    Lower numpy binary operations (add, sub, mul, div) with broadcasting support.

    Args:
        mlir_lower: The MLIR lowering instance
        target: Target variable for the result
        target_type: Numba type of the target
        args: List of argument variables
        intrinsic_op: Intrinsic operation to be generated
    """
    lhs_broadcast = None
    rhs_broadcast = None
    broadcast_dims = None

    lhs_type = mlir_lower.get_numba_type(args[0].name)
    rhs_type = mlir_lower.get_numba_type(args[1].name)

    if isinstance(lhs_type, types.Number):
        # Scalar + Array case
        rhs_dims, rhs_ndim = get_array_dims(mlir_lower, args[1], rhs_type)
        scalar_value = mlir_lower.load_var(args[0])
        empty = tensor.empty(
            sizes=rhs_dims, element_type=mlir_lower.get_mlir_type(target_type.dtype)
        )
        lhs_broadcast = linalg.fill(scalar_value, outs=[empty])
        if rhs_type.dtype != target_type.dtype:
            type_converted_memref = memref.alloc(
                memref=mlir_lower.get_mlir_type(target_type),
                dynamic_sizes=rhs_dims,
                symbol_operands=[],
            )
            rhs_broadcast = linalg.copy(
                memref_to_tensor(mlir_lower.load_var(args[1])),
                outs=[memref_to_tensor(type_converted_memref)],
            )
        else:
            rhs_broadcast = memref_to_tensor(mlir_lower.load_var(args[1]))
        broadcast_dims = rhs_dims
    elif isinstance(rhs_type, types.Number):
        # Array + Scalar case
        lhs_dims, lhs_ndim = get_array_dims(mlir_lower, args[0], lhs_type)
        scalar_value = mlir_lower.load_var(args[1])
        empty = tensor.empty(
            sizes=lhs_dims, element_type=mlir_lower.get_mlir_type(target_type.dtype)
        )
        rhs_broadcast = linalg.fill(scalar_value, outs=[empty])
        if lhs_type.dtype != target_type.dtype:
            type_converted_memref = memref.alloc(
                memref=mlir_lower.get_mlir_type(target_type),
                dynamic_sizes=lhs_dims,
                symbol_operands=[],
            )
            lhs_broadcast = linalg.copy(
                memref_to_tensor(mlir_lower.load_var(args[0])),
                outs=[memref_to_tensor(type_converted_memref)],
            )
        else:
            lhs_broadcast = memref_to_tensor(mlir_lower.load_var(args[0]))
        broadcast_dims = lhs_dims
    else:
        # Array + Array case with broadcasting
        lhs_dims, lhs_ndim = get_array_dims(mlir_lower, args[0], lhs_type)
        rhs_dims, rhs_ndim = get_array_dims(mlir_lower, args[1], rhs_type)
        result_ndim = max(lhs_ndim, rhs_ndim)

        lhs_tensor = memref_to_tensor(mlir_lower.load_var(args[0]))
        rhs_tensor = memref_to_tensor(mlir_lower.load_var(args[1]))

        lhs_shape = shape.shape_of(lhs_tensor)
        rhs_shape = shape.shape_of(rhs_tensor)

        is_broadcastable = shape.is_broadcastable([lhs_shape, rhs_shape])

        error_memref = mlir_lower._get_or_create_error_global()
        if error_memref is not None:
            not_broadcastable = arith.xori(is_broadcastable, arith.constant(result=T.i(1), value=1))
            with scf.if_ctx_manager(not_broadcastable):
                set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
                scf.yield_([])

        broadcast_shape = shape.broadcast(shapes=[rhs_shape, lhs_shape], result=lhs_shape.type)

        broadcast_dims = []
        for i in range(result_ndim):
            dim = shape.get_extent(
                broadcast_shape, arith.constant(result=ir.IndexType.get(), value=i)
            )
            broadcast_dims.append(dim)

        lhs_broadcast = broadcast_if_needed(
            mlir_lower,
            lhs_tensor,
            lhs_dims,
            lhs_ndim,
            lhs_shape,
            result_ndim,
            broadcast_shape,
            broadcast_dims,
            target_type,
        )
        rhs_broadcast = broadcast_if_needed(
            mlir_lower,
            rhs_tensor,
            rhs_dims,
            rhs_ndim,
            rhs_shape,
            result_ndim,
            broadcast_shape,
            broadcast_dims,
            target_type,
        )

    element_type = mlir_lower.get_mlir_type(target_type.dtype)
    empty = tensor.empty(sizes=broadcast_dims, element_type=element_type)

    result_tensor = linalg_op(lhs_broadcast, rhs_broadcast, outs=[empty])

    mlir_lower.store_var(target, tensor_to_memref(result_tensor))


def lower_transpose(mlir_lower, target, array):
    input_array = mlir_lower.load_var(array)
    input_type = mlir_lower.get_numba_type(array.name)

    input_tensor = memref_to_tensor(input_array)

    dims = []
    for i in range(input_type.ndim):
        dim = memref.dim(
            source=input_array, index=arith.constant(result=ir.IndexType.get(), value=i)
        )
        dims.append(dim)

    empty_tensor = tensor.empty(
        sizes=list(reversed(dims)),
        element_type=mlir_lower.get_mlir_type(input_type.dtype),
    )
    transpose_op = linalg.transpose(
        input=input_tensor,
        outs=[empty_tensor],
        permutation=list(reversed(range(input_type.ndim))),
    )
    mlir_lower.store_var(target, tensor_to_memref(transpose_op.results[0]))


def lower_linalg_dot(mlir_lower, target, target_type, args):
    lhs_type = mlir_lower.get_numba_type(args[0].name)
    rhs_type = mlir_lower.get_numba_type(args[1].name)
    if (lhs_type.ndim == 0 and rhs_type.ndim == 0) or (lhs_type.ndim == 1 and rhs_type.ndim == 1):
        lower_linalg_dot_vector(mlir_lower, target, target_type, args)
    elif (lhs_type.ndim == 0 and rhs_type.ndim == 1) or (lhs_type.ndim == 1 and rhs_type.ndim == 0):
        lower_linalg_scalar_array_dot(mlir_lower, target, target_type, args)
    elif (lhs_type.ndim == 0 and rhs_type.ndim > 1) or (lhs_type.ndim > 1 and rhs_type.ndim == 0):
        lower_np_binop(mlir_lower, target, target_type, args, linalg.mul)
    elif lhs_type.ndim >= 2 and rhs_type.ndim == 1:
        lower_linalg_nd_1d_dot(mlir_lower, target, target_type, args)
    elif lhs_type.ndim >= 1 and rhs_type.ndim >= 2:
        lower_linalg_nd_md_dot(mlir_lower, target, target_type, args)
    else:
        raise NotImplementedError(f"Not support np.dot between {lhs_type} and {rhs_type}.")


def lower_linalg_scalar_array_dot(mlir_lower, target, target_type, args):
    lhs_ndim = mlir_lower.get_numba_type(args[0].name).ndim

    if lhs_ndim == 0:
        scalar_type = mlir_lower.get_numba_type(args[0].name)
        scalar_value = tensor.extract(
            tensor=memref_to_tensor(mlir_lower.load_var(args[0])), indices=[]
        )
        array_tensor = memref_to_tensor(mlir_lower.load_var(args[1]))
        n = memref.dim(
            source=mlir_lower.load_var(args[1]),
            index=arith.constant(result=ir.IndexType.get(), value=0),
        )
    else:
        scalar_type = mlir_lower.get_numba_type(args[1].name)
        scalar_value = tensor.extract(
            tensor=memref_to_tensor(mlir_lower.load_var(args[1])), indices=[]
        )
        array_tensor = memref_to_tensor(mlir_lower.load_var(args[0]))
        n = memref.dim(
            source=mlir_lower.load_var(args[0]),
            index=arith.constant(result=ir.IndexType.get(), value=0),
        )

    # Create result tensor with same shape as array
    element_type = mlir_lower.get_mlir_type(target_type.dtype)
    empty_tensor = tensor.empty(sizes=[n], element_type=element_type)

    index_type = ir.IndexType.get()
    target_shape = shape.shape_of(array_tensor)
    index = arith.constant(result=index_type, value=0)
    target_extracts = [tensor.extract(target_shape, [index])]

    # Type convert arguments if needed
    if scalar_type != target_type.dtype:
        scalar_value = convert(scalar_value, mlir_lower.get_mlir_type(target_type.dtype))

    if target_type.dtype != array_tensor.type.element_type:
        array_tensor = linalg.copy(array_tensor, outs=[empty_tensor])

    @tensor.generate(empty_tensor.type, dynamic_extents=target_extracts)
    def generate_scalar_array_dot(i1: index_type):
        return mul(scalar_value, tensor.extract(array_tensor, [i1]))

    mlir_lower.store_var(target, tensor_to_memref(generate_scalar_array_dot))


def lower_linalg_nd_1d_dot(mlir_lower, target, target_type, args):
    lhs_ndim = mlir_lower.get_numba_type(args[0].name).ndim
    rhs_ndim = mlir_lower.get_numba_type(args[1].name).ndim

    # Ensure lhs is N-dimensional and rhs is 1D
    assert lhs_ndim > 1 and rhs_ndim == 1, (
        f"linalg_nd_1d expects lhs.ndim > 1 and rhs.ndim == 1, got lhs.ndim={lhs_ndim}, rhs.ndim={rhs_ndim}"
    )

    # Get dimensions
    lhs_dims = []
    for i in range(lhs_ndim):
        index = arith.constant(result=ir.IndexType.get(), value=i)
        dim = memref.dim(source=mlir_lower.load_var(args[0]), index=index)
        lhs_dims.append(dim)

    rhs_dim = memref.dim(
        source=mlir_lower.load_var(args[1]),
        index=arith.constant(result=ir.IndexType.get(), value=0),
    )

    # Verify that the last dimension of lhs matches the dimension of rhs
    dims_match = arith.cmpi(predicate=arith.CmpIPredicate.eq, lhs=lhs_dims[-1], rhs=rhs_dim)
    error_memref = mlir_lower._get_or_create_error_global()
    if error_memref is not None:
        dims_mismatch = arith.xori(dims_match, arith.constant(result=T.i(1), value=1))
        with scf.if_ctx_manager(dims_mismatch):
            set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
            scf.yield_([])

    # Result dimensions are all but the last dimension of lhs
    result_dims = lhs_dims[:-1]

    # Create result memref
    element_type = mlir_lower.get_mlir_type(target_type.dtype)
    result_memref = memref.alloc(
        memref=mlir_lower.get_mlir_type(target_type),
        dynamic_sizes=result_dims,
        symbol_operands=[],
    )

    # Convert inputs to tensors
    lhs_tensor = memref_to_tensor(mlir_lower.load_var(args[0]))
    rhs_tensor = memref_to_tensor(mlir_lower.load_var(args[1]))

    # Create nested loops to compute dot product
    def build_nested_dot_loops(result_memref, indices_so_far, dim_idx):
        if dim_idx == len(result_dims):
            # Base case - compute dot product over the last dimension
            last_dim_size = lhs_dims[-1]

            # Create parallel loop to iterate over the last dimension with reduction
            init_value = arith.constant(
                result=element_type,
                value=0.0 if isinstance(element_type, ir.FloatType) else 0,
            )

            parallel_op = scf.ParallelOp(
                lowerBound=[arith.constant(result=ir.IndexType.get(), value=0)],
                upperBound=[last_dim_size],
                step=[arith.constant(result=ir.IndexType.get(), value=1)],
                initVals=[init_value],
                results_=[init_value.type],
            )
            parallel_block = ir.Block.create_at_start(parallel_op.regions[0], [ir.IndexType.get()])

            with ir.InsertionPoint(parallel_block):
                k = parallel_block.arguments[0]

                # Extract elements from lhs and rhs
                lhs_indices = indices_so_far + [k]
                rhs_indices = [k]

                lhs_element = tensor.extract(tensor=lhs_tensor, indices=lhs_indices)
                rhs_element = tensor.extract(tensor=rhs_tensor, indices=rhs_indices)

                # Multiply
                product = mul(lhs_element, rhs_element)

                # Reduce by adding
                reduce_op = scf.reduce([product], num_reductions=1)
                reduction_block = ir.Block.create_at_start(
                    reduce_op.reductions[0], [product.type, product.type]
                )
                with ir.InsertionPoint(reduction_block):
                    lhs_arg = reduction_block.arguments[0]
                    rhs_arg = reduction_block.arguments[1]
                    sum_result = add(lhs_arg, rhs_arg)
                    scf.reduce_return(sum_result)

            # Store the result from the parallel operation
            final_sum = parallel_op.results[0]
            memref.store(value=final_sum, memref=result_memref, indices=indices_so_far)
            return

        # Create loop for current dimension
        lb = arith.constant(result=ir.IndexType.get(), value=0)
        ub = result_dims[dim_idx]
        step = arith.constant(result=ir.IndexType.get(), value=1)

        loop = scf.ForOp(lower_bound=lb, upper_bound=ub, step=step, iter_args=[])

        with ir.InsertionPoint(loop.body):
            iv = loop.induction_variable
            new_indices = indices_so_far + [iv]

            # Recurse to next dimension
            build_nested_dot_loops(result_memref, new_indices, dim_idx + 1)
            scf.yield_([])

    # Start the nested loop construction
    build_nested_dot_loops(result_memref, [], 0)
    mlir_lower.store_var(target, result_memref)


def lower_linalg_nd_md_dot(mlir_lower, target, target_type, args):
    lhs_ndim = mlir_lower.get_numba_type(args[0].name).ndim
    rhs_ndim = mlir_lower.get_numba_type(args[1].name).ndim

    # Get dimensions
    lhs_dims = []
    for i in range(lhs_ndim):
        index = arith.constant(result=ir.IndexType.get(), value=i)
        dim = memref.dim(source=mlir_lower.load_var(args[0]), index=index)
        lhs_dims.append(dim)

    rhs_dims = []
    for i in range(rhs_ndim):
        index = arith.constant(result=ir.IndexType.get(), value=i)
        dim = memref.dim(source=mlir_lower.load_var(args[1]), index=index)
        rhs_dims.append(dim)

    # Verify that the last dimension of lhs matches the second last dimension of rhs
    dims_match = arith.cmpi(predicate=arith.CmpIPredicate.eq, lhs=lhs_dims[-1], rhs=rhs_dims[-2])
    error_memref = mlir_lower._get_or_create_error_global()
    if error_memref is not None:
        dims_mismatch = arith.xori(dims_match, arith.constant(result=T.i(1), value=1))
        with scf.if_ctx_manager(dims_mismatch):
            set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
            scf.yield_([])

    # Result dimensions: all but the last dimension of lhs + all but the second last dimension of rhs
    result_dims = lhs_dims[:-1] + rhs_dims[:-2] + [rhs_dims[-1]]

    # Create result memref
    element_type = mlir_lower.get_mlir_type(target_type.dtype)
    result_memref = memref.alloc(
        memref=mlir_lower.get_mlir_type(target_type),
        dynamic_sizes=result_dims,
        symbol_operands=[],
    )

    # Initialize result with zeros
    zero = arith.constant(
        result=element_type, value=0.0 if isinstance(element_type, ir.FloatType) else 0
    )
    result_tensor = linalg.fill(zero, outs=[memref_to_tensor(result_memref)])
    result_memref = tensor_to_memref(result_tensor)

    # Convert inputs to tensors
    lhs_tensor = memref_to_tensor(mlir_lower.load_var(args[0]))
    rhs_tensor = memref_to_tensor(mlir_lower.load_var(args[1]))

    # Create nested loops to compute dot product
    def build_nested_dot_loops(result_memref, indices_so_far, dim_idx):
        if dim_idx == len(result_dims):
            # Base case - compute dot product over the reduction dimension
            reduction_dim_size = lhs_dims[-1]  # same as rhs_dims[-2]

            # Create parallel loop to iterate over the last dimension with reduction
            init_value = arith.constant(
                result=element_type,
                value=0.0 if isinstance(element_type, ir.FloatType) else 0,
            )

            parallel_op = scf.ParallelOp(
                lowerBound=[arith.constant(result=ir.IndexType.get(), value=0)],
                upperBound=[reduction_dim_size],
                step=[arith.constant(result=ir.IndexType.get(), value=1)],
                initVals=[init_value],
                results_=[init_value.type],
            )
            parallel_block = ir.Block.create_at_start(parallel_op.regions[0], [ir.IndexType.get()])

            with ir.InsertionPoint(parallel_block):
                k = parallel_block.arguments[0]

                lhs_indices = indices_so_far[: len(lhs_dims) - 1] + [k]
                rhs_indices = (
                    indices_so_far[len(lhs_dims) - 1 : len(lhs_dims) - 1 + len(rhs_dims) - 2]
                    + [k]
                    + [indices_so_far[-1]]
                )

                lhs_element = tensor.extract(tensor=lhs_tensor, indices=lhs_indices)
                rhs_element = tensor.extract(tensor=rhs_tensor, indices=rhs_indices)

                product = mul(lhs_element, rhs_element)

                reduce_op = scf.reduce([product], num_reductions=1)
                reduction_block = ir.Block.create_at_start(
                    reduce_op.reductions[0], [product.type, product.type]
                )
                with ir.InsertionPoint(reduction_block):
                    lhs_arg = reduction_block.arguments[0]
                    rhs_arg = reduction_block.arguments[1]
                    sum_result = add(lhs_arg, rhs_arg)
                    scf.reduce_return(sum_result)

            # Store the result from the parallel operation
            final_sum = parallel_op.results[0]
            memref.store(value=final_sum, memref=result_memref, indices=indices_so_far)
            return

        # Create loop for current dimension
        lb = arith.constant(result=ir.IndexType.get(), value=0)
        ub = result_dims[dim_idx]
        step = arith.constant(result=ir.IndexType.get(), value=1)

        loop = scf.ForOp(lower_bound=lb, upper_bound=ub, step=step, iter_args=[])

        with ir.InsertionPoint(loop.body):
            iv = loop.induction_variable
            new_indices = indices_so_far + [iv]

            # Recurse to next dimension
            build_nested_dot_loops(result_memref, new_indices, dim_idx + 1)
            scf.yield_([])

    # Start the nested loop construction
    build_nested_dot_loops(result_memref, [], 0)
    mlir_lower.store_var(target, result_memref)


def lower_linalg_dot_vector(mlir_lower, target, target_type, args):
    lhs_ndim = mlir_lower.get_numba_type(args[0].name).ndim
    rhs_ndim = mlir_lower.get_numba_type(args[1].name).ndim

    c0 = arith.constant(result=ir.IndexType.get(), value=0)
    c1 = arith.constant(result=ir.IndexType.get(), value=1)

    lhs_tensor = memref_to_tensor(mlir_lower.load_var(args[0]))
    rhs_tensor = memref_to_tensor(mlir_lower.load_var(args[1]))

    if lhs_ndim == 0 and rhs_ndim == 0:
        lhs_value = tensor.extract(tensor=lhs_tensor, indices=[])
        rhs_value = tensor.extract(tensor=rhs_tensor, indices=[])
        result = mul(lhs_value, rhs_value)
        mlir_lower.store_var(target, result)
    elif lhs_ndim == 1 and rhs_ndim == 1:
        n = memref.dim(source=mlir_lower.load_var(args[0]), index=c0)

        rhs_dim = memref.dim(source=mlir_lower.load_var(args[1]), index=c0)
        dims_match = arith.cmpi(predicate=arith.CmpIPredicate.eq, lhs=n, rhs=rhs_dim)
        error_memref = mlir_lower._get_or_create_error_global()
        if error_memref is not None:
            dims_mismatch = arith.xori(dims_match, arith.constant(result=T.i(1), value=1))
            with scf.if_ctx_manager(dims_mismatch):
                set_error_code_if_zero(error_memref, KERNEL_ERROR_CODES[ValueError])
                scf.yield_([])

        element_type = mlir_lower.get_mlir_type(target_type)
        empty = tensor.empty(sizes=[c1, c1], element_type=element_type)

        zero = arith.constant(
            result=element_type,
            value=0 if isinstance(element_type, ir.IntegerType) else 0.0,
        )
        result_tensor = linalg.fill(zero, outs=[empty])

        # TODO: use linalg.dot instead of scf.ForOp. Currently, this exposes a race condition.
        loop = scf.ForOp(lower_bound=c0, upper_bound=n, step=c1, iter_args=[result_tensor])

        with ir.InsertionPoint(loop.body):
            i = loop.induction_variable
            current_result = loop.inner_iter_args[0]

            lhs_element = tensor.extract(tensor=lhs_tensor, indices=[i])
            rhs_element = tensor.extract(tensor=rhs_tensor, indices=[i])

            product = mul(lhs_element, rhs_element)
            current_value = tensor.extract(tensor=current_result, indices=[c0, c0])
            new_value = add(current_value, product)

            updated_result = tensor.insert(scalar=new_value, dest=current_result, indices=[c0, c0])
            scf.yield_([updated_result])

        result_tensor = loop.results[0]

        scalar_value = tensor.extract(tensor=result_tensor, indices=[c0, c0])
        mlir_lower.store_var(target, scalar_value)
