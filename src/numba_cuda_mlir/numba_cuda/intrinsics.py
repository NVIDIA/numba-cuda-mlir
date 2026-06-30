# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir import numba_cuda as cuda
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core.errors import (
    RequireLiteralValue,
    TypingError,
    NumbaTypeError,
)
from numba_cuda_mlir.numba_cuda.typing import signature
from numba_cuda_mlir.numba_cuda.extending import overload_attribute, overload_method
from numba_cuda_mlir.numba_cuda.extending import intrinsic


# Each @intrinsic below provides the typing (the returned signature plus literal
# / type validation) that the MLIR type-inference pass needs. The codegen used
# to build NVVM IR with llvmlite, but on the MLIR path these grid / syncthreads
# / warp shuffle / warp vote operations are lowered by
# numba_cuda_mlir.lowering.cuda, so the codegen closures are never invoked (a
# numba_cuda_mlir.numba_cuda intrinsic builder is filtered out by MLIRLower).
# The codegen is therefore a single shared tombstone.
def _dead_codegen(context, builder, sig, args):
    raise NotImplementedError(
        "this intrinsic is lowered by numba_cuda_mlir.lowering.cuda on the MLIR "
        "path; the vendored llvmlite codegen is never invoked"
    )


# -------------------------------------------------------------------------------
# Grid functions


def _type_grid_function(ndim):
    val = ndim.literal_value
    if val == 1:
        restype = types.int64
    elif val in (2, 3):
        restype = types.UniTuple(types.int64, val)
    else:
        raise ValueError("argument can only be 1, 2, 3")

    return signature(restype, types.int32)


@intrinsic
def grid(typingctx, ndim):
    """grid(ndim)

    Return the absolute position of the current thread in the entire grid of
    blocks.  *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """

    if not isinstance(ndim, types.IntegerLiteral):
        raise RequireLiteralValue(ndim)

    sig = _type_grid_function(ndim)

    return sig, _dead_codegen


@intrinsic
def gridsize(typingctx, ndim):
    """gridsize(ndim)

    Return the absolute size (or shape) in threads of the entire grid of
    blocks. *ndim* should correspond to the number of dimensions declared when
    instantiating the kernel. If *ndim* is 1, a single integer is returned.
    If *ndim* is 2 or 3, a tuple of the given number of integers is returned.

    Computation of the first integer is as follows::

        cuda.blockDim.x * cuda.gridDim.x

    and is similar for the other two indices, but using the ``y`` and ``z``
    attributes.
    """

    if not isinstance(ndim, types.IntegerLiteral):
        raise RequireLiteralValue(ndim)

    sig = _type_grid_function(ndim)

    return sig, _dead_codegen


@intrinsic
def _warpsize(typingctx):
    sig = signature(types.int32)

    return sig, _dead_codegen


@overload_attribute(types.Module(cuda), "warpsize", target="cuda")
def cuda_warpsize(mod):
    """
    The size of a warp. All architectures implemented to date have a warp size
    of 32.
    """

    def get(mod):
        return _warpsize()

    return get


# -------------------------------------------------------------------------------
# syncthreads


@intrinsic
def syncthreads(typingctx):
    """
    Synchronize all threads in the same thread block.  This function implements
    the same pattern as barriers in traditional multi-threaded programming: this
    function waits until all threads in the block call it, at which point it
    returns control to all its callers.
    """
    sig = signature(types.none)

    return sig, _dead_codegen


def _syncthreads_predicate(typingctx, predicate, fname):
    if not isinstance(predicate, types.Integer):
        return None

    sig = signature(types.i4, types.i4)

    return sig, _dead_codegen


@intrinsic
def syncthreads_count(typingctx, predicate):
    """
    syncthreads_count(predicate)

    An extension to numba.cuda.syncthreads where the return value is a count
    of the threads where predicate is true.
    """
    fname = "llvm.nvvm.barrier0.popc"
    return _syncthreads_predicate(typingctx, predicate, fname)


@intrinsic
def syncthreads_and(typingctx, predicate):
    """
    syncthreads_and(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for all threads or 0 otherwise.
    """
    fname = "llvm.nvvm.barrier0.and"
    return _syncthreads_predicate(typingctx, predicate, fname)


@intrinsic
def syncthreads_or(typingctx, predicate):
    """
    syncthreads_or(predicate)

    An extension to numba.cuda.syncthreads where 1 is returned if predicate is
    true for any thread or 0 otherwise.
    """
    fname = "llvm.nvvm.barrier0.or"
    return _syncthreads_predicate(typingctx, predicate, fname)


@overload_method(types.Integer, "bit_count", target="cuda")
def integer_bit_count(i):
    return lambda i: cuda.popc(i)


# -------------------------------------------------------------------------------
# Warp shuffle functions
#
# References:
#
# - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-shuffle-functions
# - https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#data-movement


@intrinsic
def shfl_sync(typingctx, mask, value, src_lane):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``src_lane``. If this is outside the warp, then the given value is
    returned.
    """
    membermask_type = mask
    mode_value = 0
    a_type = value
    b_type = src_lane
    c_value = 0x1F
    return shfl_sync_intrinsic(typingctx, membermask_type, mode_value, a_type, b_type, c_value)


@intrinsic
def shfl_up_sync(typingctx, mask, value, delta):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``(laneid - delta)``. If this is outside the warp, then the given value is
    returned.
    """
    membermask_type = mask
    mode_value = 1
    a_type = value
    b_type = delta
    c_value = 0
    return shfl_sync_intrinsic(typingctx, membermask_type, mode_value, a_type, b_type, c_value)


@intrinsic
def shfl_down_sync(typingctx, mask, value, delta):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``(laneid + delta)``. If this is outside the warp, then the given value is
    returned.
    """
    membermask_type = mask
    mode_value = 2
    a_type = value
    b_type = delta
    c_value = 0x1F
    return shfl_sync_intrinsic(typingctx, membermask_type, mode_value, a_type, b_type, c_value)


@intrinsic
def shfl_xor_sync(typingctx, mask, value, lane_mask):
    """
    Shuffles ``value`` across the masked warp and returns the value from
    ``(laneid ^ lane_mask)``.
    """
    membermask_type = mask
    mode_value = 3
    a_type = value
    b_type = lane_mask
    c_value = 0x1F
    return shfl_sync_intrinsic(typingctx, membermask_type, mode_value, a_type, b_type, c_value)


def shfl_sync_intrinsic(
    typingctx,
    membermask_type,
    mode_value,
    a_type,
    b_type,
    c_value,
):
    if a_type not in (types.i4, types.i8, types.f4, types.f8):
        raise TypingError("shfl_sync only supports 32- and 64-bit ints and floats")

    sig = signature(a_type, membermask_type, a_type, b_type)

    return sig, _dead_codegen


# -------------------------------------------------------------------------------
# Warp vote functions
#
# References:
#
# - https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-vote-functions
# - https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html?highlight=data%2520movement#vote


@intrinsic
def all_sync(typingctx, mask_type, predicate_type):
    """
    If for all threads in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
    mode_value = 0
    # Validate the mask / predicate types (raises on unsupported types).
    vote_sync_intrinsic(typingctx, mask_type, mode_value, predicate_type)

    sig_outer = signature(types.b1, mask_type, predicate_type)
    return sig_outer, _dead_codegen


@intrinsic
def any_sync(typingctx, mask_type, predicate_type):
    """
    If for any thread in the masked warp the predicate is true, then
    a non-zero value is returned, otherwise 0 is returned.
    """
    mode_value = 1
    vote_sync_intrinsic(typingctx, mask_type, mode_value, predicate_type)

    sig_outer = signature(types.b1, mask_type, predicate_type)
    return sig_outer, _dead_codegen


@intrinsic
def eq_sync(typingctx, mask_type, predicate_type):
    """
    If for all threads in the masked warp the boolean predicate is the same,
    then a non-zero value is returned, otherwise 0 is returned.
    """
    mode_value = 2
    vote_sync_intrinsic(typingctx, mask_type, mode_value, predicate_type)

    sig_outer = signature(types.b1, mask_type, predicate_type)
    return sig_outer, _dead_codegen


@intrinsic
def ballot_sync(typingctx, mask_type, predicate_type):
    """
    Returns a mask of all threads in the warp whose predicate is true,
    and are within the given mask.
    """
    mode_value = 3
    vote_sync_intrinsic(typingctx, mask_type, mode_value, predicate_type)

    sig_outer = signature(types.i4, mask_type, predicate_type)
    return sig_outer, _dead_codegen


def vote_sync_intrinsic(typingctx, mask_type, mode_value, predicate_type):
    # Validate mode value
    if mode_value not in (0, 1, 2, 3):
        raise ValueError("Mode must be 0 (all), 1 (any), 2 (eq), or 3 (ballot)")

    if types.unliteral(mask_type) not in types.integer_domain:
        raise NumbaTypeError(f"Mask type must be an integer. Got {mask_type}")
    predicate_types = types.integer_domain | {types.boolean}

    if types.unliteral(predicate_type) not in predicate_types:
        raise NumbaTypeError(f"Predicate must be an integer or boolean. Got {predicate_type}")

    sig = signature(types.Tuple((types.i4, types.b1)), mask_type, predicate_type)

    return sig, _dead_codegen
