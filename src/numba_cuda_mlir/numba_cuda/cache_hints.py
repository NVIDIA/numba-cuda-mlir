# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.extending import intrinsic, overload
from numba_cuda_mlir.numba_cuda.core.errors import NumbaTypeError

# Docs references:
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld
# https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#load-functions-using-cache-hints
#
# The @intrinsic ``codegen`` closures here used to build the ld/st.global PTX
# inline asm with llvmlite. On the MLIR path the cache-hint operations are
# lowered by numba_cuda_mlir.lowering.cuda (register_cache_hint_lowerings,
# @lower against the stub functions below), so these closures are never invoked
# (a numba_cuda_mlir.numba_cuda intrinsic builder is filtered out, and an array
# argument is rejected outright by MLIRLower). The @intrinsic typing (the
# returned signature) and argument validation are retained; the codegen is a
# tombstone.


def _dead_codegen(context, builder, sig, args):
    raise NotImplementedError(
        "cache-hint codegen is provided by numba_cuda_mlir.lowering.cuda on the "
        "MLIR path; this vendored llvmlite codegen is never invoked"
    )


def ldca(array, i):
    """Generate a `ld.global.ca` instruction for element `i` of an array."""


def ldcg(array, i):
    """Generate a `ld.global.cg` instruction for element `i` of an array."""


def ldcs(array, i):
    """Generate a `ld.global.cs` instruction for element `i` of an array."""


def ldlu(array, i):
    """Generate a `ld.global.lu` instruction for element `i` of an array."""


def ldcv(array, i):
    """Generate a `ld.global.cv` instruction for element `i` of an array."""


def stcg(array, i, value):
    """Generate a `st.global.cg` instruction for element `i` of an array."""


def stcs(array, i, value):
    """Generate a `st.global.cs` instruction for element `i` of an array."""


def stwb(array, i, value):
    """Generate a `st.global.wb` instruction for element `i` of an array."""


def stwt(array, i, value):
    """Generate a `st.global.wt` instruction for element `i` of an array."""


# See
# https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#restricted-use-of-sub-word-sizes
# for background on the choice of "r" for 8-bit operands - there is
# no constraint for 8-bit operands, but the operand for loads and
# stores is permitted to be greater than 8 bits.
CONSTRAINT_MAP = {1: "b", 8: "r", 16: "h", 32: "r", 64: "l", 128: "q"}


def _validate_arguments(instruction, array, index):
    is_array = isinstance(array, types.Array)
    is_pointer = isinstance(array, types.CPointer)
    if not (is_array or is_pointer):
        msg = f"{instruction} operates on arrays or pointers. Got type {array}"
        raise NumbaTypeError(msg)

    valid_index = False

    if isinstance(index, types.Integer):
        if is_array and array.ndim != 1:
            # for pointers, any integer index is valid
            msg = f"Expected {array.ndim} indices, got a scalar"
            raise NumbaTypeError(msg)
        valid_index = True

    if isinstance(index, types.UniTuple):
        if is_pointer:
            msg = f"Pointers only support scalar indexing, got tuple of {index.count}"
            raise NumbaTypeError(msg)

        if index.count != array.ndim:
            msg = f"Expected {array.ndim} indices, got {index.count}"
            raise NumbaTypeError(msg)

        if isinstance(index.dtype, types.Integer):
            valid_index = True

    if not valid_index:
        raise NumbaTypeError(f"{index} is not a valid index")


def _validate_bitwidth(instruction, array):
    dtype = array.dtype

    if not isinstance(dtype, (types.Integer, types.Float)):
        msg = f"{instruction} requires array of integer or float type, got {dtype}"
        raise NumbaTypeError(msg)

    bitwidth = dtype.bitwidth
    if bitwidth not in CONSTRAINT_MAP:
        valid_widths = sorted(CONSTRAINT_MAP.keys())
        msg = (
            f"{instruction} requires array dtype with bitwidth "
            f"in {valid_widths}, got bitwidth {bitwidth}"
        )
        raise NumbaTypeError(msg)


def ld_cache_operator(operator):
    @intrinsic
    def impl(typingctx, array, index):
        _validate_arguments(f"ld{operator}", array, index)
        _validate_bitwidth(f"ld{operator}", array)

        signature = array.dtype(array, index)

        return signature, _dead_codegen

    return impl


ldca_intrinsic = ld_cache_operator("ca")
ldcg_intrinsic = ld_cache_operator("cg")
ldcs_intrinsic = ld_cache_operator("cs")
ldlu_intrinsic = ld_cache_operator("lu")
ldcv_intrinsic = ld_cache_operator("cv")


def st_cache_operator(operator):
    @intrinsic
    def impl(typingctx, array, index, value):
        _validate_arguments(f"st{operator}", array, index)
        _validate_bitwidth(f"st{operator}", array)

        signature = types.void(array, index, value)

        return signature, _dead_codegen

    return impl


stcg_intrinsic = st_cache_operator("cg")
stcs_intrinsic = st_cache_operator("cs")
stwb_intrinsic = st_cache_operator("wb")
stwt_intrinsic = st_cache_operator("wt")


@overload(ldca, target="cuda")
def ol_ldca(array, i):
    def impl(array, i):
        return ldca_intrinsic(array, i)

    return impl


@overload(ldcg, target="cuda")
def ol_ldcg(array, i):
    def impl(array, i):
        return ldcg_intrinsic(array, i)

    return impl


@overload(ldcs, target="cuda")
def ol_ldcs(array, i):
    def impl(array, i):
        return ldcs_intrinsic(array, i)

    return impl


@overload(ldlu, target="cuda")
def ol_ldlu(array, i):
    def impl(array, i):
        return ldlu_intrinsic(array, i)

    return impl


@overload(ldcv, target="cuda")
def ol_ldcv(array, i):
    def impl(array, i):
        return ldcv_intrinsic(array, i)

    return impl


@overload(stcg, target="cuda")
def ol_stcg(array, i, value):
    def impl(array, i, value):
        return stcg_intrinsic(array, i, value)

    return impl


@overload(stcs, target="cuda")
def ol_stcs(array, i, value):
    def impl(array, i, value):
        return stcs_intrinsic(array, i, value)

    return impl


@overload(stwb, target="cuda")
def ol_stwb(array, i, value):
    def impl(array, i, value):
        return stwb_intrinsic(array, i, value)

    return impl


@overload(stwt, target="cuda")
def ol_stwt(array, i, value):
    def impl(array, i, value):
        return stwt_intrinsic(array, i, value)

    return impl
