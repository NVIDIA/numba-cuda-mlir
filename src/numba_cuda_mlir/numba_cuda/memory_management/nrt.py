# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Removed numba-cuda NRT runtime.

The device-side NRT functions are emitted as MLIR LLVM dialect ops by
numba_cuda_mlir.memory_management.nrt_mlir, and the memsys that holds the
allocator statistics is managed by numba_cuda_mlir.memory_management.rtsys.
The NVRTC-compiled memsys.cu runtime this module used to provide has been
removed along with its <cuda/atomic> dependency on CCCL, so the entry points
below are stubs that fail loudly rather than silently allocating against a
runtime that no longer exists.
"""

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.extending import intrinsic, overload_classmethod
from numba_cuda_mlir.numba_cuda.typing.templates import signature

_REMOVED_MSG = (
    "the numba-cuda NRT runtime has been removed from numba-cuda-mlir; use "
    "numba_cuda_mlir.memory_management.rtsys instead"
)


@intrinsic
def intrin_alloc(typingctx, allocsize, align):
    """Intrinsic to call into the allocator for Array"""

    def codegen(context, builder, signature, args):
        raise NotImplementedError(_REMOVED_MSG)

    mip = types.MemInfoPointer(types.voidptr)  # return untyped pointer
    sig = signature(mip, allocsize, align)
    return sig, codegen


@overload_classmethod(types.Array, "_allocate", target="CUDA")
def _ol_array_allocate(cls, allocsize, align):
    """Implements a Numba-only CUDA-target classmethod on the array type."""

    def impl(cls, allocsize, align):
        return intrin_alloc(allocsize, align)

    return impl


class _RemovedRuntime:
    """Stand-in for the removed ``_Runtime`` singleton."""

    def __getattr__(self, name):
        raise NotImplementedError(f"{_REMOVED_MSG} (attempted rtsys.{name})")


rtsys = _RemovedRuntime()
