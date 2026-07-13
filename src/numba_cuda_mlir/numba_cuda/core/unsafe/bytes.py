# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
This file provides internal compiler utilities that support certain special
operations with bytes and workarounds for limitations enforced in userland.
"""

from numba_cuda_mlir.numba_cuda.extending import intrinsic
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda import cgutils


# grab_byte / grab_uint64_t built llvmlite load codegen; filtered out on the MLIR
# path, so the codegen is a shared tombstone (typing retained).
def _dead_codegen(context, builder, signature, args):
    raise NotImplementedError("this byte-access codegen is not used on the MLIR path")


@intrinsic
def grab_byte(typingctx, data, offset):
    # returns a byte at a given offset in data
    sig = types.uint8(types.voidptr, types.intp)
    return sig, _dead_codegen


@intrinsic
def grab_uint64_t(typingctx, data, offset):
    # returns a uint64_t at a given offset in data
    sig = types.uint64(types.voidptr, types.intp)
    return sig, _dead_codegen


@intrinsic
def memcpy_region(typingctx, dst, dst_offset, src, src_offset, nbytes, align):
    """Copy nbytes from *(src + src_offset) to *(dst + dst_offset)"""

    def codegen(context, builder, signature, args):
        [
            dst_val,
            dst_offset_val,
            src_val,
            src_offset_val,
            nbytes_val,
            align_val,
        ] = args
        src_ptr = builder.gep(src_val, [src_offset_val])
        dst_ptr = builder.gep(dst_val, [dst_offset_val])
        cgutils.raw_memcpy(builder, dst_ptr, src_ptr, nbytes_val, align_val)
        return context.get_dummy_value()

    sig = types.void(
        types.voidptr,
        types.intp,
        types.voidptr,
        types.intp,
        types.intp,
        types.intp,
    )
    return sig, codegen
