# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""This module provides the unsafe things for targets/numbers.py"""

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.extending import intrinsic


# These intrinsics' codegen closures built llvmlite bitcast/cttz/ctlz IR. They
# are filtered out on the MLIR path, so the codegen is a shared tombstone; the
# typing (returned signatures and validation) is retained.
def _dead_codegen(context, builder, signature, args):
    raise NotImplementedError(
        "this intrinsic's vendored llvmlite codegen is not used on the MLIR path"
    )


@intrinsic
def viewer(tyctx, val, viewty):
    """Bitcast a scalar 'val' to the given type 'viewty'."""
    retty = viewty.dtype
    sig = retty(val, viewty)
    return sig, _dead_codegen


@intrinsic
def trailing_zeros(typeingctx, src):
    """Counts trailing zeros in the binary representation of an integer."""
    if not isinstance(src, types.Integer):
        msg = f"trailing_zeros is only defined for integers, but value passed was '{src}'."
        raise errors.NumbaTypeError(msg)

    return src(src), _dead_codegen


@intrinsic
def leading_zeros(typeingctx, src):
    """Counts leading zeros in the binary representation of an integer."""
    if not isinstance(src, types.Integer):
        msg = f"leading_zeros is only defined for integers, but value passed was '{src}'."
        raise errors.NumbaTypeError(msg)

    return src(src), _dead_codegen
