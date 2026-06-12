# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

# This module used to hold the @lower_builtin/@lower_cast/@lower_constant
# codegen for scalar number arithmetic, comparisons, bitwise ops, complex
# operations and numeric casts/constants. All of it built llvmlite IR and was
# installed into the target context, but it is filtered out on the MLIR path
# (numbers are lowered by numba_cuda_mlir.lowering.builtins). Only the scalar
# ``.view()`` typing overload remains; it is pure Python and lowers normally.

from numba_cuda_mlir.numba_cuda.core.imputils import Registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.extending import overload_method
from numba_cuda_mlir.numba_cuda.cpython.unsafe.numbers import viewer

registry = Registry("numbers")


# -------------------------------------------------------------------------------
# View


def scalar_view(scalar, viewty):
    """Typing for the np scalar 'view' method."""
    if isinstance(scalar, (types.Float, types.Integer)) and isinstance(
        viewty, types.abstract.DTypeSpec
    ):
        if scalar.bitwidth != viewty.dtype.bitwidth:
            raise errors.TypingError(
                "Changing the dtype of a 0d array is only supported if the itemsize is unchanged"
            )

        def impl(scalar, viewty):
            return viewer(scalar, viewty)

        return impl


overload_method(types.Float, "view")(scalar_view)
overload_method(types.Integer, "view")(scalar_view)
