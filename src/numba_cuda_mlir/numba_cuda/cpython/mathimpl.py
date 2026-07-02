# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""
Provide math calls that uses intrinsics or libc math functions.

The bitcast/comparison helpers and the @lower codegen for the math module
(isnan/isinf/isfinite, copysign, frexp/ldexp, atan2/hypot, radians/degrees,
pow/nextafter/gcd, ...) built llvmlite IR and were filtered out on the MLIR
path (math is lowered by numba_cuda_mlir.lowering.math). They are gone. What
remains is pure Python: the float min/max constants (still used by
cmathimpl) and the ``_unsigned`` typing overload used by integer helpers.
"""

import numpy as np

from numba_cuda_mlir.numba_cuda.core.imputils import Registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.extending import overload

registry = Registry("mathimpl")

# Helpers, shared with cmathimpl.
_NP_FLT_FINFO = np.finfo(np.dtype("float32"))
FLT_MAX = _NP_FLT_FINFO.max
FLT_MIN = _NP_FLT_FINFO.tiny

_NP_DBL_FINFO = np.finfo(np.dtype("float64"))
DBL_MAX = _NP_DBL_FINFO.max
DBL_MIN = _NP_DBL_FINFO.tiny

FLOAT_ABS_MASK = 0x7FFFFFFF
FLOAT_SIGN_MASK = 0x80000000
DOUBLE_ABS_MASK = 0x7FFFFFFFFFFFFFFF
DOUBLE_SIGN_MASK = 0x8000000000000000


def _unsigned(T):
    """Convert integer to unsigned integer of equivalent width."""
    pass


@overload(_unsigned)
def _unsigned_impl(T):
    if T in types.unsigned_domain:
        return lambda T: T
    elif T in types.signed_domain:
        newT = getattr(types, "uint{}".format(T.bitwidth))
        return lambda T: newT(T)
