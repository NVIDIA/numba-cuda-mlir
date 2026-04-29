# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda import types
import numpy as np
from cusimt.numba_cuda.extending import intrinsic


@intrinsic
def memref_alloc(typingctx, shape, np_dtype=np.float64):
    restype = types.Array(
        dtype=np_dtype.dtype,
        ndim=shape.count if isinstance(shape, types.UniTuple) else 1,
        layout="C",
    )
    sig = restype(shape)

    return sig, None


@intrinsic
def memref_dim(typingctx, array, index):
    restype = types.int64
    sig = restype(array, index)

    return sig, None
