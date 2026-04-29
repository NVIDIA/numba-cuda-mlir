# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.numba_cuda import types

def memref_alloc(
    shape: types.UniTuple | types.Integer, np_dtype: types.DType | type
) -> types.Array: ...
