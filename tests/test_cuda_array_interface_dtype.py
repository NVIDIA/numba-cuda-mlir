# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from numba_cuda_mlir.numba_cuda.api import _dtype_from_cuda_array_interface


class _Owner:
    def __init__(self, dtype):
        self.dtype = dtype


def test_cuda_array_interface_uses_supported_owner_dtype_for_void_typestr():
    owner_dtype = np.dtype([("x", np.float16), ("y", np.float16)])
    dtype = _dtype_from_cuda_array_interface({"typestr": "|V4"}, _Owner(owner_dtype))
    assert dtype == owner_dtype


def test_cuda_array_interface_unsupported_owner_dtype_falls_back_to_typestr():
    owner_dtype = np.dtype([("bad", object)])
    dtype = _dtype_from_cuda_array_interface({"typestr": "|V8"}, _Owner(owner_dtype))
    assert dtype == np.dtype("|V8")
