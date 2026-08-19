# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device functions in a separate file for multi-file lineinfo tests."""

from numba_cuda_mlir import cuda


@cuda.jit(device=True)
def helper_scale_offset(x):
    y = x * 2.0
    y = y + 1.0
    return y
