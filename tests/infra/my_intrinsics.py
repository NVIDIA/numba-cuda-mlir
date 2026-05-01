# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir.dialects import nvvm
from numba_cuda_mlir._mlir.extras import types as T


@cuda.intrin.define
def elect_sync() -> types.boolean:
    return nvvm.elect_sync()
