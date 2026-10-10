# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda.experimental import intrin
from numba_cuda_mlir import types
from cudamlir.dialects import nvvm
from cudamlir.extras import types as T


@intrin.define
def elect_sync() -> types.boolean:
    return nvvm.elect_sync()
