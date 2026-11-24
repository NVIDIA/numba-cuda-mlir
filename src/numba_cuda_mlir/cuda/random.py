# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import warnings

from numba_cuda_mlir.numba_cuda.random import *  # noqa: F401,F403

warnings.warn(
    "numba_cuda_mlir.cuda.random is deprecated and will be removed in v0.8.0. "
    "Use the nvmath-python cuRAND device APIs instead: "
    "https://docs.nvidia.com/cuda/nvmath-python/latest/device-apis/curand.html",
    FutureWarning,
    stacklevel=2,
)
