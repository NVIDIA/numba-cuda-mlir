# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Memory management module for numba_cuda_mlir.

This module provides NRT (Numba Runtime) support including:
- Compiling NRT CUDA sources to LTOIR for linking with numba_cuda_mlir kernels
- Runtime system (rtsys) for managing device-side memory allocator state
- Configuration for NRT enablement and statistics
"""

from numba_cuda_mlir.memory_management.nrt import (
    get_include,
    compile_nrt_ltoir,
    needs_nrt_linking,
    NRT_FUNCTIONS,
)

from numba_cuda_mlir.memory_management.config import (
    is_nrt_enabled,
    is_nrt_stats_enabled,
)

from numba_cuda_mlir.memory_management.rtsys import (
    rtsys,
    _nrt_mstats,
)

__all__ = [
    # NRT LTOIR compilation
    "get_include",
    "compile_nrt_ltoir",
    "needs_nrt_linking",
    "NRT_FUNCTIONS",
    # Configuration
    "is_nrt_enabled",
    "is_nrt_stats_enabled",
    # Runtime system
    "rtsys",
    "_nrt_mstats",
]
