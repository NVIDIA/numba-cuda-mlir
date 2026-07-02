# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.extending import overload, overload_method
from numba_cuda_mlir.numba_cuda.typing import signature
from numba_cuda_mlir.numba_cuda.extending import intrinsic
from numba_cuda_mlir.numba_cuda.types.ext_types import (
    grid_group,
    GridGroup as GridGroupClass,
)


class GridGroup:
    """A cooperative group representing the entire grid"""

    def sync() -> None:
        """Synchronize this grid group"""


def this_grid() -> GridGroup:
    """Get the current grid group."""
    return GridGroup()
