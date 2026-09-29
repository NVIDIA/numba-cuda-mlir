# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.core.imputils import Registry


class LoweringRegistry(Registry):
    """
    Registry for MLIR-based lowering implementations.

    Each lowering module should create its own instance of this registry:
        registry = LoweringRegistry()
        lower = registry.lower
    """

    def __init__(self):
        super().__init__(name="numba_cuda_mlir")
