# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import wraps

from numba_cuda_mlir.numba_cuda.core.imputils import Registry


def _with_lowering_api(implementation, api):
    if api not in ("vars", "values"):
        raise ValueError(f"Unsupported lowering API: {api!r}")

    @wraps(implementation)
    def registered(*args, **kwargs):
        return implementation(*args, **kwargs)

    registered._numba_cuda_mlir_lowering_api = api
    return registered


class LoweringRegistry(Registry):
    """
    Registry for MLIR-based lowering implementations.

    Each lowering module should create its own instance of this registry:
        registry = LoweringRegistry()
        lower = registry.lower
    """

    def __init__(self):
        super().__init__(name="numba_cuda_mlir")

    def lower(self, func, *argtys, api="vars"):
        def decorate(implementation):
            registered = _with_lowering_api(implementation, api)
            super(LoweringRegistry, self).lower(func, *argtys)(registered)
            return implementation

        return decorate
