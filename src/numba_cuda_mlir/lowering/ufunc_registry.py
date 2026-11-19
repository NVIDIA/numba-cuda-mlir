# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.lowering_registry import register_lowering


class UFuncRegistry:
    def __init__(self, name: str | None = None):
        self.registry = {}

    def __str__(self):
        return f"UFuncRegistry(registry={self.registry})"

    def __repr__(self):
        return str(self)

    def register(self, ufunc):
        def decorator(lowering):
            self.registry[ufunc] = register_lowering(lowering)
            return lowering

        return decorator

    def get(self, ufunc):
        return self.registry.get(ufunc)
