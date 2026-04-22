# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
class UFuncRegistry:
    def __init__(self, name: str | None = None):
        self.registry = {}

    def __str__(self):
        return f"UFuncRegistry(registry={self.registry})"

    def __repr__(self):
        return str(self)

    def register(self, ufunc):
        def decorator(lowering):
            self.registry[ufunc] = lowering
            return lowering

        return decorator

    def get(self, ufunc):
        return self.registry.get(ufunc)
