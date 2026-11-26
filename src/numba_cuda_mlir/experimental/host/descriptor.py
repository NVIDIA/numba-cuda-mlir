# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from functools import cached_property

from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext
from numba_cuda_mlir.numba_cuda.core.target_extension import CPU, target_registry


class TargetDescriptor:
    @cached_property
    def typing_context(self):
        return MLIRTypingContext()

    @cached_property
    def target_context(self):
        return MLIRTargetContext(self.typing_context, target="numba_cuda_mlir_host")


class Target(CPU):
    pass


target_registry["numba_cuda_mlir_host"] = Target
target = TargetDescriptor()
