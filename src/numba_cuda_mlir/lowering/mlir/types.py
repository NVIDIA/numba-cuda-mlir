# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.lowering_registry import LoweringRegistry
from numba_cuda_mlir.numba_cuda import types

registry = LoweringRegistry()
lower = registry.lower


lower(bool, types.Any)


def converter(mlir_lower, target, args, kwargs):
    assert not kwargs, "bool_convert does not accept any keyword arguments"

    arg = mlir_lower.load_var(args[0])
    converted = mlir_lower.mlir_convert(arg, T.bool())
    mlir_lower.store_var(target, converted)
