# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.mlir_lowering_registry import MLIRLoweringRegistry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.extending import (
    intrinsic,
    _Intrinsic,
    overload,
    overload_attribute,
)
from numba_cuda_mlir.descriptor import MLIRTypingContext

registry = MLIRLoweringRegistry()
lower = registry.lower


def mlir_convert(ctor):
    """
    Create an mlir builder that converts a value indiscriminately to the given type.
    Note that the construction of the type is delayed until the builder itself
    is called because we need to be within an MLIR context.
    """

    def converter(mlir_lower, target, args, kwargs):
        assert not kwargs, "mlir_convert does not accept any keyword arguments"
        mlir_lower.store_var(target, mlir_lower.mlir_convert(mlir_lower.load_var(args[0]), ctor()))

    return converter


@intrinsic
def i32(typingctx: MLIRTypingContext, value):
    return types.int32(value), mlir_convert(T.i32)


@intrinsic
def i64(typingctx: MLIRTypingContext, value):
    return types.int64(value), mlir_convert(T.i64)


@intrinsic
def f32(typingctx: MLIRTypingContext, value):
    return types.float32(value), mlir_convert(T.f32)


@intrinsic
def f64(typingctx: MLIRTypingContext, value):
    return types.float64(value), mlir_convert(T.f64)


lower(bool, types.Any)(mlir_convert(T.bool))


@intrinsic
def bool(typingctx, value):
    return types.bool(value), mlir_convert(T.bool)
