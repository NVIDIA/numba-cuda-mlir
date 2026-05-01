# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# CUDA-callables that are implemented in MLIR

import sys
import inspect
import numba_cuda_mlir.runtime as rt
from numba_cuda_mlir.compiler import declare_mlir_library
from numba_cuda_mlir.errors import (
    MultipleIntrinsicFunctionsError,
    UnsupportedIntrinsicTypeError,
)
from numba_cuda_mlir.lowering_utilities.type_conversions import to_mlir_type
from numba_cuda_mlir.lowering_utilities.discover_functions import discover_functions
from numba_cuda_mlir._mlir.ir import UnitAttr


def _import_runtime():
    for lib in rt._get_all_libraries():
        lib = getattr(rt, lib)
        for func in lib.functions.keys():
            setattr(sys.modules[__name__], func, getattr(lib, func))


_import_runtime()

breakpoint = rt.util.breakpoint
nanosleep = rt.util.nanosleep


def _resolve_type_annotation(ty):
    if ty is None:
        return None  # void return type
    # Handle tuple return types (e.g., tuple[types.int32, types.int32])
    if hasattr(ty, "__origin__") and ty.__origin__ is tuple:
        return [to_mlir_type(arg) for arg in ty.__args__]
    try:
        return to_mlir_type(ty)
    except (TypeError, NotImplementedError):
        pass
    if callable(ty):
        return ty()
    raise TypeError(f"Cannot resolve type annotation: {ty}")


def _intrinsic_from_source(source: str):
    try:
        functions = discover_functions(source)
    except (TypeError, NotImplementedError) as e:
        raise UnsupportedIntrinsicTypeError(str(e)) from e

    if len(functions) == 0:
        raise ValueError(
            "No functions found in MLIR source. "
            "Ensure the source contains a func.func definition."
        )

    if len(functions) > 1:
        raise MultipleIntrinsicFunctionsError(list(functions.keys()))

    func_name = next(iter(functions.keys()))
    lib = declare_mlir_library(source)
    return getattr(lib, func_name)


def _intrinsic_from_func(func):
    from numba_cuda_mlir.mlir.context import mlir_mod_ctx
    from numba_cuda_mlir.mlir.dialect_exts.func import func as mlir_func

    sig = inspect.signature(func)

    original_annotations = {}
    for name, param in sig.parameters.items():
        ty = param.annotation
        if ty is inspect.Parameter.empty:
            raise TypeError(f"Parameter '{name}' must have a type annotation")
        original_annotations[name] = ty

    ret_ty = sig.return_annotation
    if ret_ty is not inspect.Signature.empty and ret_ty is not None:
        original_annotations["return"] = ret_ty

    with mlir_mod_ctx() as ctx:
        new_annotations = {}
        for name, ty in original_annotations.items():
            resolved = _resolve_type_annotation(ty)
            if resolved is not None:
                new_annotations[name] = resolved

        func.__annotations__ = new_annotations

        decorator = mlir_func(sym_visibility="private")
        mlir_fn = decorator(func)
        mlir_fn.func_attrs["alwaysinline"] = UnitAttr.get()
        mlir_fn.emit()

        # Capture the module string before exiting the context
        module_str = str(ctx.module)

    # Restore original annotations to avoid retaining references to MLIR
    # objects after the context is destroyed (use-after-free)
    func.__annotations__ = original_annotations

    lib = declare_mlir_library(module_str)
    return getattr(lib, func.__name__)


def define(func_or_source):
    """
    Create an intrinsic function callable from JIT kernels.

    Can be used in two ways:

    1. As a decorator for Python functions using MLIR Python bindings:

        @cuda.intrin.define
        def my_mul(a: types.int32, b: types.int32) -> types.int32:
            return arith.muli(a, b)

    2. With inline MLIR source code:

        my_add = cuda.intrin.define('''
            func.func private @my_add(%a: i32, %b: i32) -> i32 attributes {always_inline} {
                %r = arith.addi %a, %b : i32
                return %r : i32
            }
        ''')

    For the decorator form, type annotations can be:
    - numba_cuda_mlir types: `types.int32`, `types.float32`, etc. (recommended)
    - MLIR type callables: `T.f32`, `llvm.PointerType.get` (for advanced types)

    Args:
        func_or_source: Either a Python function with type annotations, or
                        MLIR source code containing exactly one func.func.

    Returns:
        An ExternMLIRLibraryFunction that can be called from JIT kernels.

    Raises:
        TypeError: If used as decorator and a parameter lacks a type annotation.
        MultipleIntrinsicFunctionsError: If MLIR source contains multiple functions.
        UnsupportedIntrinsicTypeError: If a type cannot be converted.
        ValueError: If MLIR source contains no functions.
    """
    if isinstance(func_or_source, str):
        return _intrinsic_from_source(func_or_source)
    else:
        return _intrinsic_from_func(func_or_source)
