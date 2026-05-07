# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Experimental Numba-CUDA-MLIR features.

When a kernel's defining module has imported from this package, ``@cuda.jit``
auto-enables ``experimental_ast_transforms=True`` for that kernel (unless the
user explicitly passes ``experimental_ast_transforms=False``).
"""

import importlib as _importlib

from numba_cuda_mlir.cuda import inline_ptx  # noqa: F401
from numba_cuda_mlir.cuda.experimental.struct import struct  # noqa: F401
from numba_cuda_mlir.cuda.experimental.union import union  # noqa: F401


def __getattr__(name):
    _lazy = {
        "intrin": "numba_cuda_mlir.cuda.intrin",
        "tcgen05_descriptors": "numba_cuda_mlir.cuda.experimental.tcgen05_descriptors",
    }
    if name in _lazy:
        return _importlib.import_module(_lazy[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class _ConstevalContextManager:
    """Context manager for consteval blocks -- transformed away by AST passes."""

    def __enter__(self):
        raise RuntimeError(
            "consteval() block was not transformed at compile time.\n"
            "Ensure experimental_ast_transforms is enabled:\n"
            "    from numba_cuda_mlir.cuda.experimental import consteval\n"
            "    @cuda.jit(experimental_ast_transforms=True)"
        )

    def __exit__(self, *args):
        pass


def consteval(value=None):
    """
    Evaluate an expression at compile time, or mark a block for compile-time
    execution.

    Usage as expression (returns the compile-time value)::

        x = consteval(GLOBAL_CONST * 2)

    Usage as context manager (executes block at compile time)::

        with consteval():
            config = load_config()
            N = config["block_size"]
    """
    if value is None:
        return _ConstevalContextManager()
    raise RuntimeError(
        "consteval() was not transformed at compile time.\n"
        "Ensure experimental_ast_transforms is enabled:\n"
        "    from numba_cuda_mlir.cuda.experimental import consteval\n"
        "    @cuda.jit(experimental_ast_transforms=True)"
    )


class _CurrentTargetOptionsMarker:
    def __repr__(self):
        return "current_target_options()"


def current_target_options() -> dict:
    """
    Return the current kernel's target options as a dictionary.

    Only usable inside ``consteval()`` expressions::

        @cuda.jit(chip="sm_90")
        def kernel(arr):
            chip = consteval(current_target_options()["chip"])
    """
    return _CurrentTargetOptionsMarker()


def local_array_from(iterable, dtype):
    """
    Create a local array from a generator expression or iterable.

    Transformed at AST level into ``cuda.local_array`` + assignment loop::

        arr = local_array_from((i + 1 for i in indices), dtype=np.float32)
    """
    pass


__all__ = [
    "consteval",
    "current_target_options",
    "inline_ptx",
    "intrin",
    "local_array_from",
    "struct",
    "tcgen05_descriptors",
    "union",
]
