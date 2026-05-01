# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Thunk that overrides everything in numba.cuda and overrides
everything in numba.cuda that numba_cuda_mlir _also_ supports.
"""

import importlib
import sys

# Base API comes from `numba.cuda` (which may be redirected to numba-cuda-mlir).
from numba_cuda_mlir.numba_cuda import *  # noqa: F403
from numba_cuda_mlir.numba_cuda.cudadrv.devicearray import (
    DeviceNDArray,  # ty:ignore[unresolved-import]
)  # noqa: F401
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll  # noqa: F401,E402

# numba-cuda-mlir overrides/extensions
from numba_cuda_mlir.cuda.lazy_api import *

HAS_NUMBA = False


class _ConstevalContextManager:
    """Context manager for consteval blocks - transformed away by AST passes."""

    def __enter__(self):
        raise RuntimeError(
            "consteval() block was not transformed at compile time.\n"
            "This usually means experimental_ast_transforms is not enabled.\n"
            "Add experimental_ast_transforms=True to your @jit decorator:\n"
            "    @cuda.jit(experimental_ast_transforms=True)"
        )

    def __exit__(self, *args):
        pass


def consteval(value=None):
    """
    Evaluate an expression at compile time, or mark a block for compile-time execution.

    Usage as expression (returns the compile-time value):
        x = consteval(GLOBAL_CONST * 2)

    Usage as context manager (executes block at compile time):
        with consteval():
            config = load_config()
            N = config["block_size"]
        # Use consteval(N) to access N at runtime

    Requires experimental_ast_transforms=True in the @jit decorator.
    """
    if value is None:
        return _ConstevalContextManager()
    raise RuntimeError(
        "consteval() was not transformed at compile time.\n"
        "This usually means experimental_ast_transforms is not enabled.\n"
        "Add experimental_ast_transforms=True to your @jit decorator:\n"
        "    @cuda.jit(experimental_ast_transforms=True)"
    )


# Submodules (must be modules, not class stubs) so that `numba.cuda.shared` and
# `cuda.shared.array` resolve to the same callables we register typing/lowering
# for.  Assign from importlib's return value so the star import from
# numba_cuda_mlir.numba_cuda cannot leave stub attributes on this package.
const = importlib.import_module("numba_cuda_mlir.cuda.const")
local = importlib.import_module("numba_cuda_mlir.cuda.local")
shared = importlib.import_module("numba_cuda_mlir.cuda.shared")
fp16 = importlib.import_module("numba_cuda_mlir.cuda.fp16")
libdevice = importlib.import_module("numba_cuda_mlir.cuda.libdevice")
libdevicefuncs = importlib.import_module("numba_cuda_mlir.cuda.libdevicefuncs")
vector = importlib.import_module("numba_cuda_mlir.cuda.vector")
vector_types = importlib.import_module("numba_cuda_mlir.cuda.vector_types")

# Expose vector type constructors (float32x4, int32x2, etc.) at module level
from .vector_types import *  # noqa: F401,F403

local_array = local.array  # noqa: F401
shared_array = shared.array  # noqa: F401


def local_array_from(iterable, dtype):
    """
    Create a local array from a generator expression or iterable.

    This function is transformed at AST level to:
        arr = local_array(len(iterable_source), dtype=dtype)
        for i, val in enumerate(iterable):
            arr[i] = val

    Example:
        arr = cuda.local_array_from((i+1 for i in indices), dtype=np.float32)
    """
    pass


def __getattr__(name):
    """Lazy load modules to avoid circular import issues."""
    if name in ("intrin", "tcgen05_descriptors", "tensor_map"):
        import importlib

        module = importlib.import_module(f"numba_cuda_mlir.cuda.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def vectorize(*args, **kwargs):
    raise NotImplementedError("vectorize is not implemented")


def inline_ptx(format_string: str, *args) -> None:
    """
    Add PTX code directly into the kernel.
    The format string and arguments mirror the CUDA C++ inline assembly syntax.

    ```
    TODO(ajm): examples
    ```
    """


def struct(*args, **kwargs):
    from numba_cuda_mlir.host import struct

    return struct(*args, **kwargs)


def union(*args, **kwargs):
    from numba_cuda_mlir.host import union

    return union(*args, **kwargs)


def clz(x):
    """Count leading zeros. For a 32-bit value, returns 0-32. For 64-bit, returns 0-64."""
    pass


def ffs(x):
    """Find first set bit. Returns 1-indexed position of LSB, or 0 if input is 0."""
    pass


def brev(x):
    """Reverse the bits of x."""
    pass


def popc(x):
    """Count the number of set bits in x."""
    pass


def selp(cond, a, b):
    """Select based on predicate: returns a if cond is true, else b."""
    pass


# Special registers - these are accessed as module attributes, not functions
warpsize = 32
laneid = None  # Placeholder - actual value comes from NVVM intrinsic at runtime


# Cache hint load instructions
def ldca(array, i):
    """Generate a `ld.global.ca` instruction for element `i` of an array."""
    pass


def ldcg(array, i):
    """Generate a `ld.global.cg` instruction for element `i` of an array."""
    pass


def ldcs(array, i):
    """Generate a `ld.global.cs` instruction for element `i` of an array."""
    pass


def ldlu(array, i):
    """Generate a `ld.global.lu` instruction for element `i` of an array."""
    pass


def ldcv(array, i):
    """Generate a `ld.global.cv` instruction for element `i` of an array."""
    pass


# Cache hint store instructions
def stcg(array, i, value):
    """Generate a `st.global.cg` instruction for element `i` of an array."""
    pass


def stcs(array, i, value):
    """Generate a `st.global.cs` instruction for element `i` of an array."""
    pass


def stwb(array, i, value):
    """Generate a `st.global.wb` instruction for element `i` of an array."""
    pass


def stwt(array, i, value):
    """Generate a `st.global.wt` instruction for element `i` of an array."""
    pass


class _CurrentTargetOptionsMarker:
    """Marker class for current_target_options() - replaced during consteval."""

    def __repr__(self):
        return "numba_cuda_mlir.current_target_options()"


def current_target_options() -> dict:
    """
    Return the current kernel's target options as a dictionary.

    This function can only be used inside consteval() expressions.
    At consteval time, it returns the targetoptions dict passed to @jit().

    Example:
        @cuda.jit(chip='sm_90', experimental_ast_transforms=True)
        def kernel(arr):
            chip = consteval(numba_cuda_mlir.current_target_options()['chip'])  # 'sm_90'
    """
    return _CurrentTargetOptionsMarker()
