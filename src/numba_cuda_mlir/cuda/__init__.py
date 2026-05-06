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
_EXPERIMENTAL_IMPORTED = False


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


def __getattr__(name):
    """Lazy load modules to avoid circular import issues."""
    if name in ("intrin", "tensor_map", "experimental"):
        import importlib

        module = importlib.import_module(f"numba_cuda_mlir.cuda.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def inline_ptx(format_string: str, *args) -> None:
    """
    Add PTX code directly into the kernel.
    The format string and arguments mirror the CUDA C++ inline assembly syntax.
    """


def vectorize(*args, **kwargs):
    raise NotImplementedError("vectorize is not implemented")


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
