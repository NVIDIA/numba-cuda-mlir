# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.compiler import declare_mlir_library
from functools import lru_cache
from numba_cuda_mlir.lowering_utilities import context
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.tools import get_gpu_compute_capability
from numba_cuda_mlir.lowering_utilities import link
from pathlib import Path
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.typing.externals import ExternMLIRLibrary
import sys

_FILE_ROOT_DIR = Path(__file__).parent


def _get_all_libraries():
    libs = ["nvvm", "atomics", "bitfields", "util", "round", "libdevice", "cmath"]
    # Add tcgen05 library for SM 10.0+ to avoid MLIR parsing issues on older GPUs
    cc = get_gpu_compute_capability(tuple)
    if cc >= (10, 0):
        libs.append("nvvm_tcgen05")
    return libs


# Python-generated libraries (not .mlir files) - these are handled via @property
_PYTHON_GENERATED_LIBS = {"round", "cmath"}


class _MLIRRuntimeLibraryWrapper:
    @property
    @lru_cache
    def round(self):
        from .round_intrinsics import get_round_intrinsics_module

        return declare_mlir_library(get_round_intrinsics_module())

    @property
    @lru_cache
    def cmath(self):
        from .cmath_intrinsics import get_cmath_intrinsics_module

        return declare_mlir_library(get_cmath_intrinsics_module())

    def __getattr__(self, name) -> ExternMLIRLibrary | None:
        trace(name)
        maybe_lib = (_FILE_ROOT_DIR / name).with_suffix(".mlir")
        if maybe_lib.exists():
            return declare_mlir_library(maybe_lib)
        raise AttributeError(f"No MLIR runtime library {name}")


_runtime = _MLIRRuntimeLibraryWrapper()
for lib in _get_all_libraries():
    setattr(sys.modules[__name__], lib, getattr(_runtime, lib))
