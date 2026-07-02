# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Isolate bundled LLVM/MLIR libraries from any differently-versioned LLVM
# elsewhere in the process (Triton, Halide, Mojo, ...). Must run before
# any MLIR Python binding is imported. See #170.
_mlir_deepbind_handles = []


def _preload_mlir_libs_with_deepbind():
    import sys

    if not sys.platform.startswith("linux"):
        return  # macOS / Windows have different symbol isolation semantics

    import ctypes
    import os
    from pathlib import Path

    mode = (
        os.RTLD_NOW | os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)  # missing on musl / uclibc
    )
    libs_dir = Path(__file__).parent / "_mlir" / "_mlir_libs"

    # Dependencies first: libMLIRPythonCAPI.so has no sibling DT_NEEDED
    # entries, so it loads cleanly with DEEPBIND. The other bundled libs
    # DT_NEEDED it; loading them after CAPI means the transitive resolves
    # find CAPI already in the process with DEEPBIND in place.
    for name in (
        "libMLIRPythonCAPI.so",
        "libMLIRPythonSupport-numba_cuda_mlir.so",
        "libMLIRToLLVM70.so",
        "libMLIRModernToNVVM.so",
    ):
        lib = libs_dir / name
        if lib.exists():
            _mlir_deepbind_handles.append(ctypes.CDLL(str(lib), mode=mode))


_preload_mlir_libs_with_deepbind()
del _preload_mlir_libs_with_deepbind

from numba_cuda_mlir._version import __version__
from numba_cuda_mlir.mlir import make_nanobind_metaclass_inheritable
from numba_cuda_mlir.numba_cuda.np.numpy_support import carray, farray  # noqa: F401

make_nanobind_metaclass_inheritable()

__all__ = ["cuda", "carray", "farray", "__version__"]
