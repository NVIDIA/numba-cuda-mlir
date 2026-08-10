# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# DEEPBIND-preload bundled LLVM/MLIR libs before any MLIR binding is
# imported, so their symbols cannot be preempted by another in-process
# LLVM. See #170.
_mlir_deepbind_handles = []


def _preload_mlir_libs_with_deepbind():
    import sys

    if not sys.platform.startswith("linux"):
        return  # RTLD_DEEPBIND is Linux-only

    import ctypes
    import os
    from pathlib import Path

    mode = (
        os.RTLD_NOW | os.RTLD_LOCAL | getattr(os, "RTLD_DEEPBIND", 0)  # missing on musl / uclibc
    )
    libs_dir = Path(__file__).parent / "_mlir" / "_mlir_libs"

    # CAPI first: it has no sibling DT_NEEDED so DEEPBIND applies at
    # first load; the other libs DT_NEEDED it and reuse the handle.
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
