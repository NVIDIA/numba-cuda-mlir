# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Lightweight overrides for `numba.cuda` when redirected to numba-cuda-mlir.

This module must stay importable without importing `numba.cuda` (or other heavy
parts of numba-cuda-mlir) at import time, because it is used during bootstrapping of the
`numba.cuda` package redirect.
"""

from __future__ import annotations

import importlib

__all__ = [
    "jit",
    "declare_device",
    "compile",
    "compile_ptx",
    "compile_all",
    "compile_mlir",
    "compile_for",
    "compile_cubin",
]


def jit(*args, **kwargs):
    # Lazy import to avoid import-time cycles during `import numba.cuda`.
    from numba_cuda_mlir.decorators import mlir_jit

    return mlir_jit(*args, **kwargs)


def declare_device(*args, **kwargs):
    from numba_cuda_mlir.compiler import declare_device as _declare_device

    return _declare_device(*args, **kwargs)


def compile(*args, **kwargs):
    from numba_cuda_mlir.compiler import compile as _compile

    return _compile(*args, **kwargs)


def compile_ptx(*args, **kwargs):
    from numba_cuda_mlir.compiler import compile_ptx as _compile_ptx

    return _compile_ptx(*args, **kwargs)


def compile_all(*args, **kwargs):
    from numba_cuda_mlir.compiler import compile_all as _compile_all

    return _compile_all(*args, **kwargs)


def compile_mlir(*args, **kwargs):
    from numba_cuda_mlir.compiler import compile_mlir as _compile_mlir

    return _compile_mlir(*args, **kwargs)


def compile_for(*args, **kwargs):
    from numba_cuda_mlir.compiler import compile_for as _compile_for

    return _compile_for(*args, **kwargs)


def compile_cubin(*args, **kwargs):
    from numba_cuda_mlir.compiler import compile_cubin as _compile_cubin

    return _compile_cubin(*args, **kwargs)
