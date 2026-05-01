# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NRT tests for string operations."""

import numpy as np
import pytest

from numba_cuda_mlir import cuda


def test_materialize_string_constant_mlir():
    """Verify that passing a string literal to a device function
    produces valid MLIR with the string data as a global constant."""
    from numba_cuda_mlir import compiler

    @cuda.jit
    def getstr(s):
        return s.upper()

    @cuda.jit
    def kernel(out):
        out[0] = getstr("hello") == "HELLO"

    cres = compiler.compile_for(kernel, np.zeros(1, dtype=np.bool_))
    mlir = cres.mlir_module_str
    assert "__numba_cuda_mlir_str_" in mlir


def test_new_string_nrt():
    @cuda.jit
    def getstr(s):
        return s.upper()

    @cuda.jit
    def kernel(out):
        out[0] = getstr("test") == "TEST"

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0]


def test_new_string_nrt_lto():
    @cuda.jit(lto=True)
    def getstr(s):
        return s.upper()

    @cuda.jit(lto=True)
    def kernel(out):
        out[0] = getstr("test") == "TEST"

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0]
