# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NRT tests for string operations."""

import numpy as np
import pytest

import cuda.simt as cs


def test_materialize_string_constant_mlir():
    """Verify that passing a string literal to a device function
    produces valid MLIR with the string data as a global constant."""
    from cusimt import compiler

    @cs.jit
    def getstr(s):
        return s.upper()

    @cs.jit
    def kernel(out):
        out[0] = getstr("hello") == "HELLO"

    cres = compiler.compile_for(kernel, np.zeros(1, dtype=np.bool_))
    mlir = cres.mlir_module_str
    assert "__cusimt_str_" in mlir


def test_new_string_nrt():
    @cs.jit
    def getstr(s):
        return s.upper()

    @cs.jit
    def kernel(out):
        out[0] = getstr("test") == "TEST"

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0]


@pytest.mark.xfail(reason="LLVM version mismatch: MLIR emits LLVM 23, llvmlite supports 20")
def test_new_string_nrt_lto():
    @cs.jit(lto=True)
    def getstr(s):
        return s.upper()

    @cs.jit(lto=True)
    def kernel(out):
        out[0] = getstr("test") == "TEST"

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0]
