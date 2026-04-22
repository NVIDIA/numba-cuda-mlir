# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cusimt
from cusimt.cuda import consteval
from cusimt import cuda
import numpy as np


def test_constant_if_true_branch():
    """Test that if True: ... else: ... is folded to the true branch."""

    @cusimt.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(True):
            arr[i] = 1.0
        else:
            arr[i] = 2.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "arr[i] = 1.0" in source
    assert "arr[i] = 2.0" not in source
    assert "if True" not in source
    assert "if False" not in source


def test_constant_if_false_branch():
    """Test that if False: ... else: ... is folded to the else branch."""

    @cusimt.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(False):
            arr[i] = 1.0
        else:
            arr[i] = 2.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "arr[i] = 2.0" in source
    assert "arr[i] = 1.0" not in source


def test_constant_if_no_else():
    """Test that if False: ... with no else is removed entirely."""

    @cusimt.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = 0.0
        if consteval(False):
            arr[i] = 999.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "arr[i] = 0.0" in source
    assert "arr[i] = 999.0" not in source


def test_constant_if_expression():
    """Test constant if with consteval expression."""
    USE_FAST_PATH = True

    @cusimt.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(USE_FAST_PATH):
            arr[i] = float(i)
        else:
            arr[i] = float(i) * 2.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "arr[i] = float(i)" in source
    assert "float(i) * 2.0" not in source


def test_constant_if_runs_correctly():
    """Test that constant if folding produces correct runtime behavior."""

    @cusimt.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(True):
            arr[i] = 42.0
        else:
            arr[i] = 0.0

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert all(result == 42.0)


def test_nested_constant_if():
    """Test nested constant if statements."""
    OUTER = True
    INNER = False

    @cusimt.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        if consteval(OUTER):
            if consteval(INNER):
                arr[i] = 1.0
            else:
                arr[i] = 2.0
        else:
            arr[i] = 3.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "arr[i] = 2.0" in source
    assert "arr[i] = 1.0" not in source
    assert "arr[i] = 3.0" not in source
