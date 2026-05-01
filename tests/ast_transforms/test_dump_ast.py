# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numba_cuda_mlir
from numba_cuda_mlir import cuda
import numpy as np
from numba_cuda_mlir.testing import filecheck_with_comments


def test_consteval_loop_unroll():
    """Test that consteval loop unrolling produces correct transformed source."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def k(arr):
        for i in numba_cuda_mlir.consteval(range(3)):
            arr[i] = float(i)

    # Compile first since AST transforms happen at compile time
    cres = k.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    print(source)
    # CHECK: def k(arr):
    # CHECK-NEXT: arr[0] = float(0)
    # CHECK-NEXT: arr[1] = float(1)
    # CHECK-NEXT: arr[2] = float(2)
    filecheck_with_comments(source)


def test_consteval_if_folding():
    """Test that consteval if statements are folded correctly."""
    config = {"debug": True}

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def k(arr):
        if numba_cuda_mlir.consteval(config["debug"]):
            arr[0] = 1.0
        else:
            arr[0] = 2.0

    # Compile first since AST transforms happen at compile time
    cres = k.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    print(source)
    # CHECK: def k(arr):
    # CHECK-NEXT: arr[0] = 1.0
    filecheck_with_comments(source)
    assert "else" not in source


def test_consteval_function_call():
    """Test that consteval can call functions at compile time."""

    def get_size():
        return 4

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def k(arr):
        n = numba_cuda_mlir.consteval(get_size())
        arr[0] = float(n)

    # Compile first since AST transforms happen at compile time
    cres = k.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    print(source)
    # CHECK: def k(arr):
    # CHECK-NEXT: n = 4
    # CHECK-NEXT: arr[0] = float(n)
    filecheck_with_comments(source)


def test_no_transforms_returns_none():
    """Test that inspect_transformed_source returns None when no transforms applied."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def k(arr):
        i = cuda.threadIdx.x
        arr[i] = float(i)

    # Need to compile first since AST transforms happen at compile time
    k.compile("void(float32[:])")
    sources = k.inspect_transformed_source()
    # All values in the dict should be None since there were no consteval() calls
    assert all(v is None for v in sources.values())


def test_transformed_kernel_runs():
    """Test that a transformed kernel compiles and runs correctly."""

    @numba_cuda_mlir.jit(experimental_ast_transforms=True)
    def k(arr):
        for i in numba_cuda_mlir.consteval(range(3)):
            arr[i] = float(i) * 2.0

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    k[1, 32](d_a)
    result = d_a.copy_to_host()

    assert result[0] == 0.0
    assert result[1] == 2.0
    assert result[2] == 4.0
