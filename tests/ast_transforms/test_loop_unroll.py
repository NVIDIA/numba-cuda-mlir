# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numba_cuda_mlir
from numba_cuda_mlir.cuda import consteval
from numba_cuda_mlir.ast_transforms import ConstevalError
from numba_cuda_mlir import cuda
import numpy as np
import pytest


def test_unroll_range():
    """Test basic loop unrolling with range()."""

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for j in consteval(range(3)):
            arr[i * 3 + j] = float(j)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    # Should have unrolled to 3 assignments
    assert "arr[i * 3 + 0] = float(0)" in source
    assert "arr[i * 3 + 1] = float(1)" in source
    assert "arr[i * 3 + 2] = float(2)" in source
    # Should not have a for loop
    assert "for j in" not in source


def test_unroll_range_runs():
    """Test that unrolled range loop executes correctly."""

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for j in consteval(range(4)):
            arr[i * 4 + j] = float(j * 10)

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 8](d_a)
    result = d_a.copy_to_host()

    # Thread 0: indices 0,1,2,3 -> values 0,10,20,30
    assert result[0] == 0.0
    assert result[1] == 10.0
    assert result[2] == 20.0
    assert result[3] == 30.0
    # Thread 1: indices 4,5,6,7 -> values 0,10,20,30
    assert result[4] == 0.0
    assert result[5] == 10.0


def test_unroll_tuple():
    """Test loop unrolling with a tuple."""
    SIZES = (8, 16, 32)

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        total = 0.0
        for s in consteval(SIZES):
            total = total + float(s)
        arr[i] = total

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "total = total + float(8)" in source
    assert "total = total + float(16)" in source
    assert "total = total + float(32)" in source


def test_unroll_tuple_runs():
    """Test that unrolled tuple loop executes correctly."""
    VALUES = (1, 2, 4, 8)

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        total = 0.0
        for v in consteval(VALUES):
            total = total + float(v)
        arr[i] = total

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    # 1 + 2 + 4 + 8 = 15
    assert all(result == 15.0)


def test_unroll_list():
    """Test loop unrolling with a list."""
    ITEMS = [10, 20, 30]

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        offset = 0
        for val in consteval(ITEMS):
            arr[i * 3 + offset] = float(val)
            offset = offset + 1

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert "float(10)" in source
    assert "float(20)" in source
    assert "float(30)" in source


def test_unroll_with_consteval_count():
    """Test loop unrolling where count comes from consteval."""
    N = 4

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        count = consteval(N)
        for j in consteval(range(count)):
            arr[i * count + j] = float(j)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "count = 4" in source
    # Note: count stays as a variable name, j is replaced with constants
    assert "arr[i * count + 0]" in source
    assert "arr[i * count + 3]" in source


def test_unroll_nested_consteval():
    """Test loop unrolling with consteval inside the loop body."""
    MULTIPLIERS = (2, 3, 5)

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for m in consteval(MULTIPLIERS):
            factor = consteval(m * 10)
            arr[i] = arr[i] + float(factor)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # Each iteration should have its own evaluated factor
    assert "factor = 20" in source  # 2 * 10
    assert "factor = 30" in source  # 3 * 10
    assert "factor = 50" in source  # 5 * 10


def test_unroll_conditional_inside():
    """Test loop unrolling with conditional inside based on loop var."""
    FLAGS = (True, False, True)

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        offset = 0
        for flag in consteval(FLAGS):
            if consteval(flag):
                arr[i * 3 + offset] = 1.0
            else:
                arr[i * 3 + offset] = 0.0
            offset = offset + 1

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # First iteration: flag=True, so only true branch
    # Second iteration: flag=False, so only false branch
    # Third iteration: flag=True, so only true branch
    # After constant if folding, we should see the appropriate assignments
    # Note: offset is a runtime variable, so it appears as 'offset' not 0,1,2
    # The key thing is that the if/else branches are folded correctly
    lines = source.split("\n")
    arr_lines = [l.strip() for l in lines if "arr[i * 3 + offset]" in l]
    assert len(arr_lines) == 3
    assert arr_lines[0] == "arr[i * 3 + offset] = 1.0"  # True
    assert arr_lines[1] == "arr[i * 3 + offset] = 0.0"  # False
    assert arr_lines[2] == "arr[i * 3 + offset] = 1.0"  # True


def test_unroll_string_items():
    """Test loop unrolling with string items (for compile-time selection)."""
    TYPES = ("float", "int")

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for t in consteval(TYPES):
            if consteval(t == "float"):
                arr[i] = 1.0
            else:
                arr[i] = 2.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # Both branches should appear (one for each iteration)
    assert "arr[i] = 1.0" in source
    assert "arr[i] = 2.0" in source


def test_unroll_computed_range():
    """Test loop unrolling with computed range bounds."""
    BASE = 2
    MULT = 3

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for j in consteval(range(BASE * MULT)):
            arr[i * 6 + j] = float(j)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # Should unroll to 6 iterations (2 * 3)
    for j in range(6):
        assert f"arr[i * 6 + {j}] = float({j})" in source


def test_unroll_empty_range():
    """Test loop unrolling with empty range."""

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        arr[i] = 0.0
        for j in consteval(range(0)):
            arr[i] = float(j)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # Should just have the initial assignment, no loop body
    assert "arr[i] = 0.0" in source
    # The loop body should not appear at all
    lines = [l.strip() for l in source.split("\n") if "arr[i]" in l]
    assert len(lines) == 1  # Only the arr[i] = 0.0 line


def test_unroll_non_iterable_raises():
    """Test that consteval with non-iterable raises error."""

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for j in consteval(42):
            arr[i] = float(j)

    # Error is raised at compile time (when AST transforms run)
    with pytest.raises(ConstevalError, match="must be iterable"):
        kernel.compile("void(float32[:])")


def test_unroll_nested_loops():
    """Test nested loop unrolling."""
    OUTER = (0, 1)
    INNER = (0, 1, 2)

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        for a in consteval(OUTER):
            for b in consteval(INNER):
                arr[i * 6 + a * 3 + b] = float(a * 10 + b)

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    # Should have 2 * 3 = 6 assignments
    # Note: expressions like a*10+b become 0*10+0, not evaluated to 0
    assert "arr[i * 6 + 0 * 3 + 0] = float(0 * 10 + 0)" in source
    assert "arr[i * 6 + 0 * 3 + 1] = float(0 * 10 + 1)" in source
    assert "arr[i * 6 + 0 * 3 + 2] = float(0 * 10 + 2)" in source
    assert "arr[i * 6 + 1 * 3 + 0] = float(1 * 10 + 0)" in source
    assert "arr[i * 6 + 1 * 3 + 1] = float(1 * 10 + 1)" in source
    assert "arr[i * 6 + 1 * 3 + 2] = float(1 * 10 + 2)" in source


def test_unroll_nested_loops_runs():
    """Test that nested unrolled loops execute correctly."""

    @numba_cuda_mlir.cuda.jit(experimental_ast_transforms=True)
    def kernel(arr):
        i = cuda.threadIdx.x
        if i == 0:
            for a in consteval(range(2)):
                for b in consteval(range(3)):
                    arr[a * 3 + b] = float(a * 10 + b)

    a = np.zeros(6, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 1](d_a)
    result = d_a.copy_to_host()

    expected = [0, 1, 2, 10, 11, 12]
    np.testing.assert_array_equal(result, expected)
