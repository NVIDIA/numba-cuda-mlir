# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example of using print intrinsic
"""

from textwrap import dedent

from numba_cuda_mlir.testing import run_in_subprocess


def _run_cuda_print(code: str) -> str:
    stdout, stderr = run_in_subprocess(dedent(code))
    return stdout + stderr


def test_printf_simple():
    _run_cuda_print(
        """
        from numba_cuda_mlir import cuda
        import numpy as np

        @cuda.jit(dump=True)
        def k(d: cuda.DeviceNDArray):
            print("hello from kernel ", 0)
            print("hello from kernel ", 0, 3.14, True)
            print(d)

        h = np.random.randn(2, 3, 4).astype(np.float32)
        d = cuda.to_device(h)
        stream = int(cuda.default_stream())
        k[1, 1, stream, 0](d)
        cuda.synchronize()
        """
    )


def test_print_space_separator():
    """Test that print adds spaces between arguments."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k():
            print(1, 2, 3)

        k[1, 1]()
        cuda.synchronize()
        """
    )
    assert "1 2 3" in out


def test_print_bool_true_false():
    """Test that booleans print as True/False, not 1/0."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k():
            print(True)
            print(False)

        k[1, 1]()
        cuda.synchronize()
        """
    )
    assert "True" in out
    assert "False" in out


def test_print_bool_variable():
    """Test that boolean variables print as True/False."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k(x):
            print(x == 0)

        k[1, 1](0)
        cuda.synchronize()
        """
    )
    assert "True" in out


def test_print_tuple():
    """Test printing tuples."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k(tup):
            print(tup)

        k[1, 1]((1, 2, 3))
        cuda.synchronize()
        """
    )
    assert "(1, 2, 3)" in out


def test_print_single_element_tuple():
    """Test printing single-element tuples with trailing comma."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k(tup):
            print(tup)

        k[1, 1]((42,))
        cuda.synchronize()
        """
    )
    assert "(42,)" in out


def test_print_dim3():
    """Test printing Dim3 objects like cuda.threadIdx."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k():
            print(cuda.threadIdx)

        k[1, 1]()
        cuda.synchronize()
        """
    )
    assert "(0, 0, 0)" in out


def test_print_empty():
    """Test empty print() outputs just a newline."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k():
            print()
            print("after")

        k[1, 1]()
        cuda.synchronize()
        """
    )
    assert "after" in out


def test_print_string_literal():
    """String literals in print() are materialized as MLIR unicode structs
    and printed character-by-character via _lower_string_struct_print."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k():
            print("hello world")

        k[1, 1]()
        cuda.synchronize()
        """
    )
    assert "hello world" in out


def test_print_string_literal_with_other_args():
    """String literals mixed with numeric arguments."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda
        import numpy as np

        @cuda.jit
        def k(x):
            print("value:", x[0])

        arr = cuda.to_device(np.array([42], dtype=np.int64))
        k[1, 1](arr)
        cuda.synchronize()
        """
    )
    assert "value:" in out
    assert "42" in out


def test_print_multiple_string_literals():
    """Multiple string literals in separate print calls."""

    out = _run_cuda_print(
        """
        from numba_cuda_mlir import cuda

        @cuda.jit
        def k():
            print("first")
            print("second")

        k[1, 1]()
        cuda.synchronize()
        """
    )
    assert "first" in out
    assert "second" in out


if __name__ == "__main__":
    test_printf_simple()
