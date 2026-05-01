# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example of using print intrinsic
"""

from numba_cuda_mlir import cuda
import numpy as np
import pytest


def test_printf_simple():
    @cuda.jit(dump=True)
    def k(d: cuda.DeviceNDArray):
        print("hello from kernel ", 0)
        print("hello from kernel ", 0, 3.14, True)
        print(d)

    h = np.random.randn(2, 3, 4).astype(np.float32)
    d = cuda.to_device(h)
    stream = int(cuda.default_stream())
    k[1, 1, stream, 0](d)


def test_print_space_separator(capfd):
    """Test that print adds spaces between arguments."""

    @cuda.jit
    def k():
        print(1, 2, 3)

    k[1, 1]()
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "1 2 3" in out


def test_print_bool_true_false(capfd):
    """Test that booleans print as True/False, not 1/0."""

    @cuda.jit
    def k():
        print(True)
        print(False)

    k[1, 1]()
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "True" in out
    assert "False" in out


def test_print_bool_variable(capfd):
    """Test that boolean variables print as True/False."""

    @cuda.jit
    def k(x):
        print(x == 0)

    k[1, 1](0)
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "True" in out


def test_print_tuple(capfd):
    """Test printing tuples."""

    @cuda.jit
    def k(tup):
        print(tup)

    k[1, 1]((1, 2, 3))
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "(1, 2, 3)" in out


def test_print_single_element_tuple(capfd):
    """Test printing single-element tuples with trailing comma."""

    @cuda.jit
    def k(tup):
        print(tup)

    k[1, 1]((42,))
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "(42,)" in out


def test_print_dim3(capfd):
    """Test printing Dim3 objects like cuda.threadIdx."""

    @cuda.jit
    def k():
        print(cuda.threadIdx)

    k[1, 1]()
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "(0, 0, 0)" in out


def test_print_empty(capfd):
    """Test empty print() outputs just a newline."""

    @cuda.jit
    def k():
        print()
        print("after")

    k[1, 1]()
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "after" in out


def test_print_string_literal(capfd):
    """String literals in print() are materialized as MLIR unicode structs
    and printed character-by-character via _lower_string_struct_print."""

    @cuda.jit
    def k():
        print("hello world")

    k[1, 1]()
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "hello world" in out


def test_print_string_literal_with_other_args(capfd):
    """String literals mixed with numeric arguments."""

    @cuda.jit
    def k(x):
        print("value:", x[0])

    arr = cuda.to_device(np.array([42], dtype=np.int64))
    k[1, 1](arr)
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "value:" in out
    assert "42" in out


def test_print_multiple_string_literals(capfd):
    """Multiple string literals in separate print calls."""

    @cuda.jit
    def k():
        print("first")
        print("second")

    k[1, 1]()
    cuda.synchronize()
    out = capfd.readouterr().out
    assert "first" in out
    assert "second" in out


if __name__ == "__main__":
    test_printf_simple()
