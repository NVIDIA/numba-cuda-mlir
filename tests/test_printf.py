# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Example of using print intrinsic
"""

import cuda.simt as cs
import numpy as np
import pytest


def test_printf_simple():
    @cs.jit(dump=True)
    def k(d: cs.DeviceNDArray):
        print("hello from kernel ", 0)
        print("hello from kernel ", 0, 3.14, True)
        print(d)

    h = np.random.randn(2, 3, 4).astype(np.float32)
    d = cs.to_device(h)
    stream = int(cs.default_stream())
    k[1, 1, stream, 0](d)


def test_print_space_separator(capfd):
    """Test that print adds spaces between arguments."""

    @cs.jit
    def k():
        print(1, 2, 3)

    k[1, 1]()
    cs.synchronize()
    out = capfd.readouterr().out
    assert "1 2 3" in out


def test_print_bool_true_false(capfd):
    """Test that booleans print as True/False, not 1/0."""

    @cs.jit
    def k():
        print(True)
        print(False)

    k[1, 1]()
    cs.synchronize()
    out = capfd.readouterr().out
    assert "True" in out
    assert "False" in out


def test_print_bool_variable(capfd):
    """Test that boolean variables print as True/False."""

    @cs.jit
    def k(x):
        print(x == 0)

    k[1, 1](0)
    cs.synchronize()
    out = capfd.readouterr().out
    assert "True" in out


def test_print_tuple(capfd):
    """Test printing tuples."""

    @cs.jit
    def k(tup):
        print(tup)

    k[1, 1]((1, 2, 3))
    cs.synchronize()
    out = capfd.readouterr().out
    assert "(1, 2, 3)" in out


def test_print_single_element_tuple(capfd):
    """Test printing single-element tuples with trailing comma."""

    @cs.jit
    def k(tup):
        print(tup)

    k[1, 1]((42,))
    cs.synchronize()
    out = capfd.readouterr().out
    assert "(42,)" in out


def test_print_dim3(capfd):
    """Test printing Dim3 objects like cuda.threadIdx."""

    @cs.jit
    def k():
        print(cs.threadIdx)

    k[1, 1]()
    cs.synchronize()
    out = capfd.readouterr().out
    assert "(0, 0, 0)" in out


def test_print_empty(capfd):
    """Test empty print() outputs just a newline."""

    @cs.jit
    def k():
        print()
        print("after")

    k[1, 1]()
    cs.synchronize()
    out = capfd.readouterr().out
    assert "after" in out


def test_print_string_literal(capfd):
    """String literals in print() are materialized as MLIR unicode structs
    and printed character-by-character via _lower_string_struct_print."""

    @cs.jit
    def k():
        print("hello world")

    k[1, 1]()
    cs.synchronize()
    out = capfd.readouterr().out
    assert "hello world" in out


def test_print_string_literal_with_other_args(capfd):
    """String literals mixed with numeric arguments."""

    @cs.jit
    def k(x):
        print("value:", x[0])

    arr = cs.to_device(np.array([42], dtype=np.int64))
    k[1, 1](arr)
    cs.synchronize()
    out = capfd.readouterr().out
    assert "value:" in out
    assert "42" in out


def test_print_multiple_string_literals(capfd):
    """Multiple string literals in separate print calls."""

    @cs.jit
    def k():
        print("first")
        print("second")

    k[1, 1]()
    cs.synchronize()
    out = capfd.readouterr().out
    assert "first" in out
    assert "second" in out


if __name__ == "__main__":
    test_printf_simple()
