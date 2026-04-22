# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for string literal types in CUDA kernels.

String literals are represented as compile-time constants; equality (==) is
constant-folded. The backend value type is !llvm.ptr (i8*, null-terminated)
when materialized; see cusimt.models.StringLiteralModel.
"""

import numpy as np
import pytest

import cuda.simt as cs


@pytest.mark.parametrize(
    "literal,compare_against,expected",
    [
        ("a", "a", True),
        ("a", "A", False),
        ("ABC", "a", False),
        ("ABC", "ABC", True),
        ("", "", True),
        ("", "x", False),
        ("x", "", False),
        ("hello", "hello", True),
        ("hello", "world", False),
    ],
)
def test_string_literal_eq(literal, compare_against, expected):
    """Kernel compares a string literal against another; result in out[0]."""

    @cs.jit
    def kernel(out):
        out[0] = literal == compare_against

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0] == expected


def test_string_literal_eq_device_function_same():
    """Device function receives literal and compares to fixed string; same -> True."""

    @cs.jit(device=True)
    def do_cmp(st):
        return st == "a"

    @cs.jit
    def kernel(out):
        out[0] = do_cmp("a")

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert bool(out[0]) is True


def test_string_literal_eq_device_function_different():
    """Device function receives literal and compares to fixed string; different -> False."""

    @cs.jit(device=True)
    def do_cmp(st):
        return st == "a"

    @cs.jit
    def kernel(out):
        out[0] = do_cmp("ABC")

    out = np.zeros(1, dtype=np.bool_)
    kernel[1, 1](out)
    assert bool(out[0]) is False


@pytest.mark.parametrize(
    "literal,expected",
    [
        ("a", True),
        ("A", False),
        ("", False),
        ("xyz", False),
    ],
)
def test_string_literal_eq_device_function_parametrized(literal, expected):
    """Device function do_cmp(st): return st == "a"; kernel calls with various literals."""

    @cs.jit(device=True)
    def do_cmp(st):
        return st == "a"

    @cs.jit
    def kernel(out, lit):
        out[0] = do_cmp(lit)

    # We cannot pass a runtime string from host; we only support literals.
    # So we need one kernel per literal (compiled with that literal inlined).
    # Parametrize builds a new kernel per (literal, expected) via the closure.
    def make_kernel(lit):
        @cs.jit
        def k(out):
            out[0] = do_cmp(lit)

        return k

    out = np.zeros(1, dtype=np.bool_)
    make_kernel(literal)[1, 1](out)
    assert out[0] == expected


def test_string_literal_eq_multiple_in_one_kernel():
    """Multiple string literal comparisons in a single kernel."""

    @cs.jit
    def kernel(out):
        out[0] = "a" == "a"
        out[1] = "a" == "b"
        out[2] = "" == ""
        out[3] = "x" == "x"

    out = np.zeros(4, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0] == True
    assert out[1] == False
    assert out[2] == True
    assert out[3] == True


def test_string_literal_eq_used_in_conditional():
    """String comparison result used in branch."""

    @cs.jit
    def kernel(out):
        if "yes" == "yes":
            out[0] = 1
        else:
            out[0] = 0
        if "no" == "yes":
            out[1] = 1
        else:
            out[1] = 0

    out = np.zeros(2, dtype=np.int32)
    kernel[1, 1](out)
    assert out[0] == 1
    assert out[1] == 0


def test_string_literal_eq_single_char():
    """Single-character literal comparison."""

    @cs.jit
    def kernel(out):
        out[0] = "x" == "x"
        out[1] = "x" == "y"

    out = np.zeros(2, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0] == True
    assert out[1] == False


def test_string_literal_eq_unicode_ascii_subset():
    """ASCII subset (kernel string literals are typically ASCII in practice)."""

    # Literals that are fine as ASCII
    @cs.jit
    def kernel(out):
        out[0] = "cafe" == "cafe"
        out[1] = "cafe" == "Cafe"

    out = np.zeros(2, dtype=np.bool_)
    kernel[1, 1](out)
    assert out[0] == True
    assert out[1] == False


# -----------------------------------------------------------------------------
# CharSeq / UnicodeCharSeq: numpy 'S' and 'U' dtype arrays
# These correspond to numba.core.types.CharSeq and UnicodeCharSeq.
# -----------------------------------------------------------------------------


@pytest.mark.xfail(reason="b'...' literals go through Bytes type (cast path NYI)")
class TestCharSeqAssign:
    """Assign string/byte literals to numpy S (byte string) arrays."""

    def test_assign_const_byte_string(self):
        @cs.jit
        def kernel(arr):
            i = cs.grid(1)
            if i < len(arr):
                arr[i] = b"XYZ"

        n = 8
        arr = np.zeros(n + 1, dtype="S12")
        kernel[1, n](arr)
        expected = np.zeros_like(arr)
        expected[:n] = b"XYZ"
        np.testing.assert_equal(arr, expected)

    def test_assign_const_byte_string_short(self):
        @cs.jit
        def kernel(arr):
            arr[0] = b"A"

        arr = np.zeros(1, dtype="S4")
        kernel[1, 1](arr)
        assert arr[0] == b"A"

    def test_assign_const_byte_string_empty(self):
        @cs.jit
        def kernel(arr):
            arr[0] = b""

        arr = np.array([b"hello"], dtype="S8")
        kernel[1, 1](arr)
        assert arr[0] == b""


class TestUnicodeCharSeqAssign:
    """Assign unicode string literals to numpy U (unicode) arrays."""

    def test_assign_const_unicode_string(self):
        @cs.jit
        def kernel(arr):
            i = cs.grid(1)
            if i < len(arr):
                arr[i] = "XYZ"

        n = 8
        arr = np.zeros(n + 1, dtype="<U12")
        kernel[1, n](arr)
        expected = np.zeros_like(arr)
        expected[:n] = "XYZ"
        np.testing.assert_equal(arr, expected)

    def test_assign_const_unicode_string_short(self):
        @cs.jit
        def kernel(arr):
            arr[0] = "A"

        arr = np.zeros(1, dtype="<U4")
        kernel[1, 1](arr)
        assert arr[0] == "A"

    def test_assign_const_unicode_string_empty(self):
        @cs.jit
        def kernel(arr):
            arr[0] = ""

        arr = np.array(["hello"], dtype="<U8")
        kernel[1, 1](arr)
        assert arr[0] == ""


@pytest.mark.xfail(reason="Record dtype with string fields NYI")
class TestCharSeqInRecord:
    """String fields in numpy record (structured) dtype arrays."""

    def test_assign_const_string_in_record(self):
        @cs.jit
        def kernel(a):
            a[0]["x"] = 1
            a[0]["y"] = "ABC"
            a[1]["x"] = 2
            a[1]["y"] = "XYZ"

        dt = np.dtype([("x", np.int32), ("y", np.dtype("<U12"))])
        a = np.zeros(2, dt)
        kernel[1, 1](a)
        reference = np.asarray([(1, "ABC"), (2, "XYZ")], dtype=dt)
        np.testing.assert_array_equal(reference, a)

    def test_assign_const_bytes_in_record(self):
        @cs.jit
        def kernel(a):
            a[0]["x"] = 1
            a[0]["y"] = b"ABC"
            a[1]["x"] = 2
            a[1]["y"] = b"XYZ"

        dt = np.dtype([("x", np.float32), ("y", np.dtype("S12"))])
        a = np.zeros(2, dt)
        kernel[1, 1](a)
        reference = np.asarray([(1, b"ABC"), (2, b"XYZ")], dtype=dt)
        np.testing.assert_array_equal(reference, a)
