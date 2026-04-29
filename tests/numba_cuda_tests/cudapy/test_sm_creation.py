# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numpy as np
import cusimt
import cuda.simt as cuda
from cusimt.numba_cuda.types import float32, int32, void

from cusimt.numba_cuda.core.errors import TypingError
from cusimt.testing import NumbaCUDATestCase
from .extensions_usecases import struct_model_type
import pytest

GLOBAL_CONSTANT = 5
GLOBAL_CONSTANT_2 = 6
GLOBAL_CONSTANT_TUPLE = 5, 6


def udt_global_constants(A):
    sa = cuda.shared.array(shape=GLOBAL_CONSTANT, dtype=float32)
    i = cuda.grid(1)
    A[i] = sa[i]


def udt_global_build_tuple(A):
    sa = cuda.shared.array(shape=(GLOBAL_CONSTANT, GLOBAL_CONSTANT_2), dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_global_build_list(A):
    sa = cuda.shared.array(shape=[GLOBAL_CONSTANT, GLOBAL_CONSTANT_2], dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_global_constant_tuple(A):
    sa = cuda.shared.array(shape=GLOBAL_CONSTANT_TUPLE, dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_invalid_1(A):
    sa = cuda.shared.array(shape=A[0], dtype=float32)
    i = cuda.grid(1)
    A[i] = sa[i]


def udt_invalid_2(A):
    sa = cuda.shared.array(shape=(1, A[0]), dtype=float32)
    i, j = cuda.grid(2)
    A[i, j] = sa[i, j]


def udt_invalid_3(A):
    sa = cuda.shared.array(shape=(1, A[0]), dtype=float32)
    i = cuda.grid(1)
    A[i] = sa[i, 0]


class TestSharedMemoryCreation(NumbaCUDATestCase):
    def getarg(self):
        return np.array(100, dtype=np.float32, ndmin=1)

    def getarg2(self):
        return self.getarg().reshape(1, 1)

    def test_global_constants(self):
        udt = cusimt.jit((float32[:],))(udt_global_constants)
        udt[1, 1](self.getarg())

    def test_global_build_tuple(self):
        udt = cusimt.jit((float32[:, :],))(udt_global_build_tuple)
        udt[1, 1](self.getarg2())

    def test_global_build_list(self):
        with self.assertRaises(TypingError) as raises:
            cusimt.jit((float32[:, :],))(udt_global_build_list)

        self.assertIn(
            "No implementation of function Function(<function shared.array",
            str(raises.exception),
        )
        self.assertIn(
            "found for signature:\n \n "
            ">>> array(shape=list(int64)<iv=[5, 6]>, "
            "dtype=class(float32)",
            str(raises.exception),
        )

    def test_global_constant_tuple(self):
        udt = cusimt.jit((float32[:, :],))(udt_global_constant_tuple)
        udt[1, 1](self.getarg2())

    def test_invalid_1(self):
        # Scalar shape cannot be a floating point value
        with self.assertRaises(TypingError) as raises:
            cusimt.jit((float32[:],))(udt_invalid_1)

        self.assertIn(
            "No implementation of function Function(<function shared.array",
            str(raises.exception),
        )
        self.assertIn(
            "found for signature:\n \n "
            ">>> array(shape=float32, dtype=class(float32))",
            str(raises.exception),
        )

    @pytest.mark.xfail(True, reason="ICE")
    def test_invalid_2(self):
        # Tuple shape cannot contain a floating point value
        with self.assertRaises(TypingError) as raises:
            cusimt.jit((float32[:, :],))(udt_invalid_2)

        self.assertIn(
            "No implementation of function Function(<function shared.array",
            str(raises.exception),
        )
        self.assertIn(
            "found for signature:\n \n "
            ">>> array(shape=Tuple(Literal[int](1), "
            "array(float32, 1d, A)), dtype=class(float32))",
            str(raises.exception),
        )

    @pytest.mark.xfail(True, reason="Assertion not raised")
    def test_invalid_3(self):
        # Scalar shape must be literal
        with self.assertRaises(TypingError) as raises:
            cusimt.jit((int32[:],))(udt_invalid_1)

        self.assertIn(
            "No implementation of function Function(<function shared.array",
            str(raises.exception),
        )
        self.assertIn(
            "found for signature:\n \n " ">>> array(shape=int32, dtype=class(float32))",
            str(raises.exception),
        )

    @pytest.mark.xfail(True, reason="Assertion not raised")
    def test_invalid_4(self):
        # Tuple shape must contain only literals
        with self.assertRaises(TypingError) as raises:
            cusimt.jit((int32[:],))(udt_invalid_3)

        self.assertIn(
            "No implementation of function Function(<function shared.array",
            str(raises.exception),
        )
        self.assertIn(
            "found for signature:\n \n "
            ">>> array(shape=Tuple(Literal[int](1), int32), "
            "dtype=class(float32))",
            str(raises.exception),
        )

    def check_dtype(self, f, dtype):
        # Find the typing of the dtype argument to cuda.shared.array
        annotation = next(iter(f.overloads.values()))._type_annotation
        l_dtype = annotation.typemap["s"].dtype
        # Ensure that the typing is correct
        self.assertEqual(l_dtype, dtype)

    def test_numba_dtype(self):
        # Check that Numba types can be used as the dtype of a shared array
        @cusimt.jit(void(int32[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype=int32)
            s[0] = x[0]
            x[0] = s[0]

        self.check_dtype(f, int32)

    def test_numpy_dtype(self):
        # Check that NumPy types can be used as the dtype of a shared array
        @cusimt.jit(void(int32[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype=np.int32)
            s[0] = x[0]
            x[0] = s[0]

        self.check_dtype(f, int32)

    def test_string_dtype(self):
        # Check that strings can be used to specify the dtype of a shared array
        @cusimt.jit(void(int32[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype="int32")
            s[0] = x[0]
            x[0] = s[0]

        self.check_dtype(f, int32)

    def test_invalid_string_dtype(self):
        # Check that strings of invalid dtypes cause a typing error
        re = ".*Invalid NumPy dtype specified: 'int33'.*"
        with self.assertRaisesRegex(TypingError, re):

            @cusimt.jit(void(int32[::1]))
            def f(x):
                s = cuda.shared.array(10, dtype="int33")
                s[0] = x[0]
                x[0] = s[0]

    @pytest.mark.xfail(True, reason="Typing error")
    def test_type_with_struct_data_model(self):
        @cusimt.jit(void(struct_model_type[::1]))
        def f(x):
            s = cuda.shared.array(10, dtype=struct_model_type)
            s[0] = x[0]
            x[0] = s[0]

        self.check_dtype(f, struct_model_type)
