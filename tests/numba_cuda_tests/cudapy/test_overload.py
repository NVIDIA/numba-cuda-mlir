# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda import types

from numba_cuda_mlir.numba_cuda.core.errors import TypingError

from numba_cuda_mlir.extending import overload, refresh_registries, typing_registry
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof as cpu_typeof
from numba_cuda_mlir.testing import NumbaCUDATestCase
import numpy as np
import pytest


# Dummy function definitions to overload


def generic_func_1():
    pass


def cuda_func_1():
    pass


def generic_func_2():
    pass


def cuda_func_2():
    pass


def generic_calls_generic():
    pass


def generic_calls_cuda():
    pass


def cuda_calls_generic():
    pass


def cuda_calls_cuda():
    pass


def target_overloaded():
    pass


def generic_calls_target_overloaded():
    pass


def cuda_calls_target_overloaded():
    pass


def target_overloaded_calls_target_overloaded():
    pass


def default_values_and_kwargs():
    pass


# To recognise which functions are resolved for a call, we identify each with a
# prime number. Each function called multiplies a value by its prime (starting
# with the value 1), and we can check that the result is as expected based on
# the final value after all multiplications.

GENERIC_FUNCTION_1 = 2
CUDA_FUNCTION_1 = 3
GENERIC_FUNCTION_2 = 5
CUDA_FUNCTION_2 = 7
GENERIC_CALLS_GENERIC = 11
GENERIC_CALLS_CUDA = 13
CUDA_CALLS_GENERIC = 17
CUDA_CALLS_CUDA = 19
GENERIC_TARGET_OL = 23
CUDA_TARGET_OL = 29
GENERIC_CALLS_TARGET_OL = 31
CUDA_CALLS_TARGET_OL = 37
GENERIC_TARGET_OL_CALLS_TARGET_OL = 41
CUDA_TARGET_OL_CALLS_TARGET_OL = 43


# Overload implementations


@overload(generic_func_1, target="generic", typing_registry=typing_registry)
def ol_generic_func_1(x):
    def impl(x):
        x[0] *= GENERIC_FUNCTION_1

    return impl


@overload(cuda_func_1, target="cuda", typing_registry=typing_registry)
def ol_cuda_func_1(x):
    def impl(x):
        x[0] *= CUDA_FUNCTION_1

    return impl


@overload(generic_func_2, target="generic", typing_registry=typing_registry)
def ol_generic_func_2(x):
    def impl(x):
        x[0] *= GENERIC_FUNCTION_2

    return impl


@overload(cuda_func_2, target="cuda", typing_registry=typing_registry)
def ol_cuda_func(x):
    def impl(x):
        x[0] *= CUDA_FUNCTION_2

    return impl


@overload(generic_calls_generic, target="generic", typing_registry=typing_registry)
def ol_generic_calls_generic(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_GENERIC
        generic_func_1(x)

    return impl


@overload(generic_calls_cuda, target="generic", typing_registry=typing_registry)
def ol_generic_calls_cuda(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_CUDA
        cuda_func_1(x)

    return impl


@overload(cuda_calls_generic, target="cuda", typing_registry=typing_registry)
def ol_cuda_calls_generic(x):
    def impl(x):
        x[0] *= CUDA_CALLS_GENERIC
        generic_func_1(x)

    return impl


@overload(cuda_calls_cuda, target="cuda", typing_registry=typing_registry)
def ol_cuda_calls_cuda(x):
    def impl(x):
        x[0] *= CUDA_CALLS_CUDA
        cuda_func_1(x)

    return impl


@overload(target_overloaded, target="generic", typing_registry=typing_registry)
def ol_target_overloaded_generic(x):
    def impl(x):
        x[0] *= GENERIC_TARGET_OL

    return impl


@overload(target_overloaded, target="cuda", typing_registry=typing_registry)
def ol_target_overloaded_cuda(x):
    def impl(x):
        x[0] *= CUDA_TARGET_OL

    return impl


@overload(generic_calls_target_overloaded, target="generic", typing_registry=typing_registry)
def ol_generic_calls_target_overloaded(x):
    def impl(x):
        x[0] *= GENERIC_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(cuda_calls_target_overloaded, target="cuda", typing_registry=typing_registry)
def ol_cuda_calls_target_overloaded(x):
    def impl(x):
        x[0] *= CUDA_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(
    target_overloaded_calls_target_overloaded,
    target="generic",
    typing_registry=typing_registry,
)
def ol_generic_calls_target_overloaded_generic(x):
    def impl(x):
        x[0] *= GENERIC_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(
    target_overloaded_calls_target_overloaded,
    target="cuda",
    typing_registry=typing_registry,
)
def ol_generic_calls_target_overloaded_cuda(x):
    def impl(x):
        x[0] *= CUDA_TARGET_OL_CALLS_TARGET_OL
        target_overloaded(x)

    return impl


@overload(default_values_and_kwargs, typing_registry=typing_registry)
def ol_default_values_and_kwargs(out, x, y=5, z=6):
    def impl(out, x, y=5, z=6):
        out[0], out[1] = x + y, z

    return impl


refresh_registries()


class TestOverload(NumbaCUDATestCase):
    def check_overload(self, kernel, expected):
        x = np.ones(1, dtype=np.int32)
        numba_cuda_mlir.cuda.jit(kernel)[1, 1](x)
        self.assertEqual(x[0], expected)

    def test_generic(self):
        def kernel(x):
            generic_func_1(x)

        expected = GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda(self):
        def kernel(x):
            cuda_func_1(x)

        expected = CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_and_cuda(self):
        def kernel(x):
            generic_func_1(x)
            cuda_func_1(x)

        expected = GENERIC_FUNCTION_1 * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_two_generic_calls(self):
        def kernel(x):
            generic_func_1(x)
            generic_func_2(x)

        expected = GENERIC_FUNCTION_1 * GENERIC_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_call_two_cuda_calls(self):
        def kernel(x):
            cuda_func_1(x)
            cuda_func_2(x)

        expected = CUDA_FUNCTION_1 * CUDA_FUNCTION_2
        self.check_overload(kernel, expected)

    def test_generic_calls_generic(self):
        def kernel(x):
            generic_calls_generic(x)

        expected = GENERIC_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_generic_calls_cuda(self):
        def kernel(x):
            generic_calls_cuda(x)

        expected = GENERIC_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_generic(self):
        def kernel(x):
            cuda_calls_generic(x)

        expected = CUDA_CALLS_GENERIC * GENERIC_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_cuda_calls_cuda(self):
        def kernel(x):
            cuda_calls_cuda(x)

        expected = CUDA_CALLS_CUDA * CUDA_FUNCTION_1
        self.check_overload(kernel, expected)

    def test_call_target_overloaded(self):
        def kernel(x):
            target_overloaded(x)

        expected = CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_generic_calls_target_overloaded(self):
        def kernel(x):
            generic_calls_target_overloaded(x)

        expected = GENERIC_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_cuda_calls_target_overloaded(self):
        def kernel(x):
            cuda_calls_target_overloaded(x)

        expected = CUDA_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    def test_target_overloaded_calls_target_overloaded(self):
        def kernel(x):
            target_overloaded_calls_target_overloaded(x)

        # Check the CUDA overloads are used on CUDA
        expected = CUDA_TARGET_OL_CALLS_TARGET_OL * CUDA_TARGET_OL
        self.check_overload(kernel, expected)

    @pytest.mark.xfail(True, reason="ICE")
    def test_default_values_and_kwargs(self):
        """
        Test default values and kwargs.
        """

        @numba_cuda_mlir.cuda.jit()
        def kernel(a, b, out):
            default_values_and_kwargs(out, a, z=b)

        out = np.empty(2, dtype=np.int64)
        kernel[1, 1](1, 2, out)
        self.assertEqual(tuple(out), (6, 2))
